"""ds4_search/live_app.py — live reporting dashboard (web app).

ARCHITECTURE
============
A single Python process that:

  1. Serves a single-page HTML dashboard at http://<host>:<port>/
  2. Exposes a small read-only JSON API at /api/{state, data, heartbeat, csv}
  3. Spawns one ds4_search/sweep_runner.py subprocess and streams its stdout
     into a ring buffer for the in-browser activity log
  4. Reads the per-cell flow_*.csv files the sweep writes and serves them

The page polls the API every few seconds and re-renders the chart, table,
leaderboards, and activity log. It is purely a *reporting* surface — it does
not start, stop, or otherwise manage the workers. The sweep is started
automatically on launch (--auto-start, which main.py always passes), and the
only way to stop it is Ctrl-C in the terminal, which tears down both the
server and the sweep.

This is normally launched via `python main.py` (no arguments); main.py reads
the grid from main.toml and passes it here. It is not meant to be run
directly with hand-typed grid args.
"""
import argparse
import csv as csv_mod
import glob
import hashlib
import http.server
import json
import math
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

# Reuse the existing dashboard's per-cell helpers. These are pure
# functions: filename parsing, CSV loading, classification, stats.
from ds4_search.static_dashboard import (
    parse_filename, load_flow_csv, classify_shape, compute_stats,
)

# Errors raised when the client (a browser tab) goes away mid-response —
# the user closed the tab, hit reload, or navigated elsewhere. These are
# expected and harmless; they must never be surfaced as a server "error".
CLIENT_DISCONNECT_ERRORS = (
    ConnectionResetError, BrokenPipeError, ConnectionAbortedError,
)


# ═══════════════════════════════════════════════════════════════════
#  Application state (thread-safe)
# ═══════════════════════════════════════════════════════════════════
class AppState:
    """Holds the live state of the app: subprocess handle, heartbeat
    ring buffer, CPU sample for delta calc. All access goes through
    self.lock so HTTP handler threads and the capture thread coexist.
    """
    HEARTBEAT_MAXLEN = 200    # rolling buffer size for sweep stdout

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.sweep_proc = None
        self.sweep_started_at = None
        self.heartbeat = deque(maxlen=self.HEARTBEAT_MAXLEN)
        self.heartbeat_total = 0    # cumulative; exceeds buffer size
        self.startup_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        self._app_hash = self._compute_self_hash()
        self.public_url = None          # set once the socket is bound
        self._last_disconnect_note = 0.0  # rate-limit the "browser closed" line

    def _compute_self_hash(self):
        try:
            with open(__file__, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:8]
        except OSError:
            return "unknown"

    # ── Sweep lifecycle ───────────────────────────────────────────
    def is_sweep_running(self):
        with self.lock:
            return (self.sweep_proc is not None
                    and self.sweep_proc.poll() is None)

    def start_sweep(self):
        """Spawn ds4_search/sweep_runner.py if not already running. Returns
        True if a new process was started, False if one was
        already alive.
        """
        with self.lock:
            if self.sweep_proc and self.sweep_proc.poll() is None:
                return False
            cmd = self._build_sweep_cmd()
            self._heartbeat_locked(
                f"starting sweep (PID will be assigned): "
                f"{' '.join(cmd[:6])}…")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self.args.dir,
                    bufsize=1, text=True,
                    # Own session/process group (start_new_session=True →
                    # setsid in the child). This lets stop_sweep() signal
                    # the ENTIRE group with os.killpg — the __sweep__
                    # parent AND every ProcessPoolExecutor worker it
                    # forked — instead of just the parent PID. Without
                    # this, terminating the parent orphaned the workers,
                    # which kept running (and holding gigabytes) as
                    # lingering processes. We always tear the sweep down
                    # explicitly via stop_sweep() (on Ctrl-C, on SIGTERM,
                    # and on the dashboard's stop), so giving it its own
                    # session is safe.
                    start_new_session=True,
                )
            except OSError as e:
                self._heartbeat_locked(f"[ERROR] spawn failed: {e}")
                return False
            self.sweep_proc = proc
            self.sweep_started_at = time.time()
            self._heartbeat_locked(f"sweep started, PID {proc.pid}")
            t = threading.Thread(target=self._capture_loop,
                                 args=(proc,), daemon=True)
            t.start()
            return True

    def stop_sweep(self, timeout=10):
        """SIGTERM the whole sweep process group, wait up to timeout, then
        SIGKILL the group. Signalling the group (not just the parent PID)
        is what guarantees the forked ProcessPoolExecutor workers die too
        rather than lingering as orphans. Returns True if a sweep was
        running and got stopped.
        """
        with self.lock:
            proc = self.sweep_proc
            if not proc or proc.poll() is not None:
                return False
        # Resolve the process-group id. start_sweep used start_new_session,
        # so the sweep is its own group leader and pgid == proc.pid, but we
        # fetch it explicitly rather than assume.
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return False

        def _signal_group(sig):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass  # group already gone

        # Outside lock: terminate may take a while.
        self._heartbeat(f"stopping sweep (PGID {pgid}, SIGTERM to group)")
        _signal_group(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
            self._heartbeat(f"sweep PID {proc.pid} exited "
                            f"with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            self._heartbeat(f"sweep group {pgid} did not stop in "
                            f"{timeout}s — sending SIGKILL to group")
            _signal_group(signal.SIGKILL)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._heartbeat(f"[WARN] sweep group {pgid} won't die")
        return True

    def _build_sweep_cmd(self):
        # Spawn the sweep through the project's single entry point (main.py)
        # via its hidden worker route, so the child inherits the same
        # sys.path / output-dir setup and the sweep's own os.execv
        # hot-reload re-runs a valid command.
        main_py = str(Path(__file__).resolve().parents[2] / "main.py")
        cmd = [
            sys.executable, "-u", main_py, "__sweep__",   # -u: line-buffered
        ]
        # Forward only the grid args we were actually given. The normal launch
        # from main.py passes NONE of them, so the sweep backend reads the grid
        # straight from main.toml — one source of truth, no marshalling here.
        for flag, val in (("--k", self.args.k), ("--T", self.args.T),
                          ("--lb", self.args.lb), ("--N", self.args.N)):
            if val:
                cmd += [flag, val]
        # Multi-seed sweeps pass a seed *list*; a single seed is back-compat;
        # neither given → the sweep backend uses main.toml's seeds.
        if getattr(self.args, "seeds", None):
            cmd += ["--seeds", self.args.seeds]
        elif self.args.seed is not None:
            cmd += ["--seed", str(self.args.seed)]
        cmd += [
            "--torus-d", self.args.torus_d,
            "--status-json", os.path.join(
                self.args.dir, "lb_sweep_status.json"),
        ]
        if self.args.watch_reload:
            cmd.append("--watch-reload")
        return cmd

    def _capture_loop(self, proc):
        """Read sweep stdout line-by-line, push to heartbeat ring."""
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self._heartbeat(line)
        except Exception as e:
            self._heartbeat(f"[capture err] {type(e).__name__}: {e}")
        finally:
            self._heartbeat(f"sweep PID {proc.pid} stdout closed")

    def _heartbeat(self, line):
        with self.lock:
            self._heartbeat_locked(line)

    def _heartbeat_locked(self, line):
        ts = time.strftime("%H:%M:%S")
        self.heartbeat.append(f"[{ts}] {line}")
        self.heartbeat_total += 1

    def set_public_url(self, url):
        with self.lock:
            self.public_url = url

    def note_client_disconnect(self):
        """Record (rate-limited) that a browser tab closed mid-response.

        This is harmless — the server and the sweep keep running — so we
        log one friendly line instead of letting a scary traceback reach
        the terminal. Rate-limited because a single page issues several
        concurrent polls (state + data + heartbeat) that all disconnect
        together when the tab closes, and we don't want three identical
        lines each time.
        """
        with self.lock:
            now = time.time()
            if now - self._last_disconnect_note < 2.0:
                return
            self._last_disconnect_note = now
            where = (f" To keep watching, reopen {self.public_url}"
                     if self.public_url else "")
            self._heartbeat_locked(
                "detected browser closing — the server and sweep are "
                f"still running.{where}")

# ═══════════════════════════════════════════════════════════════════
#  Data loading — read all flow_*.csv files in the dir
# ═══════════════════════════════════════════════════════════════════
def _meta_for_flow_csv(path):
    """Best-effort read of the Tier-1 meta sidecar next to a flow CSV.

    flow/flow_<tag>.csv -> flow/meta_<tag>.json (mirrors ds4_search/
    sweep_runner.py:_sidecar_path). The sidecar carries the realised graph
    properties the flow CSV does not — notably k_avg (measured mean degree),
    k_min, k_max — so the dashboard can show measured-vs-target degree without
    re-reading the graph. Returns {} when the sidecar is missing or unreadable
    (older runs, a cell still in flight, a partial write): the caller treats an
    absent measured-k as "—", so this never blocks a row from rendering.
    """
    d = os.path.dirname(path)
    base = os.path.basename(path)
    if base.startswith("flow_"):
        base = "meta_" + base[len("flow_"):]
    meta_path = os.path.join(d, base.rsplit(".", 1)[0] + ".json")
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_all_data(dir_):
    """Scan flow_*.csv in dir, return {cells: [...], toruses: [...]}.

    Each record contains the parameter fields from the filename, the
    flow time series (t, d, in_window arrays), and per-cell summary
    stats. This is the structure that drives the dashboard plot,
    table, leaderboard, and consolidated CSV export.

    Cells are additionally annotated from their meta sidecar with the
    *measured* mean degree `k_measured` (the realised k̂ of the grown
    graph), `k_min`/`k_max`, and `k_err = k_measured - k` (target). That
    deviation is the calibration/thermalisation health signal: |k_err|
    far from 0 means the graph the d_s curve was measured on did not hit
    the degree it was supposed to, so the curve belongs to a different
    point in (k, …) space than its label claims.
    """
    cells, toruses = [], []
    # Match ds4_search/static_dashboard.py: prefer flow/ subdir, fall back to top-level
    # for legacy data layouts. ds4_search/sweep_runner.py writes to flow/ by default,
    # so a top-level-only glob silently misses everything.
    patterns = [os.path.join(dir_, "flow", "flow_*.csv"),
                os.path.join(dir_, "flow_*.csv")]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    for path in paths:
        name = os.path.basename(path)
        kind, fields = parse_filename(name)
        if kind is None:
            continue
        try:
            t, d, _se, in_w = load_flow_csv(path)
        except Exception:
            continue
        # Compute per-cell summary stats. For toruses, shape isn't
        # meaningful — they're calibrated references, not basin cells.
        stats = compute_stats(np.array(t), np.array(d),
                              np.array(in_w, dtype=bool))
        rec = {
            **fields,
            "kind": kind,
            "csv": name,
            "t": [None if not math.isfinite(x) else float(x) for x in t],
            "d": [None if not math.isfinite(x) else float(x) for x in d],
            "in_window": [bool(x) for x in in_w],
            **{k: (None if (isinstance(v, float) and not math.isfinite(v))
                   else v) for k, v in stats.items()},
        }
        if kind == "cell":
            rec["shape"] = classify_shape(np.array(d),
                                          np.array(in_w, dtype=bool))
            # Annotate measured degree from the meta sidecar (best-effort).
            meta = _meta_for_flow_csv(path)
            k_meas = meta.get("k_avg")
            if isinstance(k_meas, (int, float)) and math.isfinite(k_meas):
                rec["k_measured"] = float(k_meas)
                target = fields.get("k")
                if isinstance(target, (int, float)):
                    rec["k_err"] = float(k_meas) - float(target)
                km_min, km_max = meta.get("k_min"), meta.get("k_max")
                if isinstance(km_min, (int, float)):
                    rec["k_min"] = int(km_min)
                if isinstance(km_max, (int, float)):
                    rec["k_max"] = int(km_max)
            cells.append(rec)
        else:
            toruses.append(rec)
    return {"cells": cells, "toruses": toruses}


def compute_data_hash(data):
    """SHA-256 prefix of the full dataset.

    Hashing strategy: we hash the (sorted) parameter tuples of all
    cells + toruses, plus the in-window range and ds_median for
    each. This makes the hash flip when any cell completes (new
    parameter tuple appears) AND when an existing cell's data
    changes (rare, but happens after --force re-runs).

    We deliberately avoid hashing the full t/d arrays — too noisy
    for visible-version-pill purposes, and a single recomputed
    point would flip it without any meaningful change.
    """
    h = hashlib.sha256()
    keys = []
    for c in data.get("cells", []):
        key = (c.get("k"), c.get("T"), c.get("lb"), c.get("N"),
               c.get("seed"),
               round(c.get("ds_median") or 0, 4),
               round(c.get("flatness_std") or 0, 4),
               # measured degree lands in the meta sidecar, which may be
               # written a beat after the flow CSV — fold it in so the table
               # re-renders once k̂ becomes available, not only on the next cell.
               round(c.get("k_measured") or 0, 3))
        keys.append(("cell",) + key)
    for t in data.get("toruses", []):
        key = (t.get("torus_dim"), t.get("torus_L"), t.get("seed"),
               round(t.get("ds_median") or 0, 4))
        keys.append(("torus",) + key)
    for k in sorted(keys, key=str):
        h.update(json.dumps(k, default=str).encode())
    return h.hexdigest()[:8]


def write_consolidated_csv(data, out_path):
    """Long-format CSV: one row per (cell, t_index).

    Columns: kind, k, T, lb, N, seed, torus_dim, torus_L, shape,
             t_idx, t, d_s_mean, in_window

    This is the single study-level CSV the user asked for. It's
    additional to the existing per-cell flow_*.csv files (which
    are kept as input — this consolidates them).

    Atomic write via .tmp + rename so a concurrent reader never
    sees a half-written file.
    """
    fields = ["kind", "k", "T", "lb", "N", "seed", "torus_dim",
              "torus_L", "shape", "k_measured", "k_err",
              "t_idx", "t", "d_s_mean", "in_window"]
    tmp = out_path + ".tmp"
    n_rows = 0
    with open(tmp, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cell in data.get("cells", []):
            base = {
                "kind": "cell",
                "k": cell.get("k", ""),
                "T": cell.get("T", ""),
                "lb": cell.get("lb", ""),
                "N": cell.get("N", ""),
                "seed": cell.get("seed", ""),
                "torus_dim": "", "torus_L": "",
                "shape": cell.get("shape", ""),
                "k_measured": ("" if cell.get("k_measured") is None
                               else round(cell["k_measured"], 4)),
                "k_err": ("" if cell.get("k_err") is None
                          else round(cell["k_err"], 4)),
            }
            ts = cell.get("t", []) or []
            ds = cell.get("d", []) or []
            ws = cell.get("in_window", []) or []
            for i, (tt, dd, ww) in enumerate(zip(ts, ds, ws)):
                row = dict(base)
                row.update({"t_idx": i, "t": tt,
                            "d_s_mean": dd, "in_window": int(bool(ww))})
                w.writerow(row); n_rows += 1
        for tr in data.get("toruses", []):
            base = {
                "kind": "torus",
                "k": "", "T": "", "lb": "", "N": "",
                "seed": tr.get("seed", ""),
                "torus_dim": tr.get("torus_dim", ""),
                "torus_L": tr.get("torus_L", ""),
                "shape": "torus",
            }
            ts = tr.get("t", []) or []
            ds = tr.get("d", []) or []
            ws = tr.get("in_window", []) or []
            for i, (tt, dd, ww) in enumerate(zip(ts, ds, ws)):
                row = dict(base)
                row.update({"t_idx": i, "t": tt,
                            "d_s_mean": dd, "in_window": int(bool(ww))})
                w.writerow(row); n_rows += 1
    os.replace(tmp, out_path)
    return n_rows


# ═══════════════════════════════════════════════════════════════════
#  HTTP handler
# ═══════════════════════════════════════════════════════════════════
def make_handler(state, html_template, base_path=""):
    """Return a request handler class closed over (state, html, base_path).

    `base_path` is an optional URL prefix baked into the served page so JS
    fetches and links resolve correctly if the app is ever mounted under a
    sub-path. Standalone (the default), it's empty and handlers see paths
    like `/api/state`.
    """

    class AppHandler(http.server.BaseHTTPRequestHandler):
        # Quiet down the per-request noise (we have heartbeat lines
        # for what matters)
        def log_message(self, fmt, *args):
            pass

        def _json(self, payload, code=200):
            try:
                body = json.dumps(payload, default=str).encode()
            except (TypeError, ValueError) as e:
                body = json.dumps({"error": str(e)}).encode()
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            self.wfile.write(body)

        def _html(self, html):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # ── GET ───────────────────────────────────────────────────
        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            try:
                if path in ("/", "/index.html"):
                    # Bake the URL base path into the HTML at serve time.
                    html = html_template.replace("__BASE_PATH__", base_path)
                    self._html(html)
                elif path == "/api/state":
                    self._handle_state()
                elif path == "/api/data":
                    self._handle_data()
                elif path == "/api/heartbeat":
                    self._handle_heartbeat()
                elif path == "/api/csv":
                    self._handle_csv()
                else:
                    self.send_error(404)
            except CLIENT_DISCONNECT_ERRORS:
                # The browser tab closed or navigated away mid-response.
                # The socket is already gone, so do NOT try to send a 500
                # here — that write would just raise again and produce a
                # confusing double traceback. Note it quietly and return.
                state.note_client_disconnect()
            except Exception as e:
                # A genuine server-side error: try to report it as JSON,
                # but if the client has *also* disconnected by now, treat
                # that secondary failure as the same harmless case.
                try:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                except CLIENT_DISCONNECT_ERRORS:
                    state.note_client_disconnect()

        def _handle_state(self):
            running = state.is_sweep_running()
            sweep_info = {}
            try:
                sj = os.path.join(state.args.dir,
                                  "lb_sweep_status.json")
                with open(sj) as f:
                    sweep_info = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
            self._json({
                "sweep_running": running,
                "sweep_pid": (state.sweep_proc.pid
                              if running else None),
                "sweep_started_at": state.sweep_started_at,
                "sweep_status": sweep_info,
                "build": {
                    "app_started_at": state.startup_iso,
                    "app_hash": state._app_hash,
                },
                "args": {
                    "dir": state.args.dir,
                    "k": state.args.k,
                    "T": state.args.T,
                    "lb": state.args.lb,
                    "N": state.args.N,
                    "seeds": (state.args.seeds
                              if getattr(state.args, "seeds", None)
                              else str(state.args.seed)),
                    "torus_d": state.args.torus_d,
                },
            })

        def _handle_data(self):
            data = load_all_data(state.args.dir)
            data["data_hash"] = compute_data_hash(data)
            data["n_cells"] = len(data["cells"])
            data["n_toruses"] = len(data["toruses"])
            self._json(data)

        def _handle_heartbeat(self):
            with state.lock:
                lines = list(state.heartbeat)
                total = state.heartbeat_total
            self._json({"lines": lines, "total": total,
                        "max": state.HEARTBEAT_MAXLEN})

        def _handle_csv(self):
            data = load_all_data(state.args.dir)
            out_path = os.path.join(state.args.dir, "lb_study.csv")
            n = write_consolidated_csv(data, out_path)
            self._json({"path": out_path, "rows": n,
                        "n_cells": len(data["cells"]),
                        "n_toruses": len(data["toruses"])})

    return AppHandler


# ═══════════════════════════════════════════════════════════════════
#  HTML template (single-page app)
# ═══════════════════════════════════════════════════════════════════
from ds4_search.live_app_page import HTML_TEMPLATE


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Live reporting dashboard for the d_s sweep. Normally "
                    "started via `python main.py`; serves a read-only "
                    "page and auto-starts the sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for details.",
    )
    ap.add_argument("--dir", default=".", help="Data directory")
    ap.add_argument("--port", type=int, default=1234,
                    help="HTTP port to bind (default 1234, 0 = OS-assigned)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="HTTP bind address. Default 127.0.0.1 means "
                         "localhost-only — no other machine on your "
                         "network can reach it. Use 0.0.0.0 to expose "
                         "on all interfaces (e.g. for LAN access). "
                         "There's no real authentication beyond a "
                         "per-launch CSRF token, so only bind 0.0.0.0 "
                         "on networks you trust.")
    # Grid args — optional pass-through to the spawned sweep worker. When
    # omitted (the normal launch from main.py), the sweep reads the grid from
    # main.toml itself, so these need not be specified here.
    ap.add_argument("--k", default=None,
                    help="Comma-separated k values, e.g. '7,8,9,10,11' "
                         "(omitted → main.toml [grid].k)")
    ap.add_argument("--T", default=None,
                    help="Comma-separated T values, e.g. '0,0.005' "
                         "(omitted → main.toml [grid].T)")
    ap.add_argument("--lb", default=None,
                    help="Comma-separated lb values (omitted → main.toml [grid].lb)")
    ap.add_argument("--N", default=None,
                    help="Comma-separated N values (omitted → main.toml [grid].N)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Single base seed (back-compat). Ignored when "
                         "--seeds is given; omitted → main.toml [grid] seeds.")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds to run per cell "
                         "(e.g. '42,43,44,45,46'). Forwarded to the sweep; "
                         "overrides --seed.")
    ap.add_argument("--torus-d", default="2,3,4,5",
                    help="Comma-separated torus dimensions for "
                         "calibration anchors (default 2,3,4,5)")
    ap.add_argument("--watch-reload", action="store_true",
                    help="Enable hot-reload on the spawned sweep")
    ap.add_argument("--auto-start", action="store_true",
                    help="Start the sweep immediately on launch (main.py "
                         "always passes this)")
    args = ap.parse_args(argv)
    args.dir = os.path.abspath(args.dir)

    if not os.path.isdir(args.dir):
        print(f"[ERROR] data dir does not exist: {args.dir}",
              file=sys.stderr)
        sys.exit(1)

    state = AppState(args)
    handler_cls = make_handler(state, HTML_TEMPLATE)

    # Tiny subclass so allow_reuse_address is in effect BEFORE bind() —
    # setting it on the instance after construction is too late, the
    # bind has already happened in __init__. With this, a port that
    # was held in TIME_WAIT after a previous crash becomes reusable
    # immediately instead of after the kernel's ~60s grace window.
    class _ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True   # don't let a lingering client thread block Ctrl-C

        def handle_error(self, request, client_address):
            # The default ThreadingTCPServer.handle_error prints the full
            # "Exception occurred during processing of request…" traceback
            # to stderr. For a client that simply went away mid-response
            # that's noise that reads like a crash — swallow it (it's
            # already noted in the heartbeat log via note_client_disconnect).
            exc = sys.exc_info()[1]
            if isinstance(exc, CLIENT_DISCONNECT_ERRORS):
                return
            super().handle_error(request, client_address)

    try:
        server = _ReusableTCPServer((args.host, args.port), handler_cls)
    except OSError as e:
        if e.errno == 98:  # EADDRINUSE — another process is actually
                           # listening (not just TIME_WAIT)
            print(f"\n[ERROR] port {args.port} is held by another process.",
                  file=sys.stderr)
            print(f"  to find it:   lsof -i :{args.port}",
                  file=sys.stderr)
            print(f"  to free it:   fuser -k {args.port}/tcp",
                  file=sys.stderr)
            print(f"  or pick another port: set  port = {args.port + 1}  "
                  f"under [server] in main.toml", file=sys.stderr)
            print("  (or  port = 0  to let the OS pick any free port "
                  "automatically — the chosen URL is printed at startup)",
                  file=sys.stderr)
            sys.exit(98)
        raise

    # When --port 0 is used the kernel picks a port; read it back from
    # the actual bound socket so portal mount registration knows the
    # right number.
    bound_port = server.server_address[1]
    public_url = f"http://{args.host}:{bound_port}/"
    # Hand the resolved URL to AppState so the "browser closed" log line
    # can tell the user exactly where to reopen the dashboard.
    state.set_public_url(public_url)

    print(f"  serving on {public_url}")
    print(f"  data dir: {args.dir}")
    print(f"  open in browser: {public_url}")
    if args.host in ("127.0.0.1", "localhost"):
        print("  binding: localhost-only (other machines cannot reach)")
    else:
        print(f"  ⚠ binding: {args.host} — reachable from your network "
              f"(read-only reporting page; no authentication).")
    print("  Ctrl-C to stop the server and the sweep")

    if args.auto_start:
        print("  --auto-start: starting sweep now")
        state.start_sweep()

    # Funnel SIGTERM (e.g. `kill <pid>`, a service manager, an IDE stop
    # button) into the same shutdown path as Ctrl-C, so the sweep group is
    # always torn down via stop_sweep() rather than us dying and leaving
    # the workers orphaned. The handler runs in the main thread and raises
    # KeyboardInterrupt, which unblocks serve_forever() into the cleanup
    # below — identical to the SIGINT/Ctrl-C path.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down…")
        if state.is_sweep_running():
            print(f"  stopping sweep PID {state.sweep_proc.pid}")
            state.stop_sweep(timeout=10)
        server.shutdown()
        print("  exited.")


if __name__ == "__main__":
    main()
