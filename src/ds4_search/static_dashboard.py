"""
ds4_search/static_dashboard.py — Interactive dashboard for d_s(t) flow analysis
====================================================================
Reframes the spectral-dimension search around two questions:
    1. Is d_s(t) locally flat across some t window? — the dimension
       at that scale is well-defined only if so.
    2. Does that same flat region persist across N? — only then is
       it asymptotic, not a finite-size artifact.

A globally averaged d_s value (the SLQ fit slope) is meaningless for
graphs whose d_s(t) ramps or oscillates. The flatness + N-stability
framing avoids that vulnerability. The dashboard surfaces flatness
metrics directly, lets you filter by curve shape, and overlays
multiple cells so collapse-or-divergence-across-N is visible at a
glance.

Architecture:
    1 script, 1 stats CSV (append-only), JSON-in-HTML for the
    dashboard. The CSV is grep-able persistent storage; the HTML
    embeds curve data as JSON so the JS dashboard is self-contained.

Workflow
--------
    # Mid-sweep peek (or post-sweep aggregate)
    python main.py dashboard --dir /data2/26

    # Custom output paths
    python main.py dashboard --dir /data2/26 \\
        --stats-csv lb_stats.csv --out lb_dash.html

    # Force rescan of all flow CSVs (re-derive stats from scratch)
    python main.py dashboard --dir /data2/26 --rescan

    # Quietly: just append new cells, no console summary
    python main.py dashboard --dir /data2/26 --quiet

The script is idempotent — safe to re-run any time. New flow CSVs
get their stats computed and appended to the stats CSV; cells
already in the stats CSV are skipped (unless --rescan).
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  Filename parsers
# ═══════════════════════════════════════════════════════════════════
RE_CELL = re.compile(
    r"^flow_k(\d+)_T([\d.eE+-]+)_lb([\d.eE+-]+)_N(\d+)_s(\d+)\.csv$")
RE_TORUS = re.compile(
    r"^flow_torus(\d+)d_L(\d+)_s(\d+)\.csv$")


# ═══════════════════════════════════════════════════════════════════
#  Per-cell stats
# ═══════════════════════════════════════════════════════════════════
STATS_HEADER = [
    "kind",         # "cell" or "torus"
    "k", "T", "lb", "N", "seed",
    "torus_dim", "torus_L",
    "shape",
    "n_in_window",
    "ds_median", "ds_mean",
    "ds_p10", "ds_p25", "ds_p75", "ds_p90",
    "ds_min", "ds_max",
    "flatness_std", "flatness_range", "flatness_iqr",
    "ds_at_t_min", "ds_at_t_max",
    "t_min_in_window", "t_max_in_window",
    "csv_path",
]


def classify_shape(d_s_mean, in_window):
    """Heuristic shape tag — must stay in sync with ds4_search.sweep_runner.

    Note on 'flat': the tag means BOTH (a) no significant local
    extrema AND (b) total variation small enough that the curve
    actually sits at a single dimension. Without (b), a curve that
    smoothly meanders between d=3 and d=5 would be mislabelled as
    'flat' just because it has no peaks. We use std-of-d_s < 0.7
    as the (b) criterion, matching the threshold below which
    'flatness_std' starts to look genuinely meaningful in the
    dashboard's leaderboard.
    """
    valid = in_window & np.isfinite(d_s_mean)
    if valid.sum() < 8:
        return "undetermined"
    y = d_s_mean[valid].copy()
    y_smooth = (np.convolve(y, np.ones(3) / 3, mode="valid")
                if len(y) >= 5 else y)
    n = len(y_smooth)
    peaks_idx, troughs_idx = [], []
    for i in range(1, n - 1):
        if y_smooth[i] > y_smooth[i - 1] and y_smooth[i] > y_smooth[i + 1]:
            peaks_idx.append(i)
        if y_smooth[i] < y_smooth[i - 1] and y_smooth[i] < y_smooth[i + 1]:
            troughs_idx.append(i)
    span = float(y_smooth.max() - y_smooth.min())
    std_y = float(np.std(y))
    if span < 0.5 and std_y < 0.4:
        return "flat"

    def amp(idx):
        l = max(idx - 1, 0); r = min(idx + 1, n - 1)
        return min(abs(y_smooth[idx] - y_smooth[l]),
                   abs(y_smooth[idx] - y_smooth[r]))
    peaks = [i for i in peaks_idx if amp(i) >= 0.15]
    troughs = [i for i in troughs_idx if amp(i) >= 0.15]
    n_peaks = len(peaks); n_troughs = len(troughs)
    net = float(y_smooth[-1] - y_smooth[0])
    if n_peaks == 0 and n_troughs == 0:
        if net >= 0.8: return "monotone_rising"
        if net <= -0.8: return "monotone_falling"
        # Featureless but non-zero variation: only call it "flat"
        # if the std is genuinely small. Otherwise it's "wobble" —
        # a meandering curve with no sharp features but real drift.
        if std_y < 0.7:
            return "flat"
        return "wobble"
    if n_peaks == 1 and n_troughs <= 1:
        return "single_peak"
    if n_peaks == 2:
        return "double_peak"
    return "bumpy"


def compute_stats(t, d_s_mean, in_window):
    """Per-cell summary metrics. Flatness measured on in-window region.

    `flatness_std` is the standard deviation of d_s across the in-window
    region — the headline metric. A value below ~0.3 means d_s varies
    by less than ~0.3 across all scales in the resolved window, which
    is the only situation where talking about "the spectral dimension"
    makes sense.

    `flatness_iqr` is the inter-quartile range — robust to a single
    bad point, so a curve that is mostly flat with one outlier still
    scores well on this metric.
    """
    valid = in_window & np.isfinite(d_s_mean)
    out = {col: float("nan") for col in STATS_HEADER
           if col not in ("kind", "k", "T", "lb", "N", "seed",
                          "torus_dim", "torus_L", "shape", "csv_path")}
    out["n_in_window"] = int(valid.sum())
    if valid.sum() < 4:
        return out
    d = d_s_mean[valid]
    t_w = t[valid]
    out["ds_median"] = float(np.median(d))
    out["ds_mean"] = float(d.mean())
    out["ds_p10"] = float(np.percentile(d, 10))
    out["ds_p25"] = float(np.percentile(d, 25))
    out["ds_p75"] = float(np.percentile(d, 75))
    out["ds_p90"] = float(np.percentile(d, 90))
    out["ds_min"] = float(d.min())
    out["ds_max"] = float(d.max())
    out["flatness_std"] = float(d.std())
    out["flatness_range"] = float(d.max() - d.min())
    out["flatness_iqr"] = float(out["ds_p75"] - out["ds_p25"])
    out["ds_at_t_min"] = float(d[0])
    out["ds_at_t_max"] = float(d[-1])
    out["t_min_in_window"] = float(t_w[0])
    out["t_max_in_window"] = float(t_w[-1])
    return out


def load_flow_csv(path):
    """Return (t, d_s_mean, d_s_se, in_window) arrays."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    t = np.array([float(r["t"]) for r in rows])
    d_mean = np.array([float(r["d_s_mean"]) for r in rows])
    d_se = np.array([float(r["d_s_se"]) for r in rows])
    in_w = np.array([bool(int(r["in_window"])) for r in rows])
    return t, d_mean, d_se, in_w


def parse_filename(name):
    """Return (kind, fields_dict) or (None, None) if not a flow CSV."""
    m = RE_CELL.match(name)
    if m:
        return "cell", {
            "k": int(m.group(1)),
            "T": float(m.group(2)),
            "lb": float(m.group(3)),
            "N": int(m.group(4)),
            "seed": int(m.group(5)),
            "torus_dim": "", "torus_L": "",
        }
    m = RE_TORUS.match(name)
    if m:
        d = int(m.group(1)); L = int(m.group(2)); seed = int(m.group(3))
        return "torus", {
            "k": "", "T": "", "lb": "",
            "N": L ** d, "seed": seed,
            "torus_dim": d, "torus_L": L,
        }
    return None, None


# ═══════════════════════════════════════════════════════════════════
#  Grid expansion + sweep auto-spawn
# ═══════════════════════════════════════════════════════════════════
def parse_grid_args(args):
    """Convert --k/--T/--lb/--N CSV strings into the cartesian product
    of (k, T, lb, N) tuples that constitute the expected cell list.
    Returns [] when any of the four args is missing — partial grids are
    treated as 'no expected total specified'.
    """
    if not all([args.k, args.T, args.lb, args.N]):
        return []
    k_vals = [int(x) for x in args.k.split(",") if x.strip()]
    T_vals = [float(x) for x in args.T.split(",") if x.strip()]
    lb_vals = [float(x) for x in args.lb.split(",") if x.strip()]
    N_vals = [int(x) for x in args.N.split(",") if x.strip()]
    return [(k, T, lb, N)
            for k in k_vals for T in T_vals
            for lb in lb_vals for N in N_vals]


def expected_csv_path(dir_, k, T, lb, N, seed):
    """Filename a finished cell would have. Match must be exact for
    the idempotent skip in ds4_search/sweep_runner.py to pick up an existing run.
    """
    return os.path.join(dir_,
        f"flow_k{k}_T{T}_lb{lb}_N{N}_s{seed}.csv")


def split_done_missing(expected_cells, dir_, seed):
    """Partition the expected list by whether their CSV exists on disk."""
    done, missing = [], []
    for (k, T, lb, N) in expected_cells:
        if os.path.exists(expected_csv_path(dir_, k, T, lb, N, seed)):
            done.append((k, T, lb, N))
        else:
            missing.append((k, T, lb, N))
    return done, missing


def is_sweep_running(status_json_path, max_age_sec=120):
    """Return True if the status JSON indicates an active sweep.

    Uses two checks: phase=='running' AND the 'now' timestamp in the
    JSON is within max_age_sec of the current time. A stale-but-
    running file (e.g. from a sweep that crashed without writing
    phase=done) is treated as not running, so we can re-spawn.
    """
    if not os.path.exists(status_json_path):
        return False
    try:
        with open(status_json_path) as f:
            s = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if s.get("phase") != "running":
        return False
    now_in_status = s.get("now") or 0
    return (time.time() - now_in_status) < max_age_sec


def spawn_sweep(missing_cells, args):
    """Spawn ds4_search/sweep_runner.py for the missing cells. Returns (proc, cmd).

    Foreground by default: stdout/stderr inherit from the dashboard's
    terminal, and the sweep is in the same process group, so Ctrl-C
    in the terminal exits both cleanly. The dashboard's main() will
    proc.wait() on the child after rendering the HTML — open the
    dashboard in a browser tab while the terminal shows live output.

    With --detach: stdout/stderr go to args.sweep_log, the child
    starts a new session, and the dashboard returns immediately.
    Sweep runs autonomously; needs explicit kill to stop.
    """
    import subprocess
    ks = sorted({c[0] for c in missing_cells})
    Ts = sorted({c[1] for c in missing_cells})
    lbs = sorted({c[2] for c in missing_cells})
    Ns = sorted({c[3] for c in missing_cells})
    main_py = str(Path(__file__).resolve().parents[2] / "main.py")
    cmd = [
        sys.executable, main_py, "__sweep__",
        "--k", ",".join(str(x) for x in ks),
        "--T", ",".join(str(x) for x in Ts),
        "--lb", ",".join(str(x) for x in lbs),
        "--N", ",".join(str(x) for x in Ns),
        "--seed", str(args.seed),
        "--status-json", args.status_json,
        "--torus-d", args.torus_d,
    ]
    if getattr(args, "watch_reload", False):
        cmd.append("--watch-reload")
    if args.detach:
        # Detached mode: log file, new session, no Ctrl-C propagation.
        log_f = open(args.sweep_log, "a")
        log_f.write(f"\n\n=== sweep spawned (detached) at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    f"cmd: {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=log_f,
                                stderr=subprocess.STDOUT,
                                cwd=args.dir, start_new_session=True)
    else:
        # Attached mode: inherit terminal stdio, same process group.
        # Ctrl-C in the dashboard terminal will SIGINT the sweep too.
        proc = subprocess.Popen(cmd, cwd=args.dir)
    return proc, cmd


# ═══════════════════════════════════════════════════════════════════
#  Stats CSV: read / append
# ═══════════════════════════════════════════════════════════════════
def cell_key(row):
    """Uniqueness key for a stats row (so we can dedupe on re-scan)."""
    return (row.get("kind"), str(row.get("k")), str(row.get("T")),
            str(row.get("lb")), str(row.get("N")), str(row.get("seed")),
            str(row.get("torus_dim")), str(row.get("torus_L")))


def load_stats_csv(path):
    """Return list of dicts (or [] if file absent)."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def append_stats_csv(path, new_rows):
    """Append rows; create file with header if it doesn't exist."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_HEADER)
        if not exists:
            w.writeheader()
        for row in new_rows:
            # Coerce all values to strings for safe CSV writing.
            w.writerow({k: ("" if row.get(k) is None
                            else str(row[k]))
                        for k in STATS_HEADER})


# ═══════════════════════════════════════════════════════════════════
#  HTML / JS dashboard
# ═══════════════════════════════════════════════════════════════════
from ds4_search.dashboard_page import HTML_TEMPLATE


def _json_safe(obj):
    """Convert numpy types and NaN/Inf to JSON-safe forms."""
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f): return None
        return f
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"can't JSON-encode {type(obj)}")


def render_dashboard(stats_rows, dir_, out_path, status_json_path,
                     progress_info=None):
    """Build the HTML dashboard from stats rows + the original CSVs.

    progress_info is an optional dict with grid expected/done counts.
    When present, the dashboard shows X/Y prominently in the status
    line. When None, no expected total is shown — pure exploration
    mode for ad-hoc dirs.
    """
    # We need the curve data for plotting, so for each stats row we load
    # the corresponding flow CSV. Stats rows store the path.
    cells_data, torus_data = [], []
    n_dropped_missing = 0
    n_dropped_load_err = 0
    sample_missing = []
    for row in stats_rows:
        csv_path = row.get("csv_path") or ""
        if not csv_path or not os.path.exists(csv_path):
            # Resolve against the scan dir, trying the relative path as
            # stored, then the bare basename, then the flow/ subdir
            # (covers legacy stats CSVs that stored only a basename).
            base = os.path.basename(csv_path)
            candidates = [
                os.path.join(dir_, csv_path),
                os.path.join(dir_, base),
                os.path.join(dir_, "flow", base),
            ]
            csv_path = next((c for c in candidates if os.path.exists(c)), "")
            if not csv_path:
                n_dropped_missing += 1
                if len(sample_missing) < 3:
                    sample_missing.append(row.get("csv_path") or "<empty>")
                continue
        try:
            t, d, _, in_w = load_flow_csv(csv_path)
        except Exception:
            n_dropped_load_err += 1
            continue
        rec = {
            "t": [None if not math.isfinite(x) else float(x) for x in t],
            "d": [None if not math.isfinite(x) else float(x) for x in d],
            "in_window": [bool(x) for x in in_w],
        }
        if row.get("kind") == "torus":
            rec["torus_dim"] = int(row["torus_dim"]) if row.get("torus_dim") else None
            rec["torus_L"] = int(row["torus_L"]) if row.get("torus_L") else None
            rec["seed"] = int(row["seed"]) if row.get("seed") else None
            rec["N"] = int(row["N"]) if row.get("N") else None
            torus_data.append(rec)
        else:
            for k in ("k", "N", "seed"):
                rec[k] = int(row[k]) if row.get(k) else None
            for k in ("T", "lb"):
                rec[k] = float(row[k]) if row.get(k) else None
            for k in ("ds_median", "ds_mean", "ds_p10", "ds_p25",
                     "ds_p75", "ds_p90", "ds_min", "ds_max",
                     "flatness_std", "flatness_range", "flatness_iqr",
                     "ds_at_t_min", "ds_at_t_max"):
                v = row.get(k)
                rec[k] = float(v) if v not in ("", None, "nan") else None
            rec["shape"] = row.get("shape", "undetermined")
            cells_data.append(rec)

    payload = {"cells": cells_data, "toruses": torus_data}
    json_str = json.dumps(payload, default=_json_safe, allow_nan=False)
    progress_str = json.dumps(progress_info or {}, default=_json_safe,
                              allow_nan=False)
    # Build identifier: timestamp + short hash of this script's source.
    # Lets the user verify at-a-glance which version of the dashboard
    # rendered the HTML they're looking at — useful when iterating on
    # the script while watching the dashboard.
    try:
        import hashlib
        with open(__file__, "rb") as f:
            script_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    except OSError:
        script_hash = "unknown"
    build_info = {
        "rendered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_hash": script_hash,
        "script_path": os.path.basename(__file__),
    }
    build_str = json.dumps(build_info, default=_json_safe, allow_nan=False)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json_str)
    html = html.replace("__STATUS_PATH__", json.dumps(status_json_path))
    html = html.replace("__PROGRESS_JSON__", progress_str)
    html = html.replace("__BUILD_JSON__", build_str)
    with open(out_path, "w") as f:
        f.write(html)
    return {
        "n_rendered_cells": len(cells_data),
        "n_rendered_torus": len(torus_data),
        "n_dropped_missing": n_dropped_missing,
        "n_dropped_load_err": n_dropped_load_err,
        "sample_missing_paths": sample_missing,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main: scan, append, render
# ═══════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=str, default=".",
                    help="Directory to scan for flow_*.csv (default: cwd)")
    ap.add_argument("--stats-csv", type=str, default="lb_dashboard_stats.csv",
                    help="Per-cell stats CSV (default lb_dashboard_stats.csv "
                         "in cwd; appended to across runs)")
    ap.add_argument("--out", type=str, default="lb_dashboard.html",
                    help="Dashboard HTML path (default lb_dashboard.html)")
    ap.add_argument("--status-json", type=str,
                    default="lb_sweep_status.json",
                    help="Path the dashboard polls for live sweep status. "
                         "Should match --status-json on ds4_search/sweep_runner.py. "
                         "If served from the same HTTP root as the HTML, "
                         "a relative filename works.")
    ap.add_argument("--rescan", action="store_true",
                    help="Discard existing stats CSV and rebuild from scratch")
    ap.add_argument("--quiet", action="store_true")

    # ── Grid args (optional). When given, the dashboard knows which
    # cells are "expected" and can show progress as X/Y. With
    # --auto-sweep, it spawns ds4_search/sweep_runner.py as a background process
    # to fill in any missing cells.
    ap.add_argument("--k", type=str, default=None,
                    help="Grid k values, comma-separated (e.g. '7,8,9,10,11'). "
                         "When given alongside --T, --lb, --N, the dashboard "
                         "computes total expected cells for the X/Y display.")
    ap.add_argument("--T", type=str, default=None,
                    help="Grid T values, comma-separated (e.g. '0,0.005')")
    ap.add_argument("--lb", type=str, default=None,
                    help="Grid lb values, comma-separated")
    ap.add_argument("--N", type=str, default=None,
                    help="Grid N values, comma-separated")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the sweep (default 42)")
    ap.add_argument("--auto-sweep", action="store_true",
                    help="If grid args are given and cells are missing, "
                         "spawn ds4_search/sweep_runner.py as a background subprocess "
                         "and return immediately. The dashboard's status "
                         "banner then tracks live progress. Refuses to "
                         "spawn if a sweep appears to already be running "
                         "(detected via recent status JSON updates).")
    ap.add_argument("--sweep-log", type=str, default="lb_sweep.log",
                    help="Log file for the spawned sweep's stdout/stderr "
                         "when running in --detach mode. Ignored in the "
                         "default attached mode (stdio inherits terminal).")
    ap.add_argument("--detach", action="store_true",
                    help="Run the spawned sweep as a background daemon "
                         "(new session, log file). Without this flag "
                         "(default), the sweep is a foreground child: "
                         "stdout/stderr appear in the dashboard's "
                         "terminal, and Ctrl-C exits both cleanly.")
    ap.add_argument("--torus-d", type=str, default="4",
                    help="Torus dimension(s) for reference, comma-"
                         "separated (e.g. '2,3,4,5'). The torus refs "
                         "compute FIRST so they appear in the dashboard "
                         "as soon as you open it, anchoring the basin "
                         "curves against known-flat reference geometries. "
                         "Default '4' is one 4D torus per N value.")
    ap.add_argument("--watch-reload", action="store_true",
                    help="Pass --watch-reload to spawned ds4_search/sweep_runner.py: "
                         "the sweep will hot-restart between cells when "
                         "any source file under src/ changes (cached cells "
                         "skip on restart, so no re-work). Useful for "
                         "live tuning of shape thresholds and status "
                         "metrics during long sweeps.")
    args = ap.parse_args(argv)

    # 1. Load existing stats (or skip if --rescan)
    if args.rescan and os.path.exists(args.stats_csv):
        os.remove(args.stats_csv)
    existing = load_stats_csv(args.stats_csv)
    seen_keys = {cell_key(r) for r in existing}
    if not args.quiet:
        print(f"  stats CSV: {args.stats_csv}")
        print(f"    existing rows: {len(existing)}")

    # 2. Scan directory for flow CSVs.
    # Default location is `<dir>/flow/flow_*.csv`. Legacy top-level
    # `<dir>/flow_*.csv` files are also picked up so the dashboard
    # keeps working through a migration; once all CSVs have been
    # moved into flow/, the top-level glob simply matches nothing.
    patterns = [os.path.join(args.dir, "flow", "flow_*.csv"),
                os.path.join(args.dir, "flow_*.csv")]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    if not args.quiet:
        print(f"  scanning flow/ + top-level: {len(paths)} file(s) found")

    # 3. Compute stats for any new cells
    new_rows = []
    for path in paths:
        name = os.path.basename(path)
        kind, fields = parse_filename(name)
        if kind is None:
            continue
        # Check key against existing
        probe_row = {"kind": kind, **fields}
        if cell_key(probe_row) in seen_keys and not args.rescan:
            continue
        try:
            t, d, _se, in_w = load_flow_csv(path)
        except Exception as e:
            if not args.quiet:
                print(f"    [skip] {name}: {e}", file=sys.stderr)
            continue
        stats = compute_stats(t, d, in_w)
        shape = classify_shape(d, in_w)
        # Store the path relative to the scan dir (keeping any flow/
        # subdir), not just the basename — so it resolves whether the
        # CSVs sit in {dir}/ or {dir}/flow/. The render-time fallback
        # below also tries the flow/ subdir for legacy basename-only
        # stats CSVs. Relative (not absolute) so the data dir can move.
        rel_path = os.path.relpath(path, args.dir)
        row = {"kind": kind, **fields, "shape": shape,
               "csv_path": rel_path, **stats}
        new_rows.append(row)

    if not args.quiet and new_rows:
        print(f"    new cells: {len(new_rows)} (will be appended)")
    if new_rows:
        append_stats_csv(args.stats_csv, new_rows)

    # 4. Compute expected vs done from grid args (if any)
    expected = parse_grid_args(args)
    progress_info = {"expected": len(expected), "done_in_grid": 0,
                     "missing_in_grid": 0, "loaded": 0,
                     "auto_sweep_spawned": False, "sweep_pid": None}
    spawned_proc = None
    if expected:
        done, missing = split_done_missing(expected, args.dir, args.seed)
        progress_info["done_in_grid"] = len(done)
        progress_info["missing_in_grid"] = len(missing)
        if not args.quiet:
            print(f"  grid: {len(expected)} expected cell(s), "
                  f"{len(done)} done, {len(missing)} missing")
        if args.auto_sweep and missing:
            if is_sweep_running(args.status_json):
                if not args.quiet:
                    print(f"  --auto-sweep: sweep already running "
                          f"(see {args.status_json}); skipping spawn")
            else:
                mode = ("detached → " + args.sweep_log) if args.detach \
                       else "attached"
                if not args.quiet:
                    print(f"  --auto-sweep: spawning ds4_search/sweep_runner.py for "
                          f"{len(missing)} missing cell(s) [{mode}]")
                proc, cmd = spawn_sweep(missing, args)
                spawned_proc = proc if not args.detach else None
                progress_info["auto_sweep_spawned"] = True
                progress_info["sweep_pid"] = proc.pid
                if not args.quiet:
                    if args.detach:
                        print(f"    spawned PID {proc.pid}; tail "
                              f"{args.sweep_log} to follow progress")
                    else:
                        print(f"    spawned PID {proc.pid}; sweep output "
                              f"will appear below. Ctrl-C to stop both.")

    # 5. Reload full stats CSV and render dashboard
    all_rows = load_stats_csv(args.stats_csv)
    progress_info["loaded"] = len([r for r in all_rows
                                   if r.get("kind") == "cell"])
    if not all_rows and not progress_info["auto_sweep_spawned"]:
        print(f"  no cells to render — exiting.", file=sys.stderr)
        return
    render_info = render_dashboard(all_rows, args.dir, args.out,
                                    args.status_json, progress_info)
    if not args.quiet:
        print(f"  → {args.out}")
        if all_rows:
            print(f"    {len(all_rows)} cell(s) total in stats CSV")
            cell_rows = [r for r in all_rows if r.get("kind") == "cell"]
            torus_rows = [r for r in all_rows if r.get("kind") == "torus"]
            print(f"    {len(cell_rows)} basin cell(s), "
                  f"{len(torus_rows)} torus reference(s)")

            # Diagnostics: if stats rows reference flow CSVs that
            # can't be found, that's the silent-empty-dashboard bug.
            # Surface it loudly.
            if render_info["n_dropped_missing"] > 0:
                print(f"")
                print(f"  ⚠ WARNING: {render_info['n_dropped_missing']} stats "
                      f"row(s) reference flow CSVs that don't exist on disk.")
                print(f"    The dashboard will render WITHOUT those cells.")
                print(f"    Sample missing paths:")
                for p in render_info["sample_missing_paths"]:
                    print(f"      {p}")
                print(f"")
                print(f"    Likely causes:")
                print(f"      1. The flow_*.csv files were deleted "
                      f"(check `ls {args.dir}/flow_*.csv`)")
                print(f"      2. The stats CSV is from a different "
                      f"working directory")
                print(f"      3. The data was moved")
                print(f"")
                print(f"    Fixes:")
                print(f"      • If files moved: re-run with --rescan to "
                      f"rebuild stats from current location")
                print(f"      • If files deleted: re-run sweep "
                      f"(ds4_search/sweep_runner.py with same args, or dashboard "
                      f"--auto-sweep) — cached cells will skip, missing "
                      f"ones will recompute")
                print(f"      • If sure paths are correct: "
                      f"`rm {args.stats_csv}` and re-run dashboard")
            if render_info["n_dropped_load_err"] > 0:
                print(f"  ⚠ {render_info['n_dropped_load_err']} flow CSV(s) "
                      f"could not be parsed (corrupt?). Inspect manually.")
            if render_info["n_rendered_cells"] == 0 and len(cell_rows) > 0:
                print(f"")
                print(f"  ⚠ Dashboard will be EMPTY despite {len(cell_rows)} "
                      f"stats rows present — see warnings above.")
        else:
            print(f"    (no data yet — sweep will populate as it runs; "
                  f"refresh dashboard to see new cells)")

    # 6. If we spawned a foreground sweep, wait for it. Ctrl-C
    #    propagates to the child via the shared process group, so
    #    the user gets a single clean exit signal for both.
    if spawned_proc is not None:
        if not args.quiet:
            print(f"\n  waiting for sweep (PID {spawned_proc.pid}); "
                  f"Ctrl-C to stop. Open {args.out} in a browser meanwhile.")
        try:
            spawned_proc.wait()
        except KeyboardInterrupt:
            # Sweep child receives the same SIGINT via process group.
            # Give it a moment to finish writing its current state
            # (status JSON, current flow CSV) before we exit.
            print(f"\n  Ctrl-C received — waiting for sweep to clean up...",
                  flush=True)
            try:
                spawned_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"  sweep didn't exit in 10s, sending SIGTERM",
                      flush=True)
                spawned_proc.terminate()
                spawned_proc.wait(timeout=5)
            print(f"  exited.")
            return
        rc = spawned_proc.returncode
        if rc == 0:
            print(f"\n  sweep finished normally. Final dashboard "
                  f"render → re-run `python main.py dashboard --dir "
                  f"{args.dir}` to pick up all new cells.")
        else:
            print(f"\n  sweep exited with code {rc}. Check log/status "
                  f"for details.")


if __name__ == "__main__":
    main()
