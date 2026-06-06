"""
core/disk_io.py — JSON / log / worker-side logging helpers
==========================================================
Stateless except for the open log file handle. Imports stdlib +
core.project_constants (for filenames). Provides the orchestrator log,
worker-side RSS-traced logging, and the JSON-backed μ table.
"""

import json
import os
import sys
import time
from threading import Event, Thread

from core.project_constants import LOG_FILE, MU_JSON, DATA_DIR


def _ensure_data_dir():
    """Make the output data dir on first need. Idempotent — cheap to call
    repeatedly. Centralised here so every writer (log, CSV, μ table,
    heartbeat files) goes through the same guarantee, instead of each
    path-using helper calling makedirs in its own way.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator-side log (LOG_FILE + stdout)
# ═══════════════════════════════════════════════════════════════════
_LOG_FH = None


def log(msg, end="\n"):
    """Print to stdout and append to LOG_FILE with an ISO-format
    timestamp.  The format `YYYY-MM-DD HH:MM:SS` matches the dashboard's
    rendered "updated" stamp so log lines and dashboard state line up
    visually, AND it disambiguates day boundaries on multi-day sweeps —
    `[00:55:50]` was useless on a 4-day run because you couldn't tell
    which day a given line belonged to.  The CSV's t_start/t_end
    columns remain the canonical absolute-time source (epoch float
    seconds, sub-second precision); these log timestamps are the
    human-readable companion."""
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, end=end, flush=True)
    global _LOG_FH
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + end)
            _LOG_FH.flush()
        except Exception:
            pass


def open_log():
    global _LOG_FH
    if _LOG_FH is None:
        _ensure_data_dir()
        _LOG_FH = open(LOG_FILE, "a", buffering=1)


# ═══════════════════════════════════════════════════════════════════
#  Worker-side logging (memory-trace instrumentation)
# ═══════════════════════════════════════════════════════════════════
# These run inside worker processes, so they print directly to stdout
# (the parent terminal) rather than through log() — that would race on
# the LOG_FILE handle. Every line carries the worker PID and current
# RSS so per-worker memory growth is trivially grep-able:
#     grep 'pid=12345' sweep.log
def _rss_mb():
    """Resident-set size in MB for the current process. Linux-only;
    -1 elsewhere."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1
    return -1


def _wlog(tag, msg):
    """One worker-side log line. Atomic single write so lines from
    parallel workers don't byte-interleave on the terminal.  Uses the
    same ISO format as log() — see that docstring for rationale."""
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    rss = _rss_mb()
    rss_str = f"{rss:>5d}MB" if rss >= 0 else "    ?MB"
    line = f"[{t}] [pid={os.getpid():>6d} rss={rss_str}] {tag}  {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def _start_heartbeat(tag, interval_s=60, sample_s=5):
    """Daemon thread: log RSS + current phase every interval_s seconds,
    AND sample RSS every sample_s seconds to track the peak. Returns
    (stop_fn, phase_ref, peak_rss_ref). Mutate phase_ref[0] on phase
    changes so the heartbeat reflects what the worker is currently
    doing. peak_rss_ref[0] always holds the highest RSS observed since
    the heartbeat started.

    Initial sample is taken eagerly BEFORE the first stop.wait(sample_s)
    blocks. Without this, jobs that finish in less than sample_s never
    record any RSS reading and write rss_peak_mb=0 to the CSV — which
    the dashboard's memory panel correctly filters out as no-data,
    making small-N rows invisible in the per-N peak-RSS view. Tiny
    cells (N≤1k) routinely finish in <5s, so the bug was systematic
    rather than rare.
    """
    stop = Event()
    phase = ["init"]
    peak_rss = [0]

    # Eager initial sample — guarantees rss_peak_mb is non-zero even
    # for sub-sample_s jobs. The loop below picks up subsequent peaks
    # on the regular cadence.
    rss0 = _rss_mb()
    if rss0 > peak_rss[0]:
        peak_rss[0] = rss0

    def loop():
        last_log = time.time()
        # Sample more frequently than we log so we catch transient RSS
        # spikes between phase boundaries (e.g. mid-SLQ tracelet
        # allocations) that minute-grained heartbeats would miss.
        while not stop.wait(sample_s):
            rss = _rss_mb()
            if rss > peak_rss[0]:
                peak_rss[0] = rss
            now = time.time()
            if now - last_log >= interval_s:
                _wlog(tag, f"♥ heartbeat — phase=[{phase[0]}] "
                           f"peak_rss={peak_rss[0]}MB")
                last_log = now

    Thread(target=loop, daemon=True).start()
    return stop.set, phase, peak_rss


# ═══════════════════════════════════════════════════════════════════
#  μ table — JSON-backed (k, T, lb) → μ cache
# ═══════════════════════════════════════════════════════════════════
def load_mu_table():
    if not os.path.exists(MU_JSON):
        return {}
    try:
        with open(MU_JSON, "r") as f:
            return json.load(f).get("mu_table", {})
    except Exception:
        return {}


def save_mu_table(table):
    parent = os.path.dirname(MU_JSON)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(MU_JSON, "w") as f:
        json.dump({"mu_table": table,
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2)


def mu_key(k, T, lb):
    return f"{int(k)}_{float(T)}_{float(lb)}"
