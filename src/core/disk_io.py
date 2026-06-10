"""
core/disk_io.py — JSON / log / worker-side logging helpers
==========================================================
Stateless. Imports stdlib + core.project_constants (for filenames).
Provides the orchestrator log, worker-side RSS-traced logging, and the
JSON-backed μ table.
"""

import json
import os
import sys
import time

from core.project_constants import MU_JSON


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator-side log (stdout)
# ═══════════════════════════════════════════════════════════════════
def log(msg, end="\n"):
    """Print to stdout with an ISO-format timestamp.  The format
    `YYYY-MM-DD HH:MM:SS` matches the dashboard's rendered "updated"
    stamp so log lines and dashboard state line up visually, AND it
    disambiguates day boundaries on multi-day sweeps — `[00:55:50]` was
    useless on a 4-day run because you couldn't tell which day a given
    line belonged to.  The CSV's t_start/t_end columns remain the
    canonical absolute-time source (epoch float seconds, sub-second
    precision); these log timestamps are the human-readable companion.

    The live dashboard captures this stdout stream into its in-memory
    heartbeat ring, which is the project's actual log surface; there is
    no separate on-disk logfile."""
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", end=end, flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Worker-side logging (memory-trace instrumentation)
# ═══════════════════════════════════════════════════════════════════
# These run inside worker processes, so they print directly to stdout
# (the parent terminal, captured by the live dashboard's heartbeat ring).
# Every line carries the worker PID and current RSS so per-worker memory
# growth is trivially grep-able from captured output:
#     ... | grep 'pid=12345'
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
    # Atomic write (tmp + fsync + os.replace), same guarantee the flow CSV
    # and the sidecars use: a crash mid-write must never leave a truncated
    # mu_table.json, since a corrupt table would poison every subsequent
    # resume. os.replace is atomic on POSIX, so the final path only ever
    # appears complete — a killed writer leaves at most a stray .tmp.
    parent = os.path.dirname(MU_JSON)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{MU_JSON}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump({"mu_table": table,
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, MU_JSON)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def mu_key(k, T, lb):
    return f"{int(k)}_{float(T)}_{float(lb)}"


# ═══════════════════════════════════════════════════════════════════
#  N-aware μ — the μ-vs-N map
# ═══════════════════════════════════════════════════════════════════
# μ is calibrated at N=MU_N_CAL (small, cheap). The realised k_avg can drift
# from the target k as N grows. Rather than re-running the (expensive) binary
# search at every N, the sweep measures the drift FOR FREE from the cells it
# has already built (every meta sidecar records k_avg and the μ used) and
# stores first-order Newton corrections under N-qualified keys:
#
#     "<k>_<T>_<lb>@<N>"  →  μ corrected for rungs of size ≥ N
#
# The correction uses the empirical near-hyperbola k·μ ≈ const (see
# graph_builder._mu_guess: μ ≈ C/k in every regime), so to move a realised
# k_avg back onto the target k the multiplicative update is
#
#     μ_corrected = μ_used × (k_avg / k_target)
#
# (denser than wanted ⇒ k_avg > k ⇒ raise the penalty, and vice versa).
# mu_lookup resolves a cell's μ as: exact @N key → largest @N' key with
# N' ≤ N → the base (k,T,lb) key. Corrections are only ever derived from
# completed data and only applied to rungs that have produced none yet
# (resolve_cell_mu prefers the μ recorded in an existing sidecar), so seeds
# within one (k,T,lb,N) cell can never mix different μ values.

def mu_key_n(k, T, lb, N):
    return f"{mu_key(k, T, lb)}@{int(N)}"


def mu_lookup(table, k, T, lb, N):
    """Resolve μ for a cell at size N: exact N-qualified entry, else the
    entry for the largest qualified N' ≤ N, else the base calibration.
    Returns (mu, key_used) or (None, None) when no entry exists at all."""
    base = mu_key(k, T, lb)
    exact = mu_key_n(k, T, lb, N)
    if exact in table:
        return float(table[exact]), exact
    best_n, best_key = -1, None
    prefix = base + "@"
    for key in table:
        if key.startswith(prefix):
            try:
                n = int(key[len(prefix):])
            except ValueError:
                continue
            if best_n < n <= int(N):
                best_n, best_key = n, key
    if best_key is not None:
        return float(table[best_key]), best_key
    if base in table:
        return float(table[base]), base
    return None, None


def resolve_cell_mu(table, k, T, lb, N, flow_dir="flow"):
    """μ for ONE cell (k,T,lb,N), with seed consistency: if any seed of this
    cell already wrote a meta sidecar, reuse the μ recorded there — later
    seeds of the same cell must be grown with the SAME μ as the earlier ones,
    even if a drift correction has landed in the table since. Only a cell
    with no data yet picks up corrections via mu_lookup.

    Returns (mu, source) where source is 'meta', a table key, or None."""
    import glob as _glob
    pat = os.path.join(flow_dir,
                       f"meta_k{int(k)}_T{float(T)}_lb{float(lb)}"
                       f"_N{int(N)}_s*.json")
    for p in sorted(_glob.glob(pat)):
        try:
            with open(p) as fh:
                m = json.load(fh)
            mu = m.get("mu")
            if mu is not None and mu == mu:        # not None, not NaN
                return float(mu), "meta"
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    mu, key = mu_lookup(table, k, T, lb, N)
    return mu, key
