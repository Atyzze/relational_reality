"""
ds4_search/sweep_runner.py — Map d_s(t) curve shapes across a (k, lb, N) grid
====================================================================
Sweeps locality bias lb at fixed (k, T) for multiple N values, runs
the d_s(t) flow probe on each cell, and produces an overlay HTML
per (k, N) panel where curves are color-coded by lb. Designed to
characterise how the flow shape changes as a function of lb — for
example, locating a transition between monotone-rising and
double-peak shapes.

Includes a heuristic shape classifier that tags each curve as one of
{flat, monotone_rising, single_peak, double_peak, bumpy, undetermined}
so transitions in shape across lb can be summarised at a glance.

Usage
-----
    This is the batch worker the dashboard spawns as
    `python main.py __sweep__ …` (see main.py / ds4_search.live_app).
    `__sweep__` is an internal marker, not a public subcommand, but it IS
    the way to drive this module's argparse interface by hand — main.py
    forwards everything after the marker here:

    # Single (k, T), sweep lb at one N
    python main.py __sweep__ --k 9 --T 0.005 --N 64000 \\
                             --lb 0.93,0.94,0.95,0.99

    # Compare T=0 (ground state) against T=0.005 (low-T basin) —
    # shows whether thermal noise smooths out a transition seen at T=0
    python main.py __sweep__ --k 9 --T 0,0.005 --N 64000 \\
                             --lb 0.93,0.94,0.95,0.99

    # Full (k, T, lb, N) grid
    python main.py __sweep__ --k 6,8,9 --T 0,0.005 \\
                             --N 16000,64000,256000 \\
                             --lb 0.93,0.94,0.95,0.99

    # Use an explicit lb list (file)
    python main.py __sweep__ --k 9 --T 0.005 --N 64000 \\
                             --lb-file my_lb_grid.txt

The script reuses cached per-cell flow CSVs when present (skipping
SLQ recompute), so iterating on the layout/grid is cheap once a
cell has been run.

Outputs
-------
    flow_<tag>.csv / .html               — per-cell flow data
    lb_sweep_k<K>_N<N>.html              — per-panel overlay (one per k, N)
    lb_sweep_summary.html                — grid-of-panels summary
    lb_sweep_classifications.csv         — shape tag per cell

Cost per cell: 1-10 minutes depending on N.
With 3 k × 10 lb × 3 N = 90 cells, plan for several hours.
"""

import argparse
import concurrent.futures
import threading
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

# Unix-only (peak-RSS readout). A plain `import resource` is an ImportError
# on Windows, which would kill the whole sweep at import time even though
# setup.bat is a first-class entry point — so it degrades to None instead.
try:
    import resource
except ImportError:                      # Windows
    resource = None

# ── BLAS thread pinning ───────────────────────────────────────────
# Must be set BEFORE numpy is imported (here or transitively via
# core.cell_tests). Each worker process runs one SLQ at a time, and the
# Lanczos kernel inside it is dominated by sparse matvec — BLAS
# parallelism barely helps and actively hurts when N workers ×
# M BLAS-threads oversubscribes the CPU. Setting this once in the
# parent propagates to every forked worker.
#
# `setdefault` so an explicit env override (e.g. for benchmarking
# single-process BLAS speed) still wins.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS",
          # Numba has a @njit(parallel=True) kernel. We already parallelise
          # ACROSS cells with one process per worker, so letting each worker
          # ALSO spin up NUMBA_NUM_THREADS (defaults to the CPU count) threads
          # for that kernel is nested parallelism: 32 workers × 32 numba
          # threads ≈ 1000 threads fighting over 32 hardware threads. That
          # thrashing reads as "100% busy" at low power. Pin numba to 1 thread
          # per worker too. Set before numba is imported (below) so it takes.
          "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from core.cell_tests import build_cell, build_torus_cell, run_flow_test
from core.flow_probe import write_csv as write_flow_csv, write_quad_npz
from core.disk_io import load_mu_table, mu_key
from core.project_constants import MU_JSON, EC, MAX_DEG, PROD_SWEEPS, \
    REDRAW_INTERVAL_S, WORKERS, \
    FLOW_REFRESH_INTERVAL_S, MIN_FREE_GB, MU_N_CAL, FLOW_DIR
from core.graph_builder import calibrate_missing, update_mu_table_n


def _report_thread_caps():
    """Print the per-process BLAS / OpenMP / Numba thread caps at sweep startup,
    so the log makes it OBVIOUS whether each worker is single-threaded. The sweep
    forks one worker per core; if BLAS is NOT pinned to 1 thread per worker, then
    N workers x N BLAS threads thrash over N cores and the spectral probe prints
    "[flow] SLQ:" and then appears to hang (no error, no output, no flow CSV).
    Seeing "OPENBLAS=1" here confirms the pin took; anything else is the hang
    precursor and is called out loudly below."""
    g = lambda v: os.environ.get(v, "unset")
    line = (f"[threads] OMP={g('OMP_NUM_THREADS')} OPENBLAS={g('OPENBLAS_NUM_THREADS')} "
            f"MKL={g('MKL_NUM_THREADS')} NUMBA={g('NUMBA_NUM_THREADS')}")
    try:
        import threadpoolctl
        pools = threadpoolctl.threadpool_info()
        if pools:
            line += " | live: " + ", ".join(
                f"{p.get('user_api', p.get('internal_api', '?'))}:"
                f"{p.get('num_threads', '?')}t" for p in pools)
    except Exception:
        line += " | (pip install threadpoolctl for the live BLAS thread count)"
    print(line, flush=True)
    if g("OPENBLAS_NUM_THREADS") != "1" or g("OMP_NUM_THREADS") != "1":
        print("  !!! BLAS is NOT pinned to 1 thread per worker. With one worker per "
              "core this oversubscribes and the SLQ probe will appear to hang with no "
              "error and write no flow CSV. Ensure nothing exports OMP_NUM_THREADS / "
              "OPENBLAS_NUM_THREADS before launch, then rerun.", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Output paths
# ═══════════════════════════════════════════════════════════════════
# Per-cell flow CSVs go into the FLOW_DIR subdirectory (imported above from
# core.project_constants — shared project-wide) so the project root stays
# uncluttered as the dataset grows. Existing top-level flow_*.csv files
# are still readable by the dashboard (it globs both locations).

# Failure visibility. Cells that never produce a flow_*.csv (a worker raised,
# or μ-calibration was dropped) are otherwise indistinguishable from
# not-yet-run cells in the heatmap. We persist *why* so shape_analysis can
# mark them: a per-cell sidecar for in-worker failures, and one ledger for
# calibration drops. `MU_FAILURES_JSON` lives in the output root (next to
# mu_table.json); fail sidecars live in FLOW_DIR next to the (absent) CSV.
MU_FAILURES_JSON = "mu_failures.json"


def _classify_fail_reason(msg):
    """Bucket a failure message into a glyph category. 'max_degree' is the
    one the heatmap calls out specially (the graph's k_top hit the engine
    cap MAX_DEG, so the chain stopped sampling H); everything else is a
    generic 'error'."""
    m = (msg or "").lower()
    return "max_degree" if "max_degree" in m else "error"


def _fmt_hms(seconds):
    """Compact duration: '1h 03m', '12m 40s', '45s'. For the cumulative
    compute figure, which can run to hours over a long sweep."""
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _load_wall_history(flow_dir=FLOW_DIR):
    """Seed the ETA cost model from wall times already on disk. Every cell
    that finished on a prior run wrote its total wall time to its
    meta_<tag>.json sidecar, so we can recover the (N, wall_s) cost trend
    immediately at startup instead of re-learning it from scratch each run.
    Without this, a resumed sweep (or an os.execv hot-reload) starts with an
    empty cost model right after the cheap cells are all cached — i.e. the
    ETA is cold exactly when the expensive tail begins. Best-effort; returns
    a list of (N, wall_s) tuples."""
    samples = []
    for p in glob.glob(os.path.join(flow_dir, "meta_*.json")):
        try:
            with open(p) as fh:
                m = json.load(fh)
            N, w = m.get("N"), m.get("wall_s")
            if N and w and float(w) > 0:
                samples.append((int(N), float(w)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return samples


def _load_compute_invested(flow_dir=FLOW_DIR, assume_workers=1):
    """Sum cell wall_s across all meta sidecars (= total CORE-seconds, since
    cells ran in parallel) AND estimate real elapsed wall-clock by dividing
    each cell's wall_s by the worker count that was active when it ran
    (recorded in meta as 'workers'). Cells written before we recorded that
    fall back to `assume_workers`. Returns
    (core_seconds, est_elapsed_seconds, n_with_workers, n_total)."""
    core_s = elapsed_s = 0.0
    n_with = n_total = 0
    for p in glob.glob(os.path.join(flow_dir, "meta_*.json")):
        try:
            with open(p) as fh:
                m = json.load(fh)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        w = m.get("wall_s")
        if not (w and float(w) > 0):
            continue
        n_total += 1
        core_s += float(w)
        wk = m.get("workers")
        if wk and int(wk) > 0:
            elapsed_s += float(w) / float(wk)
            n_with += 1
        else:
            elapsed_s += float(w) / max(1, int(assume_workers))
    return core_s, elapsed_s, n_with, n_total


def _self_rss_mb():
    """Resident set size of THIS process in MB (Linux /proc). 0 if unknown."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0   # kB → MB
    except Exception:
        pass
    return 0.0


def _free_mem_bytes():
    """Allocatable memory in bytes (Linux /proc/meminfo MemAvailable, psutil
    fallback). None if it can't be determined — the guard then no-ops."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024       # kB → bytes
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _load_mem_history(flow_dir=FLOW_DIR):
    """(N, rss_bytes) pairs from meta sidecars that recorded rss_mb — the
    per-cell memory cost trend, recovered across restarts like the ETA model."""
    out = []
    for p in glob.glob(os.path.join(flow_dir, "meta_*.json")):
        try:
            with open(p) as fh:
                m = json.load(fh)
            N, r = m.get("N"), m.get("rss_mb")
            if N and r and float(r) > 0:
                out.append((int(N), float(r) * 1024 * 1024))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return out


def _make_rss_model(samples):
    """Return predict(N)->bytes for one cell's resident memory. Linear fit
    rss≈a·N+b from history if it spans ≥2 distinct N; otherwise a conservative
    analytical fallback (per-worker baseline + bytes/node) that OVER-estimates,
    so the guard errs toward safety. The fit is floored by ½ the fallback so a
    degenerate fit can't make us reckless."""
    base, per_node = 300e6, 1024.0          # ~interpreter baseline; B/node
    fallback = lambda N: base + per_node * N
    Ns = sorted({n for n, _ in samples})
    if len(Ns) >= 2:
        import numpy as _np
        a, b = _np.polyfit(_np.array([n for n, _ in samples], float),
                           _np.array([r for _, r in samples], float), 1)
        return lambda N: float(max(a * N + b, 0.5 * fallback(N)))
    return fallback


# ═══════════════════════════════════════════════════════════════════
#  Shape classifier
# ═══════════════════════════════════════════════════════════════════
# Single source of truth lives in ds4_search/shape_classify.py so this
# sweep and the offline dashboard (static_dashboard.py) can't drift apart.
# Imported HERE rather than at the top of the file on purpose: the import
# pulls in numpy, and the BLAS/numba thread-pinning env vars at the top must
# be set before numpy is first imported. By this point numpy is already in.
from ds4_search.shape_classify import classify_shape  # noqa: E402,F401


# ═══════════════════════════════════════════════════════════════════
#  Status JSON: live progress / ETA / RSS for the dashboard to poll
# ═══════════════════════════════════════════════════════════════════
def get_rss_mb():
    """Current RSS in MB on Linux. Returns None elsewhere."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0  # KB → MB
    except (OSError, ValueError):
        pass
    return None


def get_rss_peak_mb():
    """Peak RSS in MB. ru_maxrss is KILOBYTES on Linux but BYTES on macOS —
    dispatch on sys.platform. (An earlier version guessed the unit from the
    value's magnitude, which misread a genuinely large Linux peak — >100 GB
    is exactly the regime the big-N warnings in the README describe — as
    macOS bytes and under-reported it 1024×.) None on Windows (no
    `resource` module) — callers already treat None as "unknown"."""
    if resource is None:
        return None
    try:
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return v / (1024.0 * 1024.0)   # bytes → MB
        return v / 1024.0                   # kB → MB (Linux and BSDs)
    except Exception:
        return None


# Cached at module load — script hash + start time. The hash flips
# on os.execv restart (because the new process re-imports this
# module from updated source), giving the dashboard a visible
# version-change signal.
_SWEEP_BUILD_INFO = {
    "script_path": os.path.basename(__file__),
    "script_hash": (lambda: hashlib.sha256(
        open(__file__, "rb").read()).hexdigest()[:8])()
        if os.path.exists(__file__) else "unknown",
    "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
}


def _work_N(w):
    """Node count N of a work-item (cell or torus)."""
    return w[4] if w[0] == "cell" else w[2]


def estimate_eta_seconds(done_samples, remaining_Ns, n_workers):
    """Wall-clock ETA (seconds) for the remaining work.

    The naive "remaining ÷ average rate" estimate is badly wrong here:
    cells are run smallest-N first, and per-cell cost grows steeply with
    N (roughly a power law, since the SLQ/RW work scales super-linearly).
    So once the small cells are done, the average rate is far too
    optimistic for the big-N cells still to come.

    Instead we fit the observed cost trend  wall ≈ a · N^b  on a log-log
    least-squares line through the completed (N, wall) samples, predict
    each remaining cell's wall from its own N, sum that, and divide by the
    worker count (work runs in parallel). The result is also floored at the
    single longest predicted item — you can't finish faster than the slowest
    remaining cell on one core.

    Returns None until there are enough samples (≥3 over ≥2 distinct N) to
    fit a trend; the caller then just omits the ETA.
    """
    if not remaining_Ns:
        return 0
    pts = [(float(n), float(w)) for (n, w) in done_samples
           if n and w and w > 0]
    distinct_N = {n for n, _ in pts}
    if len(pts) >= 3 and len(distinct_N) >= 2:
        xs = np.log(np.array([n for n, _ in pts]))
        ys = np.log(np.array([w for _, w in pts]))
        b, log_a = np.polyfit(xs, ys, 1)      # ys ≈ b·xs + log_a
        a = float(np.exp(log_a))

        def predict(n):
            return a * (n ** b)
    elif pts:
        avg = sum(w for _, w in pts) / len(pts)

        def predict(n):
            return avg
    else:
        return None
    preds = [predict(n) for n in remaining_Ns]
    total_cpu_s = sum(preds)
    wall = total_cpu_s / max(1, n_workers)
    return int(max(wall, max(preds, default=0)))


class _ShapeRefresher:
    """Runs the shape analysis (CSV + histogram + heatmaps) OFF the worker-
    dispatch thread, so regenerating plots never stalls the sweep.

    The dispatch loop calls .request() whenever new cells have landed; that
    is non-blocking. A single daemon thread does the actual (heavy, growing)
    re-scan-and-render, at most once per `min_interval_s` seconds, and
    coalesces bursts of requests into one run. If a request arrives while a
    render is in flight, exactly one more render is scheduled afterwards
    (so the final state is never missed) rather than queueing many.

    `periodic_s` (>0) adds a wall-clock heartbeat: if that many seconds pass
    with no render, the thread forces one even though no new cell finished.
    Without it, a long high-N cell could grind for many minutes during which
    nothing redraws, so edits / newly-flagged cells stay invisible. 0 turns
    the heartbeat off (pure request-driven, the original behaviour).

    Why time-throttled instead of every-N-cells: the refresh re-reads every
    flow_*.csv on disk, so its cost grows with the sweep. A fixed cell count
    made the stalls lengthen over time; a wall-clock floor keeps plot churn
    bounded no matter how large the run or how fast cells complete.
    """

    def __init__(self, min_interval_s=10.0, periodic_s=0.0):
        self.min_interval_s = float(min_interval_s)
        self.periodic_s = max(0.0, float(periodic_s))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending = False
        self._stop = False
        # 0.0 == "never rendered": with the heartbeat on this makes the first
        # loop render immediately, so a relaunch shows the current grid at once
        # instead of after one full interval.
        self._last_run = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="shape-refresh")
        self._thread.start()

    def request(self):
        """Non-blocking: mark that a refresh is wanted and wake the worker."""
        with self._lock:
            self._pending = True
        self._wake.set()

    def _loop(self):
        while True:
            if self._stop and not self._pending:
                return
            # Block until either a request wakes us or — if the heartbeat is
            # on — the periodic deadline elapses, whichever comes first.
            if self.periodic_s > 0:
                timeout = max(0.0, (self._last_run + self.periodic_s)
                              - time.time())
                self._wake.wait(timeout)
            else:
                self._wake.wait()
            if self._stop and not self._pending:
                return
            # Honour the minimum interval between renders, but stay
            # responsive to stop(): wait on the event in short slices so a
            # shutdown during the throttle window returns promptly instead
            # of blocking for the full interval.
            deadline = self._last_run + self.min_interval_s
            while time.time() < deadline:
                if self._stop and not self._pending:
                    return
                time.sleep(min(0.1, max(0.0, deadline - time.time())))
            # Render if a request is pending OR the heartbeat is due (so the
            # plots advance even with no new data).
            periodic_due = (self.periodic_s > 0 and
                            (time.time() - self._last_run) >= self.periodic_s)
            with self._lock:
                do_render = self._pending or periodic_due
                self._pending = False
                self._wake.clear()
            if self._stop and not do_render:
                return
            if do_render:
                _refresh_shape_analysis()
                self._last_run = time.time()

    def stop(self, final_refresh=False):
        """Stop the background thread. The caller is responsible for any
        final synchronous refresh (finalize does one), so by default we do
        NOT render here — we just unblock and join."""
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2.0)


def _finalize_sweep(args, classifications, cells, seed_list, t_total,
                    n_done, n_failed, phase):
    """Write the post-sweep artefacts: classifications CSV, a final shape
    refresh (so the plots reflect every finished cell), and a terminal
    status JSON. Safe to call on the normal completion path AND from the
    Ctrl-C handler, so an interrupted run still leaves up-to-date plots
    and a CSV for whatever cells actually finished — not just a bare
    flow/ dir.
    """
    cls_path = f"{args.out_prefix}_classifications.csv"
    with open(cls_path, "w") as f:
        f.write("k,T,lb,N,seed,shape,ds_min,ds_max\n")
        for row in classifications:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\n  → {cls_path}", flush=True)
    print(f"  flow data: up to {len(cells) * len(seed_list)} flow_*.csv "
          f"file(s) ({len(cells)} cells × {len(seed_list)} seed(s)) in "
          f"{FLOW_DIR}/", flush=True)
    print(f"  total wall time: {time.time()-t_total:.0f}s", flush=True)

    # Final shape refresh so the last batch of cells is reflected in the
    # summary + plots, however the sweep ended.
    _refresh_shape_analysis()

    write_status_json(args.status_json, build_status(
        phase=phase,
        started_at=t_total,
        n_total=len(cells) * len(seed_list), n_done=n_done,
        n_failed=n_failed,
    ))


def _flow_health_signal():
    """If the heatmap comes up empty, say WHY. Distinguish 'no cell has
    finished yet' (benign, early in a run) from 'cells ARE finishing but every
    one is FAILING' (loud — you are silently banking zero usable data). The
    latter shows up as fail_*.json sidecars with no flow_*.csv beside them.
    Pure read-only + best-effort: it must never perturb the sweep."""
    try:
        import collections
        ok = glob.glob("flow/flow_*.csv") + glob.glob("flow_*.csv")
        fails = glob.glob("flow/fail_*.json") + glob.glob("fail_*.json")
        if ok or not fails:
            return  # have data, or nothing has run yet -> nothing to warn about
        reasons = collections.Counter()
        sample = None
        for fp in fails:
            try:
                with open(fp) as fh:
                    rec = json.load(fh)
                reasons[rec.get("reason", "unknown")] += 1
                if sample is None:
                    sample = rec.get("detail", "")
            except Exception:
                reasons["unreadable"] += 1
        bar = "!" * 74
        top = ", ".join(f"{n}x {r}" for r, n in reasons.most_common())
        print("\n" + bar +
              f"\n  {len(fails)} cells have FINISHED but ALL FAILED - 0 produced a flow CSV,"
              "\n  so the heatmap has nothing to plot. This is NOT a plotting problem."
              f"\n  Failure reasons: {top}" +
              (f"\n  Example detail: {str(sample)[:180]}" if sample else "") +
              "\n  Per-cell records are in flow/fail_*.json (and the [tag] FAILED lines"
              "\n  above). If k_avg sits far below your target k, the graphs are"
              "\n  under-thermalised (typically the very-high-lb bootstrap) and the d_s"
              "\n  probe then fails on the near-1D / disconnected result."
              "\n" + bar + "\n", flush=True)
    except Exception:
        pass


def _refresh_shape_analysis():
    """Re-run the shape analysis on the data gathered so far, refreshing
    shape_summary.csv and the histogram/heatmap PNGs in the output dir.

    The sweep runs with the output directory as its cwd, so the default
    `--dir .` finds flow/ and writes the artefacts alongside it. Best-effort:
    a plotting hiccup must never take down the sweep, so everything is
    wrapped and failures are logged and swallowed.
    """
    try:
        import shape_analysis
        shape_analysis.main(["--dir", "."])
        print("  \u21bb shape analysis refreshed "
              "(shape_summary.csv + histogram + heatmap)", flush=True)
    except Exception as e:
        print(f"  [shape] refresh skipped: {type(e).__name__}: {e}",
              flush=True)
    _flow_health_signal()
    # Dual-mode (flat-4D vs flowing-4D) analysis on the same data. Additive
    # and best-effort: a hiccup here must never take down the sweep.
    try:
        import flow_modes
        flow_modes.main(["--dir", "."])
        print("  \u21bb flow-mode analysis refreshed "
              "(flow_modes.csv + plane + maps)", flush=True)
    except Exception as e:
        print(f"  [flow_modes] refresh skipped: {type(e).__name__}: {e}",
              flush=True)
    # Field-isotropy heatmap from the per-cell isotropy summaries stored in the
    # meta sidecars (computed once at build time — nothing is re-probed here).
    # Additive and best-effort.
    try:
        from physics_tests import isotropy as _iso
        out = _iso.render_isotropy_heatmap(".", "isotropy_heatmap.png")
        if out:
            print("  \u21bb isotropy heatmap refreshed (isotropy_heatmap.png)",
                  flush=True)
    except Exception as e:
        print(f"  [isotropy] heatmap refresh skipped: {type(e).__name__}: {e}",
              flush=True)
    # Heavier N->infinity convergence pass + its leaderboard charts, on a
    # slower throttle (only meaningful once cells have several N).
    _maybe_refresh_flow_convergence()


# Count of μ-calibration drops, surfaced in every status payload (set once
# at sweep start by _mu_failures_banner via main()).
_MU_FAIL_COUNT = [0]

# Wall-clock of the last flow-convergence refresh, so it can run on a slower
# cadence than the per-cell heatmaps/maps above.
_last_flow_conv_run = 0.0

# Wall-clock of the last μ-vs-N drift audit (same slow cadence; the audit only
# reads JSON sidecars but there is no point re-running it per cell).
_last_mu_drift_run = 0.0


def _maybe_update_mu_drift(mu_table):
    """Throttled μ-vs-N drift audit (see graph_builder.update_mu_table_n).
    Mutates `mu_table` in place — corrections become visible to every
    work-item submitted AFTER this call — and persists the merged table.
    Must run on the dispatch thread (it feeds future submits); reads only
    meta sidecars, so it is cheap. Best-effort, never takes down the sweep."""
    global _last_mu_drift_run
    iv = FLOW_REFRESH_INTERVAL_S
    if iv < 0:
        return
    now = time.time()
    if iv > 0 and (now - _last_mu_drift_run) < iv:
        return
    _last_mu_drift_run = now
    try:
        added = update_mu_table_n(mu_table, flow_dir=FLOW_DIR,
                                  log_fn=lambda m: print(m, flush=True))
        if added:
            from core.disk_io import save_mu_table
            save_mu_table(mu_table)
            print(f"  \u21bb μ-vs-N corrections saved: {len(added)} new "
                  f"N-qualified entr{'y' if len(added) == 1 else 'ies'} "
                  f"(applied only to rungs with no data yet)", flush=True)
    except Exception as e:
        print(f"  [μ-drift] audit skipped: {type(e).__name__}: {e}",
              flush=True)


def _maybe_refresh_flow_convergence():
    """Throttled refresh of flow_convergence.csv (per-cell N->infinity plateau
    extrapolation) and its leaderboard charts (flow_map/flow_scatter). Runs at
    most once per FLOW_REFRESH_INTERVAL_S; a negative interval disables it.
    Heavier than the maps above and only meaningful with several N per cell,
    so it does not run on every cell. Best-effort and quiet (the tools' own
    verbose tables are suppressed; one status line is printed)."""
    global _last_flow_conv_run
    iv = FLOW_REFRESH_INTERVAL_S
    if iv < 0:                                  # disabled
        return
    now = time.time()
    if iv > 0 and (now - _last_flow_conv_run) < iv:
        return                                  # throttled
    _last_flow_conv_run = now
    import contextlib
    import io
    try:
        import flow_convergence
        with contextlib.redirect_stdout(io.StringIO()):
            flow_convergence.main(["--dir", "."])
        if os.path.exists("flow_convergence.csv"):
            import flow_charts
            with contextlib.redirect_stdout(io.StringIO()):
                flow_charts.main(["--csv", "flow_convergence.csv"])
            print("  \u21bb flow convergence + charts refreshed "
                  "(flow_convergence.csv + flow_map/flow_scatter)", flush=True)
    except Exception as e:
        print(f"  [flow_convergence] refresh skipped: "
              f"{type(e).__name__}: {e}", flush=True)


def _mu_failures_banner():
    """μ-calibration drops mean grid cells that are PHYSICALLY UNREACHABLE
    (no μ in the search range realises the target k there) — per-cell
    information, not a program error, so the sweep must not abort. But it
    must also be impossible to miss: red banner at start and a count in the
    status JSON / dashboard."""
    try:
        with open(MU_FAILURES_JSON) as fh:
            led = json.load(fh)
    except (OSError, ValueError):
        return 0
    n = len(led) if isinstance(led, (list, dict)) else 0
    if n:
        bar = "!" * 74
        print(f"\n{bar}\n  ⚠ {n} grid cell(s) FAILED μ-calibration — those "
              f"(k, T, lb) combinations are\n  physically unreachable (no μ "
              f"realises the target k). They are skipped and\n  marked '!' "
              f"in the heatmaps. Ledger: {MU_FAILURES_JSON}\n{bar}\n",
              flush=True)
    return n


def build_status(phase, started_at, n_total, n_done, n_failed,
                 last_cell=None, next_cell=None,
                 in_flight=None,
                 calibration=None,
                 eta_sec_override=None,
                 compute_spent_sec=None,
                 cells_done_session=None):
    """Assemble a status dict for write_status_json.

    All values are JSON-safe (no numpy scalars, no NaN/Inf passed
    through unprotected). Dashboard polls this and re-renders a
    tiny status banner.

    `in_flight` is a list of dicts, one per work-item currently
    executing in the worker pool. Each is either a cell (has k, T,
    lb, N) or a torus reference (has d, L, N). The dashboard renders
    these so the user can see all parallel work, not just one
    "next" cell. `next_cell` is kept for backward compatibility:
    we set it to the first in_flight item if any.

    `compute_spent_sec` is the cumulative compute already invested across
    ALL runs (sum of every completed cell's wall time, recovered from the
    meta sidecars at startup + this session). It's CPU-time summed over
    cells, not wall-clock — cells run in parallel — so it's the "machine
    time burned" figure, independent of the per-session `elapsed_sec`.
    `cells_done_session` is how many cells THIS run finished; when given it
    drives the rate so a relaunch (where n_done already counts cached cells)
    doesn't report an absurd cells/min off a near-zero elapsed.
    """
    now = time.time()
    elapsed = now - started_at if started_at else 0
    # Rate from THIS session's completions, not the cached-inclusive n_done.
    rate_n = cells_done_session if cells_done_session is not None else n_done
    rate = (rate_n / elapsed * 60.0) if (elapsed > 0 and rate_n > 0) else None
    remaining = max(0, n_total - n_done - n_failed)
    eta = (remaining / (rate / 60.0)) if (rate and rate > 0) else None

    # Back-compat: if caller passed in_flight but not next_cell,
    # synthesise a next_cell from the first cell-kind in_flight item
    # so older dashboards still render something sensible.
    if next_cell is None and in_flight:
        for item in in_flight:
            if item.get("kind") == "cell":
                next_cell = (item["k"], item["T"], item["lb"], item["N"])
                break

    # Prefer the cost-extrapolated ETA when the caller supplies it (it
    # accounts for the steep cost-vs-N growth); fall back to the naive
    # remaining/rate estimate only before enough samples exist to fit.
    eta_final = eta_sec_override if eta_sec_override is not None else eta

    # Feasibility check: when the extrapolated ETA exceeds half a year the
    # configured grid is not going to finish on this machine — say so in
    # words, instead of trusting the user to convert a 10-digit second count.
    # The threshold is deliberately generous; multi-month sweeps are real.
    eta_warning = None
    if eta_final is not None and eta_final > 183 * 86400:
        eta_warning = (
            f"projected finish is ~{eta_final / (365.25 * 86400):.1f} YEARS "
            f"away — the remaining N ladder is infeasible on this machine. "
            f"Trim the top N rungs in main.toml (see the README note on the "
            f"N ladder); already-computed cells are kept and skipped.")

    out = {
        "phase": phase,                       # running | done | calibrating
        "started_at": started_at,
        "now": now,
        "elapsed_sec": round(elapsed, 1),
        "compute_spent_sec": (round(float(compute_spent_sec), 1)
                              if compute_spent_sec else None),
        "rss_mb": round(get_rss_mb(), 1) if get_rss_mb() else None,
        "rss_peak_mb": round(get_rss_peak_mb(), 1) if get_rss_peak_mb() else None,
        "n_total": n_total,
        "n_done": n_done,
        "n_failed": n_failed,
        "n_remaining": remaining,
        "rate_per_min": round(rate, 2) if rate else None,
        "eta_sec": round(eta_final) if eta_final else None,
        "eta_warning": eta_warning,
        "mu_failures": _MU_FAIL_COUNT[0],
        "last_cell": last_cell,
        "next_cell": (
            {"k": next_cell[0], "T": next_cell[1],
             "lb": next_cell[2], "N": next_cell[3]}
            if next_cell else None),
        "in_flight": in_flight or [],
        "build": _SWEEP_BUILD_INFO,
    }
    if calibration is not None:
        out["calibration"] = calibration
    return out


def write_status_json(path, status):
    """Atomic write: dump to .tmp then rename. Avoids partial-read
    races where the dashboard polls mid-write."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2, default=str)
        os.replace(tmp, path)   # atomic on POSIX
    except OSError:
        # Don't crash the sweep over a status write failure.
        pass


# ═══════════════════════════════════════════════════════════════════
#  Hot reload: process-level (os.execv) when source files change
# ═══════════════════════════════════════════════════════════════════
def _watched_files():
    """Return the list of .py files we'll monitor for changes.

    Watches every .py file under the src/ tree (core/, ds4_search/,
    metrics/, ...). The sweep imports several of these (core.cell_tests,
    core.flow_probe, core.graph_builder, etc.) and any of them changing should
    trigger a restart.

    Hot-reload is *not* needed for changes to data files (mu_table,
    flow_*.csv) — those are re-read on every cell anyway.
    """
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(src_root, "**", "*.py"),
                             recursive=True))
    # Always include this file even if naming convention changes
    me = os.path.abspath(__file__)
    if me not in files:
        files.append(me)
    return files


def _file_hash(path):
    """SHA-256 of file contents. None if file is unreadable.

    SHA-256 is overkill for change detection but it's stdlib, fast
    enough for a few-hundred-KB Python source, and the False-positive
    rate is zero. We don't truncate — full hex digests sit in a small
    dict for the sweep's lifetime.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def snapshot_source_hashes():
    """{path: hash} for all watched files at this moment.

    Captured once at sweep startup (after imports complete) and again
    between each cell. Comparison drives the restart decision.
    """
    return {p: _file_hash(p) for p in _watched_files()}


def detect_changes(baseline, current):
    """Return list of paths whose hash differs between baseline and
    current snapshots. Empty list = no restart needed.
    """
    changed = []
    for p, h in current.items():
        if baseline.get(p) != h:
            changed.append(p)
    # Also catch deletions — file in baseline but missing now
    for p in baseline:
        if p not in current:
            changed.append(p)
    return changed


def restart_self(reason, status_json_path, status_payload):
    """Replace the current process with a fresh invocation of itself
    using the same argv. The cached-cell skip logic in main() means
    the restart picks up exactly where this run left off without
    having to hand off any state.

    Writes a final status JSON announcing the restart so the
    dashboard's banner can show it briefly before the new process
    overwrites the file with its own status.

    os.execv is POSIX-only. On Windows you'd need subprocess+exit.
    All file descriptors are inherited; stdout/stderr stay attached
    to the same pipe (or log file, in the dashboard-spawn case).
    """
    status_payload = dict(status_payload)
    status_payload["phase"] = "restarting"
    status_payload["restart_reason"] = reason
    write_status_json(status_json_path, status_payload)
    print(f"\n  ⟳ source change detected: {reason}", flush=True)
    print("    restarting sweep via os.execv (cached cells will skip)",
          flush=True)
    # Brief pause so the status JSON shows "restarting" for at least
    # one dashboard poll cycle — otherwise it's overwritten so fast
    # the user never sees the message.
    time.sleep(1.0)
    os.execv(sys.executable, [sys.executable, *sys.argv])
    # ↑ Does not return on success. If it does return, exec failed.
    print("    [FATAL] os.execv failed; exiting with code 1",
          flush=True)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  Cell I/O
# ═══════════════════════════════════════════════════════════════════
def cell_tag(k, T, lb, N, seed):
    return f"k{k}_T{T}_lb{lb}_N{N}_s{seed}"


def torus_tag(d, L, seed):
    return f"torus{d}d_L{L}_s{seed}"


def torus_L_for_N(N: int, d: int) -> int:
    return max(6, int(round(N ** (1.0 / d))))


# ═══════════════════════════════════════════════════════════════════
#  Parallel work-item dispatch
# ═══════════════════════════════════════════════════════════════════
# A work-item is a tuple identifying one unit of expensive work that
# can run independently in a worker process. Seed is part of the
# identity — each (cell, seed) pair is its own independent SLQ run with
# its own flow_*.csv, so multi-seed sweeps parallelise and resume
# per-seed for free:
#     ("torus", d, N, seed)         — torus reference at (d, L=L(N,d))
#     ("cell",  k, T, lb, N, seed)  — sweep cell
#
# Both ultimately call run_flow_test() which is single-threaded SLQ.
# Wrapping each in a worker process and dispatching all of them to a
# ProcessPoolExecutor lets the box's many cores run many items in
# parallel — the original loops were serial, so a 16-core machine
# was idling 15 cores while one SLQ ground through L=45.
def _work_seed(w):
    """Seed of a work-item (always the last tuple element)."""
    return w[-1]


def _work_tag(w):
    """Short identifier for log prefixing (includes seed)."""
    if w[0] == "torus":
        _, d, N, seed = w
        L = torus_L_for_N(N, d)
        return f"torus{d}d_L{L}_N{N}_s{seed}"
    _, k, T, lb, N, seed = w
    return f"k{k}_T{T}_lb{lb}_N{N}_s{seed}"


def _work_cost_estimate(w):
    """Rough wall-time estimate (seconds) for ordering. SLQ scales
    roughly linearly in N at large N, sub-linear at small N. We
    submit smallest-first so the dashboard fills with results
    quickly instead of staring at a single huge L=45 cell for
    minutes before anything appears.
    """
    if w[0] == "torus":
        _, d, N, seed = w
        L = torus_L_for_N(N, d)
        n_eff = L ** d
        return 1.0 + n_eff / 50000.0
    _, k, T, lb, N, seed = w
    return 1.0 + N / 50000.0


def _csv_path_for(w):
    """Where this work-item's output CSV will live. Used both for
    cached-skip detection and for writing the result. The seed is taken
    from the work-item itself (its last element).
    """
    if w[0] == "torus":
        _, d, N, seed = w
        L = torus_L_for_N(N, d)
        return os.path.join(FLOW_DIR, f"flow_{torus_tag(d, L, seed)}.csv")
    _, k, T, lb, N, seed = w
    return os.path.join(FLOW_DIR, f"flow_{cell_tag(k, T, lb, N, seed)}.csv")


def _work_in_flight_dict(w):
    """JSON-safe in_flight entry for the status JSON. Dashboard
    renders these as a list of currently-running cells.
    """
    if w[0] == "torus":
        _, d, N, seed = w
        L = torus_L_for_N(N, d)
        return {"kind": "torus", "d": d, "L": L, "N": N, "seed": seed}
    _, k, T, lb, N, seed = w
    return {"kind": "cell", "k": k, "T": T, "lb": lb, "N": N, "seed": seed}


def _worker_init():
    """ProcessPoolExecutor initializer, run once at the top of every
    worker. On Linux, ask the kernel to send this worker SIGKILL the
    instant its parent (the __sweep__ process) dies — for ANY reason,
    including SIGKILL, a segfault, or the OOM killer, none of which give
    the parent a chance to tear the pool down itself. This is the hard
    backstop against the worker-orphaning that otherwise leaves a "ton of
    lingering python processes" when the sweep goes away unexpectedly.

    Workers only write flow CSVs atomically (see flow_probe.write_csv), so
    a hard SIGKILL here can't corrupt a cell — it leaves at most a .tmp.

    No-op on macOS/Windows (no PR_SET_PDEATHSIG); the parent-side
    process-group kill in live_app covers teardown there.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        import signal as _signal
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, _signal.SIGKILL, 0, 0, 0)
        # Race guard: if the parent already died between fork and this
        # prctl call, PDEATHSIG won't fire (it only triggers on a *future*
        # parent death). Detect the orphan-now case and exit immediately.
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        # A missing/odd libc must not stop the worker from doing its job;
        # the parent-side killpg remains the primary teardown path.
        pass
    # Belt-and-suspenders: also cap numba's thread pool at runtime, in case
    # NUMBA_NUM_THREADS (set at module import) was bypassed. We parallelise
    # across cells with one process per worker, so each worker's parallel
    # kernel must stay single-threaded or the box oversubscribes badly.
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _sidecar_path(csv_path, new_prefix, new_ext):
    """flow/flow_<tag>.csv -> flow/<new_prefix>_<tag>.<new_ext> (atomic-write
    targets live in the same dir as the flow CSV)."""
    base = os.path.basename(csv_path).replace("flow_", new_prefix + "_", 1)
    base = base.rsplit(".", 1)[0] + "." + new_ext
    return os.path.join(os.path.dirname(csv_path), base)


def _write_cell_meta(work_item, cell, res, params, mu_table, csv_path, wall_s):
    """Tier-1 sidecar: agnostic graph properties + full provenance +
    equilibration trace, written next to the flow CSV so the downstream
    LCC>=90% gate, the energy/Hamiltonian audit, and the equilibration
    check all have something on disk. These are raw measurements of the
    grown graph (not analysis), so persisting them keeps capture
    analysis-agnostic. Atomic, like the flow CSV: a killed worker leaves
    at most a stray .tmp.
    """
    s = cell.stats
    flow = res.flow if res is not None else {}
    tg = flow.get("t_grid")
    meta = {
        "schema": 2,
        "code_hash": _SWEEP_BUILD_INFO.get("script_hash"),
        "ec": EC, "max_degree": MAX_DEG, "prod_sweeps": PROD_SWEEPS,
        "n_probes": params.get("n_probes"),
        "lanczos_m": params.get("lanczos_m"),
        "half_window": params.get("half_window"),
        "N_eff": int(cell.N_eff),
        "lam_max_bound": float(cell.lam_max_bound),
        "wall_s": round(float(wall_s), 2),
        # parallel workers active for this run, so summed cell time can be
        # turned into an elapsed-wall-clock estimate later (wall_s is this
        # cell's own time; many ran at once).
        "workers": params.get("workers"),
        # resident memory of this worker after build+probe (MB) — feeds the
        # memory guard's RSS-vs-N model so it can predict the cost of a new
        # cell and avoid starting one that would exhaust RAM.
        "rss_mb": round(_self_rss_mb(), 1),
    }
    if tg is not None and len(tg):
        meta.update(t_lo=float(tg[0]), t_hi=float(tg[-1]), n_t=int(len(tg)))
    if work_item[0] == "torus":
        _, d, N, seed = work_item
        meta.update(kind="torus", d=d, N=N, seed=seed, mu=None)
    else:
        _, k, T, lb, N, seed = work_item
        # Ground truth: the μ the cell was ACTUALLY grown with (stashed by
        # build_cell), falling back to a base-table lookup only for safety —
        # after N-qualified drift corrections the two can differ.
        mu_used = getattr(cell, "mu_used", None)
        if mu_used is None:
            mu_used = float(mu_table.get(mu_key(k, T, lb), float("nan")))
        meta.update(kind="cell", k=k, T=T, lb=lb, N=N, seed=seed,
                    mu=float(mu_used),
                    therm_sweeps=int(cell.sweeps))
        ti = getattr(cell, "therm_info", None)
        if ti:
            meta.update(therm_converged=bool(ti.get("converged")),
                        therm_verified=ti.get("verified"),
                        therm_verify_blocks=int(ti.get("blocks", 0)),
                        therm_total_sweeps=int(ti.get("total_sweeps",
                                                      cell.sweeps)),
                        # evidence + provenance: live full-resolution verdicts
                        # are authoritative; reverify only upgrades metas
                        # WITHOUT this marker (its decimated-trace verdict is
                        # an approximation of this one).
                        therm_verify_stats=ti.get("detail"),
                        therm_verify_method="live-v2")
    if s is not None:
        meta.update(
            lcc_pct=float(s.lcc_pct), transitivity=float(s.transitivity),
            tri_per_edge=float(s.tri_per_edge),
            assortativity=float(s.assortativity),
            k_avg=float(s.k_avg), k_min=int(s.k_min), k_max=int(s.k_max),
            edges=int(s.edges), triangles=int(s.triangles),
            k_top=int(cell.peak_deg))
        if meta.get("kind") == "cell" and meta.get("k"):
            # realised-k drift, so the μ-vs-N audit can read it directly
            meta["k_err_pct"] = round(
                100.0 * (float(s.k_avg) - meta["k"]) / meta["k"], 3)
    else:
        # torus: regular lattice, connected, triangle-free
        meta.update(lcc_pct=100.0, transitivity=0.0, k_top=int(cell.peak_deg))
    dh = getattr(cell, "deg_hist", None)
    if dh is not None:
        meta["degree_hist"] = [int(x) for x in dh]
    sdsq = getattr(cell, "sum_deg_sq", 0)
    if sdsq:
        meta["sum_deg_sq"] = int(sdsq)
        if meta.get("mu") is not None and "edges" in meta:
            meta["energy_H"] = float(EC * meta["edges"] + meta["mu"] * sdsq)
    tr = getattr(cell, "therm_trace", None)
    if tr:
        meta["therm_trace"] = {
            "cols": ["sweep", "k_avg", "sum_deg_sq"],
            "rows": [[int(a), float(b), float(c)] for (a, b, c) in tr],
        }
    iso = getattr(cell, "isotropy", None)
    if iso:
        meta["isotropy"] = iso
    meta_path = _sidecar_path(csv_path, "meta", "json")
    tmp = f"{meta_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(meta, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, meta_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _write_fail_sidecar(work_item, reason, detail, csv_path):
    """Record an in-worker failure next to the CSV that was never written, so
    the heatmap can mark the cell instead of showing a bare 'not run' gap.
    `reason` is the bucket from _classify_fail_reason; `detail` is the raw
    message (it carries the peak degree for a max_degree hit). Atomic, and
    itself best-effort — a failed cell must not be made worse by a failed
    marker write.
    """
    rec = {"reason": reason, "detail": str(detail)[:300]}
    if work_item[0] == "torus":
        _, d, N, seed = work_item
        rec.update(kind="torus", d=d, N=N, seed=seed)
    else:
        _, k, T, lb, N, seed = work_item
        rec.update(kind="cell", k=k, T=T, lb=lb, N=N, seed=seed)
    path = _sidecar_path(csv_path, "fail", "json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _clear_fail_sidecar(csv_path):
    """Remove any stale fail sidecar for a cell that has now succeeded, so a
    cell that failed on a previous run and succeeds on a rerun stops being
    marked as failed."""
    try:
        os.remove(_sidecar_path(csv_path, "fail", "json"))
    except OSError:
        pass


def _write_mu_failures(grid_ktl, mu_table, cal_failures, path):
    """Write the calibration-failure ledger: one entry per (k, T, lb) in the
    grid that has no μ in the table (so every N for it was dropped before it
    could run). Reason comes from this run's calibration errors when present,
    so a max-degree cap hit during the calibration build is labelled as such.
    Rewritten every run from the current grid-vs-table state, so a cell that
    starts calibrating successfully drops out of the ledger automatically.
    """
    out = {}
    for (k, T, lb) in grid_ktl:
        key = mu_key(k, T, lb)
        if key in mu_table:
            continue
        msg = cal_failures.get(key, "no μ entry (calibration not run or failed)")
        out[key] = {"k": k, "T": T, "lb": lb,
                    "reason": _classify_fail_reason(msg),
                    "detail": str(msg)[:300]}
    write_status_json(path, out)   # generic atomic JSON writer (tmp + rename)


def _run_one(work_item, params, mu_table):
    """Worker entry point: run one torus ref or sweep cell to
    completion, write its flow CSV, return a small result dict.

    Lives at module top-level so ProcessPoolExecutor can pickle it.
    Returns no big arrays — only the parameter tuple, success flag,
    wall time, and (for cells) shape/ds_lo/ds_hi. The flow CSV is
    persisted to disk; the dashboard reads it from there.

    Parent has already pinned BLAS to 1 thread per worker via env
    vars set at module import time, so we don't oversubscribe.
    """
    t0 = time.time()
    tag = _work_tag(work_item)
    seed = _work_seed(work_item)
    csv_path = _csv_path_for(work_item)
    print(f"[{tag}] starting", flush=True)
    try:
        if work_item[0] == "torus":
            _, d, N, _seed = work_item
            L = torus_L_for_N(N, d)
            cell = build_torus_cell(d, L, seed)
        else:
            _, k, T, lb, N, _seed = work_item
            cell = build_cell(k, T, lb, N, seed, mu_table)

        # Field-isotropy summary on the cell's OWN Laplacian (already built for
        # the flow probe — no graph is re-evolved). Stored in the meta sidecar
        # and fed to the auto-refreshed isotropy heatmap, so uniformity is
        # measured on every graph the sweep produces. Best-effort: a hiccup
        # here must never fail a finished cell.
        try:
            from physics_tests import isotropy as _iso
            cell.isotropy = _iso.cell_isotropy_summary(cell.L_csr, seed=seed)
        except Exception as _ie:
            print(f"[{tag}] isotropy summary skipped: "
                  f"{type(_ie).__name__}: {_ie}", flush=True)

        res = run_flow_test(cell,
                            n_probes=params["n_probes"],
                            lanczos_m=params["lanczos_m"],
                            half_window=params["half_window"])
        write_flow_csv(res, csv_path)
        # This cell produced data — drop any fail marker left by a previous
        # run so it stops showing as failed in the heatmap.
        _clear_fail_sidecar(csv_path)
        # Tier-2: persist the SLQ Ritz quadrature so d_s(t) can be
        # re-extracted (re-grid / deeper-UV / re-window) without re-running
        # SLQ. Tier-1: enriched topology + provenance + equilibration trace.
        # Both best-effort — a sidecar hiccup must not fail a finished cell.
        try:
            write_quad_npz(res, _sidecar_path(csv_path, "quad", "npz"))
        except Exception as _qe:
            print(f"[{tag}] quad sidecar skipped: "
                  f"{type(_qe).__name__}: {_qe}", flush=True)
        try:
            _write_cell_meta(work_item, cell, res, params, mu_table,
                             csv_path, wall_s=time.time() - t0)
        except Exception as _me:
            print(f"[{tag}] meta sidecar skipped: "
                  f"{type(_me).__name__}: {_me}", flush=True)

        if work_item[0] == "cell":
            shape = classify_shape(res.flow["d_s_mean"],
                                   res.flow["in_window"])
            in_w = res.flow["in_window"]
            d_in = res.flow["d_s_mean"][in_w]
            d_in = d_in[np.isfinite(d_in)]
            ds_lo = float(d_in.min()) if len(d_in) else math.nan
            ds_hi = float(d_in.max()) if len(d_in) else math.nan
        else:
            shape = None
            ds_lo = ds_hi = None

        wall = time.time() - t0
        msg = f"[{tag}] done in {wall:.1f}s"
        if shape is not None:
            msg += f", shape={shape}, d_s ∈ [{ds_lo:.2f}, {ds_hi:.2f}]"
        print(msg, flush=True)

        return {"work_item": work_item, "ok": True, "wall_s": wall,
                "shape": shape, "ds_lo": ds_lo, "ds_hi": ds_hi}
    except Exception as e:
        wall = time.time() - t0
        detail = f"{type(e).__name__}: {e}"
        reason = _classify_fail_reason(str(e))
        print(f"[{tag}] FAILED in {wall:.1f}s: {detail}", flush=True)
        # Persist why, so the heatmap can mark this cell (max_degree cap hit
        # vs other error) instead of leaving an unexplained gap.
        _write_fail_sidecar(work_item, reason, detail, csv_path)
        return {"work_item": work_item, "ok": False, "wall_s": wall,
                "reason": reason, "error": detail}


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=str, default=None,
                    help="Comma-separated k values (e.g. '6,8,9'). "
                         "Omitted → the [grid] k from main.toml.")
    ap.add_argument("--T", type=str, default=None,
                    help="Temperature(s), comma-separated. Omitted → the "
                         "[grid] T from main.toml. T=0 is a useful "
                         "ground-state anchor (no thermal noise); T=0.005 "
                         "is the typical low-T basin.")
    ap.add_argument("--lb", type=str, default=None,
                    help="Comma-separated lb values "
                         "(e.g. '0.93,0.94,0.95,0.99')")
    ap.add_argument("--lb-file", type=str, default=None,
                    help="Path to file with one lb value per line "
                         "(alternative to --lb)")
    ap.add_argument("--N", type=str, default=None,
                    help="Comma-separated N values (e.g. '16000,64000'). "
                         "Omitted → the [grid] N from main.toml.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Single base seed (back-compat). Ignored if "
                         "--seeds is given; omitted → the [grid] seeds "
                         "from main.toml (seed .. seed+n_seeds-1).")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds to run per cell "
                         "(e.g. '42,43,44,45,46'). Each seed produces its "
                         "own flow_*.csv. Overrides --seed.")
    ap.add_argument("--shape-refresh-secs", type=float, default=10.0,
                    help="Minimum seconds between background regenerations "
                         "of the heatmaps/histogram while sweeping. Plots "
                         "render off the dispatch thread, so this only "
                         "bounds plot freshness, never sweep speed.")
    ap.add_argument("--n-probes", type=int, default=60)
    ap.add_argument("--lanczos-m", type=int, default=300)
    ap.add_argument("--half-window", type=int, default=10)
    ap.add_argument("--no-torus", action="store_true",
                    help="Skip torus reference per panel")
    ap.add_argument("--force", action="store_true",
                    help="Re-run all cells even if their flow_*.csv "
                         "already exists. Default is to skip cached "
                         "cells, which makes the sweep idempotent so "
                         "the dashboard can spawn it safely.")
    ap.add_argument("--watch-reload", action="store_true",
                    help="Hot-reload mode: between cells, check for "
                         "changes to any source file under src/. If "
                         "anything changed, restart the sweep via "
                         "os.execv (same PID, fresh interpreter). "
                         "Cached-cell skipping means the restart "
                         "continues from where we stopped without "
                         "re-doing work. Useful for live tweaking of "
                         "shape thresholds, status JSON contents, or "
                         "log formats during a long sweep. Note that "
                         "the currently-running cell completes with "
                         "the OLD code; reload happens at the next "
                         "cell boundary.")
    ap.add_argument("--torus-d", type=str, default="4",
                    help="Torus dimension(s) for reference, comma-"
                         "separated (e.g. '2,3,4,5'). Each becomes a "
                         "separate dashed reference curve in the "
                         "dashboard, anchoring basin curves against "
                         "known-flat geometries. Torus refs run FIRST "
                         "so they appear in the dashboard before any "
                         "basin cell completes.")
    ap.add_argument("--out-prefix", type=str, default="lb_sweep")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of parallel worker processes for the "
                         "torus + cell sweep. Default 0 means use all "
                         "cores granted by sched_getaffinity (e.g. 16 "
                         "on a 16-core box, less under cgroup/taskset). "
                         "Each worker runs one SLQ at a time; BLAS is "
                         "pinned to 1 thread per worker via env vars "
                         "set before numpy import. Reduce this if you "
                         "OOM on big-N items running concurrently.")
    ap.add_argument("--status-json", type=str,
                    default="lb_sweep_status.json",
                    help="Path for live sweep status JSON, written after "
                         "each cell completes. The dashboard polls this "
                         "file to show progress / ETA / RSS while the "
                         "sweep runs. Atomic writes via rename.")
    ap.add_argument("--show-plan", action="store_true",
                    help="Print the cell list + estimated runtime, then exit")
    args = ap.parse_args(argv)
    _report_thread_caps()

    # Build the selected graph-growth engine's native library ONCE here, in the
    # parent, before any calibration or sweep worker pool forks — so the many
    # parallel build_graph calls find it already compiled and never race to
    # build it. No-op for numba / when no compiler is present (runs on numba).
    from core.graph_builder import ensure_engine_built
    _engine = ensure_engine_built(log=print)
    print(f"  [engine] graph growth backend: {_engine}", flush=True)

    # Grid source: explicit CLI args win; anything omitted falls back to the
    # [grid] in main.toml (via core.project_constants). main.py launches the
    # sweep with NO grid args, so the normal path reads main.toml here — the
    # grid lives in exactly one place and is never marshalled through main.
    from core import project_constants as _cfg
    k_vals = parse_ints(args.k) if args.k else list(_cfg.K_ALL)
    T_vals = parse_floats(args.T) if args.T else list(_cfg.T_ALL)
    N_vals = parse_ints(args.N) if args.N else list(_cfg.N_ALL)
    # Seeds to run per cell. --seeds (a list) is the normal explicit path;
    # --seed (single) is kept for older invocations; omit both → main.toml's
    # SEEDS. Each seed is an independent unit of work with its own flow_*.csv.
    if args.seeds:
        seed_list = parse_ints(args.seeds)
    elif args.seed is not None:
        seed_list = [args.seed]
    else:
        seed_list = list(_cfg.SEEDS)
    if not seed_list:
        raise SystemExit("no seeds to run (empty --seeds).")
    if args.lb_file:
        with open(args.lb_file) as f:
            lb_vals = [float(line.strip()) for line in f
                       if line.strip() and not line.strip().startswith("#")]
    elif args.lb:
        lb_vals = parse_floats(args.lb)
    else:
        lb_vals = list(_cfg.LB_ALL)

    # Cell grid: (k, T, lb, N). Iteration order is cheap-first — N is
    # the outermost axis so we exhaust every (T, k, lb) combo at the
    # current N before paying the next ×4 memory/time jump. Within a
    # fixed N, lb cycles fastest (innermost), then k, then T. The
    # progression "lowest N, lowest k, lowest lb → crank lb high → crank
    # k high → bump N to next ×4" matches what the operator sees in the
    # heartbeat: a full (k, lb) plane fills in for each N rung before
    # the next one starts.
    cells = [(k, T, lb, N)
             for N in N_vals
             for T in T_vals
             for k in k_vals
             for lb in lb_vals]
    print(f"  plan: {len(k_vals)} k × {len(T_vals)} T × "
          f"{len(N_vals)} N × {len(lb_vals)} lb = {len(cells)} basin cells"
          f" × {len(seed_list)} seed(s) = "
          f"{len(cells) * len(seed_list)} cell-runs",
          flush=True)
    if not args.no_torus:
        torus_dims_preview = [int(x) for x in str(args.torus_d).split(",")
                              if x.strip()]
        n_torus = len(torus_dims_preview) * len(N_vals)
        print(f"        + {n_torus} torus reference cells "
              f"({len(torus_dims_preview)} dim(s) × {len(N_vals)} N)",
              flush=True)

    # Estimate runtime: SLQ at ~5 min for N=64k, scales roughly linearly
    # in N at large N, sub-linear at small N. Rough bound:
    est_min = sum(0.3 + N / 50000 for (_, _, _, N) in cells)
    if not args.no_torus:
        torus_dims_preview = [int(x) for x in str(args.torus_d).split(",")
                              if x.strip()]
        for d in torus_dims_preview:
            est_min += sum(0.3 + N / 50000 for N in N_vals)
    print(f"        estimated wall time ~{est_min:.0f} min "
          f"(very rough; depends on hardware)",
          flush=True)
    if args.show_plan:
        for c in cells:
            print(f"          {c}")
        return

    # Load μ table
    mu_table = load_mu_table()
    if not mu_table:
        print(f"  {MU_JSON} not found or empty — will calibrate from scratch.",
              flush=True)
        mu_table = {}

    # Identify (k, T, lb) tuples that need μ calibration
    needed_keys = {(k, T, lb) for (k, T, lb, _) in cells}
    missing = sorted({(k, T, lb) for (k, T, lb) in needed_keys
                      if mu_key(k, T, lb) not in mu_table})
    cal_failures = {}        # mu_key -> error message, filled if we calibrate
    if missing:
        print(f"\n  μ-calibration missing for {len(missing)} (k, T, lb) "
              f"tuple(s) — auto-calibrating before sweep starts:",
              flush=True)
        for c in missing[:10]:
            print(f"      {c}", flush=True)
        if len(missing) > 10:
            print(f"      ... and {len(missing)-10} more", flush=True)
        # Surface a 'calibrating' phase so the live page shows it instead of
        # sitting on "Initialising…" for the whole (potentially long)
        # calibration pass that runs before the sweep proper.
        write_status_json(args.status_json, build_status(
            phase="calibrating",
            started_at=time.time(),
            n_total=len(cells) * len(seed_list), n_done=0, n_failed=0,
            calibration={"n_targets": len(missing), "n_cal": MU_N_CAL},
        ))
        # Do it. calibrate_missing parallelises by default and persists
        # the table to disk after each successful cell, so a partial
        # run is recoverable.
        mu_table, n_done, n_failed, cal_failures = calibrate_missing(
            missing, ec=EC, verbose=True)
        if n_failed:
            still_missing = [c for c in missing
                             if mu_key(*c) not in mu_table]
            print(f"\n  ⚠ {n_failed} calibration(s) failed; "
                  f"{len(still_missing)} cell(s) cannot run:", flush=True)
            for c in still_missing[:5]:
                print(f"      {c}", flush=True)
            # Drop affected cells from the run rather than aborting —
            # the user may have specified a wide grid and a single bad
            # corner shouldn't kill the whole sweep.
            before = len(cells)
            cells = [(k, T, lb, N) for (k, T, lb, N) in cells
                     if mu_key(k, T, lb) in mu_table]
            print(f"  proceeding with {len(cells)}/{before} cells",
                  flush=True)
            if not cells:
                # Still record why before bailing, so the heatmap explains
                # the empty result instead of just vanishing.
                _write_mu_failures(needed_keys, mu_table, cal_failures,
                                   MU_FAILURES_JSON)
                raise SystemExit("no calibratable cells left.")
        else:
            print(f"  all {len(missing)} calibrations succeeded; "
                  f"sweep starting now.", flush=True)

    # Record (or clear) the calibration-failure ledger so shape_analysis can
    # mark cells that were dropped before they could run. Derived from the
    # full grid vs the μ table, so it's correct on resume and self-clears
    # when a previously-failed cell calibrates. `cal_failures` is empty when
    # no calibration ran this session.
    _write_mu_failures(needed_keys, mu_table, cal_failures, MU_FAILURES_JSON)
    # Red banner + status-JSON count for any calibration drops, past or
    # present — quiet ledgers get missed; unreachable grid cells shouldn't.
    _MU_FAIL_COUNT[0] = _mu_failures_banner()

    t_total = time.time()
    classifications = []  # rows: (k, T, lb, N, seed, shape, ds_min, ds_max)
    n_done = 0
    n_failed = 0

    # Hot-reload baseline: capture source file hashes now, after all
    # imports are done. We compare against this between cells; any
    # change triggers an os.execv restart. Disabled by default — opt
    # in via --watch-reload for live development.
    src_baseline = snapshot_source_hashes() if args.watch_reload else None
    if args.watch_reload:
        print(f"  --watch-reload: monitoring "
              f"{len(src_baseline)} source file(s) for changes",
              flush=True)

    # Initial status: phase=running, nothing done yet
    write_status_json(args.status_json, build_status(
        phase="running",
        started_at=t_total,
        n_total=len(cells) * len(seed_list), n_done=0, n_failed=0,
        next_cell=cells[0] if cells else None,
    ))

    # Ensure the flow/ subdirectory exists. All per-cell CSVs are
    # written there to keep the project root clean as the dataset
    # grows. Idempotent — no-op if already present.
    os.makedirs(FLOW_DIR, exist_ok=True)

    # ── Build unified work list ──────────────────────────────────────
    # Both torus refs and sweep cells are independent SLQ-bound items;
    # we used to run them in two serial nested for-loops. Pooling them
    # together lets a 16-core box actually use all 16 cores instead of
    # one, especially while the slowest item (e.g. L=45 4D, ~10 min)
    # is grinding on its single worker.
    #
    # Each seed is its own work-item with its own CSV, so the cached-skip
    # check below is per-(item, seed): a cell with 3 of 5 seeds already on
    # disk resubmits only the 2 missing ones, and a fully-done cell skips
    # all its seeds. That makes multi-seed runs resume cleanly.
    work = []
    cached_cell_count = 0

    if not args.no_torus:
        torus_dims = [int(x) for x in str(args.torus_d).split(",")
                      if x.strip()]
        for d_torus in torus_dims:
            for N in N_vals:
                for seed in seed_list:
                    w = ("torus", d_torus, N, seed)
                    if (os.path.exists(_csv_path_for(w))
                            and not args.force):
                        L = torus_L_for_N(N, d_torus)
                        print(f"[torus ref] {d_torus}D L={L} "
                              f"(N={L**d_torus}) seed={seed} "
                              f"— cached, skipping", flush=True)
                        continue
                    work.append(w)

    for (k, T, lb, N) in cells:
        for seed in seed_list:
            w = ("cell", k, T, lb, N, seed)
            if (os.path.exists(_csv_path_for(w))
                    and not args.force):
                print(f"[cell] k={k} T={T} lb={lb} N={N} seed={seed} "
                      f"— cached, skipping", flush=True)
                cached_cell_count += 1
                continue
            work.append(w)

    # Smallest-first ordering: tiny items finish in seconds and pop up
    # in the dashboard right away, while the giant L=45 ref runs in
    # parallel underneath. Operator gets early signal that the run is
    # healthy instead of waiting 10 minutes for the first row.
    work.sort(key=_work_cost_estimate)

    # ── ETA + auto-shape tracking ────────────────────────────────────
    # eta_done_samples: (N, wall_s) for every finished work-item, used to
    # fit the cost-vs-N trend. eta_remaining: how many items of each N are
    # still to finish.
    # The shape analysis (heatmaps + histogram + summary) is regenerated in
    # the background by a dedicated thread, throttled to at most once every
    # SHAPE_REFRESH_SECS, so plotting never blocks worker dispatch. The
    # dispatch loop only *requests* refreshes; the thread coalesces them.
    eta_done_samples = _load_wall_history(FLOW_DIR)
    if eta_done_samples:
        try:
            _assume = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            _assume = os.cpu_count() or 1
        _core_s, _elapsed_s, _n_w, _n_tot = _load_compute_invested(
            FLOW_DIR, assume_workers=_assume)
        print(f"  ETA warm-start: recovered {len(eta_done_samples)} prior "
              f"cell wall-time(s) from {FLOW_DIR}/meta_*.json — the cost-vs-N "
              f"trend persists across restarts", flush=True)
        print(f"  total compute: {_fmt_hms(_core_s)} of CORE time "
              f"(sum of per-cell wall over {_n_tot} cells — cells run in "
              f"parallel, so this is core-hours, NOT elapsed time)",
              flush=True)
        if _n_w == _n_tot:
            _cov = "from each cell's recorded worker count"
        elif _n_w == 0:
            _cov = (f"no cell recorded its worker count yet, so all assume "
                    f"{_assume}-way parallel — a rough estimate")
        else:
            _cov = (f"{_n_tot - _n_w} older cell(s) predate worker-count "
                    f"recording and assume {_assume}-way")
        print(f"  estimated real elapsed: ~{_fmt_hms(_elapsed_s)} "
              f"(core time ÷ workers per cell; {_cov})", flush=True)
    eta_remaining = Counter(_work_N(w) for w in work)
    SHAPE_REFRESH_SECS = float(getattr(args, "shape_refresh_secs", 10.0))
    # Periodic redraw heartbeat from main.toml ([dashboard].redraw_interval_s,
    # default 10s; 0 = off). When on, it also governs the min-interval so the
    # configured cadence is what you actually get (a small value isn't blocked
    # by the larger event-throttle); when off, fall back to the CLI throttle
    # and render only when cells land.
    redraw_s = REDRAW_INTERVAL_S
    throttle_s = redraw_s if redraw_s > 0 else SHAPE_REFRESH_SECS
    shape_refresher = _ShapeRefresher(min_interval_s=throttle_s,
                                      periodic_s=redraw_s)
    if redraw_s > 0:
        print(f"  plots redraw at least every {redraw_s:g}s "
              f"(config [dashboard].redraw_interval_s; 0 disables)", flush=True)
    else:
        print("  periodic plot redraw disabled "
              "([dashboard].redraw_interval_s = 0); plots refresh on new data",
              flush=True)

    n_done = cached_cell_count
    n_failed = 0
    classifications = []

    # Worker count. Precedence: --workers flag > [compute].workers in
    # main.toml > auto (all granted logical cores). With BLAS *and* numba
    # pinned to 1 thread per worker (env at module import), each worker is a
    # single-threaded process, so the parallelism is exactly the worker count.
    # On a memory- or cache-bound load, fewer workers than logical cores is
    # often FASTER (less contention for RAM bandwidth and shared L3) and draws
    # less power — set [compute].workers to your physical core count, or lower,
    # to dial resource use down. 0 anywhere = auto.
    if args.workers > 0:
        n_workers = args.workers
    elif WORKERS > 0:
        n_workers = WORKERS
    else:
        try:
            n_workers = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_workers = os.cpu_count() or 1
    n_workers = min(n_workers, max(1, len(work)))   # don't over-allocate

    n_torus_work = sum(1 for w in work if w[0] == "torus")
    n_cell_work = sum(1 for w in work if w[0] == "cell")
    print(f"\n  parallel dispatch: {n_workers} worker(s), "
          f"{len(work)} item(s) "
          f"({n_cell_work} cell + {n_torus_work} torus); "
          f"{cached_cell_count} cell(s) cached",
          flush=True)

    # Initial status — phase=running, in_flight empty until pool starts
    write_status_json(args.status_json, build_status(
        phase="running",
        started_at=t_total,
        n_total=len(cells) * len(seed_list), n_done=n_done, n_failed=n_failed,
        in_flight=[],
    ))

    if not work:
        # Everything was cached — nothing to dispatch, fall through
        # to classifications writing.
        print("  all items cached; nothing to compute.", flush=True)
    else:
        # Use fork start method so workers inherit the parent's already-
        # loaded modules (numpy, core modules, etc) without re-importing.
        # BLAS env vars from the parent are inherited too. On non-fork
        # platforms the default context is fine — they'll spawn fresh
        # interpreters, slightly slower startup but otherwise correct.
        ctx_name = "fork" if "fork" in mp.get_all_start_methods() else None
        ctx = mp.get_context(ctx_name) if ctx_name else None

        # in_flight: future -> work_item, drained as as_completed yields
        in_flight = {}
        params = {
            "n_probes": args.n_probes,
            "lanczos_m": args.lanczos_m,
            "half_window": args.half_window,
            # recorded into each cell's meta so a later run can estimate REAL
            # elapsed wall-clock (= summed cell time ÷ workers), which summed
            # cell time alone can't give once cells run in parallel.
            "workers": n_workers,
        }
        last_source_check = time.time()
        # Most recent CELL completion — torus completions don't touch
        # this. Persisted across loop iterations so a torus finishing
        # between cells doesn't blank the dashboard's "last:" row.
        last_cell_persistent = None

        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_worker_init)
            try:
                # Submit-on-demand pattern: keep at most `n_workers`
                # futures in flight at any moment by priming the pool
                # with the first batch and submitting the next item
                # only when one completes. This way `in_flight` mirrors
                # what's actually running — important for the dashboard,
                # which would otherwise display 1200+ "in flight" cells
                # when only 16 are really executing.
                work_iter = iter(work)
                _lookahead = []
                _MIN_FREE = MIN_FREE_GB * (1024 ** 3)
                _rss_model = _make_rss_model(_load_mem_history(FLOW_DIR))
                _guard_on = (MIN_FREE_GB > 0
                             and _free_mem_bytes() is not None)
                _last_hold = [0.0]
                _last_eta_warn = [0.0]
                if _guard_on:
                    print(f"  memory guard on: keeping ≥{MIN_FREE_GB:g} GB "
                          f"free ({(_free_mem_bytes() or 0) / 2**30:.1f} GB "
                          f"now). A cell that would breach it waits for "
                          f"running cells to finish first, so the sweep "
                          f"throttles itself down instead of being OOM-killed "
                          f"(config [compute].min_free_gb).", flush=True)
                elif MIN_FREE_GB > 0:
                    print("  memory guard requested but free memory can't be "
                          "read here; proceeding without it.", flush=True)

                def _peek():
                    if not _lookahead:
                        try:
                            _lookahead.append(next(work_iter))
                        except StopIteration:
                            return None
                    return _lookahead[0]

                def _submit_one():
                    """Submit the next item, unless the memory guard says a new
                    cell would drop free RAM below the margin — then hold it
                    back (it stays buffered) and let running cells finish. When
                    nothing is in flight we submit regardless: waiting can't
                    free more, so a single huge cell runs solo rather than
                    stalling the sweep."""
                    item = _peek()
                    if item is None:
                        return False
                    if _guard_on and in_flight:
                        free = _free_mem_bytes()
                        if (free is not None
                                and free - _rss_model(_work_N(item)) < _MIN_FREE):
                            now = time.time()
                            if now - _last_hold[0] > 20:
                                print(f"  [mem-guard] {free / 2**30:.1f} GB "
                                      f"free; holding new cells (keep ≥"
                                      f"{MIN_FREE_GB:g} GB) — waiting for "
                                      f"{len(in_flight)} running to finish.",
                                      flush=True)
                                _last_hold[0] = now
                            return False
                    _lookahead.pop(0)
                    in_flight[executor.submit(
                        _run_one, item, params, mu_table)] = item
                    return True

                def _fill():
                    """Submit until the pool is full, work runs out, or the
                    memory guard holds us back — so we ramp back up to full
                    concurrency once running cells free their RAM."""
                    while len(in_flight) < n_workers and _submit_one():
                        pass

                _fill()

                while in_flight:
                    done, _pending = concurrent.futures.wait(
                        in_flight.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        w = in_flight.pop(fut)
                        try:
                            result = fut.result()
                        except Exception as e:
                            # Worker process died unexpectedly
                            # (segfault, OOM-kill, etc). Don't let one
                            # bad item kill the sweep; record it as
                            # failed and keep going.
                            print(f"[{_work_tag(w)}] WORKER CRASHED: "
                                  f"{type(e).__name__}: {e}", flush=True)
                            if w[0] == "cell":
                                _, k, T, lb, N, seed = w
                                classifications.append(
                                    (k, T, lb, N, seed, "FAILED",
                                     math.nan, math.nan))
                                n_failed += 1
                            eta_remaining[_work_N(w)] -= 1
                            _fill()
                            continue

                        # Item finished (cell or torus): update the ETA
                        # model. wall_s is real compute time; record it
                        # even for failed cells (timing is still timing).
                        eta_remaining[_work_N(w)] -= 1
                        _wall = result.get("wall_s")
                        if _wall and _wall > 0:
                            eta_done_samples.append((_work_N(w), _wall))

                        if w[0] == "cell":
                            _, k, T, lb, N, seed = w
                            if result["ok"]:
                                n_done += 1
                                classifications.append(
                                    (k, T, lb, N, seed,
                                     result["shape"],
                                     result["ds_lo"], result["ds_hi"]))
                                last_cell_persistent = {
                                    "k": k, "T": T, "lb": lb, "N": N,
                                    "seed": seed,
                                    "shape": result["shape"],
                                    "ds_min": result["ds_lo"],
                                    "ds_max": result["ds_hi"],
                                    "wall_s": round(result["wall_s"], 1)}
                            else:
                                n_failed += 1
                                classifications.append(
                                    (k, T, lb, N, seed, "FAILED",
                                     math.nan, math.nan))
                                last_cell_persistent = {
                                    "k": k, "T": T, "lb": lb, "N": N,
                                    "shape": "FAILED",
                                    "error": result["error"][:200]}
                        # Torus completions: log only, don't touch
                        # n_done or last_cell_persistent (last_cell is
                        # reserved for sweep cells; torus refs aren't
                        # in the dashboard table).

                        # Refill the pool (memory-permitting) with pending
                        # items — back up to full concurrency as RAM frees.
                        _fill()

                        in_flight_list = [_work_in_flight_dict(ww)
                                          for ww in in_flight.values()]

                        # Periodic source-change check for --watch-reload.
                        # Done at completion boundaries, every ~5s, so we
                        # don't poll filesystem on every tick.
                        if (args.watch_reload
                                and time.time() - last_source_check > 5):
                            current_hashes = snapshot_source_hashes()
                            changed = detect_changes(src_baseline,
                                                     current_hashes)
                            if changed:
                                reason = ", ".join(
                                    os.path.basename(p)
                                    for p in changed[:3])
                                if len(changed) > 3:
                                    reason += f" (+{len(changed)-3} more)"
                                # Cancel pending futures and JOIN the
                                # in-flight ones before we execv. os.execv
                                # replaces this process image in place, so
                                # any worker still running at that moment
                                # would be orphaned (PDEATHSIG can't help —
                                # the parent PID doesn't die, it's reused).
                                # wait=True blocks only until the few
                                # in-flight cells finish; everything already
                                # on disk is skipped by cached-skip on the
                                # restart, so the wait is bounded and cheap.
                                executor.shutdown(wait=True,
                                                  cancel_futures=True)
                                restart_self(
                                    reason, args.status_json,
                                    build_status(
                                        phase="running",
                                        started_at=t_total,
                                        n_total=len(cells) * len(seed_list),
                                        n_done=n_done,
                                        n_failed=n_failed,
                                        in_flight=in_flight_list,
                                    ))
                                # restart_self does not return on success
                            last_source_check = time.time()

                        eta = estimate_eta_seconds(
                            eta_done_samples,
                            list(eta_remaining.elements()),
                            n_workers)
                        _status = build_status(
                            phase="running",
                            started_at=t_total,
                            n_total=len(cells) * len(seed_list),
                            n_done=n_done, n_failed=n_failed,
                            last_cell=last_cell_persistent,
                            in_flight=in_flight_list,
                            eta_sec_override=eta,
                            compute_spent_sec=sum(w for _, w in eta_done_samples),
                            cells_done_session=max(0, n_done - cached_cell_count),
                        )
                        write_status_json(args.status_json, _status)
                        # Surface grid infeasibility on the console too, not
                        # just in the polled JSON — throttled so it nags
                        # rather than spams.
                        if (_status.get("eta_warning")
                                and time.time() - _last_eta_warn[0] > 600):
                            print(f"  ⚠ {_status['eta_warning']}", flush=True)
                            _last_eta_warn[0] = time.time()

                        # Ask the background thread to regenerate plots.
                        # Non-blocking: it renders off the dispatch path and
                        # at most once per SHAPE_REFRESH_SECS, so finished
                        # workers never idle waiting on matplotlib.
                        shape_refresher.request()

                        # μ-vs-N drift audit, on the same slow throttle as
                        # the convergence pass. Runs in THIS thread on
                        # purpose: it mutates the live mu_table dict, which
                        # is pickled into every subsequent submit, so
                        # corrections must land between submits, not in a
                        # background thread racing them.
                        _maybe_update_mu_drift(mu_table)
            finally:
                executor.shutdown(wait=True)
                # Quiesce the background plotter before the synchronous
                # final refresh in _finalize_sweep, so they don't overlap.
                shape_refresher.stop()
        except KeyboardInterrupt:
            print("\n  Ctrl-C received; cancelling pending futures…",
                  flush=True)
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            # Still write out everything for the cells that DID finish:
            # plots + classifications CSV + a terminal status. Without this
            # a stopped sweep leaves only flow/ and no heatmaps.
            try:
                _finalize_sweep(args, classifications, cells, seed_list,
                                t_total, n_done, n_failed,
                                phase="interrupted")
            except Exception as e:
                print(f"  [finalize] skipped on interrupt: "
                      f"{type(e).__name__}: {e}", flush=True)
            return

    _finalize_sweep(args, classifications, cells, seed_list, t_total,
                    n_done, n_failed, phase="done")


if __name__ == "__main__":
    main()
