"""
core/graph_builder.py — Graph build + μ calibration
==========================================
build_graph runs adaptive thermalisation with p0's drift+noise
criterion. calibrate_mu binary-searches the degree-penalty multiplier
at small N so k_actual ≈ k. ensure_mu_calibrated parallelises the
calibration sweep across the full (k, T, lb) grid.
"""

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from core.physics_engine import PhysicsEngine
from metrics import get_graph_stats

from core.project_constants import (
    K_ALL, T_ALL, LB_ALL, MAX_DEG, SEEDS,
    MU_N_CAL, MU_LO, MU_HI, MU_TOL, MU_MAX_ITER,
    THERM_WINDOW, THERM_DRIFT_TOL, THERM_MIN_BASE, THERM_MAX_BASE,
    THERM_CONFIRM, PROD_SWEEPS,
)
from core.disk_io import _wlog, log, mu_key, load_mu_table, save_mu_table


def _mu_guess(k, T, lb, ec):
    """Closed-form μ starting point — lifted from p0._mu_guess.

    The four lb-bracketed coefficients (0.509, 0.543, 0.594, 0.628 in
    the temperature-driven branch) are empirical: fit by running the
    full calibrate_mu binary search on the (k, T, lb) grid at
    N=MU_N_CAL, then taking the median ratio of the converged μ to
    abs(ec)/2 within each lb bracket. They get the binary search to
    target inside ~3-4 iterations on average instead of ~12, which
    matters because calibrate_mu is the bottleneck phase of the
    sweep.

    Provenance / regenerating: there's no automated job for this.  If
    you change MU_N_CAL or ec, run calibrate_mu over a representative
    slice of the (k, T, lb) grid with the lb-bracket coefficients all
    set to 0.5 (so _mu_guess returns the ratio scaled only by 1/k),
    then read off the median μ·k/abs(ec) per bracket from the resulting
    mu_table.json. The values currently here were fit at MU_N_CAL=4000,
    ec=-1.0, and have been stable across two MU_TOL changes since.

    The T=0 branch falls back to closed-form expressions: ec/4 per node
    at high lb (the energy of one extra edge in a saturated cluster),
    or |ec|/(2k-1) at low lb (matching a uniform-graph mean-field).
    These are exact in their respective limits, no fitting involved.
    """
    ec4 = abs(ec) / 4.0
    ratio = abs(ec) / 2.0
    if T > 0:
        if lb >= 0.9:
            C = 0.509 * ratio
        elif lb >= 0.75:
            C = 0.543 * ratio
        elif lb >= 0.5:
            C = 0.594 * ratio
        else:
            C = 0.628 * ratio
        return C / max(k, 1)
    return ec4 / max(k, 1) if lb >= 0.9 else abs(ec) / (2 * max(k, 1) - 1)


def _thermalise(eng, N, tmin, tmax, noise_tol, log_tag=None, trace_out=None):
    """Sliding-window drift+noise convergence loop.

    If `trace_out` is a list, an equilibration trace is appended to it:
    (sweep, k_avg, sum_deg_sq) at ~120 points across the run. Both
    quantities come from node_degrees alone (no neighbour scan), so this
    stays cheap and keeps the loop testable with a minimal fake engine.
    sum_deg_sq is the degree-concentration term of the Hamiltonian
    (H = ec·E + μ·Σd²), so the trace lets equilibration be audited post-hoc
    on the *structural* observable, not just mean degree.

    Iterates the engine 1 sweep (N MC attempts) at a time, tracking
    mean degree per sweep in `hist`. After at least `tmin` sweeps and
    once `hist` is `THERM_WINDOW` long, every sweep tests whether the
    last THERM_WINDOW means have:
        • |slope of linear fit| < THERM_DRIFT_TOL  (no trend)
        • std < noise_tol                          (small fluctuation)
    Both conditions must hold for THERM_CONFIRM consecutive sweeps to
    declare convergence; a single failure resets the confirm counter.

    Returns the number of sweeps actually run. Engine state is mutated
    in place; the caller reads eng.node_degrees, eng.peak_degree, etc.

    Pulled out of build_graph so the convergence criterion is testable
    without spinning up a real PhysicsEngine — tests can pass a fake
    object exposing iterate(steps), node_degrees, peak_degree.

    k_top is the highest single-node degree observed at ANY point
    during the entire MC trajectory. Tracked inside the engine via
    the njit attempt_toggle — eng.peak_degree updates in O(1) per
    accepted edge addition. Faster than scanning node_degrees[:N].max()
    each sweep AND captures intermediate peaks within a sweep that
    an end-of-sweep scan would miss.
    """
    hist = []
    deg_confirms = 0
    sweeps = 0
    # Trace cadence: at most ~120 points over the whole run, so the sidecar
    # stays tiny regardless of tmax.
    trace_every = max(1, tmax // 120)

    last_log_t = time.time()
    SWEEP_LOG_INTERVAL_S = 60.0
    sweep_chunk_t0 = time.time()
    sweep_chunk_start = 0
    for sw in range(1, tmax + 1):
        eng.iterate(steps=N)           # 1 sweep = N single-step MC attempts
        kavg = float(np.mean(eng.node_degrees[:N]))
        hist.append(kavg)
        sweeps = sw

        if trace_out is not None and (sw == 1 or sw % trace_every == 0):
            _deg = eng.node_degrees[:N].astype(np.int64)
            trace_out.append((sw, kavg, float((_deg * _deg).sum())))

        if log_tag and (time.time() - last_log_t) >= SWEEP_LOG_INTERVAL_S:
            now = time.time()
            n_chunk = sw - sweep_chunk_start
            dt_chunk = now - sweep_chunk_t0
            sweep_chunk_t0 = now
            sweep_chunk_start = sw
            _wlog(log_tag,
                  f"therm sweep {sw}/{tmax}  k_avg={hist[-1]:.4f}  "
                  f"k_top={eng.peak_degree}  "
                  f"({n_chunk} sweeps in last {dt_chunk:.0f}s = "
                  f"{dt_chunk/max(n_chunk,1):.1f}s/sweep)")
            last_log_t = now

        if sw >= tmin and len(hist) >= THERM_WINDOW:
            w = np.asarray(hist[-THERM_WINDOW:])
            drift = abs(float(np.polyfit(np.arange(THERM_WINDOW), w, 1)[0]))
            noise = float(np.std(w))
            if drift < THERM_DRIFT_TOL and noise < noise_tol:
                deg_confirms += 1
                if log_tag:
                    _wlog(log_tag,
                          f"therm candidate at sw={sw}: drift={drift:.5f} "
                          f"noise={noise:.5f}  confirms={deg_confirms}/{THERM_CONFIRM}")
                if deg_confirms >= THERM_CONFIRM:
                    if log_tag:
                        _wlog(log_tag, f"therm CONFIRMED at sw={sw}")
                    break
            else:
                deg_confirms = 0

    return sweeps


def build_graph(N, k, T, lb, mu, ec, seed, max_deg=MAX_DEG,
                tmax_override=None, log_tag=None):
    """
    Build one thermalised graph at (N, k, T, lb, μ). Returns
    (eng, stats, sweeps_used, peak_degree).

    Thermalisation follows the drift+noise criterion (drift via linear
    fit on a sliding window, noise via window std), with tmin/tmax
    scaled by 1/(1-lb) since high-locality mixing is slow.

    If log_tag is given, prints worker-side progress (every 60 s
    inside the thermalisation loop, plus phase-transition lines).
    μ-calibration passes log_tag=None to keep that phase quiet.
    """
    if log_tag:
        _wlog(log_tag, f"build_graph: allocating PhysicsEngine "
                       f"N={N} max_deg={max_deg}")
    eng = PhysicsEngine(N, seed=seed, max_degree=max_deg)
    if log_tag:
        _wlog(log_tag, "build_graph: PhysicsEngine constructed")
    eng.temperature    = T
    eng.edge_cost      = ec
    eng.degree_penalty = mu
    eng.locality_bias  = lb

    egap = max(1.0 - lb, 0.001)
    scale = min(300.0, 1.0 / egap)
    tmin = int(min(THERM_MIN_BASE * scale, 2000))
    tmax = tmax_override or int(min(THERM_MAX_BASE * scale, 5000))
    noise_tol = max(0.5 / math.sqrt(N), 0.0001)
    if log_tag:
        _wlog(log_tag, f"build_graph: therm config tmin={tmin} "
                       f"tmax={tmax} noise_tol={noise_tol:.5f}")

    therm_trace = []
    sweeps = _thermalise(eng, N, tmin, tmax, noise_tol, log_tag=log_tag,
                         trace_out=therm_trace)

    if log_tag:
        _wlog(log_tag, f"therm done after {sweeps} sweeps  "
                       f"→  {PROD_SWEEPS} production sweeps next")
    eng.iterate(steps=N * PROD_SWEEPS)
    # Final post-production equilibration sample, and stash the trace on the
    # engine so build_cell can persist it (keeps build_graph's return arity
    # unchanged — every existing caller still unpacks the same 4-tuple).
    _degf = eng.node_degrees[:N].astype(np.int64)
    therm_trace.append((sweeps + PROD_SWEEPS,
                        float(_degf.mean()), float((_degf * _degf).sum())))
    eng.therm_trace = therm_trace
    if log_tag:
        _wlog(log_tag, f"production sweeps done  →  computing graph stats  "
                       f"(k_top across run = {eng.peak_degree})")
    stats = get_graph_stats(eng.node_neighbors, eng.node_degrees)
    if log_tag:
        _wlog(log_tag,
              f"stats: edges={stats.edges} triangles={stats.triangles} "
              f"k_avg={stats.k_avg:.4f} lcc={stats.lcc_pct:.1f}%")
    return eng, stats, sweeps, eng.peak_degree


# ═══════════════════════════════════════════════════════════════════
#  μ calibration — binary search at N=MU_N_CAL, cached to JSON
# ═══════════════════════════════════════════════════════════════════
def calibrate_mu(k, T, lb, ec, verbose=True):
    """Binary-search μ at small N so k_actual ≈ k.

    Returns (mu, err, k_actual) — the best μ found, the absolute error
    |k_actual - k| at that μ, and the realised k_avg the engine
    actually produced. The third value is what makes "err 5.598" at
    k=10 legible on the calibration log: was k_actual ≈ 4.4 (μ too
    high, graph oversuppressed) or ≈ 15.6 (μ too low, graph
    saturated)? They tell different stories about whether the binary
    search bracket needs to widen or whether the closed-form starting
    guess in _mu_guess is mis-tuned for this (T, lb) cell.
    """
    lo, hi = MU_LO, MU_HI
    guess = max(lo, min(hi, _mu_guess(k, T, lb, ec)))
    best_mu, best_err, best_k_avg = guess, float("inf"), float("nan")

    for it in range(MU_MAX_ITER):
        mu = guess if it == 0 else (lo + hi) / 2.0
        _, s, _, _ = build_graph(MU_N_CAL, k, T, lb, mu, ec,
                                 seed=SEEDS[0], tmax_override=1500)
        err = abs(s.k_avg - k)
        if err < best_err:
            best_err, best_mu, best_k_avg = err, mu, float(s.k_avg)
        if err < MU_TOL:
            break
        if s.k_avg > k + MU_TOL:
            lo = mu
        elif s.k_avg < k - MU_TOL:
            hi = mu
        else:
            break
        # Bracket-tighten on the first iter so we don't binary-search
        # a 2000× range when our closed-form guess is already close.
        if it == 0:
            if s.k_avg > k:
                lo, hi = mu * 0.8, mu * 2.0
            else:
                lo, hi = mu * 0.3, mu * 1.2

    if verbose:
        log(f"    μ cal  k={k} T={T:.3f} lb={lb:.3f}: "
            f"μ={best_mu:.5f}  k̃={best_k_avg:.3f}  "
            f"μk={best_mu*k:.4f}  err={best_err:.4f}")
    return float(best_mu), float(best_err), float(best_k_avg)


def _calib_worker(args):
    k, T, lb, ec = args
    try:
        mu, err, k_avg = calibrate_mu(k, T, lb, ec, verbose=False)
        return (k, T, lb, mu, err, k_avg, None)
    except Exception as e:
        return (k, T, lb, None, None, None, str(e)[:200])


def ensure_mu_calibrated(ec, workers=None):
    """Make sure every (k, T, lb) has a μ in mu_table.json. Runs in parallel."""
    table = load_mu_table()
    needed = []
    for k in K_ALL:
        for T in T_ALL:
            for lb in LB_ALL:
                if mu_key(k, T, lb) not in table:
                    needed.append((k, T, lb, ec))
    if not needed:
        log(f"  μ table complete ({len(table)} entries)")
        return table

    log(f"  μ calibration: {len(needed)} missing, "
        f"{len(table)} cached → running in parallel")
    wk = min(workers or multiprocessing.cpu_count(), len(needed))

    done_count = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=wk) as ex:
        futs = {ex.submit(_calib_worker, a): a for a in needed}
        for f in as_completed(futs):
            k, T, lb, mu, err, k_avg, err_msg = f.result()
            done_count += 1
            if mu is None:
                log(f"    ✗ μ cal failed k={k} T={T} lb={lb}: {err_msg}")
                continue
            table[mu_key(k, T, lb)] = mu
            save_mu_table(table)
            # Log line shows k̃ (the realised k_avg) alongside the
            # error so an operator scanning a 1300-line cal log can
            # immediately tell *which way* a misfit cell drifted —
            # k̃<<k means μ over-suppressed degree, k̃>>k means μ
            # too low and the graph saturated near max_deg. Pure
            # `err` only gives magnitude, not direction.
            log(f"    [{done_count:3d}/{len(needed)}] μ k={k} T={T:.3f} "
                f"lb={lb:.3f} → {mu:.5f} k̃={k_avg:.3f} (err {err:.4f}) "
                f"[{time.time()-t0:.0f}s total]")
    log(f"  μ calibration done ({len(table)} entries)")
    return table


def calibrate_missing(targets, ec, workers=None, verbose=True):
    """Calibrate μ for an explicit list of (k, T, lb) tuples.

    Use this when you want to ensure a specific subset of cells has
    calibrated μ — e.g. before a focused sweep that uses parameters
    outside the K_ALL × T_ALL × LB_ALL config grid. Uses the same
    parallel _calib_worker as ensure_mu_calibrated so the per-cell
    cost is identical; just the target set is narrower.

    Args:
        targets: iterable of (k, T, lb) tuples
        ec:      energy-cost coefficient (use core.project_constants.EC)
        workers: parallelism (default: cpu_count())
        verbose: emit per-cell progress log lines

    Returns:
        (table, n_done, n_failed) — the updated μ table and counts.
        Failed cells are absent from the table; the caller can decide
        whether to skip them or retry.

    The μ table is persisted to disk after each successful calibration
    so a Ctrl-C in the middle still leaves a valid (partial) table.
    """
    table = load_mu_table()
    needed = [(k, T, lb, ec) for (k, T, lb) in targets
              if mu_key(k, T, lb) not in table]
    if not needed:
        if verbose:
            log(f"  μ calibration: all {len(list(targets))} targets "
                f"already cached")
        return table, 0, 0

    if verbose:
        log(f"  μ calibration: {len(needed)} target(s) missing → "
            f"running in parallel")
    wk = min(workers or multiprocessing.cpu_count(), len(needed))

    n_done = n_failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=wk) as ex:
        futs = {ex.submit(_calib_worker, a): a for a in needed}
        for f in as_completed(futs):
            k, T, lb, mu, err, k_avg, err_msg = f.result()
            if mu is None:
                n_failed += 1
                if verbose:
                    log(f"    ✗ μ cal failed k={k} T={T} lb={lb}: "
                        f"{err_msg}")
                continue
            n_done += 1
            table[mu_key(k, T, lb)] = mu
            save_mu_table(table)
            if verbose:
                log(f"    [{n_done:3d}/{len(needed)}] μ k={k} T={T:.3f} "
                    f"lb={lb:.3f} → {mu:.5f} k̃={k_avg:.3f} "
                    f"(err {err:.4f}) [{time.time()-t0:.0f}s total]")
    if verbose:
        log(f"  μ calibration done: {n_done} added, "
            f"{n_failed} failed, {len(table)} total entries")
    return table, n_done, n_failed
