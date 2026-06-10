"""
core/graph_builder.py — Graph build + μ calibration
==========================================
build_graph runs adaptive thermalisation with p0's drift+noise
criterion. calibrate_mu binary-searches the degree-penalty multiplier
at small N so k_actual ≈ k. calibrate_missing parallelises the
calibration sweep across an explicit list of (k, T, lb) cells.
"""

import math
import os
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from core.physics_engine import PhysicsEngine
from metrics import get_graph_stats

try:
    from core import graph_store as _GRAPH_STORE   # optional persistent graph cache
except Exception:                                  # pragma: no cover
    _GRAPH_STORE = None

from core.project_constants import (
    MAX_DEG, SEEDS,
    MU_N_CAL, MU_LO, MU_HI, MU_TOL, MU_MAX_ITER,
    THERM_WINDOW, THERM_DRIFT_TOL, THERM_MIN_BASE, THERM_MAX_BASE,
    THERM_CONFIRM, PROD_SWEEPS, ENGINE_BACKEND,
)
from core.disk_io import (_wlog, log, mu_key, mu_key_n, mu_lookup,
                          load_mu_table, save_mu_table)


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

    Returns (sweeps, converged): the number of sweeps actually run and
    whether the drift+noise criterion CONFIRMED — converged=False means
    the loop ran out of budget (hit tmax) without ever confirming, which
    the caller must surface rather than silently treating as equilibrated.
    Engine state is mutated in place; the caller reads eng.node_degrees,
    eng.peak_degree, etc.

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
    converged = False
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
                    converged = True
                    break
            else:
                deg_confirms = 0

    return sweeps, converged


def _collect_series(eng, N, n_sweeps, sweep_offset=0, trace_out=None,
                    trace_decim=1):
    """Run `n_sweeps` more sweeps, sampling k_avg and q = Σd²/N EVERY sweep
    (q is the degree-concentration term of the Hamiltonian — the structural
    observable, which carries the slowest modes). Returns (sweeps, k, q) as
    float arrays. A decimated copy goes to trace_out for the meta sidecar so
    persisted traces stay small while the verdict uses full resolution."""
    sw = np.empty(n_sweeps)
    ks = np.empty(n_sweeps)
    qs = np.empty(n_sweeps)
    for i in range(n_sweeps):
        eng.iterate(steps=N)
        deg = eng.node_degrees[:N].astype(np.int64)
        sw[i] = sweep_offset + i + 1
        ks[i] = float(deg.mean())
        qs[i] = float((deg * deg).sum()) / N
        if trace_out is not None and (i % trace_decim == 0):
            trace_out.append((int(sw[i]), ks[i], qs[i] * N))
    return sw, ks, qs


def _tau_int(x):
    """Integrated autocorrelation time of a 1-D series, in sample units,
    via Sokal's adaptive window (sum normalised autocorrelations until the
    window exceeds 5·τ). Floored at 1 (white noise), capped at n/4 (beyond
    that the series cannot resolve its own correlations)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return float(max(1, n // 4))
    y = x - x.mean()
    var = float(np.dot(y, y)) / n
    if var <= 0:
        return 1.0
    tau = 1.0
    for lag in range(1, n // 4):
        rho = float(np.dot(y[:-lag], y[lag:])) / ((n - lag) * var)
        tau += 2.0 * rho
        if lag >= 5.0 * tau:
            break
    return float(min(max(tau, 1.0), n / 4.0))


def _drift_verdict(sw, y, horizon_sweeps, abs_floor):
    """Stationarity verdict for one observable series sampled per sweep.

    1. OLS drift slope over the stretch; residual σ after detrending.
    2. τ_int measured FROM THE RESIDUALS (not assumed): effective sample
       size ESS = n/(2τ); the slope's standard error is inflated by √(2τ).
    3. Significance z = slope/SE.
    4. IMPACT = how far the fitted drift would carry the observable over
       another HORIZON of evolution — the cell's own thermalisation scale
       (≈ phase-1 sweeps), i.e. "would running it as long again change the
       structure beyond its noise band?" — in units of that band:
           impact = |slope|·horizon_sweeps / max(σ_resid, abs_floor)
       (The production window itself is only PROD_SWEEPS≈3 sweeps — drift
       across it is never the issue; a transient STRUCTURE is.)

    verdict ∈ {True, False, None}:
      • False  — drift is statistically DETECTED (|z| > 3) AND it matters
                 (impact > 1): the structure moves by more than its noise
                 band during the measurement. The real failure mode.
      • True   — resolved and either no detectable drift or a drift too
                 small to matter (impact ≤ 1; e.g. T=0 neutral-manifold
                 wandering that never moves q beyond its band).
      • None   — UNRESOLVED: τ so long the stretch can't decide
                 (ESS < 8 or span < 6τ). Not a failure — a budget verdict.

    Returns (verdict, stats_dict).
    """
    n = y.size
    span = float(sw[-1] - sw[0]) if n > 1 else 0.0
    if n < 8 or span <= 0:
        return None, {"n": int(n), "reason": "too few samples"}
    A = np.vstack([sw - sw.mean(), np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    resid = y - A @ coef
    sigma = float(resid.std(ddof=1)) if n > 2 else 0.0
    tau = _tau_int(resid)
    ess = n / (2.0 * tau)
    sxx = float(np.sum((sw - sw.mean()) ** 2))
    se_slope = (sigma * math.sqrt(2.0 * tau) / math.sqrt(sxx)) if sxx > 0 else float("inf")
    z = slope / se_slope if se_slope > 0 else 0.0
    impact = abs(slope) * horizon_sweeps / max(sigma, abs_floor, 1e-300)
    stats = {"n": int(n), "span": span, "tau": round(tau, 1),
             "ess": round(ess, 1), "slope_per_ksweep": slope * 1000.0,
             "z": round(float(z), 2), "impact": round(float(impact), 3)}
    if ess < 8 or span < 6.0 * tau:
        return None, stats
    if abs(z) > 3.0 and impact > 1.0:
        return False, stats
    return True, stats


def _verify_equilibration(eng, N, s0, noise_tol, log_tag=None,
                          trace_out=None):
    """Equilibration GUARANTEE, hard-capped at ~2-3× the phase-1 cost.

    The drift+noise criterion in _thermalise watches mean degree, which can
    settle before the structure does. This pass samples BOTH k_avg and
    q = Σd²/N every sweep over a continuation block the length of phase-1
    and applies _drift_verdict — a τ-aware drift-slope test with an impact
    criterion — to each. τ is MEASURED from the residuals, never assumed:
    a fixed-τ window comparison false-fails 70-80%% of the time on
    perfectly stationary series whose slow modes have τ ~ 100-400 sweeps
    (verified by simulation), which is precisely the regime of Σd² at
    large N. The impact criterion additionally forgives real-but-harmless
    motion (e.g. T=0 neutral-manifold wandering) that would never move the
    observable beyond its own fluctuation band during the measurement.

      block 1 (≈2× total): verdict on the block.
      block 2 (≈3× total): only if block 1 said False or None — drift may
              still be decaying, so block 2 ALONE gets the final word when
              block 1 failed; the CONCATENATED series decides when block 1
              was merely unresolved (more span resolves longer τ).

    verified ∈ {True, False, None}; None means "could not resolve within
    the ×3 budget" and is recorded as such — downstream gates reject only
    explicit False, so unresolved cells stay usable but flagged.
    """
    block = max(s0, 400)
    horizon = block                       # "as long again" — see _drift_verdict
    decim = max(1, (2 * block) // 400)   # keep persisted traces ≤ ~400 rows

    def _verdict_on(sw, ks, qs):
        vk, sk = _drift_verdict(sw, ks, horizon, noise_tol)
        vq, sq = _drift_verdict(sw, qs, horizon,
                                0.005 * max(abs(float(np.median(qs))), 1e-12))
        if vk is False or vq is False:
            v = False
        elif vk is None or vq is None:
            v = None
        else:
            v = True
        return v, {"k": sk, "q": sq}

    sw1, k1, q1 = _collect_series(eng, N, block, sweep_offset=s0,
                                  trace_out=trace_out, trace_decim=decim)
    v, st = _verdict_on(sw1, k1, q1)
    extra = block
    blocks = 1
    if v is not True:
        first = (v, st)
        sw2, k2, q2 = _collect_series(eng, N, block, sweep_offset=s0 + block,
                                      trace_out=trace_out, trace_decim=decim)
        extra += block
        blocks = 2
        if first[0] is False:
            # drift was detected in block 1 — it may have been the tail of
            # relaxation, so the LATER block alone gets the final word.
            v, st = _verdict_on(sw2, k2, q2)
            st["block1"] = first[1]
        else:
            # unresolved — more span is what resolves a long τ.
            v, st = _verdict_on(np.concatenate([sw1, sw2]),
                                np.concatenate([k1, k2]),
                                np.concatenate([q1, q2]))
    if log_tag:
        _q = st.get("q", {})
        _wlog(log_tag, f"equil-verify [{blocks} block(s)] -> "
                       f"{ {True: 'PASS', False: 'DRIFT', None: 'UNRESOLVED'}[v] } "
                       f"(q: tau={_q.get('tau')}, z={_q.get('z')}, "
                       f"impact={_q.get('impact')})")
    return {"verified": v, "blocks": blocks, "extra_sweeps": extra,
            "detail": st}

def _resolve_backend():
    """Normalise ENGINE_BACKEND to the concrete native backend to *try*
    (or 'numba'). 'auto' prefers the C++ engine."""
    b = (ENGINE_BACKEND or "auto").lower()
    if b == "auto":
        return "cpp"
    return b


def ensure_engine_built(log=print):
    """Build the selected native engine's shared library ONCE, up front, before
    any worker pool forks — so the many parallel build_graph calls find it
    already compiled and never race to build it. Idempotent and safe to call
    from a single process; a no-op for the numba backend or if no compiler is
    present (the sweep then runs on numba). Returns the backend that will
    actually be used ('cpp'/'rust'/'numba')."""
    want = _resolve_backend()
    if want == "numba":
        return "numba"
    try:
        from engines import get_engine
        cls = get_engine(want)
        if cls.ensure_available(log=log):
            return cls.backend
        log(f"[engines] {want} unavailable — the sweep will run on numba.")
    except Exception as ex:                      # pragma: no cover
        log(f"[engines] {want} build error ({type(ex).__name__}: {ex}) "
            f"— the sweep will run on numba.")
    return "numba"


def _make_engine(N, T, lb, mu, ec, seed, max_deg, log_tag=None):
    """Construct the graph-growth engine for the selected back-end and return
    (engine_like, backend_name). The C++/Rust engines are wrapped in
    _NativeEngineAdapter so build_graph drives them exactly like the numba
    kernel; numba is returned raw (unchanged behaviour) and is the guaranteed
    fallback whenever a native engine can't be built or constructed."""
    want = _resolve_backend()

    def _log(msg):
        if log_tag:
            _wlog(log_tag, msg)

    if want != "numba":
        try:
            from engines import get_engine
            cls = get_engine(want)
            # ensure_available() is cheap once the .so exists (it is pre-built
            # before any pool by ensure_engine_built); a single-process caller
            # builds it here on first use.
            if cls.ensure_available(log=(lambda *a: _log(" ".join(str(x) for x in a)))):
                eng = cls(N, max_degree=max_deg, seed=seed,
                          temperature=T, degree_penalty=mu,
                          edge_cost=ec, locality_bias=lb)
                _log(f"build_graph: using {cls.backend} engine")
                return _NativeEngineAdapter(eng, N), cls.backend
            _log(f"build_graph: {want} engine unavailable — using numba")
        except Exception as ex:
            _log(f"build_graph: {want} engine error "
                 f"({type(ex).__name__}: {ex}) — using numba")

    eng = PhysicsEngine(N, seed=seed, max_degree=max_deg,
                        temperature=T, degree_penalty=mu,
                        edge_cost=ec, locality_bias=lb)
    return eng, "numba"


def build_graph(N, k, T, lb, mu, ec, seed, max_deg=MAX_DEG,
                tmax_override=None, log_tag=None, use_store=True,
                verify_equil=True):
    """
    Build one thermalised graph at (N, k, T, lb, μ). Returns
    (eng, stats, sweeps_used, peak_degree).

    Thermalisation follows the drift+noise criterion (drift via linear
    fit on a sliding window, noise via window std), with tmin/tmax
    scaled by 1/(1-lb) since high-locality mixing is slow.

    When verify_equil is True (the default for production cells), the
    drift+noise criterion is then VERIFIED by _verify_equilibration —
    a block-comparison stationarity check on both k_avg and Σd²/N,
    hard-capped at ~2-3× the phase-1 cost. The outcome is stashed on the
    engine as eng.therm_info = {sweeps, converged, verified, blocks,
    extra_sweeps, total_sweeps} so callers can persist it; cells with
    verified=False must be flagged downstream, never silently trusted.
    μ-calibration passes verify_equil=False — its throwaway trial builds
    only need k_avg to settle, and tripling that phase would dominate
    the sweep's startup cost.

    If log_tag is given, prints worker-side progress (every 60 s
    inside the thermalisation loop, plus phase-transition lines).
    μ-calibration passes log_tag=None to keep that phase quiet.

    When use_store is True (and $GRAPH_STORE != "0"), the final graph is cached
    via core.graph_store keyed by (k,T,lb,N,seed,mu,ec,max_deg): a matching cache
    entry is loaded and returned without re-evolving, and a freshly built graph is
    saved before returning. μ-calibration passes use_store=False so its many
    throwaway trial-μ builds don't pollute the cache. stats are recomputed from the
    loaded arrays (cheap, deterministic) rather than serialised.
    """
    store_on = use_store and os.environ.get("GRAPH_STORE", "1") != "0" and _GRAPH_STORE is not None
    if store_on:
        cached = _GRAPH_STORE.load_graph(k, T, lb, N, seed, max_deg, mu=mu)
        if cached is not None:
            stats = get_graph_stats(cached.node_neighbors, cached.node_degrees)
            sweeps = int(cached.meta.get("sweeps", 0))
            peak_deg = int(cached.meta.get("peak_deg", cached.peak_degree))
            # Restore the equilibration verdict recorded at build time, so a
            # cache hit carries the same guarantee flags as a fresh build.
            cached.therm_info = cached.meta.get("therm_info")
            if log_tag:
                _wlog(log_tag, f"build_graph: CACHE HIT {os.path.relpath(_GRAPH_STORE.path_for(k,T,lb,N,seed,max_deg))} "
                               f"— skipped thermalisation (sweeps={sweeps}, k_avg={stats.k_avg:.3f})")
            return cached, stats, sweeps, peak_deg

    if log_tag:
        _wlog(log_tag, f"build_graph: allocating engine "
                       f"N={N} max_deg={max_deg}")
    eng, _engine_used = _make_engine(N, T, lb, mu, ec, seed, max_deg, log_tag)
    if log_tag:
        _wlog(log_tag, f"build_graph: {_engine_used} engine constructed")

    egap = max(1.0 - lb, 0.001)
    scale = min(300.0, 1.0 / egap)
    tmin = int(min(THERM_MIN_BASE * scale, 2000))
    tmax = tmax_override or int(min(THERM_MAX_BASE * scale, 5000))
    noise_tol = max(0.5 / math.sqrt(N), 0.0001)
    if log_tag:
        _wlog(log_tag, f"build_graph: therm config tmin={tmin} "
                       f"tmax={tmax} noise_tol={noise_tol:.5f}")

    therm_trace = []
    sweeps, converged = _thermalise(eng, N, tmin, tmax, noise_tol,
                                    log_tag=log_tag, trace_out=therm_trace)
    if not converged and log_tag:
        _wlog(log_tag, f"therm WARNING: drift+noise never confirmed within "
                       f"tmax={tmax} sweeps — verification pass will decide")

    # Equilibration guarantee: verify stationarity on BOTH k_avg and Σd²/N
    # with consecutive-block comparison, capped at ~2-3× the phase-1 cost.
    therm_info = {"sweeps": int(sweeps), "converged": bool(converged),
                  "verified": None, "blocks": 0, "extra_sweeps": 0}
    if verify_equil:
        v = _verify_equilibration(eng, N, sweeps, noise_tol,
                                  log_tag=log_tag, trace_out=therm_trace)
        therm_info.update(verified=bool(v["verified"]),
                          blocks=int(v["blocks"]),
                          extra_sweeps=int(v["extra_sweeps"]),
                          detail=v["detail"])
    therm_info["total_sweeps"] = int(sweeps) + int(therm_info["extra_sweeps"])

    if log_tag:
        _wlog(log_tag, f"therm done after {therm_info['total_sweeps']} sweeps "
                       f"(phase-1 {sweeps} + verify "
                       f"{therm_info['extra_sweeps']})  "
                       f"→  {PROD_SWEEPS} production sweeps next")
    eng.iterate(steps=N * PROD_SWEEPS)
    # Final post-production equilibration sample, and stash the trace on the
    # engine so build_cell can persist it (keeps build_graph's return arity
    # unchanged — every existing caller still unpacks the same 4-tuple).
    _degf = eng.node_degrees[:N].astype(np.int64)
    therm_trace.append((therm_info["total_sweeps"] + PROD_SWEEPS,
                        float(_degf.mean()), float((_degf * _degf).sum())))
    eng.therm_trace = therm_trace
    eng.therm_info = therm_info
    if log_tag:
        _wlog(log_tag, f"production sweeps done  →  computing graph stats  "
                       f"(k_top across run = {eng.peak_degree})")
    stats = get_graph_stats(eng.node_neighbors, eng.node_degrees)
    if log_tag:
        _wlog(log_tag,
              f"stats: edges={stats.edges} triangles={stats.triangles} "
              f"k_avg={stats.k_avg:.4f} lcc={stats.lcc_pct:.1f}%")
    if store_on:
        try:
            _GRAPH_STORE.save_graph(
                eng.node_neighbors, eng.node_degrees,
                k=k, T=T, lb=lb, N=N, seed=seed, max_degree=max_deg,
                mu=mu, ec=ec, sweeps=sweeps, peak_deg=eng.peak_degree,
                therm_trace=getattr(eng, "therm_trace", None), k_avg=stats.k_avg,
                engine=_engine_used,
                therm_info=getattr(eng, "therm_info", None))
            if log_tag:
                _wlog(log_tag, "build_graph: cached final graph")
        except Exception as ex:                 # caching is best-effort, never fatal
            if log_tag:
                _wlog(log_tag, f"build_graph: cache write skipped ({ex})")
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
    sat_floor = 0.0          # highest μ found to still saturate the cap

    for it in range(MU_MAX_ITER):
        mu = guess if it == 0 else (lo + hi) / 2.0
        try:
            _, s, _, _ = build_graph(MU_N_CAL, k, T, lb, mu, ec,
                                     seed=SEEDS[0], tmax_override=1500,
                                     use_store=False, verify_equil=False)
        except RuntimeError as e:
            if "max_degree" not in str(e):
                raise
            # A cap hit means the graph wanted to grow past MAX_DEG at this μ,
            # i.e. the degree penalty is too weak — μ is too LOW. So treat it
            # exactly like "k_avg = +inf": raise the floor and keep searching
            # UPWARD instead of letting the RuntimeError abort the whole cell.
            #
            # This is what was dropping the cold-T / low-ℓ corner. The cap was
            # NOT hit at the μ those cells actually need (cold cells need a μ
            # near the T=0 value, which is safe — that's why T=0 itself fills);
            # it was only hit at a too-low *trial* μ the binary search probed
            # on the way there (the it==0 down-bracket overshoots low for very
            # small T, because the T>0 guess is T-independent). Recovering from
            # it lets the search settle on the correct, safe μ and the cell
            # fills, with no change to MAX_DEG.
            sat_floor = max(sat_floor, mu)
            lo, hi = (mu, mu * 4.0) if it == 0 else (mu, hi)
            continue
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
        # Never let the bracket dip back into a μ range known to saturate the
        # cap — that would just re-trigger the RuntimeError and waste an iter.
        lo = max(lo, sat_floor)
        if hi <= lo:
            hi = lo * 1.5

    if verbose:
        log(f"    μ cal  k={k} T={T:.3f} lb={lb:.3f} N={MU_N_CAL}: "
            f"μ={best_mu:.5f}  k̂={best_k_avg:.3f}  "
            f"μk={best_mu*k:.4f}  err={best_err:.4f}")
    return float(best_mu), float(best_err), float(best_k_avg)


def _calib_worker(args):
    k, T, lb, ec = args
    try:
        mu, err, k_avg = calibrate_mu(k, T, lb, ec, verbose=False)
        return (k, T, lb, mu, err, k_avg, None)
    except Exception as e:
        return (k, T, lb, None, None, None, str(e)[:200])


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
        (table, n_done, n_failed, failures) — the updated μ table, counts,
        and a {mu_key: error_message} dict for the cells whose calibration
        raised (e.g. a max-degree cap hit during the calibration build).
        Failed cells are absent from the table; the caller can decide
        whether to skip them or retry, and can persist `failures` so the
        dashboard can show *why* those cells produced no data.

    The μ table is persisted to disk after each successful calibration
    so a Ctrl-C in the middle still leaves a valid (partial) table.
    """
    table = load_mu_table()
    # Materialise once: `targets` may be a one-shot iterator, and we touch it
    # both in the comprehension below and in the len() further down. Consuming
    # it twice would make the count read 0 for a generator.
    targets = list(targets)
    needed = [(k, T, lb, ec) for (k, T, lb) in targets
              if mu_key(k, T, lb) not in table]
    if not needed:
        if verbose:
            log(f"  μ calibration: all {len(targets)} targets "
                f"already cached")
        return table, 0, 0, {}

    if verbose:
        log(f"  μ calibration: {len(needed)} target(s) missing → "
            f"running in parallel")
    wk = min(workers or multiprocessing.cpu_count(), len(needed))

    n_done = n_failed = 0
    failures = {}          # mu_key -> error message, for cells that raised
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=wk) as ex:
        futs = {ex.submit(_calib_worker, a): a for a in needed}
        for f in as_completed(futs):
            k, T, lb, mu, err, k_avg, err_msg = f.result()
            if mu is None:
                n_failed += 1
                failures[mu_key(k, T, lb)] = err_msg or "calibration failed"
                if verbose:
                    log(f"    ✗ μ cal failed k={k} T={T} lb={lb}: "
                        f"{err_msg}")
                continue
            n_done += 1
            table[mu_key(k, T, lb)] = mu
            save_mu_table(table)
            if verbose:
                log(f"    [{n_done:3d}/{len(needed)}] μ k={k} T={T:.3f} "
                    f"lb={lb:.3f} N={MU_N_CAL} → {mu:.5f} k̂={k_avg:.3f} "
                    f"(err {err:.4f}) [{time.time()-t0:.0f}s total]")
    if verbose:
        log(f"  μ calibration done: {n_done} added, "
            f"{n_failed} failed, {len(table)} total entries")
    return table, n_done, n_failed, failures


# ═══════════════════════════════════════════════════════════════════
#  μ-vs-N drift audit — measured for free from the sweep's own sidecars
# ═══════════════════════════════════════════════════════════════════
def mu_drift_report(flow_dir="flow"):
    """Scan the meta sidecars and report realised-k drift per (k, T, lb, N).

    Every completed cell already records the μ it was grown with and the
    k_avg it realised, so the μ(N) map costs NOTHING beyond reading JSON.
    Returns {(k, T, lb): {N: {"k_avg": mean over seeds, "mu": μ used,
    "n_seeds": count, "err_pct": 100·(k_avg−k)/k}}}, sorted by N.
    """
    import glob as _glob
    import json as _json
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(lambda: {"k_avg": [], "mu": []}))
    for p in _glob.glob(os.path.join(flow_dir, "meta_*.json")):
        try:
            with open(p) as fh:
                m = _json.load(fh)
        except (OSError, ValueError):
            continue
        if m.get("kind") != "cell":
            continue
        k_avg, mu = m.get("k_avg"), m.get("mu")
        if k_avg is None or mu is None or mu != mu:
            continue
        cell = (int(m["k"]), float(m["T"]), float(m["lb"]))
        acc[cell][int(m["N"])]["k_avg"].append(float(k_avg))
        acc[cell][int(m["N"])]["mu"].append(float(mu))
    report = {}
    for cell, by_n in acc.items():
        k_target = cell[0]
        rows = {}
        for N in sorted(by_n):
            ks = by_n[N]["k_avg"]
            mus = by_n[N]["mu"]
            mean_k = float(np.mean(ks))
            rows[N] = {
                "k_avg": mean_k,
                "mu": float(np.median(mus)),
                "n_seeds": len(ks),
                "err_pct": 100.0 * (mean_k - k_target) / k_target,
            }
        report[cell] = rows
    return report


def _load_pred_registry():
    """Set of N-qualified keys whose table entries are PREDICTIONS (see
    update_mu_table_n). Lives in MU_PRED_JSON next to mu_table.json."""
    import json as _json
    from core.project_constants import MU_PRED_JSON
    try:
        with open(MU_PRED_JSON) as fh:
            return set(_json.load(fh).get("predicted_keys", []))
    except (OSError, ValueError):
        return set()


def _save_pred_registry(keys):
    import json as _json
    from core.project_constants import MU_PRED_JSON
    tmp = f"{MU_PRED_JSON}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            _json.dump({"predicted_keys": sorted(keys),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh,
                       indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, MU_PRED_JSON)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def update_mu_table_n(table, flow_dir="flow",
                      drift_tol=None, min_seeds=None, target_Ns=None,
                      log_fn=None):
    """Derive N-qualified μ corrections AND predictions from the drift report
    and merge them into `table` IN PLACE (the caller persists with
    save_mu_table). NOTHING here builds a graph — every input is read from
    sidecars of cells the sweep already ran, so the whole audit costs a JSON
    scan.

    Two phases:

    MEASURED corrections — for each (k, T, lb) and each measured N where the
    realised k_avg drifts from the target by more than drift_tol (relative)
    with at least min_seeds seeds, store

        table["k_T_lb@N"] = clip(μ_used × k_avg / k_target,
                                 0.5·μ_used, 2.0·μ_used)

    — the first-order Newton step on the empirical k·μ ≈ const relation (see
    disk_io's module comment). Derived once per N, then PINNED (a measured
    correction is never overwritten by a later audit or by a prediction).

    PREDICTED entries — the point of the map: big rungs should START with an
    accurate μ, not wait to measure their own drift. Every measured rung
    yields the per-N ideal μ*(N) = μ_used(N)·k̂(N)/k regardless of which μ it
    actually ran with; with ≥2 distinct measured N we fit μ*(N) = a + b·ln N
    and extrapolate to each grid rung LARGER than the largest measured one
    that has produced no data yet. Predictions are conservative — clipped to
    [0.5, 2.0]× the largest-measured μ* — written only when they differ from
    what mu_lookup would already return by more than drift_tol, and tracked
    in MU_PRED_JSON so they REFRESH on every audit (the fit improves as rungs
    complete) until their rung gains real data, at which point the measured
    path takes over. Seed consistency is preserved throughout: a rung's first
    seed records its μ in its sidecar and resolve_cell_mu makes every later
    seed reuse it, so a mid-cell prediction refresh can never split a cell.

    Cells that already have data at some N are never affected (resolve_cell_mu
    reuses their recorded μ). Returns the list of keys added or refreshed.
    drift_tol / min_seeds default to the [calibration] config values;
    target_Ns defaults to the grid's N ladder.
    """
    from core.project_constants import (MU_DRIFT_TOL, MU_DRIFT_MIN_SEEDS,
                                        MU_N_CORRECTION, N_ALL)
    if not MU_N_CORRECTION:
        return []
    drift_tol = MU_DRIFT_TOL if drift_tol is None else drift_tol
    min_seeds = MU_DRIFT_MIN_SEEDS if min_seeds is None else min_seeds
    target_Ns = sorted(int(n) for n in (N_ALL if target_Ns is None
                                        else target_Ns))
    predicted = _load_pred_registry()
    added = []
    report = mu_drift_report(flow_dir)

    # ── phase 1: measured corrections (pinned once derived) ────────────
    for (k, T, lb), rows in report.items():
        for N, r in rows.items():
            if r["n_seeds"] < min_seeds:
                continue
            if abs(r["err_pct"]) / 100.0 <= drift_tol:
                continue
            key = mu_key_n(k, T, lb, N)
            if key in table and key not in predicted:
                continue           # measured pin — derived once, no feedback
            mu_used = r["mu"]
            corrected = mu_used * (r["k_avg"] / k)
            corrected = min(max(corrected, 0.5 * mu_used), 2.0 * mu_used)
            table[key] = float(corrected)
            predicted.discard(key)   # measured now; stop refreshing it
            added.append(key)
            if log_fn:
                log_fn(f"  μ-drift: k={k} T={T:g} lb={lb:g} at N={N:,} "
                       f"realised k̂={r['k_avg']:.3f} "
                       f"({r['err_pct']:+.1f}%) over {r['n_seeds']} seed(s) "
                       f"→ μ {mu_used:.5f} → {corrected:.5f} (measured)")

    # ── phase 2: predicted entries for dataless larger rungs ───────────
    for (k, T, lb), rows in report.items():
        pts = [(N, r) for N, r in rows.items() if r["n_seeds"] >= min_seeds]
        if len({N for N, _ in pts}) < 2:
            continue               # a trend needs ≥2 distinct measured N
        # Per-N ideal μ*: what μ WOULD have hit k exactly at that N.
        Ns = np.array([N for N, _ in pts], dtype=float)
        mu_star = np.array([r["mu"] * (r["k_avg"] / k) for _, r in pts])
        b, a = np.polyfit(np.log(Ns), mu_star, 1)     # μ* ≈ a + b·ln N
        n_max = int(Ns.max())
        anchor = float(mu_star[np.argmax(Ns)])        # μ* at largest measured N
        for N in target_Ns:
            if N <= n_max or N in rows:
                continue           # only dataless rungs beyond the data
            key = mu_key_n(k, T, lb, N)
            if key in table and key not in predicted:
                continue           # a measured pin exists — never overwrite
            pred = a + b * math.log(N)
            pred = min(max(pred, 0.5 * anchor), 2.0 * anchor)  # conservative
            # Compare against what the lookup would return WITHOUT this key —
            # i.e. the μ the rung would otherwise start with.
            base, _bsrc = mu_lookup(
                {kk: v for kk, v in table.items() if kk != key},
                k, T, lb, N)
            if base is None:
                continue
            if abs(pred - base) / base <= drift_tol:
                # trend says no correction needed → drop a stale prediction
                if key in predicted and key in table:
                    del table[key]
                    predicted.discard(key)
                    added.append(key)
                continue
            if key in table and abs(table[key] - pred) < 1e-12:
                continue           # unchanged — don't spam the log
            table[key] = float(pred)
            predicted.add(key)
            added.append(key)
            if log_fn:
                log_fn(f"  μ-predict: k={k} T={T:g} lb={lb:g} → "
                       f"μ({N:,}) ≈ {pred:.5f} extrapolated from "
                       f"{len(pts)} measured rung(s) ≤ {n_max:,} "
                       f"(refreshes until N={N:,} has data)")
    _save_pred_registry(predicted)
    return added
