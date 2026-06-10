"""
core/cell_tests.py — Independent geometric tests on a single graph cell
=========================================================================
Builds one graph and runs one or more independent tests on it to
characterise its emergent geometry beyond the headline d_s number from
the main sweep. Two cell-build modes:

  basin cell   — (k, T, lb, N, seed) basin candidate from the sweep,
                 thermalised via the project's standard pipeline using
                 the calibrated μ from mu_table.json.
  torus        — d-dimensional periodic hypercubic lattice, side L.
                 Reference geometry with known d_s = d. Used to sanity-
                 check the flow and field probes against a graph that
                 IS guaranteed flat-d_s, so any flow observed on a
                 basin cell can be attributed to graph behaviour rather
                 than estimator bias.

Two tests, sharing the expensive Laplacian-build step:

  flow   — d_s(t) running spectral dimension via local-slope regression
           on the SLQ heat-kernel trace. Distinguishes "flat at d_s=4"
           (4D-like geometry) from "flowing from ~4 at large t to ~2 at
           small t" (CDT/asymptotic-safety dimensional-reduction
           signature) from any other flow shape.

  field  — Massive scalar Green's function. Solves (L + m²I) G = δ_x
           for source nodes x; bins G by BFS distance from source;
           fits log G(r) = c + α·r + β·log r in the scaling window.
           Reports the polynomial exponent β and two regime-dependent
           dimension readings (d_small_mr = 2-β, d_large_mr = 1-2β).
           α should approximately match -m_input.

Usage
-----
    NOTE: `python main.py` launches only the dashboard + auto-sweep —
    there is no `main.py cells` subcommand, and this module cannot be run
    directly (its package-relative imports require the project root on
    sys.path). Import it and call main()/build_cell; the examples below
    show that interface.

    # Run both tests on a top contender
    python main.py cells --k 8 --T 0.005 --lb 0.93 --N 16000 --seed 42

    # Just one test
    python main.py cells --k 9 --T 0 --lb 0.95 --N 64000 --seed 42 --tests flow

    # 4D torus reference for the flow test (should give flat d_s ≈ 4)
    python main.py cells --torus 4,12 --seed 42 --tests flow

    # 3D torus reference for the field test (cleanest validation —
    # both small-mr and large-mr regimes give β = -1 in 3D)
    python main.py cells --torus 3,24 --seed 42 --tests field

    # Field test with custom mass grid
    python main.py cells --k 8 --T 0 --lb 0.99 --N 16000 --seed 42 \\
                         --tests field --masses 0.05,0.1,0.2,0.4,0.8

    # Higher-fidelity flow with more SLQ probes
    python main.py cells --k 9 --T 0 --lb 0.95 --N 256000 --seed 42 \\
                         --tests flow --n-probes 100 --lanczos-m 400

Outputs (in cwd, prefixed by --out-prefix):
    flow_<tag>.csv / .html       — d_s(t) curve + jackknife band
    field_<tag>_curves.csv       — G(r) per (m, r) bin
    field_<tag>_summary.csv      — d-fit summary per mass
    field_<tag>.html             — two-panel plot

where <tag> = k{k}_T{T}_lb{lb}_N{N}_s{seed}  (basin mode)
           or torus{D}d_L{L}_s{seed}          (torus mode).

Design notes
------------
- Flow test reuses helpers from core.flow_probe (slq_per_probe_Z, compute_ds_flow,
  write_html); see that file for the algorithmic rationale.
- Field test uses scipy's sparse direct solver via factorized() so the
  same factorisation is reused across multiple source vectors at fixed
  mass, then refactorised per mass.
- Source nodes for the field test are sampled uniformly from the LCC.
  Multiple sources are averaged within distance bins to suppress
  geometric inhomogeneity that any single source might happen to see.
- The Yukawa fit is regime-aware: at small mr the polynomial prefactor
  is 1/r^(d-2), at large mr it is 1/r^((d-1)/2). The script reports the
  3-parameter free fit's β and both interpretations.
"""

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Reuse all the existing flow infrastructure from core/flow_probe.py — one source of
# truth for the SLQ probe and the d_s(t) computation, so the flow test stays
# in lockstep with whatever changes we make to that script later.
from core.flow_probe import (
    slq_per_probe_Z, compute_ds_flow, FlowResult,
    build_lcc_laplacian, make_slq_t_grid,
    write_csv as write_flow_csv,
    write_html as write_flow_html,
    _quad_dict,
    PW, ML, MR, MT, MB,
)

# Shared graph-build path with the rest of the project.
from metrics.numba_kernels import _build_laplacian_csr_jit
from core.graph_builder import build_graph
from core.project_constants import MAX_DEG, MU_JSON, FLOW_DIR, MU_DRIFT_TOL
from core.disk_io import load_mu_table, mu_key, resolve_cell_mu


# ═══════════════════════════════════════════════════════════════════
#  Cell-build: produce the Laplacian once for both tests
# ═══════════════════════════════════════════════════════════════════
@dataclass
class CellState:
    """Everything the tests need about a built (k, T, lb, N, seed) cell.
    Both flow and field tests consume the same instance — the Laplacian
    build is the expensive step (thermalisation, LCC extraction, CSR
    construction) and we don't want to redo it per test."""
    k: int
    T: float
    lb: float
    N: int
    seed: int
    label: str
    L_indptr: np.ndarray
    L_indices: np.ndarray
    L_data: np.ndarray
    L_csr: sp.csr_matrix
    N_eff: int
    lam_max_bound: int
    stats: object
    sweeps: int
    peak_deg: int
    # Tier-1 extras (populated by build_cell; default-None so build_torus_cell
    # and _build_cell_finalise keyword-construct without them).
    deg_hist: object = None      # np.ndarray: full-graph degree histogram
    sum_deg_sq: int = 0          # Σ d²  (the Hamiltonian degree term)
    therm_trace: object = None   # list of (sweep, k_avg, sum_deg_sq)
    mu_used: object = None       # the μ this cell was actually grown with
    therm_info: object = None    # equilibration verdict from build_graph


def build_cell(k, T, lb, N, seed, mu_table, ec=-1.0):
    """Thermalise the graph at (k, T, lb, N, seed), extract LCC, build
    sparse Laplacian. Returns a CellState consumed by the tests.

    μ resolution is N-aware (disk_io.resolve_cell_mu): a cell that already
    has data at this exact (k, T, lb, N) reuses the μ recorded in its meta
    sidecar — later seeds must match earlier ones — while a fresh rung picks
    up any N-qualified drift correction, falling back to the base
    calibration. The μ actually used is stashed on the CellState as
    cell.mu_used so the meta writer records ground truth, not a re-lookup.
    """
    mu, mu_src = resolve_cell_mu(mu_table, k, T, lb, N, flow_dir=FLOW_DIR)
    if mu is None:
        raise SystemExit(
            f"no μ entry for {mu_key(k, T, lb)} in {MU_JSON} — run the "
            f"sweep's μ-calibration phase first.")

    print(f"  [{time.strftime('%H:%M:%S')}] building graph "
          f"k={k} T={T} lb={lb} N={N} seed={seed} μ={mu:.6f} "
          f"(from {mu_src})",
          flush=True)
    eng, stats, sweeps, peak_deg = build_graph(
        N=N, k=k, T=T, lb=lb, mu=mu, ec=ec, seed=seed,
        max_deg=MAX_DEG, log_tag=None)
    ti = getattr(eng, "therm_info", None)
    print(f"    therm sweeps={sweeps}, k_avg={stats.k_avg:.3f}, "
          f"edges={stats.edges}, lcc={stats.lcc_pct:.1f}%, "
          f"k_top={peak_deg}"
          + (f", equil-verified={ti['verified']}" if ti else ""),
          flush=True)
    # Realised-k drift cross-check: warn loudly when the graph the sweep
    # actually grew is meaningfully off its nominal k column.
    err_pct = 100.0 * (stats.k_avg - k) / k
    if abs(err_pct) > 100.0 * MU_DRIFT_TOL:
        print(f"    ⚠ realised k̂={stats.k_avg:.3f} is {err_pct:+.1f}% off "
              f"target k={k} (μ-vs-N drift — the sweep's drift audit will "
              f"correct rungs that haven't started)", flush=True)
    cell = _build_cell_finalise(eng, stats, sweeps, peak_deg,
                                k, T, lb, N, seed)
    cell.mu_used = float(mu)
    cell.therm_info = ti
    # Tier-1: full-graph degree histogram + Σd² + the equilibration trace
    # stashed on the engine by build_graph. Cheap, agnostic graph properties.
    deg = eng.node_degrees[:N].astype(np.int64)
    cell.deg_hist = np.bincount(deg, minlength=MAX_DEG + 1).astype(np.int64)
    cell.sum_deg_sq = int((deg * deg).sum())
    cell.therm_trace = getattr(eng, "therm_trace", None)
    return cell


def build_torus_cell(d, L, seed):
    """Build a d-dimensional periodic hypercubic lattice as a reference
    geometry. Skips the Markov-chain build path entirely — direct
    neighbour-list construction.

    Used for sanity-checking the flow and field probes against a graph
    with known dimension. A flat d_s(t) curve at d_s = d on this geometry
    confirms that any flow observed on a basin cell is real graph
    behaviour rather than estimator bias. Cleanest reference at d ∈
    {2, 3, 4} where the polynomial-prefactor exponent is unambiguous;
    higher d (5, 6) work but the BFS-distance vs Euclidean-distance
    discrepancy on a hypercubic lattice grows with d.

    Returns a CellState identical in shape to one from build_cell, with
    the (k, T, lb) fields repurposed: k = 2d (coordination number),
    T = 0, lb = 0. The label field carries the torus identity.
    """
    if d < 1 or d > 6:
        raise ValueError(f"torus dim must be 1-6, got {d}")
    if L < 4:
        raise ValueError(f"torus side L must be ≥4 to have meaningful "
                         f"diameter, got {L}")
    N = L ** d
    print(f"  [{time.strftime('%H:%M:%S')}] building {d}D torus "
          f"L={L} (N={N}) seed={seed}",
          flush=True)

    # Vectorised neighbour-list construction. coords[i] holds the
    # d-tuple of coordinates for flat index i; strides converts
    # coordinates back to flat indices.
    strides = np.array([L ** (d - 1 - i) for i in range(d)], dtype=np.int64)
    flat = np.arange(N, dtype=np.int64)
    coords = np.zeros((N, d), dtype=np.int64)
    for i in range(d):
        coords[:, i] = (flat // strides[i]) % L

    max_deg = 2 * d
    nn = np.zeros((N, max_deg), dtype=np.int32)
    nd_arr = np.full(N, max_deg, dtype=np.int32)
    slot = 0
    for axis in range(d):
        for delta in (-1, +1):
            shifted = coords.copy()
            shifted[:, axis] = (shifted[:, axis] + delta) % L
            v_idx = (shifted * strides[None, :]).sum(axis=1)
            nn[:, slot] = v_idx.astype(np.int32)
            slot += 1

    L_ip, L_id, L_dt = _build_laplacian_csr_jit(nn, nd_arr, N)
    L_csr = sp.csr_matrix((L_dt, L_id, L_ip), shape=(N, N))

    label = f"torus {d}D L={L} N={N} s={seed}"
    print(f"    {d}D torus built, all nodes degree {max_deg}, "
          f"BFS diameter = d·⌊L/2⌋ = {d*(L//2)}",
          flush=True)

    # Reuse CellState: k=2d (coordination), T=0, lb=0 are the natural
    # defaults for a regular lattice. Stats=None is fine because no
    # caller of CellState reads stats except for cosmetic logging.
    return CellState(
        k=2 * d, T=0.0, lb=0.0, N=N, seed=seed, label=label,
        L_indptr=L_ip, L_indices=L_id, L_data=L_dt, L_csr=L_csr,
        N_eff=N, lam_max_bound=2 * max_deg,
        stats=None, sweeps=0, peak_deg=max_deg,
    )


def _build_cell_finalise(eng, stats, sweeps, peak_deg, k, T, lb, N, seed):
    """Shared LCC-extraction and Laplacian-build path. Pulled out of
    build_cell so build_cell stays readable and so future callers
    that already have a thermalised graph can reuse it."""

    # Restrict to LCC + build the Laplacian (shared with flow_probe).
    N_eff, L_ip, L_id, L_dt, max_deg_eff = build_lcc_laplacian(eng, N)

    L_csr = sp.csr_matrix((L_dt, L_id, L_ip), shape=(N_eff, N_eff))
    label = f"k={k} T={T} lb={lb} N={N} s={seed}"

    return CellState(
        k=k, T=T, lb=lb, N=N, seed=seed, label=label,
        L_indptr=L_ip, L_indices=L_id, L_data=L_dt, L_csr=L_csr,
        N_eff=N_eff, lam_max_bound=2 * max_deg_eff,
        stats=stats, sweeps=sweeps, peak_deg=peak_deg,
    )


# ═══════════════════════════════════════════════════════════════════
#  Test 1: d_s(t) flow — thin wrapper around core.flow_probe's compute_ds_flow
# ═══════════════════════════════════════════════════════════════════
def run_flow_test(cell: CellState, n_probes, lanczos_m, half_window):
    """Compute d_s(t) on the cell's Laplacian. Returns a FlowResult
    suitable for write_flow_csv / write_flow_html.

    Implementation note: this duplicates a small amount of logic from
    core.flow_probe.run_flow_for_cell because that function builds the graph
    itself; we want to consume the already-built CellState.
    """
    t_grid, t_lo, t_hi = make_slq_t_grid(cell.N_eff, cell.lam_max_bound)

    print(f"    [flow] SLQ: n_probes={n_probes}, lanczos_m={lanczos_m}, "
          f"N_eff={cell.N_eff}, t_grid=[{t_lo:.2g}, {t_hi:.2g}]",
          flush=True)
    t0 = time.time()
    per_probe, theta_all, omega_all = slq_per_probe_Z(
        cell.L_indptr, cell.L_indices, cell.L_data,
        cell.N_eff,
        n_probes, lanczos_m, t_grid, seed=cell.seed * 1000 + 1,
        return_quad=True)
    print(f"    [flow] SLQ done in {time.time()-t0:.1f}s, computing flow...",
          flush=True)
    flow = compute_ds_flow(per_probe, t_grid, cell.N_eff,
                           cell.lam_max_bound, half_window=half_window)

    return FlowResult(
        k=cell.k, T=cell.T, lb=cell.lb, N=cell.N, seed=cell.seed,
        label=cell.label, flow=flow, stats=cell.stats,
        quad=_quad_dict(theta_all, omega_all, t_grid, cell.N_eff,
                        cell.lam_max_bound, lanczos_m, n_probes),
    )


# ═══════════════════════════════════════════════════════════════════
#  Test 2: massive scalar Green's function (Yukawa form fit)
# ═══════════════════════════════════════════════════════════════════
def _bfs_distances_csr(L_csr, source):
    """BFS distances from source on the LCC, in steps. Returns int array
    of length N_eff with -1 for unreached nodes (shouldn't happen on
    LCC). Manually rolled rather than scipy.sparse.csgraph because we
    already have the CSR structure and want explicit control.
    """
    n = L_csr.shape[0]
    indptr = L_csr.indptr
    indices = L_csr.indices
    R = np.full(n, -1, dtype=np.int32)
    R[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        du = R[u] + 1
        # CSR rows for L include the diagonal (degree entry, value > 0)
        # AND the off-diagonal -1 entries. We want neighbours, which are
        # the off-diagonal entries. The Laplacian construction in
        # _build_laplacian_csr_jit puts the diagonal first per row, so
        # we'd skip indices[indptr[u]] — but more robustly, just skip
        # any j == u (self-loop / diagonal).
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if v != u and R[v] < 0:
                R[v] = du
                queue.append(v)
    return R


def _bin_G_by_distance(per_source_G, per_source_R, min_count_per_bin=5):
    """For each integer r > 0, average G(node) across all (source, node)
    pairs at BFS distance r from their source. Returns arrays
    (r, count, G_mean, G_se) sorted by r ascending. Only positive G
    values are included (CG occasionally produces tiny negative values
    near the noise floor).
    """
    max_r = max(int(R.max()) for R in per_source_R)
    out_r, out_n, out_mean, out_se = [], [], [], []
    for r in range(1, max_r + 1):
        vals = []
        for G, R in zip(per_source_G, per_source_R):
            sel = G[R == r]
            sel = sel[sel > 0]
            if len(sel) > 0:
                vals.append(sel)
        if not vals:
            continue
        all_vals = np.concatenate(vals)
        if len(all_vals) < min_count_per_bin:
            continue
        out_r.append(r)
        out_n.append(len(all_vals))
        out_mean.append(float(all_vals.mean()))
        out_se.append(float(all_vals.std(ddof=1)) / math.sqrt(len(all_vals)))
    return (np.array(out_r, dtype=int),
            np.array(out_n, dtype=int),
            np.array(out_mean),
            np.array(out_se))


def _yukawa_fit(r_arr, n_arr, G_mean, G_se, m_input,
                r_min=2, r_max_frac=0.33):
    """Fit the lattice/graph Green's function in two complementary
    parameterisations. Both the small-mr and large-mr asymptotics of
    the d-dimensional massive Klein-Gordon propagator are pure-power
    times exponential, but with different exponents:

        small mr (r ≪ 1/m):  G(r) ≃ A / r^(d-2)
        large mr (r ≫ 1/m):  G(r) ≃ A · exp(-m·r) / r^((d-1)/2)

    These have *different* polynomial exponents — only in d=3 do they
    coincide (β = -1 in both). We do a single 3-parameter free fit
        log G(r) = c + α·r + β·log r
    on the data, recover (α, β), and report two derived dimensions:

        d_small_mr = 2 - β        (valid if observed mr ≪ 1)
        d_large_mr = 1 - 2·β      (valid if observed mr ≫ 1)

    A consistency check is that α should approximately match -m_input.
    A second check: in d=3, both formulas give d=3, so they agree
    regardless of regime; in d=4, they agree only in their dependence
    on β (small-mr expects β=-2, large-mr expects β=-1.5).

    Fit window
    ----------
    r ∈ [r_min, r_max] where r_max = min(r_max_frac · r_max_observed,
    ceil(8/m)). The first cap prevents finite-size artefacts (graph
    diameter wraparound on PBC lattices, or BFS-distance saturation
    on bounded random graphs); the second prevents fitting deep into
    the noise floor below G ≈ exp(-8) ≈ 3·10⁻⁴ relative to source.
    """
    r_max_obs = int(r_arr.max()) if len(r_arr) else 0
    r_max_finite_size = max(int(r_max_frac * r_max_obs), r_min + 2)
    r_max_noise = int(math.ceil(8.0 / max(m_input, 1e-3))) if m_input > 0 else r_max_finite_size
    r_max = min(r_max_finite_size, r_max_noise)
    in_win = ((r_arr >= r_min)
              & (r_arr <= r_max)
              & (G_mean > 0)
              & (G_se > 0))
    n_pts = int(in_win.sum())
    out = {
        "alpha": math.nan, "alpha_se": math.nan,
        "beta": math.nan, "beta_se": math.nan,
        "d_small_mr": math.nan, "d_small_mr_se": math.nan,
        "d_large_mr": math.nan, "d_large_mr_se": math.nan,
        "m_recovered": math.nan,
        "n_pts": n_pts,
        "fit_window": (float(r_arr[in_win].min()) if n_pts else math.nan,
                       float(r_arr[in_win].max()) if n_pts else math.nan),
        "in_window_mask": in_win,
    }
    if n_pts < 4:
        return out

    r_w = r_arr[in_win].astype(float)
    G_w = G_mean[in_win]
    se_w = G_se[in_win]
    log_r = np.log(r_w)
    log_G = np.log(G_w)
    relerr = se_w / G_w
    w = 1.0 / np.maximum(relerr ** 2, 1e-12)
    sw = np.sqrt(w)

    # Free 3-param fit:  log G = c + α·r + β·log r
    X = np.column_stack([np.ones_like(r_w), r_w, log_r])
    sol, _, _, _ = np.linalg.lstsq(X * sw[:, None], log_G * sw, rcond=None)
    c_fit, alpha, beta = sol

    # Covariance for SEs
    resid = log_G - (c_fit + alpha * r_w + beta * log_r)
    chi2 = float(np.sum(w * resid ** 2))
    dof = max(n_pts - 3, 1)
    sigma2 = chi2 / dof
    XtWX = X.T @ (w[:, None] * X)
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
        alpha_se = math.sqrt(max(cov[1, 1], 0.0))
        beta_se = math.sqrt(max(cov[2, 2], 0.0))
    except np.linalg.LinAlgError:
        alpha_se = beta_se = math.nan

    out.update({
        "alpha": alpha, "alpha_se": alpha_se,
        "beta": beta, "beta_se": beta_se,
        "d_small_mr": 2.0 - beta,
        "d_small_mr_se": beta_se,
        "d_large_mr": 1.0 - 2.0 * beta,
        "d_large_mr_se": 2.0 * beta_se,
        "m_recovered": -alpha,
    })
    return out


@dataclass
class FieldResult:
    cell: CellState
    masses: np.ndarray
    n_sources: int
    sources: list                 # node indices used
    per_mass: dict = field(default_factory=dict)
    # per_mass[m] = {
    #   "r": np.ndarray, "n": np.ndarray, "G_mean": np.ndarray,
    #   "G_se": np.ndarray, "fit": dict (from _yukawa_fit),
    #   "wall_s": float,
    # }


def run_field_test(cell: CellState, masses, n_sources, seed,
                   r_min=2, r_max_frac=0.33):
    """Solve (L + m²I) G = δ_x for sampled source nodes, bin G by BFS
    distance, fit free 3-parameter (const, α, β) form. Returns
    FieldResult.

    Uses scipy.sparse.linalg.factorized() so a single LU/superLU
    factorisation per mass is reused across all source vectors at that
    mass. For N_eff ≲ 10⁶ and our sparse Laplacians this is comfortably
    faster than CG.
    """
    rng = np.random.default_rng(seed * 7919 + 31)
    sources = sorted(rng.choice(cell.N_eff, size=n_sources, replace=False).tolist())
    print(f"    [field] {n_sources} sources, {len(masses)} masses, "
          f"N_eff={cell.N_eff}", flush=True)

    # Convert L to CSC once for the direct solver — factorized() builds
    # an LU via SuperLU, which prefers CSC.
    L_csc = cell.L_csr.tocsc()
    eye_csc = sp.eye(cell.N_eff, format="csc")

    # Pre-compute BFS distances for each source. These don't depend on m.
    print(f"    [field] computing BFS distances from {n_sources} sources...",
          flush=True)
    t0 = time.time()
    per_source_R = [_bfs_distances_csr(cell.L_csr, s) for s in sources]
    print(f"    [field] BFS done in {time.time()-t0:.1f}s",
          flush=True)

    result = FieldResult(cell=cell, masses=np.asarray(masses, dtype=float),
                         n_sources=n_sources, sources=sources)

    for m in result.masses:
        t0 = time.time()
        op = (L_csc + (m * m) * eye_csc).tocsc()
        try:
            solve = spla.factorized(op)
        except RuntimeError as e:
            print(f"    [field] m={m}: factorisation failed ({e}); "
                  f"falling back to spsolve per source", flush=True)
            solve = lambda b: spla.spsolve(op, b)

        per_source_G = []
        for src in sources:
            b = np.zeros(cell.N_eff, dtype=np.float64)
            b[src] = 1.0
            G = solve(b)
            G = np.asarray(G, dtype=np.float64)
            per_source_G.append(G)

        r_arr, n_arr, G_mean, G_se = _bin_G_by_distance(
            per_source_G, per_source_R)
        fit = _yukawa_fit(r_arr, n_arr, G_mean, G_se, m,
                          r_min=r_min, r_max_frac=r_max_frac)

        wall = time.time() - t0
        result.per_mass[float(m)] = {
            "r": r_arr, "n": n_arr,
            "G_mean": G_mean, "G_se": G_se,
            "fit": fit, "wall_s": wall,
        }
        print(f"    [field] m={m:.3f}: n_pts={fit['n_pts']:>3d}, "
              f"β={fit['beta']:.3f}±{fit['beta_se']:.3f}, "
              f"d_smallmr={fit['d_small_mr']:.2f}, "
              f"d_largemr={fit['d_large_mr']:.2f}, "
              f"m_rec={fit['m_recovered']:.3f} "
              f"({wall:.1f}s)", flush=True)

    return result


# ═══════════════════════════════════════════════════════════════════
#  Field-test outputs: CSV + HTML
# ═══════════════════════════════════════════════════════════════════
def write_field_csvs(res: FieldResult, prefix: str):
    """Write two CSVs: a long-format curves file (one row per (m, r)
    bin) and a summary file (one row per mass)."""
    curves_path = f"{prefix}_curves.csv"
    with open(curves_path, "w") as f:
        f.write("m,r,count,G_mean,G_se,in_fit_window\n")
        for m in res.masses:
            d = res.per_mass[float(m)]
            mask = d["fit"]["in_window_mask"]
            for i in range(len(d["r"])):
                f.write(f"{m:.6f},{d['r'][i]:d},{d['n'][i]:d},"
                        f"{d['G_mean'][i]:.6e},{d['G_se'][i]:.6e},"
                        f"{int(mask[i]) if i < len(mask) else 0}\n")

    summary_path = f"{prefix}_summary.csv"
    with open(summary_path, "w") as f:
        f.write("m_input,n_pts,alpha,alpha_se,beta,beta_se,"
                "d_small_mr,d_large_mr,m_recovered,"
                "fit_r_min,fit_r_max,wall_s\n")
        for m in res.masses:
            d = res.per_mass[float(m)]; fit = d["fit"]
            f.write(f"{m:.6f},{fit['n_pts']:d},"
                    f"{fit['alpha']:.6f},{fit['alpha_se']:.6f},"
                    f"{fit['beta']:.6f},{fit['beta_se']:.6f},"
                    f"{fit['d_small_mr']:.6f},{fit['d_large_mr']:.6f},"
                    f"{fit['m_recovered']:.6f},"
                    f"{fit['fit_window'][0]:.3f},{fit['fit_window'][1]:.3f},"
                    f"{d['wall_s']:.2f}\n")

    return curves_path, summary_path


def _render_field_svg(res: FieldResult,
                      d_min=0.0, d_max=8.0, d_ref=4.0):
    """Two-panel SVG. Top: log G(r) vs r per mass with fit overlays.
    Bottom: recovered d (fixed-m and free-m fits) vs m, with d=4
    reference line.
    """
    # Layout: top panel takes 60% of height, bottom 40%, vertical gap.
    PH_TOTAL = 720
    GAP = 30
    top_h = int((PH_TOTAL - MT - MB - GAP) * 0.6)
    bot_h = (PH_TOTAL - MT - MB - GAP) - top_h
    plot_w = PW - ML - MR

    palette = ["#4ec9b0", "#dcdcaa", "#9cdcfe", "#ce9178", "#c586c0",
               "#f48771", "#b5cea8", "#7ec96e"]

    # ── Top panel: log G(r) vs r, each mass a line ──
    # Collect all r and G values across masses to set axis ranges.
    all_r, all_G = [], []
    for m in res.masses:
        d = res.per_mass[float(m)]
        all_r.append(d["r"]); all_G.append(d["G_mean"][d["G_mean"] > 0])
    all_r_cat = np.concatenate(all_r) if all_r else np.array([1])
    all_G_cat = np.concatenate(all_G) if all_G else np.array([1.0])
    r_max = max(int(all_r_cat.max()), 4)
    log_G_min = float(np.log10(max(all_G_cat.min(), 1e-30)))
    log_G_max = float(np.log10(all_G_cat.max() * 1.2))
    if log_G_max <= log_G_min:
        log_G_max = log_G_min + 1.0

    def x_top(r):
        return ML + (r / r_max) * plot_w

    def y_top(G):
        if G <= 0 or not np.isfinite(G):
            return None
        ly = math.log10(G)
        return MT + (1 - (ly - log_G_min) / (log_G_max - log_G_min)) * top_h

    bot_y0 = MT + top_h + GAP
    m_min = float(res.masses.min())
    m_max = float(res.masses.max())
    if m_max <= m_min:
        m_max = m_min + 0.1

    def x_bot(m):
        return ML + (m - m_min) / (m_max - m_min) * plot_w

    def y_bot(d):
        return bot_y0 + (1 - (d - d_min) / (d_max - d_min)) * bot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {PW} {PH_TOTAL}" '
               f'style="font-family: ui-monospace, monospace; '
               f'background: #0d0d12;">')

    # Title
    svg.append(f'<text x="{PW/2}" y="20" fill="#ddd" font-size="13" '
               f'text-anchor="middle" font-weight="bold">'
               f'Field propagation — {res.cell.label}</text>')

    # ── Top frame ──
    svg.append(f'<rect x="{ML}" y="{MT}" width="{plot_w}" height="{top_h}" '
               f'fill="#15151b" stroke="#444" stroke-width="1"/>')
    svg.append(f'<text x="{ML+plot_w/2}" y="{MT+top_h+22}" fill="#aaa" '
               f'font-size="11" text-anchor="middle">'
               f'BFS distance r from source</text>')
    svg.append(f'<text x="20" y="{MT+top_h/2}" fill="#aaa" font-size="11" '
               f'text-anchor="middle" transform="rotate(-90 20 {MT+top_h/2})">'
               f'log10 G(r)</text>')
    # Y gridlines on top
    for ly in range(int(math.ceil(log_G_min)), int(math.floor(log_G_max)) + 1):
        y = MT + (1 - (ly - log_G_min) / (log_G_max - log_G_min)) * top_h
        svg.append(f'<line x1="{ML}" x2="{ML+plot_w}" y1="{y:.1f}" '
                   f'y2="{y:.1f}" stroke="#2a2a35" stroke-width="0.5"/>')
        svg.append(f'<text x="{ML-8}" y="{y+3:.1f}" fill="#888" '
                   f'font-size="9" text-anchor="end">{ly}</text>')
    # X ticks on top (linear r)
    n_x_ticks = 6
    for i in range(n_x_ticks + 1):
        r_tick = int(round(i / n_x_ticks * r_max))
        x = x_top(r_tick)
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" '
                   f'y2="{MT+top_h}" stroke="#2a2a35" stroke-width="0.5"/>')
        svg.append(f'<text x="{x:.1f}" y="{MT+top_h+11}" fill="#888" '
                   f'font-size="9" text-anchor="middle">{r_tick}</text>')

    # Curves on top + fit overlays
    for idx, m in enumerate(res.masses):
        d = res.per_mass[float(m)]; color = palette[idx % len(palette)]
        # Data points
        pts = []
        for i in range(len(d["r"])):
            yv = y_top(d["G_mean"][i])
            if yv is None: continue
            pts.append((x_top(d["r"][i]), yv))
        if len(pts) > 1:
            path_d = "M" + f"{pts[0][0]:.1f},{pts[0][1]:.1f}" + \
                     " " + " ".join(f"L{x:.1f},{y:.1f}" for (x, y) in pts[1:])
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" '
                       f'stroke-width="1.6" opacity="0.9"/>')
        # Fit-window markers (filled circles)
        mask = d["fit"]["in_window_mask"]
        for i in range(len(d["r"])):
            if i < len(mask) and mask[i]:
                yv = y_top(d["G_mean"][i])
                if yv is None: continue
                svg.append(f'<circle cx="{x_top(d["r"][i]):.1f}" '
                           f'cy="{yv:.1f}" r="2.5" fill="{color}" '
                           f'stroke="none"/>')
        # Fit overlay: reconstruct from the free fit's (alpha, beta).
        # log G = c + α·r + β·log r
        fit = d["fit"]
        if np.isfinite(fit["beta"]) and fit["n_pts"] >= 4:
            r_min_w, r_max_w = fit["fit_window"]
            alpha = fit["alpha"]; beta = fit["beta"]
            # Recover c by anchoring at the median data point in window
            mask_ = mask
            r_in = d["r"][mask_].astype(float)
            G_in = d["G_mean"][mask_]
            c_arr = np.log(G_in) - alpha * r_in - beta * np.log(r_in)
            c_fit = float(np.median(c_arr))
            fit_pts = []
            for r_q in np.linspace(r_min_w, r_max_w, 30):
                lg_e = c_fit + alpha * r_q + beta * math.log(r_q)
                yv = y_top(math.exp(lg_e))
                if yv is not None:
                    fit_pts.append((x_top(r_q), yv))
            if len(fit_pts) > 1:
                fp = "M" + f"{fit_pts[0][0]:.1f},{fit_pts[0][1]:.1f}" + \
                     " " + " ".join(f"L{x:.1f},{y:.1f}"
                                    for (x, y) in fit_pts[1:])
                svg.append(f'<path d="{fp}" fill="none" stroke="{color}" '
                           f'stroke-width="1" stroke-dasharray="4,3" '
                           f'opacity="0.55"/>')

        # Legend
        ly = MT + 14 + idx * 16
        svg.append(f'<rect x="{ML+12}" y="{ly-7}" width="14" height="3" '
                   f'fill="{color}"/>')
        svg.append(f'<text x="{ML+32}" y="{ly-2}" fill="#ccc" '
                   f'font-size="10">m={m:.3f} → β={fit["beta"]:.2f}'
                   f'±{fit["beta_se"]:.2f} '
                   f'(d_sm={fit["d_small_mr"]:.2f} | '
                   f'd_lg={fit["d_large_mr"]:.2f})</text>')

    # ── Bottom panel: d vs m ──
    svg.append(f'<rect x="{ML}" y="{bot_y0}" width="{plot_w}" '
               f'height="{bot_h}" fill="#15151b" stroke="#444" '
               f'stroke-width="1"/>')
    svg.append(f'<text x="{ML+plot_w/2}" y="{bot_y0+bot_h+22}" '
               f'fill="#aaa" font-size="11" text-anchor="middle">'
               f'input mass m</text>')
    svg.append(f'<text x="20" y="{bot_y0+bot_h/2}" fill="#aaa" '
               f'font-size="11" text-anchor="middle" '
               f'transform="rotate(-90 20 {bot_y0+bot_h/2})">'
               f'recovered dimension d</text>')

    # d=4 reference line
    if d_min <= d_ref <= d_max:
        y = y_bot(d_ref)
        svg.append(f'<line x1="{ML}" x2="{ML+plot_w}" y1="{y:.1f}" '
                   f'y2="{y:.1f}" stroke="#7ec96e" stroke-width="1" '
                   f'stroke-dasharray="6,4" opacity="0.7"/>')
        svg.append(f'<text x="{ML+plot_w-10}" y="{y-4:.1f}" fill="#7ec96e" '
                   f'font-size="10" text-anchor="end" opacity="0.85">'
                   f'd = 4</text>')

    # Y gridlines / labels on bottom
    for d_int in range(int(math.ceil(d_min)), int(math.floor(d_max)) + 1):
        y = y_bot(d_int)
        svg.append(f'<line x1="{ML}" x2="{ML+plot_w}" y1="{y:.1f}" '
                   f'y2="{y:.1f}" stroke="#2a2a35" stroke-width="0.5"/>')
        svg.append(f'<text x="{ML-8}" y="{y+3:.1f}" fill="#888" '
                   f'font-size="9" text-anchor="end">{d_int}</text>')
    # X ticks (mass values)
    for m in res.masses:
        x = x_bot(m)
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{bot_y0}" '
                   f'y2="{bot_y0+bot_h}" stroke="#2a2a35" '
                   f'stroke-width="0.5"/>')
        svg.append(f'<text x="{x:.1f}" y="{bot_y0+bot_h+11}" '
                   f'fill="#888" font-size="9" text-anchor="middle">'
                   f'{m:.2g}</text>')

    # d_small_mr (filled) and d_large_mr (open) markers per mass.
    for idx, m in enumerate(res.masses):
        d = res.per_mass[float(m)]["fit"]
        cx = x_bot(m)
        if np.isfinite(d["d_small_mr"]):
            cy = y_bot(np.clip(d["d_small_mr"], d_min, d_max))
            if np.isfinite(d["d_small_mr_se"]):
                cy_lo = y_bot(np.clip(d["d_small_mr"] + d["d_small_mr_se"], d_min, d_max))
                cy_hi = y_bot(np.clip(d["d_small_mr"] - d["d_small_mr_se"], d_min, d_max))
                svg.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{cy_lo:.1f}" '
                           f'y2="{cy_hi:.1f}" stroke="#9cdcfe" '
                           f'stroke-width="1.5"/>')
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" '
                       f'fill="#9cdcfe" stroke="#15151b" stroke-width="1"/>')
        if np.isfinite(d["d_large_mr"]):
            cy2 = y_bot(np.clip(d["d_large_mr"], d_min, d_max))
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy2:.1f}" r="3" '
                       f'fill="none" stroke="#dcdcaa" stroke-width="1.5"/>')

    # Legend for bottom panel
    svg.append(f'<circle cx="{ML+12}" cy="{bot_y0+12}" r="3.5" '
               f'fill="#9cdcfe" stroke="none"/>')
    svg.append(f'<text x="{ML+22}" y="{bot_y0+15}" fill="#ccc" '
               f'font-size="10">d = 2 - β  (small-mr regime)</text>')
    svg.append(f'<circle cx="{ML+200}" cy="{bot_y0+12}" r="3" fill="none" '
               f'stroke="#dcdcaa" stroke-width="1.5"/>')
    svg.append(f'<text x="{ML+210}" y="{bot_y0+15}" fill="#ccc" '
               f'font-size="10">d = 1 - 2β  (large-mr regime)</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def write_field_html(res: FieldResult, path: str):
    svg = _render_field_svg(res)
    title = f"Field propagation — {res.cell.label}"
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            f"<style>body {{ background:#0a0a0e; color:#ddd; "
            f"font-family: ui-monospace, monospace; padding: 20px; }}"
            f"h1 {{ font-size: 16px; color: #7ec9ff; }}"
            f"p {{ max-width: 900px; line-height: 1.5; color: #aaa; }}"
            f"table {{ border-collapse: collapse; margin-top: 16px; }}"
            f"th, td {{ padding: 4px 12px; text-align: right; "
            f"border-bottom: 1px solid #333; font-size: 11px; }}"
            f"th {{ color: #7ec9ff; font-weight: normal; }}"
            f"</style></head><body>"
            f"<h1>{title}</h1>"
            f"<p>Massive scalar Green's function G(r) on the graph: "
            f"solve (L + m²I) G = δ_x for source nodes x, average over "
            f"{res.n_sources} sources, fit log G(r) = c + α·r + β·log r "
            f"in the scaling window. The d-dimensional propagator has "
            f"polynomial prefactor 1/r^(d-2) at small mr (massless-like) "
            f"and 1/r^((d-1)/2) at large mr (asymptotic Yukawa); we report "
            f"both interpretations. Consistency check: recovered α should "
            f"match -m_input. <b>Top panel:</b> data (solid), fit overlay "
            f"(dashed), filled circles in fit window. <b>Bottom panel:</b> "
            f"recovered d as a function of input m, in both regime "
            f"interpretations. Filled blue circles: d_small_mr = 2 - β; "
            f"open yellow: d_large_mr = 1 - 2β. Consistent recovery of "
            f"d ≈ 4 in either regime, with α ≈ -m, is the independent "
            f"confirmation that the graph carries 4D-like geometric "
            f"information beyond the heat-kernel diffusion that defined "
            f"d_s.</p>"
            f"{svg}"
            f"<table><tr><th>m_input</th><th>n_pts</th>"
            f"<th>α (≈ -m)</th><th>β</th>"
            f"<th>d_small_mr</th><th>d_large_mr</th>"
            f"<th>fit window</th></tr>")
    for m in res.masses:
        d = res.per_mass[float(m)]; fit = d["fit"]
        html += (f"<tr><td>{m:.4f}</td><td>{fit['n_pts']:d}</td>"
                 f"<td>{fit['alpha']:.3f} ± {fit['alpha_se']:.3f}</td>"
                 f"<td>{fit['beta']:.3f} ± {fit['beta_se']:.3f}</td>"
                 f"<td>{fit['d_small_mr']:.3f}</td>"
                 f"<td>{fit['d_large_mr']:.3f}</td>"
                 f"<td>r ∈ [{fit['fit_window'][0]:.1f}, "
                 f"{fit['fit_window'][1]:.1f}]</td></tr>")
    html += "</table></body></html>"
    with open(path, "w") as f:
        f.write(html)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def parse_masses(s):
    """Parse '0.05,0.1,0.2,0.4,0.8' → [0.05, 0.1, 0.2, 0.4, 0.8]."""
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Independent geometric tests on a single graph cell. "
                    "Either a basin cell from the sweep (--k --T --lb --N) "
                    "or a periodic torus reference (--torus D,L).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # basin candidate
  core/cell_tests.py --k 9 --T 0 --lb 0.95 --N 64000 --seed 42

  # 4D-torus reference for comparison (known flat d_s = 4)
  core/cell_tests.py --torus 4,12 --seed 42 --tests flow

  # 3D-torus reference for the field-test fit (small-mr and large-mr
  # regimes give the same exponent in 3D, cleanest validation)
  core/cell_tests.py --torus 3,24 --seed 42 --tests field
""")
    # Cell selection: either basin (k, T, lb, N) or torus (D, L).
    ap.add_argument("--k", type=int,
                    help="Target mean degree (basin cell mode)")
    ap.add_argument("--T", type=float,
                    help="Metropolis temperature (basin cell mode)")
    ap.add_argument("--lb", type=float,
                    help="Locality bias (basin cell mode)")
    ap.add_argument("--N", type=int,
                    help="Node count (basin cell mode; ignored in torus mode)")
    ap.add_argument("--torus", type=str, default=None,
                    help="Reference torus 'D,L' — d-dim periodic hypercubic "
                         "lattice with side L, N=L^D nodes. Mutually "
                         "exclusive with the basin-cell args. Examples: "
                         "'4,12' (4D L=12), '3,24' (3D L=24).")
    ap.add_argument("--seed", type=int, default=42)
    # Test selection
    ap.add_argument("--tests", type=str, default="flow,field",
                    help="Comma-separated subset of {flow, field}. "
                         "Default: both.")
    # Flow-test parameters
    ap.add_argument("--n-probes", type=int, default=60,
                    help="SLQ probe vectors for flow test (default 60)")
    ap.add_argument("--lanczos-m", type=int, default=300,
                    help="Lanczos steps per probe for flow test (default 300)")
    ap.add_argument("--half-window", type=int, default=10,
                    help="Local-slope half-window for flow test (default 10)")
    # Field-test parameters
    ap.add_argument("--masses", type=str, default="0.05,0.1,0.2,0.4,0.8",
                    help="Comma-separated mass values for field test "
                         "(default 0.05,0.1,0.2,0.4,0.8)")
    ap.add_argument("--n-sources", type=int, default=8,
                    help="Source nodes per mass for field test (default 8)")
    ap.add_argument("--r-min", type=int, default=2,
                    help="Minimum BFS distance for fit (default 2)")
    ap.add_argument("--r-max-frac", type=float, default=0.33,
                    help="Fit window upper bound as fraction of "
                         "max observed BFS distance (default 0.33; "
                         "protects against finite-size wraparound)")
    # Output
    ap.add_argument("--out-prefix", type=str, default=None,
                    help="Override output filename prefix.")
    args = ap.parse_args(argv)

    tests = set(t.strip() for t in args.tests.split(",") if t.strip())
    valid = {"flow", "field"}
    if not tests <= valid:
        ap.error(f"--tests must be subset of {valid}")
    if not tests:
        ap.error("No tests requested")

    # ── Mode dispatch: torus reference vs basin cell ─────────────────
    if args.torus is not None:
        # Torus reference mode. Reject conflicting basin args.
        basin_args = {"k": args.k, "T": args.T, "lb": args.lb, "N": args.N}
        provided = [name for name, val in basin_args.items() if val is not None]
        if provided:
            ap.error(f"--torus is mutually exclusive with basin args; "
                     f"got both --torus and {provided}")
        try:
            d_str, L_str = args.torus.split(",")
            torus_d = int(d_str)
            torus_L = int(L_str)
        except ValueError:
            ap.error(f"--torus must be 'D,L' (e.g. '4,12'), got '{args.torus}'")

        cell = build_torus_cell(torus_d, torus_L, args.seed)
        tag = f"torus{torus_d}d_L{torus_L}_s{args.seed}"
    else:
        # Basin cell mode. All four basin args are required.
        missing = [name for name in ["k", "T", "lb", "N"]
                   if getattr(args, name) is None]
        if missing:
            ap.error(f"basin cell mode requires {missing}; "
                     f"or pass --torus D,L for a reference torus")

        mu_table = load_mu_table()
        if not mu_table:
            raise SystemExit(f"{MU_JSON} not found or empty — run the sweep's "
                             f"μ-calibration phase first.")

        cell = build_cell(args.k, args.T, args.lb, args.N, args.seed,
                          mu_table)
        tag = (f"k{args.k}_T{args.T}_lb{args.lb}"
               f"_N{args.N}_s{args.seed}")

    prefix = args.out_prefix or tag

    if "flow" in tests:
        flow_res = run_flow_test(cell, args.n_probes, args.lanczos_m,
                                 args.half_window)
        write_flow_csv(flow_res, f"flow_{prefix}.csv")
        write_flow_html([flow_res], f"flow_{prefix}.html",
                        title=f"d_s(t) flow — {flow_res.label}")
        print(f"    [flow] → flow_{prefix}.csv, flow_{prefix}.html")

    if "field" in tests:
        masses = parse_masses(args.masses)
        field_res = run_field_test(
            cell, masses, args.n_sources, args.seed,
            r_min=args.r_min, r_max_frac=args.r_max_frac)
        c_path, s_path = write_field_csvs(field_res, f"field_{prefix}")
        write_field_html(field_res, f"field_{prefix}.html")
        print(f"    [field] → {c_path}, {s_path}, field_{prefix}.html")


if __name__ == "__main__":
    main()
