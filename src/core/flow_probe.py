"""
core/flow_probe.py — Running spectral dimension d_s(t) flow probe
==========================================================
Measures d_s(t) = -2 · d log Z / d log t as a function of diffusion
time t, instead of fitting a single power law to the whole window.

Why: a graph that's "4D-like" globally (single d_s ≈ 4) can still show
scale-dependent dimension. The signature result of CDT and asymptotic
safety is d_s flowing from ~4 in the IR (large t, large scales) down
to ~2 in the UV (small t, short scales). A regular 4D lattice gives
flat d_s = 4. This script distinguishes the two.

Usage
-----
    NOTE: `python main.py` launches only the dashboard + auto-sweep —
    there is no `main.py flow` subcommand. This module is used as a
    library (import it and call run_flow_for_cell / main()); the example
    commands below show that interface, not a main.py route.

    # Single cell from your sweep
    python main.py flow --k 8 --T 0.005 --lb 0.94 --N 4000 --seed 42

    # Multiple cells (overlay plot)
    python main.py flow --cells "8,0.005,0.94 9,0,0.95 8,0,0.99" \\
                      --N 16000 --seed 42

    # Reuse μ from existing mu_table.json (default), or override
    python main.py flow --k 8 --T 0.005 --lb 0.94 --N 4000 --seed 42 \\
                      --n-probes 60 --lanczos-m 400

Outputs (in cwd):
    flow_k{k}_T{T}_lb{lb}_N{N}_s{seed}.csv   # raw t, Z, d_s data
    flow_k{k}_T{T}_lb{lb}_N{N}_s{seed}.html  # self-contained plot
    flow_overlay.html                         # if --cells given

Design notes
------------
- Local-slope fit: W-point sliding window weighted-linear regression on
  (log t, log Z), weights = 1/Z_se². Same regression kernel as the
  existing global SLQ fit, just applied locally. W=21 by default — wide
  enough to suppress per-t noise, narrow enough that ~3:1 t-ratio steps
  in the slope are still resolved.

- Jackknife CI: SLQ already produces n_probes independent Z trajectories
  (one per Rademacher probe vector). Leave-one-probe-out → n_probes
  pseudo-samples of d_s(t). The jackknife SE at each t is
      SE(t) = √[(n−1)/n · Σ (d_s_jack_p(t) − d_s_mean(t))²]
  This captures stochastic-trace noise propagated through the local fit.
  Does NOT capture finite-N geometric error — that's what the sweep's
  N-extrapolation handles.

- Window mask: SLQ Z bounds (Z ∈ [max(10, 3e-5·N),
  0.3·N], t·λ_max ≥ 0.5, finite, relerr < 15%). Outside the mask we
  still compute and display d_s(t) but dim it in the plot — saturation
  regions on either side are informative for sanity-checking the fit
  window, but shouldn't be read as physics.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# Local imports — all from existing modules, no changes to them.
from core.physics_engine import PhysicsEngine
from metrics import get_graph_stats
from metrics.numba_kernels import _build_laplacian_csr_jit, _extract_lcc_nodes
from metrics.stochastic_lanczos import _lanczos_slq, _lanczos_slq_scipy
from core.graph_builder import build_graph
from core.project_constants import MAX_DEG, MU_JSON, PROD_SWEEPS
from core.disk_io import load_mu_table, mu_key


# ═══════════════════════════════════════════════════════════════════
#  Per-probe Z(t) accumulation — the SLQ heat-kernel inner loop
# ═══════════════════════════════════════════════════════════════════
def slq_per_probe_Z(L_indptr, L_indices, L_data, N_eff, lam_max_bound,
                    n_probes, lanczos_m, t_grid, seed=20250419,
                    return_quad=False):
    """Run SLQ; return per_probe_Z of shape (n_probes, n_t).

    The standard stochastic-Lanczos-quadrature inner loop. Kept
    separate here so we can compute d_s(t) without invoking the global
    power-law fit. Switches to scipy CSR matvec at N_eff ≥ 500k for
    the same speed reasons as the production probe.

    If return_quad=True, also returns the per-probe Gauss quadrature
    (theta, omega) as (n_probes, lanczos_m) arrays, where for each probe
    Z_p(t) = N_eff · Σ_i omega[i]·exp(-t·theta[i]). These Ritz pairs are
    the SLQ primitive; the t-grid evaluation above is just a (lossy)
    projection of them. Persisting them lets the whole d_s(t) extraction
    (t-grid, UV reach, window, smoothing, jackknife) be redone later
    WITHOUT re-running SLQ. Probes whose Lanczos terminated early are
    zero-padded (omega=0 ⇒ no contribution).
    """
    use_scipy = N_eff >= 500_000
    L_scipy = None
    if use_scipy:
        try:
            import scipy.sparse as sp
            L_scipy = sp.csr_matrix(
                (L_data, L_indices, L_indptr), shape=(N_eff, N_eff))
        except ImportError:
            use_scipy = False

    rng = np.random.default_rng(seed)
    n_t = len(t_grid)
    per_probe = np.zeros((n_probes, n_t), dtype=np.float64)
    if return_quad:
        theta_all = np.zeros((n_probes, lanczos_m), dtype=np.float64)
        omega_all = np.zeros((n_probes, lanczos_m), dtype=np.float64)

    for p in range(n_probes):
        v = rng.choice(np.array([-1.0, 1.0]), size=N_eff).astype(np.float64)
        v /= np.linalg.norm(v)
        if use_scipy:
            alphas, betas = _lanczos_slq_scipy(L_scipy, v, lanczos_m)
        else:
            alphas, betas = _lanczos_slq(
                L_indptr, L_indices, L_data, v, lanczos_m)
        k_act = len(alphas)
        T = np.diag(alphas)
        if k_act > 1:
            off = betas[:k_act - 1]
            T = T + np.diag(off, k=1) + np.diag(off, k=-1)
        theta, U = np.linalg.eigh(T)
        theta = np.maximum(theta, 0.0)
        tau = U[0, :] ** 2
        exp_mat = np.exp(-t_grid[:, None] * theta[None, :])
        per_probe[p, :] = N_eff * (exp_mat @ tau)
        if return_quad:
            m_act = min(theta.shape[0], lanczos_m)
            theta_all[p, :m_act] = theta[:m_act]
            omega_all[p, :m_act] = tau[:m_act]

    if return_quad:
        return per_probe, theta_all, omega_all
    return per_probe


# ═══════════════════════════════════════════════════════════════════
#  Local-slope estimator: d_s(t) via sliding weighted log-log fit
# ═══════════════════════════════════════════════════════════════════
def local_slope(log_t, log_Z, weights, half_window=10):
    """For each i, fit log_Z[i-w:i+w+1] = a + slope·log_t[i-w:i+w+1]
    weighted by `weights`, return slope at each i. d_s(t) = -2 · slope.

    Uses normal equations directly (cheap, N×O(W) total). Edges of the
    array use a one-sided window (clipped). NaNs are returned where
    fewer than 3 valid weighted points remain in the window.
    """
    n = len(log_t)
    slopes = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        x = log_t[lo:hi]
        y = log_Z[lo:hi]
        w = weights[lo:hi]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if finite.sum() < 3:
            continue
        xf = x[finite]; yf = y[finite]; wf = w[finite]
        # Weighted least squares — closed form for slope only is enough.
        Sw = wf.sum()
        Sx = (wf * xf).sum()
        Sy = (wf * yf).sum()
        Sxx = (wf * xf * xf).sum()
        Sxy = (wf * xf * yf).sum()
        denom = Sw * Sxx - Sx * Sx
        if abs(denom) < 1e-30:
            continue
        slopes[i] = (Sw * Sxy - Sx * Sy) / denom
    return slopes


def compute_ds_flow(per_probe_Z, t_grid, N_eff, lam_max_bound,
                    half_window=10):
    """Return dict with t_grid, Z_mean, Z_se, d_s_mean, d_s_se,
    in_window mask. Uses jackknife over probes for d_s SE.
    """
    n_probes, n_t = per_probe_Z.shape
    Z_mean = per_probe_Z.mean(axis=0)
    Z_se = per_probe_Z.std(axis=0, ddof=1) / math.sqrt(n_probes)

    # Power-law window mask — standard SLQ window criteria.
    min_Z = max(10.0, N_eff * 3e-5)
    max_Z = 0.3 * N_eff
    min_t_lam = 0.5 / lam_max_bound
    Z_relerr = Z_se / np.maximum(np.abs(Z_mean), 1e-20)
    in_window = (
        (Z_mean > min_Z) & (Z_mean < max_Z)
        & (Z_relerr < 0.15)
        & (t_grid >= min_t_lam)
        & np.isfinite(Z_mean)
    )

    # log-log transform on the full grid (so we can also show d_s
    # outside the window, dimmed). Mask Z<=0 with NaN.
    safe_Z = np.where(Z_mean > 0, Z_mean, np.nan)
    log_t = np.log(t_grid)
    log_Z = np.log(safe_Z)
    weights = np.where(np.isfinite(Z_relerr) & (Z_relerr > 0),
                       1.0 / (Z_relerr ** 2), 0.0)

    # Mean d_s(t)
    slope_mean = local_slope(log_t, log_Z, weights, half_window)
    d_s_mean = -2.0 * slope_mean

    # Jackknife: leave-one-probe-out
    d_s_jack = np.full((n_probes, n_t), np.nan, dtype=np.float64)
    for p in range(n_probes):
        idx = np.r_[0:p, p + 1:n_probes]
        Zp = per_probe_Z[idx, :].mean(axis=0)
        Zp_se = per_probe_Z[idx, :].std(axis=0, ddof=1) / math.sqrt(n_probes - 1)
        Zp_safe = np.where(Zp > 0, Zp, np.nan)
        Zp_relerr = Zp_se / np.maximum(np.abs(Zp), 1e-20)
        wp = np.where(np.isfinite(Zp_relerr) & (Zp_relerr > 0),
                      1.0 / (Zp_relerr ** 2), 0.0)
        sl = local_slope(log_t, np.log(Zp_safe), wp, half_window)
        d_s_jack[p, :] = -2.0 * sl

    # Jackknife SE: √[(n-1)/n · Σ (jack_p - mean)²]
    diffs = d_s_jack - d_s_mean[None, :]
    finite = np.isfinite(diffs)
    n_eff = finite.sum(axis=0)
    sumsq = np.nansum(diffs ** 2, axis=0)
    d_s_se = np.where(
        n_eff >= 2,
        np.sqrt((n_eff - 1) / np.maximum(n_eff, 1) * sumsq),
        np.nan,
    )

    return {
        "t_grid": t_grid,
        "Z_mean": Z_mean,
        "Z_se": Z_se,
        "d_s_mean": d_s_mean,
        "d_s_se": d_s_se,
        "in_window": in_window,
        "n_probes": n_probes,
        "N_eff": N_eff,
        "lam_max_bound": lam_max_bound,
    }


# ═══════════════════════════════════════════════════════════════════
#  Build a graph from sweep params, then run flow probe
# ═══════════════════════════════════════════════════════════════════
@dataclass
class FlowResult:
    k: int
    T: float
    lb: float
    N: int
    seed: int
    label: str
    flow: dict     # the dict returned by compute_ds_flow
    stats: object  # GraphStats
    quad: object = None   # Tier-2: per-probe SLQ Ritz quadrature (or None)


def _quad_dict(theta_all, omega_all, t_grid, N_eff, lam_max_bound,
               lanczos_m, n_probes):
    """Bundle the per-probe Ritz quadrature for persistence. Z(t) =
    N_eff · mean_p Σ_i omega[p,i]·exp(-t·theta[p,i])."""
    return {
        "theta": theta_all, "omega": omega_all, "t_grid": np.asarray(t_grid),
        "N_eff": int(N_eff), "lam_max_bound": float(lam_max_bound),
        "lanczos_m": int(lanczos_m), "n_probes": int(n_probes),
    }


# ═══════════════════════════════════════════════════════════════════
#  Shared LCC-Laplacian + t-grid construction
# ═══════════════════════════════════════════════════════════════════
# Both run_flow_for_cell (here) and cell_tests._build_cell_finalise need
# the Laplacian restricted to the largest connected component, plus the
# d_s≈4 t-grid. Defined once here so the two callers can't drift apart.
def build_lcc_laplacian(eng, N):
    """Restrict the engine's graph to its largest connected component and
    build the CSR graph Laplacian on it.

    Returns (N_eff, L_indptr, L_indices, L_data, max_deg_eff) where N_eff
    is the LCC size (== N when the graph is already connected) and
    max_deg_eff is the maximum degree within the LCC.
    """
    lcc = _extract_lcc_nodes(eng.node_neighbors, eng.node_degrees, N)
    n_lcc = len(lcc)
    if n_lcc < N:
        old_to_new = np.full(N, -1, dtype=np.int32)
        old_to_new[lcc] = np.arange(n_lcc, dtype=np.int32)
        lcc_neighbors = np.full((n_lcc, eng.node_neighbors.shape[1]),
                                -1, dtype=np.int32)
        lcc_degrees = np.zeros(n_lcc, dtype=np.int32)
        for new_i in range(n_lcc):
            old_i = lcc[new_i]
            d = 0
            for kk in range(eng.node_degrees[old_i]):
                old_j = eng.node_neighbors[old_i, kk]
                new_j = old_to_new[old_j]
                if new_j >= 0:
                    lcc_neighbors[new_i, d] = new_j
                    d += 1
            lcc_degrees[new_i] = d
        N_eff = n_lcc
        L_ip, L_id, L_dt = _build_laplacian_csr_jit(
            lcc_neighbors, lcc_degrees, n_lcc)
        max_deg_eff = int(lcc_degrees.max())
    else:
        N_eff = N
        L_ip, L_id, L_dt = _build_laplacian_csr_jit(
            eng.node_neighbors, eng.node_degrees, N)
        max_deg_eff = int(eng.node_degrees.max())
    return N_eff, L_ip, L_id, L_dt, max_deg_eff


def make_slq_t_grid(N_eff, lam_max_bound, d_s_guess=4.0, n_points=240):
    """Log-spaced heat-kernel time grid for the SLQ flow.

    d_s_guess is nudged toward 4 because we investigate cells where
    d_s ≈ 4; this concentrates the grid in the relevant power-law window.
    Returns (t_grid, t_lo, t_hi).
    """
    t_lo = 0.2 / lam_max_bound
    t_hi = min(3.0 * float(N_eff) ** (2.0 / d_s_guess), 1e10)
    t_grid = np.logspace(np.log10(t_lo), np.log10(t_hi), n_points)
    return t_grid, t_lo, t_hi


def run_flow_for_cell(k, T, lb, N, seed, n_probes, lanczos_m, mu_table,
                      half_window=10, ec=-1.0):
    """Build a graph at (k, T, lb, N, seed) the same way the worker does,
    then run SLQ and compute d_s(t). Returns FlowResult.
    """
    key = mu_key(k, T, lb)
    if key not in mu_table:
        raise SystemExit(
            f"no μ entry for {key} in {MU_JSON} — run the sweep's "
            f"μ-calibration phase first, or pass --mu manually.")
    mu = float(mu_table[key])

    print(f"  [{time.strftime('%H:%M:%S')}] building graph "
          f"k={k} T={T} lb={lb} N={N} seed={seed} μ={mu:.6f}",
          flush=True)
    eng, stats, sweeps, peak_deg = build_graph(
        N=N, k=k, T=T, lb=lb, mu=mu, ec=ec, seed=seed,
        max_deg=MAX_DEG, log_tag=None)
    print(f"    therm sweeps={sweeps}, k_avg={stats.k_avg:.3f}, "
          f"edges={stats.edges}, lcc={stats.lcc_pct:.1f}%, "
          f"k_top={peak_deg}",
          flush=True)

    # Build Laplacian on the largest connected component.
    N_eff, L_ip, L_id, L_dt, max_deg_eff = build_lcc_laplacian(eng, N)
    lam_max_bound = 2 * max_deg_eff
    t_grid, t_lo, t_hi = make_slq_t_grid(N_eff, lam_max_bound)

    print(f"    SLQ: n_probes={n_probes}, lanczos_m={lanczos_m}, "
          f"N_eff={N_eff}, t_grid=[{t_lo:.2g}, {t_hi:.2g}]",
          flush=True)
    t0 = time.time()
    per_probe, theta_all, omega_all = slq_per_probe_Z(
        L_ip, L_id, L_dt, N_eff, lam_max_bound,
        n_probes, lanczos_m, t_grid, seed=seed * 1000 + 1, return_quad=True)
    print(f"    SLQ done in {time.time()-t0:.1f}s, computing flow...",
          flush=True)
    flow = compute_ds_flow(per_probe, t_grid, N_eff, lam_max_bound,
                           half_window=half_window)

    label = f"k={k} T={T} lb={lb} N={N} s={seed}"
    return FlowResult(k=k, T=T, lb=lb, N=N, seed=seed,
                      label=label, flow=flow, stats=stats,
                      quad=_quad_dict(theta_all, omega_all, t_grid, N_eff,
                                      lam_max_bound, lanczos_m, n_probes))


# ═══════════════════════════════════════════════════════════════════
#  Output: CSV + self-contained HTML/SVG plot
# ═══════════════════════════════════════════════════════════════════
def write_csv(res: FlowResult, path: str):
    # Atomic write: a worker killed mid-write (SIGTERM/SIGKILL from a stop,
    # a hot-reload, or an OOM) must never leave a half-written flow_*.csv.
    # The sweep's resume logic treats any existing flow_*.csv as a finished
    # cell, so a truncated file would silently poison the restart. Writing
    # to a temp file in the same directory and os.replace()-ing it into
    # place means the final path only ever appears complete (os.replace is
    # atomic on POSIX) — a killed worker leaves at most a stray .tmp.
    f = res.flow
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            fh.write("t,Z_mean,Z_se,d_s_mean,d_s_se,in_window\n")
            for i in range(len(f["t_grid"])):
                fh.write(f"{f['t_grid'][i]:.6e},"
                         f"{f['Z_mean'][i]:.6e},"
                         f"{f['Z_se'][i]:.6e},"
                         f"{f['d_s_mean'][i]:.6f},"
                         f"{f['d_s_se'][i]:.6f},"
                         f"{int(f['in_window'][i])}\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def write_quad_npz(res: FlowResult, path: str):
    """Tier-2 sidecar: persist the per-probe SLQ Ritz quadrature so the
    full d_s(t) extraction can be redone post-hoc WITHOUT re-running SLQ —
        Z(t) = N_eff · mean_p Σ_i omega[p,i]·exp(-t·theta[p,i])
    i.e. you can change the t-grid, push the UV lower, re-window, re-smooth,
    or re-jackknife straight from this file. ~150 KB/cell vs the hours SLQ
    costs. theta kept float64 (the small near-zero eigenvalues govern the IR);
    omega float32 (weights in [0,1]). Atomic; no-op if res.quad is None.
    Passing a file handle to np.savez_compressed avoids its .npz auto-suffix
    so os.replace lands exactly on `path`.
    """
    q = res.quad
    if q is None:
        return
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                theta=np.asarray(q["theta"], dtype=np.float64),
                omega=np.asarray(q["omega"], dtype=np.float32),
                t_grid=np.asarray(q["t_grid"], dtype=np.float64),
                N_eff=np.int64(q["N_eff"]),
                lam_max_bound=np.float64(q["lam_max_bound"]),
                lanczos_m=np.int64(q["lanczos_m"]),
                n_probes=np.int64(q["n_probes"]),
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def reconstruct_Z(npz_path, t_grid=None):
    """Inverse of write_quad_npz: rebuild Z(t) from a saved quadrature on
    any t-grid (defaults to the one used at capture). Returns (t_grid, Z).
    This is what makes a re-grid / deeper-UV analysis free."""
    d = np.load(npz_path)
    theta = d["theta"].astype(np.float64)         # (P, m)
    omega = d["omega"].astype(np.float64)         # (P, m)
    N_eff = float(d["N_eff"])
    t = np.asarray(t_grid, dtype=np.float64) if t_grid is not None \
        else d["t_grid"].astype(np.float64)
    # Z_p(t) = N_eff Σ_i omega[p,i] exp(-t theta[p,i]); average over probes.
    P = theta.shape[0]
    Z = np.zeros(t.shape[0], dtype=np.float64)
    for p in range(P):
        Z += N_eff * (np.exp(-t[:, None] * theta[p][None, :]) @ omega[p])
    return t, Z / P
# Plot dimensions in SVG user units (CSS scales the rendered size).
PW, PH = 1100, 540
ML, MR, MT, MB = 70, 40, 30, 60   # plot margins inside SVG


def _svg_path(xs, ys, in_win):
    """Build a polyline 'd' attribute with M/L commands, breaking the
    path on NaN values so gaps don't visually connect across them.
    Splits into in-window and out-of-window segments for separate
    styling (solid vs dashed/dimmed).
    """
    in_segments, out_segments = [], []
    cur_in, cur_out = [], []
    for x, y, w in zip(xs, ys, in_win):
        if not (np.isfinite(x) and np.isfinite(y)):
            if cur_in:  in_segments.append(cur_in);  cur_in = []
            if cur_out: out_segments.append(cur_out); cur_out = []
            continue
        if w:
            if cur_out: out_segments.append(cur_out); cur_out = []
            cur_in.append((x, y))
        else:
            if cur_in:  in_segments.append(cur_in);  cur_in = []
            cur_out.append((x, y))
    if cur_in:  in_segments.append(cur_in)
    if cur_out: out_segments.append(cur_out)

    def to_d(segs):
        parts = []
        for seg in segs:
            if not seg: continue
            parts.append(f"M{seg[0][0]:.2f},{seg[0][1]:.2f}")
            parts += [f"L{x:.2f},{y:.2f}" for (x, y) in seg[1:]]
        return " ".join(parts)
    return to_d(in_segments), to_d(out_segments)


def _render_svg(results, title, ds_min=None, ds_max=None):
    """Render one or more FlowResults into an SVG string.

    Linear y-axis on d_s, log x-axis on t. Reference lines at d_s=2
    and d_s=4 are dashed/labeled. CI band is ±1σ from jackknife.

    Y-axis bounds: by default auto-scaled to the actual in-window
    d_s range across all results, padded with a margin and floored
    at [0.5, 4.5] so the d=2 and d=4 reference lines are always
    present. Pass explicit ds_min / ds_max to override.

    Out-of-range data points are clipped at the plot edge but
    plotted in a desaturated style with a top/bottom marker so it
    is visibly clear that the rendered line does not reflect the
    underlying value — rather than silently ramming the curve flat
    at the y-axis limit, which was the behaviour of an earlier
    version that produced misleading "plateau at ds_max" readings.
    """
    # X range: union over all results' t_grids.
    t_all = np.concatenate([r.flow["t_grid"] for r in results])
    t_min, t_max = float(t_all.min()), float(t_all.max())
    log_t_min, log_t_max = math.log10(t_min), math.log10(t_max)

    # Auto y-range: take the in-window finite d_s values across all
    # results, find min/max, pad by 0.5 each side, and ensure both
    # reference lines (d_s=2 and d_s=4) remain visible.
    if ds_min is None or ds_max is None:
        all_in_window_ds = []
        for r in results:
            f = r.flow
            in_w = f["in_window"]
            d_finite = f["d_s_mean"][in_w]
            d_finite = d_finite[np.isfinite(d_finite)]
            if len(d_finite):
                all_in_window_ds.append(d_finite)
        if all_in_window_ds:
            cat = np.concatenate(all_in_window_ds)
            data_min = float(cat.min())
            data_max = float(cat.max())
        else:
            data_min, data_max = 1.0, 5.0
        if ds_min is None:
            ds_min = min(0.5, data_min - 0.5)
        if ds_max is None:
            # Round up to next 0.5 so y-axis labels are clean
            ds_max = max(4.5, math.ceil((data_max + 0.5) * 2) / 2)

    plot_w = PW - ML - MR
    plot_h = PH - MT - MB

    def x_of(t):  return ML + (math.log10(t) - log_t_min) / (log_t_max - log_t_min) * plot_w
    def y_of(d):  return MT + (1 - (d - ds_min) / (ds_max - ds_min)) * plot_h

    # Color palette for overlay mode.
    palette = ["#4ec9b0", "#dcdcaa", "#9cdcfe", "#ce9178", "#c586c0",
               "#f48771", "#b5cea8"]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {PW} {PH}" '
               f'style="font-family: ui-monospace, monospace; '
               f'background: #0d0d12;">')

    # Plot frame
    svg.append(f'<rect x="{ML}" y="{MT}" width="{plot_w}" height="{plot_h}" '
               f'fill="#15151b" stroke="#444" stroke-width="1"/>')

    # Reference lines at d_s=2 and d_s=4.
    for d_ref, label, color in [(4, "d_s = 4 (4D-like)", "#7ec96e"),
                                 (2, "d_s = 2 (UV / branched-polymer)", "#e5b06b")]:
        if ds_min <= d_ref <= ds_max:
            y = y_of(d_ref)
            svg.append(f'<line x1="{ML}" x2="{ML+plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                       f'stroke="{color}" stroke-width="1" stroke-dasharray="6,4" '
                       f'opacity="0.7"/>')
            svg.append(f'<text x="{ML+plot_w-10}" y="{y-4:.1f}" fill="{color}" '
                       f'font-size="11" text-anchor="end" opacity="0.85">{label}</text>')

    # Y gridlines + labels.
    for d in range(int(math.ceil(ds_min)), int(math.floor(ds_max)) + 1):
        y = y_of(d)
        svg.append(f'<line x1="{ML}" x2="{ML+plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="#2a2a35" stroke-width="0.5"/>')
        svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" fill="#888" font-size="10" '
                   f'text-anchor="end">{d}</text>')

    # X gridlines + labels (one per decade).
    for log_t in range(int(math.floor(log_t_min)), int(math.ceil(log_t_max)) + 1):
        x = x_of(10 ** log_t)
        if x < ML or x > ML + plot_w: continue
        svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{MT+plot_h}" '
                   f'stroke="#2a2a35" stroke-width="0.5"/>')
        svg.append(f'<text x="{x:.1f}" y="{MT+plot_h+15}" fill="#888" '
                   f'font-size="10" text-anchor="middle">10<tspan dy="-4" '
                   f'font-size="8">{log_t}</tspan></text>')

    # Axis titles.
    svg.append(f'<text x="{ML+plot_w/2}" y="{MT+plot_h+40}" fill="#aaa" '
               f'font-size="12" text-anchor="middle">diffusion time t (log)</text>')
    svg.append(f'<text x="20" y="{MT+plot_h/2}" fill="#aaa" font-size="12" '
               f'text-anchor="middle" transform="rotate(-90 20 {MT+plot_h/2})">'
               f'd_s(t) — running spectral dimension</text>')
    svg.append(f'<text x="{PW/2}" y="20" fill="#ddd" font-size="13" '
               f'text-anchor="middle" font-weight="bold">{title}</text>')

    # Each result: CI band + mean line.
    for idx, r in enumerate(results):
        f = r.flow
        color = palette[idx % len(palette)]
        t = f["t_grid"]
        d_mean = f["d_s_mean"]
        d_se = f["d_s_se"]
        in_w = f["in_window"]

        # Build CI polygon (only over in-window region for clarity).
        upper, lower = d_mean + d_se, d_mean - d_se
        poly = []
        valid = np.isfinite(d_mean) & np.isfinite(d_se) & in_w
        idx_valid = np.where(valid)[0]
        if len(idx_valid) > 1:
            for i in idx_valid:
                poly.append((x_of(t[i]), y_of(np.clip(upper[i], ds_min, ds_max))))
            for i in reversed(idx_valid):
                poly.append((x_of(t[i]), y_of(np.clip(lower[i], ds_min, ds_max))))
            pts = " ".join(f"{x:.1f},{y:.1f}" for (x, y) in poly)
            svg.append(f'<polygon points="{pts}" fill="{color}" '
                       f'opacity="0.15" stroke="none"/>')

        # Main line. We split each point into one of three categories:
        #   in_range_in_window     — solid line, full color
        #   in_range_out_of_window — dashed dim line (ballistic/saturation)
        #   out_of_range           — clipped at axis edge, drawn but with
        #                            ↑/↓ arrow marker so the reader can
        #                            see that the curve continues beyond
        #                            the visible y-axis. (This was the
        #                            bug fix: silent np.clip used to make
        #                            it look like a flat plateau.)
        xs = np.array([x_of(tt) for tt in t])
        in_range = np.isfinite(d_mean) & (d_mean >= ds_min) & (d_mean <= ds_max)
        # ys for plotting: clip at axis edge but track whether each point
        # was clipped, for marker drawing.
        ys_plot = np.where(np.isfinite(d_mean),
                           np.clip(d_mean, ds_min, ds_max),
                           np.nan)
        ys = np.array([y_of(yy) if np.isfinite(yy) else np.nan
                       for yy in ys_plot])
        in_d, out_d = _svg_path(xs, ys, in_w)
        if out_d:
            svg.append(f'<path d="{out_d}" fill="none" stroke="{color}" '
                       f'stroke-width="1" opacity="0.3" stroke-dasharray="3,3"/>')
        if in_d:
            svg.append(f'<path d="{in_d}" fill="none" stroke="{color}" '
                       f'stroke-width="2"/>')

        # Out-of-range markers — small triangles at the clipped edge
        # pointing in the direction the curve actually went.
        clipped_high = np.where(np.isfinite(d_mean) & (d_mean > ds_max))[0]
        clipped_low = np.where(np.isfinite(d_mean) & (d_mean < ds_min))[0]
        for i in clipped_high[::3]:  # one marker every 3 points to avoid clutter
            cx = x_of(t[i])
            cy = y_of(ds_max)
            svg.append(f'<polygon points="{cx:.1f},{cy+1:.1f} '
                       f'{cx-4:.1f},{cy+8:.1f} {cx+4:.1f},{cy+8:.1f}" '
                       f'fill="{color}" opacity="0.85"/>')
        for i in clipped_low[::3]:
            cx = x_of(t[i])
            cy = y_of(ds_min)
            svg.append(f'<polygon points="{cx:.1f},{cy-1:.1f} '
                       f'{cx-4:.1f},{cy-8:.1f} {cx+4:.1f},{cy-8:.1f}" '
                       f'fill="{color}" opacity="0.85"/>')

        # Legend entry
        ly = MT + 12 + idx * 18
        svg.append(f'<rect x="{ML+12}" y="{ly-8}" width="14" height="3" '
                   f'fill="{color}"/>')
        svg.append(f'<text x="{ML+32}" y="{ly-3}" fill="#ccc" font-size="11">'
                   f'{r.label}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def write_html(results, path, title):
    svg = _render_svg(results, title)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            f"<style>body {{ background:#0a0a0e; color:#ddd; "
            f"font-family: ui-monospace, monospace; padding: 20px; }}"
            f"h1 {{ font-size: 16px; color: #7ec9ff; }}"
            f"p {{ max-width: 900px; line-height: 1.5; color: #aaa; }}"
            f"</style></head><body>"
            f"<h1>{title}</h1>"
            f"<p>Running spectral dimension d_s(t) = -2·d log Z / d log t. "
            f"Solid line: in power-law window. Dashed: outside (saturation/"
            f"ballistic regimes — diagnostic only). Shaded: ±1σ from "
            f"jackknife over SLQ probe vectors. The 4→2 flow signature of "
            f"CDT and asymptotic safety would show as a curve sloping from "
            f"~4 at large t down toward ~2 at small t.</p>"
            f"{svg}</body></html>")
    with open(path, "w") as fh:
        fh.write(html)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def parse_cells(s):
    """Parse '8,0.005,0.94 9,0,0.95' → [(8, 0.005, 0.94), (9, 0, 0.95)]."""
    out = []
    for tok in s.split():
        parts = tok.split(",")
        if len(parts) != 3:
            raise SystemExit(f"bad --cells token {tok!r}; expected k,T,lb")
        out.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int)
    ap.add_argument("--T", type=float)
    ap.add_argument("--lb", type=float)
    ap.add_argument("--cells", type=str,
                    help='space-separated "k,T,lb" tuples for overlay mode')
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-probes", type=int, default=60,
                    help="more probes → tighter d_s(t) error band (default 60)")
    ap.add_argument("--lanczos-m", type=int, default=300,
                    help="Lanczos steps per probe (default 300)")
    ap.add_argument("--half-window", type=int, default=10,
                    help="local-slope half-window in t-grid steps (default 10 → 21-pt window)")
    ap.add_argument("--out-prefix", type=str, default="flow")
    args = ap.parse_args(argv)

    if args.cells:
        cells = parse_cells(args.cells)
    else:
        if args.k is None or args.T is None or args.lb is None:
            ap.error("must give either --cells or all of --k --T --lb")
        cells = [(args.k, args.T, args.lb)]

    mu_table = load_mu_table()
    if not mu_table:
        raise SystemExit(f"{MU_JSON} not found or empty — run the sweep's "
                         f"μ-calibration phase first.")

    results = []
    for (k, T, lb) in cells:
        res = run_flow_for_cell(
            k=k, T=T, lb=lb, N=args.N, seed=args.seed,
            n_probes=args.n_probes, lanczos_m=args.lanczos_m,
            mu_table=mu_table, half_window=args.half_window)
        # Per-cell CSV
        tag = f"k{k}_T{T}_lb{lb}_N{args.N}_s{args.seed}"
        csv_path = f"{args.out_prefix}_{tag}.csv"
        write_csv(res, csv_path)
        print(f"    → {csv_path}")
        quad_path = f"quad_{tag}.npz"
        write_quad_npz(res, quad_path)
        print(f"    → {quad_path}")
        # Per-cell HTML (single-cell view)
        if len(cells) == 1:
            html_path = f"{args.out_prefix}_{tag}.html"
            write_html([res], html_path,
                       title=f"d_s(t) flow — {res.label}")
            print(f"    → {html_path}")
        results.append(res)

    # Overlay HTML if multiple cells
    if len(cells) > 1:
        html_path = f"{args.out_prefix}_overlay.html"
        write_html(results, html_path,
                   title=f"d_s(t) flow — {len(cells)} cells, N={args.N}")
        print(f"    → {html_path}")


if __name__ == "__main__":
    main()
