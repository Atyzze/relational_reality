#!/usr/bin/env python3
"""isotropy.py — does a field spread as a uniform expanding sphere?

A perturbation in real space spreads isotropically: a wavefront/heat front is a
sphere, the same in every direction, with no preferred axis and no hidden
fast-path. This probe checks the emergent graphs for the same property.

It has no coordinates to rotate, so "isotropic" is tested the graph-native way:
diffuse a heat field exp(-tL)·e_s from a source node s, bucket every other node
by its graph distance r from s (BFS shells = the discrete "spheres"), and ask
whether the field is *uniform within each shell*. If it is, every direction at
radius r looks the same → isotropic. If some nodes in a shell get the field far
sooner than their neighbours, that is a hidden "highway" (a low-dimensional
fast subgraph) → anisotropy. Two companion checks:

  • ball growth  |B_r| ~ r^d  — the metric should look d-dimensional and smooth;
    a highway shows up as a kink or a too-fast volume.
  • homogeneity — repeating from many sources, the profiles should agree
    (the space looks the same everywhere), so we average over sources and
    report the spread.

Because *any* disordered graph has some shell-to-shell scatter from discreteness
and degree variation, the probe also runs on a matched reference torus (built by
the same code, isotropic by construction). The verdict is relative: is the
disordered graph **as uniform as a lattice of the same size**?

Run (from the project root):
    python src/physics_tests/isotropy.py --k 8 --T 0 --lb 0.99 --N 16000
    python src/physics_tests/isotropy.py --N 16000 --sources 16 --ref-dim 4

Best at moderate N (≤ ~64k) and a handful of sources — this is a structural
check on one graph, not a convergence study.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import expm_multiply

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)                       # …/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _adjacency_from_laplacian(L):
    """Edge-connectivity (sparsity) pattern from a combinatorial Laplacian
    L = D - A: take the negated off-diagonal. Weights are positive; for the
    unweighted BFS below only the pattern matters."""
    A = -sp.csr_matrix(L).copy()
    A.setdiag(0.0)
    A.eliminate_zeros()
    A.data = np.abs(A.data)
    return A


def isotropy_probe(L, n_sources=12, times=(5.0, 20.0, 80.0),
                   field_floor=1e-6, seed=0):
    """Heat-field isotropy on the graph with Laplacian L (scipy sparse).

    Returns a dict:
      shell_cv[t]   : array over r of the source-averaged within-shell
                      coefficient of variation of exp(-tL)e_s (lower = more
                      isotropic); NaN where the field is below `field_floor`.
      cv_summary[t] : median of shell_cv[t] over the trustworthy shells.
      ball_dim      : ball-growth dimension d from |B_r| ~ r^d (source-avg).
      r_max         : longest shell index seen.
    """
    L = sp.csr_matrix(L)
    N = L.shape[0]
    A = _adjacency_from_laplacian(L)
    rng = np.random.default_rng(seed)
    n_sources = int(min(n_sources, N))
    srcs = rng.choice(N, size=n_sources, replace=False)

    per_t_cv = {float(t): [] for t in times}          # list of per-shell CV arrays
    per_t_hi = {float(t): [] for t in times}          # per-shell max/median (highway) arrays
    ball_arrs = []                                     # per-source cumulative |B_r|
    ball_dims = []
    r_max = 0
    for s in srcs:
        dist = shortest_path(A, method="BF", unweighted=True, indices=s)
        dist = np.asarray(dist).ravel()
        finite = np.isfinite(dist)
        rr = dist[finite].astype(int)
        rmax = int(rr.max())
        r_max = max(r_max, rmax)
        # ball growth: |B_r| = # nodes within distance r
        counts = np.bincount(rr, minlength=rmax + 1)
        ball = np.cumsum(counts)
        ball_arrs.append(ball.astype(float))
        # fit log|B_r| ~ d log r over the bulk (skip r<2 and the saturating top)
        hi_fit = max(3, int(rmax * 0.7))
        rs = np.arange(2, hi_fit + 1)
        if len(rs) >= 2 and ball[hi_fit] > ball[2] > 0:
            d = np.polyfit(np.log(rs), np.log(ball[2:hi_fit + 1]), 1)[0]
            ball_dims.append(float(d))
        # heat field per t, within-shell CV + highway (max/median) ratio
        e = np.zeros(N)
        e[s] = 1.0
        for t in times:
            u = expm_multiply(-float(t) * L, e)
            u = np.asarray(u).ravel()[finite]
            umax = u.max() if u.size else 0.0
            cv = np.full(rmax + 1, np.nan)
            hir = np.full(rmax + 1, np.nan)
            for r in range(rmax + 1):
                sh = u[rr == r]
                if sh.size >= 2:
                    m = sh.mean()
                    if m > field_floor * max(umax, 1e-300):
                        cv[r] = sh.std(ddof=1) / m
                        med = np.median(sh)
                        if med > 0:
                            hir[r] = sh.max() / med
            per_t_cv[float(t)].append(cv)
            per_t_hi[float(t)].append(hir)

    # source-average each per-shell array (ragged → pad with NaN)
    def _avg_ragged(arrs):
        width = max((a.size for a in arrs), default=0)
        if not width:
            return np.array([])
        M = np.full((len(arrs), width), np.nan)
        for i, a in enumerate(arrs):
            M[i, :a.size] = a
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(M, axis=0)

    shell_cv, shell_hi, cv_summary = {}, {}, {}
    for t in times:
        mean_cv = _avg_ragged(per_t_cv[float(t)])
        shell_cv[float(t)] = mean_cv
        shell_hi[float(t)] = _avg_ragged(per_t_hi[float(t)])
        good = mean_cv[np.isfinite(mean_cv)]
        cv_summary[float(t)] = float(np.median(good)) if good.size else np.nan
    ball_mean = _avg_ragged(ball_arrs)
    return {
        "shell_cv": shell_cv,
        "shell_hi": shell_hi,
        "ball_mean": ball_mean,
        "cv_summary": cv_summary,
        "ball_dim": float(np.mean(ball_dims)) if ball_dims else float("nan"),
        "r_max": r_max,
        "n_sources": n_sources,
        "times": [float(t) for t in times],
    }


_CV_NOISE = 1e-3   # below this absolute within-shell CV the field is essentially
                   # uniform and the disordered/torus ratio is numerical noise


def default_probe_times(N, d_guess=4.0):
    """Diffusion times scaled to the graph SIZE, so the probe stays
    informative at every N. A heat front reaches graph distance ~√t and the
    graph's radius grows like N^(1/d), so a field still HAS within-shell
    structure only for t up to ~N^(2/d). Fixed times (the old 5/20/80
    default) are tuned for N≈16k: on much larger or better-mixing graphs
    they land past equilibration, where every shell is trivially uniform and
    CV≈0 reads as spuriously perfect isotropy (the bright band the
    expander corner of the old heatmap showed was exactly this artefact,
    not physics).

    The coefficients reproduce the old (5, 20, 80) at N=16000, d_guess=4
    (√16000 ≈ 126.5), so existing single-cell results stay comparable.
    """
    scale = float(N) ** (2.0 / d_guess)
    return tuple(round(c * scale, 4) for c in (0.0395, 0.1581, 0.6325))


def _informative_times(dis, tor, times, floor=_CV_NOISE):
    """Times where the comparison is trustworthy: the disordered field still has
    measurable within-shell structure (CV >= floor). At long diffusion times the
    field equilibrates to ~uniform on both graphs, so the CV ratio there blows up
    on two near-zero numbers and means nothing — those times are excluded from the
    verdict (but still reported)."""
    out = []
    for t in times:
        d = dis["cv_summary"].get(float(t), float("nan"))
        if np.isfinite(d) and d >= floor:
            out.append(float(t))
    return out


def _ratio_at(dis, tor, t):
    d = dis["cv_summary"].get(float(t), float("nan"))
    r = tor["cv_summary"].get(float(t), float("nan")) if tor else float("nan")
    return (d / r) if (tor and np.isfinite(r) and r > 0) else float("nan")


def _highway_diag(dis, t):
    """The 'why': in the shell with the worst uniformity at time t, how much faster
    did the fastest node get the field than the shell's median node? A large
    max/median is a low-dimensional fast-path ('highway')."""
    hi = dis["shell_hi"].get(float(t), np.array([]))
    cv = dis["shell_cv"].get(float(t), np.array([]))
    if not cv.size:
        return None
    finite = np.where(np.isfinite(cv))[0]
    if not finite.size:
        return None
    worst_r = int(finite[np.argmax(cv[finite])])
    return dict(shell=worst_r, cv=float(cv[worst_r]),
                max_over_median=(float(hi[worst_r]) if worst_r < hi.size and np.isfinite(hi[worst_r]) else float("nan")))


def _dump_data(dis, tor, times, out_path):
    """Write the full per-shell curves (CSV) and a summary (JSON) next to the PNG,
    so the diagnosis is fully available even when matplotlib is not installed."""
    import csv
    base = os.path.splitext(out_path)[0]
    shells_csv = base + "_shells.csv"
    with open(shells_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heat_t", "shell_r", "disordered_cv", "torus_cv",
                    "disordered_max_over_median"])
        for t in times:
            dcv = dis["shell_cv"].get(float(t), np.array([]))
            tcv = tor["shell_cv"].get(float(t), np.array([])) if tor else np.array([])
            dhi = dis["shell_hi"].get(float(t), np.array([]))
            for r in range(dcv.size):
                w.writerow([f"{t:g}", r,
                            "" if not np.isfinite(dcv[r]) else f"{dcv[r]:.6g}",
                            "" if (r >= tcv.size or not np.isfinite(tcv[r])) else f"{tcv[r]:.6g}",
                            "" if (r >= dhi.size or not np.isfinite(dhi[r])) else f"{dhi[r]:.6g}"])
    summary_json = base + "_summary.json"
    info = _informative_times(dis, tor, times)
    per_t = []
    for t in times:
        per_t.append(dict(t=float(t),
                          disordered_cv=dis["cv_summary"].get(float(t), None),
                          torus_cv=(tor["cv_summary"].get(float(t), None) if tor else None),
                          ratio=(None if not np.isfinite(_ratio_at(dis, tor, t)) else _ratio_at(dis, tor, t)),
                          informative=(float(t) in info)))
    with open(summary_json, "w") as f:
        json.dump(dict(per_time=per_t, informative_times=info,
                       ball_dim_disordered=dis.get("ball_dim"),
                       ball_dim_torus=(tor.get("ball_dim") if tor else None),
                       n_sources=dis.get("n_sources")), f, indent=2, default=str)
    return shells_csv, summary_json


def _plot(disordered, torus, times, out_path):
    """Multi-panel diagnostic: (top) within-shell CV vs r per time; (bottom) ball
    growth |B_r| log-log and the highway band (per-shell max/median of the field)
    for the most informative time. Saved as a PNG; skipped if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (plot skipped — install matplotlib to get PNGs: {e})")
        return None
    info = _informative_times(disordered, torus, times)
    focus = (max(info) if info else (times[len(times) // 2] if times else None))
    ncol = max(len(times), 2)
    fig, ax = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8), squeeze=False)

    for j, t in enumerate(times):
        a = ax[0][j]
        dcv = disordered["shell_cv"].get(float(t), np.array([]))
        tcv = torus["shell_cv"].get(float(t), np.array([])) if torus else None
        a.plot(np.arange(dcv.size), dcv, "-o", ms=3, color="#d1495b", label="disordered")
        if tcv is not None and tcv.size:
            a.plot(np.arange(tcv.size), tcv, "-o", ms=3, color="#3b7dd8", label="torus ref")
        info_tag = "  ✓ used in verdict" if float(t) in info else "  ✗ field equilibrated"
        a.set_title(f"heat time t = {t:g}{info_tag}", fontsize=9)
        a.set_xlabel("graph distance r (shell)"); a.set_ylabel("within-shell CV (0 = uniform)")
        a.grid(alpha=.3); a.legend(fontsize=8)
    for j in range(len(times), ncol):
        ax[0][j].axis("off")

    # bottom-left: ball growth (log-log)
    a = ax[1][0]
    bd = disordered.get("ball_mean", np.array([]))
    if bd.size:
        rr = np.arange(1, bd.size)
        a.loglog(rr, bd[1:], "-o", ms=3, color="#d1495b",
                 label=f"disordered  (d≈{disordered.get('ball_dim', float('nan')):.2f})")
    if torus:
        bt = torus.get("ball_mean", np.array([]))
        if bt.size:
            rt = np.arange(1, bt.size)
            a.loglog(rt, bt[1:], "-o", ms=3, color="#3b7dd8",
                     label=f"torus ref  (d≈{torus.get('ball_dim', float('nan')):.2f})")
    a.set_title("ball growth |B_r| ~ r^d  (slope = dimension)", fontsize=9)
    a.set_xlabel("graph distance r"); a.set_ylabel("|B_r| (nodes within r)")
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

    # bottom-second: highway band at the focus time
    if ncol >= 2:
        a = ax[1][1]
        if focus is not None:
            dhi = disordered["shell_hi"].get(float(focus), np.array([]))
            a.plot(np.arange(dhi.size), dhi, "-o", ms=3, color="#d1495b", label="disordered")
            if torus:
                thi = torus["shell_hi"].get(float(focus), np.array([]))
                if thi.size:
                    a.plot(np.arange(thi.size), thi, "-o", ms=3, color="#3b7dd8", label="torus ref")
            a.axhline(1.0, color="#888", lw=1, ls="--")
            a.set_title(f"fast-path ratio (max/median in shell), t={focus:g}\n"
                        f"high = a 'highway' reaches some nodes early", fontsize=9)
            a.set_xlabel("graph distance r (shell)"); a.set_ylabel("max / median field in shell")
            a.grid(alpha=.3); a.legend(fontsize=8)
        for j in range(2, ncol):
            ax[1][j].axis("off")

    fig.suptitle("Field isotropy: does a perturbation spread as a uniform sphere? "
                 "(lower CV & smooth ball growth = more isotropic)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _calibrate_if_needed(k, T, lb, ec, mu_table, save=True):
    """Ensure a μ entry exists for (k,T,lb); calibrate once and (optionally)
    persist to the μ-table JSON so future runs and the sweep reuse it."""
    from core.disk_io import mu_key, save_mu_table
    from core.graph_builder import calibrate_mu
    key = mu_key(k, T, lb)
    if key in mu_table:
        return mu_table[key]
    print(f"[isotropy] {key} not calibrated — calibrating once …", flush=True)
    mu, _err, _k = calibrate_mu(k, T, lb, ec, verbose=False)
    mu_table[key] = mu
    if save:
        try:
            save_mu_table(mu_table)
        except Exception:
            pass
    return mu


def _probe_cell(k, T, lb, N, seed, sources, times, mu_table):
    """Build (reusing the graph cache) and probe one disordered cell."""
    from core.cell_tests import build_cell
    print(f"[isotropy] building cell k={k} T={T:g} lb={lb:g} N={N:,} …", flush=True)
    cell = build_cell(k, T, lb, N, seed, mu_table)
    return isotropy_probe(cell.L_csr, n_sources=sources, times=tuple(times), seed=seed)


def _probe_torus(ref_dim, N, seed, sources, times):
    from core.cell_tests import build_torus_cell
    L = max(2, round(N ** (1.0 / ref_dim)))
    print(f"[isotropy] building reference {ref_dim}D torus L={L} "
          f"(N={L ** ref_dim:,}) …", flush=True)
    tcell = build_torus_cell(ref_dim, L, seed)
    return isotropy_probe(tcell.L_csr, n_sources=sources, times=tuple(times), seed=seed)


def _verdict_text(worst):
    if worst <= 1.5:
        return ("isotropic — the field is as uniform across shells as the reference "
                "lattice; no hidden highway.")
    if worst <= 3.0:
        return ("mildly anisotropic — shells are noticeably less uniform than the "
                "lattice; worth a closer look.")
    return ("ANISOTROPIC — the field is far less uniform than the lattice; a "
            "preferred direction / fast-path is likely.")


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--lb", type=float, default=0.99)
    ap.add_argument("--N", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources", type=int, default=12,
                    help="random source nodes to average over")
    ap.add_argument("--times", type=float, nargs="+", default=None,
                    help="heat-diffusion times to probe (default: scaled to "
                         "N via default_probe_times — ≈5/20/80 at N=16000)")
    ap.add_argument("--ref-dim", type=int, default=4,
                    help="dimension of the reference torus (0 = skip it)")
    ap.add_argument("--ec", type=float, default=-1.0)
    ap.add_argument("--out", default="isotropy.png")
    # sweep mode: vary k / lb (and optionally N) and draw a uniformity heatmap
    ap.add_argument("--sweep", action="store_true",
                    help="sweep over k/lb (and N) and emit a uniformity heatmap + CSV")
    ap.add_argument("--sweep-k", default=None,
                    help="comma-separated k values for --sweep (default: project grid)")
    ap.add_argument("--sweep-lb", default=None,
                    help="comma-separated lb values for --sweep (default: project grid)")
    ap.add_argument("--sweep-N", default=None,
                    help="comma-separated N values for --sweep (default: just --N)")
    args = ap.parse_args(argv)
    args._times_explicit = args.times is not None
    if args.times is None:
        # Scale to the (single-cell) N; the sweep path re-derives per N rung.
        args.times = list(default_probe_times(args.N))

    from core.disk_io import load_mu_table
    mu_table = load_mu_table()

    if args.sweep:
        return sweep(args, mu_table)

    _calibrate_if_needed(args.k, args.T, args.lb, args.ec, mu_table)
    dis = _probe_cell(args.k, args.T, args.lb, args.N, args.seed,
                      args.sources, args.times, mu_table)
    tor = _probe_torus(args.ref_dim, args.N, args.seed, args.sources, args.times) \
        if args.ref_dim > 0 else None

    info = _informative_times(dis, tor, args.times)
    print("\n[isotropy] within-shell CV (median over bulk shells; lower = more isotropic):")
    print(f"  {'heat t':>8}  {'disordered':>11}  {'torus ref':>10}  {'ratio':>8}  used?")
    print("  " + "-" * 52)
    worst = 0.0
    for t in args.times:
        dcv = dis["cv_summary"].get(float(t), float("nan"))
        tcv = tor["cv_summary"].get(float(t), float("nan")) if tor else float("nan")
        ratio = _ratio_at(dis, tor, t)
        used = float(t) in info
        if used and np.isfinite(ratio):
            worst = max(worst, ratio)
        mark = "  ✓" if used else "  ✗ (field flat)"
        print(f"  {t:>8g}  {dcv:>11.3f}  {tcv:>10.3f}  "
              f"{(f'{ratio:.2f}x' if np.isfinite(ratio) else '—'):>8}{mark}")
    print("  " + "-" * 52)
    print(f"  ball-growth dimension d (|B_r|~r^d): disordered {dis['ball_dim']:.2f}"
          + (f", torus {tor['ball_dim']:.2f}" if tor else ""))

    # the "why": point at the worst shell and the fast-path ratio there
    if info:
        focus = max(info)
        hw = _highway_diag(dis, focus)
        if hw:
            print(f"  worst shell at t={focus:g}: r={hw['shell']}, CV={hw['cv']:.3f}, "
                  f"fastest node got the field {hw['max_over_median']:.1f}x the shell "
                  f"median  → that gap is the 'highway'.")

    if tor:
        if not info:
            print("\n[isotropy] verdict: inconclusive — the field equilibrated to "
                  "near-uniform at every probed time (ratios are noise). Try smaller "
                  "--times or a larger --N.")
        else:
            print(f"\n[isotropy] verdict (over informative times {', '.join(f'{t:g}' for t in info)}): "
                  f"{_verdict_text(worst)}")
            print(f"           worst trustworthy ratio = {worst:.2f}x  "
                  f"(absolute disordered CV stays small: {max(dis['cv_summary'].get(t, 0) for t in info):.3f}).")

    shells_csv, summary_json = _dump_data(dis, tor, args.times, args.out)
    print(f"[isotropy] data: {os.path.basename(shells_csv)}, {os.path.basename(summary_json)}")
    out = _plot(dis, tor, args.times, args.out)
    if out:
        print(f"[isotropy] wrote {out}")
    return dis, tor


def _grid_default(name, fallback):
    """Pull a default sweep axis from the project grid if available."""
    try:
        from core import project_constants as cfg
        return list(getattr(cfg, name))
    except Exception:
        return fallback


def sweep(args, mu_table):
    """Vary k / lb (and optionally N), compute the uniformity metric for each cell
    (reusing cached graphs), write a CSV, and render a heatmap per N. The metric is
    the worst trustworthy disordered/torus within-shell-CV ratio (1 = lattice-like;
    larger = more anisotropic)."""
    import csv

    def _parse(s, default):
        if s is None:
            return default
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                out.append(int(float(tok)) if float(tok).is_integer() else float(tok))
        return out

    ks = _parse(args.sweep_k, _grid_default("K_ALL", [4, 6, 8, 10]))
    lbs = _parse(args.sweep_lb, _grid_default("LB_ALL", [0.9, 0.95, 0.99]))
    Ns = _parse(args.sweep_N, [args.N])
    print(f"[isotropy] sweep: k={ks} lb={lbs} N={Ns} T={args.T:g} "
          f"ref-dim={args.ref_dim} sources={args.sources}")

    rows = []                                   # dicts per cell
    grids = {}                                  # N -> (matrix[k,lb], ks, lbs)
    torus_cache = {}
    for N in Ns:
        # Unless --times was given explicitly, scale the probe times to THIS
        # N rung — fixed times tuned for one N land past equilibration on a
        # larger one (see default_probe_times).
        times_n = (args.times if getattr(args, "_times_explicit", True)
                   else list(default_probe_times(N)))
        tor = None
        if args.ref_dim > 0:
            L = max(2, round(N ** (1.0 / args.ref_dim)))
            if (args.ref_dim, L) not in torus_cache:
                torus_cache[(args.ref_dim, L)] = _probe_torus(
                    args.ref_dim, N, args.seed, args.sources, times_n)
            tor = torus_cache[(args.ref_dim, L)]
        M = np.full((len(ks), len(lbs)), np.nan)
        for i, k in enumerate(ks):
            for j, lb in enumerate(lbs):
                try:
                    _calibrate_if_needed(k, args.T, lb, args.ec, mu_table)
                    dis = _probe_cell(k, args.T, lb, N, args.seed,
                                      args.sources, times_n, mu_table)
                except Exception as ex:
                    print(f"  [skip] k={k} lb={lb} N={N}: {ex}")
                    rows.append(dict(k=k, T=args.T, lb=lb, N=N, metric=None,
                                     ball_dim_dis=None, error=str(ex)[:120]))
                    continue
                info = _informative_times(dis, tor, times_n)
                metric = max((_ratio_at(dis, tor, t) for t in info
                              if np.isfinite(_ratio_at(dis, tor, t))), default=float("nan"))
                M[i, j] = metric
                rows.append(dict(k=k, T=args.T, lb=lb, N=N,
                                 metric=(None if not np.isfinite(metric) else round(metric, 4)),
                                 ball_dim_dis=round(dis["ball_dim"], 3),
                                 ball_dim_tor=(round(tor["ball_dim"], 3) if tor else None),
                                 abs_cv=round(max((dis["cv_summary"].get(t, 0) for t in info), default=0.0), 5),
                                 informative=";".join(f"{t:g}" for t in info)))
                print(f"  k={k:>3} lb={lb:<5g} N={N:>8,}: "
                      f"uniformity ratio={'—' if not np.isfinite(metric) else f'{metric:.2f}x'}  "
                      f"d_dis={dis['ball_dim']:.2f}")
        grids[N] = (M, list(ks), list(lbs))

    base = os.path.splitext(args.out)[0] + "_sweep"
    csv_path = base + ".csv"
    cols = ["k", "T", "lb", "N", "metric", "ball_dim_dis", "ball_dim_tor",
            "abs_cv", "informative", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\n[isotropy] sweep data → {csv_path}")
    png = _heatmap(grids, args.T, args.ref_dim, base + ".png")
    if png:
        print(f"[isotropy] sweep heatmap → {png}")
    return rows


def _heatmap(grids, T, ref_dim, out_path):
    """Heatmap of the uniformity metric over k × lb, one panel per N. Skipped
    (CSV still written) if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception as e:
        print(f"  (heatmap skipped — install matplotlib: {e})")
        return None
    Ns = list(grids)
    fig, ax = plt.subplots(1, len(Ns), figsize=(4.8 * len(Ns), 4.4), squeeze=False)
    finite = np.array([v for (M, _k, _l) in grids.values() for v in M.ravel() if np.isfinite(v)])
    vmin = max(1.0, float(finite.min())) if finite.size else 1.0
    vmax = max(vmin * 1.01, float(finite.max())) if finite.size else 3.0
    for c, N in enumerate(Ns):
        M, ks, lbs = grids[N]
        a = ax[0][c]
        im = a.imshow(M, origin="lower", aspect="auto", cmap="magma_r",
                      norm=LogNorm(vmin=vmin, vmax=vmax))
        a.set_xticks(range(len(lbs))); a.set_xticklabels([f"{x:g}" for x in lbs])
        a.set_yticks(range(len(ks))); a.set_yticklabels([f"{x:g}" for x in ks])
        a.set_xlabel("locality bias  ℓb"); a.set_ylabel("target degree  k")
        a.set_title(f"N = {N:,}")
        for i in range(len(ks)):
            for j in range(len(lbs)):
                if np.isfinite(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                           fontsize=8, color="#222" if M[i, j] < (vmin * vmax) ** 0.5 else "#eee")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="anisotropy ratio (1 = lattice-like)")
    fig.suptitle(f"Field uniformity across the grid (T={T:g}, vs {ref_dim}D torus) — "
                 f"lower = more isotropic", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def cell_isotropy_summary(L, n_sources=6, times=None, seed=0):
    """Compact, cheap isotropy summary for ONE already-built graph (Laplacian
    L), for storing in a cell's meta sidecar and reuse by the auto-refreshed
    isotropy heatmap. Reuses isotropy_probe with a SMALL source count (the
    standalone deep-dive uses more) and the graph's own Laplacian, so nothing
    is re-evolved. Returns:

        {cv: {t: within-shell CV}, ball_dim, aniso, aniso_t, n_sources,
         equilibrated}

    `times` defaults to default_probe_times(N) — scaled to the graph size so
    the probe doesn't land past equilibration on large or fast-mixing graphs.

    `aniso` is the within-shell CV at the most-informative diffusion time —
    the largest t whose field still has measurable structure (CV >= _CV_NOISE),
    falling back to the smallest t if everything has equilibrated. It is a
    self-contained "how non-uniform is the spread" scalar (lower = more
    isotropic) that needs no reference graph, so it sidesteps the divide-by-
    near-zero blow-up the disordered/torus ratio suffers at long times.

    `equilibrated` is True when NO probe time had measurable structure — the
    field was uniform everywhere at every t, so `aniso` is a noise floor, not
    a measurement. The heatmap renders such cells as an explicit 'eq' glyph
    instead of colouring a meaningless near-zero as "perfectly isotropic".
    """
    N = sp.csr_matrix(L).shape[0]
    if times is None:
        times = default_probe_times(N)
    pr = isotropy_probe(L, n_sources=n_sources, times=times, seed=seed)
    cv = {float(t): (float(v) if (v is not None and np.isfinite(v)) else None)
          for t, v in pr["cv_summary"].items()}
    informative = [t for t, v in cv.items() if v is not None and v >= _CV_NOISE]
    at = max(informative) if informative else (min(cv) if cv else float("nan"))
    bd = pr.get("ball_dim", float("nan"))
    return {
        "cv": cv,
        "ball_dim": (float(bd) if np.isfinite(bd) else None),
        "aniso": (cv.get(at) if cv else None),
        "aniso_t": (float(at) if at == at else None),   # NaN-safe
        "n_sources": int(pr.get("n_sources", n_sources)),
        "equilibrated": (not informative),
    }


def render_isotropy_heatmap(flow_dir=".", out_path="isotropy_heatmap.png"):
    """Auto-refreshed companion to the shape/flow heatmaps: a k × ℓb map of
    within-shell field anisotropy (lower = more isotropic), one panel per
    (T, N). Reads the per-cell isotropy summary the sweep stores in each
    meta_*.json sidecar — NO graph is re-evolved or re-probed here. Best-effort:
    returns None (writes nothing) if matplotlib is absent or no cell carries an
    isotropy summary yet.
    """
    import glob
    from collections import defaultdict
    metas = []
    metas_torus = []
    for pat in (os.path.join(flow_dir, "flow", "meta_*.json"),
                os.path.join(flow_dir, "meta_*.json")):
        for p in glob.glob(pat):
            try:
                with open(p) as fh:
                    m = json.load(fh)
            except Exception:
                continue
            iso = m.get("isotropy")
            if not (isinstance(iso, dict) and iso.get("aniso") is not None):
                continue
            if m.get("kind") == "cell":
                metas.append(m)
            elif m.get("kind") == "torus":
                metas_torus.append(m)
    if not metas:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception:
        return None

    panels = defaultdict(dict)            # (T, N) -> {(k, lb): value}
    eq_cells = defaultdict(set)           # (T, N) -> {(k, lb)} equilibrated
    # Reference: the sweep's own 4D-torus cells also carry isotropy summaries
    # (their metas have kind="torus"). A matched torus is isotropic BY
    # CONSTRUCTION, so its within-shell CV at the same size is the finite-size
    # discreteness floor — the meaningful per-cell number is the RATIO to it
    # (≈1 = lattice-grade isotropy; an absolute CV alone has no zero point,
    # since ANY disordered graph carries some CV from degree fluctuations).
    torus_ref = {}                        # N -> [aniso, ...]
    for m in metas_torus:
        try:
            iso = m["isotropy"]
            a = float(iso["aniso"])
            if iso.get("equilibrated") or a < _CV_NOISE:
                continue
            if int(m.get("d", 4)) != 4:
                continue
            torus_ref.setdefault(int(m["N"]), []).append(a)
        except (KeyError, TypeError, ValueError):
            continue
    torus_ref = {N: float(np.median(v)) for N, v in torus_ref.items()}

    def _ref_for(N):
        """Nearest-N 4D-torus CV, accepted within a factor of 4 in N
        (torus N = L^4 never matches a cell N exactly)."""
        if not torus_ref:
            return None
        best = min(torus_ref, key=lambda n: abs(math.log(n / N)))
        return torus_ref[best] if abs(math.log(best / N)) <= math.log(4) \
            else None

    use_ratio = bool(torus_ref)
    for m in metas:
        try:
            T_, N_ = float(m["T"]), int(m["N"])
            klb = (int(m["k"]), float(m["lb"]))
            iso = m["isotropy"]
            a = float(iso["aniso"])
            # 'equilibrated' = the field was uniform at every probe time, so
            # `aniso` is a noise floor, not a measurement. Old sidecars
            # predate the flag; for those, an aniso below the noise floor is
            # the same artefact, so mask it the same way.
            if iso.get("equilibrated") or a < _CV_NOISE:
                eq_cells[(T_, N_)].add(klb)
                panels[(T_, N_)].setdefault(klb, np.nan)
            else:
                ref = _ref_for(N_) if use_ratio else None
                panels[(T_, N_)][klb] = (a / ref) if ref else a
        except (KeyError, TypeError, ValueError):
            continue
    keys = sorted(panels)
    finite = [v for d in panels.values() for v in d.values()
              if np.isfinite(v) and v > 0]
    if not keys or not finite:
        return None
    vmin = max(min(finite), 1e-4)
    vmax = max(max(finite), vmin * 1.01)

    # Grid layout matching the shape heatmaps: one ROW per temperature T,
    # one COLUMN per N, shared (k × ℓ) axes (the union across all panels)
    # and ONE shared colour scale + colorbar — so panels are directly
    # comparable and the figure has a sane aspect ratio. (The old layout
    # put every (T, N) combination side-by-side in a single 1×n row, which
    # for a 5T × 4N run produced a ~9000-px-wide strip.)
    Ts = sorted({T for (T, _N) in keys})
    Ns = sorted({N for (_T, N) in keys})
    all_ks = sorted({k for d in panels.values() for (k, _l) in d})
    all_lbs = sorted({l for d in panels.values() for (_k, l) in d})
    n_rows, n_cols = len(Ts), len(Ns)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.55 * n_cols + 1.6, 2.35 * n_rows + 1.2),
        squeeze=False)
    # Annotate per-cell values only while panels are still readable.
    annotate = (len(all_ks) * len(all_lbs)) <= 200 and n_cols <= 6
    im = None
    for ri, T in enumerate(Ts):
        for ci, N in enumerate(Ns):
            a = axes[ri][ci]
            d = panels.get((T, N))
            if not d:
                a.set_facecolor("#f0f0f0")
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_color("#cccccc")
                continue
            M = np.full((len(all_ks), len(all_lbs)), np.nan)
            for (k, lb), v in d.items():
                M[all_ks.index(k), all_lbs.index(lb)] = v
            im = a.imshow(M, origin="lower", aspect="auto", cmap="magma_r",
                          norm=LogNorm(vmin=vmin, vmax=vmax))
            # Equilibrated cells: the field was uniform at every probe time,
            # so there is nothing to colour — mark explicitly instead of
            # rendering a meaningless near-zero as "perfectly isotropic".
            for (k, lb) in eq_cells.get((T, N), ()):
                a.text(all_lbs.index(lb), all_ks.index(k), "eq",
                       ha="center", va="center", fontsize=6, color="#999")
            a.set_xticks(range(len(all_lbs)))
            a.set_yticks(range(len(all_ks)))
            if ri == n_rows - 1:
                a.set_xticklabels([f"{x:g}" for x in all_lbs],
                                  rotation=40, fontsize=7)
                a.set_xlabel("ℓb", fontsize=8)
            else:
                a.set_xticklabels([])
            if ci == 0:
                a.set_yticklabels([f"{x:g}" for x in all_ks], fontsize=7)
                a.set_ylabel(f"T = {T:g}\nk", fontsize=8)
            else:
                a.set_yticklabels([])
            if ri == 0:
                a.set_title(f"N = {N:,}", fontsize=9)
            if annotate:
                for i in range(len(all_ks)):
                    for j in range(len(all_lbs)):
                        if np.isfinite(M[i, j]):
                            a.text(j, i, f"{M[i, j]:.2f}", ha="center",
                                   va="center", fontsize=6,
                                   color="#222" if M[i, j] < (vmin * vmax) ** 0.5
                                   else "#eee")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("CV ratio vs matched 4D torus (log scale; ≈1 = "
                       "lattice-grade)" if use_ratio
                       else "within-shell CV (log scale)")
    if use_ratio:
        head = ("Field isotropy across the grid — within-shell CV RATIO vs "
                "the matched 4D torus\n"
                "≈1 = as uniform as a lattice of the same size (the meaningful "
                "target; a disordered graph cannot reach 0) · ≲2–3 OK · ≫1 = "
                "structural anisotropy / highway")
    else:
        head = ("Field isotropy across the grid (raw within-shell CV; lower = "
                "more isotropic)\n"
                "no torus references found — absolute CV has no zero point; "
                "run the sweep with torus refs for the ratio view")
    fig.suptitle(head + " · rows = T, columns = N · 'eq' = field "
                 "equilibrated at every probe time (no signal)",
                 fontsize=10)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run()
