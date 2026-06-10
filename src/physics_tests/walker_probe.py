#!/usr/bin/env python3
"""
walker_probe — Monte-Carlo random walkers vs the SLQ heat kernel
================================================================
The spectral dimension this project measures IS a random-walker return
probability: Z(t) = (1/N)·Tr e^{−tL} with the combinatorial Laplacian
L = D − A is exactly the average probability that a continuous-time random
walker — every edge fires at rate 1, so a node of degree d jumps at rate d —
is back at its start after time t, and d_s(t) = −2·dln Z/dln t. The SLQ
probe computes that quantity SPECTRALLY (stochastic Lanczos quadrature);
this tool computes the very same quantity by ACTUALLY SIMULATING walkers,
giving an independent estimator with completely different systematics:

  • uniformisation: every walker ticks at the global rate Λ = max degree;
    at a tick, a walker at node i jumps to a uniform random neighbour with
    probability deg(i)/Λ, else stays. The discrete tick-chain occupancies
    g_m = P[at start after m ticks], Poisson-mixed with weights
    Pois(m; Λt), reproduce e^{−tL} EXACTLY — no time-discretisation error,
    the only error is Monte-Carlo.
  • d_s(t) is then extracted with the SAME local-slope estimator the SLQ
    pipeline uses (flow_probe.local_slope), so any disagreement is about
    the Z(t) estimate, not the differentiation.

Agreement validates the SLQ implementation, the Laplacian construction and
the windowing end to end; disagreement localises a bug. As a MEASUREMENT
instrument the walker is strictly worse — its noise at fixed CPU grows like
1/√(returns) and returns die as t^{−d_s/2}, which is exactly why the sweep
uses SLQ — so this stays an on-demand cross-check, never a sweep stage.

Usage (after the cell exists in output/):
    python main.py walkers k8 T0.004 lb0.99 N16000 s42
    python main.py walkers k8 T0.004 lb0.99 N16000 s42 --walkers 400000 --tmax 8
Writes output/walker_check_<tag>.png and prints the agreement verdict.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SRC, os.path.dirname(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.graph_builder import build_graph                    # noqa: E402
from core.flow_probe import local_slope                       # noqa: E402
from core.project_constants import EC                         # noqa: E402
from metrics.numba_kernels import _extract_lcc_nodes          # noqa: E402

_TOK = re.compile(r"^(k|T|lb|N|s)([-\d.eE]+)$")


def parse_cell(tokens):
    out = {}
    for t in tokens:
        m = _TOK.match(t)
        if not m:
            raise SystemExit(f"unrecognised cell token {t!r} "
                             "(expected like: k8 T0.004 lb0.99 N16000 s42)")
        key, val = m.groups()
        out[key] = int(val) if key in ("k", "N", "s") else float(val)
    missing = {"k", "T", "lb", "N", "s"} - set(out)
    if missing:
        raise SystemExit(f"missing cell tokens: {sorted(missing)}")
    return out["k"], out["T"], out["lb"], out["N"], out["s"]


_FNAME = re.compile(r"^(flow|meta)_k(\d+)_T([-\d.eE]+)_lb([-\d.eE]+)"
                    r"_N(\d+)_s(\d+)\.(csv|json)$")


def _find_cell_files(outdir, k, T, lb, N, seed):
    """Locate the cell's flow CSV + meta sidecar by PARSING candidate
    filenames (float-tolerant: T0, T0.0 and T0.000 all match T=0)."""
    tag = f"k{k}_T{T:g}_lb{lb:g}_N{N}_s{seed}"
    flow = meta = None
    for d in (os.path.join(outdir, "flow"), outdir):
        for p in glob.glob(os.path.join(d, "*_k*_T*_lb*_N*_s*.*")):
            m = _FNAME.match(os.path.basename(p))
            if not m:
                continue
            kind, kk, tt, ll, nn, ss, _ext = m.groups()
            if (int(kk) == k and int(nn) == N and int(ss) == seed
                    and abs(float(tt) - T) < 1e-12
                    and abs(float(ll) - lb) < 1e-12):
                if kind == "flow" and flow is None:
                    flow = p
                elif kind == "meta" and meta is None:
                    meta = p
        if flow and meta:
            break
    return tag, flow, meta


def _load_slq_curve(path):
    """Read (t, Z, Z_se, d_s, in_window) from a sweep flow CSV — parsed as
    one row unit so a malformed row can never desynchronise the arrays."""
    t, z, zse, d, inw = [], [], [], [], []
    import csv as _csv
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            try:
                row = (float(r["t"]), float(r["Z_mean"]),
                       float(r.get("Z_se", 0) or 0), float(r["d_s_mean"]),
                       r.get("in_window", "1") in ("1", "True", "true"))
            except (KeyError, TypeError, ValueError):
                continue
            t.append(row[0]); z.append(row[1]); zse.append(row[2])
            d.append(row[3]); inw.append(row[4])
    return (np.array(t), np.array(z), np.array(zse), np.array(d),
            np.array(inw, dtype=bool))


def simulate_return_probability(neighbors, degrees, starts, t_grid,
                                n_walkers, rng):
    """Uniformised CTRW: returns (p_hat[t], hit matrix g (walkers×ticks),
    Lambda, M). Exact in time via Poisson mixing; MC error only."""
    Lam = float(degrees.max())
    t_max = float(t_grid.max())
    M = int(Lam * t_max + 8.0 * math.sqrt(max(Lam * t_max, 1.0)) + 10)
    W = int(n_walkers)
    start = starts[rng.integers(0, len(starts), size=W)].astype(np.int64)
    pos = start.copy()
    hits = np.empty((M + 1, W), dtype=np.uint8)
    hits[0] = 1                                     # at start at tick 0
    deg = degrees.astype(np.float64)
    for m in range(1, M + 1):
        r = rng.random(W)
        jump = r < deg[pos] / Lam
        if jump.any():
            j = np.flatnonzero(jump)
            pj = pos[j]
            idx = (rng.random(j.size) * deg[pj]).astype(np.int64)
            pos[j] = neighbors[pj, idx]
        hits[m] = (pos == start)
    g = hits.mean(axis=1)                            # ĝ_m
    # Poisson mixing: p(t) = Σ_m Pois(m; Λt) ĝ_m, in log space for stability
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, M + 1)))))
    p = np.empty_like(t_grid, dtype=np.float64)
    ms = np.arange(M + 1, dtype=np.float64)
    for i, t in enumerate(t_grid):
        lt = Lam * t
        logw = -lt + ms * (math.log(lt) if lt > 0 else -np.inf) - log_fact
        w = np.exp(logw - logw.max())
        w /= w.sum()
        p[i] = float(w @ g)
    return p, hits, Lam, M


def walker_ds_curve(p, t_grid, half_window=8):
    log_t = np.log(t_grid)
    log_p = np.log(np.clip(p, 1e-300, None))
    slope = local_slope(log_t, log_p, np.ones_like(log_t),
                        half_window=half_window)
    return -2.0 * slope


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="main.py walkers",
        description="Monte-Carlo walker cross-check of the SLQ d_s(t) curve "
                    "for one measured cell.")
    ap.add_argument("cell", nargs=5,
                    help="cell tokens, e.g.: k8 T0.004 lb0.99 N16000 s42")
    ap.add_argument("--dir", default="output")
    ap.add_argument("--walkers", type=int, default=200_000)
    ap.add_argument("--tmax", type=float, default=6.0,
                    help="largest diffusion time to simulate (walker noise "
                         "explodes in the IR — that is WHY the sweep uses "
                         "SLQ; default 6.0 covers UV through the dip)")
    ap.add_argument("--points", type=int, default=80)
    ap.add_argument("--min-returns", type=float, default=50.0,
                    help="trust walker points only where p̂·walkers ≥ this")
    args = ap.parse_args(argv)

    k, T, lb, N, seed = parse_cell(args.cell)
    tag, flow_csv, meta_p = _find_cell_files(args.dir, k, T, lb, N, seed)
    if not (flow_csv and meta_p):
        print(f"cell {tag} not found under {args.dir}/flow — run the sweep "
              f"for that cell first (the cross-check compares against its "
              f"measured SLQ curve).", file=sys.stderr)
        return 1
    with open(meta_p) as fh:
        meta = json.load(fh)
    mu = meta.get("mu")
    if mu is None:
        print(f"{meta_p} records no μ; cannot rebuild the identical graph.",
              file=sys.stderr)
        return 1

    print(f"[walkers] rebuilding {tag} (μ={mu:g} from meta, same seed ⇒ "
          f"bit-identical graph) …")
    eng, _stats, _sw, _pk = build_graph(N, k, T, lb, mu, EC, seed,
                                        use_store=True, verify_equil=False)
    nbrs = np.ascontiguousarray(eng.node_neighbors)
    degs = np.ascontiguousarray(eng.node_degrees[:N]).astype(np.int32)
    lcc = _extract_lcc_nodes(nbrs, degs, N)
    print(f"[walkers] LCC {len(lcc)}/{N} nodes; Λ = max degree = "
          f"{int(degs.max())}; simulating {args.walkers:,} walkers "
          f"(uniformised CTRW, exact via Poisson mixing) …")

    rng = np.random.default_rng(seed * 7919 + 11)
    t_grid = np.geomspace(2.0 / max(float(degs.max()), 1.0), args.tmax,
                          args.points)
    p, hits, Lam, M = simulate_return_probability(
        nbrs, degs, np.asarray(lcc), t_grid, args.walkers, rng)
    reliable = p * args.walkers >= args.min_returns

    t_s, Z_s, Zse_s, d_s, inw = _load_slq_curve(flow_csv)
    # The VERDICT compares the estimator-identical object: Z(t) itself.
    # (d_s curves smooth over different log-t window widths, so comparing
    # them mixes a differentiation choice into an estimator check.)
    N_eff = float(len(lcc))
    lnZ_slq = np.interp(np.log(t_grid), np.log(t_s),
                        np.log(np.clip(Z_s, 1e-300, None)))
    lnZ_w = np.log(np.clip(N_eff * p, 1e-300, None))
    inw_g = np.interp(np.log(t_grid), np.log(t_s),
                      inw.astype(float)) > 0.5
    joint = reliable & inw_g
    # tolerance: combined MC + SLQ standard errors in ln Z, floored at 1%
    se_w = np.sqrt(np.clip(1.0 - p, 0, 1) / np.maximum(p * args.walkers, 1))
    se_q = np.interp(np.log(t_grid), np.log(t_s),
                     Zse_s / np.clip(Z_s, 1e-300, None))
    z_dev = np.abs(lnZ_w - lnZ_slq) / np.maximum(
        np.hypot(se_w, se_q), 0.01)
    max_z = float(z_dev[joint].max()) if joint.any() else float("nan")
    # d_s overlay stays as the visual, slope window matched in LOG-T WIDTH
    # to the SLQ pipeline's (hw=8 of a 240-point grid over its own range)
    slq_dec = (np.log10(t_s[-1] / t_s[0])) * (8.0 / max(len(t_s) - 1, 1))
    w_dec = np.log10(t_grid[-1] / t_grid[0]) / max(len(t_grid) - 1, 1)
    hw_match = max(2, int(round(slq_dec / max(w_dec, 1e-12))))
    ds_w = walker_ds_curve(p, t_grid, half_window=hw_match)
    d_interp = np.interp(t_grid, t_s, d_s)
    dev = np.abs(ds_w - d_interp)[joint]
    max_dev = float(dev.max()) if dev.size else float("nan")

    # bootstrap band over walkers (resampling the stored hit matrix)
    B = 16
    ds_b = np.empty((B, len(t_grid)))
    Wn = hits.shape[1]
    log_fact = np.concatenate(([0.0],
                               np.cumsum(np.log(np.arange(1, M + 1)))))
    ms = np.arange(M + 1, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, Wn, size=Wn)
        g_b = hits[:, idx].mean(axis=1)
        p_b = np.empty_like(t_grid)
        for i, t in enumerate(t_grid):
            lt = Lam * t
            logw = -lt + ms * math.log(lt) - log_fact
            w = np.exp(logw - logw.max()); w /= w.sum()
            p_b[i] = float(w @ g_b)
        ds_b[b] = walker_ds_curve(p_b, t_grid, half_window=hw_match)
    ds_lo, ds_hi = np.percentile(ds_b, [16, 84], axis=0)


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_s, d_s, "-", color="#4aa3df", lw=1.8,
            label="SLQ heat kernel (spectral)")
    ax.fill_between(t_grid[reliable], ds_lo[reliable], ds_hi[reliable],
                    color="#e0633a", alpha=0.20)
    ax.plot(t_grid[reliable], ds_w[reliable], "o-", color="#e0633a", ms=4,
            lw=1.0, label=f"MC walkers ({args.walkers:,}, ±1σ band)")
    if (~reliable).any():
        ax.plot(t_grid[~reliable], ds_w[~reliable], "o", color="#e0633a",
                ms=4, mfc="none", alpha=0.5, label="walker (too few returns)")
    ax.axhline(4, color="#888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("diffusion time t (log)")
    ax.set_ylabel("spectral dimension d_s(t)")
    ax.set_title(f"{tag} — same quantity, two estimators: "
                 f"Tr e^{{−tL}} by Lanczos vs by simulated walkers\n"
                 f"Z agreement: max {max_z:.1f}σ; d_s overlay max |Δ| = "
                 f"{max_dev:.3f} (matched slope windows)", fontsize=10)
    ax.legend(fontsize=8)
    out_png = os.path.join(args.dir, f"walker_check_{tag}.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

    verdict = ("AGREE" if max_z < 4.0 else "DISAGREE")
    print(f"[walkers] {verdict}: Z(t) matches within {max_z:.1f}σ of the "
          f"combined MC+SLQ errors over {int(joint.sum())} jointly reliable "
          f"points (threshold 4σ); d_s overlay max |Δ| = {max_dev:.3f} "
          f"→ {out_png}")
    try:
        eng.close()
    except Exception:
        pass
    return 0 if verdict == "AGREE" else 2


if __name__ == "__main__":
    sys.exit(main())
