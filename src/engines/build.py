"""
engines.build — engine-agnostic growth helpers
==============================================
Small utilities that work against ANY engine through the interface, so the
benchmark, main.py and the probes don't each reinvent them:

  calibrate_degree_penalty(cls, k, base)  -> μ that yields average degree ≈ k
  detect_equilibrium_sweeps(cls, params)  -> sweeps until the degree plateaus

These intentionally mirror (and could eventually replace) the bespoke versions
in src/core/graph_builder.py; kept here so a probe can grow a graph from any
backend in one call.
"""
from __future__ import annotations

import math


def _avg_deg(cls, n, sweeps, seed, params, max_degree):
    eng = cls(n, max_degree=max_degree, seed=seed, **params)
    try:
        eng.sweep(sweeps)
    except RuntimeError:
        pass
    d = eng.stats()["k_avg"]
    eng.close()
    return d


def calibrate_degree_penalty(cls, target_k, base_params, *, n=30000, sweeps=300,
                             max_degree=32, tol=0.12, log=print):
    """Bisect degree_penalty (μ) so the equilibrium average degree ≈ target_k.
    avg-deg decreases monotonically in μ. Closed-form seed: μ ≈ -ec/(2(2k+1))."""
    ec = base_params.get("edge_cost", -1.0)
    seed0 = (-ec / (2.0 * (2 * target_k + 1))) if ec < 0 else 0.03
    lo, hi = seed0 * 0.4, seed0 * 2.5

    def avg(mu):
        return _avg_deg(cls, n, sweeps, 11, {**base_params, "degree_penalty": mu}, max_degree)

    d_lo, d_hi, t = avg(lo), avg(hi), 0
    while d_lo < target_k and t < 5:
        lo *= 0.6; d_lo = avg(lo); t += 1
    while d_hi > target_k and t < 10:
        hi *= 1.6; d_hi = avg(hi); t += 1
    mu, d = seed0, None
    for _ in range(9):
        mu = 0.5 * (lo + hi); d = avg(mu)
        if abs(d - target_k) <= tol:
            break
        lo, hi = (mu, hi) if d > target_k else (lo, mu)
    if log:
        log(f"[calibrate] mean_degree {target_k} -> degree_penalty={mu:.4f} (avg deg {d:.2f})")
    return mu


def _plateaued(hist, window=3, rel=0.01):
    if len(hist) < window + 1:
        return False
    recent = [d for _, d in hist[-(window + 1):]]
    if recent[-1] < 0.5:
        return False
    lo, hi = min(recent), max(recent)
    return (hi - lo) <= rel * max(hi, 1e-9)


def detect_equilibrium_sweeps(cls, params, *, n=40000, max_degree=32, cap=2000,
                              margin=1.25, log=print):
    """Grow a probe graph in chunks until its average degree plateaus; return
    sweeps × margin (rounded to 5). Off the clock — the only mid-growth sampling."""
    eng = cls(n, max_degree=max_degree, seed=7, **params)
    chunk, cum, hist = 5, 0, []
    while cum < cap:
        try:
            eng.sweep(chunk)
        except RuntimeError:
            break
        cum += chunk
        hist.append((cum, eng.stats()["k_avg"]))
        if _plateaued(hist):
            break
        chunk = min(40, chunk + 5)
    eng.close()
    eq = hist[-1][0] if hist else 300
    sweeps = int(math.ceil(eq * margin / 5.0) * 5)
    if log and hist:
        log(f"[equilibrium] avg degree plateaus by ~{eq} sweeps -> {sweeps} sweeps/cell "
            f"(final avg deg {hist[-1][1]:.2f})")
    return sweeps
