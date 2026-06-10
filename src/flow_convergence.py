#!/usr/bin/env python3
"""
flow_convergence.py — does d_s(t) stabilize at 4 as N grows?
=============================================================
Standalone read-only analysis. Sits next to main.py, digs through the
existing flow_*.csv output, and turns the "is this cell converging to a
flat 4 or just creeping up / wiggling?" eyeball-call into numbers — per
(k, T, lb) cell, across its N ladder and across seeds.

It computes the four diagnostics:

  (1) WIGGLE-TO-BAND ratio.  Plateau wiggle RMS divided by the median
      jackknife SE (d_s_se) in the plateau. If ~1, the wiggle is just
      stochastic-trace noise -> throw probes at it, not N. If >>1, the
      wiggle is real graph/finite-size structure. (Heuristic: adjacent
      d_s(t) points are correlated by the 21-pt local-slope window, so
      this is a guide, not a formal test.)

  (2) WIGGLE AMPLITUDE vs N  (log-log, with a c + a*N^-b floor fit).
      Decaying toward 0  -> finite-size; the curve is collapsing onto a
      flat plateau -> 4D-consistent. Leveling at a nonzero floor while
      the per-seed scatter keeps shrinking -> genuine log-periodic /
      discrete-scale-invariance oscillation ("4D on average").

  (3) PLATEAU VALUE vs N  (N^-b extrapolation to d_inf, echoing
      metrics/size_extrapolation). Converging to ~4 -> 4D. Marching
      upward with no sign of leveling -> the cell is really >4D and the
      expander creep hadn't surfaced at small N yet.

  (4) DEVIATION FROM 4 over the plateau, vs N. RMS of (d_s - 4) across
      the plateau window (captures offset AND dip/wiggle excursion).
      -> 0 with N == collapsing onto flat-4. Optional: pass --ref-csv to
      compare against a real reference curve (e.g. a 4D-torus d_s(t))
      instead of the flat line, for the stronger universality-class test.

The plateau is the post-peak, in-window region of d_s(t) (NOT the dip
floor the heatmap reads) — i.e. the settled part you actually want to
judge "is it 4?" on. Wiggle is the residual after removing a smooth
degree-<=2 trend in log-t, so the real dip-and-recover shape is not
counted as wiggle.

Run (from the project root)
---------------------------
    python src/flow_convergence.py                     # auto-find output/
    python src/flow_convergence.py --dir output --plots
    python src/flow_convergence.py --lb-min 0.9 --T 0  # focus the high-lb band
    python src/flow_convergence.py --N-min 256000      # mirror "start high"

Reads only; writes flow_convergence.csv (+ flow_convergence_top.png with --plots).
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Use the project's one finite-size-scaling fit instead of a private copy.
# size_extrapolation.py is self-contained (math + numpy; scipy is lazy), so we
# import the module file directly — adding src/metrics to the path rather than
# the metrics package — to avoid pulling in numba just for this analysis tool.
# This file lives in src/, so metrics/ is a sibling directory.
import sys as _sys
_METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "metrics")
if _METRICS_DIR not in _sys.path:
    _sys.path.insert(0, _METRICS_DIR)
from size_extrapolation import extrapolate_asymptote

RE_CELL = re.compile(
    r"^flow_k(\d+)_T([\d.eE+-]+)_lb([\d.eE+-]+)_N(\d+)_s(\d+)\.csv$")


# ════════════════════════════════════════════════════════════════════
#  Discovery + loading
# ════════════════════════════════════════════════════════════════════
def find_flow_files(base):
    """Glob flow_*.csv whether run from the project root or from output/."""
    pats = []
    for d in (base, os.path.join(base, "output")):
        pats.append(os.path.join(d, "flow_k*_*.csv"))
        pats.append(os.path.join(d, "flow", "flow_k*_*.csv"))
    paths = []
    for p in pats:
        paths.extend(glob.glob(p))
    return sorted(set(paths))


def load_curve(path):
    """Return (t, d_s, d_s_se, in_window) float/bool arrays. Only these
    columns are used (Z is irrelevant to the d_s flow diagnostics)."""
    t, d, se, w = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]))
            d.append(float(r["d_s_mean"]))
            se.append(float(r.get("d_s_se", "nan")))
            w.append(bool(int(r["in_window"])))
    return (np.array(t), np.array(d), np.array(se),
            np.array(w, dtype=bool))


# ════════════════════════════════════════════════════════════════════
#  Per-curve plateau extraction + wiggle
# ════════════════════════════════════════════════════════════════════
def plateau_slice(t, d, inw, min_pts=5):
    """Post-peak in-window region. Returns (idx, flag) or (None, reason).

    The plateau is everything from the in-window peak of d_s to the end
    of the window. If the peak sits at the very right edge (curve still
    rising, window too short), fall back to the second half of the
    in-window region and flag it 'short' so the caller can de-weight it.
    """
    m = inw & np.isfinite(d)
    if m.sum() < 8:                      # matches the run's <8 convention
        return None, "lt8_in_window"
    idx = np.flatnonzero(m)
    pk = int(np.argmax(d[idx]))          # position of the peak within idx
    post = idx[pk:]
    flag = "ok"
    if len(post) < min_pts:
        flag = "short"
        post = idx[max(0, len(idx) // 2):]
    elif len(post) >= 8:
        # drop the leading descent shoulder so the plateau mean isn't
        # biased upward by the peak (matters when the window is short).
        ntrim = max(1, int(0.25 * len(post)))
        if len(post) - ntrim >= 4:
            post = post[ntrim:]
    if len(post) < 3:
        return None, "plateau_too_short"
    return post, flag


def curve_metrics(t, d, se, inw):
    """One curve -> dict of plateau scalars, or None if unusable."""
    sl, flag = plateau_slice(t, d, inw)
    if sl is None:
        return None
    tp, dp = t[sl], d[sl]
    sep = se[sl]
    sep = np.where(np.isfinite(sep) & (sep > 0), sep, np.nan)
    n = len(dp)

    logt = np.log(tp)
    # weighted plateau value (inverse-variance), with a plain-mean fallback
    if np.isfinite(sep).any():
        w = np.where(np.isfinite(sep), 1.0 / sep ** 2, 0.0)
        value = float(np.sum(w * dp) / np.sum(w)) if w.sum() > 0 else float(np.mean(dp))
    else:
        value = float(np.mean(dp))

    # wiggle = residual after a smooth low-order log-t trend
    deg = 1 if n < 6 else 2
    try:
        if np.isfinite(sep).all() and (sep > 0).all():
            coef = np.polyfit(logt, dp, deg, w=1.0 / sep)
        else:
            coef = np.polyfit(logt, dp, deg)
        trend = np.polyval(coef, logt)
        wiggle = float(np.sqrt(np.mean((dp - trend) ** 2)))
    except Exception:
        wiggle = float(np.std(dp))

    band = float(np.nanmedian(sep)) if np.isfinite(sep).any() else float("nan")
    dev4 = float(np.sqrt(np.mean((dp - 4.0) ** 2)))

    return dict(value=value, wiggle=wiggle, band=band, dev4=dev4,
                npts=n, flag=flag, t_lo=float(tp[0]), t_hi=float(tp[-1]),
                tp=tp, dp=dp)


# ════════════════════════════════════════════════════════════════════
#  Fits across the N ladder
# ════════════════════════════════════════════════════════════════════
def _powlaw(N, c, a, b):
    return c + a * np.power(N, -b)


def value_extrap(Ns, ys, sigma):
    """Extrapolate plateau value to N->inf, y = d_inf + a*N^-b.

    Consolidated: for >=3 N the actual fit is the project's shared
    metrics.size_extrapolation.extrapolate_asymptote (weighted 3-param NLS,
    fixed-β fallback, Birge-ratio CI inflation) — the same routine the rest
    of the codebase uses, rather than a second private copy. The n=1 (lone
    value) and n=2 (delta only) shortcuts stay here, since an asymptotic fit
    needs at least 3 points.
    """
    Ns = np.asarray(Ns, float)
    ys = np.asarray(ys, float)
    sigma = np.asarray(sigma, float)
    n = len(Ns)
    if n == 1:
        return dict(d_inf=float(ys[0]), ci=float("nan"),
                    beta=float("nan"), model="1N")
    if n == 2:
        return dict(d_inf=float("nan"), ci=float("nan"), beta=float("nan"),
                    model="2N", delta=float(ys[1] - ys[0]))
    # >=3 points: hand off to the shared fit. It wants 95% CIs, not the
    # 1-sigma SEMs we carry, so convert; clean up non-positive SEMs first.
    finite = np.isfinite(sigma) & (sigma > 0)
    med = float(np.nanmedian(sigma[finite])) if finite.any() else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    sig = np.where(finite, sigma, med)
    guess = float(ys[-1]) if (np.isfinite(ys[-1]) and ys[-1] > 0.5) else 4.0
    r = extrapolate_asymptote(Ns, ys, 1.96 * sig, d_s_guess=guess)
    d_inf = r.get("d_inf", float("nan"))
    return dict(
        d_inf=float(d_inf) if d_inf is not None else float("nan"),
        ci=float(r.get("ci95_inf", float("nan"))),
        beta=float(r.get("beta", float("nan"))),
        model=str(r.get("regime") or r.get("reason") or "extrap"),
    )


def decay_trend(Ns, ys):
    """For a quantity expected to shrink with N (wiggle, seed-std, dev4).
    Returns log-log slope (beta_decay = -slope), direction, and a floor
    estimate c from c + a*N^-b (c~0 => ->0; c>0 => persistent floor)."""
    Ns = np.asarray(Ns, float)
    ys = np.asarray(ys, float)
    out = dict(beta=float("nan"), r2=float("nan"), dir="?",
               floor=float("nan"), first=float(ys[0]) if len(ys) else float("nan"),
               last=float(ys[-1]) if len(ys) else float("nan"))
    fin = np.isfinite(Ns) & np.isfinite(ys) & (Ns > 0)
    if fin.sum() >= 2:
        lx = np.log(Ns[fin])
        pos = ys[fin] > 1e-9
        if pos.sum() >= 2:
            ly = np.log(np.clip(ys[fin], 1e-9, None))
            A = np.polyfit(lx, ly, 1)
            slope = A[0]
            pred = np.polyval(A, lx)
            ss = np.sum((ly - ly.mean()) ** 2)
            out["beta"] = float(-slope)
            out["r2"] = float(1 - np.sum((ly - pred) ** 2) / ss) if ss > 0 else float("nan")
        f0 = ys[fin][0]
        rel = (ys[fin][-1] - f0) / f0 if f0 > 0 else (ys[fin][-1] - f0)
        out["dir"] = "down" if rel < -0.2 else ("up" if rel > 0.2 else "flat")
    if fin.sum() >= 3 and _HAVE_SCIPY:
        try:
            Nf, yf = Ns[fin], ys[fin]
            popt, _ = curve_fit(_powlaw, Nf, yf,
                                p0=[max(yf[-1], 0.0), yf[0], 0.5],
                                maxfev=20000,
                                bounds=([0.0, -50.0, 0.05],
                                        [max(yf) + 1.0, 50.0, 3.0]))
            out["floor"] = float(popt[0])
        except Exception:
            pass
    return out


# ════════════════════════════════════════════════════════════════════
#  Reference curve (optional, for diagnostic 4 upgrade)
# ════════════════════════════════════════════════════════════════════
def load_reference(path):
    t, d = [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        tcol = "t" if "t" in cols else cols[0]
        dcol = ("d_s_mean" if "d_s_mean" in cols
                else ("d_s" if "d_s" in cols else cols[1]))
        for r in rd:
            try:
                t.append(float(r[tcol])); d.append(float(r[dcol]))
            except (ValueError, KeyError):
                continue
    t, d = np.array(t), np.array(d)
    o = np.argsort(t)
    return t[o], d[o]


def residual_vs_ref(tp, dp, ref_t, ref_d):
    """RMS of (curve - reference) over the overlapping log-t range."""
    lo = max(tp.min(), ref_t.min())
    hi = min(tp.max(), ref_t.max())
    m = (tp >= lo) & (tp <= hi)
    if m.sum() < 3:
        return float("nan")
    ref_on = np.interp(np.log(tp[m]), np.log(ref_t), ref_d)
    return float(np.sqrt(np.mean((dp[m] - ref_on) ** 2)))


# ════════════════════════════════════════════════════════════════════
#  Verdict + score
# ════════════════════════════════════════════════════════════════════
def verdict_and_score(c):
    """c: per-cell aggregate dict. Returns (verdict, notes, score 0-100)."""
    Ns = c["Ns"]
    n = len(Ns)
    val = c["value"]
    seedstd = c["seedstd"]
    wig = c["wiggle"]
    band = c["band"]
    d_inf = c["val_extrap"]["d_inf"]
    wig_t = c["wig_trend"]
    seed_t = c["seed_trend"]
    notes = []

    d_target = d_inf if np.isfinite(d_inf) else (val[-1] if n else float("nan"))
    ratio = (wig[-1] / band[-1]) if (n and np.isfinite(band[-1]) and band[-1] > 0) else float("nan")

    if any(f == "short" for f in c["flags"]):
        notes.append("plateau window short at some N (raise N / widen in_window)")
    if np.isfinite(ratio) and ratio < 1.3:
        notes.append("wiggle ~= probe noise -> add probes, not N")

    if n < 2:
        v = (f"NEED >=2 N (single point @ {val[-1]:.2f})" if n
             else "no usable curves")
        sc = 0.0 if not np.isfinite(d_target) else max(0.0, 55 - 60 * min(abs(d_target - 4), 1.0))
        return v, notes, round(sc, 1)

    near4 = np.isfinite(d_target) and abs(d_target - 4.0) <= 0.30
    far_lo = np.isfinite(d_target) and d_target < 3.60
    far_hi = np.isfinite(d_target) and d_target > 4.40
    rise_abs = float(val[-1] - val[0]) if n >= 2 else 0.0
    creeping = (rise_abs > 0.15) and (val[-1] > 4.25)
    resolving = (wig_t["dir"] == "down") and (seed_t["dir"] in ("down", "flat"))
    floor_osc = (np.isfinite(wig_t["floor"]) and wig_t["floor"] > 0.12
                 and wig_t["dir"] != "down")
    good_trend = resolving or floor_osc

    if creeping:
        v = f"CREEPING >4 (->{d_target:.2f}, value rising)"
    elif near4 and resolving:
        v = f"CONVERGING->4 [check] ({d_target:.2f})"
    elif near4 and floor_osc:
        v = f"4D + oscillation (mean~{d_target:.2f}, wiggle floor~{wig_t['floor']:.2f})"
    elif near4:
        v = f"near 4 ({d_target:.2f}); trend unclear"
    elif far_lo:
        v = f"sub-4 ({d_target:.2f})"
    elif far_hi:
        v = f">4 ({d_target:.2f})"
    elif np.isfinite(d_target):
        v = f"ambiguous ({d_target:.2f})"
    else:
        v = "ambiguous"

    # transparent score: closeness dominates; reward resolving trends.
    sc = 100.0
    sc -= 60.0 * (min(abs(d_target - 4.0), 1.0) if np.isfinite(d_target) else 1.0)
    if creeping:
        sc -= 20.0
    if not good_trend:
        sc -= 12.0
    last_spread = seedstd[-1] if (n and np.isfinite(seedstd[-1])) else 0.30
    sc -= 12.0 * min(last_spread / 0.40, 1.0)
    if c["val_extrap"]["model"] in ("1N", "2N"):
        sc = min(sc, 60.0)              # can't trust closeness without a fit
    sc = max(0.0, min(100.0, sc))
    return v, notes, round(sc, 1)


# ════════════════════════════════════════════════════════════════════
#  Driver
# ════════════════════════════════════════════════════════════════════
def _therm_verdict_for(curve_path):
    """therm_verified from the meta sidecar of one flow CSV: True / False /
    None (no sidecar, or sidecar predates the verification pass)."""
    mp = os.path.join(os.path.dirname(curve_path),
                      os.path.basename(curve_path)
                      .replace("flow_", "meta_", 1)
                      .rsplit(".", 1)[0] + ".json")
    try:
        with open(mp) as fh:
            return json.load(fh).get("therm_verified")
    except (OSError, ValueError):
        return None


def build_cells(paths, args, ref=None):
    # per (k,T,lb) -> per N -> list of per-curve metric dicts
    raw = defaultdict(lambda: defaultdict(list))
    refres = defaultdict(lambda: defaultdict(list))  # ref residuals
    therm = defaultdict(lambda: defaultdict(list))   # therm_verified per seed
    n_files = n_used = 0
    for p in paths:
        m = RE_CELL.match(os.path.basename(p))
        if not m:
            continue
        k, T, lb, N, _seed = (int(m.group(1)), float(m.group(2)),
                               float(m.group(3)), int(m.group(4)), int(m.group(5)))
        # filters
        if args.k and k not in args.k:
            continue
        if args.T is not None and not any(abs(T - tt) < 1e-12 for tt in args.T):
            continue
        if lb < args.lb_min or lb > args.lb_max:
            continue
        if args.N_min and N < args.N_min:
            continue
        n_files += 1
        try:
            t, d, se, w = load_curve(p)
            cm = curve_metrics(t, d, se, w)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        if cm is None:
            continue
        n_used += 1
        raw[(k, T, lb)][N].append(cm)
        therm[(k, T, lb)][N].append(_therm_verdict_for(p))
        if ref is not None:
            refres[(k, T, lb)][N].append(
                residual_vs_ref(cm["tp"], cm["dp"], ref[0], ref[1]))

    print(f"[scan] {n_files} flow files matched filters, "
          f"{n_used} produced a usable plateau")

    cells = []
    for key, byN in raw.items():
        k, T, lb = key
        Ns = sorted(byN)
        _tv = therm.get(key, {}).get(Ns[-1], [])
        therm_ok = not (_tv and all(v is False for v in _tv))
        agg = dict(k=k, T=T, lb=lb, Ns=Ns, therm_ok=therm_ok,
                   value=[], value_sem=[], seedstd=[], wiggle=[],
                   band=[], dev4=[], nseed=[], flags=[], refres=[])
        for N in Ns:
            ms = byN[N]
            vals = np.array([x["value"] for x in ms])
            wigs = np.array([x["wiggle"] for x in ms])
            bands = np.array([x["band"] for x in ms])
            devs = np.array([x["dev4"] for x in ms])
            ns = len(ms)
            agg["value"].append(float(np.mean(vals)))
            if ns >= 2:
                sd = float(np.std(vals, ddof=1))
                agg["seedstd"].append(sd)
                agg["value_sem"].append(sd / math.sqrt(ns))
            else:
                agg["seedstd"].append(float("nan"))
                # single seed: use that curve's jackknife band as the SEM proxy
                agg["value_sem"].append(float(bands[0]) if np.isfinite(bands[0]) else float("nan"))
            agg["wiggle"].append(float(np.mean(wigs)))
            agg["band"].append(float(np.nanmean(bands)))
            agg["dev4"].append(float(np.mean(devs)))
            agg["nseed"].append(ns)
            agg["flags"].extend([x["flag"] for x in ms])
            if ref is not None:
                rr = np.array(refres[key][N], float)
                agg["refres"].append(float(np.nanmean(rr)))
        for kk in ("value", "value_sem", "seedstd", "wiggle", "band",
                   "dev4", "refres"):
            agg[kk] = np.array(agg[kk], float)

        agg["val_extrap"] = value_extrap(Ns, agg["value"], agg["value_sem"])
        agg["val_dir"] = decay_trend(Ns, agg["value"])       # dir only used
        agg["wig_trend"] = decay_trend(Ns, agg["wiggle"])
        agg["seed_trend"] = decay_trend(Ns, agg["seedstd"])
        agg["dev_trend"] = decay_trend(Ns, agg["dev4"])
        agg["verdict"], agg["notes"], agg["score"] = verdict_and_score(agg)
        cells.append(agg)

    cells.sort(key=lambda c: (-c["score"], c["k"], c["lb"], c["T"]))
    return cells


def fmt_seq(arr, fmt="{:.2f}"):
    return "[" + " ".join(("--" if not np.isfinite(x) else fmt.format(x))
                          for x in arr) + "]"


def print_report(cells, args):
    enough = [c for c in cells if len(c["Ns"]) >= 2]
    thin = [c for c in cells if len(c["Ns"]) < 2]

    print("\n" + "=" * 100)
    print("4D-CANDIDATE RANKING  (cells with >=2 N points, sorted by score)")
    print("score = 100 - 60*min(|d_inf-4|,1) - 20*creeping - 12*not_resolving "
          "- 12*min(seedstd/0.4,1)")
    print("=" * 100)
    hdr = (f"{'score':>5} {'k':>2} {'T':>6} {'lb':>6} {'d_inf':>11} "
           f"{'val@maxN':>8} {'wig@maxN':>8} {'w/band':>6} "
           f"{'seedSD':>6} {'verdict'}")
    print(hdr)
    print("-" * 100)
    for c in enough:
        e = c["val_extrap"]
        dinf = (f"{e['d_inf']:.2f}+-{e['ci']:.2f}"
                if np.isfinite(e.get("d_inf", float('nan')))
                and np.isfinite(e.get("ci", float('nan')))
                else (f"d2N:{e.get('delta', float('nan')):+.2f}"
                      if e["model"] == "2N" else "--"))
        ratio = (c["wiggle"][-1] / c["band"][-1]
                 if np.isfinite(c["band"][-1]) and c["band"][-1] > 0 else float("nan"))
        ss = c["seedstd"][-1]
        print(f"{c['score']:>5.0f} {c['k']:>2} {c['T']:>6g} {c['lb']:>6g} "
              f"{dinf:>11} {c['value'][-1]:>8.2f} {c['wiggle'][-1]:>8.3f} "
              f"{(f'{ratio:.1f}' if np.isfinite(ratio) else '--'):>6} "
              f"{(f'{ss:.3f}' if np.isfinite(ss) else '--'):>6} {c['verdict']}")

    # detail for the top candidates
    top = enough[:args.top]
    if top:
        print("\n" + "=" * 100)
        print(f"TOP {len(top)} — N-ladder detail")
        print("=" * 100)
        for c in top:
            print(f"\n  k={c['k']} T={c['T']:g} lb={c['lb']:g}   "
                  f"[{c['verdict']}]   score={c['score']:.0f}")
            print(f"    N        : {[f'{int(n):,}' for n in c['Ns']]}")
            print(f"    n_seeds  : {c['nseed']}")
            print(f"    value    : {fmt_seq(c['value'])}   "
                  f"(value->inf via {c['val_extrap']['model']}, "
                  f"beta={c['val_extrap'].get('beta', float('nan')):.2f})")
            print(f"    seed-std : {fmt_seq(c['seedstd'], '{:.3f}')}   "
                  f"trend {c['seed_trend']['dir']}")
            print(f"    wiggle   : {fmt_seq(c['wiggle'], '{:.3f}')}   "
                  f"trend {c['wig_trend']['dir']}, "
                  f"floor~{c['wig_trend']['floor']:.3f} "
                  f"(beta_decay={c['wig_trend']['beta']:.2f})")
            print(f"    band(SE) : {fmt_seq(c['band'], '{:.3f}')}")
            print(f"    dev-from4: {fmt_seq(c['dev4'], '{:.3f}')}   "
                  f"trend {c['dev_trend']['dir']}")
            if c["refres"].size:
                print(f"    ref-resid: {fmt_seq(c['refres'], '{:.3f}')}   "
                      f"trend {decay_trend(c['Ns'], c['refres'])['dir']}")
            for nt in c["notes"]:
                print(f"    note: {nt}")

    if thin:
        print("\n" + "-" * 100)
        print(f"{len(thin)} cell(s) with only 1 N (no ladder yet) — "
              f"run a second N to triage:")
        for c in sorted(thin, key=lambda c: abs(c['value'][-1] - 4)):
            print(f"    k={c['k']:>2} T={c['T']:g} lb={c['lb']:g}  "
                  f"N={int(c['Ns'][0]):,}  value={c['value'][-1]:.2f}")

    print()


def write_csv(cells, path):
    fields = ["k", "T", "lb", "n_N", "N_max", "therm_ok", "score", "verdict",
              "d_inf", "d_inf_ci", "beta_value", "extrap_model",
              "value_at_maxN", "value_dir",
              "wiggle_at_maxN", "wiggle_dir", "wiggle_floor", "wiggle_beta",
              "wiggle_to_band_at_maxN",
              "seedstd_at_maxN", "seedstd_dir",
              "dev4_at_maxN", "dev4_dir",
              "N_list", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            e = c["val_extrap"]
            ratio = (c["wiggle"][-1] / c["band"][-1]
                     if np.isfinite(c["band"][-1]) and c["band"][-1] > 0 else float("nan"))
            w.writerow({
                "k": c["k"], "T": c["T"], "lb": c["lb"],
                "n_N": len(c["Ns"]), "N_max": int(c["Ns"][-1]),
                "therm_ok": c.get("therm_ok", True),
                "score": c["score"], "verdict": c["verdict"],
                "d_inf": f"{e.get('d_inf', float('nan')):.4f}",
                "d_inf_ci": f"{e.get('ci', float('nan')):.4f}",
                "beta_value": f"{e.get('beta', float('nan')):.4f}",
                "extrap_model": e["model"],
                "value_at_maxN": f"{c['value'][-1]:.4f}",
                "value_dir": c["val_dir"]["dir"],
                "wiggle_at_maxN": f"{c['wiggle'][-1]:.4f}",
                "wiggle_dir": c["wig_trend"]["dir"],
                "wiggle_floor": f"{c['wig_trend']['floor']:.4f}",
                "wiggle_beta": f"{c['wig_trend']['beta']:.4f}",
                "wiggle_to_band_at_maxN": f"{ratio:.3f}",
                "seedstd_at_maxN": f"{c['seedstd'][-1]:.4f}",
                "seedstd_dir": c["seed_trend"]["dir"],
                "dev4_at_maxN": f"{c['dev4'][-1]:.4f}",
                "dev4_dir": c["dev_trend"]["dir"],
                "N_list": ";".join(str(int(n)) for n in c["Ns"]),
                "notes": " | ".join(c["notes"]),
            })
    print(f"[csv] wrote {path}  ({len(cells)} cells)")


def make_plots(cells, path, top_n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable ({e}); skipping")
        return
    top = [c for c in cells if len(c["Ns"]) >= 2][:top_n]
    if not top:
        print("[plots] nothing with a ladder to plot")
        return
    fig, axes = plt.subplots(len(top), 2, figsize=(11, 2.6 * len(top)),
                             squeeze=False)
    for i, c in enumerate(top):
        Ns = np.array(c["Ns"], float)
        ax = axes[i][0]
        ax.errorbar(Ns, c["value"], yerr=c["value_sem"], fmt="o-",
                    color="#7ec9ff", capsize=3)
        e = c["val_extrap"]
        if np.isfinite(e.get("d_inf", float("nan"))):
            ax.axhline(e["d_inf"], color="#c586c0", ls="--",
                       label=f"d_inf={e['d_inf']:.2f}")
            if np.isfinite(e.get("ci", float("nan"))):
                ax.axhspan(e["d_inf"] - e["ci"], e["d_inf"] + e["ci"],
                           color="#c586c0", alpha=0.12)
        ax.axhline(4.0, color="#7ec96e", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"k={c['k']} T={c['T']:g} lb={c['lb']:g} — plateau value",
                     fontsize=9)
        ax.set_xlabel("N"); ax.set_ylabel("d_s plateau"); ax.legend(fontsize=7)

        ax2 = axes[i][1]
        ax2.loglog(Ns, np.clip(c["wiggle"], 1e-4, None), "s-",
                   color="#ce9178", label="wiggle RMS")
        ss = c["seedstd"]
        if np.isfinite(ss).any():
            ax2.loglog(Ns, np.clip(ss, 1e-4, None), "^-",
                       color="#dcdcaa", label="seed-std")
        if np.isfinite(c["band"]).any():
            ax2.loglog(Ns, np.clip(c["band"], 1e-4, None), ":",
                       color="#888", label="probe band")
        ax2.set_title("wiggle / spread / band vs N", fontsize=9)
        ax2.set_xlabel("N"); ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[plots] wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                    help="project root or output/ (auto-detects flow_*.csv)")
    ap.add_argument("--out", default="flow_convergence.csv")
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="restrict to these k values")
    ap.add_argument("--T", type=float, nargs="+", default=None,
                    help="restrict to these T values")
    ap.add_argument("--lb-min", type=float, default=0.0,
                    help="lower lb bound (default 0; e.g. 0.9 for the band)")
    ap.add_argument("--lb-max", type=float, default=1.0)
    ap.add_argument("--N-min", type=int, default=16000,
                    help="drop N below this (default 16000: skip the 1k/4k "
                         "near-noise rungs but keep mid-N as fit anchors). "
                         "Set 0 to use all, or 256000 for a start-high run.")
    ap.add_argument("--ref-csv", default=None,
                    help="optional reference d_s(t) curve (cols t,d_s_mean) "
                         "to subtract for the universality-class residual")
    ap.add_argument("--top", type=int, default=8,
                    help="how many top candidates to detail / plot")
    ap.add_argument("--plots", action="store_true",
                    help="also write flow_convergence_top.png")
    args = ap.parse_args(argv)

    if not _HAVE_SCIPY:
        print("[warn] scipy not found — using fixed-beta linear fits only "
              "(less accurate d_inf, no floor estimate).")

    paths = find_flow_files(args.dir)
    if not paths:
        print(f"[scan] no flow_*.csv under {args.dir!r} "
              f"(looked in ./, ./flow, ./output, ./output/flow). "
              f"Point --dir at the project root or output/.")
        return

    ref = None
    if args.ref_csv:
        try:
            ref = load_reference(args.ref_csv)
            print(f"[ref] loaded {len(ref[0])} pts from {args.ref_csv}")
        except Exception as e:
            print(f"[ref] could not load {args.ref_csv}: {e} — "
                  f"falling back to deviation-from-4 only")

    cells = build_cells(paths, args, ref=ref)
    if not cells:
        print("[main] no cells survived filtering / plateau extraction.")
        return
    print_report(cells, args)
    write_csv(cells, args.out)
    if args.plots:
        make_plots(cells, "flow_convergence_top.png", args.top)


if __name__ == "__main__":
    main()
