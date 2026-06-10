#!/usr/bin/env python3
"""
flow_modes.py — dual-mode 4D search: flat-4D vs flowing-4D
===========================================================
Downstream, read-only meta-analysis. Lives in src/ so the sweep can
`import flow_modes` to auto-refresh it; also runnable directly:

    python src/flow_modes.py --dir output            # manual
    (auto: sweep_runner calls flow_modes.main(["--dir","."]) on refresh)

It reads every flow_*.csv (the FULL d_s(t) curve, including the
out-of-window UV points) plus the per-cell meta_*.json topology sidecar
when present, and scores each (k,T,lb) cell against the TWO distinct 4D
targets we decided to track simultaneously:

  FLAT-4D    d_s(t) = 4 at every scale (regular-lattice geometry).
             Signature: IR plateau ~4, UV shoulder ~4, curve matches the
             4D-torus reference everywhere (residual ~0 across the curve).

  FLOWING-4D the CDT / asymptotic-safety dimensional reduction: d_s ~ 4 in
             the IR (large t) flowing DOWN toward ~2 in the UV (small t).
             Signature: IR plateau ~4 (IR residual vs 4D torus ~0) but the
             UV is suppressed *below* what the lattice shows
             (UV residual vs 4D torus significantly negative).

Why the reference matters: a discrete lattice's d_s(t) ALSO rises from ~2
in the UV (lattice discreteness), so "UV~2" alone does NOT prove a flow.
The discriminator is EXCESS UV suppression beyond the 4D torus. When
matched-N torus references are present in the data (the sweep builds them
by default), this script anchors on them; otherwise it falls back to the
absolute (UV, IR) values and says so loudly.

Topology gate: cells whose largest connected component is below --lcc-min
(default 90%) are flagged and excluded from candidacy — a d_s measured on
a small island is meaningless. LCC% is read from the meta sidecar when
present, else estimated from the heat-kernel trace as max(Z_mean)/N.

Equilibration gate: cells whose meta sidecar carries an explicit
therm_verified=False (the build-time block-comparison stationarity check
failed at its ~x3 budget — see core.graph_builder._verify_equilibration)
are likewise excluded: their d_s does not measure the equilibrium
ensemble. Sidecars without the flag (runs predating the check) pass.

Outputs (cwd):
  flow_modes.csv        per-cell: both mode scores, descriptors, topology,
                        N-trend, verdict
  flow_modes_plane.png  the (UV, IR) plane — the headline picture: flat-4D
                        clusters at one corner, flowing-4D at another
  flow_modes_maps.png   (k x lb) per T, one row = flat score, one = flow

This is additive; it does not touch shape_analysis / flow_convergence.
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
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

RE_CELL = re.compile(
    r"^flow_k(\d+)_T([\d.eE+-]+)_lb([\d.eE+-]+)_N(\d+)_s(\d+)\.csv$")
RE_TORUS = re.compile(r"^flow_torus(\d+)d_L(\d+)_s(\d+)\.csv$")

# target points in the (UV shoulder, IR plateau) plane
FLAT_TARGET = (4.0, 4.0)
FLOW_TARGET = (2.0, 4.0)
LCC_MIN_DEFAULT = 90.0
IR_FLAT_TOL = 0.6           # in-IR std below this = a real plateau
UV_RESID_FLOW = -0.5        # UV residual below this vs torus = excess suppression


# ════════════════════════════════════════════════════════════════════
#  Discovery + loading
# ════════════════════════════════════════════════════════════════════
def find_files(base):
    out = []
    for d in (base, os.path.join(base, "output")):
        out += glob.glob(os.path.join(d, "flow_*.csv"))
        out += glob.glob(os.path.join(d, "flow", "flow_*.csv"))
    return sorted(set(out))


def load_curve(path):
    t, Z, d, se, w = [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]))
            Z.append(float(r.get("Z_mean", "nan")))
            d.append(float(r["d_s_mean"]))
            se.append(float(r.get("d_s_se", "nan")))
            w.append(bool(int(r["in_window"])))
    return (np.array(t), np.array(Z), np.array(d), np.array(se),
            np.array(w, dtype=bool))


def load_meta(csv_path):
    """meta_<tag>.json sidecar next to the flow csv, or None."""
    base = os.path.basename(csv_path)
    meta_name = base.replace("flow_", "meta_", 1).rsplit(".", 1)[0] + ".json"
    meta_path = os.path.join(os.path.dirname(csv_path), meta_name)
    if os.path.exists(meta_path):
        try:
            return json.load(open(meta_path))
        except Exception:
            return None
    return None


def lcc_pct_for(Z, N, meta):
    """Prefer exact LCC% from meta; else estimate from the heat-kernel
    trace: Z(t->0) = Tr(e^0) = N_eff, so max(Z) ~ LCC node count."""
    if meta and np.isfinite(meta.get("lcc_pct", float("nan"))):
        return float(meta["lcc_pct"]), "meta"
    zf = Z[np.isfinite(Z)]
    if zf.size and N > 0:
        return float(100.0 * zf.max() / N), "from_Z"
    return float("nan"), "unknown"


# ════════════════════════════════════════════════════════════════════
#  Shape descriptors: split the curve into UV (rising limb) and IR (plateau)
# ════════════════════════════════════════════════════════════════════
def describe(t, d, se, inw):
    """Return shape descriptors of one d_s(t) curve, or None if unusable.

    peak    : (t, value) of the global in-window maximum
    IR_*    : post-peak in-window plateau (value, in-plateau std, n)
    UV_*    : rising-limb shoulder. UV_min = lowest d_s before the peak;
              UV_conf flags whether that minimum is in-window (trustworthy)
              or out-of-window (UV clipped by the SLQ mask -> low confidence)
    """
    m = inw & np.isfinite(d)
    if m.sum() < 8:
        return None
    idx = np.flatnonzero(m)
    pk = int(np.argmax(d[idx]))            # peak position within in-window
    peak_i = idx[pk]
    peak_t, peak_v = float(t[peak_i]), float(d[peak_i])

    # ---- IR plateau: post-peak, trim leading 25% descent shoulder ----
    post = idx[pk:]
    ir_flag = "ok"
    if len(post) < 5:
        ir_flag = "short"
        post = idx[max(0, len(idx) // 2):]
    elif len(post) >= 8:
        nt = max(1, int(0.25 * len(post)))
        if len(post) - nt >= 4:
            post = post[nt:]
    sep = se[post]
    sep = np.where(np.isfinite(sep) & (sep > 0), sep, np.nan)
    dp = d[post]
    if np.isfinite(sep).any():
        wts = np.where(np.isfinite(sep), 1.0 / sep ** 2, 0.0)
        ir_value = float(np.sum(wts * dp) / np.sum(wts))
    else:
        ir_value = float(np.mean(dp))
    ir_flat = float(np.std(dp))            # in-plateau spread (flatness)
    ir_n = int(len(post))

    # ---- UV shoulder: lowest d_s on the rising limb (t < peak) ----
    # Prefer in-window UV points; if the window starts at/after the peak,
    # fall back to near-window UV points (flagged low-confidence) since the
    # flow signature genuinely lives in the region the mask tends to clip.
    uv_in = np.flatnonzero(inw & np.isfinite(d) & (t < peak_t))
    if uv_in.size >= 2:
        uv_min = float(np.min(d[uv_in]))
        uv_lowt = float(np.mean(d[uv_in][np.argsort(t[uv_in])[:max(2, uv_in.size // 4)]]))
        uv_conf = "in_window"
    else:
        # extend below the window, but only within ~1 decade of its left edge
        left_t = float(t[idx].min())
        uv_oo = np.flatnonzero(np.isfinite(d) & (t < peak_t)
                               & (t >= left_t / 10.0))
        if uv_oo.size >= 2:
            uv_min = float(np.min(d[uv_oo]))
            uv_lowt = float(np.mean(d[uv_oo][np.argsort(t[uv_oo])[:max(2, uv_oo.size // 4)]]))
            uv_conf = "extrapolated"
        else:
            uv_min = uv_lowt = float("nan")
            uv_conf = "none"

    # monotone-descent check (UV below IR below/near peak): flow needs it
    descends = (np.isfinite(uv_min) and uv_min < ir_value - 0.3
                and ir_value <= peak_v + 0.5)

    return dict(peak_t=peak_t, peak_v=peak_v,
                ir_value=ir_value, ir_flat=ir_flat, ir_n=ir_n, ir_flag=ir_flag,
                uv_min=uv_min, uv_lowt=uv_lowt, uv_conf=uv_conf,
                descends=bool(descends),
                t=t, d=d, inw=inw)            # keep arrays for residuals


# ════════════════════════════════════════════════════════════════════
#  4D-torus reference (anchors flat vs flow); built per N from the data
# ════════════════════════════════════════════════════════════════════
def build_ref_4d(torus_curves):
    """torus_curves: list of (N, t, d, inw) for 4D tori. Returns
    {N -> (t_sorted, d_ref)} averaging seeds at each torus N."""
    byN = defaultdict(list)
    for (N, t, d, inw) in torus_curves:
        byN[N].append((t, d))
    ref = {}
    for N, lst in byN.items():
        # all share the same t-grid construction per N; average d on it
        t0 = lst[0][0]
        ds = np.vstack([np.interp(np.log(t0), np.log(t), d)
                        for (t, d) in lst])
        ref[N] = (t0, np.nanmean(ds, axis=0))
    return ref


def nearest_ref(ref, N):
    if not ref:
        return None
    key = min(ref, key=lambda rn: abs(math.log(rn) - math.log(N)))
    # only use if within a factor of ~2 in N
    if abs(math.log(key) - math.log(N)) < math.log(2.2):
        return ref[key]
    return None


def residual_uv_ir(desc, ref_curve):
    """UV-mean and IR-mean of (cell d_s - ref d_s) over the cell's regions."""
    if ref_curve is None:
        return float("nan"), float("nan")
    rt, rd = ref_curve
    t, d, inw, pk = desc["t"], desc["d"], desc["inw"], desc["peak_t"]
    lo, hi = max(t.min(), rt.min()), min(t.max(), rt.max())
    base = np.isfinite(d) & (t >= lo) & (t <= hi)
    if base.sum() < 4:
        return float("nan"), float("nan")
    refon = np.interp(np.log(t[base]), np.log(rt), rd)
    resid = d[base] - refon
    tb = t[base]
    uv_mask = tb < pk
    ir_mask = (tb >= pk) & inw[base]
    # UV summary at the DEEPEST-UV third (smallest t), where excess
    # dimensional suppression — the flow signature — actually shows up;
    # averaging the whole rising limb dilutes it toward zero.
    if uv_mask.sum():
        uvt = tb[uv_mask]
        order = np.argsort(uvt)
        ntake = max(1, len(order) // 3)
        uv = float(np.mean(resid[uv_mask][order[:ntake]]))
    else:
        uv = float("nan")
    ir = float(np.mean(resid[ir_mask])) if ir_mask.sum() else float("nan")
    return uv, ir


# ════════════════════════════════════════════════════════════════════
#  Scoring
# ════════════════════════════════════════════════════════════════════
def mode_scores(uv, ir, ir_flat, descends, uv_resid, ir_resid, have_ref):
    """Return (flat_score, flow_score) in 0-100.

    Anchored mode (have_ref): flat = matches torus everywhere (UV & IR
    residual ~0); flow = IR matches torus (~0) but UV residual << 0.
    Absolute mode (no ref): distance in the (UV, IR) plane to each target.
    """
    flat_ok_shape = np.isfinite(ir_flat) and ir_flat < IR_FLAT_TOL
    if have_ref and np.isfinite(uv_resid) and np.isfinite(ir_resid):
        # FLAT: matches the 4D torus EVERYWHERE -> both residuals ~0
        # (product: both factors must be small to score high).
        flat = 100.0 * max(0.0, 1.0 - abs(ir_resid) / 0.6) \
                     * max(0.0, 1.0 - abs(uv_resid) / 0.6)
        # FLOW: IR matches the torus (~0) AND UV sits BELOW it (excess
        # suppression). uv_excess = how far UV is below the lattice.
        ir_match = max(0.0, 1.0 - abs(ir_resid) / 0.6)
        uv_excess = min(1.0, max(0.0, -uv_resid) / 0.8)
        flow = 100.0 * ir_match * uv_excess
        if not descends:
            flow *= 0.5
        mode = "ref"
    else:
        if not (np.isfinite(uv) and np.isfinite(ir)):
            return 0.0, 0.0, "none"
        # absolute fallback: distance to each target in the (UV, IR) plane.
        # NB: lattice discreteness pushes flat-4D UV down to ~2-3, so this
        # mode genuinely cannot separate flat from flow — hence the warning.
        flat = 100.0 * max(0.0, 1.0 - abs(ir - FLAT_TARGET[1]) / 0.6) \
                     * max(0.0, 1.0 - abs(uv - FLAT_TARGET[0]) / 0.9)
        flow = 100.0 * max(0.0, 1.0 - abs(ir - FLOW_TARGET[1]) / 0.6) \
                     * max(0.0, 1.0 - abs(uv - FLOW_TARGET[0]) / 0.9)
        if not descends:
            flow *= 0.5
        mode = "abs"
    if not flat_ok_shape:
        flat *= 0.4                          # not a flat plateau -> demote flat
    return round(flat, 1), round(flow, 1), mode


# ════════════════════════════════════════════════════════════════════
#  N-trend helper (does the descriptor stabilize as N grows?)
# ════════════════════════════════════════════════════════════════════
def _nanmean(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def trend_dir(Ns, ys):
    Ns, ys = np.asarray(Ns, float), np.asarray(ys, float)
    f = np.isfinite(Ns) & np.isfinite(ys)
    if f.sum() < 2:
        return "?"
    y = ys[f]
    rel = (y[-1] - y[0]) / abs(y[0]) if y[0] != 0 else (y[-1] - y[0])
    return "down" if rel < -0.15 else ("up" if rel > 0.15 else "flat")


# ════════════════════════════════════════════════════════════════════
#  Driver
# ════════════════════════════════════════════════════════════════════
def build(paths, args):
    torus4d, basins = [], defaultdict(lambda: defaultdict(list))
    meta_seen = 0
    for p in paths:
        b = os.path.basename(p)
        mt = RE_TORUS.match(b)
        if mt:
            d, L, seed = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
            if d == 4:
                t, Z, ds, se, w = load_curve(p)
                torus4d.append((L ** d, t, ds, w))
            continue
        mc = RE_CELL.match(b)
        if not mc:
            continue
        k, T, lb, N, seed = (int(mc.group(1)), float(mc.group(2)),
                             float(mc.group(3)), int(mc.group(4)), int(mc.group(5)))
        if args.k and k not in args.k:
            continue
        if args.T is not None and not any(abs(T - x) < 1e-12 for x in args.T):
            continue
        if lb < args.lb_min or lb > args.lb_max:
            continue
        if args.N_min and N < args.N_min:
            continue
        t, Z, ds, se, w = load_curve(p)
        desc = describe(t, ds, se, w)
        if desc is None:
            continue
        meta = load_meta(p)
        if meta:
            meta_seen += 1
        lcc, lcc_src = lcc_pct_for(Z, N, meta)
        trans = float(meta["transitivity"]) if (meta and "transitivity" in meta) else float("nan")
        # Equilibration verdict from the build: True (verified), False
        # (the block-comparison check FAILED at the x3 budget), or None
        # (no sidecar / sidecar predates the verification pass).
        therm_ok = meta.get("therm_verified") if meta else None
        basins[(k, T, lb)][N].append(
            dict(seed=seed, desc=desc, lcc=lcc, lcc_src=lcc_src, trans=trans,
                 therm_ok=therm_ok, N=N))

    ref = build_ref_4d(torus4d)
    have_any_ref = len(ref) > 0
    print(f"[scan] {sum(len(v) for d in basins.values() for v in d.values())} "
          f"basin curves over {len(basins)} cells; "
          f"{len(torus4d)} 4D-torus ref curves ({len(ref)} N); "
          f"{meta_seen} topology sidecars")
    if not have_any_ref:
        print("[ref] no 4D-torus references found -> ABSOLUTE (UV,IR) mode "
              "(can't separate lattice-discreteness UV dip from a real flow; "
              "run the sweep with torus refs to anchor).")

    cells = []
    for key, byN in basins.items():
        k, T, lb = key
        Ns = sorted(byN)
        agg = dict(k=k, T=T, lb=lb, Ns=Ns,
                   ir=[], uv=[], irflat=[], lcc=[], trans=[],
                   uvres=[], irres=[], flat=[], flow=[], nseed=[],
                   uvconf=[], lcc_src=[])
        for N in Ns:
            recs = byN[N]
            irs = np.array([r["desc"]["ir_value"] for r in recs])
            uvs = np.array([r["desc"]["uv_min"] for r in recs])
            flats = np.array([r["desc"]["ir_flat"] for r in recs])
            lccs = np.array([r["lcc"] for r in recs])
            trans = np.array([r["trans"] for r in recs])
            refc = nearest_ref(ref, N)
            uvres_l, irres_l, fl_l, fw_l = [], [], [], []
            for r in recs:
                uvr, irr = residual_uv_ir(r["desc"], refc)
                fl, fw, _ = mode_scores(
                    r["desc"]["uv_min"], r["desc"]["ir_value"],
                    r["desc"]["ir_flat"], r["desc"]["descends"],
                    uvr, irr, refc is not None)
                uvres_l.append(uvr); irres_l.append(irr)
                fl_l.append(fl); fw_l.append(fw)
            agg["ir"].append(_nanmean(irs))
            agg["uv"].append(_nanmean(uvs))
            agg["irflat"].append(_nanmean(flats))
            agg["lcc"].append(_nanmean(lccs))
            agg["trans"].append(_nanmean(trans))
            agg["uvres"].append(_nanmean(uvres_l))
            agg["irres"].append(_nanmean(irres_l))
            agg["flat"].append(_nanmean(fl_l))
            agg["flow"].append(_nanmean(fw_l))
            agg["nseed"].append(len(recs))
            agg["uvconf"].append(recs[0]["desc"]["uv_conf"])
            agg["lcc_src"].append(recs[0]["lcc_src"])
        for kk in ("ir", "uv", "irflat", "lcc", "trans", "uvres", "irres",
                   "flat", "flow"):
            agg[kk] = np.array(agg[kk], float)
        agg["have_ref"] = have_any_ref and np.isfinite(agg["uvres"][-1])
        agg["lcc_ok"] = np.isfinite(agg["lcc"][-1]) and agg["lcc"][-1] >= args.lcc_min
        # Equilibration gate (companion to the LCC gate): d_s measured on a
        # graph whose build-time stationarity check failed is not a
        # measurement of the equilibrium ensemble. Gate only when every seed
        # at the largest N is explicitly False — sidecars without the flag
        # (older runs) stay un-gated.
        _tv = [r.get("therm_ok") for r in byN[Ns[-1]]]
        agg["therm_ok"] = not (_tv and all(v is False for v in _tv))
        agg["ir_dir"] = trend_dir(Ns, agg["ir"])
        agg["uv_dir"] = trend_dir(Ns, agg["uv"])
        agg["flat_dir"] = trend_dir(Ns, agg["irflat"])
        # verdict
        fl, fw = agg["flat"][-1], agg["flow"][-1]
        if not agg["lcc_ok"]:
            agg["verdict"] = f"REJECT: LCC {agg['lcc'][-1]:.0f}% < {args.lcc_min:.0f}%"
            fl = fw = 0.0
        elif not agg["therm_ok"]:
            agg["verdict"] = "REJECT: not equilibrated (therm_verified=False)"
            fl = fw = 0.0
        elif fl >= 60 and fl >= fw:
            agg["verdict"] = f"FLAT-4D ({fl:.0f})"
        elif fw >= 60 and fw > fl:
            tag = "" if agg["uvconf"][-1] == "in_window" else " [UV low-conf]"
            agg["verdict"] = f"FLOWING-4D ({fw:.0f}){tag}"
        elif max(fl, fw) >= 35:
            agg["verdict"] = f"weak ({'flat' if fl>=fw else 'flow'} {max(fl,fw):.0f})"
        else:
            agg["verdict"] = "neither"
        agg["flat_final"], agg["flow_final"] = fl, fw
        cells.append(agg)
    cells.sort(key=lambda c: -max(c["flat_final"], c["flow_final"]))
    return cells, have_any_ref


def report(cells, have_ref, args):
    gated = [c for c in cells if not c["lcc_ok"]]
    therm_gated = [c for c in cells if c["lcc_ok"] and not c.get("therm_ok", True)]
    live = [c for c in cells if c["lcc_ok"] and c.get("therm_ok", True)]
    mode = "ref-anchored (residual vs 4D torus)" if have_ref else "ABSOLUTE (UV,IR)"
    print("\n" + "=" * 104)
    print(f"DUAL-MODE 4D RANKING   [{mode}]   "
          f"({len(live)} cells pass LCC>={args.lcc_min:.0f}%, {len(gated)} gated out)")
    print("=" * 104)
    print(f"{'k':>2} {'T':>6} {'lb':>6} | {'UV':>5} {'IR':>5} {'flat?':>5} "
          f"| {'UVres':>6} {'IRres':>6} | {'FLAT':>5} {'FLOW':>5} | "
          f"{'LCC%':>5} {'trans':>5} | verdict")
    print("-" * 104)
    for c in live[:args.top]:
        tr = f"{c['trans'][-1]:.2f}" if np.isfinite(c['trans'][-1]) else "--"
        print(f"{c['k']:>2} {c['T']:>6g} {c['lb']:>6g} | "
              f"{c['uv'][-1]:>5.2f} {c['ir'][-1]:>5.2f} {c['irflat'][-1]:>5.2f} | "
              f"{c['uvres'][-1]:>6.2f} {c['irres'][-1]:>6.2f} | "
              f"{c['flat_final']:>5.0f} {c['flow_final']:>5.0f} | "
              f"{c['lcc'][-1]:>5.0f} {tr:>5} | {c['verdict']}")
    if gated:
        print(f"\n  {len(gated)} cells rejected on LCC (sampling islands), worst first:")
        for c in sorted(gated, key=lambda c: c['lcc'][-1])[:8]:
            print(f"    k{c['k']:>2} T{c['T']:g} lb{c['lb']:g}  "
                  f"LCC={c['lcc'][-1]:.0f}% (src:{c['lcc_src'][-1]})")
    if therm_gated:
        print(f"\n  {len(therm_gated)} cells rejected as unequilibrated "
              f"(build-time stationarity check failed at its x3 budget):")
        for c in therm_gated[:8]:
            print(f"    k{c['k']:>2} T{c['T']:g} lb{c['lb']:g}")
    print()


def write_csv(cells, path, args):
    cols = ["k", "T", "lb", "n_N", "N_max", "lcc_pct", "lcc_src", "therm_ok",
            "transitivity",
            "uv_shoulder", "ir_plateau", "ir_flatness", "uv_conf",
            "uv_resid_vs_4Dtorus", "ir_resid_vs_4Dtorus",
            "flat_score", "flow_score", "ir_dir", "uv_dir", "flat_dir",
            "verdict", "N_list"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            w.writerow({
                "k": c["k"], "T": c["T"], "lb": c["lb"],
                "n_N": len(c["Ns"]), "N_max": int(c["Ns"][-1]),
                "lcc_pct": f"{c['lcc'][-1]:.1f}", "lcc_src": c["lcc_src"][-1],
                "therm_ok": c.get("therm_ok", True),
                "transitivity": f"{c['trans'][-1]:.4f}",
                "uv_shoulder": f"{c['uv'][-1]:.3f}",
                "ir_plateau": f"{c['ir'][-1]:.3f}",
                "ir_flatness": f"{c['irflat'][-1]:.3f}",
                "uv_conf": c["uvconf"][-1],
                "uv_resid_vs_4Dtorus": f"{c['uvres'][-1]:.3f}",
                "ir_resid_vs_4Dtorus": f"{c['irres'][-1]:.3f}",
                "flat_score": f"{c['flat_final']:.1f}",
                "flow_score": f"{c['flow_final']:.1f}",
                "ir_dir": c["ir_dir"], "uv_dir": c["uv_dir"],
                "flat_dir": c["flat_dir"], "verdict": c["verdict"],
                "N_list": ";".join(str(int(n)) for n in c["Ns"]),
            })
    print(f"[csv] wrote {path} ({len(cells)} cells)")


def plots(cells, have_ref, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable ({e})")
        return
    live = [c for c in cells if c["lcc_ok"] and c.get("therm_ok", True)]
    gate = [c for c in cells
            if not (c["lcc_ok"] and c.get("therm_ok", True))]

    # ---- headline: the residual plane (ref mode) or absolute (UV,IR) ----
    fig, ax = plt.subplots(figsize=(8.5, 7))
    use_res = have_ref
    def xy(c):
        return (c["uvres"][-1], c["irres"][-1]) if use_res else (c["uv"][-1], c["ir"][-1])
    for c in gate:
        x, y = xy(c)
        ax.scatter(x, y, s=18, c="#bbb", alpha=0.5, marker="x")
    for c in live:
        x, y = xy(c)
        col = "#7ec96e" if c["flat_final"] >= c["flow_final"] else "#4ec9e0"
        ax.scatter(x, y, s=30 + 0.9 * max(c["flat_final"], c["flow_final"]),
                   c=col, alpha=0.8, edgecolor="none")
    if use_res:
        # flat-4D = matches torus everywhere = origin; flow = UV excess (left)
        ax.scatter(0, 0, marker="*", s=320, c="#1a9c1a", edgecolor="k", zorder=5)
        ax.annotate("flat-4D\n(matches lattice)", (0, 0), fontsize=10,
                    fontweight="bold", xytext=(8, 8), textcoords="offset points",
                    color="#1a9c1a")
        ax.axvspan(-3, UV_RESID_FLOW, color="#4ec9e0", alpha=0.07)
        ax.annotate("flowing-4D →\n(UV below lattice)", (UV_RESID_FLOW - 0.05, 0.0),
                    fontsize=10, fontweight="bold", ha="right",
                    xytext=(-4, 8), textcoords="offset points", color="#1a8ca0")
        ax.axhline(0, color="#999", ls=":", lw=.8)
        ax.axvline(0, color="#999", ls=":", lw=.8)
        ax.axvline(UV_RESID_FLOW, color="#4ec9e0", ls="--", lw=.8)
        ax.set_xlabel("UV residual vs 4D torus  (negative = excess suppression = flow)")
        ax.set_ylabel("IR residual vs 4D torus  (0 = matches 4D at large scales)")
        ax.set_xlim(-2.2, 1.2); ax.set_ylim(-2.0, 2.0)
        ttl = ("Residual plane vs 4D torus: flat-4D at the origin, "
               "flowing-4D to the left\n(IR residual ~0 = looks 4D in the IR; "
               "UV residual < 0 = dimensional reduction)")
    else:
        for tgt, lab, col in [(FLAT_TARGET, "flat-4D", "#1a9c1a"),
                              (FLOW_TARGET, "flowing-4D", "#1a8ca0")]:
            ax.scatter(*tgt, marker="*", s=320, c=col, edgecolor="k", zorder=5)
            ax.annotate(lab, tgt, fontsize=10, fontweight="bold",
                        xytext=(8, 6), textcoords="offset points", color=col)
        ax.axhline(4, color="#999", ls=":", lw=.8)
        ax.axvline(4, color="#999", ls=":", lw=.8)
        ax.axvline(2, color="#999", ls=":", lw=.8)
        ax.set_xlabel("UV shoulder (d_s at small t)")
        ax.set_ylabel("IR plateau (d_s at large t)")
        ax.set_xlim(0.8, 6.5); ax.set_ylim(0.8, 7)
        ttl = ("ABSOLUTE (UV, IR) plane (no torus ref) — WARNING: lattice "
               "discreteness pushes flat-4D UV toward ~2,\nso this mode "
               "cannot cleanly separate flat from flowing.")
    for c in live:
        if max(c["flat_final"], c["flow_final"]) >= 55:
            x, y = xy(c)
            ax.annotate(f"k{c['k']} lb{c['lb']:g} T{c['T']:g}", (x, y),
                        fontsize=6, xytext=(3, -9), textcoords="offset points")
    import time as _t
    from matplotlib.lines import Line2D
    ttl += (f"\n{len(cells)} cells · {len(live)} pass gates "
            f"(LCC≥{args.lcc_min:g}%, equilibrated) "
            f"· {len(gate)} excluded (gray ×) · updated {_t.strftime('%H:%M:%S')}")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#7ec96e",
               markeredgecolor="none", markersize=8,
               label="flat-4D leaning (LCC ≥ gate)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#4ec9e0",
               markeredgecolor="none", markersize=8,
               label="flowing-4D leaning (LCC ≥ gate)"),
        Line2D([0], [0], marker="x", linestyle="none", color="#bbb", markersize=7,
               label=f"LCC < {args.lcc_min:g}% or unequilibrated — excluded"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#1a9c1a",
               markeredgecolor="k", markersize=13,
               label="ideal target (marker size ∝ score)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(ttl, fontsize=10)
    fig.tight_layout(); fig.savefig("flow_modes_plane.png", dpi=150); plt.close(fig)
    print("[plots] wrote flow_modes_plane.png")

    # ---- (k x lb) per T maps for each mode score ----
    Ts = sorted({c["T"] for c in cells})
    ks = sorted({c["k"] for c in cells})
    lbs = sorted({c["lb"] for c in cells})
    by = {(c["T"], c["k"], c["lb"]): c for c in cells}
    fig, axes = plt.subplots(2, len(Ts), figsize=(2.7 * len(Ts) + 1.2, 7.5),
                             squeeze=False)
    for row, (field, name) in enumerate([("flat_final", "FLAT-4D score"),
                                          ("flow_final", "FLOWING-4D score")]):
        for ci, T in enumerate(Ts):
            ax = axes[row][ci]
            Z = np.full((len(ks), len(lbs)), np.nan)
            for i, k in enumerate(ks):
                for j, lb in enumerate(lbs):
                    c = by.get((T, k, lb))
                    if c:
                        Z[i, j] = c[field]
            im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                           vmin=0, vmax=100,
                           extent=[-.5, len(lbs)-.5, -.5, len(ks)-.5])
            for i, k in enumerate(ks):
                for j, lb in enumerate(lbs):
                    c = by.get((T, k, lb))
                    if not c:
                        continue
                    v = c[field]
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=5, color="white" if v < 55 else "black")
            ax.set_xticks(range(len(lbs)))
            ax.set_xticklabels([f"{x:g}" for x in lbs], rotation=45, fontsize=6)
            ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=6)
            if ci == 0:
                ax.set_ylabel(f"{name}\nk", fontsize=8)
            if row == 0:
                ax.set_title(f"T = {T:g}", fontsize=10)
            if row == 1:
                ax.set_xlabel("lb")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    import time as _t
    fig.suptitle("Mode scores across (k, lb) per T  (top row flat-4D, "
                 f"bottom row flowing-4D)\n{len(cells)} cells · "
                 f"{len(live)} pass LCC≥{args.lcc_min:g}% · "
                 f"updated {_t.strftime('%H:%M:%S')}", fontsize=11)
    fig.savefig("flow_modes_maps.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[plots] wrote flow_modes_maps.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="flow_modes.csv")
    ap.add_argument("--lcc-min", type=float, default=LCC_MIN_DEFAULT,
                    help="reject cells whose LCC%% is below this (default 90)")
    ap.add_argument("--k", type=int, nargs="+", default=None)
    ap.add_argument("--T", type=float, nargs="+", default=None)
    ap.add_argument("--lb-min", type=float, default=0.0)
    ap.add_argument("--lb-max", type=float, default=1.0)
    ap.add_argument("--N-min", type=int, default=16000)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    paths = find_files(args.dir)
    if not paths:
        print(f"[scan] no flow_*.csv under {args.dir!r}")
        return
    cells, have_ref = build(paths, args)
    if not cells:
        print("[main] no usable cells after filtering.")
        return
    report(cells, have_ref, args)
    write_csv(cells, args.out, args)
    if not args.no_plots:
        plots(cells, have_ref, args)


if __name__ == "__main__":
    main()
