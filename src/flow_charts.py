#!/usr/bin/env python3
"""flow_charts.py — overview charts from flow_convergence.csv

Reads the summary CSV (NOT the raw flow_*.csv) and draws the two views
that make it readable:

  flow_map.png      (k x lb) grid, one panel per T, coloured by the
                    extrapolated d_inf (diverging around 4). Cells whose
                    plateau is NOT actually flat are hatched out; genuine
                    near-4 flat cells get a green box. This is the "where
                    is 4D" map, in the same layout as the d_s heatmap.

  flow_scatter.png  every cell as a point: x = d_inf, y = flatness
                    (in-plateau spread of d_s). The 4D target box is
                    d_inf in [3.7,4.3] AND flat. Points high on the y
                    axis are sloping windows masquerading as ~4.

KEY: "flatness" = sqrt(dev4^2 - (value-4)^2) = the std of d_s across the
plateau window. The CSV's `wiggle` column DETRENDS first, so a steep
smooth slope reads as ~0 wiggle even though the window isn't flat at all.
This recovers the slope the detrend hid. Small = real plateau.
"""
import argparse, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

FLAT_TOL = 0.6      # in-plateau std below this = "flat enough"
NEAR_TOL = 0.30     # |d_inf - 4| below this = "near 4"

def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        def g(k):
            try: return float(r[k])
            except: return float("nan")
        d_inf, val, dev4 = g("d_inf"), g("value_at_maxN"), g("dev4_at_maxN")
        flat = math.sqrt(max(dev4**2 - (val-4)**2, 0.0))   # in-plateau std
        ci = g("d_inf_ci")
        rows.append(dict(k=int(float(r["k"])), T=float(r["T"]),
                         lb=float(r["lb"]), d_inf=d_inf, ci=ci, val=val,
                         dev4=dev4, flat=flat,
                         therm_ok=(r.get("therm_ok", "True") != "False"),
                         is_flat=flat < FLAT_TOL,
                         near4=abs(val-4) <= NEAR_TOL,
                         trust=(np.isfinite(ci) and ci < 0.4
                                and np.isfinite(d_inf) and abs(d_inf-val) < 0.5),
                         nmax=int(float(r["N_max"])),
                         seedsd_dir=r.get("seedstd_dir",""),
                         verdict=r.get("verdict","")))
    return rows

def map_fig(rows, out):
    Ts  = sorted({r["T"]  for r in rows})
    ks  = sorted({r["k"]  for r in rows})
    lbs = sorted({r["lb"] for r in rows})
    by = {(r["T"], r["k"], r["lb"]): r for r in rows}
    norm = TwoSlopeNorm(vmin=1.5, vcenter=4.0, vmax=6.5)
    fig, axes = plt.subplots(1, len(Ts), figsize=(3.0*len(Ts)+1.2, 4.2),
                             squeeze=False)
    im = None
    for ci, T in enumerate(Ts):
        ax = axes[0][ci]
        Z = np.full((len(ks), len(lbs)), np.nan)
        for i, k in enumerate(ks):
            for j, lb in enumerate(lbs):
                r = by.get((T, k, lb))
                if r: Z[i, j] = r["val"]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r",
                       norm=norm, extent=[-.5, len(lbs)-.5, -.5, len(ks)-.5])
        for i, k in enumerate(ks):
            for j, lb in enumerate(lbs):
                r = by.get((T, k, lb))
                if not r: continue
                ax.text(j, i, f"{r['val']:.1f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(r['val']-4) > 1.4 else "black")
                if not r["therm_ok"]:       # build never equilibrated ->
                    ax.add_patch(mpatches.Rectangle(   # gated, same as
                        (j-.5, i-.5), 1, 1, fill=False, hatch="////",   # flow_modes
                        edgecolor="#bb000077", linewidth=0))
                elif not r["is_flat"]:      # not a real plateau -> hatch out
                    ax.add_patch(mpatches.Rectangle(
                        (j-.5, i-.5), 1, 1, fill=False, hatch="xxx",
                        edgecolor="#33333355", linewidth=0))
                elif r["near4"]:            # flat AND ~4 -> candidate box
                    ax.add_patch(mpatches.Rectangle(
                        (j-.5, i-.5), 1, 1, fill=False,
                        edgecolor="#00d000", linewidth=2.2))
        ax.set_xticks(range(len(lbs)))
        ax.set_xticklabels([f"{x:g}" for x in lbs], rotation=45, fontsize=6)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=6)
        ax.set_xlabel("lb"); ax.set_title(f"T = {T:g}", fontsize=10)
        if ci == 0: ax.set_ylabel("k")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label("d_s at largest N  (white = 4)")
    fig.suptitle("4D map — colour = d_inf;  green box = flat & near 4 & equilibrated;  "
                 "red //// = not equilibrated (gated);  "
                 "xxx = plateau NOT flat (sloping window, ignore the value)",
                 fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)

def scatter_fig(rows, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    Ts = sorted({r["T"] for r in rows})
    cmap = {T: c for T, c in zip(Ts, ["#4ec9b0","#9cdcfe","#dcdcaa","#f48771"])}
    for r in rows:
        ax.scatter(r["val"], r["flat"], s=22+8*(r["nmax"]>1_500_000),
                   c=cmap[r["T"]], alpha=0.7, edgecolor="none")
    # target box
    ax.add_patch(mpatches.Rectangle((3.7, 0), 0.6, FLAT_TOL, fill=True,
                 color="#00d000", alpha=0.10))
    ax.add_patch(mpatches.Rectangle((3.7, 0), 0.6, FLAT_TOL, fill=False,
                 edgecolor="#00a000", lw=1.5, ls="--"))
    ax.axvline(4, color="#888", lw=0.8, ls=":")
    for r in rows:
        if 3.7 <= r["val"] <= 4.3 and r["flat"] < FLAT_TOL:
            ax.annotate(f"k{r['k']} lb{r['lb']:g} T{r['T']:g}",
                        (r["val"], r["flat"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("d_s at largest N  (measured, not extrapolated)")
    ax.set_ylabel("flatness = in-plateau std of d_s   (low = real plateau)")
    ax.set_xlim(1, 7); ax.set_ylim(-0.05, 3.2)
    handles = [mpatches.Patch(color=cmap[T], label=f"T={T:g}") for T in Ts]
    handles.append(mpatches.Patch(color="#00d000", alpha=0.3, label="4D target box"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_title("Every cell: is the plateau flat (low y) AND at 4 (x≈4)?\n"
                 "Cells near x=4 but high y are sloping windows whose mean "
                 "merely crosses 4 — not 4D.", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)

def leaderboard(rows, n=15):
    cand = [r for r in rows if r["near4"] and r["is_flat"] and r["trust"]]
    cand.sort(key=lambda r: (abs(r["val"]-4), r["flat"]))
    print(f"\nTRUST-GATED candidates (flat, near 4, extrap agrees), best first "
          f"[{len(cand)}]:")
    print(f"  {'k':>2} {'T':>6} {'lb':>6} {'val@N':>6} {'d_inf':>6} {'ci':>5} "
          f"{'flat':>5} {'Nmax':>9} seeds")
    for r in cand[:n]:
        print(f"  {r['k']:>2} {r['T']:>6g} {r['lb']:>6g} {r['val']:>6.2f} "
              f"{r['d_inf']:>6.2f} {r['ci']:>5.2f} {r['flat']:>5.2f} "
              f"{r['nmax']:>9,} {r['seedsd_dir']}")
    near_not_flat = sum(1 for r in rows if abs(r['val']-4)<=0.3 and not r['is_flat'])
    print(f"\n  ({near_not_flat} cells sit near 4 but are NOT flat — sloping "
          f"windows the score column mislabels as 4D.)")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="flow_convergence.csv")
    a = ap.parse_args(argv)
    rows = load(a.csv)
    print(f"loaded {len(rows)} cells")
    map_fig(rows, "flow_map.png")
    scatter_fig(rows, "flow_scatter.png")
    leaderboard(rows)

if __name__ == "__main__":
    main()
