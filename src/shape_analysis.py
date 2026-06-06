#!/usr/bin/env python3
"""
shape_analysis.py — answer the integer-quantization question

Reads existing flow_*.csv files, extracts a per-cell "local effective
dimension" from the dip floor of d_s(t), and asks: does it cluster at
integers (= phase diagram of discrete dimensional regimes) or vary
continuously (= one-knob tunable continuous d)?

Outputs three things:
  1. shape_summary.csv  — per-cell row with (k, T, lb, N, seed,
        d_s_local, dip_t, has_dip, n_peaks, …)
  2. shape_histogram.png  — distribution of d_s_local across all cells.
        Multimodal at integers ⇒ quantized. Smooth ⇒ continuous.
  3. shape_heatmap.png  — d_s_local as 2D heatmap over (k, lb), one
        panel per N. Plateau-ish iso-color regions ⇒ phases.

Usage
-----
    NOTE: `python main.py` launches only the dashboard + auto-sweep —
    there is no `main.py shape` subcommand. The sweep calls this module's
    main() directly (shape_analysis.main(["--dir", "."])) to refresh the
    heatmaps; the examples below show that interface.

    python main.py shape                           # uses ./flow/
    python main.py shape --dir /data2/28/new
    python main.py shape --N 1024000 4096000       # restrict N

The dip is found with two robust criteria, not just argmin:
  (a) d_s(t) has at least one local maximum to its left (rules out
      monotone-falling chains where "dip" is just the start)
  (b) d_s(t) climbs back at least Δ_climb above the floor afterwards
      (rules out finite-size cutoff being misread as a dip)
If either fails, has_dip = False and d_s_local falls back to a window-
average over t ∈ [3, 30] so every cell still gets a number.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RE_CELL = re.compile(
    r"^flow_k(\d+)_T([\d.eE+-]+)_lb([\d.eE+-]+)_N(\d+)_s(\d+)\.csv$")


# ─── load one flow CSV ────────────────────────────────────────────────
def load_flow(path):
    """Returns (t, d_s, in_window) as numpy arrays."""
    t, d, w = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]))
            d.append(float(r["d_s_mean"]))
            w.append(bool(int(r["in_window"])))
    return np.array(t), np.array(d), np.array(w, dtype=bool)


# ─── shape extraction ─────────────────────────────────────────────────
def extract_shape(t, d, in_w, climb_threshold=0.3,
                  min_drop_from_peak=0.4,
                  fallback_window=(3.0, 30.0)):
    """Compute (d_s_local, dip_t, has_dip, n_peaks) from one curve.

    Strategy:
      • Restrict to in-window points (avoids small-t lattice spike and
        large-t finite-size climb past mixing time).
      • Smooth the curve lightly (3-point moving average in log-t) so
        local-noise wiggles don't get treated as extrema.
      • Find the *first* prominent peak (local max). The "dip" is then
        the minimum of d_s in the t-range AFTER that peak.
      • Three checks for has_dip:
          1. There's a real peak earlier (d_peak − d > min_drop_from_peak)
          2. The minimum is strictly inside the post-peak segment
             (not at the right boundary, where it'd be cutoff artifact)
          3. The curve climbs back at least climb_threshold afterward
      • d_s_local = d_s at the dip; dip_t = its t-coordinate.
      • If any check fails, fall back to mean over the fallback window
        so every cell still produces a comparable number.
    """
    valid = in_w & np.isfinite(d)
    if valid.sum() < 8:
        return float("nan"), float("nan"), False, 0

    tw = t[valid]
    dw = d[valid]

    # 3-point moving average smoothing in log-t. Edge values copied.
    ds = dw.copy()
    ds[1:-1] = (dw[:-2] + dw[1:-1] + dw[2:]) / 3.0

    # Find local extrema by sign-change of the discrete derivative
    # (in log-t since t is log-spaced).
    log_t = np.log(tw)
    dd = np.gradient(ds, log_t, edge_order=2)

    n_peaks = 0
    peaks_idx = []
    mins_idx = []
    for i in range(1, len(dd) - 1):
        if dd[i - 1] > 0 and dd[i + 1] < 0:
            peaks_idx.append(i)
            n_peaks += 1
        elif dd[i - 1] < 0 and dd[i + 1] > 0:
            mins_idx.append(i)

    has_dip = False
    dip_idx = None

    if peaks_idx:
        # consider the FIRST peak; the dip is the minimum after it
        first_peak = peaks_idx[0]
        d_peak_val = ds[first_peak]
        # candidate mins are those strictly after the first peak AND
        # strictly before the last in-window index (otherwise it's a
        # boundary artifact, not a real interior dip)
        candidates = [j for j in mins_idx
                      if first_peak < j < len(ds) - 2]
        if candidates:
            # pick the minimum-value one (deepest dip)
            j_best = min(candidates, key=lambda j: ds[j])
            d_min = ds[j_best]
            # check 1: drop below the preceding peak is meaningful
            drop_ok = (d_peak_val - d_min) >= min_drop_from_peak
            # check 3: post-dip climb is meaningful
            post_climb = ds[j_best:].max() - d_min
            climb_ok = post_climb >= climb_threshold
            if drop_ok and climb_ok:
                has_dip = True
                dip_idx = j_best

    if has_dip:
        d_s_local = float(ds[dip_idx])
        dip_t = float(tw[dip_idx])
    else:
        # fallback: window-average over [3, 30]
        m = (tw >= fallback_window[0]) & (tw <= fallback_window[1])
        if m.sum() >= 3:
            d_s_local = float(np.nanmean(ds[m]))
        else:
            d_s_local = float(np.nanmean(ds))
        dip_t = float("nan")

    return d_s_local, dip_t, has_dip, int(n_peaks)


# ─── scan all flow files in dir ───────────────────────────────────────
def scan(dir_):
    """Yield per-cell records by reading every flow_*.csv in the dir."""
    patterns = [os.path.join(dir_, "flow", "flow_k*_*.csv"),
                os.path.join(dir_, "flow_k*_*.csv")]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    print(f"[scan] {len(paths)} cell flow files in {dir_}")

    for p in paths:
        m = RE_CELL.match(os.path.basename(p))
        if not m:
            continue
        k, T, lb, N, seed = (int(m.group(1)), float(m.group(2)),
                             float(m.group(3)), int(m.group(4)),
                             int(m.group(5)))
        try:
            t, d, w = load_flow(p)
            d_s_local, dip_t, has_dip, n_peaks = extract_shape(t, d, w)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        yield {
            "k": k, "T": T, "lb": lb, "N": N, "seed": seed,
            "d_s_local": d_s_local, "dip_t": dip_t,
            "has_dip": has_dip, "n_peaks": n_peaks,
        }


# ─── failure records (why a cell has no flow CSV) ─────────────────────
def load_failures(dir_):
    """Collect *why* cells produced no flow_*.csv, so the heatmap can mark
    them instead of showing an ambiguous gap (`×` currently means both
    "not yet run" and "failed"). Two sources, both written by the sweep:

      • fail_<tag>.json sidecars (in flow/ or the dir root) — a worker raised
        while building/measuring a cell, keyed per (k, T, lb, N). reason is
        'max_degree' when the graph's k_top hit the engine cap MAX_DEG (the
        Markov chain stopped sampling H), else a generic error.
      • mu_failures.json (dir root) — (k, T, lb) tuples whose μ-calibration
        failed, so EVERY N for them was dropped before it could run.

    Returns a dict of sets:
        cap_cell / err_cell : {(k, T, lb, N)}  from the per-cell sidecars
        cap_ktl / err_ktl   : {(k, T, lb)}     from the μ-calibration ledger
    'cap' takes precedence over a generic error if a cell is flagged both.
    """
    cap_cell, err_cell, cap_ktl, err_ktl = set(), set(), set(), set()
    for pat in (os.path.join(dir_, "flow", "fail_*.json"),
                os.path.join(dir_, "fail_*.json")):
        for p in glob.glob(pat):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if r.get("kind") != "cell":
                continue
            try:
                key = (int(r["k"]), float(r["T"]),
                       float(r["lb"]), int(r["N"]))
            except (KeyError, ValueError, TypeError):
                continue
            (cap_cell if r.get("reason") == "max_degree"
             else err_cell).add(key)
    mp = os.path.join(dir_, "mu_failures.json")
    if os.path.exists(mp):
        try:
            for v in json.load(open(mp)).values():
                try:
                    ktl = (int(v["k"]), float(v["T"]), float(v["lb"]))
                except (KeyError, ValueError, TypeError):
                    continue
                (cap_ktl if v.get("reason") == "max_degree"
                 else err_ktl).add(ktl)
        except Exception:
            pass
    err_cell -= cap_cell
    err_ktl -= cap_ktl
    return {"cap_cell": cap_cell, "err_cell": err_cell,
            "cap_ktl": cap_ktl, "err_ktl": err_ktl}


# ─── plots ────────────────────────────────────────────────────────────
def plot_histogram(records, out, restrict_N=None):
    """Distribution of d_s_local. Quantized → multimodal at integers."""
    records = [r for r in records if np.isfinite(r["d_s_local"])]
    if restrict_N:
        records = [r for r in records if r["N"] in restrict_N]
    if not records:
        print(f"[hist] no data for N restriction {restrict_N}")
        return

    d_s_local_all = np.array([r["d_s_local"] for r in records])
    d_with = np.array([r["d_s_local"] for r in records if r["has_dip"]])
    d_without = np.array([r["d_s_local"] for r in records if not r["has_dip"]])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(0.5, 10.05, 0.1)
    if len(d_with):
        ax.hist(d_with, bins=bins, alpha=0.75, label=f"has dip (n={len(d_with)})",
                color="#7ec96e", edgecolor="white", linewidth=0.3)
    if len(d_without):
        ax.hist(d_without, bins=bins, alpha=0.55,
                label=f"fallback avg (n={len(d_without)})",
                color="#c96e6e", edgecolor="white", linewidth=0.3)

    for d_int in range(1, 8):
        ax.axvline(d_int, color="#888", ls=":", lw=0.8, alpha=0.6,
                   zorder=0)

    n_str = (f"N ∈ {sorted(restrict_N)}" if restrict_N
             else "all N pooled")
    ax.set_xlabel("spectral dimension d_s (dip floor, or fallback window-average)")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of local spectral dimension d_s across "
                 f"({len(records)} cells, {n_str})\n"
                 f"Multimodal peaks at integers ⇒ quantized phases.  "
                 f"Smooth ⇒ continuous tuning.", fontsize=11)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[hist] saved → {out}")

    # also report histogram-derived diagnostic
    if len(d_s_local_all) > 20:
        # crude quantization metric: fraction of mass within ±0.2 of any integer
        near_int_mask = np.zeros_like(d_s_local_all, dtype=bool)
        for d_int in range(1, 8):
            near_int_mask |= np.abs(d_s_local_all - d_int) < 0.2
        frac = near_int_mask.mean()
        print(f"[hist] {100*frac:.1f}% of cells within ±0.2 of an integer")
        if frac > 0.7:
            print("       → suggestive of quantization")
        elif frac < 0.4:
            print("       → suggestive of continuous tuning")
        else:
            print("       → ambiguous; inspect histogram visually")


def _cell_stat(finite_vals, metric):
    """Reduce a list of per-seed d_s values for one (T,k,ℓ,N) cell to a
    single scalar + a 'kind' tag that decides how the cell is drawn.

    Returns (value, kind):
      kind="value"        -> value is the number to colour/annotate
      kind="single_seed"  -> only one finite seed (σ undefined); value is
                             that lone d_s, shown faint for context
      kind="no_value"     -> measured but no finite seed (·)
    The 'no_file' case (×) is decided by the caller from CSV presence.
    """
    if not finite_vals:
        return (float("nan"), "no_value")
    if metric == "mean":
        return (float(np.mean(finite_vals)), "value")
    # seed_std: sample std (ddof=1) needs ≥2 finite seeds
    if len(finite_vals) >= 2:
        return (float(np.std(finite_vals, ddof=1)), "value")
    return (float(finite_vals[0]), "single_seed")


def plot_heatmap(records, out, restrict_N=None, metric="mean", failures=None):
    """Heatmap over (k, ℓ) as a grid of panels: one ROW per temperature T
    and one COLUMN per N, so all four swept parameters appear in a single
    image. Within each panel the axes are k (vertical) × ℓ (horizontal);
    N runs across the columns, T down the rows (labelled at the left
    margin). All panels share one (k × ℓ) grid (the union of every k and ℓ
    in the run) and one colour scale, so panels are directly comparable
    and a cell missing in one panel shows as a gap *there*.

    `metric` selects what each cell shows, both reduced over the seeds
    present for that cell:
      "mean"     -> mean d_s  (the value map; colour scale 1–7, turbo)
      "seed_std" -> seed-to-seed sample std of d_s (the spread map; the
                    disorder signal that a single seed can't reveal;
                    sequential scale 0–max, magma)

    Gaps are drawn distinctly so a blank is never a mystery:
      ·  measured but the d_s(t) curve had <8 in-window points, so no
         value could be read (common at extreme ℓ and at k=2, a near-1D
         chain);
      ×  no data file AND no recorded failure — i.e. genuinely not yet run.
      K  (red) the cell FAILED because its k_top hit the engine cap MAX_DEG
         — the Markov chain stopped sampling the Hamiltonian, so no data.
         Read from the sweep's fail_*.json / mu_failures.json (pass
         `failures` from load_failures()).
      !  (orange) the cell failed for some other reason (see the sweep log).
    In the spread map only, a third mark appears:
      n1 a single faint value — only one seed was readable, so no spread
         can be estimated for that cell.

    NOTE: a `K`/`!` only appears where that (T, k, ℓ) slot exists on the
    shared axes — i.e. at least one *other* cell at that T/k/ℓ produced data.
    A temperature row in which every cell failed has no axis row to mark; the
    sweep log still lists it.
    """
    spread = (metric == "seed_std")
    failures = failures or {}

    def fail_kind(k, T, lb, N):
        """cap / fail / None for a cell that has no data file."""
        if ((k, T, lb, N) in failures.get("cap_cell", ())
                or (k, T, lb) in failures.get("cap_ktl", ())):
            return "cap"
        if ((k, T, lb, N) in failures.get("err_cell", ())
                or (k, T, lb) in failures.get("err_ktl", ())):
            return "fail"
        return None

    # NB: keep non-finite records — needed to tell "measured but
    # unreadable" (·) apart from "never measured" (×).
    if restrict_N:
        records = [r for r in records if r["N"] in restrict_N]
    if not records:
        print(f"[map] no data for N restriction {restrict_N}")
        return

    Ns = sorted(set(r["N"] for r in records))
    Ts = sorted(set(r["T"] for r in records))
    all_ks = sorted(set(r["k"] for r in records))
    all_lbs = sorted(set(r["lb"] for r in records))

    # First pass: reduce every (T,N) panel to a (k×ℓ) value grid + a
    # parallel 'kind' grid, and (for the spread map) find the shared vmax.
    panels = {}          # (ri, ci) -> (Z values, kinds, present-set)
    max_spread = 0.0
    for ri, T in enumerate(Ts):
        for ci, N in enumerate(Ns):
            cell_recs = [r for r in records
                         if r["T"] == T and r["N"] == N]
            finite_vals = defaultdict(list)
            present = set()
            for r in cell_recs:
                present.add((r["k"], r["lb"]))
                if np.isfinite(r["d_s_local"]):
                    finite_vals[(r["k"], r["lb"])].append(r["d_s_local"])
            Z = np.full((len(all_ks), len(all_lbs)), np.nan)
            kinds = np.empty((len(all_ks), len(all_lbs)), dtype=object)
            for i, k in enumerate(all_ks):
                for j, lb in enumerate(all_lbs):
                    if (k, lb) in finite_vals or (k, lb) in present:
                        val, kind = _cell_stat(finite_vals.get((k, lb), []),
                                               metric)
                    else:
                        fk = fail_kind(k, T, lb, N)
                        val, kind = float("nan"), (fk if fk else "no_file")
                    Z[i, j] = val
                    kinds[i, j] = kind
                    if spread and kind == "value":
                        max_spread = max(max_spread, val)
            panels[(ri, ci)] = (Z, kinds, present)

    if spread:
        n_seeds_seen = len(set(r["seed"] for r in records))
        if n_seeds_seen < 2:
            print("[map] only one seed in the data — no spread to show; "
                  "skipping seed-std map.")
            return
        vmin, vmax, cmap = 0.0, max(0.05, max_spread), "magma"
    else:
        vmin, vmax, cmap = 1.0, 7.0, "turbo"
    print(f"[map] {len(Ts)} T × {len(Ns)} N grid "
          f"(T = {[f'{t:g}' for t in Ts]}); metric={metric}")

    n_rows, n_cols = len(Ts), len(Ns)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.55 * n_cols + 1.4, 2.35 * n_rows + 1.2),
        squeeze=False)

    n_no_value = n_no_file = n_single = 0
    n_cap = n_err = 0
    im = None
    for ri, T in enumerate(Ts):
        for ci, N in enumerate(Ns):
            ax = axes[ri][ci]
            Z, kinds, present = panels[(ri, ci)]
            # only "value" cells are coloured; everything else is a glyph
            Zc = np.where(np.vectorize(lambda s: s == "value")(kinds),
                          Z, np.nan)
            im = ax.imshow(Zc, origin="lower", aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[-0.5, len(all_lbs) - 0.5,
                                   -0.5, len(all_ks) - 0.5])

            ax.set_xticks(range(len(all_lbs)))
            ax.set_yticks(range(len(all_ks)))
            if ri == n_rows - 1:
                ax.set_xticklabels([f"{lb:g}" for lb in all_lbs],
                                   rotation=40, fontsize=7)
                ax.set_xlabel("ℓ", fontsize=8)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_yticklabels([str(k) for k in all_ks], fontsize=7)
                ax.set_ylabel("k", fontsize=8)
                ax.annotate(f"T = {T:g}",
                            xy=(0, 0.5), xycoords=ax.yaxis.label,
                            xytext=(-ax.yaxis.labelpad - 16, 0),
                            textcoords="offset points",
                            ha="right", va="center", rotation=90,
                            fontsize=10, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f"N = {N:,}", fontsize=9)

            for i in range(len(all_ks)):
                for j in range(len(all_lbs)):
                    kind = kinds[i, j]
                    if kind == "value":
                        v = Z[i, j]
                        if spread:
                            txt = f"{v:.2f}"
                            dark = (v / vmax) > 0.55  # magma: bright→dark txt
                            color = "black" if dark else "white"
                        else:
                            txt = f"{v:.1f}"
                            color = ("white" if v < 3.5 or v > 5.5
                                     else "black")
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=6, color=color)
                    elif kind == "single_seed":
                        # one readable seed: no spread estimate. Faint fill
                        # (not the gap hatch) + a small marker.
                        n_single += 1
                        ax.add_patch(mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1.0, 1.0,
                            facecolor="#f4f4f4", edgecolor="#dcdcdc",
                            linewidth=0.0, zorder=1.5))
                        ax.text(j, i, "n1", ha="center", va="center",
                                fontsize=6, color="#9a9a9a", zorder=2)
                    elif kind == "cap":
                        # k_top hit the engine cap -> the cell failed. Solid
                        # red fill + bold "K" so it can't be mistaken for a
                        # not-yet-run gap.
                        n_cap += 1
                        ax.add_patch(mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1.0, 1.0,
                            facecolor="#f7d4d4", edgecolor="#d23b3b",
                            linewidth=0.8, zorder=1.5))
                        ax.text(j, i, "K", ha="center", va="center",
                                fontsize=7, fontweight="bold",
                                color="#b11d1d", zorder=2)
                    elif kind == "fail":
                        # failed for some other reason (see sweep log).
                        n_err += 1
                        ax.add_patch(mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1.0, 1.0,
                            facecolor="#fde6cc", edgecolor="#e08a1e",
                            linewidth=0.8, zorder=1.5))
                        ax.text(j, i, "!", ha="center", va="center",
                                fontsize=7, fontweight="bold",
                                color="#b5650f", zorder=2)
                    else:
                        if kind == "no_value":
                            n_no_value += 1
                            glyph = "·"
                        else:
                            n_no_file += 1
                            glyph = "×"
                        ax.add_patch(mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1.0, 1.0,
                            facecolor="#ededed", edgecolor="#c9c9c9",
                            hatch="///", linewidth=0.0, zorder=1.5))
                        ax.text(j, i, glyph, ha="center", va="center",
                                fontsize=7, color="#7a7a7a", zorder=2)

    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(),
                          fraction=0.025, pad=0.02)
        cb.set_label("seed-to-seed std of d_s" if spread
                     else "spectral dimension d_s")

    if spread:
        head = ("Seed-to-seed spread of d_s across (k, ℓ)  "
                "— the disorder variance a single seed can't see")
        cellline = "cell = sample std of d_s over seeds (ddof=1)"
        extra = "    n1  only one readable seed (no spread)"
    else:
        head = "Local spectral dimension d_s across (k, ℓ)"
        cellline = "cell = mean d_s over seeds present"
        extra = ""
    import time as _t
    _stamp = _t.strftime("%Y-%m-%d %H:%M:%S")
    _nseeds = len(set(r["seed"] for r in records))
    fig.suptitle(
        f"{head}\n"
        f"rows = T (temperature)   columns = N   "
        f"panel axes: k (vertical) × ℓ (horizontal)   {cellline}\n"
        f"gaps:  ·  <8 in-window points    ×  not yet run    "
        f"K  failed: k_top hit engine cap    !  failed: other{extra}",
        fontsize=10)
    # Count/timestamp as a bottom caption (a 4th suptitle line collides with the
    # top row's per-panel "N=" titles).
    fig.text(0.5, 0.005,
             f"{len(records)} cell-records · {_nseeds} seed(s) present "
             f"· updated {_stamp}",
             ha="center", va="bottom", fontsize=8, color="#666")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    tail = f", {n_single} single-seed" if spread else ""
    cf = f", {n_cap} k-cap fail, {n_err} other fail" if (n_cap or n_err) else ""
    print(f"[map] saved → {out}  "
          f"(gaps: {n_no_value} with no readable value, "
          f"{n_no_file} not yet run{cf}{tail})")


# ─── main ─────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                    help="Directory containing flow/ subdir or flow_*.csv files")
    ap.add_argument("--N", nargs="+", type=int, default=None,
                    help="Restrict histogram to these N values "
                         "(default: all)")
    ap.add_argument("--summary-out", default="shape_summary.csv")
    ap.add_argument("--hist-out", default="shape_histogram.png")
    ap.add_argument("--map-out", default="shape_heatmap.png")
    ap.add_argument("--seed-std-out", default="shape_heatmap_seed_std.png",
                    help="Output for the seed-to-seed spread map. Written "
                         "automatically when the data has >1 seed per cell.")
    args = ap.parse_args(argv)

    records = list(scan(args.dir))
    if not records:
        print("[main] no cell records found. Check --dir.")
        return

    # write the long table
    keys = ["k", "T", "lb", "N", "seed", "d_s_local", "dip_t",
            "has_dip", "n_peaks"]
    with open(args.summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in keys})
    print(f"[csv] saved → {args.summary_out}  ({len(records)} rows)")

    # quick console summary by N
    by_N = defaultdict(list)
    for r in records:
        by_N[r["N"]].append(r)
    print("\n[summary] per-N stats:")
    print(f"  {'N':>10}  {'cells':>6}  {'has_dip':>9}  {'<d_s_local>':>10}")
    for N in sorted(by_N):
        recs = by_N[N]
        n_dip = sum(1 for r in recs if r["has_dip"])
        d_mean = np.nanmean([r["d_s_local"] for r in recs])
        print(f"  {N:>10,}  {len(recs):>6}  {n_dip:>4}/{len(recs):<4} "
              f"{d_mean:>10.2f}")
    print()

    # plots
    if args.N:
        plot_histogram(records, args.hist_out, restrict_N=set(args.N))
    else:
        plot_histogram(records, args.hist_out)

    restrict = set(args.N) if args.N else None
    # Why-it's-blank records, so the heatmap can mark failed cells (k-cap vs
    # other) distinctly from not-yet-run ones.
    failures = load_failures(args.dir)
    plot_heatmap(records, args.map_out, restrict_N=restrict, metric="mean",
                 failures=failures)
    # The spread map self-skips (with a note) when there is only one seed,
    # so this is safe to always call.
    plot_heatmap(records, args.seed_std_out, restrict_N=restrict,
                 metric="seed_std", failures=failures)


if __name__ == "__main__":
    main()
