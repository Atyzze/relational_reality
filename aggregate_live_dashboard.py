#!/usr/bin/env python3
"""
aggregate_live_dashboard.py (p_triadic overlay)

Continuously scans a target directory for all `sweep_summary_append.csv` files,
aggregates the data, and renders a single dashboard.

UPDATED:
- Colors are now keyed by `p_triadic` (so line/color corresponds to p_triadic).
- If multiple node-counts (N) exist, markers vary by N while keeping the same color per p_triadic.
- Legend shows p_triadic values explicitly.

Usage:
    python aggregate_live_dashboard.py --sweep-dir mu_sweep --interval 1.0
"""

import argparse
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

def try_fit_critical_slowing(mus, Ts):
    """Fit T ~ a / (mu - muc)^gamma for COMPLETED points."""
    mus = np.asarray(mus, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    ok = np.isfinite(mus) & np.isfinite(Ts) & (Ts > 0)
    mus, Ts = mus[ok], Ts[ok]
    if len(mus) < 6:
        return None

    mu_min = float(np.min(mus))
    mu_max = float(np.max(mus))
    span = max(1e-6, (mu_max - mu_min))
    muc_candidates = np.linspace(mu_min - 0.5 * span, mu_min - 1e-6, 60)

    best = None
    best_sse = None

    for muc in muc_candidates:
        x = mus - muc
        if np.any(x <= 0):
            continue
        lx = np.log(x)
        ly = np.log(Ts)

        A = np.vstack([np.ones_like(lx), lx]).T
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        b0, b1 = coef
        pred = A @ coef
        sse = float(np.sum((ly - pred) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            gamma = -b1
            a = float(np.exp(b0))
            best = (float(muc), float(gamma), float(a))

    return best

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return np.nan

def safe_int(val):
    try:
        return int(float(val))
    except Exception:
        return -1

def fmt_p(p):
    if not np.isfinite(p):
        return "nan"
    # compact formatting, stable in legend
    if abs(p) < 1e-12:
        return "0"
    return f"{p:.6g}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, default="mu_sweep", help="Root directory containing batch folders")
    ap.add_argument("--out-png", type=str, default="live.png", help="Output PNG path")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_png = Path(args.out_png)

    print(f"Starting live dashboard aggregator.")
    print(f"Scanning: {sweep_dir.absolute()}/**/*.csv")
    print(f"Outputting to: {out_png.absolute()}")
    print(f"Updating every {args.interval} seconds. Press Ctrl+C to stop.")

    last_mtimes = {}

    # Palette for p_triadic (colors)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    # Marker set for different N (optional visual differentiation)
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]

    while True:
        try:
            csv_files = list(sweep_dir.rglob("sweep_summary_append.csv"))

            needs_update = False
            current_mtimes = {}
            for f in csv_files:
                mtime = f.stat().st_mtime
                current_mtimes[f] = mtime
                if f not in last_mtimes or last_mtimes[f] != mtime:
                    needs_update = True

            if not needs_update and out_png.exists():
                time.sleep(args.interval)
                continue

            last_mtimes = current_mtimes

            # Read all data
            all_rows = []
            for f in csv_files:
                try:
                    with f.open("r", newline="") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            all_rows.append(row)
                except Exception:
                    # Ignore read errors from active writes
                    pass

            if not all_rows:
                time.sleep(args.interval)
                continue

            # Parse data (NOTE: now includes p_triadic)
            mu, ptri, kmean, dH, dS, trans, assort, ips, iters, status, nodes = [], [], [], [], [], [], [], [], [], [], []

            for row in all_rows:
                mu.append(safe_float(row.get("mu")))
                ptri.append(safe_float(row.get("p_triadic")))
                kmean.append(safe_float(row.get("kmean")))
                dH.append(safe_float(row.get("d_H")))
                dS.append(safe_float(row.get("d_S")))
                trans.append(safe_float(row.get("transitivity")))
                assort.append(safe_float(row.get("assortativity")))
                ips.append(safe_float(row.get("iters_per_sec")))
                iters.append(safe_int(row.get("iter")))
                status.append((row.get("status") or "").strip())
                nodes.append(safe_int(row.get("nodes")))

            mu_np = np.array(mu, dtype=float)
            p_np = np.array(ptri, dtype=float)
            k_np = np.array(kmean, dtype=float)
            dH_np = np.array(dH, dtype=float)
            dS_np = np.array(dS, dtype=float)
            trans_np = np.array(trans, dtype=float)
            assort_np = np.array(assort, dtype=float)
            ips_np = np.array(ips, dtype=float)
            it_np = np.array(iters, dtype=float)
            status_np = np.array(status)
            nodes_np = np.array(nodes, dtype=int)

            unique_nodes = sorted([int(n) for n in set(nodes_np) if n > 0])

            # Prefer to group by p_triadic; if it's missing, fall back to a single bucket.
            finite_p = [float(p) for p in set(p_np) if np.isfinite(p)]
            if finite_p:
                unique_p = sorted(finite_p)
            else:
                unique_p = [np.nan]

            # Assign colors per p_triadic
            p_to_color = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(unique_p)}

            # Assign markers per N (if multiple)
            n_to_marker = {n: marker_cycle[i % len(marker_cycle)] for i, n in enumerate(unique_nodes)}

            # --- PHASE TRANSITION ("CHAOS") DETECTION (unchanged logic, but must not interleave groups) ---
            # We'll do it within each (p, N) bucket to avoid interleaving artifacts.
            transition_mus = []
            for p in unique_p:
                for n in unique_nodes:
                    mask = (nodes_np == n)
                    if np.isfinite(p):
                        mask = mask & np.isfinite(p_np) & (p_np == p)

                    valid_mask = mask & np.isfinite(mu_np) & np.isfinite(k_np)
                    iso_mu = mu_np[valid_mask]
                    iso_k = k_np[valid_mask]
                    if len(iso_mu) < 3:
                        continue

                    s_idx = np.argsort(iso_mu)
                    s_mu, s_k = iso_mu[s_idx], iso_k[s_idx]
                    jumps = np.where(np.diff(s_k) < -0.5)[0]
                    for j in jumps:
                        transition_mus.append((s_mu[j] + s_mu[j + 1]) / 2.0)

            transition_mus = sorted(transition_mus)

            # Setup Plot - Forced X-axis alignment
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(10, 1, hspace=0.25)

            ax1_top = fig.add_subplot(gs[0, 0])
            ax1_bot = fig.add_subplot(gs[1, 0], sharex=ax1_top)
            ax2 = fig.add_subplot(gs[2, 0], sharex=ax1_top)
            ax3 = fig.add_subplot(gs[3, 0], sharex=ax1_top)
            ax4 = fig.add_subplot(gs[4, 0], sharex=ax1_top)
            ax5 = fig.add_subplot(gs[5, 0], sharex=ax1_top)
            ax6 = fig.add_subplot(gs[6, 0], sharex=ax1_top)
            ax6b = ax6.twinx()

            # Axis settings
            ax1_top.set_ylim(1.5, 1000)
            ax1_top.set_yscale("log")
            ax1_top.set_yticks([10, 100, 1000])
            ax1_bot.set_ylim(2.5, 5.5)

            ax1_top.spines["bottom"].set_visible(False)
            ax1_bot.spines["top"].set_visible(False)
            ax1_top.xaxis.tick_top()
            ax1_top.tick_params(labeltop=False)
            ax1_bot.xaxis.tick_bottom()

            d = 0.015
            kwargs = dict(transform=ax1_top.transAxes, color="k", clip_on=False)
            ax1_top.plot((-d, +d), (-d, +d), **kwargs)
            ax1_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax1_bot.transAxes)
            ax1_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax1_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            ax1_top.set_title("Degree Penality mu (μ) sweep (overlay by p_triadic)")
            ax1_bot.set_ylabel("kmean", fontsize=7)
            ax2.set_ylabel("dH (Hausdorff)", fontsize=7)
            ax3.set_ylabel("dS (Spectral)", fontsize=7)
            ax4.set_ylabel("transitivity", fontsize=7)
            ax5.set_ylabel("degree assortativity", fontsize=7)
            ax5.set_ylim(-1, 1.0)
            ax6.set_ylabel("iters/sec", fontsize=7)
            ax6b.set_ylabel("iters (final)", fontsize=7)
# --- Plot data grouped by p_triadic (color) and N (marker) ---

            # Initialize lists BEFORE the loop so they populate correctly
            p_handles = []
            p_labels = []

            # Fit lines are still drawn, but not forced into the legend to keep it readable.
            for p in unique_p:
                color = p_to_color[p]
                for n in unique_nodes:
                    mask = (nodes_np == n)
                    if np.isfinite(p):
                        mask = mask & np.isfinite(p_np) & (p_np == p)

                    if not np.any(mask):
                        continue

                    marker = n_to_marker.get(n, "o")

                    # Build the base label string with N and p_triadic
                    label_str = f"N={n}, p_triadic={fmt_p(p)}"

                    # Fit & draw predicted line on ax6b (per (p, N) bucket, for COMPLETED points)
                    comp_mask = mask & (status_np == "COMPLETED")
                    if np.any(comp_mask):
                        fit = try_fit_critical_slowing(mu_np[comp_mask], it_np[comp_mask])
                        if fit is not None:
                            muc, gamma, a = fit
                            mu_line = np.linspace(np.nanmin(mu_np[mask]), np.nanmax(mu_np[mask]), 200)
                            valid = mu_line > muc
                            predT = np.full_like(mu_line, np.nan, dtype=float)
                            predT[valid] = a / np.power((mu_line[valid] - muc), gamma)
                            ax6b.plot(mu_line, predT, color=color, linestyle="--", alpha=0.8)

                            # Append the critical slowing down fit parameters to the label
                            label_str += f", μc≈{muc:.4f}, γ≈{gamma:.4f}"

                    # Append exact color AND marker to the legend handle
                    p_handles.append(Line2D([0], [0], marker=marker, linestyle="None", color=color, markersize=8))
                    p_labels.append(label_str)

                 # Broken-axis split for kmean
                    top_mask = mask & (k_np >= 5.5)
                    bot_mask = mask & (k_np < 5.5)

                    # --- Visual crowding adjustments ---
                    base_size = 8      # Reduced from 14
                    base_alpha = 0.4   # Reduced from 0.7 so overlapping points show density
                    # -----------------------------------

                    # We use edgecolors="none" so marker borders don't muddy the plot in dense regions
                    ax1_top.scatter(mu_np[top_mask], k_np[top_mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")
                    ax1_bot.scatter(mu_np[bot_mask], k_np[bot_mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")

                    # Other plots
                    ax2.scatter(mu_np[mask], dH_np[mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")
                    ax3.scatter(mu_np[mask], dS_np[mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")
                    ax4.scatter(mu_np[mask], trans_np[mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")
                    ax5.scatter(mu_np[mask], assort_np[mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")
                    ax6.scatter(mu_np[mask], ips_np[mask], s=base_size, color=color, alpha=base_alpha, marker=marker, edgecolors="none")

                    # Make the 'iters' (final) overlay even fainter so it doesn't distract from performance metrics
                    ax6b.scatter(mu_np[mask], it_np[mask], s=6, color=color, alpha=0.25, marker="x")


            ax1_top.set_zorder(10)
            ax1_top.legend(p_handles, p_labels, loc="upper right", bbox_to_anchor=(0.99, 0.96), facecolor="white", framealpha=0.75,
                title="Legend (Color = p_triadic, Marker = N)", fontsize=8
            )

            for ax in (ax1_top, ax1_bot, ax2, ax3, ax4, ax5, ax6):
                ax.grid(True, alpha=0.25)
                if ax != ax6:
                    ax.tick_params(labelbottom=False)

            # Draw transition lines
            if transition_mus:
                # Simple draw (if too many, they will overlap lightly)
                for mid_mu in transition_mus:
                    for ax in (ax1_top, ax1_bot, ax2, ax3, ax4, ax5, ax6):
                        if mid_mu >= 0.015:
                            ax.axvline(x=mid_mu, color="red", linestyle="--", linewidth=0.25, alpha=0.3, zorder=0)
                            ax6.text(mid_mu,-0.25, f"{mid_mu:.4f}", color="red", rotation=45, va="top", ha="center", fontsize=7,
                            transform=ax6.get_xaxis_transform(), clip_on=False,            )


            # Save atomically
            temp_png = out_png.with_suffix(".tmp.png")
            fig.savefig(temp_png, dpi=400, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            temp_png.replace(out_png)

        except Exception as e:
            print(f"Error during plotting: {e}")

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
