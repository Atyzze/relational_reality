#!/usr/bin/env python3
"""
analyze_scaling.py

Scans for sweep_summary_append.csv files, calculates the critical breakout
point (mu_c) for each system size (N), and plots the finite-size scaling law
to extrapolate the thermodynamic phase transition limit.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# Standard Finite-Size Scaling Power Law
def scaling_law(N, mu_inf, A, theta):
    return mu_inf + A * np.power(N, -theta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=str, default="mu_sweep", help="Root directory containing batch folders")
    ap.add_argument("--out-png", type=str, default="finite_size_scaling.png", help="Output PNG path")
    ap.add_argument("--breakout-threshold", type=float, default=6.0,
                    help="kmean threshold to define the breakout from the plateau phase")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    csv_files = list(sweep_dir.rglob("sweep_summary_append.csv"))

    if not csv_files:
        print(f"No summary CSVs found in {sweep_dir}")
        return

    print(f"Loading data from {len(csv_files)} files...")

    # Load and concatenate all data
    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Could not read {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)

    # Clean data
    df['mu'] = pd.to_numeric(df['mu'], errors='coerce')
    df['kmean'] = pd.to_numeric(df['kmean'], errors='coerce')
    df['d_S'] = pd.to_numeric(df['d_S'], errors='coerce')
    df['nodes'] = pd.to_numeric(df['nodes'], errors='coerce')
    df = df.dropna(subset=['mu', 'nodes', 'kmean'])

    unique_nodes = sorted(df['nodes'].unique())
    print(f"Found system sizes: {unique_nodes}")

    # Calculate mu_c for each N
    scaling_data = []

    for n in unique_nodes:
        df_n = df[df['nodes'] == n]

        # Sort by mu descending (moving right to left on the graph)
        df_n_sorted = df_n.sort_values('mu', ascending=False)

        # Find the highest mu where the network "breaks out" (kmean > threshold)
        breakout_points = df_n_sorted[df_n_sorted['kmean'] > args.breakout_threshold]

        if not breakout_points.empty:
            mu_c = breakout_points['mu'].max()
            scaling_data.append({'N': n, 'mu_c': mu_c})
            print(f"N={int(n)} -> Breakout detected at mu_c = {mu_c:.5f}")
        else:
            print(f"N={int(n)} -> No breakout detected (kmean never exceeded {args.breakout_threshold})")

    scaling_df = pd.DataFrame(scaling_data)

    if len(scaling_df) < 3:
        print("Not enough N values with a detected breakout to fit a scaling curve (need at least 3).")

    # Set up the visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax_k = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1], sharex=ax_k)
    ax_scale = fig.add_subplot(gs[1, :])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot 1 & 2: kmean and d_S vs mu
    for i, n in enumerate(unique_nodes):
        c = colors[i % len(colors)]
        df_n = df[df['nodes'] == n]

        # kmean plot
        ax_k.scatter(df_n['mu'], df_n['kmean'], s=10, alpha=0.6, color=c, label=f"N={int(n)}")
        # d_S plot
        df_n_ds = df_n.dropna(subset=['d_S'])
        ax_s.scatter(df_n_ds['mu'], df_n_ds['d_S'], s=10, alpha=0.6, color=c)

        # Draw vertical lines for the breakout points we found
        if n in scaling_df['N'].values:
            mu_c_val = scaling_df.loc[scaling_df['N'] == n, 'mu_c'].values[0]
            ax_k.axvline(x=mu_c_val, color=c, linestyle='--', alpha=0.5)
            ax_s.axvline(x=mu_c_val, color=c, linestyle='--', alpha=0.5)

    ax_k.set_yscale('log')
    ax_k.set_ylabel("Mean Degree (kmean)")
    ax_k.set_title("Breakout Threshold Detection")
    ax_k.axhline(y=args.breakout_threshold, color='red', linestyle=':', label="Breakout Threshold")
    ax_k.legend()
    ax_k.grid(True, alpha=0.3)

    ax_s.set_ylabel("Spectral Dimension (d_S)")
    ax_s.set_title("Dimensional Diffusion Scatter")
    ax_s.grid(True, alpha=0.3)

    # Plot 3: Finite Size Scaling Fit
    if len(scaling_df) >= 3:
        N_vals = scaling_df['N'].values
        mu_vals = scaling_df['mu_c'].values

        # Try to fit the curve
        try:
            # P0: initial guesses [mu_inf, A, theta]
            p0 = [min(mu_vals) * 0.9, 0.1, 0.5]
            bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])

            popt, _ = curve_fit(scaling_law, N_vals, mu_vals, p0=p0, bounds=bounds, maxfev=10000)
            mu_inf, A, theta = popt

            # Generate points for the smooth fit line
            N_line = np.logspace(np.log10(min(N_vals)*0.5), np.log10(max(N_vals)*2), 100)
            mu_line = scaling_law(N_line, mu_inf, A, theta)

          # Plot against 1/N for standard scaling visualization
            inv_N = 1.0 / N_vals
            inv_N_line = 1.0 / N_line

            # FIXED: Added 'r' before the strings to handle LaTeX slashes properly
            ax_scale.scatter(inv_N, mu_vals, color='red', s=60, zorder=5, label=r'Measured Breakout Points $\mu_c(N)$')
            ax_scale.plot(inv_N_line, mu_line, color='black', linestyle='--',
                          label=rf'Fit: $\mu_c(N) = {mu_inf:.5f} + {A:.3f} \cdot N^{{-{theta:.3f}}}$')

            # Mark the extrapolated thermodynamic limit
            ax_scale.axhline(y=mu_inf, color='blue', linestyle=':',
                             label=rf'Thermodynamic Limit ($N \to \infty$): $\mu_c \approx {mu_inf:.5f}$')

            ax_scale.set_title("Finite-Size Scaling of the Geometric Phase Transition")
            ax_scale.set_xlabel(r"$1 / N$")
            ax_scale.set_ylabel(r"Critical Penalty $\mu_c$")
            ax_scale.legend()
            ax_scale.grid(True, alpha=0.3)

            print(f"\n--- SCALING RESULTS ---")
            print(f"Extrapolated infinite-size transition limit (mu_c at N=inf): {mu_inf:.6f}")
            print(f"Power-law exponent (theta): {theta:.6f}")

        except Exception as e:
            print(f"Curve fitting failed: {e}")
            ax_scale.text(0.5, 0.5, "Could not fit scaling law curve.", ha='center', va='center')
    else:
        ax_scale.text(0.5, 0.5, "Need at least 3 valid N datasets to fit scaling law.", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200, bbox_inches='tight')
    print(f"Analysis saved to {args.out_png}")

if __name__ == "__main__":
    main()
