import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import seaborn as sns

import warnings
# Suppress the specific FutureWarning about concatenation
warnings.simplefilter(action='ignore', category=FutureWarning)

# CONFIG
FILE_PATTERN = "runs/sweep_data_N*.csv"
# FILE_PATTERN = "runs/sweep_data_N*.csv" # Use this if your csvs are in a subfolder
OUTPUT_DIR = "universe_dashboard"

# Columns to exclude from being plotted as metrics
IGNORE_COLS = [
    "seed", "N_actual", "Step_measured", "N", "steps",
    "param_LAMBDA_E", "param_LAMBDA_G", "param_BETA",
    "param_KAPPA", "param_PAULI", "param_RHO0",
    "param_TEMP_START", "param_TEMP_SCALE",
    "source_file"
]

def load_data():
    files = glob.glob(FILE_PATTERN)
    if not files:
        print(f"No data files found matching '{FILE_PATTERN}'!")
        return None

    df_list = []
    for f in files:
        try:
            # Check if file is empty
            if os.path.getsize(f) == 0:
                print(f"Skipping empty file: {f}")
                continue

            df = pd.read_csv(f)

            # Ensure N_actual exists
            if 'N_actual' not in df.columns and 'N' in df.columns:
                df['N_actual'] = df['N']

            # Add source for debugging
            df['source_file'] = f

            # --- CRITICAL FIX: Force Numeric Conversion ---
            # This fixes "missing metrics" caused by corrupted lines
            for col in df.columns:
                if col not in IGNORE_COLS and col != 'source_file':
                    # Try to force conversion to numbers, turn errors into NaNs
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # ---------------------------------------------

            df_list.append(df)
            print(f">> Loaded {len(df)} samples from {f}")

        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not df_list: return None

    # Combine (handling legacy files that might miss columns)
    return pd.concat(df_list, ignore_index=True)

def plot_metric_evolution(df, col_name):
    # Filter out NaNs for this specific metric before plotting
    # This prevents empty plots if a specific run failed this metric
    valid_df = df.dropna(subset=[col_name])

    if len(valid_df) == 0:
        print(f"   [Skip] No valid data for {col_name}")
        return

    # 1. Statistics
    stats = valid_df.groupby('N_actual')[col_name].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()
    stats = stats.sort_values('N_actual')

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 7))

    # Envelope
    ax.fill_between(stats['N_actual'], stats['min'], stats['max'],
                    color='gray', alpha=0.15, label='Min-Max Range')

    # Scatter (Jittered)
    sns.stripplot(data=valid_df, x='N_actual', y=col_name, ax=ax,
                  color='black', alpha=0.3, jitter=0.1, size=4, zorder=1)

    # Mean Line
    ax.plot(range(len(stats)), stats['mean'], color='#d62728', linewidth=2.5,
            marker='o', markersize=8, label='Mean Value', zorder=2)

    # Annotations
    for i, row in stats.iterrows():
        label_txt = f"{row['mean']:.2f}\n[{row['min']:.2f}, {row['max']:.2f}]"
        ax.text(i, row['max'], label_txt,
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    # Styling
    ax.set_title(f"Metric Analysis: {col_name}", fontsize=16, pad=20)
    ax.set_xlabel("System Size (N)", fontsize=12)
    ax.set_ylabel(col_name, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats['N_actual'].astype(int))

    # Legend construction
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    scatter_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                          label='Individual Universe', markersize=6, alpha=0.5)

    if len(handles) >= 2:
        final_handles = [handles[0], handles[1], scatter_handle]
        final_labels = [labels[0], labels[1], 'Single Run']
        ax.legend(final_handles, final_labels, loc='best')

    safe_name = col_name.replace("/", "_")
    out_path = f"{OUTPUT_DIR}/{safe_name}_scaling.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"   [Graph] Saved {out_path}")
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(">> Reading Multiverse Data...")
    df = load_data()
    if df is None: return

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in IGNORE_COLS]

    print(f">> Found {len(metrics)} metrics to plot: {metrics}")

    for metric in metrics:
        plot_metric_evolution(df, metric)

    print(f"\n>> All graphs generated in folder: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
