import os
import glob
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = "runs"

def get_trajectories():
    """
    Parses log files and organizes data by (Engine_ID, N).
    Structure: data[(engine_id, N)][seed] = list of (step, k_min, k_avg, k_max)
    """
    data = {}
    log_files = glob.glob(os.path.join(RUNS_DIR, "**", "logs", "*.*"), recursive=True)
    valid_files = [f for f in log_files if f.endswith('.log') or f.endswith('.csv')]

    print(f"Parsing {len(valid_files)} log files...")

    for lf in valid_files:
        try:
            with open(lf, 'r') as f:
                lines = f.readlines()

            if len(lines) < 3: continue

            # Parse Metadata
            meta_line = lines[0].lstrip('#').strip()
            meta_parts = meta_line.split(',')
            meta = {}
            for x in meta_parts:
                if '=' in x:
                    key, val = x.split('=', 1)
                    meta[key.strip()] = val.strip()

            N = int(meta.get('N', 0))
            seed = int(meta.get('seed', 0))
            engine_id = int(meta.get('metadata_version', 1))

            if N == 0: continue

            group_key = (engine_id, N)

            if group_key not in data: data[group_key] = {}
            if seed not in data[group_key]: data[group_key][seed] = []

            # Parse Body
            for line in lines[2:]:
                if line.startswith('#'): continue
                cols = line.strip().split(',')
                try:
                    step = int(cols[1])
                    k_min = int(cols[4])
                    k_avg = float(cols[5])
                    k_max = int(cols[6])
                    data[group_key][seed].append((step, k_min, k_avg, k_max))
                except (ValueError, IndexError):
                    continue
        except Exception:
            continue

    return data

def plot_per_engine():
    trajectories = get_trajectories()
    if not trajectories:
        print("No data found.")
        return

    # Identify unique engines
    all_keys = trajectories.keys()
    unique_engines = sorted(list(set(k[0] for k in all_keys)))
    num_engines = len(unique_engines)

    if num_engines == 0:
        print("No engines detected.")
        return

    # Dynamic height: 5 inches per engine.
    # CHANGED: sharex=False to allow independent scaling
    fig, axes = plt.subplots(nrows=num_engines, ncols=1, figsize=(12, 5 * num_engines), sharex=False)

    if num_engines == 1:
        axes = [axes]

    print(f"Plotting {num_engines} unique engines...")

    for ax, engine_id in zip(axes, unique_engines):
        engine_keys = [k for k in all_keys if k[0] == engine_id]
        engine_keys.sort(key=lambda x: x[1])

        colors = plt.cm.turbo(np.linspace(0, 1, len(engine_keys)))

        for idx, key in enumerate(engine_keys):
            N = key[1]
            seeds = trajectories[key]
            color = colors[idx]

            first_run = True
            for seed, points in seeds.items():
                if not points: continue
                points.sort(key=lambda x: x[0])

                steps = [p[0] for p in points]
                k_mins = [p[1] for p in points]
                k_avgs = [p[2] for p in points]
                k_maxs = [p[3] for p in points]

                label = f"N={N}" if first_run else None

                ax.plot(steps, k_avgs, color=color, alpha=0.9, linewidth=1.5, label=label)
                ax.fill_between(steps, k_mins, k_maxs, color=color, alpha=0.1)

                first_run = False

        ax.set_title(f"Engine Version E{engine_id}", fontsize=14, pad=10)
        ax.set_ylabel('Degree (k)')
        # CHANGED: Added x-label to every chart since they now have different scales
        ax.set_xlabel('Simulation Steps')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.suptitle('Convergence Comparison by Engine Version', fontsize=16, y=0.99)
    plt.tight_layout()

    out_file = "k_trajectories_per_engine.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    plot_per_engine()
