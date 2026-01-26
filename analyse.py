import os
import re
import glob
import pandas as pd
import networkx as nx
import numpy as np
import time
import warnings
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
RUNS_DIR = "runs"
STABILITY_WINDOW = 10      # Used for data integrity validation (checking last X rows for corruption)
MIN_SNAPSHOTS = 3          # Minimum snapshots required within the 1% window to confirm stability
PROBE_COUNT = 100          # Keep low to speed up Hausdorff

# --- WARNING SUPPRESSION ---
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning

warnings.simplefilter('ignore', RankWarning)
warnings.filterwarnings("ignore", module="numpy")

# --- HELPER: FORMATTING ---
def fmt_step(step):
    """Helper to format step count to match drive.py."""
    return f"{step:011_d}"

def extract_metadata(data_folder_path):
    """Extracts Version, N, and Seed from path hierarchy."""
    try:
        path_parts = os.path.normpath(data_folder_path).split(os.sep)
        if path_parts[-1] != 'data': return 0, 0, 0
        seed_str, n_str, ver_str = path_parts[-2], path_parts[-3], path_parts[-4]
        seed = int(seed_str[1:]) if seed_str.startswith('S') and seed_str[1:].isdigit() else 0
        N = int(n_str[1:]) if n_str.startswith('N') and n_str[1:].isdigit() else 0
        version = int(ver_str[1:]) if ver_str.startswith('E') and ver_str[1:].isdigit() else 0
        return version, N, seed
    except:
        return 0, 0, 0

def get_metrics_path(data_folder_path, version, N, seed):
    parent_dir = os.path.dirname(data_folder_path)
    filename = f"E{version}_N{N}_S{seed}_metrics.csv"
    return os.path.join(parent_dir, filename)

# --- PHYSICS METRICS ---
def compute_hausdorff_dimension(G, probes=PROBE_COUNT):
    if G.number_of_nodes() < 5 or G.number_of_edges() == 0: return 0.0
    nodes = list(G.nodes())
    centers = np.random.choice(nodes, size=min(len(nodes), probes), replace=False)
    rs_all, counts_all = [], []
    for source in centers:
        dists = nx.single_source_shortest_path_length(G, source)
        max_d = max(dists.values())
        if max_d < 2: continue
        counts = np.zeros(max_d + 1)
        for d in dists.values(): counts[d] += 1
        cumulative = np.cumsum(counts)
        rs_all.extend(np.arange(1, len(cumulative)))
        counts_all.extend(cumulative[1:])
    if len(rs_all) < 5: return 0.0
    try:
        log_r, log_n = np.log(rs_all), np.log(counts_all)
        mask = (log_r > np.log(1.5)) & (log_r < np.log(8))
        slope, _ = np.polyfit(log_r[mask], log_n[mask], 1) if np.sum(mask) >= 3 else np.polyfit(log_r, log_n, 1)
        return slope
    except: return 0.0

def verify_step_integrity(folder_path, step, expected_edges, version, N, seed):
    """
    Checks if the file on disk matches the metadata in the CSV.
    Used to validate the last few entries of the metrics file.
    """
    step_str = fmt_step(step)
    fpath = os.path.join(folder_path, f"E{version}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    if not os.path.exists(fpath):
        return False

    # Fast check: If expected edges is 0, file should be just header (or very small)
    # If > 0, we can do a quick line count check if performance is okay,
    # but for now, strictness is better to avoid "garbage out".
    try:
        # Optimization: Don't parse CSV, just count lines (faster)
        with open(fpath, 'rb') as f:
            lines = sum(1 for _ in f)

        # Lines in file = edges + 1 (header)
        return (lines - 1) == expected_edges
    except:
        return False

def calculate_single_step(folder_path, step, version, N_meta, seed):
    step_str = fmt_step(step)
    fpath = os.path.join(folder_path, f"E{version}_N{N_meta}_S{seed}_iter_{step_str}_edges.csv")
    try:
        df = pd.read_csv(fpath)
        G = nx.from_pandas_edgelist(df, 'source', 'target') if not df.empty else nx.Graph()

        # --- ROBUST DEGREE CALCULATION ---
        connected_nodes = G.number_of_nodes()
        raw_degrees = [d for n, d in G.degree()]
        missing_nodes = max(0, N_meta - connected_nodes)
        degrees = raw_degrees + [0] * missing_nodes
        if len(degrees) < N_meta: degrees += [0] * (N_meta - len(degrees))

        k_mean, k_std = np.mean(degrees), np.std(degrees)
        k_min, k_max = np.min(degrees), np.max(degrees)

        if k_max > k_min:
            counts, _ = np.histogram(degrees, bins=5, range=(k_min, k_max))
        else:
            counts = np.array([len(degrees)] + [0]*4)

        current_sum = np.sum(counts)
        if (N_meta - current_sum) != 0: counts[0] += (N_meta - current_sum)

        return {
            'step': step,
            'N_actual': connected_nodes,
            'edges': len(df),
            'mean_degree': k_mean,
            'std_degree': k_std,
            'min_degree': k_min,
            'max_degree': k_max,
            'bin_1': counts[0], 'bin_2': counts[1], 'bin_3': counts[2],
            'bin_4': counts[3], 'bin_5': counts[4],
            'triangles': sum(nx.triangles(G).values()) // 3,
            'hausdorff': compute_hausdorff_dimension(G)
        }
    except Exception:
        return None

def process_folder_metrics(folder_path):
    version, N, seed = extract_metadata(folder_path)
    metrics_path = get_metrics_path(folder_path, version, N, seed)
    edge_files = glob.glob(os.path.join(folder_path, f"E{version}_N{N}_S{seed}_iter_*_edges.csv"))

    # 1. Get List of Steps on Disk
    disk_steps = []
    pattern = re.compile(rf"iter_([\d_]+)_edges")
    for f in edge_files:
        match = pattern.search(f)
        if match:
            disk_steps.append(int(match.group(1).replace('_', '')))
    disk_steps.sort()

    if not disk_steps: return 0, 0

    # 2. Smart Resume: Check existing CSV
    existing_df = pd.DataFrame()
    start_step = 0

    if os.path.exists(metrics_path):
        try:
            existing_df = pd.read_csv(metrics_path)
            if not existing_df.empty:
                existing_df = existing_df.sort_values('step')

                # Check the last entries.
                # If they match disk, we assume the rest are fine.
                # If any fail, we truncate to the last good one.

                to_validate = existing_df.tail(STABILITY_WINDOW)
                last_good_idx = -1

                all_valid = True
                failure_found = False

                for idx, row in to_validate.iterrows():
                    s = int(row['step'])
                    e = int(row['edges'])
                    if not verify_step_integrity(folder_path, s, e, version, N, seed):
                        # Found a corruption/mismatch
                        failure_found = True
                        # Truncate existing_df to everything BEFORE this index
                        existing_df = existing_df.loc[:idx-1]
                        break

                if not failure_found:
                    # All checked rows were good. Resume from end.
                    start_step = int(existing_df.iloc[-1]['step']) + 1
                else:
                    # We truncated. Resume from new end.
                    if not existing_df.empty:
                        start_step = int(existing_df.iloc[-1]['step']) + 1
                        # Save the repaired CSV immediately so we don't have bad data on disk
                        existing_df.to_csv(metrics_path, index=False)
                    else:
                        start_step = 0
                        # File was effectively corrupt/empty, clear it
                        open(metrics_path, 'w').close()

        except Exception as e:
            # If CSV is unreadable, start over
            existing_df = pd.DataFrame()
            start_step = 0

    # 3. Process Missing Steps
    steps_to_process = [s for s in disk_steps if s >= start_step]

    if not steps_to_process:
        return 0, (disk_steps[-1] if disk_steps else 0)

    new_rows = [calculate_single_step(folder_path, s, version, N, seed) for s in steps_to_process]
    new_rows = [r for r in new_rows if r is not None]

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Append logic
        if not existing_df.empty:
            # Ensure columns match before appending
            if list(df_new.columns) == list(existing_df.columns):
                df_new.to_csv(metrics_path, mode='a', header=False, index=False)
            else:
                # Columns mismatch (maybe schema update), re-write all
                combined = pd.concat([existing_df, df_new], ignore_index=True)
                combined.to_csv(metrics_path, mode='w', header=True, index=False)
        else:
            df_new.to_csv(metrics_path, mode='w', header=True, index=False)

    return len(new_rows), (disk_steps[-1] if disk_steps else 0)

def check_run_status(folder_path):
    """
    Determines if a run is 'Stable'.
    Criteria:
    1. Triangle Count identical for the last 1% (OR last MIN_SNAPSHOTS, whichever is larger).
    2. Graph must be connected.
    """
    version_id, N, seed = extract_metadata(folder_path)
    metrics_path = get_metrics_path(folder_path, version_id, N, seed)

    if not os.path.exists(metrics_path):
        return False, [], f"E{version_id}", (N, seed), 0

    try:
        df = pd.read_csv(metrics_path).sort_values('step')
        if df.empty: return False, [], f"E{version_id}", (N, seed), 0

        max_step = df['step'].max()

        # --- DYNAMIC WINDOW CALCULATION ---
        # We don't strictly know 'interval' here without parsing, but we can infer
        # or just grab the last MIN_SNAPSHOTS rows if the 1% window is too small.

        # 1. Try 1% Window first
        threshold_step = max_step - (max_step * 0.01)
        window_df = df[df['step'] >= threshold_step]

        # 2. If 1% yields too few points, fallback to the last MIN_SNAPSHOTS strictly
        if len(window_df) < MIN_SNAPSHOTS:
            window_df = df.tail(MIN_SNAPSHOTS)

        # Safety: If we STILL don't have enough data (e.g. run just started), it's not stable.
        if len(window_df) < MIN_SNAPSHOTS:
             return False, [], f"E{version_id}", (N, seed), max_step

        # 3. Check Connectivity (Must be connected at the end)
        if window_df.iloc[-1]['min_degree'] <= 0:
             return False, [], f"E{version_id}", (N, seed), max_step

        # 4. Check Stability (Identical Triangles across window)
        if window_df['triangles'].nunique() != 1:
             return False, [], f"E{version_id}", (N, seed), max_step

        # --- IF STABLE: GENERATE SLICES ---
        slices = []
        target_percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                              0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

        for p in target_percentages:
            target = int(max_step * p)
            idx = (df['step'] - target).abs().idxmin()
            row = df.loc[idx]

            if abs(row['step'] - target) <= max(1, (0.15 * max_step)):
                d = row.to_dict()
                d['slice_lbl'] = f"{int(p*100)}%"
                slices.append(d)

        return True, slices, f"E{version_id}", (N, seed), max_step
    except Exception as e:
        return False, [], f"E{version_id}", (N, seed), 0

def find_all_data_folders(root_dir):
    data_folders = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == 'data':
            ver, N, seed = extract_metadata(root)
            if N > 0:
                data_folders.append(root)
    return data_folders

def main():
    print("="*140)
    print("  RELATIONAL REALITY: EVOLUTION & DISTRIBUTION ANALYSIS")
    print("="*140)

    print(f">> Scanning runs/ directory...")
    all_folders = find_all_data_folders(RUNS_DIR)

    if not all_folders:
        print("No 'data' folders found in runs/ directory.")
        return

    # --- PHASE 1: UPDATE METRICS ---
    t0 = time.time()
    updated_count = 0
    workers = max(1, os.cpu_count() - 2)

    print(f"   Calculating missing metrics for {len(all_folders)} runs (Smart Resume)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_folder_metrics, f): f for f in all_folders}
        for i, future in enumerate(as_completed(futures)):
            try:
                cnt, _ = future.result()
                updated_count += cnt
            except Exception as e:
                print(f"   Error processing {futures[future]}: {e}")

    print(f"   Metrics update complete ({updated_count} new snapshots processed in {time.time()-t0:.1f}s).\n")

    # --- PHASE 2: GENERATE REPORT ---
    completed_results = []
    active_runs = []

    for f in all_folders:
        is_complete, slices, label, meta, last_step = check_run_status(f)

        if is_complete:
            if slices:
                for s in slices:
                    completed_results.append({'config': label, 'N': meta[0], 'seed': meta[1], 'slice': s['slice_lbl'], 'stats': s})
        else:
            active_runs.append({'config': label, 'N': meta[0], 'seed': meta[1], 'step': last_step})

    # --- TABLE PRINTING: COMPLETED RUNS ---
    print(f"COMPLETED RUNS (Stable) - Aggregated Report")
    print("-" * 148)
    print(f"{'Version':<8} | {'N':<6} | {'Slice':<5} | {'Seeds':<5} | {'Steps (Avg)':<11} | {'<k> ± σ':<11} | {'Min':<5} | {'Max':<5} | {'Distribution':<25} | {'Triangles':<9} | {'Haus.'}")
    print("-" * 148)

    df_res = pd.DataFrame(completed_results)
    if not df_res.empty:
        grouped = df_res.groupby(['config', 'N', 'slice'])

        def sort_key(k):
            ver, n, slc = k
            slc_val = int(slc.replace('%', ''))
            return (ver, n, slc_val)

        keys = sorted(list(grouped.groups.keys()), key=sort_key)

        for key in keys:
            cfg, n_val, slc = key
            group = grouped.get_group(key)

            seed_count = group['seed'].nunique()
            stats = [r['stats'] for _, r in group.iterrows()]

            avg_step = np.mean([s['step'] for s in stats])
            if avg_step > 1_000_000: step_str = f"{avg_step/1_000_000:.2f}M"
            else: step_str = f"{avg_step/1_000:.1f}K"

            k_mean = np.mean([s['mean_degree'] for s in stats])
            k_std = np.mean([s.get('std_degree',0) for s in stats])

            hist_vals = [np.mean([s.get(f'bin_{i}',0) for s in stats]) for i in range(1,6)]
            hist_str = ":".join([f"{v:.0f}" for v in hist_vals])

            print(f"{cfg:<8} | {n_val:<6} | {slc:<5} | {seed_count:<5} | {step_str:<11} | {f'{k_mean:.2f}±{k_std:.2f}':<11} | {np.mean([s.get('min_degree',0) for s in stats]):<5.1f} | {np.mean([s.get('max_degree',0) for s in stats]):<5.1f} | {hist_str:<25} | {np.mean([s['triangles'] for s in stats]):<9.0f} | {np.mean([s.get('hausdorff',0) for s in stats]):.2f}")

            if slc == "100%": print("-" * 148)
    else:
        print("   No completed runs found.")

    # --- RECENT ACTIVITY: INCOMPLETE RUNS ---
    print("\n\nRECENT ACTIVITY (Active / Unstabilized Runs)")
    print("-" * 60)
    print(f"{'Config':<15} | {'Seed':<8} | {'Last Snapshot Step'}")
    print("-" * 60)

    if active_runs:
        active_runs.sort(key=lambda x: (x['config'], x['N'], x['seed']))
        for r in active_runs:
            cfg_str = f"{r['config']} N{r['N']}"
            step_disp = f"{r['step']:,}" if r['step'] > 0 else "Initializing..."
            print(f"{cfg_str:<15} | {r['seed']:<8} | {step_disp}")
    else:
        print("   No active runs.")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
