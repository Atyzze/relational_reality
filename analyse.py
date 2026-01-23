import os
import re
import glob
import pandas as pd
import networkx as nx
import numpy as np
import time
import json
import hashlib
import warnings
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
RUNS_DIR = "runs"
STABILITY_WINDOW = 10      # Look at last 10 snapshots for final validation
STABILITY_THRESHOLD = 0.01 # 1% change allowed
PROBE_COUNT = 100          # Keep low to speed up Hausdorff
REQUIRED_FILE_COUNT = 101  # Exact number of steps (0-100) required to be considered "Finished"

# --- ACTIVITY CHECK CONFIG ---
ACTIVITY_CHECK_WINDOW = 4        # Check time diffs between the last 4 files
ACTIVITY_TIMEOUT_MULTIPLIER = 3.0 # If time since last file > Avg_Step_Time * 3.0, consider Dead.
MIN_FILES_FOR_ACTIVITY = 2       # Need at least 2 files to calculate a delta

# --- WARNING SUPPRESSION ---
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning

warnings.simplefilter('ignore', RankWarning)
warnings.filterwarnings("ignore", module="numpy")

# --- HELPER: FORMATTING ---
def fmt_step(step):
    """
    Helper to format step count to match drive.py.
    Example: 1 -> '000_000_001'
    """
    return f"{step:011_d}"

# --- HELPER: CONFIG HASHING ---
def get_config_signature(data_folder_path, version_id):
    """
    Generates a short hash based on engine.py.
    Assumes structure: runs/E<ver>/N<N>/S<seed>/data
    So engine.py is at: runs/E<ver>/engine.py
    """
    # Go up 3 levels: data -> S<seed> -> N<N> -> E<ver>
    version_dir = os.path.dirname(os.path.dirname(os.path.dirname(data_folder_path)))
    engine_path = os.path.join(version_dir, "engine.py")

    hasher = hashlib.sha256()

    if os.path.exists(engine_path):
        with open(engine_path, 'rb') as f:
            hasher.update(f.read())
    else:
        # Fallback: check if engine exists in the data folder itself (older versions)
        local_engine = os.path.join(data_folder_path, "engine.py")
        if os.path.exists(local_engine):
             with open(local_engine, 'rb') as f:
                hasher.update(f.read())
        else:
            hasher.update(b"NO_ENGINE")

    return hasher.hexdigest()[:8]

def extract_metadata(data_folder_path):
    """
    Extracts Version, N, and Seed from path hierarchy.
    Expected: runs/E<ver>/N<N>/S<seed>/data
    """
    try:
        # path parts: [..., 'runs', 'E0', 'N100', 'S1000', 'data']
        path_parts = os.path.normpath(data_folder_path).split(os.sep)

        # We expect 'data' to be the last part
        if path_parts[-1] != 'data':
            return 0, 0, 0

        seed_str = path_parts[-2] # S1000
        n_str = path_parts[-3]    # N100
        ver_str = path_parts[-4]  # E0

        # Strip prefixes
        seed = int(seed_str[1:]) if seed_str.startswith('S') and seed_str[1:].isdigit() else 0
        N = int(n_str[1:]) if n_str.startswith('N') and n_str[1:].isdigit() else 0
        version = int(ver_str[1:]) if ver_str.startswith('E') and ver_str[1:].isdigit() else 0

        return version, N, seed
    except:
        return 0, 0, 0

def get_metrics_path(data_folder_path, version, N, seed):
    """
    Constructs the metrics file path.
    Location: Parent folder of 'data' (the seed folder).
    Name: E{id}_N{N}_S{seed}_metrics.csv
    """
    parent_dir = os.path.dirname(data_folder_path)
    filename = f"E{version}_N{N}_S{seed}_metrics.csv"
    return os.path.join(parent_dir, filename)

# --- PHYSICS METRICS ---

def compute_hausdorff_dimension(G, probes=PROBE_COUNT):
    """Estimates Hausdorff dimension via Box-Counting on the graph metric."""
    if G.number_of_nodes() < 5 or G.number_of_edges() == 0:
        return 0.0

    nodes = list(G.nodes())
    centers = np.random.choice(nodes, size=min(len(nodes), probes), replace=False)

    rs_all, counts_all = [], []

    for source in centers:
        dists = nx.single_source_shortest_path_length(G, source)
        max_d = max(dists.values())
        if max_d < 2: continue

        counts = np.zeros(max_d + 1)
        for d in dists.values():
            counts[d] += 1

        cumulative = np.cumsum(counts)
        valid_r = np.arange(1, len(cumulative))
        rs_all.extend(valid_r)
        counts_all.extend(cumulative[1:])

    if len(rs_all) < 5:
        return 0.0

    try:
        log_r = np.log(rs_all)
        log_n = np.log(counts_all)
        mask = (log_r > np.log(1.5)) & (log_r < np.log(8))

        if np.sum(mask) < 3:
            slope, _ = np.polyfit(log_r, log_n, 1)
        else:
            slope, _ = np.polyfit(log_r[mask], log_n[mask], 1)

        return slope
    except:
        return 0.0

# --- PHASE 1: METRIC GENERATION (SMART UPDATE) ---

def verify_step_integrity(folder_path, step, expected_edges, version, N, seed):
    """Checks if the file on disk matches the expected edge count in the CSV."""
    # Pattern: E{v}_N{n}_S{s}_iter_{step}_edges.csv
    step_str = fmt_step(step)
    fpath = os.path.join(folder_path, f"E{version}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    if not os.path.exists(fpath):
        return False
    try:
        df = pd.read_csv(fpath)
        actual_edges = len(df)
        return actual_edges == expected_edges
    except:
        return False

def calculate_single_step(folder_path, step, version, N, seed):
    """Calculates metrics for a single step."""
    step_str = fmt_step(step)
    fpath = os.path.join(folder_path, f"E{version}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    try:
        df = pd.read_csv(fpath)
        if df.empty:
            return {'step': step, 'N_actual': 0, 'edges': 0,
                    'mean_degree': 0.0, 'triangles': 0, 'hausdorff': 0.0}

        G = nx.from_pandas_edgelist(df, 'source', 'target')

        N_curr = G.number_of_nodes()
        E_curr = G.number_of_edges()
        k_mean = (2 * E_curr) / N_curr if N_curr > 0 else 0
        tri = sum(nx.triangles(G).values()) // 3
        dim_h = compute_hausdorff_dimension(G)

        return {
            'step': step,
            'N_actual': N_curr,
            'edges': E_curr,
            'mean_degree': k_mean,
            'triangles': tri,
            'hausdorff': dim_h
        }
    except Exception:
        return None

def process_folder_metrics(folder_path):
    # Extract metadata to build filenames
    version, N, seed = extract_metadata(folder_path)
    metrics_path = get_metrics_path(folder_path, version, N, seed)

    # 1. Identify all available steps on disk
    # Look for files like E0_N100_S1000_iter_000_000_000_edges.csv
    search_pattern = os.path.join(folder_path, f"E{version}_N{N}_S{seed}_iter_*_edges.csv")
    edge_files = glob.glob(search_pattern)

    disk_steps = []
    # Regex matches: ...iter_([\d_]+)_edges.csv
    pattern = re.compile(rf"iter_([\d_]+)_edges\.csv")

    for f in edge_files:
        filename = os.path.basename(f)
        m = pattern.search(filename)
        if m:
            # removing underscores to parse integer
            step_val = int(m.group(1).replace('_', ''))
            disk_steps.append(step_val)
    disk_steps.sort()

    if not disk_steps:
        return 0

    # 2. Check existing metrics file
    existing_df = pd.DataFrame()
    start_calculation_from = 0

    if os.path.exists(metrics_path):
        try:
            existing_df = pd.read_csv(metrics_path)
            if not existing_df.empty:
                existing_df = existing_df.sort_values('step')

                # VERIFICATION: Check last 2 entries
                valid_history = True
                rows_to_check = existing_df.tail(2)

                for _, row in rows_to_check.iterrows():
                    step_check = int(row['step'])
                    edges_check = int(row['edges'])
                    if not verify_step_integrity(folder_path, step_check, edges_check, version, N, seed):
                        valid_history = False
                        break

                if valid_history:
                    last_recorded_step = int(existing_df.iloc[-1]['step'])
                    start_calculation_from = last_recorded_step + 1
                else:
                    existing_df = pd.DataFrame()
                    start_calculation_from = 0
        except Exception:
            existing_df = pd.DataFrame()
            start_calculation_from = 0

    # 3. Determine tasks
    steps_to_process = [s for s in disk_steps if s >= start_calculation_from]

    if not steps_to_process:
        return 0

    # 4. Compute Metrics for new steps
    new_rows = []
    for step in steps_to_process:
        row = calculate_single_step(folder_path, step, version, N, seed)
        if row:
            new_rows.append(row)

    # 5. Append or Overwrite
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if not existing_df.empty:
            df_new.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(metrics_path, mode='w', header=True, index=False)

    return len(new_rows)

# --- PHASE 2: TIME SERIES & STABILITY CHECK ---

def check_run_stability(folder_path):
    version_id, N, seed = extract_metadata(folder_path)
    metrics_path = get_metrics_path(folder_path, version_id, N, seed)

    # Use version_id as the config identifier primarily, but we also hash engine
    cfg_hash = get_config_signature(folder_path, version_id)

    # Combined label for display
    config_label = f"E{version_id}"

    if not os.path.exists(metrics_path):
        return "No Metrics", None, config_label, (N, seed)

    try:
        df = pd.read_csv(metrics_path)
        if len(df) < STABILITY_WINDOW:
            return "Insufficient Data", None, config_label, (N, seed)

        df = df.sort_values('step')
        tri_series = df['triangles'].values
        steps = df['step'].values
        total_steps = steps[-1] if len(steps) > 0 else 1

        # 1. Peak Analysis
        peak_idx = np.argmax(tri_series)
        peak_val = tri_series[peak_idx]
        peak_step = steps[peak_idx]
        peak_pct = (peak_step / total_steps) * 100.0

        # 2. Final Value
        final_val = tri_series[-1]

        # 3. Stability / Settling Time Analysis
        last_window = tri_series[-STABILITY_WINDOW:]
        final_avg = np.mean(last_window)

        # Guard against division by zero if graph is empty
        if final_avg == 0:
             band_low = -0.1; band_high = 0.1
        else:
             band_low = final_avg * (1.0 - STABILITY_THRESHOLD)
             band_high = final_avg * (1.0 + STABILITY_THRESHOLD)

        stabilized_step_idx = 0
        for i in range(len(tri_series)-1, -1, -1):
            val = tri_series[i]
            if val < band_low or val > band_high:
                stabilized_step_idx = i + 1
                break

        if stabilized_step_idx >= len(steps): stabilized_step_idx = len(steps) - 1

        stab_step = steps[stabilized_step_idx]
        stab_pct = (stab_step / total_steps) * 100.0

        # 4. Status Check
        window_min = np.min(last_window)
        window_max = np.max(last_window)
        window_mean = np.mean(last_window)

        if window_mean == 0:
            pct_change = 0.0
        else:
            pct_change = (window_max - window_min) / window_mean

        status = "Unstable" if pct_change > STABILITY_THRESHOLD else "Stable"

        stats = df.iloc[-1].to_dict()
        stats['peak_tri'] = peak_val
        stats['peak_pct'] = peak_pct
        stats['final_tri'] = final_val
        stats['stab_pct'] = stab_pct

        return status, stats, config_label, (N, seed)

    except Exception as e:
        return f"Error: {str(e)}", None, config_label, (N, seed)

# --- ACTIVITY ANALYSIS (NEW) ---
def analyze_run_activity(file_list):
    """
    Determines if a run is 'Active' or 'Dead' based on the update frequency
    of the last few files.

    Returns: status (str), avg_step_time (float), time_since_last (float)
    """
    if not file_list:
        return "Dead", 0, 0

    # 1. Sort files by modification time
    try:
        # Get (filename, mtime) tuples
        files_with_mtime = [(f, os.path.getmtime(f)) for f in file_list]
        files_with_mtime.sort(key=lambda x: x[1]) # Sort by time ascending
    except OSError:
        # If file access fails
        return "Unknown", 0, 0

    last_mtime = files_with_mtime[-1][1]
    now = time.time()
    time_since_last = now - last_mtime

    # If we have very few files, we can't calculate a trend,
    # but we can check if it started very recently.
    if len(files_with_mtime) < MIN_FILES_FOR_ACTIVITY:
        # If last file is less than 5 minutes old, assume active start
        if time_since_last < 300:
            return "Active", 0, time_since_last
        else:
            return "Dead", 0, time_since_last

    # 2. Get the tail for averaging
    # We want the last ACTIVITY_CHECK_WINDOW + 1 files to get 'WINDOW' intervals
    tail = files_with_mtime[-(ACTIVITY_CHECK_WINDOW + 1):]

    intervals = []
    for i in range(1, len(tail)):
        diff = tail[i][1] - tail[i-1][1]
        intervals.append(diff)

    if not intervals:
        return "Dead", 0, time_since_last

    avg_step_time = sum(intervals) / len(intervals)

    # 3. Determine Threshold
    # Extrapolate: Next step expected at last_mtime + avg_step_time.
    # Allow a tolerance buffer.

    # Ensure avg_step_time is at least non-zero to avoid logic errors
    if avg_step_time < 0.1: avg_step_time = 0.1

    max_wait_time = avg_step_time * ACTIVITY_TIMEOUT_MULTIPLIER

    # Absolute minimum wait of 30 seconds to prevent flagging fast runs that paused briefly
    max_wait_time = max(max_wait_time, 30.0)

    if time_since_last <= max_wait_time:
        return "Active", avg_step_time, time_since_last
    else:
        return "Dead", avg_step_time, time_since_last

# --- MAIN DRIVER ---

def classify_folders(root_dir):
    finished_folders = []
    active_runs = []
    dead_runs = []

    for root, dirs, files in os.walk(root_dir):
        # In the new structure, we only care about 'data' folders
        if os.path.basename(root) != 'data':
            continue

        # Extract Meta to build correct filename pattern
        version, N_meta, seed = extract_metadata(root)
        if N_meta == 0: continue

        # Check file counts
        # Pattern: E{v}_N{n}_S{s}_iter_*_edges.csv
        edge_pattern = os.path.join(root, f"E{version}_N{N_meta}_S{seed}_iter_*_edges.csv")
        node_pattern = os.path.join(root, f"E{version}_N{N_meta}_S{seed}_iter_*_nodes.csv")

        edge_files = glob.glob(edge_pattern)
        node_files = glob.glob(node_pattern)

        if not edge_files and not node_files:
            continue

        e_count = len(edge_files)
        n_count = len(node_files)

        run_info = {
            'path': root,
            'edges': e_count,
            'nodes': n_count
        }

        if e_count == REQUIRED_FILE_COUNT and n_count == REQUIRED_FILE_COUNT:
            finished_folders.append(root)
        else:
            # Analyze Activity for incomplete runs
            status, avg_time, ago = analyze_run_activity(edge_files)
            run_info['avg_step_sec'] = avg_time
            run_info['last_seen_sec'] = ago

            if status == "Active":
                active_runs.append(run_info)
            else:
                dead_runs.append(run_info)

    return finished_folders, active_runs, dead_runs

def main():
    print("="*135)
    print("  RELATIONAL REALITY: STABILITY & DIMENSION ANALYSIS (INCREMENTAL)")
    print("="*135)

    print(f">> Scanning runs/ directory for 'data' folders (Target Files: {REQUIRED_FILE_COUNT})...")
    finished_folders, active_runs, dead_runs = classify_folders(RUNS_DIR)

    if not finished_folders and not active_runs and not dead_runs:
        print("No run folders found.")
        return

    print(f"   Found {len(finished_folders)} finished runs.")
    print(f"   Found {len(active_runs)} ACTIVE runs (currently computing).")
    print(f"   Found {len(dead_runs)} DEAD/INCOMPLETE runs.\n")

    # --- PROCESS FINISHED RUNS ---
    if finished_folders:
        print(f">> Phase 1: Updating metrics for {len(finished_folders)} finished runs...")
        t0 = time.time()
        updated_count = 0
        workers = max(1, os.cpu_count() - 2)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_folder_metrics, f): f for f in finished_folders}
            for i, future in enumerate(as_completed(futures)):
                if i % 10 == 0:
                    print(f"   Processed {i}/{len(finished_folders)}...", end='\r')
                updated_count += future.result()

        print(f"   Done. Calculated {updated_count} new steps in {time.time()-t0:.1f}s.\n")

        print(f">> Phase 2: Stability Check (Threshold {STABILITY_THRESHOLD*100}%)...")

        results = []

        for f in finished_folders:
            status, row, label, meta = check_run_stability(f)
            N, seed = meta

            if row: # Only add valid results
                results.append({
                    'config': label,
                    'N': N,
                    'seed': seed,
                    'status': status,
                    'stats': row
                })

        print("-" * 135)
        # Header
        print(f"{'Version':<20} | {'N':<6} | {'Stb/Tot':<7} | {'<k>':<6} | {'Tri(Fin)':<9} | {'Tri(Pk)':<9} | {'Pk%':<6} | {'Stb%':<6} | {'Haus.'}")
        print("-" * 135)

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            grouped = df_res.groupby(['config', 'N'])

            for (cfg, n_val), group in grouped:
                total = len(group)
                stable_group = group[group['status'] == 'Stable']
                stable_count = len(stable_group)

                if stable_count > 0:
                    # Basic Stats
                    k_vals = [r['mean_degree'] for r in stable_group['stats']]
                    h_vals = [r.get('hausdorff', 0.0) for r in stable_group['stats']]

                    # Triangle Stats
                    t_fin_vals = [r['final_tri'] for r in stable_group['stats']]
                    t_pk_vals = [r['peak_tri'] for r in stable_group['stats']]
                    pk_pct_vals = [r['peak_pct'] for r in stable_group['stats']]
                    stb_pct_vals = [r['stab_pct'] for r in stable_group['stats']]

                    # Formatting
                    count_str = f"{stable_count}/{total}"
                    k_str = f"{np.mean(k_vals):.2f}"
                    h_str = f"{np.mean(h_vals):.2f}±{np.std(h_vals):.2f}"

                    t_fin_str = f"{np.mean(t_fin_vals):.0f}"
                    t_pk_str = f"{np.mean(t_pk_vals):.0f}"

                    pk_pct_str = f"{np.mean(pk_pct_vals):.0f}%"
                    stb_pct_str = f"{np.mean(stb_pct_vals):.0f}%"

                else:
                    count_str = f"0/{total}"
                    k_str = "N/A"; h_str = "N/A"
                    t_fin_str = "N/A"; t_pk_str = "N/A"
                    pk_pct_str = "N/A"; stb_pct_str = "N/A"

                print(f"{cfg:<20} | {n_val:<6} | {count_str:<7} | {k_str:<6} | {t_fin_str:<9} | {t_pk_str:<9} | {pk_pct_str:<6} | {stb_pct_str:<6} | {h_str}")

        print("-" * 135)

        # Detailed Warnings for Finished Runs
        unstable_runs = df_res[df_res['status'] == 'Unstable']
        if not unstable_runs.empty:
            print("\n[WARNING] The following finished runs were excluded due to instability (>1% change):")
            unstable_runs = unstable_runs.sort_values(by=['config', 'N', 'seed'])
            for _, row in unstable_runs.iterrows():
                print(f"  - Ver {row['config']} | N={row['N']} | Seed={row['seed']}")

    # --- PROCESS ACTIVE RUNS (INFO ONLY) ---
    if active_runs:
        print("\n" + "="*80)
        print(f"  ACTIVELY RUNNING ({len(active_runs)})")
        print("="*80)
        print(f"{'Path':<45} | {'Files':<6} | {'LastUpdate':<12} | {'AvgStep':<10}")
        print("-" * 80)

        for run in active_runs:
            disp_path = run['path']
            if len(disp_path) > 43: disp_path = "..." + disp_path[-40:]

            last_ago = f"{run['last_seen_sec']:.1f}s"
            avg_step = f"{run['avg_step_sec']:.1f}s"

            print(f"{disp_path:<45} | {run['edges']:<6} | {last_ago:<12} | {avg_step:<10}")
        print("-" * 80)

    # --- PROCESS DEAD RUNS (DELETE OPTION) ---
    if dead_runs:
        print("\n" + "="*80)
        print(f"  DEAD / INCOMPLETE RUNS DETECTED ({len(dead_runs)})")
        print("="*80)
        print(f"{'Path':<45} | {'Files':<6} | {'LastUpdate':<12} | {'AvgStep':<10}")
        print("-" * 80)

        for run in dead_runs:
            # Shorten path for display
            disp_path = run['path']
            if len(disp_path) > 43: disp_path = "..." + disp_path[-40:]

            last_ago = f"{run['last_seen_sec']:.1f}s"
            avg_step = f"{run['avg_step_sec']:.1f}s"

            print(f"{disp_path:<45} | {run['edges']:<6} | {last_ago:<12} | {avg_step:<10}")

        print("-" * 80)

        # Interactive Deletion
        ans = input(f"\n>> Do you want to DELETE these {len(dead_runs)} DEAD folders? (y/n): ").strip().lower()
        if ans == 'y':
            print("   Deleting (including parent seed folder)...")
            for run in dead_runs:
                try:
                    # run['path'] is '.../S1000/data'. We delete the S1000 folder.
                    seed_dir = os.path.dirname(run['path'])
                    shutil.rmtree(seed_dir)
                    print(f"   Deleted: {seed_dir}")
                except Exception as e:
                    print(f"   Error deleting {seed_dir}: {e}")
            print("   Cleanup complete.")
        else:
            print("   Skipping deletion.")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
