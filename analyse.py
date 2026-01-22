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
METRICS_FILE = "metrics.csv"
STABILITY_WINDOW = 10     # Look at last 10 snapshots for final validation
STABILITY_THRESHOLD = 0.01 # 1% change allowed
PROBE_COUNT = 100           # Keep low to speed up Hausdorff
REQUIRED_FILE_COUNT = 101   # Exact number of steps (0-100) required to be considered "Finished"

# --- WARNING SUPPRESSION FIX ---
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning

warnings.simplefilter('ignore', RankWarning)
warnings.filterwarnings("ignore", module="numpy")

# --- HELPER: CONFIG HASHING ---
def get_config_signature(folder_path):
    """Generates a short hash based on engine.py and physics_parameters.json."""
    engine_path = os.path.join(folder_path, "engine.py")

    if not os.path.exists(engine_path):
        engine_path = os.path.join(os.path.dirname(folder_path), "engine.py")

    hasher = hashlib.sha256()

    if os.path.exists(engine_path):
        with open(engine_path, 'rb') as f:
            hasher.update(f.read())
    else:
        hasher.update(b"NO_ENGINE")

    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            hasher.update(f.read())

    return hasher.hexdigest()[:8]

def extract_metadata(folder_path):
    """Extract N and Seed from folder string."""
    n_match = re.search(r'_N(\d+)', folder_path)
    N = int(n_match.group(1)) if n_match else 0

    s_match = re.search(r'seed_(\d+)', folder_path)
    seed = int(s_match.group(1)) if s_match else 0

    ts_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', folder_path)
    timestamp = ts_match.group(1) if ts_match else "Unknown_Date"

    return N, seed, timestamp

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

def verify_step_integrity(folder_path, step, expected_edges):
    """Checks if the file on disk matches the expected edge count in the CSV."""
    fpath = os.path.join(folder_path, f"edges_step_{step}.csv")
    if not os.path.exists(fpath):
        return False
    try:
        df = pd.read_csv(fpath)
        # Check simple edge count (rows in edge file)
        # Note: Depending on file format, empty might mean header only or 0 bytes
        actual_edges = len(df)
        return actual_edges == expected_edges
    except:
        return False

def calculate_single_step(folder_path, step):
    """Calculates metrics for a single step."""
    fpath = os.path.join(folder_path, f"edges_step_{step}.csv")
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
    metrics_path = os.path.join(folder_path, METRICS_FILE)

    # 1. Identify all available steps on disk
    edge_files = glob.glob(os.path.join(folder_path, "edges_step_*.csv"))
    disk_steps = []
    for f in edge_files:
        m = re.search(r'edges_step_(\d+).csv', f)
        if m:
            disk_steps.append(int(m.group(1)))
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
                    if not verify_step_integrity(folder_path, step_check, edges_check):
                        valid_history = False
                        break

                if valid_history:
                    last_recorded_step = int(existing_df.iloc[-1]['step'])
                    # We start calculating from the next step after the last valid one
                    # Logic: Identify disk steps strictly greater than last recorded
                    start_calculation_from = last_recorded_step + 1
                else:
                    # Corruption detected or mismatch: Recalculate everything
                    existing_df = pd.DataFrame()
                    start_calculation_from = 0
        except Exception:
             # Read error: Recalculate everything
            existing_df = pd.DataFrame()
            start_calculation_from = 0

    # 3. Determine tasks
    steps_to_process = [s for s in disk_steps if s >= start_calculation_from]

    if not steps_to_process:
        return 0 # Nothing new to do

    # 4. Compute Metrics for new steps
    new_rows = []
    for step in steps_to_process:
        row = calculate_single_step(folder_path, step)
        if row:
            new_rows.append(row)

    # 5. Append or Overwrite
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if not existing_df.empty:
            # Append mode (header=False)
            df_new.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            # Write mode (header=True)
            df_new.to_csv(metrics_path, mode='w', header=True, index=False)

    return len(new_rows)

# --- PHASE 2: TIME SERIES & STABILITY CHECK ---

def check_run_stability(folder_path):
    metrics_path = os.path.join(folder_path, METRICS_FILE)
    N, seed, ts = extract_metadata(folder_path)
    cfg_hash = get_config_signature(folder_path)

    if not os.path.exists(metrics_path):
        return "No Metrics", None, cfg_hash, (N, seed, ts)

    try:
        df = pd.read_csv(metrics_path)
        if len(df) < STABILITY_WINDOW:
            return "Insufficient Data", None, cfg_hash, (N, seed, ts)

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

        pct_change = 0.0 if window_mean == 0 else (window_max - window_min) / window_mean
        status = "Unstable" if pct_change > STABILITY_THRESHOLD else "Stable"

        stats = df.iloc[-1].to_dict()
        stats['peak_tri'] = peak_val
        stats['peak_pct'] = peak_pct
        stats['final_tri'] = final_val
        stats['stab_pct'] = stab_pct

        return status, stats, cfg_hash, (N, seed, ts)

    except Exception as e:
        return f"Error: {str(e)}", None, cfg_hash, (N, seed, ts)

# --- MAIN DRIVER ---

def classify_folders(root_dir):
    finished_folders = []
    incomplete_runs = []

    for root, dirs, files in os.walk(root_dir):
        # We look for a folder that contains at least one step file
        edge_files = glob.glob(os.path.join(root, "edges_step_*.csv"))
        node_files = glob.glob(os.path.join(root, "nodes_step_*.csv"))

        if not edge_files and not node_files:
            continue

        e_count = len(edge_files)
        n_count = len(node_files)

        if e_count == REQUIRED_FILE_COUNT and n_count == REQUIRED_FILE_COUNT:
            finished_folders.append(root)
        else:
            incomplete_runs.append({
                'path': root,
                'edges': e_count,
                'nodes': n_count
            })

    return finished_folders, incomplete_runs

def main():
    print("="*135)
    print("  RELATIONAL REALITY: STABILITY & DIMENSION ANALYSIS (INCREMENTAL)")
    print("="*135)

    print(f">> Scanning folders (Definition of Finished: Exactly {REQUIRED_FILE_COUNT} edge & node files)...")
    finished_folders, incomplete_runs = classify_folders(RUNS_DIR)

    if not finished_folders and not incomplete_runs:
        print("No run folders found.")
        return

    print(f"   Found {len(finished_folders)} finished runs and {len(incomplete_runs)} incomplete runs.\n")

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
        hash_to_date = {}

        for f in finished_folders:
            status, row, h, meta = check_run_stability(f)
            N, seed, date = meta

            if h not in hash_to_date: hash_to_date[h] = date
            elif date < hash_to_date[h]: hash_to_date[h] = date

            results.append({
                'hash': h, 'N': N, 'seed': seed,
                'status': status, 'stats': row
            })

        print("-" * 135)
        # Header
        print(f"{'Config':<20} | {'N':<6} | {'Stb/Tot':<7} | {'<k>':<6} | {'Tri(Fin)':<9} | {'Tri(Pk)':<9} | {'Pk%':<6} | {'Stb%':<6} | {'Haus.'}")
        print("-" * 135)

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            df_res['config_label'] = df_res['hash'].map(hash_to_date)
            grouped = df_res.groupby(['config_label', 'N'])

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
            unstable_runs = unstable_runs.sort_values(by=['config_label', 'N', 'seed'])
            for _, row in unstable_runs.iterrows():
                print(f"  - Config {row['config_label']} | N={row['N']} | Seed={row['seed']}")

    # --- PROCESS INCOMPLETE RUNS ---
    if incomplete_runs:
        print("\n" + "="*80)
        print(f"  INCOMPLETE RUNS DETECTED ({len(incomplete_runs)})")
        print("="*80)
        print(f"{'Path':<50} | {'Edges':<8} | {'Nodes':<8}")
        print("-" * 80)

        for run in incomplete_runs:
            # Shorten path for display
            disp_path = run['path']
            if len(disp_path) > 48: disp_path = "..." + disp_path[-45:]
            print(f"{disp_path:<50} | {run['edges']:<8} | {run['nodes']:<8}")

        print("-" * 80)

        # Interactive Deletion
        ans = input(f"\n>> Do you want to DELETE these {len(incomplete_runs)} incomplete folders? (y/n): ").strip().lower()
        if ans == 'y':
            print("   Deleting...")
            for run in incomplete_runs:
                try:
                    shutil.rmtree(run['path'])
                    print(f"   Deleted: {run['path']}")
                except Exception as e:
                    print(f"   Error deleting {run['path']}: {e}")
            print("   Cleanup complete.")
        else:
            print("   Skipping deletion.")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
