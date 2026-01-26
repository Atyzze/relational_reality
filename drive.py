import argparse
import os
import time
import datetime
import shutil
import csv
import json
import filecmp
import difflib
import re
import numpy as np
import networkx as nx
from engine import PhysicsEngine


# --- DEFAULTS ---
DEFAULT_N = 100
DEFAULT_SEED = 1000
DEFAULT_RUNS = 10
DEFAULT_INTERVAL = 100_000  # Steps between logs/snapshots/checks

# Replaced fixed window with a minimum snapshot count for safety
MIN_SNAPSHOTS_FOR_STABILITY = 10

# FIX: Cap the lookback window so it doesn't expand infinitely on long runs
MAX_STABILITY_WINDOW_STEPS = 2_000_000
# FIX: Allow slight fluctuation (0.1%) to handle metastability
STABILITY_TOLERANCE = 0.001

def fmt_time(seconds):
    """Helper to format seconds into HH:MM:SS string."""
    return str(datetime.timedelta(seconds=int(seconds)))

def fmt_step(step):
    """
    Helper to format step count.
    - Ensures at least 9 digits (padded with zeros).
    - Includes underscores for readability.
    Example: 1 -> '000_000_001'
    """
    return f"{step:011_d}"

def get_engine_version(engine_file, runs_dir):
    """
    Determines the engine version ID automatically.
    """
    if not os.path.exists(runs_dir):
        return 1, True, None

    existing_versions = []
    for d in os.listdir(runs_dir):
        if d.startswith('E') and d[1:].isdigit():
            full_path = os.path.join(runs_dir, d)
            if os.path.isdir(full_path):
                existing_versions.append(int(d[1:]))

    existing_versions.sort()

    for v_id in existing_versions:
        stored_engine = os.path.join(runs_dir, f"E{v_id}", "engine.py")
        if os.path.exists(stored_engine):
            if filecmp.cmp(engine_file, stored_engine, shallow=False):
                return v_id, False, None

    next_id = existing_versions[-1] + 1 if existing_versions else 1
    prev_id = existing_versions[-1] if existing_versions else None

    return next_id, True, prev_id

def get_diff_content(file_base, file_curr):
    """Generates the unified diff content as a string."""
    with open(file_base, 'r') as f1, open(file_curr, 'r') as f2:
        prev_lines = f1.readlines()
        curr_lines = f2.readlines()

    diff = list(difflib.unified_diff(
        prev_lines, curr_lines,
        fromfile='Base',
        tofile='Current',
        lineterm=''
    ))
    return diff

def generate_engine_diff(current_file, prev_file, output_path):
    """Generates a text file listing line changes."""
    diff = get_diff_content(prev_file, current_file)
    added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

    with open(output_path, 'w') as f:
        f.write(f"--- ENGINE DIFFERENCE REPORT ---\n")
        f.write(f"Compared against version containing: {prev_file}\n")
        f.write(f"Lines Added: {added}\n")
        f.write(f"Lines Removed: {removed}\n")
        f.write(f"Total Lines Changed: {added + removed}\n")
        f.write(f"--------------------------------\n\n")
        f.write("DETAILS:\n")
        for line in diff:
            f.write(line + "\n")

def find_latest_snapshot_step(data_dir, version_id, N, seed):
    """
    Scans directory for E{ver}_N{N}_S{seed}_iter_{step}_nodes.csv
    and returns the largest step found.
    """
    max_step = -1
    pattern = re.compile(rf"E{version_id}_N{N}_S{seed}_iter_([\d_]+)_nodes\.csv")

    if not os.path.exists(data_dir):
        return -1

    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            step_str = match.group(1).replace('_', '')
            step = int(step_str)
            if step > max_step:
                max_step = step
    return max_step

def build_graph_from_csv(node_file, edge_file):
    """Helper to reconstruct a NetworkX graph from CSVs for analysis."""
    G = nx.Graph()

    # We only need connectivity for metrics, so we can ignore Psi details for speed
    # but we need nodes to exist even if unconnected.
    with open(node_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                G.add_node(int(row[0]))

    with open(edge_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                u, v = int(row[0]), int(row[1])
                G.add_edge(u, v)
    return G

def calculate_metrics(G):
    """Returns (k_min, k_avg, k_max, triangles)."""
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return 0, 0.0, 0, 0

    k_min = np.min(degrees)
    k_max = np.max(degrees)
    k_avg = np.mean(degrees)

    tri_dict = nx.triangles(G)
    triangles = sum(tri_dict.values()) // 3

    return k_min, k_avg, k_max, triangles

def check_stability_from_disk(data_dir, version_id, N, seed, latest_step, interval):
    """
    Checks if the Triangle Count has been stable (within tolerance) for the last 1% of total steps
    OR the last MIN_SNAPSHOTS_FOR_STABILITY, capped at MAX_STABILITY_WINDOW_STEPS.

    CRITICAL: Also checks that k_min > 0 (no disconnected nodes) in the latest snapshot.
    """
    # 1. Calculate the required window size
    min_window_span = (MIN_SNAPSHOTS_FOR_STABILITY - 1) * interval

    # FIX: Use 1% (0.01) instead of 5% (0.05) to match runtime logic
    # FIX: Cap the window at MAX_STABILITY_WINDOW_STEPS
    calc_window = min(latest_step * 0.01, MAX_STABILITY_WINDOW_STEPS)
    required_window = max(calc_window, min_window_span)

    threshold_step = latest_step - required_window

    # 2. Collect metrics from snapshots within the window
    history_metrics = []
    current_check_step = latest_step

    # Scan backwards until we hit the threshold or run out of files
    while current_check_step >= threshold_step and current_check_step >= 0:
        step_str = fmt_step(current_check_step)
        node_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
        edge_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")

        # If files don't exist, we can't verify stability
        if not os.path.exists(node_file) or not os.path.exists(edge_file):
            break

        try:
            G = build_graph_from_csv(node_file, edge_file)
            metrics = calculate_metrics(G) # (k_min, k_avg, k_max, triangles)
            history_metrics.append(metrics)
        except Exception:
            break

        current_check_step -= interval

    # 3. Validation
    if len(history_metrics) < MIN_SNAPSHOTS_FOR_STABILITY:
        return False

    # NEW CONDITION: k_min must be > 0 in the LATEST snapshot.
    # history_metrics[0] is the snapshot at 'latest_step'.
    # metrics tuple is (k_min, k_avg, k_max, triangles) -> index 0 is k_min.
    latest_k_min = history_metrics[0][0]
    if latest_k_min == 0:
        # Not fully connected yet, even if triangles are stable.
        return False

    # 4. Check Criteria: Triangle Count (Index 3) must be within tolerance
    tri_values = [m[3] for m in history_metrics]

    val_min = min(tri_values)
    val_max = max(tri_values)

    # Avoid division by zero
    if val_max == 0:
        return True # Stable at 0

    variation = (val_max - val_min) / val_max

    return variation <= STABILITY_TOLERANCE

def load_engine_state(engine, step, version_id, N, seed, data_dir):
    """Reads CSVs for a specific step and populates the engine state."""
    step_str = fmt_step(step)

    node_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
    edge_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        raise FileNotFoundError(f"Missing snapshot files for step {step}")

    with open(node_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                nid = int(row[0])
                if nid < engine.N:
                    engine.psi[nid] = complex(float(row[1]), float(row[2]))

    engine.adj_matrix[:] = False
    engine.theta_matrix[:] = 0.0
    edge_count = 0

    with open(edge_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                u, v = int(row[0]), int(row[1])
                theta = float(row[2])

                engine.adj_matrix[u, v] = True
                engine.adj_matrix[v, u] = True
                engine.theta_matrix[u, v] = theta
                engine.theta_matrix[v, u] = theta
                edge_count += 1

    engine.E_tracker[0] = edge_count

def run_simulation(version_id, N, interval, seed, data_output_dir, run_idx, total_runs, batch_start_time, start_step=0):
    t0 = time.time()
    engine = PhysicsEngine(N, seed)

    # Resume or Start
    if start_step > 0:
        load_engine_state(engine, start_step, version_id, N, seed, data_output_dir)
        current_step = start_step + 1
    else:
        export_snapshot(engine.G, 0, version_id, N, seed, data_output_dir)
        current_step = 1

    t = current_step

    # Stability Tracking
    # Format: list of tuples (step, (k_min, k_avg, k_max, triangles))
    metric_history = []

    while True:
        # 1. Physics Step
        engine.step()

        # 2. Check/Log Interval
        if t % interval == 0:
            now = time.time()
            run_elapsed = now - t0

            # Calculate Metrics
            metrics = calculate_metrics(engine.G)
            k_min, k_avg, k_max, triangles = metrics

            # Log
            steps_processed_session = t - start_step
            sps = steps_processed_session / run_elapsed if run_elapsed > 0.001 else 0.0

            print(
                f"E{version_id:<3}"
                f"N{N:<6}"
                f"{sps:>7.0f}i/s "
                f"S{seed} "
                f"[{run_idx+1:<2}/{total_runs:<2}] "
                f"i:{fmt_step(t)} "
                f"Tri:{triangles:<6} "
                f"k:{k_min:<2}/{k_avg:<6.3f}/{k_max:<2} "
            )

            # Save
            export_snapshot(engine.G, t, version_id, N, seed, data_output_dir)

            # 3. Dynamic Stop Condition
            metric_history.append((t, metrics))

            # Calculate the required window size.
            min_window_span = (MIN_SNAPSHOTS_FOR_STABILITY - 1) * interval

            # FIX: Cap the window at MAX_STABILITY_WINDOW_STEPS
            calc_window = min(t * 0.01, MAX_STABILITY_WINDOW_STEPS)
            required_window = max(calc_window, min_window_span)

            threshold_step = t - required_window

            # Filter history
            relevant_snapshots = [m for s, m in metric_history if s >= threshold_step]

            # Check if we have enough data points (Safeguard)
            if len(relevant_snapshots) >= MIN_SNAPSHOTS_FOR_STABILITY:

                # Check Triangles (Index 3)
                tri_vals = [m[3] for m in relevant_snapshots]
                val_min = min(tri_vals)
                val_max = max(tri_vals)

                variation = 0.0
                if val_max > 0:
                    variation = (val_max - val_min) / val_max

                if variation <= STABILITY_TOLERANCE:
                    # NEW CONDITION: Wait for graph to be connected (k_min > 0)
                    if k_min > 0:
                        steps_covered = t - relevant_snapshots[0][0]
                        print(f"   [STABILIZED] k_min > 0 AND Triangles var {variation:.5f} <= {STABILITY_TOLERANCE} for {steps_covered} steps. Stopping.")
                        break
                    else:
                        # Log that we are waiting for connectivity despite stability
                        # (Only log this occasionally to avoid spam, or just let the standard log handle it)
                        pass
        t += 1

    dt = time.time() - t0
    return dt

def export_snapshot(G, step, version_id, N, seed, output_dir):
    step_str = fmt_step(step)

    node_file = os.path.join(output_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
    with open(node_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "psi_real", "psi_imag"])
        for n in G.nodes():
            psi = G.nodes[n]["psi"]
            writer.writerow([n, psi.real, psi.imag])

    edge_file = os.path.join(output_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")
    with open(edge_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "theta"])
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get("theta", 0.0)])

def main():
    parser = argparse.ArgumentParser(description="Relational Reality Batch Driver")
    parser.add_argument("-N", "--nodes", type=int, default=DEFAULT_N, help=f"Nodes (default: {DEFAULT_N})")
    parser.add_argument("-I", "--interval", type=int, default=DEFAULT_INTERVAL, help=f"Snapshot/Check Interval (default: {DEFAULT_INTERVAL})")
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help=f"Starting Seed (default: {DEFAULT_SEED})")
    parser.add_argument("-c", "--count", type=int, default=DEFAULT_RUNS, help=f"Number of runs/seeds (default: {DEFAULT_RUNS})")
    parser.add_argument("-v", "--version", type=int, help="Force run on specific engine version ID")

    args = parser.parse_args()
    engine_file = "engine.py"
    if not os.path.exists(engine_file):
        print("Error: engine.py not found.")
        return

    base_runs_dir = "runs"

    # 1. RESOLVE ENGINE VERSION
    if args.version is not None:
        version_id = args.version
        version_dir = os.path.join(base_runs_dir, f"E{version_id}")
        if not os.path.exists(version_dir):
            print(f"Error: Version directory {version_dir} does not exist.")
            return
        print(f"Force-Targeting Engine Version: {version_id}")
    else:
        version_id, is_new_version, prev_id = get_engine_version(engine_file, base_runs_dir)
        version_dir = os.path.join(base_runs_dir, f"E{version_id}")
        os.makedirs(version_dir, exist_ok=True)
        dest_engine_path = os.path.join(version_dir, "engine.py")

        if is_new_version:
            print(f"New Engine Config Detected. Assigning ID: {version_id}")
            shutil.copy2(engine_file, dest_engine_path)
            if prev_id is not None:
                prev_engine_path = os.path.join(base_runs_dir, f"E{prev_id}", "engine.py")
                diff_path = os.path.join(version_dir, "engine_difference.txt")
                generate_engine_diff(engine_file, prev_engine_path, diff_path)
                print(f"Diff generated against version {prev_id} -> {diff_path}")
        else:
            print(f"Using existing Engine Version: {version_id}")

    # 2. SETUP BATCH DIRECTORY
    batch_dir = os.path.join(version_dir, f"N{args.nodes}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"==================================================")
    print(f"STARTING BATCH")
    print(f"Version:   E{version_id}")
    print(f"Config:    N={args.nodes} | Interval={args.interval} | Count={args.count}")
    print(f"Output:    {batch_dir}")
    print(f"==================================================\n")

    batch_start_time = time.time()

    for i in range(args.count):
        current_seed = args.seed + i
        run_dir = os.path.join(batch_dir, f"S{current_seed}")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # 3. CHECK EXISTING DATA
        last_found_step = find_latest_snapshot_step(data_dir, version_id, args.nodes, current_seed)

        start_step = 0
        status_msg = "[STARTED]"

        # Logic: If data exists, check if it's already stable
        if last_found_step > 0:
            print(f">> Run {i+1}/{args.count} (Seed {current_seed}) Checking stability of existing data...")
            is_stable = check_stability_from_disk(data_dir, version_id, args.nodes, current_seed, last_found_step, args.interval)

            if is_stable:
                print(f">> Run {i+1}/{args.count} (Seed {current_seed}) [SKIPPED] - Already Stabilized at step {last_found_step}.")
                continue

            # If not stable, we resume
            # Safe rewind to ensure we catch the rhythm
            safe_step = last_found_step - (args.interval * 2)
            if safe_step < 0: safe_step = 0
            start_step = safe_step

            if start_step > 0:
                status_msg = f"[RESUMING] from step {start_step} (found {last_found_step})"
            else:
                status_msg = f"[RESTARTING] (found {last_found_step} but rewound)"

        print(f">> Run {i+1}/{args.count} (Seed {current_seed}) {status_msg}")

        try:
            duration = run_simulation(
                version_id,
                args.nodes, args.interval, current_seed,
                data_dir, i, args.count, batch_start_time,
                start_step=start_step
            )
            print(f"   [DONE] Time: {duration:.2f}s\n")
        except Exception as e:
            print(f"   [FAILED] Error: {e}\n")

    total_time = time.time() - batch_start_time
    avg_time = total_time / args.count if args.count > 0 else 0

    print(f"==================================================")
    print(f"BATCH COMPLETE")
    print(f"Total Time: {fmt_time(total_time)}")
    print(f"Avg per Run: {avg_time:.2f}s")
    print(f"==================================================")

if __name__ == "__main__":
    main()
