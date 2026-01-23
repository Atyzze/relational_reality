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
from engine import PhysicsEngine

# --- DEFAULTS ---
DEFAULT_N = 100
DEFAULT_RATIO = 10000
DEFAULT_SEED = 1000
DEFAULT_RUNS = 10

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
    Scans for E0, E1, E2...
    """
    if not os.path.exists(runs_dir):
        return 0, True, None

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

    next_id = existing_versions[-1] + 1 if existing_versions else 0
    prev_id = existing_versions[-1] if existing_versions else None

    return next_id, True, prev_id

def get_diff_content(file_base, file_curr):
    """Generates the unified diff content as a string (excluding headers)."""
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
    """Generates a text file listing line changes and specific differences."""
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
    # Matches: E0_N100_S1000_iter_000_000_000_nodes.csv
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

def load_engine_state(engine, step, version_id, N, seed, data_dir):
    """Reads CSVs for a specific step and populates the engine state."""
    step_str = fmt_step(step)

    # Updated paths: ..._iter_000_000_000_nodes.csv
    node_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
    edge_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        raise FileNotFoundError(f"Missing snapshot files for step {step}")

    # 1. Load Nodes (Psi)
    with open(node_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if row:
                nid = int(row[0])
                if nid < engine.N:
                    engine.psi[nid] = complex(float(row[1]), float(row[2]))

    # 2. Load Edges (Adj + Theta)
    engine.adj_matrix[:] = False
    engine.theta_matrix[:] = 0.0
    edge_count = 0

    with open(edge_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if row:
                u, v = int(row[0]), int(row[1])
                theta = float(row[2])

                engine.adj_matrix[u, v] = True
                engine.adj_matrix[v, u] = True
                engine.theta_matrix[u, v] = theta
                engine.theta_matrix[v, u] = theta
                edge_count += 1

    # 3. Update internal tracker
    engine.E_tracker[0] = edge_count

def run_simulation(version_id, N, step_ratio, seed, data_output_dir, run_idx, total_runs, batch_start_time, start_step=0):
    t0 = time.time()
    total_steps = N * step_ratio

    snapshot_interval = int(total_steps / 100)
    if snapshot_interval < 1: snapshot_interval = 1

    engine = PhysicsEngine(N, seed)

    # RESUME or START
    if start_step > 0:
        load_engine_state(engine, start_step, version_id, N, seed, data_output_dir)
        current_step = start_step + 1
    else:
        export_snapshot(engine.G, 0, version_id, N, seed, data_output_dir)
        current_step = 1

    # Main Physics Loop
    for t in range(current_step, total_steps + 1):
        engine.step()

        if t % snapshot_interval == 0:
            now = time.time()
            run_elapsed = now - t0

            pct = (t / total_steps) * 100
            steps_processed_session = t - start_step
            sps = steps_processed_session / run_elapsed if run_elapsed > 0.001 else 0.0

            steps_remaining_run = total_steps - t
            run_eta = steps_remaining_run / sps if sps > 0 else 0

            runs_remaining = total_runs - (run_idx + 1)
            total_steps_remaining_batch = steps_remaining_run + (runs_remaining * total_steps)
            batch_eta = total_steps_remaining_batch / sps if sps > 0 else 0

            print(
                f"E{version_id} "
                f"N{N} "
                f"Seed {seed} "
                f"[{run_idx+1}/{total_runs}]"
                f"[{pct:>3.0f}%] "
                f"Spd: {sps:>5.0f} stp/s | "
                f"ETA: {fmt_time(run_eta)} | "
                f"Batch ETA: {fmt_time(batch_eta)}"
            )

            export_snapshot(engine.G, t, version_id, N, seed, data_output_dir)

    dt = time.time() - t0
    return dt

def export_snapshot(G, step, version_id, N, seed, output_dir):
    step_str = fmt_step(step)

    # Updated paths: ..._iter_000_000_000_nodes.csv
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
    parser.add_argument("-R", "--ratio", type=int, default=DEFAULT_RATIO, help=f"Steps/Node (default: {DEFAULT_RATIO})")
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help=f"Starting Seed (default: {DEFAULT_SEED})")
    parser.add_argument("-c", "--count", type=int, default=DEFAULT_RUNS, help=f"Number of runs/seeds (default: {DEFAULT_RUNS})")
    parser.add_argument("-v", "--version", type=int, help="Force run on specific engine version ID (re-run/mod support)")

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
    print(f"Config:    N={args.nodes} | Ratio={args.ratio} | Count={args.count}")
    print(f"Output:    {batch_dir}")
    print(f"==================================================\n")

    batch_start_time = time.time()
    total_steps = args.nodes * args.ratio

    for i in range(args.count):
        current_seed = args.seed + i
        run_dir = os.path.join(batch_dir, f"S{current_seed}")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Check for completion
        final_step_formatted = fmt_step(total_steps)
        # Updated check path
        final_file = os.path.join(data_dir, f"E{version_id}_N{args.nodes}_S{current_seed}_iter_{final_step_formatted}_nodes.csv")

        if os.path.exists(final_file):
             print(f">> Run {i+1}/{args.count} (Seed {current_seed}) [SKIPPED] - Complete.")
             continue

        # Check for Resume
        last_found_step = find_latest_snapshot_step(data_dir, version_id, args.nodes, current_seed)

        snapshot_interval = int(total_steps / 100)
        if snapshot_interval < 1: snapshot_interval = 1

        start_step = 0
        status_msg = "[STARTED]"

        if last_found_step > 0:
            safe_step = last_found_step - (snapshot_interval * 2)
            if safe_step < 0:
                safe_step = 0

            start_step = safe_step
            if start_step > 0:
                status_msg = f"[RESUMING] from step {start_step} (found {last_found_step})"
            else:
                status_msg = f"[RESTARTING] (found {last_found_step} but rewound to 0)"

        print(f">> Run {i+1}/{args.count} (Seed {current_seed}) {status_msg}")

        try:
            duration = run_simulation(
                version_id,
                args.nodes, args.ratio, current_seed,
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
