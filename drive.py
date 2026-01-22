import argparse
import os
import time
import datetime
import shutil
import csv
import json
import hashlib
import numpy as np
from engine import PhysicsEngine

# --- DEFAULTS ---
DEFAULT_N = 100
DEFAULT_RATIO = 100000
DEFAULT_SEED = 1
DEFAULT_RUNS = 100

def fmt_time(seconds):
    """Helper to format seconds into HH:MM:SS string."""
    return str(datetime.timedelta(seconds=int(seconds)))

def get_file_hash(filepath):
    """Calculates SHA256 hash of a file to uniquely identify engine state."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot hash file: {filepath} does not exist.")

    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:12] # Return first 12 chars for brevity

def save_physics_config(engine, output_dir):
    """
    Extracts physics constants from the engine and saves them to JSON.
    """
    config = {}
    for key, value in engine.__dict__.items():
        # Heuristic: Physics constants are UPPERCASE or specific scalars
        if key.isupper() or key in ['TEMP', 'TEMP_SCALE']:
            if key == 'G': continue # Skip graph object

            # Convert Numpy types to Python native types
            if isinstance(value, (np.integer, np.int64, np.int32)):
                value = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = value.tolist()

            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config[key] = value

    config_path = os.path.join(output_dir, "physics_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def run_simulation(N, step_ratio, seed, run_output_dir, run_idx, total_runs, batch_start_time):
    t0 = time.time()
    total_steps = N * step_ratio

    # Snapshot interval: 1%
    snapshot_interval = int(total_steps / 100)
    if snapshot_interval < 1: snapshot_interval = 1

    engine = PhysicsEngine(N, seed)

    # 0. Export Initial State (0%)
    export_snapshot(engine.G, 0, run_output_dir)

    # Main Physics Loop
    for t in range(1, total_steps + 1):
        engine.step()

        # Check Intervals
        if t % snapshot_interval == 0:
            now = time.time()

            # --- TIMING CALCULATIONS ---
            run_elapsed = now - t0
            batch_elapsed = now - batch_start_time

            pct = (t / total_steps) * 100
            sps = t / run_elapsed if run_elapsed > 0.001 else 0.0
            steps_remaining_run = total_steps - t
            run_eta = steps_remaining_run / sps if sps > 0 else 0
            next_pct_eta = snapshot_interval / sps if sps > 0 else 0

            # Global Batch ETA
            total_job_steps = total_runs * total_steps
            current_job_steps = (run_idx * total_steps) + t

            if current_job_steps > 0:
                batch_eta = (batch_elapsed / current_job_steps) * (total_job_steps - current_job_steps)
            else:
                batch_eta = 0

            print(
                f"N{N} "
                f"Seed {seed} "
                f"[{run_idx+1}/{total_runs}]"
                f"[{pct:>3.0f}%] "
                f"Spd: {sps:>5.0f} stp/s | "
                f"Run ETA: {fmt_time(run_eta)} | "
                f"1% ETA: {next_pct_eta:>4.2f}s | "
                f"Batch ETA: {fmt_time(batch_eta)} | "
                f"Elapsed: {fmt_time(batch_elapsed)}"
            )

            export_snapshot(engine.G, t, run_output_dir)

    dt = time.time() - t0
    return dt

def export_snapshot(G, step, output_dir):
    # Nodes
    node_file = os.path.join(output_dir, f"nodes_step_{step}.csv")
    with open(node_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "psi_real", "psi_imag"])
        for n in G.nodes():
            psi = G.nodes[n]["psi"]
            writer.writerow([n, psi.real, psi.imag])

    # Edges
    edge_file = os.path.join(output_dir, f"edges_step_{step}.csv")
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

    args = parser.parse_args()

    # 1. IDENTIFY ENGINE VERSION
    engine_file = "engine.py"
    if not os.path.exists(engine_file):
        print("Error: engine.py not found in current directory.")
        return

    engine_hash = get_file_hash(engine_file)

    # 2. SETUP DIRECTORY STRUCTURE
    # /runs/{hash}/
    base_run_dir = os.path.join("runs", engine_hash)
    os.makedirs(base_run_dir, exist_ok=True)

    # Save copy of engine if it's the first time seeing this hash
    dest_engine_path = os.path.join(base_run_dir, "engine.py")
    if not os.path.exists(dest_engine_path):
        shutil.copy2(engine_file, dest_engine_path)
        print(f"New Engine detected. Saved reference to: {dest_engine_path}")

        # Dump config only if new engine (assuming params are tied to engine code)
        temp_engine = PhysicsEngine(args.nodes, args.seed)
        save_physics_config(temp_engine, base_run_dir)

    # /runs/{hash}/N{count}/
    # We group by N (and Ratio implicitly via the run execution, but folder is N)
    batch_dir = os.path.join(base_run_dir, f"N{args.nodes}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"==================================================")
    print(f"STARTING BATCH")
    print(f"Engine ID: {engine_hash}")
    print(f"Config:    N={args.nodes} | Ratio={args.ratio} | Count={args.count}")
    print(f"Output:    {batch_dir}")
    print(f"==================================================\n")

    batch_start_time = time.time()

    for i in range(args.count):
        current_seed = args.seed + i

        # /runs/{hash}/N{count}/seed_{seed}
        run_name = f"seed_{current_seed}"
        run_dir = os.path.join(batch_dir, run_name)

        # Skip if already exists to allow resuming
        if os.path.exists(run_dir):
            print(f">> Run {i+1}/{args.count} (Seed {current_seed}) ALREADY EXISTS. Skipping.")
            continue

        os.makedirs(run_dir, exist_ok=True)

        print(f">> Run {i+1}/{args.count} (Seed {current_seed}) started...")

        duration = run_simulation(args.nodes, args.ratio, current_seed, run_dir, i, args.count, batch_start_time)

        print(f"   [DONE] Time: {duration:.2f}s\n")

    total_time = time.time() - batch_start_time
    avg_time = total_time / args.count

    print(f"==================================================")
    print(f"BATCH COMPLETE")
    print(f"Total Time: {fmt_time(total_time)}")
    print(f"Avg per Run: {avg_time:.2f}s")
    print(f"==================================================")

if __name__ == "__main__":
    main()
