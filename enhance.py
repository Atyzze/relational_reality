import argparse
import os
import sys
import re
import time
import subprocess

# Import drive to reuse logic and defaults
try:
    import drive
except ImportError:
    print("Error: drive.py must be in the same directory.")
    sys.exit(1)

# --- HELPER: FILE SCANNING ---
def count_frames(data_dir):
    """Counts .csv files to estimate current resolution density."""
    if not os.path.exists(data_dir):
        return 0
    return len([f for f in os.listdir(data_dir) if f.endswith("_nodes.csv")])

def find_max_step(data_dir, version_id, N, seed):
    """Finds the latest step recorded."""
    steps = get_existing_steps(data_dir, version_id, N, seed)
    return steps[-1] if steps else 0

def get_existing_steps(data_dir, version_id, N, seed):
    """Scans the data directory and returns a sorted list of steps that already exist."""
    steps = set()
    pattern = re.compile(rf"E{version_id}_N{N}_S{seed}_iter_([\d_]+)_nodes\.csv")

    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            match = pattern.match(fname)
            if match:
                step_str = match.group(1).replace('_', '')
                steps.add(int(step_str))
    return sorted(list(steps))

# --- INTERACTIVE WIZARD ---
def get_valid_subfolders(path, prefix):
    """Returns sorted list of numerical IDs from folder names (e.g. 'E1' -> 1)."""
    if not os.path.exists(path): return []
    ids = []
    for d in os.listdir(path):
        if d.startswith(prefix) and os.path.isdir(os.path.join(path, d)):
            try:
                ids.append(int(d[len(prefix):]))
            except ValueError:
                continue
    return sorted(ids)

def select_id(ids, name, prefix):
    """Generic selector: Auto-selects if 1 option, else asks user."""
    if not ids:
        print(f"No {name}s found.")
        sys.exit(0)

    if len(ids) == 1:
        print(f">> Auto-selecting only {name}: {prefix}{ids[0]}")
        return ids[0]

    print(f"\n--- Available {name}s ---")
    for i, val in enumerate(ids):
        print(f"[{i+1}] {prefix}{val}")

    while True:
        try:
            choice = input(f"Select {name} (1-{len(ids)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(ids):
                return ids[idx]
        except ValueError:
            pass
        print("Invalid selection.")

def scan_and_select():
    base_runs = "runs"
    if not os.path.exists(base_runs):
        print(f"No 'runs' directory found in {os.getcwd()}")
        sys.exit(0)

    # 1. SELECT VERSION
    versions = get_valid_subfolders(base_runs, "E")
    v_id = select_id(versions, "Engine Version", "E")

    v_path = os.path.join(base_runs, f"E{v_id}")

    # 2. SELECT NODES
    nodes = get_valid_subfolders(v_path, "N")
    n_val = select_id(nodes, "System Size", "N")

    n_path = os.path.join(v_path, f"N{n_val}")

    # 3. SCAN SEEDS FOR SELECTED CONFIG
    print(f"\nScanning runs for E{v_id} N{n_val}...")

    sims = []
    # Scan S folders
    seeds = get_valid_subfolders(n_path, "S")

    for s_val in seeds:
        data_dir = os.path.join(n_path, f"S{s_val}", "data")
        if os.path.exists(data_dir):
            frame_count = count_frames(data_dir)
            max_step = find_max_step(data_dir, v_id, n_val, s_val)
            if frame_count > 0:
                sims.append({
                    "v": v_id, "n": n_val, "s": s_val,
                    "frames": frame_count, "step": max_step
                })

    if not sims:
        print("No valid data found in these runs.")
        sys.exit(0)

    # 4. DISPLAY TABLE
    print("\n" + "="*65)
    print(f"{'#':<4} | {'SEED':<6} | {'FRAMES':<8} | {'MAX STEP':<12} | {'STATUS'}")
    print("-" * 65)

    for i, s in enumerate(sims):
        status = "Sparse"
        if s['frames'] > 150: status = "Dense"
        if s['frames'] > 1000: status = "HD"

        print(f"{i+1:<4} | S{s['s']:<5} | {s['frames']:<8} | {drive.fmt_step(s['step']):<12} | {status}")
    print("="*65)

    # 5. USER INPUT FOR RUN
    target = None
    while True:
        try:
            sel = input(f"\nSelect Run to Enhance (1-{len(sims)}): ").strip()
            idx = int(sel) - 1
            if 0 <= idx < len(sims):
                target = sims[idx]
                break
        except ValueError: pass
        print("Invalid selection.")

    print(f"\nSelected: E{target['v']} N{target['n']} S{target['s']}")

    # 6. GET PARAMETERS
    res_in = input("Target Resolution (1-100) [Default: 20]: ").strip()
    res = float(res_in) if res_in else 20.0

    win_in = input("Window Percentage (0-100) [Default: 25]: ").strip()
    win = float(win_in) if win_in else 25.0

    # 7. SPAWN PROCESS
    print("\n🚀 Launching Enhancement Process...")
    print("-" * 40)

    cmd = [
        sys.executable, sys.argv[0],
        "-v", str(target['v']),
        "-N", str(target['n']),
        "-s", str(target['s']),
        "-r", str(res),
        "-w", str(win)
    ]

    subprocess.call(cmd)
    sys.exit(0)

# --- WORKER LOGIC ---
def enhance_run(args):
    # 1. SETUP PATHS
    base_runs_dir = "runs"
    version_dir = os.path.join(base_runs_dir, f"E{args.version}")
    batch_dir = os.path.join(version_dir, f"N{args.nodes}")
    run_dir = os.path.join(batch_dir, f"S{args.seed}")
    data_dir = os.path.join(run_dir, "data")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # 2. CALCULATE PARAMETERS
    # FIX: Do not calculate based on ratio. Detect actual simulation bounds from disk.
    max_step = find_max_step(data_dir, args.version, args.nodes, args.seed)

    if max_step == 0:
        print(f"Error: No existing data found (Max Step = 0). Cannot enhance an empty run.")
        return

    total_sim_steps = max_step

    # Calculate Cutoff Step based on existing data
    window_pct = max(0.0, min(100.0, args.window)) / 100.0
    cutoff_step = int(total_sim_steps * window_pct)

    # Calculate Stride
    min_frames = 100
    max_frames = total_sim_steps
    user_res = max(1.0, min(100.0, args.resolution))

    if user_res == 1.0:
        target_total_frames = min_frames
    elif user_res == 100.0:
        target_total_frames = max_frames
    else:
        pct = (user_res - 1.0) / 99.0
        target_total_frames = int(min_frames + (max_frames - min_frames) * pct)

    # Safety div check
    if target_total_frames <= 0: target_total_frames = 1

    target_stride = max(1, int(total_sim_steps / target_total_frames))

    print(f"==================================================")
    print(f"ENHANCING RESOLUTION")
    print(f"Target:     E{args.version} | N={args.nodes} | Seed={args.seed}")
    print(f"Range:      Existing Data Max Step: {drive.fmt_step(total_sim_steps)}")
    print(f"Window:     First {args.window}% (Steps 0 - {drive.fmt_step(cutoff_step)})")
    print(f"Resolution: {user_res:.1f}/100")
    print(f"Stride:     Save every {target_stride} steps")
    print(f"==================================================\n")

    # 3. SCAN & PLAN
    existing_steps = get_existing_steps(data_dir, args.version, args.nodes, args.seed)
    existing_set = set(existing_steps)

    if not existing_steps:
        print("Error: No data in folder.")
        return

    desired_steps = []
    curr = 0
    while curr <= cutoff_step:
        if curr not in existing_set:
            desired_steps.append(curr)
        curr += target_stride

    if not desired_steps:
        print(">> No new frames needed in the specified window/resolution.")
        return

    print(f">> Found {len(desired_steps)} missing frames to generate.")

    # Group by nearest previous snapshot
    work_plan = {}
    for target in desired_steps:
        start_node = 0
        for s in existing_steps:
            if s <= target:
                start_node = s
            else:
                break

        if start_node not in work_plan:
            work_plan[start_node] = []
        work_plan[start_node].append(target)

    sorted_starts = sorted(work_plan.keys())

    # 4. EXECUTE
    engine = drive.PhysicsEngine(args.nodes, args.seed)
    total_generated = 0
    t0_global = time.time()

    for start_step in sorted_starts:
        targets = sorted(work_plan[start_step])
        final_target = targets[-1]

        print(f"   [Load {drive.fmt_step(start_step)}] -> Simulating to {drive.fmt_step(final_target)} ({len(targets)} frames)...")

        drive.load_engine_state(engine, start_step, args.version, args.nodes, args.seed, data_dir)
        current_sim_step = start_step

        for t in targets:
            steps_to_go = t - current_sim_step
            if steps_to_go > 0:
                for _ in range(steps_to_go):
                    engine.step()
                current_sim_step = t

            drive.export_snapshot(engine.G, current_sim_step, args.version, args.nodes, args.seed, data_dir)
            total_generated += 1

    total_time = time.time() - t0_global
    print(f"\n==================================================")
    print(f"ENHANCEMENT COMPLETE")
    print(f"Frames Added: {total_generated}")
    print(f"Time Taken:   {drive.fmt_time(total_time)}")
    print(f"==================================================")

def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Relational Reality Resolution Enhancer")
        parser.add_argument("-N", "--nodes", type=int, default=drive.DEFAULT_N)
        parser.add_argument("-s", "--seed", type=int, default=drive.DEFAULT_SEED)
        parser.add_argument("-v", "--version", type=int, required=True)
        parser.add_argument("-r", "--resolution", type=float, default=20.0)
        parser.add_argument("-w", "--window", type=float, default=25.0)
        args = parser.parse_args()

        enhance_run(args)
    else:
        scan_and_select()

if __name__ == "__main__":
    main()
