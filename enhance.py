import argparse
import os
import sys
import re
import time
import subprocess
import multiprocessing
import math
import bisect
from datetime import timedelta

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

# --- WORKER FUNCTION (Must be top-level for multiprocessing) ---
def worker_segment(payload):
    """
    Independent worker function.
    Payload: (start_step, target_steps, version, nodes, seed, data_dir)
    """
    start_step, targets, v_id, N, seed, data_dir = payload

    try:
        # Initialize separate engine instance for this process
        engine = drive.PhysicsEngine(N, seed)

        # Load state
        drive.load_engine_state(engine, start_step, v_id, N, seed, data_dir)

        current_sim_step = start_step
        generated_count = 0

        for t in targets:
            steps_to_go = t - current_sim_step
            if steps_to_go > 0:
                for _ in range(steps_to_go):
                    engine.step()
                current_sim_step = t

            # Export
            drive.export_snapshot(engine.G, current_sim_step, v_id, N, seed, data_dir)
            generated_count += 1

        return generated_count
    except Exception as e:
        return f"Error in thread starting at {start_step}: {str(e)}"

# --- VALIDATION ---
def validate_launch_pads(data_dir, steps_to_check, version_id, N, seed):
    """
    Performs a consistency check on the snapshots we intend to use as start points.
    Calculates metrics to ensure files aren't corrupt.
    """
    print(f"\n>> Validating {len(steps_to_check)} Launch Pad Snapshots...")
    valid_steps = []

    for step in steps_to_check:
        step_str = drive.fmt_step(step)
        n_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
        e_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")

        try:
            # Reconstruct graph briefly to check integrity
            G = drive.build_graph_from_csv(n_file, e_file)
            k_min, k_avg, k_max, tri = drive.calculate_metrics(G)

            # Basic sanity check
            if len(G.nodes) != N:
                print(f"   [WARN] Step {step}: Node count mismatch ({len(G.nodes)} vs {N}). Skipping.")
                continue

            valid_steps.append(step)

        except Exception as e:
            print(f"   [FAIL] Step {step}: Corrupt or unreadable ({e})")

    return valid_steps

# --- INTERACTIVE WIZARD ---
def get_valid_subfolders(path, prefix):
    if not os.path.exists(path): return []
    ids = []
    for d in os.listdir(path):
        if d.startswith(prefix) and os.path.isdir(os.path.join(path, d)):
            try:
                ids.append(int(d[len(prefix):]))
            except ValueError: continue
    return sorted(ids)

def select_id(ids, name, prefix):
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
            if 0 <= idx < len(ids): return ids[idx]
        except ValueError: pass
        print("Invalid selection.")

def scan_and_select():
    base_runs = "runs"
    if not os.path.exists(base_runs):
        print(f"No 'runs' directory found.")
        sys.exit(0)

    v_id = select_id(get_valid_subfolders(base_runs, "E"), "Engine Version", "E")
    v_path = os.path.join(base_runs, f"E{v_id}")

    n_val = select_id(get_valid_subfolders(v_path, "N"), "System Size", "N")
    n_path = os.path.join(v_path, f"N{n_val}")

    print(f"\nScanning runs for E{v_id} N{n_val}...")
    sims = []
    seeds = get_valid_subfolders(n_path, "S")
    for s_val in seeds:
        data_dir = os.path.join(n_path, f"S{s_val}", "data")
        if os.path.exists(data_dir):
            frame_count = count_frames(data_dir)
            max_step = find_max_step(data_dir, v_id, n_val, s_val)
            if frame_count > 0:
                sims.append({"v": v_id, "n": n_val, "s": s_val, "frames": frame_count, "step": max_step})

    if not sims:
        print("No valid data found.")
        sys.exit(0)

    print("\n" + "="*65)
    print(f"{'#':<4} | {'SEED':<6} | {'FRAMES':<8} | {'MAX STEP':<12} | {'STATUS'}")
    print("-" * 65)
    for i, s in enumerate(sims):
        status = "Sparse"
        if s['frames'] > 150: status = "Dense"
        if s['frames'] > 1000: status = "HD"
        print(f"{i+1:<4} | S{s['s']:<5} | {s['frames']:<8} | {drive.fmt_step(s['step']):<12} | {status}")
    print("="*65)

    target = None
    while True:
        try:
            sel = input(f"\nSelect Run to Enhance (1-{len(sims)}): ").strip()
            idx = int(sel) - 1
            if 0 <= idx < len(sims):
                target = sims[idx]
                break
        except ValueError: pass

    # --- ENHANCEMENT SETTINGS ---
    print("\n--- Enhancement Factor ---")
    print("2 = Double resolution (insert 1 frame between existing)")
    print("4 = Quadruple resolution")
    print("High numbers (e.g. 1000) = Render every single step")
    fac_in = input("Enter Factor [Default: 4]: ").strip()
    factor = float(fac_in) if fac_in else 4.0

    print("\n--- Time Window Selection ---")
    start_in = input("Start Percentage (0-100) [Default: 0]: ").strip()
    start_pct = float(start_in) if start_in else 0.0

    end_in = input("End Percentage (0-100) [Default: 25]: ").strip()
    end_pct = float(end_in) if end_in else 25.0

    if start_pct >= end_pct:
        print(f"Warning: Start ({start_pct}%) >= End ({end_pct}%). Adjusting End to {start_pct+10}%.")
        end_pct = start_pct + 10

    # Threading input
    import multiprocessing
    default_jobs = max(1, multiprocessing.cpu_count() // 2)
    jobs_in = input(f"Threads (Default: {default_jobs}): ").strip()
    jobs = int(jobs_in) if jobs_in else default_jobs

    print("\n🚀 Launching Enhancement Process...")

    cmd = [
        sys.executable, sys.argv[0],
        "-v", str(target['v']), "-N", str(target['n']), "-s", str(target['s']),
        "-f", str(factor),
        "--start", str(start_pct),
        "--end", str(end_pct),
        "-j", str(jobs)
    ]
    subprocess.call(cmd)
    sys.exit(0)

# --- PROGRESS BAR ---
def print_progress(done, total, start_time):
    elapsed = time.time() - start_time
    pct = (done / total) * 100

    # Calculate Rate
    rate = done / elapsed if elapsed > 0 else 0

    # Calculate ETA
    remaining = total - done
    eta_seconds = remaining / rate if rate > 0 else 0
    eta_str = str(timedelta(seconds=int(eta_seconds)))

    # Visual Bar
    bar_len = 30
    filled = int(bar_len * done // total)
    bar = "█" * filled + "-" * (bar_len - filled)

    status = f"\r[{bar}] {pct:5.1f}% | {done}/{total} Frames | {rate:5.2f} f/s | ETA: {eta_str}   "
    sys.stdout.write(status)
    sys.stdout.flush()

# --- MAIN EXECUTION ---
def enhance_run(args):
    base_runs_dir = "runs"
    version_dir = os.path.join(base_runs_dir, f"E{args.version}")
    batch_dir = os.path.join(version_dir, f"N{args.nodes}")
    run_dir = os.path.join(batch_dir, f"S{args.seed}")
    data_dir = os.path.join(run_dir, "data")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # 1. SETUP BOUNDS
    max_step = find_max_step(data_dir, args.version, args.nodes, args.seed)
    if max_step == 0:
        print("Error: No existing data found.")
        return

    # Calculate Integer Steps for the requested Window
    start_step_limit = int(max_step * (args.start / 100.0))
    end_step_limit = int(max_step * (args.end / 100.0))

    if end_step_limit > max_step: end_step_limit = max_step
    if start_step_limit >= end_step_limit:
        print("Error: Start step is after End step.")
        return

    # --- STRIDE CALCULATION ---
    existing_steps = get_existing_steps(data_dir, args.version, args.nodes, args.seed)
    num_existing = len(existing_steps)

    # Calculate current average stride (gap between frames)
    if num_existing > 1:
        current_avg_stride = max_step / (num_existing - 1)
    else:
        current_avg_stride = max_step

    # Calculate target stride based on factor
    target_stride = int(current_avg_stride / args.factor)
    if target_stride < 1:
        target_stride = 1

    print(f"==================================================")
    print(f"ENHANCING RESOLUTION (MULTI-THREADED)")
    print(f"Target:     E{args.version} | N={args.nodes} | Seed={args.seed}")
    print(f"Window:     {args.start}% - {args.end}%")
    print(f"Step Range: {drive.fmt_step(start_step_limit)} -> {drive.fmt_step(end_step_limit)}")
    print(f"Factor:     {args.factor}x (New Stride: Every {target_stride} steps)")
    print(f"Threads:    {args.jobs}")
    print(f"==================================================")

    # 2. PLANNING
    existing_set = set(existing_steps)

    # Identify missing frames
    missing_steps = []

    # We must snap the grid to '0' to ensure consistent frames across different partial runs
    # Find the first multiple of target_stride that is >= start_step_limit
    first_target = math.ceil(start_step_limit / target_stride) * target_stride

    curr = first_target
    while curr <= end_step_limit:
        if curr not in existing_set:
            missing_steps.append(curr)
        curr += target_stride

    if not missing_steps:
        print(">> No new frames needed in this window.")
        # We allow visualization to prompt still
    else:
        # Group by Launch Pad
        tasks = {} # { start_step: [target1, target2...] }

        for target in missing_steps:
            # Find insertion point to get the nearest item to the left (existing snapshot)
            # This allows us to jump to 90% and start enhancing from the 89% checkpoint
            idx = bisect.bisect_right(existing_steps, target)
            if idx == 0:
                start_node = 0
            else:
                start_node = existing_steps[idx-1]

            if start_node not in tasks:
                tasks[start_node] = []
            tasks[start_node].append(target)

        sorted_starts = sorted(tasks.keys())
        print(f">> Found {len(missing_steps)} frames to generate.")
        print(f">> Optimized into {len(sorted_starts)} parallel segments.")

        # 3. CONSISTENCY CHECK
        valid_starts = validate_launch_pads(data_dir, sorted_starts, args.version, args.nodes, args.seed)

        # Build Payloads
        work_payloads = []
        total_frames_planned = 0

        for start in valid_starts:
            if start in tasks:
                targets = sorted(tasks[start])
                work_payloads.append((start, targets, args.version, args.nodes, args.seed, data_dir))
                total_frames_planned += len(targets)

        if not work_payloads:
            print("No valid tasks remaining after consistency check.")
            return

        # 4. EXECUTION
        print(f"\n>> Starting Pool with {args.jobs} workers...")
        t0 = time.time()

        with multiprocessing.Pool(processes=args.jobs) as pool:
            results = pool.imap_unordered(worker_segment, work_payloads)
            frames_completed = 0
            for res in results:
                if isinstance(res, int):
                    frames_completed += res
                    print_progress(frames_completed, total_frames_planned, t0)
                else:
                    sys.stdout.write(f"\n[ERROR] {res}\n")

        sys.stdout.write("\n") # Clear line
        total_time = time.time() - t0

        print(f"==================================================")
        print(f"BATCH COMPLETE")
        print(f"Time Taken:   {drive.fmt_time(total_time)}")
        if total_time > 0:
            print(f"Avg Speed:    {total_frames_planned/total_time:.2f} f/s")

    # 5. VISUALIZATION PROMPT
    print(f"==================================================")
    if os.path.exists("visualize.py"):
        try:
            choice = input(f"Launch Visualization for this run? [Y/n]: ").strip().lower()
            if choice != 'n':
                print(f"\n>> 🚀 Launching visualize.py ...")
                vis_cmd = [
                    sys.executable, "visualize.py",
                    "--version", str(args.version),
                    "--N", str(args.nodes),
                    "--seed", str(args.seed),
                    "--threads", str(args.jobs)
                ]
                subprocess.call(vis_cmd)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n[WARN] Could not launch visualizer: {e}")
    else:
        print("\n[NOTE] visualize.py not found. Skipping visualization.")

def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-N", "--nodes", type=int, default=drive.DEFAULT_N)
        parser.add_argument("-s", "--seed", type=int, default=drive.DEFAULT_SEED)
        parser.add_argument("-v", "--version", type=int, required=True)
        parser.add_argument("-f", "--factor", type=float, default=2.0, help="Density multiplier")

        # New Window Arguments
        parser.add_argument("--start", type=float, default=0.0, help="Start percentage (0-100)")
        parser.add_argument("--end", type=float, default=25.0, help="End percentage (0-100)")

        default_jobs = max(1, multiprocessing.cpu_count() // 2)
        parser.add_argument("-j", "--jobs", type=int, default=default_jobs,
                           help=f"Number of threads (default: {default_jobs})")

        args = parser.parse_args()
        enhance_run(args)
    else:
        scan_and_select()

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows safety
    main()
