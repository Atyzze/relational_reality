import argparse
import os
import time
import datetime
import shutil
import csv
import filecmp
import re
import collections
import math
import difflib  # Added for difference reporting
import numpy as np
import networkx as nx
from engine import PhysicsEngine

# --- DEFAULTS ---
DEFAULT_N = 100
DEFAULT_SEED = 1000
DEFAULT_RUNS = 8
DEFAULT_INTERVAL = 100_000

# --- PHYSICS GATES ---
REQUIRED_MIN_DEGREE = 2
STABILITY_STREAK_TARGET = 10
TARGET_K_MEAN = 3
REGRESSION_WINDOW = 20

# --- TEMPERATURE CONTROL ---
TEMP_START = 00
TEMP_DECAY = 1
TEMP_FLOOR = 1e-20

def get_temperature(step):
    t = TEMP_START * (TEMP_DECAY ** step)
    return max(t, TEMP_FLOOR)

def fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "---"
    return str(datetime.timedelta(seconds=int(seconds)))

def fmt_step(step):
    return f"{step:011_d}"

def get_engine_version(engine_file, runs_dir):
    """
    Scans for existing engines. Returns (id, is_new, prev_id).
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

    # Check if current engine matches ANY previous version
    for v_id in existing_versions:
        stored_engine = os.path.join(runs_dir, f"E{v_id}", "engine.py")
        if os.path.exists(stored_engine):
            if filecmp.cmp(engine_file, stored_engine, shallow=False):
                return v_id, False, None

    next_id = existing_versions[-1] + 1 if existing_versions else 1
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
    max_step = -1
    pattern = re.compile(rf"E{version_id}_N{N}_S{seed}_iter_([\d_]+)_nodes\.csv")
    if not os.path.exists(data_dir): return -1
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1).replace('_', ''))
            if step > max_step: max_step = step
    return max_step

def build_graph_from_csv(node_file, edge_file):
    G = nx.Graph()
    if os.path.exists(node_file):
        with open(node_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader: G.add_node(int(row[0]))
    if os.path.exists(edge_file):
        with open(edge_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader: G.add_edge(int(row[0]), int(row[1]))
    return G

def calculate_metrics(G):
    degrees = [d for n, d in G.degree()]
    if not degrees: return 0, 0.0, 0, 0
    k_min, k_max, k_avg = np.min(degrees), np.max(degrees), np.mean(degrees)
    triangles = sum(nx.triangles(G).values()) // 3
    return k_min, k_avg, k_max, triangles

def check_stability_from_disk(data_dir, version_id, N, seed, latest_step, interval):
    required_history = (STABILITY_STREAK_TARGET - 1) * interval
    if latest_step < required_history: return False

    metrics_window = []
    for i in range(STABILITY_STREAK_TARGET):
        step = latest_step - (i * interval)
        nf = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{fmt_step(step)}_nodes.csv")
        ef = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{fmt_step(step)}_edges.csv")
        if not os.path.exists(nf): return False
        try:
            G = build_graph_from_csv(nf, ef)
            metrics_window.append(calculate_metrics(G))
        except: return False

    if not metrics_window: return False
    if metrics_window[0][0] < REQUIRED_MIN_DEGREE: return False
    first_k = int(metrics_window[0][1] * 1000)
    return all(int(m[1] * 1000) == first_k for m in metrics_window)

def load_engine_state(engine, step, version_id, N, seed, data_dir):
    step_str = fmt_step(step)
    nf = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
    ef = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")
    if not os.path.exists(nf): raise FileNotFoundError(f"Missing snapshot {step}")

    with open(nf, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            nid = int(row[0])
            if nid < engine.N: engine.psi[nid] = complex(float(row[1]), float(row[2]))

    engine.adj_matrix[:] = False
    engine.theta_matrix[:] = 0.0
    edge_count = 0
    with open(ef, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            theta = float(row[2])
            engine.adj_matrix[u, v] = True
            engine.adj_matrix[v, u] = True
            engine.theta_matrix[u, v] = theta
            engine.theta_matrix[v, u] = theta
            edge_count += 1
    engine.E_tracker[0] = edge_count

def load_raw_state_metrics(version_id, N, seed, step, data_dir):
    step_str = fmt_step(step)
    nf = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
    ef = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")
    if not os.path.exists(nf) or not os.path.exists(ef): return None
    G = build_graph_from_csv(nf, ef)
    return calculate_metrics(G)

def verify_stitching_and_heal(version_id, N, seed, data_dir, interval, latest_step):
    safe_step = latest_step - interval
    if safe_step < 0: return True, latest_step
    print(f"   [STITCHING] Verifying causality: {safe_step} -> {latest_step} ...")
    test_engine = PhysicsEngine(N, seed)
    try:
        load_engine_state(test_engine, safe_step, version_id, N, seed, data_dir)
    except Exception as e:
        print(f"   [ERROR] Could not load safe step: {e}")
        return False, 0
    current_sim_step = safe_step
    for _ in range(interval):
        test_engine.params[0] = get_temperature(current_sim_step)
        test_engine.step()
        current_sim_step += 1
    calc_m = calculate_metrics(test_engine.G)
    disk_m = load_raw_state_metrics(version_id, N, seed, latest_step, data_dir)
    if disk_m is None: return False, safe_step
    topology_match = (calc_m[3] == disk_m[3])
    physics_match = math.isclose(calc_m[1], disk_m[1], rel_tol=1e-4)
    if topology_match and physics_match:
        print(f"   [VERIFIED] Physics signatures match.")
        return True, latest_step
    else:
        print(f"   [DRIFT] Disk vs Calc mismatch. Discarding step {latest_step}.")
        return False, safe_step

def estimate_total_eta(history, current_elapsed_time):
    if len(history) < 5: return None
    times = np.array([h[0] for h in history])
    ks = np.array([h[1] for h in history])
    try:
        m, c = np.polyfit(times, ks, 1)
        if m <= 0.000001: return None
        t_arrival_12 = (TARGET_K_MEAN - c) / m
        return (t_arrival_12 * 2.0) - current_elapsed_time
    except: return None

def run_simulation(version_id, N, interval, seed, data_output_dir, run_idx, total_runs, start_step=0, log_file=None):
    t0 = time.time()
    f_log = None
    if log_file:
        try:
            f_log = open(log_file, 'w')
            f_log.write(f"# metadata_version={version_id},N={N},seed={seed},start_step={start_step}\n")
            f_log.write("timestamp_iso,step,sps,elapsed_sec,k_min,k_avg,k_max,triangles,temp\n")
        except Exception as e:
            print(f"[WARN] Could not open log file: {e}")

    engine = PhysicsEngine(N, seed)
    if start_step > 0:
        load_engine_state(engine, start_step, version_id, N, seed, data_output_dir)
        t = start_step
    else:
        export_snapshot(engine.G, 0, version_id, N, seed, data_output_dir)
        t = 0

    current_streak = 0
    last_k_long = -1
    trend_history = collections.deque(maxlen=REGRESSION_WINDOW)
    m_init = calculate_metrics(engine.G)
    trend_history.append((0, m_init[1]))

    try:
        while True:
            current_temp = get_temperature(t)
            engine.params[0] = current_temp
            engine.step()
            t += 1
            if t % interval == 0:
                elapsed = time.time() - t0
                metrics = calculate_metrics(engine.G)
                k_min, k_avg, k_max, triangles = metrics
                trend_history.append((elapsed, k_avg))
                k_long = int(k_avg * 1000)
                if k_long == last_k_long: current_streak += 1
                else: current_streak = 1; last_k_long = k_long
                sps = (t - start_step) / elapsed if elapsed > 0 else 0
                if k_avg < TARGET_K_MEAN: status = f">> 12 ({k_avg/TARGET_K_MEAN:.0%})"
                else: status = f"STABL {current_streak}/{STABILITY_STREAK_TARGET}"
                eta = fmt_time(estimate_total_eta(trend_history, elapsed))

                print(f"E{version_id:<3} N{N:<7_} {sps:>10_.0f}i/s {t:>12_.0f} S{seed} [{run_idx+1:<2}/{total_runs:<2}] "
                      f"T:{fmt_time(elapsed)} [{status:<12}] ETA:{eta:<8} Tri:{triangles:<10_} "
                      f"k:{k_min:<2}/{k_avg:<8.5f}/{k_max:<2} Temp:{current_temp:.6f}")

                if f_log:
                    now_iso = datetime.datetime.now().isoformat()
                    f_log.write(f"{now_iso},{t},{sps:.2f},{elapsed:.2f},{k_min},{k_avg:.5f},{k_max},{triangles},{current_temp:.6g}\n")
                    f_log.flush()

                export_snapshot(engine.G, t, version_id, N, seed, data_output_dir)
                if k_min >= REQUIRED_MIN_DEGREE and current_streak >= STABILITY_STREAK_TARGET and k_avg >= (TARGET_K_MEAN - 0.2):
                    print(f"   [CRYSTALLIZED] Run Complete at step {t}")
                    if f_log: f_log.write(f"# COMPLETED at {datetime.datetime.now().isoformat()}\n")
                    break
    finally:
        if f_log: f_log.close()
    return time.time() - t0

def export_snapshot(G, step, version_id, N, seed, output_dir):
    step_str = fmt_step(step)
    with open(os.path.join(output_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "psi_real", "psi_imag"])
        for n in G.nodes():
            psi = G.nodes[n].get("psi", 0j)
            writer.writerow([n, psi.real, psi.imag])
    with open(os.path.join(output_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "theta"])
        for u, v, d in G.edges(data=True):
            writer.writerow([u, v, d.get("theta", 0.0)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--nodes", type=int, default=DEFAULT_N)
    parser.add_argument("-I", "--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-c", "--count", type=int, default=DEFAULT_RUNS)
    parser.add_argument("-v", "--version", type=int)
    args = parser.parse_args()

    base_runs_dir = "runs"

    # --- ENGINE VERSION DETECTION OR SELECTION ---
    if args.version is not None:
        # User forced a specific version
        v_id = args.version
        version_dir = os.path.join(base_runs_dir, f"E{v_id}")
        if not os.path.exists(version_dir):
             print(f"[ERROR] Forced version E{v_id} not found in {base_runs_dir}")
             return
        print(f"Force-Targeting Engine Version: {v_id}")
    else:
        # Auto-detect version
        v_id, is_new, prev_id = get_engine_version("engine.py", base_runs_dir)
        version_dir = os.path.join(base_runs_dir, f"E{v_id}")
        os.makedirs(version_dir, exist_ok=True)

        if is_new:
            shutil.copy2("engine.py", os.path.join(version_dir, "engine.py"))
            print(f"New Engine detected. Using E{v_id}")
            # Generate diff if there was a previous version
            if prev_id is not None:
                prev_engine_path = os.path.join(base_runs_dir, f"E{prev_id}", "engine.py")
                diff_path = os.path.join(version_dir, "engine_difference.txt")
                generate_engine_diff("engine.py", prev_engine_path, diff_path)
                print(f"   [DIFF] Generated report against E{prev_id} -> {diff_path}")
        else:
            print(f"Using existing Engine Version: E{v_id}")

    batch_dir = os.path.join(version_dir, f"N{args.nodes}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"=== BATCH E{v_id} | N={args.nodes} | {args.count} Runs ===")
    print(f"=== MODE: EXTERNAL TEMP CONTROL ===")

    for i in range(args.count):
        seed = args.seed + i
        run_dir = os.path.join(batch_dir, f"S{seed}")
        data_dir = os.path.join(run_dir, "data")
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        last_step = find_latest_snapshot_step(data_dir, v_id, args.nodes, seed)
        start_step = 0
        if last_step > 0:
            print(f">> Run {i+1} (S{seed}) Checking {last_step}...")
            if check_stability_from_disk(data_dir, v_id, args.nodes, seed, last_step, args.interval):
                print(f"   [SKIPPED] Already Stable.")
                continue
            valid, healed_step = verify_stitching_and_heal(v_id, args.nodes, seed, data_dir, args.interval, last_step)
            if valid: start_step = healed_step; print(f"   [RESUMING] from step {start_step}")
            else: start_step = healed_step
        else:
            print(f">> Run {i+1} (S{seed}) [STARTING]")

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M_%S")
        log_path = os.path.join(logs_dir, f"{timestamp_str}.csv")

        try:
            run_simulation(v_id, args.nodes, args.interval, seed, data_dir, i, args.count, start_step, log_path)
        except KeyboardInterrupt:
            print("\n[STOPPED]")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
