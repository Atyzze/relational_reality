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
import numpy as np
import networkx as nx
from engine import PhysicsEngine

# --- DEFAULTS ---
DEFAULT_N = 100
DEFAULT_SEED = 1000
DEFAULT_RUNS = 8
DEFAULT_INTERVAL = 100_000

# --- PHYSICS GATES ---
REQUIRED_MIN_DEGREE = 4
STABILITY_STREAK_TARGET = 10
TARGET_K_MEAN = 12.0
REGRESSION_WINDOW = 20

# --- TEMPERATURE CONTROL ---
TEMP_START = 0.20
TEMP_DECAY = 0.9999
TEMP_FLOOR = 1e-9

def get_temperature(step):
    """
    Calculates the exact temperature for a specific step.
    T(t) = T_start * (decay ^ t)
    """
    t = TEMP_START * (TEMP_DECAY ** step)
    return max(t, TEMP_FLOOR)

def fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "---"
    return str(datetime.timedelta(seconds=int(seconds)))

def fmt_step(step):
    return f"{step:011_d}"

def get_engine_version(engine_file, runs_dir):
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
    return next_id, True, (existing_versions[-1] if existing_versions else None)

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
    """
    Loads state from CSV into the engine's internal matrices.
    Does NOT manually set engine.G (as that is now a read-only property).
    """
    step_str = fmt_step(step)
    nf = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_nodes.csv")
    ef = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    if not os.path.exists(nf):
        raise FileNotFoundError(f"Missing snapshot {step}")

    # 1. Load Nodes (Psi)
    with open(nf, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            nid = int(row[0])
            if nid < engine.N:
                engine.psi[nid] = complex(float(row[1]), float(row[2]))

    # 2. Load Edges (Matrices only)
    engine.adj_matrix[:] = False
    engine.theta_matrix[:] = 0.0
    edge_count = 0

    with open(ef, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            theta = float(row[2])

            # Populate Matrices
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

    # SIMULATE FORWARD WITH EXACT HISTORICAL TEMP
    current_sim_step = safe_step
    for _ in range(interval):
        current_temp = get_temperature(current_sim_step)
        test_engine.params[0] = current_temp
        test_engine.step()
        current_sim_step += 1

    calc_m = calculate_metrics(test_engine.G)
    disk_m = load_raw_state_metrics(version_id, N, seed, latest_step, data_dir)

    if disk_m is None:
        print(f"   [WARN] Target file missing. Rolling back.")
        return False, safe_step

    topology_match = (calc_m[3] == disk_m[3])
    physics_match = math.isclose(calc_m[1], disk_m[1], rel_tol=1e-4)

    if topology_match and physics_match:
        print(f"   [VERIFIED] Physics signatures match.")
        return True, latest_step
    else:
        print(f"   [DRIFT DETECTED] Disk: Tri={disk_m[3]} k={disk_m[1]:.4f} | Calc: Tri={calc_m[3]} k={calc_m[1]:.4f}")
        print(f"   [HEALING] Discarding step {latest_step}. Resuming from {safe_step}.")
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
    except:
        return None

def run_simulation(version_id, N, interval, seed, data_output_dir, run_idx, total_runs, start_step=0):
    t0 = time.time()

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

    while True:
        # --- EXTERNAL TEMP CONTROL ---
        current_temp = get_temperature(t)
        engine.params[0] = current_temp

        # Physics
        engine.step()
        t += 1

        if t % interval == 0:
            elapsed = time.time() - t0
            metrics = calculate_metrics(engine.G)
            k_min, k_avg, k_max, triangles = metrics

            trend_history.append((elapsed, k_avg))

            k_long = int(k_avg * 1000)
            if k_long == last_k_long:
                current_streak += 1
            else:
                current_streak = 1
                last_k_long = k_long

            sps = (t - start_step) / elapsed if elapsed > 0 else 0

            if k_avg < TARGET_K_MEAN:
                status = f">> 12 ({k_avg/TARGET_K_MEAN:.0%})"
            else:
                status = f"STABL {current_streak}/{STABILITY_STREAK_TARGET}"

            eta = fmt_time(estimate_total_eta(trend_history, elapsed))

            print(
                f"E{version_id:<3} "
                f"N{N:<7_}"                        # Increased padding for underscores
                f"{sps:>10_.0f}i/s "                # Added underscores, increased width to 10
                f"{t:>12_.0f} "                  # Added underscores
                f"S{seed} "
                f"[{run_idx+1}/{total_runs}] "
                f"T:{fmt_time(elapsed)} "
                f"[{status:<12}] "
                f"ETA:{eta:<8} "
                f"Tri:{triangles:<10_} "            # Added underscores
                f"k:{k_min:<2}/{k_avg:<8.5f}/{k_max:<2} "
                f"Temp:{current_temp:.6f}"
        )

            export_snapshot(engine.G, t, version_id, N, seed, data_output_dir)

            if k_min >= REQUIRED_MIN_DEGREE and current_streak >= STABILITY_STREAK_TARGET and k_avg >= (TARGET_K_MEAN - 0.2):
                print(f"   [CRYSTALLIZED] Run Complete at step {t}")
                break
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

    v_id, is_new, _ = get_engine_version("engine.py", "runs")
    v_id = args.version if args.version else v_id
    version_dir = os.path.join("runs", f"E{v_id}")
    batch_dir = os.path.join(version_dir, f"N{args.nodes}")
    os.makedirs(batch_dir, exist_ok=True)

    if is_new:
        shutil.copy2("engine.py", os.path.join(version_dir, "engine.py"))
        print(f"New Engine detected. Using E{v_id}")

    print(f"=== BATCH E{v_id} | N={args.nodes} | {args.count} Runs ===")
    print(f"=== MODE: EXTERNAL TEMP CONTROL ===")

    for i in range(args.count):
        seed = args.seed + i
        run_dir = os.path.join(batch_dir, f"S{seed}")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        last_step = find_latest_snapshot_step(data_dir, v_id, args.nodes, seed)
        start_step = 0

        if last_step > 0:
            print(f">> Run {i+1} (S{seed}) Checking {last_step}...")
            if check_stability_from_disk(data_dir, v_id, args.nodes, seed, last_step, args.interval):
                print(f"   [SKIPPED] Already Stable.")
                continue

            valid, healed_step = verify_stitching_and_heal(
                v_id, args.nodes, seed, data_dir, args.interval, last_step
            )

            if valid:
                start_step = healed_step
                print(f"   [RESUMING] from step {start_step}")
            else:
                start_step = healed_step
        else:
            print(f">> Run {i+1} (S{seed}) [STARTING]")

        try:
            run_simulation(v_id, args.nodes, args.interval, seed, data_dir, i, args.count, start_step)
        except KeyboardInterrupt:
            print("\n[STOPPED]")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
