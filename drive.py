import argparse
import os
import sys
import time
import datetime
import csv
import numpy as np
import networkx as nx
import re
from engine import PhysicsEngine  # Imports engine.py from the SAME directory as this script

os.environ["OMP_NUM_THREADS"] = "1"

# --- CONFIG DEFAULTS ---
DEFAULT_N = 1601
DEFAULT_SEED = 42
DEFAULT_RUNS = 4
DEFAULT_DELTA_K = 0.01
DEFAULT_STEP_INTERVAL = 100_000  # For --report-mode step

# --- STABILITY CONTROL ---
STABILITY_WINDOW = 1_000_000 # Iterations
STABILITY_TOLERANCE = 1e-8   # Amount of k-mean digits to wait for stabilizing
STABILITY_K_MIN = 2          # Extra exit condition, wait for every single node to have at least 2 edges

# --- TEMPERATURE CONTROL ---
TEMP_START = 1
TEMP_DECAY = 0.99999
TEMP_FLOOR = 0.00000

# --- EARLY ZOOM SETUP ---
ZOOM_SLICES = 10  # How many extra frames for the first few delta_k milestones
ZOOM_WINDOW = 3  # How many delta_k milestones to stay zoomed in for

# ==========================================
#        HELPER FUNCTIONS
# ==========================================

def get_temperature(iter_num):
    t = TEMP_START * (TEMP_DECAY ** iter_num)
    return max(t, TEMP_FLOOR)

def fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0: return "---"
    return str(datetime.timedelta(seconds=int(seconds)))

def fmt_iter(iter_num):
    return f"{iter_num:011_d}"

def export_snapshot(G, iter_num, version_tag, N, seed, output_dir):
    iter_str = fmt_iter(iter_num)
    # Nodes
    node_file = os.path.join(output_dir, f"{version_tag}_N{N}_S{seed}_iter_{iter_str}_nodes.csv")
    with open(node_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "psi_real", "psi_imag"])
        for n in G.nodes():
            psi = G.nodes[n].get("psi", 0j)
            writer.writerow([n, psi.real, psi.imag])
    # Edges
    edge_file = os.path.join(output_dir, f"{version_tag}_N{N}_S{seed}_iter_{iter_str}_edges.csv")
    with open(edge_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "theta"])
        for u, v, d in G.edges(data=True):
            writer.writerow([u, v, d.get("theta", 0.0)])

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

def find_latest_snapshot_iter(data_dir, version_tag, N, seed):
    max_iter = -1
    pattern = re.compile(rf"{version_tag}_N{N}_S{seed}_iter_([\d_]+)_nodes\.csv")
    if not os.path.exists(data_dir): return -1
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            # remove underscores from "000_100" -> "000100"
            val = int(match.group(1).replace('_', ''))
            if val > max_iter: max_iter = val
    return max_iter

def load_engine_state(engine, iter_num, version_tag, N, seed, data_dir):
    iter_str = fmt_iter(iter_num)
    nf = os.path.join(data_dir, f"{version_tag}_N{N}_S{seed}_iter_{iter_str}_nodes.csv")
    ef = os.path.join(data_dir, f"{version_tag}_N{N}_S{seed}_iter_{iter_str}_edges.csv")
    if not os.path.exists(nf): raise FileNotFoundError(f"Missing snapshot {iter_num}")

    # Reset Engine
    engine.adj_matrix[:] = False
    engine.theta_matrix[:] = 0.0
    engine.meta[:] = 0

    # Load Nodes
    with open(nf, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            nid = int(row[0])
            if nid < engine.N:
                engine.psi[nid] = complex(float(row[1]), float(row[2]))

    # Load Edges & Rebuild Lazy List
    with open(ef, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            theta = float(row[2])
            engine.adj_matrix[u, v] = True
            engine.adj_matrix[v, u] = True
            engine.theta_matrix[u, v] = theta
            engine.theta_matrix[v, u] = -theta

            # Smart Engine edge list reconstruction
            idx = engine.meta[1]
            if idx < len(engine.edge_list):
                engine.edge_list[idx, 0] = u
                engine.edge_list[idx, 1] = v
                engine.meta[1] += 1
                engine.meta[0] += 1

def load_raw_state_metrics(version_tag, N, seed, iter_num, data_dir):
    iter_str = fmt_iter(iter_num)
    nf = os.path.join(data_dir, f"{version_tag}_N{N}_S{seed}_iter_{iter_str}_nodes.csv")
    ef = os.path.join(data_dir, f"{version_tag}_N{N}_S{seed}_iter_{iter_str}_edges.csv")
    if not os.path.exists(nf) or not os.path.exists(ef): return None
    G = build_graph_from_csv(nf, ef)
    return calculate_metrics(G)

def generate_seed_schedule(file_path, default_count, default_start_seed):
    if not os.path.exists(file_path):
        return [default_start_seed + i for i in range(default_count)], default_count
    try:
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
             return [default_start_seed + i for i in range(default_count)], default_count
        target_count = int(lines[0])
        base_seeds = []
        for line in lines[1:]:
            parts = line.replace(',', ' ').split()
            for p in parts:
                if p.isdigit(): base_seeds.append(int(p))
        if not base_seeds: base_seeds = [default_start_seed]
        final_seeds = []
        used_seeds = set()
        num_base = len(base_seeds)
        for i in range(target_count):
            base_idx = i % num_base
            cycle_offset = i // num_base
            candidate = base_seeds[base_idx] + cycle_offset
            while candidate in used_seeds: candidate += 1
            final_seeds.append(candidate)
            used_seeds.add(candidate)
        return final_seeds, target_count
    except Exception as e:
        print(f"[WARN] Failed to parse {file_path}: {e}. Using defaults.")
        return [default_start_seed + i for i in range(default_count)], default_count

def get_cumulative_time(log_path):
    if not os.path.exists(log_path): return 0.0
    last_val = 0.0
    try:
        with open(log_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # Check for valid data row (index 3 is elapsed_sec)
                # Matches format: timestamp_iso, iter, sps, elapsed_sec, ...
                if len(parts) >= 4 and parts[1].isdigit():
                    try:
                        val = float(parts[3])
                        if val >= 0: last_val = val
                    except ValueError: continue
    except Exception as e:
        pass
    return last_val

# ==========================================
#          CORE LOGIC
# ==========================================

def run_simulation(version_tag, N, delta_k, seed, data_output_dir, run_idx, total_runs, start_iter, log_file, prev_elapsed_time, report_mode, step_interval):
    t0 = time.time()
    f_log = None
    try:
        mode = 'w'
        write_header = True
        if start_iter > 0 and os.path.exists(log_file):
            mode = 'a'
            write_header = False

        f_log = open(log_file, mode)
        if write_header:
            f_log.write(f"# version={version_tag},N={N},seed={seed},delta_k={delta_k},stab_win={STABILITY_WINDOW}\n")
            f_log.write("timestamp_iso,iter,iter_per_sec,elapsed_sec,k_min,k_avg,k_max,triangles,temp,edges\n")
    except Exception as e:
        print(f"[WARN] Could not open log file: {e}")

    engine = PhysicsEngine(N, seed)

    current_iter = 0
    if start_iter > 0:
        load_engine_state(engine, start_iter, version_tag, N, seed, data_output_dir)
        current_iter = start_iter
    else:
        export_snapshot(engine.G, 0, version_tag, N, seed, data_output_dir)
        current_iter = 0

    current_edge_count = engine.meta[0]
    last_reported_k = (2.0 * current_edge_count) / N

    fine_delta_k = delta_k / ZOOM_SLICES
    zoom_threshold = delta_k * ZOOM_WINDOW

    # Stability check scheduling
    start_block = start_iter // STABILITY_WINDOW
    next_check_iter = (start_block + 1) * STABILITY_WINDOW
    if next_check_iter <= start_iter:
        next_check_iter += STABILITY_WINDOW

    metrics = calculate_metrics(engine.G)
    last_check_k_avg = metrics[1]

    try:
        while True:
            current_temp = get_temperature(current_iter)
            engine.params[0] = current_temp
            engine.iterate()
            current_iter += 1

            current_edge_count = engine.meta[0]
            current_k = (2.0 * current_edge_count) / N

            do_report = False

            # --- REPORT LOGIC: K-MEAN VS STEP ---
            if report_mode == "k":
                # Dynamic delta determination
                if last_reported_k < (zoom_threshold - 1e-9):
                    effective_delta_k = fine_delta_k
                else:
                    effective_delta_k = delta_k

                k_diff = abs(current_k - last_reported_k)
                if k_diff >= effective_delta_k:
                    do_report = True

            elif report_mode == "step":
                if current_iter > 0 and current_iter % step_interval == 0:
                    do_report = True

            # Always report on huge milestones or stability windows
            if current_iter % STABILITY_WINDOW == 0 or current_iter % 1_000_000 == 0:
                do_report = True

            # SAVE / LOG CONDITION
            if do_report:
                session_elapsed = time.time() - t0
                total_elapsed = session_elapsed + prev_elapsed_time

                metrics = calculate_metrics(engine.G)
                k_min, k_avg, k_max, triangles = metrics

                # Iterations per second (Cumulative)
                ips = current_iter / total_elapsed if total_elapsed > 0 else 0

                print(f"{version_tag:<7} N{N:<7} {ips:>10_.0f}it/s {current_iter:>12_.0f} S{seed:<4} [{run_idx+1:<2}/{total_runs:<2}] "
                      f"T:{fmt_time(total_elapsed)} E:{current_edge_count:<6} Tri:{triangles:<6_}  "
                      f"k:{k_min:<2}/{k_avg:<13.10f}/{k_max:<2} Temp:{current_temp:.1e}")

                now_iso = datetime.datetime.now().isoformat()
                if f_log:
                    f_log.write(f"{now_iso},{current_iter},{ips:.0f},{total_elapsed:.2f},{k_min},{k_avg:.20f},{k_max},{triangles},{current_temp:.6g},{current_edge_count}\n")
                    f_log.flush()

                export_snapshot(engine.G, current_iter, version_tag, N, seed, data_output_dir)

                # Only update last_reported_k if we hit the actual threshold to prevent drift
                if report_mode == "k":
                    last_reported_k = current_k

                # STABILITY CHECK
                if current_iter >= next_check_iter:
                    delta_stability = abs(k_avg - last_check_k_avg)
                    if delta_stability < STABILITY_TOLERANCE:
                        print(f"[STABLE] No k_avg fluctuation ({delta_stability:.1e}) over last {STABILITY_WINDOW} iters.")
                        if k_min >= STABILITY_K_MIN:
                            print(f"COMPLETED at {datetime.datetime.now().isoformat()}")
                            if f_log: f_log.write(f"COMPLETED at {datetime.datetime.now().isoformat()}\n")
                            break
                        else:
                            print(f"Waiting for k-min ({k_min}) >= {STABILITY_K_MIN}")
                    else:
                        next_check_iter = ((current_iter // STABILITY_WINDOW) + 1) * STABILITY_WINDOW
                        print(f"   [FLUCTUATION] k_avg changed by {delta_stability:.10f}. Extending run to {next_check_iter:_}")
                        last_check_k_avg = k_avg

    finally:
        if f_log: f_log.close()
    return time.time() - t0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", type=int, default=DEFAULT_N)
    parser.add_argument("-d", "--delta-k", type=float, default=DEFAULT_DELTA_K)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-c", "--count", type=int, default=DEFAULT_RUNS)
    parser.add_argument("-r", "--report-mode", type=str, choices=["k", "step"], default="k")
    parser.add_argument("-i", "--step-interval", type=int, default=DEFAULT_STEP_INTERVAL)
    args = parser.parse_args()

    # --- SELF-AWARE VERSION DETECTION ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_tag = os.path.basename(script_dir)

    if not re.match(r"^E\d+D\d+$", version_tag):
        print(f"[WARN] Script running in '{version_tag}', expected 'ExDy' format. Exiting.")
        sys.exit(1)

    run_seeds, total_runs_count = generate_seed_schedule("seeds.txt", args.count, args.seed)

    batch_dir = os.path.join(script_dir, f"N{args.nodes}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"=== BATCH {version_tag} | N={args.nodes} | {total_runs_count} Runs | Mode: {args.report_mode.upper()} ===")

    for i, seed in enumerate(run_seeds):
        run_dir = os.path.join(batch_dir, f"S{seed}")
        os.makedirs(run_dir, exist_ok=True)

        log_path = os.path.join(batch_dir, f"S{seed}_log.csv")

        last_iter = find_latest_snapshot_iter(run_dir, version_tag, args.nodes, seed)
        start_iter = 0
        prev_elapsed_time = 0.0

        is_already_stable = False
        if last_iter > 0 and last_iter >= STABILITY_WINDOW and (last_iter % STABILITY_WINDOW == 0):
            prev_heartbeat = last_iter - STABILITY_WINDOW
            curr_m = load_raw_state_metrics(version_tag, args.nodes, seed, last_iter, run_dir)
            prev_m = load_raw_state_metrics(version_tag, args.nodes, seed, prev_heartbeat, run_dir)
            if curr_m and prev_m and abs(curr_m[1] - prev_m[1]) < STABILITY_TOLERANCE:
                print(f">> Run {i+1} (S{seed}) [SKIPPED] Found stable heartbeat on disk.")
                is_already_stable = True

        if is_already_stable: continue

        if last_iter > 0:
            print(f">> Run {i+1} (S{seed}) [RESUMING] from iter {last_iter:_}")
            start_iter = last_iter
            prev_elapsed_time = get_cumulative_time(log_path)
        else:
            print(f">> Run {i+1} (S{seed}) [STARTING]")

        try:
            run_simulation(version_tag, args.nodes, args.delta_k, seed, run_dir, i, total_runs_count, start_iter, log_path, prev_elapsed_time, args.report_mode, args.step_interval)
        except KeyboardInterrupt:
            print("\n[STOPPED]")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
