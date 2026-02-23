import os
import glob
import csv
import re
import time
import queue
import multiprocessing
import datetime
import hashlib
import warnings
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
RUNS_DIR = "data"
SNAPSHOT_PCTS = [1.00]
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT = f"topology_results_auto_{TIMESTAMP}.csv"
PLOT_OUT = f"dimensionality_scaling_auto_{TIMESTAMP}.png"

# Automatically utilize half of the available cores
TOTAL_CORES = multiprocessing.cpu_count()
MAX_WORKERS = max(1, TOTAL_CORES // 2)

# --- AUTO-SPECTRAL CONFIG ---
MAX_WALK_LENGTH = 120       # Extended max length to give large networks room to stabilize
MIN_FIT_POINTS = 5          # Minimum number of valid steps required to form a line
NUM_WALKERS = 15000         # Increased slightly to smooth out the longer walks
PLATEAU_MULT = 15.0         # Cutoff threshold to avoid the late-time mixing plateau

# ==========================================
#        METRICS & MATH
# ==========================================

def calculate_metrics(G: nx.Graph):
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return 0, 0.0, 0, 0
    k_min = float(np.min(degrees))
    k_max = float(np.max(degrees))
    k_avg = float(np.mean(degrees))
    triangles = int(sum(nx.triangles(G).values()) // 3)
    return k_min, k_avg, k_max, triangles

def load_graph_snapshot(eng_tag: str, N: int, n_path: str, seed_str: str, target_iter: int):
    run_dir = os.path.join(n_path, seed_str)
    if not os.path.exists(run_dir): return None

    search_pattern = os.path.join(run_dir, f"{eng_tag}_N{N}_{seed_str}_iter_*_edges.csv")
    edge_files = glob.glob(search_pattern)
    if not edge_files: return None

    def extract_iter(filepath: str) -> int:
        fname = os.path.basename(filepath)
        m = re.search(r"_iter_([\d_]+)_edges\.csv", fname)
        return int(m.group(1).replace("_", "")) if m else 0

    closest_edge_file = min(edge_files, key=lambda f: abs(extract_iter(f) - target_iter))
    closest_node_file = closest_edge_file.replace("_edges.csv", "_nodes.csv")

    G = nx.Graph()
    if os.path.exists(closest_node_file):
        with open(closest_node_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row: G.add_node(int(row[0]))

    if os.path.exists(closest_edge_file):
        with open(closest_edge_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 2: G.add_edge(int(row[0]), int(row[1]))

    return G, extract_iter(closest_edge_file)

def compute_hausdorff_dimension(G: nx.Graph, num_samples: int = 50):
    if len(G) == 0: return None
    nodes = list(G.nodes())
    samples = np.random.choice(nodes, min(num_samples, len(nodes)), replace=False)

    radii, volumes = [], []
    for source in samples:
        lengths = nx.single_source_shortest_path_length(G, source)
        if not lengths: continue
        max_r = max(lengths.values())
        if max_r < 3: continue

        vol_at_r = [0] * (max_r + 1)
        for _, dist in lengths.items(): vol_at_r[dist] += 1

        cum_vol = np.cumsum(vol_at_r)
        for r in range(1, int(max_r * 0.8)):
            radii.append(r)
            volumes.append(cum_vol[r])

    if len(set(radii)) < 2: return None
    try:
        slope, _, _, _, _ = stats.linregress(np.log(radii), np.log(volumes))
        return float(slope)
    except Exception:
        return None

def _stable_int_seed(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF

def compute_3d_measure(G: nx.Graph):
    if len(G) == 0: return None
    try: return float(nx.transitivity(G))
    except Exception: return None

# ==========================================
#      AUTO-ADJUSTING SPECTRAL DIMENSION
# ==========================================

def find_best_fit_window(t_vals, p_vals, min_pts=MIN_FIT_POINTS):
    """
    Scans all possible time windows to find the most stable linear fit for d_S.
    Balances R^2 value with the length of the window to prevent overfitting noise.
    """
    best_score = -np.inf
    best_dS, best_R, best_window = None, None, (0, 0)

    log_t = np.log(t_vals)
    log_p = np.log(p_vals)
    n = len(t_vals)

    if n < min_pts:
        return None, None, None, 0

    for i in range(n - min_pts + 1):
        for j in range(i + min_pts, n + 1):
            window_t = log_t[i:j]
            window_p = log_p[i:j]

            slope, _, r_val, _, _ = stats.linregress(window_t, window_p)
            r_squared = r_val ** 2

            # Score = R^2 * sqrt(number of points)
            # This gently rewards longer stable windows over tiny 5-point anomalies
            score = r_squared * ((j - i) ** 0.5)

            if score > best_score:
                best_score = score
                best_dS = float(-2.0 * slope)
                best_R = float(r_val)
                best_window = (int(t_vals[i]), int(t_vals[j-1]))

    return best_dS, best_R, best_window, (best_window[1] - best_window[0] + 1) if best_window else 0


def auto_compute_spectral_dimension(G: nx.Graph, base_seed: int = 0, n_reps: int = 4):
    """
    Runs a longer random walk, filters out finite-size mixing and bipartite zero-returns,
    and automatically finds the optimal scaling window.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0: return None, None, 0, None, "N/A"

    n_reps = max(1, int(n_reps))
    walkers_per_rep = max(500, int(NUM_WALKERS) // n_reps)
    nbr_map = {u: list(G.neighbors(u)) for u in nodes}
    plateau_cut = PLATEAU_MULT / max(1, N)

    dS_vals, fit_pts_list, R_vals, windows = [], [], [], []

    for rep in range(n_reps):
        rng = np.random.default_rng((base_seed + rep) & 0xFFFFFFFF)
        return_counts = np.zeros(MAX_WALK_LENGTH, dtype=np.int64)
        denom = np.zeros(MAX_WALK_LENGTH, dtype=np.int64)

        for _ in range(walkers_per_rep):
            start = nodes[int(rng.integers(0, N))]
            curr = start
            for t in range(MAX_WALK_LENGTH):
                nbrs = nbr_map.get(curr, [])
                if not nbrs: break
                denom[t] += 1
                curr = nbrs[int(rng.integers(0, len(nbrs)))]
                if curr == start:
                    return_counts[t] += 1

        t_cands, p_cands = [], []
        # Filter raw walk data
        for t in range(2, MAX_WALK_LENGTH): # Skip t=0,1 (local artifacts)
            if denom[t] <= 0 or return_counts[t] < 5: continue
            p = return_counts[t] / float(denom[t])
            if p <= plateau_cut: break # We've hit the mixing time boundary, stop adding points
            t_cands.append(t + 1)
            p_cands.append(p)

        dS, R, window, pts = find_best_fit_window(t_cands, p_cands)
        if dS is not None and dS > 0:
            dS_vals.append(dS)
            R_vals.append(R)
            fit_pts_list.append(pts)
            windows.append(window)

    if not dS_vals:
        return None, None, 0, None, "N/A"

    dS_mean = float(np.mean(dS_vals))
    dS_std = float(np.std(dS_vals, ddof=1)) if len(dS_vals) > 1 else 0.0
    fit_points_mean = int(round(float(np.mean(fit_pts_list))))
    R_mean = float(np.mean(R_vals))

    # Average the start and end of the windows for reporting
    avg_start = int(np.mean([w[0] for w in windows]))
    avg_end = int(np.mean([w[1] for w in windows]))
    window_str = f"{avg_start}-{avg_end}"

    return dS_mean, dS_std, fit_points_mean, R_mean, window_str


# ==========================================
#        WORKER & PLOTTER PROCESSES
# ==========================================

def worker_process(primary_q, secondary_q, result_q, worker_id: int):
    while True:
        try: task = primary_q.get_nowait()
        except queue.Empty:
            try: task = secondary_q.get_nowait()
            except queue.Empty: break

        eng, N, seed_str, pct, target_iter, n_path = task

        load_result = load_graph_snapshot(eng, N, n_path, seed_str, target_iter)
        if load_result is None:
            result_q.put(None)
            continue

        G, actual_iter = load_result
        task_seed = _stable_int_seed(eng, N, seed_str, actual_iter, worker_id)

        t0 = time.time()
        d_H = compute_hausdorff_dimension(G)
        d_S, d_S_std, d_S_fit_pts, d_S_R, d_S_win = auto_compute_spectral_dimension(G, base_seed=task_seed)
        d_3D = compute_3d_measure(G)
        compute_sec = time.time() - t0

        _, k_avg, _, _ = calculate_metrics(G)

        result_q.put({
            "Engine": eng, "N": int(N), "Seed": seed_str,
            "Target_Pct": float(pct), "Target_Iter": int(target_iter), "Actual_Iter": int(actual_iter),
            "k_avg": float(k_avg),
            "d_H": d_H,
            "d_S": d_S, "d_S_std": d_S_std, "d_S_fit_pts": int(d_S_fit_pts), "d_S_R": d_S_R, "d_S_Window": d_S_win,
            "d_3D": d_3D,
            "Compute_Time_sec": float(compute_sec),
        })


# ==========================================
#        LIVE DEDICATED PLOTTER PROCESS
# ==========================================


def live_plotter_process(plot_q):
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    from collections import defaultdict
    import queue

    SHOW_SLOPES = True

    while True:
        data = plot_q.get()
        if data is None: break
        while not plot_q.empty():
            try:
                latest = plot_q.get_nowait()
                if latest is None: return
                data = latest
            except queue.Empty: break

        results = data
        finals = [r for r in results if r.get("Target_Pct") == 1.0 and r.get("d_H") is not None and r.get("d_S") is not None]
        if not finals: continue

        by_eng_H = defaultdict(lambda: defaultdict(list))
        by_eng_S = defaultdict(lambda: defaultdict(list))
        by_eng_k = defaultdict(list)

        for r in finals:
            eng, N = r["Engine"], r["N"]
            by_eng_H[eng][N].append(r["d_H"])
            by_eng_S[eng][N].append(r["d_S"])
            if r.get("k_avg") is not None: by_eng_k[eng].append(r["k_avg"])

        engines_sorted = sorted(by_eng_H.keys())

        # --- SINGLE COMBINED PLOT ---
        fig, ax = plt.subplots(figsize=(15, 10))

        for eng in engines_sorted:
            Ns = sorted(by_eng_H[eng].keys())
            if len(Ns) < 2: continue

            logNs = np.log10(Ns)
            avgH = np.array([np.mean(by_eng_H[eng][n]) for n in Ns], dtype=float)
            minH = np.array([np.min(by_eng_H[eng][n]) for n in Ns], dtype=float)
            maxH = np.array([np.max(by_eng_H[eng][n]) for n in Ns], dtype=float)

            avgS = np.array([np.mean(by_eng_S[eng][n]) for n in Ns], dtype=float)
            minS = np.array([np.min(by_eng_S[eng][n]) for n in Ns], dtype=float)
            maxS = np.array([np.max(by_eng_S[eng][n]) for n in Ns], dtype=float)

            k_label = f"{float(np.mean(by_eng_k[eng])):.3g}" if by_eng_k.get(eng) else "?"

            num_points_for_fit = min(4, len(Ns))
            logNs_fit = logNs[-num_points_for_fit:]

            mH, bH, rH, _, _ = stats.linregress(logNs_fit, avgH[-num_points_for_fit:])
            mS, bS, rS, _, _ = stats.linregress(logNs_fit, avgS[-num_points_for_fit:])

            # --- 95% Confidence Interval Calculation for Convergence ---
            top_Ns = Ns[-min(3, len(Ns)):]
            pool_S = []
            for n in top_Ns:
                pool_S.extend(by_eng_S[eng][n])

            converged_S = np.mean(pool_S)
            if len(pool_S) > 1:
                se_S = stats.sem(pool_S)
                ci_95 = se_S * stats.t.ppf((1 + 0.95) / 2., len(pool_S) - 1)
            else:
                ci_95 = 0.0

            # --- RENDER HAUSDORFF ---
            label_H = f"{eng} $d_H$ (k≈{k_label}) | m={mH:.3f} (R={rH:.2f})"
            p = ax.plot(logNs, avgH, marker="o", linewidth=2.5, linestyle="-", label=label_H)
            color = p[0].get_color()

            # --- RENDER SPECTRAL (Now with Conv & CI appended!) ---
            label_S = f"{eng} $d_S$ (k≈{k_label}) | m={mS:.3f} (R={rS:.2f}) | Conv={converged_S:.2f}$\\pm${ci_95:.2f}"
            ax.plot(logNs, avgS, marker="s", linewidth=2.5, linestyle="-.", color=color, label=label_S)

            if SHOW_SLOPES and len(Ns) > 2:
                ax.plot(logNs_fit, mH * logNs_fit + bH, linestyle="--", alpha=0.7, color='gray')
                ax.plot(logNs_fit, mS * logNs_fit + bS, linestyle="--", alpha=0.7, color='gray')

            ax.fill_between(logNs, minH, maxH, color=color, alpha=0.1)
            ax.fill_between(logNs, minS, maxS, color=color, alpha=0.1)

            for x, y_h, y_s in zip(logNs, avgH, avgS):
                ax.annotate(f"{y_h:.2f}", (x, y_h), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color=color, weight='bold')
                ax.annotate(f"{y_s:.2f}", (x, y_s), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=8, color=color, weight='bold')

        ax.set_title("Dimensionality Scaling: Hausdorff ($d_H$) vs Spectral ($d_S$) [Auto-Windowed]", fontsize=14)
        ax.set_xlabel("Network Size $\\log_{10}(N)$", fontsize=12)
        ax.set_ylabel("Measured Dimension", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Legend updated to clarify the sources of the math
        ax.legend(title="Engine Fits (Slope from largest 4 N | Conv limit from largest 3 N)", fontsize=9, ncol=2, loc="upper left")

        fig.tight_layout()
        # Make sure PLOT_OUT is defined in your global scope (it is in your main script)
        fig.savefig(PLOT_OUT, dpi=200)
        plt.close(fig)

# ==========================================
#               MAIN
# ==========================================

def main():
    print(f"--- Deep Topology Monitor | Auto-Spectral Engine ({MAX_WORKERS}/{TOTAL_CORES} Threads) ---")

    if not os.path.exists(RUNS_DIR):
        print(f"[ERROR] RUNS_DIR '{RUNS_DIR}' not found.")
        return

    engines = [d for d in os.listdir(RUNS_DIR) if d.startswith("E")]
    tasks = []

    for eng in engines:
        eng_path = os.path.join(RUNS_DIR, eng)
        if not os.path.isdir(eng_path): continue

        for n_dir_name in os.listdir(eng_path):
            if not n_dir_name.startswith("N"): continue
            try: N = int(n_dir_name[1:])
            except ValueError: continue

            n_path = os.path.join(eng_path, n_dir_name)
            if not os.path.isdir(n_path): continue

            for log_file in glob.glob(os.path.join(n_path, "S*_log.csv")):
                seed_str = os.path.basename(log_file).split("_")[0]
                try:
                    with open(log_file, "r") as f:
                        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                    if not lines or "COMPLETED" not in lines[-1]: continue

                    max_iter = next((int(p.split(",")[1]) for p in reversed(lines) if len(p.split(",")) >= 2 and p.split(",")[1].isdigit()), 0)
                    if max_iter <= 0: continue
                except Exception: continue

                for pct in SNAPSHOT_PCTS:
                    tasks.append((eng, N, seed_str, pct, int(max_iter * pct), n_path))

    tasks.sort(key=lambda x: (x[1], x[3]))
    total_tasks = len(tasks)
    print(f"Queued {total_tasks} snapshot analyses.")

    q_ascend, q_descend, result_q, plot_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()

    midpoint = total_tasks // 2
    for t in tasks[:midpoint]: q_ascend.put(t)
    for t in reversed(tasks[midpoint:]): q_descend.put(t)

    plot_proc = multiprocessing.Process(target=live_plotter_process, args=(plot_q,))
    plot_proc.start()

    workers = []
    ascend_count = max(1, MAX_WORKERS // 2)
    for i in range(ascend_count):
        p = multiprocessing.Process(target=worker_process, args=(q_ascend, q_descend, result_q, i))
        p.start(); workers.append(p)
    for i in range(max(0, MAX_WORKERS - ascend_count)):
        p = multiprocessing.Process(target=worker_process, args=(q_descend, q_ascend, result_q, i + ascend_count))
        p.start(); workers.append(p)

    results = []
    completed = 0
    pbar = tqdm(total=total_tasks, desc="Crunching Topology")

    while completed < total_tasks:
        res = result_q.get()
        completed += 1
        if res is not None:
            results.append(res)

            # Formatted log string to report findings along the way
            dH_str = f"{res['d_H']:.2f}" if res.get("d_H") is not None else "N/A"
            dS_str = f"{res['d_S']:.2f}" if res.get("d_S") is not None else "N/A"
            win_str = f"[{res.get('d_S_Window', 'N/A')}]"

            log_str = (
                f"[LOG] {res['Engine']} N={res['N']:<7} {res['Seed']:<4} "
                f"| dH:{dH_str:>5} dS:{dS_str:>5} Win:{win_str:<7} "
                f"| t:{res.get('Compute_Time_sec', 0.0):.1f}s"
            )
            tqdm.write(log_str)

            file_exists = os.path.isfile(CSV_OUT)
            with open(CSV_OUT, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                if not file_exists: writer.writeheader()
                writer.writerow(res)

            plot_q.put(list(results))
            pbar.update(1)

    pbar.close()
    plot_q.put(None)
    for p in workers: p.join()
    plot_proc.join()
    print(f"\n[DONE] Processed {len(results)} valid snapshots. Final data in {CSV_OUT}. Plot in {PLOT_OUT}.")

if __name__ == "__main__":
    main()
