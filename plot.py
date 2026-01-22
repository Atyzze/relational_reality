import os
import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import stats, optimize, linalg
import math
import sys
import time
import hashlib
import json
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- CONFIGURATION ---
RUNS_DIR = "runs"
OUTPUT_DIR = "analysis_dashboard"
PROBE_COUNT = 10
# Low count to ensure we catch early runs and generate their metrics.csv
MIN_FILE_COUNT = 200

# --- HELPER FUNCTIONS: FILES & METADATA ---

def find_run_folders(root_dir):
    """Recursively find all folders containing simulation data."""
    run_folders = []
    for root, dirs, files in os.walk(root_dir):
        # Optimization: fast fail if no csvs
        if any(f.startswith("edges_step_") and f.endswith(".csv") for f in files):
            run_folders.append(root)
    return run_folders

def extract_metadata(folder_path):
    """Extract N and Seed from folder string."""
    n_match = re.search(r'_N(\d+)', folder_path)
    if not n_match: return None, None
    N = int(n_match.group(1))

    s_match_batch = re.search(r'seed_(\d+)', folder_path)
    s_match_old = re.search(r'_s(\d+)', folder_path)

    if s_match_batch:
        seed = int(s_match_batch.group(1))
    elif s_match_old:
        seed = int(s_match_old.group(1))
    else:
        seed = 0

    return N, seed

def extract_timestamp(folder_path):
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', folder_path)
    if match:
        return match.group(1)

    engine_path = os.path.join(folder_path, "engine.py")
    if not os.path.exists(engine_path):
        engine_path = os.path.join(os.path.dirname(folder_path), "engine.py")

    if os.path.exists(engine_path):
        ts = os.path.getmtime(engine_path)
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

    return "9999-12-31_23-59-59"

def get_config_signature(folder_path):
    engine_path = os.path.join(folder_path, "engine.py")
    if not os.path.exists(engine_path):
        engine_path = os.path.join(os.path.dirname(folder_path), "engine.py")

    params_path = os.path.join(folder_path, "physics_parameters.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(os.path.dirname(folder_path), "physics_parameters.json")

    hasher = hashlib.sha256()
    details = ""

    if os.path.exists(engine_path):
        with open(engine_path, 'rb') as f:
            content = f.read()
            hasher.update(content)
            details += f"--- ENGINE.PY ({len(content)} bytes) ---\n"
            details += "Hash: " + hashlib.sha256(content).hexdigest() + "\n\n"
    else:
        details += "--- ENGINE.PY MISSING ---\n\n"
        hasher.update(b"NO_ENGINE")

    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            content = f.read()
            hasher.update(content)
            try:
                p_dict = json.loads(content)
                details += "--- PHYSICS PARAMETERS ---\n"
                details += json.dumps(p_dict, indent=2) + "\n"
            except:
                details += "--- PHYSICS PARAMETERS (Raw) ---\n" + content.decode('utf-8', errors='ignore') + "\n"
    else:
        details += "--- PHYSICS PARAMETERS MISSING ---\n"
        hasher.update(b"NO_PARAMS")

    full_hash = hasher.hexdigest()
    short_hash = full_hash[:8]
    return short_hash, details

def get_final_snapshot(folder_path):
    """Finds the highest step number available in the folder."""
    edge_files = glob.glob(os.path.join(folder_path, "edges_step_*.csv"))
    if not edge_files: return None, None, None

    steps = []
    for f in edge_files:
        m = re.search(r'edges_step_(\d+).csv', f)
        if m: steps.append(int(m.group(1)))

    if not steps: return None, None, None

    final_step = max(steps)
    edges_path = os.path.join(folder_path, f"edges_step_{final_step}.csv")
    nodes_path = os.path.join(folder_path, f"nodes_step_{final_step}.csv")

    if not os.path.exists(nodes_path): return None, None, None

    return final_step, edges_path, nodes_path

def scan_and_group_runs(folders):
    print(f">> Scanning {len(folders)} folders...")

    raw_data = []
    config_details_map = {}
    hash_timestamps = {}
    group_stats = {}

    for f in folders:
        N, s = extract_metadata(f)
        if N is None: continue

        cfg_hash, cfg_details = get_config_signature(f)
        config_details_map[cfg_hash] = cfg_details

        key = (cfg_hash, N)
        if key not in group_stats: group_stats[key] = {'found': 0, 'valid': 0}
        group_stats[key]['found'] += 1

        all_files = [name for name in os.listdir(f) if os.path.isfile(os.path.join(f, name))]
        file_count = len(all_files)

        ts = extract_timestamp(f)
        if cfg_hash not in hash_timestamps:
            hash_timestamps[cfg_hash] = ts
        else:
            if ts < hash_timestamps[cfg_hash]:
                hash_timestamps[cfg_hash] = ts

        # Use lower file count to ensure we generate metrics for active runs
        if file_count >= MIN_FILE_COUNT:
            group_stats[key]['valid'] += 1
            step, ef, nf = get_final_snapshot(f)

            # We append even if step is None so we can at least try to generate metrics.csv
            # But normally get_final_snapshot returns None if no edge files exist.
            if step is not None:
                raw_data.append({
                    'N': N, 'seed': s, 'step': step,
                    'efile': ef, 'nfile': nf, 'folder': f,
                    'raw_hash': cfg_hash
                })

    hash_to_label = hash_timestamps

    for row in raw_data:
        row['config_id'] = hash_to_label.get(row['raw_hash'], row['raw_hash'])

    final_details_map = {}
    for h, details in config_details_map.items():
        if h in hash_to_label:
            label = hash_to_label[h]
            final_details_map[label] = details

    unique_hashes = sorted(list(set([k[0] for k in group_stats.keys()])))

    print("-" * 100)
    print(f"{'Config (Date)':<25} | {'N':<6} | {'Found':<8} | {'Valid':<8} | {'Status'}")
    print("-" * 100)

    for h in unique_hashes:
        label = hash_to_label.get(h, h)
        relevant_ns = sorted(list(set([k[1] for k in group_stats.keys() if k[0] == h])))

        for n_val in relevant_ns:
            stats = group_stats[(h, n_val)]
            status = "Partial" if stats['valid'] < stats['found'] else "Complete"
            if stats['valid'] == 0: status = "No Valid Runs"
            print(f"{label:<25} | {n_val:<6} | {stats['found']:<8} | {stats['valid']:<8} | {status}")

    print("-" * 100 + "\n")

    if not raw_data:
        return [], {}

    df_meta = pd.DataFrame(raw_data)
    valid_candidates = []

    for _, row in df_meta.iterrows():
        valid_candidates.append(row.to_dict())

    if valid_candidates:
        valid_candidates.sort(key=lambda x: (x['config_id'], x['N'], x['seed']))

    return valid_candidates, final_details_map

# --- PHYSICS LOGIC & LOCAL METRICS ---

def get_run_evolution(folder_path, N):
    """
    1. Scans all edges_step_X.csv files in the folder.
    2. Computes Triangles and Mean Degree for each step.
    3. SAVES 'metrics.csv' inside the run folder (overwriting previous).
    4. Returns the series for global analysis.
    """
    files = glob.glob(os.path.join(folder_path, "edges_step_*.csv"))
    series = []

    step_files = []
    for f in files:
        m = re.search(r'edges_step_(\d+).csv', f)
        if m:
            step_files.append((int(m.group(1)), f))

    step_files.sort(key=lambda x: x[0])

    for step, fpath in step_files:
        try:
            # We must load the edges to calculate metrics
            df = pd.read_csv(fpath)

            # 1. Mean Degree <k> = 2E / N
            edge_count = len(df)
            k_mean = (2.0 * edge_count) / N

            # 2. Triangles (Requires Graph construction)
            G = nx.from_pandas_edgelist(df, 'source', 'target')
            if G.number_of_nodes() > 0:
                t_count = sum(nx.triangles(G).values()) // 3
            else:
                t_count = 0

            series.append((step, t_count, k_mean))
        except:
            pass

    # --- NEW: GENERATE LOCAL METRICS.CSV ---
    if series:
        try:
            local_df = pd.DataFrame(series, columns=['step', 'triangles', 'mean_degree'])
            local_metrics_path = os.path.join(folder_path, "metrics.csv")
            local_df.to_csv(local_metrics_path, index=False)
            # We don't print here to avoid clogging the parallel output,
            # but the file will be created/updated in the folder.
        except Exception as e:
            # Fail silently on file write to avoid stopping the worker
            pass

    return series

def U_ij(G, i, j):
    if i > j: u, v, sign = j, i, -1.0
    else:     u, v, sign = i, j, 1.0
    if G.has_edge(u, v): return np.exp(1j * sign * G[u][v]['theta'])
    return 0j

def apply_random_gauge_transform(G):
    rng = np.random.default_rng()
    alpha = {n: rng.uniform(-math.pi, math.pi) for n in G.nodes()}
    for n in G.nodes(): G.nodes[n]['psi'] *= np.exp(1j * alpha[n])
    for u, v in G.edges(): G[u][v]['theta'] = (G[u][v]['theta'] + alpha[u] - alpha[v]) % (2*math.pi)

def measure_gauge_observables(G):
    vals = []
    triangles = []
    nodes = sorted(list(G.nodes()))
    for i in nodes[:PROBE_COUNT]:
        nbrs = sorted(list(G.neighbors(i)))
        for j in nbrs:
            if j > i:
                common = sorted(list(set(G.neighbors(i)).intersection(G.neighbors(j))))
                for k in common:
                    if k > j:
                        triangles.append((i, j, k))
                        if len(triangles) >= 50: break
            if len(triangles) >= 50: break
        if len(triangles) >= 50: break

    for (i, j, k) in triangles:
        w = U_ij(G, i, j) * U_ij(G, j, k) * U_ij(G, k, i)
        vals.append(w.real)
    if not vals: return np.array([0.0])
    return np.array(vals)

def check_gauge_invariance(G):
    obs_before = measure_gauge_observables(G)
    G_copy = G.copy()
    apply_random_gauge_transform(G_copy)
    obs_after = measure_gauge_observables(G_copy)
    return np.max(np.abs(obs_before - obs_after))

def solve_wasserstein_scipy(p, q, C):
    n = len(p); m = len(q)
    c_vec = C.flatten()
    A_eq = np.zeros((n + m, n * m))
    b_eq = np.concatenate([p, q])
    for i in range(n): A_eq[i, i*m : (i+1)*m] = 1
    for j in range(m): A_eq[n+j, j::m] = 1
    res = optimize.linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    return res.fun if res.success else 0.0

def compute_gravity_params(G, alpha=0.5):
    if nx.is_directed(G) or not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        if not comps: return 0.0, 0.0
        G = G.subgraph(max(comps, key=len)).copy()

    try: path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    except: return 0.0, 0.0

    nodes = list(G.nodes())
    sample_nodes = np.random.choice(nodes, min(len(nodes), PROBE_COUNT), replace=False)
    node_rhos = []; node_curvatures = []

    for u in sample_nodes:
        nbrs_u = list(G.neighbors(u))
        if not nbrs_u: continue
        edge_kappas = []
        for v in nbrs_u:
            def get_dist(node, nbrs):
                deg = len(nbrs); mass = np.zeros(len(nbrs) + 1); mass[0] = alpha; mass[1:] = (1.0 - alpha) / deg
                return [node] + nbrs, mass

            nbrs_v = list(G.neighbors(v))
            locs_u, mu_u = get_dist(u, nbrs_u); locs_v, mu_v = get_dist(v, nbrs_v)
            M = np.zeros((len(locs_u), len(locs_v)))
            for i, node_i in enumerate(locs_u):
                for j, node_j in enumerate(locs_v):
                    M[i, j] = path_lengths[node_i].get(node_j, 9999)
            emd = solve_wasserstein_scipy(mu_u, mu_v, M)
            edge_kappas.append(1.0 - emd)

        R = np.mean(edge_kappas)
        psi = G.nodes[u]['psi']; rho = psi.real**2 + psi.imag**2
        node_rhos.append(rho); node_curvatures.append(R)

    if len(node_rhos) < 5: return 0.0, 0.0
    slope, intercept, _, _, _ = stats.linregress(node_rhos, node_curvatures)
    return slope, intercept

def compute_spectral_dimension(G):
    N = G.number_of_nodes()
    try:
        L = nx.laplacian_matrix(G).astype(float)
        evals = linalg.eigh(L.todense(), eigvals_only=True)
        t_vals = np.logspace(-1, 2, 40)
        p_t = []
        for t in t_vals:
            trace = np.sum(np.exp(-evals * t))
            p_t.append(trace / N)
        log_t = np.log(t_vals); log_p = np.log(p_t)
        mask = (t_vals > 0.5) & (t_vals < 5.0)
        if np.sum(mask) < 5: return np.nan
        slope, _ = np.polyfit(log_t[mask], log_p[mask], 1)
        return -2 * slope
    except: return np.nan

def compute_hausdorff_dimension(G):
    centers = np.random.choice(list(G.nodes()), size=min(len(G), PROBE_COUNT), replace=False)
    rs_all, counts_all = [], []
    for source in centers:
        dists = nx.single_source_shortest_path_length(G, source)
        max_d = max(dists.values())
        if max_d < 2: continue
        counts = np.zeros(max_d + 1)
        for d in dists.values(): counts[d] += 1
        cumulative = np.cumsum(counts)
        valid_r = np.arange(1, len(cumulative))
        rs_all.extend(valid_r); counts_all.extend(cumulative[1:])

    if len(rs_all) < 5: return 0.0
    log_r = np.log(rs_all); log_n = np.log(counts_all)
    mask = (log_r > np.log(1.5)) & (log_r < np.log(8))
    if np.sum(mask) < 3: return 0.0
    slope, _ = np.polyfit(log_r[mask], log_n[mask], 1)
    return slope

# --- ANALYSIS & REPORTING ---

def analyze_run(item):
    """
    Worker function to process a single run.
    """
    np.random.seed(item['seed'])

    # This now computes the series AND saves 'metrics.csv' locally in the run folder
    evolution_series = get_run_evolution(item['folder'], item['N'])

    try:
        df_e = pd.read_csv(item['efile'])
        df_n = pd.read_csv(item['nfile'])

        # 1. Build Final Graph State
        G = nx.Graph()
        G.add_nodes_from(range(item['N']))
        for _, row in df_n.iterrows():
            nid = int(row['node_id'])
            if nid < item['N']: G.nodes[nid]['psi'] = row['psi_real'] + 1j * row['psi_imag']
        for _, row in df_e.iterrows():
            G.add_edge(int(row['source']), int(row['target']), theta=row['theta'])

        # 2. Compute Static Metrics (Final Step)
        mean_degree = (2 * G.number_of_edges()) / item['N']
        triangles = sum(nx.triangles(G).values()) // 3

        if mean_degree > 0.1:
            gauge_delta = check_gauge_invariance(G)
            hausdorff = compute_hausdorff_dimension(G)
            spec_dim = compute_spectral_dimension(G)
            grav_G, cosmo_lambda = compute_gravity_params(G)
        else:
            gauge_delta, hausdorff, spec_dim = 0.0, 0.0, np.nan
            grav_G, cosmo_lambda = 0.0, 0.0
    except:
        mean_degree = 0
        triangles = 0
        gauge_delta = 0
        hausdorff = 0
        spec_dim = np.nan
        grav_G = 0
        cosmo_lambda = 0

    return {
        "config_id": item['config_id'],
        "N": item['N'], "seed": item['seed'], "step": item['step'],
        "mean_degree": mean_degree, "triangles": triangles,
        "hausdorff": hausdorff, "spectral_dim": spec_dim,
        "gauge_delta": gauge_delta, "gravity_G": grav_G, "cosmo_lambda": cosmo_lambda,
        "evolution_series": evolution_series
    }

def print_final_summary(df, config_details_map):
    print("\n" + "="*170)
    print(f"{'MULTI-ENGINE PHYSICS COMPARISON':^170}")
    print("="*170)

    # Group by Config AND N
    stats = df.groupby(['config_id', 'N']).agg({
        'seed': ['count'],
        'mean_degree': ['mean', 'std'], 'triangles': ['mean', 'std'],
        'hausdorff': ['mean', 'std'], 'spectral_dim': ['mean', 'std'],
        'gravity_G': ['mean', 'std'], 'cosmo_lambda': ['mean', 'std'],
        'gauge_delta': ['max']
    })

    summary_header = (f"{'Cfg (Date)':<20} | {'N':<6} | {'Cnt':<4} | {'<k>':<12} | {'Tri':<12} | {'Hausdorff':<12} | "
                      f"{'Spectral':<12} | {'Grav G':<12} | {'Lambda':<12} | {'Gauge Err':<10}")
    print(summary_header)
    print("-" * 170)
    summary_lines = []

    for (cfg, n_val), row in stats.iterrows():
        count = row[('seed', 'count')]
        k_str = f"{row[('mean_degree','mean')]:.2f}±{row[('mean_degree','std')]:.2f}"
        t_str = f"{row[('triangles','mean')]:.1f}±{row[('triangles','std')]:.1f}"
        h_str = f"{row[('hausdorff','mean')]:.2f}±{row[('hausdorff','std')]:.2f}"

        s_mean = row[('spectral_dim','mean')]
        s_std = row[('spectral_dim','std')]
        s_str = f"{s_mean:.2f}±{s_std:.2f}" if not np.isnan(s_mean) else "N/A"

        g_str = f"{row[('gravity_G','mean')]:.2f}±{row[('gravity_G','std')]:.2f}"
        l_str = f"{row[('cosmo_lambda','mean')]:.2f}±{row[('cosmo_lambda','std')]:.2f}"
        e_str = f"{row[('gauge_delta','max')]:.1e}"

        line = f"{cfg:<20} | {n_val:<6} | {int(count):<4} | {k_str:<12} | {t_str:<12} | {h_str:<12} | {s_str:<12} | {g_str:<12} | {l_str:<12} | {e_str:<10}"
        print(line)
        summary_lines.append(line)
    print("="*170 + "\n")

    for cfg, details in config_details_map.items():
        safe_cfg = str(cfg).replace(":", "-").replace(" ", "_")
        fname = os.path.join(OUTPUT_DIR, f"Config_{safe_cfg}_details.txt")
        with open(fname, 'w') as f:
            f.write(details)
        print(f">> Saved details for engine {cfg} -> {fname}")

    return summary_lines, summary_header

def save_text_report(df, summary_lines, summary_header):
    outfile = os.path.join(OUTPUT_DIR, "dashboard_comparison.txt")
    with open(outfile, "w") as f:
        f.write("RELATIONAL REALITY MULTI-ENGINE DASHBOARD\n")
        f.write("=========================================\n\n")
        f.write("--- AGGREGATE STATS ---\n")
        f.write(summary_header + "\n")
        f.write("-" * 170 + "\n")
        f.write("\n".join(summary_lines))
        f.write("\n\n")
    print(f">> Full Comparative Report Saved: {outfile}")

def plot_time_series(df, metric_idx, metric_name, filename):
    plt.figure(figsize=(12, 7))
    configs = df['config_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for i, cfg in enumerate(configs):
        subset = df[df['config_id'] == cfg]
        all_series = []

        for _, row in subset.iterrows():
            ts = row['evolution_series']
            if not ts: continue

            steps = [x[0] for x in ts]
            vals = [x[metric_idx] for x in ts]

            plt.plot(steps, vals, color=colors[i], alpha=0.1, linewidth=0.5)

            s_df = pd.DataFrame({'step': steps, 'val': vals})
            s_df.set_index('step', inplace=True)
            all_series.append(s_df)

        if all_series:
            concat_df = pd.concat(all_series)
            mean_df = concat_df.groupby(concat_df.index).mean()
            plt.plot(mean_df.index, mean_df['val'], color=colors[i], linewidth=2.5, label=f"Mean: {cfg}")

    plt.title(f"{metric_name} Over Time (Dynamics)")
    plt.xlabel("Simulation Step")
    plt.ylabel(metric_name)
    plt.legend(title="Engine Configuration")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def plot_metric(df, metric, title, fname, log_y=False):
    plt.figure(figsize=(12, 7))

    configs = df['config_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for i, cfg in enumerate(configs):
        subset = df[df['config_id'] == cfg].dropna(subset=[metric])
        if subset.empty: continue

        jitter_N = subset['N'] * (1 + np.random.uniform(-0.03, 0.03, len(subset)))

        plt.scatter(jitter_N, subset[metric], color=colors[i], alpha=0.2, s=15, label=f"_nolegend_")

        means = subset.groupby('N')[metric].mean()
        plt.plot(means.index, means.values, marker='o', lw=2, color=colors[i], label=f"Cfg: {cfg}")

        if not means.empty:
            last_n = means.index[-1]
            last_v = means.values[-1]
            lbl = f"{last_v:.2f}"
            plt.text(last_n, last_v, lbl, color=colors[i], fontweight='bold', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.xscale('log')
    all_Ns = sorted(df['N'].unique())
    plt.xticks(all_Ns, all_Ns)
    if log_y: plt.yscale('log')

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("System Size (N)")
    plt.legend(title="Engine Configuration")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

def save_evolution_csv(df):
    rows = []
    for _, row in df.iterrows():
        evo = row['evolution_series']
        if not evo: continue

        cfg = row['config_id']
        N = row['N']
        seed = row['seed']

        for step_data in evo:
            rows.append({
                'config_id': cfg, 'N': N, 'seed': seed,
                'step': step_data[0],
                'triangles': step_data[1],
                'mean_degree': step_data[2]
            })

    if rows:
        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values(by=['config_id', 'N', 'seed', 'step'])
        out_path = os.path.join(OUTPUT_DIR, "evolution_metrics.csv")
        out_df.to_csv(out_path, index=False)
        print(f">> Detailed Evolution Metrics saved: {out_path}")

# --- MAIN EXECUTION ---

def main():
    print(">> Starting Multi-Engine Comparative Analysis (Parallelized)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folders = find_run_folders(RUNS_DIR)

    # GROUPING STEP
    candidates, config_map = scan_and_group_runs(folders)

    if not candidates:
        print(">> Analysis finished: No valid runs found matching the criteria.")
        return

    total_runs = len(candidates)
    cpu_cores = (int)(os.cpu_count() / 2.0)
    if cpu_cores < 1: cpu_cores = 1

    print(f">> Processing {total_runs} runs (Active & Finished) across {len(config_map)} configurations...")
    print(f">> Spawning ProcessPoolExecutor with {cpu_cores} workers for parallel analysis.")

    valid_results = []
    start_time = time.time()

    print("\n" + "="*170)
    print(f"{'LIVE SIMULATION MATRIX':^170}")
    print("="*170)
    print(f"{'#':<4} | {'Cfg (Date)':<20} | {'N':<6} | {'Seed':<6} | {'<k>':<8} | {'Tri':<8} | {'Haus':<8} | {'Spec':<8} | {'G_grav':<8} | {'Lambda':<8} | {'Gauge Err'}")
    print("-" * 170)

    # --- PARALLEL EXECUTION ---
    processed_count = 0
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = {executor.submit(analyze_run, item): item for item in candidates}

        for future in as_completed(futures):
            item = futures[future]
            processed_count += 1
            try:
                res = future.result()
                if res:
                    valid_results.append(res)
                    s_val = f"{res['spectral_dim']:.2f}" if not np.isnan(res['spectral_dim']) else "NaN"
                    line = (f"{processed_count:<4} | {res['config_id']:<20} | {int(res['N']):<6} | {int(res['seed']):<6} | "
                            f"{res['mean_degree']:<8.2f} | {int(res['triangles']):<8} | {res['hausdorff']:<8.2f} | {s_val:<8} | "
                            f"{res['gravity_G']:<8.2f} | {res['cosmo_lambda']:<8.2f} | {res['gauge_delta']:.1e}")
                    print(line)
                else:
                    print(f"{processed_count:<4} | {item['config_id']:<20} | {item['N']:<6} | {item['seed']:<6} | [ERROR]")
            except Exception as e:
                print(f"{processed_count:<4} | {item['config_id']:<20} | {item['N']:<6} | [EXCEPTION] {e}")

            if processed_count % 10 == 0 or processed_count == total_runs:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count
                remaining = (total_runs - processed_count) * avg_time
                print(f"   >>> [Progress: {processed_count}/{total_runs}] ETA: {str(timedelta(seconds=int(remaining)))} <<<")

    print("="*170 + "\n")

    if valid_results:
        df = pd.DataFrame(valid_results).sort_values(by=['config_id', 'N', 'seed'])
        summ_lines, summ_header = print_final_summary(df, config_map)
        save_text_report(df, summ_lines, summ_header)
        save_evolution_csv(df)

        print(">> Generating Comparative Plots...")
        plot_metric(df, 'mean_degree', '<k> Topology (Comparison)', 'comp_k.png')
        plot_metric(df, 'triangles', 'Triangle Count (Comparison)', 'comp_tri.png')
        plot_time_series(df, 1, "Triangles", "growth_triangles.png")
        plot_time_series(df, 2, "Mean Degree", "growth_mean_degree.png")
        plot_metric(df, 'hausdorff', 'Hausdorff Dimension (Scaling)', 'comp_dim_haus.png')
        plot_metric(df, 'spectral_dim', 'Spectral Dimension (Heat Kernel)', 'comp_dim_spec.png')
        plot_metric(df, 'gravity_G', 'Gravitational Coupling G', 'comp_gravity_G.png')
        plot_metric(df, 'cosmo_lambda', 'Cosmological Constant Lambda', 'comp_cosmo_lambda.png')
        plot_metric(df, 'gauge_delta', 'Gauge Error', 'comp_gauge.png', log_y=True)
        print(f">> Done. Results in {OUTPUT_DIR}/")
    else:
        print("Analysis produced no valid data rows.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
