import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from mpl_toolkits.mplot3d import Axes3D
import platform
import subprocess

# --- CONFIGURATION ---
RUNS_DIR = "runs"
SEED_FIXED = 42 # For visualization consistency
MAX_NODES_LIMIT = 15000 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

# --- MENU HELPERS ---

def get_subfolders(path):
    """Returns a list of subfolders in a given path."""
    if not os.path.exists(path):
        return []
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def select_option(options, prompt_text):
    """Console menu with auto-select for single options."""
    if not options:
        print(f"No options found for: {prompt_text}")
        return None

    # --- AUTO-SELECT LOGIC ---
    if len(options) == 1:
        print(f">> Auto-selecting only option for {prompt_text}: {options[0]}")
        return options[0]
    # -------------------------

    print(f"\n--- {prompt_text} ---")
    for i, opt in enumerate(options):
        print(f"[{i+1}] {opt}")

    while True:
        try:
            choice = input(f"Select (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

def open_file_explorer(path):
    """Opens the file explorer to the given path in a cross-platform way."""
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", path])
        else:  # Linux/Unix
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"Could not open file explorer: {e}")

# --- HYBRID SAMPLING LOGIC ---

def smart_sample_sort(items):
    """
    Hybrid rendering order:
    1. Key Frames (Last, First, 50%, 25%, 75%, 12.5%, etc.)
    2. Sequential Fill for 0-25% (Early universe details)
    3. Bisection/Jump Fill for 25-100% (Broad evolution details)
    """
    n = len(items)
    if n == 0: return []
    if n <= 2: return list(reversed(items))

    ordered_indices = []
    visited = set()

    def add(idx):
        if 0 <= idx < n and idx not in visited:
            ordered_indices.append(idx)
            visited.add(idx)

    # --- PHASE 1: KEY FRAMES ---
    add(n - 1)
    add(0)
    add(int(n * 0.5))
    p25 = int(n * 0.25)
    add(p25)
    add(int(n * 0.75))
    add(int(n * 0.125))
    add(int(n * 0.375))
    add(int(n * 0.625))
    add(int(n * 0.875))

    # --- PHASE 2: SEQUENTIAL 0-25% ---
    for i in range(p25 + 1):
        add(i)

    # --- PHASE 3: JUMP AROUND REMAINING 75% ---
    queue = [(p25, n - 1)]

    while queue:
        new_queue = []
        for start, end in queue:
            mid = (start + end) // 2
            add(mid)
            if mid - start > 1:
                new_queue.append((start, mid))
            if end - mid > 1:
                new_queue.append((mid, end))
        queue = new_queue

    return [items[i] for i in ordered_indices]

# --- CORE LOGIC ---

def parse_steps(data_dir, version_id, N, seed):
    """Finds all matching node/edge CSV pairs in the specific data folder."""
    # Pattern: E{v}_N{n}_S{s}_iter_{step}_nodes.csv
    search_pattern = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_*_nodes.csv")
    node_files = glob.glob(search_pattern)

    data_map = {}
    # Regex: ...iter_([\d_]+)_nodes.csv
    pattern = re.compile(rf"iter_([\d_]+)_nodes\.csv")

    for nf in node_files:
        match = pattern.search(nf)
        if match:
            step_str = match.group(1)
            # Remove underscores for integer sorting
            step_int = int(step_str.replace('_', ''))

            # Construct edge filename
            edge_file = nf.replace("_nodes.csv", "_edges.csv")

            if os.path.exists(edge_file):
                data_map[step_int] = (nf, edge_file, step_str)

    return sorted(data_map.keys()), data_map

def load_graph(node_file, edge_file):
    """Reads CSVs and builds a NetworkX graph."""
    try:
        df_nodes = pd.read_csv(node_file)
        df_nodes.columns = [c.strip().lower() for c in df_nodes.columns]

        real_col = next((c for c in df_nodes.columns if 'real' in c or 're' in c), None)
        imag_col = next((c for c in df_nodes.columns if 'imag' in c or 'im' in c), None)

        G = nx.Graph()
        if real_col and imag_col:
            for _, row in df_nodes.iterrows():
                nid = int(row.iloc[0])
                psi = complex(row[real_col], row[imag_col])
                rho = abs(psi)**2
                G.add_node(nid, psi=psi, rho=rho)

        df_edges = pd.read_csv(edge_file)
        if not df_edges.empty:
            df_edges.columns = [c.strip().lower() for c in df_edges.columns]
            u_col, v_col = df_edges.columns[0], df_edges.columns[1]
            edges = [(int(r[u_col]), int(r[v_col])) for _, r in df_edges.iterrows()]
            G.add_edges_from(edges)

        return G
    except Exception:
        return nx.Graph()

def get_distance_matrix(G):
    """Calculates All-Pairs Shortest Path using Scipy (Fast)."""
    N = len(G)
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    row, col, data = [], [], []
    for u, v in G.edges():
        row.append(node_to_idx[u])
        col.append(node_to_idx[v])
        data.append(1.0)
        row.append(node_to_idx[v])
        col.append(node_to_idx[u])
        data.append(1.0)

    sparse_mat = csr_matrix((data, (row, col)), shape=(N, N))
    dist_matrix = shortest_path(sparse_mat, method='D', directed=False, unweighted=True)

    finite = np.isfinite(dist_matrix)
    if np.any(finite):
        max_d = dist_matrix[finite].max()
        dist_matrix[~finite] = max_d * 2.0
    else:
        dist_matrix[:] = 1.0

    return dist_matrix

def process_single_step(args):
    """Worker function to process one frame."""
    step_int, node_f, edge_f, step_str, output_dir = args

    # Output filename: viz_iter_000_000_000.png
    out_name = f"viz_iter_{step_str}.png"
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        return

    try:
        # Re-seed for consistency
        np.random.seed(SEED_FIXED + step_int)

        # 1. Load Graph
        G = load_graph(node_f, edge_f)
        N = G.number_of_nodes()
        rhos = [G.nodes[n].get('rho', 0) for n in G.nodes()]

        # --- CALCULATE LAYOUTS ---

        # A. Topology: Spring (Fast)
        pos_spring = nx.spring_layout(G, k=0.15, iterations=50, seed=SEED_FIXED)

        # B. Quantum: Spectral (Very Fast)
        if G.number_of_edges() > 0:
            try:
                pos_spec = nx.spectral_layout(G)
                pos_spec = np.array([pos_spec[n] for n in G.nodes()])
            except:
                pos_spec = np.zeros((N, 2))
        else:
            pos_spec = np.zeros((N, 2))

        # C & D. Classical & Hologram: MDS (Slow)
        if G.number_of_edges() > 0 and N <= MAX_NODES_LIMIT:
            dist_matrix = get_distance_matrix(G)

            # 2D MDS
            mds_2d = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED_FIXED, n_init=4, max_iter=300, normalized_stress="auto")
            pos_mds_2d = mds_2d.fit_transform(dist_matrix)

            # 3D MDS
            mds_3d = MDS(n_components=3, dissimilarity="precomputed", random_state=SEED_FIXED, n_init=4, max_iter=300, normalized_stress="auto")
            pos_mds_3d = mds_3d.fit_transform(dist_matrix)
        else:
            # Fallback for empty graph or too large
            pos_mds_2d = np.zeros((N, 2))
            pos_mds_3d = np.zeros((N, 3))

        # --- PLOTTING (2x2 Grid) ---
        fig = plt.figure(figsize=(20, 16))

        # 1. TOP LEFT: Topology
        ax1 = fig.add_subplot(2, 2, 1)
        nx.draw_networkx_edges(G, pos_spring, ax=ax1, alpha=0.03, edge_color="gray")
        nx.draw_networkx_nodes(G, pos_spring, ax=ax1, node_size=10, node_color=rhos, cmap="plasma")
        ax1.set_title("1. Physical Topology (Spring)", fontsize=14)
        ax1.axis("off")

        # 2. TOP RIGHT: Spectral
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(pos_spec[:, 0], pos_spec[:, 1], s=15, alpha=0.6, c=rhos, cmap="inferno")
        ax2.set_title("2. Quantum Resonance (Spectral)", fontsize=14)
        ax2.axis("off")

        # 3. BOTTOM LEFT: Classical Map (MDS 2D)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(pos_mds_2d[:, 0], pos_mds_2d[:, 1], s=15, alpha=0.6, c=rhos, cmap="viridis")
        ax3.set_title("3. Emergent Manifold (MDS 2D)", fontsize=14)
        ax3.axis("off")

        # 4. BOTTOM RIGHT: Hologram (MDS 3D)
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(pos_mds_3d[:, 0], pos_mds_3d[:, 1], pos_mds_3d[:, 2], c=rhos, cmap="viridis", s=12, alpha=0.6)
        ax4.set_title("4. The Hologram (MDS 3D)", fontsize=14)
        ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Metadata
        n_edges = G.number_of_edges()
        mean_deg = (2 * n_edges / N) if N > 0 else 0
        fig.suptitle(f"Relational Reality | Step {step_int} | N={N} | <k>={mean_deg:.2f}", fontsize=16)

        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close(fig)

    except Exception as e:
        print(f"\n[!] Error on step {step_int}: {e}")

def main():
    print("=========================================")
    print("   RELATIONAL REALITY VISUALIZER (CLI)   ")
    print("=========================================")

    # 1. Select Version
    versions = get_subfolders(RUNS_DIR)
    versions.sort()
    if not versions:
        print(f"No Engine Versions found in '{RUNS_DIR}'")
        return
    ver_str = select_option(versions, "Select Engine Version")
    if not ver_str: return

    # 2. Select N (System Size) - SORTED NUMERICALLY
    n_path = os.path.join(RUNS_DIR, ver_str)
    n_counts = get_subfolders(n_path)

    def sort_key_n(x):
        if x.startswith('N') and x[1:].isdigit():
            return int(x[1:])
        return float('inf')

    n_counts.sort(key=sort_key_n)

    if not n_counts:
        print("No N-counts found.")
        return
    n_str = select_option(n_counts, "Select System Size (N)")
    if not n_str: return

    # 3. Select Seed
    seed_path = os.path.join(n_path, n_str)
    seeds = get_subfolders(seed_path)

    def sort_key_s(x):
        if x.startswith('S') and x[1:].isdigit():
            return int(x[1:])
        return float('inf')

    seeds.sort(key=sort_key_s)

    if not seeds:
        print("No Seeds found.")
        return
    seed_str = select_option(seeds, "Select Simulation Seed")
    if not seed_str: return

    # Paths
    full_seed_path = os.path.join(seed_path, seed_str)
    data_dir = os.path.join(full_seed_path, "data")
    output_dir = os.path.join(full_seed_path, "renders")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Extract clean integers for parsing regex
    try:
        v_id = int(ver_str[1:])
        n_id = int(n_str[1:])
        s_id = int(seed_str[1:])
    except ValueError:
        print("Error parsing directory names (Expected format E0, N100, S1000)")
        return

    print(f"\n>> Scanning data in: {data_dir}...")
    steps_sorted, data_map = parse_steps(data_dir, v_id, n_id, s_id)

    if not steps_sorted:
        print("No matching data files found.")
        return

    print(f"   Found {len(steps_sorted)} snapshots.")

    # --- REORDER FOR HYBRID SAMPLING ---
    print(f">> Reordering tasks (Hybrid: Key Frames -> Seq 0-25% -> Smart Sample 75%)")
    steps_smart_order = smart_sample_sort(steps_sorted)

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print(f">> Saving renders to: {output_dir}")

    # Prepare tasks in SMART ORDER
    tasks = []
    for step in steps_smart_order:
        node_f, edge_f, step_s = data_map[step]
        tasks.append((step, node_f, edge_f, step_s, output_dir))

    # --- CORE SELECTION ---
    max_cores = cpu_count()
    default_cores = 1
    print(f"\n--- Resource Allocation ---")
    print(f"Available Cores: {max_cores}")

    while True:
        core_input = input(f"Enter number of cores to use (Default: {default_cores}): ").strip()
        if not core_input:
            n_cores = default_cores
            break
        try:
            val = int(core_input)
            if 1 <= val <= max_cores:
                n_cores = val
                break
            else:
                print(f"Please enter a number between 1 and {max_cores}.")
        except ValueError:
            print("Invalid input.")

    print(f">> Rendering on {n_cores} cores...")

    # Run
    open_file_explorer(output_dir)
    with Pool(processes=n_cores) as pool:
        list(tqdm(pool.imap(process_single_step, tasks), total=len(tasks)))

    print("\n[DONE] Visualization complete.")
    print(f"Opening folder: {output_dir}")
    open_file_explorer(output_dir)

if __name__ == "__main__":
    main()
