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
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plot

# --- CONFIGURATION ---
INPUT_DIR = "."
OUTPUT_DIR = "renders_ultimate"  # New folder for the 4-panel view
SEED = 42

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

def parse_steps(input_dir):
    """Finds all matching node/edge CSV pairs."""
    node_files = glob.glob(os.path.join(input_dir, "nodes_step_*.csv"))
    data_map = {}
    for nf in node_files:
        match = re.search(r"nodes_step_(\d+)\.csv", nf)
        if match:
            step = int(match.group(1))
            edge_file = os.path.join(input_dir, f"edges_step_{step}.csv")
            if os.path.exists(edge_file):
                data_map[step] = (nf, edge_file)
    return sorted(data_map.keys()), data_map

def load_graph(node_file, edge_file):
    """Reads CSVs and builds a NetworkX graph."""
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

    try:
        df_edges = pd.read_csv(edge_file)
        if not df_edges.empty:
            df_edges.columns = [c.strip().lower() for c in df_edges.columns]
            u_col, v_col = df_edges.columns[0], df_edges.columns[1]
            edges = [(int(r[u_col]), int(r[v_col])) for _, r in df_edges.iterrows()]
            G.add_edges_from(edges)
    except pd.errors.EmptyDataError:
        pass
    return G

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
    step, node_f, edge_f = args
    out_name = f"viz_ultimate_step_{step:09d}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if os.path.exists(out_path):
        return

    try:
        # Re-seed for consistency
        np.random.seed(SEED + step)

        # 1. Load Graph
        G = load_graph(node_f, edge_f)
        N = G.number_of_nodes()
        rhos = [G.nodes[n].get('rho', 0) for n in G.nodes()]

        # --- CALCULATE LAYOUTS ---

        # A. Topology: Spring (Fast)
        pos_spring = nx.spring_layout(G, k=0.15, iterations=50, seed=SEED)

        # B. Quantum: Spectral (Very Fast)
        # This gives the "Comet" view
        if G.number_of_edges() > 0:
            pos_spec = nx.spectral_layout(G)
            pos_spec = np.array([pos_spec[n] for n in G.nodes()])
        else:
            pos_spec = np.zeros((N, 2))

        # C & D. Classical & Hologram: MDS (Slow)
        if G.number_of_edges() > 0:
            dist_matrix = get_distance_matrix(G)

            # 2D MDS
            mds_2d = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=4, max_iter=300, normalized_stress="auto")
            pos_mds_2d = mds_2d.fit_transform(dist_matrix)

            # 3D MDS
            mds_3d = MDS(n_components=3, dissimilarity="precomputed", random_state=SEED, n_init=4, max_iter=300, normalized_stress="auto")
            pos_mds_3d = mds_3d.fit_transform(dist_matrix)
        else:
            pos_mds_2d = np.zeros((N, 2))
            pos_mds_3d = np.zeros((N, 3))

        # --- PLOTTING (2x2 Grid) ---
        fig = plt.figure(figsize=(20, 16))

        # 1. TOP LEFT: Topology
        ax1 = fig.add_subplot(2, 2, 1)
        nx.draw_networkx_edges(G, pos_spring, ax=ax1, alpha=0.03, edge_color="gray")
        nx.draw_networkx_nodes(G, pos_spring, ax=ax1, node_size=5, node_color=rhos, cmap="plasma")
        ax1.set_title("1. Physical Topology (Spring)\n[Clustering View]", fontsize=14)
        ax1.axis("off")

        # 2. TOP RIGHT: Spectral
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(pos_spec[:, 0], pos_spec[:, 1], s=10, alpha=0.6, c=rhos, cmap="inferno") # Inferno for quantum vibe
        ax2.set_title("2. Quantum Resonance (Spectral)\n[Vibrational View]", fontsize=14)
        ax2.axis("off")

        # 3. BOTTOM LEFT: Classical Map (MDS 2D)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(pos_mds_2d[:, 0], pos_mds_2d[:, 1], s=10, alpha=0.6, c=rhos, cmap="viridis")
        ax3.set_title("3. Emergent Manifold (MDS 2D)\n[Map View]", fontsize=14)
        ax3.axis("off")

        # 4. BOTTOM RIGHT: Hologram (MDS 3D)
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(pos_mds_3d[:, 0], pos_mds_3d[:, 1], pos_mds_3d[:, 2], c=rhos, cmap="viridis", s=8, alpha=0.6)
        ax4.set_title("4. The Hologram (MDS 3D)\n[Truth View]", fontsize=14)
        # Transparent background for 3D
        ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Metadata
        n_edges = G.number_of_edges()
        mean_deg = (2 * n_edges / N) if N > 0 else 0
        fig.suptitle(f"Relational Reality Ultimate View | Step {step} | N={N} | <k>={mean_deg:.2f}", fontsize=16)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150) # 150 dpi is good balance for large images
        plt.close(fig)

    except Exception as e:
        print(f"\n[!] Error on step {step}: {e}")

def main():
    print(">>> ULTIMATE REALITY VISUALIZER (4-MODE) <<<")
    steps, data_map = parse_steps(INPUT_DIR)

    if not steps:
        print("No data found!")
        return

    # Sort descending to see the latest universe state first
    tasks = [(step, *data_map[step]) for step in sorted(steps, reverse=False)]

    # Adjust cores based on RAM. 3D MDS + 2D MDS is memory intensive.
    n_cores = min(cpu_count(), 10)
    print(f"Rendering {len(tasks)} frames (4-Panel) on {n_cores} cores.")

    with Pool(processes=n_cores) as pool:
        list(tqdm(pool.imap_unordered(process_single_step, tasks), total=len(tasks)))

    print(f"\nDone! Check '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
