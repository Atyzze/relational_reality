import os
import re
import glob
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

# --- CONFIGURATION ---
N_LANDMARKS = 150
TOP_K_EIGENVALUES = 10
SEED_FIXED = 42

class FastMDS_Spectrum:
    """
    Computes the Eigen Spectrum for topological analysis.
    """
    def __init__(self, n_landmarks=150, seed=42):
        self.n_landmarks = n_landmarks
        self.seed = seed

    def get_spectrum(self, adj_matrix, N):
        rng = np.random.RandomState(self.seed)

        # 1. Select Landmarks
        actual_k = min(N, self.n_landmarks)
        if actual_k < 2: return np.zeros(actual_k)

        landmarks = rng.choice(N, size=actual_k, replace=False)
        landmarks.sort()

        # 2. Compute Geodesics (Shortest Paths)
        # unweighted=True because we care about hops/topology, not specific edge weights (theta)
        D_L = shortest_path(adj_matrix, method='D', directed=False, indices=landmarks)

        # 3. Handle Disconnected Components (Infinite distances)
        finite = np.isfinite(D_L)
        if not np.any(finite):
            return np.zeros(actual_k)

        # Replace infinity with a penalty distance (1.5x max observed)
        max_dist = np.nanmax(D_L[finite])
        if max_dist <= 0: max_dist = 1.0
        D_L[~finite] = max_dist * 1.5

        # 4. Double Centering (MDS Kernel)
        D_L_sq = D_L ** 2
        D_LL_sq = D_L_sq[:, landmarks]
        n = actual_k
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_LL_sq @ J

        # 5. Eigen Decomposition
        eigvals = np.linalg.eigvalsh(B)
        # Sort descending
        eigvals = np.sort(eigvals)[::-1]

        return eigvals

def load_graph_structure(node_file, edge_file):
    """
    Fast graph loader matching drive.py CSV format.
    """
    try:
        # Load Nodes to get N and mapping
        # drive.py header: node_id, psi_real, psi_imag
        df_nodes = pd.read_csv(node_file)
        node_ids = df_nodes.iloc[:, 0].astype(int).values
        N = len(node_ids)

        if N == 0: return None, 0

        # Mapping node_id -> index (0..N-1)
        mapping = {nid: i for i, nid in enumerate(sorted(node_ids))}

        # Load Edges
        # drive.py header: source, target, theta
        if not os.path.exists(edge_file):
            return None, 0

        df_edges = pd.read_csv(edge_file)
        if df_edges.empty:
            # Return disconnected graph of size N
            adj = csr_matrix((N, N))
            return adj, N

        # Clean column names (strip whitespace)
        df_edges.columns = [c.strip().lower() for c in df_edges.columns]

        # Identify source/target columns
        u_col = df_edges.columns[0] # source
        v_col = df_edges.columns[1] # target

        # Map IDs to matrix indices
        u_vals = df_edges[u_col].map(mapping).dropna()
        v_vals = df_edges[v_col].map(mapping).dropna()

        # Create Sparse Adjacency Matrix
        data = np.ones(len(u_vals))
        adj = csr_matrix((data, (u_vals.values, v_vals.values)), shape=(N, N))

        # Symmetrize (Graph is undirected)
        adj = adj + adj.T

        return adj, N

    except Exception as e:
        # print(f"DEBUG: Error loading {node_file}: {e}")
        return None, 0

def parse_steps(data_dir):
    """
    Finds and sorts step files matching drive.py's format:
    E{v}_N{N}_S{s}_iter_{step}_nodes.csv
    """
    # Regex to match step number in filenames like: ...iter_0000000_nodes.csv
    pattern = re.compile(r"iter_([\d_]+)_nodes\.csv")

    if not os.path.isdir(data_dir):
        return []

    files = glob.glob(os.path.join(data_dir, "*_nodes.csv"))
    steps = []

    for f in files:
        m = pattern.search(f)
        if m:
            step_str = m.group(1)
            # Remove underscores from step string (e.g. 100_000 -> 100000)
            step_int = int(step_str.replace('_', ''))

            edge_f = f.replace("_nodes.csv", "_edges.csv")
            if os.path.exists(edge_f):
                steps.append((step_int, f, edge_f))

    # Sort by step index
    return sorted(steps, key=lambda x: x[0])

def main():
    parser = argparse.ArgumentParser(description="Eigen Spectrum Tracker")
    parser.add_argument("--dir", type=str, required=True, help="Path to 'data' folder")
    parser.add_argument("--max_files", type=int, default=None, help="Limit files processed")
    parser.add_argument("--output", type=str, default="spectral_history.png", help="Output image file")
    args = parser.parse_args()

    # --- 1. Path Validation ---
    target_dir = args.dir
    # Auto-fix relative paths if user is running from 'runs' parent
    if not os.path.exists(target_dir) and not target_dir.startswith("/"):
        potential = os.path.join(os.getcwd(), target_dir)
        if os.path.exists(potential):
            target_dir = potential

    if not os.path.exists(target_dir):
        print(f"[ERROR] Directory not found: {target_dir}")
        print(f"       Current working directory: {os.getcwd()}")
        sys.exit(1)

    print(f"Scanning: {target_dir}")

    # --- 2. File Discovery ---
    all_steps = parse_steps(target_dir)

    if not all_steps:
        print(f"[ERROR] No valid CSV files found in {target_dir}")
        print("       Expected format: *_iter_XXXXXX_nodes.csv")
        print("       Check your path? (e.g., runs/E7/N6400/S1000/data)")
        sys.exit(1)

    if args.max_files:
        all_steps = all_steps[:args.max_files]
        print(f"Limiting to first {len(all_steps)} steps.")
    else:
        print(f"Found {len(all_steps)} steps to process.")

    # --- 3. Analysis Loop ---
    history_vals = []
    history_steps = []

    mds = FastMDS_Spectrum(n_landmarks=N_LANDMARKS, seed=SEED_FIXED)

    # Wrap in try/except to save progress if interrupted
    try:
        for step, node_f, edge_f in tqdm(all_steps, unit="step"):
            adj, N = load_graph_structure(node_f, edge_f)

            # Skip corrupted/empty frames
            if adj is None or N == 0:
                continue

            eigs = mds.get_spectrum(adj, N)

            # Pad if fewer eigenvalues found than TOP_K
            if len(eigs) < TOP_K_EIGENVALUES:
                eigs = np.pad(eigs, (0, TOP_K_EIGENVALUES - len(eigs)))

            history_vals.append(eigs[:TOP_K_EIGENVALUES])
            history_steps.append(step)

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted! Plotting what we have so far...")

    # --- 4. Plotting ---
    if not history_vals:
        print("[ERROR] No valid data could be processed. Exiting.")
        sys.exit(1)

    hist_arr = np.array(history_vals) # Shape: (Steps, K)

   # ... (Keep previous imports and setup)

    # --- REPLACEMENT PLOTTING SECTION ---
    print(f"Plotting {len(history_steps)} frames...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Custom "Signal vs Noise" styling
    # λ1 = Primary Dimension (Red/Orange)
    # λ2 = Secondary Dimension (Cyan/Blue)
    # λ3 = Tertiary Dimension (Green)
    # λ4+ = Noise (Faint Gray)

    styles = {
        0: {"color": "#FF3333", "width": 2.5, "alpha": 1.0, "label": "λ1 (Primary)"},    # Red
        1: {"color": "#00CCFF", "width": 2.0, "alpha": 0.9, "label": "λ2 (Secondary)"},  # Cyan
        2: {"color": "#00FF66", "width": 1.5, "alpha": 0.8, "label": "λ3 (Tertiary)"},   # Green
    }

    # Plot noise first (so it stays in background)
    noise_plotted = False
    for k in range(3, TOP_K_EIGENVALUES):
        if k < hist_arr.shape[1]:
            ax.plot(history_steps, hist_arr[:, k],
                     color="#888888",
                     linewidth=0.5,
                     alpha=0.2,
                     label="Noise (λ4+)" if not noise_plotted else None)
            noise_plotted = True

    # Plot signal on top
    for k in range(3):
        if k < hist_arr.shape[1]:
            s = styles[k]
            ax.plot(history_steps, hist_arr[:, k],
                     color=s["color"],
                     linewidth=s["width"],
                     alpha=s["alpha"],
                     label=s["label"])

    ax.set_yscale('log')
    ax.set_xlabel('Simulation Steps', fontsize=12, color='white')
    ax.set_ylabel('Eigenvalue Magnitude (Log Scale)', fontsize=12, color='white')

    # Dark mode styling for better contrast
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    # Grid and Legend
    ax.grid(True, which="major", ls="-", alpha=0.3, color='#555555')
    ax.grid(True, which="minor", ls=":", alpha=0.1, color='#555555')

    legend = ax.legend(loc='upper right', frameon=True, facecolor='#222222', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    plt.title(f'Spectral History: Signal vs Noise (N={all_steps[0][0] if all_steps else "?"})', color='white', fontsize=14, pad=15)
    plt.tight_layout()

    plt.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n[DONE] Chart saved to: {os.path.abspath(args.output)}")
    print(f"\n[DONE] Chart saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
