import os
import re
import glob
import warnings
from collections import deque
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
SEED_FIXED = 42
GLOBAL_CMAP = "viridis"
JUMP_THRESHOLD = 0.25  # Max allowed delta in <k> before alerting/flagging

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

# --- MENU & FILE HELPERS ---

def get_subfolders(path):
    if not os.path.exists(path): return []
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def select_option(options, prompt_text):
    if not options:
        print(f"No options found for: {prompt_text}")
        return None
    if len(options) == 1:
        print(f">> Auto-selecting only option for {prompt_text}: {options[0]}")
        return options[0]

    print(f"\n--- {prompt_text} ---")
    for i, opt in enumerate(options):
        print(f"[{i+1}] {opt}")

    while True:
        try:
            choice = input(f"Select (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options): return options[idx]
        except ValueError: pass
        print("Invalid selection. Try again.")

def open_file_explorer(path):
    try:
        system = platform.system()
        if system == "Windows": os.startfile(path)
        elif system == "Darwin": subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else: subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e: print(f"Could not open file explorer: {e}")

def get_rendered_steps(output_dir):
    if not os.path.exists(output_dir): return set()
    rendered = set()
    pattern = re.compile(rf"Step_([\d_]+)\.png")
    for f in os.listdir(output_dir):
        m = pattern.search(f)
        if m:
            step_val = int(m.group(1).replace('_', ''))
            rendered.add(step_val)
    return rendered

def smart_sample_sort(all_steps, rendered_steps):
    """
    Creates a prioritized task list by interleaving two strategies:
    1. BSP (Binary Space Partitioning): 0%, 100%, 50%, 25%, 75%... (Global structure)
    2. Sequential: 0, 1, 2, 3... (Smooth playback from start)
    """
    if not all_steps: return []

    N = len(all_steps)

    # --- Strategy 1: Binary Space Partitioning (The "Structure" Group) ---
    bsp_indices = []
    visited_bsp = set()
    queue = deque([(0, N - 1)])

    # Always add Start and End first
    if N > 0:
        bsp_indices.append(0)
        visited_bsp.add(0)
    if N > 1:
        bsp_indices.append(N - 1)
        visited_bsp.add(N - 1)

    while queue:
        start, end = queue.popleft()
        if end - start <= 1: continue

        mid = (start + end) // 2
        if mid not in visited_bsp:
            bsp_indices.append(mid)
            visited_bsp.add(mid)

        # Prioritize left (start->mid) then right (mid->end)
        # This gives us 25% then 75%
        queue.append((start, mid))
        queue.append((mid, end))

    # Fill gaps for BSP (in case integer math skipped any)
    for i in range(N):
        if i not in visited_bsp:
            bsp_indices.append(i)

    # --- Strategy 2: Sequential (The "Playback" Group) ---
    seq_indices = list(range(N))

    # --- Interleave (The "Two Groups" Logic) ---
    final_indices = []
    added_indices = set()

    # Zip them together: [BSP_0, SEQ_0, BSP_1, SEQ_1, ...]
    max_len = max(len(bsp_indices), len(seq_indices))
    for i in range(max_len):
        # Pick from BSP
        if i < len(bsp_indices):
            idx = bsp_indices[i]
            if idx not in added_indices:
                final_indices.append(idx)
                added_indices.add(idx)

        # Pick from Sequential
        if i < len(seq_indices):
            idx = seq_indices[i]
            if idx not in added_indices:
                final_indices.append(idx)
                added_indices.add(idx)

    # Convert indices back to step numbers and filter rendered
    # Note: We check if the *step value* is in rendered_steps
    return [all_steps[i] for i in final_indices if all_steps[i] not in rendered_steps]
# --- CORE MATH & GRAPH LOGIC ---

def parse_steps(data_dir, version_id, N, seed):
    search_pattern = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_*_nodes.csv")
    node_files = glob.glob(search_pattern)
    data_map = {}
    pattern = re.compile(rf"iter_([\d_]+)_nodes\.csv")

    for nf in node_files:
        match = pattern.search(nf)
        if match:
            step_str = match.group(1)
            step_int = int(step_str.replace('_', ''))
            edge_file = nf.replace("_nodes.csv", "_edges.csv")
            if os.path.exists(edge_file):
                data_map[step_int] = (nf, edge_file, step_str)
    return sorted(data_map.keys()), data_map

def load_graph(node_file, edge_file):
    try:
        df_nodes = pd.read_csv(node_file)
        df_nodes.columns = [c.strip().lower() for c in df_nodes.columns]
        real_col = next((c for c in df_nodes.columns if 'real' in c or 're' in c), "psi_real")
        imag_col = next((c for c in df_nodes.columns if 'imag' in c or 'im' in c), "psi_imag")

        G = nx.Graph()
        for _, row in df_nodes.iterrows():
            nid = int(row.iloc[0])
            psi = complex(row.get(real_col, 0), row.get(imag_col, 0))
            G.add_node(nid, psi=psi, rho=abs(psi)**2)

        df_edges = pd.read_csv(edge_file)
        if not df_edges.empty:
            df_edges.columns = [c.strip().lower() for c in df_edges.columns]
            u_col, v_col = df_edges.columns[0], df_edges.columns[1]
            G.add_edges_from([(int(r[u_col]), int(r[v_col])) for _, r in df_edges.iterrows()])
        return G
    except: return nx.Graph()

def get_distance_matrix(G):
    N = len(G)
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    row, col, data = [], [], []
    for u, v in G.edges():
        row.append(node_to_idx[u]); col.append(node_to_idx[v]); data.append(1.0)
        row.append(node_to_idx[v]); col.append(node_to_idx[u]); data.append(1.0)
    sparse_mat = csr_matrix((data, (row, col)), shape=(N, N))
    dist_matrix = shortest_path(sparse_mat, method='D', directed=False, unweighted=True)
    finite = np.isfinite(dist_matrix)
    if np.any(finite): dist_matrix[~finite] = dist_matrix[finite].max() * 2.0
    else: dist_matrix[:] = 1.0
    return dist_matrix

# --- RENDERING WORKER ---
def process_single_step(args):
    step_int, node_f, edge_f, step_str, output_dir, v_id, N_val, s_id = args

    try:
        np.random.seed(SEED_FIXED + step_int)
        G = load_graph(node_f, edge_f)
        N = G.number_of_nodes()
        rhos = np.array([G.nodes[n].get('rho', 0) for n in G.nodes()])

        # --- K-Distribution Logic ---
        degrees = np.array([d for _, d in G.degree()])
        n_edges = G.number_of_edges()
        mean_deg = (2 * n_edges / N) if N > 0 else 0

        out_name = f"E{v_id}_N{N_val}_S{s_id}_i{step_str}_k{mean_deg:.3f}.png"
        out_path = os.path.join(output_dir, out_name)
        # Check if exists (comment out if you want to force overwrite)
        if os.path.exists(out_path): return

        v_min, v_max = (min(rhos), max(rhos)) if len(rhos) > 0 else (0, 1)

        # Layouts
        pos_spring = nx.spring_layout(G, k=0.15, iterations=50, seed=SEED_FIXED)
        dist_matrix = get_distance_matrix(G)
        mds_2d = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED_FIXED, normalized_stress="auto")
        pos_mds_2d = mds_2d.fit_transform(dist_matrix)
        mds_3d = MDS(n_components=3, dissimilarity="precomputed", random_state=SEED_FIXED, normalized_stress="auto")
        pos_mds_3d = mds_3d.fit_transform(dist_matrix)
        try:
            # 1. Calculate the raw layout
            spec_dict = nx.spectral_layout(G)

            # 2. STABILIZATION: Pick a consistent anchor (e.g., the node with the lowest ID)
            # This ensures we are always checking the same specific node every frame.
            anchor_id = min(G.nodes())

            # 3. Check the anchor's position.
            # If it's negative, we flip the whole world to make it positive.
            anchor_x, anchor_y = spec_dict[anchor_id]
            flip_x = -1 if anchor_x < 0 else 1
            flip_y = -1 if anchor_y < 0 else 1

            # 4. Build the array applying the flip correction
            pos_spec = np.array([
                [spec_dict[n][0] * flip_x, spec_dict[n][1] * flip_y]
                for n in G.nodes()
            ])

        except: pos_spec = np.zeros((N, 2))

        # --- Plotting ---
        fig = plt.figure(figsize=(20, 17))
        scatter_args = {'c': rhos, 'cmap': GLOBAL_CMAP, 'vmin': v_min, 'vmax': v_max, 's': 20, 'alpha': 0.9}

        for i, (pos, title, proj) in enumerate([
            (pos_spring, "1. Physical Topology (Spring)", None),
            (pos_spec, "2. Quantum Resonance (Spectral)", None),
            (pos_mds_2d, "3. Emergent Manifold (MDS 2D)", None),
            (pos_mds_3d, "4. The Hologram (MDS 3D)", '3d')
        ]):
            ax = fig.add_subplot(2, 2, i+1, projection=proj)

            if proj == '3d':
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], **scatter_args)
                # Hide panes
                ax.xaxis.set_pane_color((1, 1, 1, 0))
                ax.yaxis.set_pane_color((1, 1, 1, 0))
                ax.zaxis.set_pane_color((1, 1, 1, 0))

                # --- FIX: Zoom and Center 3D Plot ---
                # 1. Force equal aspect ratio (cube)
                ax.set_box_aspect([1, 1, 1])

                # 2. Calculate centered bounds
                max_range = np.array([
                    pos[:,0].max() - pos[:,0].min(),
                    pos[:,1].max() - pos[:,1].min(),
                    pos[:,2].max() - pos[:,2].min()
                ]).max() / 2.0

                mid_x = (pos[:,0].max() + pos[:,0].min()) * 0.5
                mid_y = (pos[:,1].max() + pos[:,1].min()) * 0.5
                mid_z = (pos[:,2].max() + pos[:,2].min()) * 0.5

                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                # 3. Adjust Camera Distance (Lower = Closer/Bigger)
                # Default is ~10. Setting to 7 fills the quadrant better.
                ax.dist = 2

            elif i == 0:
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.05, edge_color="gray")
                sc_map = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20, node_color=rhos, cmap=GLOBAL_CMAP, vmin=v_min, vmax=v_max, alpha=0.9)
            else:
                ax.scatter(pos[:, 0], pos[:, 1], **scatter_args)

            ax.set_title(title, fontsize=14)
            ax.axis("off")

        fig.suptitle("Relational Reality", fontsize=22, y=0.98, fontweight='bold')
        step_formatted = f"{step_int:,}".replace(",", "_")
        sub_title_1 = f"E{v_id} | S{s_id} | N={N_val} | Step {step_formatted} | Avg <k>={mean_deg:.2f}"
        fig.text(0.5, 0.95, sub_title_1, ha='center', fontsize=15)

        # --- K-Distribution Mini Chart ---
        ax_hist = fig.add_axes([0.30, 0.89, 0.40, 0.05])
        if len(degrees) > 0:
            k_min, k_max = int(min(degrees)), int(max(degrees))

            if k_max - k_min < 1:
                bins = [k_min - 0.5, k_min + 0.5]
            elif k_max - k_min < 40:
                bins = np.arange(k_min, k_max + 2) - 0.5
            else:
                bins = 40

            counts, bin_edges = np.histogram(degrees, bins=bins)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = (bin_edges[1] - bin_edges[0]) * 0.9

            bin_colors = []
            cmap = plt.get_cmap(GLOBAL_CMAP)
            for i in range(len(bin_edges)-1):
                mask = (degrees >= bin_edges[i]) & (degrees < bin_edges[i+1])
                if np.any(mask):
                    val_rho = np.max(rhos[mask])
                    norm_val = (val_rho - v_min) / (v_max - v_min) if v_max > v_min else 0
                    bin_colors.append(cmap(norm_val))
                else:
                    bin_colors.append((0.5, 0.5, 0.5, 0.1))

            bars = ax_hist.bar(centers, counts, width=width, color=bin_colors, edgecolor='none', alpha=0.9)
            ax_hist.bar_label(bars, labels=[f'{int(v)}' if v > 0 else '' for v in counts],
                              padding=2, fontsize=7, fontweight='bold')
            ax_hist.axvline(mean_deg, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax_hist.set_title(f"Degree Distribution ($k$)", fontsize=10, loc='left', pad=2)
            ax_hist.tick_params(axis='x', labelsize=8)
            ax_hist.tick_params(axis='y', left=False, labelleft=False)
            ax_hist.set_ylim(0, max(counts) * 1.35)
            for spine in ['top', 'right', 'left']: ax_hist.spines[spine].set_visible(False)
            ax_hist.patch.set_alpha(0)
        else:
            ax_hist.axis('off')

        fig.subplots_adjust(top=0.86, bottom=0.1)
        cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        cb = fig.colorbar(sc_map, cax=cbar_ax, orientation='horizontal')
        cb.set_label(r'$\rho$ (Density)', fontsize=14)

        plt.savefig(out_path, dpi=300)
        plt.close(fig)
    except Exception as e: print(f"Error Step {step_int}: {e}")

def main():
    print("=========================================")
    print("   RELATIONAL REALITY VISUALIZER V4.2    ")
    print("=========================================")

    versions = get_subfolders(RUNS_DIR)
    versions.sort()
    ver_str = select_option(versions, "Select Engine Version")
    if not ver_str: return
    n_path = os.path.join(RUNS_DIR, ver_str)
    n_counts = get_subfolders(n_path)
    n_counts.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    n_str = select_option(n_counts, "Select System Size (N)")
    if not n_str: return
    seed_path = os.path.join(n_path, n_str)
    seeds = get_subfolders(seed_path)
    seeds.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    seed_str = select_option(seeds, "Select Simulation Seed")
    if not seed_str: return

    full_seed_path = os.path.join(seed_path, seed_str)
    data_dir = os.path.join(full_seed_path, "data")
    output_dir = os.path.join(full_seed_path, "renders")
    os.makedirs(output_dir, exist_ok=True)

    v_id = int(ver_str[1:]); n_id = int(n_str[1:]); s_id = int(seed_str[1:])

    steps_sorted, data_map = parse_steps(data_dir, v_id, n_id, s_id)
    rendered_steps = get_rendered_steps(output_dir)
    steps_to_render = smart_sample_sort(steps_sorted, rendered_steps)

    if not steps_to_render:
        print("\n>> All available frames are already rendered.")
        return

    print(f"\n>> Found {len(steps_to_render)} new frames to render.")
    tasks = [(s, data_map[s][0], data_map[s][1], data_map[s][2], output_dir, v_id, n_id, s_id) for s in steps_to_render]

    n_cores = min(cpu_count(), 4)
    print(f">> Rendering on {n_cores} cores...")
    open_file_explorer(output_dir)
    with Pool(processes=n_cores) as pool:
        list(tqdm(pool.imap(process_single_step, tasks), total=len(tasks)))

    print("\n[DONE] Visualization V4.2 complete.")

if __name__ == "__main__":
    main()
