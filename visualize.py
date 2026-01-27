import os
import sys
import re
import glob
import warnings
import argparse
import platform
import subprocess
import threading
from collections import deque
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

# --- CONFIGURATION ---
RUNS_DIR = "runs"
SEED_FIXED = 42
GLOBAL_CMAP = "viridis"

# --- TEMPORAL STABILITY CONFIG ---
ENABLE_PROCRUSTES_ALIGNMENT = True

# If enabled, applies "camera inertia": blends current alignment with previous alignment transform
# to reduce sudden global rotations between frames.
ENABLE_CAMERA_INERTIA = True
# 0.0 = no smoothing (immediate), 0.9 = very smooth/laggy
INERTIA_ALPHA = 0.80

# Cache directory inside each run's renders folder
PROCRUSTES_CACHE_DIRNAME = ".align_cache"
MIN_SHARED_NODES_FOR_ALIGNMENT = 20

# --- SPREADING / PREVIEW CONFIG ---
# Enables the "two lanes" render: forward-aligned + spread-preview concurrently.
ENABLE_DUAL_LANE_RENDER = True

# How many "spread" frames to render ASAP (BSP order). Set None to render all missing.
SPREAD_MAX_FRAMES = 250

# In aligned-forward lane: if PNG already exists but cache is missing, compute + write cache,
# but skip writing the PNG to save time.
ALIGNED_FORWARD_SKIP_PNG_IF_EXISTS = True


# --- FAST MDS / LMDS ---

class FastMDS:
    """
    Landmark MDS (LMDS) approximation to classical MDS on graph geodesic distances.
    Returns embedding + landmark-kernel eigen spectrum for explained-variance diagnostics.
    """
    def __init__(self, n_components=2, n_landmarks=150, seed=42):
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.seed = seed

    def fit_transform(self, adj_matrix, N):
        rng = np.random.RandomState(self.seed)

        actual_k = min(N, self.n_landmarks)
        landmarks = rng.choice(N, size=actual_k, replace=False)
        landmarks.sort()

        D_L = shortest_path(adj_matrix, method='D', directed=False, indices=landmarks)

        finite = np.isfinite(D_L)
        if not np.any(finite):
            embedding = np.zeros((N, self.n_components), dtype=float)
            meta = {"landmarks": actual_k, "eigvals_all": np.zeros(actual_k, dtype=float)}
            return embedding, meta

        max_dist = np.nanmax(D_L[finite])
        if not np.isfinite(max_dist) or max_dist <= 0:
            max_dist = 1.0
        D_L[~finite] = max_dist * 1.5

        D_L_sq = D_L ** 2
        D_LL_sq = D_L_sq[:, landmarks]  # (k, k)

        n = actual_k
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_LL_sq @ J

        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        eigvals_top = eigvals_sorted[:self.n_components]
        eigvecs_top = eigvecs_sorted[:, :self.n_components]

        L_k = eigvecs_top * np.sqrt(np.maximum(eigvals_top, 1e-9))
        L_k_pinv = np.linalg.pinv(L_k)

        row_means = np.mean(D_LL_sq, axis=1, keepdims=True)
        D_centered = D_L_sq - row_means

        embedding = -0.5 * (L_k_pinv @ D_centered)  # (d, N)
        embedding = embedding.T  # (N, d)

        meta = {"landmarks": actual_k, "eigvals_all": eigvals_sorted}
        return embedding, meta


# --- ARGUMENT PARSING ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Relational Reality Visualizer V5.4 (Stable + Spread)")

    # Selection Mode Arguments
    parser.add_argument("--version", type=str, help="Engine Version (e.g. '1')")
    parser.add_argument("--N", type=int, help="System Size N (e.g. 100)")
    parser.add_argument("--seed", type=str, help="Simulation Seed (e.g. '42')")
    parser.add_argument("--threads", type=int, default=None, help="Number of concurrent render threads")

    # Worker Mode Arguments
    parser.add_argument("--worker", action="store_true", help="Internal flag")
    parser.add_argument("--step", type=int, help="Step")
    parser.add_argument("--step_str", type=str, help="Step String")
    parser.add_argument("--node_file", type=str, help="Nodes")
    parser.add_argument("--edge_file", type=str, help="Edges")
    parser.add_argument("--out_dir", type=str, help="Output")

    # Worker controls
    parser.add_argument("--no_align", action="store_true",
                        help="Disable Procrustes alignment for this worker (preview lane).")
    parser.add_argument("--force_cache", action="store_true",
                        help="Compute/save alignment cache even if PNG exists (for forward lane).")
    parser.add_argument("--skip_png_if_exists", action="store_true",
                        help="If PNG exists, skip writing it (but still compute cache if force_cache).")

    return parser.parse_args()


# --- MENU & FILE HELPERS ---



def get_subfolders(path):
    if not os.path.exists(path):
        return []
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
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


def open_file_explorer(path):
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Could not open file explorer: {e}")

def _launch_subprocess_star(payload):
    """Pickle-safe wrapper for multiprocessing Pool."""
    task_args, extra_args = payload
    return launch_subprocess(task_args, extra_args=extra_args)

def get_rendered_steps(output_dir):
    if not os.path.exists(output_dir):
        return set()
    rendered = set()
    pattern = re.compile(rf"_i([\d_]+)_k")
    for f in os.listdir(output_dir):
        if f.endswith(".png"):
            m = pattern.search(f)
            if m:
                rendered.add(int(m.group(1).replace('_', '')))
    return rendered

def _launch_subprocess_star(args):
    """Pickle-safe helper for multiprocessing Pool."""
    task_args, extra_args = args
    return launch_subprocess(task_args, extra_args=extra_args)


def smart_sample_sort(all_steps, rendered_steps):
    # BSP-style order: first/last/mid, then recursively.
    if not all_steps:
        return []
    needed_steps = [s for s in all_steps if s not in rendered_steps]
    if not needed_steps:
        return []

    Nn = len(needed_steps)
    all_steps_sorted = sorted(needed_steps)
    bsp_indices = []
    visited = set()
    queue = deque([(0, Nn - 1)])

    if Nn > 0:
        bsp_indices.append(0); visited.add(0)
    if Nn > 1:
        bsp_indices.append(Nn - 1); visited.add(Nn - 1)

    while queue:
        start, end = queue.popleft()
        if end - start <= 1:
            continue
        mid = (start + end) // 2
        if mid not in visited:
            bsp_indices.append(mid); visited.add(mid)
        queue.append((start, mid))
        queue.append((mid, end))

    for i in range(Nn):
        if i not in visited:
            bsp_indices.append(i)

    return [all_steps_sorted[i] for i in bsp_indices]


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


# --- CORE LOGIC ---

def load_graph(node_file, edge_file):
    try:
        df_nodes = pd.read_csv(node_file)
        df_nodes.columns = [c.strip().lower() for c in df_nodes.columns]
        real_col = next((c for c in df_nodes.columns if 'real' in c or c in ('re',)), "psi_real")
        imag_col = next((c for c in df_nodes.columns if 'imag' in c or c in ('im',)), "psi_imag")

        G = nx.Graph()
        for _, row in df_nodes.iterrows():
            nid = int(row.iloc[0])
            psi = complex(row.get(real_col, 0), row.get(imag_col, 0))
            G.add_node(nid, psi=psi, rho=abs(psi) ** 2)

        df_edges = pd.read_csv(edge_file)
        if not df_edges.empty:
            df_edges.columns = [c.strip().lower() for c in df_edges.columns]
            u_col, v_col = df_edges.columns[0], df_edges.columns[1]
            G.add_edges_from([(int(r[u_col]), int(r[v_col])) for _, r in df_edges.iterrows()])

        return G
    except:
        return nx.Graph()


def stabilize_array(pos_array, anchor_idx):
    if pos_array.shape[0] == 0:
        return pos_array
    dims = pos_array.shape[1]

    multipliers = []
    for d in range(dims):
        anchor_val = pos_array[anchor_idx, d]
        multipliers.append(-1.0 if anchor_val < 0 else 1.0)
    pos_array = pos_array * np.array(multipliers)

    if dims == 2:
        x_val = pos_array[anchor_idx, 0]
        y_val = pos_array[anchor_idx, 1]
        if abs(y_val) > abs(x_val):
            pos_array[:, [0, 1]] = pos_array[:, [1, 0]]
    return pos_array


def stabilize_dict(pos_dict, anchor_id):
    if anchor_id not in pos_dict:
        return pos_dict
    anchor_pos = pos_dict[anchor_id]
    dims = len(anchor_pos)

    multipliers = []
    for d in range(dims):
        multipliers.append(-1.0 if anchor_pos[d] < 0 else 1.0)

    new_pos = {}
    for node, coords in pos_dict.items():
        new_pos[node] = np.array(coords) * np.array(multipliers)

    if dims == 2:
        anc_x = new_pos[anchor_id][0]
        anc_y = new_pos[anchor_id][1]
        if abs(anc_y) > abs(anc_x):
            for node in new_pos:
                new_pos[node] = new_pos[node][[1, 0]]
    return new_pos


def explained_var(eigs, d):
    eigs = np.array(eigs, dtype=float)
    eigs = eigs[eigs > 1e-12]
    if eigs.size == 0:
        return 0.0
    d = min(d, eigs.size)
    return float(np.sum(eigs[:d]) / np.sum(eigs))


# --- PROCRUSTES ALIGNMENT (ORTHOGONAL) + INERTIA ---

def compute_procrustes_transform(current_coords, current_nodes, prev_coords, prev_nodes):
    shared, idx_cur, idx_prev = np.intersect1d(current_nodes, prev_nodes, return_indices=True)
    if shared.size < MIN_SHARED_NODES_FOR_ALIGNMENT:
        return None

    X = current_coords[idx_cur, :]
    Y = prev_coords[idx_prev, :]

    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)

    Xc = X - X_mean
    Yc = Y - Y_mean

    C = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    mean_cur_all = current_coords.mean(axis=0, keepdims=True)
    t_prev = Y_mean
    return R, t_prev, mean_cur_all


def apply_transform(coords, R, t_prev, mean_cur_all):
    return (coords - mean_cur_all) @ R + t_prev


def blend_rotations(R_new, R_old, alpha):
    M = alpha * R_old + (1.0 - alpha) * R_new
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def cache_path(cache_dir, step_int):
    return os.path.join(cache_dir, f"step_{step_int}.npz")


def load_prev_alignment(cache_dir, prev_step_int):
    path = cache_path(cache_dir, prev_step_int)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        out = {
            "nodes": data["nodes"].astype(int),
            "spring": data["spring"].astype(float),
            "spec": data["spec"].astype(float),
            "mds2": data["mds2"].astype(float),
            "mds3": data["mds3"].astype(float),
            "R2": data["R2"].astype(float) if "R2" in data.files else None,
            "R3": data["R3"].astype(float) if "R3" in data.files else None,
            "Rspec": data["Rspec"].astype(float) if "Rspec" in data.files else None,
            "Rspring": data["Rspring"].astype(float) if "Rspring" in data.files else None,
        }
        return out
    except:
        return None


def save_alignment(cache_dir, step_int, nodes, spring, spec, mds2, mds3, R2=None, R3=None, Rspec=None, Rspring=None):
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, step_int)

    payload = dict(
        nodes=np.array(nodes, dtype=int),
        spring=np.array(spring, dtype=float),
        spec=np.array(spec, dtype=float),
        mds2=np.array(mds2, dtype=float),
        mds3=np.array(mds3, dtype=float),
    )
    if R2 is not None:
        payload["R2"] = np.array(R2, dtype=float)
    if R3 is not None:
        payload["R3"] = np.array(R3, dtype=float)
    if Rspec is not None:
        payload["Rspec"] = np.array(Rspec, dtype=float)
    if Rspring is not None:
        payload["Rspring"] = np.array(Rspring, dtype=float)

    try:
        np.savez_compressed(path, **payload)
    except:
        pass


# --- WORKER ---

def launch_subprocess(task_args, extra_args=None):
    step_int, node_f, edge_f, step_str, output_dir, v_id, N_val, s_id = task_args
    cmd = [
        sys.executable, __file__, "--worker",
        "--version", str(v_id), "--N", str(N_val), "--seed", str(s_id),
        "--step", str(step_int), "--step_str", str(step_str),
        "--node_file", node_f, "--edge_file", edge_f, "--out_dir", output_dir
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return step_int

def worker_main(args):
    try:
        G = load_graph(args.node_file, args.edge_file)
        N = G.number_of_nodes()
        if N == 0:
            return

        nodes_ordered = np.array(sorted(G.nodes()), dtype=int)

        rhos = np.array([G.nodes[int(n)].get('rho', 0.0) for n in nodes_ordered], dtype=float)
        n_edges = G.number_of_edges()
        mean_deg = (2.0 * n_edges / N) if N > 0 else 0.0
        degrees = np.array([G.degree(int(n)) for n in nodes_ordered], dtype=float)

        out_name = f"E{args.version}_N{args.N}_S{args.seed}_i{args.step_str}_k{mean_deg:.3f}.png"
        out_path = os.path.join(args.out_dir, out_name)

        cache_dir = os.path.join(args.out_dir, PROCRUSTES_CACHE_DIRNAME)
        aligned_mode = (ENABLE_PROCRUSTES_ALIGNMENT and (not args.no_align))

        png_exists = os.path.exists(out_path)
        cache_exists = os.path.exists(cache_path(cache_dir, args.step))

        # Early exit rules:
        if not aligned_mode:
            if png_exists:
                return
        else:
            if png_exists and cache_exists:
                return

        np.random.seed(SEED_FIXED + args.step)
        anchor_id = int(nodes_ordered[0])
        anchor_idx = 0

        # --- Spring ---
        if N < 3000:
            raw_spring = nx.spring_layout(G, k=0.15, iterations=50, seed=SEED_FIXED)
        else:
            try:
                init_pos = nx.spectral_layout(G, weight=None)
                raw_spring = nx.spring_layout(G, k=0.15, pos=init_pos, iterations=25, seed=SEED_FIXED)
            except:
                raw_spring = nx.spring_layout(G, k=0.15, iterations=20, seed=SEED_FIXED)

        pos_spring = stabilize_dict(raw_spring, anchor_id)
        spring_arr = np.array([pos_spring.get(int(n), (0.0, 0.0)) for n in nodes_ordered], dtype=float)

        # --- Spectral ---
        try:
            G_spec = G.copy()
            ignition_threshold = 1.5
            if mean_deg < ignition_threshold:
                ghost_edges = [(anchor_id, int(n), 1e-5) for n in nodes_ordered if int(n) != anchor_id]
                G_spec.add_weighted_edges_from(ghost_edges)

            raw_spec_dict = nx.spectral_layout(G_spec, weight='weight')
            spec_arr = np.array([raw_spec_dict.get(int(n), (0.0, 0.0)) for n in nodes_ordered], dtype=float)

            if mean_deg < ignition_threshold:
                target_idx = anchor_idx
            else:
                max_deg_node = max(dict(G.degree()).items(), key=lambda x: x[1])[0]
                target_idx = int(np.where(nodes_ordered == max_deg_node)[0][0])

            spec_arr = stabilize_array(spec_arr, target_idx)
        except:
            spec_arr = np.zeros((N, 2), dtype=float)

        # --- LMDS ---
        adj_mat = nx.to_scipy_sparse_array(G, nodelist=[int(n) for n in nodes_ordered], format='csr')
        k_landmarks = min(N, 200)

        fmds_2d = FastMDS(n_components=2, n_landmarks=k_landmarks, seed=SEED_FIXED)
        raw_mds_2d, mds2_meta = fmds_2d.fit_transform(adj_mat, N)
        mds2 = stabilize_array(raw_mds_2d, anchor_idx)
        ev2 = explained_var(mds2_meta["eigvals_all"], 2)

        fmds_3d = FastMDS(n_components=3, n_landmarks=k_landmarks, seed=SEED_FIXED)
        raw_mds_3d, mds3_meta = fmds_3d.fit_transform(adj_mat, N)
        mds3 = stabilize_array(raw_mds_3d, anchor_idx)
        ev3 = explained_var(mds3_meta["eigvals_all"], 3)

        # --- Align to previous (aligned lane only) ---
        R2_used = None
        R3_used = None
        Rspec_used = None
        Rspring_used = None

        if aligned_mode:
            prev = load_prev_alignment(cache_dir, args.step - 1)
            if prev is not None:
                prev_nodes = prev["nodes"]

                # Spring
                T = compute_procrustes_transform(spring_arr, nodes_ordered, prev["spring"], prev_nodes)
                if T is not None:
                    R_new, t_prev, mean_cur_all = T
                    if ENABLE_CAMERA_INERTIA and prev.get("Rspring") is not None:
                        R_new = blend_rotations(R_new, prev["Rspring"], INERTIA_ALPHA)
                    spring_arr = apply_transform(spring_arr, R_new, t_prev, mean_cur_all)
                    Rspring_used = R_new

                # Spectral
                T = compute_procrustes_transform(spec_arr, nodes_ordered, prev["spec"], prev_nodes)
                if T is not None:
                    R_new, t_prev, mean_cur_all = T
                    if ENABLE_CAMERA_INERTIA and prev.get("Rspec") is not None:
                        R_new = blend_rotations(R_new, prev["Rspec"], INERTIA_ALPHA)
                    spec_arr = apply_transform(spec_arr, R_new, t_prev, mean_cur_all)
                    Rspec_used = R_new

                # LMDS 2D
                T = compute_procrustes_transform(mds2, nodes_ordered, prev["mds2"], prev_nodes)
                if T is not None:
                    R_new, t_prev, mean_cur_all = T
                    if ENABLE_CAMERA_INERTIA and prev.get("R2") is not None:
                        R_new = blend_rotations(R_new, prev["R2"], INERTIA_ALPHA)
                    mds2 = apply_transform(mds2, R_new, t_prev, mean_cur_all)
                    R2_used = R_new

                # LMDS 3D
                T = compute_procrustes_transform(mds3, nodes_ordered, prev["mds3"], prev_nodes)
                if T is not None:
                    R_new, t_prev, mean_cur_all = T
                    if ENABLE_CAMERA_INERTIA and prev.get("R3") is not None:
                        R_new = blend_rotations(R_new, prev["R3"], INERTIA_ALPHA)
                    mds3 = apply_transform(mds3, R_new, t_prev, mean_cur_all)
                    R3_used = R_new

            # Save cache
            save_alignment(cache_dir, args.step, nodes_ordered, spring_arr, spec_arr, mds2, mds3,
                           R2=R2_used, R3=R3_used, Rspec=Rspec_used, Rspring=Rspring_used)

        if png_exists and aligned_mode and args.skip_png_if_exists:
            return

        # --- RENDER PNG ---
        v_min = float(np.min(rhos)) if rhos.size else 0.0
        v_max = float(np.max(rhos)) if rhos.size else 1.0
        if v_max <= v_min:
            v_max = v_min + 1e-12

        fig = plt.figure(figsize=(20, 17))
        # UPDATED: Pushed top down from 0.72 to 0.70 to give header more room
        plt.subplots_adjust(top=0.70, hspace=0.2, wspace=0.15)

        scatter_args = dict(c=rhos, cmap=GLOBAL_CMAP, vmin=v_min, vmax=v_max, s=20, alpha=0.9)

        plots = [
            (spring_arr, "1. Physical Topology (Spring) [Stabilized+Aligned]" if aligned_mode else
                        "1. Physical Topology (Spring) [Stabilized]", None),
            (spec_arr, "2. Quantum Resonance (Spectral) [Stabilized+Aligned]" if aligned_mode else
                      "2. Quantum Resonance (Spectral) [Stabilized]", None),
            (mds2, f"3. Emergent Manifold (Fast MDS 2D) [{'Aligned' if aligned_mode else 'Preview'}] | EV2={ev2:.1%}", None),
            (mds3, f"4. The Hologram (Fast MDS 3D) [{'Aligned' if aligned_mode else 'Preview'}] | EV3={ev3:.1%}", "3d"),
        ]

        sc_map = None

        for i, (arr, title, proj) in enumerate(plots):
            ax = fig.add_subplot(2, 2, i + 1, projection=proj)

            if proj == "3d":
                arr3 = np.array(arr, dtype=float)
                if arr3.shape[1] < 3:
                    pad = np.zeros((arr3.shape[0], 3), dtype=float)
                    pad[:, :arr3.shape[1]] = arr3
                    arr3 = pad

                ax.scatter(arr3[:, 0], arr3[:, 1], arr3[:, 2], **scatter_args)
                ax.set_box_aspect([1, 1, 1])

                max_range = (np.max(arr3, axis=0) - np.min(arr3, axis=0)).max() / 2.0
                mid = (np.max(arr3, axis=0) + np.min(arr3, axis=0)) * 0.5
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
                ax.set_title(title, fontsize=14)
                ax.axis("off")

            elif i == 0:
                edge_alpha = 0.05 if N < 5000 else 0.01
                pos_dict_for_edges = {int(n): arr[idx] for idx, n in enumerate(nodes_ordered)}
                if N < 15000:
                    nx.draw_networkx_edges(G, pos_dict_for_edges, ax=ax, alpha=edge_alpha, edge_color="gray")

                sc_map = ax.scatter(arr[:, 0], arr[:, 1], **scatter_args)
                ax.set_title(title, fontsize=14)
                ax.axis("off")
            else:
                ax.scatter(arr[:, 0], arr[:, 1], **scatter_args)
                ax.set_title(title, fontsize=14)
                ax.axis("off")

        fig.suptitle("Relational Reality", fontsize=22, y=0.98, fontweight="bold")
        step_fmt = f"{args.step:,}".replace(",", "_")
        sub_title = f"E{args.version} | N={args.N} | S{args.seed} | Step {step_fmt} | Avg <k>={mean_deg:.2f}"
        fig.text(0.5, 0.95, sub_title, ha="center", fontsize=15)

        # --- UPDATED: Degree Histogram Layout ---
        # Moved to the LEFT side: [left, bottom, width, height]
        ax_hist = fig.add_axes([0.08, 0.76, 0.40, 0.08])
        if degrees.size:
            deg_min, deg_max = degrees.min(), degrees.max()
            bins = np.arange(deg_min, deg_max + 2) - 0.5
            counts, edges = np.histogram(degrees, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            width = 0.8 * (edges[1] - edges[0])

            # Re-map colors for the histogram bars
            norm = plt.Normalize(vmin=v_min, vmax=v_max)
            cmap = plt.get_cmap(GLOBAL_CMAP)
            bin_colors = []
            for b_center in centers:
                # Average rho of nodes with this degree approx
                relevant = rhos[np.abs(degrees - b_center) < 0.5]
                if relevant.size > 0:
                    val = np.mean(relevant)
                else:
                    val = v_min
                bin_colors.append(cmap(norm(val)))

            bars = ax_hist.bar(centers, counts, width=width, color=bin_colors)

            # Adjust Y-limit so text labels don't hit the top
            if len(counts) > 0:
                ax_hist.set_ylim(0, max(counts) * 1.15)

            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    ax_hist.text(rect.get_x() + rect.get_width() / 2., height,
                                f"{int(height)}", ha="center", va="bottom", fontsize=7, color="black")

            ax_hist.axvline(mean_deg, color="red", linestyle="--", alpha=0.8)
            ax_hist.set_title(f"Degree Distribution (Avg <k>={mean_deg:.2f})", fontsize=10, pad=3)
            ax_hist.tick_params(labelsize=8)
            ax_hist.spines['top'].set_visible(False)
            ax_hist.spines['right'].set_visible(False)

        # --- UPDATED: Eigen Spectrum Layout ---
        # Moved to the RIGHT side: [left, bottom, width, height]
        try:
            eigs = np.array(mds3_meta["eigvals_all"], dtype=float)
            eigs = eigs[eigs > 1e-12][:30]
            if eigs.size > 0:
                ax_spec = fig.add_axes([0.56, 0.76, 0.35, 0.08])
                xs = np.arange(1, len(eigs) + 1)

                ax_spec.plot(xs, eigs, marker="o", linewidth=1, markersize=3, color='#444444')
                ax_spec.set_yscale("log")

                if len(eigs) >= 2: ax_spec.axvline(2, linestyle="--", color='gray', alpha=0.4)
                if len(eigs) >= 3: ax_spec.axvline(3, linestyle="--", color='gray', alpha=0.4)

                ax_spec.set_title("LMDS Eigen Spectrum", fontsize=10, pad=3)
                ax_spec.tick_params(axis="both", labelsize=7)
                ax_spec.spines['top'].set_visible(False)
                ax_spec.spines['right'].set_visible(False)

                # Info box
                ax_spec.text(
                    0.98, 0.90,
                    f"EV2={ev2:.1%}\nEV3={ev3:.1%}",
                    transform=ax_spec.transAxes,
                    ha="right", va="top",
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.2')
                )
        except:
            pass

        # Colorbar
        if sc_map is None:
            import matplotlib as mpl
            sc_map = mpl.cm.ScalarMappable(cmap=GLOBAL_CMAP, norm=mpl.colors.Normalize(vmin=v_min, vmax=v_max))
            sc_map.set_array([])

        cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        cb = fig.colorbar(sc_map, cax=cbar_ax, orientation="horizontal")
        cb.set_label(r"$\rho$ (Density)", fontsize=14)

        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"Error in Worker Frame {args.step}: {e}")
        import traceback
        traceback.print_exc()


# --- MAIN ---

def main():
    args = parse_arguments()
    warnings.filterwarnings("ignore")

    if args.worker:
        worker_main(args)
        return

    print("==================================================")
    print("  RELATIONAL REALITY VISUALIZER V5.4 (SPREAD+STB) ")
    print("==================================================")

    # 1. Version
    if args.version:
        ver_str = f"E{args.version}"
        if not os.path.exists(os.path.join(RUNS_DIR, ver_str)):
            return
    else:
        versions = get_subfolders(RUNS_DIR)
        versions.sort()
        ver_str = select_option(versions, "Select Engine Version")
        if not ver_str:
            return

    # 2. N
    n_path = os.path.join(RUNS_DIR, ver_str)
    if args.N:
        n_str = f"N{args.N}"
        if not os.path.exists(os.path.join(n_path, n_str)):
            return
    else:
        n_counts = get_subfolders(n_path)
        n_counts.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        n_str = select_option(n_counts, "Select System Size (N)")
        if not n_str:
            return
    n_val = int(n_str[1:])

    # 3. Seed
    seed_path = os.path.join(n_path, n_str)
    if args.seed:
        seed_str = f"S{args.seed}"
        if not os.path.exists(os.path.join(seed_path, seed_str)):
            return
    else:
        seeds = get_subfolders(seed_path)
        seeds.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        seed_str = select_option(seeds, "Select Simulation Seed")
        if not seed_str:
            return

    # 4. Threads (total budget)
    max_cores = cpu_count()
    if args.threads:
        n_cores = args.threads
    else:
        print(f"\n--- Thread Configuration (Max: {max_cores}) ---")
        n_cores = min(max_cores, 8)
        try:
            inp = input(f"Enter threads (Default {n_cores}): ").strip()
            if inp:
                n_cores = int(inp)
        except:
            pass

    full_seed_path = os.path.join(seed_path, seed_str)
    data_dir = os.path.join(full_seed_path, "data")
    output_dir = os.path.join(full_seed_path, "renders")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, PROCRUSTES_CACHE_DIRNAME), exist_ok=True)

    v_id = int(ver_str[1:])
    s_id = int(seed_str[1:])

    steps_sorted, data_map = parse_steps(data_dir, v_id, n_val, s_id)
    rendered_steps = get_rendered_steps(output_dir)
    missing_steps = [s for s in steps_sorted if s not in rendered_steps]

    if not missing_steps:
        print("\n>> All frames rendered.")
        return

    open_file_explorer(output_dir)

    # --- Build tasks ---
    forward_steps = sorted(missing_steps)  # aligned lane wants chronological
    spread_steps = smart_sample_sort(steps_sorted, rendered_steps)  # BSP order across timeline

    if SPREAD_MAX_FRAMES is not None:
        spread_steps = spread_steps[:SPREAD_MAX_FRAMES]

    forward_tasks = [(s, data_map[s][0], data_map[s][1], data_map[s][2], output_dir, v_id, n_val, s_id) for s in forward_steps]
    spread_tasks = [(s, data_map[s][0], data_map[s][1], data_map[s][2], output_dir, v_id, n_val, s_id) for s in spread_steps]

    print(f"\n>> Missing frames total: {len(missing_steps)}")
    print(f">> Forward aligned lane: {len(forward_tasks)} frames (chronological)")
    print(f">> Spread preview lane:  {len(spread_tasks)} frames (BSP spread)")

    if ENABLE_DUAL_LANE_RENDER and n_cores >= 2:
        # Reserve 1 "lane" for forward; use remaining for spread parallelism.
        spread_cores = max(1, n_cores - 1)

        print(f">> Dual-lane enabled. Total cores={n_cores} -> spread pool cores={spread_cores}")

        def forward_lane():
            # Forward lane: aligned; also force cache chain even if PNG exists.
            extra = ["--force_cache"]
            if ALIGNED_FORWARD_SKIP_PNG_IF_EXISTS:
                extra.append("--skip_png_if_exists")

            for t in tqdm(forward_tasks, total=len(forward_tasks), desc="Forward(aligned)", position=0, leave=True):
                launch_subprocess(t, extra_args=extra)

        def spread_lane():
            # Spread lane: no alignment, purely to get coverage ASAP.
            extra = ["--no_align"]
            if spread_cores <= 1:
                for t in tqdm(spread_tasks, total=len(spread_tasks), desc="Spread(preview)", position=1, leave=True):
                    launch_subprocess(t, extra_args=extra)
            else:
                with Pool(processes=spread_cores) as pool:
                    iterable = [(task, extra) for task in spread_tasks]
                    list(tqdm(
                        pool.imap(_launch_subprocess_star, iterable),
                        total=len(spread_tasks),
                        desc="Spread(preview)",
                        position=1,
                        leave=True
                    ))



        th_fwd = threading.Thread(target=forward_lane, daemon=False)
        th_spd = threading.Thread(target=spread_lane, daemon=False)

        th_fwd.start()
        th_spd.start()

        th_fwd.join()
        th_spd.join()

    else:
        # Fallback: if dual-lane is off or cores < 2
        if ENABLE_PROCRUSTES_ALIGNMENT:
            print("\n>> Dual-lane disabled or not enough cores; running aligned sequential.")
            extra = ["--force_cache"]
            if ALIGNED_FORWARD_SKIP_PNG_IF_EXISTS:
                extra.append("--skip_png_if_exists")
            for t in tqdm(forward_tasks, total=len(forward_tasks), desc="Aligned", leave=True):
                launch_subprocess(t, extra_args=extra)
        else:
            print("\n>> Alignment disabled; rendering spread order with pool.")
            with Pool(processes=n_cores) as pool:
                list(tqdm(pool.imap(launch_subprocess, spread_tasks), total=len(spread_tasks)))

    print("\n[DONE] Visualization complete.")


if __name__ == "__main__":
    main()
