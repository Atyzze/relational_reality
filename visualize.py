import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys, re, glob, warnings, argparse, platform, subprocess, time, shutil, math
from collections import deque
from multiprocessing import Pool, cpu_count
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.sparse.csgraph import shortest_path
import scipy.sparse.linalg

# --- CONFIGURATION ---
RUNS_DIR = "data"
SEED_FIXED = 42
GLOBAL_CMAP = "viridis"
DEFAULT_K_THRESHOLD = 0.01

# ==========================================
#        PHYSICS & MATH HELPERS (FROM v0.py)
# ==========================================

def compute_node_frustration(G, nodes_ordered):
    """
    Calculates the 'geometric stress' (curl) for each node.
    """
    node_stress = {n: 0.0 for n in nodes_ordered}
    node_counts = {n: 0 for n in nodes_ordered}
    edge_thetas = {}
    for u, v, d in G.edges(data=True):
        th = d.get('theta', 0.0)
        edge_thetas[(u, v)] = th
        edge_thetas[(v, u)] = th

    processed_triangles = set()
    for u in G.nodes():
        nbrs = list(G.neighbors(u))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v, w = nbrs[i], nbrs[j]
                if G.has_edge(v, w):
                    tri_key = tuple(sorted((u, v, w)))
                    if tri_key in processed_triangles: continue
                    processed_triangles.add(tri_key)

                    t_uv = edge_thetas.get((u, v), 0.0)
                    t_vw = edge_thetas.get((v, w), 0.0)
                    t_wu = edge_thetas.get((w, u), 0.0)

                    best_stress = 1.0
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                sum_th = s1*t_uv + s2*t_vw + s3*t_wu
                                stress = 1.0 - math.cos(2*sum_th)
                                if stress < best_stress: best_stress = stress

                    node_stress[u] += best_stress; node_counts[u] += 1
                    node_stress[v] += best_stress; node_counts[v] += 1
                    node_stress[w] += best_stress; node_counts[w] += 1

    result = []
    for n in nodes_ordered:
        c = node_counts[n]
        if c > 0: result.append(node_stress[n] / c)
        else: result.append(0.0)
    return np.array(result)

def compute_integrated_phase(G, nodes_ordered):
    N = len(nodes_ordered)
    node_to_idx = {n: i for i, n in enumerate(nodes_ordered)}
    phases = np.zeros(N, dtype=float)
    visited = np.zeros(N, dtype=bool)

    for start_node in nodes_ordered:
        start_idx = node_to_idx[start_node]
        if visited[start_idx]: continue
        queue = deque([start_node])
        visited[start_idx] = True
        phases[start_idx] = 0.0
        while queue:
            u = queue.popleft()
            u_idx = node_to_idx[u]
            u_phi = phases[u_idx]
            for v in G.neighbors(u):
                v_idx = node_to_idx[v]
                if not visited[v_idx]:
                    edge_theta = G[u][v].get('theta', 0.0)
                    phases[v_idx] = u_phi + edge_theta
                    visited[v_idx] = True
                    queue.append(v)
    return (phases + math.pi) % (2 * math.pi) - math.pi

# ==========================================
#          LAYOUT & VISUALIZATION
# ==========================================

class FastMDS:
    """ Landmark MDS approximation for faster layout calculation. """
    def __init__(self, n_components=2, n_landmarks=100, seed=42):
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
            return np.zeros((N, self.n_components), dtype=float), {"eigvals_all": []}

        max_dist = np.nanmax(D_L[finite])
        if max_dist == 0:
             return np.zeros((N, self.n_components), dtype=float), {"eigvals_all": np.zeros(actual_k)}
        if not np.isfinite(max_dist) or max_dist <= 0: max_dist = 1.0
        D_L[~finite] = max_dist * 1.5

        D_L_sq = D_L ** 2
        D_LL_sq = D_L_sq[:, landmarks]
        n = actual_k
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_LL_sq @ J

        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]

        eigvals_top = eigvals_sorted[:self.n_components]
        eigvecs_top = eigvecs[:, idx][:, :self.n_components]

        L_k = eigvecs_top * np.sqrt(np.maximum(eigvals_top, 1e-9))
        embedding = -0.5 * (np.linalg.pinv(L_k) @ (D_L_sq - np.mean(D_LL_sq, axis=1, keepdims=True)))
        return embedding.T, {"eigvals_all": eigvals_sorted}

# --- LAYOUT HELPERS ---
def stabilize_array(pos_array, ignored_arg=None):
    """
    Flips axes deterministically based on the 'Pole' (Max Absolute Value).
    This is much more stable than anchoring to Node 0.
    """
    if pos_array.shape[0] == 0: return pos_array

    multipliers = []
    for d in range(pos_array.shape[1]):
        # Find the node with the strongest signal (furthest from 0) in this dimension
        # This is the "Tip" of the shape.
        col = pos_array[:, d]
        idx_max = np.argmax(np.abs(col))
        val_max = col[idx_max]

        # If that tip is negative, flip the whole world to make it positive.
        multipliers.append(-1.0 if val_max < 0 else 1.0)

    return pos_array * np.array(multipliers)

def stabilize_dict(pos_dict, ignored_arg=None):
    """Same logic for dictionary-based layouts."""
    if not pos_dict: return pos_dict

    # Convert to array for fast calc
    nodes = sorted(pos_dict.keys())
    arr = np.array([pos_dict[n] for n in nodes])

    # Calculate multipliers using the array logic
    multipliers = []
    for d in range(arr.shape[1]):
        col = arr[:, d]
        idx_max = np.argmax(np.abs(col))
        val_max = col[idx_max]
        multipliers.append(-1.0 if val_max < 0 else 1.0)

    mult_arr = np.array(multipliers)

    # Apply back to dict
    return {n: pos * mult_arr for n, pos in pos_dict.items()}

def get_dual_spectral_init(G, nodes_ordered):
    """Calculates both dominant spectral modes (v1-v2 and v1-v3)."""

    def normalize(v):
        mn, mx = v.min(), v.max()
        return v if mx - mn < 1e-9 else 2 * ((v - mn) / (mx - mn)) - 1

    try:
        N = len(nodes_ordered)
        L = nx.laplacian_matrix(G, nodelist=nodes_ordered).toarray().astype(float)

        vals, vecs = scipy.linalg.eigh(L, subset_by_index=[1, 5]) //this is more strict/deterministic
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        v1, v2, v3 = normalize(vecs[:, 1]), normalize(vecs[:, 2]), normalize(vecs[:, 3])
        pos_A = np.column_stack((v1, v2))

        vals, vecs = scipy.sparse.linalg.eigsh(L, k=min(N-1, 5), which='SM', tol=1e-3) #here we can control the range with tol being lowered towards identical result below
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        v1, v2, v3 = normalize(vecs[:, 1]), normalize(vecs[:, 2]), normalize(vecs[:, 3])

        pos_B = np.column_stack((v1, v2))

        return {n: pos_A[i] for i, n in enumerate(nodes_ordered)}, {n: pos_B[i] for i, n in enumerate(nodes_ordered)}
    except:
        return None, None

def run_umap_layout_original(adj_mat, mean_deg, seed=42):
    """
    Runs UMAP directly on the adjacency matrix.
    Handles scipy csr_array vs csr_matrix compatibility for Numba.
    """
    from scipy.sparse import csr_matrix

    # 2. Convert 'csr_array' (NetworkX) to 'csr_matrix' (UMAP/Numba requirement)
    if not isinstance(adj_mat, csr_matrix):
        adj_mat = csr_matrix(adj_mat)

    # 3. Dynamic neighbors
    n_neighbors = int(max(2, round(mean_deg)))
    import umap

    # 4. Run UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        init='spectral',
        random_state=seed,
        n_jobs=1,
        force_approximation_algorithm=True
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs*")
        warnings.filterwarnings("ignore", message=".*Spectral initialisation failed.*")
        embedding_2d = reducer.fit_transform(adj_mat)

    return embedding_2d

def load_graph(node_file, edge_file):
    # Updated to capture Theta for new color modes
    if not os.path.exists(node_file):
        return nx.Graph()

    df_nodes = pd.read_csv(node_file)
    df_nodes.columns = [c.strip().lower() for c in df_nodes.columns]

    real_col = next((c for c in df_nodes.columns if 'real' in c), "psi_real")
    imag_col = next((c for c in df_nodes.columns if 'imag' in c), "psi_imag")
    has_theta = 'theta' in df_nodes.columns

    G = nx.Graph()
    for _, row in df_nodes.iterrows():
        c_val = complex(row.get(real_col, 0), row.get(imag_col, 0))
        rho_val = abs(c_val) ** 2
        theta_val = row['theta'] if has_theta else np.angle(c_val)
        G.add_node(int(row.iloc[0]), rho=rho_val, theta=theta_val)

    if os.path.exists(edge_file):
        df_edges = pd.read_csv(edge_file)
        if not df_edges.empty:
            df_edges.columns = [c.strip().lower() for c in df_edges.columns]
            # Capture edge theta if available
            G.add_edges_from([(int(r.iloc[0]), int(r.iloc[1]), {'theta': r.get('theta', 0.0)}) for _, r in df_edges.iterrows()])
    return G

def worker_main(args):
    try:
        # 1. Load Data
        G = load_graph(args.node_file, args.edge_file)
        N = G.number_of_nodes()
        if N == 0: return

        nodes_ordered = np.array(sorted(G.nodes()), dtype=int)
        rhos = np.array([G.nodes[n]['rho'] for n in nodes_ordered], dtype=float)
        degrees = np.array([G.degree(n) for n in nodes_ordered], dtype=float)
        mean_deg = np.mean(degrees) if N > 0 else 0.0

        out_name = f"{args.version_tag}_N{args.N}_S{args.seed}_i{args.step_str}_k{mean_deg:.4f}.png"
        out_path = os.path.join(args.out_dir, out_name)
        if os.path.exists(out_path): return

        # 2. Calculate Layouts
        np.random.seed(SEED_FIXED)
        anchor_id = int(nodes_ordered[0])

        # A. Spectral Dual Modes
        spec_A, spec_B = get_dual_spectral_init(G, nodes_ordered)

        spec_raw_array = np.zeros((N, 2))
        if spec_A is not None:
            init_main = spec_A
            init_alt = spec_B
            spec_raw_array = np.array([spec_A[n] for n in nodes_ordered])

            # INSTEAD OF stabilize_array, rotate the whole embedding so Node 0 is at angle 0
            anchor_angle = np.arctan2(spec_raw_array[0, 1], spec_raw_array[0, 0])

            # Apply a 2D rotation matrix by -anchor_angle
            cos_th = np.cos(-anchor_angle)
            sin_th = np.sin(-anchor_angle)

            x_rot = spec_raw_array[:, 0] * cos_th - spec_raw_array[:, 1] * sin_th
            y_rot = spec_raw_array[:, 0] * sin_th + spec_raw_array[:, 1] * cos_th

            spec_raw_array[:, 0] = x_rot
            spec_raw_array[:, 1] = y_rot

            # We still need to fix independent axis mirroring (chirality).
            # If a second consistent node flips its Y-axis, we flip the whole Y-axis.
            if len(nodes_ordered) > 1 and spec_raw_array[1, 1] < 0:
                spec_raw_array[:, 1] *= -1.0
        else:
            init_main = nx.random_layout(G, seed=SEED_FIXED)
            init_alt = nx.random_layout(G, seed=SEED_FIXED)

        # B. Spring Layouts
        def get_spring(init_pos, iters=25):
            try:
                raw = nx.spring_layout(G, k=0.15, pos=init_pos, iterations=iters, seed=SEED_FIXED)
                s = stabilize_dict(raw, anchor_id)
                return np.array([s.get(n, (0,0)) for n in nodes_ordered])
            except: return np.zeros((N, 2))

        spring_rand = get_spring(None)
        spring_main = get_spring(init_main)
        spring_classic = get_spring(init_alt)

        # C. MDS & UMAP
        adj_mat = nx.to_scipy_sparse_array(G, nodelist=nodes_ordered, format='csr')

        # UMAP
        if mean_deg < 0.01:
            umap_pos = np.zeros((N, 2))
            n_neighbors = 0
        else:
            umap_pos = run_umap_layout_original(adj_mat, mean_deg, seed=(SEED_FIXED))
            umap_pos = stabilize_array(umap_pos, 0)
            n_neighbors = int(max(2, round(mean_deg)))

        # MDS 2D
        mds2, mds2_meta = FastMDS(2).fit_transform(adj_mat, N)
        mds2 = stabilize_array(mds2, 0)
        evals_all = mds2_meta.get("eigvals_all", [])
        valid_evals = evals_all[evals_all > 1e-9]
        total_variance = np.sum(valid_evals) if len(valid_evals) > 0 else 1.0
        ev2_sum = np.sum(evals_all[:2]) if len(evals_all) >= 2 else np.sum(evals_all)
        ev2_pct = (ev2_sum / total_variance) * 100
        title_2d = f"4. MDS 2D (EV={ev2_pct:.1f}%)"

        # MDS 3D
        mds3, mds3_meta = FastMDS(3).fit_transform(adj_mat, N)
        mds3 = stabilize_array(mds3, 0)
        evals_3d = mds3_meta.get("eigvals_all", [])
        ev3_sum = np.sum(evals_3d[:3]) if len(evals_3d) >= 3 else np.sum(evals_3d)
        ev3_pct = (ev3_sum / (np.sum(evals_3d[evals_3d > 1e-9]) or 1.0)) * 100
        title_3d = f"6. MDS 3D (EV={ev3_pct:.1f}%)"

        # --- COLOR FIELDS (Integrated from v0.py) ---
        mode = getattr(args, "color_mode", "auto")
        local_cmap = GLOBAL_CMAP

        if mode == "phase_integration":
            cvals = compute_integrated_phase(G, nodes_ordered)
            ctitle = "Projected Phase (∫θ)"; local_cmap = "twilight"
        elif mode == "frustration":
            cvals = compute_node_frustration(G, nodes_ordered)
            ctitle = "Geom. Frustration"; local_cmap = "inferno"
        elif mode == "spectral_angle":
            if spec_A is not None:
                # Use v1 and v2
                cvals = np.arctan2(spec_raw_array[:, 1], spec_raw_array[:, 0])
                ctitle = "Spectral Angle (ψ)"; local_cmap = "hsv"
                cmin_override, cmax_override = -math.pi, math.pi
            else:
                cvals = degrees; ctitle = "Deg (No Spectral)"
        elif mode == "degree":
            cvals = degrees; ctitle = "Degree"
        elif mode == "rho":
            cvals = rhos; ctitle = "ρ (Amp)"
        else:
             # Auto-fallback logic
             def _span(x): return float(np.nanmax(x) - np.nanmin(x)) if x.size > 0 else 0.0
             if _span(rhos) > 1e-12:
                 cvals = rhos; ctitle = "ρ (Amp)"
             else:
                 cvals = degrees; ctitle = "Degree"

        cmin = float(np.nanmin(cvals)) if cvals.size else 0.0
        cmax = float(np.nanmax(cvals)) if cvals.size else 1.0
        if abs(cmax - cmin) < 1e-12: cmin, cmax = cmin - 1.0, cmin + 1.0
        cmap_args = dict(c=cvals, cmap=local_cmap, vmin=cmin, vmax=cmax, s=20, alpha=0.9)


        # 3. Render
        fig = plt.figure(figsize=(24, 24))

        # --- BACKGROUND TEXTURE ---
        ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
        ax_bg.axis('off')
        ax_bg.spy(adj_mat, markersize=0.5, color='#444444', alpha=0.15)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.25, wspace=0.25)

        grid = [
            (degrees, "1. Degree Distribution", 'hist'),
            (spring_rand, "2. Spring (Random Init)", 'scatter'),
            (mds3_meta, "3. LMDS Spectrum", 'spectrum'),
            (mds2, title_2d, 'scatter'),
            (umap_pos, f"5. UMAP (neighbors={n_neighbors})", 'scatter'),
            (mds3, title_3d, '3d'),
            (spring_main, "7. Spring (Spectral v1-v2)", 'scatter'),
            (spec_raw_array, "8. Spectral (Raw)", 'scatter'),
            (spring_classic, "9. Spring (Spectral v1-v3)", 'scatter')
        ]

        for i, (data, title, type_) in enumerate(grid):
            ax = fig.add_subplot(3, 3, i+1, projection='3d' if type_=='3d' else None)
            ax.set_facecolor((0, 0, 0, 0))
            ax.set_title(title, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if type_ == 'scatter':
                pos = {n: data[k] for k, n in enumerate(nodes_ordered)}
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.03, edge_color="gray")
                ax.scatter(data[:,0], data[:,1], **cmap_args)
                ax.axis('off')

            elif type_ == '3d':
                d3 = np.hstack([data, np.zeros((data.shape[0], 1))]) if data.shape[1] < 3 else data
                ax.scatter(d3[:,0], d3[:,1], d3[:,2], **cmap_args)
                ax.set_box_aspect([1,1,1])
                ax.axis('off')
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.set_facecolor((0, 0, 0, 0))

            elif type_ == 'hist':
                if data.size > 0:
                    bins = np.arange(data.min(), data.max() + 2) - 0.5
                    counts, edges = np.histogram(data, bins=bins)
                    bin_centers = edges[:-1] + 0.5
                    cmap = plt.get_cmap(local_cmap) # Use local_cmap here
                    norm = plt.Normalize(vmin=cmin, vmax=cmax)

                    if mode == "degree":
                         bar_colors = [cmap(norm(k)) for k in bin_centers]
                    else:
                         bar_colors = 'white'

                    ax.bar(edges[:-1], counts, width=0.8, color=bar_colors, edgecolor='black', linewidth=0.5, align='edge')
                    if len(counts) > 0: ax.set_ylim(top=max(counts) * 1.15)


                    for x_pos, y_pos, k_val in zip(edges[:-1], counts, bin_centers):
                        if y_pos > 0:
                            ax.text(x_pos + 0.4, y_pos, str(int(y_pos)), ha='center', va='bottom', fontsize=10, color='black')

                    ax.axvline(mean_deg, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    ax.set_xlim(left=-0.9)
                    if args.global_max_k > 0: ax.set_xlim(right=args.global_max_k + 0.5)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            elif type_ == 'spectrum':
                ev = np.array([]) if args.step == 0 else np.array(data.get("eigvals_all", []))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if len(ev) >= 2:
                    mask = (ev > -np.inf); mask[:3] = True
                    ev = ev[mask][:30]
                    ev = np.maximum(ev, 1e-6)
                else:
                    ev = ev[ev > 0][:30]

                if ev.size > 0:
                    xs = np.arange(1, len(ev) + 1)
                    ax.plot(xs, ev, 'o-', color='#444', markersize=4, linewidth=1)
                    ax.set_yscale("log")
                    from matplotlib.ticker import ScalarFormatter
                    ymin, ymax = np.min(ev), np.max(ev)
                    if ymax - ymin < 1.0:
                        mid = (ymax + ymin) / 2.0
                        ax.set_ylim(max(0, mid - 0.6), mid + 0.6)
                    formatter = ScalarFormatter()
                    formatter.set_scientific(False)
                    ax.yaxis.set_major_formatter(formatter)
                    if (ymax < 100 and ymin > 0.1) or ((ymax / max(ymin, 0) < 10) and ymax > 0.01):
                        ax.yaxis.set_minor_formatter(formatter)
                    else:
                        ax.yaxis.set_minor_formatter(plt.NullFormatter())

                    rounded_eigs = np.round(ev, 2)
                    unique_vals, counts_ = np.unique(rounded_eigs, return_counts=True)
                    sorted_indices = np.argsort(unique_vals)[::-1]
                    unique_vals = unique_vals[sorted_indices]
                    counts_ = counts_[sorted_indices]
                    summary_text = "modes:\n"
                    for val, count in zip(unique_vals[:30], counts_[:30]):
                        if count > 1: summary_text += f"(x{count}) {val:.2f}\n"
                        else: summary_text += f"{val:.2f}\n"
                    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                    ax.text(0.95, 0.95, summary_text, transform=ax.transAxes, fontsize=11, va='top', ha='right', bbox=props)

        stats = f"iter: {args.step_str} | k: {mean_deg:.4f} | color: {ctitle}"
        fig.suptitle(f"Relational Reality | N{N} | {args.version_tag} | S{args.seed}\n{stats}",
                     fontsize=18, y=0.98, fontweight="bold")

        pct = (args.frame_idx + 1) / max(1, args.total_frames)
        total_slots = 60
        filled = int(pct * total_slots)
        bar = f"|{'█' * filled}{'·' * (total_slots - filled)}|"
        fig.text(0.5, 0.95, bar, ha="center", fontsize=10, color='#555', family='monospace')

        plt.savefig(out_path, dpi=200)
        plt.close(fig)

    except Exception as e:
        print(f"Frame {args.step} Error: {e}")
        import traceback
        traceback.print_exc()


# --- MAIN CONTROLLER ---
def _launch_subprocess_star(args):
    return launch_subprocess(*args)

def launch_subprocess(task_args):
    step_int, node_f, edge_f, step_str, out_dir, ver, N, s_id, gk, f_idx, tot, color_mode = task_args
    print(f"[Start] Processing N{str(N)}_{ver}_S{str(s_id)} iteration {step_str} (Frame {f_idx+1}/{tot})...", flush=True)
    start_time = time.time()

    cmd = [sys.executable, __file__, "--worker",
           "--version_tag", ver, "--N", str(N), "--seed", str(s_id),
           "--step", str(step_int), "--step_str", step_str,
           "--node_file", node_f, "--edge_file", edge_f, "--out_dir", out_dir,
           "--global_max_k", str(gk), "--frame_idx", str(f_idx), "--total_frames", str(tot),
           "--color_mode", str(color_mode)]

    subprocess.run(cmd)

    duration = time.time() - start_time
    print(f"[Done]  Finished Step {step_str} in {duration:.2f}s", flush=True)

    try:
        log_path = os.path.join(out_dir, "render.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Frame: {f_idx+1:04d}/{tot} | Iter: {step_str} | Duration: {duration:.2f}s\n"
        with open(log_path, "a") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to log: {e}")

    return step_int

def get_binary_spread_order(steps):
    if not steps: return []
    tasks = list(steps)
    N = len(tasks)
    if N == 0: return []
    indices = []
    seen = set()

    if 0 not in seen: indices.append(0); seen.add(0)
    if (N - 1) not in seen and (N - 1) >= 0: indices.append(N - 1); seen.add(N - 1)

    queue = deque([(0, N - 1)])
    while queue:
        low, high = queue.popleft()
        if low + 1 >= high: continue
        mid = (low + high) // 2
        if mid not in seen: indices.append(mid); seen.add(mid)
        queue.append((low, mid)); queue.append((mid, high))
    return [tasks[i] for i in indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--version_tag", type=str); parser.add_argument("--N", type=int); parser.add_argument("--seed", type=str)
    parser.add_argument("--step", type=int); parser.add_argument("--step_str", type=str)
    parser.add_argument("--node_file", type=str); parser.add_argument("--edge_file", type=str); parser.add_argument("--out_dir", type=str)
    parser.add_argument("--global_max_k", type=float, default=0); parser.add_argument("--frame_idx", type=int, default=0); parser.add_argument("--total_frames", type=int, default=1)
    parser.add_argument("--zoom", type=float, default=None); parser.add_argument("--threads", type=int)
    parser.add_argument("--color_mode", type=str, default="auto", choices=["auto","rho","degree","phase_integration","frustration","spectral_angle"])
    args = parser.parse_args()

    if args.worker:
        worker_main(args)
        return

    print("=== RELATIONAL REALITY VISUALIZER (v2: Enhanced Modes) ===")

    def get_input(path, name):
        if not os.path.exists(path): return None
        opts = [d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
                and not d.startswith("__") and not d.startswith(".")
                and (re.match(r"^E\d+D\d+", d) if name == "Version" else True)]
        def natural_keys(text): return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
        opts = sorted(opts, key=natural_keys)
        if not opts: return None
        if len(opts) == 1: return opts[0]
        print(f"\nSelect {name}:")
        for i, o in enumerate(opts): print(f"[{i+1}] {o}")
        return opts[int(input("Select: "))-1]

    if not args.version_tag: args.version_tag = get_input(RUNS_DIR, "Version")
    path_n = os.path.join(RUNS_DIR, args.version_tag)
    if not args.N: args.N = int(get_input(path_n, "Size (N)").replace("N",""))
    path_s = os.path.join(path_n, f"N{args.N}")
    if not args.seed: args.seed = get_input(path_s, "Seed").replace("S","")

    # --- COLOR MODE MENU ---
    if args.color_mode == "auto":
        print("\nSelect Color Mode:")
        print("[1] Degree (Connectivity) (Default)")
        print("[2] Spectral Angle (Manifold Position)")
        print("[3] Rho (Amplitude)")
        print("[4] Frustration (Geometric Stress)")
        print("[5] Phase Integration (Path Winding)")
        choice = input("Select [1-5]: ").strip()
        mode_map = {
            "1": "degree",
            "2": "spectral_angle",
            "3": "rho",
            "4": "frustration",
            "5": "phase_integration",
        }
        args.color_mode = mode_map.get(choice, "auto")
        print(f"Selected Mode: {args.color_mode}")

    base_path = os.path.join(path_s, f"S{args.seed}")
    out_dir = os.path.join("renders", f"N{args.N}_{args.version_tag}_S{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    # ==========================================
    #     TRACEABILITY COPIES
    # ==========================================
    # 1. Copy the visualization script itself
    shutil.copy(__file__, os.path.join(out_dir, "visualize.py"))

    # 2. Copy the engine and drive scripts
    for script in ["engine.py", "drive.py"]:
        src_path = os.path.join(path_n, script)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(out_dir, script))

    # 3. Copy the log file
    log_file_path = os.path.join(path_s, f"S{args.seed}_log.csv")
    if os.path.exists(log_file_path):
        shutil.copy(log_file_path, os.path.join(out_dir, f"S{args.seed}_log.csv"))
    # ==========================================

    files = sorted(glob.glob(os.path.join(base_path, "*_nodes.csv")))
    all_available_map = {}
    for f in files:
        m = re.search(r"iter_([\d_]+)_nodes", f)
        if m:
            s_str = m.group(1); s_int = int(s_str.replace('_', ''))
            all_available_map[s_int] = (f, f.replace("nodes", "edges"), s_str)

    zoom = args.zoom if args.zoom else DEFAULT_K_THRESHOLD
    log_file_path = os.path.join(path_s, f"S{args.seed}_log.csv")
    target_steps = set()
    gk = 10

    if os.path.exists(log_file_path) and zoom > 0:
        try:
            df = pd.read_csv(log_file_path, comment='#')
            if 'k_avg' in df.columns:
                df_clean = df.dropna(subset=['k_avg'])
                if not df_clean.empty:
                    k_vals = df_clean['k_avg'].values
                    steps = df_clean['iter'].values
                    min_k = np.nanmin(k_vals); max_k = np.nanmax(k_vals)
                    if np.isfinite(min_k) and np.isfinite(max_k):
                       # --- EARLY ZOOM LOGIC ---
                        ZOOM_SLICES = 10
                        ZOOM_WINDOW = 10  # Match this with drive.py

                        fine_zoom = zoom / ZOOM_SLICES
                        zoom_threshold = zoom * ZOOM_WINDOW

                        # Fine targets from 0 up to the extended threshold
                        fine_targets = np.arange(fine_zoom, zoom_threshold, fine_zoom)
                        # Standard targets from the threshold up to the max
                        standard_targets = np.arange(max(zoom_threshold, min_k), max_k + zoom, zoom)

                        # Combine them (including min_k)
                        targets = np.concatenate(([min_k], fine_targets, standard_targets))
                        targets = np.unique(targets) # Strip duplicates

                        # Find closest matches in the log
                        target_idxs = [np.abs(k_vals - t).argmin() for t in targets]
                        target_steps = set(steps[target_idxs])
                        if 'k_max' in df.columns: gk = df['k_max'].max()
        except: pass

    if all_available_map:
        target_steps.add(min(all_available_map.keys()))
        target_steps.add(max(all_available_map.keys()))

    tasks_map = {}
    for s_int, (f, edge_f, s_str) in all_available_map.items():
        if not target_steps or s_int in target_steps:
             tasks_map[s_int] = (f, edge_f, s_str)


    tasks_map = all_available_map  #comment out for special render all





    sorted_steps = sorted(tasks_map.keys())
    step_rank_map = {step: i for i, step in enumerate(sorted_steps)}
    ordered_steps = get_binary_spread_order(sorted_steps)




    final_tasks = []
    total_count = len(ordered_steps)
    for s_int in ordered_steps:
        f, edge_f, s_str = tasks_map[s_int]
        rank_idx = step_rank_map[s_int]
        final_tasks.append((s_int, f, edge_f, s_str, out_dir, args.version_tag, args.N, args.seed, gk, rank_idx, total_count, args.color_mode))

    print(f">> Rendering {len(final_tasks)} frames to {out_dir}...")

    pool = Pool(args.threads or max(1, cpu_count()//4))
    try:
        iterator = pool.imap_unordered(_launch_subprocess_star, [[t] for t in final_tasks])
        for _ in tqdm(iterator, total=len(final_tasks)):
            pass
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n\n[!] Ctrl+C Detected. Executing recursive kill...")
        pool.terminate()
        current_pid = os.getpid()
        if platform.system() == 'Windows':
            subprocess.run(f"taskkill /F /T /PID {current_pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import signal
            try:
                os.killpg(os.getpgid(current_pid), signal.SIGKILL)
            except:
                pass
        sys.exit(1)
    finally:
        if 'pool' in locals() and not pool._state:
            pool.join()

if __name__ == "__main__":
    main()
