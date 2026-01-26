import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from engine import PhysicsEngine
import os
import csv
import sys
import re

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

# CONFIG
NUM_RUNS = 3
CLUSTER_SIZE = 25
CONTEXT_HOPS_MICRO = 1
CONTEXT_HOPS_MACRO = 2

# TIMELINE CONFIG
INITIAL_HEAL_STEPS = 60  # Phase 1: Rapid response
EXTRA_HEAL_STEPS = 150   # Phase 2: Long-term relaxation

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

def find_available_steps(data_dir, version_id, N, seed):
    """Scans for all available step counts in the data directory."""
    steps = []
    # Pattern: E{v}_N{n}_S{s}_iter_{step}_edges.csv
    pattern = re.compile(rf"E{version_id}_N{N}_S{seed}_iter_([\d_]+)_edges\.csv")

    if not os.path.exists(data_dir):
        return []

    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            step_str = match.group(1).replace('_', '')
            steps.append(int(step_str))

    steps.sort()
    return steps

def fmt_step(step):
    """Matches drive.py formatting (000_000_000)."""
    return f"{step:011_d}"

def load_engine_complete(N, step, version_id, seed, data_dir):
    """Loads engine state using formatting compatible with drive.py."""
    eng = PhysicsEngine(N)

    step_str = fmt_step(step)
    # File format: E{v}_N{n}_S{s}_iter_{step}_edges.csv
    edge_file = os.path.join(data_dir, f"E{version_id}_N{N}_S{seed}_iter_{step_str}_edges.csv")

    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    with open(edge_file, 'r') as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if row:
                u, v = int(row[0]), int(row[1])
                eng.adj_matrix[u, v] = eng.adj_matrix[v, u] = True

    # TURBO MODE for rapid plasticity
    eng.params[0] = 0.60; eng.params[9] = 0.005; eng.params[10] = 0.005
    return eng

def get_neighbors(adj, node):
    return [i for i, is_connected in enumerate(adj[node]) if is_connected]

def get_cluster(adj, start_node, size):
    cluster = {start_node}
    queue = [start_node]
    while len(cluster) < size and queue:
        current = queue.pop(0)
        neighbors = get_neighbors(adj, current)
        for n in neighbors:
            if n not in cluster:
                cluster.add(n)
                queue.append(n)
                if len(cluster) >= size: break
    return list(cluster)

def get_view_nodes(adj, all_victims, buffer_hops):
    view = set(all_victims)
    current_shell = set(all_victims)
    for _ in range(buffer_hops):
        next_shell = set()
        for n in current_shell:
            neighbors = get_neighbors(adj, n)
            for nb in neighbors:
                if nb not in view:
                    view.add(nb)
                    next_shell.add(nb)
        current_shell = next_shell
    return list(view)

def draw_panel(ax, G, pos, title, mode, victims=None, new_edges=None):
    ax.clear()
    ax.set_title(title, fontsize=10, color='white', fontweight='bold', pad=4)
    ax.axis('off')

    C_BG_NODE = '#005577'
    C_BG_EDGE = '#222222'
    C_VICTIM  = '#ff0044'
    C_REPAIR  = '#00ff00'

    node_colors, node_sizes = [], []
    edge_colors, edge_widths = [], []

    new_edge_set = set(tuple(sorted(e)) for e in (new_edges or []))
    victim_set = set(victims) if victims else set()

    # EDGES
    for u, v in G.edges():
        pair = tuple(sorted((u, v)))
        if mode == "repair" and pair in new_edge_set:
            edge_colors.append(C_REPAIR); edge_widths.append(2.5)
        elif mode == "target" and (u in victim_set and v in victim_set):
            edge_colors.append(C_VICTIM); edge_widths.append(1.5)
        elif mode == "target" and (u in victim_set or v in victim_set):
            edge_colors.append(C_VICTIM); edge_widths.append(1.0)
        else:
            edge_colors.append(C_BG_EDGE); edge_widths.append(0.6)

    # NODES
    for n in G.nodes():
        if mode == "target" and n in victim_set:
            node_colors.append(C_VICTIM); node_sizes.append(60)
        else:
            node_colors.append(C_BG_NODE); node_sizes.append(30)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)

def capture_single_run(run_id, N, step_to_load, version_id, seed, data_dir):
    print(f"\n⚡ Run {run_id}: initializing...")
    hrp = load_engine_complete(N, step_to_load, version_id, seed, data_dir)

    while True:
        # 1. FIND DUAL TARGETS
        start_A = np.random.randint(0, N)
        cluster_A = get_cluster(hrp.adj_matrix, start_A, CLUSTER_SIZE)

        candidates_B = []
        shell = get_view_nodes(hrp.adj_matrix, cluster_A, 5)
        for n in shell:
            if n not in cluster_A: candidates_B.append(n)
        if not candidates_B: continue
        start_B = candidates_B[np.random.randint(0, len(candidates_B))]
        cluster_B = get_cluster(hrp.adj_matrix, start_B, CLUSTER_SIZE)
        all_victims = list(set(cluster_A + cluster_B))

        if len(all_victims) < CLUSTER_SIZE * 1.5: continue
        print(f"   Target Acquired: {len(all_victims)} nodes.")

        # 2. DEFINE VIEWPORTS
        nodes_micro = get_view_nodes(hrp.adj_matrix, all_victims, CONTEXT_HOPS_MICRO)
        nodes_macro = get_view_nodes(hrp.adj_matrix, all_victims, CONTEXT_HOPS_MACRO)
        G_micro_base = nx.from_numpy_array(hrp.adj_matrix).subgraph(nodes_micro).copy()
        G_macro_base = nx.from_numpy_array(hrp.adj_matrix).subgraph(nodes_macro).copy()

        # 3. LAYOUT 1: BASELINE
        try: pos_base = nx.kamada_kawai_layout(G_macro_base)
        except: continue

        # 4. BLAST
        for v in all_victims: hrp.adj_matrix[v, :] = False; hrp.adj_matrix[:, v] = False
        G_micro_void = G_micro_base.copy(); G_micro_void.remove_nodes_from(all_victims)
        G_macro_void = G_macro_base.copy(); G_macro_void.remove_nodes_from(all_victims)

        # 5. PHASE 1: RAPID HEAL
        for _ in range(INITIAL_HEAL_STEPS):
            for _ in range(N): hrp.step()

        new_edges_initial = []
        for u in G_macro_void.nodes():
            for v in G_macro_void.nodes():
                if u < v and hrp.adj_matrix[u, v] and not G_macro_base.has_edge(u, v):
                    new_edges_initial.append((u, v))
        if not new_edges_initial: continue

        G_micro_healed = G_micro_void.copy()
        G_macro_healed = G_macro_void.copy()
        for u, v in new_edges_initial:
            if u in G_micro_healed and v in G_micro_healed: G_micro_healed.add_edge(u, v)
            if u in G_macro_healed and v in G_macro_healed: G_macro_healed.add_edge(u, v)

        # LAYOUT 2: RAPID RESPONSE (Anchored tight)
        moveable = set(u for u,v in new_edges_initial) | set(v for u,v in new_edges_initial)
        fixed = [n for n in G_macro_healed.nodes() if n not in moveable]
        if not fixed: fixed = None
        pos_healed = nx.spring_layout(G_macro_healed, pos=pos_base, fixed=fixed, k=0.2, iterations=30)

        # 6. PHASE 2: LONG-TERM INTEGRATION
        print(f"   ⏳ Long-term stabilization ({EXTRA_HEAL_STEPS} steps)...")
        for _ in range(EXTRA_HEAL_STEPS):
             for _ in range(N): hrp.step()

        # LAYOUT 3: RELAXED (Less anchoring, allows spreading)
        # We unfix more nodes to allow the scar to integrate into the surroundings
        rim_nodes = get_view_nodes(hrp.adj_matrix, moveable, 1) # Get nodes touching the scar
        relaxed_moveable = set(moveable) | set(rim_nodes)
        fixed_relaxed = [n for n in G_macro_healed.nodes() if n not in relaxed_moveable]
        if not fixed_relaxed: fixed_relaxed = None

        # Higher k value = more spreading force
        pos_longterm = nx.spring_layout(G_macro_healed, pos=pos_healed, fixed=fixed_relaxed, k=0.3, iterations=50)

        print(f"   ✅ Run {run_id} Success!")

        # Return list of 6 states for columns
        return {
            "micro": [G_micro_base, G_micro_base, G_micro_void, G_micro_healed, G_micro_healed, G_micro_healed],
            "macro": [G_macro_base, G_macro_base, G_macro_void, G_macro_healed, G_macro_healed, G_macro_healed],
            # 3 layouts corresponding to time phases
            "pos_base": pos_base,
            "pos_healed": pos_healed,
            "pos_longterm": pos_longterm,
            "victims": all_victims,
            "new_edges": new_edges_initial
        }

def run_full_matrix():
    print("=========================================")
    print("   HEALING EXPERIMENT MATRIX (CLI)       ")
    print("=========================================")

    # 1. Select Version
    versions = get_subfolders(RUNS_DIR)
    versions.sort()
    if not versions:
        print(f"No Engine Versions found in '{RUNS_DIR}'")
        return
    ver_str = select_option(versions, "Select Engine Version")
    if not ver_str: return

    # 2. Select N
    n_path = os.path.join(RUNS_DIR, ver_str)
    n_counts = get_subfolders(n_path)
    # Sort numerically by N
    n_counts.sort(key=lambda x: int(x[1:]) if x.startswith('N') and x[1:].isdigit() else float('inf'))

    if not n_counts: print("No N-counts found."); return
    n_str = select_option(n_counts, "Select System Size (N)")
    if not n_str: return

    # 3. Select Seed
    seed_path = os.path.join(n_path, n_str)
    seeds = get_subfolders(seed_path)
    # Sort numerically by Seed
    seeds.sort(key=lambda x: int(x[1:]) if x.startswith('S') and x[1:].isdigit() else float('inf'))

    if not seeds: print("No Seeds found."); return
    seed_str = select_option(seeds, "Select Simulation Seed")
    if not seed_str: return

    # PARSE IDS
    try:
        v_id = int(ver_str[1:])
        n_id = int(n_str[1:])
        s_id = int(seed_str[1:])
    except ValueError:
        print("Error parsing folder names.")
        return

    # 4. LOCATE DATA
    data_dir = os.path.join(seed_path, seed_str, "data")
    if not os.path.exists(data_dir):
        print(f"Data directory missing: {data_dir}"); return

    # 5. FIND AVAILABLE STEPS & SELECT BY PERCENTAGE
    available_steps = find_available_steps(data_dir, v_id, n_id, s_id)
    if not available_steps:
        print("No edge snapshots found in this run.")
        return

    max_step = available_steps[-1]

    print(f"\n--- Select Snapshot Position ---")
    print(f"Range available: 0 to {max_step} steps")

    while True:
        try:
            # Added prompt text and default check
            pct_input = input("Enter percentage (0-100) to sample from [Default: 25]: ").strip()

            if not pct_input:
                pct = 25.0
            else:
                pct = float(pct_input)

            if 0 <= pct <= 100:
                target_step = int(max_step * (pct / 100.0))
                # Find closest available step
                closest_step = min(available_steps, key=lambda x: abs(x - target_step))
                print(f">> Selected {pct}% -> Step ~{target_step} (Using closest: {closest_step})")
                selected_step = closest_step
                break
            else:
                print("Please enter a number between 0 and 100.")
        except ValueError:
            print("Invalid input.")

    # 6. RUN
    data_rows = []
    for i in range(1, NUM_RUNS + 1):
        data_rows.append(capture_single_run(i, n_id, selected_step, v_id, s_id, data_dir))

    print("\n🎨 Rendering Grand Matrix (36 Panels)...")
    plt.style.use('dark_background')
    # 6 Columns now
    fig, axes = plt.subplots(NUM_RUNS * 2, 6, figsize=(28, 24))

    cols = ["Baseline", "Target Locked", "Impact Crater", "Rapid Response", "Long-Term Integration", "New Normal"]
    modes = ["normal", "target", "void", "repair", "repair", "normal"]

    for i, data in enumerate(data_rows):
        r_micro = i * 2
        r_macro = i * 2 + 1

        axes[r_micro, 0].set_ylabel(f"Run {i+1} (Micro)", fontsize=12, color='white', rotation=90, labelpad=10)
        axes[r_macro, 0].set_ylabel(f"Run {i+1} (Macro)", fontsize=12, color='white', rotation=90, labelpad=10)

        for col in range(6):
            # Determine which layout to use based on column index
            if col < 3: cur_pos = data["pos_base"]
            elif col == 3: cur_pos = data["pos_healed"]
            else: cur_pos = data["pos_longterm"] # Cols 4 and 5 use relaxed layout

            # Micro Panel
            draw_panel(axes[r_micro, col], data["micro"][col], cur_pos,
                       f"{cols[col]}", modes[col],
                       victims=data["victims"], new_edges=data["new_edges"])

            # Macro Panel
            draw_panel(axes[r_macro, col], data["macro"][col], cur_pos,
                       f"{cols[col]}", modes[col],
                       victims=data["victims"], new_edges=data["new_edges"])

    plt.tight_layout()
    output_filename = f"exp0_N{n_id}_S{s_id}_step{selected_step}_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.savefig(output_filename, dpi=350)
    print(f"📸 Saved: {output_filename}")

if __name__ == "__main__":
    run_full_matrix()
