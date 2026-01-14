import os
import csv
import time
import numpy as np
import importlib.util
import argparse
import sys
import random

# -----------------------------------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Relational Reality Multiverse Sweeper")

    # Core Physics Parameters
    parser.add_argument("--N", type=int, default=100, help="Number of nodes (default: 400)")
    parser.add_argument("--ratio", type=int, default=3000, help="Steps per Node ratio (default: 3000)")

    # Sweep Control
    parser.add_argument("--runs", type=int, default=100, help="Number of universes to simulate")
    parser.add_argument("--start_seed", type=int, default=1000, help="Starting RNG seed")
    parser.add_argument("--out", type=str, default="", help="Custom output filename (optional)")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# MEASUREMENT HELPERS
# -----------------------------------------------------------------------------
def measure_backreaction(sim, G):
    """
    Calculates Pearson correlation between Matter Density (rho) and Curvature.
    This is the 'r_rho_curv' metric.
    """
    try:
        if G.number_of_edges() == 0: return np.nan

        # Use main component for stability
        comp = max(sim.nx.connected_components(G), key=len)
        nodes = list(comp)

        # Sample if too large (speed optimization)
        if len(nodes) > sim.CURV_SAMPLE:
            nodes = random.sample(nodes, sim.CURV_SAMPLE)

        rho = np.array([sim.rho_i(G, i) for i in nodes], dtype=float)
        curv = np.array([sim.node_curvature_proxy(G, i) for i in nodes], dtype=float)

        return sim.pearson_r(rho, curv)
    except Exception as e:
        print(f"   [Warn] Backreaction check failed: {e}")
        return np.nan

def measure_gauge_invariance(sim, G, rng):
    """
    Checks if physics stays the same under gauge transformation.
    Returns the max difference in triangle flux (should be near machine epsilon).
    """
    try:
        # 1. Sample Triangles
        tri_samples = sim.sample_triangles(G, max_tri=500)

        # Filter for alive triangles
        def tri_is_alive(tri):
            i, j, k = tri
            return (G.has_edge(*sim.edge_key(i, j)) and
                    G.has_edge(*sim.edge_key(j, k)) and
                    G.has_edge(*sim.edge_key(k, i)))

        valid_tris = [tri for tri in tri_samples if tri_is_alive(tri)]
        if not valid_tris: return 0.0

        # 2. Measure Before
        tri_before = np.array([sim.triangle_flux_real(G, tri) for tri in valid_tris], dtype=float)

        # 3. Apply Transform
        # We work on a copy to not mess up the main graph for other tests
        G_copy = G.copy()
        sim.apply_random_gauge_transform(G_copy, rng)

        # 4. Measure After
        tri_after = np.array([sim.triangle_flux_real(G_copy, tri) for tri in valid_tris], dtype=float)

        # 5. Diff
        return float(np.nanmax(np.abs(tri_after - tri_before)))
    except Exception as e:
        print(f"   [Warn] Gauge check failed: {e}")
        return np.nan

def measure_quench_response(sim, G, rng):
    """
    Re-runs the quench experiment: Pokes the universe and sees if density/curvature
    changes are correlated.
    """
    try:
        if not sim.DO_QUENCH or G.number_of_edges() == 0:
            return np.nan

        # 1. Setup Copy
        G_quench = G.copy()
        adj_sets_quench = {n: set(G_quench.neighbors(n)) for n in G_quench.nodes()}

        # 2. Pick Source
        comp_list = list(max(sim.nx.connected_components(G_quench), key=len))
        if not comp_list: return np.nan
        quench_src = rng.choice(comp_list)

        # 3. Baseline Profiles
        curv_before, sc1 = sim.curvature_profile_vs_distance(G_quench, quench_src, sim.MAX_DIST_QUENCH)
        rho_before, sc2 = sim.density_profile_vs_distance(G_quench, quench_src, sim.MAX_DIST_QUENCH)
        shell_counts = np.minimum(sc1, sc2)

        # 4. Inject Energy
        region_nodes, _ = sim.nodes_within_radius(G_quench, quench_src, sim.QUENCH_RADIUS)
        for v in region_nodes:
            G_quench.nodes[v]["psi"] += (sim.QUENCH_STRENGTH + 0.0j)

        # 5. Relax (Physics Engine)
        # Note: We use the sim's update functions
        for _ in range(sim.QUENCH_RELAX_MOVES_STAGE1):
            rr = rng.random()
            if rr < (sim.P_PSI / (sim.P_PSI + sim.P_THETA)):
                sim.psi_update(G_quench)
            else:
                sim.theta_update(G_quench, adj_sets=adj_sets_quench)

        for _ in range(sim.QUENCH_RELAX_MOVES_STAGE2):
            rr = rng.random()
            if sim.QUENCH_ALLOW_REWIRE and rr < sim.QUENCH_P_REWIRE:
                sim.rewire_move(G_quench, sim.MU_DEG2, adj_sets_quench, rng)
            else:
                if rr < (sim.P_PSI / (sim.P_PSI + sim.P_THETA)):
                    sim.psi_update(G_quench)
                else:
                    sim.theta_update(G_quench, adj_sets=adj_sets_quench)

        # 6. Measure After
        curv_after, sc3 = sim.curvature_profile_vs_distance(G_quench, quench_src, sim.MAX_DIST_QUENCH)
        rho_after, sc4 = sim.density_profile_vs_distance(G_quench, quench_src, sim.MAX_DIST_QUENCH)

        # 7. Calculate Correlation
        shell_counts = np.minimum.reduce([shell_counts, sc3, sc4])
        delta_curv = curv_after - curv_before
        delta_rho = rho_after - rho_before

        mask = np.isfinite(delta_rho) & np.isfinite(delta_curv) & (shell_counts >= sim.MIN_SHELL_QUENCH)
        if np.sum(mask) >= 3:
            return sim.pearson_r(delta_rho[mask], delta_curv[mask])
        return np.nan

    except Exception as e:
        print(f"   [Warn] Quench failed: {e}")
        return np.nan

# -----------------------------------------------------------------------------
# SIMULATION RUNNER
# -----------------------------------------------------------------------------
def run_simulation_instance(seed_val, n_nodes, step_ratio):
    # 1. Load module
    filename = "main.py"
    if not os.path.exists(filename):
        print(f"[ERROR] Could not find {filename}.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("sim_module", filename)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)

    # 2. OVERRIDE CONFIG
    sim.SEED = seed_val
    sim.np.random.seed(seed_val)
    sim.random.seed(seed_val)
    sim.N = n_nodes
    sim.stepToNRatio = step_ratio

    sim.STEP_COUNT = sim.N * sim.stepToNRatio
    sim.LOG_EVERY = sim.STEP_COUNT + 100
    sim.VISUALIZE_STEPS = False

    # Folder setup
    sweep_dir = f"runs/sweep_frames_N{n_nodes}"
    if not os.path.exists(sweep_dir): os.makedirs(sweep_dir, exist_ok=True)
    sim.FRAMES_DIR = sweep_dir

    # 3. Run Simulation
    G_final = sim.run()

    # 4. Harvest Data
    rng = np.random.default_rng(seed_val + 999) # Separate RNG for tests

    # A. Topology
    degrees = [d for _, d in G_final.degree()]
    k_mean = np.mean(degrees)
    tri_count = sum(sim.nx.triangles(G_final).values()) // 3 if G_final.number_of_edges() else 0

    # B. Dimensions
    d_H, d_S = sim.analyze_dimensionality(G_final, plot=False)

    # C. Advanced Physics Checks
    # We calculate these NOW using the helpers above
    gauge_diff = measure_gauge_invariance(sim, G_final, rng)
    quench_r = measure_quench_response(sim, G_final, rng)
    backreaction_r = measure_backreaction(sim, G_final)

    # D. Gravity (Ricci Slope)
    edge_curvatures = sim.compute_ollivier_ricci_flow(G_final)
    node_curvatures = {node: [] for node in G_final.nodes()}
    for (u, v), k in edge_curvatures.items():
        node_curvatures[u].append(k)
        node_curvatures[v].append(k)

    matter_density = []
    scalar_curvature = []
    for i in G_final.nodes():
        psi = G_final.nodes[i]["psi"]
        rho = float(psi.real**2 + psi.imag**2)
        matter_density.append(rho)
        if node_curvatures[i]:
            R = np.mean(node_curvatures[i])
        else:
            R = 0.0
        scalar_curvature.append(R)

    slope, intercept = np.nan, np.nan
    if len(matter_density) > 10:
        slope, intercept = np.polyfit(matter_density, scalar_curvature, 1)

    return {
        # Metadata
        "seed": seed_val,
        "N_actual": sim.N,
        "Step_measured": sim.STEP_COUNT,

        # Core Metrics
        "k_mean": k_mean,
        "triangles": tri_count,
        "hausdorff_dim": d_H,
        "spectral_dim": d_S,

        # Advanced Physics
        "gravity_G": slope,
        "cosmological_const": intercept,
        "quench_response": quench_r,
        "backreaction_r": backreaction_r,
        "gauge_diff_loop": gauge_diff,

        # Parameters
        "param_LAMBDA_E": sim.LAMBDA_E_BASE,
        "param_LAMBDA_G": sim.LAMBDA_G,
        "param_BETA": sim.BETA
    }

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    csv_filename = args.out if args.out else f"runs/sweep_data_N{args.N}.csv"

    print(f"--- 🌌 MULTIVERSE SWEEPER INITIATED 🌌 ---")
    print(f"   Nodes (N)    : {args.N}")
    print(f"   Total Steps  : {args.N * args.ratio}")
    print(f"   Output File  : {csv_filename}")
    print(f"---------------------------------------------")

    start_time = time.time()
    file_exists = os.path.isfile(csv_filename)

    # Full Column List
    fieldnames = [
        "seed", "N_actual", "Step_measured",
        "k_mean", "triangles", "hausdorff_dim", "spectral_dim",
        "gravity_G", "cosmological_const",
        "quench_response", "backreaction_r", "gauge_diff_loop",
        "param_LAMBDA_E", "param_LAMBDA_G", "param_BETA"
    ]

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i in range(args.runs):
            current_seed = args.start_seed + i
            iteration_start = time.time()

            print(f"\n>> [Run {i+1}/{args.runs}] Universe #{current_seed} (N={args.N})")
            try:
                data = run_simulation_instance(current_seed, args.N, args.ratio)
                writer.writerow(data)
                csvfile.flush()

                duration = time.time() - iteration_start
                print(f"   ✅ COMPLETE in {duration:.1f}s")
                print(f"      k={data['k_mean']:.3f} | dH={data['hausdorff_dim']:.2f} | Quench={data['quench_response']:.3f}")

            except Exception as e:
                print(f"   ❌ FAILED: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n--- SWEEP FINISHED in {(time.time() - start_time)/60:.2f} minutes ---")

if __name__ == "__main__":
    main()
