import argparse
import numpy as np
import networkx as nx
import pandas as pd
import multiprocessing
import time
import sys
from engine import PhysicsEngine

# --- CONFIGURATION ---
PARAM_MAP = {
    "temp": 0,
    "kappa": 1,
    "lam_g": 2,
    "lam_psi": 3,
    "lam_pauli": 4,
    "rho0": 5,
    "mass2": 6,
    "beta": 7,
    "lam_e": 8,
    "p_psi": 9,
    "p_theta": 10,
    "psi_step": 11,
    "theta_step": 12,
    "mu": 14,
    "p_triadic": 15
}

# --- DEFAULTS ---
DEFAULT_P1 = "p_triadic"
DEFAULT_P1_START = 1.0
DEFAULT_P1_END = 0.9
DEFAULT_P1_STEPS = 10

DEFAULT_P2 = "mu"
DEFAULT_P2_START = 1.00
DEFAULT_P2_END = 0.005
DEFAULT_P2_STEPS = 20

# --- TEMPERATURE CONTROL ---
TEMP_START = 0.20
TEMP_DECAY = 0.9999
TEMP_FLOOR = 1e-20

# --- STABILITY GATES ---
CHECK_INTERVAL = 10000         # Steps between metric checks
STABILITY_STREAK_TARGET = 100  # How many intervals metrics must remain constant
REQUIRED_MEAN_DEGREE = 1.999    # MINIMUM average degree to count as a valid stable state
REQUIRED_MIN_DEGREE = 4       # Only for "CRYSTALLIZED" label (does not affect basic success)

def get_temperature(step, decay):
    return max(TEMP_START * (decay ** step), TEMP_FLOOR)

def calculate_metrics(G):
    """Replicates the metric calculation from drive.py"""
    degrees = [d for n, d in G.degree()]
    if not degrees: return 0, 0.0, 0, 0
    k_min, k_max, k_avg = np.min(degrees), np.max(degrees), np.mean(degrees)
    triangles = sum(nx.triangles(G).values()) // 3
    return k_min, k_avg, k_max, triangles

def run_single_sim(args):
    """
    Runs until stable (converged) or max_steps (timeout).
    Metric for success: FEWEST STEPS TO VALID STABILITY (k_avg > 1.0).
    """
    p1_name, p1_val, p2_name, p2_val, N, seed, max_steps = args

    engine = PhysicsEngine(N, seed)

    # 1. Apply Parameters
    decay = TEMP_DECAY
    if p1_name == "cooling": decay = p1_val
    elif p1_name in PARAM_MAP: engine.params[PARAM_MAP[p1_name]] = p1_val

    if p2_name == "cooling": decay = p2_val
    elif p2_name in PARAM_MAP: engine.params[PARAM_MAP[p2_name]] = p2_val

    # 2. Run Physics
    step = 0
    stable_streak = 0
    last_k_long = -1

    # Metrics placeholders
    k_min, k_avg, k_max, triangles = 0, 0.0, 0, 0
    status = "TIMEOUT"

    while step < max_steps:
        # Anneal
        current_temp = get_temperature(step, decay)
        engine.params[0] = current_temp

        # Burst Execution
        for _ in range(CHECK_INTERVAL):
            engine.step()
        step += CHECK_INTERVAL

        # --- METRICS CHECK ---
        k_min, k_avg, k_max, triangles = calculate_metrics(engine.G)

        # 1. NAN CHECK (Physics breakdown)
        if np.isnan(engine.psi[0].real):
            status = "NAN_PSI"
            break

        # 2. STABILITY CHECK (drive.py logic: int(k * 1000))
        # We wait for the graph to stop changing topology significantly.
        k_long = int(k_avg * 1000)

        if k_long == last_k_long:
            stable_streak += 1
        else:
            stable_streak = 1
            last_k_long = k_long

        if stable_streak >= STABILITY_STREAK_TARGET:
            # We are stable. But is it a USEFUL stability?
            if k_avg >= REQUIRED_MEAN_DEGREE:
                status = "STABLE"
                # Label as Crystallized if high quality
                if k_min >= REQUIRED_MIN_DEGREE:
                    status = "CRYSTALLIZED"
            else:
                # Stable but disconnected/dust (k < 1.0)
                status = "COLLAPSED"
            break

    return (p1_val, p2_val, step, status, k_avg, triangles, k_min, k_max)

def main():
    parser = argparse.ArgumentParser(description="Speed-to-Equilibrium Sweep")

    # Sweep Setup
    parser.add_argument("--p1", type=str, default=None)
    parser.add_argument("--start1", type=float, default=None)
    parser.add_argument("--end1", type=float, default=None)
    parser.add_argument("--steps1", type=int, default=None)

    parser.add_argument("--p2", type=str, default=None)
    parser.add_argument("--start2", type=float, default=None)
    parser.add_argument("--end2", type=float, default=None)
    parser.add_argument("--steps2", type=int, default=None)

    # Sim Config
    parser.add_argument("-N", "--nodes", type=int, default=400)
    parser.add_argument("--max_steps", type=int, default=10_000_000, help="Safety timeout")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("-t", "--threads", type=int, default=((int)(multiprocessing.cpu_count()/2.0)))

    args = parser.parse_args()

    # Defaults logic
    p1 = args.p1 if args.p1 else DEFAULT_P1
    s1 = args.start1 if args.start1 is not None else DEFAULT_P1_START
    e1 = args.end1 if args.end1 is not None else DEFAULT_P1_END
    n1 = args.steps1 if args.steps1 else DEFAULT_P1_STEPS

    p2 = args.p2 if args.p2 else DEFAULT_P2
    s2 = args.start2 if args.start2 is not None else DEFAULT_P2_START
    e2 = args.end2 if args.end2 is not None else DEFAULT_P2_END
    n2 = args.steps2 if args.steps2 else DEFAULT_P2_STEPS

    # Generate Tasks
    vals1 = np.linspace(s1, e1, n1)
    vals2 = np.linspace(s2, e2, n2)
    tasks = []

    print(f"--- EQUILIBRIUM SPEED SWEEP ---")
    print(f"P1: {p1} ({s1:.4f} -> {e1:.4f})")
    print(f"P2: {p2} ({s2:.4f} -> {e2:.4f})")
    print(f"Objective: Find parameters that reach STABILITY (k_avg >= {REQUIRED_MEAN_DEGREE}) in FEWEST STEPS.")
    print(f"Gate: Stable Streak = {STABILITY_STREAK_TARGET} checks of {CHECK_INTERVAL} steps")
    print("-" * 60)

    for v1 in vals1:
        for v2 in vals2:
            tasks.append((p1, v1, p2, v2, args.nodes, args.seed, args.max_steps))

    # Run
    start = time.time()
    results = []

    with multiprocessing.Pool(args.threads) as pool:
        total = len(tasks)
        for i, res in enumerate(pool.imap_unordered(run_single_sim, tasks), 1):
            results.append(res)

            # Unpack for readability
            steps = res[2]
            status = res[3]
            k_avg = res[4]

            # Dynamic status symbol
            status_sym = "?"
            if status == "CRYSTALLIZED": status_sym = "★"
            elif status == "STABLE": status_sym = "✓"
            elif status == "COLLAPSED": status_sym = "x"  # Stable but k < 1
            elif status == "NAN_PSI": status_sym = "!"
            elif status == "TIMEOUT": status_sym = "~"

            sys.stdout.write(f"\r[{i}/{total}] {status_sym} Steps={steps:,} (k={k_avg:.2f})")
            sys.stdout.flush()

    print(f"\n\nDone in {time.time()-start:.1f}s")

    # Sort logic:
    # 1. Must be VALID Stable (Status == STABLE or CRYSTALLIZED)
    # 2. Fewest Steps (Ascending)
    def sort_key(x):
        # x = (p1, p2, steps, status, ...)

        # Priority 1: Convergence status
        # COLLAPSED (k<1) is treated as a failure (0)
        is_converged = 1 if x[3] in ["STABLE", "CRYSTALLIZED"] else 0

        # Priority 2: Speed (Steps).
        # We want FEWEST steps. Since sort is reverse=True, we negate steps.
        # (Small steps = -Small = Big number -> First)
        return (is_converged, -x[2])

    results.sort(key=sort_key, reverse=True)

    if results:
        best = results[0]
        print("\n=== FASTEST TO CONVERGE (Valid) ===")
        print(f"{p1}: {best[0]:.6f}")
        print(f"{p2}: {best[1]:.6f}")
        print(f"Status: {best[3]}")
        print(f"Steps: {best[2]:,}")
        print(f"Resulting k_avg: {best[4]:.4f}")
    else:
        print("\nNo results found.")

    # Save
    df = pd.DataFrame(results, columns=[p1, p2, "steps", "status", "k_avg", "triangles", "k_min", "k_max"])
    csv_name = f"sweep_speed_{p1}_{p2}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved to {csv_name}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
