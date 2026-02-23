#!/usr/bin/env python3
"""
mu_sweep_for_stability_engine_live.py

Live-sweeping "universes" using engine.py directly.

Key features:
- Batch isolation: each invocation creates a unique batch folder, e.g. mu_sweep/batch_20260220_153012/
  Nothing overwrites previous runs.
- Each run writes:
    <batch>/MU0p0xxx/N####/S###/run_log.csv
    snapshots (CSV-only): snapshot_iter_XXXXXXXXXX_nodes.csv, snapshot_iter_XXXXXXXXXX_edges.csv
- Shared append-only summary:
    <batch>/sweep_summary_append.csv   (one row per completed run job)
- Dedicated plotter thread:
    Reads sweep_summary_append.csv periodically and updates:
    <batch>/live_dashboard.png
- Ctrl-C safe:
    Stops scheduling/waiting and exits; already-finished run data is preserved.

Computation:
- kmean is tracked over time inside each run_log.csv
- Early exits label: COMPLETED / UNSTABLE_FLICKER / FROZEN_SPARSE / MAX_ITERS_REACHED
- Dimensions (dH, dS, …) attempted at END for ALL statuses by default.
  If it fails => dim_status=FAILED and dim_note has error snippet.
  Only COMPLETED+dim_status=OK is considered fully_computed.

Prediction:
- The plotter tries to fit a crude “critical slowing” curve T(mu) to COMPLETED runs:
    T ~ a / (mu - mu_c)^gamma
  by scanning candidate mu_c and doing linear regression on log(T) vs log(mu-mu_c).
  This is best-effort and will be blank until enough completed points exist.

Requirements:
- engine.py providing PhysicsEngine(N, seed) and engine.iterate(), engine.G (networkx graph)
- measure_and_plot_dimensions.py with:
    compute_hausdorff_dimension(G, num_samples)
    auto_compute_spectral_dimension(G, base_seed, n_reps)
    compute_3d_measure(G)
"""

import argparse
import csv
import os
import time
import math
import signal
import threading
import multiprocessing as mp
import queue as queue_mod
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------------
# Defaults
# -------------------------
DEFAULT_NODES = 3201
DEFAULT_MU_START = 0.10000
DEFAULT_MU_END   = 0.00001
DEFAULT_MU_STEP  = 0.00010


# p_triadic sweep (optional outer sweep)
DEFAULT_PTRIADIC_VALUES = "0.99"  # e.g. "0,0.02,0.05"
DEFAULT_PTRIADIC_START = None
DEFAULT_PTRIADIC_END = None
DEFAULT_PTRIADIC_STEP = None
DEFAULT_PTRIADIC_PARAM_INDEX = 4

DEFAULT_SEEDS_PER_MU = 3
DEFAULT_BASE_SEED = 42

DEFAULT_STEP_INTERVAL = 100_000
DEFAULT_MAX_ITERS = 1_000_000_000

DEFAULT_STABILITY_WINDOW = 1_000_000
DEFAULT_STABILITY_TOL = 1e-8
DEFAULT_STABILITY_REPEAT = 4
DEFAULT_STABILITY_K_MIN = 2

DEFAULT_DELTA_K_SNAPSHOT = 0.01
DEFAULT_MAX_SPREAD_SNAPSHOTS = 10

# early exit
DEFAULT_FLICKER_WINDOW_TICKS = 300
DEFAULT_FLICKER_BIN = 0.01
DEFAULT_FLICKER_MIN_BIN_CHANGES = 10
DEFAULT_FLICKER_MIN_RANGE = 0.02
DEFAULT_FROZEN_SPARSE_TICKS = 50
DEFAULT_FROZEN_KMIN = 1

# completion + loop-detection
DEFAULT_COMPLETION_KMIN_FRAC = 0.25
DEFAULT_NO_NEW_HIGH_TICKS = 800
DEFAULT_NEW_HIGH_EPS = 1e-4

# dimensions (attempt on all runs)
DEFAULT_COMPUTE_DIMENSIONS = True
DEFAULT_HAUSDORFF_SAMPLES = 20
DEFAULT_SPECTRAL_REPS = 2
DEFAULT_SPECTRAL_WALKERS = 15000
DEFAULT_SPECTRAL_WALK_LENGTH = 120
DEFAULT_SPECTRAL_MIN_FIT_POINTS = 5
DEFAULT_SPECTRAL_PLATEAU_MULT = 15.0

# live plot
DEFAULT_LIVE_PLOT = True
DEFAULT_PLOT_INTERVAL_SEC = 60.0 #update only once a minute, dont waste needless resources

DEFAULT_OUT_ROOT = "mu_sweep"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) // 4)


# -------------------------
# Helpers
# -------------------------
def mu_grid(mu_start: float, mu_end: float, mu_step: float):
    if mu_step <= 0:
        raise ValueError("mu_step must be positive (sweep is downward internally).")
    mus = []
    mu = mu_start
    while mu >= mu_end - 1e-12:
        mus.append(round(mu, 4))
        mu -= mu_step
    return mus

def safe_mu(mu: float) -> str:
    return f"MU{mu:.4f}".replace(".", "p")


def safe_ptriadic(p: float | None) -> str:
    if p is None:
        return "PTRIunset"
    return f"PTRI{p:.6f}".replace(".", "p")

def parse_ptriadic_values(args) -> list[float | None]:
    # Priority: explicit comma list > range triplet > none
    if getattr(args, "p_triadic_values", None):
        vals = []
        for part in str(args.p_triadic_values).split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        return vals

    ps = getattr(args, "p_triadic_start", None)
    pe = getattr(args, "p_triadic_end", None)
    pst = getattr(args, "p_triadic_step", None)
    if ps is not None or pe is not None or pst is not None:
        if ps is None or pe is None or pst is None:
            raise ValueError("If using --p-triadic-start/end/step, you must provide all three.")
        if pst <= 0:
            raise ValueError("p_triadic_step must be > 0.")
        vals = []
        p = float(ps)
        # sweep upward inclusive
        while p <= float(pe) + 1e-12:
            vals.append(float(p))
            p += float(pst)
        return vals

    # no sweep requested
    return [None]


def set_engine_param(engine, *, mu: float | None = None, p_triadic: float | None = None, p_triadic_param_index: int = DEFAULT_PTRIADIC_PARAM_INDEX):
    # Best-effort param wiring: try attribute, dict-like params, then list-like params indices.
    if mu is not None:
        # existing behavior: engine.params[1] is mu in this codebase, but keep it best-effort.
        try:
            engine.params[1] = float(mu)
        except Exception:
            try:
                setattr(engine, "mu", float(mu))
            except Exception:
                pass

    if p_triadic is None:
        return

    v = float(p_triadic)
    # 1) direct attribute
    for attr in ("p_triadic", "pTriadic", "ptriadic"):
        try:
            if hasattr(engine, attr):
                setattr(engine, attr, v)
                return
        except Exception:
            pass

    # 2) dict-like params
    try:
        if hasattr(engine, "params") and isinstance(engine.params, dict):
            for key in ("p_triadic", "pTriadic", "ptriadic"):
                if key in engine.params:
                    engine.params[key] = v
                    return
    except Exception:
        pass

    # 3) list-like params index fallback
    try:
        engine.params[int(p_triadic_param_index)] = v
        return
    except Exception:
        pass

def fmt_iter(i: int) -> str:
    return f"{int(i):010d}"

def hms(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "?"
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def progress_printer(q, stop_evt, total_runs):
    """
    Consume progress messages from worker processes and print them live.
    """
    last = {}

    while not stop_evt.is_set():
        try:
            msg = q.get(timeout=0.25)
        except queue_mod.Empty:
            continue

        if msg is None:
            break

        key = (msg.get("mu"), msg.get("seed"))
        it = msg.get("iter", 0)

        # avoid duplicate prints
        if last.get(key) == it:
            continue
        last[key] = it

        mu = msg.get("mu")
        seed = msg.get("seed")

        print(
            f"N{msg.get('N')} mu={mu:.5f} seed={seed} "
            f"iter={it:<12_} "
            f"kavg={msg.get('k_avg'):<20} "
            f"ips={msg.get('iters_per_sec'):.0f}",
            flush=True
        )


def get_k_stats(G):
    """Return graph metrics.

    Returns:
        (k_min, k_avg, k_max, triangles, edges, tri_per_edge, transitivity, assortativity)

    Notes:
      - tri_per_edge is a simple density proxy: global triangles divided by the number of edges (T / m).
      - transitivity is NetworkX's global clustering coefficient.
      - assortativity is the degree assortativity coefficient (can be NaN if variance is 0).
    """
    import networkx as nx

    degrees = [d for _, d in G.degree()]
    if not degrees:
        return 0, float("nan"), 0, 0, 0, 0.0, 0.0, float("nan")

    k_min = min(degrees)
    k_max = max(degrees)
    k_avg = sum(degrees) / len(degrees)

    # triangles + edge count
    tri = sum(nx.triangles(G).values()) // 3 if G.number_of_nodes() > 0 else 0
    m = int(G.number_of_edges())
    tri_per_edge = (float(tri) / m) if m > 0 else 0.0

    # clustering metrics
    try:
        trans = float(nx.transitivity(G)) if G.number_of_nodes() >= 3 else 0.0
    except Exception:
        trans = 0.0

    try:
        assort = float(nx.degree_assortativity_coefficient(G))
        if not math.isfinite(assort):
            assort = float("nan")
    except Exception:
        assort = float("nan")

    return int(k_min), float(k_avg), int(k_max), int(tri), m, float(tri_per_edge), trans, assort



def export_snapshot_csv(G, run_dir: Path, iter_num: int):
    """CSV-only snapshot: nodes + edges."""
    import numpy as np
    it_str = fmt_iter(iter_num)
    nodes_path = run_dir / f"snapshot_iter_{it_str}_nodes.csv"
    edges_path = run_dir / f"snapshot_iter_{it_str}_edges.csv"

    with nodes_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "psi_real", "psi_imag"])
        for n in G.nodes():
            psi = G.nodes[n].get("psi", 0 + 0j)
            w.writerow([n, float(np.real(psi)), float(np.imag(psi))])

    with edges_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v"])
        for u, v in G.edges():
            w.writerow([u, v])


def select_spread_iters(candidate_iters: list[int], max_keep: int) -> set[int]:
    if max_keep <= 0 or not candidate_iters:
        return set()
    iters = sorted(set(candidate_iters))
    if len(iters) <= max_keep:
        return set(iters)
    keep = set()
    for j in range(max_keep):
        idx = round(j * (len(iters) - 1) / (max_keep - 1))
        keep.add(iters[idx])
    return keep


def load_dimensions_module(script_dir: Path):
    import importlib.util
    dim_py = script_dir / "measure_and_plot_dimensions.py"
    if not dim_py.exists():
        return None
    spec = importlib.util.spec_from_file_location("measure_and_plot_dimensions", str(dim_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def compute_dimensions_from_graph(mod, G, base_seed: int,
                                  hausdorff_samples: int,
                                  spectral_reps: int,
                                  spectral_walkers: int,
                                  spectral_walk_length: int,
                                  spectral_min_fit_points: int,
                                  spectral_plateau_mult: float):
    mod.NUM_WALKERS = int(spectral_walkers)
    mod.MAX_WALK_LENGTH = int(spectral_walk_length)
    mod.MIN_FIT_POINTS = int(spectral_min_fit_points)
    mod.PLATEAU_MULT = float(spectral_plateau_mult)

    dH = mod.compute_hausdorff_dimension(G, num_samples=int(hausdorff_samples))
    dS, dS_std, fit_pts, R_mean, window_str = mod.auto_compute_spectral_dimension(
        G, base_seed=int(base_seed), n_reps=int(spectral_reps)
    )
    d3 = mod.compute_3d_measure(G)
    return dH, dS, dS_std, fit_pts, R_mean, window_str, d3


def flicker_detectorA(k_series: list[float],
                     window_ticks: int,
                     bin_size: float,
                     min_bin_changes: int,
                     min_range: float) -> bool:
    if len(k_series) < window_ticks:
        return False
    w = k_series[-window_ticks:]
    w = [x for x in w if isinstance(x, float) and math.isfinite(x)]
    if len(w) < window_ticks:
        return False
    kmin = min(w)
    kmax = max(w)
    if (kmax - kmin) < min_range:
        return False
    def q(x): return int(math.floor(x / bin_size))
    bins = [q(x) for x in w]
    changes = sum(1 for i in range(1, len(bins)) if bins[i] != bins[i-1])
    return changes >= min_bin_changes

def flicker_detector(
    k_series: list[float],
    step_interval: int,
    window_iters: int,
    bin_size: float,
    hysteresis: float,
    loops: int,
    min_range: float,
    min_samples: int,
    min_balance: float,
) -> tuple[bool, dict]:
    """
    Returns (is_flicker, diagnostics).
    """
    import math

    # Window sizing
    if step_interval <= 0:
        return False, {"reason": "bad_step_interval"}
    need_points = max(min_samples, window_iters // step_interval)
    if len(k_series) < need_points:
        return False, {"reason": "not_enough_points", "need": need_points, "have": len(k_series)}

    w = k_series[-need_points:]
    w = [x for x in w if isinstance(x, float) and math.isfinite(x)]
    if len(w) < need_points:
        return False, {"reason": "nonfinite_in_window"}

    kmin = min(w); kmax = max(w)
    if (kmax - kmin) < min_range:
        return False, {"reason": "range_too_small", "range": (kmax - kmin)}

    # Hysteresis binning
    def base_bin(x):  # floor bin
        return int(math.floor(x / bin_size))

    bins = []
    cur = base_bin(w[0])
    bins.append(cur)
    for x in w[1:]:
        # current bin edges:
        low_edge = cur * bin_size
        high_edge = (cur + 1) * bin_size
        # switch only if outside hysteresis band
        if x < (low_edge - hysteresis):
            cur = base_bin(x)
        elif x > (high_edge + hysteresis):
            cur = base_bin(x)
        bins.append(cur)

    # balance check
    from collections import Counter
    c = Counter(bins)
    uniq = list(c.keys())
    if len(uniq) != 2:
        return False, {"reason": "not_two_bins", "unique_bins": len(uniq)}

    total = len(bins)
    frac0 = c[uniq[0]] / total
    frac1 = c[uniq[1]] / total
    if min(frac0, frac1) < min_balance:
        return False, {"reason": "unbalanced", "fractions": (frac0, frac1)}

    # compress duplicates
    comp = [bins[0]]
    for b in bins[1:]:
        if b != comp[-1]:
            comp.append(b)

    need_comp = 2 * loops + 1
    if len(comp) < need_comp:
        return False, {"reason": "not_enough_switches", "comp_len": len(comp), "need": need_comp}

    tail = comp[-need_comp:]
    # alternation check
    for i in range(1, len(tail)):
        if tail[i] == tail[i-1]:
            return False, {"reason": "no_alternation_in_tail"}

    return True, {
        "reason": "flicker_detected",
        "range": (kmax - kmin),
        "comp_len": len(comp),
        "fractions": (frac0, frac1),
        "bins": tuple(sorted(uniq)),
    }


# -------------------------
# Run function (one mu,seed)
# -------------------------
@dataclass
class DimCfg:
    base_seed: int
    hausdorff_samples: int
    spectral_reps: int
    spectral_walkers: int
    spectral_walk_length: int
    spectral_min_fit_points: int
    spectral_plateau_mult: float


def run_one_engine(
    script_dir: Path,
    nodes: int,
    seed: int,
    mu: float,
    p_triadic: float | None,
    p_triadic_param_index: int,
    run_dir: Path,
    step_interval: int,
    max_iters: int,
    stability_window: int,
    stability_tol: float,
    stability_repeat: int,
    stability_k_min: int,
    delta_k_snapshot: float,
    max_spread_snapshots: int,
    flicker_window_ticks: int,
    flicker_bin: float,
    flicker_min_bin_changes: int,
    flicker_min_range: float,
    frozen_sparse_ticks: int,
    frozen_kmin: int,
    completion_kmin_frac: float,
    no_new_high_ticks: int,
    new_high_eps: float,
    compute_dimensions: bool,
    dim_cfg: DimCfg, progress_q=None):
    import datetime
    import numpy as np
    from engine import PhysicsEngine

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.csv"

    dim_mod = load_dimensions_module(script_dir) if compute_dimensions else None

    engine = PhysicsEngine(nodes, seed)
    # Best-effort parameter wiring
    set_engine_param(engine, mu=mu, p_triadic=p_triadic, p_triadic_param_index=p_triadic_param_index)

    # snapshots
    milestone_bucket = None
    kept_iters = set()
    candidate_iters = []
    export_snapshot_csv(engine.G, run_dir, 0)
    kept_iters.add(0)

    # log header
    fieldnames = [
        "timestamp", "iter", "elapsed_time",
        "k_min", "k_avg", "k_max", "triangles", "edges", "tri_per_edge", "transitivity", "assortativity",
        "iters_per_sec",
        "status", "note",
        "dim_status", "dim_note",
        "d_H", "d_S", "d_S_std", "d_S_fit_pts", "d_S_R", "d_S_Window", "d_3D"
    ]
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    t0 = time.time()
    best_kavg = None
    ticks_since_high = 0

    status = ""
    note = ""
    k_series = []
    frozen_sparse_streak = 0

    final_iter = 0

    for it in range(1, int(max_iters) + 1):
        engine.iterate()

        if it % step_interval != 0:
            continue

        final_iter = it
        k_min, k_avg, k_max, triangles, edges, tri_per_edge, transitivity, assortativity = get_k_stats(engine.G)
        elapsed = time.time() - t0
        ips = (it / elapsed) if elapsed > 0 else None

        k_series.append(k_avg)
        candidate_iters.append(it)

        if np.isfinite(k_avg):
            bucket = int(math.floor(k_avg / delta_k_snapshot))
            if milestone_bucket is None:
                milestone_bucket = bucket
            if bucket != milestone_bucket:
                milestone_bucket = bucket
                export_snapshot_csv(engine.G, run_dir, it)
                kept_iters.add(it)


        # COMPLETED criterion: k_min reaches a fraction of k_mean
        if np.isfinite(k_avg) and k_avg > 0 and k_min >= (completion_kmin_frac * k_avg):
            status = "COMPLETED"
            note = f"k_min>={completion_kmin_frac:g}*kmean"
            break

        # Loop detection: if k_mean stops making new highs for too long, we may be stuck
        if np.isfinite(k_avg):
            if best_kavg is None:
                best_kavg = k_avg
                ticks_since_high = 0
            else:
                rel_eps = new_high_eps * max(1.0, abs(best_kavg))
                if k_avg > best_kavg + rel_eps:
                    best_kavg = k_avg
                    ticks_since_high = 0
                else:
                    ticks_since_high += 1

            if ticks_since_high >= no_new_high_ticks:
                status = "LOOP_DETECTED"
                note = f"no new kmean highs for {ticks_since_high} ticks (best={best_kavg:.6g})"
                break

        # append progress row (dims blank by design until end)
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow({
                "timestamp": datetime.datetime.now().isoformat(),
                "iter": it,
                "elapsed_time": elapsed,
                "k_min": k_min,
                "k_avg": k_avg,
                "k_max": k_max,
                "triangles": triangles,
                "edges": edges,
                "tri_per_edge": tri_per_edge,
                "transitivity": transitivity,
                "assortativity": assortativity,
                "iters_per_sec": ips if ips is not None else "",
                "status": "",
                "note": "",
                "dim_status": "",
                "dim_note": "",
                "d_H": "",
                "d_S": "",
                "d_S_std": "",
                "d_S_fit_pts": "",
                "d_S_R": "",
                "d_S_Window": "",
                "d_3D": "",
            })

        # also stream progress to parent console
        if progress_q is not None:
            try:
                progress_q.put_nowait({
                    "mu": mu,
                    "seed": seed,
                    "iter": it,
                    "N": engine.G.number_of_nodes(),
                    "elapsed_time": elapsed,
                    "k_min": k_min,
                    "k_avg": k_avg,
                    "k_max": k_max,
                    "triangles": triangles,
                "edges": edges,
                "tri_per_edge": tri_per_edge,
                    "iters_per_sec": ips,
                })
            except Exception:
                pass

    if not status:
        status = "MAX_ITERS_REACHED"
        note = f"Hit max_iters={max_iters}"

    if final_iter not in kept_iters:
        export_snapshot_csv(engine.G, run_dir, final_iter)
        kept_iters.add(final_iter)

    # spread selection (nearest kept; no replay)
    spread_keep = select_spread_iters(candidate_iters, max_spread_snapshots)
    kept_sorted = sorted(kept_iters)
    approx_spread = set()
    for target in spread_keep:
        near = min(kept_sorted, key=lambda k: abs(k - target))
        approx_spread.add(near)

    final_keep = set(kept_iters).union(approx_spread)

    # prune snapshots
    for p in run_dir.glob("snapshot_iter_*_nodes.csv"):
        try:
            itp = int(p.name.split("snapshot_iter_")[1].split("_nodes.csv")[0])
        except Exception:
            continue
        if itp not in final_keep:
            p.unlink(missing_ok=True)
            (run_dir / f"snapshot_iter_{fmt_iter(itp)}_edges.csv").unlink(missing_ok=True)

    for p in run_dir.glob("snapshot_iter_*_edges.csv"):
        try:
            itp = int(p.name.split("snapshot_iter_")[1].split("_edges.csv")[0])
        except Exception:
            continue
        if itp not in final_keep:
            p.unlink(missing_ok=True)

    # terminal measures
    k_min, k_avg, k_max, triangles, edges, tri_per_edge, transitivity, assortativity = get_k_stats(engine.G)
    elapsed = time.time() - t0
    ips = (final_iter / elapsed) if elapsed > 0 else None

    # dimensions attempt for ALL statuses (default)
    dH = dS = dS_std = R_mean = d3 = None
    fit_pts = None
    window_str = None
    dim_status = ""
    dim_note = ""

    if compute_dimensions and dim_mod is not None:
        try:
            dH, dS, dS_std, fit_pts, R_mean, window_str, d3 = compute_dimensions_from_graph(
                dim_mod, engine.G,
                base_seed=dim_cfg.base_seed,
                hausdorff_samples=dim_cfg.hausdorff_samples,
                spectral_reps=dim_cfg.spectral_reps,
                spectral_walkers=dim_cfg.spectral_walkers,
                spectral_walk_length=dim_cfg.spectral_walk_length,
                spectral_min_fit_points=dim_cfg.spectral_min_fit_points,
                spectral_plateau_mult=dim_cfg.spectral_plateau_mult,
            )
            dim_status = "OK"
        except Exception as e:
            dim_status = "FAILED"
            dim_note = (str(e) or "dimension calc error")[:160]

    fully_computed = (status == "COMPLETED" and dim_status == "OK")

    # terminal row
    with log_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iter": final_iter,
            "elapsed_time": elapsed,
            "k_min": k_min,
            "k_avg": k_avg,
            "k_max": k_max,
            "triangles": triangles,
                "edges": edges,
                "tri_per_edge": tri_per_edge,
            "transitivity": transitivity,
            "assortativity": assortativity,
            "iters_per_sec": ips if ips is not None else "",
            "status": status,
            "note": note,
            "dim_status": dim_status,
            "dim_note": dim_note,
            "d_H": "" if dH is None else dH,
            "d_S": "" if dS is None else dS,
            "d_S_std": "" if dS_std is None else dS_std,
            "d_S_fit_pts": "" if fit_pts is None else fit_pts,
            "d_S_R": "" if R_mean is None else R_mean,
            "d_S_Window": "" if window_str is None else window_str,
            "d_3D": "" if d3 is None else d3,
        })

    return {
        "mu": mu,
        "p_triadic": p_triadic,
        "seed": seed,
        "nodes": nodes,
        "status": status,
        "iter": final_iter,
        "elapsed_sec": elapsed,
        "iters_per_sec": (final_iter / elapsed) if elapsed > 0 else None,
        "kmean": k_avg,
        "triangles": triangles,
        "edges": edges,
        "tri_per_edge": tri_per_edge,
        "transitivity": transitivity,
        "assortativity": assortativity,
        "note": note,
        "dim_status": dim_status,
        "dim_note": dim_note,
        "d_H": dH,
        "d_S": dS,
        "d_S_std": dS_std,
        "d_S_fit_pts": fit_pts,
        "d_S_R": R_mean,
        "d_S_Window": window_str,
        "d_3D": d3,
        "fully_computed": fully_computed,
        "kept_snapshots": len(final_keep),
        "run_dir": str(run_dir),
    }


# -------------------------
# Append-only summary writer
# -------------------------
SUMMARY_FIELDS = [
    "batch_id",
    "mu", "p_triadic", "seed", "nodes",
    "status", "iter", "elapsed_sec", "iters_per_sec",
    "kmean", "triangles", "edges", "tri_per_edge", "transitivity", "assortativity", "note",
    "dim_status", "dim_note",
    "d_H", "d_S", "d_S_std", "d_S_fit_pts", "d_S_R", "d_S_Window", "d_3D",
    "fully_computed",
    "kept_snapshots",
    "run_dir",
]


def ensure_summary_header(summary_csv: Path):
    if summary_csv.exists() and summary_csv.stat().st_size > 0:
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()


def append_summary_row(summary_csv: Path, row: dict):
    # atomic-ish append (single process is doing it)
    with summary_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


# -------------------------
# Plotter thread
# -------------------------
def try_fit_critical_slowing(mus, Ts):
    """
    Fit T ~ a / (mu - muc)^gamma for COMPLETED points.
    """
    import numpy as np

    mus = np.asarray(mus, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    ok = np.isfinite(mus) & np.isfinite(Ts) & (Ts > 0)
    mus, Ts = mus[ok], Ts[ok]
    if len(mus) < 6:
        return None

    mu_min = float(np.min(mus))
    mu_max = float(np.max(mus))

    # candidate muc must be below min(mu) a bit
    # scan a small range below observed min
    span = max(1e-6, (mu_max - mu_min))
    muc_candidates = np.linspace(mu_min - 0.5*span, mu_min - 1e-6, 60)

    best = None
    best_sse = None

    for muc in muc_candidates:
        x = mus - muc
        if np.any(x <= 0):
            continue
        lx = np.log(x)
        ly = np.log(Ts)

        # linear regression ly = b0 + b1*lx  where b1 = -gamma
        A = np.vstack([np.ones_like(lx), lx]).T
        # least squares
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        b0, b1 = coef
        pred = A @ coef
        sse = float(np.sum((ly - pred) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            gamma = -b1
            a = float(np.exp(b0))
            best = (float(muc), float(gamma), float(a))

    return best



def plotter_loop(stop_evt: threading.Event, summary_csv: Path, out_png: Path,
                 nodes_label: str, interval_sec: float):
    """
    Reads appended summary CSV and updates a PNG dashboard continuously.

    If p_triadic is present in the summary, plots are overlaid by p_triadic value.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last_mtime = None

    def ffloat(x):
        try:
            return float(x)
        except Exception:
            return None

    def fint(x):
        try:
            return int(float(x))
        except Exception:
            return None

    while not stop_evt.is_set():
        try:
            if summary_csv.exists():
                mtime = summary_csv.stat().st_mtime
                if last_mtime is None or mtime != last_mtime:
                    last_mtime = mtime

                    rows = []
                    with summary_csv.open("r", newline="") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            rows.append(row)

                    # columns
                    mu = []
                    ptri = []
                    kmean = []
                    dH = []
                    dS = []
                    trans = []
                    assort = []
                    iters = []
                    ips = []
                    status = []

                    for row in rows:
                        mu.append(ffloat(row.get("mu")))
                        ptri.append(ffloat(row.get("p_triadic")))
                        kmean.append(ffloat(row.get("kmean")))
                        dH.append(ffloat(row.get("d_H")))
                        dS.append(ffloat(row.get("d_S")))
                        trans.append(ffloat(row.get("transitivity")))
                        assort.append(ffloat(row.get("assortativity")))
                        iters.append(fint(row.get("iter")))
                        ips.append(ffloat(row.get("iters_per_sec")))
                        status.append((row.get("status") or "").strip())

                    mu_np = np.asarray([x if x is not None else np.nan for x in mu], dtype=float)
                    ptri_np = np.asarray([x if x is not None else np.nan for x in ptri], dtype=float)
                    k_np = np.asarray([x if x is not None else np.nan for x in kmean], dtype=float)
                    dH_np = np.asarray([x if x is not None else np.nan for x in dH], dtype=float)
                    dS_np = np.asarray([x if x is not None else np.nan for x in dS], dtype=float)
                    trans_np = np.asarray([x if x is not None else np.nan for x in trans], dtype=float)
                    assort_np = np.asarray([x if x is not None else np.nan for x in assort], dtype=float)
                    it_np = np.asarray([x if x is not None else np.nan for x in iters], dtype=float)
                    ips_np = np.asarray([x if x is not None else np.nan for x in ips], dtype=float)

                    # grouping
                    has_ptri = np.any(np.isfinite(ptri_np))
                    if has_ptri:
                        uniq = sorted({float(x) for x in ptri_np if np.isfinite(x)})
                        groups = [(p, np.isclose(ptri_np, p, rtol=0, atol=1e-12)) for p in uniq]
                        title_suffix = " (overlay by p_triadic)"
                    else:
                        groups = [(None, np.ones_like(mu_np, dtype=bool))]
                        title_suffix = ""

                    # COMPLETED set for convergence fit (pooled)
                    completed_mask = np.array([s == "COMPLETED" for s in status], dtype=bool)
                    fit = None
                    if np.any(completed_mask):
                        fit = try_fit_critical_slowing(mu_np[completed_mask], it_np[completed_mask])

                    fig = plt.figure(figsize=(12, 16))
                    gs = fig.add_gridspec(7, 1, hspace=0.35)

                    ax1_top = fig.add_subplot(gs[0, 0])
                    ax1_bot = fig.add_subplot(gs[1, 0])
                    ax2 = fig.add_subplot(gs[2, 0])
                    ax3 = fig.add_subplot(gs[3, 0])
                    ax4 = fig.add_subplot(gs[4, 0])
                    ax5 = fig.add_subplot(gs[5, 0])
                    ax6 = fig.add_subplot(gs[6, 0])

                    # broken-axis kmean
                    for p, mask in groups:
                        lab = None if p is None else f"p_triadic={p:g}"
                        ax1_top.scatter(mu_np[mask], k_np[mask], s=14, label=lab)
                        ax1_bot.scatter(mu_np[mask], k_np[mask], s=14, label=lab)

                    ax1_top.set_ylim(4.5, 1000)
                    ax1_bot.set_ylim(0, 4.5)
                    ax1_bot.set_yticks([2, 3, 4])
                    ax1_top.set_yscale("log")
                    ax1_top.spines["bottom"].set_visible(False)
                    ax1_bot.spines["top"].set_visible(False)
                    ax1_top.xaxis.tick_top()
                    ax1_top.tick_params(labeltop=False)
                    ax1_bot.xaxis.tick_bottom()

                    d = 0.015
                    kwargs = dict(transform=ax1_top.transAxes, color="k", clip_on=False)
                    ax1_top.plot((-d, +d), (-d, +d), **kwargs)
                    ax1_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                    kwargs.update(transform=ax1_bot.transAxes)
                    ax1_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                    ax1_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                    ax1_top.set_title(f"Live Degree Penality (mu) sweep dashboard ({nodes_label}){title_suffix}")
                    ax1_top.set_ylabel("kmean (>8)")
                    ax1_bot.set_ylabel("kmean (≤8)")

                    # other plots
                    for p, mask in groups:
                        lab = None if p is None else f"p_triadic={p:g}"
                        ax2.scatter(mu_np[mask], dH_np[mask], s=14, label=lab)
                        ax3.scatter(mu_np[mask], dS_np[mask], s=14, label=lab)
                        ax4.scatter(mu_np[mask], trans_np[mask], s=14, label=lab)
                        ax5.scatter(mu_np[mask], assort_np[mask], s=14, label=lab)
                        ax6.scatter(mu_np[mask], ips_np[mask], s=14, label=lab)

                    ax2.set_ylabel("dH (Hausdorff)")
                    ax3.set_ylabel("dS (Spectral)")
                    ax4.set_ylabel("transitivity")
                    ax5.set_ylabel("degree assortativity.")
                    ax6.set_ylabel("iters/sec")
                    ax6.set_xlabel("μ")

                    # iters (final) overlay on ax6 right axis (pooled)
                    ax6b = ax6.twinx()
                    ax6b.scatter(mu_np, it_np, s=10)
                    ax6b.set_ylabel("iters (final)")

                    if fit is not None:
                        muc, gamma, a = fit
                        mu_line = np.linspace(np.nanmin(mu_np), np.nanmax(mu_np), 200)
                        valid = mu_line > muc
                        predT = np.full_like(mu_line, np.nan, dtype=float)
                        predT[valid] = a / np.power((mu_line[valid] - muc), gamma)
                        ax6b.plot(mu_line, predT)
                        ax6.set_title(f"iters/sec + predicted iters (fit: mu_c≈{muc:.5f}, γ≈{gamma:.2f})")

                    for ax in (ax1_top, ax1_bot, ax2, ax3, ax4, ax5, ax6):
                        ax.grid(True, alpha=0.25)

                    # Legends only if we have multiple p_triadic groups
                    if has_ptri and len(groups) > 1:
                        ax1_top.legend(loc="best", fontsize=8)
                        ax2.legend(loc="best", fontsize=8)

                    fig.savefig(out_png, dpi=170)
                    plt.close(fig)

        except Exception:
            # Plot thread should never crash the sweep
            pass

        stop_evt.wait(interval_sec)


# -------------------------
# Main sweep

# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    ap.add_argument("--mu-start", type=float, default=DEFAULT_MU_START)
    ap.add_argument("--mu-end", type=float, default=DEFAULT_MU_END)
    ap.add_argument("--mu-step", type=float, default=DEFAULT_MU_STEP)

    # optional outer sweep over p_triadic
    ap.add_argument("--p-triadic-values", type=str, default=DEFAULT_PTRIADIC_VALUES,
                    help='Comma-separated values, e.g. "0,0.02,0.05". If set, overrides start/end/step.')
    ap.add_argument("--p-triadic-start", type=float, default=DEFAULT_PTRIADIC_START)
    ap.add_argument("--p-triadic-end", type=float, default=DEFAULT_PTRIADIC_END)
    ap.add_argument("--p-triadic-step", type=float, default=DEFAULT_PTRIADIC_STEP)
    ap.add_argument("--p-triadic-param-index", type=int, default=DEFAULT_PTRIADIC_PARAM_INDEX,
                    help="Fallback index into engine.params for p_triadic if not attribute/dict-like.")

    ap.add_argument("--seeds-per-mu", type=int, default=DEFAULT_SEEDS_PER_MU)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)

    ap.add_argument("--step-interval", type=int, default=DEFAULT_STEP_INTERVAL)
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)

    ap.add_argument("--stability-window", type=int, default=DEFAULT_STABILITY_WINDOW)
    ap.add_argument("--stability-tol", type=float, default=DEFAULT_STABILITY_TOL)
    ap.add_argument("--stability-repeat", type=int, default=DEFAULT_STABILITY_REPEAT)
    ap.add_argument("--stability-k-min", type=int, default=DEFAULT_STABILITY_K_MIN)

    ap.add_argument("--completion-kmin-frac", type=float, default=DEFAULT_COMPLETION_KMIN_FRAC,
                    help="COMPLETED when k_min >= frac * k_mean")
    ap.add_argument("--no-new-high-ticks", type=int, default=DEFAULT_NO_NEW_HIGH_TICKS,
                    help="Loop detection: if k_mean makes no new highs for this many ticks, stop")
    ap.add_argument("--new-high-eps", type=float, default=DEFAULT_NEW_HIGH_EPS,
                    help="Relative eps for considering a k_mean value a new high")

    ap.add_argument("--delta-k-snapshot", type=float, default=DEFAULT_DELTA_K_SNAPSHOT)
    ap.add_argument("--max-spread-snapshots", type=int, default=DEFAULT_MAX_SPREAD_SNAPSHOTS)

    ap.add_argument("--flicker-window-ticks", type=int, default=DEFAULT_FLICKER_WINDOW_TICKS)
    ap.add_argument("--flicker-bin", type=float, default=DEFAULT_FLICKER_BIN)
    ap.add_argument("--flicker-min-bin-changes", type=int, default=DEFAULT_FLICKER_MIN_BIN_CHANGES)
    ap.add_argument("--flicker-min-range", type=float, default=DEFAULT_FLICKER_MIN_RANGE)
    ap.add_argument("--frozen-sparse-ticks", type=int, default=DEFAULT_FROZEN_SPARSE_TICKS)
    ap.add_argument("--frozen-kmin", type=int, default=DEFAULT_FROZEN_KMIN)

    ap.add_argument("--compute-dimensions", action="store_true", default=DEFAULT_COMPUTE_DIMENSIONS)
    ap.add_argument("--hausdorff-samples", type=int, default=DEFAULT_HAUSDORFF_SAMPLES)
    ap.add_argument("--spectral-reps", type=int, default=DEFAULT_SPECTRAL_REPS)
    ap.add_argument("--spectral-walkers", type=int, default=DEFAULT_SPECTRAL_WALKERS)
    ap.add_argument("--spectral-walk-length", type=int, default=DEFAULT_SPECTRAL_WALK_LENGTH)
    ap.add_argument("--spectral-min-fit-points", type=int, default=DEFAULT_SPECTRAL_MIN_FIT_POINTS)
    ap.add_argument("--spectral-plateau-mult", type=float, default=DEFAULT_SPECTRAL_PLATEAU_MULT)

    ap.add_argument("--out", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    ap.add_argument("--live-plot", action="store_true", default=DEFAULT_LIVE_PLOT)
    ap.add_argument("--plot-interval-sec", type=float, default=DEFAULT_PLOT_INTERVAL_SEC)

    args = ap.parse_args()

    # batch id = datetime
    batch_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out)
    batch_root = out_root / f"batch_{batch_id}"
    batch_root.mkdir(parents=True, exist_ok=True)

    summary_csv = batch_root / "sweep_summary_append.csv"
    ensure_summary_header(summary_csv)

    script_dir = Path(__file__).resolve().parent

    mus = mu_grid(args.mu_start, args.mu_end, args.mu_step)
    ptri_values = parse_ptriadic_values(args)
    seeds = [args.base_seed + i for i in range(args.seeds_per_mu)]

    jobs = []
    for ptri in ptri_values:
        for mu in mus:
            for seed in seeds:
                run_dir = batch_root / safe_ptriadic(ptri) / safe_mu(mu) / f"N{args.nodes}" / f"S{seed}"
                jobs.append((ptri, mu, seed, run_dir))

    total = len(jobs)
    print(f"Batch: {batch_id}")
    print(f"Output: {batch_root}")
    print(f"Total runs: {total} | workers={args.workers}")
    if ptri_values != [None]:
        print(f"p_triadic sweep: {ptri_values}")
    print("Ctrl-C anytime: already-finished run dirs + appended summary are preserved.\n")

    dim_cfg = DimCfg(
        base_seed=args.base_seed,
        hausdorff_samples=args.hausdorff_samples,
        spectral_reps=args.spectral_reps,
        spectral_walkers=args.spectral_walkers,
        spectral_walk_length=args.spectral_walk_length,
        spectral_min_fit_points=args.spectral_min_fit_points,
        spectral_plateau_mult=args.spectral_plateau_mult,
    )

    # Plotter thread
    stop_plot = threading.Event()
    plot_thread = None
    out_png = batch_root / "live_dashboard.png"
    if args.live_plot:
        nodes_label = f"N={args.nodes}"
        plot_thread = threading.Thread(
            target=plotter_loop,
            args=(stop_plot, summary_csv, out_png, nodes_label, float(args.plot_interval_sec)),
            daemon=True,
        )
        plot_thread.start()



    # Progress printer (workers -> parent)
    mgr = mp.Manager()
    progress_q = mgr.Queue(maxsize=10_000)
    stop_progress = threading.Event()
    progress_thread = threading.Thread(
        target=progress_printer,
        args=(progress_q, stop_progress, total),  # ← add total
        daemon=True,
    )
    progress_thread.start()

    # Ctrl-C handling: stop cleanly
    stop_main = {"stop": False}

    def on_sigint(signum, frame):
        #stop_main["stop"] = True
        print("\n[CTRL-C] Stopping submission / waiting. Preserving completed run data…", flush=True)

    #signal.signal(signal.SIGINT, on_sigint)

    started = time.time()
    done = 0
    wall_times = []

    # Submit everything up front; on ctrl-c, we stop consuming results ASAP.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for ptri, mu, seed, run_dir in jobs:
            futures.append(ex.submit(
                run_one_engine,
                script_dir, args.nodes, seed, mu, ptri, args.p_triadic_param_index, run_dir,
                args.step_interval, args.max_iters,
                args.stability_window, args.stability_tol,
                args.stability_repeat, args.stability_k_min,
                args.delta_k_snapshot, args.max_spread_snapshots,
                args.flicker_window_ticks, args.flicker_bin,
                args.flicker_min_bin_changes, args.flicker_min_range,
                args.frozen_sparse_ticks, args.frozen_kmin,
                args.completion_kmin_frac, args.no_new_high_ticks, args.new_high_eps,
                args.compute_dimensions, dim_cfg,
                progress_q,
            ))

        try:
            for fut in as_completed(futures):
                if stop_main["stop"]:
                    break

                r = fut.result()
                done += 1
                wall_times.append(r["elapsed_sec"] if isinstance(r["elapsed_sec"], (int, float)) else 0.0)

                avg = sum(wall_times) / max(1, len(wall_times))
                eta = (total - done) * avg / max(1, args.workers)

                # append to shared summary (plot thread watches this)
                r2 = dict(r)
                r2["batch_id"] = batch_id
                append_summary_row(summary_csv, r2)

                k = r.get("kmean")
                k_str = f"{k:.10g}" if isinstance(k, float) and math.isfinite(k) else "?"
                ips = r.get("iters_per_sec")
                ips_str = f"{ips:.3g}" if isinstance(ips, float) and math.isfinite(ips) else "?"
                dim_part = ""
                if r.get("dim_status") == "OK":
                    dH = r.get("d_H")
                    dS = r.get("d_S")
                    dH_str = f"{dH:.5g}" if isinstance(dH, float) and math.isfinite(dH) else "?"
                    dS_str = f"{dS:.5g}" if isinstance(dS, float) and math.isfinite(dS) else "?"
                    dim_part = f" dH={dH_str} dS={dS_str}"

                print(
                    f"[{done:>4}/{total}] mu={r['mu']:.4f} seed={r['seed']} "
                    f"status={r['status']} kmean={k_str}{dim_part} "
                    f"iter={r['iter']:,} ips={ips_str} kept={r.get('kept_snapshots','?')} "
                    f"| ETA~ {hms(eta)}",
                    flush=True
                )

        finally:
            # stop plotter
            stop_plot.set()
            if plot_thread is not None:
                plot_thread.join(timeout=2.0)

    total_wall = time.time() - started
    print(f"\nBatch complete (or interrupted).")
    print(f"Summary (append-only): {summary_csv}")
    if args.live_plot:
        print(f"Dashboard PNG: {out_png}")
    print(f"Total wall time: {total_wall:.1f}s")


if __name__ == "__main__":
    main()
