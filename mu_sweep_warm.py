#!/usr/bin/env python3
"""mu_sweep_bidirectional_warmstart.py

Bidirectional (outward-from-middle) μ sweep with warm-started graphs.

What it does
- Chooses a mid-point μ within [mu_low, mu_high] (default midpoint).
- For each seed, runs ONE "bootstrap" relaxation at μ_mid.
- Clones that relaxed engine state into two branches:
    * upward branch: μ increases toward mu_high
    * downward branch: μ decreases toward mu_low

Outputs
- Same directory + CSV conventions as mu_sweep.py
- Live plot includes kmean, d_S, d_H, assortativity, and transitivity.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import multiprocessing as mp
import threading
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# -------------------------
# Defaults
# -------------------------
DEFAULT_NODES = 10001
DEFAULT_MU_LOW =  0.00001
DEFAULT_MU_HIGH = 0.10000
DEFAULT_MU_STEP = 0.00010
DEFAULT_SEEDS = 3
DEFAULT_BASE_SEED = 42

DEFAULT_PTRIADIC_VALUES = "0.99"
DEFAULT_PTRIADIC_PARAM_INDEX = 4

DEFAULT_STEP_INTERVAL = 100_000
DEFAULT_MAX_ITERS = 1_000_000_000

DEFAULT_COMPLETION_KMIN_FRAC = 0.25
DEFAULT_NO_NEW_HIGH_TICKS = 800
DEFAULT_NEW_HIGH_EPS = 1e-4

DEFAULT_COMPUTE_DIMENSIONS = True
# Kept as internal constants to declutter the CLI
DEFAULT_HAUSDORFF_SAMPLES = 50
DEFAULT_SPECTRAL_REPS = 3
DEFAULT_SPECTRAL_WALKERS = 15000
DEFAULT_SPECTRAL_WALK_LENGTH = 120
DEFAULT_SPECTRAL_MIN_FIT_POINTS = 5
DEFAULT_SPECTRAL_PLATEAU_MULT = 15.0

DEFAULT_OUT_ROOT = "mu_sweep"
DEFAULT_AGG_INTERVAL = 4.0

DEFAULT_LIVE_PLOT = True
DEFAULT_LIVE_PLOT_FILENAME = "live.png"

# Temperature (kept fixed for now; stored everywhere so we can anneal later)
DEFAULT_TEMPERATURE = 0.0

# Ricci (approx) defaults
DEFAULT_COMPUTE_RICCI = True
DEFAULT_RICCI_EDGE_SAMPLES = 100
DEFAULT_RICCI_ALPHA = 0.0
DEFAULT_RICCI_CUTOFF = 10


# -------------------------
# Helpers
# -------------------------

def safe_mu(mu: float) -> str:
    return f"MU{mu:.4f}".replace(".", "p")

def safe_ptriadic(p: float | None) -> str:
    if p is None:
        return "PTRIunset"
    return f"PTRI{p:.6f}".replace(".", "p")

def parse_ptriadic_values(s: str | None) -> list[float | None]:
    if not s:
        return [None]
    vals: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals if vals else [None]

def set_engine_param(engine: Any, *, temperature: float | None = None,
                     mu: float | None = None,
                     p_triadic: float | None = None,
                     p_triadic_param_index: int = DEFAULT_PTRIADIC_PARAM_INDEX) -> None:
    # Temperature is expected at params[0] (per user convention).
    if temperature is not None:
        try:
            engine.params[0] = float(temperature)
        except Exception:
            try:
                setattr(engine, "temperature", float(temperature))
            except Exception:
                pass

    if mu is not None:
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
    for attr in ("p_triadic", "pTriadic", "ptriadic"):
        try:
            if hasattr(engine, attr):
                setattr(engine, attr, v)
                return
        except Exception:
            pass

    try:
        if hasattr(engine, "params") and isinstance(engine.params, dict):
            for key in ("p_triadic", "pTriadic", "ptriadic"):
                if key in engine.params:
                    engine.params[key] = v
                    return
    except Exception:
        pass

    try:
        engine.params[int(p_triadic_param_index)] = v
    except Exception:
        pass

def fmt_iter(i: int) -> str:
    return f"{int(i):010d}"


def _single_item_queue(ctx: mp.context.BaseContext, item):
    """Create a ctx.Queue preloaded with a single item.

    Used for the "2 processes per seed" layout (one up + one down).
    IMPORTANT: the queue must be created from the SAME mp context (fork)
    to avoid SemLock/_rebuild issues.
    """
    q = ctx.Queue()
    q.put(item)
    return q

def get_k_stats(G):
    import networkx as nx

    degrees = [d for _, d in G.degree()]
    if not degrees:
        return 0, float("nan"), 0, 0, 0, 0.0, 0.0, float("nan")

    k_min = min(degrees)
    k_max = max(degrees)
    k_avg = sum(degrees) / len(degrees)

    tri = sum(nx.triangles(G).values()) // 3 if G.number_of_nodes() > 0 else 0
    m = int(G.number_of_edges())
    tri_per_edge = (float(tri) / m) if m > 0 else 0.0

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

def export_snapshot_csv(G, run_dir: Path, iter_num: int) -> None:
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

def _add_order_parameter(df, xaxis: str):
    """Add/return an x-axis series for plotting.

    xaxis options:
      - mu
      - transitivity
      - tri_per_edge
      - kmean
      - ollivier_ricci
      - pca1 : PCA( kmean, d_S, d_H, assortativity, transitivity, ollivier_ricci )
    """
    import numpy as np
    import pandas as pd

    xaxis = str(xaxis).strip().lower()

    if xaxis in ("mu", "transitivity", "tri_per_edge", "kmean", "ollivier_ricci"):
        if xaxis not in df.columns:
            raise KeyError(f"Requested xaxis='{xaxis}' not in CSV columns")
        return df[xaxis], xaxis

    if xaxis == "pca1":
        feats = ["kmean", "d_S", "d_H", "assortativity", "transitivity", "ollivier_ricci"]
        have = [c for c in feats if c in df.columns]
        if len(have) < 3:
            raise ValueError(f"pca1 needs >=3 features; have {have}")

        X = df[have].copy()
        for c in have:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.dropna()
        if X.empty:
            raise ValueError("pca1: no complete rows")

        # z-score
        Z = X.values.astype(float)
        mu = np.nanmean(Z, axis=0)
        sd = np.nanstd(Z, axis=0)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd

        # first principal component via SVD
        # Z = U S Vt, PC1 scores = U[:,0]*S[0]
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        scores = U[:, 0] * S[0]

        # map back into df index
        df = df.copy()
        df.loc[X.index, "pca1"] = scores
        return df["pca1"], "pca1"

    raise ValueError(f"Unknown xaxis: {xaxis}")


def update_live_plot(summary_csv: Path, out_png: Path, *,
                     title: str = "Warm bidirectional μ sweep",
                     xaxis: str = "mu") -> None:
    """Render the 2x3 live dashboard.

    Note: You still *sweep μ* to generate the states. Changing xaxis here just
    re-parameterizes the visualization (and later analysis) using the same data.
    """
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not summary_csv.exists() or summary_csv.stat().st_size == 0:
            return

        df = pd.read_csv(summary_csv)
        if df.empty:
            return

        # Ensure numeric columns we use
        cols_to_num = [
            "mu", "temperature", "kmean", "d_S", "d_H", "assortativity", "transitivity",
            "ollivier_ricci", "tri_per_edge", "nodes"
        ]
        for col in cols_to_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # keep only finished runs
        if "status" in df.columns:
            df2 = df[df["status"].isin(["COMPLETED", "LOOP_DETECTED", "MAX_ITERS_REACHED"])]
            if not df2.empty:
                df = df2

        # need kmean for most plots
        df = df.dropna(subset=["kmean"])
        if df.empty:
            return

        # x-axis
        try:
            x, xname = _add_order_parameter(df, xaxis)
            df = df.copy()
            df["_x_"] = x
        except Exception as e:
            # fallback to μ
            df = df.dropna(subset=["mu"])
            if df.empty:
                return
            df["_x_"] = df["mu"]
            xname = "mu"
            print(f"[plot] xaxis='{xaxis}' unavailable ({e}); falling back to μ", flush=True)

        df = df.dropna(subset=["_x_"])
        if df.empty:
            return

        N_val = int(df["nodes"].dropna().iloc[0]) if "nodes" in df.columns and not df["nodes"].dropna().empty else "Unknown"
        T_val = None
        if "temperature" in df.columns and not df["temperature"].dropna().empty:
            T_val = float(df["temperature"].dropna().iloc[0])

        if "branch" not in df.columns:
            df["branch"] = "all"
        branches = df["branch"].dropna().unique().tolist()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax_k, ax_dS, ax_dH, ax_assort, ax_trans, ax_ricci = axes.flatten()

        def plot_metric(ax, metric, y_label, y_scale="linear"):
            if metric in df.columns and df[metric].notna().any():
                for b in branches:
                    d = df[df["branch"] == b].dropna(subset=[metric, "_x_"])
                    if not d.empty:
                        ax.scatter(d["_x_"], d[metric], s=12, alpha=0.7, label=str(b))
                ax.set_ylabel(y_label)
                if y_scale == "log":
                    ax.set_yscale("log")
            else:
                ax.text(0.5, 0.5, f"{metric} not available", ha="center", va="center")

            ax.set_title(f"{y_label} vs {xname}")
            ax.set_xlabel(xname)
            ax.grid(True, alpha=0.3)

        plot_metric(ax_k, "kmean", "Mean degree ⟨k⟩", y_scale="log")
        plot_metric(ax_dS, "d_S", "Spectral dimension d_S")
        plot_metric(ax_dH, "d_H", "Hausdorff dimension d_H")
        plot_metric(ax_assort, "assortativity", "Assortativity")
        plot_metric(ax_trans, "transitivity", "Transitivity")
        plot_metric(ax_ricci, "ollivier_ricci", "Ollivier-Ricci (median κ)")

        ax_k.legend(loc="best", fontsize=9)

        if T_val is None:
            fig.suptitle(f"{title} | N={N_val}", fontsize=16)
        else:
            fig.suptitle(f"{title} | N={N_val} | T={T_val:g}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting error: {e}")
        pass

SUMMARY_FIELDS = [
    "batch_id", "mu", "temperature", "p_triadic", "seed", "nodes", "status", "iter",
    "elapsed_sec", "iters_per_sec", "kmean", "triangles", "edges",
    "tri_per_edge", "transitivity", "assortativity", "note", "dim_status",
    "dim_note", "d_H", "d_S", "d_S_std", "d_S_fit_pts", "d_S_R",
    "d_S_Window", "d_3D", "fully_computed", "kept_snapshots", "run_dir",
    "branch", "mu_mid",

    # approximate curvature summary
    "ollivier_ricci",
]

def ensure_summary_header(summary_csv: Path) -> None:
    if summary_csv.exists() and summary_csv.stat().st_size > 0:
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()

def append_summary_row(summary_csv: Path, row: dict) -> None:
    with summary_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

@dataclass
class DimCfg:
    base_seed: int
    hausdorff_samples: int
    spectral_reps: int
    spectral_walkers: int
    spectral_walk_length: int
    spectral_min_fit_points: int
    spectral_plateau_mult: float

@dataclass
class RunCfg:
    nodes: int
    temperature: float
    step_interval: int
    max_iters: int
    completion_kmin_frac: float
    no_new_high_ticks: int
    new_high_eps: float
    compute_dimensions: bool
    compute_ricci: bool
    ricci_edge_samples: int
    ricci_alpha: float
    ricci_cutoff: int
    dim_cfg: DimCfg


# -------------------------
# Ollivier-Ricci (approx)
# -------------------------

def _w1_transport_cost(cost_mat, a, b):
    """Compute Wasserstein-1 (Earth Mover's) cost between two discrete measures.

    Uses linear programming (HiGHS) from SciPy. Intended for *small* supports.
    """
    import numpy as np
    from scipy.optimize import linprog

    cost_mat = np.asarray(cost_mat, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = cost_mat.shape

    c = cost_mat.reshape(-1)

    A_eq = []
    b_eq = []

    for i in range(na):
        row = np.zeros(na * nb)
        row[i * nb:(i + 1) * nb] = 1.0
        A_eq.append(row)
        b_eq.append(a[i])

    for j in range(nb):
        col = np.zeros(na * nb)
        col[j::nb] = 1.0
        A_eq.append(col)
        b_eq.append(b[j])

    bounds = [(0.0, None)] * (na * nb)
    res = linprog(c, A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return float(res.fun)


def compute_ollivier_ricci_mean(G, *, edge_samples: int, alpha: float, cutoff: int, base_seed: int) -> float:
    """Approximate mean Ollivier-Ricci curvature by sampling edges.

    κ(x,y) = 1 - W1(m_x, m_y) / d(x,y). For adjacent nodes d=1.
    m_x is (1-α) uniform on neighbors of x plus α mass on x.

    Notes
    - This is an approximation: exact OR on large graphs is expensive.
    - Uses global shortest-path distances but with a cutoff; distances beyond cutoff
      are capped at (cutoff + 1).
    """
    import random
    import numpy as np
    import networkx as nx

    if G.number_of_edges() == 0:
        return float("nan")

    rnd = random.Random(int(base_seed))
    edges = list(G.edges())
    if not edges:
        return float("nan")

    k = min(int(edge_samples), len(edges))
    sampled = rnd.sample(edges, k)

    kappas = []
    cap_dist = int(cutoff) + 1

    for x, y in sampled:
        Nx = list(G.neighbors(x))
        Ny = list(G.neighbors(y))
        if not Nx or not Ny:
            continue

        supp_x = ([x] if alpha > 0 else []) + Nx
        supp_y = ([y] if alpha > 0 else []) + Ny

        ax = ([] if alpha <= 0 else [alpha]) + [(1.0 - alpha) / len(Nx)] * len(Nx)
        by = ([] if alpha <= 0 else [alpha]) + [(1.0 - alpha) / len(Ny)] * len(Ny)

        D = np.zeros((len(supp_x), len(supp_y)), dtype=float)
        for i, u in enumerate(supp_x):
            try:
                dist_map = nx.single_source_shortest_path_length(G, u, cutoff=int(cutoff))
            except Exception:
                dist_map = {}
            for j, v in enumerate(supp_y):
                D[i, j] = float(dist_map.get(v, cap_dist))

        try:
            W1 = _w1_transport_cost(D, ax, by)
            kappa = 1.0 - W1  # since d(x,y)=1
            if math.isfinite(kappa):
                kappas.append(float(kappa))
        except Exception:
            continue

    return float(np.median(kappas)) if kappas else float("nan")

# -------------------------
# Engine cloning + warm-start
# -------------------------

def clone_engine(engine: Any, *, mode: str, nodes: int, seed: int) -> Any:
    if mode == "deepcopy":
        try:
            return copy.deepcopy(engine)
        except Exception:
            mode = "graph_only"

    if mode == "graph_only":
        from engine import PhysicsEngine
        e2 = PhysicsEngine(nodes, seed)
        try:
            e2.G = engine.G.copy()
        except Exception:
            import networkx as nx
            G2 = nx.Graph()
            G2.add_nodes_from(engine.G.nodes(data=True))
            G2.add_edges_from(engine.G.edges())
            e2.G = G2

        try:
            if hasattr(engine, "params") and hasattr(e2, "params"):
                try:
                    e2.params = copy.deepcopy(engine.params)
                except Exception:
                    pass
        except Exception:
            pass

        for attr in ("rng", "random", "_rng"):
            if hasattr(engine, attr) and hasattr(e2, attr):
                try:
                    setattr(e2, attr, copy.deepcopy(getattr(engine, attr)))
                except Exception:
                    pass
        return e2

    raise ValueError(f"Unknown clone mode: {mode}")

# -------------------------
# Single run
# -------------------------

def run_at_mu(*, script_dir: Path, engine: Any, run_dir: Path, seed: int,
              temperature: float,
              mu: float, p_triadic: float | None, p_triadic_param_index: int,
              run_cfg: RunCfg, branch: str, mu_mid: float,
              reset_between_mu: bool = False,
              console_kmean_jump: float | None = None,
              console_tag: str | None = None) -> dict:

    import datetime
    from engine import PhysicsEngine

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.csv"

    dim_mod = load_dimensions_module(script_dir) if run_cfg.compute_dimensions else None

    if reset_between_mu:
        engine = PhysicsEngine(run_cfg.nodes, seed)

    set_engine_param(engine, temperature=temperature, mu=mu, p_triadic=p_triadic,
                    p_triadic_param_index=p_triadic_param_index)

    fieldnames = [
        "timestamp", "iter", "temperature", "elapsed_time", "k_min", "k_avg", "k_max", "triangles",
        "edges", "tri_per_edge", "transitivity", "assortativity", "iters_per_sec",
        "status", "note", "dim_status", "dim_note", "d_H", "d_S", "d_S_std",
        "d_S_fit_pts", "d_S_R", "d_S_Window", "d_3D",
    ]

    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    try:
        export_snapshot_csv(engine.G, run_dir, 0)
    except Exception:
        pass

    t0 = time.time()
    final_iter = 0
    status = ""
    note = ""
    best_kavg = None
    ticks_since_high = 0

    # For console kmean "jump" logging (warmup)
    last_bucket = None
    jump = float(console_kmean_jump) if (console_kmean_jump is not None and console_kmean_jump > 0) else None

    for it in range(1, int(run_cfg.max_iters) + 1):
        engine.iterate()

        if it % int(run_cfg.step_interval) != 0:
            continue

        final_iter = it
        k_min, k_avg, k_max, triangles, edges, tri_per_edge, transitivity, assortativity = get_k_stats(engine.G)
        elapsed = time.time() - t0
        ips = (it / elapsed) if elapsed > 0 else None

        if jump is not None and math.isfinite(k_avg):
            bucket = int(math.floor(float(k_avg) / jump))
            if last_bucket is None:
                last_bucket = bucket
            elif bucket != last_bucket:
                last_bucket = bucket
                tag = console_tag or branch
                print(f"[{tag}] kmean crossed {bucket * jump:.2f} at iter={it} (μ={mu:.5f} seed={seed} kmean={k_avg:.6g})", flush=True)

        if math.isfinite(k_avg) and k_avg > 0 and k_min >= (run_cfg.completion_kmin_frac * k_avg):
            status = "COMPLETED"
            note = f"k_min>={run_cfg.completion_kmin_frac:g}*kmean"
            break

        if math.isfinite(k_avg):
            if best_kavg is None:
                best_kavg = k_avg
                ticks_since_high = 0
            else:
                rel_eps = run_cfg.new_high_eps * max(1.0, abs(best_kavg))
                if k_avg > best_kavg + rel_eps:
                    best_kavg = k_avg
                    ticks_since_high = 0
                else:
                    ticks_since_high += 1

            if ticks_since_high >= run_cfg.no_new_high_ticks:
                status = "LOOP_DETECTED"
                note = f"no new kmean highs for {ticks_since_high} ticks (best={best_kavg:.6g})"
                break

        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow({
                "timestamp": datetime.datetime.now().isoformat(), "iter": it,
                "temperature": float(temperature),
                "elapsed_time": elapsed, "k_min": k_min, "k_avg": k_avg, "k_max": k_max,
                "triangles": triangles, "edges": edges, "tri_per_edge": tri_per_edge,
                "transitivity": transitivity, "assortativity": assortativity,
                "iters_per_sec": ips if ips is not None else "", "status": "",
                "note": "", "dim_status": "", "dim_note": "", "d_H": "", "d_S": "",
                "d_S_std": "", "d_S_fit_pts": "", "d_S_R": "", "d_S_Window": "", "d_3D": "",
            })

    if not status:
        status = "MAX_ITERS_REACHED"
        note = f"Hit max_iters={run_cfg.max_iters}"

    k_min, k_avg, k_max, triangles, edges, tri_per_edge, transitivity, assortativity = get_k_stats(engine.G)
    elapsed = time.time() - t0
    ips = (final_iter / elapsed) if elapsed > 0 else None

    dH = dS = dS_std = R_mean = d3 = fit_pts = window_str = None
    dim_status = dim_note = ""

    # Approximate mean Ollivier-Ricci curvature (sampled)
    ricci_mean = None
    ricci_note = ""
    if getattr(run_cfg, "compute_ricci", False):
        try:
            ricci_mean = compute_ollivier_ricci_mean(
                engine.G,
                edge_samples=int(run_cfg.ricci_edge_samples),
                alpha=float(run_cfg.ricci_alpha),
                cutoff=int(run_cfg.ricci_cutoff),
                base_seed=int(run_cfg.dim_cfg.base_seed) + 7919,
            )
        except Exception as e:
            ricci_mean = float("nan")
            ricci_note = (str(e) or "ricci calc error")[:160]

    if run_cfg.compute_dimensions and dim_mod is not None:
        try:
            dH, dS, dS_std, fit_pts, R_mean, window_str, d3 = compute_dimensions_from_graph(
                dim_mod, engine.G,
                base_seed=run_cfg.dim_cfg.base_seed,
                hausdorff_samples=run_cfg.dim_cfg.hausdorff_samples,
                spectral_reps=run_cfg.dim_cfg.spectral_reps,
                spectral_walkers=run_cfg.dim_cfg.spectral_walkers,
                spectral_walk_length=run_cfg.dim_cfg.spectral_walk_length,
                spectral_min_fit_points=run_cfg.dim_cfg.spectral_min_fit_points,
                spectral_plateau_mult=run_cfg.dim_cfg.spectral_plateau_mult,
            )
            dim_status = "OK"
        except Exception as e:
            dim_status = "FAILED"
            dim_note = (str(e) or "dimension calc error")[:160]
            print(f"\n[!] DIMENSION CALC CRASHED: {e}\n")

    fully_computed = (status == "COMPLETED" and dim_status == "OK")

    with log_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "iter": final_iter,
            "temperature": float(temperature),
            "elapsed_time": elapsed, "k_min": k_min, "k_avg": k_avg, "k_max": k_max,
            "triangles": triangles, "edges": edges, "tri_per_edge": tri_per_edge,
            "transitivity": transitivity, "assortativity": assortativity,
            "iters_per_sec": ips if ips is not None else "", "status": status,
            "note": note, "dim_status": dim_status, "dim_note": dim_note,
            "d_H": "" if dH is None else dH, "d_S": "" if dS is None else dS,
            "d_S_std": "" if dS_std is None else dS_std, "d_S_fit_pts": "" if fit_pts is None else fit_pts,
            "d_S_R": "" if R_mean is None else R_mean, "d_S_Window": "" if window_str is None else window_str,
            "d_3D": "" if d3 is None else d3,
        })

    return {
        "mu": float(mu), "temperature": float(temperature), "mu_mid": float(mu_mid), "branch": branch, "p_triadic": p_triadic,
        "seed": int(seed), "nodes": int(run_cfg.nodes), "status": status,
        "iter": int(final_iter), "elapsed_sec": float(elapsed),
        "iters_per_sec": (final_iter / elapsed) if elapsed > 0 else None,
        "kmean": float(k_avg) if math.isfinite(k_avg) else float("nan"),
        "triangles": int(triangles), "edges": int(edges),
        "tri_per_edge": float(tri_per_edge), "transitivity": float(transitivity),
        "assortativity": float(assortativity) if math.isfinite(assortativity) else float("nan"),
        "note": note, "dim_status": dim_status, "dim_note": dim_note,
        "d_H": dH, "d_S": dS, "d_S_std": dS_std, "d_S_fit_pts": fit_pts,
        "d_S_R": R_mean, "d_S_Window": window_str, "d_3D": d3,
        "fully_computed": fully_computed, "kept_snapshots": 1, "run_dir": str(run_dir),
        "ollivier_ricci": ricci_mean,
    }, engine

# -------------------------
# Thread workers
# -------------------------

def build_mu_lists(mu_low: float, mu_high: float, mu_step: float, mu_mid: float | None) -> tuple[float, list[float], list[float]]:
    if mu_step <= 0:
        raise ValueError("mu_step must be > 0")
    if mu_low > mu_high:
        mu_low, mu_high = mu_high, mu_low

    if mu_mid is None:
        mu_mid = 0.5 * (mu_low + mu_high)

    n_steps = round((mu_mid - mu_low) / mu_step)
    mu_mid = mu_low + n_steps * mu_step
    mu_mid = min(max(mu_mid, mu_low), mu_high)

    up = []
    mu = mu_mid
    while mu <= mu_high + 1e-12:
        up.append(round(mu, 4))
        mu += mu_step

    down = []
    mu = mu_mid - mu_step
    while mu >= mu_low - 1e-12:
        down.append(round(mu, 4))
        mu -= mu_step

    return float(mu_mid), up, down

def worker_branch(name: str, q_in: "queue.Queue[tuple[int, Any]]", result_q: "queue.Queue[dict]",
                  script_dir: Path, batch_root: Path, p_triadic: float | None,
                  p_triadic_param_index: int, mu_list: list[float], run_cfg: RunCfg,
                  mu_mid: float, reset_between_mu: bool, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            seed, engine = q_in.get(timeout=0.25)
        except queue.Empty:
            return

        for mu in mu_list:
            if stop_evt.is_set():
                break

            print(f"[{name}] START N={run_cfg.nodes} seed={seed} μ={mu:.5f} ptri={p_triadic if p_triadic is not None else 'unset'}", flush=True)

            run_dir = batch_root / safe_ptriadic(p_triadic) / safe_mu(mu) / f"N{run_cfg.nodes}" / f"S{seed}"

            row, engine = run_at_mu(
                script_dir=script_dir, engine=engine, run_dir=run_dir, seed=seed,
                temperature=float(run_cfg.temperature),
                mu=mu, p_triadic=p_triadic, p_triadic_param_index=p_triadic_param_index,
                run_cfg=run_cfg, branch=name, mu_mid=mu_mid, reset_between_mu=reset_between_mu,
            )

            print(f"[{name}] DONE  N={run_cfg.nodes} seed={seed} μ={mu:.5f} status={row.get('status')} "
                  f"iter={row.get('iter')} kmean={row.get('kmean'):.6g} "
                  f"dS={row.get('d_S') if row.get('d_S') is not None else ''} "
                  f"elapsed={row.get('elapsed_sec'):.2f}s", flush=True)

            result_q.put(row)

        q_in.task_done()


def warmup_worker(seed: int, nodes: int, mu_mid: float, ptri: float | None,
                  p_triadic_param_index: int, script_dir: Path, batch_root: Path,
                  run_cfg: RunCfg, clone_mode: str, reset_between_mu: bool,
                  result_q, warmed_q, kmean_jump: float, stop_evt) -> None:
    """Run the mid warmup for one seed and return the warmed engine."""
    if stop_evt.is_set():
        return

    from engine import PhysicsEngine

    eng = PhysicsEngine(nodes, seed)
    set_engine_param(eng, temperature=float(run_cfg.temperature), mu=mu_mid, p_triadic=ptri,
                    p_triadic_param_index=p_triadic_param_index)
    run_dir_mid = batch_root / safe_ptriadic(ptri) / safe_mu(mu_mid) / f"N{nodes}" / f"S{seed}"

    tag = f"warm{seed}"
    print(f"[{tag}] START N={nodes} seed={seed} μ={mu_mid:.5f} ptri={ptri if ptri is not None else 'unset'}", flush=True)
    row_mid, eng = run_at_mu(
        script_dir=script_dir, engine=eng, run_dir=run_dir_mid, seed=seed,
        temperature=float(run_cfg.temperature),
        mu=mu_mid, p_triadic=ptri, p_triadic_param_index=p_triadic_param_index,
        run_cfg=run_cfg, branch="mid", mu_mid=mu_mid, reset_between_mu=reset_between_mu,
        console_kmean_jump=float(kmean_jump), console_tag=tag,
    )
    result_q.put(row_mid)
    print(f"[{tag}] DONE  N={nodes} seed={seed} μ={mu_mid:.5f} status={row_mid.get('status')} "
          f"iter={row_mid.get('iter')} kmean={row_mid.get('kmean'):.6g} "
          f"ricci={row_mid.get('ollivier_ricci') if row_mid.get('ollivier_ricci') is not None else ''} "
          f"dS={row_mid.get('d_S') if row_mid.get('d_S') is not None else ''} "
          f"elapsed={row_mid.get('elapsed_sec'):.2f}s", flush=True)

    # Return warmed engine for branching
    warmed_q.put((seed, eng))

def aggregator_loop(summary_csv: Path, result_q: "queue.Queue[dict]", batch_id: str,
                    stop_evt: threading.Event, interval_sec: float, total_expected: int | None = None,
                    live_plot: bool = True, live_plot_path: Path | None = None,
                    plot_xaxis: str = "mu"):
    ensure_summary_header(summary_csv)
    done = 0
    last_print = time.time()

    while not stop_evt.is_set() or not result_q.empty():
        rows = []
        t_deadline = time.time() + max(0.1, interval_sec)

        while time.time() < t_deadline:
            try:
                rows.append(result_q.get_nowait())
            except queue.Empty:
                break

        if rows:
            for r in rows:
                r2 = dict(r)
                r2["batch_id"] = batch_id
                append_summary_row(summary_csv, r2)
                done += 1

            if live_plot and live_plot_path is not None:
                update_live_plot(summary_csv, live_plot_path, xaxis=plot_xaxis)

        now = time.time()
        if now - last_print >= max(1.0, interval_sec):
            if total_expected is not None:
                print(f"[agg] wrote {done}/{total_expected} completed μ-runs | queue={result_q.qsize()}", flush=True)
            else:
                print(f"[agg] wrote {done} completed μ-runs | queue={result_q.qsize()}", flush=True)
            last_print = now

        stop_evt.wait(timeout=0.25)

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Core parameters kept for CLI
    ap.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="Fixed temperature to set on the engine (stored in CSVs for future annealing).")
    ap.add_argument("--mu-low", type=float, default=DEFAULT_MU_LOW)
    ap.add_argument("--mu-high", type=float, default=DEFAULT_MU_HIGH)
    ap.add_argument("--mu-step", type=float, default=DEFAULT_MU_STEP)
    ap.add_argument("--mu-mid", type=float, default=None,
                    help="Optional explicit midpoint μ. If omitted, midpoint of [mu-low, mu-high], snapped to grid.")

    ap.add_argument("--p-triadic-values", type=str, default=DEFAULT_PTRIADIC_VALUES)
    ap.add_argument("--p-triadic-param-index", type=int, default=DEFAULT_PTRIADIC_PARAM_INDEX)
    ap.add_argument("--seeds-per-mu", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)

    ap.add_argument("--step-interval", type=int, default=DEFAULT_STEP_INTERVAL)
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    ap.add_argument("--completion-kmin-frac", type=float, default=DEFAULT_COMPLETION_KMIN_FRAC)
    ap.add_argument("--no-new-high-ticks", type=int, default=DEFAULT_NO_NEW_HIGH_TICKS)
    ap.add_argument("--new-high-eps", type=float, default=DEFAULT_NEW_HIGH_EPS)

    ap.add_argument("--compute-dimensions", action="store_true", default=DEFAULT_COMPUTE_DIMENSIONS)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--agg-interval", type=float, default=DEFAULT_AGG_INTERVAL)

    ap.add_argument("--plot-xaxis", type=str, default="mu", choices=["mu","transitivity","tri_per_edge","kmean","ollivier_ricci","pca1"],
                    help="X-axis for live plots (data is still generated by sweeping μ).")

    ap.add_argument("--clone-mode", type=str, default="deepcopy", choices=["deepcopy", "graph_only"])
    ap.add_argument("--reset-between-mu", action="store_true", default=False)

    ap.add_argument("--warmup-kmean-jump", type=float, default=0.01,
                    help="During warmup (μ_mid), print when kmean crosses multiples of this value (0 disables).")

    ap.add_argument("--compute-ricci", action="store_true", default=DEFAULT_COMPUTE_RICCI)
    ap.add_argument("--ricci-edge-samples", type=int, default=DEFAULT_RICCI_EDGE_SAMPLES)
    ap.add_argument("--ricci-alpha", type=float, default=DEFAULT_RICCI_ALPHA)
    ap.add_argument("--ricci-cutoff", type=int, default=DEFAULT_RICCI_CUTOFF)

    args = ap.parse_args()
    # Robust multiprocessing context: avoid SemLock/_rebuild issues under heavy churn.
    # On Linux, 'fork' is the most stable choice for this workload.
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context("fork")


    batch_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out)
    batch_root = out_root / f"N{args.nodes}_{batch_id}"
    batch_root.mkdir(parents=True, exist_ok=True)
    summary_csv = batch_root / "sweep_summary_append.csv"
    ensure_summary_header(summary_csv)

    script_dir = Path(__file__).resolve().parent
    mu_mid, mus_up, mus_down = build_mu_lists(args.mu_low, args.mu_high, args.mu_step, args.mu_mid)

    seeds = [int(args.base_seed + i) for i in range(int(args.seeds_per_mu))]
    ptri_values = parse_ptriadic_values(args.p_triadic_values)

    # DimCfg now relies on internal constants rather than CLI args
    dim_cfg = DimCfg(
        base_seed=int(args.base_seed),
        hausdorff_samples=DEFAULT_HAUSDORFF_SAMPLES,
        spectral_reps=DEFAULT_SPECTRAL_REPS,
        spectral_walkers=DEFAULT_SPECTRAL_WALKERS,
        spectral_walk_length=DEFAULT_SPECTRAL_WALK_LENGTH,
        spectral_min_fit_points=DEFAULT_SPECTRAL_MIN_FIT_POINTS,
        spectral_plateau_mult=DEFAULT_SPECTRAL_PLATEAU_MULT,
    )

    run_cfg = RunCfg(
        nodes=int(args.nodes),
        temperature=float(args.temperature),
        step_interval=int(args.step_interval),
        max_iters=int(args.max_iters),
        completion_kmin_frac=float(args.completion_kmin_frac),
        no_new_high_ticks=int(args.no_new_high_ticks),
        new_high_eps=float(args.new_high_eps),
        compute_dimensions=bool(args.compute_dimensions),
        compute_ricci=bool(args.compute_ricci),
        ricci_edge_samples=int(args.ricci_edge_samples),
        ricci_alpha=float(args.ricci_alpha),
        ricci_cutoff=int(args.ricci_cutoff),
        dim_cfg=dim_cfg,
    )

    print(f"\nBatch: {batch_id}")
    print(f"System Size (N): {run_cfg.nodes}")
    print(f"Output: {batch_root}")
    print(f"μ range: [{min(args.mu_low, args.mu_high):.5f}, {max(args.mu_low, args.mu_high):.5f}] step={args.mu_step:.5f}")
    print(f"μ_mid (snapped): {mu_mid:.5f}")
    print(f"Warmup: all seeds concurrent; kmean jump logging={args.warmup_kmean_jump:g}")
    print(f"After warmup: 2 processes per seed (one up, one down)")
    print(f"Warm-start across μ: {'OFF' if args.reset_between_mu else 'ON'}")
    print(f"Ricci: {'ON' if run_cfg.compute_ricci else 'OFF'} (edge_samples={run_cfg.ricci_edge_samples}, alpha={run_cfg.ricci_alpha:g}, cutoff={run_cfg.ricci_cutoff})")
    print("Ctrl-C to stop; completed rows are preserved.\n")

    result_q: mp.Queue = ctx.Queue()
    stop_evt = ctx.Event()

    expected_per_seed = len(mus_up) + len(mus_down)
    total_expected = len(ptri_values) * len(seeds) * expected_per_seed
    live_plot_path = batch_root / str(DEFAULT_LIVE_PLOT_FILENAME)

    agg_thr = ctx.Process(
        target=aggregator_loop,
        args=(summary_csv, result_q, batch_id, stop_evt, float(args.agg_interval),
              total_expected, True, live_plot_path, str(args.plot_xaxis)), daemon=False
    )
    agg_thr.start()

    try:
        for ptri in ptri_values:
            # --- Warmup: all seeds concurrently at μ_mid ---
            warmed_q: mp.Queue = ctx.Queue()
            warm_procs: list[mp.Process] = []
            for seed in seeds:
                p = ctx.Process(
                    target=warmup_worker,
                    args=(seed, run_cfg.nodes, mu_mid, ptri, args.p_triadic_param_index,
                          script_dir, batch_root, run_cfg, args.clone_mode, args.reset_between_mu,
                          result_q, warmed_q, float(args.warmup_kmean_jump), stop_evt),
                    daemon=False,
                )
                p.start()
                warm_procs.append(p)

            warmed: dict[int, Any] = {}
            while len(warmed) < len(seeds):
                try:
                    s, eng = warmed_q.get(timeout=0.5)
                    warmed[int(s)] = eng
                except Exception:
                    if not any(p.is_alive() for p in warm_procs) and warmed_q.empty():
                        break

            for p in warm_procs:
                p.join()

            if len(warmed) != len(seeds):
                missing = sorted(set(seeds) - set(warmed.keys()))
                print(f"[!] Warmup missing seeds: {missing}", flush=True)

            # --- Branching: 2 processes per seed (up + down) ---
            mus_up_branch = mus_up[1:]
            mus_down_branch = mus_down

            branch_procs: list[mp.Process] = []
            for seed, eng in warmed.items():
                eng_up = clone_engine(eng, mode=args.clone_mode, nodes=run_cfg.nodes, seed=seed)
                eng_down = clone_engine(eng, mode=args.clone_mode, nodes=run_cfg.nodes, seed=seed)

                p_up = ctx.Process(
                    target=worker_branch,
                    args=(f"up_s{seed}", _single_item_queue(ctx, (seed, eng_up)), result_q, script_dir, batch_root, ptri,
                          args.p_triadic_param_index, mus_up_branch, run_cfg, mu_mid, args.reset_between_mu, stop_evt),
                    daemon=False,
                )
                p_down = ctx.Process(
                    target=worker_branch,
                    args=(f"down_s{seed}", _single_item_queue(ctx, (seed, eng_down)), result_q, script_dir, batch_root, ptri,
                          args.p_triadic_param_index, mus_down_branch, run_cfg, mu_mid, args.reset_between_mu, stop_evt),
                    daemon=False,
                )
                p_up.start(); p_down.start()
                branch_procs.extend([p_up, p_down])

            while any(p.is_alive() for p in branch_procs):
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[CTRL-C] Stopping… preserving completed data.", flush=True)
    finally:
        stop_evt.set()
        agg_thr.join(timeout=5.0)

    print("\nDone.")
    print(f"Summary: {summary_csv}")

if __name__ == "__main__":
    main()
