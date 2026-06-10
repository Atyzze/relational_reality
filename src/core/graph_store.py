"""
core.graph_store — persistent cache of grown graphs (final thermalised state).

Thermalising a graph is the expensive step (thousands of sweeps x N Metropolis
moves); everything downstream (LCC + Laplacian, the d_s flow probe, the isotropy
probe) is cheap by comparison. This module saves the FINAL graph for a given
(k, T, lb, N, seed, mu, ec, max_degree) so any later analysis can reload it
instead of re-evolving it. build_graph() reads/writes this transparently, so a
sweep started from main.py populates the cache and the uniformity tool (and any
other probe) reuses it for free.

Format: one compressed .npz per graph, holding the graph as a compact CSR
neighbour list (degrees are recovered from the row pointers) plus a small JSON
metadata blob. No new dependency — numpy only, deliberately (the project keeps to
four packages). HDF5 would let many graphs share one file, but a directory of
.npz files is simpler, append-only, trivially rsync-able, and needs nothing
extra; the per-graph overhead is negligible next to the array data.

Layout:
    <root>/output/graphs/<k_T_lb>/N<N>_seed<seed>_md<max_degree>.npz
The directory is anchored to the project root (this file's location), NOT the
current working directory, so the sweep (which chdir's into output/) and a tool
run from the repo root share exactly one library.

Public API:
    save_graph(neighbors, degrees, *, k,T,lb,N,seed,max_degree,mu,ec,
               sweeps,peak_deg, therm_trace=None, k_avg=None) -> path
    load_graph(k,T,lb,N,seed,max_degree, *, mu=None) -> StoredGraph | None
    find_path(...) -> str | None
"""
from __future__ import annotations

import datetime
import json
import os
import socket

import numpy as np

try:                                            # canonical key helper (k_T_lb)
    from core.disk_io import mu_key
except Exception:                               # pragma: no cover - standalone fallback
    def mu_key(k, T, lb):
        return f"{k}_{T}_{lb}"

FORMAT_VERSION = 1
_MU_TOL = 1e-9                                   # stored mu must match requested mu this closely

# Project root = three levels up from this file (.../src/core/graph_store.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def store_dir() -> str:
    """Absolute graph-cache directory. Override with $GRAPH_STORE_DIR; otherwise
    <project_root>/output/graphs (independent of the current working directory)."""
    d = os.environ.get("GRAPH_STORE_DIR") or os.path.join(_ROOT, "output", "graphs")
    return os.path.abspath(d)


def _param_dir(k, T, lb) -> str:
    return os.path.join(store_dir(), mu_key(k, T, lb))


def path_for(k, T, lb, N, seed, max_degree) -> str:
    return os.path.join(_param_dir(k, T, lb),
                        f"N{int(N)}_seed{int(seed)}_md{int(max_degree)}.npz")


# ───────────────────────── (de)serialisation ─────────────────────────
def _to_csr(neighbors: np.ndarray, degrees: np.ndarray):
    """Pack an N x max_degree (-1 padded) neighbour table into compact CSR
    (indptr int64, indices int32). Robust to the -1 sentinel: selects exactly
    the first degrees[i] entries of each row."""
    N = int(degrees.shape[0])
    deg = degrees[:N].astype(np.int64)
    cols = np.arange(neighbors.shape[1])
    mask = cols[None, :] < deg[:, None]
    indices = neighbors[:N][mask].astype(np.int32)
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    return indptr, indices


def _from_csr(indptr: np.ndarray, indices: np.ndarray, max_degree: int):
    """Rebuild the N x max_degree (-1 padded) neighbour table + degrees array."""
    N = int(indptr.shape[0] - 1)
    deg = np.diff(indptr).astype(np.int32)
    neighbors = np.full((N, int(max_degree)), -1, dtype=np.int32)
    cols = np.arange(int(max_degree))
    mask = cols[None, :] < deg[:, None]
    neighbors[mask] = indices
    return neighbors, deg


class StoredGraph:
    """A graph loaded from the cache, shaped to stand in for a thermalised
    PhysicsEngine where the downstream code only needs the final arrays.
    build_graph() recomputes stats from these, so we don't serialise stats."""
    def __init__(self, neighbors, degrees, meta):
        self.node_neighbors = neighbors
        self.node_degrees = degrees
        self.meta = meta
        self.therm_trace = meta.get("_therm_trace")  # list[(sweep,k_avg,sumdsq)] or None

    @property
    def peak_degree(self) -> int:
        return int(self.meta.get("peak_deg", int(self.node_degrees.max()) if self.node_degrees.size else 0))


# ───────────────────────── save / load ─────────────────────────
def save_graph(neighbors, degrees, *, k, T, lb, N, seed, max_degree, mu, ec,
               sweeps, peak_deg, therm_trace=None, k_avg=None, engine=None,
               therm_info=None) -> str:
    """Persist the final graph. Atomic (writes to .tmp then renames)."""
    path = path_for(k, T, lb, N, seed, max_degree)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    indptr, indices = _to_csr(np.asarray(neighbors), np.asarray(degrees))
    meta = dict(fmt=FORMAT_VERSION, k=int(k), T=float(T), lb=float(lb), N=int(N),
                seed=int(seed), max_degree=int(max_degree), mu=float(mu), ec=float(ec),
                sweeps=int(sweeps), peak_deg=int(peak_deg),
                edges=int(indices.size // 2), n_nodes=int(degrees.shape[0]),
                k_avg=(float(k_avg) if k_avg is not None else None),
                engine=(str(engine) if engine is not None else None),
                # equilibration verdict from build time (None for graphs cached
                # before the verification pass existed) — restored on load so a
                # cache hit carries the same guarantee flags as a fresh build.
                therm_info=(dict(therm_info) if therm_info else None),
                created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    arrays = dict(indptr=indptr, indices=indices,
                  meta=np.array(json.dumps(meta)))
    if therm_trace is not None and len(therm_trace):
        arrays["therm_trace"] = np.asarray(therm_trace, dtype=np.float64)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)                       # np.savez_compressed appends .npz; tmp already has it
    return path


def _read_npz(path: str):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        indptr = z["indptr"]; indices = z["indices"]
        if "therm_trace" in z.files:
            meta["_therm_trace"] = [tuple(row) for row in z["therm_trace"]]
    return meta, indptr, indices


def find_path(k, T, lb, N, seed, max_degree, *, mu=None) -> str | None:
    """Return the cache path iff a usable graph exists (right format, matching
    params, and — if mu is given — a matching mu). Otherwise None (rebuild)."""
    path = path_for(k, T, lb, N, seed, max_degree)
    if not os.path.exists(path):
        return None
    try:
        meta, _ip, _id = _read_npz(path)
    except Exception:
        return None
    if meta.get("fmt") != FORMAT_VERSION:
        return None
    if (int(meta.get("N", -1)) != int(N) or int(meta.get("max_degree", -1)) != int(max_degree)):
        return None
    if mu is not None and abs(float(meta.get("mu", float("nan"))) - float(mu)) > _MU_TOL:
        return None
    return path


def load_graph(k, T, lb, N, seed, max_degree, *, mu=None) -> StoredGraph | None:
    path = find_path(k, T, lb, N, seed, max_degree, mu=mu)
    if path is None:
        return None
    try:
        meta, indptr, indices = _read_npz(path)
        neighbors, degrees = _from_csr(indptr, indices, max_degree)
        return StoredGraph(neighbors, degrees, meta)
    except Exception:
        return None
