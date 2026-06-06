"""
metrics — graph metrics + spectral-dimension building blocks
=============================================================

Public surface:

    Topology snapshot:
        GraphStats              — namedtuple
        get_graph_stats(neighbors, degrees) -> GraphStats
        compute_graph_stats     — njit core

    Graph container:
        Graph                   — dataclass (neighbors, degrees, N, …)

    Asymptotic extrapolation:
        extrapolate_asymptote(N_values, ds_values, ci95_values, ...)

    Reference graphs (validation against analytically-known d_s):
        REFERENCE_FAMILIES, all_families, build_family,
        build_hypercubic_{1..6}d, build_cycle, build_triangular_2d,
        build_honeycomb_2d, build_sierpinski_gasket,
        build_sierpinski_tetrahedron, build_t_fractal,
        build_sierpinski_carpet

Module layout (downstream of __init__):
    numba_kernels.py         Numba kernels (LCC extraction, CSR matvec,
                             Laplacian build) shared by core's flow probe
    stochastic_lanczos.py    Lanczos tridiagonalization kernels (SLQ)
    graph_container.py       Graph dataclass
    graph_topology.py        GraphStats + structural metrics
    size_extrapolation.py    Finite-size extrapolation
    reference_lattices.py    Reference graph builders + registry

The low-level SLQ/LCC kernels in numba_kernels and stochastic_lanczos are
imported directly by core.flow_probe / core.cell_tests via their submodule
paths; they keep a leading underscore to mark them internal and are not
re-exported here.
"""

# ─── Topology ────────────────────────────────────────────────────────
from .graph_topology import (
    GraphStats,
    get_graph_stats,
    compute_graph_stats,
)

# ─── Graph dataclass ─────────────────────────────────────────────────
from .graph_container import Graph

# ─── Extrapolation ───────────────────────────────────────────────────
from .size_extrapolation import extrapolate_asymptote

# ─── Reference graphs ────────────────────────────────────────────────
from .reference_lattices import (
    REFERENCE_FAMILIES, all_families, build_family,
    build_hypercubic_1d, build_hypercubic_2d, build_hypercubic_3d,
    build_hypercubic_4d, build_hypercubic_5d, build_hypercubic_6d,
    build_cycle, build_triangular_2d, build_honeycomb_2d,
    build_sierpinski_gasket, build_sierpinski_tetrahedron,
    build_t_fractal, build_sierpinski_carpet,
)

__all__ = [
    # Topology
    "GraphStats", "get_graph_stats", "compute_graph_stats",
    # Graph
    "Graph",
    # Extrap
    "extrapolate_asymptote",
    # Refs
    "REFERENCE_FAMILIES", "all_families", "build_family",
    "build_hypercubic_1d", "build_hypercubic_2d", "build_hypercubic_3d",
    "build_hypercubic_4d", "build_hypercubic_5d", "build_hypercubic_6d",
    "build_cycle", "build_triangular_2d", "build_honeycomb_2d",
    "build_sierpinski_gasket", "build_sierpinski_tetrahedron",
    "build_t_fractal", "build_sierpinski_carpet",
]
