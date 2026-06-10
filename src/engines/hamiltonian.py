"""
engines.hamiltonian — parameter spec for the current growth Hamiltonian
=======================================================================
H = edge_cost·E + degree_penalty·Σ d²

These four knobs are shared by the Numba, C++ and Rust engines because they all
implement this same energy. They live in ONE place so configs, the benchmark and
any UI agree on names, defaults and help text. A future engine with a different
Hamiltonian simply returns its own list from `parameters()` instead of this one.
"""
from .base import Param

HAMILTONIAN_PARAMS = [
    Param("temperature", 0.004, "Metropolis temperature T (0.0 = cold ground state)",
          low=0.0),
    Param("degree_penalty", 0.0311,
          "μ — coefficient of Σ d²; sets the equilibrium average degree "
          "(usually auto-calibrated from a target mean degree k)", low=0.0),
    Param("edge_cost", -1.0, "ec — energy per edge (negative favours edges)"),
    Param("locality_bias", 0.99,
          "ℓb — fraction of moves that are local (triangle) moves; the rest are global",
          low=0.0, high=1.0),
]
