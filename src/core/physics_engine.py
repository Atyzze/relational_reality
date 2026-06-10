"""
core.physics_engine — COMPATIBILITY SHIM
========================================
The Numba engine kernel moved to the top-level `engines` package, where it now
sits beside the C++ and Rust engines behind a common interface (see
engines/base.py). This shim re-exports the same names so existing callers
(graph_builder, flow_probe, cell_tests, …) keep working unchanged.

New code should prefer:
    from engines import get_engine
    eng = get_engine("numba")(N, max_degree=32, seed=0, temperature=..., ...)
or, for the raw class:
    from engines.numba.kernel import PhysicsEngine
"""
import os as _os
import sys as _sys

_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # .../src
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

from engines.numba.kernel import (   # noqa: F401,E402  (re-export)
    PhysicsEngine,
    seed_numba,
    find_idx,
    add_edge,
    remove_edge,
    attempt_toggle,
    perform_iterations,
)
