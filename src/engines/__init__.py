"""
engines — registry of selectable graph-growth back-ends
=======================================================
    from engines import get_engine, available_engines, ensure_built

    Eng = get_engine("cpp")              # or "numba" / "rust" (aliases: py, c++, rs)
    eng = Eng(N=16000, max_degree=32, seed=1, temperature=0.0, locality_bias=0.99,
              degree_penalty=0.031)
    eng.sweep(300)                       # grow
    g = eng.snapshot()                   # -> metrics.graph_container.Graph

Pick programmatically:
    for name, cls in available_engines().items():
        print(name, "->", [p.name for p in cls.parameters()])

To add an engine: implement engines.base.Engine and add it to _REGISTRY below.
"""
from .base import Engine, Param, Graph                  # noqa: F401
from .numba.engine import NumbaEngine
from .cpp.engine import CppEngine
from .rust.engine import RustEngine
from . import build                                     # noqa: F401  (grow/calibrate helpers)

_REGISTRY = {cls.backend: cls for cls in (NumbaEngine, CppEngine, RustEngine)}
_ALIASES = {"py": "numba", "python": "numba", "c++": "cpp", "cxx": "cpp", "rs": "rust"}

__all__ = ["Engine", "Param", "Graph", "get_engine", "all_engines",
           "available_engines", "ensure_built", "build",
           "NumbaEngine", "CppEngine", "RustEngine"]


def _key(name: str) -> str:
    k = str(name).lower()
    return _ALIASES.get(k, k)


def get_engine(name: str):
    """Return the Engine *class* for a backend name (raises KeyError if unknown)."""
    k = _key(name)
    if k not in _REGISTRY:
        raise KeyError(f"unknown engine {name!r}; have {sorted(_REGISTRY)} "
                       f"(aliases {sorted(_ALIASES)})")
    return _REGISTRY[k]


def all_engines() -> dict:
    """All registered engines, regardless of availability."""
    return dict(_REGISTRY)


def available_engines(ensure: bool = False, log=print) -> dict:
    """Engines runnable right now. If ensure=True, try to build native libs first."""
    out = {}
    for k, cls in _REGISTRY.items():
        ok = cls.ensure_available(log=log) if ensure else cls.available()
        if ok:
            out[k] = cls
    return out


def ensure_built(backends=None, log=print) -> dict:
    """Compile native engines (cpp/rust) if needed; return {backend: available?}."""
    backends = backends or list(_REGISTRY)
    status = {}
    for name in backends:
        cls = get_engine(name)
        status[cls.backend] = cls.ensure_available(log=log)
    return status
