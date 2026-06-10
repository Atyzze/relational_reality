"""
engines.base — the Engine interface
===================================
Every graph-growth back-end (Numba reference, C++, Rust, or a future custom
Hamiltonian) implements this one interface, so the rest of the project —
graph builders, the memory benchmark, and every diagnostic probe — can drive
any engine the same way and never has to know which one it is talking to.

The two contracts that make engines swappable:

  1. PARAMETERS ARE SELF-DECLARED.  An engine returns its own list of `Param`s
     from `parameters()`. The current Hamiltonian has four (temperature,
     degree_penalty, edge_cost, locality_bias); a different Hamiltonian can
     declare as many as it likes. Configs and UIs read this list instead of
     hard-coding four knobs, so adding a parameter never touches the callers.

  2. OUTPUT IS UNIFORM.  `snapshot()` returns the project's existing
     `metrics.graph_container.Graph` (int32 neighbours (N, max_degree), int32
     degrees, N). The spectral-dimension library, the reference lattices and
     the isotropy probe already speak `Graph`, so any engine's output drops
     straight into all of them.

To add a new engine: subclass `Engine`, set `backend`/`name`, implement
`parameters()`, `step()`, `neighbors_degrees()`, `degree_array()` and
`peak_degree`, then register it in `engines/__init__.py`. That's the whole
surface a new Hamiltonian has to satisfy.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Reuse the project's existing Graph container so engine output is identical to
# what the reference lattices and probes already consume.
# engines now lives under src/, so the package parent IS the src dir.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    from metrics.graph_container import Graph            # canonical container
except Exception:                                        # pragma: no cover - standalone fallback
    @dataclass
    class Graph:                                         # type: ignore
        neighbors: np.ndarray
        degrees: np.ndarray
        N: int
        name: str = ""
        family: str = ""
        known_ds: Optional[float] = None
        known_ds_kind: str = ""

        @property
        def max_deg(self) -> int:
            return int(self.neighbors.shape[1]) if self.neighbors.size else 0


@dataclass(frozen=True)
class Param:
    """One tunable knob an engine exposes. `kind` is 'float' or 'int'; `low`/
    `high` are advisory bounds for config validation and UI sliders."""
    name: str
    default: float
    help: str = ""
    kind: str = "float"
    low: Optional[float] = None
    high: Optional[float] = None

    def coerce(self, v):
        v = int(round(v)) if self.kind == "int" else float(v)
        if self.low is not None:
            v = max(v, self.low)
        if self.high is not None:
            v = min(v, self.high)
        return v


class Engine(ABC):
    """A graph-growth Markov chain. Construct with the node count and the
    structural options, then advance with `step`/`sweep` and read out the
    graph with `snapshot`/`stats`."""

    name: str = "engine"      # human label, e.g. "C++ (engine_core.cpp)"
    backend: str = ""         # registry key, e.g. "numba" / "cpp" / "rust"

    # ---- declaration (class-level) ----
    @classmethod
    @abstractmethod
    def parameters(cls) -> list[Param]:
        """The Hamiltonian knobs this engine accepts (any number)."""

    @classmethod
    def defaults(cls) -> dict:
        return {p.name: p.default for p in cls.parameters()}

    @classmethod
    def available(cls) -> bool:
        """True if this engine can run right now (deps present / lib built)."""
        return True

    @classmethod
    def ensure_available(cls, log=print) -> bool:
        """Make the engine runnable if possible (e.g. compile a native lib)."""
        return cls.available()

    # ---- lifecycle ----
    def __init__(self, n, *, max_degree: int = 32, seed: int = 0, **params):
        self.n = int(n)
        self.max_degree = int(max_degree)
        self.seed = int(seed)
        spec = {p.name: p for p in self.parameters()}
        self.params = {name: p.default for name, p in spec.items()}
        for k, v in params.items():
            if k in spec:
                self.params[k] = spec[k].coerce(v)
        # kwargs that aren't Hamiltonian params (e.g. an engine 'mode' flag)
        self._extra = {k: v for k, v in params.items() if k not in spec}

    # ---- evolution ----
    @abstractmethod
    def step(self, n_steps: int) -> None:
        """Advance the chain by `n_steps` single Metropolis moves."""

    def sweep(self, n_sweeps: int = 1) -> None:
        """One sweep = N single moves."""
        self.step(self.n * int(n_sweeps))

    # ---- readout ----
    @abstractmethod
    def neighbors_degrees(self):
        """Return (neighbours int32 (N, max_degree), degrees int32 (N,)) as
        fresh arrays in the canonical layout (-1 padded)."""

    @abstractmethod
    def degree_array(self) -> np.ndarray:
        """Cheap degrees-only readout (for thermalisation loops)."""

    @property
    @abstractmethod
    def peak_degree(self) -> int:
        """Highest single-node degree seen at any point in the trajectory."""

    def snapshot(self, name: str = "", family: str = "") -> Graph:
        nb, dg = self.neighbors_degrees()
        return Graph(neighbors=nb, degrees=dg, N=self.n,
                     name=name or self.backend, family=family)

    def stats(self) -> dict:
        dg = self.degree_array()
        edges = int(dg.sum()) // 2
        return dict(edges=edges, k_avg=2 * edges / max(self.n, 1),
                    peak_degree=self.peak_degree)

    # ---- resource management ----
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<{self.backend} engine N={self.n} max_degree={self.max_degree}>"
