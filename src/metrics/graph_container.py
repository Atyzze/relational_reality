"""
metrics.graph_container — Graph dataclass
=========================================
The (neighbors, degrees, N) container that the reference-lattice builders
produce and that the spectral-dimension library takes as input. Tiny, no
algorithmic content.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Graph:
    neighbors: np.ndarray        # int32 (N, max_deg), -1 padded
    degrees: np.ndarray          # int32 (N,)
    N: int
    name: str = ""
    family: str = ""
    known_ds: float | None = None
    known_ds_kind: str = ""      # "analytic" | "consensus" | ""

    @property
    def max_deg(self) -> int:
        return int(self.neighbors.shape[1]) if self.neighbors.size else 0
