"""engines.numba.engine — the reference engine (Numba JIT)."""
from __future__ import annotations

import numpy as np

from ..base import Engine, Param
from ..hamiltonian import HAMILTONIAN_PARAMS
from .kernel import perform_iterations, seed_numba


class NumbaEngine(Engine):
    name = "Numba (reference)"
    backend = "numba"

    @classmethod
    def parameters(cls) -> list[Param]:
        return list(HAMILTONIAN_PARAMS)

    @classmethod
    def available(cls) -> bool:
        try:
            import numba  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self, n, *, max_degree: int = 32, seed: int = 0, **params):
        super().__init__(n, max_degree=max_degree, seed=seed, **params)
        self.nb = np.full((self.n, self.max_degree), -1, dtype=np.int32)
        self.dg = np.zeros(self.n, dtype=np.int32)
        self._peak = np.zeros(1, dtype=np.int32)
        seed_numba(self.seed)

    def step(self, n_steps: int) -> None:
        p = self.params
        try:
            perform_iterations(int(n_steps), self.n, self.nb, self.dg,
                               p["temperature"], p["degree_penalty"],
                               p["edge_cost"], p["locality_bias"],
                               self.max_degree, self._peak)
        except Exception as e:
            if "max_degree exceeded" in str(e):
                at_cap = int((self.dg >= self.max_degree).sum())
                raise RuntimeError(
                    f"max_degree={self.max_degree} hit — {at_cap} node(s) at cap, "
                    f"peak={int(self.dg.max())}. Raise max_degree.") from None
            raise

    def neighbors_degrees(self):
        return self.nb.copy(), self.dg.copy()

    def degree_array(self) -> np.ndarray:
        return self.dg

    @property
    def peak_degree(self) -> int:
        return int(self._peak[0])
