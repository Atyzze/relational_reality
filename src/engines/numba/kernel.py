from numba import njit
import numpy as np
import math

from ..hamiltonian import HAMILTONIAN_PARAMS

# Canonical defaults for the four Hamiltonian knobs, taken from the ONE place that
# defines them (engines/hamiltonian.py), shared with the C++/Rust engines and every
# config. Callers in the flow pipeline pass the swept values explicitly; these
# defaults only apply to ad-hoc construction.
_HAM_DEFAULTS = {p.name: p.default for p in HAMILTONIAN_PARAMS}


class PhysicsEngine:
    def __init__(self, N, seed, max_degree=32, *,
                 temperature=None, degree_penalty=None,
                 edge_cost=None, locality_bias=None):
        np.random.seed(seed); seed_numba(seed)
        self.N = int(N); self.max_degree = int(max_degree)
        self.node_neighbors = np.full((self.N, self.max_degree), -1, dtype=np.int32)
        self.node_degrees = np.zeros(self.N, dtype=np.int32)
        self.temperature    = _HAM_DEFAULTS["temperature"]    if temperature    is None else float(temperature)
        self.degree_penalty = _HAM_DEFAULTS["degree_penalty"] if degree_penalty is None else float(degree_penalty)
        self.edge_cost      = _HAM_DEFAULTS["edge_cost"]      if edge_cost      is None else float(edge_cost)
        self.locality_bias  = _HAM_DEFAULTS["locality_bias"]  if locality_bias  is None else float(locality_bias)
        self._peak = np.zeros(1, dtype=np.int32)

    @property
    def peak_degree(self): return int(self._peak[0])

    def iterate(self, steps=100_000):
        try:
            perform_iterations(
                steps, self.N, self.node_neighbors, self.node_degrees,
                self.temperature, self.degree_penalty, self.edge_cost,
                self.locality_bias, self.max_degree, self._peak)
        except Exception as e:
            if "max_degree exceeded" in str(e):
                at_cap = int((self.node_degrees >= self.max_degree).sum())
                raise RuntimeError(
                    f"max_degree={self.max_degree} hit — {at_cap} node(s) at cap, "
                    f"peak={int(self.node_degrees.max())}. Markov chain is no "
                    f"longer sampling the Hamiltonian; raise max_degree.") from None
            raise


@njit(cache=True)
def seed_numba(seed): np.random.seed(seed)

@njit(cache=True)
def find_idx(u, v, neighbors, degrees):
    for i in range(degrees[u]):
        if neighbors[u, i] == v: return i
    return -1

@njit(cache=True)
def add_edge(u, v, neighbors, degrees):
    neighbors[u, degrees[u]] = v; degrees[u] += 1
    neighbors[v, degrees[v]] = u; degrees[v] += 1

@njit(cache=True)
def remove_edge(u, v, neighbors, degrees):
    i = find_idx(u, v, neighbors, degrees)
    if i != -1:
        last = degrees[u] - 1
        neighbors[u, i] = neighbors[u, last]; neighbors[u, last] = -1; degrees[u] = last
    j = find_idx(v, u, neighbors, degrees)
    if j != -1:
        last = degrees[v] - 1
        neighbors[v, j] = neighbors[v, last]; neighbors[v, last] = -1; degrees[v] = last

@njit(cache=True)
def attempt_toggle(u, v, neighbors, degrees, temperature, mu, ec, max_deg, peak):
    exists = find_idx(u, v, neighbors, degrees) != -1
    du, dv = degrees[u], degrees[v]
    if not exists and (du >= max_deg or dv >= max_deg):
        raise RuntimeError("max_degree exceeded")
    if exists: dE = -ec + mu * (2 - 2*(du + dv))
    else:      dE =  ec + mu * (2 + 2*(du + dv))
    if dE <= 0.0 or (temperature > 0.0 and np.random.random() < math.exp(-dE / temperature)):
        if exists: remove_edge(u, v, neighbors, degrees)
        else:
            add_edge(u, v, neighbors, degrees)
            if degrees[u] > peak[0]: peak[0] = degrees[u]
            if degrees[v] > peak[0]: peak[0] = degrees[v]

@njit(cache=True)
def perform_iterations(steps, N, neighbors, degrees, temperature, mu, ec, lb, max_deg, peak):
    for _ in range(steps):
        if np.random.random() < lb:
            k = np.random.randint(0, N)
            cnt = degrees[k]
            if cnt >= 2:
                i1 = np.random.randint(0, cnt)
                i2 = np.random.randint(0, cnt - 1)
                if i2 >= i1: i2 += 1
                attempt_toggle(neighbors[k, i1], neighbors[k, i2], neighbors, degrees, temperature, mu, ec, max_deg, peak)
        else:
            u = np.random.randint(0, N)
            v = np.random.randint(0, N)
            if u != v:
                attempt_toggle(u, v, neighbors, degrees, temperature, mu, ec, max_deg, peak)
