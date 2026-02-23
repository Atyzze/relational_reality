import math
import numpy as np
import networkx as nx
from numba import njit

P_THETA   = 0   # 1 = always wiggle edge phases, never rewire, 0 = always rewire, never wiggle, 0.5 = 50%-50% of both
TH_STEP   = 0.00   # Brownian step size for edge phase

# --- NEW SPARSE CONSTRAINT ---
# Hard limit on maximum neighbors per node.
# Since your network averages k~4 and peaks ~8, 128 is extremely safe and keeps memory tiny.
MAX_DEGREE = 1024


class PhysicsEngine:
    """Sparse physics/rewiring engine.

    Compatibility notes:
    - Nodes can carry a complex-valued state 'psi' (two floats per node).
      This is stored internally as a (N,2) float array: psi[:,0]=real, psi[:,1]=imag.
    - The exported NetworkX graph (engine.G) includes node attribute 'psi' as a Python complex.
    - For snapshot loading, use .clear() + .add_edge_snapshot(...) + direct writes to .psi.
    """

    def __init__(self, N: int, seed=None, edge_capacity: int | None = None):
        self.N = int(N)
        if seed is not None:
            np.random.seed(seed)

        # --- NODE STATE (psi) ---
        # Two numbers per node (real, imag). Stored as float64 for speed/numba friendliness.
        self.psi = np.zeros((self.N, 2), dtype=np.float64)

        # --- SPARSE ARCHITECTURE ---
        # adj_nodes stores the ID of the neighbor.
        # adj_thetas stores the phase angle directed FROM the current node TO the neighbor.
        self.adj_nodes = np.full((self.N, MAX_DEGREE), -1, dtype=np.int32)
        self.adj_thetas = np.zeros((self.N, MAX_DEGREE), dtype=np.float64)
        self.deg = np.zeros(self.N, dtype=np.int32)

        # --- EDGE TRACKING ---
        if edge_capacity is None:
            edge_capacity = max(1024, 8 * self.N)
            edge_capacity = min(edge_capacity, (self.N * (self.N - 1)) // 2 if self.N > 1 else 1)

        self.edge_list = np.zeros((int(edge_capacity), 2), dtype=np.int32)
        # meta = [count_valid_edges, used_slots]
        self.meta = np.array([0, 0], dtype=np.int32)

        # --- DYNAMIC CONTROLS ---
        # 0: TEMP      (temperature)
        # 1: MU_DEG    (degree-regularization strength)
        # 2: BETA      (triadic closure / triangle frustration strength)
        # 3: LAM_E     (base cost of an edge; + => sparse, - => dense)
        # 4: P_TRIADIC (probability of local peer introduction vs blind vacuum tethering)
        self.params = np.array([0, 0, 0, -1.0, 0.99], dtype=np.float64)
    # ---------------------------
    # Public helpers
    # ---------------------------

    def clear(self, clear_psi: bool = False) -> None:
        """Reset topology (and optionally psi) to empty."""
        self.adj_nodes.fill(-1)
        self.adj_thetas.fill(0.0)
        self.deg.fill(0)
        self.meta[:] = 0
        if clear_psi:
            self.psi.fill(0.0)

    def set_psi(self, node: int, real: float, imag: float) -> None:
        self.psi[int(node), 0] = float(real)
        self.psi[int(node), 1] = float(imag)

    def get_psi(self, node: int) -> complex:
        n = int(node)
        return complex(float(self.psi[n, 0]), float(self.psi[n, 1]))

    def add_edge_snapshot(self, u: int, v: int, theta: float) -> bool:
        """Add an edge exactly (used for snapshot/state restore).

        Unlike the Monte Carlo toggles, this does not compute energies or accept/reject.
        It is safe to call many times when reconstructing a saved graph.

        Returns True if the edge was added, False if it already existed or max degree hit.
        """
        u = int(u); v = int(v)
        if u == v:
            return False
        if find_idx(u, v, self.adj_nodes, self.deg) != -1:
            return False

        self._ensure_edge_capacity(extra_slots_needed=1)
        ok = add_edge(u, v, float(theta), self.adj_nodes, self.adj_thetas, self.deg, MAX_DEGREE)
        if not ok:
            return False

        idx = int(self.meta[1])
        if idx < self.edge_list.shape[0]:
            self.edge_list[idx, 0] = u
            self.edge_list[idx, 1] = v
            self.meta[1] += 1
        self.meta[0] += 1
        return True

    @property
    def G(self) -> nx.Graph:
        """Export for visualization (fast sparse iteration)."""
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(self.N))

        # Attach psi to nodes (keeps downstream tooling compatible)
        # 'psi' is a complex number; also include psi_real/psi_imag for convenience.
        psi_real = self.psi[:, 0]
        psi_imag = self.psi[:, 1]
        for n in range(self.N):
            G_nx.nodes[n]["psi"] = complex(float(psi_real[n]), float(psi_imag[n]))
            G_nx.nodes[n]["psi_real"] = float(psi_real[n])
            G_nx.nodes[n]["psi_imag"] = float(psi_imag[n])

        for u in range(self.N):
            for i in range(self.deg[u]):
                v = self.adj_nodes[u, i]
                # Only add each edge once (canonical ordering)
                if u < v:
                    G_nx.add_edge(u, v, theta=float(self.adj_thetas[u, i]))
        return G_nx

    def _ensure_edge_capacity(self, extra_slots_needed: int = 1) -> None:
        """Ensure edge_list has room for new entries."""
        used = int(self.meta[1])
        cap = self.edge_list.shape[0]
        if used + extra_slots_needed < cap:
            return

        # First try compaction
        rebuild_edge_list(self.edge_list, self.adj_nodes, self.deg, self.meta)

        used = int(self.meta[1])
        if used + extra_slots_needed < cap:
            return

        # Grow
        max_edges = (self.N * (self.N - 1)) // 2 if self.N > 1 else 1
        new_cap = min(max_edges, max(cap * 2, used + extra_slots_needed + 16))
        if new_cap <= cap:
            return

        new_list = np.zeros((new_cap, 2), dtype=np.int32)
        new_list[:used] = self.edge_list[:used]
        self.edge_list = new_list

    def iterate(self) -> None:
        valid_edges = int(self.meta[0])
        used_slots = int(self.meta[1])

        # Periodic compaction
        if used_slots > 2 * max(100, valid_edges):
            rebuild_edge_list(self.edge_list, self.adj_nodes, self.deg, self.meta)

        self._ensure_edge_capacity(extra_slots_needed=2)

        perform_iteration(
            self.N,
            self.adj_nodes,
            self.adj_thetas,
            self.deg,
            self.edge_list,
            self.meta,
            self.params,
            MAX_DEGREE
        )


# ---------------------------
# COMPILED CORE (SPARSE OPTIMIZED)
# ---------------------------

@njit(cache=True)
def find_idx(u, v, adj_nodes, deg):
    """O(k) search for a neighbor instead of O(N) matrix lookup."""
    for i in range(deg[u]):
        if adj_nodes[u, i] == v:
            return i
    return -1


@njit(cache=True)
def add_edge(u, v, th, adj_nodes, adj_thetas, deg, max_deg):
    if deg[u] >= max_deg or deg[v] >= max_deg:
        return False

    idx_u = deg[u]
    adj_nodes[u, idx_u] = v
    adj_thetas[u, idx_u] = th
    deg[u] += 1

    idx_v = deg[v]
    adj_nodes[v, idx_v] = u
    adj_thetas[v, idx_v] = -th  # Antisymmetric enforcement
    deg[v] += 1
    return True


@njit(cache=True)
def remove_edge(u, v, adj_nodes, adj_thetas, deg):
    """O(k) removal using swap-and-pop."""
    i = find_idx(u, v, adj_nodes, deg)
    if i != -1:
        last = deg[u] - 1
        adj_nodes[u, i] = adj_nodes[u, last]
        adj_thetas[u, i] = adj_thetas[u, last]
        deg[u] = last

    j = find_idx(v, u, adj_nodes, deg)
    if j != -1:
        last = deg[v] - 1
        adj_nodes[v, j] = adj_nodes[v, last]
        adj_thetas[v, j] = adj_thetas[v, last]
        deg[v] = last


@njit(cache=True)
def rebuild_edge_list(edge_list, adj_nodes, deg, meta):
    """Perfect O(N + E) compaction directly from the source of truth."""
    write_ptr = 0
    N = len(deg)
    for u in range(N):
        for i in range(deg[u]):
            v = adj_nodes[u, i]
            if u < v:  # Deduplicate inherently
                edge_list[write_ptr, 0] = u
                edge_list[write_ptr, 1] = v
                write_ptr += 1
    meta[0] = write_ptr
    meta[1] = write_ptr


@njit(cache=True)
def get_energy_edge_topology(u, v, th_uv, adj_nodes, adj_thetas, deg, beta):
    """Triangle frustration. Runs in O(k) time."""
    tri_term = 0.0
    for i in range(deg[u]):
        k = adj_nodes[u, i]

        # Is k also a neighbor of v?
        j = find_idx(v, k, adj_nodes, deg)
        if j != -1:
            th_uk = adj_thetas[u, i]
            th_vk = adj_thetas[v, j]

            # phi = theta[u,v] + theta[v,k] + theta[k,u]
            # Since theta[k,u] is exactly -theta[u,k]
            phi = th_uv + th_vk - th_uk
            tri_term += 1.0 - math.cos(phi)

    return beta * tri_term


@njit(cache=True)
def attempt_toggle(u, v, adj_nodes, adj_thetas, deg, params, edge_list, meta, max_deg):
    temp = params[0]
    mu = params[1]
    beta = params[2]
    lambda_e = params[3]

    idx_u = find_idx(u, v, adj_nodes, deg)
    exists = (idx_u != -1)

    d_u = deg[u]
    d_v = deg[v]

    if exists:
        th_val = adj_thetas[u, idx_u]
    else:
        if d_u >= max_deg or d_v >= max_deg:
            return False  # Hit sparse memory limit
        th_val = np.random.uniform(-math.pi, math.pi)

    E_topo = get_energy_edge_topology(u, v, th_val, adj_nodes, adj_thetas, deg, beta)
    k0 = 0  # target coordination
    # --- DEGREE PENALTY ---
    if exists:
        du = 1 - 2 * (d_u - k0)
        dv = 1 - 2 * (d_v - k0)
        dE = -(E_topo + lambda_e) + mu * (du + dv)
    else:
        du = 1 + 2 * (d_u - k0)
        dv = 1 + 2 * (d_v - k0)
        dE = (E_topo + lambda_e) + mu * (du + dv)

    accept = False
    if dE <= 0.0:
        accept = True
    else:
        if temp > 0.0 and np.random.random() < math.exp(-dE / temp):
            accept = True

    if not accept:
        return False

    if exists:
        remove_edge(u, v, adj_nodes, adj_thetas, deg)
        meta[0] -= 1
    else:
        add_edge(u, v, th_val, adj_nodes, adj_thetas, deg, max_deg)
        idx = meta[1]
        if idx < edge_list.shape[0]:
            edge_list[idx, 0] = u
            edge_list[idx, 1] = v
            meta[1] += 1
        meta[0] += 1

    return True


@njit(cache=True)
def perform_iteration(N, adj_nodes, adj_thetas, deg, edge_list, meta, params, max_deg):
    r = np.random.random()
    temp = params[0]
    beta = params[2]
    p_triadic = params[4]

    # 1) EDGE PHASE ADJUSTMENT ("wiggling")
    if r < P_THETA:
        if meta[0] > 0:
            for _ in range(100):  # Cap retries to avoid infinite loops in edge cases
                idx = np.random.randint(0, meta[1])
                u = edge_list[idx, 0]
                v = edge_list[idx, 1]

                idx_u = find_idx(u, v, adj_nodes, deg)
                if idx_u != -1:
                    th0 = adj_thetas[u, idx_u]
                    E0 = get_energy_edge_topology(u, v, th0, adj_nodes, adj_thetas, deg, beta)

                    new_th = (th0 + np.random.normal(0.0, TH_STEP) + math.pi) % (2 * math.pi) - math.pi

                    # Apply temporarily for evaluation
                    adj_thetas[u, idx_u] = new_th
                    idx_v = find_idx(v, u, adj_nodes, deg)
                    adj_thetas[v, idx_v] = -new_th

                    E1 = get_energy_edge_topology(u, v, new_th, adj_nodes, adj_thetas, deg, beta)
                    dE = E1 - E0

                    if dE > 0.0:
                        if temp == 0.0 or np.random.random() >= math.exp(-dE / temp):
                            # Reject: revert changes
                            adj_thetas[u, idx_u] = th0
                            adj_thetas[v, idx_v] = -th0
                    break

    # 2) TOPOLOGY CHANGE
    else:
        if np.random.random() < p_triadic:
            k = np.random.randint(0, N)
            cnt = deg[k]
            if cnt >= 2:
                i1 = np.random.randint(0, cnt)
                i2 = np.random.randint(0, cnt - 1)
                if i2 >= i1:
                    i2 += 1
                u = adj_nodes[k, i1]
                v = adj_nodes[k, i2]
                attempt_toggle(u, v, adj_nodes, adj_thetas, deg, params, edge_list, meta, max_deg)
        else:
            u = np.random.randint(0, N)
            v = np.random.randint(0, N)
            if u != v:
                attempt_toggle(u, v, adj_nodes, adj_thetas, deg, params, edge_list, meta, max_deg)
