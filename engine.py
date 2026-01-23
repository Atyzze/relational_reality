import math
import numpy as np
import networkx as nx
from numba import njit

class PhysicsEngine:
    def __init__(self, N, seed=None):
        self.N = N
        if seed:
            np.random.seed(seed)
        self.psi = np.zeros(N, dtype=np.complex128)
        self.theta_matrix = np.zeros((N, N), dtype=np.float64)
        self.adj_matrix = np.zeros((N, N), dtype=np.bool_)

        # Track edge count to avoid expensive O(N^2) scans every step
        self.E_tracker = np.array([0], dtype=np.int32)

        for i in range(N):
            #self.psi[i] = np.random.normal(0, 0.1) + 1j * np.random.normal(0, 0.1) #init all as random instead
            self.psi[i] = 0 #init to 0, all the same instead of all random

        self.params = np.array([
            0.20,   # 0: TEMP
            0.90,   # 1: KAPPA
            0.60,   # 2: LAM_G
            0.08,   # 3: LAM_PSI
            0.05,   # 4: LAM_PAULI
            4.0,    # 5: RHO0
            0.35,   # 6: MASS2
            0.50,   # 7: BETA
            -0.50,  # 8: LAM_E
            0.55,   # 9: P_PSI
            0.35,   # 10: P_THETA
            0.25,   # 11: PSI_STEP
            0.35,   # 12: THETA_STEP
            0.9999, # 13: TEMP_SCALE
            0.01,   # 14: MU_DEG2
            0.99    # 15: P_TRIADIC_TOGGLE
        ], dtype=np.float64)

    @property
    def G(self):
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(self.N))
        nx.set_node_attributes(G_nx, {i: self.psi[i] for i in range(self.N)}, "psi")
        rows, cols = np.where(self.adj_matrix)
        edges = []
        for r, c in zip(rows, cols):
            if r < c:
                edges.append((r, c, {"theta": self.theta_matrix[r, c]}))
        G_nx.add_edges_from(edges)
        return G_nx

    def step(self):
        # Pass edge tracker to compiled function
        attempt_step(self.N, self.psi, self.theta_matrix, self.adj_matrix, self.params, self.E_tracker)

# --- COMPILED CORE FUNCTIONS ---
@njit(cache=True)
def get_energy_local_psi(i, psi, theta_matrix, adj_matrix, N, params):
    kappa = params[1]
    lambda_g = params[2]
    lambda_psi = params[3]
    lambda_pauli = params[4]
    rho0 = params[5]
    mass2 = params[6]

    p = psi[i]
    r = p.real**2 + p.imag**2
    denom = rho0 if rho0 > 1e-9 else 1e-9
    E_self = mass2 * r + lambda_psi * r**2 + lambda_pauli * math.exp(r / denom)

    E_interaction = 0.0
    for j in range(N):
        if adj_matrix[i, j]:
            p_j = psi[j]
            th = theta_matrix[i, j]
            # Handle orientation for consistent phase difference
            if i > j:
                u_real = math.cos(-th); u_imag = math.sin(-th)
            else:
                u_real = math.cos(th); u_imag = math.sin(th)

            t1_r = p.real * u_real + p.imag * u_imag
            t1_i = p.real * u_imag - p.imag * u_real
            res_real = t1_r * p_j.real - t1_i * p_j.imag
            E_interaction += -kappa * res_real

            rho_j = p_j.real**2 + p_j.imag**2
            E_interaction += lambda_g * (r + rho_j)

    return E_self + E_interaction

@njit(cache=True)
def get_energy_edge_terms(u, v, theta_val, psi, theta_matrix, adj_matrix, N, params):
    kappa = params[1]
    beta = params[7]
    lambda_g = params[2]

    if u > v:
        u_uv_r = math.cos(-theta_val); u_uv_i = math.sin(-theta_val)
    else:
        u_uv_r = math.cos(theta_val); u_uv_i = math.sin(theta_val)

    p_u = psi[u]; p_v = psi[v]
    t1_r = p_u.real * u_uv_r + p_u.imag * u_uv_i
    t1_i = p_u.real * u_uv_i - p_u.imag * u_uv_r
    hop_real = t1_r * p_v.real - t1_i * p_v.imag
    E_hop = -kappa * hop_real

    rho_u = p_u.real**2 + p_u.imag**2
    rho_v = p_v.real**2 + p_v.imag**2
    E_stress = lambda_g * (rho_u + rho_v)

    E_tri = 0.0
    for k in range(N):
        if adj_matrix[u, k] and adj_matrix[v, k]:
            th_vk = theta_matrix[v, k]
            if v > k:
                u_vk_r = math.cos(-th_vk); u_vk_i = math.sin(-th_vk)
            else:
                u_vk_r = math.cos(th_vk); u_vk_i = math.sin(th_vk)

            th_ku = theta_matrix[k, u]
            if k > u:
                u_ku_r = math.cos(-th_ku); u_ku_i = math.sin(-th_ku)
            else:
                u_ku_r = math.cos(th_ku); u_ku_i = math.sin(th_ku)

            t2_r = u_uv_r * u_vk_r - u_uv_i * u_vk_i
            t2_i = u_uv_r * u_vk_i + u_uv_i * u_vk_r
            loop_r = t2_r * u_ku_r - t2_i * u_ku_i
            E_tri += beta * (1.0 - loop_r)

    return E_hop + E_stress + E_tri

@njit(cache=True)
def attempt_toggle(u, v, psi, theta_matrix, adj_matrix, N, params, log_q_correction):
    temp = params[0]
    lambda_e = params[8]
    mu = params[14]

    exists = adj_matrix[u, v]

    d_u = 0; d_v = 0
    for k in range(N):
        if adj_matrix[u, k]: d_u += 1
        if adj_matrix[v, k]: d_v += 1

    if exists:
        theta_val = theta_matrix[u, v]
    else:
        theta_val = np.random.uniform(-math.pi, math.pi)

    E_edge_topo = get_energy_edge_terms(u, v, theta_val, psi, theta_matrix, adj_matrix, N, params)

    if exists:
        # REMOVE
        dE_topo = -(E_edge_topo + lambda_e)
        dE_deg = mu * ((1 - 2*d_u) + (1 - 2*d_v))
    else:
        # ADD
        dE_topo = (E_edge_topo + lambda_e)
        dE_deg = mu * ((1 + 2*d_u) + (1 + 2*d_v))

    dE = dE_topo + dE_deg

    exponent = (-dE / temp) + log_q_correction

    accept = False
    if exponent >= 0:
        accept = True
    elif temp > 1e-12:
         if np.random.random() < math.exp(exponent):
             accept = True
    elif dE <= 0:
        accept = True

    if accept:
        if exists:
            adj_matrix[u, v] = False; adj_matrix[v, u] = False
            theta_matrix[u, v] = 0.0; theta_matrix[v, u] = 0.0
            return -1 # Return change in edge count
        else:
            adj_matrix[u, v] = True; adj_matrix[v, u] = True
            theta_matrix[u, v] = theta_val; theta_matrix[v, u] = theta_val
            return 1
    return 0

@njit(cache=True)
def get_edges(adj_matrix, N):
    # Optimized helper to get all edges for uniform sampling
    # Count first to allocate array
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if adj_matrix[i, j]:
                count += 1

    edges = np.empty((count, 2), dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if adj_matrix[i, j]:
                edges[idx, 0] = i
                edges[idx, 1] = j
                idx += 1
    return edges, count

@njit(cache=True)
def attempt_step(N, psi, theta_matrix, adj_matrix, params, E_tracker):
    r = np.random.random()

    temp = params[0]
    p_psi = params[9]
    p_theta = params[10]

    # --- PSI MOVE ---
    if r < p_psi:
        i = np.random.randint(0, N)
        E0 = get_energy_local_psi(i, psi, theta_matrix, adj_matrix, N, params)
        old_val = psi[i]
        psi[i] = old_val + complex(np.random.normal(0, params[11]), np.random.normal(0, params[11]))
        E1 = get_energy_local_psi(i, psi, theta_matrix, adj_matrix, N, params)
        if (E1 - E0) > 0 and (temp < 1e-12 or np.random.random() >= math.exp(-(E1 - E0) / temp)):
            psi[i] = old_val

    # --- THETA MOVE ---
    elif r < (p_psi + p_theta):
        # Sample edge robustly
        if E_tracker[0] > 0:
            # We don't need full edge list, just retry a few times to find an edge
            u, v = -1, -1
            for _ in range(20):
                uc = np.random.randint(0, N); vc = np.random.randint(0, N)
                if uc != vc and adj_matrix[uc, vc]:
                    u, v = uc, vc; break

            if u != -1:
                th0 = theta_matrix[u, v]
                E0 = get_energy_edge_terms(u, v, th0, psi, theta_matrix, adj_matrix, N, params)
                new_th = (th0 + np.random.normal(0, params[12]) + math.pi) % (2*math.pi) - math.pi
                theta_matrix[u, v] = new_th; theta_matrix[v, u] = new_th
                E1 = get_energy_edge_terms(u, v, new_th, psi, theta_matrix, adj_matrix, N, params)
                if (E1 - E0) > 0 and (temp < 1e-12 or np.random.random() >= math.exp(-(E1 - E0) / temp)):
                    theta_matrix[u, v] = th0; theta_matrix[v, u] = th0

    # --- REWIRE MOVE ---
    else:
        # 1. TRIADIC TOGGLE (Local)
        if np.random.random() < params[15]:
            k = np.random.randint(0, N)
            deg_k = 0
            # Quick neighbor count
            for x in range(N):
                if adj_matrix[k, x]: deg_k += 1

            if deg_k >= 2:
                idx_1 = np.random.randint(0, deg_k)
                idx_2 = np.random.randint(0, deg_k - 1)
                if idx_2 >= idx_1: idx_2 += 1

                n1 = -1; n2 = -1; curr = 0
                for x in range(N):
                    if adj_matrix[k, x]:
                        if curr == idx_1: n1 = x
                        if curr == idx_2: n2 = x
                        curr += 1

                if n1 != -1 and n2 != -1:
                    delta = attempt_toggle(n1, n2, psi, theta_matrix, adj_matrix, N, params, 0.0)
                    E_tracker[0] += delta

        # 2. GLOBAL REWIRE (Global)
        else:
            E_curr = E_tracker[0]
            total_pairs = (N * (N - 1)) // 2
            NE_curr = total_pairs - E_curr

            # Determine direction balanced:
            # If graph is empty, must ADD. If full, must REMOVE. Else 50/50.
            p_remove = 0.5
            if E_curr == 0: p_remove = 0.0
            if NE_curr == 0: p_remove = 1.0

            if np.random.random() < p_remove:
                # TRY REMOVE
                if E_curr > 0:
                    # ONLY build edge list if we are removing
                    edges, _ = get_edges(adj_matrix, N)
                    idx = np.random.randint(0, E_curr)
                    u = edges[idx, 0]
                    v = edges[idx, 1]

                    log_q = math.log(max(1, E_curr) / max(1, NE_curr + 1))
                    delta = attempt_toggle(u, v, psi, theta_matrix, adj_matrix, N, params, log_q)
                    E_tracker[0] += delta
            else:
                # TRY ADD
                # Pick uniformly from non-edges.
                u, v = -1, -1
                for _ in range(50):
                    uc = np.random.randint(0, N); vc = np.random.randint(0, N)
                    if uc != vc and not adj_matrix[uc, vc]:
                        u, v = uc, vc; break

                if u != -1:
                    log_q = math.log(max(1, NE_curr) / max(1, E_curr + 1))
                    delta = attempt_toggle(u, v, psi, theta_matrix, adj_matrix, N, params, log_q)
                    E_tracker[0] += delta

    # Cool down
    params[0] = temp * params[13]

