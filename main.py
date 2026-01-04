import math
import random
from collections import defaultdict, deque

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ot  # Requires: pip install POT

# -----------------------
# CONFIG
# -----------------------
SEED = 44
random.seed(SEED)
np.random.seed(SEED)

N = 4444

# Monte Carlo schedule
EQUIL_MOVES = 4_444_444
PROP_MOVES  = 12_000
LOG_EVERY   = 250

# Move mix
P_PSI    = 0.55
P_THETA  = 0.35
P_REWIRE = 0.10  # you can raise (0.15–0.25) once stable

# Action parameters
KAPPA      = 0.6
BETA       = 0.9
MASS2      = 0.35
LAMBDA_PSI = 0.08
LAMBDA_G   = 0.02

# Pauli exclusion (anti-crunch)
LAMBDA_PAULI = 0.15   # strength of exp(rho/rho0)
RHO0         = 4.0     # density scale for the exponential

# Temperature + steps
TEMP       = 0.5
PSI_STEP   = 0.25
THETA_STEP = 0.35

# Geometry / degree costs
MU_DEG2 = 1.2  # start a touch higher than 1.35 to fight runaway degree

# Triangle shaping (optional)
USE_TRIANGLE_REWARD = False
GAMMA_T = 0.00

# Rewire proposal mix
USE_TRIADIC_TOGGLE = True
P_TRIADIC_TOGGLE   = 0.65  # remainder uses balanced MH global kernel

# Degree control target
USE_DEG_CONTROL    = False
DEG_TARGET         = 7.0
DEG_CONTROL_EVERY  = 2 * LOG_EVERY

# Expansion + controller decomposition:
# lambda_E_total = LAMBDA_E_BASE + DELTA_LAMBDA_E
# (lambda_E_total multiplies E = #edges in the action)
LAMBDA_E_BASE = -44.0     # "cosmological expansion" (reward edges)
DELTA_LAMBDA_E = 0.0      # PI controller state
DELTA_LAMBDA_E_MIN = -100.0
DELTA_LAMBDA_E_MAX =  100.0

# PI gains (tuned to be calm)
DEG_KP      = 0.005
DEG_KI      = 0.000002
DEG_I_CLAMP = 12_000.0

# Correlator
MAX_DIST_CORR = 20
CORR_SOURCES = 260
CORR_SAMPLES_PER_D_PER_SRC = 2
CORR_MIN_SHELL = 5
AVG_K_SHORTEST_PATHS = 1

# Light cone
PERTURB_EPS      = 1.5
MAX_DIST_LIGHTCONE = 10
FRONT_Q            = 0.80

# Curvature probe
CURV_SAMPLE = 600

# Quench
DO_QUENCH = True
QUENCH_RADIUS = 2
QUENCH_STRENGTH = 0.5
MAX_DIST_QUENCH = 10
QUENCH_TWO_STAGE = True
QUENCH_ALLOW_REWIRE = False
QUENCH_P_REWIRE = 0.15
QUENCH_RELAX_MOVES_STAGE1 = 4000
QUENCH_RELAX_MOVES_STAGE2 = 20000
MU_QUENCH = 0.20
DELTA_LAMBDA_E_QUENCH = None
MIN_SHELL_QUENCH = 8

# Fit
XI_FIT_MIN = 2
XI_FIT_MAX = 10
XI_MIN_COUNT = 140

def compute_ollivier_ricci_flow(G):
    """
    Computes the 'Ollivier-Ricci Curvature' for every edge.
    This measures if the space is 'spherical' (positive curvature, gravity)
    or 'hyperbolic' (negative curvature, expansion).

    Theory:
    It compares the distance between two nodes (d) vs the distance
    between their "clouds of neighbors" (W).
    Kappa = 1 - (Wasserstein_Distance / Edge_Length)
    """
    print(">> Calculating Einstein Geometry (Ollivier-Ricci)...")

    curvature_map = {}
    nodes = list(G.nodes())

    # 1. Precompute all-pairs shortest paths (needed for Earth Mover's Distance)
    # For N=1000 this is fast. For N=4000 it takes a moment.
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    for u, v in G.edges():
        # Define the "Mass Distributions" (Ink Drops) at u and v
        # We assume uniform distribution over neighbors (Standard Ricci)
        # Advanced: You could weight this by |psi|^2 if you wanted.

        nbrs_u = list(G.neighbors(u))
        nbrs_v = list(G.neighbors(v))

        # Mass per neighbor (Sum must be 1.0)
        # We include the node itself (lazy walk) to stabilize calculation
        mu_u = np.ones(len(nbrs_u)) / len(nbrs_u)
        mu_v = np.ones(len(nbrs_v)) / len(nbrs_v)

        # Create the Cost Matrix (Distance between every neighbor of u and every neighbor of v)
        M = np.zeros((len(nbrs_u), len(nbrs_v)))
        for i, nu in enumerate(nbrs_u):
            for j, nv in enumerate(nbrs_v):
                # Distance from path_lengths
                try:
                    M[i, j] = path_lengths[nu][nv]
                except KeyError:
                    M[i, j] = 999 # Should not happen in connected graph

        # Calculate Earth Mover's Distance (Wasserstein)
        # "How much work to move the ink drop from U's neighbors to V's neighbors?"
        emd = ot.emd2(mu_u, mu_v, M)

        # Geometric Distance (Hop distance is always 1 for an edge)
        d_uv = 1.0

        # Ricci Curvature
        kappa = 1.0 - (emd / d_uv)
        curvature_map[(u, v)] = kappa

    return curvature_map

def plot_general_relativity_check(G):
    # 1. Get T_mu_nu (Energy/Matter Density)
    # We use rho = |psi|^2
    matter_density = []

    # 2. Get R_mu_nu (Geometry/Curvature)
    # We average the edge curvatures to get a scalar value for the node
    edge_curvatures = compute_ollivier_ricci_flow(G)
    node_curvatures = {node: [] for node in G.nodes()}

    for (u, v), k in edge_curvatures.items():
        node_curvatures[u].append(k)
        node_curvatures[v].append(k)

    scalar_curvature = []

    # Align the lists
    for i in G.nodes():
        # Matter
        psi = G.nodes[i]["psi"]
        rho = float(psi.real**2 + psi.imag**2)
        matter_density.append(rho)

        # Geometry
        if node_curvatures[i]:
            R = np.mean(node_curvatures[i])
        else:
            R = 0.0
        scalar_curvature.append(R)

    # 3. The "Truth" Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(matter_density, scalar_curvature, alpha=0.6, c="teal", s=15, label="Node Data")

    # Fit a line (The "Gravitational Constant")
    m, b = np.polyfit(matter_density, scalar_curvature, 1)

    plt.plot(matter_density, m*np.array(matter_density) + b, color="orange", lw=2,
             label=f"Einstein Fit: R = {m:.3f}*T + {b:.3f}")

    plt.title(f"The Einstein Test (N={len(G)})", fontsize=14)
    plt.xlabel("Matter Density (T) -> |psi|^2", fontsize=12)
    plt.ylabel("Ricci Curvature (R) -> Geometry", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    print("-" * 40)
    print(f"RESULTS:")
    print(f"Slope (Gravitational Coupling G): {m:.4f}")
    print(f"Intercept (Cosmological Constant): {b:.4f}")
    print("-" * 40)

    if m < -0.0001:
         print(">> VERDICT: SUCCESS. Negative slope confirms Gravity (Mass creates Convergence).")
    elif m > 0.05:
         print(">> VERDICT: FAILURE. Positive slope implies Anti-Gravity.")
    else:
         print(">> VERDICT: NEUTRAL. No coupling detected.")

# --- TO RUN IT ---
# Assuming 'G' is your annealed N=1000 graph:
# plot_general_relativity_check(G)


# -----------------------
# INIT: BIG BANG (0 edges)
# -----------------------
def init_big_bang_graph():
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in G.nodes():
        G.nodes[i]["psi"] = np.random.normal(0.0, 0.1) + 1j * np.random.normal(0.0, 0.1)
    return G


# -----------------------
# UTILITIES: GAUGE
# -----------------------
def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi

def edge_key(i: int, j: int):
    return (i, j) if i < j else (j, i)

def oriented_theta(G: nx.Graph, i: int, j: int) -> float:
    a, b = edge_key(i, j)
    th = G[a][b]["theta"]
    return th if (i == a and j == b) else -th

def U_ij(G: nx.Graph, i: int, j: int) -> complex:
    return np.exp(1j * oriented_theta(G, i, j))

def apply_random_gauge_transform(G: nx.Graph, rng: np.random.Generator):
    alpha = {i: float(rng.uniform(-math.pi, math.pi)) for i in G.nodes()}
    for i in G.nodes():
        G.nodes[i]["psi"] *= np.exp(1j * alpha[i])
    for a, b in G.edges():
        G[a][b]["theta"] = wrap_angle(G[a][b]["theta"] + alpha[a] - alpha[b])


# -----------------------
# ACTION TERMS
# -----------------------
def rho_i(G: nx.Graph, i: int) -> float:
    psi = G.nodes[i]["psi"]
    return float(psi.real * psi.real + psi.imag * psi.imag)

def pauli_energy_node(G: nx.Graph, i: int) -> float:
    # exp(rho / rho0) penalty prevents hyper-dense nugget formation
    r = rho_i(G, i)
    return float(LAMBDA_PAULI * math.exp(r / max(1e-9, RHO0)))

def self_energy_node(G: nx.Graph, i: int) -> float:
    r = rho_i(G, i)
    # mass + quartic + Pauli
    return float(MASS2 * r + LAMBDA_PSI * (r * r) + pauli_energy_node(G, i))

def hop_energy_edge(G: nx.Graph, i: int, j: int) -> float:
    psi_i = G.nodes[i]["psi"]
    psi_j = G.nodes[j]["psi"]
    val = np.conjugate(psi_i) * U_ij(G, i, j) * psi_j
    return -KAPPA * float(np.real(val))

def triangle_energy(G: nx.Graph, i: int, j: int, k: int) -> float:
    loop = U_ij(G, i, j) * U_ij(G, j, k) * U_ij(G, k, i)
    return BETA * (1.0 - float(np.real(loop)))

def deg2_energy_node(G: nx.Graph, i: int, mu_deg2: float) -> float:
    d = G.degree(i)
    return float(mu_deg2 * (d * d))

def stress_energy_edge(G: nx.Graph, i: int, j: int) -> float:
    # "gravity" coupling to density along edges
    return float(LAMBDA_G * (rho_i(G, i) + rho_i(G, j)))

def lambda_e_total() -> float:
    # expansion bias + controller correction
    return float(LAMBDA_E_BASE + DELTA_LAMBDA_E)

def local_energy_around_node_for_psi(G: nx.Graph, i: int) -> float:
    e = self_energy_node(G, i)
    for j in G.neighbors(i):
        e += hop_energy_edge(G, i, j)
        e += stress_energy_edge(G, i, j)
    return float(e)

def local_energy_around_edge_for_theta(G: nx.Graph, i: int, j: int, adj_sets=None) -> float:
    a, b = edge_key(i, j)
    if not G.has_edge(a, b):
        return 0.0
    e = hop_energy_edge(G, i, j)
    common = adj_sets[i].intersection(adj_sets[j]) if adj_sets is not None else set(G.neighbors(i)).intersection(set(G.neighbors(j)))
    for k in common:
        e += triangle_energy(G, i, j, k)
    return float(e)

def local_energy_for_toggle_edge(G: nx.Graph, i: int, j: int, mu_deg2: float, adj_sets=None) -> float:
    # energy terms that change when toggling an edge (i,j)
    e = 0.0
    e += deg2_energy_node(G, i, mu_deg2) + deg2_energy_node(G, j, mu_deg2)
    e += lambda_e_total() * float(G.number_of_edges())

    a, b = edge_key(i, j)
    if G.has_edge(a, b):
        e += hop_energy_edge(G, i, j)
        e += stress_energy_edge(G, i, j)
        common = adj_sets[i].intersection(adj_sets[j]) if adj_sets is not None else set(G.neighbors(i)).intersection(set(G.neighbors(j)))
        for k in common:
            e += triangle_energy(G, i, j, k)
    return float(e)


# -----------------------
# METROPOLIS / MH
# -----------------------
def metropolis_accept(dS: float, temp: float) -> bool:
    if dS <= 0:
        return True
    if temp <= 1e-12:
        return False
    return random.random() < math.exp(-dS / temp)

def metropolis_hastings_accept(dS: float, log_qratio: float, temp: float) -> bool:
    if temp <= 1e-12:
        return dS <= 0
    exponent = (-dS / temp) + log_qratio
    if exponent >= 0:
        return True
    return random.random() < math.exp(exponent)


# -----------------------
# DISTANCES / SHELLS
# -----------------------
def bfs_distances(G: nx.Graph, source: int, max_d: int):
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_d:
            continue
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = du + 1
                if dist[v] <= max_d:
                    q.append(v)
    return dist

def shell_buckets(dist_map, max_d):
    buckets = [[] for _ in range(max_d + 1)]
    for v, d in dist_map.items():
        if 0 <= d <= max_d:
            buckets[d].append(v)
    return buckets


# -----------------------
# TRIANGLE / GAUGE PROBES
# -----------------------
def sample_triangles(G: nx.Graph, max_tri=1200):
    triangles = []
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for i in nodes:
        Ni = list(G.neighbors(i))
        if len(Ni) < 2:
            continue
        random.shuffle(Ni)
        for a in Ni[:12]:
            Na = set(G.neighbors(a))
            common = Na.intersection(Ni)
            for k in list(common)[:6]:
                tri = tuple(sorted((i, a, k)))
                triangles.append(tri)
                if len(triangles) >= max_tri:
                    return list(set(triangles))
    return list(set(triangles))

def triangle_flux_real(G: nx.Graph, tri):
    i, j, k = tri
    if not (G.has_edge(*edge_key(i, j)) and G.has_edge(*edge_key(j, k)) and G.has_edge(*edge_key(k, i))):
        return np.nan
    loop = U_ij(G, i, j) * U_ij(G, j, k) * U_ij(G, k, i)
    return float(np.real(loop))

def wilson_line_transporter(G: nx.Graph, path):
    prod = 1.0 + 0.0j
    for a, b in zip(path[:-1], path[1:]):
        prod *= U_ij(G, a, b)
    return prod

def k_shortest_paths_unweighted(G: nx.Graph, s: int, t: int, k: int):
    try:
        gen = nx.shortest_simple_paths(G, s, t)
        out = []
        for _ in range(k):
            out.append(next(gen))
        return out
    except Exception:
        return []

def gauge_invariant_two_point(G: nx.Graph, i: int, j: int, k_paths: int = 1) -> complex:
    psi_i = G.nodes[i]["psi"]
    psi_j = G.nodes[j]["psi"]
    denom = (abs(psi_i) * abs(psi_j) + 1e-12)
    if i == j:
        return (np.conjugate(psi_i) * psi_i) / denom
    try:
        if k_paths <= 1:
            path = nx.shortest_path(G, i, j)
            return (np.conjugate(psi_i) * wilson_line_transporter(G, path) * psi_j) / denom
        paths = k_shortest_paths_unweighted(G, i, j, k_paths)
        if not paths:
            return 0.0 + 0.0j
        vals = [np.conjugate(psi_i) * wilson_line_transporter(G, p) * psi_j for p in paths]
        return (np.mean(vals) / denom)
    except nx.NetworkXNoPath:
        return 0.0 + 0.0j


# -----------------------
# CORRELATOR
# -----------------------
def correlator_vs_distance_shellsample_fast(G, max_d, n_sources, samples_per_d_per_src, min_shell, seed=0, k_paths=1):
    rng = np.random.default_rng(seed)
    nodes = np.array(list(G.nodes()), dtype=int)
    bins = defaultdict(list)
    counts = np.zeros(max_d + 1, dtype=int)

    for _ in range(n_sources):
        i = int(rng.choice(nodes))
        dist_map = bfs_distances(G, i, max_d)
        shells = shell_buckets(dist_map, max_d)
        for d in range(1, max_d + 1):
            shell = shells[d]
            if len(shell) < min_shell:
                continue
            take = min(samples_per_d_per_src, len(shell))
            js = rng.choice(shell, size=take, replace=False)
            for j in js:
                val = gauge_invariant_two_point(G, i, int(j), k_paths=k_paths)
                bins[d].append(float(np.real(val)))
                counts[d] += 1

    ds = np.arange(max_d + 1, dtype=int)
    means = np.array([float(np.mean(bins[d])) if len(bins[d]) else np.nan for d in ds], dtype=float)
    return ds, means, counts

def estimate_corr_length_masked(ds, corr, counts, fit_min=2, fit_max=8, min_count=50):
    ds = np.asarray(ds, float)
    corr = np.asarray(corr, float)
    counts = np.asarray(counts, int)
    mask = (ds >= fit_min) & (ds <= fit_max) & (counts >= min_count) & np.isfinite(corr) & (np.abs(corr) > 1e-6)
    x = ds[mask]
    y = np.log(np.abs(corr[mask]))
    if len(x) < 3:
        return float("nan")
    Afit = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(Afit, y, rcond=None)[0]
    if m >= -1e-9:
        return float("nan")
    return float(-1.0 / m)


# -----------------------
# LIGHT CONE
# -----------------------
def lightcone_measurement(dist_map, baseline_abspsi, G: nx.Graph, max_d: int):
    buckets = [[] for _ in range(max_d + 1)]
    for v, d in dist_map.items():
        if d <= max_d:
            buckets[d].append(abs(G.nodes[v]["psi"]) - baseline_abspsi[v])
    out = np.zeros(max_d + 1, dtype=float)
    for d in range(max_d + 1):
        out[d] = float(np.mean(buckets[d])) if buckets[d] else 0.0
    return out

def quantile_radius(row_abs, q=0.8):
    row_abs = np.asarray(row_abs, float)
    tot = float(np.sum(row_abs))
    if tot <= 1e-12:
        return float("nan")
    target = q * tot
    c = 0.0
    for d, v in enumerate(row_abs):
        c += float(v)
        if c >= target:
            return float(d)
    return float(len(row_abs) - 1)

def fit_front_scaling(Rs, ts):
    ts = np.asarray(ts, float)
    Rs = np.asarray(Rs, float)
    if len(ts) < 5:
        return {}
    A1 = np.vstack([ts, np.ones_like(ts)]).T
    a1, b1 = np.linalg.lstsq(A1, Rs, rcond=None)[0]
    pred1 = a1 * ts + b1
    sse1 = float(np.sum((Rs - pred1) ** 2))

    st = np.sqrt(ts)
    A2 = np.vstack([st, np.ones_like(st)]).T
    a2, b2 = np.linalg.lstsq(A2, Rs, rcond=None)[0]
    pred2 = a2 * st + b2
    sse2 = float(np.sum((Rs - pred2) ** 2))
    return {"lin_a": float(a1), "lin_b": float(b1), "lin_sse": sse1,
            "sqrt_a": float(a2), "sqrt_b": float(b2), "sqrt_sse": sse2}


# -----------------------
# CURVATURE PROXY
# -----------------------
def forman_edge_curvature(G: nx.Graph, u: int, v: int) -> float:
    return 4.0 - float(G.degree(u)) - float(G.degree(v))

def node_curvature_proxy(G: nx.Graph, node: int) -> float:
    vals = [forman_edge_curvature(G, node, nbr) for nbr in G.neighbors(node)]
    return float(np.mean(vals)) if vals else 0.0

def pearson_r(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
    return float(np.dot(x, y) / denom)


# -----------------------
# QUENCH HELPERS
# -----------------------
def nodes_within_radius(G, center, radius):
    dist = bfs_distances(G, center, radius)
    return list(dist.keys()), dist

def curvature_profile_vs_distance(G, source, max_d):
    dist_map = bfs_distances(G, source, max_d)
    shells = shell_buckets(dist_map, max_d)
    prof = np.zeros(max_d + 1, dtype=float)
    counts = np.zeros(max_d + 1, dtype=int)
    for d in range(max_d + 1):
        counts[d] = len(shells[d])
        prof[d] = float(np.mean([node_curvature_proxy(G, v) for v in shells[d]])) if shells[d] else np.nan
    return prof, counts

def density_profile_vs_distance(G, source, max_d):
    dist_map = bfs_distances(G, source, max_d)
    shells = shell_buckets(dist_map, max_d)
    prof = np.zeros(max_d + 1, dtype=float)
    counts = np.zeros(max_d + 1, dtype=int)
    for d in range(max_d + 1):
        counts[d] = len(shells[d])
        prof[d] = float(np.mean([rho_i(G, v) for v in shells[d]])) if shells[d] else np.nan
    return prof, counts


# -----------------------
# MOVES
# -----------------------
def psi_update(G: nx.Graph):
    i = random.randrange(N)
    e0 = local_energy_around_node_for_psi(G, i)
    psi0 = G.nodes[i]["psi"]
    dpsi = np.random.normal(0.0, PSI_STEP) + 1j * np.random.normal(0.0, PSI_STEP)
    G.nodes[i]["psi"] = psi0 + dpsi
    e1 = local_energy_around_node_for_psi(G, i)
    if metropolis_accept(e1 - e0, TEMP):
        return True
    G.nodes[i]["psi"] = psi0
    return False

def theta_update(G: nx.Graph, adj_sets=None):
    if G.number_of_edges() == 0:
        return False
    a, b = random.choice(list(G.edges()))
    e0 = local_energy_around_edge_for_theta(G, a, b, adj_sets=adj_sets)
    th0 = G[a][b]["theta"]
    G[a][b]["theta"] = wrap_angle(th0 + float(np.random.normal(0.0, THETA_STEP)))
    e1 = local_energy_around_edge_for_theta(G, a, b, adj_sets=adj_sets)
    if metropolis_accept(e1 - e0, TEMP):
        return True
    G[a][b]["theta"] = th0
    return False


# -----------------------
# REWIRING kernels
# -----------------------
def total_pairs(n: int) -> int:
    return (n * (n - 1)) // 2

def sample_non_edge_pair(G: nx.Graph, rng: np.random.Generator, max_tries: int = 20000):
    for _ in range(max_tries):
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N - 1))
        if j >= i:
            j += 1
        a, b = edge_key(i, j)
        if not G.has_edge(a, b):
            return i, j
    return None

def rewire_mh_balanced_global(G: nx.Graph, mu_deg2: float, adj_sets, rng: np.random.Generator):
    E = G.number_of_edges()
    P = total_pairs(N)
    NE = P - E

    if E == 0:
        do_remove = False
    elif NE == 0:
        do_remove = True
    else:
        do_remove = (rng.random() < 0.5)

    if do_remove:
        a, b = random.choice(list(G.edges()))
        i, j = int(a), int(b)
        a, b = edge_key(i, j)
        th0 = G[a][b]["theta"]
        common = adj_sets[i].intersection(adj_sets[j])
        n_common = len(common)

        e0 = local_energy_for_toggle_edge(G, i, j, mu_deg2, adj_sets=adj_sets)

        G.remove_edge(a, b)
        adj_sets[i].discard(j); adj_sets[j].discard(i)

        e1 = local_energy_for_toggle_edge(G, i, j, mu_deg2, adj_sets=adj_sets)

        dS_tri = (GAMMA_T * float(n_common)) if (USE_TRIANGLE_REWARD and GAMMA_T != 0.0) else 0.0
        dS = (e1 - e0) + dS_tri

        E_after = E - 1
        NE_after = P - E_after
        q_fwd = 0.5 * (1.0 / max(1, E))
        q_rev = 0.5 * (1.0 / max(1, NE_after))
        log_qratio = math.log(q_rev) - math.log(q_fwd)

        if metropolis_hastings_accept(dS, log_qratio, TEMP):
            return True

        G.add_edge(a, b)
        G[a][b]["theta"] = th0
        adj_sets[i].add(j); adj_sets[j].add(i)
        return False

    pair = sample_non_edge_pair(G, rng)
    if pair is None:
        return False
    i, j = pair
    a, b = edge_key(i, j)

    common = adj_sets[i].intersection(adj_sets[j])
    n_common = len(common)

    e0 = local_energy_for_toggle_edge(G, i, j, mu_deg2, adj_sets=adj_sets)

    G.add_edge(a, b)
    G[a][b]["theta"] = float(rng.uniform(-math.pi, math.pi))
    adj_sets[i].add(j); adj_sets[j].add(i)

    e1 = local_energy_for_toggle_edge(G, i, j, mu_deg2, adj_sets=adj_sets)

    dS_tri = (-GAMMA_T * float(n_common)) if (USE_TRIANGLE_REWARD and GAMMA_T != 0.0) else 0.0
    dS = (e1 - e0) + dS_tri

    E_after = E + 1
    NE_after = P - E_after
    q_fwd = 0.5 * (1.0 / max(1, NE))
    q_rev = 0.5 * (1.0 / max(1, E_after))
    log_qratio = math.log(q_rev) - math.log(q_fwd)

    if metropolis_hastings_accept(dS, log_qratio, TEMP):
        return True

    if G.has_edge(a, b):
        G.remove_edge(a, b)
    adj_sets[i].discard(j); adj_sets[j].discard(i)
    return False

def rewire_symmetric_triadic_toggle(G: nx.Graph, mu_deg2: float, adj_sets, rng: np.random.Generator):
    k = int(rng.integers(0, N))
    Nk = list(adj_sets[k])
    if len(Nk) < 2:
        return False
    i, j = rng.choice(Nk, size=2, replace=False)
    i = int(i); j = int(j)
    if i == j:
        return False
    a, b = edge_key(i, j)
    existed = G.has_edge(a, b)

    common = adj_sets[i].intersection(adj_sets[j])
    n_common = len(common)

    e0 = local_energy_for_toggle_edge(G, i, j, mu_deg2, adj_sets=adj_sets)

    if existed:
        th0 = G[a][b]["theta"]
        G.remove_edge(a, b)
        adj_sets[i].discard(j); adj_sets[j].discard(i)
    else:
        th0 = None
        G.add_edge(a, b)
        G[a][b]["theta"] = float(rng.uniform(-math.pi, math.pi))
        adj_sets[i].add(j); adj_sets[j].add(i)

    e1 = local_energy_for_toggle_edge(G, i, j, mu_deg2, adj_sets=adj_sets)

    if USE_TRIANGLE_REWARD and GAMMA_T != 0.0:
        dS_tri = (GAMMA_T * float(n_common)) if existed else (-GAMMA_T * float(n_common))
    else:
        dS_tri = 0.0

    dS = (e1 - e0) + dS_tri

    if metropolis_accept(dS, TEMP):
        return True

    # rollback
    if existed:
        G.add_edge(a, b)
        G[a][b]["theta"] = th0
        adj_sets[i].add(j); adj_sets[j].add(i)
    else:
        if G.has_edge(a, b):
            G.remove_edge(a, b)
        adj_sets[i].discard(j); adj_sets[j].discard(i)
    return False

def rewire_move(G, mu_deg2, adj_sets, rng):
    if USE_TRIADIC_TOGGLE and (rng.random() < P_TRIADIC_TOGGLE):
        return rewire_symmetric_triadic_toggle(G, mu_deg2, adj_sets, rng)
    return rewire_mh_balanced_global(G, mu_deg2, adj_sets, rng)


# -----------------------
# MAIN RUN
# -----------------------
def run(show_plot=True):
    global MU_DEG2, DELTA_LAMBDA_E

    G = init_big_bang_graph()
    nodes = list(G.nodes())
    rng = np.random.default_rng(SEED + 123)
    adj_sets = {u: set() for u in G.nodes()}  # starts empty

    print(f">> Relational Physics v3.3.5 | N={N} | init edges={G.number_of_edges()}")
    print(f">> Expansion baseline λE_base={LAMBDA_E_BASE:.3f} | target <k>={DEG_TARGET:.2f}")

    eq_log = {"moves": [], "mean_deg": [], "tri_flux": [], "corr_d1": [],
              "acc_rewire": [], "triangles": [], "lambda_e_total": [], "delta_lambda_e": []}

    tri_samples = []

    acc_rew = tot_rew = 0
    integ_err = 0.0

    for t in range(1, EQUIL_MOVES + 1):
        r = random.random()
        if r < P_PSI:
            psi_update(G)
        elif r < P_PSI + P_THETA:
            theta_update(G, adj_sets=adj_sets)
        else:
            tot_rew += 1
            ok = rewire_move(G, MU_DEG2, adj_sets, rng)
            acc_rew += int(ok)

        # ---- PI control on mean degree (acts on DELTA_LAMBDA_E only)
        if USE_DEG_CONTROL and (t % DEG_CONTROL_EVERY == 0):
            md_now = float(np.mean([G.degree(i) for i in nodes]))
            err = md_now - DEG_TARGET  # >0 => too many edges => increase lambda_E (more penalty)

            d_p = DEG_KP * err

            at_min = (DELTA_LAMBDA_E <= DELTA_LAMBDA_E_MIN + 1e-12)
            at_max = (DELTA_LAMBDA_E >= DELTA_LAMBDA_E_MAX - 1e-12)
            allow_int = (not at_min and not at_max) or (at_min and err > 0) or (at_max and err < 0)

            if allow_int:
                integ_err = float(np.clip(integ_err + err * DEG_CONTROL_EVERY, -DEG_I_CLAMP, DEG_I_CLAMP))

            d_i = DEG_KI * integ_err
            DELTA_LAMBDA_E = float(np.clip(DELTA_LAMBDA_E + d_p + d_i,
                                           DELTA_LAMBDA_E_MIN, DELTA_LAMBDA_E_MAX))

        if t % LOG_EVERY == 0 or t == EQUIL_MOVES:
            md = float(np.mean([G.degree(i) for i in nodes]))

            if (t % (10 * LOG_EVERY) == 0) or (t == LOG_EVERY) or (len(tri_samples) < 200):
                tri_samples = sample_triangles(G, max_tri=2500)

            tri_vals = []
            for tri in tri_samples:
                v = triangle_flux_real(G, tri)
                if not np.isnan(v):
                    tri_vals.append(v)
                if len(tri_vals) >= 250:
                    break
            tri_flux = float(np.mean(tri_vals)) if tri_vals else 0.0

            if G.number_of_edges() > 0:
                es = random.sample(list(G.edges()), k=min(300, G.number_of_edges()))
                vals = []
                for a, b in es:
                    psi_a = G.nodes[a]["psi"]
                    psi_b = G.nodes[b]["psi"]
                    denom = (abs(psi_a) * abs(psi_b) + 1e-12)
                    vals.append(float(np.real(np.conjugate(psi_a) * U_ij(G, a, b) * psi_b) / denom))
                corr1 = float(np.mean(vals))
            else:
                corr1 = 0.0

            tri_count = sum(nx.triangles(G).values()) // 3 if G.number_of_edges() else 0
            ar = acc_rew / max(1, tot_rew)

            eq_log["moves"].append(t)
            eq_log["mean_deg"].append(md)
            eq_log["tri_flux"].append(tri_flux)
            eq_log["corr_d1"].append(corr1)
            eq_log["acc_rewire"].append(ar)
            eq_log["triangles"].append(tri_count)
            eq_log["lambda_e_total"].append(lambda_e_total())
            eq_log["delta_lambda_e"].append(DELTA_LAMBDA_E)

            print(f"  eq {t:6d}/{EQUIL_MOVES} | <k>={md:.2f} | tri#={tri_count:6d} | Re(loop)={tri_flux:.3f} "
                  f"| d=1 corr={corr1:.3f} | λE={lambda_e_total():+.3f} (Δ={DELTA_LAMBDA_E:+.3f}) | acc(rew)={ar:.2f}")

            acc_rew = tot_rew = 0

    print(">> Equilibration done.")
    tri_count_final = sum(nx.triangles(G).values()) // 3 if G.number_of_edges() else 0
    print(">> triangle count (exact):", tri_count_final)

    # Gauge invariance check
    print(">> Gauge invariance check...")
    tri_samples = sample_triangles(G, max_tri=2500)

    def tri_is_alive(tri):
        i, j, k = tri
        return (G.has_edge(*edge_key(i, j)) and
                G.has_edge(*edge_key(j, k)) and
                G.has_edge(*edge_key(k, i)))

    tri_check = [tri for tri in tri_samples if tri_is_alive(tri)][:300]
    pair_check = [tuple(rng.choice(nodes, size=2, replace=False)) for _ in range(250)]

    tri_before = np.array([triangle_flux_real(G, tri) for tri in tri_check], dtype=float)
    pair_before = np.array([gauge_invariant_two_point(G, i, j, k_paths=AVG_K_SHORTEST_PATHS) for i, j in pair_check], dtype=complex)

    apply_random_gauge_transform(G, rng)

    tri_after = np.array([triangle_flux_real(G, tri) for tri in tri_check], dtype=float)
    pair_after = np.array([gauge_invariant_two_point(G, i, j, k_paths=AVG_K_SHORTEST_PATHS) for i, j in pair_check], dtype=complex)

    tri_diff = float(np.nanmax(np.abs(tri_after - tri_before))) if len(tri_before) else 0.0
    pair_diff_re = float(np.max(np.abs(np.real(pair_after) - np.real(pair_before)))) if len(pair_before) else 0.0
    pair_diff_im = float(np.max(np.abs(np.imag(pair_after) - np.imag(pair_before)))) if len(pair_before) else 0.0
    print(f"  gauge: max |Δ loop| = {tri_diff:.3e}")
    print(f"  gauge: max |Δ corr Re| = {pair_diff_re:.3e} | max |Δ corr Im| = {pair_diff_im:.3e}")

    # Correlator
    print(">> Correlator vs hop distance...")
    ds, corr, corr_counts = correlator_vs_distance_shellsample_fast(
        G,
        max_d=MAX_DIST_CORR,
        n_sources=CORR_SOURCES,
        samples_per_d_per_src=CORR_SAMPLES_PER_D_PER_SRC,
        min_shell=CORR_MIN_SHELL,
        seed=SEED + 999,
        k_paths=AVG_K_SHORTEST_PATHS,
    )
    xi_est = estimate_corr_length_masked(ds, corr, corr_counts,
                                         fit_min=XI_FIT_MIN,
                                         fit_max=min(XI_FIT_MAX, MAX_DIST_CORR),
                                         min_count=XI_MIN_COUNT)

    # Curvature–density
    if G.number_of_edges():
        comp = max(nx.connected_components(G), key=len)
    else:
        comp = set(G.nodes())
    sample_nodes = random.sample(list(comp), k=min(CURV_SAMPLE, len(comp)))
    rho = np.array([rho_i(G, i) for i in sample_nodes], dtype=float)
    curv = np.array([node_curvature_proxy(G, i) for i in sample_nodes], dtype=float)
    r_rho_curv = pearson_r(rho, curv)

    # Quench
    r_resp = float("nan")
    delta_curv_prof = delta_rho_prof = None
    shell_counts_quench = None

    if DO_QUENCH and G.number_of_edges():
        print(">> Quench response (Δρ vs Δcurv)...")
        comp_list = list(comp)
        quench_src = random.choice(comp_list)

        curv_before, sc1 = curvature_profile_vs_distance(G, quench_src, MAX_DIST_QUENCH)
        rho_before, sc2 = density_profile_vs_distance(G, quench_src, MAX_DIST_QUENCH)
        shell_counts_quench = np.minimum(sc1, sc2)

        region_nodes, _ = nodes_within_radius(G, quench_src, QUENCH_RADIUS)
        for v in region_nodes:
            G.nodes[v]["psi"] += (QUENCH_STRENGTH + 0.0j)

        # relax with geometry frozen
        for _ in range(QUENCH_RELAX_MOVES_STAGE1):
            rr = random.random()
            if rr < (P_PSI / (P_PSI + P_THETA)):
                psi_update(G)
            else:
                theta_update(G, adj_sets=adj_sets)

        MU0 = MU_DEG2
        dLE0 = DELTA_LAMBDA_E
        if QUENCH_TWO_STAGE:
            MU_DEG2 = MU_QUENCH
            if DELTA_LAMBDA_E_QUENCH is not None:
                DELTA_LAMBDA_E = DELTA_LAMBDA_E_QUENCH

        for _ in range(QUENCH_RELAX_MOVES_STAGE2):
            rr = random.random()
            if QUENCH_ALLOW_REWIRE and rr < QUENCH_P_REWIRE:
                rewire_move(G, MU_DEG2, adj_sets, rng)
            else:
                if rr < (P_PSI / (P_PSI + P_THETA)):
                    psi_update(G)
                else:
                    theta_update(G, adj_sets=adj_sets)

        MU_DEG2 = MU0
        DELTA_LAMBDA_E = dLE0

        curv_after, sc3 = curvature_profile_vs_distance(G, quench_src, MAX_DIST_QUENCH)
        rho_after, sc4 = density_profile_vs_distance(G, quench_src, MAX_DIST_QUENCH)
        shell_counts_quench = np.minimum.reduce([shell_counts_quench, sc3, sc4])

        delta_curv_prof = curv_after - curv_before
        delta_rho_prof = rho_after - rho_before

        mask = np.isfinite(delta_rho_prof) & np.isfinite(delta_curv_prof) & (shell_counts_quench >= MIN_SHELL_QUENCH)
        if np.sum(mask) >= 3:
            r_resp = pearson_r(delta_rho_prof[mask], delta_curv_prof[mask])
        print(f"   quench r(Δρ,Δcurv) ≈ {r_resp:.3f}")

    # Light cone
    print(">> Light cone (geometry frozen)...")
    src = random.choice(list(comp))
    dist_map = bfs_distances(G, src, MAX_DIST_LIGHTCONE)
    baseline_abspsi = {i: abs(G.nodes[i]["psi"]) for i in G.nodes()}
    G.nodes[src]["psi"] += (PERTURB_EPS + 0.0j)

    cone_times, cone_matrix = [], []
    for t in range(1, PROP_MOVES + 1):
        rr = random.random()
        if rr < (P_PSI / (P_PSI + P_THETA)):
            psi_update(G)
        else:
            theta_update(G, adj_sets=adj_sets)
        if t % LOG_EVERY == 0 or t == PROP_MOVES:
            cone_times.append(t)
            cone_matrix.append(lightcone_measurement(dist_map, baseline_abspsi, G, MAX_DIST_LIGHTCONE))

    cone_matrix = np.array(cone_matrix)
    A = np.abs(cone_matrix)
    dgrid = np.arange(A.shape[1], dtype=float)

    Rs_q, ts_q = [], []
    for ti, t in enumerate(cone_times):
        Rq = quantile_radius(A[ti], q=FRONT_Q)
        if np.isfinite(Rq):
            Rs_q.append(Rq); ts_q.append(t)
    fit_q = fit_front_scaling(Rs_q, ts_q) if len(ts_q) >= 5 else {}
    if fit_q:
        print(f"   front fit q={FRONT_Q:.2f}: linear SSE={fit_q['lin_sse']:.3g}, sqrt SSE={fit_q['sqrt_sse']:.3g}")

    md_final = float(np.mean([G.degree(i) for i in G.nodes()]))
    print(">> Summary:")
    print(f"   <k>={md_final:.2f} target={DEG_TARGET:.2f} | λE_total={lambda_e_total():+.3f} (base={LAMBDA_E_BASE:+.3f}, Δ={DELTA_LAMBDA_E:+.3f})")
    print(f"   backreaction r(ρ,curv) ≈ {r_rho_curv:.3f}")
    print(f"   xi ≈ {xi_est:.3f} hops")
    print(f"   triangles = {tri_count_final}")

    if not show_plot:
        return G

    # -------- plots --------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    axs[0, 0].plot(eq_log["moves"], eq_log["mean_deg"], label="Mean degree")
    axs[0, 0].plot(eq_log["moves"], eq_log["tri_flux"], label="Mean Re(triangle loop)")
    axs[0, 0].plot(eq_log["moves"], eq_log["corr_d1"], label="Mean d=1 corr (Re, norm)")
    axs[0, 0].plot(eq_log["moves"], eq_log["acc_rewire"], label="Acc(rewire)")
    axs[0, 0].plot(eq_log["moves"], eq_log["lambda_e_total"], label="λE_total")
    axs[0, 0].plot(eq_log["moves"], eq_log["delta_lambda_e"], label="ΔλE (PI)")
    axs[0, 0].set_title("Equilibration diagnostics")
    axs[0, 0].set_xlabel("Moves")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(ds, corr, marker="o")
    axs[0, 1].set_title("Gauge-invariant correlator vs hop distance")
    axs[0, 1].set_xlabel("Hop distance d")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].set_title("Quench response: Δρ(d) and Δcurv(d)")
    axs[0, 2].set_xlabel("Hop distance d")
    axs[0, 2].grid(True, alpha=0.3)
    if DO_QUENCH and (delta_curv_prof is not None) and (delta_rho_prof is not None):
        dgrid_qc = np.arange(len(delta_curv_prof))
        axs[0, 2].plot(dgrid_qc, delta_rho_prof, marker="o", label="Δρ(d)")
        axs[0, 2].plot(dgrid_qc, delta_curv_prof, marker="o", label="Δcurv(d)")
        axs[0, 2].legend()

    im = axs[1, 0].imshow(
        cone_matrix,
        aspect="auto",
        origin="lower",
        extent=[0, MAX_DIST_LIGHTCONE, cone_times[0], cone_times[-1]],
    )
    axs[1, 0].set_title("Light cone: Δ|psi| vs distance/time")
    axs[1, 0].set_xlabel("Hop distance from source")
    axs[1, 0].set_ylabel("Moves (time)")
    plt.colorbar(im, ax=axs[1, 0], label="Mean Δ|psi|")

    axs[1, 1].scatter(rho, curv, s=18, alpha=0.7)
    axs[1, 1].set_title("Curvature–density diagnostic")
    axs[1, 1].set_xlabel("ρ = |psi|^2")
    axs[1, 1].set_ylabel("Node curvature proxy")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].axis("off")
    axs[1, 2].text(
        0.02, 0.95,
        "Key checks:\n"
        f"- Gauge: maxΔloop={tri_diff:.2e}\n"
        f"- Gauge: maxΔcorr(Re)={pair_diff_re:.2e}, maxΔcorr(Im)={pair_diff_im:.2e}\n"
        f"- Backreaction: r(ρ,curv)≈{r_rho_curv:.2f}\n"
        f"- Quench: r(Δρ,Δcurv)≈{r_resp:.2f}\n"
        f"- xi≈{xi_est:.2f} | triangles={tri_count_final}\n"
        f"- <k>≈{md_final:.2f} | λE_total={lambda_e_total():+.3f}\n"
        f"- Pauli: λP={LAMBDA_PAULI}, ρ0={RHO0}\n",
        va="top",
        fontsize=10,
    )

    fig.suptitle(
        f"Relational Physics v3.3.5 | N={N} | κ={KAPPA}, β={BETA}, T={TEMP}, μ={MU_DEG2:.3g}, λG={LAMBDA_G}\n"
        f"λE_base={LAMBDA_E_BASE:+.3g} + Δ={DELTA_LAMBDA_E:+.3g} | Pauli λP={LAMBDA_PAULI}, ρ0={RHO0}",
        fontsize=12,
    )
    #plt.show()

    return G


# -----------------------
# Emergent Geometry (Spring + MDS)
# -----------------------
def plot_emergent_geometry(G: nx.Graph):
    print(">>> Calculating Emergent Geometry...")
    pos_spring = nx.spring_layout(G, k=0.15, iterations=50)

    dist_matrix = nx.floyd_warshall_numpy(G)
    finite = np.isfinite(dist_matrix)
    if np.any(finite):
        dist_matrix[~finite] = dist_matrix[finite].max() * 2
    else:
        dist_matrix[:] = 1.0

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED)
    pos_mds = mds.fit_transform(dist_matrix)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    nx.draw_networkx_edges(G, pos_spring, ax=axs[0], alpha=0.03, edge_color="gray")
    nx.draw_networkx_nodes(
        G, pos_spring, ax=axs[0],
        node_size=5,
        node_color=[d for _, d in G.degree()],
        cmap="plasma"
    )
    axs[0].set_title("Physical Topology: Spring-Mass Embedding")
    axs[0].axis("off")

    sc = axs[1].scatter(
        pos_mds[:, 0], pos_mds[:, 1],
        s=10, alpha=0.6,
        c=[rho_i(G, i) for i in G.nodes()],
        cmap="viridis"
    )
    plt.colorbar(sc, ax=axs[1], label="Matter Density ρ=|psi|²")
    axs[1].set_title("MDS Projection: The Emergent Manifold")
    axs[1].axis("off")

    plt.show()


if __name__ == "__main__":
    print(">>> STARTING TOPOLOGICAL BIG BANG (v3.3.5)...")
    G = run(show_plot=True)
    plot_emergent_geometry(G)
    plot_general_relativity_check(G)





