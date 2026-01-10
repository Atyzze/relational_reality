import ot  # Requires: pip install POT
import os, math, random, time, datetime, shutil, threading, warnings
import scipy.sparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from sklearn.manifold import MDS
from collections import defaultdict, deque

# CONFIG

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

N = 1000
STEP_COUNT  = 1_000_000
LOG_EVERY   = 10_000
VISUALIZE_STEPS = False
VISUALIZE_ALL_NODES = False


# Move mix
P_PSI    = 0.55
P_THETA  = 0.35
P_REWIRE = 0.10

# Temperature + steps
TEMP       = 0.20
PSI_STEP   = 0.25
THETA_STEP = 0.35

# Action parameters / geometry / degree costs
LAMBDA_E_BASE = -2.50
LAMBDA_G      = 0.60
LAMBDA_PSI    = 0.08
KAPPA         = 0.90
BETA          = 0.90
MASS2         = 0.35

LAMBDA_PAULI  = 0.05   
RHO0          = 4.0    
MU_DEG2       = 0.1  

# Rewire proposal mix
USE_TRIADIC_TOGGLE = True
P_TRIADIC_TOGGLE   = 0.65 

#parameters below are solely for diagnostics and dont affect the actual graph evolution

# Correlator
MAX_DIST_CORR = 20
CORR_SOURCES = 260
CORR_SAMPLES_PER_D_PER_SRC = 2
CORR_MIN_SHELL = 5
AVG_K_SHORTEST_PATHS = 1

# Light cone
PROPAGATE_MOVES  = 10_000
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
QUENCH_ALLOW_REWIRE = True
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

def analyze_dimensionality(G, plot=True):
    N = G.number_of_nodes()
    centers = np.random.choice(list(G.nodes()), size=min(10, N), replace=False)

    rs_all = []
    counts_all = []

    for source in centers:
        dists = dict(nx.single_source_shortest_path_length(G, source))
        max_d = max(dists.values())

        # Cumulative count N(r)
        counts = np.zeros(max_d + 1)
        for d in dists.values():
            counts[d] += 1
        cumulative = np.cumsum(counts)

        valid_r = np.arange(1, len(cumulative))
        rs_all.extend(valid_r)
        counts_all.extend(cumulative[1:])

    # Log-Log Fit
    log_r = np.log(rs_all)
    log_n = np.log(counts_all)

    mask = (log_r > np.log(1.5)) & (log_r < np.log(8))

    d_H = np.nan
    if np.sum(mask) > 5:
        slope, intercept = np.polyfit(log_r[mask], log_n[mask], 1)
        d_H = slope

    # --- 2. Spectral Dimension (via Laplacian Eigenvalues) ---
    # L = D - A
    L = nx.laplacian_matrix(G).astype(float)

    evals = scipy.linalg.eigh(L.todense(), eigvals_only=True)

    # Heat Kernel Trace: P(t) = (1/N) * sum(exp(-lambda * t))
    # We scan t from small to large
    t_vals = np.logspace(-1, 2, 50)
    p_t = []
    for t in t_vals:
        # Sum of exponentials
        trace = np.sum(np.exp(-evals * t))
        p_t.append(trace / N)

    p_t = np.array(p_t)

    # Slope of log(P(t)) vs log(t) is -dS/2
    # We look for a linear region in log-log
    log_t = np.log(t_vals)
    log_p = np.log(p_t)

    # For N=1000, valid region is usually t=0.5 to t=5.0
    mask_spec = (t_vals > 0.5) & (t_vals < 5.0)

    d_S = np.nan
    if np.sum(mask_spec) > 5:
        slope_s, _ = np.polyfit(log_t[mask_spec], log_p[mask_spec], 1)
        d_S = -2 * slope_s

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Hausdorff Plot
        ax[0].scatter(log_r, log_n, alpha=0.1, color='k')
        if not np.isnan(d_H):
            ax[0].plot(log_r[mask], slope*log_r[mask] + intercept, 'r-', lw=2)
            ax[0].set_title(f"Hausdorff: Volume ~ r^{d_H:.2f}")
        ax[0].set_xlabel("log(radius)")
        ax[0].set_ylabel("log(cumulative nodes)")

        # Spectral Plot
        ax[1].plot(log_t, log_p, 'k.-')
        if not np.isnan(d_S):
            ax[1].plot(log_t[mask_spec], slope_s*log_t[mask_spec] + _, 'r-', lw=2)
            ax[1].set_title(f"Spectral: P(t) ~ t^{{-{d_S:.2f}/2}}")
        ax[1].set_xlabel("log(diffusion time)")
        ax[1].set_ylabel("log(return prob)")

        plt.tight_layout()
        plt.savefig(FRAMES_DIR+"/N="+str(N)+"_S="+str(STEP_COUNT) + "_d.png", dpi=300, bbox_inches="tight")
        plt.close()

    return d_H, d_S

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
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    for u, v in G.edges():
        # Define the "Mass Distributions" (Ink Drops) at u and v
        # We assume uniform distribution over neighbors (Standard Ricci)
        nbrs_u = [u] + list(G.neighbors(u))
        nbrs_v = [v] + list(G.neighbors(v))
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
    plt.tight_layout()
    plt.savefig(FRAMES_DIR+"/N="+str(N)+"_S="+str(STEP_COUNT) + "_g.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("-" * 40)
    print(f"RESULTS:")
    print(f"Slope (Gravitational Coupling G): {m:.4f}")
    print(f"Intercept (Cosmological Constant): {b:.4f}")
    print("-" * 40)


def init_graph():
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
    return float(LAMBDA_E_BASE)

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


        dS = (e1 - e0)


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


    dS = (e1 - e0)

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



    dS = (e1 - e0)

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
def run():
    global MU_DEG2, DELTA_LAMBDA_E
    G = init_graph()
    nodes = list(G.nodes())
    rng = np.random.default_rng(SEED + 123)
    adj_sets = {u: set() for u in G.nodes()}  # starts empty

    eq_log = {"moves": [], "mean_deg": [], "tri_flux": [], "corr_d1": [],
              "acc_rewire": [], "triangles": [], "lambda_e_total": [], "delta_lambda_e": []}

    tri_samples = []

    acc_rew = tot_rew = 0
    integ_err = 0.0

    start_time = time.time()

    for t in range(1, STEP_COUNT + 1):
        r = random.random()
        if r < P_PSI:
            psi_update(G)
        elif r < P_PSI + P_THETA:
            theta_update(G, adj_sets=adj_sets)
        else:
            tot_rew += 1
            ok = rewire_move(G, MU_DEG2, adj_sets, rng)
            acc_rew += int(ok)



        if t % LOG_EVERY == 0 or t == STEP_COUNT:
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
            elapsed = time.time() - start_time
            sps = t / max(1e-9, elapsed)  # steps per second
            eta = (STEP_COUNT - t) / sps # estimated seconds remaining

            eq_log["moves"].append(t)
            eq_log["mean_deg"].append(md)
            eq_log["tri_flux"].append(tri_flux)
            eq_log["corr_d1"].append(corr1)
            eq_log["acc_rewire"].append(ar)
            eq_log["triangles"].append(tri_count)
            eq_log["lambda_e_total"].append(lambda_e_total())

            print(f"{t/1e6:.3f}/{STEP_COUNT/1e6:.1f}M | <k>={md:.3f} | tri#={tri_count:6d} | Re(loop)={tri_flux:.3f} "
                  f"| d=1 corr={corr1:.3f}  acc(rew)={ar:.3f} | {sps:.0f} it/s | ETA {eta:.0f}s" )


            acc_rew = tot_rew = 0
            if VISUALIZE_STEPS:
                plot_emergent_geometry(G, t)





    #main loop done
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
        seed=SEED,
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

   # -----------------------
    # QUENCH (ISOLATED COPY)
    # -----------------------
    r_resp = float("nan")
    delta_curv_prof = delta_rho_prof = None
    shell_counts_quench = None

    if DO_QUENCH and G.number_of_edges():
        print(">> Quench response (Δρ vs Δcurv) [Running on Copy]...")

        # 1. Create a Deep Copy of the Universe
        G_quench = G.copy()

        # 2. Rebuild adj_sets for the copy (Crucial for rewiring to work)
        adj_sets_quench = {n: set(G_quench.neighbors(n)) for n in G_quench.nodes()}

        comp_list = list(max(nx.connected_components(G_quench), key=len))
        quench_src = random.choice(comp_list)

        # Baseline profiles (measured on the copy)
        curv_before, sc1 = curvature_profile_vs_distance(G_quench, quench_src, MAX_DIST_QUENCH)
        rho_before, sc2 = density_profile_vs_distance(G_quench, quench_src, MAX_DIST_QUENCH)
        shell_counts_quench = np.minimum(sc1, sc2)

        # 3. Inject Energy into the Copy
        region_nodes, _ = nodes_within_radius(G_quench, quench_src, QUENCH_RADIUS)
        for v in region_nodes:
            G_quench.nodes[v]["psi"] += (QUENCH_STRENGTH + 0.0j)

        # 4. Relax Stage 1 (Geometry Frozen)
        # Note: We pass G_quench and adj_sets_quench
        for _ in range(QUENCH_RELAX_MOVES_STAGE1):
            rr = random.random()
            if rr < (P_PSI / (P_PSI + P_THETA)):
                psi_update(G_quench)
            else:
                theta_update(G_quench, adj_sets=adj_sets_quench)

        MU0 = MU_DEG2
        if QUENCH_TWO_STAGE:
            MU_DEG2 = MU_QUENCH
            if DELTA_LAMBDA_E_QUENCH is not None:
                DELTA_LAMBDA_E = DELTA_LAMBDA_E_QUENCH

        # 5. Relax Stage 2 (Geometry Active)
        for _ in range(QUENCH_RELAX_MOVES_STAGE2):
            rr = random.random()
            if QUENCH_ALLOW_REWIRE and rr < QUENCH_P_REWIRE:
                # Use the copy's adjacency sets!
                rewire_move(G_quench, MU_DEG2, adj_sets_quench, rng)
            else:
                if rr < (P_PSI / (P_PSI + P_THETA)):
                    psi_update(G_quench)
                else:
                    theta_update(G_quench, adj_sets=adj_sets_quench)

        # Restore globals (though they didn't affect G, they affect the config)
        MU_DEG2 = MU0

        # 6. Measure Response on the Copy
        curv_after, sc3 = curvature_profile_vs_distance(G_quench, quench_src, MAX_DIST_QUENCH)
        rho_after, sc4 = density_profile_vs_distance(G_quench, quench_src, MAX_DIST_QUENCH)

        # Combine counts
        shell_counts_quench = np.minimum.reduce([shell_counts_quench, sc3, sc4])

        delta_curv_prof = curv_after - curv_before
        delta_rho_prof = rho_after - rho_before

        mask = np.isfinite(delta_rho_prof) & np.isfinite(delta_curv_prof) & (shell_counts_quench >= MIN_SHELL_QUENCH)
        if np.sum(mask) >= 3:
            r_resp = pearson_r(delta_rho_prof[mask], delta_curv_prof[mask])

        print(f"   quench r(Δρ,Δcurv) ≈ {r_resp:.3f}")

        # G_quench is discarded here; the main G remains pristine for the Light Cone test.

    # Light cone
    print(">> Light cone (geometry frozen)...")
    src = random.choice(list(comp))
    dist_map = bfs_distances(G, src, MAX_DIST_LIGHTCONE)
    baseline_abspsi = {i: abs(G.nodes[i]["psi"]) for i in G.nodes()}
    #G.nodes[src]["psi"] += (PERTURB_EPS + 0.0j)
    G.nodes[src]["psi"] *= (1.0 + PERTURB_EPS)
    cone_times, cone_matrix = [], []

    # NEW: Define a specific logging interval for the light cone
    LC_LOG_STEP = 100

    for t in range(1, PROPAGATE_MOVES + 1):
        rr = random.random()
        if rr < (P_PSI / (P_PSI + P_THETA)):
            psi_update(G)
        else:
            theta_update(G, adj_sets=adj_sets)

        # CHANGED: Use LC_LOG_STEP instead of global LOG_EVERY
        if t % LC_LOG_STEP == 0:
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
    print(f"   <k>={md_final:.2f} | λE_total={lambda_e_total():+.3f} ")
    print(f"   backreaction r(ρ,curv) ≈ {r_rho_curv:.3f}")
    print(f"   xi ≈ {xi_est:.3f} hops")
    print(f"   triangles = {tri_count_final}")


    # -------- plots --------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. Equilibration (Dual Axis)
    # Primary Y-axis: Degree, Flux, Correlation, Acceptance
    ln1 = axs[0, 0].plot(eq_log["moves"], eq_log["mean_deg"], label="Mean degree", color="tab:blue")
    ln2 = axs[0, 0].plot(eq_log["moves"], eq_log["tri_flux"], label="Mean Re(loop)", color="tab:orange")
    ln3 = axs[0, 0].plot(eq_log["moves"], eq_log["corr_d1"], label="d=1 corr", color="tab:green")
    ln4 = axs[0, 0].plot(eq_log["moves"], eq_log["acc_rewire"], label="Acc(rewire)", color="tab:red", linestyle="--")

    axs[0, 0].set_xlabel("Moves")
    axs[0, 0].set_ylabel("Degree / Flux / Prob")
    axs[0, 0].set_title("Equilibration: Topology vs Triangles")
    axs[0, 0].grid(True, alpha=0.3)

    # Secondary Y-axis (Right): Triangle Count
    ax_tri = axs[0, 0].twinx()
    ln5 = ax_tri.plot(eq_log["moves"], eq_log["triangles"], label="Total Triangles", color="tab:purple", linewidth=2, alpha=0.5)
    ax_tri.set_ylabel("Triangle Count")

    # Combined Legend
    lns = ln1 + ln2 + ln3 + ln4 + ln5
    labs = [l.get_label() for l in lns]
    axs[0, 0].legend(lns, labs, loc="center right")

    # 2. Correlator
    axs[0, 1].plot(ds, corr, marker="o")
    axs[0, 1].set_title("Gauge-invariant correlator vs hop distance")
    axs[0, 1].set_xlabel("Hop distance d")
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Quench Response
    axs[0, 2].set_title("Quench response: Δρ(d) and Δcurv(d)")
    axs[0, 2].set_xlabel("Hop distance d")
    axs[0, 2].grid(True, alpha=0.3)
    if DO_QUENCH and (delta_curv_prof is not None) and (delta_rho_prof is not None):
        dgrid_qc = np.arange(len(delta_curv_prof))
        axs[0, 2].plot(dgrid_qc, delta_rho_prof, marker="o", label="Δρ(d)")
        axs[0, 2].plot(dgrid_qc, delta_curv_prof, marker="o", label="Δcurv(d)")
        axs[0, 2].legend()

    # 4. Light Cone
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

    # 5. Curvature vs Density
    axs[1, 1].scatter(rho, curv, s=18, alpha=0.7)
    axs[1, 1].set_title("Curvature–density diagnostic")
    axs[1, 1].set_xlabel("ρ = |psi|^2")
    axs[1, 1].set_ylabel("Node curvature proxy")
    axs[1, 1].grid(True, alpha=0.3)

    # 6. Text Stats
    axs[1, 2].axis("off")
    axs[1, 2].text(
        0.02, 0.95,
        "Key checks:\n"
        f"- Gauge: maxΔloop={tri_diff:.2e}\n"
        f"- Gauge: maxΔcorr(Re)={pair_diff_re:.2e}, maxΔcorr(Im)={pair_diff_im:.2e}\n"
        f"- Backreaction: r(ρ,curv)≈{r_rho_curv:.2f}\n"
        f"- Quench: r(Δρ,Δcurv)≈{r_resp:.2f}\n"
        f"- xi≈{xi_est:.2f} | triangles={tri_count_final}\n"
        f"- <k>≈{md_final:.2f} | λE_base={LAMBDA_E_BASE:+.3g}\n"
        f"- Pauli: λP={LAMBDA_PAULI}, ρ0={RHO0}\n",
        va="top",
        fontsize=10,
    )

    fig.suptitle(
        f"Relational Physics | N={N} | κ={KAPPA}, β={BETA}, T={TEMP}, μ={MU_DEG2:.3g}, λG={LAMBDA_G}\n"
        f"λE_base={LAMBDA_E_BASE:+.3g} | Pauli λP={LAMBDA_PAULI}, ρ0={RHO0}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(FRAMES_DIR+"/N="+str(N)+"_S="+str(STEP_COUNT) + "_a.png", dpi=300, bbox_inches="tight")
    plt.close()

    return G

# Emergent Geometry (Spring + MDS)
# -----------------------
def plot_emergent_geometry(G: nx.Graph, step):
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    if VISUALIZE_ALL_NODES:
        G_vis = G
    else:
        G_vis = G.subgraph(largest_cc_nodes).copy()

    pos_spring = nx.spring_layout(G_vis, k=0.15, iterations=50)

    if G_vis.number_of_edges() == 0:
        pos_mds = np.zeros((len(G_vis.nodes()), 2))
    else:
        dist_matrix = nx.floyd_warshall_numpy(G_vis)
        finite = np.isfinite(dist_matrix)

        if np.any(finite):
            flat_finite = dist_matrix[finite]
            # Avoid zero-max error if only self-loops exist
            max_d = flat_finite.max() if flat_finite.max() > 0 else 1.0
            dist_matrix[~finite] = max_d * 2.0
        else:
            dist_matrix[:] = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=SEED,
                n_init=4,
                normalized_stress="auto"
            )
            pos_mds = mds.fit_transform(dist_matrix)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Topology
    nx.draw_networkx_edges(G_vis, pos_spring, ax=axs[0], alpha=0.03, edge_color="gray")
    nx.draw_networkx_nodes(
        G_vis, pos_spring, ax=axs[0],
        node_size=5,
        node_color=[d for _, d in G_vis.degree()],
        cmap="plasma"
    )
    axs[0].set_title("Physical Topology: Spring-Mass Embedding")
    axs[0].axis("off")

    # Right: Emergent Geometry
    sc = axs[1].scatter(
        pos_mds[:, 0], pos_mds[:, 1],
        s=10, alpha=0.6,
        c=[rho_i(G_vis, i) for i in G_vis.nodes()],
        cmap="viridis"
    )
    plt.colorbar(sc, ax=axs[1], label="Matter Density ρ=|psi|²")
    axs[1].set_title("MDS Projection: The Emergent Manifold")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(FRAMES_DIR + "/N="+str(N)+"_S="+str(step) + "_m.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    FRAMES_DIR = f"runs/{timestamp}_N{N}_S{STEP_COUNT}_s{SEED}"
    os.makedirs(FRAMES_DIR, exist_ok=True)
    shutil.copy2(__file__, FRAMES_DIR)
    print(f">> Output directory created: {FRAMES_DIR}")

    G = run() #main loop, this can take a long while depending on N and step count.
    print(f">> Calculacting general relativity.. .")
    plot_general_relativity_check(G)
    print(f">> Calculacting dimensionality.. .")
    analyze_dimensionality(G)
    print(f">> Calculating geometry.. .")
    plot_emergent_geometry(G, STEP_COUNT)
    print(f">> Finished, find all data in folder: {FRAMES_DIR}")
