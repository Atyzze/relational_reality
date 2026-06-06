"""
metrics.graph_topology — Topology snapshots
===========================================
get_graph_stats and friends. Pure, fast, side-effect-free graph
measurements used by the orchestrator for every cell.

Public surface:
    GraphStats              namedtuple with field names
    get_graph_stats         user-facing wrapper returning GraphStats
    compute_graph_stats     njit-compiled core (returns 9-tuple)
"""

from collections import namedtuple
import numpy as np
from numba import njit

# ═══════════════════════════════════════════════════════════════════
#  Named return type for compute_graph_stats
# ═══════════════════════════════════════════════════════════════════
GraphStats = namedtuple('GraphStats', [
    'k_min', 'k_avg', 'k_max', 'triangles', 'edges',
    'tri_per_edge', 'transitivity', 'assortativity', 'lcc_pct',
])


def get_graph_stats(node_neighbors, node_degrees):
    """Wrapper around njit compute_graph_stats that returns a GraphStats namedtuple."""
    raw = compute_graph_stats(node_neighbors, node_degrees)
    return GraphStats(*raw)

@njit(cache=True)
def compute_graph_stats(node_neighbors, node_degrees):
    """Numba-compiled calculation of degree stats, triangles, transitivity, assortativity, and LCC.

    Returns 9-tuple: (k_min, k_avg, k_max, triangles, edges, tri_per_edge,
                       transitivity, assortativity, lcc_pct)

    For named access, use get_graph_stats() which wraps this in a GraphStats namedtuple.
    """
    N = len(node_degrees)
    if N == 0:
        return 0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0

    k_min = N
    k_max = 0
    k_sum = 0
    edges = 0
    triangles = 0

    sum_de = 0.0
    sum_de2 = 0.0
    sum_de_de = 0.0
    triplets = 0

    # 1. Standard Metrics Pass
    for u in range(N):
        d_u = node_degrees[u]
        if d_u < k_min: k_min = d_u
        if d_u > k_max: k_max = d_u
        k_sum += d_u
        triplets += d_u * (d_u - 1) // 2

        for i in range(d_u):
            v = node_neighbors[u, i]
            if u < v:
                edges += 1
                d_v = node_degrees[v]

                # Assortativity accumulators
                sum_de += (d_u + d_v)
                sum_de2 += (d_u**2 + d_v**2)
                sum_de_de += (d_u * d_v)

                # Triangle counting (brute force neighbor intersection)
                for j in range(d_v):
                    w = node_neighbors[v, j]
                    if v < w:
                        for k in range(d_u):
                            if node_neighbors[u, k] == w:
                                triangles += 1
                                break

    k_avg = k_sum / N if N > 0 else 0.0
    tri_per_edge = triangles / edges if edges > 0 else 0.0
    transitivity = (3.0 * triangles / triplets) if triplets > 0 else 0.0

    # Pearson degree assortativity
    if edges > 0:
        M = float(edges)
        term1 = sum_de_de / M
        term2 = (sum_de / (2.0 * M)) ** 2
        term3 = (sum_de2 / (2.0 * M))
        denominator = term3 - term2
        if denominator > 1e-9:
            assortativity = (term1 - term2) / denominator
        else:
            assortativity = float('nan')
    else:
        assortativity = float('nan')

    # 2. Largest Connected Component via DFS
    visited = np.zeros(N, dtype=np.bool_)
    stack = np.empty(N, dtype=np.int32)
    lcc = 0

    for i in range(N):
        if not visited[i]:
            comp_size = 0
            stack_ptr = 0
            stack[stack_ptr] = i
            visited[i] = True

            while stack_ptr >= 0:
                u = stack[stack_ptr]
                stack_ptr -= 1
                comp_size += 1

                for j in range(node_degrees[u]):
                    v = node_neighbors[u, j]
                    if not visited[v]:
                        visited[v] = True
                        stack_ptr += 1
                        stack[stack_ptr] = v

            if comp_size > lcc:
                lcc = comp_size

    lcc_pct = (lcc / N) * 100.0 if N > 0 else 0.0
    return k_min, k_avg, k_max, triangles, edges, tri_per_edge, transitivity, assortativity, lcc_pct
