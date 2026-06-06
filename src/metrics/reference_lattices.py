"""
metrics.reference_lattices — Reference-graph builders + REFERENCE_FAMILIES registry
=======================================================================
Deterministic graphs whose spectral dimension is known exactly,
analytically, or by numerical consensus. Used to validate the d_s
probes against ground truth.

Note: the live sweep builds its torus references directly via
core.cell_tests.build_torus_cell, tuned to match the sweep's N grid.
The set here is broader (t-fractal, sierpinski_carpet,
sierpinski_tetrahedron, etc.) and is kept as a general validation library.
"""

import math
import numpy as np

from .graph_container import Graph

def _edges_to_csr(edges, N):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        if u == v: continue
        adj[u].add(v); adj[v].add(u)
    max_deg = max((len(s) for s in adj), default=0)
    neighbors = np.full((N, max_deg), -1, dtype=np.int32)
    degrees = np.zeros(N, dtype=np.int32)
    for i, s in enumerate(adj):
        for j, nb in enumerate(sorted(s)):
            neighbors[i, j] = nb
        degrees[i] = len(s)
    return neighbors, degrees

def _finalize(edges, N, family, name, known_ds, known_ds_kind):
    neighbors, degrees = _edges_to_csr(edges, N)
    return Graph(neighbors=neighbors, degrees=degrees, N=N,
                 name=name, family=family,
                 known_ds=known_ds, known_ds_kind=known_ds_kind)

def _hypercubic(d: int, N_target: int):
    # Choose L closest to N_target^(1/d)
    L = max(4, int(round(N_target ** (1.0 / d))))
    N = L ** d
    edges = []
    strides = [L ** a for a in range(d)]
    for lin in range(N):
        coords = [(lin // strides[a]) % L for a in range(d)]
        for a in range(d):
            nb = list(coords)
            nb[a] = (coords[a] + 1) % L
            lin_nb = 0
            for b in range(d):
                lin_nb += nb[b] * strides[b]
            edges.append((lin, lin_nb))
    return _finalize(edges, N,
                     family=f"hypercubic_{d}d",
                     name=f"hypercubic_{d}d_L{L}_pbc",
                     known_ds=float(d),
                     known_ds_kind="analytic")


def build_hypercubic_1d(N_target): return _hypercubic(1, N_target)
def build_hypercubic_2d(N_target): return _hypercubic(2, N_target)
def build_hypercubic_3d(N_target): return _hypercubic(3, N_target)
def build_hypercubic_4d(N_target): return _hypercubic(4, N_target)
def build_hypercubic_5d(N_target): return _hypercubic(5, N_target)
def build_hypercubic_6d(N_target): return _hypercubic(6, N_target)


# ═══════════════════════════════════════════════════════════════════
#  Cycle
# ═══════════════════════════════════════════════════════════════════
def build_cycle(N_target):
    N = max(4, int(N_target))
    edges = [(i, (i + 1) % N) for i in range(N)]
    return _finalize(edges, N,
                     family="cycle",
                     name=f"cycle_N{N}",
                     known_ds=1.0,
                     known_ds_kind="analytic")


# ═══════════════════════════════════════════════════════════════════
#  Triangular & honeycomb 2D lattices (d_s = 2)
# ═══════════════════════════════════════════════════════════════════
def build_triangular_2d(N_target):
    L = max(4, int(round(math.sqrt(N_target))))
    N = L * L
    edges = []
    for i in range(L):
        for j in range(L):
            u = i * L + j
            for di, dj in [(0, 1), (1, 0), (1, 1)]:
                ni, nj = (i + di) % L, (j + dj) % L
                edges.append((u, ni * L + nj))
    return _finalize(edges, N,
                     family="triangular",
                     name=f"triangular_L{L}_pbc",
                     known_ds=2.0,
                     known_ds_kind="analytic")


def build_honeycomb_2d(N_target):
    # Brick-wall honeycomb: 2 sublattices × L×L -> N = 2L²
    L = max(4, int(round(math.sqrt(N_target / 2))))
    N = 2 * L * L
    def idx(r, c, s): return (r * L + c) * 2 + s
    edges = []
    for r in range(L):
        for c in range(L):
            a = idx(r, c, 0); b = idx(r, c, 1)
            edges.append((a, b))
            edges.append((a, idx(r, (c - 1) % L, 1)))
            edges.append((a, idx((r - 1) % L, c, 1)))
    return _finalize(edges, N,
                     family="honeycomb",
                     name=f"honeycomb_L{L}",
                     known_ds=2.0,
                     known_ds_kind="analytic")


# ═══════════════════════════════════════════════════════════════════
#  Sierpinski gasket 2D  (d_s = 2 log 3 / log 5)
# ═══════════════════════════════════════════════════════════════════
def build_sierpinski_gasket(N_target):
    # depth d → N = 3(3^d + 1)/2; invert to choose depth
    depth = max(1, int(round(math.log(2 * N_target / 3 - 1) / math.log(3))))
    tris = [((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2))]
    for _ in range(depth):
        new = []
        for (a, b, c) in tris:
            ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            bc = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
            ca = ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
            new.append((a, ab, ca))
            new.append((ab, b, bc))
            new.append((ca, bc, c))
        tris = new
    point_idx = {}
    def pid(p):
        k = (round(p[0], 9), round(p[1], 9))
        if k not in point_idx: point_idx[k] = len(point_idx)
        return point_idx[k]
    edges = []
    for (a, b, c) in tris:
        ia, ib, ic = pid(a), pid(b), pid(c)
        edges.extend([(ia, ib), (ib, ic), (ic, ia)])
    N = len(point_idx)
    return _finalize(edges, N,
                     family="sierpinski_gasket",
                     name=f"sierpinski_gasket_d{depth}",
                     known_ds=2 * math.log(3) / math.log(5),
                     known_ds_kind="analytic")


# ═══════════════════════════════════════════════════════════════════
#  Sierpinski tetrahedron 3D  (d_s = 2 log 4 / log 6)
# ═══════════════════════════════════════════════════════════════════
def build_sierpinski_tetrahedron(N_target):
    # depth d has approximately 4^d * 4 / 3 distinct vertices (rough)
    depth = max(1, int(round(math.log(max(N_target * 0.75, 4)) / math.log(4))))
    a = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    c = (0.5, math.sqrt(3) / 2, 0.0)
    dtop = (0.5, math.sqrt(3) / 6, math.sqrt(2 / 3))
    tets = [(a, b, c, dtop)]
    for _ in range(depth):
        new = []
        for (p0, p1, p2, p3) in tets:
            m01 = tuple((p0[i] + p1[i]) / 2 for i in range(3))
            m02 = tuple((p0[i] + p2[i]) / 2 for i in range(3))
            m03 = tuple((p0[i] + p3[i]) / 2 for i in range(3))
            m12 = tuple((p1[i] + p2[i]) / 2 for i in range(3))
            m13 = tuple((p1[i] + p3[i]) / 2 for i in range(3))
            m23 = tuple((p2[i] + p3[i]) / 2 for i in range(3))
            new.append((p0, m01, m02, m03))
            new.append((m01, p1, m12, m13))
            new.append((m02, m12, p2, m23))
            new.append((m03, m13, m23, p3))
        tets = new
    point_idx = {}
    def pid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in point_idx: point_idx[k] = len(point_idx)
        return point_idx[k]
    edges = []
    for (p0, p1, p2, p3) in tets:
        i0, i1, i2, i3 = pid(p0), pid(p1), pid(p2), pid(p3)
        edges.extend([(i0, i1), (i0, i2), (i0, i3),
                      (i1, i2), (i1, i3), (i2, i3)])
    N = len(point_idx)
    return _finalize(edges, N,
                     family="sierpinski_tetra",
                     name=f"sierpinski_tetra_d{depth}",
                     known_ds=2 * math.log(4) / math.log(6),
                     known_ds_kind="analytic")


# ═══════════════════════════════════════════════════════════════════
#  T-fractal  (d_s = 2 log 3 / log 6)
# ═══════════════════════════════════════════════════════════════════
def build_t_fractal(N_target):
    # depth d → N = 2^(d+1) + 1 roughly (2 new terminals per step)
    depth = max(1, int(round(math.log2(max(N_target / 2, 2)))))
    adj = [[], []]
    adj[0].append(1); adj[1].append(0)
    terminals = [1]
    for _ in range(depth):
        new_t = []
        for term in terminals:
            w = len(adj); adj.append([])
            adj[w].append(term); adj[term].append(w)
            t1 = len(adj); adj.append([])
            adj[w].append(t1); adj[t1].append(w)
            t2 = len(adj); adj.append([])
            adj[w].append(t2); adj[t2].append(w)
            new_t.append(t1); new_t.append(t2)
        terminals = new_t
    N = len(adj)
    edges = [(u, v) for u in range(N) for v in adj[u] if u < v]
    return _finalize(edges, N,
                     family="t_fractal",
                     name=f"t_fractal_d{depth}",
                     known_ds=2 * math.log(3) / math.log(6),
                     known_ds_kind="analytic")


# ═══════════════════════════════════════════════════════════════════
#  Sierpinski carpet 2D  (d_s ≈ 1.805, numerical consensus)
# ═══════════════════════════════════════════════════════════════════
def build_sierpinski_carpet(N_target):
    # depth d → N = 8^d sites (roughly); invert
    depth = max(1, int(round(math.log(N_target) / math.log(8))))
    L = 3 ** depth
    present = np.zeros((L, L), dtype=bool)
    for x in range(L):
        tb = []
        t = x
        for _ in range(depth):
            tb.append(t % 3); t //= 3
        for y in range(L):
            tb2 = []
            t = y
            for _ in range(depth):
                tb2.append(t % 3); t //= 3
            ok = True
            for k in range(depth):
                if tb[k] == 1 and tb2[k] == 1:
                    ok = False; break
            if ok:
                present[x, y] = True
    idx = np.full((L, L), -1, dtype=np.int64)
    nid = 0
    for x in range(L):
        for y in range(L):
            if present[x, y]:
                idx[x, y] = nid; nid += 1
    edges = []
    for x in range(L):
        for y in range(L):
            if not present[x, y]: continue
            if x + 1 < L and present[x + 1, y]:
                edges.append((int(idx[x, y]), int(idx[x + 1, y])))
            if y + 1 < L and present[x, y + 1]:
                edges.append((int(idx[x, y]), int(idx[x, y + 1])))
    return _finalize(edges, int(nid),
                     family="sierpinski_carpet",
                     name=f"sierpinski_carpet_d{depth}",
                     known_ds=1.805,
                     known_ds_kind="consensus")


# ═══════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════
REFERENCE_FAMILIES = {
    "cycle":                build_cycle,
    "hypercubic_1d":        build_hypercubic_1d,
    "hypercubic_2d":        build_hypercubic_2d,
    "hypercubic_3d":        build_hypercubic_3d,
    "hypercubic_4d":        build_hypercubic_4d,
    "hypercubic_5d":        build_hypercubic_5d,
    "hypercubic_6d":        build_hypercubic_6d,
    "triangular":           build_triangular_2d,
    "honeycomb":            build_honeycomb_2d,
    "t_fractal":            build_t_fractal,
    "sierpinski_gasket":    build_sierpinski_gasket,
    "sierpinski_tetra":     build_sierpinski_tetrahedron,
    "sierpinski_carpet":    build_sierpinski_carpet,
}


def all_families():
    return list(REFERENCE_FAMILIES.keys())


def build_family(name: str, N_target: int) -> Graph:
    if name not in REFERENCE_FAMILIES:
        raise KeyError(f"unknown family: {name}")
    return REFERENCE_FAMILIES[name](N_target)
