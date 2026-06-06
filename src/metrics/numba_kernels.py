"""
metrics.numba_kernels — Numba njit kernels
=======================================
The compiled hot paths used by the SLQ (stochastic Lanczos quadrature)
spectral-dimension probe. Kept in one place so the JIT-cache files
(numba writes them to ~/.numba) live next to each other and the import
order is obvious.

Names exposed here: _extract_lcc_nodes, _csr_matvec,
_build_laplacian_csr_jit. The functions keep a leading underscore to
mark them as internal kernels — direct callers should be inside the
metrics package (or core, which shares these compiled kernels). The
module file itself is public-named.
"""

import numpy as np
from numba import njit, prange

@njit(cache=True)
def _extract_lcc_nodes(neighbors, degrees, N):
    visited = np.zeros(N, dtype=np.bool_)
    queue = np.empty(N, dtype=np.int32)
    comp = np.empty(N, dtype=np.int32)
    best = np.empty(0, dtype=np.int32)
    best_size = 0
    for start in range(N):
        if visited[start]: continue
        head = 0; tail = 0
        queue[tail] = start; tail += 1
        visited[start] = True
        csize = 0
        while head < tail:
            u = queue[head]; head += 1
            comp[csize] = u; csize += 1
            deg = degrees[u]
            for k in range(deg):
                v = neighbors[u, k]
                if not visited[v]:
                    visited[v] = True
                    queue[tail] = v; tail += 1
        if csize > best_size:
            best_size = csize
            best = comp[:csize].copy()
    return best


@njit(cache=True, parallel=True)
def _csr_matvec(indptr, indices, data, x, out):
    """Compute out = L @ x for CSR-stored L. Parallelized over rows."""
    n = len(x)
    for i in prange(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        out[i] = s


@njit(cache=True)
def _build_laplacian_csr_jit(neighbors, degrees, N):
    """Build CSR form of the combinatorial Laplacian L = D - A."""
    # Row i: 1 diagonal + degrees[i] off-diagonal
    total = N
    for i in range(N):
        total += degrees[i]

    indptr = np.zeros(N + 1, dtype=np.int64)
    indices = np.empty(total, dtype=np.int32)
    data = np.empty(total, dtype=np.float64)

    pos = 0
    for i in range(N):
        indptr[i] = pos
        # Diagonal entry
        indices[pos] = i
        data[pos] = float(degrees[i])
        pos += 1
        # Off-diagonal entries (-1 each)
        for k in range(degrees[i]):
            indices[pos] = neighbors[i, k]
            data[pos] = -1.0
            pos += 1
    indptr[N] = pos

    return indptr, indices, data
