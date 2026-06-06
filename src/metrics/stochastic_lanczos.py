"""
metrics.stochastic_lanczos — Lanczos tridiagonalization kernels for SLQ
==========================================================================
The two low-level Lanczos kernels (_lanczos_slq, _lanczos_slq_scipy) that
the spectral-dimension flow probe uses to build the heat-kernel partition
function Z(t) = (1/N)·tr exp(-t·L) via Hutchinson's trick (random probes)
+ Lanczos quadrature on each probe.

These are the m-step tridiagonalization primitives only; the probe loop,
t-grid construction, and power-law fitting live in core.flow_probe, which
imports both kernels from here.
"""

import numpy as np

from .numba_kernels import _csr_matvec

def _lanczos_slq(L_indptr, L_indices, L_data, v0, m):
    """m-step Lanczos from unit-norm vector v0 on symmetric L.

    Naive (no reorthogonalization). This is the standard SLQ kernel —
    ghost eigenvalues don't bias the quadrature Σ τ_j f(θ_j) because the
    spurious θ_j come paired with spurious weights that self-correct.

    Returns (alphas, betas) of lengths (m, m-1). If Lanczos terminates
    early (β collapses to 0), returns the shorter arrays.
    """
    n = len(v0)
    alphas = np.zeros(m, dtype=np.float64)
    betas = np.zeros(max(m - 1, 0), dtype=np.float64)

    v_prev = np.zeros(n, dtype=np.float64)
    v_curr = v0.astype(np.float64).copy()
    w = np.empty(n, dtype=np.float64)

    for step in range(m):
        _csr_matvec(L_indptr, L_indices, L_data, v_curr, w)
        alpha = float(np.dot(v_curr, w))
        alphas[step] = alpha

        if step < m - 1:
            if step == 0:
                w = w - alpha * v_curr
            else:
                w = w - alpha * v_curr - betas[step - 1] * v_prev

            beta = float(np.linalg.norm(w))
            if beta < 1e-10:
                return alphas[:step + 1], betas[:step]
            betas[step] = beta

            v_prev = v_curr
            v_curr = w / beta
            w = np.empty(n, dtype=np.float64)

    return alphas, betas


def _lanczos_slq_scipy(L_sparse, v0, m):
    """Lanczos using scipy.sparse matvec, which at large N is typically
    2–5× faster than the numba version because scipy's CSR matvec is
    C-coded with SIMD vectorization and better cache behaviour on long
    vectors. Same algorithmic semantics as _lanczos_slq.

    L_sparse: scipy.sparse.csr_matrix (symmetric, the graph Laplacian).
    """
    n = L_sparse.shape[0]
    alphas = np.zeros(m, dtype=np.float64)
    betas = np.zeros(max(m - 1, 0), dtype=np.float64)

    v_prev = np.zeros(n, dtype=np.float64)
    v_curr = v0.astype(np.float64, copy=True)

    for step in range(m):
        w = L_sparse @ v_curr  # scipy CSR matvec — the fast path
        alpha = float(np.dot(v_curr, w))
        alphas[step] = alpha

        if step < m - 1:
            if step == 0:
                w = w - alpha * v_curr
            else:
                w = w - alpha * v_curr - betas[step - 1] * v_prev

            beta = float(np.linalg.norm(w))
            if beta < 1e-10:
                return alphas[:step + 1], betas[:step]
            betas[step] = beta

            v_prev = v_curr
            v_curr = w / beta

    return alphas, betas
