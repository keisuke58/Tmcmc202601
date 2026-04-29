# -*- coding: utf-8 -*-
"""
hamilton_ode_numba_5sp.py — Numba-parallel Hamilton ODE for 5-species TMCMC.

Provides two entry points:
  simulate_0d_5sp_numba(theta, ...)  → phibar trajectory (serial)
  batch_logL_numba(theta_batch, ...) → logL array (parallel via numba.prange)

theta layout: JAX column-major upper-triangle, then b.
  [A[0,0], A[0,1],A[1,1], A[0,2],..,A[4,4], b[0]..b[4]]  (20 params total)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parent.parent
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

try:
    import numba
    from numba import njit, prange
    from improved_5species_jit import (
        _run_deterministic_jit_5s,
        _compute_Q_vector_numba_5s,
        _compute_jacobian_numba_5s,
        _newton_step_jit_5s,
        HAS_NUMBA,
    )

    _HAS_NUMBA = HAS_NUMBA
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Numba-compatible theta → A, b conversion (JAX column-major upper-triangle)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=False, fastmath=True, nogil=True)
    def _theta_to_Ab(theta):
        A = np.zeros((5, 5))
        idx = 0
        for j in range(5):
            for i in range(j + 1):
                A[i, j] = theta[idx]
                A[j, i] = theta[idx]
                idx += 1
        b = theta[15:20].copy()
        return A, b

    @njit(cache=False, fastmath=True, nogil=True)
    def _make_g0(phi_init):
        g0 = np.zeros(12)
        s = 0.0
        for i in range(5):
            g0[i] = phi_init[i]
            s += phi_init[i]
        g0[5] = 1.0 - s
        for i in range(5):
            g0[6 + i] = 0.999
        return g0

    # ------------------------------------------------------------------
    # Batch logL with OpenMP parallelism (numba.prange)
    # ------------------------------------------------------------------

    @njit(parallel=True, fastmath=True, nogil=True, cache=False)
    def _batch_logL_kernel(
        theta_batch,  # (n_batch, 20)
        phi_init,  # (5,)
        idx_sparse,  # (n_obs,) int
        data,  # (n_obs, 5)
        weights,  # (n_obs, 5) or None-like (shape (0,0))
        has_weights,  # bool
        sigma2,  # scalar
        dt,  # scalar
        maxtimestep,  # int
        Eta,  # (5,)
        EtaPhi,  # (5,)
        c_const,  # scalar
        alpha_const,  # scalar
        Kp1,  # scalar
        K_hill,  # scalar
        n_hill,  # scalar
        max_newton_iter,  # int
        eps,  # scalar
    ):
        n_batch = theta_batch.shape[0]
        n_obs = idx_sparse.shape[0]
        logL_arr = np.empty(n_batch)

        for p in prange(n_batch):
            A, b_diag = _theta_to_Ab(theta_batch[p])
            g0 = _make_g0(phi_init)

            t_arr, g_arr = _run_deterministic_jit_5s(
                g0,
                dt,
                maxtimestep,
                eps,
                Kp1,
                Eta,
                EtaPhi,
                c_const,
                alpha_const,
                A,
                b_diag,
                max_newton_iter,
                K_hill,
                n_hill,
            )

            # phibar = phi * psi
            logL = 0.0
            any_nan = False
            for j in range(n_obs):
                tidx = idx_sparse[j]
                for k in range(5):
                    phi_k = g_arr[tidx, k]
                    psi_k = g_arr[tidx, 6 + k]
                    phibar_jk = phi_k * psi_k
                    if not (phibar_jk == phibar_jk):  # isnan
                        any_nan = True
                        break
                    r = phibar_jk - data[j, k]
                    if has_weights:
                        logL -= 0.5 * weights[j, k] * r * r / sigma2
                    else:
                        logL -= 0.5 * r * r / sigma2
                if any_nan:
                    break

            logL_arr[p] = -1e20 if any_nan else logL

        return logL_arr

else:
    _batch_logL_kernel = None  # type: ignore
    _theta_to_Ab = None  # type: ignore


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------


def _warmup_once():
    """Trigger Numba JIT compilation with minimal inputs."""
    if not _HAS_NUMBA:
        return
    phi0 = np.full(5, 0.2)
    th = np.zeros(20)
    _w = np.zeros((0, 0))
    try:
        _batch_logL_kernel(
            th[np.newaxis, :],
            phi0,
            np.array([1], dtype=np.int64),
            np.zeros((1, 5)),
            _w,
            False,
            1.0,
            1e-4,
            5,  # tiny n_steps for warmup
            np.ones(5),
            np.ones(5),
            25.0,
            100.0,
            1e-4,
            0.0,
            2.0,
            5,
            1e-5,
        )
    except Exception:
        pass


_warmed_up = False


def _ensure_warm():
    global _warmed_up
    if not _warmed_up:
        _warmup_once()
        _warmed_up = True


def simulate_0d_5sp_numba(
    theta: np.ndarray,
    n_steps: int = 2500,
    dt: float = 1e-4,
    phi_init: np.ndarray | None = None,
    c_const: float = 25.0,
    alpha_const: float = 100.0,
    K_hill: float = 0.0,
    n_hill: float = 2.0,
) -> np.ndarray:
    """Run 5-species Hamilton ODE via Numba.  Returns phibar (n_steps+1, 5)."""
    if not _HAS_NUMBA:
        raise RuntimeError("numba / improved_5species_jit.py not available")
    _ensure_warm()

    if phi_init is None:
        phi_init = np.full(5, 0.2, dtype=np.float64)
    phi_init = np.asarray(phi_init, dtype=np.float64)
    s = phi_init.sum()
    if s >= 1.0:
        phi_init = phi_init * (0.999 / s)

    theta = np.asarray(theta, dtype=np.float64)
    A, b_diag = _theta_to_Ab(theta)

    g0 = _make_g0(phi_init)
    _t, g_arr = _run_deterministic_jit_5s(
        g0,
        dt,
        n_steps,
        1e-5,
        1e-4,
        np.ones(5),
        np.ones(5),
        c_const,
        alpha_const,
        A,
        b_diag,
        50,
        K_hill,
        n_hill,
    )
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    return phi * psi


def batch_logL_numba(
    theta_batch: np.ndarray,
    phi_init: np.ndarray,
    idx_sparse: np.ndarray,
    data: np.ndarray,
    sigma2: float,
    n_steps: int = 2500,
    dt: float = 1e-4,
    weights: np.ndarray | None = None,
    c_const: float = 25.0,
    alpha_const: float = 100.0,
    K_hill: float = 0.0,
    n_hill: float = 2.0,
) -> np.ndarray:
    """Evaluate logL for all particles in parallel (numba.prange / OpenMP)."""
    if not _HAS_NUMBA:
        raise RuntimeError("numba not available")
    _ensure_warm()

    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    phi_init = np.asarray(phi_init, dtype=np.float64)
    idx_sparse = np.asarray(idx_sparse, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)

    has_weights = weights is not None
    w = (
        np.asarray(weights[: data.shape[0], : data.shape[1]], dtype=np.float64)
        if has_weights
        else np.zeros((0, 0), dtype=np.float64)
    )

    return _batch_logL_kernel(
        theta_batch,
        phi_init,
        idx_sparse,
        data,
        w,
        has_weights,
        float(sigma2),
        float(dt),
        int(n_steps),
        np.ones(5, dtype=np.float64),
        np.ones(5, dtype=np.float64),
        float(c_const),
        float(alpha_const),
        1e-4,  # Kp1
        float(K_hill),
        float(n_hill),
        50,  # max_newton_iter
        1e-5,  # eps
    )
