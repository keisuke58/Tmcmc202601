# -*- coding: utf-8 -*-
"""
hamilton_ode_numba_5sp.py — Numba wrapper for the 5-species Hamilton ODE.

Converts theta from JAX column-major upper-triangle layout to A, b matrices
and delegates to the existing _run_deterministic_jit_5s Numba kernel from
improved_5species_jit.py.

Typical speed: ~2-5 ms/particle on CPU, vs ~44 s/particle with JAX on GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parent.parent
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

# Import the Numba JIT loop and Jacobian from the existing solver module.
# BiofilmNewtonSolver5S is instantiated once to trigger Numba compilation.
try:
    from improved_5species_jit import BiofilmNewtonSolver5S as _Solver5S

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _theta_to_Ab_jax_layout(theta: np.ndarray):
    """Convert 20-parameter JAX-layout theta to symmetric A(5,5) and b(5).

    JAX layout (column-major upper triangle, columns first):
      col j covers A[0..j, j]
      so index order: A[0,0], A[0,1],A[1,1], A[0,2],A[1,2],A[2,2], ...
    Then b = theta[15:20].
    """
    A = np.zeros((5, 5), dtype=np.float64)
    idx = 0
    for j in range(5):
        for i in range(j + 1):
            A[i, j] = theta[idx]
            A[j, i] = theta[idx]
            idx += 1
    b = np.asarray(theta[15:20], dtype=np.float64)
    return A, b


# Module-level singleton solver for Numba JIT compilation.
_solver_cache: dict = {}


def _get_solver(
    maxtimestep: int,
    dt: float,
    c_const: float,
    alpha_const: float,
    phi_init: np.ndarray,
    K_hill: float,
    n_hill: float,
) -> "_Solver5S":
    key = (maxtimestep, dt, c_const, alpha_const, K_hill, n_hill, tuple(phi_init.tolist()))
    if key not in _solver_cache:
        solver = _Solver5S(
            maxtimestep=maxtimestep,
            dt=dt,
            c_const=c_const,
            alpha_const=alpha_const,
            phi_init=phi_init,
            active_species=list(range(5)),
            K_hill=K_hill,
            n_hill=n_hill,
            use_numba=True,
        )
        # Warm up Numba JIT with a dummy call.
        _dummy = np.zeros(20, dtype=np.float64)
        try:
            solver.run_deterministic(_dummy)
        except Exception:
            pass
        _solver_cache[key] = solver
    return _solver_cache[key]


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
    """Run 5-species Hamilton ODE via Numba.  Returns phibar (n_steps+1, 5).

    Parameters
    ----------
    theta : (20,) array, JAX column-major upper-triangle layout
    n_steps : int
    dt : float
    phi_init : (5,) initial species fractions (Day 1 data), or None for uniform
    """
    if not _HAS_NUMBA:
        raise RuntimeError("numba / improved_5species_jit.py not available")

    if phi_init is None:
        phi_init = np.full(5, 1.0 / 5, dtype=np.float64)
    phi_init = np.asarray(phi_init, dtype=np.float64)

    # Clip phi_init so phi0 = 1-sum > 0
    phi_sum = phi_init.sum()
    if phi_sum >= 1.0:
        phi_init = phi_init * (0.999 / phi_sum)

    solver = _get_solver(n_steps, dt, c_const, alpha_const, phi_init, K_hill, n_hill)

    # Convert theta to A, b using JAX layout
    A, b_diag = _theta_to_Ab_jax_layout(theta)

    # Build initial state
    g_prev = np.zeros(12, dtype=np.float64)
    for i in range(5):
        g_prev[i] = phi_init[i]
    g_prev[5] = 1.0 - phi_init.sum()
    for i in range(5):
        g_prev[6 + i] = 0.999

    # Run Numba JIT loop
    _t_arr, g_arr = solver._run_loop_jit(None, g_prev, A, b_diag)

    # Observable: phibar = phi * psi
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    phibar = phi * psi
    return phibar
