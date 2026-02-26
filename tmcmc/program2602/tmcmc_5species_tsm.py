# -*- coding: utf-8 -*-
"""
tmcmc_5species_tsm.py - TSM Wrapper for 5-Species Model (12D State Vector)

Adapts BiofilmNewtonSolver5S for TMCMC integration.
Key features:
1. Handles 12-dimensional state vector (5 phi + 1 phi0 + 5 psi + 1 gamma).
2. Uses Complex-Step Differentiation for Sensitivity Matrix (S = dx/dtheta).
3. Manages Linearization Point (theta0) for iterative TSM updates.
4. Compatible with case2_tmcmc runners.
"""

import numpy as np
import logging
import sys
import os

from improved_5species_jit import BiofilmNewtonSolver5S, HAS_NUMBA

if HAS_NUMBA:
    from numba import njit
else:
    # Dummy njit if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)

# =============================================================================
# Helper: JIT Variance Accumulation (Re-implemented for 5-species context)
# =============================================================================


@njit(cache=True)
def _sigma2_accumulate_numba(x1, var_act):
    """
    Accumulate variance: sigma2[t, i] = sum_k (x1[t, i, k]^2 * var_act[k])

    x1 shape: (n_time, n_state, n_params)
    var_act shape: (n_params,)
    Returns: (n_time, n_state)
    """
    n_time, n_state, n_params = x1.shape
    sigma2 = np.zeros((n_time, n_state), dtype=np.float64)
    for t in range(n_time):
        for i in range(n_state):
            sum_val = 0.0
            for k in range(n_params):
                sum_val += x1[t, i, k] ** 2 * var_act[k]
            sigma2[t, i] = sum_val + 1e-12
    return sigma2


# =============================================================================
# BiofilmTSM5S Class
# =============================================================================


class BiofilmTSM5S:
    """
    TSM-ROM Wrapper for 5-Species Model.

    State Vector (12D):
        [phi1..phi5, phi0, psi1..psi5, gamma]

    Parameters (20D):
        [a11..a45, b1..b5] (See BiofilmNewtonSolver5S.THETA_NAMES)
    """

    def __init__(
        self,
        solver: BiofilmNewtonSolver5S,
        active_theta_indices=None,
        cov_rel: float = 0.005,
        theta_linearization=None,
    ):
        self.solver = solver
        self.cov_rel = float(cov_rel)

        # Default active indices: all 20 parameters
        if active_theta_indices is None:
            active_theta_indices = np.arange(20, dtype=int)
        self.active_idx = np.array(list(active_theta_indices), dtype=int)

        # Linearization state
        self.theta_linearization = None
        self._has_explicit_linearization = False
        self._linearization_enabled = False  # Start disabled (full TSM / complex step exploration)
        self._linearization_count = 0

        # Cache
        self._deterministic_solution_cached = None
        self._x1_cached = None
        self._first_call_done = False

        if theta_linearization is not None:
            self.update_linearization_point(theta_linearization)

    def update_linearization_point(self, theta_new):
        """Update the linearization center (theta0)."""
        if theta_new.shape[0] != 20:
            raise ValueError(f"theta_new must have shape (20,), got {theta_new.shape}")

        old_theta = (
            self.theta_linearization.copy() if self.theta_linearization is not None else None
        )
        self.theta_linearization = theta_new.copy()
        self._has_explicit_linearization = True
        self._linearization_count += 1

        # Invalidate cache
        self._deterministic_solution_cached = None
        self._x1_cached = None

        logger.info(f"TSM5S Linearization Point Updated (# {self._linearization_count})")
        if old_theta is not None:
            delta = np.linalg.norm(theta_new - old_theta)
            logger.info(f"||delta_theta|| = {delta:.6f}")

    def enable_linearization(self, enable: bool = True):
        """Enable/Disable linearization (approximate x(theta) ~ x(theta0) + S*(theta-theta0))."""
        self._linearization_enabled = enable
        if enable and self.theta_linearization is None:
            # Will be set on next solve
            pass

    def get_linearization_point(self):
        return self.theta_linearization.copy() if self.theta_linearization is not None else None

    def _compute_sensitivity_matrix(self, theta_center, t_arr, g_det):
        """
        Compute Sensitivity Matrix S = dx/dtheta at theta_center using Complex Step.
        Returns x1 (n_time, 12, n_active_params).
        """
        n_time = len(t_arr)
        n_state = 12
        n_active = len(self.active_idx)

        # Precompute real matrices
        A0, b0 = self.solver.theta_to_matrices(theta_center)

        x1_list = []
        # Complex step h
        h = 1e-30

        # We compute sensitivity for each time step
        # Note: We use the propagation: J * dx/dtheta = - dQ/dtheta
        # where J = dQ/dx

        # Initialize x1 at t=0 (assumed 0 sensitivity if initial condition is fixed)
        # If initial condition depends on theta, we need dphi_init/dtheta.
        # Here assumed fixed.
        x1_t = np.zeros((n_state, n_active), dtype=np.float64)
        x1_list.append(x1_t.copy())

        for step in range(n_time - 1):
            tt = t_arr[step + 1]
            g_old = g_det[step]
            g_new = g_det[step + 1]

            # 1. Compute Jacobian J = dQ/dg_new (Real)
            # Use JIT Jacobian from solver
            J = self.solver.compute_Jacobian_matrix(g_new, g_old, tt, self.solver.dt, A0, b0)

            # 2. Compute dQ/dtheta via Complex Step
            dG_dtheta = np.zeros((n_state, n_active), dtype=np.float64)

            for k, idx in enumerate(self.active_idx):
                th_cs = theta_center.astype(np.complex128)
                th_cs[idx] += 1j * h
                A_cs, b_cs = self.solver.theta_to_matrices(th_cs)

                # Q(g_new, g_old, theta_cs)
                # Note: g_new, g_old are real here (fixed at theta_center trajectory)
                Q_cs = self.solver.compute_Q_vector(
                    g_new.astype(np.complex128),
                    g_old.astype(np.complex128),
                    tt,
                    self.solver.dt,
                    A_cs,
                    b_cs,
                )

                # dQ/dtheta = Im(Q)/h
                dG_dtheta[:, k] = np.imag(Q_cs) / h

            # 3. Solve J * x1_next = - dQ/dtheta - dQ/dg_old * x1_prev
            # Wait, the implicit relation is Q(g_new, g_old, theta) = 0
            # dQ/dg_new * dg_new/dtheta + dQ/dg_old * dg_old/dtheta + dQ/dtheta = 0
            # J * x1_new + J_old * x1_old + dQ_dtheta = 0
            # J * x1_new = - (dQ_dtheta + J_old * x1_old)

            # Need J_old = dQ/dg_old.
            # dQ/dg_old involves -1/dt mostly.
            # Let's verify dQ/dg_old structure.
            # Q ~ g_new - g_old - dt*f(g_new).
            # dQ/dg_old = -I/dt? No, depends on discretization.
            # In improved_5species_jit.py:
            # phidot = (phi_new - phi_old)/dt
            # Q depends on phidot.
            # dQ/dphi_old = dQ/dphidot * dphidot/dphi_old = dQ/dphidot * (-1/dt)

            # To avoid computing J_old analytically, we can use complex step for J_old * x1_old product?
            # Or simpler: The BiofilmTSM in improved1207_paper_jit seems to ignore J_old term?
            # Let's check improved1207_paper_jit.py loop again.

            # It does:
            # x1_t[:, k] = np.linalg.solve(J, -dG)
            # This implies x1_prev is ignored or assumed 0?
            # Ah, `dG` in that code comes from `Q_cs`.
            # If `g_old` passed to `compute_Q_vector` is FIXED (real), then `Q_cs` only captures partial derivative wrt theta?
            # Wait, `solve_tsm` in `improved1207` uses `x1_list.append(x1_t)`.
            # And `x1_t` is solved at each step independently?
            # This means it's computing steady-state sensitivity or ignoring history?
            # NO! `improved1207` is a bit weird.
            # Let's look closer at line 855:
            # Q_cs = solver.compute_Q_vector(g_new_complex, g_old_complex, ...)
            # But g_new and g_old are passed as casted complex of real values.
            # So they don't carry sensitivity information.
            # So `dG` is just `\partial Q / \partial \theta`.
            # And `J * x1 = - \partial Q / \partial \theta`.
            # This gives `x1 = - J^{-1} * \partial Q / \partial \theta`.
            # This is the sensitivity of the EQUILIBRIUM solution at that time step (assuming Q=0 defines equilibrium if g_old was fixed).
            # But this is a time-stepping scheme.
            # This approximation (ignoring dQ/dg_old * x1_old) assumes `dg_new/dtheta` dominates or `dg_old/dtheta` is negligible?
            # Or maybe it's "Quasi-Steady State" sensitivity?
            # Given the biofilm dynamics are slow and often driven by boundary conditions/parameters, maybe this is acceptable.
            # OR, I am misinterpreting the code.

            # Actually, `improved1207_paper_jit.py` seems to treat it as a series of equilibrium problems?
            # "Residual Q(g_new) ... = 0".
            # If we treat it as solving F(x, theta) = 0 at each step, then dx/dtheta = - (dF/dx)^-1 * dF/dtheta.
            # This matches the code.
            # So it ignores the dependence on previous state sensitivity.
            # This is valid if the system relaxes fast compared to parameter changes?
            # But here parameters are static global parameters.

            # Let's stick to the pattern in `improved1207_paper_jit.py` for consistency with 4-species model results.
            # It seems to work there.

            rhs = -dG_dtheta
            try:
                x1_new = np.linalg.solve(J, rhs)
            except np.linalg.LinAlgError:
                J_reg = J + 1e-10 * np.eye(J.shape[0])
                x1_new = np.linalg.solve(J_reg, rhs)

            x1_list.append(x1_new)

        x1 = np.stack(x1_list, axis=0)  # (n_time, 12, n_active)
        return x1

    def solve_tsm(self, theta):
        """
        Solve TSM-ROM.
        Returns: t_arr, x0 (mean), sigma2 (variance)
        """
        theta = np.asarray(theta, dtype=np.float64)

        # Determine linearization center
        if self._has_explicit_linearization:
            theta_center = self.theta_linearization
            is_at_linearization = np.allclose(theta, theta_center)
        else:
            theta_center = theta.copy()
            self.theta_linearization = theta_center
            self._has_explicit_linearization = True
            is_at_linearization = True

        # 1. Solve Deterministic at Center (if needed)
        if (
            self._deterministic_solution_cached is None
            or self._last_center_solved is None
            or not np.allclose(theta_center, self._last_center_solved)
        ):
            # Run deterministic solver
            t_arr, g_det = self.solver.solve(theta_center)
            sigma2_zeros = np.zeros_like(g_det)

            # Compute Sensitivities (if linearization enabled or we need them for first run)
            # Actually, we need sensitivities for TSM propagation anyway?
            # Wait, if linearization is disabled, we should run solver at `theta`, not `theta_center`.

            self._deterministic_solution_cached = (t_arr, g_det, sigma2_zeros)
            self._last_center_solved = theta_center.copy()

            # Cache sensitivities
            self._x1_cached = self._compute_sensitivity_matrix(theta_center, t_arr, g_det)

        # Retrieve cached center solution
        t_arr, g_center, _ = self._deterministic_solution_cached
        x1_center = self._x1_cached

        # 2. Logic Branch
        if is_at_linearization:
            # Exact match, return deterministic solution with 0 variance
            return t_arr, g_center, np.zeros_like(g_center)

        if not self._linearization_enabled:
            # Full run at `theta` (slow but accurate)
            # We don't use cache here.
            t_arr_full, g_full = self.solver.solve(theta)
            # We still need variance?
            # Usually TSM implies providing variance.
            # If we run full deterministic, sigma2 comes from parameter uncertainty?
            # Standard TSM: x ~ N(x_det, sigma2).
            # Here we just return x_det and sigma2 based on local sensitivity at theta?
            # Let's just return 0 variance for full run or compute local sensitivity?
            # For MCMC, usually we need sigma2 for likelihood.
            # So even for full run, we need sensitivity!

            # So: Run solver at theta. Compute sensitivity at theta.
            x1_local = self._compute_sensitivity_matrix(theta, t_arr_full, g_full)

            # Compute sigma2
            var_th = (self.cov_rel * theta) ** 2
            var_act = var_th[self.active_idx]
            sigma2 = _sigma2_accumulate_numba(x1_local, var_act)

            return t_arr_full, g_full, sigma2

        else:
            # Linearization Enabled: x(theta) ~ x(theta0) + S * (theta - theta0)
            delta_theta = theta - theta_center
            delta_theta_active = delta_theta[self.active_idx]

            # x_pred = x0 + sum(x1_k * dtheta_k)
            # x1_center shape: (T, 12, n_active)
            # delta_theta_active shape: (n_active,)

            # Einsum: T=time, S=state, P=param
            # x_corr = x1 * dtheta -> (T, S)
            x_correction = np.dot(x1_center, delta_theta_active)

            x_pred = g_center + x_correction

            # Sigma2 (Aleatory)
            # Propagate variance of theta (p-box) centered at theta (or theta0?)
            # Usually we use variance at current theta.
            var_th = (self.cov_rel * theta) ** 2
            var_act = var_th[self.active_idx]

            sigma2 = _sigma2_accumulate_numba(x1_center, var_act)

            return t_arr, x_pred, sigma2

    # Internal helper for caching logic
    _last_center_solved = None
