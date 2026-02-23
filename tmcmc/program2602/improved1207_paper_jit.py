# -*- coding: utf-8 -*-
"""
improved1207_paperjit_exact.py

✅ Goal
- "improved1207.py" と **数式・状態ベクトル定義・パラメータ順序を完全一致** させたまま、
  主要ホットループ（Newton + 時間積分）だけを Numba JIT 化した版。

✅ Paper / code convention (same as improved1207.py)
State g (10,):
    [phi1, phi2, phi3, phi4, phi0,  psi1, psi2, psi3, psi4, gamma]
Active species: subset of {0,1,2,3}  (phi0, gamma are always present)

Theta (14,) order (same as improved1207.py THETA_NAMES)
    a11,a12,a22,b1,b2,  a33,a34,a44,b3,b4,  a13,a14,a23,a24

Notes
- Complex-step in theta is preserved via the pure NumPy residual (_compute_Q_vector_numpy).
- JIT path uses the original Numba residual/Jacobian kernels from improved1207.py equations.
- If numba is not available, this file still runs (falls back to pure Python path).

Usage
    from improved1207_paperjit_exact import BiofilmNewtonSolver, BiofilmTSM, get_theta_true
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    njit = None


# =============================================================================
# 0. TRUE THETA (same as improved1207.py)
# =============================================================================

def get_theta_true() -> np.ndarray:
    """
    True parameter vector used for synthetic data generation (Case II).
    Order follows THETA_NAMES.
    """
    return np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,  # M1 block
        1.5, 1.0, 2.0, 0.3, 0.4,  # M2 block
        2.0, 1.0, 2.0, 1.0        # M3 cross-interaction block
    ], dtype=np.float64)


# =============================================================================
# 1. RESIDUAL + JACOBIAN KERNELS (verbatim equations from improved1207.py)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=False, fastmath=True)
    def _compute_Q_vector_numba(phi_new, phi0_new, psi_new, gamma_new,
                                phi_old, phi0_old, psi_old,
                                dt, Kp1, Eta_vec, Eta_phi_vec,
                                c_val, alpha_val, A, b_diag, active_mask):
        # --- 安定化: phi, phi0, psi を [eps, 1-eps] にクリップ ---
        # 且つ、Inactive species を 0 に固定 (Locking)
        eps = 1e-12
        for i in range(4):
            if active_mask[i] == 1:
                if phi_new[i] < eps:
                    phi_new[i] = eps
                elif phi_new[i] > 1.0 - eps:
                    phi_new[i] = 1.0 - eps

                if psi_new[i] < eps:
                    psi_new[i] = eps
                elif psi_new[i] > 1.0 - eps:
                    psi_new[i] = 1.0 - eps
            else:
                phi_new[i] = 0.0
                psi_new[i] = 0.0

        if phi0_new < eps:
            phi0_new = eps
        elif phi0_new > 1.0 - eps:
            phi0_new = 1.0 - eps

        Q = np.zeros(10)
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt

        Interaction = A @ (phi_new * psi_new)

        # Q[0..3] for phi_i
        for i in range(4):
            if active_mask[i] == 1:
                term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
                term2 = (1.0 / Eta_vec[i]) * (gamma_new
                                              + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i]
                                              + Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
                term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
                Q[i] = term1 + term2 - term3
            else:
                # Inactive species: equation is phi = 0
                Q[i] = phi_new[i]

        # Q[4] for phi0
        Q[4] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot

        # Q[5..8] for psi_i
        for i in range(4):
            if active_mask[i] == 1:
                term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) \
                        - (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
                term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
                term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
                term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
                Q[5+i] = term1 + term2 + term3 - term4
            else:
                # Inactive species: equation is psi = 0
                Q[5+i] = psi_new[i]

        # constraint
        Q[9] = np.sum(phi_new) + phi0_new - 1.0
        return Q


    @njit(cache=False, fastmath=True)
    def _compute_jacobian_numba(phi_new, phi0_new, psi_new, gamma_new,
                                phi_old, psi_old,
                                dt, Kp1, Eta_vec, Eta_phi_vec,
                                c_val, alpha_val, A, b_diag, active_mask):
        K = np.zeros((10, 10))

        # --- 安定化: phi, phi0, psi を [eps, 1-eps] にクリップ ---
        # 且つ、Inactive species を 0 に固定 (Locking)
        eps = 1e-12
        for i in range(4):
            if active_mask[i] == 1:
                if phi_new[i] < eps:
                    phi_new[i] = eps
                elif phi_new[i] > 1.0 - eps:
                    phi_new[i] = 1.0 - eps

                if psi_new[i] < eps:
                    psi_new[i] = eps
                elif psi_new[i] > 1.0 - eps:
                    psi_new[i] = 1.0 - eps
            else:
                phi_new[i] = 0.0
                psi_new[i] = 0.0

        if phi0_new < eps:
            phi0_new = eps
        elif phi0_new > 1.0 - eps:
            phi0_new = 1.0 - eps

        phidot = (phi_new - phi_old) / dt
        psidot = (psi_new - psi_old) / dt
        Interaction = A @ (phi_new * psi_new)

        phi_p = np.zeros(4)
        psi_p = np.zeros(4)
        for i in range(4):
            if active_mask[i] == 1:
                v = phi_new[i]
                phi_p[i] = (Kp1*(-4.+8.*v))/((v**3)*(v-1.)**3) \
                           - (Kp1*(2.-4.*v))*(3./((v**4)*(v-1.)**3)
                                              +3./((v**3)*(v-1.)**4))
                v = psi_new[i]
                psi_p[i] = (4.*Kp1*(3.-5.*v+5.*v**2))/((v**4)*(v-1.)**4)

        v0 = phi0_new
        phi0_p = (Kp1*(-4.+8.*v0))/((v0**3)*(v0-1.)**3) \
                 - (Kp1*(2.-4.*v0))*(3./((v0**4)*(v0-1.)**3)
                                     +3./((v0**3)*(v0-1.)**4))

        # dQ/dphi, dQ/dpsi, dQ/dgamma
        for i in range(4):
            if active_mask[i] == 1:
                for j in range(4):
                    # Only interact with active species in A matrix effectively,
                    # but mathematically Q depends on all.
                    # However inactive phi_j/psi_j are 0.
                    # For safety, we check active_mask[j] or rely on 0 values.
                    # Relying on 0 values is safer if A is dense.
                    # But K entries for inactive columns should be 0?
                    # No, K is dQi/dXj.
                    # If i is active, Qi is biological.
                    # If j is inactive, Xj is forced to 0.
                    # Does Qi depend on Xj? Yes via Interaction.
                    # But Xj is locked to 0. So dQi/dXj is irrelevant?
                    # Actually, if Xj is locked to 0, dXj is 0.
                    # But in Newton step, we solve K * delta = -Q.
                    # If j is inactive, we want delta_j = 0.
                    # So the j-th row of K (equation for Xj) should be delta_j = 0 -> K[j,j]=1, others 0.
                    # The j-th column of K (dependence of other eqs on Xj):
                    # If we set K[i, j] correctly, it reflects dependence.
                    # But since we want delta_j=0, the j-th column values don't matter much if Q[j]=0.
                    # However, let's keep physical dependence just in case.
                    # Wait, if Xj is locked, dXj is not a variable.
                    # But we treat it as variable constrained to 0.
                    # So we should probably zero out K[i, j] if j is inactive?
                    # No, let's stick to:
                    # Row i (active): Standard derivatives.
                    # Row i (inactive): Dummy equation.
                    
                    K[i, j] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
                
                K[i, i] += phi_p[i] \
                           + (1./Eta_vec[i])*((Eta_phi_vec[i] + Eta_vec[i]*psi_new[i]**2)/dt
                                              + Eta_vec[i]*psi_new[i]*psidot[i]) \
                           - (c_val/Eta_vec[i])*(psi_new[i]*(Interaction[i] + A[i,i]*psi_new[i]))

                for j in range(4):
                    K[i, j+5] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
                
                K[i, i+5] += (1./Eta_vec[i])*(2.*Eta_vec[i]*psi_new[i]*phidot[i]
                                              + Eta_vec[i]*phi_new[i]*psidot[i]
                                              + Eta_vec[i]*phi_new[i]*psi_new[i]/dt) \
                             - (c_val/Eta_vec[i])*((Interaction[i] + A[i,i]*phi_new[i]*psi_new[i])
                                                   + psi_new[i]*(A[i,i]*phi_new[i]))
                K[i, 9] = 1./Eta_vec[i]
            else:
                # Inactive row i: Q[i] = phi[i]. dQ[i]/dphi[i] = 1.
                K[i, i] = 1.0

        K[4, 4] = phi0_p + 1./dt
        K[4, 9] = 1.0

        for i in range(4):
            k = i+5
            if active_mask[i] == 1:
                for j in range(4):
                    K[k, j] = -(c_val/Eta_vec[i])*(A[i,j]*psi_new[j]*phi_new[i]
                                                   + Interaction[i]*(1.0 if i==j else 0.0))
                K[k, i] += (psi_new[i]*phidot[i] + psi_new[i]*phi_new[i]/dt + 2.*phi_new[i]*psidot[i]) \
                           - (c_val/Eta_vec[i])*(A[i,i]*psi_new[i]*phi_new[i]
                                                 + Interaction[i] + phi_new[i]*A[i,i]*psi_new[i])
                for j in range(4):
                    K[k, j+5] = -(c_val/Eta_vec[i])*phi_new[i]*A[i,j]*phi_new[j]
                
                K[k, i+5] += psi_p[i] + (b_diag[i]*alpha_val/Eta_vec[i]) \
                             + (phi_new[i]*phidot[i] + phi_new[i]**2/dt) \
                             - (c_val/Eta_vec[i])*phi_new[i]*A[i,i]*phi_new[i]
            else:
                # Inactive row i+5: Q[i+5] = psi[i]. dQ/dpsi[i] = 1.
                K[k, k] = 1.0

        K[9, 0:5] = 1.0
        return K


    @njit(cache=False, fastmath=True)
    def _sigma2_accumulate_numba(x1, var_theta_active):
        n_time, n_state, n_theta = x1.shape
        sigma2 = np.zeros((n_time, n_state)) + 1e-12
        for k in range(n_theta):
            for t in range(n_time):
                for s in range(n_state):
                    sigma2[t, s] += (x1[t, s, k]**2) * var_theta_active[k]
        return sigma2


def _compute_Q_vector_numpy(phi_new, phi0_new, psi_new, gamma_new,
                            phi_old, phi0_old, psi_old,
                            dt, Kp1, Eta_vec, Eta_phi_vec,
                            c_val, alpha_val, A, b_diag, active_mask):
    """
    Same equations as Numba version, but pure NumPy and supports complex dtype.
    Used for complex-step differentiation in theta.
    """
    phi_new = np.asarray(phi_new)
    psi_new = np.asarray(psi_new)
    phi_old = np.asarray(phi_old)
    psi_old = np.asarray(psi_old)
    Eta_vec = np.asarray(Eta_vec)
    Eta_phi_vec = np.asarray(Eta_phi_vec)
    A = np.asarray(A)
    b_diag = np.asarray(b_diag)

    eps = 1e-12
    # Apply locking and clipping
    # Note: For complex step, we must be careful with clipping.
    # Usually complex step uses very small imaginary part. Clipping might kill it if close to bounds.
    # But here we assume we are inside bounds (eps..1-eps).
    # If inactive, force to 0 (real and imag).
    
    # We construct masks
    # active_mask is (4,) int
    # We iterate 4 species
    
    Q = np.zeros(10, dtype=np.result_type(phi_new, psi_new, phi0_new, gamma_new, A, b_diag))

    # Clipping and Locking logic
    for i in range(4):
        if active_mask[i] == 1:
            phi_new[i] = np.clip(phi_new[i].real, eps, 1.0 - eps) + 1j*phi_new[i].imag if np.iscomplexobj(phi_new) else np.clip(phi_new[i], eps, 1.0 - eps)
            psi_new[i] = np.clip(psi_new[i].real, eps, 1.0 - eps) + 1j*psi_new[i].imag if np.iscomplexobj(psi_new) else np.clip(psi_new[i], eps, 1.0 - eps)
        else:
            phi_new[i] = 0.0
            psi_new[i] = 0.0

    phi0_new = np.clip(phi0_new.real, eps, 1.0 - eps) + 1j*phi0_new.imag if np.iscomplexobj(phi0_new) else np.clip(phi0_new, eps, 1.0 - eps)

    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Interaction = A @ (phi_new * psi_new)

    for i in range(4):
        if active_mask[i] == 1:
            term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
            term2 = (1.0 / Eta_vec[i]) * (gamma_new
                                          + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i]
                                          + Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
            term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
            Q[i] = term1 + term2 - term3
        else:
            Q[i] = phi_new[i]

    Q[4] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot

    for i in range(4):
        if active_mask[i] == 1:
            term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) \
                    - (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
            term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
            term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
            term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
            Q[5+i] = term1 + term2 + term3 - term4
        else:
            Q[5+i] = psi_new[i]

    Q[9] = np.sum(phi_new) + phi0_new - 1.0
    return Q


# =============================================================================
# 2. JIT NEWTON + TIME INTEGRATION (exactly matching improved1207.py logic)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=False, fastmath=True)
    def _newton_step_jit(g_prev, dt, Kp1, Eta_vec, Eta_phi_vec, c_val, alpha_val, A, b_diag,
                         eps_tol, max_newton_iter, active_mask):
        """
        Damped Newton / line-search step (same structure as improved1207.py _newton_step).
        """
        g_new = g_prev.copy()
        
        # Force lock initially
        for i in range(4):
            if active_mask[i] == 0:
                g_new[i] = 0.0
                g_new[i+5] = 0.0

        for _ in range(max_newton_iter):
            # slices (copies) because numba kernels clip in-place
            phi_new = g_new[0:4].copy()
            phi0_new = g_new[4]
            psi_new = g_new[5:9].copy()
            gamma_new = g_new[9]

            phi_old = g_prev[0:4]
            phi0_old = g_prev[4]
            psi_old = g_prev[5:9]

            Q = _compute_Q_vector_numba(phi_new, phi0_new, psi_new, gamma_new,
                                        phi_old, phi0_old, psi_old,
                                        dt, Kp1, Eta_vec, Eta_phi_vec,
                                        c_val, alpha_val, A, b_diag, active_mask)
            # write back clipped/locked values so the iteration is consistent with kernels
            g_new[0:4] = phi_new
            g_new[4] = phi0_new
            g_new[5:9] = psi_new
            g_new[9] = gamma_new

            # NaN guard
            nan_found = False
            for i in range(10):
                if np.isnan(Q[i]):
                    nan_found = True
                    break
            if nan_found:
                break

            K = _compute_jacobian_numba(phi_new.copy(), phi0_new, psi_new.copy(), gamma_new,
                                        phi_old, psi_old,
                                        dt, Kp1, Eta_vec, Eta_phi_vec,
                                        c_val, alpha_val, A, b_diag, active_mask)

            # solve K * delta = -Q
            try:
                delta = np.linalg.solve(K, -Q)
            except Exception:
                K_reg = K + 1e-10 * np.eye(10)
                delta = np.linalg.solve(K_reg, -Q)

            # norm_Q = max(abs(Q))
            norm_Q = 0.0
            for i in range(10):
                v = abs(Q[i])
                if v > norm_Q:
                    norm_Q = v

            step = 1.0
            improved = False
            while step > 1e-4:
                g_trial = g_new + step * delta
                
                # Force lock on trial
                for i in range(4):
                    if active_mask[i] == 0:
                        g_trial[i] = 0.0
                        g_trial[i+5] = 0.0

                phi_t = g_trial[0:4].copy()
                phi0_t = g_trial[4]
                psi_t = g_trial[5:9].copy()
                gamma_t = g_trial[9]

                Q_trial = _compute_Q_vector_numba(phi_t, phi0_t, psi_t, gamma_t,
                                                  phi_old, phi0_old, psi_old,
                                                  dt, Kp1, Eta_vec, Eta_phi_vec,
                                                  c_val, alpha_val, A, b_diag, active_mask)

                # NaN check
                nan2 = False
                for i in range(10):
                    if np.isnan(Q_trial[i]):
                        nan2 = True
                        break
                if not nan2:
                    norm_trial = 0.0
                    for i in range(10):
                        v = abs(Q_trial[i])
                        if v > norm_trial:
                            norm_trial = v
                    if norm_trial < norm_Q:
                        g_new = g_trial
                        norm_Q = norm_trial
                        improved = True
                        break

                step *= 0.5

            if not improved:
                g_new = g_new + delta
                # Force lock on final step fallback
                for i in range(4):
                    if active_mask[i] == 0:
                        g_new[i] = 0.0
                        g_new[i+5] = 0.0

            if norm_Q < eps_tol:
                break

        return g_new


    @njit(cache=False, fastmath=True)
    def _run_deterministic_jit(theta, dt, maxtimestep, eps_base, Kp1,
                               Eta_vec, Eta_phi_vec, c_val, alpha_val,
                               phi_init, active_species_mask):
        """
        Deterministic time integration (same as improved1207.py run_deterministic).
        active_species_mask: (4,) boolean-like int array (1 active, 0 inactive).
        """
        # build initial state
        g_prev = np.zeros(10)
        # phi1..phi4
        for i in range(4):
            g_prev[i] = phi_init if active_species_mask[i] == 1 else 0.0
        # phi0
        g_prev[4] = 1.0 - np.sum(g_prev[0:4])
        # psi1..psi4
        for i in range(4):
            g_prev[5+i] = 0.999 if active_species_mask[i] == 1 else 0.0
        # gamma
        g_prev[9] = 0.0

        # theta -> A, b_diag (same as solver.theta_to_matrices)
        A = np.zeros((4, 4))
        b_diag = np.zeros(4)

        # M1 block
        A[0, 0] = theta[0]
        A[0, 1] = theta[1]; A[1, 0] = theta[1]
        A[1, 1] = theta[2]
        b_diag[0] = theta[3]
        b_diag[1] = theta[4]

        # M2 block
        A[2, 2] = theta[5]
        A[2, 3] = theta[6]; A[3, 2] = theta[6]
        A[3, 3] = theta[7]
        b_diag[2] = theta[8]
        b_diag[3] = theta[9]

        # M3 cross
        A[0, 2] = theta[10]; A[2, 0] = theta[10]
        A[0, 3] = theta[11]; A[3, 0] = theta[11]
        A[1, 2] = theta[12]; A[2, 1] = theta[12]
        A[1, 3] = theta[13]; A[3, 1] = theta[13]

        t_arr = np.empty(maxtimestep + 1, dtype=np.float64)
        g_arr = np.empty((maxtimestep + 1, 10), dtype=np.float64)
        t_arr[0] = 0.0
        g_arr[0] = g_prev

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            # time-dependent tolerance (verbatim structure: eps*(1+10*tt))
            tol_t = eps_base * (1.0 + 10.0 * tt)

            g_new = _newton_step_jit(
                g_prev, dt, Kp1, Eta_vec, Eta_phi_vec,
                c_val, alpha_val, A, b_diag,
                tol_t, 50, active_species_mask
            )

            g_prev = g_new
            t_arr[step + 1] = tt
            g_arr[step + 1] = g_prev

        return t_arr, g_arr


# =============================================================================
# 3. SOLVER CLASSES (API compatible with improved1207.py)
# =============================================================================

class BiofilmNewtonSolver:
    THETA_NAMES = [
        "a11","a12","a22","b1","b2",
        "a33","a34","a44","b3","b4",
        "a13","a14","a23","a24"
    ]

    def __init__(
        self,
        dt: float = 1e-5,
        maxtimestep: int = 500,
        eps: float = 1e-6,
        Kp1: float = 1e-4,
        eta_vec=None,
        eta_phi_vec=None,
        c_const: float = 100.0,
        alpha_const: float = 100.0,
        alpha_schedule: dict | None = None,
        phi_init: float = 0.2,
        active_species=None,
        use_numba: bool = True,
        max_newton_iter: int = 50,
    ):
        self.dt = float(dt)
        self.maxtimestep = int(maxtimestep)
        self.eps = float(eps)
        self.Kp1 = float(Kp1)
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)
        # Optional time-dependent antibiotics schedule (piecewise constant).
        self.alpha_schedule = alpha_schedule
        self.phi_init = float(phi_init)
        self.max_newton_iter = int(max_newton_iter)

        if active_species is None:
            active_species = [0, 1, 2, 3]
        self.active_species = list(active_species)
        
        # Precompute active mask
        self.active_mask = np.zeros(4, dtype=np.int64)
        for i in self.active_species:
            if 0 <= i < 4:
                self.active_mask[i] = 1

        if eta_vec is None:
            eta_vec = np.ones(4, dtype=float)
        if eta_phi_vec is None:
            eta_phi_vec = np.ones(4, dtype=float)

        self.Eta_vec = np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = np.asarray(eta_phi_vec, dtype=float)

        self.use_numba = bool(use_numba and HAS_NUMBA)

    def c(self, t: float) -> float:
        return float(self.c_const)

    def alpha(self, t: float) -> float:
        if not self.alpha_schedule:
            return float(self.alpha_const)

        sched = self.alpha_schedule
        try:
            alpha_before = float(sched.get("alpha_before", self.alpha_const))
            alpha_after = float(sched.get("alpha_after", self.alpha_const))
        except Exception:
            return float(self.alpha_const)

        # Switch by time
        if "switch_time" in sched:
            try:
                t_switch = float(sched["switch_time"])
                return alpha_after if float(t) >= t_switch else alpha_before
            except Exception:
                return float(self.alpha_const)

        # Switch by fraction of total duration (0..1)
        if "switch_frac" in sched:
            try:
                frac = float(sched["switch_frac"])
                frac = min(max(frac, 0.0), 1.0)
                t_switch = frac * float(self.maxtimestep) * float(self.dt)
                return alpha_after if float(t) >= t_switch else alpha_before
            except Exception:
                return float(self.alpha_const)

        # Switch by discrete step index
        if "switch_step" in sched:
            try:
                step_switch = int(sched["switch_step"])
                # t is produced by run_deterministic as step*dt
                step = int(np.floor(float(t) / float(self.dt) + 1e-12))
                return alpha_after if step >= step_switch else alpha_before
            except Exception:
                return float(self.alpha_const)

        return float(self.alpha_const)

    def theta_to_matrices(self, theta):
        theta = np.asarray(theta)
        dtype = np.complex128 if np.iscomplexobj(theta) else np.float64

        A = np.zeros((4, 4), dtype=dtype)
        b_diag = np.zeros(4, dtype=dtype)

        # M1
        A[0, 0] = theta[0]
        A[0, 1] = theta[1]; A[1, 0] = theta[1]
        A[1, 1] = theta[2]
        b_diag[0] = theta[3]
        b_diag[1] = theta[4]

        # M2
        A[2, 2] = theta[5]
        A[2, 3] = theta[6]; A[3, 2] = theta[6]
        A[3, 3] = theta[7]
        b_diag[2] = theta[8]
        b_diag[3] = theta[9]

        # M3
        A[0, 2] = theta[10]; A[2, 0] = theta[10]
        A[0, 3] = theta[11]; A[3, 0] = theta[11]
        A[1, 2] = theta[12]; A[2, 1] = theta[12]
        A[1, 3] = theta[13]; A[3, 1] = theta[13]

        return A, b_diag

    def get_initial_state(self):
        g = np.zeros(10, dtype=float)
        for i in range(4):
            g[i] = self.phi_init if (i in self.active_species) else 0.0
        g[4] = 1.0 - np.sum(g[0:4])  # phi0
        for i in range(4):
            g[5+i] = 0.999 if (i in self.active_species) else 0.0
        g[9] = 0.0  # gamma
        return g

    def compute_Q_vector(self, g_new, g_old, t, dt, A, b_diag):
        """
        Residual Q(g_new) in the same order as g.
        If g/theta is complex -> uses NumPy backend (complex-step safe).
        """
        phi_new = g_new[0:4]
        phi0_new = g_new[4]
        psi_new = g_new[5:9]
        gamma_new = g_new[9]

        phi_old = g_old[0:4]
        phi0_old = g_old[4]
        psi_old = g_old[5:9]

        if np.iscomplexobj(g_new) or np.iscomplexobj(A) or np.iscomplexobj(b_diag):
            return _compute_Q_vector_numpy(
                phi_new, phi0_new, psi_new, gamma_new,
                phi_old, phi0_old, psi_old,
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag, self.active_mask
            )

        if not self.use_numba:
            return _compute_Q_vector_numpy(
                phi_new, phi0_new, psi_new, gamma_new,
                phi_old, phi0_old, psi_old,
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag, self.active_mask
            )

        # numba backend (real)
        return _compute_Q_vector_numba(
            phi_new.copy(), phi0_new, psi_new.copy(), gamma_new,
            phi_old, phi0_old, psi_old,
            dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
            self.c(t), self.alpha(t), A.real.astype(np.float64), b_diag.real.astype(np.float64),
            self.active_mask
        )

    def compute_Jacobian_matrix(self, g_new, g_old, t, dt, A, b_diag):
        if not self.use_numba:
            # fallback finite-difference Jacobian (same as improved1207.py structure)
            x = g_new
            n = len(x)
            J = np.zeros((n, n), dtype=float)
            Q0 = self.compute_Q_vector(g_new, g_old, t, dt, A, b_diag)
            eps = 1e-8
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] += eps
                Qp = self.compute_Q_vector(x_pert, g_old, t, dt, A, b_diag)
                J[:, j] = (Qp - Q0) / eps
            return J

        return _compute_jacobian_numba(
            g_new[0:4].copy(), g_new[4], g_new[5:9].copy(), g_new[9],
            g_old[0:4], g_old[5:9],
            dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
            self.c(t), self.alpha(t), A.real.astype(np.float64), b_diag.real.astype(np.float64),
            self.active_mask
        )

    def _newton_step(self, g_prev, t, dt, A, b_diag):
        """
        Keep the original Python version for safety / fallback.
        The JIT path is used in run_deterministic when use_numba=True.
        """
        g_new = g_prev.copy()
        for _ in range(self.max_newton_iter):
            Q = self.compute_Q_vector(g_new, g_prev, t, dt, A, b_diag)
            if np.isnan(Q).any():
                break
            K = self.compute_Jacobian_matrix(g_new, g_prev, t, dt, A, b_diag)

            try:
                delta = np.linalg.solve(K, -Q)
            except np.linalg.LinAlgError:
                K_reg = K + 1e-10 * np.eye(K.shape[0])
                delta = np.linalg.solve(K_reg, -Q)

            norm_Q = np.max(np.abs(Q))
            step = 1.0
            improved = False
            while step > 1e-4:
                g_trial = g_new + step * delta
                Q_trial = self.compute_Q_vector(g_trial, g_prev, t, dt, A, b_diag)
                if not np.isnan(Q_trial).any():
                    norm_trial = np.max(np.abs(Q_trial))
                    if norm_trial < norm_Q:
                        g_new = g_trial
                        norm_Q = norm_trial
                        improved = True
                        break
                step *= 0.5

            if not improved:
                g_new = g_new + delta

            if norm_Q < self.eps:
                break

        return g_new

    def run_deterministic(self, theta, show_progress: bool = False):
        """
        Deterministic run.
        - If use_numba=True: uses the JIT integrator for speed (same math).
        - Else: uses the original Python loop.
        """
        theta = np.asarray(theta, dtype=np.float64)
        A, b_diag = self.theta_to_matrices(theta)

        if self.use_numba and HAS_NUMBA:
            t_arr, g_arr = _run_deterministic_jit(
                theta, self.dt, self.maxtimestep, self.eps, self.Kp1,
                self.Eta_vec.astype(np.float64), self.Eta_phi_vec.astype(np.float64),
                self.c_const, self.alpha_const, self.phi_init, self.active_mask
            )
            return t_arr, g_arr

        # fallback python
        dt = self.dt
        g_prev = self.get_initial_state()
        t_list, g_list = [0.0], [g_prev.copy()]
        for step in range(self.maxtimestep):
            tt = (step + 1) * dt
            tol_t = self.eps * (1.0 + 10.0 * tt)
            g_new = self._newton_step(g_prev, tt, dt, A, b_diag)
            g_prev = g_new.copy()
            t_list.append(tt)
            g_list.append(g_prev.copy())
        return np.array(t_list), np.vstack(g_list)


class BiofilmTSM:
    """
    TSM-ROM solver (same mathematical structure as improved1207.py).
    - Deterministic trajectory uses solver.run_deterministic (JIT accelerated if enabled).
    - x1 propagation uses complex-step dG/dtheta (paper-consistent, robust).
    """

    def __init__(self, solver: BiofilmNewtonSolver, active_theta_indices=None,
                 cov_rel: float = 0.005, use_complex_step: bool = True):
        self.solver = solver
        self.cov_rel = float(cov_rel)
        self.use_complex_step = bool(use_complex_step)
        if active_theta_indices is None:
            active_theta_indices = np.arange(14, dtype=int)
        self.active_idx = np.array(list(active_theta_indices), dtype=int)

    def solve_tsm(self, theta: np.ndarray):
        theta = np.asarray(theta)

        # deterministic
        t_arr, g_det = self.solver.run_deterministic(np.real(theta).astype(np.float64))
        n_time = len(t_arr)

        # store x0 (= mean) and x1 (sensitivities)
        x0_list = [g_det[0].astype(np.complex128 if np.iscomplexobj(theta) else np.float64)]
        x1_list = []

        # precompute A, b_diag possibly complex for complex-step in theta
        A0, b0 = self.solver.theta_to_matrices(theta)

        h = 1e-30  # complex-step
        # ★ FIX: use actual time steps (n_time - 1) instead of maxtimestep
        # This guarantees consistency between t_arr, x0, and sigma2 shapes.
        for step in range(n_time - 1):
            tt = t_arr[step+1]
            g_old = g_det[step]
            g_new = g_det[step+1]

            # Jacobian at (g_new, g_old)
            J = self.solver.compute_Jacobian_matrix(g_new, g_old, tt, self.solver.dt, A0, b0)

            # dG/dtheta via complex-step (robust, no subtraction cancellation)
            x1_t = np.zeros((10, len(self.active_idx)), dtype=np.float64)

            for k, idx in enumerate(self.active_idx):
                th_cs = theta.astype(np.complex128)
                th_cs[idx] += 1j * h
                A_cs, b_cs = self.solver.theta_to_matrices(th_cs)

                Q_cs = self.solver.compute_Q_vector(
                    g_new.astype(np.complex128), g_old.astype(np.complex128),
                    tt, self.solver.dt, A_cs, b_cs
                )
                dG = np.imag(Q_cs) / h  # (10,)

                rhs = -dG
                try:
                    x1_t[:, k] = np.linalg.solve(J, rhs)
                except np.linalg.LinAlgError:
                    J_reg = J + 1e-10 * np.eye(J.shape[0])
                    x1_t[:, k] = np.linalg.solve(J_reg, rhs)

            x1_list.append(x1_t)
            x0_list.append(g_new.copy())

        x0 = np.vstack(x0_list).real  # (n_time, 10)

        # x1_list has length n_time - 1 (sensitivities for t[1:] based on steps 0..n_time-2)
        x1_core = np.stack(x1_list, axis=0)  # (n_time - 1, 10, n_active)

        # ★ FIX: align sigma2 with all time points in t_arr
        # Define x1_full such that:
        #   x1_full[1:] ↔ sensitivities at t[1:], x1_full[0] copied from first step.
        x1 = np.zeros((n_time, 10, len(self.active_idx)), dtype=np.float64)
        x1[1:, :, :] = x1_core
        x1[0, :, :] = x1_core[0, :, :]  # harmless; idx_sparse never uses very early times

        # variance of theta (p-box: mean ± CoV)  (same as improved1207.py)
        var_th = (self.cov_rel * np.real(theta))**2
        var_act = var_th[self.active_idx]

        if HAS_NUMBA:
            sigma2 = _sigma2_accumulate_numba(x1, var_act)
        else:
            sigma2 = np.sum((x1**2) * var_act[None, None, :], axis=2) + 1e-12

        # Expose last sensitivities/variances for downstream covariance calculations.
        # This is useful for computing Cov(phi, psi) without changing the public API.
        self._last_x1 = x1
        self._last_var_act = var_act
        self._last_active_idx = self.active_idx
        self._last_theta = np.real(theta).astype(np.float64, copy=False)
        self._last_t_arr = t_arr

        return t_arr, x0, sigma2


__all__ = [
    "HAS_NUMBA",
    "get_theta_true",
    "BiofilmNewtonSolver",
    "BiofilmTSM",
]
