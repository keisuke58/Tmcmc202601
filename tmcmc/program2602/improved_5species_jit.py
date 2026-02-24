# -*- coding: utf-8 -*-
"""
improved_5species_jit.py

JIT-compiled Newton solver for 5-species biofilm model (N=5).
Extends improved1207_paper_jit.py (N=4) to include P.gingivalis (Species 5).

State g (12,):
    [phi1, phi2, phi3, phi4, phi5, phi0,  psi1, psi2, psi3, psi4, psi5, gamma]
    - phi: 5 species (indices 0-4)
    - phi0: 1 variable (index 5)
    - psi: 5 species (indices 6-10)
    - gamma: 1 variable (index 11)

Theta (20,):
    0-13: Same as 4-species model (M1, M2, M3 blocks)
    14: a55 (P.g self)
    15: b5 (P.g decay)
    16: a15 (S.o-P.g)
    17: a25 (A.n-P.g)
    18: a35 (Vei-P.g)
    19: a45 (F.n-P.g)
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
# 1. RESIDUAL + JACOBIAN KERNELS (N=5)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=False, fastmath=True)
    def _compute_Q_vector_numba_5s(phi_new, phi0_new, psi_new, gamma_new,
                                   phi_old, phi0_old, psi_old,
                                   dt, Kp1, Eta_vec, Eta_phi_vec,
                                   c_val, alpha_val, A, b_diag):
        # Clip to [eps, 1-eps]
        eps = 1e-12
        for i in range(5):
            if phi_new[i] < eps:
                phi_new[i] = eps
            elif phi_new[i] > 1.0 - eps:
                phi_new[i] = 1.0 - eps

            if psi_new[i] < eps:
                psi_new[i] = eps
            elif psi_new[i] > 1.0 - eps:
                psi_new[i] = 1.0 - eps

        if phi0_new < eps:
            phi0_new = eps
        elif phi0_new > 1.0 - eps:
            phi0_new = 1.0 - eps

        Q = np.zeros(12)
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt

        Interaction = A @ (phi_new * psi_new)

        # Q[0..4] for phi_i
        for i in range(5):
            term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
            term2 = (1.0 / Eta_vec[i]) * (gamma_new
                                          + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i]
                                          + Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
            term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
            Q[i] = term1 + term2 - term3

        # Q[5] for phi0
        Q[5] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot

        # Q[6..10] for psi_i
        for i in range(5):
            term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) \
                    - (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
            term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
            term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
            term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
            Q[6+i] = term1 + term2 + term3 - term4

        # constraint
        Q[11] = np.sum(phi_new) + phi0_new - 1.0
        return Q

    @njit(cache=False, fastmath=True)
    def _compute_jacobian_numba_5s(phi_new, phi0_new, psi_new, gamma_new,
                                   phi_old, psi_old,
                                   dt, Kp1, Eta_vec, Eta_phi_vec,
                                   c_val, alpha_val, A, b_diag):
        K = np.zeros((12, 12))

        # Clip
        eps = 1e-12
        for i in range(5):
            if phi_new[i] < eps: phi_new[i] = eps
            elif phi_new[i] > 1.0 - eps: phi_new[i] = 1.0 - eps
            if psi_new[i] < eps: psi_new[i] = eps
            elif psi_new[i] > 1.0 - eps: psi_new[i] = 1.0 - eps
        if phi0_new < eps: phi0_new = eps
        elif phi0_new > 1.0 - eps: phi0_new = 1.0 - eps

        phidot = (phi_new - phi_old) / dt
        psidot = (psi_new - psi_old) / dt
        Interaction = A @ (phi_new * psi_new)

        phi_p = np.zeros(5)
        psi_p = np.zeros(5)
        for i in range(5):
            v = phi_new[i]
            phi_p[i] = (Kp1*(-4.+8.*v))/((v**3)*(v-1.)**3) \
                       - (Kp1*(2.-4.*v))*(3./((v**4)*(v-1.)**3) + 3./((v**3)*(v-1.)**4))
            v = psi_new[i]
            psi_p[i] = (4.*Kp1*(3.-5.*v+5.*v**2))/((v**4)*(v-1.)**4)

        v0 = phi0_new
        phi0_p = (Kp1*(-4.+8.*v0))/((v0**3)*(v0-1.)**3) \
                 - (Kp1*(2.-4.*v0))*(3./((v0**4)*(v0-1.)**3) + 3./((v0**3)*(v0-1.)**4))

        # dQ/dphi, dQ/dpsi, dQ/dgamma
        for i in range(5):
            for j in range(5):
                K[i, j] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] += phi_p[i] \
                       + (1./Eta_vec[i])*((Eta_phi_vec[i] + Eta_vec[i]*psi_new[i]**2)/dt
                                          + Eta_vec[i]*psi_new[i]*psidot[i]) \
                       - (c_val/Eta_vec[i])*(psi_new[i]*(Interaction[i] + A[i,i]*psi_new[i]))

            for j in range(5):
                K[i, j+6] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
            K[i, i+6] += (1./Eta_vec[i])*(2.*Eta_vec[i]*psi_new[i]*phidot[i]
                                          + Eta_vec[i]*phi_new[i]*psidot[i]
                                          + Eta_vec[i]*phi_new[i]*psi_new[i]/dt) \
                         - (c_val/Eta_vec[i])*((Interaction[i] + A[i,i]*phi_new[i]*psi_new[i])
                                               + psi_new[i]*(A[i,i]*phi_new[i]))
            K[i, 11] = 1./Eta_vec[i]

        K[5, 5] = phi0_p + 1./dt
        K[5, 11] = 1.0

        for i in range(5):
            k = i+6
            for j in range(5):
                K[k, j] = -(c_val/Eta_vec[i])*(A[i,j]*psi_new[j]*phi_new[i]
                                               + Interaction[i]*(1.0 if i==j else 0.0))
            K[k, i] += (psi_new[i]*phidot[i] + psi_new[i]*phi_new[i]/dt + 2.*phi_new[i]*psidot[i]) \
                       - (c_val/Eta_vec[i])*(A[i,i]*psi_new[i]*phi_new[i]
                                             + Interaction[i] + phi_new[i]*A[i,i]*psi_new[i])
            for j in range(5):
                K[k, j+6] = -(c_val/Eta_vec[i])*phi_new[i]*A[i,j]*phi_new[j]
            K[k, i+6] += psi_p[i] + (b_diag[i]*alpha_val/Eta_vec[i]) \
                         + (phi_new[i]*phidot[i] + phi_new[i]**2/dt) \
                         - (c_val/Eta_vec[i])*phi_new[i]*A[i,i]*phi_new[i]

        K[11, 0:6] = 1.0
        return K

    @njit(cache=False, fastmath=True)
    def _newton_step_jit_5s(g_prev, dt, Kp1, Eta_vec, Eta_phi_vec, c_val, alpha_val, A, b_diag,
                            eps_tol, max_newton_iter):
        g_new = g_prev.copy()
        for _ in range(max_newton_iter):
            phi_new = g_new[0:5].copy()
            phi0_new = g_new[5]
            psi_new = g_new[6:11].copy()
            gamma_new = g_new[11]

            phi_old = g_prev[0:5]
            phi0_old = g_prev[5]
            psi_old = g_prev[6:11]

            Q = _compute_Q_vector_numba_5s(phi_new, phi0_new, psi_new, gamma_new,
                                           phi_old, phi0_old, psi_old,
                                           dt, Kp1, Eta_vec, Eta_phi_vec,
                                           c_val, alpha_val, A, b_diag)
            g_new[0:5] = phi_new
            g_new[5] = phi0_new
            g_new[6:11] = psi_new
            g_new[11] = gamma_new

            nan_found = False
            for i in range(12):
                if np.isnan(Q[i]):
                    nan_found = True
                    break
            if nan_found:
                break

            K = _compute_jacobian_numba_5s(phi_new.copy(), phi0_new, psi_new.copy(), gamma_new,
                                           phi_old, psi_old,
                                           dt, Kp1, Eta_vec, Eta_phi_vec,
                                           c_val, alpha_val, A, b_diag)

            try:
                delta = np.linalg.solve(K, -Q)
            except Exception:
                K_reg = K + 1e-10 * np.eye(12)
                delta = np.linalg.solve(K_reg, -Q)

            norm_Q = 0.0
            for i in range(12):
                v = abs(Q[i])
                if v > norm_Q: norm_Q = v

            step = 1.0
            improved = False
            while step > 1e-4:
                g_trial = g_new + step * delta
                phi_t = g_trial[0:5].copy()
                phi0_t = g_trial[5]
                psi_t = g_trial[6:11].copy()
                gamma_t = g_trial[11]

                Q_trial = _compute_Q_vector_numba_5s(phi_t, phi0_t, psi_t, gamma_t,
                                                     phi_old, phi0_old, psi_old,
                                                     dt, Kp1, Eta_vec, Eta_phi_vec,
                                                     c_val, alpha_val, A, b_diag)
                nan2 = False
                for i in range(12):
                    if np.isnan(Q_trial[i]):
                        nan2 = True
                        break
                if not nan2:
                    norm_trial = 0.0
                    for i in range(12):
                        v = abs(Q_trial[i])
                        if v > norm_trial: norm_trial = v
                    if norm_trial < norm_Q:
                        g_new = g_trial
                        norm_Q = norm_trial
                        improved = True
                        break
                step *= 0.5

            if not improved:
                g_new = g_new + delta

            if norm_Q < eps_tol:
                break
        return g_new

# =============================================================================
# 2. SOLVER CLASS (N=5)
# =============================================================================

class BiofilmNewtonSolver5S:
    THETA_NAMES = [
        "a11","a12","a22","b1","b2",
        "a33","a34","a44","b3","b4",
        "a13","a14","a23","a24",
        "a55","b5","a15","a25","a35","a45"
    ]

    def __init__(self,
                 dt=1e-5, maxtimestep=500, eps=1e-6, Kp1=1e-4,
                 eta_vec=None, eta_phi_vec=None,
                 c_const=100.0, alpha_const=100.0,
                 alpha_schedule=None, phi_init=0.2,
                 active_species=None, use_numba=True,
                 max_newton_iter=50):
        self.dt = float(dt)
        self.maxtimestep = int(maxtimestep)
        self.eps = float(eps)
        self.Kp1 = float(Kp1)
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)
        self.alpha_schedule = alpha_schedule
        self.phi_init = float(phi_init)
        self.max_newton_iter = int(max_newton_iter)

        if active_species is None:
            active_species = [0, 1, 2, 3, 4]
        self.active_species = list(active_species)

        if eta_vec is None: eta_vec = np.ones(5, dtype=float)
        if eta_phi_vec is None: eta_phi_vec = np.ones(5, dtype=float)

        self.Eta_vec = np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = np.asarray(eta_phi_vec, dtype=float)
        self.use_numba = bool(use_numba and HAS_NUMBA)

    def theta_to_matrices(self, theta):
        """Map 20 parameters to A(5x5) and b(5)."""
        A = np.zeros((5, 5))
        b_diag = np.zeros(5)

        # M1 (0,1)
        A[0,0] = theta[0]
        A[0,1] = theta[1]; A[1,0] = theta[1]
        A[1,1] = theta[2]
        b_diag[0] = theta[3]
        b_diag[1] = theta[4]

        # M2 (2,3)
        A[2,2] = theta[5]
        A[2,3] = theta[6]; A[3,2] = theta[6]
        A[3,3] = theta[7]
        b_diag[2] = theta[8]
        b_diag[3] = theta[9]

        # M3 Cross (0,1) <-> (2,3)
        A[0,2] = theta[10]; A[2,0] = theta[10]
        A[0,3] = theta[11]; A[3,0] = theta[11]
        A[1,2] = theta[12]; A[2,1] = theta[12]
        A[1,3] = theta[13]; A[3,1] = theta[13]

        # Species 5 (index 4)
        A[4,4] = theta[14]
        b_diag[4] = theta[15]

        # Cross 5 with others
        A[0,4] = theta[16]; A[4,0] = theta[16] # S.o-P.g
        A[1,4] = theta[17]; A[4,1] = theta[17] # A.n-P.g
        A[2,4] = theta[18]; A[4,2] = theta[18] # Vei-P.g
        A[3,4] = theta[19]; A[4,3] = theta[19] # F.n-P.g

        return A, b_diag

    def run_deterministic(self, theta):
        A, b_diag = self.theta_to_matrices(theta)
        
        # Initial state
        g_prev = np.zeros(12)
        active_mask = np.zeros(5, dtype=np.int32)
        for i in self.active_species:
            if 0 <= i < 5:
                active_mask[i] = 1

        for i in range(5):
            g_prev[i] = self.phi_init if active_mask[i] else 0.0
        g_prev[5] = 1.0 - np.sum(g_prev[0:5])
        for i in range(5):
            g_prev[6+i] = 0.999 if active_mask[i] else 0.0
        g_prev[11] = 0.0

        if self.use_numba:
            # Need JIT wrapper for loop
            return self._run_loop_jit(theta, g_prev, A, b_diag)
        else:
            # Fallback python loop (not implemented fully for brevity, assume numba)
            # Or implement simple python loop if needed
            raise NotImplementedError("Pure Python fallback not implemented for 5S solver yet.")

    def _run_loop_jit(self, theta, g_prev, A, b_diag):
        return _run_deterministic_jit_5s(
            g_prev, self.dt, self.maxtimestep, self.eps, self.Kp1,
            self.Eta_vec, self.Eta_phi_vec, self.c_const, self.alpha_const,
            A, b_diag, self.max_newton_iter
        )

@njit(cache=False, fastmath=True)
def _run_deterministic_jit_5s(g_prev, dt, maxtimestep, eps_base, Kp1,
                              Eta_vec, Eta_phi_vec, c_val, alpha_val,
                              A, b_diag, max_newton_iter):
    t_arr = np.empty(maxtimestep + 1, dtype=np.float64)
    g_arr = np.empty((maxtimestep + 1, 12), dtype=np.float64)
    t_arr[0] = 0.0
    g_arr[0] = g_prev

    for step in range(maxtimestep):
        tt = (step + 1) * dt
        tol_t = eps_base * (1.0 + 10.0 * tt)
        
        g_new = _newton_step_jit_5s(
            g_prev, dt, Kp1, Eta_vec, Eta_phi_vec,
            c_val, alpha_val, A, b_diag,
            tol_t, max_newton_iter
        )
        g_prev = g_new
        t_arr[step + 1] = tt
        g_arr[step + 1] = g_prev
        
    return t_arr, g_arr
