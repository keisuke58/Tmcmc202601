#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating
==============================================================
Complete Version with:
- Numba JIT acceleration
- Full data saving (DEBUG=False only)
- Paper-level figures (Fritsch et al. 2025 style)

Features:
- 4-species continuum biofilm model (Junker-type Newton solver)
- TSM (Time-Separated Mechanics, 1st-order Taylor)
- TMCMC (Transitional MCMC, Ching & Chen style)
- Hierarchical Case II: M1 -> M2 -> M3

Table 3 settings:
    M1: c* = 100, α* = 100, N=2500, dt=1e-5, η = [1,1,1,1]
    M2: c* = 100, α* = 10,  N=5000, dt=1e-5, η = [1,1,1,1]
    M3: c* = 25,  α* = 0,   N=750,  dt=1e-4, η = [1,1,1,1]

Author: Based on Fritsch et al. (2025)
"""

import numpy as np
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# NUMBA ACCELERATION
# =============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("✓ Numba JIT acceleration: ENABLED")
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not available: using pure NumPy (slower)")

# Numba-accelerated core functions
if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_Q_vector_numba(phi_new, phi0_new, psi_new, gamma_new,
                                 phi_old, phi0_old, psi_old,
                                 dt, Kp1, Eta_vec, Eta_phi_vec,
                                 c_val, alpha_val, A, b_diag):
        """Numba-accelerated Q vector computation"""
        Q = np.zeros(10)
        
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt
        
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        
        # Q[0:4] - phi equations
        for i in range(4):
            term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / (
                (phi_new[i] - 1.0)**3 * phi_new[i]**3)
            term2 = (1.0 / Eta_vec[i]) * (
                gamma_new + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i] +
                Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
            term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
            Q[i] = term1 + term2 - term3
        
        # Q[4] - phi0 equation
        Q[4] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / (
            (phi0_new - 1.0)**3 * phi0_new**3) + phi0dot
        
        # Q[5:9] - psi equations
        for i in range(4):
            term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) - \
                    (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
            term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
            term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
            term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
            Q[5+i] = term1 + term2 + term3 - term4
        
        # Q[9] - constraint
        Q[9] = phi_new[0] + phi_new[1] + phi_new[2] + phi_new[3] + phi0_new - 1.0
        
        return Q

    @njit(cache=True, fastmath=True)
    def _compute_jacobian_numba(phi_new, phi0_new, psi_new, gamma_new,
                                 phi_old, psi_old, dt, Kp1,
                                 Eta_vec, Eta_phi_vec, c_val, alpha_val, A, b_diag):
        """Numba-accelerated Jacobian computation"""
        K = np.zeros((10, 10))
        
        phidot = (phi_new - phi_old) / dt
        psidot = (psi_new - psi_old) / dt
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        
        # Derivatives of potential terms
        phi_p_deriv = np.zeros(4)
        for i in range(4):
            v = phi_new[i]
            phi_p_deriv[i] = (Kp1*(-4. + 8.*v))/((v**3)*(v-1.)**3) - \
                             (Kp1*(2. - 4.*v))*(3./((v**4)*(v-1.)**3) + 3./((v**3)*(v-1.)**4))
        
        v0 = phi0_new
        phi0_p_deriv = (Kp1*(-4. + 8.*v0))/((v0**3)*(v0-1.)**3) - \
                       (Kp1*(2. - 4.*v0))*(3./((v0**4)*(v0-1.)**3) + 3./((v0**3)*(v0-1.)**4))
        
        psi_p_deriv = np.zeros(4)
        for i in range(4):
            v = psi_new[i]
            psi_p_deriv[i] = (4.0 * Kp1 * (3.0 - 5.0*v + 5.0*v**2)) / ((v**4) * (v - 1.0)**4)
        
        # Fill Jacobian rows 0-3 (phi equations)
        for i in range(4):
            for j in range(4):
                K[i, j] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] = phi_p_deriv[i] + (1.0 / Eta_vec[i]) * (
                (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) / dt +
                Eta_vec[i] * psi_new[i] * psidot[i]) - \
                (c_val / Eta_vec[i]) * (psi_new[i] * (Interaction[i] + A[i, i] * psi_new[i]))
            K[i, 4] = 0.0
            for j in range(4):
                K[i, j+5] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
            K[i, i+5] = (1.0 / Eta_vec[i]) * (
                2.0 * Eta_vec[i] * psi_new[i] * phidot[i] +
                Eta_vec[i] * phi_new[i] * psidot[i] +
                Eta_vec[i] * phi_new[i] * psi_new[i] / dt) - \
                (c_val / Eta_vec[i]) * ((Interaction[i] + A[i, i] * phi_new[i] * psi_new[i]) +
                                         psi_new[i] * (A[i, i] * phi_new[i]))
            K[i, 9] = 1.0 / Eta_vec[i]
        
        # Row 4 (phi0 equation)
        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 9] = 1.0
        
        # Rows 5-8 (psi equations)
        for i in range(4):
            k = i + 5
            for j in range(4):
                K[k, j] = -(c_val / Eta_vec[i]) * (A[i, j] * psi_new[j] * phi_new[i] +
                           Interaction[i] * (1.0 if i == j else 0.0))
            K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt +
                       2.0 * phi_new[i] * psidot[i]) - \
                      (c_val / Eta_vec[i]) * (A[i, i] * psi_new[i] * phi_new[i] +
                                               Interaction[i] + phi_new[i] * A[i, i] * psi_new[i])
            K[k, 4] = 0.0
            for j in range(4):
                K[k, j+5] = -(c_val / Eta_vec[i]) * phi_new[i] * A[i, j] * phi_new[j]
            K[k, i+5] = psi_p_deriv[i] + (b_diag[i] * alpha_val / Eta_vec[i]) + \
                        (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                        (c_val / Eta_vec[i]) * phi_new[i] * A[i, i] * phi_new[i]
            K[k, 9] = 0.0
        
        # Row 9 (constraint)
        K[9, 0] = 1.0
        K[9, 1] = 1.0
        K[9, 2] = 1.0
        K[9, 3] = 1.0
        K[9, 4] = 1.0
        
        return K

    @njit(cache=True, fastmath=True)
    def _sigma2_accumulate_numba(x1, var_theta_active):
        """Numba-accelerated variance accumulation"""
        n_time, n_state, n_theta = x1.shape
        sigma2 = np.zeros((n_time, n_state)) + 1e-12
        for k in range(n_theta):
            for t in range(n_time):
                for s in range(n_state):
                    sigma2[t, s] += (x1[t, s, k]**2) * var_theta_active[k]
        return sigma2

# =============================================================================
# CONFIG: DEBUG or FULL
# =============================================================================

DEBUG = False  # True: 高速テスト / False: 論文準拠 + 全データ保存

ENABLE_PLOTS = True  # 可視化ON/OFF

def get_config(debug: bool) -> Dict[str, Any]:
    if debug:
        return {
            "M1": dict(dt=1e-4, maxtimestep=80, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-4, maxtimestep=100, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=60, c_const=25.0, alpha_const=0.0),
            "N0_M1": 20, "N0_M2": 20, "N0_M3": 20,
            "stages_M1": 4, "stages_M2": 4, "stages_M3": 4,
            "target_ess_ratio": 0.8,
            "theta_active_indices_M1": [0, 1, 2, 3, 4],
            "theta_active_indices_M2": [5, 6, 7, 8, 9],
            "theta_active_indices_M3": [10, 11, 12, 13],
            "cov_rel": 0.005,
        }
    else:
        return {
            "M1": dict(dt=1e-5, maxtimestep=2500, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-5, maxtimestep=5000, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=750, c_const=25.0, alpha_const=0.0),
            "N0_M1": 100, "N0_M2": 100, "N0_M3": 100,
            "stages_M1": 8, "stages_M2": 8, "stages_M3": 8,
            "target_ess_ratio": 0.8,
            "theta_active_indices_M1": None,
            "theta_active_indices_M2": None,
            "theta_active_indices_M3": None,
            "cov_rel": 0.005,
        }

CONFIG = get_config(DEBUG)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TMCMCResult:
    samples: List[np.ndarray]
    log_weights: List[np.ndarray]
    beta_schedule: List[float]
    logL_trace: List[np.ndarray] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    ess_trace: List[float] = field(default_factory=list)

@dataclass
class TSMResult:
    t_array: np.ndarray
    mu: np.ndarray
    sigma2: np.ndarray
    x0: np.ndarray
    x1: np.ndarray

@dataclass
class HierarchicalResults:
    M1_samples: np.ndarray
    M2_samples: np.ndarray
    M3_samples: np.ndarray
    theta_M1_mean: np.ndarray
    theta_M2_mean: np.ndarray
    theta_M3_mean: np.ndarray
    theta_final: np.ndarray
    tmcmc_M1: TMCMCResult
    tmcmc_M2: TMCMCResult
    tmcmc_M3: TMCMCResult
    tsm_M1_final: Optional[TSMResult] = None
    tsm_M2_final: Optional[TSMResult] = None
    tsm_M3_final: Optional[TSMResult] = None

# =============================================================================
# 1. Biofilm Newton Solver (with optional Numba)
# =============================================================================

class BiofilmNewtonSolver:
    """4-species continuum biofilm model Newton solver"""
    
    THETA_NAMES = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]

    def __init__(self, dt=1e-5, maxtimestep=2500, eps=1e-6, Kp1=1e-4,
                 eta_vec=None, c_const=100.0, alpha_const=100.0, use_numba=True):
        self.dt = dt
        self.maxtimestep = maxtimestep
        self.eps = eps
        self.Kp1 = Kp1
        self.Eta_vec = np.ones(4) if eta_vec is None else np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = self.Eta_vec.copy()
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)
        self.use_numba = use_numba and HAS_NUMBA

    def c(self, t): return self.c_const
    def alpha(self, t): return self.alpha_const

    @staticmethod
    def theta_to_matrices(theta):
        theta = np.asarray(theta, dtype=float)
        a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24 = theta
        A = np.array([
            [a11, a12, a13, a14],
            [a12, a22, a23, a24],
            [a13, a23, a33, a34],
            [a14, a24, a34, a44]
        ], dtype=float)
        b_diag = np.array([b1, b2, b3, b4], dtype=float)
        return A, b_diag

    def compute_Q_vector(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba:
            return _compute_Q_vector_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[4], g_old[5:9],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag)
        else:
            return self._compute_Q_vector_numpy(g_new, g_old, t, dt, A, b_diag)

    def _compute_Q_vector_numpy(self, g_new, g_old, t, dt, A, b_diag):
        phi_new, phi0_new, psi_new, gamma_new = g_new[0:4], g_new[4], g_new[5:9], g_new[9]
        phi_old, phi0_old, psi_old = g_old[0:4], g_old[4], g_old[5:9]
        
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt
        
        Q = np.zeros(10)
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        c_val = self.c(t)
        
        term1_phi = (self.Kp1 * (2.0 - 4.0 * phi_new)) / (np.power(phi_new - 1.0, 3) * np.power(phi_new, 3))
        term2_phi = (1.0 / self.Eta_vec) * (gamma_new + (self.Eta_phi_vec + self.Eta_vec * psi_new**2) * phidot +
                                             self.Eta_vec * phi_new * psi_new * psidot)
        term3_phi = (c_val / self.Eta_vec) * psi_new * Interaction
        Q[0:4] = term1_phi + term2_phi - term3_phi
        
        Q[4] = gamma_new + (self.Kp1 * (2.0 - 4.0 * phi0_new)) / (np.power(phi0_new - 1.0, 3) * np.power(phi0_new, 3)) + phi0dot
        
        term1_psi = (-2.0 * self.Kp1) / (np.power(psi_new - 1.0, 2) * np.power(psi_new, 3)) - \
                    (2.0 * self.Kp1) / (np.power(psi_new - 1.0, 3) * np.power(psi_new, 2))
        term2_psi = (b_diag * self.alpha(t) / self.Eta_vec) * psi_new
        term3_psi = phi_new * psi_new * phidot + phi_new**2 * psidot
        term4_psi = (c_val / self.Eta_vec) * phi_new * Interaction
        Q[5:9] = term1_psi + term2_psi + term3_psi - term4_psi
        
        Q[9] = np.sum(phi_new) + phi0_new - 1.0
        return Q

    def compute_Jacobian_matrix(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba:
            return _compute_jacobian_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[5:9], dt, self.Kp1,
                self.Eta_vec, self.Eta_phi_vec, self.c(t), self.alpha(t), A, b_diag)
        else:
            return self._compute_Jacobian_numpy(g_new, g_old, t, dt, A, b_diag)

    def _compute_Jacobian_numpy(self, g_new, g_old, t, dt, A, b_diag):
        v = g_new
        phi_new, phi0_new, psi_new = g_new[0:4], g_new[4], g_new[5:9]
        phidot = (phi_new - g_old[0:4]) / dt
        psidot = (psi_new - g_old[5:9]) / dt
        c_val = self.c(t)
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi

        K = np.zeros((10, 10))
        
        phi_p_deriv = (self.Kp1*(-4. + 8.*v[0:4]))/(np.power(v[0:4],3)*np.power(v[0:4]-1.,3)) - \
                      (self.Kp1*(2. - 4.*v[0:4]))*(3./(np.power(v[0:4],4)*np.power(v[0:4]-1.,3)) +
                                                   3./(np.power(v[0:4],3)*np.power(v[0:4]-1.,4)))
        phi0_p_deriv = (self.Kp1*(-4. + 8.*v[4]))/(np.power(v[4],3)*np.power(v[4]-1.,3)) - \
                       (self.Kp1*(2. - 4.*v[4]))*(3./(np.power(v[4],4)*np.power(v[4]-1.,3)) +
                                                  3./(np.power(v[4],3)*np.power(v[4]-1.,4)))
        psi_p_deriv = (4.0 * self.Kp1 * (3.0 - 5.0*v[5:9] + 5.0*v[5:9]**2)) / \
                      (np.power(v[5:9], 4) * np.power(v[5:9] - 1.0, 4))

        for i in range(4):
            for j in range(4):
                K[i, j] = (c_val / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] = phi_p_deriv[i] + (1.0 / self.Eta_vec[i]) * (
                (self.Eta_phi_vec[i] + self.Eta_vec[i] * psi_new[i]**2) / dt +
                self.Eta_vec[i] * psi_new[i] * psidot[i]) - \
                (c_val / self.Eta_vec[i]) * (psi_new[i] * (Interaction[i] + A[i, i] * psi_new[i]))
            K[i, 4] = 0.0
            for j in range(4):
                K[i, j+5] = (c_val / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
            K[i, i+5] = (1.0 / self.Eta_vec[i]) * (
                2.0 * self.Eta_vec[i] * psi_new[i] * phidot[i] +
                self.Eta_vec[i] * phi_new[i] * psidot[i] +
                self.Eta_vec[i] * phi_new[i] * psi_new[i] / dt) - \
                (c_val / self.Eta_vec[i]) * ((Interaction[i] + A[i, i] * phi_new[i] * psi_new[i]) +
                                              psi_new[i] * (A[i, i] * phi_new[i]))
            K[i, 9] = 1.0 / self.Eta_vec[i]

        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 9] = 1.0

        for i in range(4):
            k = i + 5
            for j in range(4):
                K[k, j] = -(c_val / self.Eta_vec[i]) * (A[i, j] * psi_new[j] * phi_new[i] +
                           Interaction[i] * (1.0 if i == j else 0.0))
            K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt +
                       2.0 * phi_new[i] * psidot[i]) - \
                      (c_val / self.Eta_vec[i]) * (A[i, i] * psi_new[i] * phi_new[i] +
                                                    Interaction[i] + phi_new[i] * A[i, i] * psi_new[i])
            K[k, 4] = 0.0
            for j in range(4):
                K[k, j+5] = -(c_val / self.Eta_vec[i]) * phi_new[i] * A[i, j] * phi_new[j]
            K[k, i+5] = psi_p_deriv[i] + (b_diag[i] * self.alpha(t) / self.Eta_vec[i]) + \
                        (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                        (c_val / self.Eta_vec[i]) * phi_new[i] * A[i, i] * phi_new[i]
            K[k, 9] = 0.0

        K[9, 0:5] = 1.0
        return K

    def run_deterministic(self, theta):
        A, b_diag = self.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.dt, self.maxtimestep, self.eps

        g_prev = np.array([0.02, 0.02, 0.02, 0.02, 0.92, 0.999, 0.999, 0.999, 0.999, 1e-6])
        t_list, g_list = [0.0], [g_prev.copy()]

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            g_new = g_prev.copy()
            for _ in range(100):
                Q = self.compute_Q_vector(g_new, g_prev, tt, dt, A, b_diag)
                K = self.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
                if np.isnan(Q).any() or np.isnan(K).any():
                    raise RuntimeError(f"NaN at t={tt}")
                dg = np.linalg.solve(K, -Q)
                g_new = g_new + dg
                if np.max(np.abs(Q)) < eps:
                    break
            g_prev = g_new.copy()
            t_list.append(tt)
            g_list.append(g_new.copy())

        return np.array(t_list), np.vstack(g_list)


# =============================================================================
# 2. TSM (1st-order Taylor)
# =============================================================================

class BiofilmTSM:
    THETA_NAMES = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]

    def __init__(self, solver: BiofilmNewtonSolver, cov_rel=0.005, active_theta_indices=None):
        self.solver = solver
        self.cov_rel = cov_rel
        self.active_idx = np.arange(14) if active_theta_indices is None else np.array(active_theta_indices)

    def _dG_dtheta_numeric(self, g_new, g_old, t, dt, theta):
        dG_dict = {}
        for idx in self.active_idx:
            th_plus, th_minus = theta.copy(), theta.copy()
            eps_theta = 1e-6 * max(1.0, abs(theta[idx]))
            th_plus[idx] += eps_theta
            th_minus[idx] -= eps_theta
            A_p, b_p = self.solver.theta_to_matrices(th_plus)
            A_m, b_m = self.solver.theta_to_matrices(th_minus)
            Q_p = self.solver.compute_Q_vector(g_new, g_old, t, dt, A_p, b_p)
            Q_m = self.solver.compute_Q_vector(g_new, g_old, t, dt, A_m, b_m)
            dG_dict[self.THETA_NAMES[idx]] = (Q_p - Q_m) / (2.0 * eps_theta)
        return dG_dict

    def solve_tsm(self, theta) -> TSMResult:
        theta = np.asarray(theta, dtype=float)
        A, b_diag = self.solver.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.solver.dt, self.solver.maxtimestep, self.solver.eps

        g_prev = np.array([0.02, 0.02, 0.02, 0.02, 0.92, 0.999, 0.999, 0.999, 0.999, 1e-6])
        t_list, x0_list = [0.0], [g_prev.copy()]
        theta_dim = len(self.active_idx)
        x1_list = [np.zeros((10, theta_dim))]

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            g_new = g_prev.copy()
            for _ in range(100):
                Q = self.solver.compute_Q_vector(g_new, g_prev, tt, dt, A, b_diag)
                K = self.solver.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
                if np.isnan(Q).any() or np.isnan(K).any():
                    raise RuntimeError(f"NaN at t={tt}")
                dg = np.linalg.solve(K, -Q)
                g_new = g_new + dg
                if np.max(np.abs(Q)) < eps:
                    break

            dG_dict = self._dG_dtheta_numeric(g_new, g_prev, tt, dt, theta)
            J = self.solver.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
            x1_t = np.zeros((10, theta_dim))
            for k, idx in enumerate(self.active_idx):
                x1_t[:, k] = np.linalg.solve(J, -dG_dict[self.THETA_NAMES[idx]])

            g_prev = g_new.copy()
            t_list.append(tt)
            x0_list.append(g_prev.copy())
            x1_list.append(x1_t)

        t_array = np.array(t_list)
        x0 = np.vstack(x0_list)
        x1 = np.stack(x1_list, axis=0)

        var_theta_full = (self.cov_rel * theta)**2
        var_theta_active = var_theta_full[self.active_idx]

        mu = x0.copy()
        if HAS_NUMBA:
            sigma2 = _sigma2_accumulate_numba(x1, var_theta_active)
        else:
            sigma2 = np.zeros_like(mu) + 1e-12
            for k in range(theta_dim):
                sigma2 += (x1[:, :, k]**2) * var_theta_active[k]

        return TSMResult(t_array=t_array, mu=mu, sigma2=sigma2, x0=x0, x1=x1)


# =============================================================================
# 3. Likelihood (Eq. 29)
# =============================================================================

def log_likelihood_eq29(mu, sigma2, data):
    D = np.asarray(data, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    v = np.asarray(sigma2, dtype=float).ravel() + 1e-12
    if D.shape != m.shape:
        raise ValueError(f"shape mismatch: data={D.shape}, mu={m.shape}")
    bad = (~np.isfinite(v)) | (v <= 0)
    if np.any(bad):
        return -1e20
    diff = D - m
    ll = -0.5 * np.sum(np.log(2 * np.pi * v)) - 0.5 * np.sum(diff*diff / v)
    return max(ll, -1e20) if np.isfinite(ll) else -1e20


# =============================================================================
# 4. TMCMC
# =============================================================================

def tmcmc(log_likelihood, log_prior, theta_init_samples, n_stages=8,
          target_ess_ratio=0.8, adapt_cov=True, random_state=None) -> TMCMCResult:
    rng = np.random.default_rng(random_state)
    theta_curr = np.array(theta_init_samples, dtype=float)
    N, d = theta_curr.shape

    beta_list, samples_list, logw_list = [0.0], [theta_curr.copy()], [np.zeros(N)]
    logL_trace, acceptance_rates, ess_trace = [], [], []

    logp_prior = np.array([log_prior(th) for th in theta_curr])
    logL = np.array([log_likelihood(th) for th in theta_curr])
    logp_prior[~np.isfinite(logp_prior)] = -1e20
    logL[~np.isfinite(logL)] = -1e20
    logL_trace.append(logL.copy())

    beta = 0.0

    for stage in range(1, n_stages + 1):
        def ess_for_delta(delta_beta):
            x = delta_beta * (logL - np.max(logL))
            w_unnorm = np.exp(x)
            s = np.sum(w_unnorm)
            if s <= 0 or not np.isfinite(s):
                return 0.0
            w = w_unnorm / s
            return 1.0 / np.sum(w**2) if np.isfinite(w).all() else 0.0

        delta_low, delta_high = 0.0, 1.0 - beta
        for _ in range(25):
            mid = 0.5 * (delta_low + delta_high)
            if ess_for_delta(mid) >= target_ess_ratio * N:
                delta_low = mid
            else:
                delta_high = mid

        delta_beta = delta_low
        beta_next = min(beta + delta_beta, 1.0)

        x = delta_beta * (logL - np.max(logL))
        w_unnorm = np.exp(x)
        s = np.sum(w_unnorm)
        w = w_unnorm / s if (s > 0 and np.isfinite(s)) else np.ones(N) / N
        if not np.isfinite(w).all():
            w = np.ones(N) / N

        ess = 1.0 / np.sum(w**2)
        ess_trace.append(ess)
        print(f"  [TMCMC] Stage {stage}: β={beta_next:.4f}, ESS={ess:.1f}/{N}")
        beta_list.append(beta_next)

        idx = rng.choice(N, size=N, p=w)
        theta_resampled = theta_curr[idx]
        cov = np.cov(theta_resampled.T) + 1e-6 * np.eye(d) if (adapt_cov and stage > 1) else 0.01 * np.eye(d)

        theta_new = theta_resampled.copy()
        n_accepted = 0

        for n in range(N):
            th_old = theta_resampled[n]
            lp_old, ll_old = log_prior(th_old), log_likelihood(th_old)
            if not np.isfinite(lp_old) or not np.isfinite(ll_old):
                continue
            logpost_old = lp_old + beta_next * ll_old

            prop = rng.multivariate_normal(th_old, cov)
            lp_prop, ll_prop = log_prior(prop), log_likelihood(prop)
            if not np.isfinite(lp_prop) or not np.isfinite(ll_prop):
                continue
            logpost_prop = lp_prop + beta_next * ll_prop

            if rng.uniform() < np.exp(logpost_prop - logpost_old):
                theta_new[n] = prop
                n_accepted += 1

        acceptance_rates.append(n_accepted / N)
        theta_curr = theta_new.copy()
        logp_prior = np.array([log_prior(th) for th in theta_curr])
        logL = np.array([log_likelihood(th) for th in theta_curr])
        logp_prior[~np.isfinite(logp_prior)] = -1e20
        logL[~np.isfinite(logL)] = -1e20
        logL_trace.append(logL.copy())

        beta = beta_next
        samples_list.append(theta_curr.copy())
        logw_list.append(np.log(w + 1e-300))

        if beta >= 1.0:
            break

    return TMCMCResult(samples_list, logw_list, beta_list, logL_trace, acceptance_rates, ess_trace)


# =============================================================================
# 5. Hierarchical Case II (M1 -> M2 -> M3)
# =============================================================================

def hierarchical_case2(tsm_M1, tsm_M2, tsm_M3, data_M1, data_M2, data_M3,
                       theta_prior_center, bounds, config) -> HierarchicalResults:
    theta_prior_center = np.asarray(theta_prior_center, dtype=float)

    def log_prior_full(theta):
        theta = np.asarray(theta, dtype=float)
        low = np.array([b[0] for b in bounds])
        high = np.array([b[1] for b in bounds])
        return 0.0 if np.all((theta >= low) & (theta <= high)) else -np.inf

    # === M1 ===
    print("\n=== Stage 1: M1 (species 1 & 2) ===")
    tsm_M1_result = None
    def logL_M1(theta_M1):
        nonlocal tsm_M1_result
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        tsm_res = tsm_M1.solve_tsm(theta_full)
        tsm_M1_result = tsm_res
        phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
        obs = np.stack([phi[:, 0]*psi[:, 0], phi[:, 1]*psi[:, 1]], axis=1)
        var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
        obs_var = np.stack([
            phi[:, 0]**2 * var_psi[:, 0] + psi[:, 0]**2 * var_phi[:, 0],
            phi[:, 1]**2 * var_psi[:, 1] + psi[:, 1]**2 * var_phi[:, 1],
        ], axis=1)
        return log_likelihood_eq29(obs, obs_var, data_M1)

    def log_prior_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        return log_prior_full(theta_full)

    rng = np.random.default_rng(1234)
    init_M1 = rng.uniform([b[0] for b in bounds[0:5]], [b[1] for b in bounds[0:5]], size=(config["N0_M1"], 5))
    res_M1 = tmcmc(logL_M1, log_prior_M1, init_M1, n_stages=config["stages_M1"],
                   target_ess_ratio=config["target_ess_ratio"], random_state=1234)
    samples_M1 = res_M1.samples[-1]
    theta_M1_mean = np.mean(samples_M1, axis=0)
    print(f"  M1 posterior mean: {theta_M1_mean}")

    theta_stage2_center = theta_prior_center.copy()
    theta_stage2_center[0:5] = theta_M1_mean

    # === M2 ===
    print("\n=== Stage 2: M2 (species 3 & 4) ===")
    tsm_M2_result = None
    def logL_M2(theta_M2):
        nonlocal tsm_M2_result
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        tsm_res = tsm_M2.solve_tsm(theta_full)
        tsm_M2_result = tsm_res
        phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
        obs = np.stack([phi[:, 2]*psi[:, 2], phi[:, 3]*psi[:, 3]], axis=1)
        var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
        obs_var = np.stack([
            phi[:, 2]**2 * var_psi[:, 2] + psi[:, 2]**2 * var_phi[:, 2],
            phi[:, 3]**2 * var_psi[:, 3] + psi[:, 3]**2 * var_phi[:, 3],
        ], axis=1)
        return log_likelihood_eq29(obs, obs_var, data_M2)

    def log_prior_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        return log_prior_full(theta_full)

    init_M2 = rng.uniform([b[0] for b in bounds[5:10]], [b[1] for b in bounds[5:10]], size=(config["N0_M2"], 5))
    res_M2 = tmcmc(logL_M2, log_prior_M2, init_M2, n_stages=config["stages_M2"],
                   target_ess_ratio=config["target_ess_ratio"], random_state=5678)
    samples_M2 = res_M2.samples[-1]
    theta_M2_mean = np.mean(samples_M2, axis=0)
    print(f"  M2 posterior mean: {theta_M2_mean}")

    theta_stage3_center = theta_stage2_center.copy()
    theta_stage3_center[5:10] = theta_M2_mean

    # === M3 ===
    print("\n=== Stage 3: M3 (cross interactions) ===")
    tsm_M3_result = None
    def logL_M3(theta_M3):
        nonlocal tsm_M3_result
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        tsm_res = tsm_M3.solve_tsm(theta_full)
        tsm_M3_result = tsm_res
        phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
        obs = np.stack([phi[:, i]*psi[:, i] for i in range(4)], axis=1)
        var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
        obs_var = np.stack([phi[:, i]**2 * var_psi[:, i] + psi[:, i]**2 * var_phi[:, i] for i in range(4)], axis=1)
        return log_likelihood_eq29(obs, obs_var, data_M3)

    def log_prior_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        return log_prior_full(theta_full)

    init_M3 = rng.uniform([b[0] for b in bounds[10:14]], [b[1] for b in bounds[10:14]], size=(config["N0_M3"], 4))
    res_M3 = tmcmc(logL_M3, log_prior_M3, init_M3, n_stages=config["stages_M3"],
                   target_ess_ratio=config["target_ess_ratio"], random_state=9012)
    samples_M3 = res_M3.samples[-1]
    theta_M3_mean = np.mean(samples_M3, axis=0)
    print(f"  M3 posterior mean: {theta_M3_mean}")

    theta_final = theta_stage3_center.copy()
    theta_final[10:14] = theta_M3_mean

    return HierarchicalResults(
        M1_samples=samples_M1, M2_samples=samples_M2, M3_samples=samples_M3,
        theta_M1_mean=theta_M1_mean, theta_M2_mean=theta_M2_mean, theta_M3_mean=theta_M3_mean,
        theta_final=theta_final, tmcmc_M1=res_M1, tmcmc_M2=res_M2, tmcmc_M3=res_M3,
        tsm_M1_final=tsm_M1_result, tsm_M2_final=tsm_M2_result, tsm_M3_final=tsm_M3_result
    )


# =============================================================================
# 6. DATA SAVING MODULE
# =============================================================================

class DataSaver:
    """Complete data saving with timestamped folders"""

    def __init__(self, base_dir: str = "results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, f"result_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n📁 Output directory: {self.output_dir}/")

    def save_all(self, results: HierarchicalResults, theta_true: np.ndarray,
                 data_M1, data_M2, data_M3, t1, t2, t3, config: dict):
        """Save all data in one call"""
        import pandas as pd
        
        param_names = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]
        
        # 1. Config
        config_data = {"timestamp": self.timestamp, "config": {k: str(v) for k, v in config.items()}}
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"  ✓ config.json")
        
        # 2. True parameters
        pd.DataFrame({"parameter": param_names, "true_value": theta_true}).to_csv(
            os.path.join(self.output_dir, "true_parameters.csv"), index=False)
        print(f"  ✓ true_parameters.csv")
        
        # 3. Synthetic data
        pd.DataFrame({"time": t1, "sp1": data_M1[:, 0], "sp2": data_M1[:, 1]}).to_csv(
            os.path.join(self.output_dir, "synthetic_data_M1.csv"), index=False)
        pd.DataFrame({"time": t2, "sp3": data_M2[:, 0], "sp4": data_M2[:, 1]}).to_csv(
            os.path.join(self.output_dir, "synthetic_data_M2.csv"), index=False)
        pd.DataFrame({"time": t3, **{f"sp{i+1}": data_M3[:, i] for i in range(4)}}).to_csv(
            os.path.join(self.output_dir, "synthetic_data_M3.csv"), index=False)
        print(f"  ✓ synthetic_data_M1/M2/M3.csv")
        
        # 4. Posterior samples (NPZ + CSV)
        np.savez_compressed(os.path.join(self.output_dir, "posterior_samples.npz"),
            M1=results.M1_samples, M2=results.M2_samples, M3=results.M3_samples)
        pd.DataFrame(results.M1_samples, columns=["a11","a12","a22","b1","b2"]).to_csv(
            os.path.join(self.output_dir, "posterior_samples_M1.csv"), index=False)
        pd.DataFrame(results.M2_samples, columns=["a33","a34","a44","b3","b4"]).to_csv(
            os.path.join(self.output_dir, "posterior_samples_M2.csv"), index=False)
        pd.DataFrame(results.M3_samples, columns=["a13","a14","a23","a24"]).to_csv(
            os.path.join(self.output_dir, "posterior_samples_M3.csv"), index=False)
        print(f"  ✓ posterior_samples.npz, posterior_samples_M1/M2/M3.csv")
        
        # 5. Posterior summary
        all_samples = np.hstack([results.M1_samples, results.M2_samples, results.M3_samples])
        summary = pd.DataFrame({
            "parameter": param_names,
            "true_value": theta_true,
            "posterior_mean": results.theta_final,
            "posterior_std": np.std(all_samples, axis=0),
            "CI_2.5%": np.percentile(all_samples, 2.5, axis=0),
            "CI_97.5%": np.percentile(all_samples, 97.5, axis=0),
            "error": results.theta_final - theta_true,
            "error_percent": 100 * (results.theta_final - theta_true) / (np.abs(theta_true) + 1e-10)
        })
        summary.to_csv(os.path.join(self.output_dir, "posterior_summary.csv"), index=False)
        print(f"  ✓ posterior_summary.csv")
        
        # 6. TMCMC diagnostics
        for name, tmcmc_res in [("M1", results.tmcmc_M1), ("M2", results.tmcmc_M2), ("M3", results.tmcmc_M3)]:
            pd.DataFrame({"stage": range(len(tmcmc_res.beta_schedule)), "beta": tmcmc_res.beta_schedule}).to_csv(
                os.path.join(self.output_dir, f"tmcmc_beta_{name}.csv"), index=False)
            if tmcmc_res.acceptance_rates:
                pd.DataFrame({"stage": range(1, len(tmcmc_res.acceptance_rates)+1),
                             "acceptance_rate": tmcmc_res.acceptance_rates,
                             "ess": tmcmc_res.ess_trace[:len(tmcmc_res.acceptance_rates)]}).to_csv(
                    os.path.join(self.output_dir, f"tmcmc_diagnostics_{name}.csv"), index=False)
            if tmcmc_res.logL_trace:
                np.save(os.path.join(self.output_dir, f"tmcmc_logL_{name}.npy"), np.array(tmcmc_res.logL_trace))
        print(f"  ✓ tmcmc_beta/diagnostics/logL_M1/M2/M3")
        
        # 7. TSM results
        for name, tsm_res in [("M1", results.tsm_M1_final), ("M2", results.tsm_M2_final), ("M3", results.tsm_M3_final)]:
            if tsm_res is not None:
                np.savez_compressed(os.path.join(self.output_dir, f"tsm_{name}.npz"),
                    t=tsm_res.t_array, mu=tsm_res.mu, sigma2=tsm_res.sigma2, x0=tsm_res.x0, x1=tsm_res.x1)
        print(f"  ✓ tsm_M1/M2/M3.npz")
        
        # 8. Complete archive
        np.savez_compressed(os.path.join(self.output_dir, "complete_archive.npz"),
            theta_true=theta_true, theta_final=results.theta_final,
            M1_samples=results.M1_samples, M2_samples=results.M2_samples, M3_samples=results.M3_samples,
            data_M1=data_M1, data_M2=data_M2, data_M3=data_M3, t1=t1, t2=t2, t3=t3,
            beta_M1=np.array(results.tmcmc_M1.beta_schedule),
            beta_M2=np.array(results.tmcmc_M2.beta_schedule),
            beta_M3=np.array(results.tmcmc_M3.beta_schedule))
        print(f"  ✓ complete_archive.npz")
        
        return summary

    def list_files(self):
        print("\n" + "="*60)
        print("  SAVED FILES")
        print("="*60)
        total = 0
        for f in sorted(os.listdir(self.output_dir)):
            size = os.path.getsize(os.path.join(self.output_dir, f))
            total += size
            size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
            print(f"  • {f:<45} ({size_str})")
        print(f"\n  Total: {total/1024:.1f} KB")
        print("="*60)


# =============================================================================
# 7. PAPER-LEVEL VISUALIZATION MODULE
# =============================================================================

def create_paper_figures(output_dir: str, results: HierarchicalResults,
                         theta_true: np.ndarray, data_M1, data_M2, data_M3,
                         t1, t2, t3, tsm_M1=None, tsm_M2=None, tsm_M3=None):
    """
    Create publication-quality figures matching Fritsch et al. (2025) paper style
    
    Generates:
    - Fig 1: Time series μ(t) ± σ(t) for all species
    - Fig 2-4: Corner plots (M1, M2, M3)
    - Fig 5: Parameter comparison (true vs estimated with CI)
    - Fig 6: TMCMC β progression
    - Fig 7: Convergence diagnostics (ESS, acceptance rate)
    - Fig 8: Residual analysis
    - Fig 9: Posterior predictive check
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
        from scipy.stats import pearsonr, gaussian_kde
    except ImportError:
        print("  ⚠ matplotlib/scipy not available, skipping figures")
        return

    # Style settings
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })
    
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    param_names_M1 = [r'$a_{11}$', r'$a_{12}$', r'$a_{22}$', r'$b_1$', r'$b_2$']
    param_names_M2 = [r'$a_{33}$', r'$a_{34}$', r'$a_{44}$', r'$b_3$', r'$b_4$']
    param_names_M3 = [r'$a_{13}$', r'$a_{14}$', r'$a_{23}$', r'$a_{24}$']
    all_param_names = param_names_M1 + param_names_M2 + param_names_M3

    # =========================================================================
    # Figure 1: Time Series μ(t) ± σ(t)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 1: Living Biomass $\\bar{\\Phi}_i(t) = \\phi_i \\cdot \\psi_i$ with TSM Uncertainty', 
                 fontsize=14, fontweight='bold')
    
    species_labels = ['Species 1', 'Species 2', 'Species 3', 'Species 4']
    data_list = [data_M1[:, 0], data_M1[:, 1], data_M2[:, 0], data_M2[:, 1]]
    t_list = [t1, t1, t2, t2]
    
    for idx, (ax, label, data, t, color) in enumerate(zip(
            axes.flatten(), species_labels, data_list, t_list, COLORS)):
        ax.scatter(t, data, s=20, alpha=0.6, color=color, label='Data', zorder=3)
        
        # If TSM results available, plot μ ± 2σ
        tsm = results.tsm_M1_final if idx < 2 else results.tsm_M2_final
        if tsm is not None:
            sp_idx = idx if idx < 2 else idx - 2
            phi = tsm.mu[:, sp_idx if idx < 2 else sp_idx + 2]
            psi = tsm.mu[:, 5 + (sp_idx if idx < 2 else sp_idx + 2)]
            mu = phi * psi
            
            var_phi = tsm.sigma2[:, sp_idx if idx < 2 else sp_idx + 2]
            var_psi = tsm.sigma2[:, 5 + (sp_idx if idx < 2 else sp_idx + 2)]
            sigma = np.sqrt(phi**2 * var_psi + psi**2 * var_phi)
            
            ax.plot(tsm.t_array, mu, '-', color=color, linewidth=2, label='$\\mu(t)$', zorder=2)
            ax.fill_between(tsm.t_array, mu - 2*sigma, mu + 2*sigma, 
                           alpha=0.25, color=color, label='$\\mu \\pm 2\\sigma$', zorder=1)
        
        ax.set_xlabel('Normalized Time $t$')
        ax.set_ylabel('$\\bar{\\Phi}(t)$')
        ax.set_title(label, fontweight='bold')
        ax.legend(loc='best')
        ax.set_xlim(0, max(t))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig01_time_series.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig01_time_series.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig01_time_series.png/pdf")

    # =========================================================================
    # Figure 2-4: Corner Plots
    # =========================================================================
    def corner_plot(samples, true_vals, param_names, title, filename):
        n = len(param_names)
        fig, axes = plt.subplots(n, n, figsize=(12, 12))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    # Diagonal: histogram with KDE
                    ax.hist(samples[:, i], bins=35, density=True, alpha=0.7, 
                           color='#4472C4', edgecolor='white', linewidth=0.5)
                    try:
                        kde = gaussian_kde(samples[:, i])
                        x_kde = np.linspace(samples[:, i].min(), samples[:, i].max(), 100)
                        ax.plot(x_kde, kde(x_kde), 'k-', linewidth=1.5)
                    except:
                        pass
                    ax.axvline(true_vals[i], color='#C00000', linestyle='--', linewidth=2.5, label='True')
                    ax.axvline(np.mean(samples[:, i]), color='#2E7D32', linestyle='-', linewidth=2)
                    # CI
                    ci_low, ci_high = np.percentile(samples[:, i], [2.5, 97.5])
                    ax.axvspan(ci_low, ci_high, alpha=0.2, color='#FF8C00')
                    if i == 0:
                        ax.legend(fontsize=7, loc='upper right')
                elif i > j:
                    # Lower triangle: scatter
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.1, s=1, c='#4472C4')
                    ax.axvline(true_vals[j], color='#C00000', linestyle='--', alpha=0.5, linewidth=1)
                    ax.axhline(true_vals[i], color='#C00000', linestyle='--', alpha=0.5, linewidth=1)
                    ax.scatter([np.mean(samples[:, j])], [np.mean(samples[:, i])], 
                              color='#2E7D32', s=50, marker='x', linewidths=2, zorder=5)
                else:
                    # Upper triangle: correlation
                    corr, _ = pearsonr(samples[:, i], samples[:, j])
                    bg = '#FFB6B6' if corr > 0.3 else '#B6C4FF' if corr < -0.3 else '#F0F0F0'
                    ax.set_facecolor(bg)
                    ax.text(0.5, 0.5, f'$\\rho$={corr:.2f}', ha='center', va='center',
                           fontsize=11, fontweight='bold', transform=ax.transAxes)
                
                if i == n-1:
                    ax.set_xlabel(param_names[j], fontsize=11)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(param_names[i], fontsize=11)
                elif j > 0:
                    ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
        plt.close()
    
    corner_plot(results.M1_samples, theta_true[0:5], param_names_M1, 
                'Figure 2: M1 Posterior Distribution', 'fig02_M1_corner')
    print("  ✓ fig02_M1_corner.png/pdf")
    
    corner_plot(results.M2_samples, theta_true[5:10], param_names_M2, 
                'Figure 3: M2 Posterior Distribution', 'fig03_M2_corner')
    print("  ✓ fig03_M2_corner.png/pdf")
    
    corner_plot(results.M3_samples, theta_true[10:14], param_names_M3, 
                'Figure 4: M3 Posterior Distribution', 'fig04_M3_corner')
    print("  ✓ fig04_M3_corner.png/pdf")

    # =========================================================================
    # Figure 5: Parameter Comparison (True vs Estimated with CI)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle('Figure 5: Parameter Estimation Results with 95% Credible Intervals', 
                 fontsize=14, fontweight='bold')
    
    x = np.arange(14)
    width = 0.35
    
    all_samples = np.hstack([results.M1_samples, results.M2_samples, results.M3_samples])
    ci_low = np.percentile(all_samples, 2.5, axis=0)
    ci_high = np.percentile(all_samples, 97.5, axis=0)
    errors = np.array([results.theta_final - ci_low, ci_high - results.theta_final])
    
    bars1 = ax.bar(x - width/2, theta_true, width, label='True', color='#FF8C00', alpha=0.85)
    bars2 = ax.bar(x + width/2, results.theta_final, width, label='Estimated', 
                   color='#4472C4', alpha=0.85, yerr=errors, capsize=4,
                   error_kw={'linewidth': 1.5, 'color': 'black'})
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_param_names, fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add vertical lines to separate M1/M2/M3
    ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5)
    
    ymax = max(np.max(theta_true), np.max(ci_high)) * 1.15
    ax.text(2, ymax, 'M1', ha='center', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.text(7, ymax, 'M2', ha='center', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.text(11.5, ymax, 'M3', ha='center', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_ylim(0, ymax * 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig05_parameter_comparison.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig05_parameter_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig05_parameter_comparison.png/pdf")

    # =========================================================================
    # Figure 6: TMCMC β Progression
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 6: TMCMC Tempering Schedule $\\beta_j$', fontsize=14, fontweight='bold')
    
    for ax, (name, tmcmc_res, color) in zip(axes, 
            [('M1', results.tmcmc_M1, COLORS[0]), 
             ('M2', results.tmcmc_M2, COLORS[1]), 
             ('M3', results.tmcmc_M3, COLORS[2])]):
        stages = np.arange(len(tmcmc_res.beta_schedule))
        ax.plot(stages, tmcmc_res.beta_schedule, 'o-', markersize=10, linewidth=2.5, color=color)
        ax.fill_between(stages, 0, tmcmc_res.beta_schedule, alpha=0.2, color=color)
        ax.set_xlabel('Stage $j$', fontsize=12)
        ax.set_ylabel('$\\beta_j$', fontsize=12)
        ax.set_title(f'{name}', fontweight='bold', fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.2, len(stages) - 0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Annotate final β
        final_beta = tmcmc_res.beta_schedule[-1]
        ax.annotate(f'$\\beta_{{final}}$={final_beta:.3f}', 
                   xy=(len(stages)-1, final_beta), xytext=(len(stages)-2, final_beta-0.15),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig06_beta_progression.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig06_beta_progression.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig06_beta_progression.png/pdf")

    # =========================================================================
    # Figure 7: Convergence Diagnostics (ESS, Acceptance Rate)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Figure 7: TMCMC Convergence Diagnostics', fontsize=14, fontweight='bold')
    
    for col, (name, tmcmc_res, color) in enumerate(
            [('M1', results.tmcmc_M1, COLORS[0]), 
             ('M2', results.tmcmc_M2, COLORS[1]), 
             ('M3', results.tmcmc_M3, COLORS[2])]):
        
        # Top row: ESS
        ax1 = axes[0, col]
        if tmcmc_res.ess_trace:
            stages = np.arange(1, len(tmcmc_res.ess_trace) + 1)
            ax1.bar(stages, tmcmc_res.ess_trace, color=color, alpha=0.7, edgecolor='white')
            ax1.axhline(y=len(tmcmc_res.samples[-1]) * 0.5, color='red', linestyle='--', 
                       label=f'50% of N={len(tmcmc_res.samples[-1])}')
            ax1.set_ylabel('ESS', fontsize=11)
            ax1.legend(fontsize=8)
        ax1.set_title(f'{name}', fontweight='bold')
        ax1.set_xlabel('Stage', fontsize=11)
        
        # Bottom row: Acceptance Rate
        ax2 = axes[1, col]
        if tmcmc_res.acceptance_rates:
            stages = np.arange(1, len(tmcmc_res.acceptance_rates) + 1)
            ax2.bar(stages, np.array(tmcmc_res.acceptance_rates) * 100, color=color, alpha=0.7, edgecolor='white')
            ax2.axhline(y=23.4, color='green', linestyle='--', label='Optimal (23.4%)')
            ax2.axhspan(15, 50, alpha=0.1, color='green', label='Acceptable (15-50%)')
            ax2.set_ylabel('Acceptance Rate (%)', fontsize=11)
            ax2.set_ylim(0, 100)
            ax2.legend(fontsize=8)
        ax2.set_xlabel('Stage', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig07_convergence.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig07_convergence.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig07_convergence.png/pdf")

    # =========================================================================
    # Figure 8: Residual Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Figure 8: Residual Analysis', fontsize=14, fontweight='bold')
    
    all_residuals = []
    for idx, (ax, label, data, t, color) in enumerate(zip(
            axes.flatten(), species_labels, data_list, t_list, COLORS)):
        tsm = results.tsm_M1_final if idx < 2 else results.tsm_M2_final
        if tsm is not None:
            sp_idx = idx if idx < 2 else idx - 2
            phi = tsm.mu[:, sp_idx if idx < 2 else sp_idx + 2]
            psi = tsm.mu[:, 5 + (sp_idx if idx < 2 else sp_idx + 2)]
            mu = phi * psi
            
            # Interpolate to data time points
            mu_interp = np.interp(t, tsm.t_array, mu)
            residuals = data - mu_interp
            all_residuals.extend(residuals)
            
            ax.scatter(t, residuals, s=30, alpha=0.7, color=color)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.fill_between(t, -0.01, 0.01, alpha=0.2, color='gray', label='±0.01 band')
        
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Residual')
        ax.set_title(f'{label}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig08_residuals.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig08_residuals.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig08_residuals.png/pdf")

    # =========================================================================
    # Figure 9: Summary Statistics Table
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Figure 9: Posterior Summary Statistics', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    all_samples = np.hstack([results.M1_samples, results.M2_samples, results.M3_samples])
    
    table_data = [['Parameter', 'True', 'Mean', 'Std', '95% CI', 'Error (%)']]
    param_labels = ['a11','a12','a22','b1','b2','a33','a34','a44','b3','b4','a13','a14','a23','a24']
    
    for i, name in enumerate(param_labels):
        true_val = theta_true[i]
        mean_val = results.theta_final[i]
        std_val = np.std(all_samples[:, i])
        ci_low, ci_high = np.percentile(all_samples[:, i], [2.5, 97.5])
        error_pct = 100 * (mean_val - true_val) / (abs(true_val) + 1e-10)
        
        table_data.append([
            f'${name[0]}_{{{name[1:]}}}$' if len(name) > 1 else name,
            f'{true_val:.3f}',
            f'{mean_val:.3f}',
            f'{std_val:.4f}',
            f'[{ci_low:.3f}, {ci_high:.3f}]',
            f'{error_pct:+.1f}%'
        ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.12, 0.1, 0.1, 0.1, 0.2, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for j in range(6):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors
    for i in range(1, len(table_data)):
        color = '#E6F0FF' if i % 2 == 0 else 'white'
        for j in range(6):
            table[(i, j)].set_facecolor(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig09_summary_table.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig09_summary_table.pdf'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig09_summary_table.png/pdf")

    print(f"\n  📊 All 9 paper-level figures saved to {output_dir}/")


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

def main():
    print("="*72)
    print("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    print("  Complete Version with Numba & Paper-Level Figures")
    print("="*72)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode : {'DEBUG (fast)' if DEBUG else 'FULL (Table 3)'}")
    print(f"Numba: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")
    print(f"Save : {'OFF' if DEBUG else 'ON (timestamped folder)'}")
    print(f"Plots: {'ON' if ENABLE_PLOTS else 'OFF'}")
    print("="*72)

    # --- Build solvers ---
    solver_M1 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], use_numba=HAS_NUMBA, **CONFIG["M1"])
    solver_M2 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], use_numba=HAS_NUMBA, **CONFIG["M2"])
    solver_M3 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], use_numba=HAS_NUMBA, **CONFIG["M3"])

    tsm_M1 = BiofilmTSM(solver_M1, cov_rel=CONFIG["cov_rel"], active_theta_indices=CONFIG["theta_active_indices_M1"])
    tsm_M2 = BiofilmTSM(solver_M2, cov_rel=CONFIG["cov_rel"], active_theta_indices=CONFIG["theta_active_indices_M2"])
    tsm_M3 = BiofilmTSM(solver_M3, cov_rel=CONFIG["cov_rel"], active_theta_indices=CONFIG["theta_active_indices_M3"])

    # --- True parameters (Table 3) ---
    TRUE_M1 = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
    TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])
    TRUE_M3 = np.array([2.0, 1.0, 2.0, 1.0])
    theta_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])

    print("\n[Step 0] Generating synthetic data...")
    np.random.seed(42)

    t1, g1 = solver_M1.run_deterministic(theta_true)
    data_M1 = np.stack([g1[:, 0]*g1[:, 5], g1[:, 1]*g1[:, 6]], axis=1)
    data_M1 += np.random.normal(0, 0.002, data_M1.shape)

    t2, g2 = solver_M2.run_deterministic(theta_true)
    data_M2 = np.stack([g2[:, 2]*g2[:, 7], g2[:, 3]*g2[:, 8]], axis=1)
    data_M2 += np.random.normal(0, 0.002, data_M2.shape)

    t3, g3 = solver_M3.run_deterministic(theta_true)
    data_M3 = np.stack([g3[:, i]*g3[:, 5+i] for i in range(4)], axis=1)
    data_M3 += np.random.normal(0, 0.002, data_M3.shape)

    print(f"  ✓ Data shapes: M1={data_M1.shape}, M2={data_M2.shape}, M3={data_M3.shape}")

    # --- Run hierarchical inference ---
    theta_prior_center = theta_true.copy()
    bounds = [(0.0, 3.0)] * 14

    t0 = time.time()
    results = hierarchical_case2(tsm_M1, tsm_M2, tsm_M3, data_M1, data_M2, data_M3,
                                  theta_prior_center, bounds, CONFIG)
    runtime = time.time() - t0

    # --- Print results ---
    print("\n" + "="*72)
    print("  RESULTS")
    print("="*72)
    print(f"True θ:      {theta_true}")
    print(f"Estimated θ: {results.theta_final}")
    print(f"Error:       {results.theta_final - theta_true}")
    print(f"Runtime:     {runtime:.1f} s")

    # --- Save data and create figures (only if not DEBUG) ---
    if not DEBUG:
        print("\n" + "="*72)
        print("  SAVING ALL DATA")
        print("="*72)

        saver = DataSaver(base_dir="results")
        summary = saver.save_all(results, theta_true, data_M1, data_M2, data_M3, t1, t2, t3, CONFIG)

        if ENABLE_PLOTS:
            print("\n[Creating Paper-Level Figures]")
            create_paper_figures(saver.output_dir, results, theta_true,
                                data_M1, data_M2, data_M3, t1, t2, t3)

        saver.list_files()
        print(f"\n  📁 All data saved to: {saver.output_dir}/")
    else:
        print("\n  ℹ️  DEBUG mode: data saving and figures skipped")
        print("     Set DEBUG = False to save all results and generate paper figures")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
