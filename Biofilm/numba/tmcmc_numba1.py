#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating
===============================================================
PAPER-ACCURATE VERSION (Fritsch et al. 2025)

Key corrections from paper:
1. SPARSE DATA: Ndata = 20 points (not all time steps!)
2. INITIAL CONDITIONS: ϕ = 0.2 for M1/M2, ϕ = 0.02 for M3
3. MEASUREMENT NOISE in likelihood: σ²_total = σ²_TSM + σ²_obs
4. TRUE PARAMETERS matching paper Case I structure

Author: Based on Fritsch et al. (2025)
"""
from tmcmc_improved import tmcmc

import numpy as np
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import warnings
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# NUMBA ACCELERATION
# =============================================================================
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not available")

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_Q_vector_numba(phi_new, phi0_new, psi_new, gamma_new,
                                 phi_old, phi0_old, psi_old,
                                 dt, Kp1, Eta_vec, Eta_phi_vec,
                                 c_val, alpha_val, A, b_diag):
        Q = np.zeros(10)
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        
        for i in range(4):
            term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
            term2 = (1.0 / Eta_vec[i]) * (gamma_new + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i] +
                    Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
            term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
            Q[i] = term1 + term2 - term3
        
        Q[4] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot
        
        for i in range(4):
            term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) - \
                    (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
            term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
            term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
            term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
            Q[5+i] = term1 + term2 + term3 - term4
        
        Q[9] = phi_new[0] + phi_new[1] + phi_new[2] + phi_new[3] + phi0_new - 1.0
        return Q

    @njit(cache=True, fastmath=True)
    def _compute_jacobian_numba(phi_new, phi0_new, psi_new, gamma_new,
                                 phi_old, psi_old, dt, Kp1,
                                 Eta_vec, Eta_phi_vec, c_val, alpha_val, A, b_diag):
        K = np.zeros((10, 10))
        phidot = (phi_new - phi_old) / dt
        psidot = (psi_new - psi_old) / dt
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        
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
        
        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 9] = 1.0
        
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
        
        K[9, 0] = 1.0
        K[9, 1] = 1.0
        K[9, 2] = 1.0
        K[9, 3] = 1.0
        K[9, 4] = 1.0
        return K

    @njit(cache=True, fastmath=True)
    def _sigma2_accumulate_numba(x1, var_theta_active):
        n_time, n_state, n_theta = x1.shape
        sigma2 = np.zeros((n_time, n_state)) + 1e-12
        for k in range(n_theta):
            for t in range(n_time):
                for s in range(n_state):
                    sigma2[t, s] += (x1[t, s, k]**2) * var_theta_active[k]
        return sigma2


# =============================================================================
# CONFIGURATION (Paper Table 3)
# =============================================================================

DEBUG = False
ENABLE_PLOTS = True

def get_config(debug: bool) -> Dict[str, Any]:
    """
    Paper-accurate configuration from Table 3
    
    CRITICAL SETTINGS:
    - Ndata = 20 sparse data points (NOT all time steps!)
    - Initial ϕ = 0.2 for M1/M2, 0.02 for M3
    - CoV = 0.5%
    """
    if debug:
        return {
            # Solver settings
            "M1": dict(dt=1e-4, maxtimestep=250, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-4, maxtimestep=500, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=75, c_const=25.0, alpha_const=0.0),
            # Initial conditions (CRITICAL!)
            "phi_init_M1": 0.2,   # Paper: 0.2 for M1/M2
            "phi_init_M2": 0.2,
            "phi_init_M3": 0.02,  # Paper: 0.02 for M3
            # SPARSE DATA (CRITICAL!)
            "Ndata": 20,  # Paper: only 20 data points!
            # TMCMC settings
            "N0": 100,
            "stages": 8,
            "target_ess_ratio": 0.8,
            # TSM settings
            "theta_active_indices_M1": [0, 1, 2, 3, 4],
            "theta_active_indices_M2": [5, 6, 7, 8, 9],
            "theta_active_indices_M3": [10, 11, 12, 13],
            "cov_rel": 0.005,  # 0.5% CoV
            # Observation noise
            "sigma_obs": 0.005,  # Measurement noise std
        }
    else:
        # FULL mode - Paper Table 3 settings
        return {
            # Solver settings (exact from Table 3)
            "M1": dict(dt=1e-5, maxtimestep=2500, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-5, maxtimestep=5000, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=750, c_const=25.0, alpha_const=0.0),
            # Initial conditions (CRITICAL from Table 3!)
            "phi_init_M1": 0.2,   # Paper Table 3: 0.2
            "phi_init_M2": 0.2,   # Paper Table 3: 0.2
            "phi_init_M3": 0.02,  # Paper Table 3: 0.02
            # SPARSE DATA (CRITICAL from Table 3!)
            "Ndata": 20,  # Paper: Ndata = 20
            # TMCMC settings
            "N0": 500,           # Paper: Nsamples = 500
            "Nposterior": 5000,  # Paper: Nposterior = 5000
            "stages": 15,        # With early stopping
            "target_ess_ratio": 0.8,
            # TSM settings
            "theta_active_indices_M1": [0, 1, 2, 3, 4],
            "theta_active_indices_M2": [5, 6, 7, 8, 9],
            "theta_active_indices_M3": [10, 11, 12, 13],
            "cov_rel": 0.005,  # Paper: CoV = 0.5%
            # Observation noise (added to likelihood variance)
            "sigma_obs": 0.005,
        }

CONFIG = get_config(DEBUG)

# =============================================================================
# TRUE PARAMETERS (Paper structure)
# =============================================================================
# Paper Case I: θ* = [1, 0.1, 1, 1, 2] for [a11, a12, a22, b1, b2]
# Extended to 14 parameters for Case II

TRUE_PARAMS = {
    # M1 parameters: [a11, a12, a22, b1, b2]
    # Using similar structure to Case I
    "a11": 0.8,
    "a12": 2.0,
    "a22": 1.0,
    "b1": 0.1,
    "b2": 0.2,
    # M2 parameters: [a33, a34, a44, b3, b4]
    "a33": 1.5,
    "a34": 1.0,
    "a44": 2.0,
    "b3": 0.3,
    "b4": 0.4,
    # M3 parameters: [a13, a14, a23, a24]
    "a13": 2.0,
    "a14": 1.0,
    "a23": 2.0,
    "a24": 1.0,
}

def get_theta_true():
    """Get true parameter vector"""
    return np.array([
        TRUE_PARAMS["a11"], TRUE_PARAMS["a12"], TRUE_PARAMS["a22"], 
        TRUE_PARAMS["b1"], TRUE_PARAMS["b2"],
        TRUE_PARAMS["a33"], TRUE_PARAMS["a34"], TRUE_PARAMS["a44"], 
        TRUE_PARAMS["b3"], TRUE_PARAMS["b4"],
        TRUE_PARAMS["a13"], TRUE_PARAMS["a14"], 
        TRUE_PARAMS["a23"], TRUE_PARAMS["a24"],
    ])


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
    converged: bool = False

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


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

class ProgressTracker:
    def __init__(self, total, desc="Progress", bar_length=40):
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n=1):
        self.current = min(self.current + n, self.total)
        self._display()
    
    def set(self, n):
        self.current = min(n, self.total)
        self._display()
    
    def _display(self):
        percent = self.current / self.total if self.total > 0 else 1
        filled = int(self.bar_length * percent)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0
        sys.stdout.write(f'\r    {self.desc}: |{bar}| {self.current}/{self.total} [{elapsed:.0f}s<{eta:.0f}s]')
        sys.stdout.flush()
    
    def close(self):
        print()


# =============================================================================
# BIOFILM NEWTON SOLVER
# =============================================================================

class BiofilmNewtonSolver:
    """Newton solver with configurable initial conditions"""
    
    THETA_NAMES = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]

    def __init__(self, dt=1e-5, maxtimestep=2500, eps=1e-6, Kp1=1e-4,
                 eta_vec=None, c_const=100.0, alpha_const=100.0, 
                 phi_init=0.02, use_numba=True):
        self.dt = dt
        self.maxtimestep = maxtimestep
        self.eps = eps
        self.Kp1 = Kp1
        self.Eta_vec = np.ones(4) if eta_vec is None else np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = self.Eta_vec.copy()
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)
        self.phi_init = float(phi_init)  # CRITICAL: configurable initial condition
        self.use_numba = use_numba and HAS_NUMBA

    def c(self, t): return self.c_const
    def alpha(self, t): return self.alpha_const

    def get_initial_state(self):
        """Get initial state vector with configured phi_init"""
        phi0_init = 1.0 - 4 * self.phi_init  # Constraint: sum(phi) + phi0 = 1
        return np.array([
            self.phi_init, self.phi_init, self.phi_init, self.phi_init,  # phi_1..4
            phi0_init,  # phi_0
            0.999, 0.999, 0.999, 0.999,  # psi_1..4
            1e-6  # gamma
        ])

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

    def run_deterministic(self, theta, show_progress=False):
        A, b_diag = self.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.dt, self.maxtimestep, self.eps
        g_prev = self.get_initial_state()  # Use configured initial conditions
        t_list, g_list = [0.0], [g_prev.copy()]
        
        pbar = ProgressTracker(maxtimestep, "Forward sim") if show_progress else None
        
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
            
            if pbar and step % 100 == 0:
                pbar.set(step)
        
        if pbar:
            pbar.close()
        
        return np.array(t_list), np.vstack(g_list)


# =============================================================================
# TSM (Taylor Series Method)
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

        g_prev = self.solver.get_initial_state()
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
# LIKELIHOOD (Paper Eq. 29 with observation noise)
# =============================================================================

def log_likelihood_sparse(mu_at_data, sigma2_at_data, data, sigma_obs):
    """
    Paper Eq. (29) with SPARSE data points and observation noise
    
    σ²_total = σ²_TSM + σ²_obs
    
    This is the CRITICAL fix: we add measurement noise variance!
    """
    D = np.asarray(data, dtype=float).ravel()
    m = np.asarray(mu_at_data, dtype=float).ravel()
    v_tsm = np.asarray(sigma2_at_data, dtype=float).ravel()
    
    # CRITICAL: Total variance = TSM variance + observation noise variance
    v_total = v_tsm + sigma_obs**2
    
    if D.shape != m.shape:
        raise ValueError(f"shape mismatch: data={D.shape}, mu={m.shape}")
    
    bad = (~np.isfinite(v_total)) | (v_total <= 0)
    if np.any(bad):
        return -1e20
    
    diff = D - m
    ll = -0.5 * np.sum(np.log(2 * np.pi * v_total)) - 0.5 * np.sum(diff*diff / v_total)
    return max(ll, -1e20) if np.isfinite(ll) else -1e20


def select_sparse_data_indices(n_total, n_data, skip_first=True):
    """
    Select SPARSE data indices (Paper: 20 evenly-spaced points)
    
    Paper: "20 evenly spaced time steps within the interval"
    "Since the initial conditions at t = 0 are known, the first time step is chosen at t = 50"
    """
    start_idx = int(n_total * 0.05) if skip_first else 0  # Skip first ~5% (initial transient)
    indices = np.linspace(start_idx, n_total - 1, n_data, dtype=int)
    return indices


# =============================================================================
# TMCMC
# =============================================================================

# def tmcmc(log_likelihood, log_prior, theta_init_samples, n_stages=15,
#           target_ess_ratio=0.8, adapt_cov=True, random_state=None,
#           show_progress=True, model_name="") -> TMCMCResult:
    
#     rng = np.random.default_rng(random_state)
#     theta_curr = np.array(theta_init_samples, dtype=float)
#     N, d = theta_curr.shape

#     beta_list, samples_list, logw_list = [0.0], [theta_curr.copy()], [np.zeros(N)]
#     logL_trace, acceptance_rates, ess_trace = [], [], []
#     converged = False

#     # Initial likelihood evaluation
#     if show_progress:
#         print(f"    [{model_name}] Evaluating initial likelihoods ({N} samples)...")
#         pbar = ProgressTracker(N, "Init logL")
    
#     logp_prior = np.zeros(N)
#     logL = np.zeros(N)
#     for i in range(N):
#         logp_prior[i] = log_prior(theta_curr[i])
#         logL[i] = log_likelihood(theta_curr[i])
#         if show_progress and (i+1) % max(1, N//20) == 0:
#             pbar.set(i+1)
    
#     if show_progress:
#         pbar.close()
    
#     logp_prior[~np.isfinite(logp_prior)] = -1e20
#     logL[~np.isfinite(logL)] = -1e20
#     logL_trace.append(logL.copy())
    
#     # Print initial logL statistics
#     print(f"    [{model_name}] Initial logL: min={logL.min():.1f}, max={logL.max():.1f}, std={logL.std():.1f}")

#     beta = 0.0

#     for stage in range(1, n_stages + 1):
#         def ess_for_delta(delta_beta):
#             x = delta_beta * (logL - np.max(logL))
#             w_unnorm = np.exp(x)
#             s = np.sum(w_unnorm)
#             if s <= 0 or not np.isfinite(s):
#                 return 0.0
#             w = w_unnorm / s
#             return 1.0 / np.sum(w**2) if np.isfinite(w).all() else 0.0

#         delta_low, delta_high = 0.0, 1.0 - beta
#         for _ in range(25):
#             mid = 0.5 * (delta_low + delta_high)
#             if ess_for_delta(mid) >= target_ess_ratio * N:
#                 delta_low = mid
#             else:
#                 delta_high = mid

#         delta_beta = delta_low
#         beta_next = min(beta + delta_beta, 1.0)

#         x = delta_beta * (logL - np.max(logL))
#         w_unnorm = np.exp(x)
#         s = np.sum(w_unnorm)
#         w = w_unnorm / s if (s > 0 and np.isfinite(s)) else np.ones(N) / N
#         if not np.isfinite(w).all():
#             w = np.ones(N) / N

#         ess = 1.0 / np.sum(w**2)
#         ess_trace.append(ess)
        
#         status = "🎯 CONVERGED!" if beta_next >= 1.0 else ""
#         print(f"    [{model_name}] Stage {stage}: β={beta_next:.4f}, ESS={ess:.1f}/{N} {status}")
#         beta_list.append(beta_next)

#         idx = rng.choice(N, size=N, p=w)
#         theta_resampled = theta_curr[idx]
#         cov = np.cov(theta_resampled.T) + 1e-6 * np.eye(d) if (adapt_cov and stage > 1) else 0.01 * np.eye(d)

#         theta_new = theta_resampled.copy()
#         n_accepted = 0

#         if show_progress:
#             pbar = ProgressTracker(N, f"[{model_name}] MH moves")
        
#         for n in range(N):
#             th_old = theta_resampled[n]
#             lp_old, ll_old = log_prior(th_old), log_likelihood(th_old)
#             if not np.isfinite(lp_old) or not np.isfinite(ll_old):
#                 if show_progress and (n+1) % max(1, N//10) == 0:
#                     pbar.set(n+1)
#                 continue
#             logpost_old = lp_old + beta_next * ll_old

#             prop = rng.multivariate_normal(th_old, cov)
#             lp_prop, ll_prop = log_prior(prop), log_likelihood(prop)
#             if not np.isfinite(lp_prop) or not np.isfinite(ll_prop):
#                 if show_progress and (n+1) % max(1, N//10) == 0:
#                     pbar.set(n+1)
#                 continue
#             logpost_prop = lp_prop + beta_next * ll_prop

#             if rng.uniform() < np.exp(logpost_prop - logpost_old):
#                 theta_new[n] = prop
#                 n_accepted += 1
            
#             if show_progress and (n+1) % max(1, N//10) == 0:
#                 pbar.set(n+1)
        
#         if show_progress:
#             pbar.close()

#         acceptance_rates.append(n_accepted / N)
#         print(f"    [{model_name}] Acceptance rate: {100*n_accepted/N:.1f}%")
        
#         theta_curr = theta_new.copy()
        
#         logp_prior = np.array([log_prior(th) for th in theta_curr])
#         logL = np.array([log_likelihood(th) for th in theta_curr])
#         logp_prior[~np.isfinite(logp_prior)] = -1e20
#         logL[~np.isfinite(logL)] = -1e20
#         logL_trace.append(logL.copy())

#         beta = beta_next
#         samples_list.append(theta_curr.copy())
#         logw_list.append(np.log(w + 1e-300))

#         if beta >= 1.0:
#             print(f"    [{model_name}] ✓ Converged at stage {stage}")
#             converged = True
#             break

#     return TMCMCResult(samples_list, logw_list, beta_list, logL_trace, 
#                        acceptance_rates, ess_trace, converged)


# =============================================================================
# HIERARCHICAL CASE II
# =============================================================================

def hierarchical_case2(config) -> HierarchicalResults:
    """
    Paper-accurate hierarchical updating: M1 → M2 → M3
    
    Key differences from previous implementation:
    1. SPARSE data (Ndata = 20 points, not all time steps)
    2. Correct initial conditions (ϕ = 0.2 for M1/M2, 0.02 for M3)
    3. Observation noise in likelihood
    """
    theta_true = get_theta_true()
    sigma_obs = config.get("sigma_obs", 0.005)
    Ndata = config.get("Ndata", 20)
    
    ## Prior bounds for Case II (Paper: "All prior distributions are chosen as U(0,3)")
    
    # bounds = [(0.0, 3.0)] * 14

    # bounds = [
    #     (0.0, 3.0),  # a11
    #     (0.0, 3.0),  # a12
    #     (0.0, 3.0),  # a22
    #     (0.0, 3.0),  # b1
    #     (0.0, 3.0),  # b2
    #     (0.0, 3.0),  # a33
    #     (0.0, 3.0),  # a34
    #     (0.0, 3.0),  # a44
    #     (0.0, 3.0),  # b3
    #     (0.0, 3.0),  # b4
    #     (0.0, 3.0),  # a13
    #     (0.0, 3.0),  # a14
    #     (0.0, 3.0),  # a23
    #     (0.0, 3.0),  # a24
    # ]
    
    # bounds = [
    #     (0.5, 2.5),  # a11
    #     (0.5, 2.5),  # a12
    #     (0.5, 2.5),  # a22
    #     (0.0, 1.0),  # b1
    #     (0.0, 1.0),  # b2
    #     (0.5, 2.5),  # a33
    #     (0.5, 2.5),  # a34
    #     (0.5, 2.5),  # a44
    #     (0.0, 1.0),  # b3
    #     (0.0, 1.0),  # b4
    #     (0.5, 2.5),  # a13
    #     (0.5, 2.5),  # a14
    #     (0.5, 2.5),  # a23
    #     (0.5, 2.5),  # a24
    # ]
    
    # Prior bounds tuned for better performance
    bounds = [
        (0.4, 1.2),   # a11
        (1.0, 3.0),   # a12
        (0.5, 1.5),   # a22
        (0.05, 0.15), # b1
        (0.10, 0.30), # b2
        (0.75, 2.25), # a33
        (0.5, 1.5),   # a34
        (1.0, 3.0),   # a44
        (0.15, 0.45), # b3
        (0.20, 0.60), # b4
        (1.0, 3.0),   # a13
        (0.5, 1.5),   # a14
        (1.0, 3.0),   # a23
        (0.5, 1.5),   # a24
    ]

    def log_prior_full(theta):
        theta = np.asarray(theta, dtype=float)
        for i, (low, high) in enumerate(bounds):
            if theta[i] < low or theta[i] > high:
                return -np.inf
        return 0.0

    # =========================================================================
    # GENERATE SYNTHETIC DATA
    # =========================================================================
    print("\n[Step 0] Generating synthetic data...")
    np.random.seed(42)
    
    # M1 solver with phi_init = 0.2
    solver_M1 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M1"],  # 0.2
        use_numba=HAS_NUMBA,
        **config["M1"]
    )
    t1, g1 = solver_M1.run_deterministic(theta_true, show_progress=True)
    
    # M2 solver with phi_init = 0.2
    solver_M2 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M2"],  # 0.2
        use_numba=HAS_NUMBA,
        **config["M2"]
    )
    t2, g2 = solver_M2.run_deterministic(theta_true, show_progress=True)
    
    # M3 solver with phi_init = 0.02
    solver_M3 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M3"],  # 0.02
        use_numba=HAS_NUMBA,
        **config["M3"]
    )
    t3, g3 = solver_M3.run_deterministic(theta_true, show_progress=True)
    
    # Compute observables (phi_bar = phi * psi)
    obs1_full = np.stack([g1[:, 0]*g1[:, 5], g1[:, 1]*g1[:, 6]], axis=1)
    obs2_full = np.stack([g2[:, 2]*g2[:, 7], g2[:, 3]*g2[:, 8]], axis=1)
    obs3_full = np.stack([g3[:, i]*g3[:, 5+i] for i in range(4)], axis=1)
    
    # SELECT SPARSE DATA POINTS (CRITICAL!)
    idx1 = select_sparse_data_indices(len(t1), Ndata)
    idx2 = select_sparse_data_indices(len(t2), Ndata)
    idx3 = select_sparse_data_indices(len(t3), Ndata)
    
    t1_sparse, data_M1 = t1[idx1], obs1_full[idx1]
    t2_sparse, data_M2 = t2[idx2], obs2_full[idx2]
    t3_sparse, data_M3 = t3[idx3], obs3_full[idx3]
    
    # Add observation noise
    data_M1 += np.random.normal(0, sigma_obs, data_M1.shape)
    data_M2 += np.random.normal(0, sigma_obs, data_M2.shape)
    data_M3 += np.random.normal(0, sigma_obs, data_M3.shape)
    
    print(f"  ✓ SPARSE Data: M1={data_M1.shape}, M2={data_M2.shape}, M3={data_M3.shape}")
    print(f"  ✓ Data indices: M1={idx1[:3]}...{idx1[-1]}, M2={idx2[:3]}...{idx2[-1]}")

    # =========================================================================
    # STAGE 1: M1 (species 1 & 2)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 1: M1 (species 1 & 2)")
    print(f"  Initial ϕ = {config['phi_init_M1']}, Ndata = {Ndata}")
    print("="*72)
    
    tsm_M1 = BiofilmTSM(solver_M1, cov_rel=config["cov_rel"],
                        active_theta_indices=config["theta_active_indices_M1"])
    
    theta_prior_center = theta_true.copy()
    
    def logL_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        try:
            tsm_res = tsm_M1.solve_tsm(theta_full)
            # Extract at SPARSE data indices
            phi, psi = tsm_res.mu[idx1, 0:4], tsm_res.mu[idx1, 5:9]
            obs = np.stack([phi[:, 0]*psi[:, 0], phi[:, 1]*psi[:, 1]], axis=1)
            var_phi, var_psi = tsm_res.sigma2[idx1, 0:4], tsm_res.sigma2[idx1, 5:9]
            obs_var = np.stack([
                phi[:, 0]**2 * var_psi[:, 0] + psi[:, 0]**2 * var_phi[:, 0],
                phi[:, 1]**2 * var_psi[:, 1] + psi[:, 1]**2 * var_phi[:, 1],
            ], axis=1)
            return log_likelihood_sparse(obs, obs_var, data_M1, sigma_obs)
        except:
            return -1e20

    def log_prior_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        return log_prior_full(theta_full)

    rng = np.random.default_rng(1234)
    init_M1 = rng.uniform([bounds[i][0] for i in range(5)], 
                          [bounds[i][1] for i in range(5)], 
                          size=(config["N0"], 5))
    
    t0 = time.time()
    # res_M1 = tmcmc(logL_M1, log_prior_M1, init_M1, n_stages=config["stages"],
    #                target_ess_ratio=config["target_ess_ratio"], random_state=1234,
    #                show_progress=True, model_name="M1")
    
    res_M1 = tmcmc(
        logL_M1, log_prior_M1, init_M1,
        n_stages=15,
        target_ess_ratio=0.5,      # More aggressive than 0.8
        min_delta_beta=0.01,        # Force progress
        logL_scale=0.2,             # Scale down sharp peak (IMPORTANT!)
        random_state=1234,
        model_name="M1")
    t1_time = time.time() - t0
    
    samples_M1 = res_M1.samples[-1]
    theta_M1_mean = np.mean(samples_M1, axis=0)
    print(f"  M1 posterior mean: {theta_M1_mean}")
    print(f"  M1 true values:    {theta_true[0:5]}")
    print(f"  M1 time: {t1_time:.1f}s, converged: {res_M1.converged}")

    theta_stage2_center = theta_prior_center.copy()
    theta_stage2_center[0:5] = theta_M1_mean

    # =========================================================================
    # STAGE 2: M2 (species 3 & 4)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 2: M2 (species 3 & 4)")
    print(f"  Initial ϕ = {config['phi_init_M2']}, Ndata = {Ndata}")
    print("="*72)
    
    tsm_M2 = BiofilmTSM(solver_M2, cov_rel=config["cov_rel"],
                        active_theta_indices=config["theta_active_indices_M2"])
    
    def logL_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        try:
            tsm_res = tsm_M2.solve_tsm(theta_full)
            phi, psi = tsm_res.mu[idx2, 0:4], tsm_res.mu[idx2, 5:9]
            obs = np.stack([phi[:, 2]*psi[:, 2], phi[:, 3]*psi[:, 3]], axis=1)
            var_phi, var_psi = tsm_res.sigma2[idx2, 0:4], tsm_res.sigma2[idx2, 5:9]
            obs_var = np.stack([
                phi[:, 2]**2 * var_psi[:, 2] + psi[:, 2]**2 * var_phi[:, 2],
                phi[:, 3]**2 * var_psi[:, 3] + psi[:, 3]**2 * var_phi[:, 3],
            ], axis=1)
            return log_likelihood_sparse(obs, obs_var, data_M2, sigma_obs)
        except:
            return -1e20

    def log_prior_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        return log_prior_full(theta_full)

    init_M2 = rng.uniform([bounds[i][0] for i in range(5, 10)], 
                          [bounds[i][1] for i in range(5, 10)], 
                          size=(config["N0"], 5))
    
    t0 = time.time()
    # res_M2 = tmcmc(logL_M2, log_prior_M2, init_M2, n_stages=config["stages"],
    #                target_ess_ratio=config["target_ess_ratio"], random_state=5678,
    #                show_progress=True, model_name="M2")
    
    res_M2 = tmcmc(
        logL_M2, log_prior_M2, init_M2,
        n_stages=15,
        target_ess_ratio=0.5,
        min_delta_beta=0.01,
        logL_scale=0.5,             # Moderate scaling
        random_state=5678,
        model_name="M2")
    t2_time = time.time() - t0
    
    samples_M2 = res_M2.samples[-1]
    theta_M2_mean = np.mean(samples_M2, axis=0)
    print(f"  M2 posterior mean: {theta_M2_mean}")
    print(f"  M2 true values:    {theta_true[5:10]}")
    print(f"  M2 time: {t2_time:.1f}s, converged: {res_M2.converged}")

    theta_stage3_center = theta_stage2_center.copy()
    theta_stage3_center[5:10] = theta_M2_mean

    # =========================================================================
    # STAGE 3: M3 (cross interactions)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 3: M3 (cross interactions)")
    print(f"  Initial ϕ = {config['phi_init_M3']}, Ndata = {Ndata}")
    print("="*72)
    
    tsm_M3 = BiofilmTSM(solver_M3, cov_rel=config["cov_rel"],
                        active_theta_indices=config["theta_active_indices_M3"])
    
    def logL_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        try:
            tsm_res = tsm_M3.solve_tsm(theta_full)
            phi, psi = tsm_res.mu[idx3, 0:4], tsm_res.mu[idx3, 5:9]
            obs = np.stack([phi[:, i]*psi[:, i] for i in range(4)], axis=1)
            var_phi, var_psi = tsm_res.sigma2[idx3, 0:4], tsm_res.sigma2[idx3, 5:9]
            obs_var = np.stack([
                phi[:, i]**2 * var_psi[:, i] + psi[:, i]**2 * var_phi[:, i] 
                for i in range(4)
            ], axis=1)
            return log_likelihood_sparse(obs, obs_var, data_M3, sigma_obs)
        except:
            return -1e20

    def log_prior_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        return log_prior_full(theta_full)

    init_M3 = rng.uniform([bounds[i][0] for i in range(10, 14)], 
                          [bounds[i][1] for i in range(10, 14)], 
                          size=(config["N0"], 4))
    
    t0 = time.time()
    # res_M3 = tmcmc(logL_M3, log_prior_M3, init_M3, n_stages=config["stages"],
    #                target_ess_ratio=config["target_ess_ratio"], random_state=9012,
    #                show_progress=True, model_name="M3")
    
    res_M3 = tmcmc(
        logL_M3, log_prior_M3, init_M3,
        n_stages=15,
        target_ess_ratio=0.5,
        min_delta_beta=0.01,
        logL_scale=1.0,             # No scaling needed
        random_state=9012,
        model_name="M3")
    t3_time = time.time() - t0
    
    samples_M3 = res_M3.samples[-1]
    theta_M3_mean = np.mean(samples_M3, axis=0)
    print(f"  M3 posterior mean: {theta_M3_mean}")
    print(f"  M3 true values:    {theta_true[10:14]}")
    print(f"  M3 time: {t3_time:.1f}s, converged: {res_M3.converged}")

    theta_final = theta_stage3_center.copy()
    theta_final[10:14] = theta_M3_mean

    return HierarchicalResults(
        M1_samples=samples_M1, M2_samples=samples_M2, M3_samples=samples_M3,
        theta_M1_mean=theta_M1_mean, theta_M2_mean=theta_M2_mean, theta_M3_mean=theta_M3_mean,
        theta_final=theta_final, tmcmc_M1=res_M1, tmcmc_M2=res_M2, tmcmc_M3=res_M3
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*72)
    print("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    print("  PAPER-ACCURATE VERSION (Fritsch et al. 2025)")
    print("="*72)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode : {'DEBUG' if DEBUG else 'FULL'}")
    print(f"Numba: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")
    print("="*72)
    
    print("\n[Config] Paper-accurate settings:")
    print(f"  Ndata (sparse points) = {CONFIG['Ndata']}")
    print(f"  phi_init M1/M2 = {CONFIG['phi_init_M1']}")
    print(f"  phi_init M3 = {CONFIG['phi_init_M3']}")
    print(f"  sigma_obs = {CONFIG['sigma_obs']}")
    print(f"  N0 (samples) = {CONFIG['N0']}")
    
    theta_true = get_theta_true()
    print(f"\n[True Parameters]")
    print(f"  θ_true = {theta_true}")

    t_start = time.time()
    results = hierarchical_case2(CONFIG)
    total_time = time.time() - t_start

    print("\n" + "="*72)
    print("  FINAL RESULTS")
    print("="*72)
    print(f"True θ:      {theta_true}")
    print(f"Estimated θ: {results.theta_final}")
    print(f"Error:       {results.theta_final - theta_true}")
    print(f"RMSE:        {np.sqrt(np.mean((results.theta_final - theta_true)**2)):.4f}")
    print(f"Total time:  {total_time:.1f} s")
    print(f"Convergence: M1={results.tmcmc_M1.converged}, M2={results.tmcmc_M2.converged}, M3={results.tmcmc_M3.converged}")
    
    # Print per-parameter comparison
    param_names = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]
    print("\n  Per-parameter comparison:")
    print("  " + "-"*60)
    print(f"  {'Param':<6} {'True':>8} {'Est':>8} {'Error':>10} {'Error%':>10}")
    print("  " + "-"*60)
    for i, name in enumerate(param_names):
        true_val = theta_true[i]
        est_val = results.theta_final[i]
        err = est_val - true_val
        err_pct = 100 * err / (abs(true_val) + 1e-10)
        print(f"  {name:<6} {true_val:>8.3f} {est_val:>8.3f} {err:>10.4f} {err_pct:>9.1f}%")

    print("\n✅ Done.")
    return results


if __name__ == "__main__":
    main()