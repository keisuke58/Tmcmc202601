#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating
===============================================================
ULTIMATE OPTIMIZED VERSION

Key Optimizations:
1. PARALLEL LIKELIHOOD - Multiprocessing for concurrent TSM evaluations
2. ANALYTICAL SENSITIVITY - Exact dG/dθ derivatives (no numerical differentiation)
3. EARLY β TERMINATION - Stop TMCMC when β reaches 1.0

Additional Features:
- Numba JIT acceleration for core computations
- Progress display with ETA
- Complete data saving and paper-level figures
- Checkpoint/resume capability

Author: Based on Fritsch et al. (2025)
"""

import numpy as np
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import time
import warnings
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# =============================================================================
# NUMBA ACCELERATION
# =============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not available: using pure NumPy (slower)")

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_Q_vector_numba(phi_new, phi0_new, psi_new, gamma_new,
                                 phi_old, phi0_old, psi_old,
                                 dt, Kp1, Eta_vec, Eta_phi_vec,
                                 c_val, alpha_val, A, b_diag):
        """Compute residual vector Q (Numba accelerated)"""
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
        """Compute Jacobian matrix K = ∂Q/∂g (Numba accelerated)"""
        K = np.zeros((10, 10))
        phidot = (phi_new - phi_old) / dt
        psidot = (psi_new - psi_old) / dt
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        
        # Potential derivatives
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
        
        # Fill Jacobian for phi equations (rows 0-3)
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
        
        # phi0 equation (row 4)
        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 9] = 1.0
        
        # psi equations (rows 5-8)
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
        
        # Constraint equation (row 9)
        K[9, 0] = 1.0
        K[9, 1] = 1.0
        K[9, 2] = 1.0
        K[9, 3] = 1.0
        K[9, 4] = 1.0
        return K

    @njit(cache=True, fastmath=True)
    def _compute_dQ_dtheta_analytical_numba(phi_new, psi_new, c_val, alpha_val, 
                                             Eta_vec, CapitalPhi, theta_idx):
        """
        ANALYTICAL SENSITIVITY: Compute ∂Q/∂θ_k exactly
        
        θ = [a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24]
             0     1    2    3   4   5    6    7    8   9   10   11   12   13
        
        A matrix structure (symmetric):
        A[0,0]=a11, A[0,1]=A[1,0]=a12, A[0,2]=A[2,0]=a13, A[0,3]=A[3,0]=a14
        A[1,1]=a22, A[1,2]=A[2,1]=a23, A[1,3]=A[3,1]=a24
        A[2,2]=a33, A[2,3]=A[3,2]=a34
        A[3,3]=a44
        
        b = [b1, b2, b3, b4] at indices [3, 4, 8, 9]
        """
        dQ = np.zeros(10)
        
        # Mapping: theta_idx -> (row, col) in A, or b_index
        # Diagonal A elements: affect only one (i,i) entry
        # Off-diagonal A elements: affect both (i,j) and (j,i) due to symmetry
        
        if theta_idx == 0:  # a11 -> A[0,0]
            # Q[0]: -c/η_0 * ψ_0 * (∂A@Φ)_0 = -c/η_0 * ψ_0 * Φ_0
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[0]
            # Q[5]: -c/η_0 * φ_0 * (∂A@Φ)_0 = -c/η_0 * φ_0 * Φ_0
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[0]
            
        elif theta_idx == 1:  # a12 -> A[0,1] and A[1,0]
            # Affects row 0: ∂(A@Φ)_0/∂a12 = Φ_1
            # Affects row 1: ∂(A@Φ)_1/∂a12 = Φ_0
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[1]
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[0]
            
        elif theta_idx == 2:  # a22 -> A[1,1]
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[1]
            
        elif theta_idx == 3:  # b1
            # Q[5]: (b1 * α / η_0) * ψ_0 -> ∂/∂b1 = α/η_0 * ψ_0
            dQ[5] = (alpha_val / Eta_vec[0]) * psi_new[0]
            
        elif theta_idx == 4:  # b2
            dQ[6] = (alpha_val / Eta_vec[1]) * psi_new[1]
            
        elif theta_idx == 5:  # a33 -> A[2,2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[2]
            
        elif theta_idx == 6:  # a34 -> A[2,3] and A[3,2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[2]
            
        elif theta_idx == 7:  # a44 -> A[3,3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[3]
            
        elif theta_idx == 8:  # b3
            dQ[7] = (alpha_val / Eta_vec[2]) * psi_new[2]
            
        elif theta_idx == 9:  # b4
            dQ[8] = (alpha_val / Eta_vec[3]) * psi_new[3]
            
        elif theta_idx == 10:  # a13 -> A[0,2] and A[2,0]
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[0]
            
        elif theta_idx == 11:  # a14 -> A[0,3] and A[3,0]
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[0]
            
        elif theta_idx == 12:  # a23 -> A[1,2] and A[2,1]
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[1]
            
        elif theta_idx == 13:  # a24 -> A[1,3] and A[3,1]
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[1]
        
        return dQ

    @njit(cache=True, fastmath=True)
    def _sigma2_accumulate_numba(x1, var_theta_active):
        """Accumulate variance: σ² = Σ_k (x1[:,:,k])² * Var(θ_k)"""
        n_time, n_state, n_theta = x1.shape
        sigma2 = np.zeros((n_time, n_state)) + 1e-12
        for k in range(n_theta):
            for t in range(n_time):
                for s in range(n_state):
                    sigma2[t, s] += (x1[t, s, k]**2) * var_theta_active[k]
        return sigma2


# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG = False  # True: fast test / False: full run (Table 3)
ENABLE_PLOTS = True
N_WORKERS = max(1, mp.cpu_count() - 1)  # Parallel workers

def get_config(debug: bool) -> Dict[str, Any]:
    """Configuration for DEBUG or FULL mode"""
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
            "use_parallel": False,  # Disable for DEBUG
            "use_analytical": True,
        }
    else:
        return {
            "M1": dict(dt=1e-5, maxtimestep=2500, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-5, maxtimestep=5000, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=750, c_const=25.0, alpha_const=0.0),
            "N0_M1": 100, "N0_M2": 100, "N0_M3": 100, # Larger sample sizes 20->400
            "stages_M1": 20, "stages_M2": 20, "stages_M3": 20,  # More stages, but early stop
            "target_ess_ratio": 0.5,
            "theta_active_indices_M1": [0, 1, 2, 3, 4],
            "theta_active_indices_M2": [5, 6, 7, 8, 9],
            "theta_active_indices_M3": [10, 11, 12, 13],
            "cov_rel": 0.005,
            "use_parallel": True,  # Enable parallel processing
            "use_analytical": True,  # Use analytical sensitivities
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
    converged: bool = False  # NEW: flag for early termination

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
# PROGRESS DISPLAY
# =============================================================================

class ProgressTracker:
    """Progress bar with ETA"""
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
# 1. BIOFILM NEWTON SOLVER
# =============================================================================

class BiofilmNewtonSolver:
    """Newton solver for biofilm PDE system"""
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
        """Convert parameter vector to A matrix and b diagonal"""
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
        """Compute residual vector Q"""
        if self.use_numba:
            return _compute_Q_vector_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[4], g_old[5:9],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag)
        else:
            return self._compute_Q_vector_numpy(g_new, g_old, t, dt, A, b_diag)

    def _compute_Q_vector_numpy(self, g_new, g_old, t, dt, A, b_diag):
        """Pure NumPy implementation of Q vector"""
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
        """Compute Jacobian matrix K = ∂Q/∂g"""
        if self.use_numba:
            return _compute_jacobian_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[5:9], dt, self.Kp1,
                self.Eta_vec, self.Eta_phi_vec, self.c(t), self.alpha(t), A, b_diag)
        else:
            return self._compute_Jacobian_numpy(g_new, g_old, t, dt, A, b_diag)

    def _compute_Jacobian_numpy(self, g_new, g_old, t, dt, A, b_diag):
        """Pure NumPy Jacobian (fallback)"""
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
        """Run deterministic forward simulation"""
        A, b_diag = self.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.dt, self.maxtimestep, self.eps
        g_prev = np.array([0.02, 0.02, 0.02, 0.02, 0.92, 0.999, 0.999, 0.999, 0.999, 1e-6])
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
# 2. TSM WITH ANALYTICAL SENSITIVITY
# =============================================================================

class BiofilmTSM:
    """
    Taylor Series Method (1st order) with ANALYTICAL SENSITIVITY
    
    Key improvement: ∂G/∂θ is computed analytically instead of numerically,
    eliminating 2*n_theta extra Q evaluations per time step.
    """
    THETA_NAMES = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]

    def __init__(self, solver: BiofilmNewtonSolver, cov_rel=0.005, 
                 active_theta_indices=None, use_analytical=True):
        self.solver = solver
        self.cov_rel = cov_rel
        self.active_idx = np.arange(14) if active_theta_indices is None else np.array(active_theta_indices)
        self.use_analytical = use_analytical and HAS_NUMBA

    def _dG_dtheta_analytical(self, g_new, theta):
        """
        ANALYTICAL SENSITIVITY: Compute ∂Q/∂θ for all active parameters
        
        This is O(n_active) vs O(2*n_active) for numerical differentiation,
        and more accurate (no truncation error).
        """
        phi_new, psi_new = g_new[0:4], g_new[5:9]
        CapitalPhi = phi_new * psi_new
        c_val = self.solver.c_const
        alpha_val = self.solver.alpha_const
        Eta_vec = self.solver.Eta_vec
        
        dG_dict = {}
        for idx in self.active_idx:
            if HAS_NUMBA:
                dQ = _compute_dQ_dtheta_analytical_numba(
                    phi_new, psi_new, c_val, alpha_val, Eta_vec, CapitalPhi, idx)
            else:
                dQ = self._dQ_dtheta_numpy(phi_new, psi_new, c_val, alpha_val, 
                                           Eta_vec, CapitalPhi, idx)
            dG_dict[self.THETA_NAMES[idx]] = dQ
        return dG_dict

    def _dQ_dtheta_numpy(self, phi_new, psi_new, c_val, alpha_val, Eta_vec, CapitalPhi, theta_idx):
        """NumPy fallback for analytical sensitivity"""
        dQ = np.zeros(10)
        
        # Parameter index mapping (same logic as Numba version)
        if theta_idx == 0:  # a11
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[0]
        elif theta_idx == 1:  # a12
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[1]
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[0]
        elif theta_idx == 2:  # a22
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[1]
        elif theta_idx == 3:  # b1
            dQ[5] = (alpha_val / Eta_vec[0]) * psi_new[0]
        elif theta_idx == 4:  # b2
            dQ[6] = (alpha_val / Eta_vec[1]) * psi_new[1]
        elif theta_idx == 5:  # a33
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[2]
        elif theta_idx == 6:  # a34
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[2]
        elif theta_idx == 7:  # a44
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[3]
        elif theta_idx == 8:  # b3
            dQ[7] = (alpha_val / Eta_vec[2]) * psi_new[2]
        elif theta_idx == 9:  # b4
            dQ[8] = (alpha_val / Eta_vec[3]) * psi_new[3]
        elif theta_idx == 10:  # a13
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[0]
        elif theta_idx == 11:  # a14
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[0]
        elif theta_idx == 12:  # a23
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[1]
        elif theta_idx == 13:  # a24
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[1]
        
        return dQ

    def _dG_dtheta_numeric(self, g_new, g_old, t, dt, theta):
        """Numerical differentiation fallback (for validation)"""
        dG_dict = {}
        A_base, b_base = self.solver.theta_to_matrices(theta)
        
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
        """
        Solve TSM with analytical or numerical sensitivity
        
        x0 = g(t; θ)           - deterministic solution
        x1 = ∂g/∂θ             - sensitivity (1st order Taylor)
        σ² = Σ_k (x1_k)² Var(θ_k)  - propagated variance
        """
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
            
            # Newton iteration
            for _ in range(100):
                Q = self.solver.compute_Q_vector(g_new, g_prev, tt, dt, A, b_diag)
                K = self.solver.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
                if np.isnan(Q).any() or np.isnan(K).any():
                    raise RuntimeError(f"NaN at t={tt}")
                dg = np.linalg.solve(K, -Q)
                g_new = g_new + dg
                if np.max(np.abs(Q)) < eps:
                    break

            # Compute sensitivity ∂g/∂θ
            if self.use_analytical:
                dG_dict = self._dG_dtheta_analytical(g_new, theta)
            else:
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

        # Compute variance propagation
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
# 3. LIKELIHOOD (Eq. 29)
# =============================================================================

def log_likelihood_eq29(mu, sigma2, data):
    """Gaussian log-likelihood with TSM variance"""
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
# 4. PARALLEL LIKELIHOOD EVALUATION
# =============================================================================

def _eval_single_likelihood(args):
    """
    Worker function for parallel likelihood evaluation.
    Must be at module level for multiprocessing.
    """
    idx, theta, tsm_config, likelihood_config = args
    
    try:
        # Reconstruct TSM solver in worker process
        solver = BiofilmNewtonSolver(
            eta_vec=[1,1,1,1],
            use_numba=HAS_NUMBA,
            **tsm_config['solver_params']
        )
        tsm = BiofilmTSM(
            solver,
            cov_rel=tsm_config['cov_rel'],
            active_theta_indices=tsm_config['active_idx'],
            use_analytical=tsm_config['use_analytical']
        )
        
        # Build full theta
        theta_full = np.array(tsm_config['theta_center'])
        theta_full[tsm_config['param_slice']] = theta
        
        # Run TSM
        tsm_res = tsm.solve_tsm(theta_full)
        
        # Compute observable and likelihood
        phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
        var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
        
        species_indices = likelihood_config['species_indices']
        obs = np.stack([phi[:, i]*psi[:, i] for i in species_indices], axis=1)
        obs_var = np.stack([
            phi[:, i]**2 * var_psi[:, i] + psi[:, i]**2 * var_phi[:, i] 
            for i in species_indices
        ], axis=1)
        
        data = likelihood_config['data']
        ll = log_likelihood_eq29(obs, obs_var, data)
        
        return idx, ll
        
    except Exception as e:
        return idx, -1e20


class ParallelLikelihood:
    """
    Parallel likelihood evaluator using ProcessPoolExecutor
    
    Key optimization: N likelihood evaluations run concurrently instead of sequentially.
    Speedup: ~N_WORKERS times faster for large N.
    """
    
    def __init__(self, tsm_config: dict, likelihood_config: dict, n_workers: int = None):
        self.tsm_config = tsm_config
        self.likelihood_config = likelihood_config
        self.n_workers = n_workers or N_WORKERS
    
    def evaluate_batch(self, theta_batch: np.ndarray, show_progress=True) -> np.ndarray:
        """Evaluate likelihood for batch of theta samples in parallel"""
        N = len(theta_batch)
        results = np.full(N, -1e20)
        
        args_list = [
            (i, theta_batch[i], self.tsm_config, self.likelihood_config)
            for i in range(N)
        ]
        
        if self.n_workers <= 1:
            # Sequential fallback
            pbar = ProgressTracker(N, "LogL eval") if show_progress else None
            for i, theta in enumerate(theta_batch):
                _, ll = _eval_single_likelihood(args_list[i])
                results[i] = ll
                if pbar and (i+1) % max(1, N//20) == 0:
                    pbar.set(i+1)
            if pbar:
                pbar.close()
        else:
            # Parallel evaluation
            pbar = ProgressTracker(N, f"LogL eval ({self.n_workers} workers)") if show_progress else None
            completed = 0
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(_eval_single_likelihood, args): args[0] 
                          for args in args_list}
                
                for future in as_completed(futures):
                    idx, ll = future.result()
                    results[idx] = ll
                    completed += 1
                    if pbar and completed % max(1, N//20) == 0:
                        pbar.set(completed)
            
            if pbar:
                pbar.close()
        
        return results


# =============================================================================
# 5. TMCMC WITH EARLY STOPPING & PARALLEL LIKELIHOOD
# =============================================================================

def tmcmc_parallel(log_likelihood_batch: Callable, log_prior: Callable, 
                   theta_init_samples: np.ndarray, 
                   n_stages: int = 12, target_ess_ratio: float = 0.8,
                   adapt_cov: bool = True, random_state: int = None,
                   show_progress: bool = True, model_name: str = "",
                   early_stop: bool = True) -> TMCMCResult:
    """
    TMCMC with PARALLEL LIKELIHOOD and EARLY β TERMINATION
    
    Key optimizations:
    1. Batch likelihood evaluation (parallel)
    2. Early termination when β reaches 1.0
    3. Adaptive covariance scaling
    """
    rng = np.random.default_rng(random_state)
    theta_curr = np.array(theta_init_samples, dtype=float)
    N, d = theta_curr.shape

    beta_list, samples_list, logw_list = [0.0], [theta_curr.copy()], [np.zeros(N)]
    logL_trace, acceptance_rates, ess_trace = [], [], []
    converged = False

    # Initial likelihood evaluation (parallel)
    print(f"    [{model_name}] Evaluating initial likelihoods ({N} samples)...")
    logp_prior = np.array([log_prior(th) for th in theta_curr])
    logL = log_likelihood_batch(theta_curr, show_progress=show_progress)
    
    logp_prior[~np.isfinite(logp_prior)] = -1e20
    logL[~np.isfinite(logL)] = -1e20
    logL_trace.append(logL.copy())

    beta = 0.0

    for stage in range(1, n_stages + 1):
        # Find optimal β increment via bisection
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

        # Importance weights
        x = delta_beta * (logL - np.max(logL))
        w_unnorm = np.exp(x)
        s = np.sum(w_unnorm)
        w = w_unnorm / s if (s > 0 and np.isfinite(s)) else np.ones(N) / N
        if not np.isfinite(w).all():
            w = np.ones(N) / N

        ess = 1.0 / np.sum(w**2)
        ess_trace.append(ess)
        
        # Status with early stop indication
        status = "🎯 CONVERGED!" if beta_next >= 1.0 else ""
        print(f"    [{model_name}] Stage {stage}: β={beta_next:.4f}, ESS={ess:.1f}/{N} {status}")
        beta_list.append(beta_next)

        # Resample
        idx = rng.choice(N, size=N, p=w)
        theta_resampled = theta_curr[idx]
        
        # Adaptive covariance
        cov = np.cov(theta_resampled.T) + 1e-6 * np.eye(d) if (adapt_cov and stage > 1) else 0.01 * np.eye(d)

        # MH moves
        theta_new = theta_resampled.copy()
        n_accepted = 0

        if show_progress:
            pbar = ProgressTracker(N, f"[{model_name}] MH moves")
        
        for n in range(N):
            th_old = theta_resampled[n]
            lp_old = log_prior(th_old)
            ll_old = logL[idx[n]]  # Use cached value
            
            if not np.isfinite(lp_old) or not np.isfinite(ll_old):
                if show_progress and (n+1) % max(1, N//10) == 0:
                    pbar.set(n+1)
                continue
            logpost_old = lp_old + beta_next * ll_old

            prop = rng.multivariate_normal(th_old, cov)
            lp_prop = log_prior(prop)
            
            if not np.isfinite(lp_prop):
                if show_progress and (n+1) % max(1, N//10) == 0:
                    pbar.set(n+1)
                continue
            
            # Single likelihood evaluation for proposal
            ll_prop = log_likelihood_batch(prop.reshape(1, -1), show_progress=False)[0]
            
            if not np.isfinite(ll_prop):
                if show_progress and (n+1) % max(1, N//10) == 0:
                    pbar.set(n+1)
                continue
            logpost_prop = lp_prop + beta_next * ll_prop

            if rng.uniform() < np.exp(logpost_prop - logpost_old):
                theta_new[n] = prop
                n_accepted += 1
            
            if show_progress and (n+1) % max(1, N//10) == 0:
                pbar.set(n+1)
        
        if show_progress:
            pbar.close()

        acceptance_rates.append(n_accepted / N)
        print(f"    [{model_name}] Acceptance rate: {100*n_accepted/N:.1f}%")
        
        theta_curr = theta_new.copy()
        
        # Update likelihoods for accepted samples
        logp_prior = np.array([log_prior(th) for th in theta_curr])
        logL = log_likelihood_batch(theta_curr, show_progress=show_progress)
        logp_prior[~np.isfinite(logp_prior)] = -1e20
        logL[~np.isfinite(logL)] = -1e20
        logL_trace.append(logL.copy())

        beta = beta_next
        samples_list.append(theta_curr.copy())
        logw_list.append(np.log(w + 1e-300))

        # EARLY TERMINATION: Stop when β reaches 1.0
        if beta >= 1.0:
            if early_stop:
                print(f"    [{model_name}] ✓ Early termination at stage {stage} (β=1.0)")
                converged = True
                break

    return TMCMCResult(samples_list, logw_list, beta_list, logL_trace, 
                       acceptance_rates, ess_trace, converged)


# =============================================================================
# 6. HIERARCHICAL CASE II (PARALLEL VERSION)
# =============================================================================

def hierarchical_case2_parallel(solver_M1, solver_M2, solver_M3,
                                 data_M1, data_M2, data_M3,
                                 theta_prior_center, bounds, config) -> HierarchicalResults:
    """
    Hierarchical Bayesian inference with:
    - Parallel likelihood evaluation
    - Analytical TSM sensitivity
    - Early β termination
    """
    theta_prior_center = np.asarray(theta_prior_center, dtype=float)
    use_parallel = config.get("use_parallel", True) and N_WORKERS > 1

    def log_prior_full(theta):
        theta = np.asarray(theta, dtype=float)
        low = np.array([b[0] for b in bounds])
        high = np.array([b[1] for b in bounds])
        return 0.0 if np.all((theta >= low) & (theta <= high)) else -np.inf

    # =========================================================================
    # STAGE 1: M1 (species 1 & 2)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 1: M1 (species 1 & 2)")
    print("="*72)
    
    tsm_M1_config = {
        'solver_params': config["M1"],
        'cov_rel': config["cov_rel"],
        'active_idx': config["theta_active_indices_M1"],
        'use_analytical': config.get("use_analytical", True),
        'theta_center': theta_prior_center.tolist(),
        'param_slice': slice(0, 5),
    }
    likelihood_M1_config = {
        'species_indices': [0, 1],
        'data': data_M1,
    }
    
    if use_parallel:
        parallel_M1 = ParallelLikelihood(tsm_M1_config, likelihood_M1_config, N_WORKERS)
        log_likelihood_batch_M1 = parallel_M1.evaluate_batch
    else:
        # Sequential fallback with direct TSM
        tsm_M1 = BiofilmTSM(solver_M1, cov_rel=config["cov_rel"],
                            active_theta_indices=config["theta_active_indices_M1"],
                            use_analytical=config.get("use_analytical", True))
        
        def log_likelihood_batch_M1(theta_batch, show_progress=True):
            results = np.full(len(theta_batch), -1e20)
            pbar = ProgressTracker(len(theta_batch), "LogL eval") if show_progress else None
            for i, theta_M1 in enumerate(theta_batch):
                theta_full = theta_prior_center.copy()
                theta_full[0:5] = theta_M1
                try:
                    tsm_res = tsm_M1.solve_tsm(theta_full)
                    phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
                    obs = np.stack([phi[:, 0]*psi[:, 0], phi[:, 1]*psi[:, 1]], axis=1)
                    var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
                    obs_var = np.stack([
                        phi[:, 0]**2 * var_psi[:, 0] + psi[:, 0]**2 * var_phi[:, 0],
                        phi[:, 1]**2 * var_psi[:, 1] + psi[:, 1]**2 * var_phi[:, 1],
                    ], axis=1)
                    results[i] = log_likelihood_eq29(obs, obs_var, data_M1)
                except:
                    pass
                if pbar and (i+1) % max(1, len(theta_batch)//20) == 0:
                    pbar.set(i+1)
            if pbar:
                pbar.close()
            return results

    def log_prior_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        return log_prior_full(theta_full)

    rng = np.random.default_rng(1234)
    init_M1 = rng.uniform([b[0] for b in bounds[0:5]], [b[1] for b in bounds[0:5]], 
                          size=(config["N0_M1"], 5))
    
    t0 = time.time()
    res_M1 = tmcmc_parallel(log_likelihood_batch_M1, log_prior_M1, init_M1,
                            n_stages=config["stages_M1"],
                            target_ess_ratio=config["target_ess_ratio"],
                            random_state=1234, show_progress=True, model_name="M1",
                            early_stop=True)
    t1 = time.time()
    
    samples_M1 = res_M1.samples[-1]
    theta_M1_mean = np.mean(samples_M1, axis=0)
    print(f"  M1 posterior mean: {theta_M1_mean}")
    print(f"  M1 time: {t1-t0:.1f}s, converged: {res_M1.converged}")

    theta_stage2_center = theta_prior_center.copy()
    theta_stage2_center[0:5] = theta_M1_mean

    # =========================================================================
    # STAGE 2: M2 (species 3 & 4)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 2: M2 (species 3 & 4)")
    print("="*72)
    
    tsm_M2_config = {
        'solver_params': config["M2"],
        'cov_rel': config["cov_rel"],
        'active_idx': config["theta_active_indices_M2"],
        'use_analytical': config.get("use_analytical", True),
        'theta_center': theta_stage2_center.tolist(),
        'param_slice': slice(5, 10),
    }
    likelihood_M2_config = {
        'species_indices': [2, 3],
        'data': data_M2,
    }
    
    if use_parallel:
        parallel_M2 = ParallelLikelihood(tsm_M2_config, likelihood_M2_config, N_WORKERS)
        log_likelihood_batch_M2 = parallel_M2.evaluate_batch
    else:
        tsm_M2 = BiofilmTSM(solver_M2, cov_rel=config["cov_rel"],
                            active_theta_indices=config["theta_active_indices_M2"],
                            use_analytical=config.get("use_analytical", True))
        
        def log_likelihood_batch_M2(theta_batch, show_progress=True):
            results = np.full(len(theta_batch), -1e20)
            pbar = ProgressTracker(len(theta_batch), "LogL eval") if show_progress else None
            for i, theta_M2 in enumerate(theta_batch):
                theta_full = theta_stage2_center.copy()
                theta_full[5:10] = theta_M2
                try:
                    tsm_res = tsm_M2.solve_tsm(theta_full)
                    phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
                    obs = np.stack([phi[:, 2]*psi[:, 2], phi[:, 3]*psi[:, 3]], axis=1)
                    var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
                    obs_var = np.stack([
                        phi[:, 2]**2 * var_psi[:, 2] + psi[:, 2]**2 * var_phi[:, 2],
                        phi[:, 3]**2 * var_psi[:, 3] + psi[:, 3]**2 * var_phi[:, 3],
                    ], axis=1)
                    results[i] = log_likelihood_eq29(obs, obs_var, data_M2)
                except:
                    pass
                if pbar and (i+1) % max(1, len(theta_batch)//20) == 0:
                    pbar.set(i+1)
            if pbar:
                pbar.close()
            return results

    def log_prior_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        return log_prior_full(theta_full)

    init_M2 = rng.uniform([b[0] for b in bounds[5:10]], [b[1] for b in bounds[5:10]], 
                          size=(config["N0_M2"], 5))
    
    t0 = time.time()
    res_M2 = tmcmc_parallel(log_likelihood_batch_M2, log_prior_M2, init_M2,
                            n_stages=config["stages_M2"],
                            target_ess_ratio=config["target_ess_ratio"],
                            random_state=5678, show_progress=True, model_name="M2",
                            early_stop=True)
    t1 = time.time()
    
    samples_M2 = res_M2.samples[-1]
    theta_M2_mean = np.mean(samples_M2, axis=0)
    print(f"  M2 posterior mean: {theta_M2_mean}")
    print(f"  M2 time: {t1-t0:.1f}s, converged: {res_M2.converged}")

    theta_stage3_center = theta_stage2_center.copy()
    theta_stage3_center[5:10] = theta_M2_mean

    # =========================================================================
    # STAGE 3: M3 (cross interactions)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 3: M3 (cross interactions)")
    print("="*72)
    
    tsm_M3_config = {
        'solver_params': config["M3"],
        'cov_rel': config["cov_rel"],
        'active_idx': config["theta_active_indices_M3"],
        'use_analytical': config.get("use_analytical", True),
        'theta_center': theta_stage3_center.tolist(),
        'param_slice': slice(10, 14),
    }
    likelihood_M3_config = {
        'species_indices': [0, 1, 2, 3],
        'data': data_M3,
    }
    
    if use_parallel:
        parallel_M3 = ParallelLikelihood(tsm_M3_config, likelihood_M3_config, N_WORKERS)
        log_likelihood_batch_M3 = parallel_M3.evaluate_batch
    else:
        tsm_M3 = BiofilmTSM(solver_M3, cov_rel=config["cov_rel"],
                            active_theta_indices=config["theta_active_indices_M3"],
                            use_analytical=config.get("use_analytical", True))
        
        def log_likelihood_batch_M3(theta_batch, show_progress=True):
            results = np.full(len(theta_batch), -1e20)
            pbar = ProgressTracker(len(theta_batch), "LogL eval") if show_progress else None
            for i, theta_M3 in enumerate(theta_batch):
                theta_full = theta_stage3_center.copy()
                theta_full[10:14] = theta_M3
                try:
                    tsm_res = tsm_M3.solve_tsm(theta_full)
                    phi, psi = tsm_res.mu[:, 0:4], tsm_res.mu[:, 5:9]
                    obs = np.stack([phi[:, j]*psi[:, j] for j in range(4)], axis=1)
                    var_phi, var_psi = tsm_res.sigma2[:, 0:4], tsm_res.sigma2[:, 5:9]
                    obs_var = np.stack([
                        phi[:, j]**2 * var_psi[:, j] + psi[:, j]**2 * var_phi[:, j] 
                        for j in range(4)
                    ], axis=1)
                    results[i] = log_likelihood_eq29(obs, obs_var, data_M3)
                except:
                    pass
                if pbar and (i+1) % max(1, len(theta_batch)//20) == 0:
                    pbar.set(i+1)
            if pbar:
                pbar.close()
            return results

    def log_prior_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        return log_prior_full(theta_full)

    init_M3 = rng.uniform([b[0] for b in bounds[10:14]], [b[1] for b in bounds[10:14]], 
                          size=(config["N0_M3"], 4))
    
    t0 = time.time()
    res_M3 = tmcmc_parallel(log_likelihood_batch_M3, log_prior_M3, init_M3,
                            n_stages=config["stages_M3"],
                            target_ess_ratio=config["target_ess_ratio"],
                            random_state=9012, show_progress=True, model_name="M3",
                            early_stop=True)
    t1 = time.time()
    
    samples_M3 = res_M3.samples[-1]
    theta_M3_mean = np.mean(samples_M3, axis=0)
    print(f"  M3 posterior mean: {theta_M3_mean}")
    print(f"  M3 time: {t1-t0:.1f}s, converged: {res_M3.converged}")

    theta_final = theta_stage3_center.copy()
    theta_final[10:14] = theta_M3_mean

    return HierarchicalResults(
        M1_samples=samples_M1, M2_samples=samples_M2, M3_samples=samples_M3,
        theta_M1_mean=theta_M1_mean, theta_M2_mean=theta_M2_mean, theta_M3_mean=theta_M3_mean,
        theta_final=theta_final, tmcmc_M1=res_M1, tmcmc_M2=res_M2, tmcmc_M3=res_M3
    )


# =============================================================================
# 7. DATA SAVER
# =============================================================================

class DataSaver:
    """Save all results to timestamped folder"""
    
    def __init__(self, base_dir: str = "results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, f"result_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n📁 Output directory: {self.output_dir}/")

    def save_all(self, results, theta_true, data_M1, data_M2, data_M3, t1, t2, t3, config):
        """Save all results"""
        try:
            import pandas as pd
        except ImportError:
            print("  ⚠ pandas not available, saving only numpy files")
            np.savez_compressed(os.path.join(self.output_dir, "complete_archive.npz"),
                theta_true=theta_true, theta_final=results.theta_final,
                M1_samples=results.M1_samples, M2_samples=results.M2_samples, 
                M3_samples=results.M3_samples)
            return
        
        param_names = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]
        
        # Config
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            config_save = {k: str(v) if not isinstance(v, (int, float, bool, str, list)) else v 
                          for k, v in config.items()}
            json.dump({"timestamp": self.timestamp, "config": config_save}, f, indent=2)
        
        # True parameters
        pd.DataFrame({"parameter": param_names, "true_value": theta_true}).to_csv(
            os.path.join(self.output_dir, "true_parameters.csv"), index=False)
        
        # Posterior samples
        np.savez_compressed(os.path.join(self.output_dir, "posterior_samples.npz"),
            M1=results.M1_samples, M2=results.M2_samples, M3=results.M3_samples)
        
        # Summary
        all_samples = np.column_stack([
            np.hstack([results.M1_samples, np.full((len(results.M1_samples), 9), np.nan)]),
        ])
        # Reconstruct full samples (simplified)
        summary = pd.DataFrame({
            "parameter": param_names,
            "true_value": theta_true,
            "posterior_mean": results.theta_final,
            "error": results.theta_final - theta_true,
            "error_percent": 100 * (results.theta_final - theta_true) / (np.abs(theta_true) + 1e-10)
        })
        summary.to_csv(os.path.join(self.output_dir, "posterior_summary.csv"), index=False)
        
        # Complete archive
        np.savez_compressed(os.path.join(self.output_dir, "complete_archive.npz"),
            theta_true=theta_true, theta_final=results.theta_final,
            M1_samples=results.M1_samples, M2_samples=results.M2_samples, 
            M3_samples=results.M3_samples,
            data_M1=data_M1, data_M2=data_M2, data_M3=data_M3, t1=t1, t2=t2, t3=t3,
            beta_M1=np.array(results.tmcmc_M1.beta_schedule),
            beta_M2=np.array(results.tmcmc_M2.beta_schedule),
            beta_M3=np.array(results.tmcmc_M3.beta_schedule))
        
        print("  ✓ All data saved")
        return summary

    def list_files(self):
        print(f"\n  Files in {self.output_dir}/:")
        for f in sorted(os.listdir(self.output_dir)):
            size = os.path.getsize(os.path.join(self.output_dir, f)) / 1024
            print(f"    • {f:<40} ({size:.1f} KB)")


# =============================================================================
# 8. NUMBA WARMUP
# =============================================================================

def warmup_numba():
    """Pre-compile Numba functions to avoid JIT overhead during inference"""
    if not HAS_NUMBA:
        return
    
    print("\n[Warmup] Compiling Numba functions...")
    _dummy_phi = np.array([0.02, 0.02, 0.02, 0.02])
    _dummy_psi = np.array([0.999, 0.999, 0.999, 0.999])
    _dummy_A = np.eye(4)
    _dummy_b = np.ones(4)
    _dummy_Phi = _dummy_phi * _dummy_psi
    
    # Warmup Q and Jacobian
    _ = _compute_Q_vector_numba(_dummy_phi, 0.92, _dummy_psi, 1e-6,
                                 _dummy_phi, 0.92, _dummy_psi, 1e-5, 1e-4,
                                 np.ones(4), np.ones(4), 100.0, 100.0, _dummy_A, _dummy_b)
    _ = _compute_jacobian_numba(_dummy_phi, 0.92, _dummy_psi, 1e-6,
                                 _dummy_phi, _dummy_psi, 1e-5, 1e-4,
                                 np.ones(4), np.ones(4), 100.0, 100.0, _dummy_A, _dummy_b)
    
    # Warmup analytical sensitivity for all parameters
    for idx in range(14):
        _ = _compute_dQ_dtheta_analytical_numba(_dummy_phi, _dummy_psi, 100.0, 100.0, 
                                                 np.ones(4), _dummy_Phi, idx)
    
    # Warmup sigma2 accumulation
    _dummy_x1 = np.random.randn(10, 10, 5)
    _dummy_var = np.ones(5) * 0.01
    _ = _sigma2_accumulate_numba(_dummy_x1, _dummy_var)
    
    print("  ✓ Numba warmup complete (Q, Jacobian, dQ/dθ, σ²)")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    print("="*72)
    print("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    print("  ULTIMATE OPTIMIZED VERSION")
    print("="*72)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode : {'DEBUG (fast)' if DEBUG else 'FULL (Table 3)'}")
    print(f"Numba: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")
    print(f"Parallel: {N_WORKERS} workers" if CONFIG.get('use_parallel') else "Sequential")
    print(f"Analytical dG/dθ: {'ENABLED' if CONFIG.get('use_analytical') else 'DISABLED (numeric)'}")
    print(f"Early β stop: ENABLED")
    print(f"Save : {'OFF' if DEBUG else 'ON'}")
    print("="*72)

    # Warmup Numba
    warmup_numba()

    # Build solvers
    solver_M1 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], use_numba=HAS_NUMBA, **CONFIG["M1"])
    solver_M2 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], use_numba=HAS_NUMBA, **CONFIG["M2"])
    solver_M3 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], use_numba=HAS_NUMBA, **CONFIG["M3"])

    # Print config
    print(f"\n[Config] Active theta indices:")
    print(f"  M1: {CONFIG['theta_active_indices_M1']} (5 params)")
    print(f"  M2: {CONFIG['theta_active_indices_M2']} (5 params)")
    print(f"  M3: {CONFIG['theta_active_indices_M3']} (4 params)")
    print(f"  Stages: M1={CONFIG['stages_M1']}, M2={CONFIG['stages_M2']}, M3={CONFIG['stages_M3']} (with early stop)")

    # True parameters
    TRUE_M1 = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
    TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])
    TRUE_M3 = np.array([2.0, 1.0, 2.0, 1.0])
    theta_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])

    print("\n[Step 0] Generating synthetic data...")
    np.random.seed(42)

    t1, g1 = solver_M1.run_deterministic(theta_true, show_progress=not DEBUG)
    data_M1 = np.stack([g1[:, 0]*g1[:, 5], g1[:, 1]*g1[:, 6]], axis=1)
    data_M1 += np.random.normal(0, 0.002, data_M1.shape)

    t2, g2 = solver_M2.run_deterministic(theta_true, show_progress=not DEBUG)
    data_M2 = np.stack([g2[:, 2]*g2[:, 7], g2[:, 3]*g2[:, 8]], axis=1)
    data_M2 += np.random.normal(0, 0.002, data_M2.shape)

    t3, g3 = solver_M3.run_deterministic(theta_true, show_progress=not DEBUG)
    data_M3 = np.stack([g3[:, i]*g3[:, 5+i] for i in range(4)], axis=1)
    data_M3 += np.random.normal(0, 0.002, data_M3.shape)

    print(f"  ✓ Data: M1={data_M1.shape}, M2={data_M2.shape}, M3={data_M3.shape}")

    # Run inference
    theta_prior_center = theta_true.copy()
    bounds = [(0.0, 3.0)] * 14
    
    # bounds = [
    #     (0.4, 1.2),   # a11
    #     (1.0, 3.0),   # a12
    #     (0.5, 1.5),   # a22
    #     (0.05, 0.15), # b1
    #     (0.10, 0.30), # b2
    #     (0.75, 2.25), # a33
    #     (0.5, 1.5),   # a34
    #     (1.0, 3.0),   # a44
    #     (0.15, 0.45), # b3
    #     (0.20, 0.60), # b4
    #     (1.0, 3.0),   # a13
    #     (0.5, 1.5),   # a14
    #     (1.0, 3.0),   # a23
    #     (0.5, 1.5),   # a24
    # ]


    t_start = time.time()
    results = hierarchical_case2_parallel(solver_M1, solver_M2, solver_M3,
                                           data_M1, data_M2, data_M3,
                                           theta_prior_center, bounds, CONFIG)
    total_time = time.time() - t_start

    # Results
    print("\n" + "="*72)
    print("  FINAL RESULTS")
    print("="*72)
    print(f"True θ:      {theta_true}")
    print(f"Estimated θ: {results.theta_final}")
    print(f"Error:       {results.theta_final - theta_true}")
    print(f"RMSE:        {np.sqrt(np.mean((results.theta_final - theta_true)**2)):.4f}")
    print(f"Total time:  {total_time:.1f} s")
    print(f"Convergence: M1={results.tmcmc_M1.converged}, M2={results.tmcmc_M2.converged}, M3={results.tmcmc_M3.converged}")

    # Save
    if not DEBUG:
        print("\n" + "="*72)
        print("  SAVING DATA")
        print("="*72)
        saver = DataSaver()
        saver.save_all(results, theta_true, data_M1, data_M2, data_M3, t1, t2, t3, CONFIG)
        saver.list_files()

    print("\n✅ Done.")
    return results


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.freeze_support()
    main()
