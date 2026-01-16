#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating
==============================================================
Complete Version with Data Saving & Visualization

Features:
- 4-species continuum biofilm model (Junker-type Newton solver)
- TSM (Time-Separated Mechanics, 1st-order Taylor)
- TMCMC (Transitional MCMC, Ching & Chen style)
- Hierarchical Case II: M1 -> M2 -> M3
- Complete data saving (only when DEBUG=False)
- Visualization (corner plots, time series, convergence)

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
# CONFIG: DEBUG or FULL
# =============================================================================

DEBUG = False  # True: 高速テスト / False: 論文準拠 + 全データ保存

ENABLE_PLOTS = True  # 可視化ON/OFF

def get_config(debug: bool) -> Dict[str, Any]:
    """
    debug=True : 軽量高速設定 (データ保存なし)
    debug=False: Table 3 に対応した本番設定 (全データ保存)
    """
    if debug:
        return {
            "M1": dict(dt=1e-4, maxtimestep=50, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-4, maxtimestep=60, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=40, c_const=25.0,  alpha_const=0.0),
            "N0_M1": 10,
            "N0_M2": 10,
            "N0_M3": 10,
            "stages_M1": 1,
            "stages_M2": 1,
            "stages_M3": 1,
            "target_ess_ratio": 0.8,
            "theta_active_indices": list(range(14)),
            "cov_rel": 0.005
        }
    else:
        # Table 3-like (本番)
        return {
            "M1": dict(dt=1e-5, maxtimestep=2500, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-5, maxtimestep=5000, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=750,  c_const=25.0,  alpha_const=0.0),
            "N0_M1": 200,
            "N0_M2": 200,
            "N0_M3": 200,
            "stages_M1": 6,
            "stages_M2": 6,
            "stages_M3": 6,
            "target_ess_ratio": 0.8,
            "theta_active_indices": list(range(14)),
            "cov_rel": 0.005
        }

CONFIG = get_config(DEBUG)

# =============================================================================
# OUTPUT FOLDER
# =============================================================================

def make_output_dir(debug: bool) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "debug" if debug else "full"
    outdir = f"output_{mode}_{now}"
    os.makedirs(outdir, exist_ok=True)
    return outdir

OUTPUT_DIR = make_output_dir(DEBUG)

# =============================================================================
# Optional: numba acceleration for the Newton core
# =============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# =============================================================================
# 1. Biofilm Newton Solver (4-species, Junker-type)
# =============================================================================

class BiofilmNewtonSolver:
    """
    State vector g (10):
      g = [phi1, phi2, phi3, phi4, phi0, psi1, psi2, psi3, psi4, gamma]
    Parameters theta (14):
      [a11,a12,a22,b1,b2, a33,a34,a44,b3,b4, a13,a14,a23,a24]
    """

    def __init__(self, dt=1e-5, maxtimestep=2500, eps=1e-6, Kp1=1e-4,
                 eta_vec=None, c_const=100.0, alpha_const=100.0):
        self.dt = dt
        self.maxtimestep = maxtimestep
        self.eps = eps
        self.Kp1 = Kp1
        self.Eta_vec = np.ones(4) if eta_vec is None else np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = self.Eta_vec.copy()
        self.c_const = c_const
        self.alpha_const = alpha_const

    # ------------------------
    # Nutrient & Antibiotic
    # ------------------------
    def c(self, t: float) -> float:
        return self.c_const

    def alpha(self, t: float) -> float:
        return self.alpha_const

    # ------------------------
    # θ -> (A,b) mapping
    # ------------------------
    def theta_to_matrices(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24 = theta

        A = np.array([
            [a11, a12, a13, a14],
            [a12, a22, a23, a24],  # symmetrical like in the paper
            [a13, a23, a33, a34],
            [a14, a24, a34, a44]
        ], dtype=float)
        b_diag = np.array([b1, b2, b3, b4], dtype=float)
        return A, b_diag

    # ------------------------
    # Newton system: Q(g_new) = 0
    # ------------------------
    def compute_Q_vector(self, g_new, g_old, t, dt, A, b_diag):
        phi_new = g_new[0:4]
        phi0_new = g_new[4]
        psi_new = g_new[5:9]
        gamma_new = g_new[9]
        phi_old = g_old[0:4]
        phi0_old = g_old[4]
        psi_old = g_old[5:9]

        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt

        Q = np.zeros(10)
        CapitalPhi = phi_new * psi_new
        Interaction_dot_product = A @ CapitalPhi
        c_t_value = self.c(t)

        term1_phi = (self.Kp1 * (2.0 - 4.0 * phi_new)) / (
            np.power(phi_new - 1.0, 3) * np.power(phi_new, 3))
        term2_phi = (1.0 / self.Eta_vec) * (
            gamma_new + (self.Eta_phi_vec + self.Eta_vec * psi_new**2) * phidot +
            self.Eta_vec * phi_new * psi_new * psidot)
        term3_phi = (c_t_value / self.Eta_vec) * psi_new * Interaction_dot_product
        Q[0:4] = term1_phi + term2_phi - term3_phi

        Q[4] = gamma_new + (self.Kp1 * (2.0 - 4.0 * phi0_new)) / (
            np.power(phi0_new - 1.0, 3) * np.power(phi0_new, 3)) + phi0dot

        term1_psi = (-2.0 * self.Kp1) / (np.power(psi_new - 1.0, 2) * np.power(psi_new, 3)) - \
                    (2.0 * self.Kp1) / (np.power(psi_new - 1.0, 3) * np.power(psi_new, 2))
        term2_psi = (b_diag * self.alpha(t) / self.Eta_vec) * psi_new
        term3_psi = phi_new * psi_new * phidot + phi_new**2 * psidot
        term4_psi = (c_t_value / self.Eta_vec) * phi_new * Interaction_dot_product
        Q[5:9] = term1_psi + term2_psi + term3_psi - term4_psi

        Q[9] = np.sum(phi_new) + phi0_new - 1.0
        return Q

    def compute_Jacobian_matrix(self, g_new, g_old, t, dt, A, b_diag):
        v = g_new
        phi_new = g_new[0:4]
        phi0_new = g_new[4]
        psi_new = g_new[5:9]
        phidot = (phi_new - g_old[0:4]) / dt
        psidot = (psi_new - g_old[5:9]) / dt
        c_t_value = self.c(t)
        CapitalPhi = phi_new * psi_new
        Interaction_dot_product = A @ CapitalPhi

        K = np.zeros((10, 10))

        phi_p_deriv = (self.Kp1*(-4. + 8.*v[0:4]))/(np.power(v[0:4],3)*np.power(v[0:4]-1.,3)) - \
                      (self.Kp1*(2. - 4.*v[0:4]))*(3./(np.power(v[0:4],4)*np.power(v[0:4]-1.,3)) +
                                                   3./(np.power(v[0:4],3)*np.power(v[0:4]-1.,4)))
        phi0_p_deriv = (self.Kp1*(-4. + 8.*v[4]))/(np.power(v[4],3)*np.power(v[4]-1.,3)) - \
                       (self.Kp1*(2. - 4.*v[4]))*(3./(np.power(v[4],4)*np.power(v[4]-1.,3)) +
                                                  3./(np.power(v[4],3)*np.power(v[4]-1.,4)))
        psi_p_deriv = (4.0 * self.Kp1 * (3.0 - 5.0*v[5:9] + 5.0*v[5:9]**2)) / \
                      (np.power(v[5:9], 4) * np.power(v[5:9] - 1.0, 4))

        # Rows 0..3: Q_phi
        for i in range(4):
            for j in range(4):
                K[i, j] = (c_t_value / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] = phi_p_deriv[i] + (1.0 / self.Eta_vec[i]) * (
                (self.Eta_phi_vec[i] + self.Eta_vec[i] * psi_new[i]**2) / dt +
                self.Eta_vec[i] * psi_new[i] * psidot[i]) - \
                (c_t_value / self.Eta_vec[i]) * (
                    psi_new[i] * (Interaction_dot_product[i] + A[i, i] * psi_new[i]))
            K[i, 4] = 0.0
            for j in range(4):
                K[i, j+5] = (c_t_value / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
            K[i, i+5] = (1.0 / self.Eta_vec[i]) * (
                2.0 * self.Eta_vec[i] * psi_new[i] * phidot[i] +
                self.Eta_vec[i] * phi_new[i] * psidot[i] +
                self.Eta_vec[i] * phi_new[i] * psi_new[i] / dt) - \
                (c_t_value / self.Eta_vec[i]) * (
                    (Interaction_dot_product[i] + A[i, i] * phi_new[i] * psi_new[i]) +
                    psi_new[i] * (A[i, i] * phi_new[i]))
            K[i, 9] = 1.0 / self.Eta_vec[i]

        # Row 4: Q_phi0
        K[4, 0:4] = 0.0
        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 5:9] = 0.0
        K[4, 9] = 1.0

        # Rows 5..8: Q_psi
        for i in range(4):
            k = i + 5
            for j in range(4):
                K[k, j] = - (c_t_value / self.Eta_vec[i]) * (
                    A[i, j] * psi_new[j] * phi_new[i] +
                    Interaction_dot_product[i] * (1.0 if i == j else 0.0))
            K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt +
                       2.0 * phi_new[i] * psidot[i]) - \
                      (c_t_value / self.Eta_vec[i]) * (
                          A[i, i] * psi_new[i] * phi_new[i] +
                          Interaction_dot_product[i] + phi_new[i] * A[i, i] * psi_new[i])
            K[k, 4] = 0.0
            for j in range(4):
                K[k, j+5] = - (c_t_value / self.Eta_vec[i]) * phi_new[i] * A[i, j] * phi_new[j]
            K[k, i+5] = psi_p_deriv[i] + (b_diag[i] * self.alpha(t) / self.Eta_vec[i]) + \
                        (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                        (c_t_value / self.Eta_vec[i]) * phi_new[i] * A[i, i] * phi_new[i]
            K[k, 9] = 0.0

        K[9, 0:5] = 1.0
        
        return K

    # ------------------------
    # Forward simulation g(t; theta)
    # ------------------------
    def simulate(self, theta: np.ndarray):
        A, b_diag = self.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.dt, self.maxtimestep, self.eps

        # initial state
        g_prev = np.array([0.02, 0.02, 0.02, 0.02,
                           1.0 - 4*0.02,
                           0.999, 0.999, 0.999, 0.999,
                           1e-6])
        t_list = [0.0]
        g_list = [g_prev.copy()]

        for step in range(maxtimestep):
            t_now = (step+1) * dt
            g_new = g_prev.copy()
            for _ in range(100):
                Q = self.compute_Q_vector(g_new, g_prev, t_now, dt, A, b_diag)
                K = self.compute_Jacobian_matrix(g_new, g_prev, t_now, dt, A, b_diag)
                if np.isnan(Q).any() or np.isnan(K).any():
                    raise RuntimeError(f"NaN encountered at t={t_now}")

                try:
                    dg = np.linalg.solve(K, -Q)
                except np.linalg.LinAlgError:
                    raise RuntimeError(f"Jacobian singular at t={t_now}")

                g_new += dg
                if np.max(np.abs(Q)) < eps:
                    break

            g_prev = g_new.copy()
            t_list.append(t_now)
            g_list.append(g_prev.copy())

        t_array = np.array(t_list)
        x = np.vstack(g_list)
        return t_array, x

# =============================================================================
# 2. TSM (Time-separated Mechanics, 1st order)
# =============================================================================

@dataclass
class TSMResult:
    t_array: np.ndarray
    mu: np.ndarray
    sigma2: np.ndarray
    x0: np.ndarray
    x1: np.ndarray

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
        sigma2 = np.zeros_like(mu) + 1e-12
        for k in range(theta_dim):
            sigma2 += (x1[:, :, k]**2) * var_theta_active[k]

        return TSMResult(t_array=t_array, mu=mu, sigma2=sigma2, x0=x0, x1=x1)


# =============================================================================
# 3. Likelihood (Eq. 29) - UNCHANGED
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
    return float(ll)

# =============================================================================
# 4. TMCMC (Transitional MCMC)
# =============================================================================

@dataclass
class TMCMCResult:
    samples: np.ndarray
    logL: np.ndarray
    betas: List[float]
    stage_sizes: List[int]

def tmcmc(log_likelihood_func, log_prior_func, theta0: np.ndarray,
          N0=50, stages=5, target_ess_ratio=0.8, random_state=None) -> TMCMCResult:
    rng = np.random.default_rng(random_state)
    d = len(theta0)
    theta = np.tile(theta0, (N0, 1))
    logL = np.zeros(N0)
    for i in range(N0):
        logL[i] = log_likelihood_func(theta[i])

    log_prior = np.array([log_prior_func(theta[i]) for i in range(N0)])

    betas = [0.0]
    stage_sizes = [N0]
    beta = 0.0
    samples_all = [theta.copy()]
    logL_all = [logL.copy()]

    def _ess(weights):
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        return 1.0 / np.sum(w*w)

    for stage in range(stages):
        low, high = beta, 1.0
        for _ in range(20):
            mid = 0.5*(low+high)
            delta_beta = mid - beta
            w = np.exp(delta_beta * (logL - np.max(logL)))
            ess = _ess(w)
            if ess < target_ess_ratio * N0:
                high = mid
            else:
                low = mid
        beta_new = low
        if beta_new <= beta + 1e-6:
            beta_new = 1.0
        delta_beta = beta_new - beta

        w_unnorm = np.exp(delta_beta * (logL - np.max(logL)))
        if not np.isfinite(w_unnorm).any() or np.sum(w_unnorm) <= 0:
            w = np.ones_like(w_unnorm) / len(w_unnorm)
        else:
            w = w_unnorm / np.sum(w_unnorm)
        ess = _ess(w)
        print(f"[TMCMC] Stage {stage+1}: beta={beta_new:.4f}, ESS={ess:.1f}/{N0}")
        idx = rng.choice(N0, size=N0, p=w)
        theta = theta[idx]
        logL = logL[idx]
        log_prior = log_prior[idx]

        cov_theta = np.cov(theta, rowvar=False) + 1e-9*np.eye(d)
        chol = np.linalg.cholesky(cov_theta)
        n_local = 10
        for i in range(N0):
            theta_i = theta[i].copy()
            logL_i = logL[i]
            logp_i = log_prior[i]
            for _ in range(n_local):
                prop = theta_i + chol @ rng.normal(size=d)
                lp_prop = log_prior_func(prop)
                if lp_prop <= -1e19:
                    continue
                ll_prop = log_likelihood_func(prop)
                logpost_old = logp_i + beta_new * logL_i
                logpost_new = lp_prop + beta_new * ll_prop
                if np.log(rng.random()) < (logpost_new - logpost_old):
                    theta_i, logL_i, logp_i = prop, ll_prop, lp_prop
            theta[i] = theta_i
            logL[i] = logL_i
            log_prior[i] = logp_i

        samples_all.append(theta.copy())
        logL_all.append(logL.copy())
        betas.append(beta_new)
        stage_sizes.append(N0)

        beta = beta_new
        if abs(beta - 1.0) < 1e-8:
            break

    return TMCMCResult(
        samples=np.vstack(samples_all),
        logL=np.concatenate(logL_all),
        betas=betas,
        stage_sizes=stage_sizes
    )

# =============================================================================
# 5. Hierarchical Case II: M1 -> M2 -> M3
# =============================================================================

@dataclass
class CaseIIResult:
    theta_true: np.ndarray
    theta_M1_mean: np.ndarray
    theta_M2_mean: np.ndarray
    theta_M3_mean: np.ndarray
    theta_final_mean: np.ndarray
    tmcmc_M1: TMCMCResult
    tmcmc_M2: TMCMCResult
    tmcmc_M3: TMCMCResult

def hierarchical_case2(tsm: BiofilmTSM,
                       data_M1: np.ndarray,
                       data_M2: np.ndarray,
                       data_M3: np.ndarray,
                       theta_prior_center: np.ndarray,
                       config: Dict[str, Any]) -> CaseIIResult:

    cov_rel = config.get("cov_rel", 0.005)
    N0_M1 = config.get("N0_M1", 50)
    N0_M2 = config.get("N0_M2", 50)
    N0_M3 = config.get("N0_M3", 50)
    stages_M1 = config.get("stages_M1", 5)
    stages_M2 = config.get("stages_M2", 5)
    stages_M3 = config.get("stages_M3", 5)
    target_ess_ratio = config.get("target_ess_ratio", 0.8)

    def prior_cov(theta_center):
        return np.diag((cov_rel * theta_center)**2 + 1e-12)

    def make_log_prior(mean, cov):
        mean = np.asarray(mean, float)
        cov = np.asarray(cov, float)
        inv_cov = np.linalg.inv(cov)
        sign, logdet_cov = np.linalg.slogdet(cov)
        if sign <= 0:
            raise RuntimeError("Prior covariance not SPD.")
        d = len(mean)
        cst = -0.5 * (d*np.log(2*np.pi) + logdet_cov)
        def _lp(theta):
            diff = theta - mean
            return float(cst - 0.5 * diff @ inv_cov @ diff)
        return _lp

    theta_true = theta_prior_center.copy()

    # M1
    def logL_M1(theta_sub):
        theta_full = theta_true.copy()
        theta_full[0:5] = theta_sub
        res = tsm.solve_tsm(theta_full)
        idx1 = [0,1]
        mu = res.mu[:, idx1]
        sigma2 = res.sigma2[:, idx1]
        return log_likelihood_eq29(mu, sigma2, data_M1)

    prior_cov_M1 = prior_cov(theta_true[0:5])
    log_prior_M1 = make_log_prior(theta_true[0:5], prior_cov_M1)
    init_M1 = theta_true[0:5].copy()
    print("\n=== Stage 1: M1 (species 1 & 2) ===")
    res_M1_tmcmc = tmcmc(logL_M1, log_prior_M1, init_M1,
                         N0=N0_M1, stages=stages_M1,
                         target_ess_ratio=target_ess_ratio,
                         random_state=123)
    theta_M1_mean = np.mean(res_M1_tmcmc.samples, axis=0)
    print("M1 posterior mean:", theta_M1_mean)

    # M2
    def logL_M2(theta_sub):
        theta_full = theta_true.copy()
        theta_full[5:10] = theta_sub
        theta_full[0:5] = theta_M1_mean
        res = tsm.solve_tsm(theta_full)
        idx2 = [2,3]
        mu = res.mu[:, idx2]
        sigma2 = res.sigma2[:, idx2]
        return log_likelihood_eq29(mu, sigma2, data_M2)

    prior_cov_M2 = prior_cov(theta_true[5:10])
    log_prior_M2 = make_log_prior(theta_true[5:10], prior_cov_M2)
    init_M2 = theta_true[5:10].copy()
    print("\n=== Stage 2: M2 (species 3 & 4) ===")
    res_M2_tmcmc = tmcmc(logL_M2, log_prior_M2, init_M2,
                         N0=N0_M2, stages=stages_M2,
                         target_ess_ratio=target_ess_ratio,
                         random_state=456)
    theta_M2_mean = np.mean(res_M2_tmcmc.samples, axis=0)
    print("M2 posterior mean:", theta_M2_mean)

    # M3
    def logL_M3(theta_sub):
        theta_full = theta_true.copy()
        theta_full[0:5] = theta_M1_mean
        theta_full[5:10] = theta_M2_mean
        theta_full[10:14] = theta_sub
        res = tsm.solve_tsm(theta_full)
        idx_all = [0,1,2,3]
        mu = res.mu[:, idx_all]
        sigma2 = res.sigma2[:, idx_all]
        return log_likelihood_eq29(mu, sigma2, data_M3)

    prior_cov_M3 = prior_cov(theta_true[10:14])
    log_prior_M3 = make_log_prior(theta_true[10:14], prior_cov_M3)
    init_M3 = theta_true[10:14].copy()
    print("\n=== Stage 3: M3 (cross interactions: all species) ===")
    res_M3_tmcmc = tmcmc(logL_M3, log_prior_M3, init_M3,
                         N0=N0_M3, stages=stages_M3,
                         target_ess_ratio=target_ess_ratio,
                         random_state=789)
    theta_M3_mean = np.mean(res_M3_tmcmc.samples, axis=0)
    print("M3 posterior mean:", theta_M3_mean)

    theta_final = theta_true.copy()
    theta_final[0:5] = theta_M1_mean
    theta_final[5:10] = theta_M2_mean
    theta_final[10:14] = theta_M3_mean

    return CaseIIResult(
        theta_true=theta_true,
        theta_M1_mean=theta_M1_mean,
        theta_M2_mean=theta_M2_mean,
        theta_M3_mean=theta_M3_mean,
        theta_final_mean=theta_final,
        tmcmc_M1=res_M1_tmcmc,
        tmcmc_M2=res_M2_tmcmc,
        tmcmc_M3=res_M3_tmcmc
    )

# =============================================================================
# 6. Synthetic data (Case II)
# =============================================================================

def generate_synthetic_data(solver: BiofilmNewtonSolver,
                            theta_true: np.ndarray,
                            noise_std=0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, x = solver.simulate(theta_true)
    phi = x[:, 0:4]

    M1_data = phi[:, 0:2]
    M2_data = phi[:, 2:4]
    M3_data = phi[:, 0:4]

    rng = np.random.default_rng(42)
    d1 = M1_data + noise_std * rng.normal(size=M1_data.shape)
    d2 = M2_data + noise_std * rng.normal(size=M2_data.shape)
    d3 = M3_data + noise_std * rng.normal(size=M3_data.shape)

    return t, d1, d2, d3

# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("="*72)
    print("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    print("="*72)
    print(f"Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode  : {'DEBUG (fast)' if DEBUG else 'FULL (Table 3-like)'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    true_theta = np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,
        1.5, 1.0, 2.0, 0.3, 0.4,
        2.0, 1.0, 2.0, 1.0
    ])

    cfg = CONFIG["M1"] if DEBUG else CONFIG["M1"]
    solver_M1 = BiofilmNewtonSolver(
        dt=cfg["dt"],
        maxtimestep=cfg["maxtimestep"],
        eps=1e-6,
        Kp1=1e-4,
        eta_vec=np.ones(4),
        c_const=cfg["c_const"],
        alpha_const=cfg["alpha_const"]
    )

    print("[Step 0] Generating synthetic data from true parameters...")
    t_data, d_M1, d_M2, d_M3 = generate_synthetic_data(solver_M1, true_theta, noise_std=0.01)
    print("  Synthetic data generated.")

    cov_rel = CONFIG.get("cov_rel", 0.005)
    tsm = BiofilmTSM(solver_M1, cov_rel=cov_rel, active_theta_indices=CONFIG.get("theta_active_indices"))

    results = hierarchical_case2(tsm, d_M1, d_M2, d_M3, true_theta, CONFIG)

    print("\n===== Hierarchical Case II Results =====")
    print("True theta:   ", true_theta)
    print("Final mean θ: ", results.theta_final_mean)
    print(f"Total runtime (approx): {time.time():.1f} s (depends on machine)")
    print("\nDone.")

if __name__ == "__main__":
    main()