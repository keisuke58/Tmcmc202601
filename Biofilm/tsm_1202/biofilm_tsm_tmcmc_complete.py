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
            "N0_M1": 15, "N0_M2": 15, "N0_M3": 15,
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
            "M3": dict(dt=1e-4, maxtimestep=750,  c_const=25.0,  alpha_const=0.0),
            "N0_M1": 50, "N0_M2": 50, "N0_M3": 50,
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

@dataclass
class TSMResult:
    t_array: np.ndarray
    mu: np.ndarray          # (n_time, 10)
    sigma2: np.ndarray      # (n_time, 10)
    x0: np.ndarray          # 0th order solution
    x1: np.ndarray          # 1st order sensitivities

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
    tsm_traces: Dict[str, List[TSMResult]] = field(default_factory=dict)

# =============================================================================
# 1. Biofilm Newton Solver (4-species) - UNCHANGED
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
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)

    def c(self, t):
        return self.c_const

    def alpha(self, t):
        return self.alpha_const

    @staticmethod
    def theta_to_matrices(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape[0] != 14:
            raise ValueError("theta must be length 14")
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

        for i in range(4):
            for j in range(4):
                K[i, j] = (c_t_value / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] = phi_p_deriv[i] + (1.0 / self.Eta_vec[i]) * (
                (self.Eta_phi_vec[i] + self.Eta_vec[i] * psi_new[i]**2) / dt +
                self.Eta_vec[i] * psi_new[i] * psidot[i]) - \
                (c_t_value / self.Eta_vec[i]) * (psi_new[i] * (Interaction_dot_product[i] + A[i, i] * psi_new[i]))
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

        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 9] = 1.0

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
# 2. TSM (1st-order Taylor) - UNCHANGED
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
    return max(ll, -1e20) if np.isfinite(ll) else -1e20


# =============================================================================
# 4. TMCMC - UNCHANGED (with tracking)
# =============================================================================

def tmcmc(log_likelihood, log_prior, theta_init_samples, n_stages=8,
          target_ess_ratio=0.8, adapt_cov=True, random_state=None) -> TMCMCResult:
    rng = np.random.default_rng(random_state)
    theta_curr = np.array(theta_init_samples, dtype=float)
    N, d = theta_curr.shape

    beta_list, samples_list, logw_list = [0.0], [theta_curr.copy()], [np.zeros(N)]
    logL_trace, acceptance_rates = [], []

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

    return TMCMCResult(samples_list, logw_list, beta_list, logL_trace, acceptance_rates)


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

    tsm_traces = {"M1": [], "M2": [], "M3": []}

    # === M1 ===
    print("\n=== Stage 1: M1 (species 1 & 2) ===")
    def logL_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        tsm_res = tsm_M1.solve_tsm(theta_full)
        tsm_traces["M1"].append(tsm_res)
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
    def logL_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        tsm_res = tsm_M2.solve_tsm(theta_full)
        tsm_traces["M2"].append(tsm_res)
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
    def logL_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        tsm_res = tsm_M3.solve_tsm(theta_full)
        tsm_traces["M3"].append(tsm_res)
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
        tsm_traces=tsm_traces
    )


# =============================================================================
# 6. DATA SAVING MODULE (only when DEBUG=False)
# =============================================================================

class DataSaver:
    """Complete data saving with timestamped folders"""

    def __init__(self, base_dir: str = "results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, f"result_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n📁 Output directory: {self.output_dir}/")

    def save_config(self, config: dict, debug: bool):
        """Save configuration as JSON"""
        config_data = {
            "timestamp": self.timestamp,
            "debug_mode": debug,
            "config": {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in config.items()}
        }
        path = os.path.join(self.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        print(f"  ✓ config.json")

    def save_synthetic_data(self, data_M1, data_M2, data_M3, t1, t2, t3):
        """Save synthetic observation data"""
        import pandas as pd
        pd.DataFrame({"time": t1, **{f"sp{i+1}": data_M1[:, i] for i in range(data_M1.shape[1])}}).to_csv(
            os.path.join(self.output_dir, "synthetic_data_M1.csv"), index=False)
        pd.DataFrame({"time": t2, **{f"sp{i+3}": data_M2[:, i] for i in range(data_M2.shape[1])}}).to_csv(
            os.path.join(self.output_dir, "synthetic_data_M2.csv"), index=False)
        pd.DataFrame({"time": t3, **{f"sp{i+1}": data_M3[:, i] for i in range(data_M3.shape[1])}}).to_csv(
            os.path.join(self.output_dir, "synthetic_data_M3.csv"), index=False)
        print(f"  ✓ synthetic_data_M1/M2/M3.csv")

    def save_true_parameters(self, theta_true: np.ndarray):
        """Save true parameters"""
        param_names = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]
        import pandas as pd
        pd.DataFrame({"parameter": param_names, "true_value": theta_true}).to_csv(
            os.path.join(self.output_dir, "true_parameters.csv"), index=False)
        print(f"  ✓ true_parameters.csv")

    def save_posterior_samples(self, results: HierarchicalResults):
        """Save all posterior samples"""
        np.savez_compressed(
            os.path.join(self.output_dir, "posterior_samples.npz"),
            M1_samples=results.M1_samples,
            M2_samples=results.M2_samples,
            M3_samples=results.M3_samples)
        print(f"  ✓ posterior_samples.npz")

        # Also CSV
        import pandas as pd
        for name, samples, params in [
            ("M1", results.M1_samples, ["a11","a12","a22","b1","b2"]),
            ("M2", results.M2_samples, ["a33","a34","a44","b3","b4"]),
            ("M3", results.M3_samples, ["a13","a14","a23","a24"])]:
            pd.DataFrame(samples, columns=params).to_csv(
                os.path.join(self.output_dir, f"posterior_samples_{name}.csv"), index=False)
        print(f"  ✓ posterior_samples_M1/M2/M3.csv")

    def save_posterior_summary(self, results: HierarchicalResults, theta_true: np.ndarray):
        """Save posterior mean, std, CI"""
        import pandas as pd
        param_names = ["a11","a12","a22","b1","b2","a33","a34","a44","b3","b4","a13","a14","a23","a24"]
        all_samples = np.hstack([results.M1_samples, results.M2_samples, results.M3_samples])

        summary = pd.DataFrame({
            "parameter": param_names,
            "true_value": theta_true,
            "posterior_mean": results.theta_final,
            "posterior_std": np.std(all_samples, axis=0),
            "CI_2.5%": np.percentile(all_samples, 2.5, axis=0),
            "CI_97.5%": np.percentile(all_samples, 97.5, axis=0),
            "error": results.theta_final - theta_true,
            "error_percent": 100 * (results.theta_final - theta_true) / (theta_true + 1e-10)
        })
        summary.to_csv(os.path.join(self.output_dir, "posterior_summary.csv"), index=False)
        print(f"  ✓ posterior_summary.csv")
        return summary

    def save_tmcmc_diagnostics(self, results: HierarchicalResults):
        """Save TMCMC beta progression and logL traces"""
        import pandas as pd
        for name, tmcmc_res in [("M1", results.tmcmc_M1), ("M2", results.tmcmc_M2), ("M3", results.tmcmc_M3)]:
            # Beta schedule
            pd.DataFrame({"stage": range(len(tmcmc_res.beta_schedule)), "beta": tmcmc_res.beta_schedule}).to_csv(
                os.path.join(self.output_dir, f"tmcmc_beta_{name}.csv"), index=False)
            # Acceptance rates
            if tmcmc_res.acceptance_rates:
                pd.DataFrame({"stage": range(1, len(tmcmc_res.acceptance_rates)+1),
                             "acceptance_rate": tmcmc_res.acceptance_rates}).to_csv(
                    os.path.join(self.output_dir, f"tmcmc_acceptance_{name}.csv"), index=False)
            # LogL trace (final stage)
            if tmcmc_res.logL_trace:
                np.save(os.path.join(self.output_dir, f"tmcmc_logL_trace_{name}.npy"),
                       np.array(tmcmc_res.logL_trace))
        print(f"  ✓ tmcmc_beta/acceptance/logL_trace_M1/M2/M3")

    def save_tsm_results(self, results: HierarchicalResults):
        """Save TSM μ(t), σ²(t), x0, x1 for last evaluation"""
        for name in ["M1", "M2", "M3"]:
            if results.tsm_traces.get(name) and len(results.tsm_traces[name]) > 0:
                tsm_res = results.tsm_traces[name][-1]
                np.savez_compressed(
                    os.path.join(self.output_dir, f"tsm_results_{name}.npz"),
                    t=tsm_res.t_array, mu=tsm_res.mu, sigma2=tsm_res.sigma2,
                    x0=tsm_res.x0, x1=tsm_res.x1)
        print(f"  ✓ tsm_results_M1/M2/M3.npz")

    def save_complete_archive(self, results: HierarchicalResults, theta_true: np.ndarray,
                              data_M1, data_M2, data_M3, config: dict):
        """Save everything in one NPZ"""
        np.savez_compressed(
            os.path.join(self.output_dir, "complete_archive.npz"),
            theta_true=theta_true, theta_final=results.theta_final,
            theta_M1_mean=results.theta_M1_mean, theta_M2_mean=results.theta_M2_mean,
            theta_M3_mean=results.theta_M3_mean,
            M1_samples=results.M1_samples, M2_samples=results.M2_samples, M3_samples=results.M3_samples,
            data_M1=data_M1, data_M2=data_M2, data_M3=data_M3,
            beta_M1=np.array(results.tmcmc_M1.beta_schedule),
            beta_M2=np.array(results.tmcmc_M2.beta_schedule),
            beta_M3=np.array(results.tmcmc_M3.beta_schedule))
        print(f"  ✓ complete_archive.npz")

    def list_saved_files(self):
        """List all saved files with sizes"""
        print("\n" + "="*60)
        print("  SAVED FILES")
        print("="*60)
        total = 0
        for f in sorted(os.listdir(self.output_dir)):
            size = os.path.getsize(os.path.join(self.output_dir, f))
            total += size
            size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
            print(f"  • {f:<40} ({size_str})")
        print(f"\n  Total: {total/1024:.1f} KB")
        print("="*60)


# =============================================================================
# 7. VISUALIZATION MODULE
# =============================================================================

def create_visualizations(output_dir: str, results: HierarchicalResults,
                         theta_true: np.ndarray, data_M1, data_M2, data_M3,
                         t1, t2, t3):
    """Create all visualization plots"""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr
    except ImportError:
        print("  ⚠ matplotlib/scipy not available, skipping plots")
        return

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120, 'figure.facecolor': 'white'})
    param_names_M1 = ['a11', 'a12', 'a22', 'b1', 'b2']
    param_names_M2 = ['a33', 'a34', 'a44', 'b3', 'b4']
    param_names_M3 = ['a13', 'a14', 'a23', 'a24']

    def corner_plot(samples, true_vals, param_names, title, filename):
        n = len(param_names)
        fig, axes = plt.subplots(n, n, figsize=(12, 12))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples[:, i], bins=30, color='#4472C4', alpha=0.7, density=True)
                    ax.axvline(true_vals[i], color='red', linestyle='--', linewidth=2, label='True')
                    ax.axvline(np.mean(samples[:, i]), color='green', linestyle='-', linewidth=1.5)
                elif i > j:
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.1, s=1, c='#4472C4')
                    ax.axvline(true_vals[j], color='red', linestyle='--', alpha=0.5)
                    ax.axhline(true_vals[i], color='red', linestyle='--', alpha=0.5)
                else:
                    corr, _ = pearsonr(samples[:, i], samples[:, j])
                    ax.text(0.5, 0.5, f'ρ={corr:.2f}', ha='center', va='center', fontsize=9)
                    ax.set_facecolor('#f0f0f0' if abs(corr) < 0.3 else '#ffcccc' if corr > 0 else '#ccccff')
                if i == n-1:
                    ax.set_xlabel(param_names[j])
                if j == 0 and i > 0:
                    ax.set_ylabel(param_names[i])
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    # Corner plots
    corner_plot(results.M1_samples, theta_true[0:5], param_names_M1, "M1 Posterior", "fig_M1_corner.png")
    corner_plot(results.M2_samples, theta_true[5:10], param_names_M2, "M2 Posterior", "fig_M2_corner.png")
    corner_plot(results.M3_samples, theta_true[10:14], param_names_M3, "M3 Posterior", "fig_M3_corner.png")
    print(f"  ✓ Corner plots: fig_M1/M2/M3_corner.png")

    # Beta progression
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, tmcmc_res) in zip(axes, [("M1", results.tmcmc_M1), ("M2", results.tmcmc_M2), ("M3", results.tmcmc_M3)]):
        ax.plot(tmcmc_res.beta_schedule, 'o-', markersize=8, linewidth=2)
        ax.set_xlabel("Stage")
        ax.set_ylabel("β")
        ax.set_title(f"{name} TMCMC β Progression")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_beta_progression.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig_beta_progression.png")

    # Parameter comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    all_names = param_names_M1 + param_names_M2 + param_names_M3
    x = np.arange(14)
    width = 0.35
    ax.bar(x - width/2, theta_true, width, label='True', color='#FF8C00', alpha=0.8)
    ax.bar(x + width/2, results.theta_final, width, label='Estimated', color='#4472C4', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names)
    ax.set_ylabel('Value')
    ax.set_title('Parameter Estimation Results')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_parameter_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig_parameter_comparison.png")


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

def main():
    print("="*72)
    print("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    print("="*72)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode : {'DEBUG (fast)' if DEBUG else 'FULL (Table 3)'}")
    print(f"Save : {'OFF' if DEBUG else 'ON (timestamped folder)'}")
    print(f"Plots: {'ON' if ENABLE_PLOTS and not DEBUG else 'OFF'}")
    print("="*72)

    # --- Build solvers ---
    solver_M1 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], **CONFIG["M1"])
    solver_M2 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], **CONFIG["M2"])
    solver_M3 = BiofilmNewtonSolver(eta_vec=[1,1,1,1], **CONFIG["M3"])

    tsm_M1 = BiofilmTSM(solver_M1, cov_rel=CONFIG["cov_rel"], active_theta_indices=CONFIG["theta_active_indices_M1"])
    tsm_M2 = BiofilmTSM(solver_M2, cov_rel=CONFIG["cov_rel"], active_theta_indices=CONFIG["theta_active_indices_M2"])
    tsm_M3 = BiofilmTSM(solver_M3, cov_rel=CONFIG["cov_rel"], active_theta_indices=CONFIG["theta_active_indices_M3"])

    # --- True parameters ---
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

    # --- Save data (only if not DEBUG) ---
    if not DEBUG:
        print("\n" + "="*72)
        print("  SAVING ALL DATA")
        print("="*72)

        saver = DataSaver(base_dir="results")
        saver.save_config(CONFIG, DEBUG)
        saver.save_true_parameters(theta_true)
        saver.save_synthetic_data(data_M1, data_M2, data_M3, t1, t2, t3)
        saver.save_posterior_samples(results)
        summary = saver.save_posterior_summary(results, theta_true)
        saver.save_tmcmc_diagnostics(results)
        saver.save_tsm_results(results)
        saver.save_complete_archive(results, theta_true, data_M1, data_M2, data_M3, CONFIG)

        if ENABLE_PLOTS:
            print("\n[Visualization]")
            create_visualizations(saver.output_dir, results, theta_true,
                                 data_M1, data_M2, data_M3, t1, t2, t3)

        saver.list_saved_files()

        print(f"\n  📁 All data saved to: {saver.output_dir}/")
        print("  To reload: data = np.load('complete_archive.npz')")
    else:
        print("\n  ℹ️  DEBUG mode: data saving skipped")
        print("     Set DEBUG = False to save all results")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
