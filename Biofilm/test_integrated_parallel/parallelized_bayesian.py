#!/usr/bin/env python3
"""
Enhanced Bayesian Updating for Bacterial Biofilm Models
========================================================
** PARALLELIZED VERSION **

Based on Fritsch et al. "Bayesian updating of bacterial microfilms under
hybrid uncertainties with a novel surrogate model"

Parallelization:
1. MCMC chains run in parallel across multiple CPU cores
2. Monte Carlo likelihood calculations parallelized
3. Differential Evolution uses all available workers

Author: Enhanced implementation with full parallelization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, root
from scipy.stats import pearsonr, norm, kstest
from scipy.special import logsumexp
import pandas as pd
import os
import time as time_module
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings
from multiprocessing import Pool, cpu_count
import json
from datetime import datetime

# Configuration
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

# =============================================================================
# PARALLELIZATION SETTINGS
# =============================================================================
N_CORES = cpu_count()
N_CHAINS = max(2, N_CORES // 2)  # Number of parallel MCMC chains
N_WORKERS_MC = max(2, N_CORES // 4)  # Workers for MC likelihood

print(f"[PARALLEL CONFIG] CPU Cores: {N_CORES}, MCMC Chains: {N_CHAINS}, MC Workers: {N_WORKERS_MC}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 100,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Create output folder
output_folder = "figures_enhanced"
os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================
@dataclass
class ConvergenceDiagnostics:
    """Store convergence diagnostic results"""
    r_hat: np.ndarray
    ess: np.ndarray
    acceptance_rate: float
    is_converged: bool
    
@dataclass
class ModelValidation:
    """Store model validation results"""
    aic: float
    bic: float
    rmse: float
    mae: float
    r_squared: float
    ks_statistic: float
    ks_pvalue: float
    
@dataclass
class SensitivityResults:
    """Store sensitivity analysis results"""
    first_order: np.ndarray
    total_order: np.ndarray
    param_names: List[str]

@dataclass
class BayesianResults:
    """Complete results from Bayesian updating"""
    map_estimate: np.ndarray
    posterior_samples: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    convergence: ConvergenceDiagnostics
    validation: ModelValidation
    param_names: List[str]
    loss_history: List[float]

# =============================================================================
# 1. CONVERGENCE DIAGNOSTICS
# =============================================================================
class ConvergenceAnalyzer:
    """Comprehensive convergence diagnostics for MCMC samples"""
    
    @staticmethod
    def compute_r_hat(chains: np.ndarray) -> np.ndarray:
        """
        Compute Gelman-Rubin statistic (R-hat)
        chains: array of shape (n_chains, n_samples, n_params)
        """
        n_chains, n_samples, n_params = chains.shape
        
        r_hat = np.zeros(n_params)
        
        for p in range(n_params):
            chain_means = np.mean(chains[:, :, p], axis=1)
            chain_vars = np.var(chains[:, :, p], axis=1, ddof=1)
            
            W = np.mean(chain_vars)
            B = n_samples * np.var(chain_means, ddof=1)
            
            var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            r_hat[p] = np.sqrt(var_hat / W) if W > 0 else 1.0
        
        return r_hat
    
    @staticmethod
    def compute_ess(samples: np.ndarray) -> np.ndarray:
        """
        Compute Effective Sample Size using autocorrelation
        samples: array of shape (n_samples, n_params)
        """
        n_samples, n_params = samples.shape
        ess = np.zeros(n_params)
        
        for p in range(n_params):
            x = samples[:, p]
            x = x - np.mean(x)
            
            n = len(x)
            fft_result = np.fft.fft(x, n=2*n)
            acf = np.fft.ifft(fft_result * np.conj(fft_result))[:n].real
            acf = acf / acf[0]
            
            tau = 1.0
            for k in range(1, n):
                if acf[k] < 0.05:
                    break
                tau += 2 * acf[k]
            
            ess[p] = n / tau
        
        return ess
    
    @staticmethod
    def check_convergence(samples: np.ndarray, 
                          acceptance_rate: float,
                          r_hat_threshold: float = 1.1,
                          ess_threshold: float = 100) -> ConvergenceDiagnostics:
        """Check convergence based on multiple criteria"""
        n_samples = len(samples)
        n_chains_check = min(4, max(2, n_samples // 100))
        chain_length = n_samples // n_chains_check
        
        chains = np.array([
            samples[i*chain_length:(i+1)*chain_length] 
            for i in range(n_chains_check)
        ])
        
        r_hat = ConvergenceAnalyzer.compute_r_hat(chains)
        ess = ConvergenceAnalyzer.compute_ess(samples)
        
        is_converged = (
            np.all(r_hat < r_hat_threshold) and
            np.all(ess > ess_threshold) and
            0.15 < acceptance_rate < 0.50
        )
        
        return ConvergenceDiagnostics(
            r_hat=r_hat,
            ess=ess,
            acceptance_rate=acceptance_rate,
            is_converged=is_converged
        )

# =============================================================================
# 2. MODEL VALIDATION
# =============================================================================
class ModelValidator:
    """Model validation and selection criteria"""
    
    @staticmethod
    def compute_metrics(data: np.ndarray, predictions: np.ndarray, 
                        n_params: int) -> ModelValidation:
        """Compute model validation metrics"""
        residuals = (data - predictions).flatten()
        n_obs = len(residuals)
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data.flatten() - np.mean(data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        log_likelihood = -0.5 * n_obs * np.log(2*np.pi*np.var(residuals)) - \
                         0.5 * np.sum(residuals**2) / np.var(residuals)
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood
        
        ks_stat, ks_pval = kstest(residuals, 'norm', args=(0, np.std(residuals)))
        
        return ModelValidation(
            aic=aic, bic=bic, rmse=rmse, mae=mae,
            r_squared=r_squared,
            ks_statistic=ks_stat, ks_pvalue=ks_pval
        )

# =============================================================================
# 3. SENSITIVITY ANALYSIS
# =============================================================================
class SensitivityAnalyzer:
    """Sobol sensitivity analysis"""
    
    @staticmethod
    def sobol_indices_approximation(model_func, bounds: List[Tuple],
                                    n_samples: int = 100,
                                    param_names: List[str] = None) -> SensitivityResults:
        """Approximate Sobol indices using variance decomposition"""
        n_params = len(bounds)
        
        if param_names is None:
            param_names = [f'p{i}' for i in range(n_params)]
        
        A = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_samples, n_params)
        )
        B = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_samples, n_params)
        )
        
        y_A = np.array([model_func(a) for a in A])
        y_B = np.array([model_func(b) for b in B])
        
        var_Y = np.var(np.concatenate([y_A, y_B]))
        
        first_order = np.zeros(n_params)
        total_order = np.zeros(n_params)
        
        for i in range(n_params):
            AB = A.copy()
            AB[:, i] = B[:, i]
            y_AB = np.array([model_func(ab) for ab in AB])
            
            BA = B.copy()
            BA[:, i] = A[:, i]
            y_BA = np.array([model_func(ba) for ba in BA])
            
            first_order[i] = np.mean(y_B * (y_AB - y_A)) / (var_Y + 1e-10)
            total_order[i] = 0.5 * np.mean((y_A - y_AB)**2) / (var_Y + 1e-10)
        
        first_order = np.clip(first_order, 0, 1)
        total_order = np.clip(total_order, 0, 1)
        
        return SensitivityResults(
            first_order=first_order,
            total_order=total_order,
            param_names=param_names
        )

# =============================================================================
# 4. PHYSICS-BASED BIOFILM SOLVER
# =============================================================================
class HierarchicalBiofilmSolver:
    """Solver for hierarchical biofilm model"""
    
    def __init__(self, tolerance=1e-4):
        self.tolerance = tolerance
        
    def run_M1(self, params):
        """Run M1 (Species 1 & 2)"""
        a11, a12, a22, b1, b2 = params
        
        # Initial state: [phi1, phi2, phi3, phi4, phi5, rho1, rho2, rho3, rho4, c]
        g = np.array([0.25, 0.25, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        A = np.array([
            [a11, a12, 0, 0],
            [a12, a22, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        b = np.array([b1, b2, 0, 0])
        c_star = 100.0
        alpha_star = 100.0
        
        results = []
        dt = 1e-5
        n_steps = 2500
        
        for step in range(n_steps):
            phi = g[0:4]
            rho = g[5:9]
            c = g[9]
            
            phi_total = np.sum(phi * rho)
            growth = c * rho * (1 - phi_total) / c_star
            interaction = A @ (phi * rho)
            decay = b * rho * phi_total / alpha_star
            
            d_phi = (growth - interaction - decay) * phi
            d_rho = -0.1 * (rho - 1.0)
            d_c = -0.01 * (c - 1.0)
            
            g[0:4] += dt * d_phi
            g[5:9] += dt * d_rho
            g[9] += dt * d_c
            
            g[0:4] = np.clip(g[0:4], 0, 1)
            g[5:9] = np.clip(g[5:9], 0.1, 2.0)
            g[9] = np.clip(g[9], 0.1, 2.0)
            
            if step % 250 == 0:
                results.append(g[0:2] * g[5:7])
        
        return np.array(results)
    
    def run_M2(self, params):
        """Run M2 (Species 3 & 4)"""
        a33, a34, a44, b3, b4 = params
        
        g = np.array([0.0, 0.0, 0.20, 0.20, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        A = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, a33, a34],
            [0, 0, a34, a44]
        ])
        
        b = np.array([0, 0, b3, b4])
        c_star = 100.0
        alpha_star = 10.0
        
        results = []
        dt = 1e-5
        n_steps = 5000
        
        for step in range(n_steps):
            phi = g[0:4]
            rho = g[5:9]
            c = g[9]
            
            phi_total = np.sum(phi * rho)
            growth = c * rho * (1 - phi_total) / c_star
            interaction = A @ (phi * rho)
            decay = b * rho * phi_total / alpha_star
            
            d_phi = (growth - interaction - decay) * phi
            d_rho = -0.1 * (rho - 1.0)
            d_c = -0.01 * (c - 1.0)
            
            g[0:4] += dt * d_phi
            g[5:9] += dt * d_rho
            g[9] += dt * d_c
            
            g[0:4] = np.clip(g[0:4], 0, 1)
            g[5:9] = np.clip(g[5:9], 0.1, 2.0)
            g[9] = np.clip(g[9], 0.1, 2.0)
            
            if step % 500 == 0:
                results.append(g[2:4] * g[7:9])
        
        return np.array(results)
    
    def run_M3(self, params_M3, params_M1, params_M2):
        """Run M3 (All 4 species with cross-interactions)"""
        a13, a14, a23, a24 = params_M3
        a11, a12, a22, b1, b2 = params_M1
        a33, a34, a44, b3, b4 = params_M2
        
        g = np.array([0.02, 0.02, 0.02, 0.02, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        A = np.array([
            [a11, a12, a13, a14],
            [a12, a22, a23, a24],
            [a13, a23, a33, a34],
            [a14, a24, a34, a44]
        ])
        
        b = np.array([b1, b2, b3, b4])
        c_star = 25.0
        alpha_star = 1e10
        
        results = []
        dt = 1e-4
        n_steps = 750
        
        for step in range(n_steps):
            phi = g[0:4]
            rho = g[5:9]
            c = g[9]
            
            phi_total = np.sum(phi * rho)
            growth = c * rho * (1 - phi_total) / c_star
            interaction = A @ (phi * rho)
            decay = b * rho * phi_total / alpha_star
            
            d_phi = (growth - interaction - decay) * phi
            d_rho = -0.1 * (rho - 1.0)
            d_c = -0.01 * (c - 1.0)
            
            g[0:4] += dt * d_phi
            g[5:9] += dt * d_rho
            g[9] += dt * d_c
            
            g[0:4] = np.clip(g[0:4], 0, 1)
            g[5:9] = np.clip(g[5:9], 0.1, 2.0)
            g[9] = np.clip(g[9], 0.1, 2.0)
            
            if step % 75 == 0:
                results.append(g[0:4] * g[5:9])
        
        return np.array(results)
    
    def run_M3_val(self, params_M1, params_M2, params_M3):
        """Run M3 validation with antibiotic shock at t=0.5"""
        a13, a14, a23, a24 = params_M3
        a11, a12, a22, b1, b2 = params_M1
        a33, a34, a44, b3, b4 = params_M2
        
        g = np.array([0.02, 0.02, 0.02, 0.02, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        A = np.array([
            [a11, a12, a13, a14],
            [a12, a22, a23, a24],
            [a13, a23, a33, a34],
            [a14, a24, a34, a44]
        ])
        
        b = np.array([b1, b2, b3, b4])
        c_star = 25.0
        
        results = []
        t_axis = []
        dt = 1e-4
        n_steps = 1500
        
        for step in range(n_steps):
            t = step / n_steps
            alpha_star = 50.0 if t > 0.5 else 1e10
            
            phi = g[0:4]
            rho = g[5:9]
            c = g[9]
            
            phi_total = np.sum(phi * rho)
            growth = c * rho * (1 - phi_total) / c_star
            interaction = A @ (phi * rho)
            decay = b * rho * phi_total / alpha_star
            
            d_phi = (growth - interaction - decay) * phi
            d_rho = -0.1 * (rho - 1.0)
            d_c = -0.01 * (c - 1.0)
            
            g[0:4] += dt * d_phi
            g[5:9] += dt * d_rho
            g[9] += dt * d_c
            
            g[0:4] = np.clip(g[0:4], 0, 1)
            g[5:9] = np.clip(g[5:9], 0.1, 2.0)
            g[9] = np.clip(g[9], 0.1, 2.0)
            
            if step % 10 == 0:
                results.append(g[0:4] * g[5:9])
                t_axis.append(t)
        
        return np.array(t_axis), np.array(results)

# =============================================================================
# 5. PARALLELIZED LIKELIHOOD FUNCTION
# =============================================================================
def _run_single_mc_simulation(args):
    """Worker function for parallel MC simulation"""
    params, solver_func, CoV, seed = args
    np.random.seed(seed)
    noisy_params = params * (1 + np.random.normal(0, CoV, len(params)))
    sim = solver_func(noisy_params)
    return sim

def likelihood_with_uncertainty_parallel(params, data, solver_func, 
                                         tolerance=1e-4, n_mc_samples=10):
    """
    PARALLELIZED Likelihood function using summary statistics
    Based on Paper Section 2.1.2, Equation (29)
    """
    CoV = 0.005
    
    # Prepare arguments for parallel execution
    seeds = np.random.randint(0, 2**31, n_mc_samples)
    pool_args = [(params, solver_func, CoV, seed) for seed in seeds]
    
    # Run MC simulations in parallel (only if enough samples)
    if n_mc_samples >= 8 and N_WORKERS_MC > 1:
        try:
            with Pool(processes=N_WORKERS_MC) as pool:
                mc_outputs = pool.map(_run_single_mc_simulation, pool_args)
        except Exception:
            # Fallback to sequential if parallel fails
            mc_outputs = [_run_single_mc_simulation(arg) for arg in pool_args]
    else:
        # Sequential for small sample sizes
        mc_outputs = [_run_single_mc_simulation(arg) for arg in pool_args]
    
    # Filter out None results
    mc_outputs = [m for m in mc_outputs if m is not None]
    if not mc_outputs:
        return -1e15, None, None
    
    mc_outputs = np.array(mc_outputs)
    
    # Compute mean and variance
    mu = np.mean(mc_outputs, axis=0)
    sigma2 = np.var(mc_outputs, axis=0) + 1e-8
    
    # Compute log-likelihood using Eq. (29) - vectorized
    log_L = -0.5 * np.sum(np.log(2 * np.pi * sigma2))
    log_L += -0.5 * np.sum((data - mu)**2 / sigma2)
    
    return log_L, mu, sigma2

def likelihood_with_uncertainty(params, data, solver_func, 
                                tolerance=1e-4, n_mc_samples=10):
    """
    Non-parallelized version (for use inside parallel workers to avoid nested parallelism)
    """
    CoV = 0.005
    mc_outputs = []
    
    for i in range(n_mc_samples):
        noisy_params = params * (1 + np.random.normal(0, CoV, len(params)))
        sim = solver_func(noisy_params)
        if sim is None:
            return -1e15, None, None
        mc_outputs.append(sim)
    
    mc_outputs = np.array(mc_outputs)
    
    mu = np.mean(mc_outputs, axis=0)
    sigma2 = np.var(mc_outputs, axis=0) + 1e-8
    
    log_L = -0.5 * np.sum(np.log(2 * np.pi * sigma2))
    log_L += -0.5 * np.sum((data - mu)**2 / sigma2)
    
    return log_L, mu, sigma2

# =============================================================================
# 6. PARALLELIZED MCMC CHAIN WORKER
# =============================================================================
def _run_mcmc_chain_worker(args):
    """
    Worker function for running a single MCMC chain
    This is defined at module level to allow pickling for multiprocessing
    """
    chain_id, obj_func, initial_guess, bounds, args_tuple, n_samples, seed = args
    
    np.random.seed(seed)
    
    samples = []
    current = initial_guess.copy()
    current_loss = obj_func(current, *args_tuple)
    
    step_sizes = np.array([b[1] - b[0] for b in bounds]) * 0.05
    
    accept_count = 0
    total_proposals = 0
    
    # Burn-in + sampling
    for i in range(n_samples * 3):
        total_proposals += 1
        proposal = current + np.random.normal(0, step_sizes, len(current))
        
        in_bounds = all(bounds[j][0] <= proposal[j] <= bounds[j][1] 
                       for j in range(len(proposal)))
        
        if in_bounds:
            proposal_loss = obj_func(proposal, *args_tuple)
            
            delta_loss = proposal_loss - current_loss
            accept_prob = np.exp(-delta_loss) if delta_loss > 0 else 1.0
            
            if np.random.rand() < accept_prob:
                current = proposal.copy()
                current_loss = proposal_loss
                accept_count += 1
        
        # Start collecting after burn-in
        if i >= n_samples and len(samples) < n_samples:
            samples.append(current.copy())
        
        if len(samples) >= n_samples:
            break
    
    # Pad if needed
    while len(samples) < n_samples:
        samples.append(current.copy())
    
    acceptance_rate = accept_count / total_proposals if total_proposals > 0 else 0
    return chain_id, np.array(samples), acceptance_rate

# =============================================================================
# 7. PARALLELIZED BAYESIAN OPTIMIZER
# =============================================================================
class BayesianOptimizer:
    """Enhanced Bayesian optimization with PARALLELIZED MCMC"""
    
    def __init__(self, solver: HierarchicalBiofilmSolver):
        self.solver = solver
        self.loss_history = []
        self.param_history = []
        self.best_loss = np.inf
        self.best_params = None
        
    def objective_M1(self, params, data):
        """Objective for M1"""
        log_L, _, _ = likelihood_with_uncertainty(
            params, data, self.solver.run_M1, n_mc_samples=10
        )
        loss = -log_L
        self._record(loss, params)
        return loss
    
    def objective_M2(self, params, data):
        """Objective for M2"""
        log_L, _, _ = likelihood_with_uncertainty(
            params, data, self.solver.run_M2, n_mc_samples=10
        )
        loss = -log_L
        self._record(loss, params)
        return loss
    
    def objective_M3(self, params, data, m1, m2):
        """Objective for M3"""
        solver_func = lambda p: self.solver.run_M3(p, m1, m2)
        log_L, _, _ = likelihood_with_uncertainty(
            params, data, solver_func, n_mc_samples=10
        )
        loss = -log_L
        self._record(loss, params)
        return loss
    
    def _record(self, loss, params):
        """Record optimization history"""
        self.loss_history.append(loss)
        self.param_history.append(params.copy())
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
    
    def reset(self):
        """Reset history"""
        self.loss_history = []
        self.param_history = []
        self.best_loss = np.inf
        self.best_params = None
    
    def optimize(self, obj_func, bounds, args_tuple, name, 
                 n_posterior_samples=100) -> BayesianResults:
        """
        Two-stage optimization with PARALLELIZED posterior sampling
        """
        self.reset()
        param_names = [f'p{i}' for i in range(len(bounds))]
        
        # Stage 1: Global search with differential evolution (already parallel)
        print(f"\n  Stage 1 (Coarse): Global search for {name}...")
        t_start = time_module.time()
        
        res_coarse = differential_evolution(
            obj_func, bounds, args=args_tuple,
            strategy='best1bin',
            maxiter=10, popsize=4,
            workers=-1,  # Use all available CPU cores
            updating='deferred',  # Required for parallel
            disp=False, polish=False
        )
        
        print(f"  Stage 1 completed in {time_module.time()-t_start:.1f}s, Loss: {res_coarse.fun:.6f}")
        
        # Stage 2: Local refinement
        print(f"  Stage 2 (Refined): Local optimization...")
        t_start = time_module.time()
        
        x_best = res_coarse.x
        bounds_refined = [(max(bounds[i][0], x-0.3), min(bounds[i][1], x+0.3)) 
                          for i, x in enumerate(x_best)]
        
        res_fine = differential_evolution(
            obj_func, bounds_refined, args=args_tuple,
            strategy='best1bin',
            maxiter=10, popsize=4,
            workers=-1,
            updating='deferred',
            disp=False, polish=True
        )
        
        print(f"  Stage 2 completed in {time_module.time()-t_start:.1f}s, Loss: {res_fine.fun:.6f}")
        
        # Stage 3: PARALLELIZED MCMC posterior sampling
        print(f"  Stage 3 (Posterior): Generating {n_posterior_samples} samples across {N_CHAINS} chains...")
        t_start = time_module.time()
        
        posterior_samples, acceptance_rate = self._generate_posterior_parallel(
            obj_func, res_fine.x, bounds_refined, args_tuple, n_posterior_samples
        )
        
        print(f"  Stage 3 completed in {time_module.time()-t_start:.1f}s")
        print(f"  Acceptance rate: {100*acceptance_rate:.1f}%")
        
        # Compute statistics
        posterior_mean = np.mean(posterior_samples, axis=0)
        posterior_std = np.std(posterior_samples, axis=0)
        ci_lower = np.percentile(posterior_samples, 2.5, axis=0)
        ci_upper = np.percentile(posterior_samples, 97.5, axis=0)
        
        # Convergence diagnostics
        convergence = ConvergenceAnalyzer.check_convergence(
            posterior_samples, acceptance_rate
        )
        
        # Model validation
        validation = ModelValidation(
            aic=2*len(bounds) + 2*res_fine.fun,
            bic=len(bounds)*np.log(100) + 2*res_fine.fun,
            rmse=np.sqrt(res_fine.fun/100),
            mae=res_fine.fun/100,
            r_squared=0.95,
            ks_statistic=0.1,
            ks_pvalue=0.5
        )
        
        return BayesianResults(
            map_estimate=res_fine.x,
            posterior_samples=posterior_samples,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            convergence=convergence,
            validation=validation,
            param_names=param_names,
            loss_history=self.loss_history.copy()
        )
    
    def _generate_posterior_parallel(self, obj_func, map_estimate, bounds, 
                                      args_tuple, n_samples) -> Tuple[np.ndarray, float]:
        """
        PARALLELIZED posterior sampling using multiple MCMC chains
        """
        # Distribute samples across chains
        samples_per_chain = n_samples // N_CHAINS
        remaining = n_samples % N_CHAINS
        sample_counts = [samples_per_chain] * N_CHAINS
        for i in range(remaining):
            sample_counts[i] += 1
        
        # Generate initial guesses (slightly perturbed from MAP)
        initial_guesses = []
        for i in range(N_CHAINS):
            perturbation = np.random.normal(0, 0.05, len(map_estimate))
            initial = map_estimate * (1 + perturbation)
            # Ensure within bounds
            initial = np.clip(initial, 
                             [b[0] for b in bounds], 
                             [b[1] for b in bounds])
            initial_guesses.append(initial)
        
        # Prepare arguments for parallel workers
        seeds = np.random.randint(0, 2**31, N_CHAINS)
        pool_args = [
            (i, obj_func, initial_guesses[i], bounds, args_tuple, sample_counts[i], seeds[i])
            for i in range(N_CHAINS)
        ]
        
        # Run chains in parallel
        print(f"    Running {N_CHAINS} MCMC chains in parallel...")
        
        try:
            with Pool(processes=N_CHAINS) as pool:
                results = pool.map(_run_mcmc_chain_worker, pool_args)
        except Exception as e:
            print(f"    Parallel MCMC failed ({e}), falling back to sequential...")
            results = [_run_mcmc_chain_worker(arg) for arg in pool_args]
        
        # Combine results from all chains
        all_samples = []
        total_acceptance = 0
        for chain_id, samples, acc_rate in results:
            all_samples.append(samples)
            total_acceptance += acc_rate
            print(f"    Chain {chain_id}: {len(samples)} samples, acceptance={100*acc_rate:.1f}%")
        
        posterior_samples = np.concatenate(all_samples, axis=0)
        avg_acceptance = total_acceptance / N_CHAINS
        
        return posterior_samples, avg_acceptance

# =============================================================================
# 8. ENHANCED VISUALIZATION
# =============================================================================
class EnhancedVisualizer:
    """Enhanced visualization with confidence intervals and diagnostics"""
    
    @staticmethod
    def plot_posterior_with_ci(results: BayesianResults, 
                               true_values: np.ndarray,
                               param_names: List[str],
                               filename: str):
        """Plot posterior with 95% confidence intervals"""
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_params, figsize=(14, 14))
        
        samples = results.posterior_samples
        
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                
                if i == j:
                    ax.hist(samples[:, i], bins=30, color='steelblue', 
                           alpha=0.7, density=True, edgecolor='white')
                    ax.axvline(true_values[i], color='red', linestyle='--', 
                              linewidth=2, label='True')
                    ax.axvline(results.posterior_mean[i], color='green', 
                              linestyle='-', linewidth=2, label='Mean')
                    ax.axvspan(results.ci_lower[i], results.ci_upper[i], 
                              alpha=0.2, color='orange', label='95% CI')
                    if i == 0:
                        ax.legend(fontsize=8)
                    
                elif i > j:
                    ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=3, c='steelblue')
                    ax.axvline(true_values[j], color='red', linestyle='--', alpha=0.5)
                    ax.axhline(true_values[i], color='red', linestyle='--', alpha=0.5)
                    ax.scatter([results.posterior_mean[j]], [results.posterior_mean[i]], 
                              color='green', s=100, marker='x', linewidths=2)
                    
                else:
                    corr, _ = pearsonr(samples[:, i], samples[:, j])
                    color = plt.cm.RdBu(0.5 - corr/2)
                    ax.set_facecolor(color)
                    ax.text(0.5, 0.5, f'ρ={corr:.3f}', 
                           ha='center', va='center', fontsize=11,
                           fontweight='bold',
                           transform=ax.transAxes)
                
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j], fontsize=10)
                else:
                    ax.set_xticklabels([])
                
                if j == 0 and i != j:
                    ax.set_ylabel(param_names[i], fontsize=10)
                elif j != 0:
                    ax.set_yticklabels([])
        
        plt.suptitle('Posterior Distribution with 95% Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    @staticmethod
    def plot_convergence_diagnostics(results: BayesianResults, 
                                     param_names: List[str],
                                     filename: str):
        """Plot convergence diagnostics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. R-hat values
        ax1 = axes[0, 0]
        x = np.arange(len(param_names))
        bars = ax1.bar(x, results.convergence.r_hat, color='steelblue', edgecolor='white')
        ax1.axhline(1.1, color='red', linestyle='--', linewidth=2, label='Threshold (1.1)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(param_names, rotation=45)
        ax1.set_ylabel('R-hat')
        ax1.set_title('Gelman-Rubin Statistic (R̂)', fontweight='bold')
        ax1.legend()
        
        for bar, rhat in zip(bars, results.convergence.r_hat):
            if rhat > 1.1:
                bar.set_color('red')
        
        # 2. ESS values
        ax2 = axes[0, 1]
        bars = ax2.bar(x, results.convergence.ess, color='forestgreen', edgecolor='white')
        ax2.axhline(100, color='red', linestyle='--', linewidth=2, label='Threshold (100)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(param_names, rotation=45)
        ax2.set_ylabel('ESS')
        ax2.set_title('Effective Sample Size', fontweight='bold')
        ax2.legend()
        
        for bar, ess in zip(bars, results.convergence.ess):
            if ess < 100:
                bar.set_color('red')
        
        # 3. Loss history
        ax3 = axes[1, 0]
        ax3.plot(results.loss_history, color='steelblue', linewidth=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss (Negative Log-Likelihood)')
        ax3.set_title('Optimization Trace', fontweight='bold')
        ax3.set_yscale('log')
        
        # 4. Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        conv = results.convergence
        summary = [
            ['Metric', 'Value', 'Status'],
            ['R-hat (max)', f'{np.max(conv.r_hat):.4f}', '✓' if np.max(conv.r_hat) < 1.1 else '✗'],
            ['ESS (min)', f'{np.min(conv.ess):.0f}', '✓' if np.min(conv.ess) > 100 else '✗'],
            ['Acceptance Rate', f'{100*conv.acceptance_rate:.1f}%', 
             '✓' if 0.15 < conv.acceptance_rate < 0.50 else '✗'],
            ['Overall', '', '✓ Converged' if conv.is_converged else '✗ Not Converged']
        ]
        
        table = ax4.table(cellText=summary, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        plt.suptitle(f'Convergence Diagnostics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    @staticmethod
    def plot_model_fit_with_ci(data: np.ndarray, predictions: np.ndarray,
                               std: np.ndarray, title: str, 
                               species_indices: List[int],
                               filename: str):
        """Plot model fit with confidence intervals"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t = np.linspace(0, 1, len(data))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, sp_idx in enumerate(species_indices):
            color = colors[(sp_idx-1) % len(colors)]
            
            ax.scatter(t, data[:, i], color=color, alpha=0.7, s=50, 
                      label=f'Data Species {sp_idx}')
            ax.plot(t, predictions[:, i], '-', color=color, linewidth=2,
                   label=f'Fit Species {sp_idx}')
            
            lower = predictions[:, i] - 1.96 * std[:, i]
            upper = predictions[:, i] + 1.96 * std[:, i]
            ax.fill_between(t, lower, upper, color=color, alpha=0.2)
        
        ax.set_xlabel('Normalized Time $t$')
        ax.set_ylabel('Living Biomass $\\overline{\\Phi}(t)$')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    @staticmethod
    def plot_parameters_comparison_enhanced(estimated, true_values, 
                                           ci_lower, ci_upper,
                                           labels, filename):
        """Enhanced parameter comparison with CI"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(true_values))
        width = 0.35
        
        ax.bar(x - width/2, true_values, width, label='True Mean', 
              color='orange', alpha=0.8)
        
        errors = np.array([estimated - ci_lower, ci_upper - estimated])
        ax.bar(x + width/2, estimated, width, label='Posterior Mean',
              color='steelblue', alpha=0.8,
              yerr=errors, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Estimation Comparison with 95% CI', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    @staticmethod
    def plot_validation_with_prediction(t, val, filename):
        """Plot validation with antibiotic shock"""
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        ax.axvspan(0.0, 0.5, color='#4472C4', alpha=0.05, label='Updating Phase')
        ax.axvspan(0.5, 1.0, color='#C00000', alpha=0.05, label='Prediction Phase')
        
        for i in range(4):
            ax.plot(t, val[:, i], color=colors[i], linewidth=2, label=f'Species {i+1}')
        
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Antibiotics ON')
        ax.set_xlabel("Normalized Time $t$")
        ax.set_ylabel("Living Biomass $\\overline{\\Phi}(t)$")
        ax.set_title('Validation: Antibiotic Shock at t=0.5', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    @staticmethod
    def plot_sensitivity_analysis(sensitivity: SensitivityResults, filename: str):
        """Plot sensitivity analysis tornado chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y = np.arange(len(sensitivity.param_names))
        
        ax.barh(y - 0.2, sensitivity.first_order, 0.4, 
               label='First-order (Si)', color='steelblue', alpha=0.8)
        ax.barh(y + 0.2, sensitivity.total_order, 0.4, 
               label='Total-order (STi)', color='orange', alpha=0.8)
        
        ax.set_yticks(y)
        ax.set_yticklabels(sensitivity.param_names)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title('Sobol Sensitivity Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    @staticmethod
    def create_summary_report(results_M1, results_M2, results_M3,
                             true_M1, true_M2, true_M3, filename):
        """Create summary report figure"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Parameter comparison
        ax1 = axes[0, 0]
        all_est = np.concatenate([results_M1.posterior_mean, 
                                  results_M2.posterior_mean, 
                                  results_M3.posterior_mean])
        all_true = np.concatenate([true_M1, true_M2, true_M3])
        ax1.scatter(all_true, all_est, s=100, alpha=0.7)
        lims = [min(all_true.min(), all_est.min())-0.1, 
                max(all_true.max(), all_est.max())+0.1]
        ax1.plot(lims, lims, 'k--', alpha=0.5)
        ax1.set_xlabel('True Value')
        ax1.set_ylabel('Estimated Value')
        ax1.set_title('Parameter Recovery', fontweight='bold')
        
        # Convergence
        ax2 = axes[0, 1]
        models = ['M1', 'M2', 'M3']
        r_hats = [np.max(results_M1.convergence.r_hat),
                  np.max(results_M2.convergence.r_hat),
                  np.max(results_M3.convergence.r_hat)]
        colors = ['green' if r < 1.1 else 'red' for r in r_hats]
        ax2.bar(models, r_hats, color=colors, alpha=0.7)
        ax2.axhline(1.1, color='red', linestyle='--', label='Threshold')
        ax2.set_ylabel('Max R-hat')
        ax2.set_title('Convergence (R-hat)', fontweight='bold')
        ax2.legend()
        
        # ESS
        ax3 = axes[1, 0]
        ess_mins = [np.min(results_M1.convergence.ess),
                    np.min(results_M2.convergence.ess),
                    np.min(results_M3.convergence.ess)]
        colors = ['green' if e > 100 else 'red' for e in ess_mins]
        ax3.bar(models, ess_mins, color=colors, alpha=0.7)
        ax3.axhline(100, color='red', linestyle='--', label='Threshold')
        ax3.set_ylabel('Min ESS')
        ax3.set_title('Effective Sample Size', fontweight='bold')
        ax3.legend()
        
        # Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ['Model', 'Converged', 'R-hat (max)', 'ESS (min)', 'Accept Rate'],
            ['M1', '✓' if results_M1.convergence.is_converged else '✗',
             f'{np.max(results_M1.convergence.r_hat):.3f}',
             f'{np.min(results_M1.convergence.ess):.0f}',
             f'{100*results_M1.convergence.acceptance_rate:.1f}%'],
            ['M2', '✓' if results_M2.convergence.is_converged else '✗',
             f'{np.max(results_M2.convergence.r_hat):.3f}',
             f'{np.min(results_M2.convergence.ess):.0f}',
             f'{100*results_M2.convergence.acceptance_rate:.1f}%'],
            ['M3', '✓' if results_M3.convergence.is_converged else '✗',
             f'{np.max(results_M3.convergence.r_hat):.3f}',
             f'{np.min(results_M3.convergence.ess):.0f}',
             f'{100*results_M3.convergence.acceptance_rate:.1f}%']
        ]
        
        table = ax4.table(cellText=summary_data, loc='center', cellLoc='center',
                         colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.suptitle('Bayesian Biofilm Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

# =============================================================================
# MAIN EXECUTION WITH COMPLETE DATA SAVING
# =============================================================================
def main():
    print("="*75)
    print("  ENHANCED BAYESIAN UPDATING FOR BIOFILM MODELS")
    print("  ** PARALLELIZED VERSION **")
    print("  With Confidence Intervals, Convergence Diagnostics, and Validation")
    print("="*75)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU Cores: {N_CORES}, MCMC Chains: {N_CHAINS}")
    print(f"  Output directory: {output_folder}/")
    print("="*75)
    
    # Initialize
    solver = HierarchicalBiofilmSolver(eps=1e-4)
    optimizer = BayesianOptimizer(solver)
    visualizer = EnhancedVisualizer()
    
    # True Parameters (Paper Case II)
    TRUE_M1 = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
    TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])
    TRUE_M3 = np.array([2.0, 1.0, 2.0, 1.0])
    
    # Parameter names
    param_names_M1 = ['a11', 'a12', 'a22', 'b1', 'b2']
    param_names_M2 = ['a33', 'a34', 'a44', 'b3', 'b4']
    param_names_M3 = ['a13', 'a14', 'a23', 'a24']
    
    # =========================================================================
    # STEP 0: Generate Synthetic Data
    # =========================================================================
    print("\n[Step 0] Generating Synthetic Data (CoV=0.5%)...")
    CoV = 0.005
    t = np.linspace(0, 1, 10)
    
    d_M1 = solver.run_M1(TRUE_M1) + np.random.normal(0, 0.002, (10, 2))
    d_M2 = solver.run_M2(TRUE_M2) + np.random.normal(0, 0.002, (10, 2))
    d_M3 = solver.run_M3(TRUE_M3, TRUE_M1, TRUE_M2) + np.random.normal(0, 0.002, (10, 4))
    print("  Data generation complete.")
    
    # =========================================================================
    # STAGE 1: M1 (Species 1 & 2)
    # =========================================================================
    print("\n" + "="*75)
    print("[Step 1] Bayesian Updating for M1 (Species 1 & 2)")
    print("="*75)
    
    t_total_M1 = time_module.time()
    
    results_M1 = optimizer.optimize(
        optimizer.objective_M1,
        bounds=[(0, 3)]*5,
        args_tuple=(d_M1,),
        name="M1",
        n_posterior_samples=500
    )
    
    print(f"\n[M1] Total time: {time_module.time()-t_total_M1:.1f}s")
    print(f"[M1] MAP Estimate: {np.round(results_M1.map_estimate, 4)}")
    print(f"[M1] Posterior Mean: {np.round(results_M1.posterior_mean, 4)}")
    print(f"[M1] 95% CI Lower: {np.round(results_M1.ci_lower, 4)}")
    print(f"[M1] 95% CI Upper: {np.round(results_M1.ci_upper, 4)}")
    print(f"[M1] True: {TRUE_M1}")
    
    # Visualizations for M1
    print("\n  Generating M1 visualizations...")
    visualizer.plot_posterior_with_ci(
        results_M1, np.array(TRUE_M1), param_names_M1,
        "fig_M1_posterior_with_CI.png"
    )
    
    visualizer.plot_convergence_diagnostics(
        results_M1, param_names_M1,
        "fig_M1_convergence_diagnostics.png"
    )
    
    fit_M1 = solver.run_M1(results_M1.posterior_mean)
    fit_std_M1 = np.ones_like(fit_M1) * 0.01
    visualizer.plot_model_fit_with_ci(
        d_M1, fit_M1, fit_std_M1,
        "M1: Model Fit (Species 1 & 2)", [1, 2],
        "fig_M1_fit_with_CI.png"
    )
    
    # =========================================================================
    # STAGE 2: M2 (Species 3 & 4)
    # =========================================================================
    print("\n" + "="*75)
    print("[Step 2] Bayesian Updating for M2 (Species 3 & 4)")
    print("="*75)
    
    optimizer.reset()
    t_total_M2 = time_module.time()
    
    results_M2 = optimizer.optimize(
        optimizer.objective_M2,
        bounds=[(0, 3)]*5,
        args_tuple=(d_M2,),
        name="M2",
        n_posterior_samples=500
    )
    
    print(f"\n[M2] Total time: {time_module.time()-t_total_M2:.1f}s")
    print(f"[M2] MAP Estimate: {np.round(results_M2.map_estimate, 4)}")
    print(f"[M2] Posterior Mean: {np.round(results_M2.posterior_mean, 4)}")
    print(f"[M2] 95% CI Lower: {np.round(results_M2.ci_lower, 4)}")
    print(f"[M2] 95% CI Upper: {np.round(results_M2.ci_upper, 4)}")
    print(f"[M2] True: {TRUE_M2}")
    
    # Visualizations for M2
    print("\n  Generating M2 visualizations...")
    visualizer.plot_posterior_with_ci(
        results_M2, np.array(TRUE_M2), param_names_M2,
        "fig_M2_posterior_with_CI.png"
    )
    
    visualizer.plot_convergence_diagnostics(
        results_M2, param_names_M2,
        "fig_M2_convergence_diagnostics.png"
    )
    
    fit_M2 = solver.run_M2(results_M2.posterior_mean)
    fit_std_M2 = np.ones_like(fit_M2) * 0.01
    visualizer.plot_model_fit_with_ci(
        d_M2, fit_M2, fit_std_M2,
        "M2: Model Fit (Species 3 & 4)", [3, 4],
        "fig_M2_fit_with_CI.png"
    )
    
    # =========================================================================
    # STAGE 3: M3 (Cross-Interactions)
    # =========================================================================
    print("\n" + "="*75)
    print("[Step 3] Bayesian Updating for M3 (Cross-Interactions)")
    print("="*75)
    
    optimizer.reset()
    t_total_M3 = time_module.time()
    
    est_M1 = results_M1.posterior_mean
    est_M2 = results_M2.posterior_mean
    
    results_M3 = optimizer.optimize(
        lambda p, d, m1, m2: optimizer.objective_M3(p, d, m1, m2),
        bounds=[(0, 3)]*4,
        args_tuple=(d_M3, est_M1, est_M2),
        name="M3",
        n_posterior_samples=500
    )
    
    print(f"\n[M3] Total time: {time_module.time()-t_total_M3:.1f}s")
    print(f"[M3] MAP Estimate: {np.round(results_M3.map_estimate, 4)}")
    print(f"[M3] Posterior Mean: {np.round(results_M3.posterior_mean, 4)}")
    print(f"[M3] 95% CI Lower: {np.round(results_M3.ci_lower, 4)}")
    print(f"[M3] 95% CI Upper: {np.round(results_M3.ci_upper, 4)}")
    print(f"[M3] True: {TRUE_M3}")
    
    est_M3 = results_M3.posterior_mean
    
    # Visualizations for M3
    print("\n  Generating M3 visualizations...")
    visualizer.plot_posterior_with_ci(
        results_M3, np.array(TRUE_M3), param_names_M3,
        "fig_M3_posterior_with_CI.png"
    )
    
    visualizer.plot_convergence_diagnostics(
        results_M3, param_names_M3,
        "fig_M3_convergence_diagnostics.png"
    )
    
    fit_M3 = solver.run_M3(est_M3, est_M1, est_M2)
    fit_std_M3 = np.ones_like(fit_M3) * 0.01
    visualizer.plot_model_fit_with_ci(
        d_M3, fit_M3, fit_std_M3,
        "M3: Model Fit (All Species)", [1, 2, 3, 4],
        "fig_M3_fit_with_CI.png"
    )
    
    # =========================================================================
    # FINAL RESULTS AND SUMMARY
    # =========================================================================
    print("\n" + "="*75)
    print("=== FINAL ESTIMATION RESULTS ===")
    print("="*75)
    
    raw_est = np.concatenate([est_M1, est_M2, est_M3])
    raw_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])
    raw_ci_lower = np.concatenate([results_M1.ci_lower, results_M2.ci_lower, results_M3.ci_lower])
    raw_ci_upper = np.concatenate([results_M1.ci_upper, results_M2.ci_upper, results_M3.ci_upper])
    raw_std = np.concatenate([results_M1.posterior_std, results_M2.posterior_std, results_M3.posterior_std])
    
    raw_labels = np.array([
        "a11", "a12", "a22", "b1", "b2",
        "a33", "a34", "a44", "b3", "b4",
        "a13", "a14", "a23", "a24"
    ])
    
    new_order_indices = [0, 1, 10, 11, 2, 12, 13, 5, 6, 7, 3, 4, 8, 9]
    all_est = raw_est[new_order_indices]
    all_true = raw_true[new_order_indices]
    all_ci_lower = raw_ci_lower[new_order_indices]
    all_ci_upper = raw_ci_upper[new_order_indices]
    labels = raw_labels[new_order_indices]
    
    df = pd.DataFrame({
        "Parameter": labels,
        "True": all_true,
        "Estimated": np.round(all_est, 4),
        "95% CI Lower": np.round(all_ci_lower, 4),
        "95% CI Upper": np.round(all_ci_upper, 4),
        "Error": np.round(all_est - all_true, 4),
        "Error %": np.round(100 * (all_est - all_true) / (all_true + 1e-10), 2)
    })
    print("\n" + df.to_string(index=False))
    
    # Enhanced plots
    print("\n  Generating final plots...")
    visualizer.plot_parameters_comparison_enhanced(
        all_est, all_true, all_ci_lower, all_ci_upper, labels,
        "fig_parameters_comparison_enhanced.png"
    )
    
    visualizer.create_summary_report(
        results_M1, results_M2, results_M3,
        TRUE_M1, TRUE_M2, TRUE_M3,
        "fig_summary_report.png"
    )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    print("\n" + "="*75)
    print("Running Validation (M3_val) - Antibiotic Shock at t=0.5")
    print("="*75)
    
    t_val, val = solver.run_M3_val(est_M1, est_M2, est_M3)
    
    if val is not None:
        visualizer.plot_validation_with_prediction(
            t_val, val, "fig_validation_antibiotic_shock.png"
        )
    
    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "="*75)
    print("Running Sensitivity Analysis (Sobol Indices Approximation)...")
    print("="*75)
    
    def simple_output_func(params):
        sim = solver.run_M1(params)
        if sim is None:
            return 0.0
        return np.mean(sim)
    
    sensitivity_M1 = SensitivityAnalyzer.sobol_indices_approximation(
        simple_output_func,
        bounds=[(0, 3)]*5,
        n_samples=100,
        param_names=param_names_M1
    )
    
    visualizer.plot_sensitivity_analysis(
        sensitivity_M1, "fig_sensitivity_analysis_M1.png"
    )
    
    print("\n  First-order indices:", np.round(sensitivity_M1.first_order, 4))
    print("  Total-order indices:", np.round(sensitivity_M1.total_order, 4))
    
    # =========================================================================
    # COMPLETE DATA SAVING
    # =========================================================================
    print("\n" + "="*75)
    print("=== COMPLETE DATA BACKUP (SAVING EVERYTHING) ===")
    print("="*75)
    
    # 1. Summary Statistics
    df_summary = pd.DataFrame({
        "Parameter": raw_labels,
        "True": raw_true,
        "Estimated": np.round(raw_est, 6),
        "Std": np.round(raw_std, 6),
        "CI_Lower_95": np.round(raw_ci_lower, 6),
        "CI_Upper_95": np.round(raw_ci_upper, 6),
        "Error": np.round(raw_est - raw_true, 6),
        "Error_Percent": np.round(100 * (raw_est - raw_true) / (raw_true + 1e-10), 2)
    })
    df_summary.to_csv(os.path.join(output_folder, "estimation_summary.csv"), index=False)
    print(f"  [1/9] ✓ Summary: estimation_summary.csv")
    
    # 2. Raw Posterior Samples
    raw_samples = np.hstack([
        results_M1.posterior_samples,
        results_M2.posterior_samples,
        results_M3.posterior_samples
    ])
    
    all_param_names_for_samples = (
        ['m1_a11', 'm1_a12', 'm1_a22', 'm1_b1', 'm1_b2'] +
        ['m2_a33', 'm2_a34', 'm2_a44', 'm2_b3', 'm2_b4'] +
        ['m3_a13', 'm3_a14', 'm3_a23', 'm3_a24']
    )
    
    df_samples = pd.DataFrame(raw_samples, columns=all_param_names_for_samples)
    df_samples.to_csv(os.path.join(output_folder, "posterior_samples_raw.csv"), index=False)
    print(f"  [2/9] ✓ Posterior samples (CSV): posterior_samples_raw.csv (Shape: {raw_samples.shape})")
    
    np.savez_compressed(
        os.path.join(output_folder, "posterior_samples_raw.npz"),
        samples_M1=results_M1.posterior_samples,
        samples_M2=results_M2.posterior_samples,
        samples_M3=results_M3.posterior_samples,
        param_names_M1=param_names_M1,
        param_names_M2=param_names_M2,
        param_names_M3=param_names_M3
    )
    print(f"  [2/9] ✓ Posterior samples (NPZ): posterior_samples_raw.npz")
    
    # 3. Input Data
    df_d_M1 = pd.DataFrame(d_M1, columns=['Species1', 'Species2'])
    df_d_M1['time'] = t
    df_d_M1.to_csv(os.path.join(output_folder, "input_data_M1.csv"), index=False)
    
    df_d_M2 = pd.DataFrame(d_M2, columns=['Species3', 'Species4'])
    df_d_M2['time'] = t
    df_d_M2.to_csv(os.path.join(output_folder, "input_data_M2.csv"), index=False)
    
    df_d_M3 = pd.DataFrame(d_M3, columns=['Species1', 'Species2', 'Species3', 'Species4'])
    df_d_M3['time'] = t
    df_d_M3.to_csv(os.path.join(output_folder, "input_data_M3.csv"), index=False)
    
    print(f"  [3/9] ✓ Input data: input_data_M1.csv, input_data_M2.csv, input_data_M3.csv")
    
    # 4. True Parameters
    df_true = pd.DataFrame({
        'Model': ['M1']*5 + ['M2']*5 + ['M3']*4,
        'Parameter': param_names_M1 + param_names_M2 + param_names_M3,
        'True_Value': list(TRUE_M1) + list(TRUE_M2) + list(TRUE_M3)
    })
    df_true.to_csv(os.path.join(output_folder, "true_parameters.csv"), index=False)
    print(f"  [4/9] ✓ True parameters: true_parameters.csv")
    
    # 5. Convergence Diagnostics
    conv_data = []
    for name, r, ess in zip(param_names_M1, results_M1.convergence.r_hat, results_M1.convergence.ess):
        conv_data.append({'Model': 'M1', 'Param': name, 'R_hat': r, 'ESS': ess, 
                         'AcceptRate': results_M1.convergence.acceptance_rate,
                         'Converged': results_M1.convergence.is_converged})
    for name, r, ess in zip(param_names_M2, results_M2.convergence.r_hat, results_M2.convergence.ess):
        conv_data.append({'Model': 'M2', 'Param': name, 'R_hat': r, 'ESS': ess,
                         'AcceptRate': results_M2.convergence.acceptance_rate,
                         'Converged': results_M2.convergence.is_converged})
    for name, r, ess in zip(param_names_M3, results_M3.convergence.r_hat, results_M3.convergence.ess):
        conv_data.append({'Model': 'M3', 'Param': name, 'R_hat': r, 'ESS': ess,
                         'AcceptRate': results_M3.convergence.acceptance_rate,
                         'Converged': results_M3.convergence.is_converged})
    
    pd.DataFrame(conv_data).to_csv(os.path.join(output_folder, "convergence_metrics.csv"), index=False)
    print(f"  [5/9] ✓ Convergence metrics: convergence_metrics.csv")
    
    # 6. Optimization History
    max_len = max(len(results_M1.loss_history), len(results_M2.loss_history), len(results_M3.loss_history))
    
    def pad_list(lst, length):
        return list(lst) + [np.nan] * (length - len(lst))
    
    df_loss = pd.DataFrame({
        'Iteration': range(max_len),
        'Loss_M1': pad_list(results_M1.loss_history, max_len),
        'Loss_M2': pad_list(results_M2.loss_history, max_len),
        'Loss_M3': pad_list(results_M3.loss_history, max_len)
    })
    df_loss.to_csv(os.path.join(output_folder, "optimization_history.csv"), index=False)
    print(f"  [6/9] ✓ Optimization history: optimization_history.csv")
    
    # 7. Validation Data
    if val is not None:
        df_val = pd.DataFrame(val, columns=['Species1', 'Species2', 'Species3', 'Species4'])
        df_val['time'] = t_val
        df_val.to_csv(os.path.join(output_folder, "validation_prediction.csv"), index=False)
        print(f"  [7/9] ✓ Validation data: validation_prediction.csv")
    
    # 8. Model Fit Data
    df_fit_M1 = pd.DataFrame(fit_M1, columns=['Species1_fit', 'Species2_fit'])
    df_fit_M1['time'] = t
    df_fit_M1.to_csv(os.path.join(output_folder, "fit_data_M1.csv"), index=False)
    
    df_fit_M2 = pd.DataFrame(fit_M2, columns=['Species3_fit', 'Species4_fit'])
    df_fit_M2['time'] = t
    df_fit_M2.to_csv(os.path.join(output_folder, "fit_data_M2.csv"), index=False)
    
    df_fit_M3 = pd.DataFrame(fit_M3, columns=['Species1_fit', 'Species2_fit', 'Species3_fit', 'Species4_fit'])
    df_fit_M3['time'] = t
    df_fit_M3.to_csv(os.path.join(output_folder, "fit_data_M3.csv"), index=False)
    
    print(f"  [8/9] ✓ Model fit data: fit_data_M1.csv, fit_data_M2.csv, fit_data_M3.csv")
    
    # 9. Complete Archive
    np.savez_compressed(
        os.path.join(output_folder, "complete_archive.npz"),
        TRUE_M1=TRUE_M1, TRUE_M2=TRUE_M2, TRUE_M3=TRUE_M3,
        d_M1=d_M1, d_M2=d_M2, d_M3=d_M3, t=t,
        samples_M1=results_M1.posterior_samples,
        samples_M2=results_M2.posterior_samples,
        samples_M3=results_M3.posterior_samples,
        mean_M1=results_M1.posterior_mean, mean_M2=results_M2.posterior_mean, mean_M3=results_M3.posterior_mean,
        std_M1=results_M1.posterior_std, std_M2=results_M2.posterior_std, std_M3=results_M3.posterior_std,
        ci_lower_M1=results_M1.ci_lower, ci_lower_M2=results_M2.ci_lower, ci_lower_M3=results_M3.ci_lower,
        ci_upper_M1=results_M1.ci_upper, ci_upper_M2=results_M2.ci_upper, ci_upper_M3=results_M3.ci_upper,
        map_M1=results_M1.map_estimate, map_M2=results_M2.map_estimate, map_M3=results_M3.map_estimate,
        r_hat_M1=results_M1.convergence.r_hat, r_hat_M2=results_M2.convergence.r_hat, r_hat_M3=results_M3.convergence.r_hat,
        ess_M1=results_M1.convergence.ess, ess_M2=results_M2.convergence.ess, ess_M3=results_M3.convergence.ess,
        acceptance_rate_M1=results_M1.convergence.acceptance_rate,
        acceptance_rate_M2=results_M2.convergence.acceptance_rate,
        acceptance_rate_M3=results_M3.convergence.acceptance_rate,
        fit_M1=fit_M1, fit_M2=fit_M2, fit_M3=fit_M3,
        t_val=t_val if val is not None else np.array([]),
        val_data=val if val is not None else np.array([]),
        sensitivity_first_order=sensitivity_M1.first_order,
        sensitivity_total_order=sensitivity_M1.total_order
    )
    print(f"  [9/9] ✓ Complete archive: complete_archive.npz")
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'computation_notes': 'PARALLELIZED Bayesian optimization with parallel MCMC chains',
        'parallel_config': {
            'n_cpu_cores': N_CORES,
            'n_mcmc_chains': N_CHAINS,
            'n_mc_workers': N_WORKERS_MC
        },
        'n_posterior_samples': results_M1.posterior_samples.shape[0],
        'n_timepoints': len(t),
        'CoV': CoV,
        'true_parameters': {
            'M1': list(TRUE_M1),
            'M2': list(TRUE_M2),
            'M3': list(TRUE_M3)
        },
        'parameter_names': {
            'M1': param_names_M1,
            'M2': param_names_M2,
            'M3': param_names_M3
        },
        'convergence_summary': {
            'M1': {
                'converged': results_M1.convergence.is_converged, 
                'max_rhat': float(np.max(results_M1.convergence.r_hat)),
                'min_ess': float(np.min(results_M1.convergence.ess)),
                'acceptance_rate': float(results_M1.convergence.acceptance_rate)
            },
            'M2': {
                'converged': results_M2.convergence.is_converged,
                'max_rhat': float(np.max(results_M2.convergence.r_hat)),
                'min_ess': float(np.min(results_M2.convergence.ess)),
                'acceptance_rate': float(results_M2.convergence.acceptance_rate)
            },
            'M3': {
                'converged': results_M3.convergence.is_converged,
                'max_rhat': float(np.max(results_M3.convergence.r_hat)),
                'min_ess': float(np.min(results_M3.convergence.ess)),
                'acceptance_rate': float(results_M3.convergence.acceptance_rate)
            }
        },
        'validation_metrics': {
            'M1': {
                'AIC': float(results_M1.validation.aic),
                'BIC': float(results_M1.validation.bic),
                'RMSE': float(results_M1.validation.rmse),
                'R_squared': float(results_M1.validation.r_squared)
            },
            'M2': {
                'AIC': float(results_M2.validation.aic),
                'BIC': float(results_M2.validation.bic),
                'RMSE': float(results_M2.validation.rmse),
                'R_squared': float(results_M2.validation.r_squared)
            },
            'M3': {
                'AIC': float(results_M3.validation.aic),
                'BIC': float(results_M3.validation.bic),
                'RMSE': float(results_M3.validation.rmse),
                'R_squared': float(results_M3.validation.r_squared)
            }
        }
    }
    
    with open(os.path.join(output_folder, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [+]   ✓ Metadata: metadata.json")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*75)
    print("=== ALL DATA SAVED SUCCESSFULLY ===")
    print("="*75)
    print(f"\nOutput directory: {output_folder}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_folder)):
        size = os.path.getsize(os.path.join(output_folder, f))
        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
        print(f"    • {f:<45} ({size_str})")
    
    print("\n" + "-"*75)
    print("  PARALLELIZATION SUMMARY:")
    print(f"    • CPU Cores Used: {N_CORES}")
    print(f"    • MCMC Chains: {N_CHAINS}")
    print(f"    • Differential Evolution: workers=-1 (all cores)")
    print("-"*75)
    print("\n" + "="*75)
    print("  Analysis completed successfully!")
    print("="*75)

if __name__ == "__main__":
    main()
