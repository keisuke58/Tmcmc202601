"""
=================================================================================
Case II: Hierarchical Bayesian Parameter Estimation with MCMC (Aligned Version)
=================================================================================

Complete execution script for hierarchical Bayesian updating of biofilm parameters
using adaptive MCMC with TSM-ROM.

🚀 ALIGNED WITH: improved1207_paper_jit.py (authoritative reference)

State Vector Definition (from authoritative file):
    g (10,) = [phi1, phi2, phi3, phi4, phi0, psi1, psi2, psi3, psi4, gamma]
    Active species: subset of {0,1,2,3}  (phi0, gamma are always present)

Theta (14,) order:
    [a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24]

True parameters (Case II):
    θ* = [0.8, 2.0, 1.0, 0.1, 0.2, 1.5, 1.0, 2.0, 0.3, 0.4, 2.0, 1.0, 2.0, 1.0]

Features:
  1. Hierarchical Bayesian estimation: M1 → M2 → M3
  2. TSM-ROM for uncertainty propagation (BiofilmTSM from authoritative file)
  3. Adaptive MCMC with proposal covariance learning
  4. Two-phase MCMC for M3 (coarse → refined)
  5. Comprehensive visualization and diagnostics

Usage:
    python case2_tmcmc_refined.py

Author: Keisuke (keisuke58)
Date: December 2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from config import setup_logging

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class MCMCConfig:
    """MCMC sampling configuration."""
    n_samples: int = 5000
    n_burn_in: int = 1000
    n_chains: int = 1
    initial_scale: float = 0.02 # 0.05
    target_accept: float = 0.30
    adapt_start: int = 100
    adapt_interval: int = 50


@dataclass
class ExperimentConfig:
    """Experiment configuration for synthetic data generation."""
    cov_rel: float = 0.005       # TSM relative covariance
    n_data: int = 20             # Number of observations
    sigma_obs: float = 0.001     # Observation noise
    output_dir: str = "case2_tmcmc_results2"
    random_seed: int = 42


# Model-specific configurations
MODEL_CONFIGS = {
    "M1": {
        "dt": 1e-5,
        "maxtimestep": 2500,
        "c_const": 100.0,
        "alpha_const": 100.0,
        "phi_init": 0.2,
        "active_species": [0, 1],     # Species indices in state vector
        "active_indices": list(range(5)),  # θ indices [a11,a12,a22,b1,b2]
        "param_names": ["a11", "a12", "a22", "b1", "b2"],
    },
    "M2": {
        "dt": 1e-5,
        "maxtimestep": 5000,
        "c_const": 100.0,
        "alpha_const": 10.0,
        "phi_init": 0.2,
        "active_species": [2, 3],
        "active_indices": list(range(5, 10)),  # θ indices [a33,a34,a44,b3,b4]
        "param_names": ["a33", "a34", "a44", "b3", "b4"],
    },
    "M3": {
        "dt": 1e-4,
        "maxtimestep": 750,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": 0.02,
        "active_species": [0, 1, 2, 3],
        "active_indices": list(range(10, 14)),  # θ indices [a13,a14,a23,a24]
        "param_names": ["a13", "a14", "a23", "a24"],
    },
}

# Prior bounds (uniform)
PRIOR_BOUNDS_DEFAULT = (0.0, 3.0)


# ==============================================================================
# IMPORTS (from authoritative file)
# ==============================================================================

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from AUTHORITATIVE reference file
from improved1207_paper_jit import (
    BiofilmNewtonSolver,
    BiofilmTSM,
    get_theta_true,
    HAS_NUMBA,
)

# Import diagnostics
from mcmc_diagnostics import MCMCDiagnostics

# Verify complex-step compatibility
from bugfix_theta_to_matrices import patch_biofilm_solver

# NOTE: Avoid import-time side effects; call patch in main() when running as script.


# ==============================================================================
# HELPER FUNCTIONS (added since not in authoritative file)
# ==============================================================================


def select_sparse_data_indices(n_total: int, n_obs: int) -> np.ndarray:
    """
    Select evenly spaced indices for sparse observations.
    
    Parameters
    ----------
    n_total : int
        Total number of time steps
    n_obs : int
        Number of observations to select
        
    Returns
    -------
    indices : ndarray
        Array of selected indices (skipping first ~10% of trajectory)
    """
    start_idx = int(0.1 * n_total)  # Skip initial transient
    indices = np.linspace(start_idx, n_total - 1, n_obs)
    indices = np.floor(indices).astype(int)
    indices = np.clip(indices, 0, n_total - 1)
    return indices



def log_likelihood_sparse(
    mu: np.ndarray,
    sig: np.ndarray,
    data: np.ndarray,
    sigma_obs: float,
) -> float:
    """
    Compute log-likelihood for sparse observations.
    
    Using Gaussian likelihood with combined TSM + observation variance.
    
    Parameters
    ----------
    mu : ndarray (n_obs, n_species)
        Model predicted mean (φ̄ = φψ)
    sig : ndarray (n_obs, n_species)
        Model predicted variance (from TSM)
    data : ndarray (n_obs, n_species)
        Observed data
    sigma_obs : float
        Observation noise standard deviation
        
    Returns
    -------
    logL : float
        Log-likelihood value
    """
    n_obs, n_species = data.shape
    logL = 0.0
    
    for i in range(n_obs):
        for j in range(n_species):
            # Total variance = TSM variance + observation variance
            var_total = max(sig[i, j] + sigma_obs**2, 1e-20)
            
            # Gaussian log-likelihood
            residual = data[i, j] - mu[i, j]
            logL -= 0.5 * np.log(2 * np.pi * var_total)
            logL -= 0.5 * (residual**2) / var_total
    
    return logL


def compute_phibar(x0: np.ndarray, active_species: List[int]) -> np.ndarray:
    n_t = x0.shape[0]
    n_sp = len(active_species)
    phibar = np.zeros((n_t, n_sp))

    n_state = x0.shape[1]          # = 10
    n_total_species = (n_state - 2) // 2   # = 4
    psi_offset = n_total_species + 1       # = 5

    for i, sp in enumerate(active_species):
        phibar[:, i] = x0[:, sp] * x0[:, psi_offset + sp]

    return phibar



def compute_MAP_with_uncertainty(
    samples: np.ndarray, logL: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute MAP estimate and posterior statistics.
    """
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    idx_map = np.argmax(logL)
    MAP = samples[idx_map]
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)
    
    return {
        "mean": mean,
        "std": std,
        "MAP": MAP,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================


class PlotManager:
    """Manages plot generation and file tracking."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_figs: List[Path] = []
    
    def save_figure(self, filename: str, dpi: int = 150):
        """Save current figure and track it."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        self.generated_figs.append(path)
        logger.info("Saved figure: %s", path.name)
    
    def plot_TSM_simulation(
        self,
        t_arr: np.ndarray,
        x0: np.ndarray,
        active_species: List[int],
        name: str,
        data: Optional[np.ndarray] = None,
        idx_sparse: Optional[np.ndarray] = None,
    ):
        """Plot φ̄ time series from TSM simulation."""
        phibar = compute_phibar(x0, active_species)
        
        plt.figure(figsize=(10, 6))
        for i, sp in enumerate(active_species):
            plt.plot(t_arr, phibar[:, i], label=f"φ̄{sp+1} (model)", linewidth=2)
        
        if data is not None and idx_sparse is not None:
            t_obs = t_arr[idx_sparse]
            for i, sp in enumerate(active_species):
                plt.scatter(
                    t_obs, data[:, i], s=40, edgecolor="k",
                    label=f"Data φ̄{sp+1}", alpha=0.8, zorder=10,
                )
        
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("φ̄ = φ * ψ", fontsize=12)
        plt.title(f"TSM Simulation (φ̄) - {name}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        suffix = "_with_data" if data is not None else ""
        self.save_figure(f"TSM_simulation_{name}{suffix}.png")
    
    def plot_posterior(
        self,
        samples: np.ndarray,
        theta_true: np.ndarray,
        param_names: List[str],
        name_tag: str,
        MAP: np.ndarray,
        mean: np.ndarray,
    ):
        """Plot posterior distributions with true, MAP, and mean values."""
        n_params = samples.shape[1]
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            
            if i >= n_params:
                ax.axis("off")
                continue
            
            ax.hist(samples[:, i], bins=40, alpha=0.7, density=True, color="steelblue")
            ax.axvline(theta_true[i], color="red", linestyle="--", linewidth=2, label="True")
            ax.axvline(MAP[i], color="green", linestyle="-", linewidth=2, label="MAP")
            ax.axvline(mean[i], color="orange", linestyle=":", linewidth=2, label="Mean")
            ax.set_xlabel(param_names[i], fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        fig.suptitle(f"Posterior Distributions ({name_tag})", fontsize=14)
        fig.tight_layout()
        self.save_figure(f"posterior_{name_tag}.png")
    
    def plot_parameter_comparison(
        self,
        theta_true: np.ndarray,
        theta_map: np.ndarray,
        theta_mean: np.ndarray,
        param_names: List[str],
    ):
        """Bar plot comparing all parameters."""
        idx = np.arange(len(param_names))
        width = 0.25
        
        plt.figure(figsize=(14, 6))
        plt.bar(idx - width, theta_true, width, label="True", alpha=0.8)
        plt.bar(idx, theta_map, width, label="MAP", alpha=0.8)
        plt.bar(idx + width, theta_mean, width, label="Mean", alpha=0.8)
        
        plt.xticks(idx, param_names, rotation=45, ha="right")
        plt.ylabel("Parameter Value", fontsize=12)
        plt.title("All Parameters: True vs MAP vs Mean", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        self.save_figure("posterior_all_parameters.png")


# ==============================================================================
# LIKELIHOOD FUNCTION
# ==============================================================================


class LogLikelihoodEvaluator:
    """
    Log-likelihood evaluator using TSM-ROM from authoritative file.
    """
    
    def __init__(
        self,
        solver_kwargs: Dict[str, Any],
        active_species: List[int],
        active_indices: List[int],
        theta_base: np.ndarray,
        data: np.ndarray,
        idx_sparse: np.ndarray,
        sigma_obs: float,
        cov_rel: float,
    ):
        """
        Initialize likelihood evaluator.
        
        Parameters
        ----------
        solver_kwargs : dict
            Arguments for BiofilmNewtonSolver
        active_species : List[int]
            Active species indices {0,1,2,3}
        active_indices : List[int]
            Parameter indices to estimate
        theta_base : ndarray (14,)
            Base parameter vector
        data : ndarray (n_obs, n_species)
            Observed data
        idx_sparse : ndarray
            Observation time indices
        sigma_obs : float
            Observation noise std
        cov_rel : float
            Relative covariance for TSM
        """
        self.active_species = list(active_species)
        self.active_indices = list(active_indices)
        self.theta_base = theta_base.copy()
        self.data = data
        self.idx_sparse = idx_sparse
        self.sigma_obs = sigma_obs
        self.cov_rel = cov_rel
        self.n_species = len(active_species)
        
        # Tracking
        self.call_count = 0
        self.theta_history = []
        self.logL_history = []
        
        # Create solver from authoritative file
        self.solver = BiofilmNewtonSolver(
            **solver_kwargs,
            active_species=self.active_species,
            use_numba=HAS_NUMBA,
        )
        
        # Create TSM from authoritative file
        self.tsm = BiofilmTSM(
            self.solver,
            active_theta_indices=self.active_indices,
            cov_rel=self.cov_rel,
            use_complex_step=True,
        )
    
    def __call__(self, theta_sub: np.ndarray) -> float:
        """
        Evaluate log-likelihood for given parameter subset.
        
        Parameters
        ----------
        theta_sub : ndarray
            Parameter subset to evaluate
            
        Returns
        -------
        logL : float
            Log-likelihood value
        """
        self.call_count += 1
        
        # Construct full parameter vector
        full_theta = self.theta_base.copy()
        for i, idx in enumerate(self.active_indices):
            full_theta[idx] = theta_sub[i]
        
        # Solve TSM
        try:
            t_arr, x0, sig2 = self.tsm.solve_tsm(full_theta)
        except Exception as e:
            logger.warning("TSM failed: %s", e)
            return -1e20
        
        # Compute predicted mean and variance at observation times
        mu = np.zeros((len(self.idx_sparse), self.n_species))
        sig = np.zeros((len(self.idx_sparse), self.n_species))
        
        for i, sp in enumerate(self.active_species):
            # ★ ここが重要：必ず clip
            idx = np.clip(self.idx_sparse, 0, sig2.shape[0] - 1)

            phi = x0[idx, sp]
            psi = x0[idx, 5 + sp]
            sig2_phi = sig2[idx, sp]
            sig2_psi = sig2[idx, 5 + sp]

            mu[:, i] = phi * psi
            var_phibar = phi**2 * sig2_psi + psi**2 * sig2_phi
            x1 = getattr(self.tsm, "_last_x1", None)
            var_act = getattr(self.tsm, "_last_var_act", None)
            if x1 is not None and var_act is not None:
                try:
                    x1_phi = x1[idx, sp, :]
                    x1_psi = x1[idx, 5 + sp, :]
                    cov_phi_psi = np.sum(x1_phi * x1_psi * var_act[None, :], axis=1)
                    var_phibar = var_phibar + 2.0 * phi * psi * cov_phi_psi
                except Exception:
                    pass
            sig[:, i] = var_phibar
        
        # Evaluate log-likelihood
        logL = log_likelihood_sparse(mu, sig, self.data, self.sigma_obs)
        
        # Track evaluation
        self.theta_history.append(theta_sub.copy())
        self.logL_history.append(logL)
        
        return logL
    
    def get_MAP(self) -> Tuple[np.ndarray, float]:
        """Get MAP estimate from evaluation history."""
        if len(self.logL_history) == 0:
            raise ValueError("No evaluations yet")
        
        idx_max = np.argmax(self.logL_history)
        theta_MAP = self.theta_history[idx_max]
        logL_MAP = self.logL_history[idx_max]
        
        return theta_MAP, logL_MAP


# ==============================================================================
# ADAPTIVE MCMC
# ==============================================================================


def run_adaptive_MCMC(
    log_likelihood: callable,
    prior_bounds: List[Tuple[float, float]],
    n_samples: int,
    initial_scale: float = 0.05,
    burn_in: int = 500,
    target_accept: float = 0.3,
    adapt_start: int = 100,
    adapt_interval: int = 50,
    proposal_cov: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Adaptive random-walk Metropolis-Hastings MCMC with MAP estimation.
    
    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function
    prior_bounds : List[Tuple]
        (lower, upper) bounds for each parameter
    n_samples : int
        Total samples (including burn-in)
    initial_scale : float
        Initial proposal std
    burn_in : int
        Burn-in samples to discard
    target_accept : float
        Target acceptance rate
    adapt_start : int
        Step to start adaptation
    adapt_interval : int
        Interval between adaptations
    proposal_cov : ndarray, optional
        Covariance matrix for multivariate proposals
    seed : int, optional
        Random seed
        
    Returns
    -------
    samples : ndarray (n_samples - burn_in, n_params)
        MCMC samples
    logL_values : ndarray
        Log-likelihood values
    theta_MAP : ndarray
        MAP estimate
    acceptance_rate : float
        Overall acceptance rate
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_params = len(prior_bounds)
    
    # ★ 重要: 初期点を 1.5 ± ε に設定（ε = proposal σ）
    theta_center = np.array([(low + high) / 2 for low, high in prior_bounds])
    epsilon = initial_scale  # proposal σ と同じ値
    theta_current = theta_center + np.random.randn(n_params) * epsilon
    proposal_std = np.full(n_params, initial_scale)
    
    def log_prior(theta: np.ndarray) -> float:
        """Uniform log-prior."""
        for i, (low, high) in enumerate(prior_bounds):
            if not (low <= theta[i] <= high):
                return -np.inf
        return 0.0
    
    def log_posterior(theta: np.ndarray) -> float:
        """Log-posterior = log-prior + log-likelihood."""
        lp = log_prior(theta)
        if np.isinf(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    # Evaluate initial posterior
    log_post_current = log_posterior(theta_current)
    
    # Storage
    samples_all = np.zeros((n_samples, n_params))
    logL_all = np.zeros(n_samples)
    n_accepted = 0
    
    logger.info("Initial log posterior = %.2f", log_post_current)
    
    # MCMC loop
    for i in range(n_samples):
        # Generate proposal
        if proposal_cov is None:
            eps = np.random.randn(n_params) * proposal_std
        else:
            eps = np.random.multivariate_normal(np.zeros(n_params), proposal_cov)
        
        theta_proposed = theta_current + eps
        
        # Evaluate proposal
        log_post_proposed = log_posterior(theta_proposed)
        log_alpha = log_post_proposed - log_post_current
        
        # Accept/reject
        if np.log(np.random.rand()) < log_alpha:
            theta_current = theta_proposed
            log_post_current = log_post_proposed
            n_accepted += 1
        
        # Store sample
        samples_all[i] = theta_current
        logL_all[i] = log_post_current  # Store posterior (includes prior)
        
        # Progress update
        if (i + 1) % 500 == 0:
            acc_rate = n_accepted / (i + 1)
            logger.info("%s/%s samples, acceptance: %.1f%%", i + 1, n_samples, acc_rate * 100.0)
        
        # Adapt proposal scale (diagonal case)
        if proposal_cov is None:
            if (i + 1) >= adapt_start and (i + 1) % adapt_interval == 0:
                acc_rate = n_accepted / (i + 1)
                adjustment = np.exp(0.5 * (acc_rate - target_accept))
                proposal_std *= adjustment
                proposal_std = np.clip(proposal_std, 1e-4, 1.0)
    
    # Remove burn-in
    samples = samples_all[burn_in:]
    logL_values = logL_all[burn_in:]
    acceptance_rate = n_accepted / n_samples
    
    # Find MAP
    idx_MAP = np.argmax(logL_values)
    theta_MAP = samples[idx_MAP]
    
    logger.info("MCMC complete! Acceptance rate: %.1f%%", acceptance_rate * 100.0)
    logger.info("MAP: %s", theta_MAP)
    
    return samples, logL_values, theta_MAP, acceptance_rate


def run_multi_chain_MCMC(
    model_tag: str,
    evaluator_factory: callable,
    prior_bounds: List[Tuple[float, float]],
    mcmc_config: MCMCConfig,
    proposal_cov: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Run multiple MCMC chains sequentially with diagnostics.
    
    Note: Sequential execution avoids pickling issues with TSM objects.
    """
    logger.info("[%s] Running %s MCMC chains...", model_tag, mcmc_config.n_chains)
    
    all_samples = []
    all_logL = []
    all_MAP = []
    all_acc = []
    
    for chain_idx in range(mcmc_config.n_chains):
        seed = mcmc_config.n_chains * 1000 + chain_idx
        logger.info("Chain %s/%s", chain_idx + 1, mcmc_config.n_chains)
        
        # Create fresh evaluator for each chain
        evaluator = evaluator_factory()
        
        samples, logL, MAP, acc = run_adaptive_MCMC(
            log_likelihood=evaluator,
            prior_bounds=prior_bounds,
            n_samples=mcmc_config.n_samples,
            initial_scale=mcmc_config.initial_scale,
            burn_in=mcmc_config.n_burn_in,
            target_accept=mcmc_config.target_accept,
            adapt_start=mcmc_config.adapt_start,
            adapt_interval=mcmc_config.adapt_interval,
            proposal_cov=proposal_cov,
            seed=seed,
        )
        
        all_samples.append(samples)
        all_logL.append(logL)
        all_MAP.append(MAP)
        all_acc.append(acc)
    
    # Compute diagnostics
    diag = MCMCDiagnostics(all_samples, [f"θ{i}" for i in range(len(prior_bounds))])
    diag.compute_all()
    
    # Find global MAP
    best_logL = -np.inf
    best_theta = None
    for c, logL in enumerate(all_logL):
        idx = np.argmax(logL)
        if logL[idx] > best_logL:
            best_logL = logL[idx]
            best_theta = all_samples[c][idx]
    
    diagnostics = {
        "Rhat": diag.Rhat,
        "ESS": diag.ESS,
        "acc_rate_mean": float(np.mean(all_acc)),
        "MAP_global": best_theta,
        "MAP_logL": best_logL,
    }
    
    logger.info("[%s] Summary:", model_tag)
    logger.info("Mean acceptance rate: %.1f%%", diagnostics["acc_rate_mean"] * 100.0)
    logger.info("R-hat: %s", diag.Rhat)
    logger.info("ESS: %s", diag.ESS)
    logger.info("Global MAP: %s", best_theta)
    
    return all_samples, all_logL, best_theta, diagnostics


# ==============================================================================
# DATA GENERATION
# ==============================================================================


def generate_synthetic_data(
    config: Dict[str, Any],
    theta_true: np.ndarray,
    exp_config: ExperimentConfig,
    name: str,
    plot_mgr: PlotManager,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data from TSM simulation.
    
    Returns
    -------
    data : ndarray (n_obs, n_species)
        Synthetic observations
    idx_sparse : ndarray
        Observation time indices
    t_arr : ndarray
        Time array
    x0 : ndarray
        State trajectory
    sig2 : ndarray
        TSM variance trajectory
    """
    logger.info("[%s] Generating synthetic data...", name)
    
    active_species = config["active_species"]
    active_indices = config["active_indices"]
    n_species = len(active_species)
    
    # Create solver
    solver_kwargs = {
        k: v for k, v in config.items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    
    solver = BiofilmNewtonSolver(
        **solver_kwargs,
        active_species=active_species,
        use_numba=HAS_NUMBA,
    )
    
    tsm = BiofilmTSM(
        solver,
        active_theta_indices=active_indices,
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
    )
    
    # Solve TSM
    start = time.time()
    t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
    elapsed = time.time() - start
    logger.info("TSM computation time: %.2fs", elapsed)
    
    # Plot simulation
    plot_mgr.plot_TSM_simulation(t_arr, x0, active_species, name)
    
    # Generate sparse observations
    phibar = compute_phibar(x0, active_species)
    idx_sparse = select_sparse_data_indices(len(t_arr), exp_config.n_data)
    
    np.random.seed(exp_config.random_seed)
    data = np.zeros((exp_config.n_data, n_species))
    
    for i, idx in enumerate(idx_sparse):
        idx = int(np.clip(idx, 0, sig2.shape[0] - 1))

        mu = phibar[idx]
        var = np.zeros(n_species)
        
        # ===== ADD THIS (TSM solve の直後に1回だけ) =====
        n_state = x0.shape[1]                 # = 10
        n_total_species = (n_state - 2) // 2  # = 4
        psi_offset = n_total_species + 1      # = 5
        # ===============================================

        for j, sp in enumerate(active_species):
            var[j] = (
                x0[idx, sp]**2 * sig2[idx, psi_offset + sp]
                + x0[idx, psi_offset + sp]**2 * sig2[idx, sp]
            )

        
        std = np.sqrt(var + exp_config.sigma_obs**2)
        data[i] = np.random.normal(mu, std)
    
    logger.info("Data shape: %s", data.shape)
    
    # Plot with data
    plot_mgr.plot_TSM_simulation(t_arr, x0, active_species, name, data, idx_sparse)
    
    return data, idx_sparse, t_arr, x0, sig2


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    """Main execution function."""
    
    # ===== Setup =====
    setup_logging("INFO")
    patch_biofilm_solver()
    logger.info("%s", "=" * 80)
    logger.info("Case II: Hierarchical Bayesian Parameter Estimation (Aligned with Paper)")
    logger.info("%s", "=" * 80)
    logger.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Numba: %s", "enabled" if HAS_NUMBA else "disabled")
    
    # Configuration
    exp_config = ExperimentConfig()
    mcmc_config = MCMCConfig()
    
    # Create output directory
    output_dir = Path(exp_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize plot manager
    plot_mgr = PlotManager(exp_config.output_dir)
    
    # Get TRUE parameters from authoritative file
    theta_true = get_theta_true()
    
    logger.info("Configuration:")
    logger.info("True θ = %s", theta_true)
    logger.info("MCMC samples per chain = %s", mcmc_config.n_samples)
    logger.info("Number of chains = %s", mcmc_config.n_chains)
    logger.info("Output directory = %s/", exp_config.output_dir)
    
    # ===== STEP 1: Generate Synthetic Data =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 1: Synthetic Data Generation")
    logger.info("%s", "=" * 80)
    
    data_M1, idx_M1, t_M1, x0_M1, sig2_M1 = generate_synthetic_data(
        MODEL_CONFIGS["M1"], theta_true, exp_config, "M1", plot_mgr
    )
    
    data_M2, idx_M2, t_M2, x0_M2, sig2_M2 = generate_synthetic_data(
        MODEL_CONFIGS["M2"], theta_true, exp_config, "M2", plot_mgr
    )
    
    data_M3, idx_M3, t_M3, x0_M3, sig2_M3 = generate_synthetic_data(
        MODEL_CONFIGS["M3"], theta_true, exp_config, "M3", plot_mgr
    )
    
    logger.info("Data generation complete")
    
    # ===== STEP 2: M1 Bayesian Updating =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 2: M1 Bayesian Updating (Multi-chain MCMC)")
    logger.info("%s", "=" * 80)
    
    solver_kwargs_M1 = {
        k: v for k, v in MODEL_CONFIGS["M1"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    
    def make_evaluator_M1():
        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M1,
            active_species=MODEL_CONFIGS["M1"]["active_species"],
            active_indices=MODEL_CONFIGS["M1"]["active_indices"],
            theta_base=theta_true,
            data=data_M1,
            idx_sparse=idx_M1,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
        )
    
    prior_bounds_M1 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M1"]["param_names"])
    
    start_M1 = time.time()
    chains_M1, logL_M1, MAP_M1, diag_M1 = run_multi_chain_MCMC(
        "M1", make_evaluator_M1, prior_bounds_M1, mcmc_config
    )
    time_M1 = time.time() - start_M1
    
    samples_M1 = np.concatenate(chains_M1, axis=0)
    logL_M1_all = np.concatenate(logL_M1, axis=0)
    results_M1 = compute_MAP_with_uncertainty(samples_M1, logL_M1_all)
    mean_M1 = results_M1["mean"]
    
    logger.info("[M1] Results:")
    logger.info("Computation time: %.2f min", time_M1 / 60.0)
    logger.info("MAP: %s", MAP_M1)
    logger.info("Mean: %s", mean_M1)
    logger.info("True: %s", theta_true[0:5])
    logger.info("MAP error: %.6f", np.linalg.norm(MAP_M1 - theta_true[0:5]))
    
    plot_mgr.plot_posterior(
        samples_M1, theta_true[0:5],
        MODEL_CONFIGS["M1"]["param_names"], "M1", MAP_M1, mean_M1
    )
    
    # ===== STEP 3: M2 Bayesian Updating =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 3: M2 Bayesian Updating (Multi-chain MCMC)")
    logger.info("%s", "=" * 80)
    
    solver_kwargs_M2 = {
        k: v for k, v in MODEL_CONFIGS["M2"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    
    def make_evaluator_M2():
        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M2,
            active_species=MODEL_CONFIGS["M2"]["active_species"],
            active_indices=MODEL_CONFIGS["M2"]["active_indices"],
            theta_base=theta_true,
            data=data_M2,
            idx_sparse=idx_M2,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
        )
    
    prior_bounds_M2 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M2"]["param_names"])
    
    start_M2 = time.time()
    chains_M2, logL_M2, MAP_M2, diag_M2 = run_multi_chain_MCMC(
        "M2", make_evaluator_M2, prior_bounds_M2, mcmc_config
    )
    time_M2 = time.time() - start_M2
    
    samples_M2 = np.concatenate(chains_M2, axis=0)
    logL_M2_all = np.concatenate(logL_M2, axis=0)
    results_M2 = compute_MAP_with_uncertainty(samples_M2, logL_M2_all)
    mean_M2 = results_M2["mean"]
    
    logger.info("[M2] Results:")
    logger.info("Computation time: %.2f min", time_M2 / 60.0)
    logger.info("MAP: %s", MAP_M2)
    logger.info("Mean: %s", mean_M2)
    logger.info("True: %s", theta_true[5:10])
    logger.info("MAP error: %.6f", np.linalg.norm(MAP_M2 - theta_true[5:10]))
    
    plot_mgr.plot_posterior(
        samples_M2, theta_true[5:10],
        MODEL_CONFIGS["M2"]["param_names"], "M2", MAP_M2, mean_M2
    )
    
    # ===== STEP 4: M3 Two-Stage Bayesian Updating =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 4: M3 Two-Stage Bayesian Updating")
    logger.info("%s", "=" * 80)
    
    # Use MAP estimates from M1 and M2 as base
    theta_base_M3 = theta_true.copy()
    theta_base_M3[0:5] = MAP_M1
    theta_base_M3[5:10] = MAP_M2
    
    solver_kwargs_M3 = {
        k: v for k, v in MODEL_CONFIGS["M3"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    
    def make_evaluator_M3():
        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M3,
            active_species=MODEL_CONFIGS["M3"]["active_species"],
            active_indices=MODEL_CONFIGS["M3"]["active_indices"],
            theta_base=theta_base_M3,
            data=data_M3,
            idx_sparse=idx_M3,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
        )
    
    prior_bounds_M3 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M3"]["param_names"])
    
    # ---- Phase 1: Coarse MCMC ----
    logger.info("[M3] Phase 1: Coarse MCMC (diagonal proposal)...")
    
    mcmc_config_p1 = MCMCConfig(
        n_samples=mcmc_config.n_samples // 2,
        n_burn_in=mcmc_config.n_burn_in // 2,
        n_chains=mcmc_config.n_chains,
        initial_scale=0.10,
    )
    
    chains_M3_p1, logL_M3_p1, MAP_M3_p1, _ = run_multi_chain_MCMC(
        "M3_phase1", make_evaluator_M3, prior_bounds_M3, mcmc_config_p1
    )
    
    samples_M3_p1 = np.concatenate(chains_M3_p1, axis=0)
    logger.info("Phase 1 MAP: %s", MAP_M3_p1)
    
    # ---- Phase 2: Refined MCMC with Covariance ----
    logger.info("[M3] Phase 2: Refined MCMC (covariance proposal)...")
    
    cov_M3 = np.cov(samples_M3_p1.T)
    proposal_cov_M3 = cov_M3 + 1e-6 * np.eye(len(MODEL_CONFIGS["M3"]["param_names"]))
    
    start_M3 = time.time()
    chains_M3, logL_M3, MAP_M3, diag_M3 = run_multi_chain_MCMC(
        "M3_refined", make_evaluator_M3, prior_bounds_M3, mcmc_config,
        proposal_cov=proposal_cov_M3
    )
    time_M3 = time.time() - start_M3
    
    samples_M3 = np.concatenate(chains_M3, axis=0)
    logL_M3_all = np.concatenate(logL_M3, axis=0)
    results_M3 = compute_MAP_with_uncertainty(samples_M3, logL_M3_all)
    mean_M3 = results_M3["mean"]
    
    logger.info("[M3 Refined] Results:")
    logger.info("Computation time: %.2f min", time_M3 / 60.0)
    logger.info("MAP refined: %s", MAP_M3)
    logger.info("Mean refined: %s", mean_M3)
    logger.info("True: %s", theta_true[10:14])
    logger.info("MAP error: %.6f", np.linalg.norm(MAP_M3 - theta_true[10:14]))
    
    plot_mgr.plot_posterior(
        samples_M3, theta_true[10:14],
        MODEL_CONFIGS["M3"]["param_names"], "M3_refined", MAP_M3, mean_M3
    )
    
    # ===== STEP 5: Final Summary =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 5: Final Summary")
    logger.info("%s", "=" * 80)
    
    theta_MAP_full = theta_true.copy()
    theta_MAP_full[0:5] = MAP_M1
    theta_MAP_full[5:10] = MAP_M2
    theta_MAP_full[10:14] = MAP_M3
    
    theta_mean_full = theta_true.copy()
    theta_mean_full[0:5] = mean_M1
    theta_mean_full[5:10] = mean_M2
    theta_mean_full[10:14] = mean_M3
    
    param_names_all = (
        MODEL_CONFIGS["M1"]["param_names"]
        + MODEL_CONFIGS["M2"]["param_names"]
        + MODEL_CONFIGS["M3"]["param_names"]
    )
    
    logger.info("Final Parameters:")
    logger.info("%s", "=" * 80)
    logger.info("%s", f"{'Param':<8} {'True':<12} {'MAP':<12} {'Mean':<12}")
    logger.info("%s", "-" * 80)
    
    for i, name in enumerate(param_names_all):
        logger.info("%s", f"{name:<8} {theta_true[i]:<12.6f} {theta_MAP_full[i]:<12.6f} {theta_mean_full[i]:<12.6f}")
    
    total_map_error = np.linalg.norm(theta_MAP_full - theta_true)
    total_mean_error = np.linalg.norm(theta_mean_full - theta_true)
    
    logger.info("Total Parameter Error:")
    logger.info("MAP error: %.6f", total_map_error)
    logger.info("Mean error: %.6f", total_mean_error)
    
    total_time = (time_M1 + time_M2 + time_M3) / 60.0
    logger.info("Total computation time: %.2f min", total_time)
    
    # Final comparison plot
    plot_mgr.plot_parameter_comparison(theta_true, theta_MAP_full, theta_mean_full, param_names_all)
    
    # ===== Save Results =====
    logger.info("Saving results...")
    
    np.savez(
        output_dir / "results_MAP.npz",
        theta_true=theta_true,
        theta_MAP_full=theta_MAP_full,
        theta_mean_full=theta_mean_full,
        MAP_M1=MAP_M1, MAP_M2=MAP_M2, MAP_M3=MAP_M3,
        mean_M1=mean_M1, mean_M2=mean_M2, mean_M3=mean_M3,
        samples_M1=samples_M1, samples_M2=samples_M2, samples_M3=samples_M3,
    )
    
    logger.info("Results saved to: %s/results_MAP.npz", output_dir)
    
    # ===== Completion =====
    logger.info("%s", "=" * 80)
    logger.info("Case II Complete!")
    logger.info("%s", "=" * 80)
    logger.info("Summary:")
    logger.info("Total parameter error (MAP): %.6f", total_map_error)
    logger.info("Total parameter error (Mean): %.6f", total_mean_error)
    logger.info("Total computation time: %.2f min", total_time)
    logger.info("Generated %s figures in %s/", len(plot_mgr.generated_figs), output_dir)
    logger.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
