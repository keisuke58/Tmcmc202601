"""
Adaptive MCMC and 2-phase MCMC implementations.

Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path

try:
    from tmcmc.core.evaluator import LogLikelihoodEvaluator
except ImportError:
    try:
        from core.evaluator import LogLikelihoodEvaluator
    except ImportError:
        try:
            from .evaluator import LogLikelihoodEvaluator
        except ImportError:
            # Fallback: add current directory to path
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from evaluator import LogLikelihoodEvaluator

logger = logging.getLogger(__name__)


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
    Adaptive random-walk Metropolis-Hastings MCMC.
    
    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function
    prior_bounds : List[Tuple[float, float]]
        (lower, upper) bounds for each parameter
    n_samples : int
        Number of samples to generate
    initial_scale : float
        Initial proposal scale
    burn_in : int
        Number of burn-in samples
    target_accept : float
        Target acceptance rate for adaptation
    adapt_start : int
        Start adaptation after this many samples
    adapt_interval : int
        Adapt every N samples
    proposal_cov : np.ndarray, optional
        Fixed proposal covariance matrix
    seed : int, optional
        Random seed
        
    Returns
    -------
    samples : np.ndarray
        MCMC samples (after burn-in)
    logL_values : np.ndarray
        Log-posterior values (after burn-in)
    theta_MAP : np.ndarray
        MAP estimate
    acceptance_rate : float
        Acceptance rate
    """
    # CRITICAL FIX: Use default_rng consistently
    rng = np.random.default_rng(seed)
    
    n_params = len(prior_bounds)
    # Important: Initialize at center ± epsilon
    theta_center = np.array([(low + high) / 2 for low, high in prior_bounds])
    epsilon = initial_scale

    theta_current = theta_center + rng.standard_normal(n_params) * epsilon

    # Force into prior bounds
    for i, (low, high) in enumerate(prior_bounds):
        theta_current[i] = np.clip(theta_current[i], low, high)

    proposal_std = np.full(n_params, initial_scale)

    def log_prior(theta: np.ndarray) -> float:
        for i, (low, high) in enumerate(prior_bounds):
            if not (low <= theta[i] <= high):
                return -np.inf
        return 0.0
    
    def log_posterior(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if np.isinf(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    log_post_current = log_posterior(theta_current)
    
    samples_all = np.zeros((n_samples, n_params))
    logpost_all = np.zeros(n_samples)  # Store log-posterior, not log-likelihood
    n_accepted = 0
    
    logger.info("      [MCMC] Initial log posterior = %.2f", log_post_current)
    
    for i in range(n_samples):
        if proposal_cov is None:
            eps = rng.standard_normal(n_params) * proposal_std
        else:
            eps = rng.multivariate_normal(np.zeros(n_params), proposal_cov)
        
        theta_proposed = theta_current + eps
        log_post_proposed = log_posterior(theta_proposed)
        log_alpha = log_post_proposed - log_post_current
        
        if np.log(rng.random()) < log_alpha:
            theta_current = theta_proposed
            log_post_current = log_post_proposed
            n_accepted += 1
        
        samples_all[i] = theta_current
        logpost_all[i] = log_post_current  # Store log-posterior
        
        if (i + 1) % 500 == 0:
            acc_rate = n_accepted / (i + 1)
            logger.info("      %s/%s samples, acceptance: %.1f%%", i + 1, n_samples, acc_rate * 100.0)
        
        if proposal_cov is None:
            if (i + 1) >= adapt_start and (i + 1) % adapt_interval == 0:
                acc_rate = n_accepted / (i + 1)
                adjustment = np.exp(0.5 * (acc_rate - target_accept))
                proposal_std *= adjustment
                proposal_std = np.clip(proposal_std, 1e-4, 1.0)
    
    samples = samples_all[burn_in:]
    logL_values = logpost_all[burn_in:]  # Use renamed variable (log-posterior values)
    acceptance_rate = n_accepted / n_samples
    
    idx_MAP = np.argmax(logL_values)
    theta_MAP = samples[idx_MAP]
    
    logger.info("      [MCMC] Complete. Acceptance rate: %.1f%%", acceptance_rate * 100.0)
    logger.info("      [MCMC] MAP: %s", theta_MAP)
    
    return samples, logL_values, theta_MAP, acceptance_rate


def run_two_phase_MCMC_with_linearization(
    model_tag: str,
    make_evaluator: callable,
    prior_bounds: List[Tuple[float, float]],
    mcmc_config: Any,  # MCMCConfig (imported locally to avoid circular dependency)
    theta_base: np.ndarray,
    active_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Run 2-phase MCMC with TSM linearization point update.
    
    CRITICAL ALGORITHM:
    1. Phase 1: Rough MCMC with initial linearization (prior mean)
    2. Update linearization point to MAP from Phase 1
    3. Phase 2: Refined MCMC with improved TSM approximation
    
    Parameters
    ----------
    model_tag : str
        Model identifier
    make_evaluator : callable
        Factory function that creates LogLikelihoodEvaluator
        (takes theta_linearization as argument)
    prior_bounds : list
        Parameter bounds
    mcmc_config : MCMCConfig
        MCMC configuration
    theta_base : ndarray (14,)
        Base parameter vector
    active_indices : list
        Indices of active parameters
    
    Returns
    -------
    samples_phase1, samples_phase2 : ndarray
        Samples from both phases
    MAP_phase1, MAP_phase2 : ndarray
        MAP estimates from both phases
    diagnostics : dict
        Combined diagnostics
    """
    # Import locally to avoid circular dependency
    from case2_tmcmc_linearization import run_multi_chain_MCMC, MCMCConfig
    
    logger.info("%s", "=" * 70)
    logger.info("[%s] 2-Phase MCMC with Linearization Update", model_tag)
    logger.info("%s", "=" * 70)
    
    # ===== PHASE 1: Rough MCMC with initial linearization =====
    logger.info("%s", "─" * 50)
    logger.info("PHASE 1: Initial MCMC (linearization at prior mean)")
    logger.info("%s", "─" * 50)
    
    # Initial linearization at prior center
    theta_lin_init = theta_base.copy()
    for idx in active_indices:
        theta_lin_init[idx] = (prior_bounds[0][0] + prior_bounds[0][1]) / 2
    
    mcmc_config_p1 = MCMCConfig(
        n_samples=mcmc_config.n_samples // 2,
        n_burn_in=mcmc_config.n_burn_in // 2,
        n_chains=mcmc_config.n_chains,
        initial_scale=0.10,
    )
    
    def make_evaluator_p1():
        return make_evaluator(theta_linearization=theta_lin_init)
    
    chains_p1, logL_p1, MAP_p1, diag_p1 = run_multi_chain_MCMC(
        f"{model_tag}_Phase1", make_evaluator_p1, prior_bounds, mcmc_config_p1
    )
    
    samples_p1 = np.concatenate(chains_p1, axis=0)
    
    logger.info("Phase 1 MAP: %s", MAP_p1)
    logger.info("Phase 1 ||θ - θ_lin||: %.6f", np.linalg.norm(MAP_p1 - theta_lin_init[active_indices]))
    
    # ===== UPDATE LINEARIZATION POINT =====
    logger.info("%s", "─" * 50)
    logger.info("UPDATING LINEARIZATION POINT → Phase 1 MAP")
    logger.info("%s", "─" * 50)
    
    # Construct full parameter vector for new linearization
    theta_lin_new = theta_base.copy()
    for i, idx in enumerate(active_indices):
        theta_lin_new[idx] = MAP_p1[i]
    
    logger.info("Old θ₀: %s", theta_lin_init[active_indices])
    logger.info("New θ₀: %s", theta_lin_new[active_indices])
    logger.info("||Δθ₀||: %.6f", np.linalg.norm(theta_lin_new - theta_lin_init))
    
    # ===== PHASE 2: Refined MCMC with updated linearization =====
    logger.info("%s", "─" * 50)
    logger.info("PHASE 2: Refined MCMC (linearization at Phase 1 MAP)")
    logger.info("%s", "─" * 50)
    
    # Use covariance from Phase 1 as proposal
    cov_p1 = np.cov(samples_p1.T)
    proposal_cov = cov_p1 + 1e-6 * np.eye(len(prior_bounds))
    
    def make_evaluator_p2():
        return make_evaluator(theta_linearization=theta_lin_new)
    
    chains_p2, logL_p2, MAP_p2, diag_p2 = run_multi_chain_MCMC(
        f"{model_tag}_Phase2", make_evaluator_p2, prior_bounds, mcmc_config,
        proposal_cov=proposal_cov
    )
    
    samples_p2 = np.concatenate(chains_p2, axis=0)
    
    logger.info("Phase 2 MAP: %s", MAP_p2)
    logger.info("Phase 2 ||θ - θ_lin||: %.6f", np.linalg.norm(MAP_p2 - theta_lin_new[active_indices]))
    
    # ===== SUMMARY =====
    logger.info("%s", "─" * 50)
    logger.info("2-Phase MCMC Summary")
    logger.info("%s", "─" * 50)
    
    delta_MAP = np.linalg.norm(MAP_p2 - MAP_p1)
    logger.info("||MAP_p2 - MAP_p1||: %.6f", delta_MAP)
    
    if delta_MAP < 0.01:
        logger.info("Excellent convergence: small change after linearization update")
    elif delta_MAP < 0.05:
        logger.info("Good convergence")
    else:
        logger.warning("Significant change - consider additional iteration")
    
    diagnostics = {
        "phase1": diag_p1,
        "phase2": diag_p2,
        "MAP_phase1": MAP_p1,
        "MAP_phase2": MAP_p2,
        "delta_MAP": delta_MAP,
        "theta_lin_init": theta_lin_init[active_indices],
        "theta_lin_updated": theta_lin_new[active_indices],
    }
    
    return samples_p1, samples_p2, MAP_p1, MAP_p2, diagnostics
