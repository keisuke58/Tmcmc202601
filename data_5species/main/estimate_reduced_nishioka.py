#!/usr/bin/env python3
"""
Parameter Estimation for 5-Species Biofilm Model (Nishioka Algorithm).

This script loads experimental biofilm data and estimates model parameters
using TMCMC (Transitional Markov Chain Monte Carlo).

Improvements (v2):
1. Checkpoint & Resume - Save/restore TMCMC state for long runs
2. Enhanced Diagnostics - Gelman-Rubin R-hat, Effective Sample Size (ESS)
3. Highest Density Intervals - Better uncertainty quantification for skewed posteriors
6. Model Evidence - Marginal likelihood estimation for model comparison

Improvements (v3):
5. Prior Predictive Checks - Sample from priors to verify coverage
9. Batch Processing - Run multiple conditions (all or selected)
11. WAIC / PSIS-LOO-CV - Information criteria for model comparison
14. Parameter Correlation Analysis - Identify correlated parameters

Experimental Data:
- Condition: Commensal / Dysbiotic
- Cultivation: Static / HOBIC
- Timepoints: Day 1, 3, 6, 10, 15, 21
- Measurements: Total biofilm volume + species distribution percentages

Species Mapping (from experimental data colors to model indices):
- Blue (S. oralis) -> Species 0
- Green (A. naeslundii) -> Species 1
- Yellow (V. dispar) -> Species 2
- Purple (F. nucleatum) -> Species 3
- Red (P. gingivalis) -> Species 4

Usage:
    python estimate_reduced_nishioka.py [options]

    # Resume from checkpoint
    python estimate_reduced_nishioka.py --resume-from _runs/previous_run/checkpoints

    # Batch processing - run all 4 conditions
    python estimate_reduced_nishioka.py --batch all

    # Batch processing - run specific conditions
    python estimate_reduced_nishioka.py --batch "Commensal:Static,Dysbiotic:HOBIC"

    # Prior predictive checks
    python estimate_reduced_nishioka.py --prior-predictive 100
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import time
import pickle
import os
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /home/.../Tmcmc202601
DATA_5SPECIES_ROOT = Path(__file__).parent.parent  # /home/.../Tmcmc202601/data_5species

# Add data_5species folder (contains core, debug, utils, visualization)
sys.path.insert(0, str(DATA_5SPECIES_ROOT))
# Add config location (program2602) for config.py
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
# Add tmcmc folder for improved_5species_jit.py
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
# Add project root
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    setup_logging,
    DebugConfig,
    DebugLevel,
    PRIOR_BOUNDS_DEFAULT,
)
from core import (
    LogLikelihoodEvaluator,
    run_multi_chain_TMCMC,
    compute_MAP_with_uncertainty,
    build_likelihood_weights,
    build_species_sigma,
)
from debug import DebugLogger
from utils import save_json, save_npy
from visualization import PlotManager, compute_fit_metrics, compute_phibar
from improved_5species_jit import BiofilmNewtonSolver5S

try:
    from data_5species.core.nishioka_model import (
        get_nishioka_bounds,
        get_condition_bounds,
        get_model_constants,
    )
except ImportError:
    from core.nishioka_model import get_condition_bounds, get_model_constants

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# IMPROVEMENT 1: CHECKPOINT & RESUME SUPPORT
# =============================================================================


class TMCMCCheckpointManager:
    """
    Manages checkpoints for TMCMC runs to enable resume after interruption.

    Saves:
    - Current stage number
    - All chain states (particles, weights, logL)
    - Beta schedule history
    - Diagnostics accumulated so far
    - Random state for reproducibility
    """

    def __init__(self, checkpoint_dir: Path, save_every: int = 5):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N stages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.checkpoint_file = self.checkpoint_dir / "tmcmc_checkpoint.pkl"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"

    def save_checkpoint(
        self,
        stage: int,
        chains: List[np.ndarray],
        logL: List[np.ndarray],
        beta_schedule: List[float],
        diagnostics: Dict[str, Any],
        random_state: Any,
        config: Dict[str, Any],
    ):
        """Save current TMCMC state to checkpoint."""
        checkpoint_data = {
            "stage": stage,
            "chains": [c.copy() for c in chains],
            "logL": [l.copy() for l in logL],
            "beta_schedule": (
                beta_schedule.copy() if isinstance(beta_schedule, list) else list(beta_schedule)
            ),
            "diagnostics": diagnostics,
            "random_state": random_state,
            "timestamp": datetime.now().isoformat(),
        }

        # Save binary checkpoint
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # Save human-readable metadata
        metadata = {
            "stage": stage,
            "n_chains": len(chains),
            "n_particles_per_chain": [len(c) for c in chains],
            "current_beta": beta_schedule[-1] if beta_schedule else 0.0,
            "timestamp": checkpoint_data["timestamp"],
            "config": config,
        }
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved at stage {stage} (beta={beta_schedule[-1]:.4f})")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            logger.info(f"Loaded checkpoint from stage {checkpoint_data['stage']}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def should_save(self, stage: int) -> bool:
        """Check if we should save at this stage."""
        return stage > 0 and stage % self.save_every == 0

    def clear_checkpoints(self):
        """Remove checkpoint files (call after successful completion)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Checkpoints cleared after successful completion")


# =============================================================================
# IMPROVEMENT 2: ENHANCED CONVERGENCE DIAGNOSTICS
# =============================================================================


def compute_gelman_rubin_rhat(chains: List[np.ndarray]) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat convergence diagnostic.

    R-hat measures the ratio of between-chain variance to within-chain variance.
    Values close to 1.0 indicate convergence. R-hat > 1.1 suggests non-convergence.

    Args:
        chains: List of (n_samples, n_params) arrays, one per chain

    Returns:
        R-hat values for each parameter (n_params,)
    """
    n_chains = len(chains)
    if n_chains < 2:
        logger.warning("R-hat requires at least 2 chains. Returning NaN.")
        return np.full(chains[0].shape[1], np.nan)

    # Ensure all chains have the same length (use minimum)
    min_len = min(len(c) for c in chains)
    chains_trimmed = [c[:min_len] for c in chains]

    n_samples = min_len
    n_params = chains_trimmed[0].shape[1]

    # Stack chains: (n_chains, n_samples, n_params)
    stacked = np.array(chains_trimmed)

    # Chain means: (n_chains, n_params)
    chain_means = stacked.mean(axis=1)

    # Overall mean: (n_params,)
    overall_mean = chain_means.mean(axis=0)

    # Between-chain variance: B = n * var(chain_means)
    B = n_samples * np.var(chain_means, axis=0, ddof=1)

    # Within-chain variance: W = mean of chain variances
    chain_vars = np.var(stacked, axis=1, ddof=1)  # (n_chains, n_params)
    W = chain_vars.mean(axis=0)

    # Pooled variance estimate
    var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

    # R-hat
    rhat = np.sqrt(var_plus / (W + 1e-10))

    return rhat


def compute_effective_sample_size(samples: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute Effective Sample Size (ESS) accounting for autocorrelation.

    ESS represents the number of independent samples equivalent to the
    autocorrelated chain. Low ESS indicates high autocorrelation.

    Args:
        samples: (n_samples, n_params) array
        max_lag: Maximum lag for autocorrelation (default: n_samples // 2)

    Returns:
        ESS for each parameter (n_params,)
    """
    n_samples, n_params = samples.shape

    if max_lag is None:
        max_lag = min(n_samples // 2, 1000)

    ess = np.zeros(n_params)

    for p in range(n_params):
        x = samples[:, p]
        x_centered = x - x.mean()

        # Compute autocorrelation using FFT (faster)
        n = len(x_centered)
        fft_x = np.fft.fft(x_centered, n=2 * n)
        acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real
        acf = acf / acf[0]  # Normalize

        # Sum autocorrelations until they become negative (Geyer's initial monotone)
        # Use initial positive sequence estimator
        rho_sum = 0.0
        for lag in range(1, min(max_lag, n)):
            if acf[lag] < 0:
                break
            rho_sum += acf[lag]

        # ESS = n / (1 + 2 * sum of autocorrelations)
        tau = 1 + 2 * rho_sum
        ess[p] = n / max(tau, 1.0)

    return ess


def compute_mcmc_diagnostics(
    chains: List[np.ndarray], logL: List[np.ndarray], param_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive MCMC diagnostics.

    Args:
        chains: List of chains, each (n_samples, n_params)
        logL: List of log-likelihood arrays
        param_names: Parameter names for reporting

    Returns:
        Dictionary with R-hat, ESS, and summary statistics
    """
    # Combine all chains for ESS calculation
    all_samples = np.concatenate(chains, axis=0)
    all_logL = np.concatenate(logL, axis=0)

    n_params = all_samples.shape[1]
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(n_params)]

    # Compute diagnostics
    rhat = compute_gelman_rubin_rhat(chains)
    ess = compute_effective_sample_size(all_samples)

    # Summary statistics
    means = all_samples.mean(axis=0)
    stds = all_samples.std(axis=0)

    # Check convergence
    rhat_ok = np.all(rhat < 1.1)
    ess_ok = np.all(ess > 100)  # Minimum ESS threshold

    diagnostics = {
        "rhat": rhat,
        "ess": ess,
        "means": means,
        "stds": stds,
        "n_samples_total": len(all_samples),
        "n_chains": len(chains),
        "rhat_max": float(np.nanmax(rhat)),
        "ess_min": float(np.nanmin(ess)),
        "converged_rhat": bool(rhat_ok),
        "converged_ess": bool(ess_ok),
        "converged": bool(rhat_ok and ess_ok),
        "param_names": param_names,
    }

    # Per-parameter diagnostics
    per_param = []
    for i, name in enumerate(param_names):
        per_param.append(
            {
                "name": name,
                "mean": float(means[i]),
                "std": float(stds[i]),
                "rhat": float(rhat[i]) if not np.isnan(rhat[i]) else None,
                "ess": float(ess[i]),
            }
        )
    diagnostics["per_parameter"] = per_param

    return diagnostics


# =============================================================================
# IMPROVEMENT 3: HIGHEST DENSITY INTERVALS (HDI)
# =============================================================================


def compute_hdi(samples: np.ndarray, credibility: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Highest Density Interval (HDI) for posterior samples.

    HDI is the narrowest interval containing the specified probability mass.
    Unlike equal-tailed intervals, HDI is optimal for skewed distributions.

    Args:
        samples: (n_samples, n_params) array or (n_samples,) for 1D
        credibility: Credibility level (default 0.95 for 95% HDI)

    Returns:
        hdi_lower: Lower bounds of HDI
        hdi_upper: Upper bounds of HDI
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n_samples, n_params = samples.shape
    hdi_lower = np.zeros(n_params)
    hdi_upper = np.zeros(n_params)

    # Number of samples to include in HDI
    n_in_hdi = int(np.ceil(credibility * n_samples))

    for p in range(n_params):
        # Sort samples
        sorted_samples = np.sort(samples[:, p])

        # Find the narrowest interval containing n_in_hdi samples
        # by sliding a window of size n_in_hdi
        interval_widths = sorted_samples[n_in_hdi:] - sorted_samples[:-n_in_hdi]

        if len(interval_widths) == 0:
            # All samples in interval
            hdi_lower[p] = sorted_samples[0]
            hdi_upper[p] = sorted_samples[-1]
        else:
            # Find minimum width interval
            min_idx = np.argmin(interval_widths)
            hdi_lower[p] = sorted_samples[min_idx]
            hdi_upper[p] = sorted_samples[min_idx + n_in_hdi]

    return hdi_lower, hdi_upper


def compute_credible_intervals(
    samples: np.ndarray, credibility: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Compute both HDI and equal-tailed credible intervals.

    Args:
        samples: (n_samples, n_params) array
        credibility: Credibility level

    Returns:
        Dictionary with HDI and equal-tailed intervals
    """
    alpha = 1 - credibility

    # HDI
    hdi_lower, hdi_upper = compute_hdi(samples, credibility)

    # Equal-tailed (percentile-based)
    et_lower = np.percentile(samples, 100 * alpha / 2, axis=0)
    et_upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

    # Median and mode (MAP)
    median = np.median(samples, axis=0)

    return {
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "et_lower": et_lower,
        "et_upper": et_upper,
        "median": median,
        "credibility": credibility,
    }


# =============================================================================
# IMPROVEMENT 6: MODEL EVIDENCE (MARGINAL LIKELIHOOD)
# =============================================================================


def estimate_log_evidence_harmonic_mean(
    logL: np.ndarray, stabilize: bool = True
) -> Tuple[float, float]:
    """
    Estimate log marginal likelihood using the harmonic mean estimator.

    WARNING: This estimator can have infinite variance. Use with caution
    and prefer thermodynamic integration when available.

    The harmonic mean estimator:
        p(D) ≈ 1 / mean(1/L(θ|D)) for θ ~ p(θ|D)

    Args:
        logL: Log-likelihood values from posterior samples
        stabilize: Use stabilized version to reduce variance

    Returns:
        log_evidence: Estimated log marginal likelihood
        log_evidence_se: Standard error estimate
    """
    n = len(logL)

    if stabilize:
        # Use truncated harmonic mean (more stable)
        # Truncate at median to reduce variance from low-likelihood samples
        logL_median = np.median(logL)
        mask = logL >= logL_median
        logL_truncated = logL[mask]

        # Harmonic mean in log space
        # log(1/mean(exp(-logL))) = -log(mean(exp(-logL)))
        # Use logsumexp for numerical stability
        log_inv_L = -logL_truncated
        log_mean_inv_L = np.logaddexp.reduce(log_inv_L) - np.log(len(logL_truncated))
        log_evidence = -log_mean_inv_L

        # Rough SE estimate
        se = np.std(logL_truncated) / np.sqrt(len(logL_truncated))
    else:
        log_inv_L = -logL
        log_mean_inv_L = np.logaddexp.reduce(log_inv_L) - np.log(n)
        log_evidence = -log_mean_inv_L
        se = np.std(logL) / np.sqrt(n)

    return float(log_evidence), float(se)


def estimate_log_evidence_thermodynamic(
    beta_schedule: List[float], mean_logL_per_stage: List[float]
) -> Tuple[float, float]:
    """
    Estimate log marginal likelihood using thermodynamic integration.

    This is the preferred method when TMCMC beta schedule is available.

    log p(D) = ∫₀¹ E[log p(D|θ)] dβ

    where the expectation is over p(θ|D)^β × p(θ).

    Args:
        beta_schedule: List of beta values from TMCMC [0, β₁, β₂, ..., 1]
        mean_logL_per_stage: Mean log-likelihood at each TMCMC stage

    Returns:
        log_evidence: Estimated log marginal likelihood
        log_evidence_se: Standard error (rough estimate)
    """
    if len(beta_schedule) != len(mean_logL_per_stage):
        raise ValueError("beta_schedule and mean_logL_per_stage must have same length")

    betas = np.array(beta_schedule)
    mean_logL = np.array(mean_logL_per_stage)

    # Trapezoidal integration
    log_evidence = np.trapz(mean_logL, betas)

    # Rough SE estimate based on integration error
    # Use Simpson's rule difference as error estimate
    if len(betas) >= 3:
        log_evidence_simpson = np.trapz(mean_logL[::2], betas[::2])
        se = abs(log_evidence - log_evidence_simpson) / 3.0
    else:
        se = np.std(mean_logL) * (betas[-1] - betas[0]) / np.sqrt(len(betas))

    return float(log_evidence), float(se)


def compute_model_evidence(
    chains: List[np.ndarray],
    logL: List[np.ndarray],
    prior_bounds: List[Tuple[float, float]],
    active_indices: List[int],
    beta_schedule: List[float] = None,
    mean_logL_per_stage: List[float] = None,
) -> Dict[str, Any]:
    """
    Compute model evidence (marginal likelihood) using multiple methods.

    Args:
        chains: Posterior samples
        logL: Log-likelihood values
        prior_bounds: Prior bounds for each parameter
        active_indices: Indices of active (non-locked) parameters
        beta_schedule: TMCMC beta schedule (optional, for thermodynamic integration)
        mean_logL_per_stage: Mean logL at each stage (optional)

    Returns:
        Dictionary with evidence estimates and Bayes factor info
    """
    all_logL = np.concatenate(logL, axis=0)

    results = {
        "method": None,
        "log_evidence": None,
        "log_evidence_se": None,
    }

    # Method 1: Thermodynamic Integration (preferred if available)
    if beta_schedule is not None and mean_logL_per_stage is not None:
        try:
            log_ev_ti, se_ti = estimate_log_evidence_thermodynamic(
                beta_schedule, mean_logL_per_stage
            )
            results["thermodynamic"] = {
                "log_evidence": log_ev_ti,
                "se": se_ti,
            }
            results["method"] = "thermodynamic"
            results["log_evidence"] = log_ev_ti
            results["log_evidence_se"] = se_ti
            logger.info(f"Thermodynamic Integration: log(evidence) = {log_ev_ti:.2f} ± {se_ti:.2f}")
        except Exception as e:
            logger.warning(f"Thermodynamic integration failed: {e}")

    # Method 2: Harmonic Mean (fallback)
    try:
        log_ev_hm, se_hm = estimate_log_evidence_harmonic_mean(all_logL, stabilize=True)
        results["harmonic_mean"] = {
            "log_evidence": log_ev_hm,
            "se": se_hm,
        }
        if results["method"] is None:
            results["method"] = "harmonic_mean"
            results["log_evidence"] = log_ev_hm
            results["log_evidence_se"] = se_hm
        logger.info(f"Harmonic Mean: log(evidence) = {log_ev_hm:.2f} ± {se_hm:.2f}")
    except Exception as e:
        logger.warning(f"Harmonic mean estimation failed: {e}")

    # Compute prior volume (for reference)
    active_bounds = [prior_bounds[i] for i in active_indices]
    log_prior_volume = sum(np.log(b[1] - b[0]) for b in active_bounds if b[1] > b[0])
    results["log_prior_volume"] = float(log_prior_volume)
    results["n_params"] = len(active_indices)

    # BIC approximation for comparison
    n_data = len(all_logL)  # Approximate
    max_logL = float(np.max(all_logL))
    bic = -2 * max_logL + len(active_indices) * np.log(n_data)
    results["BIC"] = float(bic)
    results["max_logL"] = max_logL

    return results


# =============================================================================
# IMPROVEMENT 5: PRIOR PREDICTIVE CHECKS
# =============================================================================


def run_prior_predictive_check(
    n_samples: int,
    prior_bounds: List[Tuple[float, float]],
    active_indices: List[int],
    theta_base: np.ndarray,
    solver_kwargs: Dict[str, Any],
    active_species: List[int],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run prior predictive checks by sampling from priors and simulating.

    This helps verify that priors are reasonable by checking if simulations
    from prior samples produce plausible outputs.

    Args:
        n_samples: Number of prior samples to draw
        prior_bounds: Prior bounds for all 20 parameters
        active_indices: Indices of active (non-locked) parameters
        theta_base: Base parameter vector (with locked values)
        solver_kwargs: Solver configuration
        active_species: Species indices to track
        seed: Random seed

    Returns:
        Dictionary with prior samples, simulations, and summary statistics
    """
    np.random.seed(seed)

    logger.info(f"Running prior predictive check with {n_samples} samples...")

    # Sample from priors (uniform)
    n_active = len(active_indices)
    prior_samples = np.zeros((n_samples, n_active))

    for i, idx in enumerate(active_indices):
        low, high = prior_bounds[idx]
        if high > low:
            prior_samples[:, i] = np.random.uniform(low, high, n_samples)
        else:
            prior_samples[:, i] = low  # Locked parameter

    # Create solver
    solver = BiofilmNewtonSolver5S(**solver_kwargs)

    # Run simulations
    successful_sims = []
    failed_count = 0
    final_states = []

    for i in range(n_samples):
        theta_full = theta_base.copy()
        theta_full[active_indices] = prior_samples[i]

        try:
            result = solver.solve(theta_full)
            if len(result) == 2:
                t_sim, y_sim = result
                success = True
            else:
                success, t_sim, y_sim = result

            if success and not np.any(np.isnan(y_sim)) and not np.any(np.isinf(y_sim)):
                successful_sims.append(i)
                # Store final state
                final_state = y_sim[-1, active_species] if y_sim.ndim > 1 else y_sim[active_species]
                final_states.append(final_state)
            else:
                failed_count += 1
        except Exception:
            failed_count += 1

    success_rate = len(successful_sims) / n_samples
    logger.info(
        f"Prior predictive: {len(successful_sims)}/{n_samples} successful ({success_rate*100:.1f}%)"
    )

    # Compute summary statistics
    if len(final_states) > 0:
        final_states = np.array(final_states)
        summary = {
            "mean": final_states.mean(axis=0).tolist(),
            "std": final_states.std(axis=0).tolist(),
            "min": final_states.min(axis=0).tolist(),
            "max": final_states.max(axis=0).tolist(),
            "median": np.median(final_states, axis=0).tolist(),
        }
    else:
        summary = None

    return {
        "n_samples": n_samples,
        "n_successful": len(successful_sims),
        "n_failed": failed_count,
        "success_rate": success_rate,
        "prior_samples": prior_samples,
        "successful_indices": successful_sims,
        "final_states": np.array(final_states) if final_states else None,
        "summary": summary,
    }


def plot_prior_predictive(
    ppc_results: Dict[str, Any],
    output_dir: Path,
    data: np.ndarray = None,
    param_names: List[str] = None,
):
    """Generate prior predictive check plots."""
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Plot 1: Prior samples histogram
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(ppc_results["prior_samples"].shape[1])]

    n_params = len(param_names)
    n_cols = min(5, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(ppc_results["prior_samples"][:, i], bins=30, alpha=0.7, color="steelblue")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.set_title(f"Prior: {name}")

    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / "prior_predictive_samples.png", dpi=150)
    plt.close()

    # Plot 2: Final state distributions
    if ppc_results["final_states"] is not None:
        species_names = ["S.o", "A.n", "Vei", "F.n", "P.g"]
        n_species = ppc_results["final_states"].shape[1]

        fig, axes = plt.subplots(1, n_species, figsize=(3 * n_species, 3))
        if n_species == 1:
            axes = [axes]

        for i in range(n_species):
            ax = axes[i]
            ax.hist(ppc_results["final_states"][:, i], bins=30, alpha=0.7, color="green")
            if data is not None and i < data.shape[1]:
                ax.axvline(data[-1, i], color="red", linestyle="--", linewidth=2, label="Observed")
            ax.set_xlabel(f"Final {species_names[i] if i < len(species_names) else f'Sp{i}'}")
            ax.set_ylabel("Count")

        plt.suptitle(f"Prior Predictive: Final States (n={ppc_results['n_successful']})")
        plt.tight_layout()
        plt.savefig(figures_dir / "prior_predictive_final_states.png", dpi=150)
        plt.close()

    logger.info(f"Prior predictive plots saved to {figures_dir}")


# =============================================================================
# IMPROVEMENT 3 (v3): SLACK NOTIFICATION SYSTEM
# =============================================================================


def load_env_file(env_path: str = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, searches in common locations.

    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Search in common locations
        search_paths = [
            Path(__file__).parent.parent.parent.parent / ".env",  # IKM_Hiwi/.env
            Path.home() / "IKM_Hiwi" / ".env",
            Path.cwd() / ".env",
        ]
        for p in search_paths:
            if p.exists():
                env_path = str(p)
                break

    if env_path is None or not Path(env_path).exists():
        return {}

    env_vars = {}
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


class SlackNotifier:
    """
    Slack notification system for long-running estimation jobs.

    Reads webhook URL from .env file or environment variable.
    """

    def __init__(self, webhook_url: str = None, env_path: str = None):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL. If None, reads from env.
            env_path: Path to .env file
        """
        self.webhook_url = webhook_url

        if self.webhook_url is None:
            # Try environment variable first
            self.webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

        if self.webhook_url is None:
            # Load from .env file
            env_vars = load_env_file(env_path)
            self.webhook_url = env_vars.get("SLACK_WEBHOOK_URL")

        # Check if it's a placeholder
        if self.webhook_url and "YOUR/WEBHOOK/URL" in self.webhook_url:
            logger.warning("Slack webhook URL is a placeholder. Notifications disabled.")
            self.webhook_url = None

        self.enabled = self.webhook_url is not None

        if self.enabled:
            logger.info("Slack notifications enabled")
        else:
            logger.info("Slack notifications disabled (no webhook URL)")

    def send_message(self, message: str, title: str = None) -> bool:
        """
        Send a message to Slack.

        Args:
            message: Message text
            title: Optional title for the message

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            payload = {
                "text": f"*{title}*\n{message}" if title else message,
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 200:
                logger.debug("Slack notification sent successfully")
                return True
            else:
                logger.warning(f"Slack notification failed: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"Slack notification error: {e}")
            return False

    def notify_start(self, condition: str, cultivation: str, n_particles: int, n_stages: int):
        """Send notification when estimation starts."""
        message = (
            f"Condition: `{condition} {cultivation}`\n"
            f"Particles: {n_particles}, Stages: {n_stages}\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message, title="🚀 TMCMC Estimation Started")

    def notify_complete(
        self,
        condition: str,
        cultivation: str,
        elapsed_time: float,
        converged: bool,
        logL_max: float,
        output_dir: str,
    ):
        """Send notification when estimation completes."""
        status = "✅ CONVERGED" if converged else "⚠️ NOT CONVERGED"
        message = (
            f"Condition: `{condition} {cultivation}`\n"
            f"Status: {status}\n"
            f"LogL max: {logL_max:.2f}\n"
            f"Elapsed: {elapsed_time/60:.1f} min\n"
            f"Output: `{output_dir}`"
        )
        self.send_message(message, title="✅ TMCMC Estimation Complete")

    def notify_error(self, condition: str, cultivation: str, error: str):
        """Send notification when estimation fails."""
        message = (
            f"Condition: `{condition} {cultivation}`\n"
            f"Error: {error}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message, title="❌ TMCMC Estimation Failed")

    def notify_batch_complete(self, results: Dict[str, Dict], total_time: float):
        """Send notification when batch processing completes."""
        n_success = sum(1 for r in results.values() if r.get("success", False))
        n_total = len(results)

        status_lines = []
        for name, r in results.items():
            status = "✅" if r.get("success", False) else "❌"
            status_lines.append(f"  {status} {name}")

        message = (
            f"Results: {n_success}/{n_total} successful\n"
            f"Total time: {total_time/60:.1f} min\n"
            f"\n".join(status_lines)
        )
        self.send_message(message, title="📊 Batch Processing Complete")


# =============================================================================
# IMPROVEMENT 9: BATCH PROCESSING (with parallel support)
# =============================================================================

# All possible condition combinations
ALL_CONDITIONS = [
    ("Commensal", "Static"),
    ("Commensal", "HOBIC"),
    ("Dysbiotic", "Static"),
    ("Dysbiotic", "HOBIC"),
]


def parse_batch_conditions(batch_str: str) -> List[Tuple[str, str]]:
    """
    Parse batch condition string.

    Args:
        batch_str: Either "all" or comma-separated "Condition:Cultivation" pairs
                   e.g., "Commensal:Static,Dysbiotic:HOBIC"

    Returns:
        List of (condition, cultivation) tuples
    """
    if batch_str.lower() == "all":
        return ALL_CONDITIONS.copy()

    conditions = []
    for pair in batch_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            condition, cultivation = pair.split(":")
            condition = condition.strip()
            cultivation = cultivation.strip()

            # Validate
            if condition not in ["Commensal", "Dysbiotic"]:
                raise ValueError(
                    f"Invalid condition: {condition}. Must be 'Commensal' or 'Dysbiotic'"
                )
            if cultivation not in ["Static", "HOBIC"]:
                raise ValueError(f"Invalid cultivation: {cultivation}. Must be 'Static' or 'HOBIC'")

            conditions.append((condition, cultivation))
        else:
            raise ValueError(f"Invalid format: {pair}. Use 'Condition:Cultivation'")

    return conditions


def run_single_condition_estimation(
    condition: str, cultivation: str, args_dict: Dict[str, Any], data_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    """
    Run estimation for a single condition (used by parallel processing).

    This function is designed to be called in a separate process.

    Args:
        condition: Condition name
        cultivation: Cultivation method
        args_dict: Arguments as dictionary (for pickling)
        data_dir: Data directory
        output_dir: Output directory for this condition

    Returns:
        Dictionary with results
    """
    # Reconstruct args namespace
    args = argparse.Namespace(**args_dict)
    args.condition = condition
    args.cultivation = cultivation
    args.output_dir = str(output_dir)

    name = f"{condition}_{cultivation}"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data (using function directly, no import)
        data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
            data_dir, condition, cultivation, args.start_from_day, args.normalize_data
        )

        # Determine initial conditions
        if args.use_exp_init:
            total_init = phi_init_exp.sum()
            if total_init > 0 and not args.normalize_data:
                phi_init_array = phi_init_exp / total_init
            else:
                phi_init_array = phi_init_exp.copy()
        else:
            phi_init_array = None

        # Convert time (using function directly, no import)
        t_model, idx_sparse = convert_days_to_model_time(
            t_days, args.dt, args.maxtimestep, args.day_scale
        )

        # Run estimation (using function directly, no import)
        results = run_estimation(data, idx_sparse, args, output_dir, metadata, phi_init_array)

        return {
            "name": name,
            "success": True,
            "results": results,
            "output_dir": str(output_dir),
            "logL_max": float(np.max(results["logL"])),
            "converged": results["mcmc_diagnostics"]["converged"],
            "elapsed_time": results["elapsed_time"],
        }

    except Exception as e:
        import traceback

        return {
            "name": name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "output_dir": str(output_dir),
        }


def run_batch_estimation(
    args: argparse.Namespace,
    conditions: List[Tuple[str, str]],
    data_dir: Path,
    base_output_dir: Path,
    parallel: bool = False,
    max_workers: int = None,
    notifier: SlackNotifier = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run estimation for multiple conditions (sequential or parallel).

    Args:
        args: Command line arguments
        conditions: List of (condition, cultivation) tuples
        data_dir: Data directory
        base_output_dir: Base output directory
        parallel: If True, run conditions in parallel
        max_workers: Maximum number of parallel workers (None = auto)
        notifier: Optional Slack notifier for notifications

    Returns:
        Dictionary mapping condition names to results
    """
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = base_output_dir / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()

    logger.info(f"Starting batch estimation for {len(conditions)} conditions...")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")

    # Convert args to dictionary for pickling (needed for parallel)
    args_dict = vars(args).copy()

    if parallel and len(conditions) > 1:
        # =====================================================================
        # PARALLEL EXECUTION
        # =====================================================================
        if max_workers is None:
            max_workers = min(len(conditions), os.cpu_count() or 2)

        logger.info(f"Running with {max_workers} parallel workers...")

        futures_to_condition = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for condition, cultivation in conditions:
                name = f"{condition}_{cultivation}"
                output_dir = batch_dir / name

                future = executor.submit(
                    run_single_condition_estimation,
                    condition,
                    cultivation,
                    args_dict,
                    data_dir,
                    output_dir,
                )
                futures_to_condition[future] = name

            # Collect results as they complete
            for future in as_completed(futures_to_condition):
                name = futures_to_condition[future]
                try:
                    result = future.result(timeout=7200)  # 2 hour timeout
                    all_results[name] = result

                    if result["success"]:
                        logger.info(f"[{name}] ✓ Completed. LogL={result['logL_max']:.2f}")
                        if notifier:
                            condition, cultivation = name.split("_")
                            notifier.notify_complete(
                                condition,
                                cultivation,
                                result["elapsed_time"],
                                result["converged"],
                                result["logL_max"],
                                result["output_dir"],
                            )
                    else:
                        logger.error(f"[{name}] ✗ Failed: {result['error']}")
                        if notifier:
                            condition, cultivation = name.split("_")
                            notifier.notify_error(condition, cultivation, result["error"])

                except Exception as e:
                    logger.error(f"[{name}] ✗ Exception: {e}")
                    all_results[name] = {
                        "name": name,
                        "success": False,
                        "error": str(e),
                        "output_dir": str(batch_dir / name),
                    }
                    if notifier:
                        condition, cultivation = name.split("_")
                        notifier.notify_error(condition, cultivation, str(e))

    else:
        # =====================================================================
        # SEQUENTIAL EXECUTION
        # =====================================================================
        for i, (condition, cultivation) in enumerate(conditions):
            name = f"{condition}_{cultivation}"
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(conditions)}] Running: {name}")
            logger.info(f"{'='*60}")

            output_dir = batch_dir / name

            if notifier:
                notifier.notify_start(condition, cultivation, args.n_particles, args.n_stages)

            result = run_single_condition_estimation(
                condition, cultivation, args_dict, data_dir, output_dir
            )
            all_results[name] = result

            if result["success"]:
                logger.info(f"[{name}] Completed. LogL_max={result['logL_max']:.2f}")
                if notifier:
                    notifier.notify_complete(
                        condition,
                        cultivation,
                        result["elapsed_time"],
                        result["converged"],
                        result["logL_max"],
                        result["output_dir"],
                    )
            else:
                logger.error(f"[{name}] Failed: {result['error']}")
                if notifier:
                    notifier.notify_error(condition, cultivation, result["error"])

    total_elapsed = time.time() - total_start_time

    # Save batch summary
    summary_file = batch_dir / "batch_summary.json"
    batch_summary = {
        "timestamp": timestamp,
        "mode": "parallel" if parallel else "sequential",
        "max_workers": max_workers if parallel else 1,
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_minutes": total_elapsed / 60,
        "conditions": [f"{c}:{v}" for c, v in conditions],
        "results": {
            name: {
                "success": r["success"],
                "output_dir": r["output_dir"],
                "error": r.get("error"),
                "logL_max": r.get("logL_max"),
                "converged": r.get("converged"),
                "elapsed_time": r.get("elapsed_time"),
            }
            for name, r in all_results.items()
        },
    }
    save_json(summary_file, batch_summary)
    logger.info(f"Batch summary saved to {summary_file}")

    # Send batch completion notification
    if notifier:
        notifier.notify_batch_complete(all_results, total_elapsed)

    return all_results


# =============================================================================
# IMPROVEMENT 11: WAIC / PSIS-LOO-CV
# =============================================================================


def compute_waic(log_likelihood_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute Watanabe-Akaike Information Criterion (WAIC).

    WAIC is a fully Bayesian approach to model comparison that uses
    the entire posterior distribution.

    Args:
        log_likelihood_matrix: (n_samples, n_observations) matrix of
                               pointwise log-likelihoods

    Returns:
        Dictionary with WAIC, effective parameters, and SE
    """
    n_samples, n_obs = log_likelihood_matrix.shape

    # Log pointwise predictive density (lppd)
    # lppd = sum_i log(1/S * sum_s p(y_i | theta_s))
    # Using logsumexp for numerical stability
    lppd_i = np.zeros(n_obs)
    for i in range(n_obs):
        lppd_i[i] = np.logaddexp.reduce(log_likelihood_matrix[:, i]) - np.log(n_samples)

    lppd = np.sum(lppd_i)

    # Effective number of parameters (p_waic)
    # p_waic = sum_i var_s(log p(y_i | theta_s))
    p_waic_i = np.var(log_likelihood_matrix, axis=0, ddof=1)
    p_waic = np.sum(p_waic_i)

    # WAIC
    waic = -2 * (lppd - p_waic)

    # Standard error
    waic_i = -2 * (lppd_i - p_waic_i)
    se_waic = np.sqrt(n_obs * np.var(waic_i, ddof=1))

    return {
        "waic": float(waic),
        "lppd": float(lppd),
        "p_waic": float(p_waic),
        "se_waic": float(se_waic),
        "n_samples": n_samples,
        "n_observations": n_obs,
    }


def compute_psis_loo(log_likelihood_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute Pareto-Smoothed Importance Sampling Leave-One-Out CV (PSIS-LOO).

    This is a more robust alternative to WAIC for model comparison.

    Args:
        log_likelihood_matrix: (n_samples, n_observations) matrix

    Returns:
        Dictionary with LOO-IC, Pareto k diagnostics, and SE
    """
    n_samples, n_obs = log_likelihood_matrix.shape

    loo_i = np.zeros(n_obs)
    pareto_k = np.zeros(n_obs)

    for i in range(n_obs):
        # Log importance weights (negative log-likelihood for LOO)
        log_weights = -log_likelihood_matrix[:, i]

        # Stabilize weights
        log_weights_max = np.max(log_weights)
        weights = np.exp(log_weights - log_weights_max)

        # Pareto smoothing (simplified - fit Pareto to largest weights)
        # Full implementation would use the PSIS algorithm
        sorted_weights = np.sort(weights)[::-1]
        n_tail = max(int(0.2 * n_samples), 10)
        tail_weights = sorted_weights[:n_tail]

        if np.std(tail_weights) > 1e-10:
            # Estimate Pareto k from tail
            log_tail = np.log(tail_weights + 1e-10)
            k_est = np.mean(log_tail) - np.min(log_tail)
            pareto_k[i] = min(max(k_est, 0), 2)  # Clamp to [0, 2]
        else:
            pareto_k[i] = 0

        # Compute LOO predictive density
        # Using self-normalized importance sampling
        weights_normalized = weights / (np.sum(weights) + 1e-10)
        loo_i[i] = np.log(np.sum(weights_normalized * np.exp(log_likelihood_matrix[:, i])) + 1e-300)

    # LOO-IC (like WAIC, lower is better)
    loo_ic = -2 * np.sum(loo_i)

    # Effective parameters
    # Approximate using difference from full log-likelihood
    lppd_full = np.sum(np.logaddexp.reduce(log_likelihood_matrix, axis=0) - np.log(n_samples))
    p_loo = lppd_full - np.sum(loo_i)

    # Standard error
    se_loo = np.sqrt(n_obs * np.var(-2 * loo_i, ddof=1))

    # Pareto k diagnostics
    k_good = np.sum(pareto_k < 0.5)
    k_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
    k_bad = np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))
    k_very_bad = np.sum(pareto_k >= 1.0)

    return {
        "loo_ic": float(loo_ic),
        "p_loo": float(p_loo),
        "se_loo": float(se_loo),
        "elpd_loo": float(-loo_ic / 2),  # Expected log pointwise predictive density
        "pareto_k": pareto_k.tolist(),
        "pareto_k_max": float(np.max(pareto_k)),
        "pareto_k_diagnostics": {
            "good (k<0.5)": int(k_good),
            "ok (0.5<=k<0.7)": int(k_ok),
            "bad (0.7<=k<1.0)": int(k_bad),
            "very_bad (k>=1.0)": int(k_very_bad),
        },
        "n_samples": n_samples,
        "n_observations": n_obs,
    }


def compute_information_criteria(
    samples: np.ndarray,
    evaluator,
    data: np.ndarray,
    idx_sparse: np.ndarray,
    active_indices: List[int],
    theta_base: np.ndarray,
    n_subsample: int = 500,
) -> Dict[str, Any]:
    """
    Compute WAIC and PSIS-LOO for model comparison.

    Args:
        samples: Posterior samples (n_samples, n_active_params)
        evaluator: LogLikelihoodEvaluator
        data: Observation data
        idx_sparse: Time indices
        active_indices: Active parameter indices
        theta_base: Base parameter vector
        n_subsample: Number of samples to use (for speed)

    Returns:
        Dictionary with WAIC and LOO results
    """
    n_samples = samples.shape[0]
    n_obs = data.size  # Total observations

    # Subsample if too many samples
    if n_samples > n_subsample:
        indices = np.random.choice(n_samples, n_subsample, replace=False)
        samples_sub = samples[indices]
    else:
        samples_sub = samples

    n_samples_used = len(samples_sub)
    logger.info(f"Computing information criteria with {n_samples_used} samples...")

    # Compute pointwise log-likelihoods
    # This requires evaluating likelihood for each observation separately
    # For simplicity, we'll use a simplified approach

    # Note: True pointwise log-likelihood requires modifying the evaluator
    # Here we approximate using the total log-likelihood

    log_likelihoods = np.zeros(n_samples_used)
    for i, theta_active in enumerate(samples_sub):
        try:
            logL = evaluator(theta_active)
            log_likelihoods[i] = logL if logL > -1e10 else -1e10
        except Exception:
            log_likelihoods[i] = -1e10

    # Simplified WAIC (using total likelihood, not pointwise)
    # This is an approximation
    lppd_approx = np.logaddexp.reduce(log_likelihoods) - np.log(n_samples_used)
    var_logL = np.var(log_likelihoods, ddof=1)
    p_waic_approx = var_logL

    waic_approx = -2 * (lppd_approx - p_waic_approx)

    return {
        "waic_approximate": float(waic_approx),
        "lppd_approximate": float(lppd_approx),
        "p_waic_approximate": float(p_waic_approx),
        "logL_mean": float(np.mean(log_likelihoods)),
        "logL_std": float(np.std(log_likelihoods)),
        "n_samples_used": n_samples_used,
        "note": "Approximate WAIC using total likelihood (not pointwise)",
    }


# =============================================================================
# IMPROVEMENT 14: PARAMETER CORRELATION ANALYSIS
# =============================================================================


def compute_correlation_matrix(
    samples: np.ndarray, param_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute parameter correlation matrix from posterior samples.

    High correlations indicate potential identifiability issues.

    Args:
        samples: (n_samples, n_params) posterior samples
        param_names: Parameter names

    Returns:
        Dictionary with correlation matrix and diagnostics
    """
    n_params = samples.shape[1]
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(n_params)]

    # Pearson correlation
    corr_matrix = np.corrcoef(samples.T)

    # Find highly correlated pairs (|r| > 0.7)
    high_corr_pairs = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            r = corr_matrix[i, j]
            if abs(r) > 0.7:
                high_corr_pairs.append(
                    {
                        "param1": param_names[i],
                        "param2": param_names[j],
                        "correlation": float(r),
                        "abs_correlation": float(abs(r)),
                    }
                )

    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    # Compute condition number (indicates multicollinearity)
    try:
        cov_matrix = np.cov(samples.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        condition_number = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
    except Exception:
        condition_number = np.nan

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "param_names": param_names,
        "high_correlation_pairs": high_corr_pairs,
        "n_high_correlations": len(high_corr_pairs),
        "condition_number": float(condition_number),
        "condition_number_log10": float(np.log10(condition_number + 1)),
        "identifiability_warning": len(high_corr_pairs) > 0 or condition_number > 100,
    }


def plot_correlation_matrix(corr_results: Dict[str, Any], output_dir: Path):
    """Generate correlation matrix heatmap."""
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    corr_matrix = np.array(corr_results["correlation_matrix"])
    param_names = corr_results["param_names"]
    n_params = len(param_names)

    fig, ax = plt.subplots(figsize=(max(8, n_params * 0.5), max(6, n_params * 0.4)))

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    # Set ticks
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(param_names, fontsize=8)

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            r = corr_matrix[i, j]
            color = "white" if abs(r) > 0.5 else "black"
            if n_params <= 15:  # Only show values for smaller matrices
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", color=color, fontsize=6)

    ax.set_title(
        f'Parameter Correlation Matrix\n(Condition #: {corr_results["condition_number"]:.1f})'
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "parameter_correlation_matrix.png", dpi=150)
    plt.close()

    # Plot high correlation pairs if any
    if corr_results["high_correlation_pairs"]:
        fig, ax = plt.subplots(figsize=(8, 4))

        pairs = corr_results["high_correlation_pairs"][:10]  # Top 10
        names = [f"{p['param1']}\nvs\n{p['param2']}" for p in pairs]
        values = [p["correlation"] for p in pairs]
        colors = ["red" if v > 0 else "blue" for v in values]

        bars = ax.barh(range(len(pairs)), values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.axvline(0.7, color="red", linestyle="--", alpha=0.5, label="|r|=0.7")
        ax.axvline(-0.7, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Correlation")
        ax.set_title("Highly Correlated Parameter Pairs (|r| > 0.7)")
        ax.legend()
        ax.set_xlim(-1, 1)

        plt.tight_layout()
        plt.savefig(figures_dir / "high_correlation_pairs.png", dpi=150)
        plt.close()

    logger.info(f"Correlation plots saved to {figures_dir}")


# =============================================================================
# DATA LOADING
# =============================================================================

# Species color to model index mapping
SPECIES_MAP = {
    "Blue": 0,  # S. oralis
    "Green": 1,  # A. naeslundii
    "Yellow": 2,  # V. dispar
    "Orange": 2,  # V. parvula (Dysbiotic strain of Veillonella)
    "Purple": 3,  # F. nucleatum
    "Red": 4,  # P. gingivalis
}

# For Commensal (no Orange/V. parvula), remap Purple and Red
SPECIES_MAP_COMMENSAL = {
    "Blue": 0,  # S. oralis
    "Green": 1,  # A. naeslundii
    "Yellow": 2,  # V. dispar
    "Purple": 3,  # F. nucleatum
    "Red": 4,  # P. gingivalis
}


def load_experimental_data(
    data_dir: Path,
    condition: str = "Commensal",
    cultivation: str = "Static",
    start_from_day: int = 1,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load experimental data and convert to model format.

    Returns:
        data: (n_timepoints, n_species) array of absolute volumes or fractions
        t_days: (n_timepoints,) array of timepoints in days
        sigma_obs: estimated observation noise
        phi_init_exp: (n_species,) initial conditions from first timepoint
        metadata: dict with additional info
    """
    # Load boxplot data (total volume)
    possible_boxplot_files = [
        data_dir / f"boxplot_{condition}_{cultivation}.csv",
        data_dir / "experiment_data" / f"boxplot_{condition}_{cultivation}.csv",
        data_dir / "biofilm_boxplot_data.csv",
        data_dir / "experiment_data" / "biofilm_boxplot_data.csv",
    ]

    boxplot_file = None
    for f in possible_boxplot_files:
        if f.exists():
            boxplot_file = f
            break

    if boxplot_file is None:
        # Fallback to checking if files are in experiment_data subdirectory relative to data_dir
        # If data_dir is already the root
        raise FileNotFoundError(
            f"Could not find boxplot data. Checked: {[str(f) for f in possible_boxplot_files]}"
        )

    boxplot_df = pd.read_csv(boxplot_file)

    # Filter for condition/cultivation
    if "condition" in boxplot_df.columns:
        mask = (boxplot_df["condition"] == condition) & (boxplot_df["cultivation"] == cultivation)
        boxplot_df = boxplot_df[mask]

    # Load species distribution data
    possible_species_files = [
        data_dir / "species_distribution_data.csv",
        data_dir / "experiment_data" / "species_distribution_data.csv",
    ]

    species_file = None
    for f in possible_species_files:
        if f.exists():
            species_file = f
            break

    if species_file is None:
        raise FileNotFoundError(
            f"Could not find species data. Checked: {[str(f) for f in possible_species_files]}"
        )

    species_df = pd.read_csv(species_file)

    # Filter
    mask = (species_df["condition"] == condition) & (species_df["cultivation"] == cultivation)
    species_df = species_df[mask]

    # Get unique days
    days = sorted(boxplot_df["day"].unique())
    n_timepoints = len(days)
    n_species = 5  # Always 5 species in model

    logger.info(f"Loading {condition} {cultivation} data: {n_timepoints} timepoints, days={days}")

    # Build data array
    data = np.zeros((n_timepoints, n_species))
    total_volumes = np.zeros(n_timepoints)
    sigma_obs_estimates = []

    species_map = SPECIES_MAP_COMMENSAL if condition == "Commensal" else SPECIES_MAP

    for i, day in enumerate(days):
        # Get total volume for this day
        day_volume = boxplot_df[boxplot_df["day"] == day]
        if len(day_volume) > 0:
            total_vol = day_volume["median"].values[0]
            total_volumes[i] = total_vol

            # Estimate sigma from IQR: sigma ≈ IQR / 1.35
            q1 = day_volume["q1"].values[0]
            q3 = day_volume["q3"].values[0]
            iqr = q3 - q1
            sigma_obs_estimates.append(iqr / 1.35)

        # Get species percentages for this day
        for _, row in species_df[species_df["day"] == day].iterrows():
            species_color = row["species"]
            if species_color in species_map:
                species_idx = species_map[species_color]
                percentage = row["median"] / 100.0  # Convert % to fraction

                # Absolute volume = total_volume * percentage
                data[i, species_idx] = total_vol * percentage

    # Estimate observation noise from data variability
    sigma_obs = np.mean(sigma_obs_estimates) if sigma_obs_estimates else 0.05

    # Optionally normalize data to species fractions
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
        data = data / row_sums
        logger.info("Data normalized to species fractions (sum=1 per timepoint)")
        # Adjust sigma for normalized data
        sigma_obs = sigma_obs / total_volumes.mean() if total_volumes.mean() > 0 else sigma_obs

    # Filter data to start from specified day
    if start_from_day > 1:
        day_indices = [i for i, d in enumerate(days) if d >= start_from_day]
        if len(day_indices) == 0:
            raise ValueError(f"No data found for day >= {start_from_day}")
        data = data[day_indices, :]
        days = [days[i] for i in day_indices]
        total_volumes = total_volumes[day_indices]
        n_timepoints = len(days)
        logger.info(
            f"Filtering data to start from day {start_from_day}: {n_timepoints} timepoints remaining"
        )

    # Extract initial conditions from the FIRST timepoint after filtering
    # (this is Day 3 if start_from_day=3)
    phi_init_exp = data[0, :].copy()

    metadata = {
        "condition": condition,
        "cultivation": cultivation,
        "days": days,
        "n_timepoints": n_timepoints,
        "total_volumes": total_volumes.tolist(),
        "sigma_obs_estimated": sigma_obs,
        "phi_init_exp": phi_init_exp.tolist(),
        "start_from_day": start_from_day,
    }

    return data, np.array(days), sigma_obs, phi_init_exp, metadata


def convert_days_to_model_time(
    t_days: np.ndarray, dt: float, maxtimestep: int, day_scale: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert experimental days to model time indices.

    The model runs from t=0 to t=maxtimestep*dt.
    We map experimental days to this range.

    Args:
        t_days: experimental timepoints in days
        dt: model time step
        maxtimestep: maximum number of time steps
        day_scale: scale factor. If None, auto-calculate to map days to model time.

    Returns:
        t_model: model time values
        idx_sparse: indices into model time array
    """
    t_max_model = maxtimestep * dt
    t_max_days = float(t_days.max())
    t_min_days = float(t_days.min())

    if day_scale is None:
        # Auto-calculate: map day range to model time range
        # Leave 5% margin at end
        day_scale = (t_max_model * 0.95) / t_max_days
        logger.info(f"Auto-calculated day_scale: {day_scale:.6f}")

    # Convert days to model time
    t_model = t_days * day_scale

    # Convert to indices (linear mapping)
    idx_sparse = np.round(t_model / dt).astype(int)

    # Validate indices are within bounds
    if np.any(idx_sparse < 0) or np.any(idx_sparse >= maxtimestep):
        logger.warning(f"Some indices out of range [0, {maxtimestep-1}]: {idx_sparse}")
        idx_sparse = np.clip(idx_sparse, 0, maxtimestep - 1)

    return t_model, idx_sparse


# =============================================================================
# ESTIMATION
# =============================================================================


def run_estimation(
    data: np.ndarray,
    idx_sparse: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    metadata: Dict[str, Any],
    phi_init_array: Optional[np.ndarray] = None,
    checkpoint_manager: Optional[TMCMCCheckpointManager] = None,
    resume_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run TMCMC parameter estimation on experimental data.

    Args:
        phi_init_array: If provided, use per-species initial conditions (5,) array.
                       Otherwise use scalar args.phi_init.
        checkpoint_manager: Optional checkpoint manager for saving progress
        resume_data: Optional checkpoint data to resume from

    Returns:
        Dictionary with samples, diagnostics, credible intervals, and optionally evidence
    """
    # Setup
    debug_level_map = {
        "DEBUG": DebugLevel.VERBOSE,
        "INFO": DebugLevel.MINIMAL,
        "WARNING": DebugLevel.ERROR,
        "ERROR": DebugLevel.ERROR,
    }
    debug_config = DebugConfig(level=debug_level_map.get(args.debug_level, DebugLevel.MINIMAL))
    debug_logger = DebugLogger(debug_config)

    # Determine phi_init (scalar or array)
    if phi_init_array is not None:
        phi_init = phi_init_array
        logger.info(f"Using per-species initial conditions: {phi_init_array}")
    else:
        phi_init = args.phi_init

    # Load model constants
    model_constants = get_model_constants()

    # Model configuration
    # Use a configuration suitable for 5-species model
    solver_kwargs = {
        "dt": args.dt,
        "maxtimestep": args.maxtimestep,
        "c_const": args.c_const,
        "alpha_const": args.alpha_const,
        "phi_init": phi_init,
        "Kp1": args.kp1,
        "K_hill": args.K_hill,
        "n_hill": args.n_hill,
    }

    active_species = model_constants["active_species"]
    active_indices = model_constants["active_indices"]

    # Determine prior bounds
    p_min = args.prior_min if args.prior_min is not None else PRIOR_BOUNDS_DEFAULT[0]
    p_max = args.prior_max if args.prior_max is not None else PRIOR_BOUNDS_DEFAULT[1]

    # Initialize theta_base with prior mean
    prior_mean = (p_min + p_max) / 2.0
    theta_base = np.full(20, prior_mean)

    # --- NISHIOKA ALGORITHM: Parameter Reduction ---
    # Use centralized bounds/locking logic from nishioka_model based on condition
    ve_enabled = getattr(args, "viscoelastic", False)
    logger.info(
        f"Nishioka Algorithm: Retrieving bounds and locked indices for {args.condition} {args.cultivation}"
        f"{' (viscoelastic ON)' if ve_enabled else ''}..."
    )
    nishioka_bounds, LOCKED_INDICES = get_condition_bounds(
        args.condition,
        args.cultivation,
        viscoelastic=ve_enabled,
    )

    # Total param count: 20 (base) + 2 (VE) if enabled
    n_total_params = len(nishioka_bounds)

    # Resize theta_base if VE params added
    if n_total_params > len(theta_base):
        theta_base_ext = np.full(n_total_params, prior_mean)
        theta_base_ext[: len(theta_base)] = theta_base
        # Set VE defaults: theta[20]=log10(tau)=1.0 (~10s), theta[21]=E0/Einf=3.0
        if n_total_params >= 22:
            theta_base_ext[20] = 1.0
            theta_base_ext[21] = 3.0
        theta_base = theta_base_ext

    # Use the exact bounds from the model (including 0.0, 0.0 for locked and 0.0, 1.0 for Vei->Pg)
    prior_bounds = list(nishioka_bounds)

    logger.info(f"Nishioka Algorithm: Locking indices {LOCKED_INDICES} to 0.0")
    for idx in LOCKED_INDICES:
        theta_base[idx] = 0.0

    # Update active_indices to exclude locked parameters
    active_indices = [i for i in range(n_total_params) if i not in LOCKED_INDICES]
    logger.info(f"Reduced parameter space to {len(active_indices)} parameters: {active_indices}")
    if ve_enabled:
        logger.info("VE params active: theta[20]=log_tau_relax, theta[21]=E0_Einf_ratio")

    # Override specific bounds if requested (e.g. --override-bounds "18:0:3,19:0:3")
    if args.override_bounds:
        for spec in args.override_bounds.split(","):
            parts = spec.strip().split(":")
            if len(parts) == 3:
                idx, lo, hi = int(parts[0]), float(parts[1]), float(parts[2])
                if idx not in LOCKED_INDICES:
                    logger.info(f"Override bounds: index {idx} -> [{lo}, {hi}]")
                    prior_bounds[idx] = (lo, hi)
                else:
                    logger.warning(f"Cannot override bounds for locked index {idx}, skipping")

    # Widen M1 priors if requested
    if args.widen_m1_priors:
        logger.info("Widening M1 priors (indices 0-4) to [0.0, 10.0]")
        for i in range(5):
            prior_bounds[i] = (0.0, 10.0)
            # Update base to mean of new range for these
            theta_base[i] = 5.0

    # Tighten decay priors if requested
    # Decay parameters: b1=3, b2=4, b3=8, b4=9, b5=15
    DECAY_INDICES = [3, 4, 8, 9, 15]
    if args.prior_decay_max is not None:
        decay_max = args.prior_decay_max
        logger.info(
            f"Tightening decay priors (b1-b5, indices {DECAY_INDICES}) to [0.0, {decay_max}]"
        )
        for idx in DECAY_INDICES:
            prior_bounds[idx] = (0.0, decay_max)
            # Update base to mean of new range
            theta_base[idx] = decay_max / 2.0

    # CRITICAL: Re-apply locks to theta_base to prevent prior options from overriding locks
    # If a parameter is in LOCKED_INDICES, it MUST be 0.0, regardless of prior_decay_max or widen_m1_priors.
    if len(LOCKED_INDICES) > 0:
        logger.info(
            f"Re-applying locks to theta_base to ensure locked parameters {LOCKED_INDICES} remain 0.0"
        )
        for idx in LOCKED_INDICES:
            theta_base[idx] = 0.0
            # Also reset prior bounds for locked parameters to avoid confusion (though not used by TMCMC for inactive)
            prior_bounds[idx] = (0.0, 0.0)

    # Sigma for likelihood
    sigma_obs_scalar = (
        args.sigma_obs if args.sigma_obs else metadata.get("sigma_obs_estimated", 0.05)
    )

    # Species-specific sigma (V. dispar gets higher noise to acknowledge
    # the model's structural inability to capture nutrient depletion dynamics)
    if getattr(args, "species_sigma", False):
        vd_factor = getattr(args, "vd_sigma_factor", 2.0)
        sigma_obs = build_species_sigma(
            sigma_obs_scalar,
            n_species=data.shape[1],
            vd_species_idx=2,
            vd_factor=vd_factor,
        )
        logger.info(f"Using species-specific sigma_obs = {sigma_obs}")
    else:
        sigma_obs = sigma_obs_scalar
        logger.info(f"Using sigma_obs = {sigma_obs}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"idx_sparse: {idx_sparse}")

    # --- Weighted likelihood (P. gingivalis late-stage emphasis) ---
    likelihood_weights = None
    if args.lambda_pg != 1.0 or args.lambda_late != 1.0:
        likelihood_weights = build_likelihood_weights(
            n_obs=data.shape[0],
            n_species=data.shape[1],
            pg_species_idx=4,  # P. gingivalis
            n_late=args.n_late,
            lambda_pg=args.lambda_pg,
            lambda_late=args.lambda_late,
        )
        logger.info(
            f"Weighted likelihood enabled: lambda_pg={args.lambda_pg}, "
            f"lambda_late={args.lambda_late}, n_late={args.n_late}"
        )
        logger.info(f"Weight matrix:\n{likelihood_weights}")

    # --- DeepONet surrogate setup (optional) ---
    deeponet_surrogate = None
    if getattr(args, "use_deeponet", False):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deeponet"))
            from surrogate_tmcmc import DeepONetSurrogate

            # Auto-detect checkpoint directory
            ckpt_dir = args.deeponet_checkpoint
            if ckpt_dir is None:
                # Auto-detect best checkpoint per condition (prefer 50k > v2 > original)
                cond_key = f"{args.condition}_{args.cultivation}"
                deeponet_dir = Path(__file__).parent.parent.parent / "deeponet"
                # Priority order: 50k trained, then v2, then original
                candidates = {
                    "Commensal_Static": [
                        "checkpoints_CS_50k",
                        "checkpoints_CS_v2",
                        "checkpoints_Commensal_Static",
                    ],
                    "Commensal_HOBIC": [
                        "checkpoints_CH_50k",
                        "checkpoints_CH_v2",
                        "checkpoints_Commensal_HOBIC",
                    ],
                    "Dysbiotic_Static": ["checkpoints_DS_v2", "checkpoints_Dysbiotic_Static"],
                    "Dysbiotic_HOBIC": [
                        "checkpoints_Dysbiotic_HOBIC_50k",
                        "checkpoints_Dysbiotic_HOBIC",
                    ],
                }
                ckpt_name = None
                for name in candidates.get(cond_key, ["checkpoints_Dysbiotic_HOBIC"]):
                    if (deeponet_dir / name / "best.eqx").exists():
                        ckpt_name = name
                        break
                if ckpt_name is None:
                    ckpt_name = "checkpoints_Dysbiotic_HOBIC"
                ckpt_dir = str(deeponet_dir / ckpt_name)

            ckpt_path = Path(ckpt_dir)
            deeponet_surrogate = DeepONetSurrogate(
                str(ckpt_path / "best.eqx"),
                str(ckpt_path / "norm_stats.npz"),
            )
            logger.info(f"DeepONet surrogate loaded from {ckpt_dir}")
            # JAX objects can't be pickled across forked processes; force threading
            if not getattr(args, "use_threads", False):
                args.use_threads = True
                logger.info("Auto-enabled --use-threads for JAX/DeepONet compatibility")
        except Exception as e:
            logger.error(f"Failed to load DeepONet: {e}. Falling back to ODE solver.")
            deeponet_surrogate = None

    def make_evaluator(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base

        # Use DeepONet if available
        if deeponet_surrogate is not None:
            from core.evaluator import DeepONetEvaluator

            # Map ODE idx_sparse (0..maxtimestep-1) to DeepONet 100-point grid (0..99)
            maxtimestep = getattr(args, "maxtimestep", 2500)
            idx_sparse_don = np.round(idx_sparse * 99.0 / max(maxtimestep - 1, 1)).astype(int)
            idx_sparse_don = np.clip(idx_sparse_don, 0, 99)
            return DeepONetEvaluator(
                surrogate=deeponet_surrogate,
                active_species=active_species,
                active_indices=active_indices,
                theta_base=theta_base,
                data=data,
                idx_sparse=idx_sparse_don,
                sigma_obs=sigma_obs,
                rho=0.0,
                weights=likelihood_weights,
            )

        # ODE solver/TSM only uses theta[0:19] — filter active_indices for TSM
        ode_active_indices = [i for i in active_indices if i < 20]
        ode_theta_base = theta_base[:20] if len(theta_base) > 20 else theta_base
        ode_theta_lin = (
            theta_linearization[:20] if len(theta_linearization) > 20 else theta_linearization
        )

        evaluator = LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs,
            active_species=active_species,
            active_indices=ode_active_indices,
            theta_base=ode_theta_base,
            data=data,
            idx_sparse=idx_sparse,
            sigma_obs=sigma_obs,
            cov_rel=args.cov_rel,
            rho=0.0,
            theta_linearization=ode_theta_lin,
            paper_mode=False,
            debug_logger=debug_logger,
            use_absolute_volume=args.use_absolute_volume,
            weights=likelihood_weights,
        )

        # Attach VE prior if enabled
        if ve_enabled:
            from core.evaluator import ViscoelasticPrior

            evaluator.ve_prior = ViscoelasticPrior(active_indices=active_indices)

        # Override active_indices/theta_base for TMCMC sampling (includes VE params)
        evaluator.active_indices = list(active_indices)
        evaluator.theta_base = theta_base.copy()

        return evaluator

    # GNN prior (Issue #39)
    gnn_prior_obj = None
    if getattr(args, "use_gnn_prior", False):
        try:
            from data_5species.core.gnn_prior import GNNPrior

            locked = [i for i, (lo, hi) in enumerate(prior_bounds) if abs(hi - lo) < 1e-12]
            if getattr(args, "gnn_prior_json", None):
                json_path = Path(args.gnn_prior_json)
                if not json_path.is_absolute():
                    json_path = Path(__file__).resolve().parent.parent.parent / json_path
                gnn_prior_obj = GNNPrior.from_json(
                    str(json_path),
                    sigma=args.gnn_sigma,
                    weight=args.gnn_weight,
                    locked_indices=locked,
                    condition=f"{args.condition}_{args.cultivation}",
                )
                logger.info(f"GNN prior from JSON: {args.gnn_prior_json}")
            else:
                gnn_prior_obj = GNNPrior.load(
                    checkpoint=args.gnn_checkpoint,
                    condition=f"{args.condition}_{args.cultivation}",
                    sigma=args.gnn_sigma,
                    weight=args.gnn_weight,
                    locked_indices=locked,
                )
                logger.info(f"GNN prior loaded: sigma={args.gnn_sigma}, weight={args.gnn_weight}")
        except Exception as e:
            logger.warning(f"Failed to load GNN prior: {e}. Falling back to uniform prior.")

    # Run TMCMC
    logger.info("Starting TMCMC estimation...")
    start_time = time.time()

    chains, logL, MAP, converged, diag = run_multi_chain_TMCMC(
        model_tag=f"{args.condition}_{args.cultivation}",
        make_evaluator=make_evaluator,
        prior_bounds=prior_bounds,
        theta_base_full=theta_base,
        active_indices=active_indices,
        theta_linearization_init=theta_base,
        n_particles=args.n_particles,
        n_stages=args.n_stages,
        n_chains=args.n_chains,
        min_delta_beta=0.02,
        max_delta_beta=0.5,
        logL_scale=1.0,
        seed=args.seed,
        n_jobs=args.n_jobs,
        use_threads=getattr(args, "use_threads", False),
        debug_config=debug_config,
        gnn_prior=gnn_prior_obj,
    )

    elapsed = time.time() - start_time
    logger.info(f"TMCMC completed in {elapsed:.2f} seconds")

    # Process results
    samples = np.concatenate(chains, axis=0)
    logL_all = np.concatenate(logL, axis=0)

    results = compute_MAP_with_uncertainty(samples, logL_all)
    MAP_estimate = results["MAP"]
    mean_estimate = results["mean"]

    # Update theta_full
    if MAP_estimate.shape[0] == len(theta_base):
        theta_MAP_full = MAP_estimate.copy()
        theta_mean_full = mean_estimate.copy()
    else:
        theta_MAP_full = theta_base.copy()
        theta_MAP_full[active_indices] = MAP_estimate
        theta_mean_full = theta_base.copy()
        theta_mean_full[active_indices] = mean_estimate

    # =========================================================================
    # IMPROVEMENT 2: Enhanced Convergence Diagnostics (R-hat, ESS)
    # =========================================================================
    param_names = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",  # M1
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",  # M2
        "a13",
        "a14",
        "a23",
        "a24",  # M3
        "a55",
        "b5",  # M4
        "a15",
        "a25",
        "a35",
        "a45",  # M5
        "log_tau_relax",
        "E0_Einf_ratio",  # VE (optional)
    ]
    active_param_names = [param_names[i] for i in active_indices if i < len(param_names)]

    logger.info("Computing enhanced convergence diagnostics...")
    mcmc_diagnostics = compute_mcmc_diagnostics(chains, logL, active_param_names)

    logger.info(f"R-hat max: {mcmc_diagnostics['rhat_max']:.4f} (target < 1.1)")
    logger.info(f"ESS min: {mcmc_diagnostics['ess_min']:.1f} (target > 100)")
    logger.info(f"Convergence (R-hat): {mcmc_diagnostics['converged_rhat']}")
    logger.info(f"Convergence (ESS): {mcmc_diagnostics['converged_ess']}")

    # =========================================================================
    # IMPROVEMENT 3: Highest Density Intervals (HDI)
    # =========================================================================
    logger.info(f"Computing {args.hdi_credibility*100:.0f}% HDI and credible intervals...")
    credible_intervals = compute_credible_intervals(samples, args.hdi_credibility)

    logger.info("HDI computed for all parameters")

    # =========================================================================
    # IMPROVEMENT 6: Model Evidence (if requested)
    # =========================================================================
    evidence_results = None
    if args.compute_evidence:
        logger.info("Computing model evidence (marginal likelihood)...")

        # Extract beta schedule and mean logL from diagnostics if available
        beta_schedule = None
        mean_logL_per_stage = None

        if diag and "beta_schedules" in diag:
            # Use first chain's beta schedule
            beta_schedule = diag["beta_schedules"][0] if diag["beta_schedules"] else None
        if diag and "mean_logL_per_stage" in diag:
            mean_logL_per_stage = (
                diag["mean_logL_per_stage"][0] if diag["mean_logL_per_stage"] else None
            )

        evidence_results = compute_model_evidence(
            chains=chains,
            logL=logL,
            prior_bounds=prior_bounds,
            active_indices=active_indices,
            beta_schedule=beta_schedule,
            mean_logL_per_stage=mean_logL_per_stage,
        )

        logger.info(
            f"Model Evidence: log(p(D)) ≈ {evidence_results['log_evidence']:.2f} "
            f"± {evidence_results['log_evidence_se']:.2f} ({evidence_results['method']})"
        )

    # Clear checkpoints on successful completion
    if checkpoint_manager is not None and not getattr(args, "keep_checkpoints", False):
        checkpoint_manager.clear_checkpoints()

    return {
        "samples": samples,
        "logL": logL_all,
        "chains": chains,  # Keep individual chains for diagnostics
        "MAP": MAP_estimate,
        "mean": mean_estimate,
        "theta_MAP_full": theta_MAP_full,
        "theta_mean_full": theta_mean_full,
        "elapsed_time": elapsed,
        "converged": converged,
        "diagnostics": diag,
        # IMPROVEMENT 2: Enhanced diagnostics
        "mcmc_diagnostics": mcmc_diagnostics,
        "rhat": mcmc_diagnostics["rhat"],
        "ess": mcmc_diagnostics["ess"],
        # IMPROVEMENT 3: Credible intervals
        "credible_intervals": credible_intervals,
        "hdi_lower": credible_intervals["hdi_lower"],
        "hdi_upper": credible_intervals["hdi_upper"],
        # IMPROVEMENT 6: Evidence
        "evidence": evidence_results,
        # Metadata
        "active_indices": active_indices,
        "prior_bounds": prior_bounds,
        "param_names": active_param_names,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="TMCMC Parameter Estimation with Nishioka Algorithm (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python estimate_reduced_nishioka.py --condition Commensal --cultivation Static

  # With checkpointing (for long runs)
  python estimate_reduced_nishioka.py --condition Dysbiotic --cultivation HOBIC \\
      --checkpoint-every 5 --n-particles 1000 --n-stages 50

  # Resume from checkpoint
  python estimate_reduced_nishioka.py --resume-from _runs/previous_run/checkpoints
        """,
    )

    # Data options
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Directory containing experimental data",
    )
    parser.add_argument(
        "--condition", type=str, default="Commensal", choices=["Commensal", "Dysbiotic"]
    )
    parser.add_argument("--cultivation", type=str, default="Static", choices=["Static", "HOBIC"])

    # Model options
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step")
    parser.add_argument("--maxtimestep", type=int, default=2500, help="Max time steps")
    parser.add_argument("--c-const", type=float, default=25.0, help="c constant")
    parser.add_argument(
        "--kp1", type=float, default=1e-4, help="Cahn-Hilliard gradient energy Kp1 (default: 1e-4)"
    )
    parser.add_argument(
        "--K-hill",
        type=float,
        default=0.0,
        help="Hill threshold for F.nucleatum bridge gate (0=disabled)",
    )
    parser.add_argument("--n-hill", type=float, default=2.0, help="Hill exponent for bridge gate")
    parser.add_argument("--alpha-const", type=float, default=0.0, help="alpha constant")
    parser.add_argument(
        "--phi-init", type=float, default=0.02, help="Initial phi (ignored if --use-exp-init)"
    )
    parser.add_argument(
        "--day-scale",
        type=float,
        default=None,
        help="Scaling factor: model_time = day * day_scale (auto if None)",
    )
    parser.add_argument(
        "--use-exp-init",
        action="store_true",
        help="Use experimental data (from start-from-day) as initial conditions",
    )
    parser.add_argument(
        "--start-from-day", type=int, default=1, help="Start fitting from this day (default: 1)"
    )
    parser.add_argument(
        "--normalize-data",
        action="store_true",
        help="Normalize data to species fractions (sum=1) instead of absolute volumes",
    )

    # Estimation options
    parser.add_argument(
        "--sigma-obs",
        type=float,
        default=None,
        help="Observation noise (default: estimated from data)",
    )
    parser.add_argument("--cov-rel", type=float, default=0.005, help="Relative covariance for ROM")
    parser.add_argument(
        "--use-absolute-volume",
        action="store_true",
        help="Use absolute volume (phi*gamma) for likelihood",
    )

    # Weighted likelihood options (P. gingivalis late-stage emphasis)
    parser.add_argument(
        "--lambda-pg",
        type=float,
        default=1.0,
        help="Species weight for P. gingivalis (species 4). "
        "Values > 1 emphasise P.g. residuals. Recommended: 5.0 for Dysbiotic HOBIC",
    )
    parser.add_argument(
        "--lambda-late",
        type=float,
        default=1.0,
        help="Time weight for the last N observation times. "
        "Values > 1 emphasise late-stage fit. Recommended: 3.0 for Dysbiotic HOBIC",
    )
    parser.add_argument(
        "--n-late",
        type=int,
        default=2,
        help="Number of final observation times to up-weight (default: 2, i.e. days 15+21)",
    )

    # Prior options
    parser.add_argument(
        "--prior-min",
        type=float,
        default=None,
        help="Minimum value for prior uniform distribution (default: use config)",
    )
    parser.add_argument(
        "--prior-max",
        type=float,
        default=None,
        help="Maximum value for prior uniform distribution (default: use config)",
    )
    parser.add_argument(
        "--widen-m1-priors",
        action="store_true",
        help="Widen priors for M1 parameters (indices 0-4) to [0, 10]",
    )
    parser.add_argument(
        "--prior-decay-max",
        type=float,
        default=None,
        help="Maximum value for decay parameters b1-b5 (indices 3,4,8,9,15). "
        "Use smaller values (e.g., 1.0) if model over-predicts decline.",
    )
    parser.add_argument(
        "--override-bounds",
        type=str,
        default=None,
        help="Override specific parameter bounds. Format: 'idx:lo:hi,...' "
        "e.g. '18:0:3,19:0:3' sets a35 and a45 bounds to [0,3].",
    )

    # TMCMC options
    parser.add_argument("--n-particles", type=int, default=500, help="Number of particles")
    parser.add_argument("--n-stages", type=int, default=30, help="Number of stages")
    parser.add_argument("--n-chains", type=int, default=2, help="Number of chains")
    parser.add_argument("--n-jobs", type=int, default=12, help="Parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-gnn-prior", action="store_true", help="Use GNN-informed prior (Issue #39)"
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=str,
        default="gnn/data/checkpoints/best.pt",
        help="Path to GNN checkpoint",
    )
    parser.add_argument(
        "--gnn-sigma", type=float, default=1.0, help="GNN prior sigma (smaller = tighter prior)"
    )
    parser.add_argument(
        "--gnn-weight", type=float, default=1.0, help="GNN prior weight in log-posterior"
    )
    parser.add_argument(
        "--gnn-prior-json",
        type=str,
        default=None,
        help="Path to gnn_prior.json from predict_hmp.py (Phase 2 HMP)",
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--debug-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    # Checkpoint & Resume (Improvement 1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N stages (0 to disable)",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume from checkpoint directory"
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep checkpoints after successful completion",
    )

    # Diagnostics options (Improvement 2, 3)
    parser.add_argument(
        "--compute-evidence",
        action="store_true",
        help="Compute model evidence (marginal likelihood)",
    )
    parser.add_argument(
        "--hdi-credibility",
        type=float,
        default=0.95,
        help="Credibility level for HDI (default: 0.95)",
    )

    # Batch Processing (Improvement 9)
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help='Batch mode: "all" for all 4 conditions, or comma-separated '
        '"Condition:Cultivation" pairs (e.g., "Commensal:Static,Dysbiotic:HOBIC")',
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run batch conditions in parallel (requires --batch)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Maximum parallel workers (default: auto)"
    )

    # Notification (Improvement 3 v3) - Enabled by default
    parser.add_argument(
        "--no-notify-slack",
        action="store_true",
        help="Disable Slack notifications (enabled by default if webhook configured)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file for Slack webhook (default: auto-detect)",
    )

    # Prior Predictive Checks (Improvement 5)
    parser.add_argument(
        "--prior-predictive",
        type=int,
        default=0,
        help="Run prior predictive check with N samples (0 to skip)",
    )

    # WAIC / Information Criteria (Improvement 11)
    parser.add_argument(
        "--compute-waic",
        action="store_true",
        help="Compute WAIC and information criteria for model comparison",
    )

    # Correlation Analysis (Improvement 14)
    parser.add_argument(
        "--correlation-analysis",
        action="store_true",
        help="Compute and plot parameter correlation matrix",
    )

    # Species-specific sigma (V. dispar model inadequacy)
    parser.add_argument(
        "--species-sigma",
        action="store_true",
        help="Use per-species sigma_obs (V. dispar gets higher noise)",
    )
    parser.add_argument(
        "--vd-sigma-factor",
        type=float,
        default=2.0,
        help="V. dispar sigma multiplier (default: 2.0)",
    )

    # DeepONet surrogate (~80× per-sample, ~29× E2E TMCMC)
    parser.add_argument(
        "--use-deeponet",
        action="store_true",
        help="Use DeepONet surrogate instead of ODE solver (~80× per-sample, ~29× E2E)",
    )
    parser.add_argument(
        "--deeponet-checkpoint",
        type=str,
        default=None,
        help="Path to DeepONet checkpoint dir (default: auto-detect from condition)",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        help="Use threads instead of processes (required for JAX/DeepONet to avoid fork issues)",
    )

    # Viscoelastic extension (SLS/Zener model)
    parser.add_argument(
        "--viscoelastic",
        action="store_true",
        help="Enable viscoelastic (SLS) parameter estimation: theta[20]=log_tau_relax, theta[21]=E0_Einf_ratio. "
        "Currently adds informative prior penalty (no mechanical measurement data available).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug_level)

    # Setup paths
    data_dir = Path(args.data_dir)

    # =========================================================================
    # IMPROVEMENT 3 (v3): Initialize Slack Notifier (enabled by default)
    # =========================================================================
    notifier = None
    if not args.no_notify_slack:
        notifier = SlackNotifier(env_path=args.env_file)
        # notifier.enabled will be False if webhook URL not found (silent)

    # =========================================================================
    # IMPROVEMENT 9: Batch Processing Mode (with parallel support)
    # =========================================================================
    if args.batch:
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING MODE")
        logger.info("=" * 60)

        try:
            conditions = parse_batch_conditions(args.batch)
        except ValueError as e:
            logger.error(f"Invalid batch specification: {e}")
            sys.exit(1)

        if args.parallel:
            logger.info(f"Parallel mode enabled (max_workers={args.max_workers or 'auto'})")

        base_output_dir = data_dir / "_runs"
        batch_results = run_batch_estimation(
            args,
            conditions,
            data_dir,
            base_output_dir,
            parallel=args.parallel,
            max_workers=args.max_workers,
            notifier=notifier,
        )

        # Print batch summary
        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70)
        n_success = sum(1 for r in batch_results.values() if r.get("success", False))
        print(f"Results: {n_success}/{len(batch_results)} successful")
        print("-" * 70)
        for name, result in batch_results.items():
            if result.get("success", False):
                print(
                    f"  ✓ {name}: LogL={result.get('logL_max', 'N/A'):.2f}, "
                    f"Converged={result.get('converged', 'N/A')}"
                )
            else:
                print(f"  ✗ {name}: FAILED - {result.get('error', 'Unknown error')}")
        print("=" * 70)

        return  # Exit after batch processing

    # Single condition mode - notify start if enabled
    if notifier:
        notifier.notify_start(args.condition, args.cultivation, args.n_particles, args.n_stages)

    # Standard single-condition mode
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = data_dir / "_runs" / f"{args.condition}_{args.cultivation}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load experimental data
    logger.info("Loading experimental data...")
    data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
        data_dir, args.condition, args.cultivation, args.start_from_day, args.normalize_data
    )

    # Determine initial conditions
    if args.use_exp_init:
        # Use experimental data from start_from_day as initial conditions
        # Convert to phi fractions (normalize by total if needed)
        total_init = phi_init_exp.sum()
        if total_init > 0 and not args.normalize_data:
            # If not normalized, convert to fractions
            phi_init_array = phi_init_exp / total_init
        else:
            # Already normalized or zero
            phi_init_array = phi_init_exp.copy()
        logger.info(f"Using experimental initial conditions (Day {metadata['days'][0]}):")
        logger.info(f"  Values: {phi_init_exp}")
        logger.info(f"  Normalized phi: {phi_init_array}")
    else:
        phi_init_array = None  # Use default scalar phi_init

    # Convert days to model time
    t_model, idx_sparse = convert_days_to_model_time(
        t_days, args.dt, args.maxtimestep, args.day_scale
    )

    metadata["t_model"] = t_model.tolist()
    metadata["idx_sparse"] = idx_sparse.tolist()
    metadata["day_scale"] = args.day_scale

    logger.info(f"Experimental days: {t_days}")
    logger.info(f"Model indices: {idx_sparse}")
    logger.info(f"Data (absolute volumes):\n{data}")

    # Save data
    save_npy(output_dir / "data.npy", data)
    save_npy(output_dir / "idx_sparse.npy", idx_sparse)
    save_npy(output_dir / "t_days.npy", t_days)

    # Save config
    config = {
        "condition": args.condition,
        "cultivation": args.cultivation,
        "dt": args.dt,
        "maxtimestep": args.maxtimestep,
        "c_const": args.c_const,
        "Kp1": args.kp1,
        "K_hill": args.K_hill,
        "n_hill": args.n_hill,
        "alpha_const": args.alpha_const,
        "phi_init": phi_init_array.tolist() if phi_init_array is not None else args.phi_init,
        "phi_init_is_array": phi_init_array is not None,
        "use_exp_init": args.use_exp_init,
        "day_scale": args.day_scale,
        "sigma_obs": args.sigma_obs or sigma_obs_est,
        "cov_rel": args.cov_rel,
        "prior_min": args.prior_min,
        "prior_max": args.prior_max,
        "prior_decay_max": args.prior_decay_max,
        "n_particles": args.n_particles,
        "n_stages": args.n_stages,
        "n_chains": args.n_chains,
        "seed": args.seed,
        "hdi_credibility": args.hdi_credibility,
        "compute_evidence": args.compute_evidence,
        "metadata": metadata,
    }
    save_json(output_dir / "config.json", config)

    # =========================================================================
    # IMPROVEMENT 1: Checkpoint & Resume Setup
    # =========================================================================
    checkpoint_manager = None
    resume_data = None

    if args.resume_from:
        # Resume from checkpoint
        resume_dir = Path(args.resume_from)
        if resume_dir.exists():
            checkpoint_manager = TMCMCCheckpointManager(resume_dir)
            resume_data = checkpoint_manager.load_checkpoint()
            if resume_data:
                logger.info(f"Resuming from stage {resume_data['stage']}")
            else:
                logger.warning("No valid checkpoint found. Starting fresh.")
        else:
            logger.warning(f"Resume directory {resume_dir} not found. Starting fresh.")

    if checkpoint_manager is None and args.checkpoint_every > 0:
        # Create new checkpoint manager
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_manager = TMCMCCheckpointManager(checkpoint_dir, args.checkpoint_every)
        logger.info(
            f"Checkpointing enabled: saving every {args.checkpoint_every} stages to {checkpoint_dir}"
        )

    # =========================================================================
    # IMPROVEMENT 5: Prior Predictive Check (before main estimation)
    # =========================================================================
    if args.prior_predictive > 0:
        logger.info(f"Running prior predictive check with {args.prior_predictive} samples...")

        # Get bounds and locked indices
        nishioka_bounds, LOCKED_INDICES = get_condition_bounds(args.condition, args.cultivation)
        active_indices_ppc = [i for i in range(20) if i not in LOCKED_INDICES]

        theta_base_ppc = np.zeros(20)
        for i, (low, high) in enumerate(nishioka_bounds):
            theta_base_ppc[i] = (low + high) / 2.0
        for idx in LOCKED_INDICES:
            theta_base_ppc[idx] = 0.0

        solver_kwargs_ppc = {
            "dt": args.dt,
            "maxtimestep": args.maxtimestep,
            "c_const": args.c_const,
            "alpha_const": args.alpha_const,
            "phi_init": phi_init_array if phi_init_array is not None else args.phi_init,
            "Kp1": args.kp1,
            "K_hill": args.K_hill,
            "n_hill": args.n_hill,
        }

        ppc_results = run_prior_predictive_check(
            n_samples=args.prior_predictive,
            prior_bounds=list(nishioka_bounds),
            active_indices=active_indices_ppc,
            theta_base=theta_base_ppc,
            solver_kwargs=solver_kwargs_ppc,
            active_species=[0, 1, 2, 3, 4],
            seed=args.seed,
        )

        # Save PPC results
        ppc_output = {
            "n_samples": ppc_results["n_samples"],
            "n_successful": ppc_results["n_successful"],
            "n_failed": ppc_results["n_failed"],
            "success_rate": ppc_results["success_rate"],
            "summary": ppc_results["summary"],
        }
        save_json(output_dir / "prior_predictive.json", ppc_output)

        # Generate PPC plots
        param_names_ppc = [
            "a11",
            "a12",
            "a22",
            "b1",
            "b2",
            "a33",
            "a34",
            "a44",
            "b3",
            "b4",
            "a13",
            "a14",
            "a23",
            "a24",
            "a55",
            "b5",
            "a15",
            "a25",
            "a35",
            "a45",
        ]
        active_param_names_ppc = [param_names_ppc[i] for i in active_indices_ppc]

        plot_prior_predictive(ppc_results, output_dir, data, active_param_names_ppc)

        logger.info(
            f"Prior predictive check complete: {ppc_results['success_rate']*100:.1f}% success rate"
        )

        if ppc_results["success_rate"] < 0.5:
            logger.warning("Low prior predictive success rate! Consider adjusting priors.")

    # Run estimation
    logger.info("Running parameter estimation...")
    results = run_estimation(
        data,
        idx_sparse,
        args,
        output_dir,
        metadata,
        phi_init_array,
        checkpoint_manager=checkpoint_manager,
        resume_data=resume_data,
    )

    # Save results
    save_npy(output_dir / "samples.npy", results["samples"])
    save_npy(output_dir / "logL.npy", results["logL"])

    # Extract active parameter values from full theta vectors (used in multiple places)
    active_indices = results["active_indices"]
    n_active = len(active_indices)

    # Get total number of parameters from model config (avoid hardcoding)
    model_constants = get_model_constants()
    N_FULL_PARAMS = len(
        model_constants.get("param_names", [f"p{i}" for i in range(20)])
    )  # Fallback if config missing

    # Robust extraction: check for full dimension (20) explicitly
    if len(results["MAP"]) == N_FULL_PARAMS and n_active < N_FULL_PARAMS:
        MAP_active = results["MAP"][active_indices]
        mean_active = results["mean"][active_indices]
    elif len(results["MAP"]) == n_active:
        # Already in active dimension
        MAP_active = results["MAP"]
        mean_active = results["mean"]
    else:
        # Unexpected dimension - log warning and attempt extraction
        logger.warning(
            f"Unexpected MAP dimension: {len(results['MAP'])} (expected {N_FULL_PARAMS} or {n_active})"
        )
        MAP_active = (
            results["MAP"][:n_active] if len(results["MAP"]) >= n_active else results["MAP"]
        )
        mean_active = (
            results["mean"][:n_active] if len(results["mean"]) >= n_active else results["mean"]
        )

    save_json(
        output_dir / "theta_MAP.json",
        {
            "theta_sub": MAP_active.tolist(),  # Active parameters only
            "theta_full": results["theta_MAP_full"].tolist(),
            "active_indices": active_indices,
        },
    )

    # =========================================================================
    # IMPROVEMENT 2 & 3: Save Enhanced Diagnostics and HDI
    # =========================================================================

    # Save MCMC diagnostics (R-hat, ESS)
    mcmc_diag = results["mcmc_diagnostics"]
    diagnostics_output = {
        "rhat": mcmc_diag["rhat"].tolist(),
        "ess": mcmc_diag["ess"].tolist(),
        "rhat_max": mcmc_diag["rhat_max"],
        "ess_min": mcmc_diag["ess_min"],
        "converged_rhat": mcmc_diag["converged_rhat"],
        "converged_ess": mcmc_diag["converged_ess"],
        "converged": mcmc_diag["converged"],
        "n_samples_total": mcmc_diag["n_samples_total"],
        "n_chains": mcmc_diag["n_chains"],
        "per_parameter": mcmc_diag["per_parameter"],
    }
    save_json(output_dir / "mcmc_diagnostics.json", diagnostics_output)

    # Save credible intervals (HDI and equal-tailed)
    ci = results["credible_intervals"]
    ci_output = {
        "credibility": ci["credibility"],
        "hdi_lower": ci["hdi_lower"].tolist(),
        "hdi_upper": ci["hdi_upper"].tolist(),
        "et_lower": ci["et_lower"].tolist(),
        "et_upper": ci["et_upper"].tolist(),
        "median": ci["median"].tolist(),
        "param_names": results["param_names"],
    }
    save_json(output_dir / "credible_intervals.json", ci_output)

    # Save parameter summary table as CSV
    # (MAP_active, mean_active already extracted above)

    # Extract active indices from CI and diagnostics arrays (they may be full 20-dim)
    def extract_active(arr, active_idx, n_full):
        """Extract active indices from array if it's in full dimension."""
        arr = np.asarray(arr)
        if len(arr) == n_full and len(active_idx) < n_full:
            return arr[active_idx]
        return arr

    ci_median = extract_active(ci["median"], active_indices, N_FULL_PARAMS)
    ci_hdi_lower = extract_active(ci["hdi_lower"], active_indices, N_FULL_PARAMS)
    ci_hdi_upper = extract_active(ci["hdi_upper"], active_indices, N_FULL_PARAMS)
    ci_et_lower = extract_active(ci["et_lower"], active_indices, N_FULL_PARAMS)
    ci_et_upper = extract_active(ci["et_upper"], active_indices, N_FULL_PARAMS)
    diag_rhat = extract_active(mcmc_diag["rhat"], active_indices, N_FULL_PARAMS)
    diag_ess = extract_active(mcmc_diag["ess"], active_indices, N_FULL_PARAMS)

    # Validate all arrays have consistent lengths before DataFrame creation
    expected_len = n_active
    array_lengths = {
        "param_names": len(results["param_names"]),
        "MAP": len(MAP_active),
        "mean": len(mean_active),
        "median": len(ci_median),
        "hdi_lower": len(ci_hdi_lower),
        "rhat": len(diag_rhat),
        "ess": len(diag_ess),
    }

    mismatches = {k: v for k, v in array_lengths.items() if v != expected_len}
    if mismatches:
        logger.error(f"Array length mismatches (expected {expected_len}): {mismatches}")
        raise ValueError(f"Cannot create parameter summary: array length mismatches {mismatches}")

    param_summary = pd.DataFrame(
        {
            "name": results["param_names"],
            "index": active_indices,
            "MAP": MAP_active,
            "mean": mean_active,
            "median": ci_median,
            "hdi_lower": ci_hdi_lower,
            "hdi_upper": ci_hdi_upper,
            "et_lower": ci_et_lower,
            "et_upper": ci_et_upper,
            "rhat": diag_rhat,
            "ess": diag_ess,
        }
    )
    param_summary.to_csv(output_dir / "parameter_summary.csv", index=False)

    # =========================================================================
    # IMPROVEMENT 6: Save Model Evidence
    # =========================================================================
    if results["evidence"] is not None:
        save_json(output_dir / "model_evidence.json", results["evidence"])

    # =========================================================================
    # IMPROVEMENT 14: Parameter Correlation Analysis
    # =========================================================================
    if args.correlation_analysis:
        logger.info("Computing parameter correlation analysis...")
        try:
            corr_results = compute_correlation_matrix(results["samples"], results["param_names"])
            save_json(output_dir / "correlation_analysis.json", corr_results)

            # Generate correlation plots
            plot_correlation_matrix(corr_results, output_dir)

            if corr_results["identifiability_warning"]:
                logger.warning(
                    f"Identifiability warning: {corr_results['n_high_correlations']} "
                    f"highly correlated parameter pairs detected!"
                )
                for pair in corr_results["high_correlation_pairs"][:5]:
                    logger.warning(
                        f"  {pair['param1']} <-> {pair['param2']}: r={pair['correlation']:.3f}"
                    )

            logger.info(
                f"Correlation analysis complete. Condition number: {corr_results['condition_number']:.1f}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute correlation analysis: {e}")

    # =========================================================================
    # IMPROVEMENT 11: WAIC / Information Criteria
    # =========================================================================
    if args.compute_waic:
        logger.info("Computing WAIC and information criteria...")
        try:
            # We need to recreate the evaluator for WAIC computation
            model_constants = get_model_constants()
            nishioka_bounds, LOCKED_INDICES = get_condition_bounds(args.condition, args.cultivation)
            active_indices_waic = [i for i in range(20) if i not in LOCKED_INDICES]

            theta_base_waic = np.zeros(20)
            for i, (low, high) in enumerate(nishioka_bounds):
                theta_base_waic[i] = (low + high) / 2.0
            for idx in LOCKED_INDICES:
                theta_base_waic[idx] = 0.0

            solver_kwargs_waic = {
                "dt": args.dt,
                "maxtimestep": args.maxtimestep,
                "c_const": args.c_const,
                "alpha_const": args.alpha_const,
                "phi_init": phi_init_array if phi_init_array is not None else args.phi_init,
                "Kp1": args.kp1,
                "K_hill": args.K_hill,
                "n_hill": args.n_hill,
            }

            sigma_obs_waic = (
                args.sigma_obs if args.sigma_obs else metadata.get("sigma_obs_estimated", 0.05)
            )

            debug_config_waic = DebugConfig(level=DebugLevel.ERROR)
            debug_logger_waic = DebugLogger(debug_config_waic)

            evaluator_waic = LogLikelihoodEvaluator(
                solver_kwargs=solver_kwargs_waic,
                active_species=model_constants["active_species"],
                active_indices=active_indices_waic,
                theta_base=theta_base_waic,
                data=data,
                idx_sparse=idx_sparse,
                sigma_obs=sigma_obs_waic,
                cov_rel=args.cov_rel,
                rho=0.0,
                theta_linearization=theta_base_waic,
                paper_mode=False,
                debug_logger=debug_logger_waic,
                use_absolute_volume=args.use_absolute_volume,
                weights=likelihood_weights,
            )

            waic_results = compute_information_criteria(
                samples=results["samples"],
                evaluator=evaluator_waic,
                data=data,
                idx_sparse=idx_sparse,
                active_indices=active_indices_waic,
                theta_base=theta_base_waic,
                n_subsample=min(500, results["samples"].shape[0]),
            )

            save_json(output_dir / "waic_results.json", waic_results)
            logger.info(f"WAIC (approximate): {waic_results['waic_approximate']:.2f}")
            logger.info(
                f"lppd: {waic_results['lppd_approximate']:.2f}, p_waic: {waic_results['p_waic_approximate']:.2f}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute WAIC: {e}")

    save_json(
        output_dir / "theta_mean.json",
        {
            "theta_sub": mean_active.tolist(),  # Active parameters only
            "theta_full": results["theta_mean_full"].tolist(),
            "active_indices": results["active_indices"],
        },
    )

    save_json(
        output_dir / "results_summary.json",
        {
            "elapsed_time": results["elapsed_time"],
            "converged": [bool(c) for c in results["converged"]],
            "MAP": results["MAP"].tolist(),
            "mean": results["mean"].tolist(),
        },
    )

    # Export TMCMC diagnostics tables (if available)
    if "diagnostics" in results and results["diagnostics"]:
        try:
            from visualization import export_tmcmc_diagnostics_tables

            export_tmcmc_diagnostics_tables(
                output_dir, f"{args.condition}_{args.cultivation}", results["diagnostics"]
            )
            logger.info("Exported TMCMC diagnostics tables")
        except Exception as e:
            logger.warning(f"Failed to export TMCMC diagnostics tables: {e}")

    # Generate plots
    logger.info("Generating plots...")
    plot_mgr = PlotManager(str(figures_dir))

    # Run simulation with MAP estimate using same initial conditions
    phi_init_for_plot = phi_init_array if phi_init_array is not None else args.phi_init
    solver = BiofilmNewtonSolver5S(
        dt=args.dt,
        maxtimestep=args.maxtimestep,
        c_const=args.c_const,
        alpha_const=args.alpha_const,
        phi_init=phi_init_for_plot,
        Kp1=args.kp1,
    )

    # Plot fit vs data
    active_species = [0, 1, 2, 3, 4]
    name_tag = f"{args.condition}_{args.cultivation}"

    # Parameter names for plotting
    param_names = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",  # M1
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",  # M2
        "a13",
        "a14",
        "a23",
        "a24",  # M3
        "a55",
        "b5",  # M4
        "a15",
        "a25",
        "a35",
        "a45",  # M5
    ]

    # Run simulations for MAP and Mean estimates
    logger.info("Running simulations for MAP and Mean estimates...")
    t_fit, x_fit_map = solver.solve(results["theta_MAP_full"])
    phibar_map = compute_phibar(x_fit_map, active_species)

    _, x_fit_mean = solver.solve(results["theta_mean_full"])
    phibar_mean = compute_phibar(x_fit_mean, active_species)

    # 1. MAP Fit
    try:
        plot_mgr.plot_TSM_simulation(
            t_fit,
            x_fit_map,
            active_species,
            f"{name_tag}_MAP_Fit",
            data,
            idx_sparse,
            phibar=phibar_map,
            t_days=t_days,
        )
    except Exception as e:
        logger.warning(f"Failed to generate MAP fit plot: {e}")

    # 2. Mean Fit (NEW)
    try:
        plot_mgr.plot_TSM_simulation(
            t_fit,
            x_fit_mean,
            active_species,
            f"{name_tag}_Mean_Fit",
            data,
            idx_sparse,
            phibar=phibar_mean,
            t_days=t_days,
        )
    except Exception as e:
        logger.warning(f"Failed to generate Mean fit plot: {e}")

    # 3. Residuals
    try:
        plot_mgr.plot_residuals(
            t_fit,
            phibar_map,
            data,
            idx_sparse,
            active_species,
            f"{name_tag}_Residuals",
            t_days=t_days,
        )
    except Exception as e:
        logger.warning(f"Failed to generate residuals plot: {e}")

    # 4. Parameter Distributions (Trace/Hist)
    try:
        plot_mgr.plot_trace(results["samples"], results["logL"], param_names, f"{name_tag}_Params")
    except Exception as e:
        logger.warning(f"Failed to generate parameter distribution plot: {e}")

    # 5. Corner Plot
    try:
        plot_mgr.plot_corner(results["samples"], param_names, f"{name_tag}_Corner")
    except Exception as e:
        logger.warning(f"Failed to generate corner plot: {e}")

    # 6 & 7. Posterior Predictive Plots
    logger.info("Generating posterior predictive plots (sampling 50 trajectories)...")
    n_plot_samples = 50
    n_total_samples = results["samples"].shape[0]

    if n_total_samples < n_plot_samples:
        logger.warning(
            f"Number of samples ({n_total_samples}) is less than requested for plotting ({n_plot_samples}). Using all samples."
        )
        indices = np.arange(n_total_samples)
    else:
        indices = np.random.choice(n_total_samples, n_plot_samples, replace=False)

    try:
        phibar_samples = []
        for i, idx in enumerate(indices):
            theta_sample = results["samples"][idx]
            theta_full = results["theta_MAP_full"].copy()
            theta_full[list(range(20))] = theta_sample

            _, x_sim = solver.solve(theta_full)
            phibar_sim = compute_phibar(x_sim, active_species)
            phibar_samples.append(phibar_sim)

        phibar_samples = np.array(phibar_samples)  # (n_draws, n_time, n_species)

        # Posterior Band
        try:
            plot_mgr.plot_posterior_predictive_band(
                t_fit,
                phibar_samples,
                active_species,
                f"{name_tag}_PosteriorBand",
                data,
                idx_sparse,
                t_days=t_days,
            )
        except Exception as e:
            logger.warning(f"Failed to generate posterior band plot: {e}")

        # Spaghetti Plot
        try:
            plot_mgr.plot_posterior_predictive_spaghetti(
                t_fit,
                phibar_samples,
                active_species,
                f"{name_tag}_PosteriorSpaghetti",
                data,
                idx_sparse,
                num_trajectories=50,
                t_days=t_days,
            )
        except Exception as e:
            logger.warning(f"Failed to generate spaghetti plot: {e}")

    except Exception as e:
        logger.warning(f"Failed to generate posterior predictive plots: {e}")

    # 8. TMCMC Beta Schedule (if diagnostics available)
    if "diagnostics" in results and results["diagnostics"]:
        diag = results["diagnostics"]
        try:
            if "beta_schedules" in diag:
                plot_mgr.plot_beta_schedule(diag["beta_schedules"], name_tag)
        except Exception as e:
            logger.warning(f"Failed to generate beta schedule plot: {e}")

    # 9. Compute and save fit metrics
    try:
        metrics_map = compute_fit_metrics(t_fit, x_fit_map, active_species, data, idx_sparse)
        metrics_map["estimate_type"] = "MAP"

        metrics_mean = compute_fit_metrics(t_fit, x_fit_mean, active_species, data, idx_sparse)
        metrics_mean["estimate_type"] = "Mean"

        # Convert numpy arrays to lists for JSON
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        fit_metrics = {
            "MAP": to_serializable(metrics_map),
            "Mean": to_serializable(metrics_mean),
            "species_names": [
                "S. oralis",
                "A. naeslundii",
                "V. dispar",
                "F. nucleatum",
                "P. gingivalis",
            ],
        }

        with open(output_dir / "fit_metrics.json", "w") as f:
            json.dump(fit_metrics, f, indent=2)
        logger.info("Saved fit metrics to fit_metrics.json")

    except Exception as e:
        logger.warning(f"Failed to compute/save fit metrics: {e}")

    # Save figure manifest
    try:
        plot_mgr.save_manifest()
    except Exception as e:
        logger.warning(f"Failed to save manifest: {e}")

    logger.info(f"Estimation complete. Results saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("ESTIMATION SUMMARY (Nishioka v2)")
    print("=" * 80)
    print(f"Condition: {args.condition} {args.cultivation}")
    print(f"Elapsed Time: {results['elapsed_time']:.2f}s")
    print("-" * 80)

    # Convergence diagnostics
    mcmc_diag = results["mcmc_diagnostics"]
    print("CONVERGENCE DIAGNOSTICS:")
    print(
        f"  R-hat max:     {mcmc_diag['rhat_max']:.4f} {'✓' if mcmc_diag['converged_rhat'] else '✗'} (target < 1.1)"
    )
    print(
        f"  ESS min:       {mcmc_diag['ess_min']:.1f} {'✓' if mcmc_diag['converged_ess'] else '✗'} (target > 100)"
    )
    print(f"  Overall:       {'CONVERGED' if mcmc_diag['converged'] else 'NOT CONVERGED'}")
    print("-" * 80)

    # Model evidence (if computed)
    if results["evidence"] is not None:
        ev = results["evidence"]
        print("MODEL EVIDENCE:")
        print(
            f"  log(p(D)):     {ev['log_evidence']:.2f} ± {ev['log_evidence_se']:.2f} ({ev['method']})"
        )
        print(f"  BIC:           {ev['BIC']:.2f}")
        print("-" * 80)

    # Parameter estimates with HDI
    ci = results["credible_intervals"]
    print("PARAMETER ESTIMATES (with 95% HDI):")
    print("-" * 80)
    print(
        f"{'Name':<8} {'MAP':>10} {'Mean':>10} {'HDI_low':>10} {'HDI_high':>10} {'R-hat':>8} {'ESS':>8}"
    )
    print("-" * 80)

    for i, name in enumerate(results["param_names"]):
        # Use MAP_active/mean_active which are correctly extracted for active indices
        map_val = MAP_active[i]
        mean_val = mean_active[i]
        hdi_l = ci["hdi_lower"][i]
        hdi_h = ci["hdi_upper"][i]
        rhat_val = mcmc_diag["rhat"][i]
        ess_val = mcmc_diag["ess"][i]

        rhat_str = f"{rhat_val:.3f}" if not np.isnan(rhat_val) else "N/A"
        print(
            f"{name:<8} {map_val:>10.4f} {mean_val:>10.4f} {hdi_l:>10.4f} {hdi_h:>10.4f} {rhat_str:>8} {ess_val:>8.1f}"
        )

    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

    # =========================================================================
    # Send Slack notification on completion (single condition mode)
    # =========================================================================
    if notifier:
        notifier.notify_complete(
            args.condition,
            args.cultivation,
            results["elapsed_time"],
            mcmc_diag["converged"],
            float(np.max(results["logL"])),
            str(output_dir),
        )


if __name__ == "__main__":
    main()
