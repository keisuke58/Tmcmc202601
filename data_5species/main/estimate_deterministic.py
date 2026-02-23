#!/usr/bin/env python3
"""
Deterministic Parameter Estimation (MLE) for 5-Species Biofilm Model.

This script uses standard numerical optimization (scipy.optimize) to find
the Maximum Likelihood Estimate (MLE) of the parameters.
This is a deterministic alternative to TMCMC (stochastic sampling).

Method:
- Objective Function: Negative Log-Likelihood (equivalent to weighted SSE)
- Optimizer: L-BFGS-B, Nelder-Mead, Differential Evolution, Basin-Hopping, Dual Annealing
- Result: Single "best fit" parameter set (Point Estimate)

Improvements (Nishioka v2):
1. Adaptive Linearization - Update linearization point during optimization
2. Latin Hypercube Sampling - Better coverage for multi-start
3. Confidence Intervals - Hessian-based standard errors
4. Model Selection - AIC/BIC computation
5. Global Optimization - Multiple optimizer options

Usage:
    python estimate_deterministic.py --condition Commensal --cultivation Static
    python estimate_deterministic.py --condition Dysbiotic --cultivation HOBIC --optimizer de --num-starts 10
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
from scipy.stats import qmc  # Latin Hypercube Sampling
from scipy.optimize import approx_fprime
import warnings

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_5SPECIES_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_CONFIGS,
    setup_logging,
    DebugConfig,
    DebugLevel,
    PRIOR_BOUNDS_DEFAULT,
)
from core import LogLikelihoodEvaluator
from debug import DebugLogger
from utils import save_json, save_npy
from improved_5species_jit import BiofilmNewtonSolver5S, HAS_NUMBA
try:
    from data_5species.core.nishioka_model import get_condition_bounds, get_model_constants
except ImportError:
    from core.nishioka_model import get_condition_bounds, get_model_constants

# Reuse data loading from existing script
from estimate_reduced_nishioka import load_experimental_data, convert_days_to_model_time

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Improvement 2: Latin Hypercube Sampling for Multi-Start
# =============================================================================
def generate_lhs_samples(bounds: List[Tuple[float, float]], n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube Samples within given bounds.

    This provides better coverage of the parameter space compared to random sampling.

    Args:
        bounds: List of (low, high) tuples for each dimension
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, n_dims) with samples in bounds
    """
    n_dims = len(bounds)
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples_unit = sampler.random(n=n_samples)  # Samples in [0, 1]^d

    # Scale to actual bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    samples_scaled = qmc.scale(samples_unit, lower, upper)

    return samples_scaled


# =============================================================================
# Improvement 3: Confidence Intervals via Hessian (IMPROVED)
# =============================================================================
def compute_hessian_and_ci(
    objective_func,
    theta_mle: np.ndarray,
    bounds: List[Tuple[float, float]],
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Compute Hessian at MLE and derive confidence intervals.

    Uses numerical differentiation with ADAPTIVE step sizes.
    The inverse of the negative Hessian gives the covariance matrix (Fisher Information).

    Args:
        objective_func: Negative log-likelihood function
        theta_mle: MLE parameter estimate
        bounds: Parameter bounds (for numerical stability)
        confidence_level: Confidence level for intervals (default 95%)

    Returns:
        Dictionary with Hessian, covariance, standard errors, and CIs
    """
    from scipy.stats import norm

    n_params = len(theta_mle)
    f0 = objective_func(theta_mle)

    # Adaptive step sizes based on parameter magnitude and bounds
    eps_array = np.zeros(n_params)
    for i in range(n_params):
        low, high = bounds[i]
        param_scale = max(abs(theta_mle[i]), (high - low) / 10, 1e-6)
        eps_array[i] = min(1e-4 * param_scale, (high - low) / 100)
        eps_array[i] = max(eps_array[i], 1e-8)  # Minimum step

    # Compute Hessian via finite differences with adaptive steps
    hessian = np.zeros((n_params, n_params))

    for i in range(n_params):
        eps_i = eps_array[i]
        for j in range(i, n_params):
            eps_j = eps_array[j]

            # Four-point formula for mixed partial derivatives
            theta_pp = theta_mle.copy()
            theta_pm = theta_mle.copy()
            theta_mp = theta_mle.copy()
            theta_mm = theta_mle.copy()

            theta_pp[i] += eps_i
            theta_pp[j] += eps_j
            theta_pm[i] += eps_i
            theta_pm[j] -= eps_j
            theta_mp[i] -= eps_i
            theta_mp[j] += eps_j
            theta_mm[i] -= eps_i
            theta_mm[j] -= eps_j

            # Clip to bounds
            for k in range(n_params):
                low, high = bounds[k]
                theta_pp[k] = np.clip(theta_pp[k], low, high)
                theta_pm[k] = np.clip(theta_pm[k], low, high)
                theta_mp[k] = np.clip(theta_mp[k], low, high)
                theta_mm[k] = np.clip(theta_mm[k], low, high)

            f_pp = objective_func(theta_pp)
            f_pm = objective_func(theta_pm)
            f_mp = objective_func(theta_mp)
            f_mm = objective_func(theta_mm)

            # Check for numerical issues
            if any(f > 1e15 for f in [f_pp, f_pm, f_mp, f_mm]):
                hessian[i, j] = 0.0  # Mark as unidentifiable
            else:
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps_i * eps_j)
            hessian[j, i] = hessian[i, j]  # Symmetric

    # Check Hessian quality
    eigenvalues = np.linalg.eigvalsh(hessian)
    is_positive_definite = np.all(eigenvalues > 0)
    condition_number = max(eigenvalues) / max(min(eigenvalues), 1e-15)

    # Covariance matrix = inverse of Hessian (for negative log-likelihood)
    try:
        if not is_positive_definite:
            # Regularize to make positive definite
            min_eig = min(eigenvalues)
            regularization = abs(min_eig) + 1e-6
            hessian_reg = hessian + np.eye(n_params) * regularization
            logger.warning(f"Hessian not positive definite. Added regularization: {regularization:.2e}")
        else:
            hessian_reg = hessian + np.eye(n_params) * 1e-10  # Small regularization

        cov_matrix = np.linalg.inv(hessian_reg)

        # Check for valid variances
        variances = np.diag(cov_matrix)

        # Detect problematic parameters (very large or negative variance)
        problematic = (variances < 0) | (variances > 1e6)

        std_errors = np.zeros(n_params)
        for i in range(n_params):
            if problematic[i] or variances[i] <= 0:
                # Use bound-based fallback for unidentifiable parameters
                low, high = bounds[i]
                std_errors[i] = (high - low) / 4  # Conservative estimate
                logger.warning(f"Parameter {i} may be unidentifiable (var={variances[i]:.2e})")
            else:
                std_errors[i] = np.sqrt(variances[i])

        # Confidence intervals
        z_score = norm.ppf((1 + confidence_level) / 2)
        ci_lower = theta_mle - z_score * std_errors
        ci_upper = theta_mle + z_score * std_errors

        # Clip to bounds
        for i, (low, high) in enumerate(bounds):
            ci_lower[i] = max(ci_lower[i], low)
            ci_upper[i] = min(ci_upper[i], high)

        success = True
        message = f"Hessian computed (cond={condition_number:.1e}, pos_def={is_positive_definite})"

    except np.linalg.LinAlgError as e:
        logger.warning(f"Hessian inversion failed: {e}. Using bound-based fallback.")
        cov_matrix = np.eye(n_params) * 0.1
        std_errors = np.array([(b[1]-b[0])/4 for b in bounds])
        ci_lower = np.array([max(theta_mle[i] - 1.96*std_errors[i], bounds[i][0]) for i in range(n_params)])
        ci_upper = np.array([min(theta_mle[i] + 1.96*std_errors[i], bounds[i][1]) for i in range(n_params)])
        success = False
        message = f"Hessian singular: {e}"
        is_positive_definite = False
        condition_number = np.inf

    return {
        "hessian": hessian,
        "covariance": cov_matrix,
        "std_errors": std_errors,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "success": success,
        "message": message,
        "is_positive_definite": is_positive_definite,
        "condition_number": condition_number,
        "eigenvalues": eigenvalues.tolist()
    }


# =============================================================================
# Improvement 6: Model Selection Metrics (AIC/BIC)
# =============================================================================
def compute_model_selection_metrics(
    logL: float,
    n_params: int,
    n_data_points: int
) -> Dict[str, float]:
    """
    Compute AIC, AICc, and BIC for model selection.

    Args:
        logL: Log-likelihood at MLE
        n_params: Number of free parameters
        n_data_points: Number of data points

    Returns:
        Dictionary with AIC, AICc, BIC values
    """
    k = n_params
    n = n_data_points

    # Akaike Information Criterion
    AIC = 2 * k - 2 * logL

    # Corrected AIC (for small samples)
    if n - k - 1 > 0:
        AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    else:
        AICc = np.inf

    # Bayesian Information Criterion
    BIC = k * np.log(n) - 2 * logL

    return {
        "AIC": AIC,
        "AICc": AICc,
        "BIC": BIC,
        "n_params": k,
        "n_data_points": n,
        "logL": logL
    }


# =============================================================================
# NEW Improvement: Fit Quality Metrics (R², RMSE, NRMSE)
# =============================================================================
def compute_fit_quality_metrics(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    species_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive fit quality metrics.

    Args:
        y_obs: Observed data (n_timepoints, n_species)
        y_pred: Predicted data (n_timepoints, n_species)
        species_names: Names for each species

    Returns:
        Dictionary with R², RMSE, NRMSE, MAE per species and overall
    """
    n_species = y_obs.shape[1]
    if species_names is None:
        species_names = [f"Species_{i}" for i in range(n_species)]

    metrics = {
        "per_species": {},
        "overall": {}
    }

    all_residuals = []

    for i in range(n_species):
        obs = y_obs[:, i]
        pred = y_pred[:, i]
        residuals = obs - pred

        # R² (coefficient of determination)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0

        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))

        # NRMSE (normalized by range)
        data_range = np.max(obs) - np.min(obs)
        nrmse = rmse / (data_range + 1e-10) if data_range > 1e-10 else rmse

        # MAE
        mae = np.mean(np.abs(residuals))

        # Max absolute error
        max_error = np.max(np.abs(residuals))

        metrics["per_species"][species_names[i]] = {
            "R2": float(r2),
            "RMSE": float(rmse),
            "NRMSE": float(nrmse),
            "MAE": float(mae),
            "max_error": float(max_error),
            "mean_obs": float(np.mean(obs)),
            "mean_pred": float(np.mean(pred))
        }

        all_residuals.extend(residuals.tolist())

    # Overall metrics
    all_residuals = np.array(all_residuals)
    all_obs = y_obs.flatten()
    all_pred = y_pred.flatten()

    ss_res_total = np.sum((all_obs - all_pred)**2)
    ss_tot_total = np.sum((all_obs - np.mean(all_obs))**2)

    metrics["overall"] = {
        "R2": float(1 - ss_res_total / (ss_tot_total + 1e-10)),
        "RMSE": float(np.sqrt(np.mean(all_residuals**2))),
        "MAE": float(np.mean(np.abs(all_residuals))),
        "max_error": float(np.max(np.abs(all_residuals)))
    }

    return metrics


# =============================================================================
# NEW Improvement: Convergence Diagnostics
# =============================================================================
def check_convergence_quality(
    objective_func,
    theta_mle: np.ndarray,
    bounds: List[Tuple[float, float]],
    tol: float = 1e-5
) -> Dict[str, Any]:
    """
    Check if the optimizer truly converged to a local minimum.

    Args:
        objective_func: Objective function (negative log-likelihood)
        theta_mle: Candidate MLE
        bounds: Parameter bounds
        tol: Tolerance for gradient check

    Returns:
        Dictionary with convergence diagnostics
    """
    n_params = len(theta_mle)

    # 1. Gradient check (should be near zero at minimum)
    gradient = approx_fprime(theta_mle, objective_func, epsilon=1e-6)
    grad_norm = np.linalg.norm(gradient)

    # 2. Check if at boundary (common cause of false convergence)
    at_boundary = []
    for i in range(n_params):
        low, high = bounds[i]
        if abs(theta_mle[i] - low) < 1e-6:
            at_boundary.append((i, "lower"))
        elif abs(theta_mle[i] - high) < 1e-6:
            at_boundary.append((i, "upper"))

    # 3. Check if at midpoint (suspicious - may indicate no sensitivity)
    at_midpoint = []
    for i in range(n_params):
        low, high = bounds[i]
        mid = (low + high) / 2
        if abs(theta_mle[i] - mid) < 0.01 * (high - low):
            at_midpoint.append(i)

    # 4. Local perturbation test
    n_worse = 0
    n_better = 0
    f_mle = objective_func(theta_mle)

    for i in range(n_params):
        for direction in [-1, 1]:
            theta_perturb = theta_mle.copy()
            low, high = bounds[i]
            step = 0.01 * (high - low) * direction
            theta_perturb[i] = np.clip(theta_mle[i] + step, low, high)

            f_perturb = objective_func(theta_perturb)
            if f_perturb < f_mle - 1e-8:
                n_better += 1
            elif f_perturb > f_mle + 1e-8:
                n_worse += 1

    is_local_minimum = (n_better == 0) and (grad_norm < tol * 100)

    return {
        "gradient_norm": float(grad_norm),
        "is_gradient_small": grad_norm < tol,
        "at_boundary": at_boundary,
        "n_at_boundary": len(at_boundary),
        "at_midpoint": at_midpoint,
        "n_at_midpoint": len(at_midpoint),
        "is_local_minimum": is_local_minimum,
        "perturbation_test": {
            "n_better": n_better,
            "n_worse": n_worse
        },
        "warnings": []
    }


# =============================================================================
# NEW Improvement: Profile Likelihood CI (more robust)
# =============================================================================
# =============================================================================
# NEW Improvement: Automated Recommendations
# =============================================================================
def generate_recommendations(result: Dict[str, Any], fit_metrics: Dict[str, Any]) -> List[str]:
    """
    Generate actionable recommendations based on estimation results.

    Args:
        result: Estimation results dictionary
        fit_metrics: Fit quality metrics dictionary

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # 1. Check overall fit quality
    overall_r2 = fit_metrics.get("overall", {}).get("R2", -999)
    if overall_r2 < 0:
        recommendations.append(
            "CRITICAL: Negative R² indicates model fit is worse than mean. "
            "Consider: (1) different model structure, (2) data quality check, "
            "(3) global optimizer (--optimizer dual_annealing)"
        )
    elif overall_r2 < 0.5:
        recommendations.append(
            "Poor fit (R² < 0.5). Try global optimizer: --optimizer DE --maxiter 1000"
        )

    # 2. Check per-species fit
    poor_species = []
    for sp_name, metrics in fit_metrics.get("per_species", {}).items():
        if metrics.get("R2", 0) < 0:
            poor_species.append(sp_name)
    if poor_species:
        recommendations.append(
            f"Species with negative R²: {poor_species}. "
            f"These species may need different dynamics or may be near detection limit."
        )

    # 3. Check convergence
    conv = result.get("convergence", {})
    if conv.get("n_at_midpoint", 0) > 0:
        n_mid = conv["n_at_midpoint"]
        recommendations.append(
            f"{n_mid} parameters at midpoint (unidentifiable). "
            f"Consider: (1) fixing these parameters, (2) using informative priors in TMCMC, "
            f"(3) collecting more data points"
        )

    if conv.get("n_at_boundary", 0) > 0:
        recommendations.append(
            "Parameters at boundary. Check if bounds are biologically reasonable."
        )

    if not conv.get("is_local_minimum", True):
        recommendations.append(
            "May not be a true minimum. Use global optimizer: --optimizer dual_annealing"
        )

    # 4. Check Hessian quality
    if not result.get("hessian_is_positive_definite", True):
        recommendations.append(
            "Hessian not positive definite (flat likelihood). "
            "Parameter uncertainties are unreliable. Consider TMCMC for proper posterior."
        )

    cond_num = result.get("hessian_condition_number", 1)
    if cond_num > 1e10:
        recommendations.append(
            f"Very high Hessian condition number ({cond_num:.1e}). "
            f"Parameters are highly correlated or unidentifiable."
        )

    # 5. Suggest next steps
    if overall_r2 < 0.8:
        recommendations.append(
            "Try TMCMC Bayesian estimation for uncertainty quantification: "
            "python estimate_reduced_nishioka.py --condition <cond> --cultivation <cult>"
        )

    if not recommendations:
        recommendations.append("Results look reasonable. Consider running TMCMC for full posterior.")

    return recommendations


def compute_profile_likelihood_ci(
    objective_func,
    theta_mle: np.ndarray,
    bounds: List[Tuple[float, float]],
    param_idx: int,
    confidence_level: float = 0.95,
    n_points: int = 20
) -> Tuple[float, float]:
    """
    Compute confidence interval using profile likelihood.

    More robust than Hessian-based CIs for non-quadratic likelihoods.

    Args:
        objective_func: Negative log-likelihood function
        theta_mle: MLE parameter estimate
        bounds: Parameter bounds
        param_idx: Index of parameter to profile
        confidence_level: Confidence level
        n_points: Number of points for profiling

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy.stats import chi2

    # Chi-squared threshold for confidence interval
    delta_chi2 = chi2.ppf(confidence_level, df=1) / 2  # Divide by 2 for log-likelihood

    f_mle = objective_func(theta_mle)
    theta_mle_val = theta_mle[param_idx]
    low, high = bounds[param_idx]

    # Profile in both directions
    ci_lower = low
    ci_upper = high

    # Search for lower bound
    for frac in np.linspace(0, 1, n_points):
        test_val = theta_mle_val - frac * (theta_mle_val - low)
        theta_test = theta_mle.copy()
        theta_test[param_idx] = test_val

        f_test = objective_func(theta_test)

        if f_test - f_mle > delta_chi2:
            # Found the boundary
            ci_lower = test_val
            break

    # Search for upper bound
    for frac in np.linspace(0, 1, n_points):
        test_val = theta_mle_val + frac * (high - theta_mle_val)
        theta_test = theta_mle.copy()
        theta_test[param_idx] = test_val

        f_test = objective_func(theta_test)

        if f_test - f_mle > delta_chi2:
            ci_upper = test_val
            break

    return ci_lower, ci_upper


# =============================================================================
# Improvement 7: Automated Analysis Report
# =============================================================================
def generate_analysis_report(result: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate a text report interpreting the estimated parameters.
    """
    theta_map = result["theta_MAP_full"]
    # active_indices = result["active_indices"] # Not used directly, using full map
    
    # Define parameter names and indices (based on BiofilmNewtonSolver5S)
    # 0-4: M1, 5-9: M2, 10-13: M3, 14-15: M4, 16-19: M5
    
    lines = []
    lines.append("==================================================")
    lines.append("       BIOFILM MODEL PARAMETER ANALYSIS           ")
    lines.append("==================================================")
    lines.append(f"Condition: {result.get('condition', 'N/A')} {result.get('cultivation', '')}")
    lines.append(f"Log-Likelihood: {result['logL']:.4f}")
    lines.append(f"AICc: {result.get('AICc', np.nan):.2f}")
    lines.append("")
    
    # 1. Growth/Decay Analysis
    lines.append("--- SPECIES GROWTH & DECAY ---")
    species_names = ["S1 (S.o)", "S2 (A.n)", "S3 (Vei)", "S4 (F.n)", "S5 (P.g)"]
    growth_indices = [0, 2, 5, 7, 14] # a11, a22, a33, a44, a55
    decay_indices = [3, 4, 8, 9, 15]  # b1, b2, b3, b4, b5
    
    for i, name in enumerate(species_names):
        g_idx = growth_indices[i]
        d_idx = decay_indices[i]
        g_val = theta_map[g_idx]
        d_val = theta_map[d_idx]
        
        status = []
        if g_val > 1.5: status.append("Rapid Growth")
        elif g_val < 0.0: status.append("Suppressed")
        else: status.append("Moderate")
        
        if d_val > 1.0: status.append("High Decay")
        elif d_val < 0.2: status.append("Low Decay")
        
        status_str = ", ".join(status)
        lines.append(f"{name:<10}: Growth={g_val:>6.3f}, Decay={d_val:>6.3f} -> {status_str}")

    lines.append("")
    
    # 2. Interaction Analysis
    lines.append("--- STRONG INTERACTIONS (|a_ij| > 0.1) ---")
    # Map of interaction indices to names
    interactions = {
        1: "S1 <-> S2",
        6: "S3 <-> S4",
        10: "S1 <-> S3", 11: "S1 <-> S4",
        12: "S2 <-> S3", 13: "S2 <-> S4",
        16: "S1 <-> S5", 17: "S2 <-> S5",
        18: "S3 <-> S5", 19: "S4 <-> S5"
    }
    
    found_interaction = False
    # Sort by absolute magnitude
    sorted_interactions = sorted(interactions.items(), key=lambda x: abs(theta_map[x[0]]), reverse=True)
    
    for idx, name in sorted_interactions:
        val = theta_map[idx]
        if abs(val) > 0.1:
            type_str = "Cooperation (+)" if val > 0 else "Competition (-)"
            lines.append(f"{name:<10}: {val:>6.3f} ({type_str})")
            found_interaction = True
            
    if not found_interaction:
        lines.append("No strong interactions found.")
        
    lines.append("")
    
    # 3. Parameter Reliability
    if result.get("ci_success", False) and "std_errors" in result:
        lines.append("--- UNCERTAIN PARAMETERS (SE > 0.5) ---")
        THETA_NAMES_FULL = [
            "a11","a12","a22","b1","b2",
            "a33","a34","a44","b3","b4",
            "a13","a14","a23","a24",
            "a55","b5",
            "a15","a25","a35","a45"
        ]
        
        # result["std_errors"] corresponds to active_indices only?
        # Yes, usually. Need to map back.
        # Check if std_errors is full or active.
        # In main, it is saved as list corresponding to active_indices.
        
        std_errors = result["std_errors"]
        active_indices = result["active_indices"]
        
        has_uncertainty = False
        if len(std_errors) == len(active_indices):
            for i, idx in enumerate(active_indices):
                se = std_errors[i]
                if se > 0.5:
                    p_name = THETA_NAMES_FULL[idx]
                    lines.append(f"{p_name:<10}: SE={se:.3f}")
                    has_uncertainty = True
        
        if not has_uncertainty:
            lines.append("All active parameters have SE <= 0.5")

    report_text = "\n".join(lines)
    
    # Save to file
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report_text)
        
    return report_text


# =============================================================================
# Improvement 4: Adaptive Linearization Wrapper
# =============================================================================
class AdaptiveLinearizationOptimizer:
    """
    Wrapper that updates linearization point during optimization.

    TSM (Taylor Series Method) relies on linearization around a point.
    If the optimizer moves far from this point, accuracy degrades.
    This class tracks movement and re-linearizes when needed.
    """

    def __init__(
        self,
        evaluator,
        theta_base: np.ndarray,
        active_indices: List[int],
        relinearization_threshold: float = 0.5,
        min_relinearization_interval: int = 50
    ):
        self.evaluator = evaluator
        self.theta_base = theta_base.copy()
        self.active_indices = active_indices
        self.threshold = relinearization_threshold
        self.min_interval = min_relinearization_interval

        # State
        self.last_linearization_point = None
        self.n_calls = 0
        self.n_relinearizations = 0
        self.calls_since_relinearization = 0

    def _get_full_theta(self, theta_active: np.ndarray) -> np.ndarray:
        """Convert active parameters to full 20D vector."""
        theta_full = self.theta_base.copy()
        theta_full[self.active_indices] = theta_active
        return theta_full

    def _should_relinearize(self, theta_active: np.ndarray) -> bool:
        """Check if we should update linearization point."""
        if self.last_linearization_point is None:
            return True

        if self.calls_since_relinearization < self.min_interval:
            return False

        # Compute relative distance
        diff = theta_active - self.last_linearization_point
        rel_dist = np.linalg.norm(diff) / (np.linalg.norm(self.last_linearization_point) + 1e-8)

        return rel_dist > self.threshold

    def _update_linearization(self, theta_active: np.ndarray):
        """Update the linearization point in the evaluator."""
        theta_full = self._get_full_theta(theta_active)

        # Update evaluator's linearization point if it has the method
        if hasattr(self.evaluator, 'update_linearization_point'):
            self.evaluator.update_linearization_point(theta_full)
        elif hasattr(self.evaluator, 'theta_linearization'):
            self.evaluator.theta_linearization = theta_full

        self.last_linearization_point = theta_active.copy()
        self.n_relinearizations += 1
        self.calls_since_relinearization = 0
        logger.debug(f"Relinearization #{self.n_relinearizations} at call {self.n_calls}")

    def __call__(self, theta_active: np.ndarray) -> float:
        """
        Evaluate log-likelihood with adaptive linearization.

        Returns NEGATIVE log-likelihood for minimization.
        """
        self.n_calls += 1
        self.calls_since_relinearization += 1

        # Check and update linearization if needed
        if self._should_relinearize(theta_active):
            self._update_linearization(theta_active)

        # Evaluate
        logL = self.evaluator(theta_active)

        if logL <= -1e10:  # Failure case
            return 1e20

        return -logL  # Return negative for minimization

    def get_stats(self) -> Dict[str, int]:
        """Return optimization statistics."""
        return {
            "n_calls": self.n_calls,
            "n_relinearizations": self.n_relinearizations
        }


# =============================================================================
# Improvement 1: Multiple Optimization Methods
# =============================================================================
def run_optimizer(
    objective_func,
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    method: str = "L-BFGS-B",
    maxiter: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run optimization with specified method.

    Args:
        objective_func: Function to minimize (negative log-likelihood)
        x0: Initial guess
        bounds: Parameter bounds
        method: One of 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'DE', 'basinhopping', 'dual_annealing'
        maxiter: Maximum iterations
        seed: Random seed for stochastic methods

    Returns:
        Dictionary with optimization result
    """
    np.random.seed(seed)
    start_time = time.time()

    bounds_array = np.array(bounds)

    try:
        if method == "L-BFGS-B":
            res = minimize(
                objective_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': False, 'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-9}
            )

        elif method == "Nelder-Mead":
            # Nelder-Mead doesn't support bounds directly, use penalty
            def bounded_objective(x):
                penalty = 0.0
                for i, (low, high) in enumerate(bounds):
                    if x[i] < low:
                        penalty += 1e6 * (low - x[i])**2
                    elif x[i] > high:
                        penalty += 1e6 * (x[i] - high)**2
                return objective_func(x) + penalty

            res = minimize(
                bounded_objective,
                x0,
                method='Nelder-Mead',
                options={'disp': False, 'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )

        elif method == "Powell":
            res = minimize(
                objective_func,
                x0,
                method='Powell',
                bounds=bounds,
                options={'disp': False, 'maxiter': maxiter, 'ftol': 1e-9}
            )

        elif method == "DE" or method == "differential_evolution":
            res = differential_evolution(
                objective_func,
                bounds=bounds,
                x0=x0,
                maxiter=maxiter,
                seed=seed,
                polish=True,  # Use L-BFGS-B to polish final result
                updating='deferred',
                workers=1,
                disp=False
            )

        elif method == "basinhopping":
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {'maxiter': 500}
            }
            res = basinhopping(
                objective_func,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=maxiter // 10,
                seed=seed,
                disp=False
            )

        elif method == "dual_annealing":
            res = dual_annealing(
                objective_func,
                bounds=bounds,
                x0=x0,
                maxiter=maxiter,
                seed=seed,
                no_local_search=False
            )

        else:
            raise ValueError(f"Unknown optimizer method: {method}")

        elapsed = time.time() - start_time

        return {
            "success": bool(getattr(res, 'success', True)),
            "x": res.x,
            "fun": res.fun,
            "nfev": getattr(res, 'nfev', -1),
            "nit": getattr(res, 'nit', -1),
            "message": str(getattr(res, 'message', '')),
            "elapsed": elapsed,
            "method": method
        }

    except Exception as e:
        logger.error(f"Optimizer {method} failed: {e}")
        return {
            "success": False,
            "x": x0,
            "fun": 1e20,
            "nfev": 0,
            "nit": 0,
            "message": str(e),
            "elapsed": time.time() - start_time,
            "method": method
        }


# =============================================================================
# Main Estimation Function (Improved)
# =============================================================================
def run_deterministic_estimation(
    data: np.ndarray,
    idx_sparse: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    metadata: Dict[str, Any],
    phi_init_array: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run deterministic parameter estimation with improvements:
    - Adaptive linearization
    - Latin Hypercube Sampling for multi-start
    - Multiple optimizer options
    - Confidence interval computation
    - Model selection metrics (AIC/BIC)
    """

    # Setup Logger
    debug_config = DebugConfig(level=DebugLevel.MINIMAL)
    debug_logger = DebugLogger(debug_config)

    # 1. Model Configuration
    if phi_init_array is not None:
        phi_init = phi_init_array
    else:
        phi_init = args.phi_init

    model_constants = get_model_constants()

    solver_kwargs = {
        "dt": args.dt,
        "maxtimestep": args.maxtimestep,
        "c_const": args.c_const,
        "alpha_const": args.alpha_const,
        "phi_init": phi_init,
    }

    active_species = model_constants["active_species"]

    # 2. Parameter Reduction (Nishioka Algorithm)
    logger.info(f"Retrieving bounds for {args.condition} {args.cultivation}...")
    nishioka_bounds, LOCKED_INDICES = get_condition_bounds(args.condition, args.cultivation)

    # Create base parameter vector (20D)
    theta_base = np.zeros(20)
    for i, (low, high) in enumerate(nishioka_bounds):
        theta_base[i] = (low + high) / 2.0

    # Apply locks
    for idx in LOCKED_INDICES:
        theta_base[idx] = 0.0

    # Active indices (free parameters)
    active_indices = [i for i in range(20) if i not in LOCKED_INDICES]
    n_params = len(active_indices)
    logger.info(f"Optimizing {n_params} free parameters. Locked: {LOCKED_INDICES}")

    # Extract bounds for active parameters
    active_bounds = [nishioka_bounds[i] for i in active_indices]

    # 3. Create Evaluator
    sigma_obs = args.sigma_obs if args.sigma_obs else metadata.get('sigma_obs_estimated', 0.05)

    evaluator = LogLikelihoodEvaluator(
        solver_kwargs=solver_kwargs,
        active_species=active_species,
        active_indices=active_indices,
        theta_base=theta_base,
        data=data,
        idx_sparse=idx_sparse,
        sigma_obs=sigma_obs,
        cov_rel=0.02,
        rho=0.0,
        theta_linearization=theta_base,
        debug_logger=debug_logger,
        use_absolute_volume=args.use_absolute_volume,
    )

    # 4. Create Adaptive Linearization Wrapper (Improvement 4)
    adaptive_optimizer = AdaptiveLinearizationOptimizer(
        evaluator=evaluator,
        theta_base=theta_base,
        active_indices=active_indices,
        relinearization_threshold=args.relinearization_threshold,
        min_relinearization_interval=args.min_relinearization_interval
    )

    # 5. Generate Initial Points using LHS (Improvement 2)
    logger.info(f"Generating {args.num_starts} initial points using Latin Hypercube Sampling...")

    if args.num_starts == 1:
        # Single start: use midpoint
        x0_samples = np.array([[(b[0] + b[1])/2.0 for b in active_bounds]])
    else:
        x0_samples = generate_lhs_samples(active_bounds, args.num_starts, seed=args.seed)
        # Replace first sample with midpoint (deterministic reference)
        x0_samples[0] = np.array([(b[0] + b[1])/2.0 for b in active_bounds])

    # 6. Run Multi-Start Optimization (Improvement 1)
    logger.info(f"Starting {args.optimizer} optimization with {args.num_starts} starts...")
    total_start_time = time.time()

    all_results = []
    best_res = None
    best_logL = -np.inf

    for i_start in range(args.num_starts):
        logger.info(f"--- Optimization Run {i_start+1}/{args.num_starts} ---")

        x0 = x0_samples[i_start]

        # Reset adaptive optimizer state for each run
        adaptive_optimizer.last_linearization_point = None
        adaptive_optimizer.n_calls = 0
        adaptive_optimizer.n_relinearizations = 0
        adaptive_optimizer.calls_since_relinearization = 0

        res = run_optimizer(
            objective_func=adaptive_optimizer,
            x0=x0,
            bounds=active_bounds,
            method=args.optimizer,
            maxiter=args.maxiter,
            seed=args.seed + i_start
        )

        # Get adaptive optimizer stats
        res["adaptive_stats"] = adaptive_optimizer.get_stats()

        current_logL = -res["fun"]
        logger.info(
            f"Run {i_start+1}: LogL={current_logL:.4f}, "
            f"Success={res['success']}, Time={res['elapsed']:.2f}s, "
            f"Relinearizations={res['adaptive_stats']['n_relinearizations']}"
        )

        all_results.append(res)

        if current_logL > best_logL:
            best_logL = current_logL
            best_res = res

    elapsed = time.time() - total_start_time

    if best_res is None or best_logL <= -1e10:
        logger.error("All optimization runs failed.")
        return {
            "success": False,
            "message": "All optimization runs failed",
            "MAP": x0_samples[0],
            "theta_MAP_full": theta_base,
            "logL": -1e20,
            "elapsed_time": elapsed,
            "n_evaluations": 0
        }

    logger.info(f"Best LogL: {best_logL:.4f}")

    # 7. Process Best Result
    optimized_theta_active = best_res["x"]
    theta_full = theta_base.copy()
    theta_full[active_indices] = optimized_theta_active

    # 8. Compute Confidence Intervals (Improvement 3 - IMPROVED)
    logger.info("Computing confidence intervals via Hessian...")

    # Use a clean objective for Hessian (without adaptive updates)
    def clean_objective(theta_active):
        logL = evaluator(theta_active)
        return -logL if logL > -1e10 else 1e20

    ci_results = compute_hessian_and_ci(
        clean_objective,
        optimized_theta_active,
        active_bounds,
        confidence_level=0.95
    )

    if ci_results["success"]:
        logger.info(f"CI computation: {ci_results['message']}")
        logger.info(f"Standard errors: {ci_results['std_errors']}")
    else:
        logger.warning(f"CI computation issue: {ci_results['message']}")

    # 8b. Check convergence quality (NEW)
    logger.info("Checking convergence quality...")
    convergence_check = check_convergence_quality(
        clean_objective,
        optimized_theta_active,
        active_bounds
    )

    if convergence_check["n_at_midpoint"] > 0:
        logger.warning(f"WARNING: {convergence_check['n_at_midpoint']} parameters at midpoint - may indicate poor sensitivity")
    if convergence_check["n_at_boundary"] > 0:
        logger.warning(f"WARNING: {convergence_check['n_at_boundary']} parameters at boundary")
    if not convergence_check["is_local_minimum"]:
        logger.warning("WARNING: May not be a true local minimum!")

    # 9. Compute Model Selection Metrics (Improvement 6)
    n_data_points = data.size  # Total number of observations
    model_metrics = compute_model_selection_metrics(
        logL=best_logL,
        n_params=n_params,
        n_data_points=n_data_points
    )

    logger.info(f"Model Selection: AIC={model_metrics['AIC']:.2f}, BIC={model_metrics['BIC']:.2f}")

    # 10. Compile Results
    result = {
        "success": bool(best_res["success"]),
        "message": best_res["message"],
        "optimizer": args.optimizer,
        "MAP": optimized_theta_active,
        "theta_MAP_full": theta_full,
        "logL": best_logL,
        "elapsed_time": elapsed,
        "n_evaluations": sum(r.get("nfev", 0) for r in all_results),
        "n_starts": args.num_starts,
        "all_results_logL": [-r["fun"] for r in all_results],

        # Confidence Intervals (Improvement 3 - IMPROVED)
        "std_errors": ci_results["std_errors"],
        "ci_lower": ci_results["ci_lower"],
        "ci_upper": ci_results["ci_upper"],
        "ci_success": ci_results["success"],
        "hessian_condition_number": ci_results.get("condition_number", np.inf),
        "hessian_is_positive_definite": ci_results.get("is_positive_definite", False),

        # Model Selection (Improvement 6)
        "AIC": model_metrics["AIC"],
        "AICc": model_metrics["AICc"],
        "BIC": model_metrics["BIC"],
        "n_params": n_params,
        "n_data_points": n_data_points,

        # Adaptive Linearization Stats (Improvement 4)
        "total_relinearizations": sum(r.get("adaptive_stats", {}).get("n_relinearizations", 0) for r in all_results),

        # Convergence Diagnostics (NEW)
        "convergence": convergence_check,

        # Metadata
        "active_indices": active_indices,
        "locked_indices": LOCKED_INDICES,
        "data": data,  # Store for fit quality computation
        "idx_sparse": idx_sparse,
    }

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Parameter Estimation (Improved v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with L-BFGS-B
  python estimate_deterministic.py --condition Commensal --cultivation Static

  # Multi-start with Differential Evolution
  python estimate_deterministic.py --condition Dysbiotic --cultivation HOBIC \\
      --optimizer DE --num-starts 10

  # Global search with Basin-Hopping
  python estimate_deterministic.py --condition Dysbiotic --cultivation Static \\
      --optimizer basinhopping --num-starts 5
        """
    )

    # Experimental condition
    parser.add_argument("--condition", type=str, default="Commensal",
                        choices=["Commensal", "Dysbiotic"],
                        help="Experimental condition")
    parser.add_argument("--cultivation", type=str, default="Static",
                        choices=["Static", "HOBIC"],
                        help="Cultivation method")

    # Solver parameters
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="Time step for solver")
    parser.add_argument("--maxtimestep", type=int, default=2500,
                        help="Maximum number of time steps")
    parser.add_argument("--c-const", type=float, default=25.0,
                        help="c constant for model")
    parser.add_argument("--alpha-const", type=float, default=0.0,
                        help="alpha constant for model")
    parser.add_argument("--phi-init", type=float, default=0.02,
                        help="Initial phi value")

    # Data loading
    parser.add_argument("--start-from-day", type=int, default=1,
                        help="Start fitting from this day")
    parser.add_argument("--use-absolute-volume", action="store_true",
                        help="Use absolute volume instead of fractions")
    parser.add_argument("--sigma-obs", type=float, default=None,
                        help="Observation noise (auto-estimated if not provided)")

    # Output
    parser.add_argument("--output-dir", type=str, default="deterministic_results",
                        help="Output directory for results")

    # Optimization settings (Improvement 1)
    parser.add_argument("--optimizer", type=str, default="L-BFGS-B",
                        choices=["L-BFGS-B", "Nelder-Mead", "Powell", "DE",
                                 "differential_evolution", "basinhopping", "dual_annealing"],
                        help="Optimization method")
    parser.add_argument("--num-starts", type=int, default=1,
                        help="Number of multi-start runs (LHS sampling)")
    parser.add_argument("--maxiter", type=int, default=2000,
                        help="Maximum iterations per optimization run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Adaptive Linearization (Improvement 4)
    parser.add_argument("--relinearization-threshold", type=float, default=0.5,
                        help="Relative distance threshold for relinearization")
    parser.add_argument("--min-relinearization-interval", type=int, default=50,
                        help="Minimum calls between relinearizations")

    # Legacy (kept for compatibility)
    parser.add_argument("--random-init", action="store_true",
                        help="[Deprecated] Use --num-starts > 1 instead")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Output setup
    output_dir = DATA_5SPECIES_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    data, days, sigma_obs, phi_init_exp, metadata = load_experimental_data(
        DATA_5SPECIES_ROOT, 
        condition=args.condition,
        cultivation=args.cultivation,
        start_from_day=args.start_from_day
    )
    
    # Time conversion
    t_model, idx_sparse = convert_days_to_model_time(days, args.dt, args.maxtimestep)
    
    # Run Estimation
    result = run_deterministic_estimation(
        data, idx_sparse, args, output_dir, metadata, phi_init_array=phi_init_exp
    )
    
    # Save Results
    save_json(output_dir / "theta_MLE.json", result["theta_MAP_full"].tolist())

    # Save comprehensive results
    results_summary = {
        "condition": args.condition,
        "cultivation": args.cultivation,
        "optimizer": args.optimizer,
        "num_starts": args.num_starts,
        "success": result["success"],
        "logL": result["logL"],
        "AIC": result["AIC"],
        "AICc": result["AICc"],
        "BIC": result["BIC"],
        "n_params": result["n_params"],
        "n_data_points": result["n_data_points"],
        "elapsed_time": result["elapsed_time"],
        "n_evaluations": result["n_evaluations"],
        "total_relinearizations": result["total_relinearizations"],
        "theta_MAP_full": result["theta_MAP_full"].tolist(),
        "active_indices": result["active_indices"],
        "locked_indices": result["locked_indices"],
    }

    # Add CI results if available
    if result["ci_success"]:
        results_summary["std_errors"] = result["std_errors"].tolist()
        results_summary["ci_lower"] = result["ci_lower"].tolist()
        results_summary["ci_upper"] = result["ci_upper"].tolist()

    save_json(output_dir / "estimation_summary.json", results_summary)

    # Save parameter table with CIs as CSV
    param_names = [f"theta_{i}" for i in result["active_indices"]]
    param_df = pd.DataFrame({
        "index": result["active_indices"],
        "name": param_names,
        "MLE": result["MAP"],
        "std_error": result["std_errors"],
        "ci_lower_95": result["ci_lower"],
        "ci_upper_95": result["ci_upper"],
    })
    param_df.to_csv(output_dir / "parameter_estimates.csv", index=False)

    # Print readable result
    print("\n" + "="*70)
    print("DETERMINISTIC ESTIMATION RESULTS (Improved v2)")
    print("="*70)
    print(f"Condition:      {args.condition} {args.cultivation}")
    print(f"Optimizer:      {args.optimizer} ({args.num_starts} starts)")
    print(f"Success:        {result['success']}")
    print(f"Elapsed Time:   {result['elapsed_time']:.2f}s")
    print("-"*70)
    print("LIKELIHOOD & MODEL SELECTION:")
    print(f"  Log-Likelihood: {result['logL']:.4f}")
    print(f"  AIC:            {result['AIC']:.2f}")
    print(f"  AICc:           {result['AICc']:.2f}")
    print(f"  BIC:            {result['BIC']:.2f}")
    print(f"  n_params:       {result['n_params']}")
    print(f"  n_data_points:  {result['n_data_points']}")
    print("-"*70)
    print("ADAPTIVE LINEARIZATION:")
    print(f"  Total Relinearizations: {result['total_relinearizations']}")
    print("-"*70)
    print("PARAMETER ESTIMATES (with 95% CI):")
    print("-"*70)
    print(f"{'Index':<8} {'MLE':>12} {'Std.Err':>12} {'CI_low':>12} {'CI_high':>12}")
    print("-"*70)
    for i, idx in enumerate(result["active_indices"]):
        mle_val = result["MAP"][i]
        se_val = result["std_errors"][i]
        ci_l = result["ci_lower"][i]
        ci_h = result["ci_upper"][i]
        print(f"theta_{idx:<3} {mle_val:>12.6f} {se_val:>12.6f} {ci_l:>12.6f} {ci_h:>12.6f}")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("="*70)

    # 11. Automated Analysis (Improvement 7)
    try:
        analysis_text = generate_analysis_report(results_summary, output_dir)
        print("\n" + analysis_text)
    except Exception as e:
        logger.error(f"Analysis generation failed: {e}")

    # 12. Generate Recommendations (NEW)
    recommendations = generate_recommendations(result, fit_metrics)
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print("="*70)

    # 7. Visualization
    logger.info("Generating fit plots...")

    # Re-creating solver for visualization
    model_constants = get_model_constants()

    solver_kwargs = {
        "dt": args.dt,
        "maxtimestep": args.maxtimestep,
        "c_const": args.c_const,
        "alpha_const": args.alpha_const,
    }

    if phi_init_exp is not None:
        solver_kwargs["phi_init"] = phi_init_exp
    else:
        solver_kwargs["phi_init"] = args.phi_init

    active_species_idx = model_constants['active_species']

    logger.info("Running full simulation for plotting...")
    solver = BiofilmNewtonSolver5S(**solver_kwargs)

    # Construct full parameter vector
    theta_full = result["theta_MAP_full"]

    # Run solver
    result_tuple = solver.solve(theta_full)
    if len(result_tuple) == 2:
        t_sim, y_sim = result_tuple
        sim_success = True
    else:
        sim_success, t_sim, y_sim = result_tuple

    if not sim_success:
        logger.warning("Simulation failed with estimated parameters!")

    # Compute fit quality metrics (NEW)
    logger.info("Computing fit quality metrics...")
    species_names = ['S.o', 'A.n', 'Vei', 'F.n', 'P.g']

    # Extract predicted values at observation times
    y_pred_at_obs = np.zeros_like(data)
    for i, idx in enumerate(idx_sparse):
        if idx < len(y_sim):
            y_pred_at_obs[i, :] = y_sim[idx, active_species_idx]

    fit_metrics = compute_fit_quality_metrics(data, y_pred_at_obs, species_names)

    # Save fit metrics
    save_json(output_dir / "fit_quality_metrics.json", fit_metrics)

    # Print fit quality
    print("\n" + "-"*70)
    print("FIT QUALITY METRICS:")
    print("-"*70)
    print(f"{'Species':<10} {'R²':>8} {'RMSE':>10} {'NRMSE':>10} {'MAE':>10}")
    print("-"*70)
    for sp_name in species_names:
        if sp_name in fit_metrics["per_species"]:
            m = fit_metrics["per_species"][sp_name]
            print(f"{sp_name:<10} {m['R2']:>8.4f} {m['RMSE']:>10.6f} {m['NRMSE']:>10.4f} {m['MAE']:>10.6f}")
    print("-"*70)
    print(f"{'Overall':<10} {fit_metrics['overall']['R2']:>8.4f} {fit_metrics['overall']['RMSE']:>10.6f} {'-':>10} {fit_metrics['overall']['MAE']:>10.6f}")
    print("="*70)

    # Convergence warnings
    conv = result.get("convergence", {})
    if conv.get("n_at_midpoint", 0) > 0:
        print(f"\n⚠️  WARNING: {conv['n_at_midpoint']} parameters at midpoint (indices: {conv.get('at_midpoint', [])})")
        print("   This may indicate parameter unidentifiability or flat likelihood surface.")
    if conv.get("n_at_boundary", 0) > 0:
        print(f"\n⚠️  WARNING: {conv['n_at_boundary']} parameters at boundary")
        print(f"   Boundaries: {conv.get('at_boundary', [])}")

    # Manual plot
    import matplotlib.pyplot as plt

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']  # Better colors
    species_names = ['S.o', 'A.n', 'Vei', 'F.n', 'P.g']

    # Calculate day scale
    day_scale = t_model[-1] / days[-1] if days[-1] > 0 else 1.0
    t_days_sim = t_sim / day_scale

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Plot 1: Model Fit =====
    ax1 = axes[0]
    for i, sp_idx in enumerate(active_species_idx):
        if sp_idx >= 5:
            continue
        # Experimental data
        ax1.scatter(days, data[:, i], label=f"{species_names[sp_idx]} (Exp)",
                    color=colors[sp_idx], marker='o', s=50, alpha=0.8)
        # Simulation
        ax1.plot(t_days_sim, y_sim[:, sp_idx], label=f"{species_names[sp_idx]} (Sim)",
                 color=colors[sp_idx], linestyle='-', linewidth=2)

    ax1.set_title(f"Model Fit: {args.condition} {args.cultivation}\n"
                  f"LogL={result['logL']:.2f}, AIC={result['AIC']:.1f}, BIC={result['BIC']:.1f}",
                  fontsize=11)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Volume Fraction")
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # ===== Plot 2: Parameter Estimates with CI =====
    ax2 = axes[1]

    n_params = len(result["active_indices"])
    x_pos = np.arange(n_params)
    mle_vals = result["MAP"]
    ci_lower = result["ci_lower"]
    ci_upper = result["ci_upper"]

    # Error bars
    errors = np.array([mle_vals - ci_lower, ci_upper - mle_vals])

    bars = ax2.bar(x_pos, mle_vals, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.errorbar(x_pos, mle_vals, yerr=errors, fmt='none', ecolor='red',
                 capsize=3, capthick=1.5, linewidth=1.5)

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"$\\theta_{{{i}}}$" for i in result["active_indices"]],
                        fontsize=9, rotation=45)
    ax2.set_ylabel("Parameter Value")
    ax2.set_title("MLE Estimates with 95% CI", fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = output_dir / "fit_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved fit plot to {plot_path}")

    # ===== Additional Plot: Multi-Start Results =====
    if args.num_starts > 1:
        fig2, ax3 = plt.subplots(figsize=(8, 4))

        all_logL = result["all_results_logL"]
        x_starts = np.arange(1, len(all_logL) + 1)

        ax3.bar(x_starts, all_logL, color='lightblue', edgecolor='black')
        ax3.axhline(y=result["logL"], color='red', linestyle='--',
                    label=f'Best: {result["logL"]:.2f}', linewidth=2)

        ax3.set_xlabel("Start #")
        ax3.set_ylabel("Log-Likelihood")
        ax3.set_title(f"Multi-Start Results ({args.optimizer})")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        multistart_path = output_dir / "multistart_comparison.png"
        plt.savefig(multistart_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved multi-start plot to {multistart_path}")

    # ===== NEW: Residual Analysis Plot =====
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
    axes3 = axes3.flatten()

    residuals = data - y_pred_at_obs

    for i, sp_name in enumerate(species_names):
        if i >= 5:
            break
        ax = axes3[i]

        # Residual plot
        ax.scatter(days, residuals[:, i], color=colors[i], s=60, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.fill_between(days, -0.02, 0.02, alpha=0.2, color='green', label='±0.02')

        ax.set_xlabel("Days")
        ax.set_ylabel("Residual")
        ax.set_title(f"{sp_name}: R²={fit_metrics['per_species'][sp_name]['R2']:.3f}")
        ax.grid(True, alpha=0.3)

    # Summary in last subplot
    ax_sum = axes3[5]
    ax_sum.axis('off')

    summary_text = "FIT QUALITY SUMMARY\n" + "="*30 + "\n\n"
    for sp_name in species_names:
        m = fit_metrics['per_species'][sp_name]
        status = "✓" if m['R2'] > 0.8 else ("~" if m['R2'] > 0.5 else "✗")
        summary_text += f"{status} {sp_name}: R²={m['R2']:.3f}, RMSE={m['RMSE']:.4f}\n"

    summary_text += f"\n{'='*30}\n"
    summary_text += f"Overall R²: {fit_metrics['overall']['R2']:.4f}\n"
    summary_text += f"Overall RMSE: {fit_metrics['overall']['RMSE']:.5f}\n"

    # Convergence warnings
    conv = result.get("convergence", {})
    if conv.get("n_at_midpoint", 0) > 0:
        summary_text += f"\n⚠️ {conv['n_at_midpoint']} params at midpoint"
    if not conv.get("is_local_minimum", True):
        summary_text += f"\n⚠️ May not be local minimum"

    ax_sum.text(0.1, 0.9, summary_text, transform=ax_sum.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f"Residual Analysis: {args.condition} {args.cultivation}", fontsize=12)
    plt.tight_layout()

    residual_path = output_dir / "residual_analysis.png"
    plt.savefig(residual_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved residual analysis to {residual_path}")

    # ===== NEW: Parameter Sensitivity Heatmap =====
    if result.get("ci_success", False):
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        # Compute relative sensitivity (1 / relative_std_error)
        rel_std_errors = result["std_errors"] / (np.abs(result["MAP"]) + 1e-10)
        sensitivity = 1 / (rel_std_errors + 1e-10)
        sensitivity = np.clip(sensitivity, 0, 100)  # Cap for visualization

        param_labels = [f"θ{i}" for i in result["active_indices"]]

        bars = ax4.barh(param_labels, sensitivity, color='steelblue', alpha=0.7)
        ax4.set_xlabel("Relative Sensitivity (higher = more identifiable)")
        ax4.set_title("Parameter Identifiability")
        ax4.grid(True, alpha=0.3, axis='x')

        # Color code by identifiability
        for bar, sens in zip(bars, sensitivity):
            if sens < 1:
                bar.set_color('red')
                bar.set_alpha(0.5)
            elif sens < 10:
                bar.set_color('orange')
            else:
                bar.set_color('green')

        sensitivity_path = output_dir / "parameter_sensitivity.png"
        plt.savefig(sensitivity_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sensitivity plot to {sensitivity_path}")

if __name__ == "__main__":
    main()
