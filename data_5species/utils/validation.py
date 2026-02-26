"""
Input validation utilities for tmcmc package.

Provides validation functions for MCMC/TMCMC inputs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


def validate_tmcmc_inputs(
    log_likelihood: callable,
    prior_bounds: List[Tuple[float, float]],
    n_particles: int,
    n_stages: int,
    target_ess_ratio: float,
    evaluator: Optional[Any],
    theta_base_full: Optional[np.ndarray],
    active_indices: Optional[List[int]],
) -> None:
    """
    Validate inputs for run_TMCMC.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function
    prior_bounds : List[Tuple[float, float]]
        Prior bounds for each parameter
    n_particles : int
        Number of particles
    n_stages : int
        Number of TMCMC stages
    target_ess_ratio : float
        Target ESS ratio
    evaluator : Any, optional
        LogLikelihoodEvaluator instance
    theta_base_full : np.ndarray, optional
        Full parameter base vector
    active_indices : List[int], optional
        Active parameter indices

    Raises
    ------
    TypeError
        If input types are incorrect
    ValueError
        If input values are invalid
    """
    if not callable(log_likelihood):
        raise TypeError("log_likelihood must be callable")

    if not isinstance(prior_bounds, list) or len(prior_bounds) == 0:
        raise ValueError("prior_bounds must be a non-empty list")

    for i, (low, high) in enumerate(prior_bounds):
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError(f"prior_bounds[{i}] must be numeric tuple")
        if low > high:
            raise ValueError(f"prior_bounds[{i}]: lower bound must be <= upper bound")

    if n_particles <= 0:
        raise ValueError(f"n_particles must be > 0, got {n_particles}")

    if n_stages <= 0:
        raise ValueError(f"n_stages must be > 0, got {n_stages}")

    if not (0 < target_ess_ratio <= 1):
        raise ValueError(f"target_ess_ratio must be in (0, 1], got {target_ess_ratio}")

    if evaluator is not None:
        if theta_base_full is None:
            raise ValueError("theta_base_full must be provided when evaluator is provided")
        if active_indices is None:
            raise ValueError("active_indices must be provided when evaluator is provided")
        if not isinstance(theta_base_full, np.ndarray):
            raise TypeError("theta_base_full must be numpy array")
        if not isinstance(active_indices, list):
            raise TypeError("active_indices must be list")
