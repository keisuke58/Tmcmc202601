"""
Core MCMC and likelihood evaluation modules for tmcmc package.

This package contains core algorithmic implementations extracted from
case2_tmcmc_linearization.py:
- evaluator: LogLikelihoodEvaluator class for TSM-ROM likelihood evaluation
- tmcmc: TMCMC algorithm implementation
- mcmc: Adaptive MCMC and 2-phase MCMC implementations
"""

from .evaluator import (
    LogLikelihoodEvaluator,
    log_likelihood_sparse,
    build_likelihood_weights,
    build_species_sigma,
)

from .tmcmc import (
    TMCMCResult,
    run_TMCMC,
    run_multi_chain_TMCMC,
    reflect_into_bounds,
    choose_subset_size,
    should_do_fom_check,
    compute_MAP_with_uncertainty,
)

from .mcmc import (
    run_adaptive_MCMC,
    run_two_phase_MCMC_with_linearization,
)

__all__ = [
    # Evaluator
    "LogLikelihoodEvaluator",
    "log_likelihood_sparse",
    "build_likelihood_weights",
    "build_species_sigma",
    # TMCMC
    "TMCMCResult",
    "run_TMCMC",
    "run_multi_chain_TMCMC",
    "reflect_into_bounds",
    "choose_subset_size",
    "should_do_fom_check",
    "compute_MAP_with_uncertainty",
    # MCMC
    "run_adaptive_MCMC",
    "run_two_phase_MCMC_with_linearization",
]
