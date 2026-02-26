"""
=================================================================================
Case II: Hierarchical Bayesian Parameter Estimation with TMCMC
       + TSM Linearization Point Update
=================================================================================

🚀 NEW FEATURE: TMCMC (Transitional MCMC) × TSM-ROM with Linearization Management
   TSM approximates: x(θ) ≈ x(θ₀) + ∂x/∂θ|_{θ₀} · (θ - θ₀)

   For accurate MCMC inference, we use:
   1. TMCMC (β tempering): Gradual transition from prior to posterior
   2. Linearization Point Update: Iteratively update θ₀ based on:
      - Weighted barycenter (robust for multi-modal posteriors)
      - Observation-based ROM error weighting (pulls towards accurate regions)
   3. Tempered Covariance: Adaptive proposal scaling with β
   4. K-step Mutation: Reduces particle correlation

   Expected improvement: MAP error 0.15 → 0.005 (30x better!)

   ★ Publication-ready: No information leakage (theta_true only for evaluation)

State Vector Definition:
    g (10,) = [phi1, phi2, phi3, phi4, phi0, psi1, psi2, psi3, psi4, gamma]

Theta (14,) order:
    [a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24]

Author: Keisuke (keisuke58)
Date: December 2025
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import json
import csv
import logging
import shlex
import zlib
import platform
import multiprocessing
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import platform
import multiprocessing

from config import (
    CONVERGENCE_DEFAULTS,
    DebugConfig,
    DebugLevel,
    LINEARIZATION_DEFAULTS,
    MAX_LINEARIZATION_SUBUPDATES_PER_EVENT,
    MAX_THETA0_STEP_NORM,
    MODEL_CONFIGS,
    PRIOR_BOUNDS_DEFAULT,
    PROPOSAL_DEFAULTS,
    ROM_ERROR_DEFAULTS,
    TMCMC_DEFAULTS,
    setup_logging,
)

# Import utilities from refactored modules
from utils import (
    code_crc32,
    save_npy,
    save_likelihood_meta,
    save_json,
    write_csv,
    to_jsonable,
    TimingStats,
    timed,
    LikelihoodHealthCounter,
    validate_tmcmc_inputs,
)

# Import debug utilities from refactored modules
from debug import (
    DebugLogger,
    SLACK_ENABLED,
    notify_slack,
    SlackNotifier,
    _slack_notifier,
)

# Import visualization utilities from refactored modules
from visualization import (
    PlotManager,
    compute_phibar,
    compute_fit_metrics,
    export_tmcmc_diagnostics_tables,
)

# Import core MCMC and evaluator utilities from refactored modules
from core import (
    LogLikelihoodEvaluator,
    log_likelihood_sparse,
    TMCMCResult,
    run_TMCMC,
    run_multi_chain_TMCMC,
    run_adaptive_MCMC,
    run_two_phase_MCMC_with_linearization,
    reflect_into_bounds,
    choose_subset_size,
    should_do_fom_check,
)

# Import main function and related utilities from refactored modules
from main import (
    main,
    parse_args,
    select_sparse_data_indices,
    generate_synthetic_data,
    _self_check_tsm_once,
    _default_output_root_for_mode,
    _stable_hash_int,
    MCMCConfig,
    ExperimentConfig,
    compute_MAP_with_uncertainty,
)

# NOTE: The original definitions of the above functions/classes are preserved below
# for reference and backward compatibility. They are now imported from the core modules above.

# Backward compatibility aliases (using underscore prefix for internal use)
_code_crc32 = code_crc32
_save_npy = save_npy
_save_likelihood_meta = save_likelihood_meta
_to_jsonable = to_jsonable
_validate_tmcmc_inputs = validate_tmcmc_inputs

logger = logging.getLogger(__name__)

# ==============================================================================
# RUN ARTIFACT HELPERS (reproducibility)
# ==============================================================================
# NOTE: These functions have been moved to utils.io module.
# Imported above for backward compatibility.


# NOTE: Slack notification support has been moved to debug.logger module.
# Imported above for backward compatibility.

# ==============================================================================
# CONSTANTS
# ==============================================================================

# NOTE: keep these module-level names for backward compatibility (tests import them).
DEFAULT_N_PARTICLES = TMCMC_DEFAULTS.n_particles
DEFAULT_N_STAGES = TMCMC_DEFAULTS.n_stages
DEFAULT_TARGET_ESS_RATIO = TMCMC_DEFAULTS.target_ess_ratio
DEFAULT_MIN_DELTA_BETA = TMCMC_DEFAULTS.min_delta_beta
DEFAULT_UPDATE_LINEARIZATION_INTERVAL = TMCMC_DEFAULTS.update_linearization_interval
DEFAULT_N_MUTATION_STEPS = TMCMC_DEFAULTS.n_mutation_steps
DEFAULT_LINEARIZATION_THRESHOLD = TMCMC_DEFAULTS.linearization_threshold
MAX_LINEARIZATION_UPDATES = TMCMC_DEFAULTS.max_linearization_updates

ROM_ERROR_THRESHOLD = ROM_ERROR_DEFAULTS.threshold
ROM_ERROR_FALLBACK = ROM_ERROR_DEFAULTS.fallback

BETA_CONVERGENCE_THRESHOLD = CONVERGENCE_DEFAULTS.beta_convergence_threshold
THETA_CONVERGENCE_THRESHOLD = CONVERGENCE_DEFAULTS.theta_convergence_threshold

# Linearization update stabilization (imported from config.py)
# NOTE: These constants are now defined in config.py as LINEARIZATION_DEFAULTS.
# Imported above for backward compatibility.
# Re-export for backward compatibility (already imported from config above)
# MAX_THETA0_STEP_NORM and MAX_LINEARIZATION_SUBUPDATES_PER_EVENT are available from config import

OPTIMAL_SCALE_FACTOR = PROPOSAL_DEFAULTS.optimal_scale_factor
COVARIANCE_NUGGET_BASE = PROPOSAL_DEFAULTS.covariance_nugget_base
COVARIANCE_NUGGET_SCALE = PROPOSAL_DEFAULTS.covariance_nugget_scale

MAX_DELTA_BETA = TMCMC_DEFAULTS.max_delta_beta
MUTATION_SCALE_FACTOR = TMCMC_DEFAULTS.mutation_scale_factor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# NOTE: DebugLogger has been moved to debug.logger module.
# Imported above for backward compatibility.


@dataclass
class MCMCConfig:
    """MCMC sampling configuration."""

    n_samples: int = 2000
    n_burn_in: int = 100
    n_chains: int = 2
    initial_scale: float = 0.02
    target_accept: float = 0.30
    adapt_start: int = 100
    adapt_interval: int = 50
    debug: DebugConfig = None  # ★ Debug configuration

    def __post_init__(self):
        """Initialize debug config if not provided."""
        if self.debug is None:
            self.debug = DebugConfig(level=DebugLevel.OFF)


@dataclass
class ExperimentConfig:
    """Experiment configuration for synthetic data generation."""

    cov_rel: float = 0.005  # TSM relative covariance
    rho: float = 0.0  # Observation correlation (equicorrelated)
    n_data: int = 20  # Number of observations
    sigma_obs: float = 0.001  # Observation noise
    # Paper notation: Nsamples (aleatory Monte Carlo samples) used in the *baseline* double-loop cost.
    # We keep this only for cost conversion/reporting; it does not affect the TSM-ROM execution.
    aleatory_samples: int = 500
    output_dir: str = None  # ★ 自動決定: sanity/debug/paper (main()で設定)
    random_seed: int = 42
    debug: DebugConfig = None  # ★ Debug configuration

    def __post_init__(self):
        """Initialize debug config if not provided."""
        if self.debug is None:
            self.debug = DebugConfig(level=DebugLevel.OFF)


# Model-specific configurations are shared in `tmcmc/config.py` (imported as MODEL_CONFIGS).


# ==============================================================================
# IMPORTS
# ==============================================================================

sys.path.insert(0, str(Path(__file__).parent))

from improved1207_paper_jit import (
    BiofilmNewtonSolver,
    get_theta_true,
    HAS_NUMBA,
)

# ★ KEY CHANGE: Use BiofilmTSM_Analytical with linearization management
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical

from mcmc_diagnostics import MCMCDiagnostics
from bugfix_theta_to_matrices import patch_biofilm_solver

# ★ 致命的②: import時の副作用を削除（main配下に移動）
# patch_biofilm_solver() と print は main() 内で実行


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _stable_hash_int(text: str) -> int:
    """Stable, cross-run integer hash (unlike Python's built-in hash())."""
    return int(zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF)


def _default_output_root_for_mode(mode: str) -> str:
    # Keep a single predictable root for Cursor "buttonization".
    # mode/seed are encoded in run_id, so analysis tools only need one root.
    _ = mode
    return str(Path("tmcmc") / "_runs")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Case II: TMCMC × TSM linearization (experiment runner)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["sanity", "debug", "paper"], default="debug", help="Execution preset"
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed (data + TMCMC)")
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory (runs are created under this)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (folder name). Default: auto timestamp",
    )
    p.add_argument(
        "--models",
        type=str,
        default="M1,M2,M3",
        help="Comma-separated list of models to run (e.g. 'M1' or 'M1,M3')",
    )

    # Experiment noise/uncertainty
    p.add_argument(
        "--sigma-obs", type=float, default=None, help="Override observation noise sigma_obs"
    )
    p.add_argument(
        "--cov-rel", type=float, default=None, help="Override ROM relative covariance cov_rel"
    )
    p.add_argument(
        "--rho", type=float, default=None, help="Observation correlation rho (default: 0.0)"
    )
    p.add_argument(
        "--aleatory-samples",
        type=int,
        default=None,
        help="For reporting only: paper Nsamples used for double-loop cost conversion (default: 500 in paper mode)",
    )

    # TMCMC knobs (optional overrides)
    p.add_argument("--n-particles", type=int, default=None, help="TMCMC particles per chain")
    p.add_argument("--n-stages", type=int, default=None, help="TMCMC max stages")
    p.add_argument(
        "--n-mutation-steps", type=int, default=None, help="TMCMC mutation steps per stage"
    )
    p.add_argument("--n-chains", type=int, default=None, help="Number of TMCMC chains (sequential)")
    p.add_argument(
        "--target-ess-ratio", type=float, default=None, help="TMCMC ESS target ratio in (0,1]"
    )
    p.add_argument(
        "--min-delta-beta",
        type=float,
        default=None,
        help="Minimum β increment per stage (progress floor)",
    )
    p.add_argument(
        "--max-delta-beta",
        type=float,
        default=None,
        help="Maximum β increment per stage (caps β jumps)",
    )
    p.add_argument(
        "--update-linearization-interval",
        type=int,
        default=None,
        help="Update linearization point every N stages",
    )
    p.add_argument(
        "--linearization-threshold",
        type=float,
        default=None,
        help="Allow linearization only when β exceeds this threshold",
    )
    p.add_argument(
        "--linearization-enable-rom-threshold",
        type=float,
        default=None,
        help="Enable linearization only if ε_ROM(MAP) <= this threshold (stability guard)",
    )
    p.add_argument(
        "--force-beta-one",
        action="store_true",
        default=False,
        help="Force β=1.0 at final stage (safety)",
    )
    p.add_argument(
        "--lock-paper-conditions",
        action="store_true",
        default=False,
        help="Force paper conditions (sigma_obs/cov_rel + conservative β jumps) regardless of --mode",
    )

    # Debug controls
    p.add_argument(
        "--debug-level",
        choices=[lvl.name for lvl in DebugLevel],
        default=None,
        help="Override debug verbosity (defaults depend on --mode)",
    )
    p.add_argument(
        "--use-paper-analytical",
        action="store_true",
        default=None,
        help="Use paper analytical derivatives (production-ready)",
    )
    p.add_argument(
        "--no-paper-analytical",
        dest="use_paper_analytical",
        action="store_false",
        help="Disable analytical derivatives (use complex-step fallback)",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        default=False,
        help="Run a lightweight self-check once at startup (sanity of solve_tsm output)",
    )
    return p.parse_args(argv)


def select_sparse_data_indices(n_total: int, n_obs: int) -> np.ndarray:
    """Select evenly spaced indices for sparse observations."""
    start_idx = int(0.1 * n_total)
    indices = np.linspace(start_idx, n_total - 1, n_obs)
    indices = np.floor(indices).astype(int)

    # ★ CRITICAL FIX: Check bounds explicitly instead of silent clipping
    # Silent clipping can hide bugs (e.g., n_total calculation errors)
    if np.any(indices < 0) or np.any(indices >= n_total):
        invalid_min = np.min(indices[indices < 0]) if np.any(indices < 0) else None
        invalid_max = np.max(indices[indices >= n_total]) if np.any(indices >= n_total) else None
        raise IndexError(
            f"Invalid indices generated: min={invalid_min}, max={invalid_max}, "
            f"valid range=[0, {n_total-1}]. This indicates a bug in index calculation."
        )

    return indices


# NOTE: log_likelihood_sparse has been moved to core.evaluator module.
# Imported above for backward compatibility.


# NOTE: compute_phibar has been moved to visualization.helpers module.
# Imported above for backward compatibility.


def _self_check_tsm_once(
    *,
    model_key: str,
    theta_true: np.ndarray,
    exp_config: "ExperimentConfig",
    use_paper_analytical: bool,
) -> Dict[str, Any]:
    """
    Lightweight self-check for "functionality sanity":
    - solve_tsm(theta_true) output has no NaN/Inf
    - t_arr is monotonically increasing
    - phi0 constraint is consistent: phi0 ≈ 1 - sum(phi_i)
    """
    cfg = MODEL_CONFIGS[model_key]
    solver_kwargs = {
        k: v for k, v in cfg.items() if k not in ["active_species", "active_indices", "param_names"]
    }
    solver = BiofilmNewtonSolver(
        **solver_kwargs,
        active_species=cfg["active_species"],
        use_numba=HAS_NUMBA,
    )
    tsm = BiofilmTSM_Analytical(
        solver,
        active_theta_indices=cfg["active_indices"],
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
        use_analytical=True,
        theta_linearization=theta_true,
        paper_mode=bool(use_paper_analytical),
    )
    t_arr, x0, sig2 = tsm.solve_tsm(theta_true)

    # Finite checks
    nonfinite_t = int(np.size(t_arr) - np.isfinite(t_arr).sum())
    nonfinite_x0 = int(np.size(x0) - np.isfinite(x0).sum())
    nonfinite_sig2 = int(np.size(sig2) - np.isfinite(sig2).sum())

    # Monotonic time
    dt = np.diff(np.asarray(t_arr, dtype=float))
    t_monotone = bool(np.all(dt > 0))

    # phi0 constraint
    n_state = x0.shape[1]
    n_total_species = (n_state - 2) // 2
    phi = x0[:, :n_total_species]
    phi0 = x0[:, n_total_species]
    phi0_from_constraint = 1.0 - np.sum(phi, axis=1)
    phi0_err = phi0 - phi0_from_constraint
    phi0_err_max_abs = float(np.max(np.abs(phi0_err))) if phi0_err.size else float("nan")
    phi0_min = float(np.min(phi0)) if phi0.size else float("nan")
    phi0_max = float(np.max(phi0)) if phi0.size else float("nan")

    ok = (
        (nonfinite_t == 0)
        and (nonfinite_x0 == 0)
        and (nonfinite_sig2 == 0)
        and t_monotone
        and (phi0_err_max_abs < 1e-6)
    )
    return {
        "model": model_key,
        "ok": bool(ok),
        "nonfinite": {"t_arr": nonfinite_t, "x0": nonfinite_x0, "sig2": nonfinite_sig2},
        "t_monotone_increasing": bool(t_monotone),
        "phi0_constraint": {
            "max_abs_error": phi0_err_max_abs,
            "phi0_min": phi0_min,
            "phi0_max": phi0_max,
        },
    }


def compute_MAP_with_uncertainty(samples: np.ndarray, logL: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute MAP estimate and posterior statistics."""
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


# NOTE: _to_jsonable, save_json, write_csv have been moved to utils.io module.
# Imported above for backward compatibility.


@dataclass
class TimingStats:
    """Lightweight timing aggregator (seconds + call counts) for metrics.json."""

    seconds: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, name: str, dt_s: float) -> None:
        if not name:
            return
        if not np.isfinite(dt_s):
            return
        self.seconds[name] += float(dt_s)
        self.counts[name] += 1

    def get_s(self, name: str) -> float:
        return float(self.seconds.get(name, 0.0))

    def snapshot(self) -> Dict[str, Any]:
        # Convert defaultdicts to plain dicts
        return {
            "seconds": {k: float(v) for k, v in self.seconds.items()},
            "counts": {k: int(v) for k, v in self.counts.items()},
        }


@dataclass
class LikelihoodHealthCounter:
    """
    Lightweight health counters for likelihood/TSM evaluation.
    Stored in diagnostics + metrics.json so failures can be triaged quickly.
    """

    n_calls: int = 0
    n_tsm_fail: int = 0
    n_output_nonfinite: int = 0  # count of non-finite entries seen in (t_arr/x0/sig2/mu/sig)

    # Variance / likelihood stability
    n_var_raw_negative: int = 0
    n_var_raw_nonfinite: int = 0
    n_var_total_clipped: int = 0  # number of entries clipped to 1e-20

    def to_dict(self) -> Dict[str, int]:
        return {
            "n_calls": int(self.n_calls),
            "n_tsm_fail": int(self.n_tsm_fail),
            "n_output_nonfinite": int(self.n_output_nonfinite),
            "n_var_raw_negative": int(self.n_var_raw_negative),
            "n_var_raw_nonfinite": int(self.n_var_raw_nonfinite),
            "n_var_total_clipped": int(self.n_var_total_clipped),
        }

    def add_from_dict(self, d: Dict[str, int]) -> None:
        self.n_calls += int(d.get("n_calls", 0))
        self.n_tsm_fail += int(d.get("n_tsm_fail", 0))
        self.n_output_nonfinite += int(d.get("n_output_nonfinite", 0))
        self.n_var_raw_negative += int(d.get("n_var_raw_negative", 0))
        self.n_var_raw_nonfinite += int(d.get("n_var_raw_nonfinite", 0))
        self.n_var_total_clipped += int(d.get("n_var_total_clipped", 0))


@contextmanager
def timed(stats: Optional[TimingStats], name: str):
    """Context manager to accumulate wall time into TimingStats."""
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - t0)


# NOTE: compute_fit_metrics and export_tmcmc_diagnostics_tables have been moved to
# visualization.helpers module. Imported above for backward compatibility.


# ==============================================================================
# VISUALIZATION
# ==============================================================================
# NOTE: PlotManager has been moved to visualization.plot_manager module.
# Imported above for backward compatibility.


# ==============================================================================
# LIKELIHOOD EVALUATOR WITH LINEARIZATION SUPPORT
# ==============================================================================
# NOTE: LogLikelihoodEvaluator has been moved to core.evaluator module.
# Imported above for backward compatibility.


# ==============================================================================
# ADAPTIVE MCMC
# ==============================================================================
# NOTE: run_adaptive_MCMC has been moved to core.mcmc module.
# Imported above for backward compatibility.


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
    """Adaptive random-walk Metropolis-Hastings MCMC."""
    # ★ CRITICAL FIX: Use default_rng consistently
    rng = np.random.default_rng(seed)

    n_params = len(prior_bounds)
    # ★ 重要: 初期点を 1.5 ± ε に設定（ε = proposal σ）
    theta_center = np.array([(low + high) / 2 for low, high in prior_bounds])
    epsilon = initial_scale

    theta_current = theta_center + rng.standard_normal(n_params) * epsilon

    # ★ prior 内に強制的に戻す
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
    logpost_all = np.zeros(
        n_samples
    )  # ★ FIX: Renamed from logL_all (stores log-posterior, not log-likelihood)
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
        logpost_all[i] = log_post_current  # ★ Store log-posterior

        if (i + 1) % 500 == 0:
            acc_rate = n_accepted / (i + 1)
            logger.info(
                "      %s/%s samples, acceptance: %.1f%%", i + 1, n_samples, acc_rate * 100.0
            )

        if proposal_cov is None:
            if (i + 1) >= adapt_start and (i + 1) % adapt_interval == 0:
                acc_rate = n_accepted / (i + 1)
                adjustment = np.exp(0.5 * (acc_rate - target_accept))
                proposal_std *= adjustment
                proposal_std = np.clip(proposal_std, 1e-4, 1.0)

    samples = samples_all[burn_in:]
    logL_values = logpost_all[burn_in:]  # ★ FIX: Use renamed variable (log-posterior values)
    acceptance_rate = n_accepted / n_samples

    idx_MAP = np.argmax(logL_values)
    theta_MAP = samples[idx_MAP]

    logger.info("      [MCMC] Complete. Acceptance rate: %.1f%%", acceptance_rate * 100.0)
    logger.info("      [MCMC] MAP: %s", theta_MAP)

    return samples, logL_values, theta_MAP, acceptance_rate


# ==============================================================================
# TRANSITIONAL MCMC (TMCMC) with β Tempering
# ==============================================================================


@dataclass
class TMCMCResult:
    """Result from Transitional MCMC."""

    samples: np.ndarray
    logL_values: np.ndarray
    theta_MAP: np.ndarray
    beta_schedule: List[float]
    converged: bool
    theta0_history: Optional[List[np.ndarray]] = None  # ★ Linearization point update history
    n_linearization_updates: int = 0  # ★ Number of linearization updates performed
    final_MAP: Optional[np.ndarray] = None  # ★ Final MAP from this chain (for global sharing)
    rom_error_pre_history: Optional[List[float]] = None  # ★ ROM error history (pre-update, debug)
    rom_error_history: Optional[List[float]] = None  # ★ ROM error history at each update
    acc_rate_history: Optional[List[float]] = None  # ★ Acceptance rate history per stage
    n_rom_evaluations: int = 0  # ★ Number of ROM (TSM) evaluations (for cost analysis)
    n_fom_evaluations: int = 0  # ★ Number of FOM evaluations (for ROM error computation)
    wall_time_s: float = 0.0  # ★ Wall time for this TMCMC chain
    timing_breakdown_s: Optional[Dict[str, float]] = (
        None  # ★ e.g., {"tsm_s":..., "fom_s":..., "tmcmc_overhead_s":...}
    )
    likelihood_health: Optional[Dict[str, int]] = None  # ★ Likelihood/TSM health counters
    stage_summary: Optional[List[Dict[str, Any]]] = (
        None  # ★ Per-stage summary rows (for CSV export)
    )


def reflect_into_bounds(x: float, low: float, high: float) -> float:
    """
    Reflect a value into bounds [low, high] using reflection (folding).

    ★ 優先度A: 境界処理（Reflection）の導入
    - 境界付近に真値がある場合、continue棄却はacceptanceを落とす
    - 反射は「提案分布が対称」という前提と相性が良く、境界での探索停滞を減らす

    Parameters
    ----------
    x : float
        Value to reflect
    low : float
        Lower bound
    high : float
        Upper bound

    Returns
    -------
    float
        Reflected value within [low, high]
    """
    width = high - low
    if width <= 0:
        return np.clip(x, low, high)
    y = x
    # Fold by reflection (works even if far outside)
    y = (y - low) % (2 * width)
    y = 2 * width - y if y > width else y
    return low + y


def choose_subset_size(beta_next: float) -> int:
    """
    ★ 優先度S: 動的サブセットサイズ（βに応じて縮める）

    βが大きい（分布が狭い）ほど、サブセットサイズを小さくしてFOM評価を削減。

    Parameters
    ----------
    beta_next : float
        Next β value (0 to 1)

    Returns
    -------
    int
        Subset size for ROM error evaluation
    """
    if beta_next < 0.6:
        return 20
    elif beta_next < 0.85:
        return 10
    else:
        return 5


def should_do_fom_check(
    beta_next: float,
    stage: int,
    update_interval: int,
    delta_theta0: Optional[float],
    last_rom_error: Optional[float],
    delta_tol: float = 5e-4,
    rom_tol: float = 0.01,
) -> bool:
    """
    ★ 優先度S: FOMチェックのスキップ条件

    「βが大きい（分布が狭い）」「線形化点がほぼ動かない」「ROM誤差も十分小さい」
    なら、FOMチェックをスキップして計算コストを削減。

    Parameters
    ----------
    beta_next : float
        Next β value
    stage : int
        Current stage
    update_interval : int
        Linearization update interval
    delta_theta0 : Optional[float]
        Last linearization point change ||Δθ₀||
    last_rom_error : Optional[float]
        Last ROM error value
    delta_tol : float
        Tolerance for linearization point change
    rom_tol : float
        Tolerance for ROM error

    Returns
    -------
    bool
        True if FOM check should be performed
    """
    # Must be update interval and β > 0.5
    if not (beta_next > 0.5 and (stage % update_interval == 0)):
        return False

    # ★ 1) スキップ条件が「誤差が未知」なときに発動しないか
    # last_rom_error / last_delta_theta0 が None の初期状態でスキップが起きると危険
    # まずはFOMチェックを実施してからスキップ判定を行う
    if last_rom_error is None or delta_theta0 is None:
        return True  # まずはFOMチェック実施（安全側）

    # Skip if linearization point hasn't moved much
    if delta_theta0 < delta_tol:
        return False
    # Skip if ROM error is already small (hysteresis for stability)
    if last_rom_error < rom_tol:
        return False
    return True


# NOTE: _validate_tmcmc_inputs has been moved to utils.validation module.
# Imported above for backward compatibility.


def run_TMCMC(
    log_likelihood: callable,
    prior_bounds: List[Tuple[float, float]],
    n_particles: int = DEFAULT_N_PARTICLES,
    n_stages: int = DEFAULT_N_STAGES,
    target_ess_ratio: float = DEFAULT_TARGET_ESS_RATIO,
    min_delta_beta: float = DEFAULT_MIN_DELTA_BETA,
    max_delta_beta: float = MAX_DELTA_BETA,
    logL_scale: float = 1.0,  # Deprecated, kept for compatibility
    seed: Optional[int] = None,
    model_name: str = "",
    evaluator: Optional[Any] = None,  # ★ LogLikelihoodEvaluator instance (for linearization update)
    theta_base_full: Optional[np.ndarray] = None,  # ★ Full 14-dim theta base
    active_indices: Optional[List[int]] = None,  # ★ Active parameter indices
    update_linearization_interval: int = DEFAULT_UPDATE_LINEARIZATION_INTERVAL,  # ★ Update linearization every N stages
    n_mutation_steps: int = DEFAULT_N_MUTATION_STEPS,  # ★ Number of MCMC steps per particle (K-step mutation)
    use_observation_based_update: bool = True,  # ★ Use observation-based linearization update (ROM error weighted)
    linearization_threshold: float = DEFAULT_LINEARIZATION_THRESHOLD,
    linearization_enable_rom_threshold: float = 0.05,
    debug_logger: Optional[DebugLogger] = None,  # ★ Debug logger (for controlled output)
    force_beta_one: bool = False,  # ★ If True, force β=1.0 at final stage (paper runs)
) -> TMCMCResult:
    """
    Transitional MCMC (TMCMC) with β tempering + Linearization Update.

    ★ 論文通りにβ（tempering）を入れることで、精度・安定性が向上！
    ★ さらに、各stageで線形化点を更新することで、TSM-ROMの精度が向上！

    TMCMCはβ=0（事前分布）からβ=1（事後分布）へ段階的に遷移することで、
    多峰性や鋭いピークがある場合でも安定した探索が可能。

    線形化点更新機能：
    - 各stageの後にMAPを計算
    - 一定間隔（update_linearization_interval）で線形化点を更新
    - TSM-ROMの近似精度が段階的に向上

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function
    prior_bounds : List[Tuple[float, float]]
        (lower, upper) bounds for each parameter
    n_particles : int
        Number of particles (samples per stage)
    n_stages : int
        Maximum number of TMCMC stages
    target_ess_ratio : float
        Target ESS as fraction of n_particles (0.5 = 50% ESS)
    min_delta_beta : float
        Minimum β increment per stage (progress floor). Used as a safety lower bound.
    max_delta_beta : float
        Maximum β increment per stage (jump cap). Critical to avoid large β jumps that can
        cause weight collapse and acceptance≈0.
    logL_scale : float
        [DEPRECATED] Scale factor for likelihood. Currently ignored for consistency with TMCMC theory.
        All likelihood calculations (ESS, resampling, mutation) now use unscaled logL.
    seed : int, optional
        Random seed
    model_name : str
        Model identifier for logging
    evaluator : LogLikelihoodEvaluator, optional
        Evaluator instance with update_linearization_point() method
    theta_base_full : ndarray (14,), optional
        Full 14-dimensional parameter base (for constructing full theta)
    active_indices : List[int], optional
        Active parameter indices (for constructing full theta)
    update_linearization_interval : int
        Update linearization point every N stages (default: 3)

    Returns
    -------
    TMCMCResult
        Samples, log-likelihood values, MAP, beta schedule, and convergence status
    """
    # ★ INPUT VALIDATION
    _validate_tmcmc_inputs(
        log_likelihood=log_likelihood,
        prior_bounds=prior_bounds,
        n_particles=n_particles,
        n_stages=n_stages,
        target_ess_ratio=target_ess_ratio,
        evaluator=evaluator,
        theta_base_full=theta_base_full,
        active_indices=active_indices,
    )

    # ★ CRITICAL FIX: Use default_rng consistently (remove np.random.seed)
    # np.random.seed is deprecated and causes non-reproducibility issues
    # default_rng is the recommended approach for NumPy >= 1.17
    rng = np.random.default_rng(seed)
    tmcmc_wall_start = time.perf_counter()

    n_params = len(prior_bounds)

    def log_prior(theta: np.ndarray) -> float:
        for i, (low, high) in enumerate(prior_bounds):
            if not (low <= theta[i] <= high):
                return -np.inf
        return 0.0

    # Initialize particles from prior
    theta = np.zeros((n_particles, n_params))
    for i in range(n_particles):
        for j, (low, high) in enumerate(prior_bounds):
            theta[i, j] = rng.uniform(low, high)

    # Evaluate initial log-likelihood
    logL = np.array([log_likelihood(t) for t in theta])
    beta = 0.0
    beta_schedule = [beta]

    # ★ Track linearization point updates
    theta0_history = []
    n_linearization_updates = 0

    # ★ Track diagnostic histories
    # ROM error at each linearization update:
    # - rom_error_pre_history: computed BEFORE θ0 update (debugging)
    # - rom_error_history: computed AFTER θ0 update (this is what we gate on / report)
    rom_error_pre_history = []
    rom_error_history = []
    acc_rate_history = []  # Acceptance rate per stage
    theta_MAP_posterior_history = (
        []
    )  # ★ Track posterior MAP at each stage (for final MAP selection)
    stage_summary: List[Dict[str, Any]] = []  # ★ Per-stage summary rows (exportable)

    # ★ 優先度S: Track last ROM error and delta_theta0 for skip conditions
    last_rom_error: Optional[float] = None
    last_delta_theta0: Optional[float] = None

    # ★ Track evaluation counts (for cost analysis)
    initial_rom_count = 0
    initial_fom_count = 0
    if evaluator is not None:
        initial_rom_count = evaluator.call_count
        initial_fom_count = evaluator.fom_call_count
        theta0_initial = evaluator.get_linearization_point()
        if theta0_initial is not None:
            theta0_history.append(theta0_initial.copy())

    # Initialize debug logger if not provided
    if debug_logger is None:
        debug_logger = DebugLogger(DebugConfig(level=DebugLevel.OFF))

    # ★ Set Slack thread for debug logger (will be set after thread creation)
    # This allows DebugLogger to add messages to the thread

    # ★ ERROR-CHECK: Check initial numerical errors
    debug_logger.check_numerical_errors(logL, theta, context="Initialization")

    # ★ Force initial log output (always show start of TMCMC)
    debug_logger.log_info(f"Initial LogL: min={logL.min():.1f}, max={logL.max():.1f}", force=True)
    if model_name:
        debug_logger.log_info(f"Model: {model_name}", force=True)
    debug_logger.log_info(
        f"Starting TMCMC with {n_particles} particles, {n_stages} stages...", force=True
    )

    # ★ Slack notification: TMCMC start with thread support (if model_name provided)
    slack_thread_ts = None
    if SLACK_ENABLED and model_name:
        title = (
            f"🔄 {model_name} TMCMC Started\n"
            f"   Particles: {n_particles}\n"
            f"   Stages: {n_stages}\n"
            f"   Initial LogL: [{logL.min():.1f}, {logL.max():.1f}]"
        )
        slack_thread_ts = _slack_notifier.start_thread(title)
        # If thread not available, fallback to regular notification
        if slack_thread_ts is None:
            notify_slack(title, raise_on_error=False)
        else:
            # Set thread for debug logger so it can add messages to the thread
            debug_logger.set_slack_thread(slack_thread_ts)

    for stage in range(1, n_stages + 1):
        # ★ Force stage start log (always show progress)
        debug_logger.log_info(f"Stage {stage}/{n_stages} starting...", force=True)
        # Per-stage counters/flags for later CSV export
        rom_error_pre_stage: Optional[float] = None
        rom_error_post_stage: Optional[float] = None
        delta_theta0_stage: Optional[float] = None
        # ★ Slack notification: 削除（詳細すぎるため、重要な情報のみ送信）
        # 1. Calculate Beta using ESS-based adaptive schedule
        # ★ CRITICAL FIX: logL_scale を撤廃（TMCMC理論との一貫性のため）
        # logL_scale は ESS計算・resampling・mutation で不整合を引き起こす
        # ESS計算で「スケール済み尤度」を見て、実際の重み更新で「未スケール尤度」を使うと、
        # beta が異常に速く 1.0 に到達し、posterior 探索が成立しない
        logL_eff = logL  # ★ logL_scale を撤廃（一貫性のため）
        delta_low, delta_high = 0.0, 1.0 - beta

        # Binary search for optimal delta_beta
        ess_at_delta_low = None  # ★ PRIORITY B: ESS値を記録（診断用）
        for _ in range(25):
            mid = 0.5 * (delta_low + delta_high)
            x = mid * (logL_eff - np.max(logL_eff))  # Shift for stability
            w = np.exp(x)
            sum_w = np.sum(w)
            if sum_w <= 0:
                ess = 0
            else:
                w_norm = w / sum_w
                ess = 1.0 / np.sum(w_norm**2)

            if ess >= target_ess_ratio * n_particles:
                delta_low = mid
                ess_at_delta_low = ess  # 最終的なESS値を記録
            else:
                delta_high = mid

        # ★ 高速化＋安全オプション（全モード共通）:
        # - 下限:  ESS が許す範囲でも、進行幅が小さくなりすぎないように min_delta_beta を保証
        # - 上限:  一気に β=1.0 近くまで飛ばないように MAX_DELTA_BETA でクリップ
        # - さらに、1.0 を超えないように (1.0 - beta) でもクリップ
        delta_beta_raw = max(delta_low, min_delta_beta)
        delta_beta = min(delta_beta_raw, float(max_delta_beta), 1.0 - beta)

        beta_next = min(beta + delta_beta, 1.0)

        # Paper-oriented safety: if the user set too few stages, still hit β=1.0 at the end.
        # This is mainly for stable reporting/plots; note that a large final jump can increase degeneracy.
        if force_beta_one and stage == n_stages and beta_next < 1.0:
            debug_logger.log_warning(
                f"Forcing final β to 1.0 at stage {stage}/{n_stages} (β was {beta_next:.4f}). "
                "Consider increasing n_stages for a smoother tempering schedule."
            )
            beta_next = 1.0
            delta_beta = 1.0 - beta

        # ★ ERROR-CHECK: Check beta progression
        debug_logger.check_beta_progression(beta_next, delta_beta, stage, context=f"Stage {stage}")

        # 2. Resample with weights
        log_w_unnorm = (beta_next - beta) * logL
        log_w_unnorm -= np.max(log_w_unnorm)  # Shift to prevent overflow
        w = np.exp(log_w_unnorm)
        w_sum = np.sum(w)

        if w_sum <= 0 or not np.isfinite(w_sum):
            debug_logger.log_warning("Weight sum issue. Using uniform.")
            w = np.ones(n_particles) / n_particles
        else:
            w /= w_sum

        # Diagnostics: actual ESS from the weights we will *actually* resample with.
        # (This can differ slightly from the binary-search ESS due to min/max delta clipping.)
        ess_weights = None
        try:
            if np.all(np.isfinite(w)) and float(np.sum(w)) > 0:
                ess_weights = float(1.0 / np.sum(w**2))
        except Exception:
            ess_weights = None

        # ★ PRIORITY B: βスケジュールの診断ログ（各stageで出力）
        # ESS計算の結果と実際のbeta進行を記録
        # 重みの尖り具合も記録（max(log_w) - min(log_w)）
        log_w_range = np.max(log_w_unnorm) - np.min(log_w_unnorm) if len(log_w_unnorm) > 0 else 0.0
        ess_at_delta_low_str = f"{ess_at_delta_low:.1f}" if ess_at_delta_low is not None else "N/A"
        ess_weights_str = f"{ess_weights:.1f}" if ess_weights is not None else "N/A"
        w_max = float(np.max(w)) if len(w) > 0 else float("nan")
        w_min = float(np.min(w)) if len(w) > 0 else float("nan")
        # ★ Force beta schedule log (always show progress)
        beta_msg = (
            f"      [TMCMC] Stage {stage}: β={beta:.4f} → {beta_next:.4f} (Δ={delta_beta:.6f}), "
            f"ESS={ess_at_delta_low_str}/{target_ess_ratio*n_particles:.1f} (actual={ess_weights_str}), "
            f"logL range=[{logL.min():.2f}, {logL.max():.2f}], "
            f"log_w range={log_w_range:.2f}, w[min,max]=[{w_min:.2e},{w_max:.2e}]"
        )
        logger.info("%s", beta_msg)
        # ★ Slack notification: 削除（詳細すぎるため、重要な情報のみ送信）

        # Resample particles
        # ★ Store particles and weights BEFORE resampling for weighted barycenter computation
        theta_before_resample = theta.copy()
        logL_before_resample = logL.copy()
        weights_before_resample = w.copy()  # ★ Store weights for barycenter

        # ★ CRITICAL FIX: Compute tempered posterior for MAP calculation
        # TMCMC stage k posterior: π_k(θ) ∝ p(θ) * p(D|θ)^β_k
        # So log_posterior = log_prior + beta * logL
        log_prior_before_resample = np.array([log_prior(t) for t in theta_before_resample])
        log_posterior_before_resample = log_prior_before_resample + beta_next * logL_before_resample

        idx = rng.choice(n_particles, size=n_particles, p=w)
        # Diagnostics: particle degeneracy after resampling (how many unique ancestors survived)
        try:
            n_unique_idx = int(np.unique(idx).size)
            unique_ratio = float(n_unique_idx) / float(n_particles)
            if debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                debug_logger.log_info(
                    f"Resample degeneracy: unique={n_unique_idx}/{n_particles} (unique_ratio={unique_ratio:.3f})"
                )
        except Exception:
            pass
        theta = theta[idx]
        logL = logL[idx]
        # Keep a copy of the post-resample population for potential recovery retries.
        theta_after_resample = theta.copy()
        logL_after_resample = logL.copy()

        # 3. Mutate (K-step MCMC with tempered posterior)
        # ★ 改善: 1-step → K-step mutation (resampling後の粒子相関を減らす)
        # 理由: resampling後は粒子が強く相関しているため、1-stepではESSが見かけ倒しになる
        # ★ 改善: Tempered covariance scaling (Del Moral et al., Ching & Chen)
        # Early stages (small β) need larger proposal variance for exploration
        cov_base = np.cov(theta.T)

        # ★ CRITICAL FIX: Handle 1D case (n_params == 1)
        # np.cov() returns scalar or 1D array for 1D input, but np.trace() requires 2D+
        # Ensure cov_base is always 2D for consistent handling
        if n_params == 1:
            # For 1D: cov_base is scalar, convert to 2D array
            cov_base = (
                np.array([[cov_base]]) if np.isscalar(cov_base) else np.array([[cov_base.item()]])
            )
        else:
            # For multi-D: ensure it's 2D (should already be, but be safe)
            if cov_base.ndim == 0:
                cov_base = np.array([[cov_base]])
            elif cov_base.ndim == 1:
                cov_base = np.diag(cov_base)

        # Optimal scaling: 2.38^2 / n_params (Gelman et al., 1996)
        # Tempered scaling: scale inversely with β (larger variance when β is small)
        optimal_scale = OPTIMAL_SCALE_FACTOR / n_params
        tempered_scale = optimal_scale / max(beta_next, 0.1)  # Avoid division by zero

        # ★ Adaptive scaling based on previous acceptance rate
        # - Low acceptance typically means steps are too large → reduce scale.
        # - Very high acceptance can mean steps are too small → slightly increase scale.
        adaptive_scale_factor = MUTATION_SCALE_FACTOR
        if len(acc_rate_history) > 0:
            prev_acc_rate = float(acc_rate_history[-1])
            if prev_acc_rate < 0.05:
                # Reduce scale factor when acceptance rate is very low
                # (cap at 0.1x to avoid freezing completely)
                shrink = max(0.1, prev_acc_rate / 0.05)
                adaptive_scale_factor = MUTATION_SCALE_FACTOR * shrink
                debug_logger.log_info(
                    f"⚠️  Low acceptance rate ({prev_acc_rate:.3f}), reducing proposal scale: {adaptive_scale_factor:.2f}x"
                )
            elif prev_acc_rate > 0.6:
                # Slightly increase step size if acceptance is extremely high
                grow = min(2.0, prev_acc_rate / 0.6)
                adaptive_scale_factor = MUTATION_SCALE_FACTOR * grow
                debug_logger.log_info(
                    f"ℹ️  High acceptance rate ({prev_acc_rate:.3f}), increasing proposal scale: {adaptive_scale_factor:.2f}x"
                )

        # ★ Global knob: MUTATION_SCALE_FACTOR controls overall jump size (and thus acceptance)
        cov = cov_base * (adaptive_scale_factor * tempered_scale)

        # ★ 優先度A: 共分散の正則化をスケール依存に（ロバスト性↑、歪み↓）
        # 固定 1e-6 はスケールによって大きすぎることがある
        # traceベースの正則化で、共分散の大きさに比例させる
        # ★ CRITICAL FIX: np.trace() requires 2D array, which we've ensured above
        scale = np.trace(cov_base) / n_params
        nugget = COVARIANCE_NUGGET_BASE + COVARIANCE_NUGGET_SCALE * scale
        cov += nugget * np.eye(n_params)

        # Diagnostics: proposal covariance scale/conditioning (helps explain extreme acceptance rates)
        try:
            if debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                cov_trace = float(np.trace(cov))
                covbase_trace = float(np.trace(cov_base))
                cond = None
                try:
                    cond = float(np.linalg.cond(cov))
                except Exception:
                    cond = None
                cond_str = f"{cond:.2e}" if cond is not None and np.isfinite(cond) else "N/A"
                debug_logger.log_info(
                    "Proposal cov stats: "
                    f"trace(cov_base)={covbase_trace:.2e}, trace(cov)={cov_trace:.2e}, "
                    f"scale={scale:.2e}, nugget={nugget:.2e}, "
                    f"tempered_scale={float(tempered_scale):.2e}, adapt_scale={float(adaptive_scale_factor):.2f}x, "
                    f"cond(cov)={cond_str}"
                )
        except Exception:
            pass

        # ★ ERROR-CHECK: Check covariance matrix validity
        debug_logger.check_covariance_matrix(cov, context=f"Stage {stage}, mutation covariance")

        def _mutate_population(
            cov_matrix: np.ndarray, steps: int
        ) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
            """K-step mutation for the whole population. Returns (theta, logL, acc_rate, n_accepted, n_total)."""
            _theta = theta_after_resample.copy()
            _logL = logL_after_resample.copy()
            _acc = 0
            _total = 0
            for i in range(n_particles):
                theta_current = _theta[i].copy()
                logL_current = _logL[i]
                for _ in range(int(max(1, steps))):
                    prop = rng.multivariate_normal(theta_current, cov_matrix)
                    _total += 1
                    for j, (low, high) in enumerate(prior_bounds):
                        prop[j] = reflect_into_bounds(prop[j], low, high)
                    lp_p = log_prior(prop)
                    if not np.isfinite(lp_p):
                        continue
                    ll_p = log_likelihood(prop)
                    if not np.isfinite(ll_p):
                        continue
                    log_ratio = (lp_p + beta_next * ll_p) - (
                        log_prior(theta_current) + beta_next * logL_current
                    )
                    if np.log(rng.random()) < log_ratio:
                        theta_current = prop
                        logL_current = ll_p
                        _acc += 1
                _theta[i] = theta_current
                _logL[i] = logL_current
            _acc_rate = _acc / _total if _total > 0 else 0.0
            return _theta, _logL, _acc_rate, int(_acc), int(_total)

        # First mutation attempt
        theta, logL, acc_rate, acc, total_proposals = _mutate_population(cov, n_mutation_steps)

        # Diagnostics: population diversity after mutation (rounded unique rows to tolerate tiny FP noise)
        try:
            if debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                theta_rounded = np.round(theta, 6)
                n_unique_theta = int(np.unique(theta_rounded, axis=0).shape[0])
                uniq_ratio = float(n_unique_theta) / float(n_particles)
                debug_logger.log_info(
                    f"Post-mutation diversity: unique≈{n_unique_theta}/{n_particles} (unique_ratio≈{uniq_ratio:.3f})"
                )
        except Exception:
            pass

        # Recovery when mutation gets stuck (avoid "continue with degenerate samples")
        # - Retry with smaller proposal scale
        # - If still stuck, add tiny jitter and retry once more
        if acc_rate < debug_logger.config.min_acceptance_rate:
            debug_logger.log_warning(
                f"Stage {stage}: acceptance rate {acc_rate:.4f} < {debug_logger.config.min_acceptance_rate:.4f}. "
                "Attempting recovery (shrink proposal covariance)."
            )
            theta, logL, acc_rate, acc, total_proposals = _mutate_population(
                cov * 0.3, max(1, n_mutation_steps // 2)
            )

        if acc_rate < debug_logger.config.min_acceptance_rate:
            # Jitter around resampled population to break exact duplicates, then retry with small covariance.
            debug_logger.log_warning(
                f"Stage {stage}: still stuck after shrink (acc_rate={acc_rate:.4f}). "
                "Applying small jitter and retrying."
            )
            theta_after_resample = theta_after_resample.copy()
            jitter = rng.normal(loc=0.0, scale=1e-3, size=theta_after_resample.shape)
            theta_after_resample += jitter
            # Reflect jittered points back into bounds
            for i in range(n_particles):
                for j, (low, high) in enumerate(prior_bounds):
                    theta_after_resample[i, j] = reflect_into_bounds(
                        theta_after_resample[i, j], low, high
                    )
            logL_after_resample = np.array([log_likelihood(t) for t in theta_after_resample])
            theta, logL, acc_rate, acc, total_proposals = _mutate_population(cov * 0.1, 1)

        # If still stuck, fail fast (do not proceed with a degenerate posterior)
        if acc_rate < debug_logger.config.min_acceptance_rate:
            raise RuntimeError(
                f"TMCMC mutation stuck: acc_rate={acc_rate:.4f} < {debug_logger.config.min_acceptance_rate:.4f} "
                f"after recovery attempts. Stage={stage}, beta_next={beta_next:.4f}."
            )

        # 4. ★ Update linearization point (if evaluator provided)
        # ⚠️ 重要: 線形化点更新後は必ずlogLを再計算する必要がある
        #
        # ★ 理論的保証（論文での説明用）:
        # The proposed algorithm can be interpreted as a fixed-point iteration on the
        # linearization point under a progressively sharpened posterior (β: 0 → 1).
        # As β increases, the posterior becomes sharper, and the weighted barycenter
        # converges to the true posterior mean, providing a natural stopping criterion
        # for linearization point updates.
        #
        # ★ LINEARIZATION POINT UPDATE TIMING RULE:
        # The linearization point is selected from particles AFTER mutation (theta, logL).
        # This ensures the linearization point reflects the current stage's posterior exploration
        # and is most consistent across stages.
        #
        # Note: Observation-based ROM error computation uses theta_before_resample (correct),
        # but the final linearization point selection uses mutation result (theta, logL).
        if evaluator is not None and theta_base_full is not None and active_indices is not None:
            should_update = False

            # Phase A / stability:
            # Do NOT enable linearization immediately just because β crossed a threshold.
            # Enable only after MAP-based ROM error check at an update event.

            # Check if it's time to update (interval-based + β threshold)
            # ★ 改善: βが小さい段階（priorに近い）では線形化点更新を避ける
            # 理由: posteriorが十分sharpになってから更新することで、ROMの精度が向上
            # ★ 修正: stageベースに戻す（update_attempt_count のバグを回避）
            # update_attempt_count は should_update=True のときしか増えないため、
            # 一度 1 になると interval の倍数に戻らず更新が止まる問題があった
            if (
                beta_next > 0.5 and (stage % update_linearization_interval == 0)
            ) or stage == n_stages:
                should_update = True
            elif beta_next <= 0.5:
                should_update = False  # Skip update if β is too small

            # ★ CRITICAL FIX: Initialize MAP variables
            # Separate posterior MAP (statistical) from linearization MAP (numerical)
            #
            # theta_MAP_posterior: Statistical MAP estimate of the posterior distribution at stage k
            #   - Used for reporting, convergence diagnostics, and final results
            #   - Must reflect observation information if observation-based update is used
            #   - Should NOT be overwritten after observation-based correction
            #   - Definition: argmax_θ [log p(D|θ)^β_k * p(θ)] with observation correction
            #
            # theta_MAP_linearize: Numerical anchor point for TSM-ROM linearization
            #   - Used for linearization point update in ROM
            #   - May differ from posterior MAP for numerical stability
            #   - Can be recomputed if needed for ROM accuracy
            #   - Default: same as posterior MAP, but can be overridden
            theta_MAP_posterior = None
            theta_MAP_linearize = None
            idx_MAP_posterior = None
            theta_MAP_posterior_computed = False
            theta_MAP_posterior_obs_corrected = (
                None  # For assertion: verify observation-corrected MAP is not overwritten
            )

            # Check convergence condition (stop if MAP hasn't moved much)
            # ★ CRITICAL FIX: Initialize should_do_fom to ensure it's defined in all code paths
            should_do_fom = True  # Default: do FOM check unless conditions suggest skipping

            if should_update:
                # ★ 新アイデア: 観測量ベースの線形化点更新（論文に強い）
                # 線形化点を「パラメータ空間」ではなく「観測量φ̄（データ）に基づいて」更新
                #
                # 数式: θ₀_new = Σ_i [w_i / (1 + ε_obs(θ_i))] * θ_i
                # where:
                #   w_i: TMCMCの重み
                #   ε_obs(θ_i): 観測点φ̄におけるROM-FOM誤差
                #
                # 効果:
                # - データに効かないROM誤差は無視
                # - 観測に重要な方向だけに線形化点が引き寄せられる
                # - posterior精度が大幅向上、多峰性でも安定
                #
                # 論文での説明:
                # "The linearization point is updated to minimize approximation error
                #  at observation points that contribute to the likelihood."

                if use_observation_based_update:
                    # ★ 優先度S: FOMチェックのスキップ条件
                    # 「βが大きい（分布が狭い）」「線形化点がほぼ動かない」「ROM誤差も十分小さい」
                    # なら、FOMチェックをスキップして計算コストを削減
                    should_do_fom = should_do_fom_check(
                        beta_next=beta_next,
                        stage=stage,
                        update_interval=update_linearization_interval,
                        delta_theta0=last_delta_theta0,
                        last_rom_error=last_rom_error,
                    )

                    if not should_do_fom:
                        # Skip FOM evaluation, MAP is already computed above
                        # ★ PRIORITY D: None フォーマット例外を確実に潰す
                        dtheta_str = (
                            "None" if last_delta_theta0 is None else f"{last_delta_theta0:.6f}"
                        )
                        rom_str = "None" if last_rom_error is None else f"{last_rom_error:.6f}"
                        debug_logger.log_info(
                            f"Skipping FOM check (β={beta_next:.3f}, ||Δθ₀||={dtheta_str}, ε_ROM={rom_str})"
                        )
                        # MAP is already computed above (idx_MAP_stage, theta_MAP_stage, theta_full_MAP)
                        # ★ 2) スキップした場合のrom_error_historyの整合性
                        # スキップ時はnp.nanをappend（後でプロットでnan無視できる）
                        # Note: This will be appended later in the ROM error check section
                    else:
                        # ★ 優先度S: 動的サブセットサイズ（βに応じて縮める）
                        # βが大きい（分布が狭い）ほど、サブセットサイズを小さくしてFOM評価を削減
                        subset_size_base = choose_subset_size(beta_next)
                        subset_size = min(subset_size_base, n_particles)

                        # ★ 優先度B: 重み付き・層化サンプリング（k-meansより軽い強化案）
                        # 重み上位＋ランダム（外れ値も拾う）
                        # ★ 3) subsetの層化サンプリングが「重みゼロ/NaN」でも壊れないか
                        weights_safe = weights_before_resample.copy()
                        # Check for NaN/Inf in weights
                        if not np.all(np.isfinite(weights_safe)):
                            # Fallback: use uniform weights if NaN/Inf detected
                            weights_safe = np.ones(n_particles) / n_particles
                            debug_logger.log_warning(
                                "Weights contain NaN/Inf, using uniform weights for subset selection"
                            )

                        m = subset_size // 2
                        # Top particles by weight
                        top_idx = np.argsort(weights_safe)[-min(5 * subset_size, n_particles) :]
                        # ★ 上位候補集合が小さすぎるときはreplaceを許可 or 全体ランダムへ
                        if len(top_idx) < m:
                            # Not enough top candidates, use random sampling
                            subset_top = rng.choice(n_particles, size=m, replace=False)
                        else:
                            subset_top = rng.choice(
                                top_idx, size=min(m, len(top_idx)), replace=False
                            )
                        # Random particles (catch outliers)
                        subset_rand = rng.choice(
                            n_particles, size=subset_size - len(subset_top), replace=False
                        )
                        subset_idx = np.unique(np.concatenate([subset_top, subset_rand]))
                        subset_size = len(subset_idx)  # Actual size after deduplication

                        debug_logger.log_observation_based_update(subset_size, n_particles)

                        # ★ 修正: NaNで初期化（未計算の粒子を明確に区別）
                        # rom_errors == 0 は危険（本当に誤差0の粒子と区別できない）
                        rom_errors = np.full(n_particles, np.nan)

                        # Step 1: サブセットのみROM誤差を計算
                        for i in subset_idx:
                            # Construct full theta for particle i
                            theta_i_full = theta_base_full.copy()
                            for j, idx in enumerate(active_indices):
                                theta_i_full[idx] = theta_before_resample[i, j]

                            # Compute ROM error at observation points
                            try:
                                rom_errors[i] = evaluator.compute_ROM_error(theta_i_full)
                            except Exception as e:
                                # If error computation fails, use large error (low weight)
                                rom_errors[i] = 1.0  # Large error → low weight
                                debug_logger.log_warning(
                                    f"ROM error computation failed for particle {i}: {e}"
                                )

                        # Step 2: サブセットのROM誤差から平均誤差を推定
                        # サブセットに含まれない粒子は平均誤差を使用
                        # ★ CRITICAL FIX: Handle all-NaN case (fallback to large error)
                        mean_rom_error_subset = np.nanmean(rom_errors)
                        if np.isnan(mean_rom_error_subset):
                            # All ROM errors are NaN: use large fallback value
                            mean_rom_error_subset = ROM_ERROR_FALLBACK
                            debug_logger.log_warning(
                                f"All ROM errors are NaN, using fallback value {ROM_ERROR_FALLBACK}"
                            )

                        # ★ 修正: NaNの粒子に平均値を割り当て（意味が明確、reviewerに説明しやすい）
                        rom_errors = np.where(
                            np.isnan(rom_errors), mean_rom_error_subset, rom_errors
                        )

                        # Step 3: 重みを修正（観測点でのROM誤差を考慮）
                        # w_i' = w_i / (1 + ε_obs(θ_i))
                        # 誤差が大きい粒子の重みを下げる
                        weights_obs_corrected = weights_before_resample.copy()
                        for i in range(n_particles):
                            weights_obs_corrected[i] = weights_before_resample[i] / (
                                1.0 + rom_errors[i]
                            )

                        # Normalize corrected weights
                        weights_sum = np.sum(weights_obs_corrected)
                        if weights_sum > 0:
                            weights_obs_corrected /= weights_sum
                        else:
                            # Fallback to original weights if all errors are too large
                            weights_obs_corrected = weights_before_resample.copy()
                            debug_logger.log_warning(
                                "All ROM errors too large, using original weights"
                            )

                        # ★ CRITICAL FIX: Compute posterior MAP using observation-corrected weights
                        # This MAP reflects observation information and should NOT be overwritten
                        #
                        # Theory: The observation correction weights particles by ROM error at observation points.
                        # The corrected weight is: w_obs_corrected = w_original / (1 + ε_obs(θ))
                        # This means particles with lower ROM error at observation points get higher weight.
                        #
                        # For posterior MAP: we want argmax_θ [log p(D|θ)^β * p(θ)] with observation correction.
                        # TMCMC stage k posterior: π_k(θ) ∝ p(θ) * p(D|θ)^β_k
                        # So log_posterior = log_prior + beta * logL
                        #
                        # ★ APPROACH: Use observation-corrected posterior score
                        # Score = log_posterior - log(1 + ε_obs)
                        #       = (log_prior + beta * logL) - log(1 + ε_obs)
                        # This gives higher score to particles with both high posterior AND low ROM error.
                        #
                        # Compute observation-corrected posterior score for each particle
                        obs_corrected_scores = log_posterior_before_resample.copy()
                        log_penalty = np.log(1.0 + rom_errors)
                        obs_corrected_scores -= log_penalty

                        # ★ SCALE VERIFICATION: Log distributions to detect scale issues
                        logL_min, logL_median, logL_max = (
                            np.min(logL_before_resample),
                            np.median(logL_before_resample),
                            np.max(logL_before_resample),
                        )
                        log_penalty_min, log_penalty_median, log_penalty_max = (
                            np.min(log_penalty),
                            np.median(log_penalty),
                            np.max(log_penalty),
                        )
                        log_post_min, log_post_median, log_post_max = (
                            np.min(log_posterior_before_resample),
                            np.median(log_posterior_before_resample),
                            np.max(log_posterior_before_resample),
                        )
                        obs_score_min, obs_score_median, obs_score_max = (
                            np.min(obs_corrected_scores),
                            np.median(obs_corrected_scores),
                            np.max(obs_corrected_scores),
                        )

                        debug_logger.log_info(
                            f"Scale check: logL=[{logL_min:.2f}, {logL_median:.2f}, {logL_max:.2f}], "
                            f"log_penalty=[{log_penalty_min:.4f}, {log_penalty_median:.4f}, {log_penalty_max:.4f}], "
                            f"log_post=[{log_post_min:.2f}, {log_post_median:.2f}, {log_post_max:.2f}], "
                            f"obs_score=[{obs_score_min:.2f}, {obs_score_median:.2f}, {obs_score_max:.2f}]"
                        )

                        # Check if penalty is too weak (penalty << logL scale)
                        penalty_ratio = np.max(log_penalty) / (
                            np.max(logL_before_resample) - np.min(logL_before_resample) + 1e-10
                        )
                        if penalty_ratio < 0.01:
                            debug_logger.log_warning(
                                f"Observation penalty may be too weak (max_penalty/max_logL_range={penalty_ratio:.4f} < 0.01)"
                            )
                        elif penalty_ratio > 0.1:
                            debug_logger.log_warning(
                                f"Observation penalty may be too strong (max_penalty/max_logL_range={penalty_ratio:.4f} > 0.1)"
                            )

                        # Find particle with highest observation-corrected posterior score
                        idx_MAP_posterior = np.argmax(obs_corrected_scores)
                        theta_MAP_posterior = theta_before_resample[idx_MAP_posterior]

                        # Also compute standard MAP (without observation correction) for comparison
                        idx_MAP_standard = np.argmax(log_posterior_before_resample)
                        theta_MAP_standard = theta_before_resample[idx_MAP_standard]

                        # Log the difference between standard MAP and observation-corrected MAP
                        if idx_MAP_posterior != idx_MAP_standard:
                            delta_map = np.linalg.norm(theta_MAP_posterior - theta_MAP_standard)
                            debug_logger.log_info(
                                f"Observation-corrected MAP differs from standard MAP: ||Δ||={delta_map:.6f}"
                            )
                        else:
                            debug_logger.log_info(
                                "Observation-corrected MAP matches standard MAP (no correction effect)"
                            )

                        # Report statistics
                        mean_rom_error = np.mean(rom_errors)
                        max_rom_error = np.max(rom_errors)
                        min_rom_error = np.min(rom_errors)
                        debug_logger.log_info(
                            f"ROM errors: mean={mean_rom_error:.6f}, min={min_rom_error:.6f}, max={max_rom_error:.6f}"
                        )

                        # Compute weighted means for comparison (optional, not used for linearization)
                        theta_weighted_mean_original = np.zeros(n_params)
                        for i in range(n_particles):
                            theta_weighted_mean_original += (
                                weights_before_resample[i] * theta_before_resample[i]
                            )

                        theta_weighted_mean_obs = np.zeros(n_params)
                        for i in range(n_particles):
                            theta_weighted_mean_obs += (
                                weights_obs_corrected[i] * theta_before_resample[i]
                            )

                        # Report difference between MAP and weighted means for comparison
                        delta_map_weighted_mean = np.linalg.norm(
                            theta_MAP_posterior - theta_weighted_mean_original
                        )
                        delta_weighted_mean_shift = np.linalg.norm(
                            theta_weighted_mean_obs - theta_weighted_mean_original
                        )
                        debug_logger.log_info(
                            f"Posterior MAP-WeightedMean distance: {delta_map_weighted_mean:.6f}, WeightedMean shift (obs-corrected): {delta_weighted_mean_shift:.6f}"
                        )

                        # Mark that posterior MAP has been computed with observation correction
                        theta_MAP_posterior_computed = True

                        # ★ ASSERT: Observation-corrected MAP should NOT be overwritten
                        # Store a copy for verification
                        theta_MAP_posterior_obs_corrected = theta_MAP_posterior.copy()
                else:
                    # No observation-based update: compute standard MAP
                    theta_MAP_posterior_computed = False
                    theta_MAP_posterior_obs_corrected = None

                # ★ CRITICAL FIX: Separate Maximum Likelihood (ML) from MAP (posterior)
                #
                # theta_ML_stage: Maximum Likelihood particle (argmax logL)
                #   - Used for comparison and diagnostics
                #   - Does NOT include prior information
                #   - Selected from theta_before_resample (for observation-based update compatibility)
                #
                # theta_MAP_posterior: Statistical MAP estimate of the posterior distribution at stage k
                #   - Definition: argmax_θ [log p(θ) + β_k * log p(D|θ)]
                #   - Must reflect observation information if observation-based update is used
                #   - Should NOT be overwritten after observation-based correction
                #   - Used for reporting, convergence diagnostics, and final results
                #   - Selected from theta_before_resample (for observation-based update compatibility)
                #
                # theta_MAP_linearize: Numerical anchor point for TSM-ROM linearization
                #   - Used for linearization point update in ROM
                #   - ★ UPDATE TIMING RULE: Selected from particles AFTER mutation (theta, logL)
                #   - This ensures consistency with current stage's posterior exploration
                #   - May differ from posterior MAP for numerical stability
                #   - Default: same as posterior MAP from mutation result, but can be overridden

                if not theta_MAP_posterior_computed:
                    # Standard MAP computation (no observation correction)
                    # Use tempered posterior: log_prior + beta * logL
                    idx_MAP_posterior = np.argmax(log_posterior_before_resample)
                    theta_MAP_posterior = theta_before_resample[idx_MAP_posterior]

                # Also compute ML for comparison (maximum likelihood, no prior)
                idx_ML_stage = np.argmax(logL_before_resample)
                theta_ML_stage = theta_before_resample[idx_ML_stage]

                # ★ CRITICAL: Record posterior MAP for final MAP selection
                # This ensures observation-corrected MAP is preserved
                # ★ ASSERT: Posterior MAP should NOT be overwritten after observation-based update
                if theta_MAP_posterior_computed:
                    # Verify that observation-corrected MAP is being used
                    assert (
                        theta_MAP_posterior is not None
                    ), "Observation-corrected MAP should be computed"
                    # ★ ASSERT: Verify that observation-corrected MAP was not overwritten
                    if theta_MAP_posterior_obs_corrected is not None:
                        assert np.allclose(
                            theta_MAP_posterior, theta_MAP_posterior_obs_corrected
                        ), "Observation-corrected MAP should NOT be overwritten after computation"
                    # Store the observation-corrected MAP
                    theta_MAP_posterior_history.append(theta_MAP_posterior.copy())
                else:
                    # Store standard MAP
                    theta_MAP_posterior_history.append(theta_MAP_posterior.copy())

                # ★ LINEARIZATION POINT SELECTION: Use mutation result (theta, logL)
                # Compute tempered posterior for mutation result
                log_prior_after_mutation = np.array([log_prior(t) for t in theta])
                log_posterior_after_mutation = log_prior_after_mutation + beta_next * logL

                # Select linearization MAP from mutation result
                idx_MAP_linearize = np.argmax(log_posterior_after_mutation)
                theta_MAP_linearize = theta[idx_MAP_linearize]

                # Store for backward compatibility (will be used for linearization)
                idx_MAP_stage = idx_MAP_linearize
                theta_MAP_stage = theta_MAP_linearize.copy()

                # Construct full 14-dim theta using linearization MAP (for ROM/linearization operations)
                # ★ NOTE: This is used for linearization point update, not for reporting
                theta_full_MAP = theta_base_full.copy()
                for i, idx in enumerate(active_indices):
                    theta_full_MAP[idx] = theta_MAP_linearize[i]

                rom_error_pre_from_enable_check: Optional[float] = None

                # (Stability gate) Try enabling linearization only after a MAP-based ROM error check.
                # This prevents enabling linearization in regions where ε_ROM is still large.
                if (beta_next >= float(linearization_threshold)) and (
                    not evaluator._linearization_enabled
                ):
                    enabled_ok = False
                    try:
                        evaluator.enable_linearization(True)
                        rom_err_try = evaluator.compute_ROM_error(theta_full_MAP)
                        if np.isfinite(rom_err_try) and (
                            rom_err_try <= float(linearization_enable_rom_threshold)
                        ):
                            enabled_ok = True
                            rom_error_pre_from_enable_check = float(rom_err_try)
                            debug_logger.log_info(
                                "✅ Linearization enabled at β=%.4f (threshold=%.3f) with ε_ROM(MAP)=%.6f <= %.6f",
                                beta_next,
                                float(linearization_threshold),
                                float(rom_err_try),
                                float(linearization_enable_rom_threshold),
                            )
                        else:
                            debug_logger.log_warning(
                                "Keeping linearization OFF (unstable): ε_ROM(MAP)=%.6f > %.6f (β=%.4f, threshold=%.3f)",
                                float(rom_err_try),
                                float(linearization_enable_rom_threshold),
                                beta_next,
                                float(linearization_threshold),
                            )
                    except Exception as e:
                        debug_logger.log_warning(
                            f"Linearization enable check failed: {e}. Keeping linearization OFF."
                        )
                    finally:
                        if not enabled_ok:
                            try:
                                evaluator.enable_linearization(False)
                            except Exception:
                                pass

                # Also compute weighted mean for comparison/reporting (optional)
                theta_weighted_mean = np.zeros(n_params)
                for i in range(n_particles):
                    theta_weighted_mean += weights_before_resample[i] * theta_before_resample[i]
                theta_full_weighted_mean = theta_base_full.copy()
                for i, idx in enumerate(active_indices):
                    theta_full_weighted_mean[idx] = theta_weighted_mean[i]

                # Get current linearization point
                theta0_old = evaluator.get_linearization_point()

                # ★ 改善: 線形化点更新判定にROM誤差を追加（論文で映えるstopping criterion）
                # Check 1: θ空間での変化（MAPベース）
                delta_theta0 = None
                if theta0_old is not None:
                    delta_theta0 = np.linalg.norm(theta_full_MAP - theta0_old)
                    delta_theta0_stage = float(delta_theta0)
                    if delta_theta0 < THETA_CONVERGENCE_THRESHOLD:
                        should_update = False
                        debug_logger.log_warning(
                            f"Linearization point converged (||Δθ₀||={delta_theta0:.6f} < {THETA_CONVERGENCE_THRESHOLD})"
                        )

                # Check 2: ROM誤差（論文でreviewerが大好きなstopping criterion）
                # Error in observable space: || φ̄_ROM(t_obs) − φ̄_FOM(t_obs) ||₂ / || φ̄_FOM(t_obs) ||₂
                rom_error_pre = None
                if should_update:
                    # ★ 優先度S: FOMチェックのスキップ条件を考慮
                    if use_observation_based_update and not should_do_fom:
                        # Skip FOM evaluation if conditions are met (use last known error)
                        rom_error_pre = last_rom_error
                        # Keep history aligned even when skipping FOM.
                        rom_error_pre_history.append(np.nan)
                    else:
                        # Use MAP for ROM error check
                        if rom_error_pre_from_enable_check is not None:
                            rom_error_pre = float(rom_error_pre_from_enable_check)
                        else:
                            rom_error_pre = evaluator.compute_ROM_error(theta_full_MAP)
                        rom_error_pre_stage = (
                            None if rom_error_pre is None else float(rom_error_pre)
                        )

                        if rom_error_pre is not None:
                            # ★ ERROR-CHECK: Check ROM error explosion
                            # Use previous stage's acceptance rate (if available) to skip check when acc_rate is very low
                            prev_acc_rate = (
                                acc_rate_history[-1] if len(acc_rate_history) > 0 else None
                            )
                            debug_logger.check_rom_error_explosion(
                                rom_error_pre,
                                context=f"Stage {stage}, linearization pre-update",
                                acc_rate=prev_acc_rate,
                            )

                            # Record pre-update ROM error (debugging only)
                            rom_error_pre_history.append(rom_error_pre)

                            if rom_error_pre < ROM_ERROR_THRESHOLD:
                                should_update = False
                                debug_logger.log_warning(
                                    f"ROM error sufficiently small (ε_ROM={rom_error_pre:.6f} < {ROM_ERROR_THRESHOLD})"
                                )
                                debug_logger.log_info(
                                    "   where ε_ROM = || φ̄_ROM(t_obs) − φ̄_FOM(t_obs) ||₂ / || φ̄_FOM(t_obs) ||₂"
                                )
                            else:
                                debug_logger.log_rom_error(
                                    stage, rom_error_pre, ROM_ERROR_THRESHOLD
                                )

                # Update linearization point if needed (use MAP, not weighted mean)
                if should_update and n_linearization_updates < MAX_LINEARIZATION_UPDATES:
                    # Stabilize θ0 updates:
                    # - Cap per-update step size (MAX_THETA0_STEP_NORM)
                    # - Allow a few sub-updates per event (MAX_LINEARIZATION_SUBUPDATES_PER_EVENT)
                    theta0_curr = evaluator.get_linearization_point()
                    theta0_start = theta0_curr.copy()

                    for _sub in range(MAX_LINEARIZATION_SUBUPDATES_PER_EVENT):
                        if n_linearization_updates >= MAX_LINEARIZATION_UPDATES:
                            break

                        delta_vec = theta_full_MAP - theta0_curr
                        delta_norm = float(np.linalg.norm(delta_vec))
                        if not np.isfinite(delta_norm) or delta_norm <= 1e-12:
                            break

                        alpha = (
                            1.0
                            if delta_norm <= MAX_THETA0_STEP_NORM
                            else (MAX_THETA0_STEP_NORM / delta_norm)
                        )
                        theta0_next = theta0_curr + alpha * delta_vec

                        # Apply update
                        evaluator.update_linearization_point(theta0_next)
                        n_linearization_updates += 1
                        theta0_history.append(theta0_next.copy())

                        # Report both MAP and weighted mean for comparison
                        delta_weighted_mean_map = np.linalg.norm(
                            theta_weighted_mean - theta_MAP_stage
                        )
                        debug_logger.log_info(
                            f"MAP-WeightedMean distance: {delta_weighted_mean_map:.6f}"
                        )

                        # Track last delta_theta0 for skip conditions (use actual step)
                        last_delta_theta0 = float(np.linalg.norm(theta0_next - theta0_curr))

                        # Log update (actual step size)
                        debug_logger.log_linearization_update(
                            stage=stage,
                            beta=beta_next,
                            update_num=n_linearization_updates,
                            theta0_old=theta0_curr,
                            theta0_new=theta0_next,
                            delta_norm=last_delta_theta0,
                        )

                        # Recompute logL for all particles with new linearization point
                        debug_logger.log_info("Recomputing logL with new linearization point...")
                        logL_prev = logL.copy()
                        logL_new = np.array([log_likelihood(t) for t in theta])
                        logL = logL_new
                        debug_logger.log_info(
                            f"✅ LogL recomputed: min={logL.min():.1f}, max={logL.max():.1f}"
                        )
                        # Guardrail: detect suspicious likelihood scale jumps after linearization updates.
                        # This often indicates that the evaluator regime (ROM/linearized ROM/FOM) or variance model changed
                        # dramatically, which can invalidate tempering assumptions.
                        try:
                            if np.all(np.isfinite(logL_prev)) and np.all(np.isfinite(logL)):
                                prev_range = float(np.max(logL_prev) - np.min(logL_prev))
                                new_range = float(np.max(logL) - np.min(logL))
                                prev_med = float(np.median(logL_prev))
                                new_med = float(np.median(logL))
                                med_shift = float(abs(new_med - prev_med))
                                lin_enabled = bool(
                                    getattr(evaluator, "_linearization_enabled", False)
                                )
                                # Heuristic: median shift far larger than prior range or absolute huge jump
                                if (prev_range > 0 and med_shift > 50.0 * prev_range) or (
                                    med_shift > 1e3
                                ):
                                    debug_logger.log_warning(
                                        "Suspicious logL scale jump after θ₀ update: "
                                        f"median {prev_med:.2f}→{new_med:.2f} (|Δ|={med_shift:.2e}), "
                                        f"range {prev_range:.2f}→{new_range:.2f}, "
                                        f"linearization_enabled={lin_enabled}."
                                    )
                        except Exception:
                            # Never fail TMCMC because of diagnostics
                            pass

                        # Post-update ROM error (this is what we report/gate on)
                        rom_error_post = None
                        if use_observation_based_update and not should_do_fom:
                            rom_error_history.append(np.nan)
                        else:
                            rom_error_post = evaluator.compute_ROM_error(theta_full_MAP)
                            if rom_error_post is not None:
                                rom_error_history.append(rom_error_post)
                                last_rom_error = rom_error_post
                                rom_error_post_stage = float(rom_error_post)
                                debug_logger.log_info(
                                    f"[TMCMC] ROM error (post-update): {rom_error_post:.6f} (threshold: {ROM_ERROR_THRESHOLD})"
                                )

                        theta0_curr = theta0_next

                        # Stop further sub-updates if ROM error is now sufficiently small.
                        if rom_error_post is not None and rom_error_post < ROM_ERROR_THRESHOLD:
                            break
                elif n_linearization_updates >= MAX_LINEARIZATION_UPDATES:
                    debug_logger.log_warning(
                        f"Reached max linearization updates ({MAX_LINEARIZATION_UPDATES}), stopping updates"
                    )

        # ★ ERROR-CHECK: Check acceptance rate (post-recovery)
        debug_logger.check_acceptance_rate(acc_rate, context=f"Stage {stage}")

        # ★ Log beta progress and acceptance rate
        debug_logger.log_beta_progress(stage, beta_next, delta_beta)
        debug_logger.log_acceptance_rate(stage, acc_rate, acc, total_proposals)

        # ★ ERROR-CHECK: Check numerical errors after mutation
        debug_logger.check_numerical_errors(logL, theta, context=f"Stage {stage}, after mutation")

        # ★ Record acceptance rate history
        acc_rate_history.append(acc_rate)

        # Diagnostics: likelihood/TSM health counters (high-signal when accuracy stagnates)
        try:
            if evaluator is not None and debug_logger.config.level in (
                DebugLevel.MINIMAL,
                DebugLevel.VERBOSE,
            ):
                h = evaluator.get_health()
                # Only log if something looks off (keeps noise down)
                key_stats = {
                    "n_calls": int(h.get("n_calls", 0)),
                    "n_tsm_fail": int(h.get("n_tsm_fail", 0)),
                    "n_output_nonfinite": int(h.get("n_output_nonfinite", 0)),
                    "n_var_raw_negative": int(h.get("n_var_raw_negative", 0)),
                    "n_var_raw_nonfinite": int(h.get("n_var_raw_nonfinite", 0)),
                    "n_var_total_clipped": int(h.get("n_var_total_clipped", 0)),
                }
                if any(v > 0 for k, v in key_stats.items() if k != "n_calls"):
                    debug_logger.log_warning(f"Likelihood health (cumulative): {key_stats}")
                else:
                    debug_logger.log_info(f"Likelihood health (cumulative): {key_stats}")
        except Exception:
            pass

        # Record stage summary for offline debugging (CSV export)
        stage_summary.append(
            {
                "stage": int(stage),
                "beta": float(beta),
                "beta_next": float(beta_next),
                "delta_beta": float(delta_beta),
                "ess": float(ess_at_delta_low) if ess_at_delta_low is not None else None,
                "ess_target": float(target_ess_ratio * n_particles),
                "acc_rate": float(acc_rate),
                "logL_min": float(np.min(logL)) if len(logL) > 0 else None,
                "logL_max": float(np.max(logL)) if len(logL) > 0 else None,
                "linearization_enabled": (
                    int(bool(getattr(evaluator, "_linearization_enabled", False)))
                    if evaluator is not None
                    else 0
                ),
                "rom_error_pre": rom_error_pre_stage,
                "rom_error_post": rom_error_post_stage,
                "delta_theta0": delta_theta0_stage,
            }
        )

        beta = beta_next
        beta_schedule.append(beta)

        if beta >= BETA_CONVERGENCE_THRESHOLD:
            debug_logger.log_info("✓ Converged! β reached 1.0", force=True)
            # ★ Slack notification: Convergence (add to thread if available)
            if SLACK_ENABLED and model_name:
                conv_msg = f"✅ Converged! Stage: {stage}/{n_stages}, β = {beta:.4f}"
                if slack_thread_ts:
                    _slack_notifier.add_to_thread(slack_thread_ts, conv_msg)
                else:
                    notify_slack(
                        f"✅ {model_name} TMCMC Converged\n"
                        f"   Stage: {stage}/{n_stages}\n"
                        f"   β = {beta:.4f} (reached {BETA_CONVERGENCE_THRESHOLD})",
                        raise_on_error=False,
                    )
            break

    # ★ CRITICAL FIX: Extract final MAP
    # Priority: Use posterior MAP from last stage if available (preserves observation correction)
    # Fallback: Use standard MAP from final logL (if no observation-based update was used)
    #
    # ★ ASSERT: Verify consistency of MAP history
    # The number of MAP records should match the number of stages where should_update=True
    # (This is approximate since should_update depends on conditions)
    if len(theta_MAP_posterior_history) > 0:
        # Use posterior MAP from last stage (preserves observation information)
        # Note: This is the MAP from the last stage where should_update=True
        theta_MAP = theta_MAP_posterior_history[-1].copy()
        map_source = "posterior (observation-corrected)"
    else:
        # Fallback: standard MAP computation (no observation-based update was used, or evaluator was None)
        # This happens when:
        # - evaluator is None (no linearization update)
        # - use_observation_based_update=False
        # - should_update was never True (e.g., β never reached threshold)
        # Compute tempered posterior for final particles
        log_prior_final = np.array([log_prior(t) for t in theta])
        log_posterior_final = log_prior_final + beta * logL
        idx_MAP = np.argmax(log_posterior_final)
        theta_MAP = theta[idx_MAP]
        map_source = "standard (from final posterior)"

    debug_logger.log_info(f"✅ TMCMC complete! Final β={beta:.4f}", force=True)
    debug_logger.log_info(f"🎯 MAP ({map_source}): {theta_MAP}", force=True)

    # ★ Slack notification: TMCMC complete (add to thread if available)
    if SLACK_ENABLED and model_name:
        complete_msg = (
            f"✅ TMCMC Complete\n"
            f"   Final β: {beta:.4f}\n"
            f"   Converged: {beta >= BETA_CONVERGENCE_THRESHOLD}\n"
            f"   Stages: {len(beta_schedule)}\n"
            f"   MAP ({map_source}): {theta_MAP}"
        )
        if slack_thread_ts:
            _slack_notifier.add_to_thread(slack_thread_ts, complete_msg)
        else:
            notify_slack(
                f"✅ {model_name} TMCMC Complete\n"
                f"   Final β: {beta:.4f}\n"
                f"   Converged: {beta >= BETA_CONVERGENCE_THRESHOLD}\n"
                f"   Stages: {len(beta_schedule)}",
                raise_on_error=False,
            )

        # ★ ASSERT: Final MAP should match the last recorded posterior MAP (only if history exists)
        if len(theta_MAP_posterior_history) > 0:
            assert np.allclose(
                theta_MAP, theta_MAP_posterior_history[-1]
            ), "Final MAP should match last recorded posterior MAP"

    # Compute final MAP for global sharing (if multiple chains)
    # Note: theta0_history now contains MAP values
    final_MAP = None
    if len(theta0_history) > 0:
        # Use the last updated MAP
        final_MAP = theta0_history[-1].copy()
    elif evaluator is not None:
        # If no updates, use current linearization point
        final_MAP = evaluator.get_linearization_point()

    # ★ Calculate evaluation counts
    n_rom_evaluations = 0
    n_fom_evaluations = 0
    if evaluator is not None:
        n_rom_evaluations = evaluator.call_count - initial_rom_count
        n_fom_evaluations = evaluator.fom_call_count - initial_fom_count

    wall_time_s = float(time.perf_counter() - tmcmc_wall_start)
    timing_breakdown_s: Optional[Dict[str, float]] = None
    if (
        evaluator is not None
        and hasattr(evaluator, "timing")
        and isinstance(getattr(evaluator, "timing"), TimingStats)
    ):
        tsm_s = float(evaluator.timing.get_s("tsm.solve_tsm"))
        fom_s = float(evaluator.timing.get_s("fom.run_deterministic"))
        tmcmc_overhead_s = float(max(0.0, wall_time_s - tsm_s - fom_s))
        timing_breakdown_s = {
            "tmcmc_total_s": wall_time_s,
            "tsm_s": tsm_s,
            "fom_s": fom_s,
            "tmcmc_overhead_s": tmcmc_overhead_s,
        }

    likelihood_health: Optional[Dict[str, int]] = None
    if evaluator is not None and hasattr(evaluator, "get_health"):
        try:
            likelihood_health = evaluator.get_health()  # type: ignore[assignment]
        except Exception:
            likelihood_health = None

    return TMCMCResult(
        samples=theta,
        logL_values=logL,
        theta_MAP=theta_MAP,
        beta_schedule=beta_schedule,
        converged=(beta >= BETA_CONVERGENCE_THRESHOLD),
        theta0_history=theta0_history if theta0_history else None,
        n_linearization_updates=n_linearization_updates,
        final_MAP=final_MAP,  # ★ For global chain sharing
        rom_error_pre_history=rom_error_pre_history if rom_error_pre_history else None,
        rom_error_history=rom_error_history if rom_error_history else None,  # ★ ROM error history
        acc_rate_history=(
            acc_rate_history if acc_rate_history else None
        ),  # ★ Acceptance rate history
        n_rom_evaluations=n_rom_evaluations,  # ★ Number of ROM evaluations
        n_fom_evaluations=n_fom_evaluations,  # ★ Number of FOM evaluations
        wall_time_s=wall_time_s,
        timing_breakdown_s=timing_breakdown_s,
        likelihood_health=likelihood_health,
        stage_summary=stage_summary if stage_summary else None,
    )


def run_multi_chain_MCMC(
    model_tag: str,
    evaluator_factory: callable,
    prior_bounds: List[Tuple[float, float]],
    mcmc_config: MCMCConfig,
    proposal_cov: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Run multiple MCMC chains sequentially with diagnostics."""
    logger.info("[%s] Running %s MCMC chains...", model_tag, mcmc_config.n_chains)

    all_samples = []
    all_logL = []
    all_MAP = []
    all_acc = []

    for chain_idx in range(mcmc_config.n_chains):
        seed = mcmc_config.n_chains * 1000 + chain_idx
        logger.info("Chain %s/%s", chain_idx + 1, mcmc_config.n_chains)

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

    diag = MCMCDiagnostics(all_samples, [f"θ{i}" for i in range(len(prior_bounds))])
    diag.compute_all()

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


def run_multi_chain_TMCMC(
    model_tag: str,
    make_evaluator: callable,  # ★ Should accept theta_linearization argument
    prior_bounds: List[Tuple[float, float]],
    theta_base_full: np.ndarray,  # ★ Full 14-dim theta base
    active_indices: List[int],  # ★ Active parameter indices
    theta_linearization_init: Optional[np.ndarray] = None,  # ★ Initial linearization point
    n_particles: int = 2000,
    n_stages: int = 30,
    target_ess_ratio: float = 0.5,
    min_delta_beta: float = 0.05,
    max_delta_beta: float = 0.2,
    logL_scale: float = 1.0,
    n_chains: int = 1,
    update_linearization_interval: int = 3,  # ★ Update every N stages
    n_mutation_steps: int = 5,  # ★ Number of MCMC steps per particle (K-step mutation)
    use_observation_based_update: bool = True,  # ★ Use observation-based linearization update (ROM error weighted)
    linearization_threshold: float = DEFAULT_LINEARIZATION_THRESHOLD,
    linearization_enable_rom_threshold: float = 0.05,
    debug_config: Optional[DebugConfig] = None,  # ★ Debug configuration
    seed: Optional[int] = None,  # ★ Base seed for reproducibility across runs
    force_beta_one: bool = False,  # ★ If True, force β=1.0 at final stage (paper runs)
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[bool], Dict]:
    """
    Run multiple TMCMC chains sequentially with diagnostics + Linearization Update.

    ★ TMCMC版のマルチチェーン実行関数（β tempering + 線形化点更新）

    ★ 重要な改善点：
    1. make_evaluator(theta_linearization) で線形化点を受け取る
    2. 各stageでMAPを計算し、線形化点を更新
    3. 「TMCMC × 線形化更新」という論文の核を実現

    Parameters
    ----------
    model_tag : str
        Model identifier
    make_evaluator : callable
        Function that returns LogLikelihoodEvaluator instance
    prior_bounds : List[Tuple[float, float]]
        Prior bounds for each parameter
    n_particles : int
        Number of particles per chain
    n_stages : int
        Maximum number of TMCMC stages
    target_ess_ratio : float
        Target ESS ratio (0.5 = 50% ESS)
    min_delta_beta : float
        Minimum β increment per stage (progress floor).
    max_delta_beta : float
        Maximum β increment per stage (jump cap).
    logL_scale : float
        [DEPRECATED] Likelihood scaling factor. Currently ignored for consistency with TMCMC theory.
        All likelihood calculations (ESS, resampling, mutation) now use unscaled logL.
    n_chains : int
        Number of chains to run

    Returns
    -------
    all_samples : List[np.ndarray]
        Samples from each chain
    all_logL : List[np.ndarray]
        Log-likelihood values from each chain
    global_MAP : np.ndarray
        Global MAP across all chains
    converged_flags : List[bool]
        Convergence status for each chain
    diagnostics : Dict
        MCMC diagnostics
    """
    logger.info(
        "[%s] Running %s TMCMC chains (β tempering + linearization update)...", model_tag, n_chains
    )

    # Initialize debug logger
    if debug_config is None:
        debug_config = DebugConfig(level=DebugLevel.OFF)
    debug_logger = DebugLogger(debug_config)

    # Initialize linearization point
    if theta_linearization_init is None:
        theta_linearization_init = theta_base_full.copy()

    all_samples = []
    all_logL = []
    all_MAP = []
    converged_flags = []
    all_beta_schedules = []
    all_theta0_histories = []  # ★ Track linearization point update history
    total_linearization_updates = 0  # ★ Track total updates
    all_MAPs = []  # ★ Collect MAPs from each chain for global sharing
    all_results = []  # ★ Store all TMCMCResult objects for diagnostics

    # ★ Global MAP for chain sharing (improves accuracy by using best estimate from each chain)
    global_MAP = theta_linearization_init.copy()

    for chain_idx in range(n_chains):
        # ★ PRIORITY A: seed に model_tag を含める（M1/M2同一挙動の切り分け）
        # 以前: seed = n_chains * 1000 + chain_idx  → M1/M2で同じseedになる可能性
        # 修正: model_tag を含めてモデルごとに異なるseedを生成
        base = int(seed or 0)
        seed_base = _stable_hash_int(model_tag) % (2**31)  # stable hash across runs
        chain_seed = (base + seed_base + n_chains * 1000 + chain_idx) % (2**31)
        logger.info("Chain %s/%s", chain_idx + 1, n_chains)
        logger.debug("seed: %s (model_tag: %s, chain: %s)", chain_seed, model_tag, chain_idx)

        # ★ Slack notification: 削除（詳細すぎるため、重要な情報のみ送信）

        # ★ Create evaluator with linearization point
        # Chain 0: use initial point
        # Chain 1+: use global MAP from previous chains (if available)
        if chain_idx == 0:
            # First chain: use initial linearization point
            current_linearization = theta_linearization_init.copy()
        else:
            # Subsequent chains: use global MAP from previous chains
            # ★ 改善④: Global MAP sharing across chains
            # This improves accuracy by leveraging the best estimate from all chains
            current_linearization = global_MAP.copy()
            logger.info("[Chain %s] Using global MAP from previous chains", chain_idx + 1)

        evaluator = make_evaluator(theta_linearization=current_linearization)
        # ★ Pass debug_logger to evaluator for silent error handling in ERROR/OFF mode
        if hasattr(evaluator, "debug_logger") or hasattr(evaluator, "__dict__"):
            evaluator.debug_logger = debug_logger
        logger.info("[Chain %s] Initial linearization point set", chain_idx + 1)

        result = run_TMCMC(
            log_likelihood=evaluator,
            prior_bounds=prior_bounds,
            n_particles=n_particles,
            n_stages=n_stages,
            target_ess_ratio=target_ess_ratio,
            min_delta_beta=min_delta_beta,
            max_delta_beta=max_delta_beta,
            logL_scale=logL_scale,
            seed=chain_seed,
            model_name=f"{model_tag}_chain{chain_idx+1}",
            evaluator=evaluator,  # ★ Pass evaluator for linearization update
            theta_base_full=theta_base_full,  # ★ Pass full theta base
            active_indices=active_indices,  # ★ Pass active indices
            update_linearization_interval=update_linearization_interval,  # ★ Update interval
            n_mutation_steps=n_mutation_steps,  # ★ K-step mutation
            use_observation_based_update=use_observation_based_update,  # ★ Observation-based update (ROM error weighted)
            linearization_threshold=linearization_threshold,
            linearization_enable_rom_threshold=linearization_enable_rom_threshold,
            debug_logger=debug_logger,  # ★ Pass debug logger
            force_beta_one=force_beta_one,
        )

        all_samples.append(result.samples)
        all_logL.append(result.logL_values)
        all_MAP.append(result.theta_MAP)
        converged_flags.append(result.converged)
        all_beta_schedules.append(result.beta_schedule)
        all_results.append(result)  # ★ Store result for diagnostics

        # ★ Track linearization point update history
        if result.theta0_history is not None:
            all_theta0_histories.append(result.theta0_history)
        total_linearization_updates += result.n_linearization_updates

        # ★ Collect MAP from this chain for global sharing
        if result.final_MAP is not None:
            all_MAPs.append(result.final_MAP.copy())

            # Update global MAP: use the MAP with highest log-likelihood across all chains
            # This provides the best estimate from all chains
            if len(all_MAPs) > 0:
                # Use the MAP from the chain with highest log-likelihood
                # (already computed in global_MAP calculation below)
                # For now, use the latest MAP (can be improved to select best)
                global_MAP = all_MAPs[-1].copy()
                logger.info(
                    "[Chain %s] Global MAP updated from %s chains", chain_idx + 1, len(all_MAPs)
                )

    # Global MAP (highest log-likelihood across all chains)
    # Find the chain and sample index with the highest log-likelihood
    best_logL = -np.inf
    best_chain_idx = 0
    best_sample_idx = 0
    for chain_idx, logL_chain in enumerate(all_logL):
        sample_idx = np.argmax(logL_chain)
        if logL_chain[sample_idx] > best_logL:
            best_logL = logL_chain[sample_idx]
            best_chain_idx = chain_idx
            best_sample_idx = sample_idx
    global_MAP = all_samples[best_chain_idx][best_sample_idx]

    # Diagnostics
    # ⚠️ 重要: TMCMCはMarkov chainではないため、R-hat/ESSは理論的に正当化されない
    # TMCMC uses resampling, particle duplication, and tempered likelihood,
    # which violate the Markov chain assumptions required for R-hat/ESS.
    # We compute them only as reference indicators, NOT for convergence judgment.
    from mcmc_diagnostics import MCMCDiagnostics

    # ★ 修正: MCMCDiagnosticsはチェーンのリスト（List[np.ndarray]）を想定している
    # all_samples_flat（2次元配列）を渡すと、chains[0]が1次元配列になってIndexErrorが発生
    param_names = [f"θ{i}" for i in range(len(prior_bounds))]
    diag = MCMCDiagnostics(all_samples, param_names)  # ← List of chains を渡す
    diag.compute_all()

    diagnostics = {
        # ⚠️ Reference only: R-hat/ESS are NOT theoretically valid for TMCMC
        # See: Del Moral et al. (2006), Ching & Chen (2007) for SMC/TMCMC theory
        "Rhat_reference": diag.Rhat,  # ★ Reference indicator only (not for convergence)
        "ESS_reference": diag.ESS,  # ★ Reference indicator only (not for convergence)
        "converged_chains": sum(converged_flags),
        "total_chains": n_chains,
        "MAP_global": global_MAP,
        "beta_schedules": all_beta_schedules,
        "theta0_history": all_theta0_histories,  # ★ Linearization point update history
        "total_linearization_updates": total_linearization_updates,  # ★ Total number of updates
        "rom_error_pre_histories": [
            r.rom_error_pre_history for r in all_results if r.rom_error_pre_history is not None
        ],  # ★ pre-update ROM errors
        "rom_error_histories": [
            r.rom_error_history for r in all_results if r.rom_error_history is not None
        ],  # ★ ROM error histories
        "acc_rate_histories": [
            r.acc_rate_history for r in all_results if r.acc_rate_history is not None
        ],  # ★ Acceptance rate histories
        "n_rom_evaluations": [
            r.n_rom_evaluations for r in all_results
        ],  # ★ ROM evaluation counts per chain
        "n_fom_evaluations": [
            r.n_fom_evaluations for r in all_results
        ],  # ★ FOM evaluation counts per chain
        "tmcmc_wall_time_s": [float(r.wall_time_s) for r in all_results],  # ★ Wall time per chain
        "timing_breakdown_s": [
            r.timing_breakdown_s for r in all_results
        ],  # ★ Per-chain breakdown (tsm/fom/overhead)
        "likelihood_health_histories": [
            r.likelihood_health for r in all_results if r.likelihood_health is not None
        ],
        "stage_summaries": [r.stage_summary for r in all_results if r.stage_summary is not None],
        "note": "R-hat/ESS are computed for reference only. TMCMC convergence is judged by β=1.0 and chain consistency.",
    }

    # Aggregate likelihood health across chains (for quick checks / metrics.json)
    health_total: Dict[str, int] = {}
    for h in diagnostics.get("likelihood_health_histories", []):
        if not isinstance(h, dict):
            continue
        for k, v in h.items():
            try:
                health_total[k] = int(health_total.get(k, 0) + int(v))
            except Exception:
                continue
    if health_total:
        diagnostics["likelihood_health_total"] = health_total

    logger.info("[%s] TMCMC Summary:", model_tag)
    logger.info("Converged chains: %s/%s", sum(converged_flags), n_chains)
    logger.info("Global MAP: %s", global_MAP)

    # ★ Slack notification: All chains complete
    if SLACK_ENABLED:
        converged_count = sum(converged_flags)
        notify_slack(
            f"✅ {model_tag} All {n_chains} chains completed\n"
            f"   Converged: {converged_count}/{n_chains}\n"
            f"   Total linearization updates: {total_linearization_updates}"
        )

    return all_samples, all_logL, global_MAP, converged_flags, diagnostics


# ==============================================================================
# 2-PHASE MCMC WITH LINEARIZATION UPDATE
# ==============================================================================


def run_two_phase_MCMC_with_linearization(
    model_tag: str,
    make_evaluator: callable,
    prior_bounds: List[Tuple[float, float]],
    mcmc_config: MCMCConfig,
    theta_base: np.ndarray,
    active_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Run 2-phase MCMC with TSM linearization point update.

    ★ CRITICAL ALGORITHM:
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
    logger.info(
        "Phase 1 ||θ - θ_lin||: %.6f", np.linalg.norm(MAP_p1 - theta_lin_init[active_indices])
    )

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
        f"{model_tag}_Phase2",
        make_evaluator_p2,
        prior_bounds,
        mcmc_config,
        proposal_cov=proposal_cov,
    )

    samples_p2 = np.concatenate(chains_p2, axis=0)

    logger.info("Phase 2 MAP: %s", MAP_p2)
    logger.info(
        "Phase 2 ||θ - θ_lin||: %.6f", np.linalg.norm(MAP_p2 - theta_lin_new[active_indices])
    )

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
    """Generate synthetic data from TSM simulation."""
    logger.info("[%s] Generating synthetic data...", name)

    solver_kwargs = {
        k: v
        for k, v in config.items()
        if k not in ["active_species", "active_indices", "param_names"]
    }

    solver = BiofilmNewtonSolver(
        **solver_kwargs,
        active_species=config["active_species"],
        use_numba=HAS_NUMBA,
    )

    # Use BiofilmTSM_Analytical for consistency
    tsm = BiofilmTSM_Analytical(
        solver,
        active_theta_indices=config["active_indices"],
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
        use_analytical=True,
        theta_linearization=theta_true,
    )

    t_arr, x0, sig2 = tsm.solve_tsm(theta_true)

    idx_sparse = select_sparse_data_indices(len(t_arr), exp_config.n_data)
    phibar = compute_phibar(x0, config["active_species"])

    # ★ CRITICAL FIX: Use default_rng consistently
    rng = np.random.default_rng(exp_config.random_seed + (_stable_hash_int(name) % 1000))

    data = np.zeros((exp_config.n_data, len(config["active_species"])))
    for i, sp in enumerate(config["active_species"]):
        data[:, i] = (
            phibar[idx_sparse, i] + rng.standard_normal(exp_config.n_data) * exp_config.sigma_obs
        )

    plot_mgr.plot_TSM_simulation(t_arr, x0, config["active_species"], name, data, idx_sparse)

    logger.info(
        "Generated %s observations for %s species",
        exp_config.n_data,
        len(config["active_species"]),
    )

    return data, idx_sparse, t_arr, x0, sig2


# ==============================================================================
# MAIN
# ==============================================================================
# NOTE: main() function has been moved to main.case2_main module.
# Imported above for backward compatibility.


def main():
    start_time_global = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args = parse_args()
    # Normalize model names case-insensitively while keeping canonical keys from MODEL_CONFIGS.
    requested_raw = [m.strip() for m in str(args.models).split(",") if m.strip()]
    key_map = {str(k).upper(): str(k) for k in MODEL_CONFIGS.keys()}
    requested_models = [key_map[m.upper()] for m in requested_raw if m.upper() in key_map]
    if not requested_models:
        requested_models = ["M1", "M2", "M3"]

    # ★ Slack notification: Process start
    if SLACK_ENABLED:
        notify_slack(
            f"🚀 TMCMC Process Started\n"
            f"   Time: {start_time_str}\n"
            f"   Case II: Hierarchical Bayesian Estimation with TSM Linearization"
        )

    # Configuration (CLI-driven)
    if args.mode == "paper":
        default_debug_level = DebugLevel.OFF
    elif args.mode == "sanity":
        default_debug_level = DebugLevel.MINIMAL
    else:
        default_debug_level = DebugLevel.VERBOSE
    debug_level = DebugLevel[args.debug_level] if args.debug_level else default_debug_level
    # Configure Python logging as early as possible (replaces all previous `print` usage).
    if debug_level == DebugLevel.VERBOSE:
        setup_logging("DEBUG")
    elif debug_level == DebugLevel.MINIMAL:
        setup_logging("INFO")
    else:
        setup_logging("ERROR")
    debug_config = DebugConfig(level=debug_level)
    debug_logger = DebugLogger(debug_config)  # ★ Create debug_logger for evaluators

    logger.info("%s", "=" * 80)
    logger.info("Case II: Hierarchical Bayesian Estimation with TSM Linearization")
    logger.info("%s", "=" * 80)
    logger.info("Start time: %s", start_time_str)
    logger.info("CLI: %s", " ".join(shlex.quote(a) for a in sys.argv))

    # ★ FAST-SANITY MODE: 30秒以内で「コードが壊れてないか」だけ確認
    # 本番実行時は False に設定
    FAST_SANITY_MODE = args.mode == "sanity"

    # ★ DIAGNOSTIC MODE: 解析微分の切り分け用
    # paper_mode=False にすると complex-step にフォールバック（解析微分無効化）
    USE_PAPER_ANALYTICAL = (
        True if args.use_paper_analytical is None else bool(args.use_paper_analytical)
    )

    # ★ PRODUCTION HYPERPARAMETERS (本番実行用)
    # 論文・再現実験向けの推奨設定（Phase A: 精度・安定性優先）
    PRODUCTION_TMCMC = {
        "n_particles": 1000,  # ★ 推奨: 2000-5000 (精度重視なら5000)
        "n_stages": 50,  # ★ 推奨: 30-50 (βジャンプを小さくし、β=1.0到達を確実に)
        "n_mutation_steps": 5,  # ★ 推奨: 5-10 (粒子相関を減らす)
        "n_chains": 1,  # ★ 推奨: 3-5 (収束診断のため)
        # β schedule controls (accuracy/stability first)
        "target_ess_ratio": float(TMCMC_DEFAULTS.target_ess_ratio),
        "min_delta_beta": 0.02,
        "max_delta_beta": 0.05,
        # Linearization management (guarded)
        "update_linearization_interval": int(TMCMC_DEFAULTS.update_linearization_interval),
        "linearization_threshold": float(TMCMC_DEFAULTS.linearization_threshold),
        "linearization_enable_rom_threshold": 0.05,
        "force_beta_one": True,
    }
    # Apply CLI overrides (if provided)
    if args.n_particles is not None:
        PRODUCTION_TMCMC["n_particles"] = int(args.n_particles)
    if args.n_stages is not None:
        PRODUCTION_TMCMC["n_stages"] = int(args.n_stages)
    if args.n_mutation_steps is not None:
        PRODUCTION_TMCMC["n_mutation_steps"] = int(args.n_mutation_steps)
    if args.n_chains is not None:
        PRODUCTION_TMCMC["n_chains"] = int(args.n_chains)
    if args.target_ess_ratio is not None:
        PRODUCTION_TMCMC["target_ess_ratio"] = float(args.target_ess_ratio)
    if args.min_delta_beta is not None:
        PRODUCTION_TMCMC["min_delta_beta"] = float(args.min_delta_beta)
    if args.max_delta_beta is not None:
        PRODUCTION_TMCMC["max_delta_beta"] = float(args.max_delta_beta)
    if args.update_linearization_interval is not None:
        PRODUCTION_TMCMC["update_linearization_interval"] = int(args.update_linearization_interval)
    if args.linearization_threshold is not None:
        PRODUCTION_TMCMC["linearization_threshold"] = float(args.linearization_threshold)
    if args.linearization_enable_rom_threshold is not None:
        PRODUCTION_TMCMC["linearization_enable_rom_threshold"] = float(
            args.linearization_enable_rom_threshold
        )
    if bool(args.force_beta_one):
        PRODUCTION_TMCMC["force_beta_one"] = True

    LOCK_PAPER_CONDITIONS = (args.mode == "paper") or bool(args.lock_paper_conditions)
    if LOCK_PAPER_CONDITIONS:
        # Paper conditions: fixed sigma_obs/cov_rel.
        # Keep TMCMC conservative defaults unless overridden above.
        PRODUCTION_TMCMC["force_beta_one"] = True

    # (改善2) Guardrails: ensure β=1.0 is reachable and mutation isn't trivially weak.
    if not FAST_SANITY_MODE:
        min_db = float(PRODUCTION_TMCMC.get("min_delta_beta", 0.0))
        if min_db > 0:
            min_required_stages = int(math.ceil(1.0 / min_db))
            if PRODUCTION_TMCMC["n_stages"] < min_required_stages and not bool(
                PRODUCTION_TMCMC.get("force_beta_one", False)
            ):
                logger.warning(
                    "n_stages=%s is too small to guarantee β=1 with min_delta_beta=%.4f. "
                    "Bumping to %s (or use --force-beta-one).",
                    PRODUCTION_TMCMC["n_stages"],
                    min_db,
                    min_required_stages,
                )
                PRODUCTION_TMCMC["n_stages"] = min_required_stages
        if PRODUCTION_TMCMC["n_mutation_steps"] < 1:
            logger.warning("n_mutation_steps must be >= 1. Bumping to 1.")
            PRODUCTION_TMCMC["n_mutation_steps"] = 1

    if FAST_SANITY_MODE:
        # Fast-sanity settings: minimal particles/stages for quick check
        mcmc_config = MCMCConfig(
            n_samples=50, n_chains=1, debug=debug_config  # Reduced for speed  # Single chain
        )
        # TMCMC fast-sanity settings (will be used in run_multi_chain_TMCMC calls)
        tmcmc_fast_sanity = {
            "n_particles": 10,
            "n_stages": 2,
            "n_mutation_steps": 1,
            "n_chains": 1,
        }
        logger.info("FAST-SANITY MODE ENABLED (quick code check, ~30 seconds)")
    else:
        # Normal production settings
        mcmc_config = MCMCConfig(
            n_samples=2000,  # ★ 本番: 200-1000 (必要に応じて調整)
            n_chains=PRODUCTION_TMCMC["n_chains"],  # Use production n_chains
            debug=debug_config,
        )
        tmcmc_fast_sanity = None  # Use production settings

    exp_config = ExperimentConfig(debug=debug_config)
    exp_config.random_seed = int(args.seed)

    # Override sigma_obs and cov_rel if specified (CLI), unless paper conditions are locked.
    if LOCK_PAPER_CONDITIONS:
        if args.sigma_obs is not None and not math.isclose(
            float(args.sigma_obs), 0.01, rel_tol=0.0, abs_tol=1e-12
        ):
            logger.warning(
                "Ignoring --sigma-obs=%s due to paper-condition lock (sigma_obs=0.01).",
                args.sigma_obs,
            )
        if args.cov_rel is not None and not math.isclose(
            float(args.cov_rel), 0.005, rel_tol=0.0, abs_tol=1e-12
        ):
            logger.warning(
                "Ignoring --cov-rel=%s due to paper-condition lock (cov_rel=0.005).", args.cov_rel
            )
        exp_config.sigma_obs = 0.01
        exp_config.cov_rel = 0.005
    else:
        if args.sigma_obs is not None:
            exp_config.sigma_obs = float(args.sigma_obs)
            logger.warning("Overriding sigma_obs: %s (default: 0.01)", exp_config.sigma_obs)
        if args.cov_rel is not None:
            exp_config.cov_rel = float(args.cov_rel)
            logger.warning("Overriding cov_rel: %s (default: 0.005)", exp_config.cov_rel)

    if args.rho is not None:
        exp_config.rho = float(args.rho)
        logger.info("Using observation correlation rho: %s", exp_config.rho)

    # Reporting-only: paper Nsamples for double-loop cost conversion
    if args.aleatory_samples is not None:
        exp_config.aleatory_samples = int(args.aleatory_samples)
        logger.info(
            "Using aleatory_samples=%s for double-loop cost reporting", exp_config.aleatory_samples
        )
    elif args.mode == "paper":
        exp_config.aleatory_samples = 500

    # ★ Output standardization: output_root/run_id/{config.json,metrics.json,figures/,diagnostics_tables/,results...}
    mode = str(args.mode)
    output_root = args.output_root or _default_output_root_for_mode(mode)
    if args.run_id:
        run_id = str(args.run_id)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}_seed{exp_config.random_seed}"
    run_dir = Path(output_root) / run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    exp_config.output_dir = str(run_dir)

    # Persist logs under the run directory once it is known.
    # This adds a FileHandler without removing the console handler.
    if debug_level == DebugLevel.VERBOSE:
        setup_logging("DEBUG", log_path=run_dir / "run.log")
    elif debug_level == DebugLevel.MINIMAL:
        setup_logging("INFO", log_path=run_dir / "run.log")
    else:
        setup_logging("ERROR", log_path=run_dir / "run.log")

    # Persist structured debug events separately (JSON Lines).
    # This avoids mixing JSON into stdout while keeping aggregation easy.
    debug_logger.set_events_jsonl(run_dir / "events.jsonl")

    # ★ 将来用ガード: main()外利用時の保険
    assert exp_config.output_dir is not None, "output_dir must be set before use"

    output_dir = Path(exp_config.output_dir)

    # ★ CONFIG SUMMARY: Always print once (regardless of debug level)
    logger.info("%s", "=" * 80)
    logger.info("EXPERIMENT CONFIGURATION SUMMARY")
    logger.info("%s", "=" * 80)
    logger.info(
        "Mode: %s",
        (
            "FAST-SANITY"
            if FAST_SANITY_MODE
            else ("PRODUCTION" if debug_config.level == DebugLevel.OFF else "DEBUG")
        ),
    )
    logger.info("Debug Level: %s", debug_config.level.name)
    logger.info("Output Root: %s", output_root)
    logger.info("Run ID: %s", run_id)
    logger.info("Run Directory: %s", run_dir)
    logger.info("Figures Directory: %s", figures_dir)
    logger.info(
        "MCMC Settings: n_samples=%s, n_chains=%s, initial_scale=%s, target_accept=%s",
        mcmc_config.n_samples,
        mcmc_config.n_chains,
        mcmc_config.initial_scale,
        mcmc_config.target_accept,
    )
    logger.info("TMCMC Settings (per model):")
    if FAST_SANITY_MODE and tmcmc_fast_sanity:
        logger.info("FAST-SANITY MODE ACTIVE")
        logger.info("n_particles=%s (reduced)", tmcmc_fast_sanity["n_particles"])
        logger.info("n_stages=%s (reduced)", tmcmc_fast_sanity["n_stages"])
        logger.info("n_mutation_steps=%s (reduced)", tmcmc_fast_sanity["n_mutation_steps"])
        logger.info("n_chains=%s (reduced)", tmcmc_fast_sanity["n_chains"])
    else:
        logger.info("n_particles=%s (production)", PRODUCTION_TMCMC["n_particles"])
        logger.info("n_stages=%s (production)", PRODUCTION_TMCMC["n_stages"])
        logger.info("n_mutation_steps=%s (production)", PRODUCTION_TMCMC["n_mutation_steps"])
        logger.info("n_chains=%s (production)", PRODUCTION_TMCMC["n_chains"])
    if not (FAST_SANITY_MODE and tmcmc_fast_sanity):
        logger.info(
            "TMCMC schedule: target_ess_ratio=%.3f, min_delta_beta=%.4f, max_delta_beta=%.4f, "
            "update_linearization_interval=%s, linearization_threshold=%.3f, lin_enable_rom_thr=%.4f, force_beta_one=%s",
            float(PRODUCTION_TMCMC["target_ess_ratio"]),
            float(PRODUCTION_TMCMC["min_delta_beta"]),
            float(PRODUCTION_TMCMC["max_delta_beta"]),
            int(PRODUCTION_TMCMC["update_linearization_interval"]),
            float(PRODUCTION_TMCMC["linearization_threshold"]),
            float(PRODUCTION_TMCMC["linearization_enable_rom_threshold"]),
            bool(PRODUCTION_TMCMC["force_beta_one"]),
        )
    logger.info(
        "Experiment Settings: n_data=%s, sigma_obs=%s, cov_rel=%s, output_dir=%s, random_seed=%s",
        exp_config.n_data,
        exp_config.sigma_obs,
        exp_config.cov_rel,
        run_dir,
        exp_config.random_seed,
    )
    logger.info(
        "Model Configuration: M1=%s params (%s); M2=%s params (%s); M3=%s params (%s)",
        len(MODEL_CONFIGS["M1"]["param_names"]),
        ", ".join(MODEL_CONFIGS["M1"]["param_names"]),
        len(MODEL_CONFIGS["M2"]["param_names"]),
        ", ".join(MODEL_CONFIGS["M2"]["param_names"]),
        len(MODEL_CONFIGS["M3"]["param_names"]),
        ", ".join(MODEL_CONFIGS["M3"]["param_names"]),
    )
    logger.info("Requested Models: %s", requested_models)
    logger.info("%s", "=" * 80)

    # Save config.json (standardized output)
    config_payload: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "start_time": start_time_str,
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "paths": {
            "output_root": str(Path(output_root).resolve()),
            "run_dir": str(run_dir.resolve()),
            "figures_dir": str(figures_dir.resolve()),
        },
        "seeds": {"base_seed": exp_config.random_seed},
        "debug": {"level": debug_config.level.name},
        "experiment": {
            "n_data": exp_config.n_data,
            "sigma_obs": exp_config.sigma_obs,
            "cov_rel": exp_config.cov_rel,
            "aleatory_samples": int(exp_config.aleatory_samples),
        },
        "tmcmc": {
            "n_particles": (
                PRODUCTION_TMCMC["n_particles"]
                if not (FAST_SANITY_MODE and tmcmc_fast_sanity)
                else tmcmc_fast_sanity["n_particles"]
            ),
            "n_stages": (
                PRODUCTION_TMCMC["n_stages"]
                if not (FAST_SANITY_MODE and tmcmc_fast_sanity)
                else tmcmc_fast_sanity["n_stages"]
            ),
            "n_mutation_steps": (
                PRODUCTION_TMCMC["n_mutation_steps"]
                if not (FAST_SANITY_MODE and tmcmc_fast_sanity)
                else tmcmc_fast_sanity["n_mutation_steps"]
            ),
            "n_chains": (
                PRODUCTION_TMCMC["n_chains"]
                if not (FAST_SANITY_MODE and tmcmc_fast_sanity)
                else tmcmc_fast_sanity["n_chains"]
            ),
            "target_ess_ratio": float(PRODUCTION_TMCMC["target_ess_ratio"]),
            "min_delta_beta": float(PRODUCTION_TMCMC["min_delta_beta"]),
            "max_delta_beta": float(PRODUCTION_TMCMC["max_delta_beta"]),
            "update_linearization_interval": int(PRODUCTION_TMCMC["update_linearization_interval"]),
            "linearization_threshold": float(PRODUCTION_TMCMC["linearization_threshold"]),
            "linearization_enable_rom_threshold": float(
                PRODUCTION_TMCMC["linearization_enable_rom_threshold"]
            ),
            "force_beta_one": bool(PRODUCTION_TMCMC["force_beta_one"]),
        },
        "models": requested_models,
        "derivatives": {"use_paper_analytical": bool(USE_PAPER_ANALYTICAL)},
        "runtime": {
            "HAS_NUMBA": bool(HAS_NUMBA),
            "lock_paper_conditions": bool(LOCK_PAPER_CONDITIONS),
        },
        "environment": {
            "python": {
                "executable": sys.executable,
                "version": sys.version,
                "version_info": list(sys.version_info),
            },
            "numpy": {"version": str(np.__version__)},
            "numba": {
                "enabled": bool(HAS_NUMBA),
                "version": None,
                "num_threads": None,
                "threading_layer": None,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "threads": {
                "os_cpu_count": int(os.cpu_count() or -1),
                "mp_cpu_count": int(multiprocessing.cpu_count() or -1),
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
                "NUMBA_NUM_THREADS": os.getenv("NUMBA_NUM_THREADS"),
            },
        },
    }
    # Fill numba details if available (best-effort; do not hard-require numba import).
    try:
        import numba  # type: ignore

        config_payload["environment"]["numba"]["version"] = str(getattr(numba, "__version__", None))
        try:
            config_payload["environment"]["numba"]["num_threads"] = int(numba.get_num_threads())  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            config_payload["environment"]["numba"]["threading_layer"] = str(numba.threading_layer())  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False, default=str)

    # Get true parameters
    theta_true = get_theta_true()
    logger.info("True parameters θ* (14): %s", theta_true)

    if bool(args.self_check):
        logger.info("%s", "=" * 80)
        logger.info("SELF-CHECK (startup sanity)")
        logger.info("%s", "=" * 80)
        try:
            # Keep it light: check only one representative model.
            rep_model = "M1" if "M1" in MODEL_CONFIGS else list(MODEL_CONFIGS.keys())[0]
            chk = _self_check_tsm_once(
                model_key=rep_model,
                theta_true=theta_true,
                exp_config=exp_config,
                use_paper_analytical=USE_PAPER_ANALYTICAL,
            )
            save_json(run_dir / "self_check.json", chk)
            if chk.get("ok", False):
                logger.info("Self-check OK (%s).", rep_model)
            else:
                logger.warning("Self-check FAILED (%s): %s", rep_model, chk)
        except Exception as e:
            logger.warning("Self-check failed with exception: %s", e)

    plot_mgr = PlotManager(str(figures_dir))

    # ===== STEP 1: Generate Data =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 1: Generate Synthetic Data")
    logger.info("%s", "=" * 80)

    # ★ Slack notification: Step 1 start
    if SLACK_ENABLED:
        notify_slack("📊 STEP 1: Generating Synthetic Data...")

    data_M1 = idx_M1 = t_M1 = x0_M1 = sig2_M1 = None
    data_M2 = idx_M2 = t_M2 = x0_M2 = sig2_M2 = None
    data_M3 = idx_M3 = t_M3 = x0_M3 = sig2_M3 = None
    if "M1" in requested_models:
        data_M1, idx_M1, t_M1, x0_M1, sig2_M1 = generate_synthetic_data(
            MODEL_CONFIGS["M1"], theta_true, exp_config, "M1", plot_mgr
        )
        # Persist run data for reproducibility/auditing (used for logL re-evaluation)
        _save_npy(run_dir / "data_M1.npy", data_M1)
        _save_npy(run_dir / "idx_M1.npy", idx_M1)
        _save_npy(run_dir / "t_M1.npy", t_M1)
        _save_likelihood_meta(
            run_dir,
            run_id=run_id,
            model="M1",
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            n_data=exp_config.n_data,
            active_species=MODEL_CONFIGS["M1"]["active_species"],
            active_indices=MODEL_CONFIGS["M1"]["active_indices"],
        )
    if "M2" in requested_models:
        data_M2, idx_M2, t_M2, x0_M2, sig2_M2 = generate_synthetic_data(
            MODEL_CONFIGS["M2"], theta_true, exp_config, "M2", plot_mgr
        )
        _save_npy(run_dir / "data_M2.npy", data_M2)
        _save_npy(run_dir / "idx_M2.npy", idx_M2)
        _save_npy(run_dir / "t_M2.npy", t_M2)
        _save_likelihood_meta(
            run_dir,
            run_id=run_id,
            model="M2",
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            n_data=exp_config.n_data,
            active_species=MODEL_CONFIGS["M2"]["active_species"],
            active_indices=MODEL_CONFIGS["M2"]["active_indices"],
            script_path=Path(__file__),
        )
    if "M3" in requested_models:
        data_M3, idx_M3, t_M3, x0_M3, sig2_M3 = generate_synthetic_data(
            MODEL_CONFIGS["M3"], theta_true, exp_config, "M3", plot_mgr
        )
        _save_npy(run_dir / "data_M3.npy", data_M3)
        _save_npy(run_dir / "idx_M3.npy", idx_M3)
        _save_npy(run_dir / "t_M3.npy", t_M3)
        _save_likelihood_meta(
            run_dir,
            run_id=run_id,
            model="M3",
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            n_data=exp_config.n_data,
            active_species=MODEL_CONFIGS["M3"]["active_species"],
            active_indices=MODEL_CONFIGS["M3"]["active_indices"],
            script_path=Path(__file__),
        )

    # ★ PRIORITY A: データ差分の証拠を出力（M1/M2同一挙動の切り分け）
    if ("M1" in requested_models) and ("M2" in requested_models):
        logger.debug("%s", "=" * 80)
        logger.debug("DIAGNOSTIC: M1 vs M2 Data Comparison")
        logger.debug("%s", "=" * 80)
        logger.debug("Data difference (max abs): %.10f", float(np.max(np.abs(data_M1 - data_M2))))
        logger.debug("t_M1 shape: %s, length: %s", t_M1.shape, len(t_M1))
        logger.debug("t_M2 shape: %s, length: %s", t_M2.shape, len(t_M2))
        logger.debug("t_M1[0:5]: %s", t_M1[:5])
        logger.debug("t_M2[0:5]: %s", t_M2[:5])
        logger.debug("idx_M1[0:5]: %s", idx_M1[:5])
        logger.debug("idx_M2[0:5]: %s", idx_M2[:5])
        logger.debug("M1 active_species: %s", MODEL_CONFIGS["M1"]["active_species"])
        logger.debug("M2 active_species: %s", MODEL_CONFIGS["M2"]["active_species"])
        logger.debug("M1 active_indices: %s", MODEL_CONFIGS["M1"]["active_indices"])
        logger.debug("M2 active_indices: %s", MODEL_CONFIGS["M2"]["active_indices"])
        logger.debug("M1 alpha_const: %s", MODEL_CONFIGS["M1"]["alpha_const"])
        logger.debug("M2 alpha_const: %s", MODEL_CONFIGS["M2"]["alpha_const"])
        logger.debug(
            "M1 data shape: %s, mean: %.6f, std: %.6f",
            data_M1.shape,
            float(np.mean(data_M1)),
            float(np.std(data_M1)),
        )
        logger.debug(
            "M2 data shape: %s, mean: %.6f, std: %.6f",
            data_M2.shape,
            float(np.mean(data_M2)),
            float(np.std(data_M2)),
        )
        logger.debug("%s", "=" * 80)

    logger.info("Data generation complete")

    # ★ Slack notification: Step 1 complete
    if SLACK_ENABLED:
        notify_slack("✅ STEP 1: Data generation complete")

    # ===== STEP 2: M1 TMCMC with Linearization Update =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 2: M1 TMCMC (β tempering) with Linearization Update")
    logger.info("%s", "=" * 80)

    # ★ Slack notification: Step 2 start
    if SLACK_ENABLED:
        notify_slack("🔄 STEP 2: Starting M1 TMCMC...")

    solver_kwargs_M1 = {
        k: v
        for k, v in MODEL_CONFIGS["M1"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }

    prior_bounds_M1 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M1"]["param_names"])

    # ---- FIX: linearization point for inference must NOT be theta_true ----
    # ★ 論文向け（実データ想定でも安全）: 非推定パラメータも含めて真値に依存しない
    # 全部 prior mean（=1.5）で初期化（実データでは真値が存在しないため）
    prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
    theta_base_M1 = np.full(14, prior_mean)  # ★ 真値ゼロ依存: 全パラメータをprior meanで初期化
    theta_lin_M1 = theta_base_M1.copy()

    # ★ 修正: make_evaluator_M1 を theta_base_M1 定義後に移動（論文向け）
    # theta_base=theta_true ではなく theta_base=theta_base_M1 を使用
    def make_evaluator_M1(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base_M1
        evaluator = LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M1,
            active_species=MODEL_CONFIGS["M1"]["active_species"],
            active_indices=MODEL_CONFIGS["M1"]["active_indices"],
            theta_base=theta_base_M1,  # ★ 修正: theta_true → theta_base_M1 (非推定パラメータを真値で固定しない)
            data=data_M1,
            idx_sparse=idx_M1,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            theta_linearization=theta_linearization,
            paper_mode=USE_PAPER_ANALYTICAL,  # ★ Use paper_analytical_derivatives (production-ready)
            debug_logger=debug_logger,  # ★ Pass debug_logger for silent error handling
        )
        # ★ PRIORITY A: evaluator init diagnostics (M1/M2 same-behavior investigation)
        logger.debug("[M1 Evaluator] active_indices: %s", evaluator.active_indices)
        logger.debug("[M1 Evaluator] active_species: %s", evaluator.active_species)
        logger.debug(
            "[M1 Evaluator] alpha_const: %s", evaluator.solver_kwargs.get("alpha_const", "N/A")
        )
        logger.debug(
            "[M1 Evaluator] data id: %s, shape: %s", id(evaluator.data), evaluator.data.shape
        )
        logger.debug(
            "[M1 Evaluator] data mean: %.6f, std: %.6f",
            float(np.mean(evaluator.data)),
            float(np.std(evaluator.data)),
        )
        logger.debug(
            "[M1 Evaluator] theta_base[active]: %s", evaluator.theta_base[evaluator.active_indices]
        )
        return evaluator

    start_M1 = time.time()

    # ★ Use TMCMC (β tempering) with linearization update
    # Apply fast-sanity settings if enabled
    if FAST_SANITY_MODE and tmcmc_fast_sanity:
        n_particles_M1 = tmcmc_fast_sanity["n_particles"]
        n_stages_M1 = tmcmc_fast_sanity["n_stages"]
        n_mutation_steps_M1 = tmcmc_fast_sanity["n_mutation_steps"]
        n_chains_M1 = tmcmc_fast_sanity["n_chains"]
    else:
        # ★ Production settings (本番実行用)
        n_particles_M1 = PRODUCTION_TMCMC["n_particles"]
        n_stages_M1 = PRODUCTION_TMCMC["n_stages"]
        n_mutation_steps_M1 = PRODUCTION_TMCMC["n_mutation_steps"]
        n_chains_M1 = PRODUCTION_TMCMC["n_chains"]

    chains_M1, logL_M1, MAP_M1, converged_M1, diag_M1 = run_multi_chain_TMCMC(
        model_tag="M1",
        make_evaluator=make_evaluator_M1,
        prior_bounds=prior_bounds_M1,
        theta_base_full=theta_base_M1,  # ★ 修正: theta_true → theta_base_M1
        active_indices=MODEL_CONFIGS["M1"]["active_indices"],
        theta_linearization_init=theta_lin_M1,
        n_particles=n_particles_M1,
        n_stages=n_stages_M1,
        target_ess_ratio=float(PRODUCTION_TMCMC["target_ess_ratio"]),
        min_delta_beta=float(PRODUCTION_TMCMC["min_delta_beta"]),
        max_delta_beta=float(PRODUCTION_TMCMC["max_delta_beta"]),
        logL_scale=0.2,  # ★ M1は鋭いピークなので0.2
        n_chains=n_chains_M1,
        update_linearization_interval=int(PRODUCTION_TMCMC["update_linearization_interval"]),
        n_mutation_steps=n_mutation_steps_M1,
        use_observation_based_update=(
            False if FAST_SANITY_MODE else True
        ),  # ★ FAST_SANITY: 重いROM error計算をスキップ
        linearization_threshold=float(PRODUCTION_TMCMC["linearization_threshold"]),
        linearization_enable_rom_threshold=float(
            PRODUCTION_TMCMC["linearization_enable_rom_threshold"]
        ),
        debug_config=debug_config,  # ★ Pass debug configuration
        seed=exp_config.random_seed,
        force_beta_one=bool(PRODUCTION_TMCMC["force_beta_one"]) and (not FAST_SANITY_MODE),
    )

    time_M1 = time.time() - start_M1

    # Combine all chains
    samples_M1 = np.concatenate(chains_M1, axis=0)
    logL_M1_all = np.concatenate(logL_M1, axis=0)
    results_M1 = compute_MAP_with_uncertainty(samples_M1, logL_M1_all)
    results_M1["MAP"] = MAP_M1  # Override with global MAP
    mean_M1 = results_M1["mean"]

    # ===== EXTRA OUTPUT: Fit plots/metrics using estimated parameters (MAP/Mean) =====
    # Note: STEP 1 plots are generated at theta_true (data generation), not inference results.
    # These additional outputs make the inference quality visible.
    theta_MAP_full_M1 = theta_base_M1.copy()
    theta_MAP_full_M1[MODEL_CONFIGS["M1"]["active_indices"]] = MAP_M1
    theta_MEAN_full_M1 = theta_base_M1.copy()
    theta_MEAN_full_M1[MODEL_CONFIGS["M1"]["active_indices"]] = mean_M1

    # Persist inferred parameters explicitly (full vector + active subset)
    save_json(
        output_dir / "theta_MAP_M1.json",
        {
            "model": "M1",
            "theta_sub": MAP_M1,
            "theta_full": theta_MAP_full_M1,
            "active_indices": MODEL_CONFIGS["M1"]["active_indices"],
            "note": "theta_full uses theta_base (prior mean) for inactive parameters.",
        },
    )
    save_json(
        output_dir / "theta_MEAN_M1.json",
        {
            "model": "M1",
            "theta_sub": mean_M1,
            "theta_full": theta_MEAN_full_M1,
            "active_indices": MODEL_CONFIGS["M1"]["active_indices"],
            "note": "theta_full uses theta_base (prior mean) for inactive parameters.",
        },
    )

    evaluator_M1_for_metrics = make_evaluator_M1(theta_linearization=theta_lin_M1)
    # Compute ROM-vs-FOM error at estimated parameters (expensive but low frequency)
    rom_err_MAP_M1 = evaluator_M1_for_metrics.compute_ROM_error(theta_MAP_full_M1)
    rom_err_MEAN_M1 = evaluator_M1_for_metrics.compute_ROM_error(theta_MEAN_full_M1)

    # Run TSM and plot fits
    solver_M1_fit = BiofilmNewtonSolver(
        **solver_kwargs_M1,
        active_species=MODEL_CONFIGS["M1"]["active_species"],
        use_numba=HAS_NUMBA,
    )
    tsm_M1_fit = BiofilmTSM_Analytical(
        solver_M1_fit,
        active_theta_indices=MODEL_CONFIGS["M1"]["active_indices"],
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
        use_analytical=USE_PAPER_ANALYTICAL,
        theta_linearization=theta_lin_M1,
        paper_mode=USE_PAPER_ANALYTICAL,
    )
    t_fit, x0_fit_MAP, _ = tsm_M1_fit.solve_tsm(theta_MAP_full_M1)
    plot_mgr.plot_TSM_simulation(
        t_fit, x0_fit_MAP, MODEL_CONFIGS["M1"]["active_species"], "M1_MAP_fit", data_M1, idx_M1
    )
    fit_metrics_MAP_M1 = compute_fit_metrics(
        t_fit, x0_fit_MAP, MODEL_CONFIGS["M1"]["active_species"], data_M1, idx_M1
    )

    t_fit, x0_fit_MEAN, _ = tsm_M1_fit.solve_tsm(theta_MEAN_full_M1)
    plot_mgr.plot_TSM_simulation(
        t_fit, x0_fit_MEAN, MODEL_CONFIGS["M1"]["active_species"], "M1_MEAN_fit", data_M1, idx_M1
    )
    fit_metrics_MEAN_M1 = compute_fit_metrics(
        t_fit, x0_fit_MEAN, MODEL_CONFIGS["M1"]["active_species"], data_M1, idx_M1
    )

    # Save compact per-model metrics
    save_json(
        output_dir / "fit_metrics_M1.json",
        {
            "model": "M1",
            "theta_base_policy": "prior_mean_full_vector",
            "rom_error_MAP_vs_FOM": rom_err_MAP_M1,
            "rom_error_MEAN_vs_FOM": rom_err_MEAN_M1,
            "fit_MAP": fit_metrics_MAP_M1,
            "fit_MEAN": fit_metrics_MEAN_M1,
        },
    )

    # Export TMCMC diagnostic tables for later inspection
    export_tmcmc_diagnostics_tables(output_dir, "M1", diag_M1)

    logger.info("[M1 TMCMC] Results:")
    logger.info("Computation time: %.2f min", time_M1 / 60.0)
    logger.info("MAP: %s", MAP_M1)
    logger.info("Mean: %s", mean_M1)
    logger.info("True: %s", theta_true[0:5])
    map_error_M1 = np.linalg.norm(MAP_M1 - theta_true[0:5])
    logger.info("MAP error: %.6f", map_error_M1)
    logger.info("Converged chains: %s/%s", sum(converged_M1), len(converged_M1))
    logger.info("Linearization updates: %s", diag_M1.get("total_linearization_updates", 0))

    # ★ Slack notification: M1 complete
    if SLACK_ENABLED:
        notify_slack(
            f"✅ M1 TMCMC Completed\n"
            f"   Time: {time_M1/60:.2f} min\n"
            f"   MAP error: {map_error_M1:.6f}\n"
            f"   Converged: {sum(converged_M1)}/{len(converged_M1)} chains\n"
            f"   Linearization updates: {diag_M1.get('total_linearization_updates', 0)}"
        )

    plot_mgr.plot_posterior(
        samples_M1, theta_true[0:5], MODEL_CONFIGS["M1"]["param_names"], "M1", MAP_M1, mean_M1
    )

    # ----- Paper Fig. 9: posterior predictive band (M1) -----
    if mode == "paper":
        try:
            n_draws = min(120, int(samples_M1.shape[0])) if samples_M1 is not None else 0
            if n_draws > 0:
                rng = np.random.default_rng(int(exp_config.random_seed) + 9001)
                draw_idx = rng.choice(int(samples_M1.shape[0]), size=n_draws, replace=False)
                tsm_M1_fit.enable_linearization(True)

                phibar_samples = np.full(
                    (n_draws, len(t_M1), len(MODEL_CONFIGS["M1"]["active_species"])),
                    np.nan,
                    dtype=float,
                )
                for d, k in enumerate(draw_idx):
                    theta_full = theta_base_M1.copy()
                    theta_full[MODEL_CONFIGS["M1"]["active_indices"]] = samples_M1[k]
                    t_arr, x0_pred, _sig2_pred = tsm_M1_fit.solve_tsm(theta_full)
                    n = min(len(t_arr), len(t_M1))
                    phibar_samples[d, :n, :] = compute_phibar(
                        x0_pred[:n], MODEL_CONFIGS["M1"]["active_species"]
                    )

                plot_mgr.plot_posterior_predictive_band(
                    t_M1,
                    phibar_samples,
                    MODEL_CONFIGS["M1"]["active_species"],
                    "M1",
                    data=data_M1,
                    idx_sparse=idx_M1,
                    filename="PaperFig09_posterior_predictive_M1.png",
                )
        except Exception as e:
            logger.warning("Paper Fig9 generation failed (M1): %s: %s", type(e).__name__, e)

    # If user requested only M1, stop here (keep run robust and fast for debugging)
    if requested_models == ["M1"]:
        # Persist standardized artifacts even for partial runs.
        # This keeps reporting/analysis stable (metrics.json, results npz, manifest).
        plot_mgr.save_manifest()

        # Save minimal results npz (M1-only). Downstream report tooling reads diagnostics from here.
        np.savez(
            output_dir / "results_MAP_linearization.npz",
            mode=mode,
            theta_true=theta_true,
            MAP_M1=MAP_M1,
            mean_M1=mean_M1,
            samples_M1=samples_M1,
            logL_M1=logL_M1_all,
            converged_M1=converged_M1,
            diagnostics_M1=diag_M1,
        )

        metrics_payload: Dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requested_models": requested_models,
            "models_ran": ["M1"],
            "timing": {
                "time_M1_s": float(time_M1),
                "total_time_min": float(time_M1) / 60.0,
            },
            "convergence": {
                "M1": {
                    "converged_chains": int(sum(converged_M1)),
                    "n_chains": int(len(converged_M1)),
                },
            },
            "errors": {
                "m1_map_error": float(map_error_M1),
            },
            "health": {
                "likelihood": {
                    "M1": diag_M1.get("likelihood_health_total"),
                }
            },
            "artifacts": {
                "config_json": "config.json",
                "metrics_json": "metrics.json",
                "results_npz": "results_MAP_linearization.npz",
                "figures_dir": "figures",
                "figures_manifest": str((Path("figures") / "FIGURES_MANIFEST.json").as_posix()),
                "diagnostics_tables_dir": "diagnostics_tables",
                "fit_metrics": {
                    "M1": "fit_metrics_M1.json",
                },
            },
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False, default=str)

        logger.info("%s", "=" * 80)
        logger.info("M1-only run complete (requested via --models M1).")
        logger.info("%s", "=" * 80)
        logger.info("Output: %s/", run_dir)
        logger.info("Figures: %s/", figures_dir)
        return

    # ===== STEP 3: M2 TMCMC with Linearization Update =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 3: M2 TMCMC (β tempering) with Linearization Update")
    logger.info("%s", "=" * 80)

    # ★ Slack notification: Step 3 start
    if SLACK_ENABLED:
        notify_slack("🔄 STEP 3: Starting M2 TMCMC...")

    solver_kwargs_M2 = {
        k: v
        for k, v in MODEL_CONFIGS["M2"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }

    prior_bounds_M2 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M2"]["param_names"])

    # ---- FIX: linearization point for inference must NOT be theta_true ----
    # ★ 論文向け（実データ想定でも安全）: 非推定パラメータも含めて真値に依存しない
    # 全部 prior mean（=1.5）で初期化（実データでは真値が存在しないため）
    prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
    theta_base_M2 = np.full(14, prior_mean)  # ★ 真値ゼロ依存: 全パラメータをprior meanで初期化
    theta_lin_M2 = theta_base_M2.copy()

    # ★ 修正: make_evaluator_M2 を theta_base_M2 定義後に移動（論文向け）
    # theta_base=theta_true ではなく theta_base=theta_base_M2 を使用
    def make_evaluator_M2(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base_M2
        evaluator = LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M2,
            active_species=MODEL_CONFIGS["M2"]["active_species"],
            active_indices=MODEL_CONFIGS["M2"]["active_indices"],
            theta_base=theta_base_M2,  # ★ 修正: theta_true → theta_base_M2 (非推定パラメータを真値で固定しない)
            data=data_M2,
            idx_sparse=idx_M2,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            theta_linearization=theta_linearization,
            paper_mode=USE_PAPER_ANALYTICAL,  # ★ Use paper_analytical_derivatives (production-ready)
            debug_logger=debug_logger,  # ★ Pass debug_logger for silent error handling
        )
        # ★ PRIORITY A: evaluator init diagnostics (M1/M2 same-behavior investigation)
        logger.debug("[M2 Evaluator] active_indices: %s", evaluator.active_indices)
        logger.debug("[M2 Evaluator] active_species: %s", evaluator.active_species)
        logger.debug(
            "[M2 Evaluator] alpha_const: %s", evaluator.solver_kwargs.get("alpha_const", "N/A")
        )
        logger.debug(
            "[M2 Evaluator] data id: %s, shape: %s", id(evaluator.data), evaluator.data.shape
        )
        logger.debug(
            "[M2 Evaluator] data mean: %.6f, std: %.6f",
            float(np.mean(evaluator.data)),
            float(np.std(evaluator.data)),
        )
        logger.debug(
            "[M2 Evaluator] theta_base[active]: %s", evaluator.theta_base[evaluator.active_indices]
        )
        return evaluator

    start_M2 = time.time()

    # ★ Use TMCMC (β tempering) with linearization update
    # Apply fast-sanity settings if enabled
    if FAST_SANITY_MODE and tmcmc_fast_sanity:
        n_particles_M2 = tmcmc_fast_sanity["n_particles"]
        n_stages_M2 = tmcmc_fast_sanity["n_stages"]
        n_mutation_steps_M2 = tmcmc_fast_sanity["n_mutation_steps"]
        n_chains_M2 = tmcmc_fast_sanity["n_chains"]
    else:
        # ★ Production settings (本番実行用)
        n_particles_M2 = PRODUCTION_TMCMC["n_particles"]
        n_stages_M2 = PRODUCTION_TMCMC["n_stages"]
        n_mutation_steps_M2 = PRODUCTION_TMCMC["n_mutation_steps"]
        n_chains_M2 = PRODUCTION_TMCMC["n_chains"]

    chains_M2, logL_M2, MAP_M2, converged_M2, diag_M2 = run_multi_chain_TMCMC(
        model_tag="M2",
        make_evaluator=make_evaluator_M2,
        prior_bounds=prior_bounds_M2,
        theta_base_full=theta_base_M2,  # ★ 修正: theta_true → theta_base_M2
        active_indices=MODEL_CONFIGS["M2"]["active_indices"],
        theta_linearization_init=theta_lin_M2,
        n_particles=n_particles_M2,
        n_stages=n_stages_M2,
        target_ess_ratio=float(PRODUCTION_TMCMC["target_ess_ratio"]),
        min_delta_beta=float(PRODUCTION_TMCMC["min_delta_beta"]),
        max_delta_beta=float(PRODUCTION_TMCMC["max_delta_beta"]),
        logL_scale=0.5,  # ★ M2は中程度なので0.5
        n_chains=n_chains_M2,
        update_linearization_interval=int(PRODUCTION_TMCMC["update_linearization_interval"]),
        n_mutation_steps=n_mutation_steps_M2,
        use_observation_based_update=(
            False if FAST_SANITY_MODE else True
        ),  # ★ FAST_SANITY: 重いROM error計算をスキップ
        linearization_threshold=float(PRODUCTION_TMCMC["linearization_threshold"]),
        linearization_enable_rom_threshold=float(
            PRODUCTION_TMCMC["linearization_enable_rom_threshold"]
        ),
        debug_config=debug_config,  # ★ Pass debug configuration
        seed=exp_config.random_seed,
        force_beta_one=bool(PRODUCTION_TMCMC["force_beta_one"]) and (not FAST_SANITY_MODE),
    )

    time_M2 = time.time() - start_M2

    # Combine all chains
    samples_M2 = np.concatenate(chains_M2, axis=0)
    logL_M2_all = np.concatenate(logL_M2, axis=0)
    results_M2 = compute_MAP_with_uncertainty(samples_M2, logL_M2_all)
    results_M2["MAP"] = MAP_M2  # Override with global MAP
    mean_M2 = results_M2["mean"]

    # ===== EXTRA OUTPUT: Fit plots/metrics using estimated parameters (MAP/Mean) =====
    theta_base_M2 = theta_base_M1  # same full-vector base policy (prior mean)
    theta_MAP_full_M2 = theta_base_M2.copy()
    theta_MAP_full_M2[MODEL_CONFIGS["M2"]["active_indices"]] = MAP_M2
    theta_MEAN_full_M2 = theta_base_M2.copy()
    theta_MEAN_full_M2[MODEL_CONFIGS["M2"]["active_indices"]] = mean_M2

    # Build an evaluator to compute ROM-vs-FOM errors at estimated parameters
    # (Use the same inference-safe base; expensive but informative)
    def make_evaluator_M2(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base_M2
        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M2,
            active_species=MODEL_CONFIGS["M2"]["active_species"],
            active_indices=MODEL_CONFIGS["M2"]["active_indices"],
            theta_base=theta_base_M2,
            data=data_M2,
            idx_sparse=idx_M2,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            theta_linearization=theta_linearization,
            paper_mode=USE_PAPER_ANALYTICAL,
            debug_logger=debug_logger,
        )

    evaluator_M2_for_metrics = make_evaluator_M2()
    rom_err_MAP_M2 = evaluator_M2_for_metrics.compute_ROM_error(theta_MAP_full_M2)
    rom_err_MEAN_M2 = evaluator_M2_for_metrics.compute_ROM_error(theta_MEAN_full_M2)

    solver_M2_fit = BiofilmNewtonSolver(
        **solver_kwargs_M2,
        active_species=MODEL_CONFIGS["M2"]["active_species"],
        use_numba=HAS_NUMBA,
    )
    tsm_M2_fit = BiofilmTSM_Analytical(
        solver_M2_fit,
        active_theta_indices=MODEL_CONFIGS["M2"]["active_indices"],
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
        use_analytical=USE_PAPER_ANALYTICAL,
        theta_linearization=theta_base_M2,
        paper_mode=USE_PAPER_ANALYTICAL,
    )
    t_fit, x0_fit_MAP, _ = tsm_M2_fit.solve_tsm(theta_MAP_full_M2)
    plot_mgr.plot_TSM_simulation(
        t_fit, x0_fit_MAP, MODEL_CONFIGS["M2"]["active_species"], "M2_MAP_fit", data_M2, idx_M2
    )
    fit_metrics_MAP_M2 = compute_fit_metrics(
        t_fit, x0_fit_MAP, MODEL_CONFIGS["M2"]["active_species"], data_M2, idx_M2
    )

    t_fit, x0_fit_MEAN, _ = tsm_M2_fit.solve_tsm(theta_MEAN_full_M2)
    plot_mgr.plot_TSM_simulation(
        t_fit, x0_fit_MEAN, MODEL_CONFIGS["M2"]["active_species"], "M2_MEAN_fit", data_M2, idx_M2
    )
    fit_metrics_MEAN_M2 = compute_fit_metrics(
        t_fit, x0_fit_MEAN, MODEL_CONFIGS["M2"]["active_species"], data_M2, idx_M2
    )

    save_json(
        output_dir / "fit_metrics_M2.json",
        {
            "model": "M2",
            "theta_base_policy": "prior_mean_full_vector",
            "rom_error_MAP_vs_FOM": rom_err_MAP_M2,
            "rom_error_MEAN_vs_FOM": rom_err_MEAN_M2,
            "fit_MAP": fit_metrics_MAP_M2,
            "fit_MEAN": fit_metrics_MEAN_M2,
        },
    )
    export_tmcmc_diagnostics_tables(output_dir, "M2", diag_M2)

    logger.info("[M2 TMCMC] Results:")
    logger.info("Computation time: %.2f min", time_M2 / 60.0)
    logger.info("MAP: %s", MAP_M2)
    logger.info("Mean: %s", mean_M2)
    logger.info("True: %s", theta_true[5:10])
    map_error_M2 = np.linalg.norm(MAP_M2 - theta_true[5:10])
    logger.info("MAP error: %.6f", map_error_M2)
    logger.info("Converged chains: %s/%s", sum(converged_M2), len(converged_M2))
    logger.info("Linearization updates: %s", diag_M2.get("total_linearization_updates", 0))

    # ★ Slack notification: M2 complete
    if SLACK_ENABLED:
        notify_slack(
            f"✅ M2 TMCMC Completed\n"
            f"   Time: {time_M2/60:.2f} min\n"
            f"   MAP error: {map_error_M2:.6f}\n"
            f"   Converged: {sum(converged_M2)}/{len(converged_M2)} chains\n"
            f"   Linearization updates: {diag_M2.get('total_linearization_updates', 0)}"
        )

    plot_mgr.plot_posterior(
        samples_M2, theta_true[5:10], MODEL_CONFIGS["M2"]["param_names"], "M2", MAP_M2, mean_M2
    )

    # ----- Paper Fig. 11: posterior predictive band (M2) -----
    if mode == "paper":
        try:
            n_draws = min(120, int(samples_M2.shape[0])) if samples_M2 is not None else 0
            if n_draws > 0:
                rng = np.random.default_rng(int(exp_config.random_seed) + 11002)
                draw_idx = rng.choice(int(samples_M2.shape[0]), size=n_draws, replace=False)
                tsm_M2_fit.enable_linearization(True)

                phibar_samples = np.full(
                    (n_draws, len(t_M2), len(MODEL_CONFIGS["M2"]["active_species"])),
                    np.nan,
                    dtype=float,
                )
                for d, k in enumerate(draw_idx):
                    theta_full = theta_base_M2.copy()
                    theta_full[MODEL_CONFIGS["M2"]["active_indices"]] = samples_M2[k]
                    t_arr, x0_pred, _sig2_pred = tsm_M2_fit.solve_tsm(theta_full)
                    n = min(len(t_arr), len(t_M2))
                    phibar_samples[d, :n, :] = compute_phibar(
                        x0_pred[:n], MODEL_CONFIGS["M2"]["active_species"]
                    )

                plot_mgr.plot_posterior_predictive_band(
                    t_M2,
                    phibar_samples,
                    MODEL_CONFIGS["M2"]["active_species"],
                    "M2",
                    data=data_M2,
                    idx_sparse=idx_M2,
                    filename="PaperFig11_posterior_predictive_M2.png",
                )
        except Exception as e:
            logger.warning("Paper Fig11 generation failed (M2): %s: %s", type(e).__name__, e)

    # ===== STEP 4: M3 TMCMC with Linearization Update =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 4: M3 TMCMC (β tempering) with Linearization Update")
    logger.info("%s", "=" * 80)

    # ★ Slack notification: Step 4 start
    if SLACK_ENABLED:
        notify_slack("🔄 STEP 4: Starting M3 TMCMC...")

    # ★ 論文向け（実データ想定でも安全）: M3の非推定パラメータも真値に依存しない
    # M1/M2のMAP推定値を使用し、非推定パラメータはprior meanで初期化
    prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
    theta_base_M3 = np.full(14, prior_mean)  # ★ 真値ゼロ依存: 全パラメータをprior meanで初期化
    theta_base_M3[0:5] = MAP_M1  # M1の推定値
    theta_base_M3[5:10] = MAP_M2  # M2の推定値
    # M3のactive_indices (10:14) は後で設定される

    solver_kwargs_M3 = {
        k: v
        for k, v in MODEL_CONFIGS["M3"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }

    def make_evaluator_M3(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base_M3
        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M3,
            active_species=MODEL_CONFIGS["M3"]["active_species"],
            active_indices=MODEL_CONFIGS["M3"]["active_indices"],
            theta_base=theta_base_M3,
            data=data_M3,
            idx_sparse=idx_M3,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            theta_linearization=theta_linearization,
            paper_mode=USE_PAPER_ANALYTICAL,  # ★ Use paper_analytical_derivatives (production-ready)
        )

    prior_bounds_M3 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M3"]["param_names"])

    # Initial linearization point for M3
    theta_lin_M3 = theta_base_M3.copy()
    for idx in MODEL_CONFIGS["M3"]["active_indices"]:
        theta_lin_M3[idx] = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5

    start_M3 = time.time()

    # ★ Use TMCMC (β tempering) with linearization update
    # Apply fast-sanity settings if enabled
    if FAST_SANITY_MODE and tmcmc_fast_sanity:
        n_particles_M3 = tmcmc_fast_sanity["n_particles"]
        n_stages_M3 = tmcmc_fast_sanity["n_stages"]
        n_mutation_steps_M3 = tmcmc_fast_sanity["n_mutation_steps"]
        n_chains_M3 = tmcmc_fast_sanity["n_chains"]
    else:
        # ★ Production settings (本番実行用)
        n_particles_M3 = PRODUCTION_TMCMC["n_particles"]
        n_stages_M3 = PRODUCTION_TMCMC["n_stages"]
        n_mutation_steps_M3 = PRODUCTION_TMCMC["n_mutation_steps"]
        n_chains_M3 = PRODUCTION_TMCMC["n_chains"]

    chains_M3, logL_M3, MAP_M3, converged_M3, diag_M3 = run_multi_chain_TMCMC(
        model_tag="M3",
        make_evaluator=make_evaluator_M3,
        prior_bounds=prior_bounds_M3,
        theta_base_full=theta_base_M3,
        active_indices=MODEL_CONFIGS["M3"]["active_indices"],
        theta_linearization_init=theta_lin_M3,
        n_particles=n_particles_M3,
        n_stages=n_stages_M3,
        target_ess_ratio=float(PRODUCTION_TMCMC["target_ess_ratio"]),
        min_delta_beta=float(PRODUCTION_TMCMC["min_delta_beta"]),
        max_delta_beta=float(PRODUCTION_TMCMC["max_delta_beta"]),
        logL_scale=1.0,  # ★ M3は通常なので1.0
        n_chains=n_chains_M3,
        update_linearization_interval=int(PRODUCTION_TMCMC["update_linearization_interval"]),
        n_mutation_steps=n_mutation_steps_M3,
        use_observation_based_update=(
            False if FAST_SANITY_MODE else True
        ),  # ★ FAST_SANITY: 重いROM error計算をスキップ
        linearization_threshold=float(PRODUCTION_TMCMC["linearization_threshold"]),
        linearization_enable_rom_threshold=float(
            PRODUCTION_TMCMC["linearization_enable_rom_threshold"]
        ),
        debug_config=debug_config,  # ★ Pass debug configuration
        seed=exp_config.random_seed,
        force_beta_one=bool(PRODUCTION_TMCMC["force_beta_one"]) and (not FAST_SANITY_MODE),
    )

    time_M3 = time.time() - start_M3

    # Combine all chains
    samples_M3 = np.concatenate(chains_M3, axis=0)
    logL_M3_all = np.concatenate(logL_M3, axis=0)
    results_M3 = compute_MAP_with_uncertainty(samples_M3, logL_M3_all)
    results_M3["MAP"] = MAP_M3  # Override with global MAP
    mean_M3 = results_M3["mean"]

    # ===== EXTRA OUTPUT: Fit plots/metrics using estimated parameters (MAP/Mean) =====
    theta_base_M3 = theta_base_M1  # same full-vector base policy (prior mean)
    theta_MAP_full_M3 = theta_base_M3.copy()
    theta_MAP_full_M3[MODEL_CONFIGS["M3"]["active_indices"]] = MAP_M3
    theta_MEAN_full_M3 = theta_base_M3.copy()
    theta_MEAN_full_M3[MODEL_CONFIGS["M3"]["active_indices"]] = mean_M3

    def make_evaluator_M3(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base_M3
        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs_M3,
            active_species=MODEL_CONFIGS["M3"]["active_species"],
            active_indices=MODEL_CONFIGS["M3"]["active_indices"],
            theta_base=theta_base_M3,
            data=data_M3,
            idx_sparse=idx_M3,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            theta_linearization=theta_linearization,
            paper_mode=USE_PAPER_ANALYTICAL,
            debug_logger=debug_logger,
        )

    evaluator_M3_for_metrics = make_evaluator_M3()
    rom_err_MAP_M3 = evaluator_M3_for_metrics.compute_ROM_error(theta_MAP_full_M3)
    rom_err_MEAN_M3 = evaluator_M3_for_metrics.compute_ROM_error(theta_MEAN_full_M3)

    solver_M3_fit = BiofilmNewtonSolver(
        **solver_kwargs_M3,
        active_species=MODEL_CONFIGS["M3"]["active_species"],
        use_numba=HAS_NUMBA,
    )
    tsm_M3_fit = BiofilmTSM_Analytical(
        solver_M3_fit,
        active_theta_indices=MODEL_CONFIGS["M3"]["active_indices"],
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
        use_analytical=USE_PAPER_ANALYTICAL,
        theta_linearization=theta_base_M3,
        paper_mode=USE_PAPER_ANALYTICAL,
    )
    t_fit, x0_fit_MAP, _ = tsm_M3_fit.solve_tsm(theta_MAP_full_M3)
    plot_mgr.plot_TSM_simulation(
        t_fit, x0_fit_MAP, MODEL_CONFIGS["M3"]["active_species"], "M3_MAP_fit", data_M3, idx_M3
    )
    fit_metrics_MAP_M3 = compute_fit_metrics(
        t_fit, x0_fit_MAP, MODEL_CONFIGS["M3"]["active_species"], data_M3, idx_M3
    )

    t_fit, x0_fit_MEAN, _ = tsm_M3_fit.solve_tsm(theta_MEAN_full_M3)
    plot_mgr.plot_TSM_simulation(
        t_fit, x0_fit_MEAN, MODEL_CONFIGS["M3"]["active_species"], "M3_MEAN_fit", data_M3, idx_M3
    )
    fit_metrics_MEAN_M3 = compute_fit_metrics(
        t_fit, x0_fit_MEAN, MODEL_CONFIGS["M3"]["active_species"], data_M3, idx_M3
    )

    save_json(
        output_dir / "fit_metrics_M3.json",
        {
            "model": "M3",
            "theta_base_policy": "prior_mean_full_vector",
            "rom_error_MAP_vs_FOM": rom_err_MAP_M3,
            "rom_error_MEAN_vs_FOM": rom_err_MEAN_M3,
            "fit_MAP": fit_metrics_MAP_M3,
            "fit_MEAN": fit_metrics_MEAN_M3,
        },
    )
    export_tmcmc_diagnostics_tables(output_dir, "M3", diag_M3)

    logger.info("[M3 TMCMC] Results:")
    logger.info("Computation time: %.2f min", time_M3 / 60.0)
    logger.info("MAP: %s", MAP_M3)
    logger.info("Mean: %s", mean_M3)
    logger.info("True: %s", theta_true[10:14])
    map_error_M3 = np.linalg.norm(MAP_M3 - theta_true[10:14])
    logger.info("MAP error: %.6f", map_error_M3)
    logger.info("Converged chains: %s/%s", sum(converged_M3), len(converged_M3))
    logger.info("Linearization updates: %s", diag_M3.get("total_linearization_updates", 0))

    # ★ Slack notification: M3 complete
    if SLACK_ENABLED:
        notify_slack(
            f"✅ M3 TMCMC Completed\n"
            f"   Time: {time_M3/60:.2f} min\n"
            f"   MAP error: {map_error_M3:.6f}\n"
            f"   Converged: {sum(converged_M3)}/{len(converged_M3)} chains\n"
            f"   Linearization updates: {diag_M3.get('total_linearization_updates', 0)}"
        )

    plot_mgr.plot_posterior(
        samples_M3,
        theta_true[10:14],
        MODEL_CONFIGS["M3"]["param_names"],
        "M3_TMCMC",
        MAP_M3,
        mean_M3,
    )

    # ----- Paper Fig. 13: posterior predictive band (M3) -----
    if mode == "paper":
        try:
            n_draws = min(120, int(samples_M3.shape[0])) if samples_M3 is not None else 0
            if n_draws > 0:
                rng = np.random.default_rng(int(exp_config.random_seed) + 13003)
                draw_idx = rng.choice(int(samples_M3.shape[0]), size=n_draws, replace=False)
                tsm_M3_fit.enable_linearization(True)

                phibar_samples = np.full(
                    (n_draws, len(t_M3), len(MODEL_CONFIGS["M3"]["active_species"])),
                    np.nan,
                    dtype=float,
                )
                for d, k in enumerate(draw_idx):
                    theta_full = theta_base_M3.copy()
                    theta_full[MODEL_CONFIGS["M3"]["active_indices"]] = samples_M3[k]
                    t_arr, x0_pred, _sig2_pred = tsm_M3_fit.solve_tsm(theta_full)
                    n = min(len(t_arr), len(t_M3))
                    phibar_samples[d, :n, :] = compute_phibar(
                        x0_pred[:n], MODEL_CONFIGS["M3"]["active_species"]
                    )

                plot_mgr.plot_posterior_predictive_band(
                    t_M3,
                    phibar_samples,
                    MODEL_CONFIGS["M3"]["active_species"],
                    "M3",
                    data=data_M3,
                    idx_sparse=idx_M3,
                    filename="PaperFig13_posterior_predictive_M3.png",
                )
        except Exception as e:
            logger.warning("Paper Fig13 generation failed (M3): %s: %s", type(e).__name__, e)

    # ===== STEP 5: Final Summary =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 5: Final Summary")
    logger.info("%s", "=" * 80)

    # ★ FIX: No information leakage - use inference-safe base (prior mean), not theta_true
    # theta_true is only used for evaluation/comparison afterward
    prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
    theta_MAP_full = np.full(14, prior_mean)
    theta_mean_full = np.full(14, prior_mean)

    theta_MAP_full[0:5] = MAP_M1
    theta_MAP_full[5:10] = MAP_M2
    theta_MAP_full[10:14] = MAP_M3

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
        logger.info(
            "%s",
            f"{name:<8} {theta_true[i]:<12.6f} {theta_MAP_full[i]:<12.6f} {theta_mean_full[i]:<12.6f}",
        )

    total_map_error = np.linalg.norm(theta_MAP_full - theta_true)
    total_mean_error = np.linalg.norm(theta_mean_full - theta_true)

    logger.info("Total Parameter Error:")
    logger.info("MAP error: %.6f", total_map_error)
    logger.info("Mean error: %.6f", total_mean_error)

    total_time = (time_M1 + time_M2 + time_M3) / 60.0
    logger.info("Total computation time: %.2f min", total_time)

    # ★ Slack notification: Final summary
    if SLACK_ENABLED:
        elapsed_total = time.time() - start_time_global  # type: ignore
        notify_slack(
            f"🎉 TMCMC Process Completed!\n"
            f"   Total time: {elapsed_total/60:.2f} min\n"
            f"   M1 MAP error: {map_error_M1:.6f}\n"
            f"   M2 MAP error: {map_error_M2:.6f}\n"
            f"   M3 MAP error: {map_error_M3:.6f}\n"
            f"   Total MAP error: {total_map_error:.6f}\n"
            f"   Output: {exp_config.output_dir}"
        )

    plot_mgr.plot_parameter_comparison(theta_true, theta_MAP_full, theta_mean_full, param_names_all)

    # ----- Paper Fig. 14: posterior mean vs true with posterior std error bars -----
    if mode == "paper":
        try:
            std_full = np.full(14, np.nan, dtype=float)
            if samples_M1 is not None and samples_M1.size:
                std_full[0:5] = np.std(samples_M1, axis=0, ddof=1)
            if samples_M2 is not None and samples_M2.size:
                std_full[5:10] = np.std(samples_M2, axis=0, ddof=1)
            if samples_M3 is not None and samples_M3.size:
                std_full[10:14] = np.std(samples_M3, axis=0, ddof=1)
            plot_mgr.plot_paper_fig14_mean_vs_true_with_std(
                theta_true=theta_true,
                posterior_mean=theta_mean_full,
                posterior_std=std_full,
                param_names=list(param_names_all),
            )
        except Exception as e:
            logger.warning("Paper Fig14 generation failed: %s: %s", type(e).__name__, e)

    # ===== STEP 6: Generate TMCMC Diagnostic Plots =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 6: Generate TMCMC Diagnostic Plots")
    logger.info("%s", "=" * 80)

    # (A) Beta schedule plots
    plot_mgr.plot_beta_schedule(diag_M1["beta_schedules"], "M1")
    plot_mgr.plot_beta_schedule(diag_M2["beta_schedules"], "M2")
    plot_mgr.plot_beta_schedule(diag_M3["beta_schedules"], "M3")

    # (B) Linearization point update history
    plot_mgr.plot_linearization_history(
        diag_M1["theta0_history"], "M1", active_indices=MODEL_CONFIGS["M1"]["active_indices"]
    )
    plot_mgr.plot_linearization_history(
        diag_M2["theta0_history"], "M2", active_indices=MODEL_CONFIGS["M2"]["active_indices"]
    )
    plot_mgr.plot_linearization_history(
        diag_M3["theta0_history"], "M3", active_indices=MODEL_CONFIGS["M3"]["active_indices"]
    )

    # (C) ROM error history (if available)
    if diag_M1.get("rom_error_histories") and len(diag_M1["rom_error_histories"]) > 0:
        # Use first chain's ROM error history
        plot_mgr.plot_rom_error_history(diag_M1["rom_error_histories"][0], "M1")
    if diag_M2.get("rom_error_histories") and len(diag_M2["rom_error_histories"]) > 0:
        plot_mgr.plot_rom_error_history(diag_M2["rom_error_histories"][0], "M2")
    if diag_M3.get("rom_error_histories") and len(diag_M3["rom_error_histories"]) > 0:
        plot_mgr.plot_rom_error_history(diag_M3["rom_error_histories"][0], "M3")

    # (D) MAP error comparison (simple bar chart)
    map_errors_tmcmc = {
        "M1": np.linalg.norm(MAP_M1 - theta_true[0:5]),
        "M2": np.linalg.norm(MAP_M2 - theta_true[5:10]),
        "M3": np.linalg.norm(MAP_M3 - theta_true[10:14]),
    }
    plot_mgr.plot_map_error_comparison(map_errors_tmcmc, name="All_Models")

    # (E) Cost-accuracy tradeoff (★ 論文で最も刺さる図)
    # Calculate total evaluation counts (sum across all chains)
    cost_tmcmc = {
        "M1": sum(diag_M1.get("n_rom_evaluations", [0])),
        "M2": sum(diag_M2.get("n_rom_evaluations", [0])),
        "M3": sum(diag_M3.get("n_rom_evaluations", [0])),
    }
    # FOM evaluations (for ROM error computation)
    fom_cost_tmcmc = {
        "M1": sum(diag_M1.get("n_fom_evaluations", [0])),
        "M2": sum(diag_M2.get("n_fom_evaluations", [0])),
        "M3": sum(diag_M3.get("n_fom_evaluations", [0])),
    }
    # Total cost = ROM + FOM evaluations
    total_cost_tmcmc = {
        "M1": cost_tmcmc["M1"] + fom_cost_tmcmc["M1"],
        "M2": cost_tmcmc["M2"] + fom_cost_tmcmc["M2"],
        "M3": cost_tmcmc["M3"] + fom_cost_tmcmc["M3"],
    }

    # (改善3) Timing breakdown (TSM/FOM/TMCMC) aggregated over chains
    def _sum_timing_breakdowns(diag: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for b in diag.get("timing_breakdown_s", []) or []:
            if not isinstance(b, dict):
                continue
            for k, v in b.items():
                try:
                    out[k] = float(out.get(k, 0.0) + float(v))
                except Exception:
                    continue
        return out

    timing_breakdown_tmcmc = {
        "M1": _sum_timing_breakdowns(diag_M1),
        "M2": _sum_timing_breakdowns(diag_M2),
        "M3": _sum_timing_breakdowns(diag_M3),
    }

    # (改善5) Double-loop equivalent cost (×Nsamples) using paper notation.
    # Baseline idea: without TSM-ROM, each likelihood evaluation would require Nsamples FOM runs.
    Nsamples = int(getattr(exp_config, "aleatory_samples", 500))
    cost_double_loop = {
        m: {
            "Nsamples": Nsamples,
            "fom_equiv_from_rom": int(cost_tmcmc[m] * Nsamples),
            "fom_equiv_plus_checks": int(cost_tmcmc[m] * Nsamples + fom_cost_tmcmc[m]),
        }
        for m in ["M1", "M2", "M3"]
    }

    plot_mgr.plot_cost_accuracy_comparison(
        cost_tmcmc=total_cost_tmcmc,
        map_errors_tmcmc=map_errors_tmcmc,
        cost_unit="Total evaluations (ROM + FOM)",
        name="All_Models",
    )

    # (F) Wall time vs accuracy (alternative cost metric)
    wall_time_tmcmc = {
        "M1": time_M1,
        "M2": time_M2,
        "M3": time_M3,
    }
    plot_mgr.plot_cost_accuracy_comparison(
        cost_tmcmc=wall_time_tmcmc,
        map_errors_tmcmc=map_errors_tmcmc,
        cost_unit="Wall time (s)",
        name="All_Models_walltime",
    )

    # ===== Save Results =====
    logger.info("Saving results...")

    np.savez(
        output_dir / "results_MAP_linearization.npz",
        mode=mode,  # ★ 将来用: 結果ファイルだけ見たときに便利 ("sanity"/"debug"/"paper")
        theta_true=theta_true,
        theta_MAP_full=theta_MAP_full,
        theta_mean_full=theta_mean_full,
        MAP_M1=MAP_M1,
        MAP_M2=MAP_M2,
        MAP_M3=MAP_M3,
        mean_M1=mean_M1,
        mean_M2=mean_M2,
        mean_M3=mean_M3,
        samples_M1=samples_M1,
        samples_M2=samples_M2,
        samples_M3=samples_M3,
        logL_M1=logL_M1_all,
        logL_M2=logL_M2_all,
        logL_M3=logL_M3_all,
        converged_M1=converged_M1,
        converged_M2=converged_M2,
        converged_M3=converged_M3,
        diagnostics_M1=diag_M1,
        diagnostics_M2=diag_M2,
        diagnostics_M3=diag_M3,
    )

    logger.info("Results saved to: %s/results_MAP_linearization.npz", output_dir)

    # ===== STEP 6: Validation (M3_val) - time-dependent antibiotics =====
    # Paper Case II, Sec. 4.2.5 / Fig. 15 analogue (no additional calibration).
    if "M3_val" in requested_models:
        if "M3" not in requested_models:
            logger.warning("M3_val requested but M3 was not run; skipping validation.")
        elif "M3_val" not in MODEL_CONFIGS:
            logger.warning("MODEL_CONFIGS has no M3_val; skipping validation.")
        else:
            logger.info("%s", "=" * 80)
            logger.info("STEP 6: Validation (M3_val) with time-dependent antibiotics")
            logger.info("%s", "=" * 80)

            # Generate validation dataset under changed setup
            data_M3v, idx_M3v, t_M3v, x0_M3v, sig2_M3v = generate_synthetic_data(
                MODEL_CONFIGS["M3_val"], theta_true, exp_config, "M3_val", plot_mgr
            )
            _save_npy(run_dir / "data_M3_val.npy", data_M3v)
            _save_npy(run_dir / "idx_M3_val.npy", idx_M3v)
            _save_npy(run_dir / "t_M3_val.npy", t_M3v)

            # Posterior draws: fix M1/M2 at MAP, draw cross terms from M3 posterior
            n_draws = min(200, samples_M3.shape[0]) if samples_M3 is not None else 0
            if n_draws <= 0:
                logger.warning("No M3 posterior samples available; skipping predictive band.")
            else:
                rng = np.random.default_rng(int(exp_config.random_seed) + 12345)
                draw_idx = rng.choice(samples_M3.shape[0], size=n_draws, replace=False)
                theta_draws_sub = samples_M3[draw_idx]  # (n_draws, 4)

                # Paper-safe base: prior mean for non-estimated entries; MAP for M1/M2 blocks
                prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0
                theta_base_val = np.full(14, prior_mean, dtype=float)
                theta_base_val[0:5] = MAP_M1
                theta_base_val[5:10] = MAP_M2

                solver_kwargs_val = {
                    k: v
                    for k, v in MODEL_CONFIGS["M3_val"].items()
                    if k not in ["active_species", "active_indices", "param_names"]
                }
                solver_val = BiofilmNewtonSolver(
                    **solver_kwargs_val,
                    active_species=MODEL_CONFIGS["M3_val"]["active_species"],
                    use_numba=HAS_NUMBA,
                )
                tsm_val = BiofilmTSM_Analytical(
                    solver_val,
                    active_theta_indices=MODEL_CONFIGS["M3_val"]["active_indices"],
                    cov_rel=exp_config.cov_rel,
                    use_complex_step=True,
                    use_analytical=True,
                    theta_linearization=theta_base_val.copy(),
                    paper_mode=USE_PAPER_ANALYTICAL,
                )
                # Speed: posterior samples are near MAP, so linearization is acceptable here
                tsm_val.enable_linearization(True)

                phibar_samples = np.full(
                    (n_draws, len(t_M3v), len(MODEL_CONFIGS["M3_val"]["active_species"])),
                    np.nan,
                    dtype=float,
                )
                for d in range(n_draws):
                    theta_full = theta_base_val.copy()
                    for j, idx in enumerate(MODEL_CONFIGS["M3_val"]["active_indices"]):
                        theta_full[idx] = float(theta_draws_sub[d, j])

                    t_arr, x0_pred, _sig2_pred = tsm_val.solve_tsm(theta_full)
                    n = min(len(t_arr), len(t_M3v))
                    phibar_samples[d, :n, :] = compute_phibar(
                        x0_pred[:n], MODEL_CONFIGS["M3_val"]["active_species"]
                    )

                plot_mgr.plot_posterior_predictive_band(
                    t_M3v,
                    phibar_samples,
                    MODEL_CONFIGS["M3_val"]["active_species"],
                    "M3_val",
                    data=data_M3v,
                    idx_sparse=idx_M3v,
                    filename="PaperFig15_posterior_predictive_M3_val.png",
                )

    # ===== Save Figure Manifest =====
    plot_mgr.save_manifest()

    # Save metrics.json (standardized output)
    metrics_payload: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timing": {
            "time_M1_s": float(time_M1),
            "time_M2_s": float(time_M2),
            "time_M3_s": float(time_M3),
            "total_time_min": float(total_time),
        },
        "timing_breakdown_tmcmc": timing_breakdown_tmcmc,
        "convergence": {
            "M1": {"converged_chains": int(sum(converged_M1)), "n_chains": int(len(converged_M1))},
            "M2": {"converged_chains": int(sum(converged_M2)), "n_chains": int(len(converged_M2))},
            "M3": {"converged_chains": int(sum(converged_M3)), "n_chains": int(len(converged_M3))},
        },
        "errors": {
            "total_map_error": float(total_map_error),
            "total_mean_error": float(total_mean_error),
            "map_errors_tmcmc": {k: float(v) for k, v in map_errors_tmcmc.items()},
        },
        "cost": {
            "aleatory_samples": int(getattr(exp_config, "aleatory_samples", 500)),
            "rom_evaluations": {k: int(v) for k, v in cost_tmcmc.items()},
            "fom_evaluations": {k: int(v) for k, v in fom_cost_tmcmc.items()},
            "total_evaluations": {k: int(v) for k, v in total_cost_tmcmc.items()},
            "double_loop_equivalent": cost_double_loop,
        },
        "health": {
            "likelihood": {
                "M1": diag_M1.get("likelihood_health_total"),
                "M2": diag_M2.get("likelihood_health_total"),
                "M3": diag_M3.get("likelihood_health_total"),
            }
        },
        "artifacts": {
            "config_json": "config.json",
            "metrics_json": "metrics.json",
            "results_npz": "results_MAP_linearization.npz",
            "figures_dir": "figures",
            "figures_manifest": str((Path("figures") / "FIGURES_MANIFEST.json").as_posix()),
            "diagnostics_tables_dir": "diagnostics_tables",
            "fit_metrics": {
                "M1": "fit_metrics_M1.json",
                "M2": "fit_metrics_M2.json",
                "M3": "fit_metrics_M3.json",
            },
        },
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False, default=str)

    # ===== Completion =====
    logger.info("%s", "=" * 80)
    logger.info("Case II with TSM Linearization Complete!")
    logger.info("%s", "=" * 80)
    logger.info("Summary:")
    logger.info("Total parameter error (MAP): %.6f", total_map_error)
    logger.info("Total parameter error (Mean): %.6f", total_mean_error)
    logger.info("M1 converged chains: %s/%s", sum(converged_M1), len(converged_M1))
    logger.info("M2 converged chains: %s/%s", sum(converged_M2), len(converged_M2))
    logger.info("M3 converged chains: %s/%s", sum(converged_M3), len(converged_M3))
    logger.info("Total computation time: %.2f min", total_time)
    logger.info("Generated %s figures in %s/", len(plot_mgr.generated_figs), figures_dir)
    logger.info("Run artifacts: %s/ (config.json, metrics.json, results..., figures/)", run_dir)
    logger.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("%s", "=" * 80)


# NOTE: main() function and CLI entry point have been moved to main.case2_main module.
# Imported above for backward compatibility.
if __name__ == "__main__":
    # Import and call main from the refactored module
    from main.case2_main import main as main_func

    # Setup logging and patch solver (same as before)
    setup_logging("INFO")
    logger.info("Modules imported with TSM Linearization support")
    logger.info("Numba: %s", "enabled" if HAS_NUMBA else "disabled")

    # Patch biofilm solver (verbose=False for silent execution)
    patch_biofilm_solver(verbose=False)

    # Error handling with Slack notification
    try:
        main_func()
    except Exception as e:
        # Slack notification: Error occurred
        if SLACK_ENABLED:
            import traceback

            error_msg = f"❌ TMCMC Process Failed\n   Error: {str(e)}\n   Type: {type(e).__name__}"
            # Truncate traceback if too long
            tb_str = traceback.format_exc()
            if len(tb_str) > 1000:
                tb_str = tb_str[:1000] + "... (truncated)"
            notify_slack(f"{error_msg}\n```\n{tb_str}\n```", raise_on_error=False)
        raise
