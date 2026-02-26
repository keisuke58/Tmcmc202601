"""
Main entry point and CLI processing for Case II TMCMC with TSM Linearization.

Extracted from case2_tmcmc_linearization.py for better modularity.
Contains main() function, CLI argument parsing, and orchestration logic.
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import time
import sys
import json
import shlex
import os
import platform
import multiprocessing
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Add project root to sys.path to allow imports from config, core, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scipy.interpolate import interp1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import from refactored modules
from config import (
    DebugConfig,
    DebugLevel,
    MODEL_CONFIGS,
    PRIOR_BOUNDS_DEFAULT,
    TMCMC_DEFAULTS,
    setup_logging,
)

from utils import (
    code_crc32,
    save_npy,
    save_likelihood_meta,
    save_json,
)

from debug import (
    DebugLogger,
    SLACK_ENABLED,
    notify_slack,
)

from visualization import (
    PlotManager,
    compute_phibar,
    compute_fit_metrics,
    export_tmcmc_diagnostics_tables,
)

from core import (
    LogLikelihoodEvaluator,
    run_multi_chain_TMCMC,
)

from core.tmcmc import _stable_hash_int

# Backward compatibility aliases
_save_npy = save_npy
_save_likelihood_meta = save_likelihood_meta
_code_crc32 = code_crc32

# Import external dependencies
# Note: sys and Path are already imported above (lines 14, 25)
sys.path.insert(0, str(Path(__file__).parent.parent))
from improved1207_paper_jit import (
    BiofilmNewtonSolver,
    get_theta_true as get_theta_true_4s,
    HAS_NUMBA,
)
from improved_5species_jit import (
    BiofilmNewtonSolver5S,
    get_theta_true as get_theta_true_5s,
)
from tmcmc_5species_tsm import BiofilmTSM5S
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from bugfix_theta_to_matrices import patch_biofilm_solver

import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _default_output_root_for_mode(mode: str) -> str:
    """Get default output root directory for mode."""
    # Keep a single predictable root for Cursor "buttonization".
    # mode/seed are encoded in run_id, so analysis tools only need one root.
    _ = mode
    return str(Path("tmcmc") / "_runs")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
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
    p.add_argument(
        "--init-strategy",
        choices=["map", "mean"],
        default="map",
        help="Initialization strategy for skipped models (map or mean)",
    )

    # Experiment noise/uncertainty
    p.add_argument(
        "--sigma-obs", type=float, default=None, help="Override observation noise sigma_obs"
    )
    p.add_argument(
        "--no-noise",
        action="store_true",
        default=False,
        help="Generate data without noise (for training data)",
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
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for particle evaluation (None = auto, 1 = sequential)",
    )
    p.add_argument(
        "--use-threads",
        action="store_true",
        default=False,
        help="Use threads instead of processes for parallel evaluation",
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


def select_sparse_data_indices(
    n_total: int, n_obs: int, t_arr: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Select evenly spaced indices for sparse observations.

    Now selects indices based on normalized time [0.0, 1.0] divided into n_obs equal parts.
    This ensures data points are evenly distributed across the normalized time range.

    Parameters
    ----------
    n_total : int
        Total number of time steps
    n_obs : int
        Number of observations to select
    t_arr : Optional[np.ndarray]
        Time array. If provided, indices are selected to match normalized time positions exactly.
        If None, indices are calculated assuming uniform time spacing.

    Returns
    -------
    indices : np.ndarray
        Array of selected indices corresponding to normalized time positions [0.0, 1.0] divided into n_obs parts
    """
    # Divide normalized time [0.0, 1.0] with 0.05 interval
    # Data points: 0.05, 0.10, 0.15, ..., 0.95, 1.00 (0.05 interval, 20 points)
    # Graph will show with margin on both sides (left: space before 0.0, right: space after 1.0)
    if n_obs == 20:
        # Use 0.05 interval: 0.05, 0.10, 0.15, ..., 0.95, 1.00
        normalized_times = np.arange(0.05, 1.0 + 0.001, 0.05)
        # Ensure exactly n_obs points
        if len(normalized_times) > n_obs:
            normalized_times = normalized_times[:n_obs]
        elif len(normalized_times) < n_obs:
            # Fill if needed
            normalized_times = np.linspace(0.05, 1.0, n_obs)
    else:
        # For other n_obs values, use 0.05 interval pattern
        # Calculate how many 0.05 intervals fit
        n_intervals = n_obs - 1  # n_obs points means n_obs-1 intervals
        if n_intervals > 0:
            interval_size = 0.95 / n_intervals  # Use range from 0.05 to 1.00
            normalized_times = np.array(
                [0.05] + [0.05 + i * interval_size for i in range(1, n_intervals)] + [1.0]
            )
        else:
            normalized_times = np.array([0.05])

    if t_arr is not None:
        # Use actual time array to find indices closest to normalized time positions
        t_min = t_arr.min()
        t_max = t_arr.max()
        if t_max > t_min:
            t_normalized = (t_arr - t_min) / (t_max - t_min)
        else:
            t_normalized = np.zeros_like(t_arr)

        # Find indices closest to each normalized time position
        indices = np.zeros(n_obs, dtype=int)
        for i, t_norm in enumerate(normalized_times):
            # Find index with normalized time closest to t_norm
            distances = np.abs(t_normalized - t_norm)
            idx = np.argmin(distances)
            indices[i] = idx

            # Only ensure first and last indices are within bounds (don't force to 0 or n_total-1)
            # This allows margin to be applied
            if indices[i] < 0:
                indices[i] = 0
            elif indices[i] >= n_total:
                indices[i] = n_total - 1
    else:
        # Convert normalized time positions to indices (assuming uniform spacing)
        # normalized_time * (n_total - 1) gives the index position
        indices = np.round(normalized_times * (n_total - 1)).astype(int)

        # Ensure last index is exactly n_total - 1 (for normalized_time = 1.0)
        indices[-1] = n_total - 1

    # CRITICAL FIX: Check bounds explicitly instead of silent clipping
    # Silent clipping can hide bugs (e.g., n_total calculation errors)
    if np.any(indices < 0) or np.any(indices >= n_total):
        invalid_min = np.min(indices[indices < 0]) if np.any(indices < 0) else None
        invalid_max = np.max(indices[indices >= n_total]) if np.any(indices >= n_total) else None
        raise IndexError(
            f"Invalid indices generated: min={invalid_min}, max={invalid_max}, "
            f"valid range=[0, {n_total-1}]. This indicates a bug in index calculation."
        )

    return indices


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

    # Check if we need 5-species solver
    active_species = cfg["active_species"]
    is_5species = (len(active_species) == 5) or (max(active_species) >= 4)

    if is_5species:
        solver = BiofilmNewtonSolver5S(
            **solver_kwargs,
            active_species=active_species,
            use_numba=HAS_NUMBA,
        )
        # Use BiofilmTSM5S for 5-species model
        tsm = BiofilmTSM5S(
            solver,
            active_theta_indices=cfg["active_indices"],
            cov_rel=exp_config.cov_rel,
            theta_linearization=theta_true,
        )
    else:
        solver = BiofilmNewtonSolver(
            **solver_kwargs,
            active_species=active_species,
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
    debug: DebugConfig = None  # Debug configuration

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
    no_noise: bool = False  # If True, generate data without noise (for training data)
    # Paper notation: Nsamples (aleatory Monte Carlo samples) used in the *baseline* double-loop cost.
    # We keep this only for cost conversion/reporting; it does not affect the TSM-ROM execution.
    aleatory_samples: int = 500
    output_dir: str = None  # Auto-determined: sanity/debug/paper (set in main())
    random_seed: int = 42
    debug: DebugConfig = None  # Debug configuration

    def __post_init__(self):
        """Initialize debug config if not provided."""
        if self.debug is None:
            self.debug = DebugConfig(level=DebugLevel.OFF)


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

    # Check if we need 5-species solver
    active_species = config["active_species"]
    is_5species = (len(active_species) == 5) or (max(active_species) >= 4)

    if is_5species:
        solver = BiofilmNewtonSolver5S(
            **solver_kwargs,
            active_species=active_species,
            use_numba=HAS_NUMBA,
        )

        # Use BiofilmTSM5S for 5-species model
        tsm = BiofilmTSM5S(
            solver,
            active_theta_indices=config["active_indices"],
            cov_rel=exp_config.cov_rel,
            theta_linearization=theta_true,
        )
    else:
        solver = BiofilmNewtonSolver(
            **solver_kwargs,
            active_species=active_species,
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

    # CRITICAL FIX: Use t_arr to select indices based on normalized time positions
    # This ensures data points are evenly distributed across normalized time [0.0, 1.0]
    idx_sparse = select_sparse_data_indices(len(t_arr), exp_config.n_data, t_arr=t_arr)
    phibar = compute_phibar(x0, config["active_species"])

    # CRITICAL FIX: Use default_rng consistently
    rng = np.random.default_rng(exp_config.random_seed + (_stable_hash_int(name) % 1000))

    data = np.zeros((exp_config.n_data, len(config["active_species"])))
    for i, sp in enumerate(config["active_species"]):
        if exp_config.no_noise:
            # Generate data without noise (for training data)
            data[:, i] = phibar[idx_sparse, i]
            logger.info("[%s] Generating data WITHOUT noise (training data mode)", name)
        else:
            # Generate data with noise (default)
            data[:, i] = (
                phibar[idx_sparse, i]
                + rng.standard_normal(exp_config.n_data) * exp_config.sigma_obs
            )

    # CRITICAL FIX: Pass pre-computed phibar to ensure plot uses the same phibar as data generation
    plot_mgr.plot_TSM_simulation(
        t_arr, x0, config["active_species"], name, data, idx_sparse, phibar=phibar
    )

    logger.info(
        "Generated %s observations for %s species",
        exp_config.n_data,
        len(config["active_species"]),
    )

    return data, idx_sparse, t_arr, x0, sig2


# NOTE: main() function is added below (extracted from case2_tmcmc_linearization.py)
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

    # Set no_noise flag if specified
    if args.no_noise:
        exp_config.no_noise = True
        logger.info("Noise-free data generation enabled (training data mode)")
        # Auto-adjust sigma_obs for noise-free data if not explicitly set
        # User observation: 0.001 yields better accuracy than default 0.01 for noise-free cases
        if args.sigma_obs is None and not LOCK_PAPER_CONDITIONS:
            exp_config.sigma_obs = 0.0001
            logger.info(
                "Auto-tuning sigma_obs to 0.0001 for noise-free mode (improved accuracy setting)"
            )

    # Override sigma_obs and cov_rel if specified (CLI), unless paper conditions are locked.
    if LOCK_PAPER_CONDITIONS:
        if args.sigma_obs is not None and not math.isclose(
            float(args.sigma_obs), 0.001, rel_tol=0.0, abs_tol=1e-12
        ):
            logger.warning(
                "Ignoring --sigma-obs=%s due to paper-condition lock (sigma_obs=0.001).",
                args.sigma_obs,
            )
        if args.cov_rel is not None and not math.isclose(
            float(args.cov_rel), 0.005, rel_tol=0.0, abs_tol=1e-12
        ):
            logger.warning(
                "Ignoring --cov-rel=%s due to paper-condition lock (cov_rel=0.005).", args.cov_rel
            )
        exp_config.sigma_obs = 0.001
        exp_config.cov_rel = 0.005
    else:
        if args.sigma_obs is not None:
            exp_config.sigma_obs = float(args.sigma_obs)
            logger.warning("Overriding sigma_obs: %s (default: 0.0001)", exp_config.sigma_obs)
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
        if args.init_strategy == "mean":
            run_id += "_MEAN"
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
    theta_true_4s = get_theta_true_4s()
    theta_true_5s = get_theta_true_5s()
    logger.info("True parameters θ* (4s): %s", theta_true_4s)

    # Decide which theta to use for default self-check
    # Ideally we should use the one corresponding to the model we check

    if bool(args.self_check):
        logger.info("%s", "=" * 80)
        logger.info("SELF-CHECK (startup sanity)")
        logger.info("%s", "=" * 80)
        try:
            # Keep it light: check only one representative model.
            rep_model = "M1" if "M1" in requested_models else requested_models[0]
            if rep_model in ["M4", "M5"]:
                theta_for_check = theta_true_5s
            else:
                theta_for_check = theta_true_4s

            chk = _self_check_tsm_once(
                model_key=rep_model,
                theta_true=theta_for_check,
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
    data_M4 = idx_M4 = t_M4 = x0_M4 = sig2_M4 = None
    data_M5 = idx_M5 = t_M5 = x0_M5 = sig2_M5 = None

    # Helper for generating data
    def _gen_and_save(model_key, theta_t, var_suffix):
        d, idx, t, x0, sig2 = generate_synthetic_data(
            MODEL_CONFIGS[model_key], theta_t, exp_config, model_key, plot_mgr
        )
        _save_npy(run_dir / f"data_{var_suffix}.npy", d)
        _save_npy(run_dir / f"idx_{var_suffix}.npy", idx)
        _save_npy(run_dir / f"t_{var_suffix}.npy", t)
        _save_likelihood_meta(
            run_dir,
            run_id=run_id,
            model=model_key,
            sigma_obs=exp_config.sigma_obs,
            cov_rel=exp_config.cov_rel,
            rho=exp_config.rho,
            n_data=exp_config.n_data,
            active_species=MODEL_CONFIGS[model_key]["active_species"],
            active_indices=MODEL_CONFIGS[model_key]["active_indices"],
            script_path=Path(__file__),
        )
        return d, idx, t, x0, sig2

    if "M1" in requested_models:
        data_M1, idx_M1, t_M1, x0_M1, sig2_M1 = _gen_and_save("M1", theta_true_4s, "M1")
    if "M2" in requested_models:
        data_M2, idx_M2, t_M2, x0_M2, sig2_M2 = _gen_and_save("M2", theta_true_4s, "M2")
    if "M3" in requested_models:
        data_M3, idx_M3, t_M3, x0_M3, sig2_M3 = _gen_and_save("M3", theta_true_4s, "M3")
    if "M4" in requested_models:
        data_M4, idx_M4, t_M4, x0_M4, sig2_M4 = _gen_and_save("M4", theta_true_5s, "M4")
    if "M5" in requested_models:
        data_M5, idx_M5, t_M5, x0_M5, sig2_M5 = _gen_and_save("M5", theta_true_5s, "M5")

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

    # Get parallelization settings (Global)
    n_jobs = args.n_jobs if args.n_jobs is not None else None
    use_threads = args.use_threads

    # ===== STEP 2: M1 TMCMC with Linearization Update =====
    # Initialize theta_base (prior mean) for all models (extended to 20 for 5-species support)
    # ★ 論文向け（実データ想定でも安全）: 非推定パラメータも含めて真値に依存しない
    # 全部 prior mean（=1.5）で初期化（実データでは真値が存在しないため）
    prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
    theta_base_M1 = np.full(20, prior_mean)  # ★ 真値ゼロ依存: 全パラメータをprior meanで初期化

    if "M1" not in requested_models:
        logger.info("Skipping M1 TMCMC (not requested)")

        # Try to load previous results from the specific run directory
        prev_run_dir = Path(
            "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/parallel_fixed_M1M2M3_20260126_210657"
        )

        # Determine which file to load based on strategy
        if args.init_strategy == "mean":
            map_file_M1 = prev_run_dir / "theta_MEAN_M1.json"
            load_key = "theta_sub"  # Assuming structure is same
        else:
            map_file_M1 = prev_run_dir / "theta_MAP_M1.json"
            load_key = "theta_sub"

        if map_file_M1.exists():
            logger.info(f"Loading M1 {args.init_strategy.upper()} from {map_file_M1}")
            with open(map_file_M1, "r") as f:
                data_M1_loaded = json.load(f)
                MAP_M1 = np.array(data_M1_loaded[load_key])
        else:
            logger.warning(f"Could not find {map_file_M1}. Using NaNs for M1.")
            MAP_M1 = np.full(5, np.nan)

        samples_M1 = np.zeros((0, 5))
        logL_M1 = np.zeros(0)
        mean_M1 = np.full(5, np.nan)
        converged_M1 = []
        diag_M1 = {"beta_schedules": [], "theta0_history": []}
        time_M1 = 0.0
        map_error_M1 = 0.0
    else:
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

        # theta_base_M1 is already initialized globally above
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
                "[M1 Evaluator] theta_base[active]: %s",
                evaluator.theta_base[evaluator.active_indices],
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

        # Parallelization settings are defined globally

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
            n_jobs=n_jobs,
            use_threads=use_threads,
        )

        time_M1 = time.time() - start_M1

        # Combine all chains
        samples_M1 = np.concatenate(chains_M1, axis=0)
        logL_M1_all = np.concatenate(logL_M1, axis=0)
        results_M1 = compute_MAP_with_uncertainty(samples_M1, logL_M1_all)
        results_M1["MAP"] = MAP_M1  # Override with global MAP
        mean_M1 = results_M1["mean"]

        # Save posterior samples for spaghetti plot
        np.save(output_dir / "trace_M1.npy", samples_M1)

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
        # CRITICAL FIX: Use original t_M1 (from data generation) instead of t_fit to ensure
        # correct time mapping for observations. If time arrays differ, interpolate.
        if len(t_fit) != len(t_M1) or not np.allclose(t_fit, t_M1):
            if not HAS_SCIPY:
                raise ImportError("scipy is required for time array interpolation")
            x0_fit_MAP_interp = np.zeros((len(t_M1), x0_fit_MAP.shape[1]))
            for j in range(x0_fit_MAP.shape[1]):
                interp_func = interp1d(
                    t_fit,
                    x0_fit_MAP[:, j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                x0_fit_MAP_interp[:, j] = interp_func(t_M1)
            x0_fit_MAP = x0_fit_MAP_interp
        plot_mgr.plot_TSM_simulation(
            t_M1, x0_fit_MAP, MODEL_CONFIGS["M1"]["active_species"], "M1_MAP_fit", data_M1, idx_M1
        )
        fit_metrics_MAP_M1 = compute_fit_metrics(
            t_M1, x0_fit_MAP, MODEL_CONFIGS["M1"]["active_species"], data_M1, idx_M1
        )

        t_fit, x0_fit_MEAN, _ = tsm_M1_fit.solve_tsm(theta_MEAN_full_M1)
        # CRITICAL FIX: Use original t_M1 (from data generation) instead of t_fit
        if len(t_fit) != len(t_M1) or not np.allclose(t_fit, t_M1):
            if not HAS_SCIPY:
                raise ImportError("scipy is required for time array interpolation")
            x0_fit_MEAN_interp = np.zeros((len(t_M1), x0_fit_MEAN.shape[1]))
            for j in range(x0_fit_MEAN.shape[1]):
                interp_func = interp1d(
                    t_fit,
                    x0_fit_MEAN[:, j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                x0_fit_MEAN_interp[:, j] = interp_func(t_M1)
            x0_fit_MEAN = x0_fit_MEAN_interp
        plot_mgr.plot_TSM_simulation(
            t_M1, x0_fit_MEAN, MODEL_CONFIGS["M1"]["active_species"], "M1_MEAN_fit", data_M1, idx_M1
        )
        fit_metrics_MEAN_M1 = compute_fit_metrics(
            t_M1, x0_fit_MEAN, MODEL_CONFIGS["M1"]["active_species"], data_M1, idx_M1
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

        # Generate pairplot for M1
        plot_mgr.plot_pairplot_posterior(
            samples_M1, theta_true[0:5], MAP_M1, mean_M1, MODEL_CONFIGS["M1"]["param_names"], "M1"
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

                        # Interpolate if time arrays differ (critical for alignment with data)
                        if len(t_arr) != len(t_M1) or not np.allclose(t_arr, t_M1):
                            if HAS_SCIPY:
                                x0_pred_interp = np.zeros((len(t_M1), x0_pred.shape[1]))
                                for j in range(x0_pred.shape[1]):
                                    interp_func = interp1d(
                                        t_arr,
                                        x0_pred[:, j],
                                        kind="linear",
                                        bounds_error=False,
                                        fill_value="extrapolate",
                                    )
                                    x0_pred_interp[:, j] = interp_func(t_M1)
                                x0_pred = x0_pred_interp
                                t_arr = t_M1

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
                    # Spaghetti plot for detailed visualization
                    plot_mgr.plot_posterior_predictive_spaghetti(
                        t_M1,
                        phibar_samples,
                        MODEL_CONFIGS["M1"]["active_species"],
                        "M1",
                        data=data_M1,
                        idx_sparse=idx_M1,
                        filename="PaperFig09_spaghetti_M1.png",
                        use_paper_naming=False,
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
    if "M2" not in requested_models:
        logger.info("Skipping M2 TMCMC (not requested)")

        # Try to load previous results from the specific run directory
        prev_run_dir = Path(
            "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/parallel_fixed_M1M2M3_20260126_210657"
        )

        # Determine which file to load based on strategy
        if args.init_strategy == "mean":
            map_file_M2 = prev_run_dir / "theta_MEAN_M2.json"
            load_key = "theta_sub"
        else:
            map_file_M2 = prev_run_dir / "theta_MAP_M2.json"
            load_key = "theta_sub"

        if map_file_M2.exists():
            logger.info(f"Loading M2 {args.init_strategy.upper()} from {map_file_M2}")
            with open(map_file_M2, "r") as f:
                data_M2_loaded = json.load(f)
                MAP_M2 = np.array(data_M2_loaded[load_key])
        else:
            logger.warning(f"Could not find {map_file_M2}. Using NaNs for M2.")
            MAP_M2 = np.full(5, np.nan)

        samples_M2 = np.zeros((0, 5))
        logL_M2 = np.zeros(0)
        mean_M2 = np.full(5, np.nan)
        converged_M2 = []
        diag_M2 = {"beta_schedules": [], "theta0_history": []}
        time_M2 = 0.0
        map_error_M2 = 0.0
    else:
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
        theta_base_M2 = np.full(20, prior_mean)  # ★ 真値ゼロ依存: 全パラメータをprior meanで初期化
        # M2の減衰パラメータ(b3, b4)を0.5に設定 (数値安定性のため)
        theta_base_M2[8] = 0.5
        theta_base_M2[9] = 0.5

        if "M1" not in requested_models:
            # Note: If M1 was run, MAP_M1 is available. If not, it was loaded or NaN.
            # If available, we should ideally use it, but original code didn't.
            pass
        # For M2, inactive species 0,1 are locked to 0, so M1 params (interactions with 0,1)
        # technically don't affect M2 dynamics. So staying with prior_mean is fine.

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
                sigma_obs=0.001,  # User requested 0.001
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
                "[M2 Evaluator] theta_base[active]: %s",
                evaluator.theta_base[evaluator.active_indices],
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
            n_particles_M2 = 2000  # User requested
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
            n_jobs=n_jobs,
            use_threads=use_threads,
        )

        time_M2 = time.time() - start_M2

        # Combine all chains
        samples_M2 = np.concatenate(chains_M2, axis=0)
        logL_M2_all = np.concatenate(logL_M2, axis=0)
        results_M2 = compute_MAP_with_uncertainty(samples_M2, logL_M2_all)
        results_M2["MAP"] = MAP_M2  # Override with global MAP
        mean_M2 = results_M2["mean"]

        # Save posterior samples for spaghetti plot
        np.save(output_dir / "trace_M2.npy", samples_M2)

        # Persist inferred parameters explicitly (full vector + active subset)
        # This allows downstream tasks (e.g. M3) to load them robustly.
        save_json(
            output_dir / "theta_MAP_M2.json",
            {
                "model": "M2",
                "theta_sub": MAP_M2,
                "theta_full": theta_MAP_full_M2,
                "active_indices": MODEL_CONFIGS["M2"]["active_indices"],
                "note": "theta_full uses theta_base (prior mean) for inactive parameters.",
            },
        )
        save_json(
            output_dir / "theta_MEAN_M2.json",
            {
                "model": "M2",
                "theta_sub": mean_M2,
                "theta_full": theta_MEAN_full_M2,
                "active_indices": MODEL_CONFIGS["M2"]["active_indices"],
                "note": "theta_full uses theta_base (prior mean) for inactive parameters.",
            },
        )

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
                sigma_obs=0.001,
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
        # CRITICAL FIX: Use original t_M2 (from data generation) instead of t_fit to ensure
        # correct time mapping for observations. If time arrays differ, interpolate.
        if len(t_fit) != len(t_M2) or not np.allclose(t_fit, t_M2):
            if not HAS_SCIPY:
                raise ImportError("scipy is required for time array interpolation")
            x0_fit_MAP_interp = np.zeros((len(t_M2), x0_fit_MAP.shape[1]))
            for j in range(x0_fit_MAP.shape[1]):
                interp_func = interp1d(
                    t_fit,
                    x0_fit_MAP[:, j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                x0_fit_MAP_interp[:, j] = interp_func(t_M2)
            x0_fit_MAP = x0_fit_MAP_interp
        plot_mgr.plot_TSM_simulation(
            t_M2, x0_fit_MAP, MODEL_CONFIGS["M2"]["active_species"], "M2_MAP_fit", data_M2, idx_M2
        )
        fit_metrics_MAP_M2 = compute_fit_metrics(
            t_M2, x0_fit_MAP, MODEL_CONFIGS["M2"]["active_species"], data_M2, idx_M2
        )

        t_fit, x0_fit_MEAN, _ = tsm_M2_fit.solve_tsm(theta_MEAN_full_M2)
        # CRITICAL FIX: Use original t_M2 (from data generation) instead of t_fit
        if len(t_fit) != len(t_M2) or not np.allclose(t_fit, t_M2):
            if not HAS_SCIPY:
                raise ImportError("scipy is required for time array interpolation")
            x0_fit_MEAN_interp = np.zeros((len(t_M2), x0_fit_MEAN.shape[1]))
            for j in range(x0_fit_MEAN.shape[1]):
                interp_func = interp1d(
                    t_fit,
                    x0_fit_MEAN[:, j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                x0_fit_MEAN_interp[:, j] = interp_func(t_M2)
            x0_fit_MEAN = x0_fit_MEAN_interp
        plot_mgr.plot_TSM_simulation(
            t_M2, x0_fit_MEAN, MODEL_CONFIGS["M2"]["active_species"], "M2_MEAN_fit", data_M2, idx_M2
        )
        fit_metrics_MEAN_M2 = compute_fit_metrics(
            t_M2, x0_fit_MEAN, MODEL_CONFIGS["M2"]["active_species"], data_M2, idx_M2
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

        # Generate pairplot for M2
        plot_mgr.plot_pairplot_posterior(
            samples_M2, theta_true[5:10], MAP_M2, mean_M2, MODEL_CONFIGS["M2"]["param_names"], "M2"
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

                        # Interpolate if time arrays differ
                        if len(t_arr) != len(t_M2) or not np.allclose(t_arr, t_M2):
                            if HAS_SCIPY:
                                x0_pred_interp = np.zeros((len(t_M2), x0_pred.shape[1]))
                                for j in range(x0_pred.shape[1]):
                                    interp_func = interp1d(
                                        t_arr,
                                        x0_pred[:, j],
                                        kind="linear",
                                        bounds_error=False,
                                        fill_value="extrapolate",
                                    )
                                    x0_pred_interp[:, j] = interp_func(t_M2)
                                x0_pred = x0_pred_interp
                                t_arr = t_M2

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
                    # Spaghetti plot for detailed visualization
                    plot_mgr.plot_posterior_predictive_spaghetti(
                        t_M2,
                        phibar_samples,
                        MODEL_CONFIGS["M2"]["active_species"],
                        "M2",
                        data=data_M2,
                        idx_sparse=idx_M2,
                        filename="PaperFig11_spaghetti_M2.png",
                        use_paper_naming=False,
                    )
            except Exception as e:
                logger.warning("Paper Fig11 generation failed (M2): %s: %s", type(e).__name__, e)

    # ===== STEP 4: M3 TMCMC with Linearization Update =====
    if "M3" not in requested_models:
        logger.info("Skipping M3 TMCMC (not requested)")
        samples_M3 = np.zeros((0, 4))
        logL_M3 = np.zeros(0)
        MAP_M3 = np.full(4, np.nan)
        mean_M3 = np.full(4, np.nan)
        converged_M3 = []
        diag_M3 = {"beta_schedules": [], "theta0_history": []}
        time_M3 = 0.0
        map_error_M3 = 0.0
    else:
        logger.info("%s", "=" * 80)
        logger.info("STEP 4: M3 TMCMC (β tempering) with Linearization Update")
        logger.info("%s", "=" * 80)

        # ★ Slack notification: Step 4 start
        if SLACK_ENABLED:
            notify_slack("🔄 STEP 4: Starting M3 TMCMC...")

            # ★ 論文向け（実データ想定でも安全）: M3の非推定パラメータも真値に依存しない
        # M1/M2のMAP推定値を使用し、非推定パラメータはprior meanで初期化
        prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
        theta_base_M3 = np.full(20, prior_mean)  # ★ 真値ゼロ依存: 全パラメータをprior meanで初期化
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
            n_jobs=n_jobs,
            use_threads=use_threads,
        )

        time_M3 = time.time() - start_M3

        # Combine all chains
        samples_M3 = np.concatenate(chains_M3, axis=0)
        logL_M3_all = np.concatenate(logL_M3, axis=0)
        results_M3 = compute_MAP_with_uncertainty(samples_M3, logL_M3_all)
        results_M3["MAP"] = MAP_M3  # Override with global MAP
        mean_M3 = results_M3["mean"]

        # Save posterior samples for spaghetti plot
        np.save(output_dir / "trace_M3.npy", samples_M3)

        # Persist inferred parameters explicitly (full vector + active subset)
        save_json(
            output_dir / "theta_MAP_M3.json",
            {
                "model": "M3",
                "theta_sub": MAP_M3,
                "theta_full": theta_MAP_full_M3,
                "active_indices": MODEL_CONFIGS["M3"]["active_indices"],
                "note": "theta_full uses theta_base (prior mean) for inactive parameters.",
            },
        )
        save_json(
            output_dir / "theta_MEAN_M3.json",
            {
                "model": "M3",
                "theta_sub": mean_M3,
                "theta_full": theta_MEAN_full_M3,
                "active_indices": MODEL_CONFIGS["M3"]["active_indices"],
                "note": "theta_full uses theta_base (prior mean) for inactive parameters.",
            },
        )

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
        # CRITICAL FIX: Use original t_M3 (from data generation) instead of t_fit to ensure
        # correct time mapping for observations. If time arrays differ, interpolate.
        if len(t_fit) != len(t_M3) or not np.allclose(t_fit, t_M3):
            if not HAS_SCIPY:
                raise ImportError("scipy is required for time array interpolation")
            x0_fit_MAP_interp = np.zeros((len(t_M3), x0_fit_MAP.shape[1]))
            for j in range(x0_fit_MAP.shape[1]):
                interp_func = interp1d(
                    t_fit,
                    x0_fit_MAP[:, j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                x0_fit_MAP_interp[:, j] = interp_func(t_M3)
            x0_fit_MAP = x0_fit_MAP_interp
        plot_mgr.plot_TSM_simulation(
            t_M3, x0_fit_MAP, MODEL_CONFIGS["M3"]["active_species"], "M3_MAP_fit", data_M3, idx_M3
        )
        fit_metrics_MAP_M3 = compute_fit_metrics(
            t_M3, x0_fit_MAP, MODEL_CONFIGS["M3"]["active_species"], data_M3, idx_M3
        )

        t_fit, x0_fit_MEAN, _ = tsm_M3_fit.solve_tsm(theta_MEAN_full_M3)
        # CRITICAL FIX: Use original t_M3 (from data generation) instead of t_fit
        if len(t_fit) != len(t_M3) or not np.allclose(t_fit, t_M3):
            if not HAS_SCIPY:
                raise ImportError("scipy is required for time array interpolation")
            x0_fit_MEAN_interp = np.zeros((len(t_M3), x0_fit_MEAN.shape[1]))
            for j in range(x0_fit_MEAN.shape[1]):
                interp_func = interp1d(
                    t_fit,
                    x0_fit_MEAN[:, j],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                x0_fit_MEAN_interp[:, j] = interp_func(t_M3)
            x0_fit_MEAN = x0_fit_MEAN_interp
        plot_mgr.plot_TSM_simulation(
            t_M3, x0_fit_MEAN, MODEL_CONFIGS["M3"]["active_species"], "M3_MEAN_fit", data_M3, idx_M3
        )
        fit_metrics_MEAN_M3 = compute_fit_metrics(
            t_M3, x0_fit_MEAN, MODEL_CONFIGS["M3"]["active_species"], data_M3, idx_M3
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

        # Generate pairplot for M3
        plot_mgr.plot_pairplot_posterior(
            samples_M3, theta_true[10:14], MAP_M3, mean_M3, MODEL_CONFIGS["M3"]["param_names"], "M3"
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

                        # Interpolate if time arrays differ
                        if len(t_arr) != len(t_M3) or not np.allclose(t_arr, t_M3):
                            if HAS_SCIPY:
                                x0_pred_interp = np.zeros((len(t_M3), x0_pred.shape[1]))
                                for j in range(x0_pred.shape[1]):
                                    interp_func = interp1d(
                                        t_arr,
                                        x0_pred[:, j],
                                        kind="linear",
                                        bounds_error=False,
                                        fill_value="extrapolate",
                                    )
                                    x0_pred_interp[:, j] = interp_func(t_M3)
                                x0_pred = x0_pred_interp
                                t_arr = t_M3

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

    # ===== STEP 4b: M4 TMCMC =====
    if "M4" not in requested_models:
        logger.info("Skipping M4 TMCMC (not requested)")
        samples_M4 = np.zeros((0, 2))
        logL_M4 = np.zeros(0)
        MAP_M4 = np.full(2, np.nan)
        mean_M4 = np.full(2, np.nan)
        converged_M4 = []
        diag_M4 = {"beta_schedules": [], "theta0_history": []}
        time_M4 = 0.0
        map_error_M4 = 0.0
    else:
        logger.info("%s", "=" * 80)
        logger.info("STEP 4b: M4 TMCMC (5-Species: a55, b5)")
        logger.info("%s", "=" * 80)

        if SLACK_ENABLED:
            notify_slack("🔄 STEP 4b: Starting M4 TMCMC...")

        prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0
        theta_base_M4 = np.full(20, prior_mean)
        theta_base_M4[0:5] = MAP_M1
        theta_base_M4[5:10] = MAP_M2
        theta_base_M4[10:14] = MAP_M3

        solver_kwargs_M4 = {
            k: v
            for k, v in MODEL_CONFIGS["M4"].items()
            if k not in ["active_species", "active_indices", "param_names"]
        }

        def make_evaluator_M4(theta_linearization=None):
            if theta_linearization is None:
                theta_linearization = theta_base_M4
            return LogLikelihoodEvaluator(
                solver_kwargs=solver_kwargs_M4,
                active_species=MODEL_CONFIGS["M4"]["active_species"],
                active_indices=MODEL_CONFIGS["M4"]["active_indices"],
                theta_base=theta_base_M4,
                data=data_M4,
                idx_sparse=idx_M4,
                sigma_obs=exp_config.sigma_obs,
                cov_rel=exp_config.cov_rel,
                rho=exp_config.rho,
                theta_linearization=theta_linearization,
                paper_mode=False,
                debug_logger=debug_logger,
            )

        prior_bounds_M4 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M4"]["param_names"])
        theta_lin_M4 = theta_base_M4.copy()

        start_M4 = time.time()

        if FAST_SANITY_MODE and tmcmc_fast_sanity:
            n_particles_M4 = tmcmc_fast_sanity["n_particles"]
            n_stages_M4 = tmcmc_fast_sanity["n_stages"]
            n_mutation_steps_M4 = tmcmc_fast_sanity["n_mutation_steps"]
            n_chains_M4 = tmcmc_fast_sanity["n_chains"]
        else:
            n_particles_M4 = PRODUCTION_TMCMC["n_particles"]
            n_stages_M4 = PRODUCTION_TMCMC["n_stages"]
            n_mutation_steps_M4 = PRODUCTION_TMCMC["n_mutation_steps"]
            n_chains_M4 = PRODUCTION_TMCMC["n_chains"]

        chains_M4, logL_M4, MAP_M4, converged_M4, diag_M4 = run_multi_chain_TMCMC(
            model_tag="M4",
            make_evaluator=make_evaluator_M4,
            prior_bounds=prior_bounds_M4,
            theta_base_full=theta_base_M4,
            active_indices=MODEL_CONFIGS["M4"]["active_indices"],
            theta_linearization_init=theta_lin_M4,
            n_particles=n_particles_M4,
            n_stages=n_stages_M4,
            target_ess_ratio=float(PRODUCTION_TMCMC["target_ess_ratio"]),
            min_delta_beta=float(PRODUCTION_TMCMC["min_delta_beta"]),
            max_delta_beta=float(PRODUCTION_TMCMC["max_delta_beta"]),
            logL_scale=1.0,
            n_chains=n_chains_M4,
            update_linearization_interval=int(PRODUCTION_TMCMC["update_linearization_interval"]),
            n_mutation_steps=n_mutation_steps_M4,
            use_observation_based_update=False if FAST_SANITY_MODE else True,
            linearization_threshold=float(PRODUCTION_TMCMC["linearization_threshold"]),
            linearization_enable_rom_threshold=float(
                PRODUCTION_TMCMC["linearization_enable_rom_threshold"]
            ),
            debug_config=debug_config,
            seed=exp_config.random_seed,
            force_beta_one=bool(PRODUCTION_TMCMC["force_beta_one"]) and (not FAST_SANITY_MODE),
            n_jobs=n_jobs,
            use_threads=use_threads,
        )

        time_M4 = time.time() - start_M4
        samples_M4 = np.concatenate(chains_M4, axis=0)
        logL_M4_all = np.concatenate(logL_M4, axis=0)
        results_M4 = compute_MAP_with_uncertainty(samples_M4, logL_M4_all)
        results_M4["MAP"] = MAP_M4
        mean_M4 = results_M4["mean"]

        np.save(output_dir / "trace_M4.npy", samples_M4)

        # M4 Theta Full Construction
        theta_MAP_full_M4 = theta_base_M4.copy()
        theta_MAP_full_M4[MODEL_CONFIGS["M4"]["active_indices"]] = MAP_M4
        theta_MEAN_full_M4 = theta_base_M4.copy()
        theta_MEAN_full_M4[MODEL_CONFIGS["M4"]["active_indices"]] = mean_M4

        save_json(
            output_dir / "theta_MAP_M4.json",
            {
                "model": "M4",
                "theta_sub": MAP_M4,
                "theta_full": theta_MAP_full_M4,
                "active_indices": MODEL_CONFIGS["M4"]["active_indices"],
            },
        )
        save_json(
            output_dir / "theta_MEAN_M4.json",
            {
                "model": "M4",
                "theta_sub": mean_M4,
                "theta_full": theta_MEAN_full_M4,
                "active_indices": MODEL_CONFIGS["M4"]["active_indices"],
            },
        )

        # Diagnostics and Plots for M4
        evaluator_M4_for_metrics = make_evaluator_M4()
        rom_err_MAP_M4 = evaluator_M4_for_metrics.compute_ROM_error(theta_MAP_full_M4)
        rom_err_MEAN_M4 = evaluator_M4_for_metrics.compute_ROM_error(theta_MEAN_full_M4)

        solver_M4_fit = BiofilmNewtonSolver5S(
            **solver_kwargs_M4,
            active_species=MODEL_CONFIGS["M4"]["active_species"],
            use_numba=HAS_NUMBA,
        )
        tsm_M4_fit = BiofilmTSM5S(
            solver_M4_fit,
            active_theta_indices=MODEL_CONFIGS["M4"]["active_indices"],
            cov_rel=exp_config.cov_rel,
            theta_linearization=theta_base_M4,
        )

        t_fit, x0_fit_MAP, _ = tsm_M4_fit.solve_tsm(theta_MAP_full_M4)
        if len(t_fit) != len(t_M4) or not np.allclose(t_fit, t_M4):
            if HAS_SCIPY:
                x0_fit_MAP_interp = np.zeros((len(t_M4), x0_fit_MAP.shape[1]))
                for j in range(x0_fit_MAP.shape[1]):
                    interp_func = interp1d(
                        t_fit,
                        x0_fit_MAP[:, j],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    x0_fit_MAP_interp[:, j] = interp_func(t_M4)
                x0_fit_MAP = x0_fit_MAP_interp

        plot_mgr.plot_TSM_simulation(
            t_M4, x0_fit_MAP, MODEL_CONFIGS["M4"]["active_species"], "M4_MAP_fit", data_M4, idx_M4
        )
        fit_metrics_MAP_M4 = compute_fit_metrics(
            t_M4, x0_fit_MAP, MODEL_CONFIGS["M4"]["active_species"], data_M4, idx_M4
        )

        t_fit, x0_fit_MEAN, _ = tsm_M4_fit.solve_tsm(theta_MEAN_full_M4)
        if len(t_fit) != len(t_M4) or not np.allclose(t_fit, t_M4):
            if HAS_SCIPY:
                x0_fit_MEAN_interp = np.zeros((len(t_M4), x0_fit_MEAN.shape[1]))
                for j in range(x0_fit_MEAN.shape[1]):
                    interp_func = interp1d(
                        t_fit,
                        x0_fit_MEAN[:, j],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    x0_fit_MEAN_interp[:, j] = interp_func(t_M4)
                x0_fit_MEAN = x0_fit_MEAN_interp

        plot_mgr.plot_TSM_simulation(
            t_M4, x0_fit_MEAN, MODEL_CONFIGS["M4"]["active_species"], "M4_MEAN_fit", data_M4, idx_M4
        )
        fit_metrics_MEAN_M4 = compute_fit_metrics(
            t_M4, x0_fit_MEAN, MODEL_CONFIGS["M4"]["active_species"], data_M4, idx_M4
        )

        save_json(
            output_dir / "fit_metrics_M4.json",
            {
                "model": "M4",
                "rom_error_MAP": rom_err_MAP_M4,
                "rom_error_MEAN": rom_err_MEAN_M4,
                "fit_MAP": fit_metrics_MAP_M4,
                "fit_MEAN": fit_metrics_MEAN_M4,
            },
        )
        export_tmcmc_diagnostics_tables(output_dir, "M4", diag_M4)

        logger.info("[M4 TMCMC] Results:")
        logger.info("Computation time: %.2f min", time_M4 / 60.0)
        logger.info("MAP: %s", MAP_M4)
        logger.info(
            "True (14,15): %s", theta_true_5s[14:16] if "theta_true_5s" in globals() else "N/A"
        )
        map_error_M4 = np.linalg.norm(
            MAP_M4 - (theta_true_5s[14:16] if "theta_true_5s" in globals() else MAP_M4)
        )
        logger.info("MAP error: %.6f", map_error_M4)

        if SLACK_ENABLED:
            notify_slack(f"✅ M4 TMCMC Completed\nTime: {time_M4/60:.2f} min\nMAP: {MAP_M4}")

        plot_mgr.plot_posterior(
            samples_M4,
            theta_true_5s[14:16] if "theta_true_5s" in globals() else None,
            MODEL_CONFIGS["M4"]["param_names"],
            "M4_TMCMC",
            MAP_M4,
            mean_M4,
        )
        plot_mgr.plot_pairplot_posterior(
            samples_M4,
            theta_true_5s[14:16] if "theta_true_5s" in globals() else None,
            MAP_M4,
            mean_M4,
            MODEL_CONFIGS["M4"]["param_names"],
            "M4",
        )

    # ===== STEP 4c: M5 TMCMC =====
    if "M5" not in requested_models:
        logger.info("Skipping M5 TMCMC (not requested)")
        samples_M5 = np.zeros((0, 4))
        logL_M5 = np.zeros(0)
        MAP_M5 = np.full(4, np.nan)
        mean_M5 = np.full(4, np.nan)
        converged_M5 = []
        diag_M5 = {"beta_schedules": [], "theta0_history": []}
        time_M5 = 0.0
        map_error_M5 = 0.0
    else:
        logger.info("%s", "=" * 80)
        logger.info("STEP 4c: M5 TMCMC (5-Species: a15..a45)")
        logger.info("%s", "=" * 80)

        if SLACK_ENABLED:
            notify_slack("🔄 STEP 4c: Starting M5 TMCMC...")

        prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0
        theta_base_M5 = np.full(20, prior_mean)
        theta_base_M5[0:5] = MAP_M1
        theta_base_M5[5:10] = MAP_M2
        theta_base_M5[10:14] = MAP_M3
        theta_base_M5[14:16] = MAP_M4

        solver_kwargs_M5 = {
            k: v
            for k, v in MODEL_CONFIGS["M5"].items()
            if k not in ["active_species", "active_indices", "param_names"]
        }

        def make_evaluator_M5(theta_linearization=None):
            if theta_linearization is None:
                theta_linearization = theta_base_M5
            return LogLikelihoodEvaluator(
                solver_kwargs=solver_kwargs_M5,
                active_species=MODEL_CONFIGS["M5"]["active_species"],
                active_indices=MODEL_CONFIGS["M5"]["active_indices"],
                theta_base=theta_base_M5,
                data=data_M5,
                idx_sparse=idx_M5,
                sigma_obs=exp_config.sigma_obs,
                cov_rel=exp_config.cov_rel,
                rho=exp_config.rho,
                theta_linearization=theta_linearization,
                paper_mode=False,
                debug_logger=debug_logger,
            )

        prior_bounds_M5 = [PRIOR_BOUNDS_DEFAULT] * len(MODEL_CONFIGS["M5"]["param_names"])
        theta_lin_M5 = theta_base_M5.copy()

        start_M5 = time.time()

        if FAST_SANITY_MODE and tmcmc_fast_sanity:
            n_particles_M5 = tmcmc_fast_sanity["n_particles"]
            n_stages_M5 = tmcmc_fast_sanity["n_stages"]
            n_mutation_steps_M5 = tmcmc_fast_sanity["n_mutation_steps"]
            n_chains_M5 = tmcmc_fast_sanity["n_chains"]
        else:
            n_particles_M5 = PRODUCTION_TMCMC["n_particles"]
            n_stages_M5 = PRODUCTION_TMCMC["n_stages"]
            n_mutation_steps_M5 = PRODUCTION_TMCMC["n_mutation_steps"]
            n_chains_M5 = PRODUCTION_TMCMC["n_chains"]

        chains_M5, logL_M5, MAP_M5, converged_M5, diag_M5 = run_multi_chain_TMCMC(
            model_tag="M5",
            make_evaluator=make_evaluator_M5,
            prior_bounds=prior_bounds_M5,
            theta_base_full=theta_base_M5,
            active_indices=MODEL_CONFIGS["M5"]["active_indices"],
            theta_linearization_init=theta_lin_M5,
            n_particles=n_particles_M5,
            n_stages=n_stages_M5,
            target_ess_ratio=float(PRODUCTION_TMCMC["target_ess_ratio"]),
            min_delta_beta=float(PRODUCTION_TMCMC["min_delta_beta"]),
            max_delta_beta=float(PRODUCTION_TMCMC["max_delta_beta"]),
            logL_scale=1.0,
            n_chains=n_chains_M5,
            update_linearization_interval=int(PRODUCTION_TMCMC["update_linearization_interval"]),
            n_mutation_steps=n_mutation_steps_M5,
            use_observation_based_update=False if FAST_SANITY_MODE else True,
            linearization_threshold=float(PRODUCTION_TMCMC["linearization_threshold"]),
            linearization_enable_rom_threshold=float(
                PRODUCTION_TMCMC["linearization_enable_rom_threshold"]
            ),
            debug_config=debug_config,
            seed=exp_config.random_seed,
            force_beta_one=bool(PRODUCTION_TMCMC["force_beta_one"]) and (not FAST_SANITY_MODE),
            n_jobs=n_jobs,
            use_threads=use_threads,
        )

        time_M5 = time.time() - start_M5
        samples_M5 = np.concatenate(chains_M5, axis=0)
        logL_M5_all = np.concatenate(logL_M5, axis=0)
        results_M5 = compute_MAP_with_uncertainty(samples_M5, logL_M5_all)
        results_M5["MAP"] = MAP_M5
        mean_M5 = results_M5["mean"]

        np.save(output_dir / "trace_M5.npy", samples_M5)

        # M5 Theta Full Construction
        theta_MAP_full_M5 = theta_base_M5.copy()
        theta_MAP_full_M5[MODEL_CONFIGS["M5"]["active_indices"]] = MAP_M5
        theta_MEAN_full_M5 = theta_base_M5.copy()
        theta_MEAN_full_M5[MODEL_CONFIGS["M5"]["active_indices"]] = mean_M5

        save_json(
            output_dir / "theta_MAP_M5.json",
            {
                "model": "M5",
                "theta_sub": MAP_M5,
                "theta_full": theta_MAP_full_M5,
                "active_indices": MODEL_CONFIGS["M5"]["active_indices"],
            },
        )
        save_json(
            output_dir / "theta_MEAN_M5.json",
            {
                "model": "M5",
                "theta_sub": mean_M5,
                "theta_full": theta_MEAN_full_M5,
                "active_indices": MODEL_CONFIGS["M5"]["active_indices"],
            },
        )

        # Diagnostics and Plots for M5
        evaluator_M5_for_metrics = make_evaluator_M5()
        rom_err_MAP_M5 = evaluator_M5_for_metrics.compute_ROM_error(theta_MAP_full_M5)
        rom_err_MEAN_M5 = evaluator_M5_for_metrics.compute_ROM_error(theta_MEAN_full_M5)

        solver_M5_fit = BiofilmNewtonSolver5S(
            **solver_kwargs_M5,
            active_species=MODEL_CONFIGS["M5"]["active_species"],
            use_numba=HAS_NUMBA,
        )
        tsm_M5_fit = BiofilmTSM5S(
            solver_M5_fit,
            active_theta_indices=MODEL_CONFIGS["M5"]["active_indices"],
            cov_rel=exp_config.cov_rel,
            theta_linearization=theta_base_M5,
        )

        t_fit, x0_fit_MAP, _ = tsm_M5_fit.solve_tsm(theta_MAP_full_M5)
        if len(t_fit) != len(t_M5) or not np.allclose(t_fit, t_M5):
            if HAS_SCIPY:
                x0_fit_MAP_interp = np.zeros((len(t_M5), x0_fit_MAP.shape[1]))
                for j in range(x0_fit_MAP.shape[1]):
                    interp_func = interp1d(
                        t_fit,
                        x0_fit_MAP[:, j],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    x0_fit_MAP_interp[:, j] = interp_func(t_M5)
                x0_fit_MAP = x0_fit_MAP_interp

        plot_mgr.plot_TSM_simulation(
            t_M5, x0_fit_MAP, MODEL_CONFIGS["M5"]["active_species"], "M5_MAP_fit", data_M5, idx_M5
        )
        fit_metrics_MAP_M5 = compute_fit_metrics(
            t_M5, x0_fit_MAP, MODEL_CONFIGS["M5"]["active_species"], data_M5, idx_M5
        )

        t_fit, x0_fit_MEAN, _ = tsm_M5_fit.solve_tsm(theta_MEAN_full_M5)
        if len(t_fit) != len(t_M5) or not np.allclose(t_fit, t_M5):
            if HAS_SCIPY:
                x0_fit_MEAN_interp = np.zeros((len(t_M5), x0_fit_MEAN.shape[1]))
                for j in range(x0_fit_MEAN.shape[1]):
                    interp_func = interp1d(
                        t_fit,
                        x0_fit_MEAN[:, j],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    x0_fit_MEAN_interp[:, j] = interp_func(t_M5)
                x0_fit_MEAN = x0_fit_MEAN_interp

        plot_mgr.plot_TSM_simulation(
            t_M5, x0_fit_MEAN, MODEL_CONFIGS["M5"]["active_species"], "M5_MEAN_fit", data_M5, idx_M5
        )
        fit_metrics_MEAN_M5 = compute_fit_metrics(
            t_M5, x0_fit_MEAN, MODEL_CONFIGS["M5"]["active_species"], data_M5, idx_M5
        )

        save_json(
            output_dir / "fit_metrics_M5.json",
            {
                "model": "M5",
                "rom_error_MAP": rom_err_MAP_M5,
                "rom_error_MEAN": rom_err_MEAN_M5,
                "fit_MAP": fit_metrics_MAP_M5,
                "fit_MEAN": fit_metrics_MEAN_M5,
            },
        )
        export_tmcmc_diagnostics_tables(output_dir, "M5", diag_M5)

        logger.info("[M5 TMCMC] Results:")
        logger.info("Computation time: %.2f min", time_M5 / 60.0)
        logger.info("MAP: %s", MAP_M5)
        logger.info(
            "True (16..19): %s", theta_true_5s[16:20] if "theta_true_5s" in globals() else "N/A"
        )
        map_error_M5 = np.linalg.norm(
            MAP_M5 - (theta_true_5s[16:20] if "theta_true_5s" in globals() else MAP_M5)
        )
        logger.info("MAP error: %.6f", map_error_M5)

        if SLACK_ENABLED:
            notify_slack(f"✅ M5 TMCMC Completed\nTime: {time_M5/60:.2f} min\nMAP: {MAP_M5}")

        plot_mgr.plot_posterior(
            samples_M5,
            theta_true_5s[16:20] if "theta_true_5s" in globals() else None,
            MODEL_CONFIGS["M5"]["param_names"],
            "M5_TMCMC",
            MAP_M5,
            mean_M5,
        )
        plot_mgr.plot_pairplot_posterior(
            samples_M5,
            theta_true_5s[16:20] if "theta_true_5s" in globals() else None,
            MAP_M5,
            mean_M5,
            MODEL_CONFIGS["M5"]["param_names"],
            "M5",
        )

    # ===== STEP 5: Final Summary =====
    logger.info("%s", "=" * 80)
    logger.info("STEP 5: Final Summary (Updated)")
    logger.info("%s", "=" * 80)

    # ★ FIX: No information leakage - use inference-safe base (prior mean), not theta_true
    # theta_true is only used for evaluation/comparison afterward
    prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0  # 1.5
    theta_MAP_full = np.full(20, prior_mean)
    theta_mean_full = np.full(20, prior_mean)

    theta_MAP_full[0:5] = MAP_M1
    theta_MAP_full[5:10] = MAP_M2
    theta_MAP_full[10:14] = MAP_M3
    if "M4" in requested_models:
        theta_MAP_full[14:16] = MAP_M4
    if "M5" in requested_models:
        theta_MAP_full[16:20] = MAP_M5

    theta_mean_full[0:5] = mean_M1
    theta_mean_full[5:10] = mean_M2
    theta_mean_full[10:14] = mean_M3
    if "M4" in requested_models:
        theta_mean_full[14:16] = mean_M4
    if "M5" in requested_models:
        theta_mean_full[16:20] = mean_M5

    # Determine theta_true for comparison
    if "theta_true_5s" in globals():
        theta_true = theta_true_5s
    else:
        theta_true = np.full(20, np.nan)
        if "theta_true_4s" in globals():
            theta_true[0:14] = theta_true_4s

    param_names_all = (
        MODEL_CONFIGS["M1"]["param_names"]
        + MODEL_CONFIGS["M2"]["param_names"]
        + MODEL_CONFIGS["M3"]["param_names"]
        + MODEL_CONFIGS["M4"]["param_names"]
        + MODEL_CONFIGS["M5"]["param_names"]
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

    # Calculate errors only for valid indices (where theta_true is not nan)
    valid_indices = ~np.isnan(theta_true)
    total_map_error = np.linalg.norm((theta_MAP_full - theta_true)[valid_indices])
    total_mean_error = np.linalg.norm((theta_mean_full - theta_true)[valid_indices])

    logger.info("Total Parameter Error (valid indices):")
    logger.info("MAP error: %.6f", total_map_error)
    logger.info("Mean error: %.6f", total_mean_error)

    total_time = (time_M1 + time_M2 + time_M3 + time_M4 + time_M5) / 60.0
    logger.info("Total computation time: %.2f min", total_time)

    # ★ Slack notification: Final summary
    if SLACK_ENABLED:
        elapsed_total = time.time() - start_time_global  # type: ignore
        msg = (
            f"🎉 TMCMC Process Completed!\n"
            f"   Total time: {elapsed_total/60:.2f} min\n"
            f"   M1 MAP error: {map_error_M1:.6f}\n"
            f"   M2 MAP error: {map_error_M2:.6f}\n"
            f"   M3 MAP error: {map_error_M3:.6f}\n"
        )
        if "M4" in requested_models:
            msg += f"   M4 MAP error: {map_error_M4:.6f}\n"
        if "M5" in requested_models:
            msg += f"   M5 MAP error: {map_error_M5:.6f}\n"

        msg += f"   Total MAP error: {total_map_error:.6f}\n" f"   Output: {exp_config.output_dir}"
        notify_slack(msg)

    plot_mgr.plot_parameter_comparison(theta_true, theta_MAP_full, theta_mean_full, param_names_all)

    # ----- Paper Fig. 14: posterior mean vs true with posterior std error bars -----
    if mode == "paper":
        try:
            std_full = np.full(20, np.nan, dtype=float)
            if samples_M1 is not None and samples_M1.size:
                std_full[0:5] = np.std(samples_M1, axis=0, ddof=1)
            if samples_M2 is not None and samples_M2.size:
                std_full[5:10] = np.std(samples_M2, axis=0, ddof=1)
            if samples_M3 is not None and samples_M3.size:
                std_full[10:14] = np.std(samples_M3, axis=0, ddof=1)
            if samples_M4 is not None and samples_M4.size:
                std_full[14:16] = np.std(samples_M4, axis=0, ddof=1)
            if samples_M5 is not None and samples_M5.size:
                std_full[16:20] = np.std(samples_M5, axis=0, ddof=1)

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
    if "M4" in requested_models:
        plot_mgr.plot_beta_schedule(diag_M4["beta_schedules"], "M4")
    if "M5" in requested_models:
        plot_mgr.plot_beta_schedule(diag_M5["beta_schedules"], "M5")

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
    if "M4" in requested_models:
        plot_mgr.plot_linearization_history(
            diag_M4["theta0_history"], "M4", active_indices=MODEL_CONFIGS["M4"]["active_indices"]
        )
    if "M5" in requested_models:
        plot_mgr.plot_linearization_history(
            diag_M5["theta0_history"], "M5", active_indices=MODEL_CONFIGS["M5"]["active_indices"]
        )

    # (C) ROM error history (if available)
    if diag_M1.get("rom_error_histories") and len(diag_M1["rom_error_histories"]) > 0:
        # Use first chain's ROM error history
        plot_mgr.plot_rom_error_history(diag_M1["rom_error_histories"][0], "M1")
    if diag_M2.get("rom_error_histories") and len(diag_M2["rom_error_histories"]) > 0:
        plot_mgr.plot_rom_error_history(diag_M2["rom_error_histories"][0], "M2")
    if diag_M3.get("rom_error_histories") and len(diag_M3["rom_error_histories"]) > 0:
        plot_mgr.plot_rom_error_history(diag_M3["rom_error_histories"][0], "M3")
    if (
        "M4" in requested_models
        and diag_M4.get("rom_error_histories")
        and len(diag_M4["rom_error_histories"]) > 0
    ):
        plot_mgr.plot_rom_error_history(diag_M4["rom_error_histories"][0], "M4")
    if (
        "M5" in requested_models
        and diag_M5.get("rom_error_histories")
        and len(diag_M5["rom_error_histories"]) > 0
    ):
        plot_mgr.plot_rom_error_history(diag_M5["rom_error_histories"][0], "M5")

    # (D) MAP error comparison (simple bar chart)
    map_errors_tmcmc = {
        "M1": np.linalg.norm(MAP_M1 - theta_true[0:5]),
        "M2": np.linalg.norm(MAP_M2 - theta_true[5:10]),
        "M3": np.linalg.norm(MAP_M3 - theta_true[10:14]),
    }
    if "M4" in requested_models:
        map_errors_tmcmc["M4"] = np.linalg.norm(MAP_M4 - theta_true[14:16])
    if "M5" in requested_models:
        map_errors_tmcmc["M5"] = np.linalg.norm(MAP_M5 - theta_true[16:20])

    plot_mgr.plot_map_error_comparison(map_errors_tmcmc, name="All_Models")

    # (E) Cost-accuracy tradeoff (★ 論文で最も刺さる図)
    # Calculate total evaluation counts (sum across all chains)
    cost_tmcmc = {
        "M1": sum(diag_M1.get("n_rom_evaluations", [0])),
        "M2": sum(diag_M2.get("n_rom_evaluations", [0])),
        "M3": sum(diag_M3.get("n_rom_evaluations", [0])),
    }
    if "M4" in requested_models:
        cost_tmcmc["M4"] = sum(diag_M4.get("n_rom_evaluations", [0]))
    if "M5" in requested_models:
        cost_tmcmc["M5"] = sum(diag_M5.get("n_rom_evaluations", [0]))

    # FOM evaluations (for ROM error computation)
    fom_cost_tmcmc = {
        "M1": sum(diag_M1.get("n_fom_evaluations", [0])),
        "M2": sum(diag_M2.get("n_fom_evaluations", [0])),
        "M3": sum(diag_M3.get("n_fom_evaluations", [0])),
    }
    if "M4" in requested_models:
        fom_cost_tmcmc["M4"] = sum(diag_M4.get("n_fom_evaluations", [0]))
    if "M5" in requested_models:
        fom_cost_tmcmc["M5"] = sum(diag_M5.get("n_fom_evaluations", [0]))

    # Total cost = ROM + FOM evaluations
    total_cost_tmcmc = {
        "M1": cost_tmcmc["M1"] + fom_cost_tmcmc["M1"],
        "M2": cost_tmcmc["M2"] + fom_cost_tmcmc["M2"],
        "M3": cost_tmcmc["M3"] + fom_cost_tmcmc["M3"],
    }
    if "M4" in requested_models:
        total_cost_tmcmc["M4"] = cost_tmcmc["M4"] + fom_cost_tmcmc["M4"]
    if "M5" in requested_models:
        total_cost_tmcmc["M5"] = cost_tmcmc["M5"] + fom_cost_tmcmc["M5"]

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
    if "M4" in requested_models:
        timing_breakdown_tmcmc["M4"] = _sum_timing_breakdowns(diag_M4)
    if "M5" in requested_models:
        timing_breakdown_tmcmc["M5"] = _sum_timing_breakdowns(diag_M5)

    # (改善5) Double-loop equivalent cost (×Nsamples) using paper notation.
    # Baseline idea: without TSM-ROM, each likelihood evaluation would require Nsamples FOM runs.
    Nsamples = int(getattr(exp_config, "aleatory_samples", 500))

    active_models_for_cost = ["M1", "M2", "M3"]
    if "M4" in requested_models:
        active_models_for_cost.append("M4")
    if "M5" in requested_models:
        active_models_for_cost.append("M5")

    cost_double_loop = {
        m: {
            "Nsamples": Nsamples,
            "fom_equiv_from_rom": int(cost_tmcmc[m] * Nsamples),
            "fom_equiv_plus_checks": int(cost_tmcmc[m] * Nsamples + fom_cost_tmcmc[m]),
        }
        for m in active_models_for_cost
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
    if "M4" in requested_models:
        wall_time_tmcmc["M4"] = time_M4
    if "M5" in requested_models:
        wall_time_tmcmc["M5"] = time_M5

    plot_mgr.plot_cost_accuracy_comparison(
        cost_tmcmc=wall_time_tmcmc,
        map_errors_tmcmc=map_errors_tmcmc,
        cost_unit="Wall time (s)",
        name="All_Models_walltime",
    )

    # ===== Save Results =====
    logger.info("Saving results...")

    # Prepare dictionary for np.savez to handle conditional keys easily
    save_dict = {
        "mode": mode,
        "theta_true": theta_true,
        "theta_MAP_full": theta_MAP_full,
        "theta_mean_full": theta_mean_full,
        "MAP_M1": MAP_M1,
        "MAP_M2": MAP_M2,
        "MAP_M3": MAP_M3,
        "mean_M1": mean_M1,
        "mean_M2": mean_M2,
        "mean_M3": mean_M3,
        "samples_M1": samples_M1,
        "samples_M2": samples_M2,
        "samples_M3": samples_M3,
        "logL_M1": logL_M1_all,
        "logL_M2": logL_M2_all,
        "logL_M3": logL_M3_all,
        "converged_M1": converged_M1,
        "converged_M2": converged_M2,
        "converged_M3": converged_M3,
        "diagnostics_M1": diag_M1,
        "diagnostics_M2": diag_M2,
        "diagnostics_M3": diag_M3,
    }

    if "M4" in requested_models:
        save_dict.update(
            {
                "MAP_M4": MAP_M4,
                "mean_M4": mean_M4,
                "samples_M4": samples_M4,
                "logL_M4": logL_M4,
                "converged_M4": converged_M4,
                "diagnostics_M4": diag_M4,
            }
        )
    if "M5" in requested_models:
        save_dict.update(
            {
                "MAP_M5": MAP_M5,
                "mean_M5": mean_M5,
                "samples_M5": samples_M5,
                "logL_M5": logL_M5,
                "converged_M5": converged_M5,
                "diagnostics_M5": diag_M5,
            }
        )

    np.savez(output_dir / "results_MAP_linearization.npz", **save_dict)

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
                theta_base_val = np.full(20, prior_mean, dtype=float)
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
                # Spaghetti plot for detailed visualization
                plot_mgr.plot_posterior_predictive_spaghetti(
                    t_M3v,
                    phibar_samples,
                    MODEL_CONFIGS["M3_val"]["active_species"],
                    "M3_val",
                    data=data_M3v,
                    idx_sparse=idx_M3v,
                    filename="PaperFig15_spaghetti_M3_val.png",
                    use_paper_naming=False,
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
            "time_M4_s": float(time_M4),
            "time_M5_s": float(time_M5),
            "total_time_min": float(total_time),
        },
        "timing_breakdown_tmcmc": timing_breakdown_tmcmc,
        "convergence": {
            "M1": {"converged_chains": int(sum(converged_M1)), "n_chains": int(len(converged_M1))},
            "M2": {"converged_chains": int(sum(converged_M2)), "n_chains": int(len(converged_M2))},
            "M3": {"converged_chains": int(sum(converged_M3)), "n_chains": int(len(converged_M3))},
            "M4": {
                "converged_chains": int(sum(converged_M4)) if converged_M4 else 0,
                "n_chains": int(len(converged_M4)) if converged_M4 else 0,
            },
            "M5": {
                "converged_chains": int(sum(converged_M5)) if converged_M5 else 0,
                "n_chains": int(len(converged_M5)) if converged_M5 else 0,
            },
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
                "M4": diag_M4.get("likelihood_health_total") if "M4" in requested_models else None,
                "M5": diag_M5.get("likelihood_health_total") if "M5" in requested_models else None,
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
                "M4": "fit_metrics_M4.json",
                "M5": "fit_metrics_M5.json",
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
    if "M4" in requested_models:
        logger.info("M4 converged chains: %s/%s", sum(converged_M4), len(converged_M4))
    if "M5" in requested_models:
        logger.info("M5 converged chains: %s/%s", sum(converged_M5), len(converged_M5))
    logger.info("Total computation time: %.2f min", total_time)
    logger.info("Generated %s figures in %s/", len(plot_mgr.generated_figs), figures_dir)
    logger.info("Run artifacts: %s/ (config.json, metrics.json, results..., figures/)", run_dir)
    logger.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("%s", "=" * 80)


# Entry point for command-line execution
if __name__ == "__main__":
    # Setup logging and patch solver
    setup_logging("INFO")
    logger.info("Modules imported with TSM Linearization support")
    logger.info("Numba: %s", "enabled" if HAS_NUMBA else "disabled")

    # Patch biofilm solver (verbose=False for silent execution)
    patch_biofilm_solver(verbose=False)

    # Error handling with Slack notification
    try:
        main()
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
