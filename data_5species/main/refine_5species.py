#!/usr/bin/env python3
"""
Global Refinement Script for 5-Species Biofilm Model.

This script performs a "Global Relaxation" step after the stepwise estimation (M1->M5).
It unlocks ALL 20 parameters simultaneously, starting from the MAP estimate obtained
in the M5 stage, and refines them within a narrow prior bound (e.g., +/- 10-20%).

Usage:
    python tmcmc/main/refine_5species.py --run_id <RUN_ID> [options]

Strategy:
    1. Load M5 MAP estimate from a previous run.
    2. Define a tight Uniform prior around the MAP estimate.
    3. Run TMCMC on all 20 parameters (active_indices = 0..19).
    4. Save refined parameters and plots.
"""

import argparse
import sys
import json
import numpy as np
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tmcmc.config import (
    MODEL_CONFIGS,
    setup_logging,
    DebugConfig,
    DebugLevel,
)
from tmcmc.core import (
    LogLikelihoodEvaluator,
    run_multi_chain_TMCMC,
    compute_MAP_with_uncertainty,
)
from tmcmc.debug import DebugLogger
from tmcmc.utils import save_json
from tmcmc.visualization import PlotManager, compute_fit_metrics
from tmcmc.improved_5species_jit import (
    BiofilmNewtonSolver5S,
)

import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Global Refinement for 5-Species Model")
    parser.add_argument(
        "--run-id", type=str, required=True, help="Run ID of the completed M5 estimation"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for outputs (default: derived from config)",
    )
    parser.add_argument(
        "--refine-range",
        type=float,
        default=0.2,
        help="Relative range for refinement prior (default: 0.2 = +/- 20%%)",
    )
    parser.add_argument(
        "--n-particles", type=int, default=1000, help="Number of particles for refinement"
    )
    parser.add_argument("--n-stages", type=int, default=30, help="Number of TMCMC stages")
    parser.add_argument("--n-chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument(
        "--sigma-obs", type=float, default=0.001, help="Observation noise (sigma_obs)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug_level)

    # Determine paths
    if args.output_root:
        root_dir = Path(args.output_root)
    else:
        # Try to infer from default location
        root_dir = Path(__file__).parent.parent / "_runs"  # Default fallback

    run_dir = root_dir / args.run_id
    if not run_dir.exists():
        # Try checking if run_id is a full path
        if Path(args.run_id).exists():
            run_dir = Path(args.run_id)
        else:
            logger.error(f"Run directory not found: {run_dir}")
            sys.exit(1)

    logger.info(f"Refining Run: {run_dir}")

    # Load M5 MAP estimate
    theta_map_file = run_dir / "theta_MAP_M5.json"
    if not theta_map_file.exists():
        logger.error(f"M5 MAP file not found: {theta_map_file}")
        logger.error("Please ensure the M5 stage was completed in the specified run.")
        sys.exit(1)

    with open(theta_map_file, "r") as f:
        map_data = json.load(f)

    theta_map_full = np.array(map_data["theta_full"])
    logger.info("Loaded M5 MAP estimate (20 params).")

    # Define refinement bounds
    # Global Relaxation: Unlock ALL 20 parameters
    active_indices = list(range(20))
    active_species = [0, 1, 2, 3, 4]  # All 5 species

    # Construct Prior Bounds around MAP
    # lower = MAP * (1 - delta), upper = MAP * (1 + delta)
    # Clip to physical bounds (e.g. >= 0)
    delta = args.refine_range
    lower_bounds = theta_map_full * (1.0 - delta)
    upper_bounds = theta_map_full * (1.0 + delta)

    # Enforce hard constraints (non-negative)
    lower_bounds = np.maximum(lower_bounds, 0.0)
    # Optional: Enforce upper limits if known (e.g. max growth rate)
    upper_bounds = np.minimum(upper_bounds, 50.0)  # Reasonable upper bound for bio params

    prior_bounds = list(zip(lower_bounds, upper_bounds))

    logger.info(f"Refinement Range: +/- {delta*100:.1f}% around M5 MAP")

    # Load Data
    # We need the data used for M5.
    # It should be saved as data_M5.npy, idx_M5.npy, etc.
    try:
        data = np.load(run_dir / "data_M5.npy")
        idx_sparse = np.load(run_dir / "idx_M5.npy")
        t_arr = np.load(run_dir / "t_M5.npy")
    except FileNotFoundError as e:
        logger.error(f"Could not load M5 data: {e}")
        sys.exit(1)

    # Setup Evaluator
    # We use M5 configuration but override active_indices
    solver_kwargs = {
        k: v
        for k, v in MODEL_CONFIGS["M5"].items()
        if k not in ["active_species", "active_indices", "param_names"]
    }

    # Base theta is just the MAP (used for any inactive params, but here all are active)
    theta_base = theta_map_full.copy()

    debug_config = DebugConfig(level=DebugLevel[args.debug_level])
    debug_logger = DebugLogger(debug_config)

    def make_evaluator(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base

        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs,
            active_species=active_species,
            active_indices=active_indices,  # ALL 20 params
            theta_base=theta_base,
            data=data,
            idx_sparse=idx_sparse,
            sigma_obs=args.sigma_obs,
            cov_rel=0.005,  # Default relative covariance
            rho=0.0,
            theta_linearization=theta_linearization,
            paper_mode=False,  # Use standard differentiation if needed, or paper mode if preferred
            debug_logger=debug_logger,
        )

    # Run TMCMC
    logger.info("Starting Global Refinement TMCMC...")
    start_time = time.time()

    chains, logL, MAP, converged, diag = run_multi_chain_TMCMC(
        model_tag="Refine_M5",
        make_evaluator=make_evaluator,
        prior_bounds=prior_bounds,
        theta_base_full=theta_base,
        active_indices=active_indices,
        theta_linearization_init=theta_base,  # Linearize around MAP initially
        n_particles=args.n_particles,
        n_stages=args.n_stages,
        n_chains=args.n_chains,
        # Refinement specific settings
        min_delta_beta=0.01,  # Go slow
        max_delta_beta=0.1,
        logL_scale=1.0,
        seed=args.seed,
        debug_config=debug_config,
    )

    elapsed = time.time() - start_time
    logger.info(f"Refinement completed in {elapsed:.2f} seconds")

    # Process Results
    samples = np.concatenate(chains, axis=0)
    logL_all = np.concatenate(logL, axis=0)

    results = compute_MAP_with_uncertainty(samples, logL_all)
    refined_map = results["MAP"]
    refined_mean = results["mean"]

    # Update theta_full with refined values
    theta_map_refined_full = theta_base.copy()
    theta_map_refined_full[active_indices] = refined_map

    theta_mean_refined_full = theta_base.copy()
    theta_mean_refined_full[active_indices] = refined_mean

    # Save outputs
    output_subdir = run_dir / "refinement"
    output_subdir.mkdir(exist_ok=True)

    save_json(
        output_subdir / "theta_MAP_refined.json",
        {
            "model": "Refine_M5",
            "theta_sub": refined_map,
            "theta_full": theta_map_refined_full,
            "active_indices": active_indices,
            "base_run_id": args.run_id,
        },
    )

    np.save(output_subdir / "trace_refined.npy", samples)

    # Plotting
    plot_mgr = PlotManager(str(output_subdir))

    # Re-run simulation with refined parameters
    solver = BiofilmNewtonSolver5S(**solver_kwargs)

    # For plotting, we need to run the solver
    # BiofilmNewtonSolver5S.solve() returns (t, x, ...)
    # But PlotManager expects (t, x, active_species, ...)

    # Run solver at MAP
    t_fit, x_fit, _, _ = solver.solve(theta_map_refined_full)

    # Plot
    plot_mgr.plot_TSM_simulation(t_fit, x_fit, active_species, "Refined_MAP_Fit", data, idx_sparse)

    # Compute metrics
    metrics = compute_fit_metrics(t_fit, x_fit, active_species, data, idx_sparse)
    save_json(output_subdir / "metrics_refined.json", metrics)

    logger.info(f"Refinement results saved to {output_subdir}")


if __name__ == "__main__":
    main()
