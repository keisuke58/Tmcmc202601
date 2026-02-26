#!/usr/bin/env python3
"""
Global Refinement Script for Biofilm Models (2-Species, 4-Species, 5-Species).

This script performs a "Global Relaxation" step after the stepwise estimation.
It unlocks ALL relevant parameters simultaneously for the specified model type,
starting from the MAP estimate obtained in the previous stages.

Feature:
    - Uses Linearized TSM (ROM) by default for extreme speedup.
    - Can fallback to full model via --no-linearization.
    - Supports 2-species (M1), 4-species (M1-M3), and 5-species (M1-M5) models.

Usage:
    python tmcmc/main/refine_5species_linearized.py --run-id <RUN_ID> --model <2s|4s|5s> [options]
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
from tmcmc.visualization import PlotManager, compute_fit_metrics, compute_phibar
from tmcmc.improved_5species_jit import (
    BiofilmNewtonSolver5S,
    get_theta_true,
)

try:
    from scipy.interpolate import interp1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global Refinement for Biofilm Models (Linearized)"
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="Run ID of the completed estimation"
    )
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for outputs")
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
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of parallel jobs")
    parser.add_argument(
        "--debug-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--no-linearization",
        action="store_true",
        help="Disable linearization (use full model, slower)",
    )

    # ★ New Argument for Model Selection
    parser.add_argument(
        "--model",
        type=str,
        default="5s",
        choices=["2s", "4s", "5s"],
        help="Model type: 2s (Species 1-2), 4s (Species 1-4), 5s (Species 1-5)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["M1", "M2", "M3", "M4", "M5"],
        help="Specific stage to run (M1..M5). Overrides --model if set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="refine",
        choices=["refine", "stage"],
        help="'refine' = Unlock ALL params involved. 'stage' = Unlock ONLY stage-specific params (fix others).",
    )

    # ★ New Argument for Absolute Volume Model
    parser.add_argument(
        "--use-absolute-volume",
        action="store_true",
        help="Use Absolute Volume (phi * gamma) for likelihood instead of default (phi * psi).",
    )

    return parser.parse_args()


def get_model_config(model_type: str):
    """
    Returns (active_species, active_indices, target_stage_name) for the given model type.
    """
    if model_type == "2s":
        # 2 Species (S1, S2) -> Refine M1 params
        active_species = [0, 1]
        active_indices = list(range(5))  # a11, a12, a22, b1, b2
        stage_name = "M1"
    elif model_type == "4s":
        # 4 Species (S1-S4) -> Refine M1, M2, M3 params
        active_species = [0, 1, 2, 3]
        active_indices = list(range(14))  # M1(5) + M2(5) + M3(4) = 14 params
        stage_name = "M3"  # Usually 4-species run ends at M3 (or M3_val)
    elif model_type == "5s":
        # 5 Species (S1-S5) -> Refine M1-M5 params
        active_species = [0, 1, 2, 3, 4]
        active_indices = list(range(20))  # All 20 params
        stage_name = "M5"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return active_species, active_indices, stage_name


def main():
    args = parse_args()
    setup_logging(args.debug_level)

    # Determine paths
    if args.output_root:
        root_dir = Path(args.output_root)
    else:
        root_dir = Path(__file__).parent.parent / "_runs"

    run_dir = root_dir / args.run_id
    if not run_dir.exists():
        if Path(args.run_id).exists():
            run_dir = Path(args.run_id)
        else:
            logger.error(f"Run directory not found: {run_dir}")
            sys.exit(1)

    logger.info(f"Refining Run: {run_dir} (Model: {args.model})")

    # Get model configuration
    active_species, active_indices, stage_name = get_model_config(args.model)

    if args.stage:
        stage_name = args.stage
        if args.stage == "M1":
            active_species = [0, 1]
            # M1: a11, a12, a22, b1, b2 (indices 0-4)
            active_indices = [0, 1, 2, 3, 4]
        elif args.stage == "M2":
            active_species = [0, 1, 2, 3, 4]
            if args.mode == "stage":
                # M2 only: a33, a34, a44, b3, b4 (indices 5-9)
                active_indices = [5, 6, 7, 8, 9]
            else:
                # M1+M2
                active_indices = list(range(10))
        elif args.stage == "M3":
            active_species = [0, 1, 2, 3]
            if args.mode == "stage":
                # M3 only: a13, a14, a23, a24 (indices 10-13)
                active_indices = [10, 11, 12, 13]
            else:
                # M1+M2+M3
                active_indices = list(range(14))
        elif args.stage == "M4":
            active_species = [0, 1, 2, 3, 4]
            if args.mode == "stage":
                # M4 (a55, b5) + M5 (a15, a25, a35, a45) -> indices 14-19
                # Merged per user request for joint estimation stability
                active_indices = [14, 15, 16, 17, 18, 19]
            else:
                # M1+M2+M3+M4+M5
                active_indices = list(range(20))
        elif args.stage == "M5":
            active_species = [0, 1, 2, 3, 4]
            if args.mode == "stage":
                # M5 only: a15, a25, a35, a45 (indices 16-19)
                active_indices = [16, 17, 18, 19]
            else:
                # All
                active_indices = list(range(20))

    # Locate MAP file
    # Try generic 'theta_MAP.json' first, then specific stage 'theta_MAP_{stage}.json'
    map_files = [
        run_dir / f"theta_MAP_{stage_name}.json",
        run_dir / "theta_MAP.json",
        run_dir / "theta_MAP_refined.json",  # If re-refining
    ]

    theta_map_file = None
    for f in map_files:
        if f.exists():
            theta_map_file = f
            break

    if not theta_map_file:
        logger.error(f"MAP file not found. Checked: {[f.name for f in map_files]}")
        sys.exit(1)

    logger.info(f"Loading MAP from: {theta_map_file.name}")
    with open(theta_map_file, "r") as f:
        map_data = json.load(f)

    # Load and pad theta
    theta_loaded = np.array(map_data["theta_full"])

    # Ensure theta is 20D for the 5-species solver
    theta_base = np.zeros(20, dtype=np.float64)
    # Default values for un-estimated parameters (taken from improved_5species_jit.py defaults if needed)
    # Here we just copy what we have.
    n_loaded = len(theta_loaded)
    if n_loaded >= 20:
        theta_base[:] = theta_loaded[:20]
    else:
        theta_base[:n_loaded] = theta_loaded
        # Fill rest with defaults if they are zero (though they shouldn't affect the model if species are inactive)
        # But for safety, let's keep them 0 or set to some nominal value to avoid division by zero if accessed
        # M4(1.2, 0.25), M5(1.0...)
        if n_loaded < 14:  # Missing M3+
            theta_base[10:14] = [2.0, 1.0, 2.0, 1.0]  # M3 defaults
        if n_loaded < 16:  # Missing M4
            theta_base[14:16] = [1.2, 0.25]  # M4 defaults
        if n_loaded < 20:  # Missing M5
            theta_base[16:20] = [1.0, 1.0, 1.0, 1.0]  # M5 defaults

    logger.info(f"Theta Base Shape: {theta_base.shape} (Active Indices: {len(active_indices)})")

    # Refinement Bounds
    theta_target = theta_base[active_indices]
    delta = args.refine_range
    lower_bounds_sub = theta_target * (1.0 - delta)
    upper_bounds_sub = theta_target * (1.0 + delta)

    lower_bounds_sub = np.maximum(lower_bounds_sub, 0.0)
    upper_bounds_sub = np.minimum(upper_bounds_sub, 50.0)

    # Construct full 20D bounds list (needed for compatibility, though only active_indices are sampled)
    # Wait, run_multi_chain_TMCMC expects prior_bounds to match the dimension of *sampled* parameters (active_indices)?
    # No, usually it expects bounds for the active parameters.
    prior_bounds = list(zip(lower_bounds_sub, upper_bounds_sub))

    # Load Data
    # Try to find data file corresponding to the stage
    data_files = [
        run_dir / f"data_{stage_name}.npy",
        run_dir / "data.npy",
    ]
    data_file = None
    for f in data_files:
        if f.exists():
            data_file = f
            break

    if not data_file:
        logger.error(f"Data file not found. Checked: {[f.name for f in data_files]}")
        sys.exit(1)

    data = np.load(data_file)

    # Load Index (idx_sparse)
    idx_files = [run_dir / f"idx_{stage_name}.npy", run_dir / "idx.npy", run_dir / "t_idx.npy"]
    idx_file = None
    for f in idx_files:
        if f.exists():
            idx_file = f
            break

    if not idx_file:
        logger.error(f"Index file not found. Checked: {[f.name for f in idx_files]}")
        sys.exit(1)

    idx_sparse = np.load(idx_file)

    # Setup Evaluator
    # Use M5 config base but override active params
    # Note: For 2s/4s, we should technically use M1/M3 configs for dt/maxtimestep if they differ?
    # M1: dt=1e-5, max=2500
    # M2: dt=1e-5, max=5000
    # M3: dt=1e-4, max=750
    # M5: likely similar to M3 or M2?

    # Let's try to load config from run if possible, or infer from args.model
    # For now, we use a hybrid approach:
    # If 2s (M1) -> Use M1 config
    # If 4s (M3) -> Use M3 config
    # If 5s (M5) -> Use M5 config

    config_name = stage_name
    if config_name not in MODEL_CONFIGS:
        # Fallback to M5 if M5 not in config (it might be custom)
        # Using M2 defaults as safe fallback for 5-species often
        config_name = "M2"

    base_config = MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["M2"])

    solver_kwargs = {
        k: v
        for k, v in base_config.items()
        if k not in ["active_species", "active_indices", "param_names"]
    }

    logger.info(f"Using solver config from {config_name}: {solver_kwargs}")

    # Map string log level to DebugLevel enum
    debug_level_map = {
        "DEBUG": DebugLevel.VERBOSE,
        "INFO": DebugLevel.MINIMAL,
        "WARNING": DebugLevel.ERROR,
        "ERROR": DebugLevel.ERROR,
    }
    debug_config = DebugConfig(level=debug_level_map.get(args.debug_level, DebugLevel.MINIMAL))
    debug_logger = DebugLogger(debug_config)

    use_linearization = not args.no_linearization

    # Slice data for evaluator to match active_species
    data_for_eval = data
    if data.shape[1] > len(active_species):
        if max(active_species) < data.shape[1]:
            data_for_eval = data[:, active_species]
        else:
            logger.warning(
                f"Data shape {data.shape} vs Active Species {active_species}. Slicing first {len(active_species)} columns."
            )
            data_for_eval = data[:, : len(active_species)]

    def make_evaluator(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base

        evaluator = LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs,
            active_species=active_species,
            active_indices=active_indices,
            theta_base=theta_base,
            data=data_for_eval,
            idx_sparse=idx_sparse,
            sigma_obs=args.sigma_obs,
            cov_rel=0.005,
            rho=0.0,
            theta_linearization=theta_linearization,
            paper_mode=False,
            debug_logger=debug_logger,
            use_absolute_volume=args.use_absolute_volume,
        )

        evaluator.enable_linearization(use_linearization)
        return evaluator

    # Run TMCMC
    model_tag = f"Refine_{args.model}_{'Lin' if use_linearization else 'Full'}"
    logger.info(f"Starting {model_tag} TMCMC...")

    start_time = time.time()

    chains, logL, MAP, converged, diag = run_multi_chain_TMCMC(
        model_tag=model_tag,
        make_evaluator=make_evaluator,
        prior_bounds=prior_bounds,
        theta_base_full=theta_base,
        active_indices=active_indices,
        theta_linearization_init=theta_base,
        n_particles=args.n_particles,
        n_stages=args.n_stages,
        n_chains=args.n_chains,
        min_delta_beta=0.01,
        max_delta_beta=0.5,
        logL_scale=1.0,
        seed=args.seed,
        n_jobs=args.n_jobs,
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

    # Update theta_full
    theta_map_refined_full = theta_base.copy()
    theta_map_refined_full[active_indices] = refined_map

    # Save outputs
    if use_linearization:
        output_subdir = run_dir / "refined"
    else:
        output_subdir = run_dir / f"refinement_{args.model}_full"
    output_subdir.mkdir(exist_ok=True)

    save_json(
        output_subdir / "theta_MAP_refined.json",
        {
            "model": args.stage if args.stage else args.model,
            "linearization": use_linearization,
            "theta_sub": refined_map,
            "theta_full": theta_map_refined_full,
            "active_indices": active_indices,
            "base_run_id": args.run_id,
        },
    )

    np.save(output_subdir / "trace_refined.npy", samples)

    # Plotting
    plot_mgr = PlotManager(str(output_subdir))

    # Re-run simulation
    solver = BiofilmNewtonSolver5S(**solver_kwargs)
    t_fit, x_fit = solver.solve(theta_map_refined_full)

    # --- Prepare Data for Plotting & Metrics ---
    # Slice data to match active species columns if necessary
    data_for_plots = data
    if data.shape[1] > len(active_species):
        if max(active_species) < data.shape[1]:
            logger.info(f"Slicing data columns {active_species} from shape {data.shape}")
            data_for_plots = data[:, active_species]
        else:
            logger.warning(
                f"Data shape {data.shape} vs Active Species {active_species}. Cannot slice by index. Using first {len(active_species)} columns."
            )
            data_for_plots = data[:, : len(active_species)]

    plot_mgr.plot_TSM_simulation(
        t_fit, x_fit, active_species, "Refined_MAP_Fit", data_for_plots, idx_sparse
    )

    # --- Extended Plotting (Corner, Band, Spaghetti) ---
    logger.info("Generating extended plots...")

    # 1. Corner Plot
    try:
        theta_true_full = get_theta_true()
        theta_true_sub = theta_true_full[active_indices]
    except Exception:
        theta_true_sub = None

    param_names = base_config.get("param_names", [f"theta_{i}" for i in active_indices])
    if len(param_names) != len(active_indices):
        param_names = [f"theta_{i}" for i in active_indices]

    plot_mgr.plot_pairplot_posterior(
        samples,
        theta_true_sub,
        refined_map,
        refined_mean,
        param_names,
        f"Refined_{args.stage or args.model}",
    )

    # 2. Posterior Predictive
    n_draws = min(100, len(samples))
    if n_draws > 0:
        rng = np.random.default_rng(args.seed + 999)
        draw_idx = rng.choice(len(samples), size=n_draws, replace=False)

        phibar_samples = np.full((n_draws, len(t_fit), len(active_species)), np.nan)

        for i, k in enumerate(draw_idx):
            theta_s = samples[k]
            theta_full_s = theta_base.copy()
            theta_full_s[active_indices] = theta_s

            t_s, x_s = solver.solve(theta_full_s)

            # Interpolate to t_fit if needed
            if len(t_s) != len(t_fit) or not np.allclose(t_s, t_fit):
                if HAS_SCIPY:
                    x_s_interp = np.zeros((len(t_fit), x_s.shape[1]))
                    for j in range(x_s.shape[1]):
                        interp_func = interp1d(
                            t_s,
                            x_s[:, j],
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        x_s_interp[:, j] = interp_func(t_fit)
                    x_s = x_s_interp
                else:
                    n = min(len(t_s), len(t_fit))
                    x_s_new = np.zeros((len(t_fit), x_s.shape[1]))
                    x_s_new[:n] = x_s[:n]
                    x_s = x_s_new

            phibar_samples[i, :, :] = compute_phibar(x_s, active_species)

        plot_mgr.plot_posterior_predictive_band(
            t_fit,
            phibar_samples,
            active_species,
            f"Refined_{args.stage or args.model}",
            data=data_for_plots,
            idx_sparse=idx_sparse,
            filename="Posterior_Predictive_Band.png",
        )

        plot_mgr.plot_posterior_predictive_spaghetti(
            t_fit,
            phibar_samples,
            active_species,
            f"Refined_{args.stage or args.model}",
            data=data_for_plots,
            idx_sparse=idx_sparse,
            filename="Posterior_Predictive_Spaghetti.png",
            use_paper_naming=False,
        )

    # Compute metrics
    metrics = compute_fit_metrics(t_fit, x_fit, active_species, data_for_plots, idx_sparse)
    save_json(output_subdir / "metrics_refined.json", metrics)
    logger.info(f"Refinement Metrics: {metrics}")

    logger.info(f"Refinement results saved to {output_subdir}")


if __name__ == "__main__":
    main()
