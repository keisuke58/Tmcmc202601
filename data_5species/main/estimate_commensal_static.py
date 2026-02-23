#!/usr/bin/env python3
"""
Parameter Estimation for Commensal Static Experimental Data.

This script loads experimental biofilm data and estimates model parameters
using TMCMC (Transitional Markov Chain Monte Carlo).

Experimental Data:
- Condition: Commensal
- Cultivation: Static
- Timepoints: Day 1, 3, 6, 10, 15, 21
- Measurements: Total biofilm volume + species distribution percentages

Species Mapping (from experimental data colors to model indices):
- Blue (S. oralis) -> Species 0
- Green (A. naeslundii) -> Species 1
- Yellow (V. dispar) -> Species 2
- Purple (F. nucleatum) -> Species 3
- Red (P. gingivalis) -> Species 4

Usage:
    python estimate_commensal_static.py [options]
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /home/.../Tmcmc202601
DATA_5SPECIES_ROOT = Path(__file__).parent.parent    # /home/.../Tmcmc202601/data_5species

# Add data_5species folder (contains core, debug, utils, visualization)
sys.path.insert(0, str(DATA_5SPECIES_ROOT))
# Add config location (program2602) for config.py
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
# Add tmcmc folder for improved_5species_jit.py
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
# Add project root
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_CONFIGS,
    setup_logging,
    DebugConfig,
    DebugLevel,
    PRIOR_BOUNDS_DEFAULT,
)
from core import (
    LogLikelihoodEvaluator,
    run_multi_chain_TMCMC,
    compute_MAP_with_uncertainty,
)
from debug import DebugLogger
from utils import save_json, save_npy
from visualization import PlotManager, compute_fit_metrics, compute_phibar
from improved_5species_jit import BiofilmNewtonSolver5S, HAS_NUMBA

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

# Species color to model index mapping
SPECIES_MAP = {
    "Blue": 0,      # S. oralis
    "Green": 1,     # A. naeslundii
    "Yellow": 2,    # V. dispar
    "Orange": 2,    # V. parvula (Dysbiotic strain of Veillonella)
    "Purple": 3,    # F. nucleatum
    "Red": 4,       # P. gingivalis
}

# For Commensal (no Orange/V. parvula), remap Purple and Red
SPECIES_MAP_COMMENSAL = {
    "Blue": 0,      # S. oralis
    "Green": 1,     # A. naeslundii
    "Yellow": 2,    # V. dispar
    "Purple": 3,    # F. nucleatum
    "Red": 4,       # P. gingivalis
}


def load_experimental_data(
    data_dir: Path,
    condition: str = "Commensal",
    cultivation: str = "Static",
    start_from_day: int = 1,
    normalize: bool = False
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
        raise FileNotFoundError(f"Could not find boxplot data. Checked: {[str(f) for f in possible_boxplot_files]}")

    boxplot_df = pd.read_csv(boxplot_file)

    # Filter for condition/cultivation
    if 'condition' in boxplot_df.columns:
        mask = (boxplot_df['condition'] == condition) & (boxplot_df['cultivation'] == cultivation)
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
        raise FileNotFoundError(f"Could not find species data. Checked: {[str(f) for f in possible_species_files]}")

    species_df = pd.read_csv(species_file)

    # Filter
    mask = (species_df['condition'] == condition) & (species_df['cultivation'] == cultivation)
    species_df = species_df[mask]

    # Get unique days
    days = sorted(boxplot_df['day'].unique())
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
        day_volume = boxplot_df[boxplot_df['day'] == day]
        if len(day_volume) > 0:
            total_vol = day_volume['median'].values[0]
            total_volumes[i] = total_vol

            # Estimate sigma from IQR: sigma ≈ IQR / 1.35
            q1 = day_volume['q1'].values[0]
            q3 = day_volume['q3'].values[0]
            iqr = q3 - q1
            sigma_obs_estimates.append(iqr / 1.35)

        # Get species percentages for this day
        for _, row in species_df[species_df['day'] == day].iterrows():
            species_color = row['species']
            if species_color in species_map:
                species_idx = species_map[species_color]
                percentage = row['median'] / 100.0  # Convert % to fraction

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
        logger.info(f"Filtering data to start from day {start_from_day}: {n_timepoints} timepoints remaining")

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
    t_days: np.ndarray,
    dt: float,
    maxtimestep: int,
    day_scale: float = None
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
    phi_init_array: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run TMCMC parameter estimation on experimental data.

    Args:
        phi_init_array: If provided, use per-species initial conditions (5,) array.
                       Otherwise use scalar args.phi_init.
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

    # Model configuration
    # Use a configuration suitable for 5-species model
    solver_kwargs = {
        "dt": args.dt,
        "maxtimestep": args.maxtimestep,
        "c_const": args.c_const,
        "alpha_const": args.alpha_const,
        "phi_init": phi_init,
    }

    active_species = [0, 1, 2, 3, 4]  # All 5 species
    active_indices = list(range(20))  # All 20 parameters

    # Determine prior bounds
    p_min = args.prior_min if args.prior_min is not None else PRIOR_BOUNDS_DEFAULT[0]
    p_max = args.prior_max if args.prior_max is not None else PRIOR_BOUNDS_DEFAULT[1]
    
    # Initialize theta_base with prior mean
    prior_mean = (p_min + p_max) / 2.0
    theta_base = np.full(20, prior_mean)

    # Prior bounds for all parameters
    prior_bounds = [(p_min, p_max) for _ in range(20)]
    
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
        logger.info(f"Tightening decay priors (b1-b5, indices {DECAY_INDICES}) to [0.0, {decay_max}]")
        for idx in DECAY_INDICES:
            prior_bounds[idx] = (0.0, decay_max)
            # Update base to mean of new range
            theta_base[idx] = decay_max / 2.0

    # Sigma for likelihood
    sigma_obs = args.sigma_obs if args.sigma_obs else metadata.get('sigma_obs_estimated', 0.05)

    logger.info(f"Using sigma_obs = {sigma_obs}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"idx_sparse: {idx_sparse}")

    def make_evaluator(theta_linearization=None):
        if theta_linearization is None:
            theta_linearization = theta_base

        return LogLikelihoodEvaluator(
            solver_kwargs=solver_kwargs,
            active_species=active_species,
            active_indices=active_indices,
            theta_base=theta_base,
            data=data,
            idx_sparse=idx_sparse,
            sigma_obs=sigma_obs,
            cov_rel=args.cov_rel,
            rho=0.0,
            theta_linearization=theta_linearization,
            paper_mode=False,
            debug_logger=debug_logger,
            use_absolute_volume=args.use_absolute_volume,
        )

    # Run TMCMC
    logger.info("Starting TMCMC estimation...")
    start_time = time.time()

    chains, logL, MAP, converged, diag = run_multi_chain_TMCMC(
        model_tag="Commensal_Static",
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
        debug_config=debug_config,
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
    theta_MAP_full = theta_base.copy()
    theta_MAP_full[active_indices] = MAP_estimate

    theta_mean_full = theta_base.copy()
    theta_mean_full[active_indices] = mean_estimate

    return {
        "samples": samples,
        "logL": logL_all,
        "MAP": MAP_estimate,
        "mean": mean_estimate,
        "theta_MAP_full": theta_MAP_full,
        "theta_mean_full": theta_mean_full,
        "elapsed_time": elapsed,
        "converged": converged,
        "diagnostics": diag,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate parameters from Commensal Static data")

    # Data options
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent.parent),
                        help="Directory containing experimental data")
    parser.add_argument("--condition", type=str, default="Commensal",
                        choices=["Commensal", "Dysbiotic"])
    parser.add_argument("--cultivation", type=str, default="Static",
                        choices=["Static", "HOBIC"])

    # Model options
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step")
    parser.add_argument("--maxtimestep", type=int, default=2500, help="Max time steps")
    parser.add_argument("--c-const", type=float, default=100.0, help="c constant (interaction strength, increased from 25.0 to overcome CH barrier)")
    parser.add_argument("--alpha-const", type=float, default=1.0, help="alpha constant (enabled decay terms)")
    parser.add_argument("--phi-init", type=float, default=0.02, help="Initial phi (ignored if --use-exp-init)")
    parser.add_argument("--day-scale", type=float, default=None,
                        help="Scaling factor: model_time = day * day_scale (auto if None)")
    parser.add_argument("--use-exp-init", action="store_true",
                        help="Use experimental data (from start-from-day) as initial conditions")
    parser.add_argument("--start-from-day", type=int, default=1,
                        help="Start fitting from this day (default: 1)")
    parser.add_argument("--normalize-data", action="store_true",
                        help="Normalize data to species fractions (sum=1) instead of absolute volumes")

    # Estimation options
    parser.add_argument("--sigma-obs", type=float, default=None,
                        help="Observation noise (default: estimated from data)")
    parser.add_argument("--cov-rel", type=float, default=0.005,
                        help="Relative covariance for ROM")
    parser.add_argument("--use-absolute-volume", action="store_true",
                        help="Use absolute volume (phi*gamma) for likelihood")

    # Prior options
    parser.add_argument("--prior-min", type=float, default=None,
                        help="Minimum value for prior uniform distribution (default: use config)")
    parser.add_argument("--prior-max", type=float, default=None,
                        help="Maximum value for prior uniform distribution (default: use config)")
    parser.add_argument("--widen-m1-priors", action="store_true",
                        help="Widen priors for M1 parameters (indices 0-4) to [0, 10]")
    parser.add_argument("--prior-decay-max", type=float, default=None,
                        help="Maximum value for decay parameters b1-b5 (indices 3,4,8,9,15). "
                             "Use smaller values (e.g., 1.0) if model over-predicts decline.")

    # TMCMC options
    parser.add_argument("--n-particles", type=int, default=500, help="Number of particles")
    parser.add_argument("--n-stages", type=int, default=30, help="Number of stages")
    parser.add_argument("--n-chains", type=int, default=2, help="Number of chains")
    parser.add_argument("--n-jobs", type=int, default=12, help="Parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output options
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--debug-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug_level)

    # Setup paths
    data_dir = Path(args.data_dir)

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
        "metadata": metadata,
    }
    save_json(output_dir / "config.json", config)

    # Run estimation
    logger.info("Running parameter estimation...")
    results = run_estimation(data, idx_sparse, args, output_dir, metadata, phi_init_array)

    # Save results
    save_npy(output_dir / "samples.npy", results["samples"])
    save_npy(output_dir / "logL.npy", results["logL"])

    save_json(output_dir / "theta_MAP.json", {
        "theta_sub": results["MAP"].tolist(),
        "theta_full": results["theta_MAP_full"].tolist(),
        "active_indices": list(range(20)),
    })

    save_json(output_dir / "theta_mean.json", {
        "theta_sub": results["mean"].tolist(),
        "theta_full": results["theta_mean_full"].tolist(),
        "active_indices": list(range(20)),
    })

    save_json(output_dir / "results_summary.json", {
        "elapsed_time": results["elapsed_time"],
        "converged": [bool(c) for c in results["converged"]],
        "MAP": results["MAP"].tolist(),
        "mean": results["mean"].tolist(),
    })

    # Export TMCMC diagnostics tables (if available)
    if "diagnostics" in results and results["diagnostics"]:
        try:
            from visualization import export_tmcmc_diagnostics_tables
            export_tmcmc_diagnostics_tables(
                output_dir,
                f"{args.condition}_{args.cultivation}",
                results["diagnostics"]
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
    )

    # Plot fit vs data
    active_species = [0, 1, 2, 3, 4]
    name_tag = f"{args.condition}_{args.cultivation}"

    # Parameter names for plotting
    param_names = [
        "a11", "a12", "a22", "b1", "b2",      # M1
        "a33", "a34", "a44", "b3", "b4",      # M2
        "a13", "a14", "a23", "a24",           # M3
        "a55", "b5",                          # M4
        "a15", "a25", "a35", "a45",           # M5
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
            t_fit, x_fit_map, active_species,
            f"{name_tag}_MAP_Fit",
            data, idx_sparse,
            phibar=phibar_map
        )
    except Exception as e:
        logger.warning(f"Failed to generate MAP fit plot: {e}")

    # 2. Mean Fit (NEW)
    try:
        plot_mgr.plot_TSM_simulation(
            t_fit, x_fit_mean, active_species,
            f"{name_tag}_Mean_Fit",
            data, idx_sparse,
            phibar=phibar_mean
        )
    except Exception as e:
        logger.warning(f"Failed to generate Mean fit plot: {e}")

    # 3. Residuals
    try:
        plot_mgr.plot_residuals(
            t_fit, phibar_map, data, idx_sparse, active_species,
            f"{name_tag}_Residuals"
        )
    except Exception as e:
        logger.warning(f"Failed to generate residuals plot: {e}")

    # 4. Parameter Distributions (Trace/Hist)
    try:
        plot_mgr.plot_trace(
            results["samples"], results["logL"], param_names,
            f"{name_tag}_Params"
        )
    except Exception as e:
        logger.warning(f"Failed to generate parameter distribution plot: {e}")

    # 5. Corner Plot
    try:
        plot_mgr.plot_corner(
            results["samples"], param_names,
            f"{name_tag}_Corner"
        )
    except Exception as e:
        logger.warning(f"Failed to generate corner plot: {e}")

    # 6 & 7. Posterior Predictive Plots
    logger.info("Generating posterior predictive plots (sampling 50 trajectories)...")
    n_plot_samples = 50
    n_total_samples = results["samples"].shape[0]

    if n_total_samples < n_plot_samples:
        logger.warning(f"Number of samples ({n_total_samples}) is less than requested for plotting ({n_plot_samples}). Using all samples.")
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
                t_fit, phibar_samples, active_species,
                f"{name_tag}_PosteriorBand",
                data, idx_sparse
            )
        except Exception as e:
            logger.warning(f"Failed to generate posterior band plot: {e}")

        # Spaghetti Plot
        try:
            plot_mgr.plot_posterior_predictive_spaghetti(
                t_fit, phibar_samples, active_species,
                f"{name_tag}_PosteriorSpaghetti",
                data, idx_sparse,
                num_trajectories=50
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
            "species_names": ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"],
        }

        with open(output_dir / "fit_metrics.json", 'w') as f:
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
    print("\n" + "="*60)
    print("ESTIMATION SUMMARY")

    for i, (name, val) in enumerate(zip(param_names, results["MAP"])):
        print(f"  {name}: {val:.4f}")

    print("="*60)


if __name__ == "__main__":
    main()
