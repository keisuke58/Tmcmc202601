#!/usr/bin/env python3
"""
Post-processing script to regenerate all figures from completed TMCMC runs.

This script loads results from an existing run directory and generates
all visualization figures without re-running the TMCMC estimation.

Usage:
    python generate_all_figures.py --run-dir _runs/Commensal_Static_20260204_062733

Figures generated:
1. TSM_simulation_*_MAP_Fit_with_data.png - MAP estimate fit
2. TSM_simulation_*_Mean_Fit_with_data.png - Mean estimate fit
3. Residuals_*.png - Per-species residuals
4. parameter_distributions_*.png - Parameter trace/histogram plots
5. corner_plot_*.png - Corner plot (20 params)
6. posterior_predictive_*.png - 5-95% confidence bands
7. posterior_predictive_spaghetti_*.png - Sample trajectories
8. TMCMC_beta_schedule_*.png - Beta progression (if diagnostics available)
9. fit_metrics.json - RMSE, MAE per species
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /home/.../Tmcmc202601
DATA_5SPECIES_ROOT = Path(__file__).parent.parent    # /home/.../Tmcmc202601/data_5species

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

from visualization import PlotManager, compute_fit_metrics, compute_phibar, export_tmcmc_diagnostics_tables
from improved_5species_jit import BiofilmNewtonSolver5S

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_run_results(run_dir: Path) -> Dict[str, Any]:
    """Load all results from a completed run directory."""
    results = {}

    # Load numpy arrays
    for npy_file in ["samples.npy", "logL.npy", "data.npy", "idx_sparse.npy", "t_days.npy"]:
        path = run_dir / npy_file
        if path.exists():
            results[npy_file.replace(".npy", "")] = np.load(path)
            logger.info(f"Loaded {npy_file}: shape {results[npy_file.replace('.npy', '')].shape}")
        else:
            logger.warning(f"Missing file: {npy_file}")

    # Load JSON files
    for json_file in ["config.json", "theta_MAP.json", "theta_mean.json", "results_summary.json"]:
        path = run_dir / json_file
        if path.exists():
            with open(path, 'r') as f:
                results[json_file.replace(".json", "")] = json.load(f)
            logger.info(f"Loaded {json_file}")
        else:
            logger.warning(f"Missing file: {json_file}")

    # Load diagnostics if available
    diag_file = run_dir / "diagnostics.json"
    if diag_file.exists():
        with open(diag_file, 'r') as f:
            results["diagnostics"] = json.load(f)
        logger.info("Loaded diagnostics.json")

    return results


def generate_all_figures(
    run_dir: Path,
    n_posterior_samples: int = 50,
    force: bool = False,
) -> None:
    """
    Generate all figures for a completed TMCMC run.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing results
    n_posterior_samples : int
        Number of posterior samples to use for predictive plots
    force : bool
        If True, regenerate figures even if they exist
    """
    logger.info(f"Loading results from: {run_dir}")
    results = load_run_results(run_dir)

    # Validate required files
    required = ["samples", "logL", "data", "idx_sparse", "config", "theta_MAP", "theta_mean"]
    missing = [r for r in required if r not in results]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    # Extract configuration
    config = results["config"]
    condition = config["condition"]
    cultivation = config["cultivation"]
    name_tag = f"{condition}_{cultivation}"

    # Setup output directory
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Initialize plot manager
    plot_mgr = PlotManager(str(figures_dir))

    # Get model parameters
    dt = config["dt"]
    maxtimestep = config["maxtimestep"]
    c_const = config["c_const"]
    alpha_const = config["alpha_const"]
    phi_init = config["phi_init"]
    # Convert list to numpy array if needed
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init)

    # Get estimation results
    samples = results["samples"]
    logL = results["logL"]
    data = results["data"]
    idx_sparse = results["idx_sparse"]

    theta_MAP_full = np.array(results["theta_MAP"]["theta_full"])
    theta_mean_full = np.array(results["theta_mean"]["theta_full"])

    active_species = [0, 1, 2, 3, 4]
    active_indices = list(range(20))

    # Parameter names
    param_names = [
        "a11", "a12", "a22", "b1", "b2",      # M1
        "a33", "a34", "a44", "b3", "b4",      # M2
        "a13", "a14", "a23", "a24",           # M3
        "a55", "b5",                          # M4
        "a15", "a25", "a35", "a45",           # M5
    ]

    # Initialize solver
    logger.info("Initializing solver...")
    solver = BiofilmNewtonSolver5S(
        dt=dt,
        maxtimestep=maxtimestep,
        c_const=c_const,
        alpha_const=alpha_const,
        phi_init=phi_init,
    )

    # Run simulation with MAP estimate
    logger.info("Running simulation with MAP estimate...")
    t_fit, x_fit_map = solver.solve(theta_MAP_full)
    phibar_map = compute_phibar(x_fit_map, active_species)

    # Run simulation with Mean estimate
    logger.info("Running simulation with Mean estimate...")
    _, x_fit_mean = solver.solve(theta_mean_full)
    phibar_mean = compute_phibar(x_fit_mean, active_species)

    # =========================================================================
    # Generate figures with error handling
    # =========================================================================

    # 1. MAP Fit
    logger.info("Generating MAP fit plot...")
    try:
        plot_mgr.plot_TSM_simulation(
            t_fit, x_fit_map, active_species,
            f"{name_tag}_MAP_Fit",
            data, idx_sparse,
            phibar=phibar_map
        )
    except Exception as e:
        logger.error(f"Failed to generate MAP fit plot: {e}")

    # 2. Mean Fit (NEW)
    logger.info("Generating Mean fit plot...")
    try:
        plot_mgr.plot_TSM_simulation(
            t_fit, x_fit_mean, active_species,
            f"{name_tag}_Mean_Fit",
            data, idx_sparse,
            phibar=phibar_mean
        )
    except Exception as e:
        logger.error(f"Failed to generate Mean fit plot: {e}")

    # 3. Residuals
    logger.info("Generating residuals plot...")
    try:
        plot_mgr.plot_residuals(
            t_fit, phibar_map, data, idx_sparse, active_species,
            f"{name_tag}_Residuals"
        )
    except Exception as e:
        logger.error(f"Failed to generate residuals plot: {e}")

    # 4. Parameter Distributions (Trace/Hist)
    logger.info("Generating parameter distribution plots...")
    try:
        plot_mgr.plot_trace(
            samples, logL, param_names,
            f"{name_tag}_Params"
        )
    except Exception as e:
        logger.error(f"Failed to generate parameter distribution plot: {e}")

    # 5. Corner Plot
    logger.info("Generating corner plot (this may take a while for 20 params)...")
    try:
        plot_mgr.plot_corner(
            samples, param_names,
            f"{name_tag}_Corner"
        )
    except Exception as e:
        logger.error(f"Failed to generate corner plot: {e}")

    # 6 & 7. Posterior Predictive Plots
    logger.info(f"Generating posterior predictive plots ({n_posterior_samples} trajectories)...")
    try:
        n_total_samples = samples.shape[0]
        if n_total_samples < n_posterior_samples:
            logger.warning(f"Only {n_total_samples} samples available, using all")
            indices = np.arange(n_total_samples)
        else:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(n_total_samples, n_posterior_samples, replace=False)

        phibar_samples = []
        for idx in indices:
            theta_sample = samples[idx]
            theta_full = theta_MAP_full.copy()
            theta_full[active_indices] = theta_sample

            _, x_sim = solver.solve(theta_full)
            phibar_sim = compute_phibar(x_sim, active_species)
            phibar_samples.append(phibar_sim)

        phibar_samples = np.array(phibar_samples)  # (n_draws, n_time, n_species)

        # Posterior Band
        logger.info("Generating posterior band plot...")
        try:
            plot_mgr.plot_posterior_predictive_band(
                t_fit, phibar_samples, active_species,
                f"{name_tag}_PosteriorBand",
                data, idx_sparse
            )
        except Exception as e:
            logger.error(f"Failed to generate posterior band plot: {e}")

        # Spaghetti Plot
        logger.info("Generating spaghetti plot...")
        try:
            plot_mgr.plot_posterior_predictive_spaghetti(
                t_fit, phibar_samples, active_species,
                f"{name_tag}_PosteriorSpaghetti",
                data, idx_sparse,
                num_trajectories=n_posterior_samples
            )
        except Exception as e:
            logger.error(f"Failed to generate spaghetti plot: {e}")

    except Exception as e:
        logger.error(f"Failed to generate posterior predictive plots: {e}")

    # 8. TMCMC Beta Schedule (if diagnostics available)
    if "diagnostics" in results:
        logger.info("Generating TMCMC diagnostics plots...")
        diag = results["diagnostics"]

        try:
            if "beta_schedules" in diag:
                plot_mgr.plot_beta_schedule(diag["beta_schedules"], name_tag)
        except Exception as e:
            logger.error(f"Failed to generate beta schedule plot: {e}")

        try:
            # Export diagnostics tables
            export_tmcmc_diagnostics_tables(run_dir, name_tag, diag)
        except Exception as e:
            logger.error(f"Failed to export diagnostics tables: {e}")
    else:
        logger.info("No diagnostics data available - skipping TMCMC diagnostics plots")

    # 9. Fit Metrics
    logger.info("Computing and saving fit metrics...")
    try:
        # MAP fit metrics
        metrics_map = compute_fit_metrics(t_fit, x_fit_map, active_species, data, idx_sparse)
        metrics_map["estimate_type"] = "MAP"

        # Mean fit metrics
        metrics_mean = compute_fit_metrics(t_fit, x_fit_mean, active_species, data, idx_sparse)
        metrics_mean["estimate_type"] = "Mean"

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        metrics = {
            "MAP": convert_to_serializable(metrics_map),
            "Mean": convert_to_serializable(metrics_mean),
            "species_names": ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"],
        }

        metrics_path = run_dir / "fit_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved fit metrics to {metrics_path}")

        # Print summary
        print("\n" + "="*60)
        print("FIT METRICS SUMMARY")
        print("="*60)
        print("\nMAP Estimate:")
        print(f"  Total RMSE: {metrics_map['rmse_total']:.6f}")
        print(f"  Total MAE:  {metrics_map['mae_total']:.6f}")
        print("  Per-species RMSE:", [f"{v:.4f}" for v in metrics_map['rmse_per_species']])

        print("\nMean Estimate:")
        print(f"  Total RMSE: {metrics_mean['rmse_total']:.6f}")
        print(f"  Total MAE:  {metrics_mean['mae_total']:.6f}")
        print("  Per-species RMSE:", [f"{v:.4f}" for v in metrics_mean['rmse_per_species']])
        print("="*60)

    except Exception as e:
        logger.error(f"Failed to compute/save fit metrics: {e}")

    # Save manifest
    try:
        plot_mgr.save_manifest()
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")

    # Summary
    n_figs = len(plot_mgr.generated_figs)
    logger.info(f"\nGenerated {n_figs} figures in {figures_dir}")
    for fig_path in plot_mgr.generated_figs:
        logger.info(f"  - {fig_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate all figures from completed TMCMC run")
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to the run directory (e.g., _runs/Commensal_Static_20260204_062733)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of posterior samples for predictive plots (default: 50)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration of all figures"
    )

    args = parser.parse_args()

    # Handle relative paths
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        # Try relative to data_5species directory
        run_dir = DATA_5SPECIES_ROOT / run_dir

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    generate_all_figures(run_dir, n_posterior_samples=args.n_samples, force=args.force)


if __name__ == "__main__":
    main()
