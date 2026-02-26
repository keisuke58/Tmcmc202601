#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_results.py

Post-calculation analysis script for 5-species TMCMC results.
Performs:
1. Uncertainty Quantification (Forward simulation of posterior samples)
2. Parameter Correlation Analysis (Focus on a13 vs a35)
3. Biological Signature Verification (Automated check)

Usage:
    python3 data_5species/main/analyze_results.py --run-dir _runs/Dysbiotic_HOBIC_...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import solver
# Always add module path to sys.path to ensure Numba can find 'improved_5species_jit'
sys.path.insert(0, str(project_root / "tmcmc" / "program2602"))
try:
    from tmcmc.program2602.improved_5species_jit import BiofilmNewtonSolver5S
except ImportError:
    from improved_5species_jit import BiofilmNewtonSolver5S


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze TMCMC results")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: run_dir/analysis)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for uncertainty quantification",
    )
    return parser.parse_args()


def load_data(run_dir):
    run_path = Path(run_dir)

    # Load samples
    samples_path = run_path / "samples.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.npy not found in {run_dir}")
    samples = np.load(samples_path)

    # Load theta_MAP
    map_path = run_path / "theta_MAP.json"
    if not map_path.exists():
        raise FileNotFoundError(f"theta_MAP.json not found in {run_dir}")
    with open(map_path, "r") as f:
        map_data = json.load(f)

    # Load config for solver params
    config_path = run_path / "config.json"
    if not config_path.exists():
        # Fallback: try to guess or use defaults if config missing (less ideal)
        print("Warning: config.json not found, using default solver parameters.")
        config = {}
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    return samples, map_data, config


def run_uncertainty_quantification(samples, map_data, config, n_samples=100):
    print(f"Running Uncertainty Quantification with {n_samples} samples...")

    # Setup solver parameters
    # Extract from config or use defaults matching the project
    solver_kwargs = {
        "dt": config.get("dt", 0.01),
        "maxtimestep": config.get("maxtimestep", 2500),  # Default 2500 steps ~ 21 days
        "c_const": config.get("c_const", 1.0),
        "alpha_const": config.get("alpha_const", 10.0),
        "phi_init": (
            np.array(config["phi_init"])
            if isinstance(config.get("phi_init"), list)
            else config.get("phi_init", 0.01)
        ),
        "use_numba": True,
    }

    # Initialize solver
    solver = BiofilmNewtonSolver5S(**solver_kwargs)

    # Prepare parameter reconstruction
    theta_full_base = np.array(map_data["theta_full"])
    active_indices = map_data["active_indices"]

    # Select random samples
    n_total = samples.shape[0]
    if n_samples > n_total:
        n_samples = n_total

    indices = np.random.choice(n_total, n_samples, replace=False)
    selected_samples = samples[indices]

    # Storage for time series
    # Shape: (n_samples, n_steps, n_species)
    # We only care about relative abundance usually
    n_steps = solver.maxtimestep + 1
    phi_rel_history = np.zeros((n_samples, n_steps, 5))

    for i, sample in enumerate(selected_samples):
        # Reconstruct full theta
        theta = theta_full_base.copy()
        theta[active_indices] = sample

        # Run simulation
        t_arr, g_arr = solver.solve(theta)

        # Calculate relative abundance
        phi_abs = g_arr[:, 0:5]
        phi_total = np.sum(phi_abs, axis=1)
        # Avoid division by zero
        phi_total[phi_total < 1e-9] = 1e-9
        phi_rel = phi_abs / phi_total[:, None]

        phi_rel_history[i] = phi_rel

        if (i + 1) % 10 == 0:
            print(f"  Simulated {i+1}/{n_samples} samples...")

    # Compute percentiles
    percentiles = [2.5, 50.0, 97.5]
    results = {}

    for species_idx in range(5):
        # Shape: (3, n_steps)
        p_data = np.percentile(phi_rel_history[:, :, species_idx], percentiles, axis=0)
        results[f"S{species_idx+1}_p2.5"] = p_data[0]
        results[f"S{species_idx+1}_p50"] = p_data[1]
        results[f"S{species_idx+1}_p97.5"] = p_data[2]

    results["time"] = np.linspace(0, solver.maxtimestep * solver.dt, n_steps)

    return results, phi_rel_history


def analyze_correlations(samples, map_data):
    print("Analyzing Parameter Correlations...")

    active_indices = map_data["active_indices"]
    # We know the names from BiofilmNewtonSolver5S.THETA_NAMES
    all_names = BiofilmNewtonSolver5S.THETA_NAMES
    active_names = [all_names[i] for i in active_indices]

    # Create DataFrame
    df = pd.DataFrame(samples, columns=active_names)

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Check key biological correlation: a13 (S1->S3) vs a35 (S3->S5)
    # Note: In solver, a31 is stored as a13 (idx 10), a53 is stored as a35 (idx 18)
    key_corr = None
    if "a13" in active_names and "a35" in active_names:
        key_corr = df["a13"].corr(df["a35"])
        print(f"  Key Correlation (a13 vs a35): {key_corr:.4f}")

    return corr_matrix, key_corr


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Analyzing results in: {run_dir}")
    print(f"Output directory: {output_dir}")

    # 1. Load Data
    try:
        samples, map_data, config = load_data(run_dir)
        print(f"  Loaded {samples.shape[0]} samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. Uncertainty Quantification
    try:
        uq_results, raw_history = run_uncertainty_quantification(
            samples, map_data, config, args.n_samples
        )

        # Save UQ results
        np.savez(output_dir / "uncertainty_quantification.npz", **uq_results)
        np.save(output_dir / "raw_history_subset.npy", raw_history)
        print("  Saved uncertainty quantification results.")

    except Exception as e:
        print(f"Error in Uncertainty Quantification: {e}")
        import traceback

        traceback.print_exc()

    # 3. Correlation Analysis
    try:
        corr_matrix, key_corr = analyze_correlations(samples, map_data)

        # Save correlations
        corr_matrix.to_csv(output_dir / "parameter_correlations.csv")
        print("  Saved parameter correlations.")

        # Save key correlation summary
        with open(output_dir / "key_correlations.txt", "w") as f:
            f.write(f"a13_a35_correlation: {key_corr}\n")

    except Exception as e:
        print(f"Error in Correlation Analysis: {e}")

    print("\nAnalysis Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
