#!/usr/bin/env python3
"""
Update specific figures in the figures directory with correct colors and axis settings.
Target files:
- posterior_predictive_Commensal_Static_PosteriorBand.png
- TSM_simulation_Commensal_Static_MAP_Fit_with_data.png

Requirements:
- Use Socransky colors (with Orange for S3 in Dysbiotic if applicable, though this is Commensal).
- X-axis in Days, starting at 0.
- Shared Y-axis (or at least consistent).
- Publication quality.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

# Project path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_5SPECIES_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing the solver
try:
    from improved_5species_jit import BiofilmNewtonSolver5S
except ImportError:
    # Fallback for different directory structures
    sys.path.append(str(PROJECT_ROOT / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

# Set seaborn style for publication
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Standard Socransky Colors
COLORS = [
    "#1f77b4",  # S1: Blue (Health)
    "#2ca02c",  # S2: Green (Early)
    "#bcbd22",  # S3: Yellow (V. dispar) - Commensal Default
    "#9467bd",  # S4: Purple (Bridge)
    "#d62728",  # S5: Red (Red Complex)
]

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]


def get_colors(condition: str) -> List[str]:
    """Get color palette based on condition."""
    colors = list(COLORS)
    # If Dysbiotic, S3 is Orange (V. parvula)
    if "Dysbiotic" in condition:
        colors[2] = "#ff7f0e"  # Orange
    return colors


def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """Load necessary data for simulation and plotting."""
    data = {}

    # Load JSONs
    with open(run_dir / "config.json", "r") as f:
        data["config"] = json.load(f)

    with open(run_dir / "theta_MAP.json", "r") as f:
        data["theta_MAP"] = json.load(f)

    # Load NPYs
    data["data_points"] = np.load(run_dir / "data.npy")
    data["t_days"] = np.load(run_dir / "t_days.npy")

    # Load samples for posterior band
    samples_path = run_dir / "samples.npy"
    if samples_path.exists():
        data["samples"] = np.load(samples_path)
    else:
        print(f"Warning: samples.npy not found at {samples_path}")
        data["samples"] = None

    return data


def compute_phibar_local(x0: np.ndarray) -> np.ndarray:
    """
    Compute observable φ̄ = φ * ψ (living bacteria volume fraction).
    x0: [phi1..5, phi0, psi1..5, gamma]
    """
    # Assuming x0 shape (n_time, n_state)
    # n_state should be 12 for 5 species
    # Indices:
    # phi: 0-4
    # phi0: 5
    # psi: 6-10
    # gamma: 11

    phi = x0[:, 0:5]
    psi = x0[:, 6:11]
    return phi * psi


def get_time_in_days(t_arr: np.ndarray, t_days: np.ndarray) -> np.ndarray:
    """Map simulation time to days."""
    t_min, t_max = t_arr.min(), t_arr.max()
    day_min, day_max = t_days.min(), t_days.max()

    if t_max > t_min:
        t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
    else:
        t_plot = t_arr
    return t_plot


def setup_solver(config: Dict[str, Any]) -> BiofilmNewtonSolver5S:
    """Initialize solver from config."""
    phi_init = config["phi_init"]
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init)

    return BiofilmNewtonSolver5S(
        dt=config["dt"],
        maxtimestep=config["maxtimestep"],
        c_const=config["c_const"],
        alpha_const=config["alpha_const"],
        phi_init=phi_init,
    )


def plot_tsm_simulation_map(
    run_dir: Path, data: Dict[str, Any], solver: BiofilmNewtonSolver5S, output_path: Path
):
    """Update TSM_simulation_Commensal_Static_MAP_Fit_with_data.png"""
    print(f"Generating MAP fit plot: {output_path.name}")

    theta_full = np.array(data["theta_MAP"]["theta_full"])
    t_arr, x_arr = solver.solve(theta_full)
    phibar_map = compute_phibar_local(x_arr)

    t_days = data["t_days"]
    t_plot = get_time_in_days(t_arr, t_days)
    data_points = data["data_points"]
    condition = data["config"].get("condition", "Commensal")
    current_colors = get_colors(condition)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)

    for i in range(5):
        ax = axes[i]
        color = current_colors[i]

        # Plot MAP
        ax.plot(t_plot, phibar_map[:, i], color=color, linewidth=3, label="MAP")

        # Plot Data
        ax.scatter(
            t_days, data_points[:, i], color=color, s=80, edgecolor="black", zorder=10, label="Data"
        )

        ax.set_title(SPECIES_NAMES[i], fontsize=14, fontweight="bold", color=color)
        ax.set_xlabel("Time (Days)", fontsize=12)
        ax.set_xlim(left=0)  # Force start at 0

        if i == 0:
            ax.set_ylabel("Relative Abundance", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_posterior_band(
    run_dir: Path,
    data: Dict[str, Any],
    solver: BiofilmNewtonSolver5S,
    output_path: Path,
    n_samples: int = 50,
):
    """Update posterior_predictive_Commensal_Static_PosteriorBand.png"""
    print(f"Generating Posterior Band plot: {output_path.name}")

    samples = data.get("samples")
    if samples is None:
        print("Skipping Posterior Band plot (no samples)")
        return

    # Check indices
    theta_map_data = data["theta_MAP"]
    active_indices = theta_map_data.get("active_indices", list(range(20)))
    theta_full_template = np.array(theta_map_data["theta_full"])

    # Select random samples
    n_total = len(samples)
    if n_total > n_samples:
        indices = np.random.choice(n_total, n_samples, replace=False)
    else:
        indices = np.arange(n_total)

    # Simulate
    phibar_collection = []
    # Use the first simulation time array for interpolation/alignment if needed
    # But usually t_arr is consistent if dt/maxtimestep are constant

    # We need a common time grid for percentiles
    # Let's run MAP first to get the reference time grid
    t_ref, _ = solver.solve(theta_full_template)

    for idx in indices:
        theta_sample = samples[idx]
        theta_curr = theta_full_template.copy()

        # Assign active params
        # If sample size matches active_indices size
        if len(theta_sample) == len(active_indices):
            theta_curr[active_indices] = theta_sample
        elif len(theta_sample) == 20:
            theta_curr = theta_sample
        else:
            # Fallback or error
            continue

        t_s, x_s = solver.solve(theta_curr)

        # Interpolate to t_ref if needed
        if len(t_s) != len(t_ref) or not np.allclose(t_s, t_ref):
            x_s_interp = np.zeros((len(t_ref), x_s.shape[1]))
            for k in range(x_s.shape[1]):
                x_s_interp[:, k] = np.interp(t_ref, t_s, x_s[:, k])
            phibar_s = compute_phibar_local(x_s_interp)
        else:
            phibar_s = compute_phibar_local(x_s)

        phibar_collection.append(phibar_s)

    phibar_stack = np.array(phibar_collection)  # (n_samples, n_time, n_species)

    # Compute Percentiles
    p05 = np.percentile(phibar_stack, 5, axis=0)
    p50 = np.percentile(phibar_stack, 50, axis=0)
    p95 = np.percentile(phibar_stack, 95, axis=0)

    # Plotting
    t_days = data["t_days"]
    t_plot = get_time_in_days(t_ref, t_days)
    data_points = data["data_points"]
    condition = data["config"].get("condition", "Commensal")
    current_colors = get_colors(condition)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)

    for i in range(5):
        ax = axes[i]
        color = current_colors[i]

        # Plot Band
        ax.fill_between(t_plot, p05[:, i], p95[:, i], color=color, alpha=0.3, label="90% CI")

        # Plot Median
        ax.plot(t_plot, p50[:, i], color=color, linestyle="--", linewidth=2, label="Median")

        # Plot Data
        ax.scatter(t_days, data_points[:, i], color=color, s=80, edgecolor="black", zorder=10)

        ax.set_title(SPECIES_NAMES[i], fontsize=14, fontweight="bold", color=color)
        ax.set_xlabel("Time (Days)", fontsize=12)
        ax.set_xlim(left=0)

        if i == 0:
            ax.set_ylabel("Relative Abundance", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Directory {args.run_dir} does not exist.")
        return

    data = load_run_data(args.run_dir)
    solver = setup_solver(data["config"])

    figures_dir = args.run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Update TSM simulation (MAP)
    # Filename pattern: TSM_simulation_{condition}_{cultivation}_MAP_Fit_with_data.png
    # But user gave specific filename: TSM_simulation_Commensal_Static_MAP_Fit_with_data.png
    # We can reconstruct it or just use the one user asked for.
    # To be safe, let's use the one that matches the run config, which should be Commensal_Static
    cond = data["config"].get("condition", "Commensal")
    cult = data["config"].get("cultivation", "Static")
    name_tag = f"{cond}_{cult}"

    tsm_filename = f"TSM_simulation_{name_tag}_MAP_Fit_with_data.png"
    plot_tsm_simulation_map(args.run_dir, data, solver, figures_dir / tsm_filename)

    # 2. Update Posterior Band
    band_filename = f"posterior_predictive_{name_tag}_PosteriorBand.png"
    plot_posterior_band(args.run_dir, data, solver, figures_dir / band_filename)

    print("Done updating figures.")


if __name__ == "__main__":
    main()
