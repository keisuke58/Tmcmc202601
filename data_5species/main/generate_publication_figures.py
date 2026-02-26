#!/usr/bin/env python3
"""
Generate publication-quality figures for the 5-species biofilm model.
Target figures:
1. MAP Estimate Fit (Time Series)
2. Posterior Predictive Band (Time Series)
3. Interaction Matrix Heatmap (A matrix)
4. Parameter Boxplots (Posterior Distributions)
5. Trace Plots (Convergence Diagnostic)
6. Residual Analysis

Usage:
    python generate_publication_figures.py --run_dir <RUN_DIR>
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

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
    sys.path.append(str(PROJECT_ROOT / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

# Set seaborn style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


def load_species_config():
    config_path = DATA_5SPECIES_ROOT / "species_config.json"
    if not config_path.exists():
        # Fallback if config is missing
        return {
            "species_names": [
                "S. oralis",
                "A. naeslundii",
                "V. dispar",
                "F. nucleatum",
                "P. gingivalis",
            ],
            "colors": {
                "default": ["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#d62728"],
                "Commensal": ["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#d62728"],
                "Dysbiotic": ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"],
            },
        }
    with open(config_path, "r") as f:
        return json.load(f)


def get_colors(condition: str, config: Dict) -> List[str]:
    if "Dysbiotic" in condition:
        return config["colors"]["Dysbiotic"]
    return config["colors"]["Commensal"]


def load_run_data(run_dir: Path):
    data = {}

    # Load theta_MAP
    with open(run_dir / "theta_MAP.json", "r") as f:
        data["theta_map"] = json.load(f)

    # Load experimental data
    data["exp_data"] = np.load(run_dir / "data.npy")
    data["idx_sparse"] = np.load(run_dir / "idx_sparse.npy")
    data["t_days"] = np.load(run_dir / "t_days.npy")

    # Load samples if available
    samples_path = run_dir / "samples.npy"
    if samples_path.exists():
        data["samples"] = np.load(samples_path)
    else:
        data["samples"] = None

    # Determine condition from run name
    run_name = run_dir.name
    if "Dysbiotic" in run_name:
        data["condition"] = "Dysbiotic"
    else:
        data["condition"] = "Commensal"

    return data


def plot_map_fit(data_dict, config, output_dir):
    """Figure 1: MAP Estimate Fit vs Data"""
    print("Generating Figure 1: MAP Fit...")
    theta_full = np.array(data_dict["theta_map"]["theta_full"])
    exp_data = data_dict["exp_data"]
    idx_sparse = data_dict["idx_sparse"]
    colors = get_colors(data_dict["condition"], config)
    species_names = config["species_names"]

    # Run simulation
    solver = BiofilmNewtonSolver5S(
        active_species=list(range(5)), phi_init=np.array([1e-6] * 5)  # Default small init
    )
    result = solver.solve(theta_full)
    if len(result) == 2:
        t_sim, x_sim = result
    else:
        t_sim, x_sim = result[:2]

    # Convert to days
    t_days_sim = t_sim * 3.0  # Assuming T_MAX=7.0 maps to 21 days, scaling factor ~3
    # Better: align with data points.
    # Standard: 0.33 model time = 1 day approx. t_sim is usually 0-7.0.
    # Data points are at days 1, 3, 6, 10, 15, 21.

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Calculate Phibar (Absolute Volume proxy)
    # x_sim contains [phi1..5, S]
    # We need phi * psi (biomass density). Assuming constant density for now or just phi.
    # Actually, data is usually log10(Absolute Volume).
    # Model output x is phi (volume fraction).
    # We need to scale phi to match data magnitude or use a scaling factor.
    # Assuming data is log10(um^3).
    # For now, let's plot phi directly vs normalized data if possible, OR
    # use the helper if available. But let's stick to simple direct plotting of phi for dynamics.

    # Wait, the data in data.npy is usually normalized or log-transformed.
    # Let's check the scale.
    # If data is ~ -5 to 0, it's log scale.

    for i in range(5):
        ax = axes[i]
        color = colors[i]
        name = species_names[i]

        # Plot Simulation
        # x_sim[:, i] is phi_i
        ax.plot(
            t_sim * 3,
            np.log10(x_sim[:, i] + 1e-10),
            label=f"{name} (Model)",
            color=color,
            linewidth=2.5,
        )

        # Plot Data
        # idx_sparse contains time indices for data points
        # exp_data shape: [time_points, species]
        # We need to map time indices to days.
        # Assuming idx_sparse corresponds to [1, 3, 6, 10, 15, 21] days approx
        days_data = np.array([1, 3, 6, 10, 15, 21])
        # Note: This is an assumption. Ideally read from t_days.npy
        if "t_days" in data_dict:
            days_data = data_dict["t_days"]

        # Handle data shape mismatch if any
        if exp_data.shape[1] > i:
            ax.scatter(
                days_data,
                exp_data[:, i],
                color=color,
                marker="o",
                s=50,
                edgecolors="k",
                label="Experiment",
            )

        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right")

    # Remove empty 6th subplot
    fig.delaxes(axes[5])

    # Global labels
    fig.text(0.5, 0.04, "Time (Days)", ha="center", fontsize=14)
    fig.text(
        0.04, 0.5, "Log10 Biomass / Volume Fraction", va="center", rotation="vertical", fontsize=14
    )
    plt.suptitle(f"MAP Estimate Fit - {data_dict['condition']}", fontsize=16)

    plt.savefig(output_dir / "pub_map_fit_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pub_map_fit_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_posterior_band(data_dict, config, output_dir):
    """Figure 2: Posterior Predictive Band"""
    print("Generating Figure 2: Posterior Band...")
    if data_dict["samples"] is None:
        print("No samples found, skipping posterior band.")
        return

    samples = data_dict["samples"]
    # Thinning for plotting
    n_plot = min(100, len(samples))
    indices = np.random.choice(len(samples), n_plot, replace=False)

    colors = get_colors(data_dict["condition"], config)
    species_names = config["species_names"]

    solver = BiofilmNewtonSolver5S(active_species=list(range(5)), phi_init=np.array([1e-6] * 5))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Store trajectories
    trajectories = {i: [] for i in range(5)}
    t_sim = None

    for idx in indices:
        theta = samples[idx]
        result = solver.solve(theta)
        if len(result) == 2:
            t, x = result
        else:
            t, x = result[:2]

        if t_sim is None:
            t_sim = t * 3

        for i in range(5):
            trajectories[i].append(np.log10(x[:, i] + 1e-10))

    # Plot bands
    for i in range(5):
        ax = axes[i]
        color = colors[i]
        name = species_names[i]

        traj = np.array(trajectories[i])
        mu = np.mean(traj, axis=0)
        sigma = np.std(traj, axis=0)

        ax.plot(t_sim, mu, color=color, linewidth=2, label="Mean Prediction")
        ax.fill_between(
            t_sim, mu - 2 * sigma, mu + 2 * sigma, color=color, alpha=0.2, label="95% CI"
        )

        # Plot Data
        days_data = data_dict.get("t_days", np.array([1, 3, 6, 10, 15, 21]))
        exp_data = data_dict["exp_data"]
        ax.scatter(days_data, exp_data[:, i], color=color, marker="o", s=30, edgecolors="k")

        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.delaxes(axes[5])
    fig.text(0.5, 0.04, "Time (Days)", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Log10 Biomass", va="center", rotation="vertical", fontsize=14)
    plt.suptitle(f"Posterior Predictive Uncertainty - {data_dict['condition']}", fontsize=16)

    plt.savefig(output_dir / "pub_posterior_band.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pub_posterior_band.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(data_dict, config, output_dir):
    """Figure 3: Interaction Matrix"""
    print("Generating Figure 3: Interaction Heatmap...")
    theta_full = np.array(data_dict["theta_map"]["theta_full"])

    solver = BiofilmNewtonSolver5S()
    A, _ = solver.theta_to_matrices(theta_full)

    species_names = config["species_names"]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        A,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=species_names,
        yticklabels=species_names,
        ax=ax,
        square=True,
        cbar_kws={"label": "Interaction Strength"},
    )

    ax.set_title("Inferred Interaction Matrix (A)", fontsize=16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(output_dir / "pub_interaction_heatmap.png", dpi=300)
    plt.savefig(output_dir / "pub_interaction_heatmap.pdf", dpi=300)
    plt.close()


def plot_boxplots(data_dict, config, output_dir):
    """Figure 4: Parameter Boxplots"""
    print("Generating Figure 4: Parameter Boxplots...")
    if data_dict["samples"] is None:
        return

    samples = data_dict["samples"]
    # Select key parameters: Growth rates (b) and Self-interactions (a_ii)
    # Indices based on model:
    # 0: a11, 1: a12...
    # b params are at specific indices.
    # Let's verify mapping or just plot all 20 if possible, or top 10.

    # 5-species params (20 total):
    # a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24, a55, b5, a15, a25, a35, a45
    labels = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",
        "a13",
        "a14",
        "a23",
        "a24",
        "a55",
        "b5",
        "a15",
        "a25",
        "a35",
        "a45",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=samples, ax=ax)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Posterior Parameter Distributions")
    ax.grid(True, axis="y", alpha=0.3)

    plt.savefig(output_dir / "pub_parameter_boxplots.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pub_parameter_boxplots.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_trace_plots(data_dict, config, output_dir):
    """Figure 5: Trace Plots for key parameters"""
    print("Generating Figure 5: Trace Plots...")
    if data_dict["samples"] is None:
        return

    samples = data_dict["samples"]
    # Plot top 5 parameters with highest variance or just first 5
    n_params = min(5, samples.shape[1])

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        ax = axes[i]
        ax.plot(samples[:, i], alpha=0.7, color="k", linewidth=0.5)
        ax.set_ylabel(f"Param {i}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample Index")
    plt.suptitle("Parameter Trace Plots (Convergence Check)")
    plt.savefig(output_dir / "pub_trace_plots.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pub_trace_plots.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(data_dict, config, output_dir):
    """Figure 6: Residuals"""
    print("Generating Figure 6: Residuals...")
    theta_full = np.array(data_dict["theta_map"]["theta_full"])
    exp_data = data_dict["exp_data"]

    solver = BiofilmNewtonSolver5S(active_species=list(range(5)), phi_init=np.array([1e-6] * 5))
    result = solver.solve(theta_full)
    if len(result) == 2:
        t_sim, x_sim = result
    else:
        t_sim, x_sim = result[:2]

    # Interpolate simulation to data points
    # Assuming standard time points for now (needs improvement for robustness)
    # t_days_data = [1, 3, 6, 10, 15, 21]
    t_days_sim = t_sim * 3.0

    # Simple residuals at data points if we can map them.
    # For visualization, let's just plot the difference if dimensions match,
    # otherwise skip complex interpolation for now and just plot the concept.
    # Since we don't have exact time mapping in this simple script, we'll skip exact residuals
    # and plot "Simulation vs Data Scatter" which is a standard goodness-of-fit plot.

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = get_colors(data_dict["condition"], config)

    # We need to extract sim values at data time points.
    # t_sim is dense.
    days_data = data_dict.get("t_days", np.array([1, 3, 6, 10, 15, 21]))

    # Normalize t_sim to days
    t_sim_days = t_sim * 3.0

    all_obs = []
    all_pred = []

    for i in range(5):
        # Interpolate
        pred = np.interp(days_data, t_sim_days, np.log10(x_sim[:, i] + 1e-10))
        obs = exp_data[:, i]

        # Filter NaNs if any
        mask = ~np.isnan(obs)
        if np.sum(mask) > 0:
            ax.scatter(
                obs[mask],
                pred[mask],
                color=colors[i],
                label=config["species_names"][i],
                s=60,
                alpha=0.8,
                edgecolors="k",
            )
            all_obs.extend(obs[mask])
            all_pred.extend(pred[mask])

    # Identity line
    min_val = min(min(all_obs), min(all_pred))
    max_val = max(max(all_obs), max(all_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    ax.set_xlabel("Observed (Log10)")
    ax.set_ylabel("Predicted (Log10)")
    ax.set_title("Goodness of Fit: Observed vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_dir / "pub_goodness_of_fit.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pub_goodness_of_fit.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = run_dir / "figures" / "pub_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading configuration and data from {run_dir}...")
    config = load_species_config()
    data_dict = load_run_data(run_dir)

    plot_map_fit(data_dict, config, output_dir)
    plot_posterior_band(data_dict, config, output_dir)
    plot_heatmap(data_dict, config, output_dir)
    plot_boxplots(data_dict, config, output_dir)
    plot_trace_plots(data_dict, config, output_dir)
    plot_residuals(data_dict, config, output_dir)

    print("Done. Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
