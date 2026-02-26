"""
Sigma（ノイズレベル）比較テスト

sigma を変えて、M1からM3まで全てまとめて図に保存
sigma: 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from visualization.plot_manager import PlotManager
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig
from core.tmcmc import _stable_hash_int

print("=" * 80)
print("Sigma（ノイズレベル）比較テスト")
print("=" * 80)
print()

# Sigma values to test
sigma_values = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.0]

print(f"Sigma values: {sigma_values}")
print()

# Configuration
exp_config = ExperimentConfig()
exp_config.cov_rel = 0.005
exp_config.n_data = 20
exp_config.random_seed = 42

theta_true = get_theta_true()

# Output directory
output_dir = Path("_runs/test_sigma_comparison")
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Store results for each model and sigma
results = {}

# ==============================================================================
# Generate data for each model and sigma
# ==============================================================================
for model_name in ["M1", "M2", "M3"]:
    print("=" * 80)
    print(f"Processing {model_name}")
    print("=" * 80)

    config = MODEL_CONFIGS[model_name]
    results[model_name] = {}

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

    tsm = BiofilmTSM_Analytical(
        solver,
        active_theta_indices=config["active_indices"],
        cov_rel=exp_config.cov_rel,
        use_complex_step=True,
        use_analytical=True,
        theta_linearization=theta_true,
    )

    # Generate TSM solution (same for all sigma values)
    t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
    phibar = compute_phibar(x0, config["active_species"])
    idx_sparse = select_sparse_data_indices(len(t_arr), exp_config.n_data, t_arr=t_arr)

    print(f"  t_arr shape: {t_arr.shape}")
    print(f"  phibar shape: {phibar.shape}")
    print(f"  idx_sparse: {idx_sparse[:5]} ... {idx_sparse[-5:]}")
    print()

    # Generate data for each sigma value
    for sigma in sigma_values:
        print(f"  Sigma = {sigma:.6f}...", end=" ")

        # Generate data with noise
        rng = np.random.default_rng(exp_config.random_seed + (_stable_hash_int(model_name) % 1000))
        data = np.zeros((exp_config.n_data, len(config["active_species"])))

        for i, sp in enumerate(config["active_species"]):
            if sigma == 0.0:
                data[:, i] = phibar[idx_sparse, i]  # No noise
            else:
                data[:, i] = phibar[idx_sparse, i] + rng.standard_normal(exp_config.n_data) * sigma

        # Calculate max difference
        phibar_at_obs = phibar[idx_sparse]
        max_diff = np.max(np.abs(data - phibar_at_obs))
        mean_diff = np.mean(np.abs(data - phibar_at_obs))

        results[model_name][sigma] = {
            "t_arr": t_arr,
            "x0": x0,
            "phibar": phibar,
            "data": data,
            "idx_sparse": idx_sparse,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }

        print(f"Max diff: {max_diff:.6f}")

    print()

# ==============================================================================
# Create comparison plots for each model
# ==============================================================================
print("=" * 80)
print("Creating comparison plots")
print("=" * 80)

for model_name in ["M1", "M2", "M3"]:
    print(f"\nCreating plots for {model_name}...")

    config = MODEL_CONFIGS[model_name]
    n_species = len(config["active_species"])

    # Create a large figure with subplots for each sigma value
    n_sigma = len(sigma_values)
    cols = 4
    rows = (n_sigma + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, sigma in enumerate(sigma_values):
        ax = axes[idx]
        result = results[model_name][sigma]

        t_arr = result["t_arr"]
        phibar = result["phibar"]
        data = result["data"]
        idx_sparse = result["idx_sparse"]
        max_diff = result["max_diff"]

        # Normalize time
        t_min = t_arr.min()
        t_max = t_arr.max()
        t_normalized = (t_arr - t_min) / (t_max - t_min) if t_max > t_min else t_arr
        t_obs_normalized = t_normalized[idx_sparse]

        # Plot all species
        for i, sp in enumerate(config["active_species"]):
            # Plot line
            ax.plot(
                t_normalized,
                phibar[:, i],
                label=f"phibar{sp+1}" if i == 0 else "",
                linewidth=1.5,
                alpha=0.7,
                color=plt.cm.tab10(i),
            )

            # Plot data points
            ax.scatter(
                t_obs_normalized,
                data[:, i],
                s=30,
                alpha=0.8,
                zorder=10,
                marker="x",
                color=plt.cm.tab10(i),
                linewidths=1.5,
            )

        ax.set_title(f"σ = {sigma:.6f}\nMax diff: {max_diff:.6f}", fontsize=10)
        ax.set_xlabel("Normalized Time", fontsize=9)
        ax.set_ylabel("phibar", fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(n_sigma, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"{model_name}: Sigma Comparison (All Species)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = figures_dir / f"{model_name}_sigma_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Saved: {fig_path}")

    # Create individual plots for each species
    for species_idx, sp in enumerate(config["active_species"]):
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, sigma in enumerate(sigma_values):
            ax = axes[idx]
            result = results[model_name][sigma]

            t_arr = result["t_arr"]
            phibar = result["phibar"]
            data = result["data"]
            idx_sparse = result["idx_sparse"]
            max_diff = result["max_diff"]

            # Normalize time
            t_min = t_arr.min()
            t_max = t_arr.max()
            t_normalized = (t_arr - t_min) / (t_max - t_min) if t_max > t_min else t_arr
            t_obs_normalized = t_normalized[idx_sparse]

            # Plot line
            ax.plot(
                t_normalized,
                phibar[:, species_idx],
                linewidth=2,
                alpha=0.7,
                color="blue",
                label="Model",
            )

            # Plot data points
            ax.scatter(
                t_obs_normalized,
                data[:, species_idx],
                s=50,
                alpha=0.9,
                zorder=10,
                marker="x",
                color="red",
                linewidths=2,
                label="Data",
            )

            # Calculate difference for this species
            phibar_at_obs = phibar[idx_sparse, species_idx]
            data_at_obs = data[:, species_idx]
            diff = np.abs(phibar_at_obs - data_at_obs)
            max_diff_sp = np.max(diff)

            ax.set_title(f"σ = {sigma:.6f}\nMax diff: {max_diff_sp:.6f}", fontsize=10)
            ax.set_xlabel("Normalized Time", fontsize=9)
            ax.set_ylabel(f"phibar{sp+1}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_sigma, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"{model_name} - Species {sp} (phibar{sp+1}): Sigma Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        fig_path = figures_dir / f"{model_name}_species{sp}_sigma_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✅ Saved: {fig_path}")

# ==============================================================================
# Create summary comparison plot
# ==============================================================================
print("\n" + "=" * 80)
print("Creating summary comparison plot")
print("=" * 80)

# Plot max difference vs sigma for all models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for model_idx, model_name in enumerate(["M1", "M2", "M3"]):
    ax = axes[model_idx]

    max_diffs = []
    mean_diffs = []
    sigmas_plot = []

    for sigma in sigma_values:
        if sigma in results[model_name]:
            max_diffs.append(results[model_name][sigma]["max_diff"])
            mean_diffs.append(results[model_name][sigma]["mean_diff"])
            sigmas_plot.append(sigma)

    ax.loglog(
        sigmas_plot, max_diffs, "o-", linewidth=2, markersize=8, label="Max difference", color="red"
    )
    ax.loglog(
        sigmas_plot,
        mean_diffs,
        "s-",
        linewidth=2,
        markersize=8,
        label="Mean difference",
        color="blue",
    )

    # Add reference line (sigma itself)
    ax.loglog(
        sigmas_plot, sigmas_plot, "--", linewidth=1, alpha=0.5, color="gray", label="σ (reference)"
    )

    ax.set_xlabel("Sigma (noise level)", fontsize=12)
    ax.set_ylabel("Difference", fontsize=12)
    ax.set_title(f"{model_name}: Difference vs Sigma", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)

plt.suptitle("Summary: Data-Model Difference vs Noise Level", fontsize=16, fontweight="bold")
plt.tight_layout()

fig_path = figures_dir / "summary_sigma_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  ✅ Saved: {fig_path}")

# ==============================================================================
# Save results summary
# ==============================================================================
print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)

summary_path = output_dir / "sigma_comparison_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Sigma Comparison Results\n")
    f.write("=" * 80 + "\n\n")

    for model_name in ["M1", "M2", "M3"]:
        f.write(f"{model_name}:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Sigma':<15} {'Max Diff':<15} {'Mean Diff':<15}\n")
        f.write("-" * 80 + "\n")

        for sigma in sigma_values:
            if sigma in results[model_name]:
                max_diff = results[model_name][sigma]["max_diff"]
                mean_diff = results[model_name][sigma]["mean_diff"]
                f.write(f"{sigma:<15.6f} {max_diff:<15.6f} {mean_diff:<15.6f}\n")

        f.write("\n")

print(f"  ✅ Saved: {summary_path}")

print("\n" + "=" * 80)
print("All figures saved to:")
print(f"  {figures_dir}")
print("=" * 80)
