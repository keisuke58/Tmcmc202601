import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from pathlib import Path
import sys

# Set plotting style for publication
plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["lines.linewidth"] = 2.0


def load_data(run_dir):
    run_path = Path(run_dir)
    analysis_path = run_path / "analysis"

    # Load UQ results
    uq_data = np.load(analysis_path / "uncertainty_quantification.npz")

    # Load raw history for spaghetti if available
    history_path = analysis_path / "raw_history_subset.npy"
    raw_history = np.load(history_path) if history_path.exists() else None

    # Load observed data
    data_path = run_path / "data.npy"
    data = np.load(data_path)

    # Load time points
    t_path = run_path / "t_days.npy"
    t_days = np.load(t_path)

    # Load config for condition
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        condition = config.get("condition", "Commensal")
    else:
        condition = "Commensal"

    return uq_data, raw_history, data, t_days, condition


def plot_publication_fit(run_dir, output_dir):
    uq_data, raw_history, data, t_days, condition = load_data(run_dir)
    time = uq_data["time"]

    # Convert simulation steps to days (approximate scaling)
    sim_days = np.linspace(0, np.max(t_days), len(time))

    species_names = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]

    # BIOLOGICALLY CORRECT COLORS (Socransky Complexes)
    # S1 (S. oralis) -> Blue (Health/Early)
    # S2 (A. naeslundii) -> Green (Early)
    # S3 (V. dispar) -> Yellow (Commensal) / Orange (Dysbiotic)
    # S4 (F. nucleatum) -> Purple (Bridge)
    # S5 (P. gingivalis) -> Red (Red Complex)

    colors = [
        "#1f77b4",  # S1: Blue
        "#2ca02c",  # S2: Green
        "#bcbd22",  # S3: Yellow (Default/Commensal)
        "#9467bd",  # S4: Purple
        "#d62728",  # S5: Red
    ]

    # Adjust S3 color for Dysbiotic
    if "Dysbiotic" in condition:
        colors[2] = "#ff7f0e"  # Orange

    # 1. Combined Spaghetti Plot with CI Band
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for i, ax in enumerate(axes):
        species = f"S{i+1}"
        name = species_names[i]
        color = colors[i]

        # Plot spaghetti lines (thin, transparent)
        if raw_history is not None:
            for sample_idx in range(min(50, raw_history.shape[0])):
                ax.plot(
                    sim_days, raw_history[sample_idx, :, i], color=color, alpha=0.1, linewidth=0.5
                )

        # Plot Median
        ax.plot(sim_days, uq_data[f"{species}_p50"], color=color, linewidth=2.5, label="Median")

        # Plot 95% CI
        ax.fill_between(
            sim_days,
            uq_data[f"{species}_p2.5"],
            uq_data[f"{species}_p97.5"],
            color=color,
            alpha=0.2,
            label="95% CI",
        )

        # Plot Observed Data
        ax.errorbar(
            t_days,
            data[:, i],
            yerr=0.05,
            fmt="o",
            color="black",
            ecolor="black",
            capsize=3,
            markersize=6,
            label="Observed",
        )

        ax.set_title(name, style="italic", color=color, fontweight="bold")
        ax.set_xlabel("Time (days)")
        if i == 0:
            ax.set_ylabel("Relative Abundance")
            ax.legend(loc="upper right")

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pub_fit_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/pub_fit_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Generated: {output_dir}/pub_fit_comparison.png")


def plot_interaction_heatmap(run_dir, output_dir):
    run_path = Path(run_dir)
    with open(run_path / "theta_MAP.json", "r") as f:
        map_data = json.load(f)
    theta = np.array(map_data["theta_full"])

    # Use standard project path resolution
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    sys.path.insert(0, str(project_root / "tmcmc" / "program2602"))

    try:
        try:
            from tmcmc.program2602.improved_5species_jit import BiofilmNewtonSolver5S
        except ImportError:
            from improved_5species_jit import BiofilmNewtonSolver5S

        solver = BiofilmNewtonSolver5S()
        A, _ = solver.theta_to_matrices(theta)

        species_names = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]

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
        plt.savefig(f"{output_dir}/pub_interaction_heatmap.pdf", dpi=300)
        plt.savefig(f"{output_dir}/pub_interaction_heatmap.png", dpi=300)
        print(f"Generated: {output_dir}/pub_interaction_heatmap.png")

    except ImportError:
        print("Could not import solver to reconstruct matrix. Skipping heatmap.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_pub_plots.py <run_dir> <output_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_dir = sys.argv[2]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_publication_fit(run_dir, output_dir)
    plot_interaction_heatmap(run_dir, output_dir)
