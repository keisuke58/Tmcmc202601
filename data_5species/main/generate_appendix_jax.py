#!/usr/bin/env python3
"""
Appendix figures for JAX TMCMC results — same visual style as the paper figures.

Uses JAX ODE (same version as training) for trajectories, and the same
visual style as generate_pub_map_plot.py and generate_pub_plots.py.

Generates per-condition:
  - pub_map_fit_comparison.{png,pdf}   (same style as paper fitting figure)
  - pub_interaction_heatmap.{png,pdf}  (same style as paper A matrix figure)

Usage (on Vancouver02 with JAX 0.6.x):
    python generate_appendix_jax.py --output-dir figures_appendix
    python generate_appendix_jax.py --output-dir figures_appendix \
        --run-CH /path --run-CS /path --run-DH /path --run-DS /path
"""

import argparse
import json
import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hamilton_ode_jax import simulate_0d, theta_to_matrices
from estimate_reduced_nishioka import convert_days_to_model_time, load_experimental_data

# Exact same colors as generate_pub_map_plot.py
COLORS_BASE = ["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#d62728"]
SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]

CONDITIONS = [
    ("Commensal", "HOBIC", "CH"),
    ("Commensal", "Static", "CS"),
    ("Dysbiotic", "HOBIC", "DH"),
    ("Dysbiotic", "Static", "DS"),
]


def get_colors(condition: str):
    colors = list(COLORS_BASE)
    if "Dysbiotic" in condition:
        colors[2] = "#ff7f0e"
    return colors


def load_theta(run_dir: Path) -> np.ndarray:
    with open(run_dir / "theta_MAP.json") as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(20)], dtype=np.float64)


def best_run_for_condition(runs_dir: Path, condition: str, cultivation: str) -> Path:
    tag = f"jax_ode_nuts_{condition}_{cultivation}_"
    candidates = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir()
        and tag in p.name
        and (p / "theta_MAP.json").exists()
        and (p / "logL.npy").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed JAX run for {condition}_{cultivation}")
    return max(candidates, key=lambda p: np.load(p / "logL.npy").max())


def simulate_and_load(
    theta: np.ndarray,
    condition: str,
    cultivation: str,
    n_steps: int = 2500,
    dt: float = 1e-4,
    K_hill: float = 0.05,
    n_hill: float = 4.0,
):
    """Run JAX ODE and return simulation trajectory + experimental data."""
    data, t_days, _, phi_init_exp, _ = load_experimental_data(
        DATA_DIR, condition, cultivation, normalize=True
    )
    phi_init = phi_init_exp / phi_init_exp.sum()
    phi_init = np.clip(phi_init, 0.01, 0.99)

    t_model, idx_sparse = convert_days_to_model_time(t_days, dt, n_steps, day_scale=None)
    idx_sparse = np.clip(idx_sparse, 0, n_steps)

    phi_traj = simulate_0d(
        jnp.array(theta),
        n_steps=n_steps,
        dt=dt,
        phi_init=jnp.array(phi_init),
        K_hill=K_hill,
        n_hill=n_hill,
    )
    phi_traj = np.array(phi_traj)

    # Normalize to fractions (same as training)
    phi_sum = phi_traj.sum(axis=1, keepdims=True)
    phi_traj = phi_traj / np.maximum(phi_sum, 1e-12)

    rmse = float(np.sqrt(np.mean((phi_traj[idx_sparse] - data) ** 2)))

    # Map simulation steps to day axis (same as generate_pub_map_plot.py)
    t_sim = np.linspace(t_days.min(), t_days.max(), phi_traj.shape[0])

    return t_sim, phi_traj, data, t_days, rmse


def save_plot_with_preview(fig, output_path: Path):
    """Same save pattern as generate_pub_map_plot.py."""
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(
        output_path.with_name(output_path.stem + "_preview.png"), dpi=100, bbox_inches="tight"
    )
    print(f"Generated: {output_path.with_suffix('.png')} (+preview)")
    print(f"Generated: {output_path.with_suffix('.pdf')}")


def plot_map_fit_jax(
    theta: np.ndarray,
    condition: str,
    cultivation: str,
    output_dir: Path,
    K_hill: float = 0.05,
    n_hill: float = 4.0,
):
    """
    Generate fitting figure matching generate_pub_map_plot.py style exactly.
    Uses JAX ODE for trajectory computation.
    """
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    t_sim, phi_traj, data_points, t_days, rmse = simulate_and_load(
        theta, condition, cultivation, K_hill=K_hill, n_hill=n_hill
    )

    colors = get_colors(condition)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=False)

    for i in range(5):
        ax = axes[i]
        color = colors[i]

        ax.plot(t_sim, phi_traj[:, i], color=color, linewidth=3, label="MAP Model")
        ax.scatter(
            t_days,
            data_points[:, i],
            color=color,
            s=100,
            edgecolor="black",
            zorder=10,
            label="Experiment",
        )

        ax.set_title(SPECIES_NAMES[i], fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (Days)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Abundance (φ̄)", fontsize=12)

        ax.set_xticks(t_days)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

        if i == 0:
            ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plot_with_preview(fig, output_dir / "pub_map_fit_comparison")
    plt.close()
    print(f"  RMSE={rmse:.4f}")


def plot_interaction_heatmap_jax(theta: np.ndarray, condition: str, output_dir: Path):
    """
    Generate A matrix heatmap matching generate_pub_plots.py style exactly.
    """
    A_jax, _ = theta_to_matrices(jnp.array(theta))
    A = np.array(A_jax)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        A,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=SPECIES_NAMES,
        yticklabels=SPECIES_NAMES,
        ax=ax,
        square=True,
        cbar_kws={"label": "Interaction Strength"},
    )
    ax.set_title("Inferred Interaction Matrix (A)", fontsize=16)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / "pub_interaction_heatmap.pdf"), dpi=300)
    fig.savefig(str(output_dir / "pub_interaction_heatmap.png"), dpi=300)
    plt.close()
    print(f"Generated: {output_dir}/pub_interaction_heatmap.png")


def main():
    parser = argparse.ArgumentParser(
        description="Appendix figures using JAX ODE + paper visual style"
    )
    parser.add_argument("--runs-dir", default=None, help="Dir with jax_ode_nuts_* run dirs")
    parser.add_argument("--run-CH", default=None)
    parser.add_argument("--run-CS", default=None)
    parser.add_argument("--run-DH", default=None)
    parser.add_argument("--run-DS", default=None)
    parser.add_argument("--output-dir", default="figures_appendix")
    parser.add_argument("--n-hill", type=float, default=4.0)
    parser.add_argument("--K-hill", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    runs_dir = Path(args.runs_dir) if args.runs_dir else SCRIPT_DIR / "_runs"
    explicit = {"CH": args.run_CH, "CS": args.run_CS, "DH": args.run_DH, "DS": args.run_DS}

    for condition, cultivation, key in CONDITIONS:
        print(f"\n=== {key} ({condition} {cultivation}) ===")

        jax_dir = (
            Path(explicit[key])
            if explicit[key]
            else best_run_for_condition(runs_dir, condition, cultivation)
        )
        logL_best = np.load(jax_dir / "logL.npy").max()
        print(f"  Run: {jax_dir.name}  (logL={logL_best:.3f})")

        theta = load_theta(jax_dir)
        cond_out = output_dir / f"{condition}_{cultivation}"

        print("  Generating fitting figure...")
        plot_map_fit_jax(
            theta, condition, cultivation, cond_out, K_hill=args.K_hill, n_hill=args.n_hill
        )

        print("  Generating A matrix heatmap...")
        plot_interaction_heatmap_jax(theta, condition, cond_out)

    print(f"\nDone. All figures in: {output_dir}/")


if __name__ == "__main__":
    main()
