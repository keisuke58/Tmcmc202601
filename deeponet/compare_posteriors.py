#!/usr/bin/env python3
"""
Compare DeepONet-TMCMC posterior vs ODE-TMCMC posterior.

Loads samples from both runs and generates comparison plots:
  - Marginal posteriors overlay (per active parameter)
  - DI / E summary statistics comparison
  - Trajectory comparison (MAP from each)
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPER_FIG_DIR = PROJECT_ROOT / "FEM" / "figures" / "paper_final"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]

# Condition → (ODE run dir, DeepONet run dir)
# Use _posterior runs which have full samples (from Feb 25 server runs)
CONDITION_DIRS = {
    "Commensal_Static": ("commensal_static_posterior", "deeponet_Commensal_Static"),
    "Commensal_HOBIC": ("commensal_hobic_posterior", "deeponet_Commensal_HOBIC"),
    "Dysbiotic_Static": ("dysbiotic_static_posterior", "deeponet_Dysbiotic_Static"),
    "Dysbiotic_HOBIC": ("dh_baseline", "deeponet_Dysbiotic_HOBIC"),
}

PARAM_NAMES = [
    "μ_So",
    "μ_An",
    "μ_Vd",
    "μ_Fn",
    "μ_Pg",
    "a₁₂",
    "a₁₃",
    "a₁₄",
    "a₂₁",
    "a₂₃",
    "a₂₄",
    "a₃₁",
    "a₃₂",
    "a₃₄",
    "a₄₁",
    "a₁₅",
    "a₂₅",
    "a₃₅_inv",
    "a₃₅",
    "a₄₅",
]


def load_samples(run_dir: Path):
    """Load posterior samples and logL from run dir."""
    # Try samples.npy
    for fname in ["samples.npy", "posterior/samples.npy"]:
        p = run_dir / fname
        if p.exists():
            samples = np.load(str(p))
            logL_path = run_dir / fname.replace("samples", "logL")
            logL = np.load(str(logL_path)) if logL_path.exists() else None
            return samples, logL

    # Try theta_MAP.json only
    for fname in ["theta_MAP.json", "posterior/theta_MAP.json"]:
        p = run_dir / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                theta = np.array(data["theta_full"])
            else:
                theta = np.array(data)
            return theta.reshape(1, -1), None

    return None, None


def compute_di(phi_final):
    """Shannon DI from final species fractions."""
    phi = np.clip(phi_final[:5], 1e-12, 1.0)
    phi = phi / phi.sum()
    H = -np.sum(phi * np.log(phi))
    H_max = np.log(5)
    return H / H_max


def plot_marginal_comparison(condition, ode_samples, don_samples, active_indices):
    """Plot marginal posterior comparison for active parameters."""
    n_active = len(active_indices)
    ncols = min(5, n_active)
    nrows = (n_active + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    axes = np.atleast_2d(axes)

    for k, idx in enumerate(active_indices):
        row, col = k // ncols, k % ncols
        ax = axes[row, col]

        if ode_samples.shape[0] > 1:
            ax.hist(
                ode_samples[:, idx],
                bins=30,
                alpha=0.5,
                density=True,
                color="#E53935",
                label="ODE-TMCMC",
            )
        else:
            ax.axvline(ode_samples[0, idx], color="#E53935", lw=2, label="ODE MAP")

        if don_samples.shape[0] > 1:
            ax.hist(
                don_samples[:, idx],
                bins=30,
                alpha=0.5,
                density=True,
                color="#1E88E5",
                label="DeepONet-TMCMC",
            )
        else:
            ax.axvline(don_samples[0, idx], color="#1E88E5", lw=2, ls="--", label="DeepONet MAP")

        ax.set_title(PARAM_NAMES[idx], fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7)

    # Hide empty subplots
    for k in range(n_active, nrows * ncols):
        row, col = k // ncols, k % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Posterior Comparison: {condition}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    runs_base = PROJECT_ROOT / "data_5species" / "_runs"

    # Load prior bounds to identify active indices
    bounds_file = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(bounds_file) as f:
        cfg = json.load(f)

    print("=" * 60)
    print("DeepONet vs ODE Posterior Comparison")
    print("=" * 60)

    for condition, (ode_dir_name, don_dir_name) in CONDITION_DIRS.items():
        print(f"\n--- {condition} ---")

        ode_dir = runs_base / ode_dir_name
        don_dir = runs_base / don_dir_name

        if not ode_dir.exists():
            print(f"  [SKIP] ODE run not found: {ode_dir}")
            continue
        if not don_dir.exists():
            print(f"  [SKIP] DeepONet run not found: {don_dir}")
            continue

        ode_samples, ode_logL = load_samples(ode_dir)
        don_samples, don_logL = load_samples(don_dir)

        if ode_samples is None:
            print("  [SKIP] No ODE samples found")
            continue
        if don_samples is None:
            print("  [SKIP] No DeepONet samples found")
            continue

        print(f"  ODE samples: {ode_samples.shape}")
        print(f"  DeepONet samples: {don_samples.shape}")

        # Identify active parameters
        strategy = cfg["strategies"].get(condition, {})
        locks = set(strategy.get("locks", []))
        active_indices = [i for i in range(20) if i not in locks]

        # MAP comparison
        if ode_logL is not None:
            ode_map = ode_samples[np.argmax(ode_logL)]
        else:
            ode_map = ode_samples[0]

        if don_logL is not None:
            don_map = don_samples[np.argmax(don_logL)]
        else:
            don_map = don_samples[0]

        # Parameter difference
        diff = np.abs(ode_map - don_map)
        rel_diff = diff / (np.abs(ode_map) + 1e-10) * 100
        print(f"  MAP parameter diff (max): {diff.max():.4f} ({rel_diff.max():.1f}%)")
        print(f"  MAP parameter diff (mean active): {diff[active_indices].mean():.4f}")

        # Marginal comparison figure
        fig = plot_marginal_comparison(condition, ode_samples, don_samples, active_indices)
        fig_path = PAPER_FIG_DIR / f"posterior_comparison_{condition}.png"
        fig.savefig(str(fig_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
