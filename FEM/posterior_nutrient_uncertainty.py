"""Issue #7: Posterior → Nutrient Uncertainty Propagation (2D)

50 posterior samples → Hamilton 0D → 2D biofilm profile → steady-state nutrient PDE
→ c_min distribution + spatial credible bands.

Quantifies how TMCMC parameter uncertainty propagates to nutrient depletion predictions.

Environment: klempt_fem (Python 3.11, jax 0.9.0.1)
Run: ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
     Tmcmc202601/FEM/posterior_nutrient_uncertainty.py
"""

import os
import sys
import time
import json

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# Import core functions from the 2D pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jax_hamilton_to_rd_2d_pipeline import (
    run_hamilton_0d,
    make_biofilm_profile_2d,
    solve_2d_nutrient,
    compute_di_2d,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 50
SEED = 42

# Hamilton 0D time integration (must match 1D multiscale pipeline scale)
T_FINAL = 25.0  # T* dimensionless time (same as multiscale_coupling_1d.py)
DT_H = 0.01  # timestep (same as multiscale_coupling_1d.py)

# Grid (same as 2D pipeline)
NX, NY = 40, 40
LX, LY = 1.0, 1.0
DEPTH_SCALE = 0.4
G_EFF = 50.0
D_C = 1.0
K_MONOD = 1.0
C_INF = 1.0

# TMCMC run directories
_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_5species", "_runs")

CONDITIONS = {
    "Commensal_Static": {
        "run": "Commensal_Static_20260208_002100",
        "color": "#2ca02c",
        "label": "Commensal",
    },
    "Dysbiotic_Static": {
        "run": "Dysbiotic_Static_20260207_203752",
        "color": "#d62728",
        "label": "Dysbiotic",
    },
}

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "klempt2024_results", "posterior_uncertainty"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_posterior_samples(condition_key, n_samples=N_SAMPLES, seed=SEED):
    """Load n_samples random posterior theta vectors from samples.npy."""
    run_name = CONDITIONS[condition_key]["run"]
    run_dir = os.path.join(_RUNS_DIR, run_name)
    samples = np.load(os.path.join(run_dir, "samples.npy"))  # (2000, 20)
    logL = np.load(os.path.join(run_dir, "logL.npy"))  # (2000,)
    rng = np.random.default_rng(seed)
    indices = rng.choice(samples.shape[0], size=n_samples, replace=False)
    # Also load MAP for reference
    with open(os.path.join(run_dir, "theta_MAP.json")) as f:
        d = json.load(f)
    theta_map = np.array(d.get("theta_sub", d) if isinstance(d, dict) else d, dtype=np.float64)[:20]
    return samples[indices], logL[indices], theta_map


# ---------------------------------------------------------------------------
# Ensemble run
# ---------------------------------------------------------------------------


def run_ensemble(condition_key):
    """Run the 2D pipeline for N_SAMPLES posterior samples.

    Returns dict with stacked results:
      c_stack      : (n, Nx, Ny) nutrient fields
      c_min_arr    : (n,)        minimum nutrient per sample
      phi_final_arr: (n, 5)      0D compositions
      phi_total_stack: (n, Nx, Ny) total biofilm fraction
      di_stack     : (n, Nx, Ny)  dysbiosis index
      theta_samples: (n, 20)
      theta_map    : (20,)
    """
    x_grid = np.linspace(0, LX, NX)
    y_grid = np.linspace(0, LY, NY)

    samples, logL, theta_map = load_posterior_samples(condition_key)
    n = samples.shape[0]

    c_stack = np.zeros((n, NX, NY))
    phi_final_arr = np.zeros((n, 5))
    phi_total_stack = np.zeros((n, NX, NY))
    di_stack = np.zeros((n, NX, NY))

    for i in range(n):
        theta = jnp.array(samples[i])
        # Step 1: Hamilton 0D (T*=25, matching multiscale_coupling_1d.py)
        _, phi_final = run_hamilton_0d(theta, t_final=T_FINAL, dt_h=DT_H)
        phi_final_arr[i] = phi_final

        # Step 2: 2D spatial profile
        phi_sp, phi_tot = make_biofilm_profile_2d(
            phi_final, x_grid, y_grid, depth_scale=DEPTH_SCALE
        )
        phi_total_stack[i] = phi_tot
        di_stack[i] = compute_di_2d(phi_sp)

        # Step 3: 2D nutrient PDE
        c_sol = solve_2d_nutrient(
            phi_tot, x_grid, y_grid, D_c=D_C, k_monod=K_MONOD, g_eff=G_EFF, c_inf=C_INF, n_newton=40
        )
        c_stack[i] = c_sol

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{n}] c_min={c_sol.min():.4f}, " f"Pg={phi_final[4]:.4f}")

    return {
        "c_stack": c_stack,
        "c_min_arr": c_stack.min(axis=(1, 2)),
        "phi_final_arr": phi_final_arr,
        "phi_total_stack": phi_total_stack,
        "di_stack": di_stack,
        "theta_samples": samples,
        "theta_map": theta_map,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cmin_distribution(results, out_dir):
    """Figure 1: c_min histogram with KDE for each condition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (cond_key, res) in enumerate(results.items()):
        ax = axes[ax_idx]
        cfg = CONDITIONS[cond_key]
        c_min = res["c_min_arr"]
        color = cfg["color"]

        ax.hist(
            c_min,
            bins=15,
            density=True,
            alpha=0.6,
            color=color,
            edgecolor="white",
            label=f"Posterior ({N_SAMPLES} samples)",
        )
        ax.axvline(
            np.median(c_min), color=color, ls="--", lw=2, label=f"Median: {np.median(c_min):.3f}"
        )
        ax.axvline(
            np.percentile(c_min, 5),
            color=color,
            ls=":",
            lw=1.5,
            label=f"5th %ile: {np.percentile(c_min, 5):.3f}",
        )
        ax.axvline(
            np.percentile(c_min, 95),
            color=color,
            ls=":",
            lw=1.5,
            label=f"95th %ile: {np.percentile(c_min, 95):.3f}",
        )

        # MAP c_min
        c_min_map = res.get("c_min_map")
        if c_min_map is not None:
            ax.axvline(c_min_map, color="black", ls="-", lw=2, label=f"MAP: {c_min_map:.3f}")

        ax.set_xlabel("c_min (minimum nutrient concentration)")
        ax.set_ylabel("Density")
        ax.set_title(f"{cfg['label']}: c_min distribution")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Posterior → Nutrient Uncertainty (N={N_SAMPLES}, g_eff={G_EFF})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "fig1_cmin_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_spatial_credible_bands(results, out_dir):
    """Figure 2: Cross-section c(x) at y=0.5 with 90% credible band."""
    x_grid = np.linspace(0, LX, NX)
    j_mid = NY // 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (cond_key, res) in enumerate(results.items()):
        ax = axes[ax_idx]
        cfg = CONDITIONS[cond_key]
        c_stack = res["c_stack"]  # (n, Nx, Ny)
        c_cross = c_stack[:, :, j_mid]  # (n, Nx)

        p05 = np.percentile(c_cross, 5, axis=0)
        p25 = np.percentile(c_cross, 25, axis=0)
        p50 = np.percentile(c_cross, 50, axis=0)
        p75 = np.percentile(c_cross, 75, axis=0)
        p95 = np.percentile(c_cross, 95, axis=0)

        ax.fill_between(x_grid, p05, p95, alpha=0.15, color=cfg["color"], label="90% CI")
        ax.fill_between(x_grid, p25, p75, alpha=0.3, color=cfg["color"], label="50% CI")
        ax.plot(x_grid, p50, color=cfg["color"], lw=2, label="Median")

        ax.set_xlabel("x (0=tooth, 1=saliva)")
        ax.set_ylabel("Nutrient c(x, y=0.5)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{cfg['label']}: Nutrient Credible Band")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Spatial Uncertainty: c(x) cross-section at y=Ly/2\n"
        f"({N_SAMPLES} posterior samples, g_eff={G_EFF})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(out_dir, "fig2_spatial_credible_bands.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_both_conditions_overlay(results, out_dir):
    """Figure 3: Overlay both conditions on one axis with credible bands."""
    x_grid = np.linspace(0, LX, NX)
    j_mid = NY // 2
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond_key, res in results.items():
        cfg = CONDITIONS[cond_key]
        c_cross = res["c_stack"][:, :, j_mid]  # (n, Nx)
        p05 = np.percentile(c_cross, 5, axis=0)
        p50 = np.percentile(c_cross, 50, axis=0)
        p95 = np.percentile(c_cross, 95, axis=0)

        ax.fill_between(x_grid, p05, p95, alpha=0.15, color=cfg["color"])
        ax.plot(x_grid, p50, color=cfg["color"], lw=2, label=cfg["label"])

    ax.set_xlabel("x (0=tooth, 1=saliva)")
    ax.set_ylabel("Nutrient c(x, y=0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Commensal vs Dysbiotic: Nutrient with 90% CI\n" f"(N={N_SAMPLES}, g_eff={G_EFF})"
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_overlay_credible_bands.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_theta_vs_cmin_scatter(results, out_dir):
    """Figure 4: Scatter of key theta params vs c_min (Spearman correlation)."""
    # Key parameters for Pg ecology
    param_indices = [18, 19, 14, 15, 5]  # a35, a45, a55, b5, a33
    param_names = ["a35 (Vd→Pg)", "a45 (Fn→Pg)", "a55 (Pg self)", "b5 (Pg growth)", "a33 (Vd self)"]
    n_params = len(param_indices)

    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8), squeeze=False)

    for row, (cond_key, res) in enumerate(results.items()):
        cfg = CONDITIONS[cond_key]
        c_min = res["c_min_arr"]
        theta = res["theta_samples"]

        for col, (pidx, pname) in enumerate(zip(param_indices, param_names)):
            ax = axes[row, col]
            x = theta[:, pidx]
            ax.scatter(x, c_min, s=15, alpha=0.6, color=cfg["color"])

            # Spearman rank correlation
            from scipy.stats import spearmanr

            rho, pval = spearmanr(x, c_min)
            ax.set_title(f"{cfg['label']}\nρ={rho:.2f} (p={pval:.2e})", fontsize=9)
            ax.set_xlabel(pname, fontsize=8)
            if col == 0:
                ax.set_ylabel("c_min")

    plt.suptitle("Parameter Sensitivity: θ vs c_min", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "fig4_theta_vs_cmin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_2d_uncertainty_map(results, out_dir):
    """Figure 5: 2D heatmap of median c and CI width."""
    x_grid = np.linspace(0, LX, NX)
    y_grid = np.linspace(0, LY, NY)
    Y, X = np.meshgrid(y_grid, x_grid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, (cond_key, res) in enumerate(results.items()):
        cfg = CONDITIONS[cond_key]
        c_stack = res["c_stack"]  # (n, Nx, Ny)
        c_median = np.median(c_stack, axis=0)
        c_width = np.percentile(c_stack, 95, axis=0) - np.percentile(c_stack, 5, axis=0)

        # Median
        ax = axes[0, col]
        im = ax.pcolormesh(X, Y, c_median, cmap="plasma", shading="auto", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{cfg['label']}: Median c(x,y)")
        ax.set_xlabel("x (depth)")
        ax.set_ylabel("y (lateral)")
        ax.set_aspect("equal")

        # CI width
        ax = axes[1, col]
        im = ax.pcolormesh(X, Y, c_width, cmap="YlOrRd", shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8, label="90% CI width")
        ax.set_title(f"{cfg['label']}: CI Width (p95-p05)")
        ax.set_xlabel("x (depth)")
        ax.set_ylabel("y (lateral)")
        ax.set_aspect("equal")

    plt.suptitle(
        f"2D Nutrient Uncertainty Map (N={N_SAMPLES}, g_eff={G_EFF})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "fig5_2d_uncertainty_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Issue #7: Posterior → Nutrient Uncertainty Propagation (2D)")
    print(f"  N_SAMPLES={N_SAMPLES}, g_eff={G_EFF}, grid={NX}x{NY}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    x_grid = np.linspace(0, LX, NX)
    y_grid = np.linspace(0, LY, NY)

    for cond_key, cfg in CONDITIONS.items():
        print(f"\n--- {cfg['label']} ({cond_key}) ---")
        print(f"  Run: {cfg['run']}")
        t0 = time.time()
        res = run_ensemble(cond_key)
        elapsed = time.time() - t0
        print(f"  Ensemble done in {elapsed:.1f}s")

        # Also run MAP for reference
        theta_map = jnp.array(res["theta_map"])
        _, phi_map = run_hamilton_0d(theta_map, t_final=T_FINAL, dt_h=DT_H)
        _, phi_tot_map = make_biofilm_profile_2d(phi_map, x_grid, y_grid, depth_scale=DEPTH_SCALE)
        c_map = solve_2d_nutrient(
            phi_tot_map, x_grid, y_grid, D_c=D_C, k_monod=K_MONOD, g_eff=G_EFF, c_inf=C_INF
        )
        res["c_min_map"] = float(c_map.min())

        # Summary statistics
        c_min = res["c_min_arr"]
        print("\n  c_min statistics:")
        print(f"    MAP:    {res['c_min_map']:.4f}")
        print(f"    Median: {np.median(c_min):.4f}")
        print(f"    Mean:   {np.mean(c_min):.4f}")
        print(f"    Std:    {np.std(c_min):.4f}")
        print(f"    5th:    {np.percentile(c_min, 5):.4f}")
        print(f"    95th:   {np.percentile(c_min, 95):.4f}")
        print(f"    Range:  [{c_min.min():.4f}, {c_min.max():.4f}]")

        results[cond_key] = res

    # --- Plotting ---
    print("\n[Plotting] ...")
    plot_cmin_distribution(results, OUT_DIR)
    plot_spatial_credible_bands(results, OUT_DIR)
    plot_both_conditions_overlay(results, OUT_DIR)
    plot_theta_vs_cmin_scatter(results, OUT_DIR)
    plot_2d_uncertainty_map(results, OUT_DIR)

    # --- Save numerical results ---
    summary = {}
    for cond_key, res in results.items():
        c_min = res["c_min_arr"]
        summary[cond_key] = {
            "n_samples": int(N_SAMPLES),
            "g_eff": G_EFF,
            "c_min_MAP": res["c_min_map"],
            "c_min_median": float(np.median(c_min)),
            "c_min_mean": float(np.mean(c_min)),
            "c_min_std": float(np.std(c_min)),
            "c_min_p05": float(np.percentile(c_min, 5)),
            "c_min_p95": float(np.percentile(c_min, 95)),
            "c_min_range": [float(c_min.min()), float(c_min.max())],
        }
    summary_path = os.path.join(OUT_DIR, "uncertainty_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    # Save raw c_min arrays
    for cond_key, res in results.items():
        np.save(os.path.join(OUT_DIR, f"c_min_{cond_key}.npy"), res["c_min_arr"])
        np.save(os.path.join(OUT_DIR, f"c_stack_{cond_key}.npy"), res["c_stack"])

    print(f"\nAll results in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
