"""Issue #6: a35 Sweep Sensitivity — Pg vs c_min Nonlinear Relationship

Sweep a35 (Vd→Pg coupling) from 0 to 25 at the Dysbiotic MAP,
keeping all other parameters fixed.

For each a35:
  Hamilton 0D (T*=25) → 2D biofilm profile → steady-state nutrient PDE
  → track Pg fraction, phi_total, c_min, DI

Demonstrates the nonlinear threshold behavior of Pg emergence
and its downstream effect on nutrient depletion.

Environment: klempt_fem (Python 3.11, jax 0.9.0.1)
Run: ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
     Tmcmc202601/FEM/a35_sweep_sensitivity.py
"""

import os
import sys
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jax_hamilton_to_rd_2d_pipeline import (
    run_hamilton_0d,
    make_biofilm_profile_2d,
    solve_2d_nutrient,
    SPECIES_NAMES,
    SPECIES_COLORS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
A35_VALUES = np.linspace(0, 25, 51)  # 51 points: 0, 0.5, 1.0, ..., 25.0

# Hamilton 0D time integration (matching multiscale_coupling_1d.py)
T_FINAL = 25.0
DT_H = 0.01

# 2D grid
NX, NY = 40, 40
LX, LY = 1.0, 1.0
DEPTH_SCALE = 0.4
G_EFF = 50.0
D_C = 1.0
K_MONOD = 1.0
C_INF = 1.0

# Base theta: Dysbiotic Static MAP
_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_5species", "_runs")

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "klempt2024_results", "a35_sweep"
)


def load_theta_map(run_name):
    """Load MAP theta from a TMCMC run."""
    path = os.path.join(_RUNS_DIR, run_name, "theta_MAP.json")
    with open(path) as f:
        d = json.load(f)
    theta = d.get("theta_sub", d) if isinstance(d, dict) else d
    return np.array(theta[:20], dtype=np.float64)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_a35_sweep():
    """Sweep a35 and compute observables."""
    theta_base = load_theta_map("Dysbiotic_Static_20260207_203752")
    a35_original = theta_base[18]
    print(f"  Base theta: Dysbiotic Static MAP (a35_original = {a35_original:.4f})")

    x_grid = np.linspace(0, LX, NX)
    y_grid = np.linspace(0, LY, NY)

    n_vals = len(A35_VALUES)
    results = {
        "a35": A35_VALUES.copy(),
        "pg_final": np.zeros(n_vals),
        "phi_total_mean": np.zeros(n_vals),
        "phi_final_all": np.zeros((n_vals, 5)),
        "c_min": np.zeros(n_vals),
        "c_tooth": np.zeros(n_vals),  # c at (x=0, y=Ly/2)
        "c_mean": np.zeros(n_vals),
        "di_0d": np.zeros(n_vals),  # DI from 0D composition
    }

    for i, a35 in enumerate(A35_VALUES):
        theta = theta_base.copy()
        theta[18] = a35
        theta_jnp = jnp.array(theta)

        # Step 1: Hamilton 0D
        _, phi_final = run_hamilton_0d(theta_jnp, t_final=T_FINAL, dt_h=DT_H)

        # Step 2: 2D spatial profile
        phi_sp, phi_tot = make_biofilm_profile_2d(
            phi_final, x_grid, y_grid, depth_scale=DEPTH_SCALE
        )

        # Step 3: 2D nutrient PDE
        c_sol = solve_2d_nutrient(
            phi_tot, x_grid, y_grid, D_c=D_C, k_monod=K_MONOD, g_eff=G_EFF, c_inf=C_INF, n_newton=40
        )

        # DI from 0D composition
        p = phi_final / max(phi_final.sum(), 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        H = -np.sum(p * np.log(p))
        di_0d = max(0.0, 1.0 - H / np.log(5.0))

        results["pg_final"][i] = phi_final[4]
        results["phi_total_mean"][i] = float(phi_tot.mean())
        results["phi_final_all"][i] = phi_final
        results["c_min"][i] = float(c_sol.min())
        results["c_tooth"][i] = float(c_sol[0, NY // 2])
        results["c_mean"][i] = float(c_sol.mean())
        results["di_0d"][i] = di_0d

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    [{i+1}/{n_vals}] a35={a35:.1f}: "
                f"Pg={phi_final[4]:.4f}, c_min={c_sol.min():.4f}, "
                f"DI={di_0d:.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweep(results, out_dir):
    a35 = results["a35"]
    pg = results["pg_final"]
    c_min = results["c_min"]
    c_tooth = results["c_tooth"]
    di = results["di_0d"]
    phi_total = results["phi_total_mean"]

    # ---- Figure 1: 4-panel overview ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0): Pg vs a35
    ax = axes[0, 0]
    ax.plot(a35, pg, "o-", color="#9467bd", markersize=3, lw=1.5)
    ax.set_xlabel("a₃₅ (Vd→Pg coupling)")
    ax.set_ylabel("P. gingivalis fraction (φ_Pg)")
    ax.set_title("Pg Abundance vs a₃₅")
    ax.axhline(y=0.01, color="gray", ls=":", alpha=0.5, label="Pg = 1%")
    ax.legend(fontsize=8)

    # (0,1): c_min vs a35
    ax = axes[0, 1]
    ax.plot(a35, c_min, "s-", color="#d62728", markersize=3, lw=1.5)
    ax.set_xlabel("a₃₅ (Vd→Pg coupling)")
    ax.set_ylabel("c_min (minimum nutrient)")
    ax.set_title("Nutrient Depletion vs a₃₅")

    # (1,0): DI vs a35
    ax = axes[1, 0]
    ax.plot(a35, di, "D-", color="#ff7f0e", markersize=3, lw=1.5)
    ax.set_xlabel("a₃₅ (Vd→Pg coupling)")
    ax.set_ylabel("Dysbiotic Index (DI)")
    ax.set_title("Dysbiosis vs a₃₅")

    # (1,1): c_min vs Pg (the key nonlinear relationship)
    ax = axes[1, 1]
    sc = ax.scatter(pg, c_min, c=a35, cmap="viridis", s=30, edgecolors="k", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="a₃₅", shrink=0.8)
    ax.set_xlabel("P. gingivalis fraction (φ_Pg)")
    ax.set_ylabel("c_min (minimum nutrient)")
    ax.set_title("c_min vs Pg (colored by a₃₅)")

    plt.suptitle(
        "Issue #6: a₃₅ Sweep Sensitivity (Dysbiotic Static MAP)\n"
        f"a₃₅ ∈ [0, 25], {len(a35)} points, T*=25, g_eff={G_EFF}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "fig1_a35_sweep_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ---- Figure 2: Species composition stacked area ----
    fig, ax = plt.subplots(figsize=(10, 5))
    phi_all = results["phi_final_all"]  # (n, 5)
    # Stacked area plot
    y_bottom = np.zeros(len(a35))
    for s in range(5):
        ax.fill_between(
            a35,
            y_bottom,
            y_bottom + phi_all[:, s],
            color=SPECIES_COLORS[s],
            alpha=0.7,
            label=SPECIES_NAMES[s],
        )
        y_bottom += phi_all[:, s]
    ax.set_xlabel("a₃₅ (Vd→Pg coupling)")
    ax.set_ylabel("Volume fraction")
    ax.set_title("Species Composition vs a₃₅ (Stacked Area)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_species_composition_stacked.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ---- Figure 3: Dual-axis Pg and c_min ----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_pg = "#9467bd"
    color_c = "#d62728"

    ax1.plot(a35, pg, "o-", color=color_pg, markersize=3, lw=1.5, label="Pg")
    ax1.set_xlabel("a₃₅ (Vd→Pg coupling)", fontsize=11)
    ax1.set_ylabel("P. gingivalis fraction (φ_Pg)", color=color_pg, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_pg)

    ax2 = ax1.twinx()
    ax2.plot(a35, c_min, "s-", color=color_c, markersize=3, lw=1.5, label="c_min")
    ax2.set_ylabel("c_min (minimum nutrient)", color=color_c, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_c)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(
        f"a₃₅ Sweep: Pg Abundance & Nutrient Depletion\n"
        f"(Dysbiotic Static MAP, T*=25, g_eff={G_EFF})"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_dual_axis_pg_cmin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Issue #6: a35 Sweep Sensitivity")
    print(f"  a35 range: [{A35_VALUES[0]:.1f}, {A35_VALUES[-1]:.1f}], " f"n={len(A35_VALUES)}")
    print(f"  T*={T_FINAL}, g_eff={G_EFF}, grid={NX}x{NY}")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    results = run_a35_sweep()
    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f}s")

    # Plot
    print("\n[Plotting] ...")
    plot_sweep(results, OUT_DIR)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "a35_sweep_results.csv")
    header = "a35,pg_final,phi_total_mean,c_min,c_tooth,c_mean,di_0d"
    data = np.column_stack(
        [
            results["a35"],
            results["pg_final"],
            results["phi_total_mean"],
            results["c_min"],
            results["c_tooth"],
            results["c_mean"],
            results["di_0d"],
        ]
    )
    np.savetxt(csv_path, data, header=header, delimiter=",", fmt="%.6f", comments="")
    print(f"  CSV saved: {csv_path}")

    # Summary
    print("\n--- Key Observations ---")
    pg = results["pg_final"]
    threshold_idx = np.argmax(pg > 0.05)  # first index where Pg > 5%
    if pg[threshold_idx] > 0.05:
        print(f"  Pg > 5% threshold: a35 ≈ {results['a35'][threshold_idx]:.1f}")
    else:
        print("  Pg never exceeds 5%")

    max_pg_idx = np.argmax(pg)
    print(f"  Max Pg = {pg[max_pg_idx]:.4f} at a35 = {results['a35'][max_pg_idx]:.1f}")
    print(f"  c_min range: [{results['c_min'].min():.4f}, {results['c_min'].max():.4f}]")
    print(f"  DI range: [{results['di_0d'].min():.3f}, {results['di_0d'].max():.3f}]")

    print(f"\nAll results in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
