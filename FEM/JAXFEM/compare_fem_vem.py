#!/usr/bin/env python3
"""
compare_fem_vem.py — Run staggered coupling with both FEM and VEM solvers,
                     then generate a comparison figure.

Usage:
    python compare_fem_vem.py [--condition dh_baseline] [--nx 20] [--ny 20]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from core_hamilton_2d_nutrient import Config2D
from run_coupled_staggered import run_staggered_coupled


def run_comparison(condition, nx=20, ny=20, n_growth_steps=30, e_model="di"):
    """Run FEM and VEM for one condition, return both snapshots."""
    _TMCMC = _HERE.parent.parent
    _RUNS = _TMCMC / "data_5species" / "_runs"

    theta = None
    theta_path = _RUNS / condition / "theta_MAP.json"
    if theta_path.exists():
        with open(theta_path) as f:
            d = json.load(f)
        theta = np.array(d.get("theta_full", d.get("theta_sub")), dtype=np.float64)

    if theta is None:
        from core_hamilton_2d_nutrient import THETA_DEMO

        theta = THETA_DEMO
        print(f"  WARNING: No theta found for {condition}, using demo")

    cfg = Config2D(Nx=nx, Ny=ny)

    results = {}
    for solver in ["fem", "vem"]:
        print(f"\n{'='*60}")
        print(f"  Running {solver.upper()} for {condition} ({nx}×{ny})")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        snaps = run_staggered_coupled(
            theta,
            cfg,
            nu=0.30,
            k_alpha=0.05,
            e_model=e_model,
            dt_growth=0.1,
            n_growth_steps=n_growth_steps,
            ode_init_steps=2500,
            ode_adjust_steps=100,
            stress_type="plane_stress",
            solver=solver,
        )
        elapsed = time.perf_counter() - t0
        snaps["solver_time"] = elapsed
        results[solver] = snaps
        print(f"  {solver.upper()} done in {elapsed:.1f}s")

    return results


def plot_comparison(results, condition, outpath):
    """Generate FEM vs VEM comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fem = results["fem"]
    vem = results["vem"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    t_fem = fem["t"]
    t_vem = vem["t"]

    # Row 0: Time series comparison
    # (a) σ_vm max
    ax = axes[0, 0]
    ax.plot(t_fem, fem["sigma_vm_max"], "b-o", ms=3, lw=1.5, label="FEM (Q4)")
    ax.plot(t_vem, vem["sigma_vm_max"], "r-s", ms=3, lw=1.5, label="VEM (Voronoi)")
    ax.set_xlabel("Growth time")
    ax.set_ylabel("σ_vm,max [Pa]")
    ax.set_title("(a) Max von Mises stress")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) u max
    ax = axes[0, 1]
    ax.plot(t_fem, fem["u_max"], "b-o", ms=3, lw=1.5, label="FEM")
    ax.plot(t_vem, vem["u_max"], "r-s", ms=3, lw=1.5, label="VEM")
    ax.set_xlabel("Growth time")
    ax.set_ylabel("|u|_max [m]")
    ax.set_title("(b) Max displacement")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Relative error
    ax = axes[0, 2]
    n_common = min(len(fem["u_max"]), len(vem["u_max"]))
    u_fem_arr = np.array(fem["u_max"][:n_common])
    u_vem_arr = np.array(vem["u_max"][:n_common])
    svm_fem_arr = np.array(fem["sigma_vm_max"][:n_common])
    svm_vem_arr = np.array(vem["sigma_vm_max"][:n_common])

    # Avoid division by zero
    mask_u = u_fem_arr > 1e-15
    mask_s = svm_fem_arr > 1e-3

    rel_err_u = np.zeros(n_common)
    rel_err_s = np.zeros(n_common)
    rel_err_u[mask_u] = np.abs(u_vem_arr[mask_u] - u_fem_arr[mask_u]) / u_fem_arr[mask_u]
    rel_err_s[mask_s] = np.abs(svm_vem_arr[mask_s] - svm_fem_arr[mask_s]) / svm_fem_arr[mask_s]

    t_common = np.array(t_fem[:n_common]) if isinstance(t_fem, list) else t_fem[:n_common]
    ax.plot(t_common, rel_err_u * 100, "g-^", ms=3, lw=1.5, label="|u| rel. err.")
    ax.plot(t_common, rel_err_s * 100, "m-v", ms=3, lw=1.5, label="σ_vm rel. err.")
    ax.set_xlabel("Growth time")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("(c) VEM vs FEM relative error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 1: Final-state field comparison
    Nx, Ny = fem["Nx"], fem["Ny"]

    # (d) FEM σ_vm final
    ax = axes[1, 0]
    svm_fem_final = fem["sigma_vm"][-1]
    n_ex, n_ey = Nx - 1, Ny - 1
    if svm_fem_final.size == n_ex * n_ey:
        im = ax.imshow(
            svm_fem_final.reshape(n_ex, n_ey).T, origin="lower", cmap="hot", aspect="auto"
        )
        plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
    ax.set_title(f"(d) FEM σ_vm (final)")

    # (e) VEM σ_vm final
    ax = axes[1, 1]
    svm_vem_final = vem["sigma_vm"][-1]
    if svm_vem_final.size == n_ex * n_ey:
        im = ax.imshow(
            svm_vem_final.reshape(n_ex, n_ey).T, origin="lower", cmap="hot", aspect="auto"
        )
        plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
    ax.set_title(f"(e) VEM σ_vm (final)")

    # (f) Summary table
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"Condition: {condition}\n"
        f"Grid: {Nx}×{Ny}\n\n"
        f"FEM (Q4):\n"
        f"  u_max = {u_fem_arr[-1]:.3e} m\n"
        f"  σ_vm,max = {svm_fem_arr[-1]:.1f} Pa\n"
        f"  time = {fem['solver_time']:.1f}s\n\n"
        f"VEM (Voronoi):\n"
        f"  u_max = {u_vem_arr[-1]:.3e} m\n"
        f"  σ_vm,max = {svm_vem_arr[-1]:.1f} Pa\n"
        f"  time = {vem['solver_time']:.1f}s\n\n"
        f"Displacement agreement:\n"
        f"  max rel. err = {rel_err_u[mask_u].max()*100:.2f}%\n"
        f"Stress agreement:\n"
        f"  max rel. err = {rel_err_s[mask_s].max()*100:.1f}%"
    )
    ax.text(
        0.05,
        0.95,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("(f) Summary")

    fig.suptitle(f"FEM vs VEM Staggered Coupling — {condition}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    print(f"\n  Saved comparison: {outpath}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="dh_baseline")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--n-growth-steps", type=int, default=30)
    ap.add_argument("--e-model", default="di")
    ap.add_argument("--all", action="store_true", help="Run all 4 conditions")
    args = ap.parse_args()

    outdir = _HERE.parent / "figures" / "fem_vs_vem"
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = (
        ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
        if args.all
        else [args.condition]
    )

    all_results = {}
    for cond in conditions:
        results = run_comparison(cond, args.nx, args.ny, args.n_growth_steps, args.e_model)
        all_results[cond] = results
        plot_comparison(results, cond, outdir / f"fem_vs_vem_{cond}.png")

    # If multiple conditions, generate summary
    if len(all_results) > 1:
        _plot_multi_condition_summary(all_results, outdir / "fem_vs_vem_summary.png")


def _plot_multi_condition_summary(all_results, outpath):
    """4-condition comparison: FEM vs VEM displacement and stress."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "commensal_static": "#2196F3",
        "commensal_hobic": "#4CAF50",
        "dh_baseline": "#FF9800",
        "dysbiotic_static": "#F44336",
    }
    labels = {
        "commensal_static": "CS",
        "commensal_hobic": "CH",
        "dh_baseline": "DH",
        "dysbiotic_static": "DS",
    }

    for cond, results in all_results.items():
        col = colors.get(cond, "gray")
        lab = labels.get(cond, cond)
        fem_t = results["fem"]["t"]
        vem_t = results["vem"]["t"]

        axes[0].plot(fem_t, results["fem"]["u_max"], "-", color=col, lw=2, label=f"{lab} FEM")
        axes[0].plot(vem_t, results["vem"]["u_max"], "--", color=col, lw=2, label=f"{lab} VEM")

        axes[1].plot(
            fem_t, results["fem"]["sigma_vm_max"], "-", color=col, lw=2, label=f"{lab} FEM"
        )
        axes[1].plot(
            vem_t, results["vem"]["sigma_vm_max"], "--", color=col, lw=2, label=f"{lab} VEM"
        )

    axes[0].set_xlabel("Growth time")
    axes[0].set_ylabel("|u|_max [m]")
    axes[0].set_title("Max displacement: FEM (solid) vs VEM (dashed)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Growth time")
    axes[1].set_ylabel("σ_vm,max [Pa]")
    axes[1].set_title("Max von Mises stress: FEM vs VEM")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("FEM vs VEM Staggered Coupling — 4 Conditions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    print(f"\n  Saved summary: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
