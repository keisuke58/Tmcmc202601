#!/usr/bin/env python3
"""
run_2d_stress_hybrid.py — 2D FEM stress from hybrid CSV (0D DI + 1D spatial)
=============================================================================

Reads pre-computed hybrid CSV (from generate_hybrid_macro_csv.py) and runs
2D FEM stress analysis for all conditions.

The hybrid approach preserves condition-dependent E(DI) from 0D while using
spatial alpha_Monod(x) from the 1D PDE.

Usage:
    python run_2d_stress_hybrid.py
    python run_2d_stress_hybrid.py --e-model virulence
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "JAXFEM"))

from JAXFEM.solve_stress_2d import solve_2d_fem

_CSV_DIR = _HERE / "_multiscale_results"
_OUTDIR = _HERE / "_stress_2d_hybrid"

CONDITIONS = {
    "commensal_static": {"label": "Commensal Static", "color": "#1f77b4"},
    "commensal_hobic": {"label": "Commensal HOBIC", "color": "#2ca02c"},
    "dysbiotic_static": {"label": "Dysbiotic Static", "color": "#ff7f0e"},
    "dysbiotic_hobic": {"label": "Dysbiotic HOBIC", "color": "#d62728"},
}

E_MODEL_COLS = {
    "di": "E_di",
    "phi_pg": "E_phi_pg",
    "virulence": "E_virulence",
}


def load_hybrid_csv(condition):
    """Load hybrid CSV and parse into arrays."""
    path = _CSV_DIR / f"macro_eigenstrain_{condition}_hybrid.csv"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None

    lines = [l.strip() for l in open(path) if not l.startswith("#") and l.strip()]
    header = lines[0].split(",")
    data = np.array([[float(x) for x in l.split(",")] for l in lines[1:]])

    result = {}
    for i, col in enumerate(header):
        result[col] = data[:, i]
    return result


def extrude_1d_to_2d(field_1d, Ny):
    """Extrude 1D depth profile to 2D: uniform in y."""
    Nx = len(field_1d)
    return np.tile(field_1d[:, None], (1, Ny))


def run_condition(condition, e_model, Ny=20, nu=0.30):
    """Run 2D FEM for one condition using hybrid CSV data."""
    data = load_hybrid_csv(condition)
    if data is None:
        return None

    Nx = len(data["depth_norm"])
    depth_norm = data["depth_norm"]
    Lx = 1.0  # normalized depth

    # E field from chosen model (0D-based, spatially uniform in 1D → uniform in 2D)
    e_col = E_MODEL_COLS[e_model]
    E_1d = data[e_col]
    E_2d = extrude_1d_to_2d(E_1d, Ny)

    # Eigenstrain from 1D spatial α_Monod
    eps_1d = data["eps_growth"]
    eps_2d = extrude_1d_to_2d(eps_1d, Ny)

    # Extract metadata from CSV comments
    E_val = float(E_1d.mean())
    eps_max = float(eps_1d.max())

    print(f"\n  Condition: {condition}")
    print(f"  Grid: {Nx}×{Ny}")
    print(f"  E ({e_model}): {E_val:.1f} Pa (uniform in space)")
    print(f"  ε_growth: min={eps_1d.min():.6f}, max={eps_max:.6f}")

    # FEM solve
    result = solve_2d_fem(E_2d, nu, eps_2d, Nx, Ny, Lx, 1.0,
                          bc_type="bottom_fixed")

    # Add metadata
    result["condition"] = condition
    result["e_model"] = e_model
    result["E_mean"] = E_val
    result["eps_growth_max"] = eps_max
    result["Nx"] = Nx
    result["Ny"] = Ny
    result["depth_norm"] = depth_norm
    result["E_1d"] = E_1d
    result["eps_1d"] = eps_1d

    svm = result["sigma_vm"]
    print(f"  σ_vm: mean={svm.mean():.4f} Pa, max={svm.max():.4f} Pa")
    print(f"  |u|_max: {np.max(np.sqrt(result['u'][:, 0]**2 + result['u'][:, 1]**2)):.6e}")

    return result


def plot_all_conditions(results, outdir, e_model):
    """Publication-quality comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    n = len(conds)
    if n == 0:
        return

    fig, axes = plt.subplots(2, max(n, 2), figsize=(5 * max(n, 2), 10))
    if n == 1:
        axes = axes[:, :1]

    # Top row: σ_vm spatial fields
    vmax_global = max(r["sigma_vm"].max() for r in results.values())
    for idx, cond in enumerate(conds):
        ax = axes[0, idx]
        r = results[cond]
        Nx, Ny = r["Nx"], r["Ny"]
        svm = r["sigma_vm"].reshape(Nx - 1, Ny - 1)
        im = ax.imshow(svm.T, origin="lower", cmap="jet",
                       aspect="equal", extent=[0, 1, 0, 1],
                       vmin=0, vmax=vmax_global)
        plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
        info = CONDITIONS.get(cond, {})
        ax.set_title(f"{info.get('label', cond)}\nσ_vm max={svm.max():.3f} Pa",
                     fontsize=10)
        ax.set_xlabel("depth (norm.)")
        ax.set_ylabel("lateral")

    # Bottom row: 1D profiles along mid-y
    for idx, cond in enumerate(conds):
        ax = axes[1, idx]
        r = results[cond]
        Nx, Ny = r["Nx"], r["Ny"]
        n_ex, n_ey = Nx - 1, Ny - 1
        mid_j = n_ey // 2

        sxx = r["sigma_xx"].reshape(n_ex, n_ey)[:, mid_j]
        syy = r["sigma_yy"].reshape(n_ex, n_ey)[:, mid_j]
        svm = r["sigma_vm"].reshape(n_ex, n_ey)[:, mid_j]
        cx = r["elem_centers"][:, 0].reshape(n_ex, n_ey)[:, mid_j]

        ax.plot(cx, svm, "k-", lw=2, label="σ_vm")
        ax.plot(cx, sxx, "b--", lw=1.5, label="σ_xx")
        ax.plot(cx, syy, "r:", lw=1.5, label="σ_yy")
        ax.set_xlabel("depth (norm.)")
        ax.set_ylabel("Stress [Pa]")
        ax.set_title(f"E={r['E_mean']:.0f} Pa, ε_max={r['eps_growth_max']:.4f}",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"2D FEM Stress: Hybrid Approach (E model: {e_model})\n"
        f"0D condition-dependent E + 1D spatial eigenstrain → 2D FEM",
        fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = outdir / f"hybrid_stress_comparison_{e_model}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nComparison figure: {out}")

    # Bar chart summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) σ_vm max
    ax = axes[0]
    vals = [results[c]["sigma_vm"].max() for c in conds]
    cols = [CONDITIONS.get(c, {}).get("color", "#888") for c in conds]
    labels = [CONDITIONS.get(c, {}).get("label", c) for c in conds]
    ax.bar(range(n), vals, color=cols, alpha=0.8, edgecolor="k")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("σ_vm max [Pa]")
    ax.set_title("(a) Peak von Mises Stress")
    ax.grid(True, alpha=0.3, axis="y")

    # (b) E (material stiffness)
    ax = axes[1]
    vals = [results[c]["E_mean"] for c in conds]
    ax.bar(range(n), vals, color=cols, alpha=0.8, edgecolor="k")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel(f"E ({e_model}) [Pa]")
    ax.set_title("(b) Young's Modulus (from 0D)")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Max displacement
    ax = axes[2]
    vals = [np.max(np.sqrt(results[c]["u"][:, 0]**2 + results[c]["u"][:, 1]**2))
            for c in conds]
    ax.bar(range(n), vals, color=cols, alpha=0.8, edgecolor="k")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("|u|_max [m]")
    ax.set_title("(c) Peak Displacement")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Cross-Condition Summary ({e_model})",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = outdir / f"hybrid_stress_bars_{e_model}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Bar chart: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=list(CONDITIONS.keys()))
    ap.add_argument("--e-model", choices=["di", "phi_pg", "virulence"],
                    default="phi_pg")
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--all-models", action="store_true",
                    help="Run all 3 E models")
    args = ap.parse_args()

    _OUTDIR.mkdir(parents=True, exist_ok=True)

    models = ["di", "phi_pg", "virulence"] if args.all_models else [args.e_model]

    for e_model in models:
        print("=" * 60)
        print(f"2D FEM Stress: Hybrid Approach (E model: {e_model})")
        print("=" * 60)

        results = {}
        summaries = {}

        for cond in args.conditions:
            r = run_condition(cond, e_model, Ny=args.ny, nu=args.nu)
            if r is not None:
                results[cond] = r
                summaries[cond] = {
                    "condition": cond,
                    "e_model": e_model,
                    "E_mean_pa": float(r["E_mean"]),
                    "eps_growth_max": float(r["eps_growth_max"]),
                    "sigma_vm_max_pa": float(r["sigma_vm"].max()),
                    "sigma_vm_mean_pa": float(r["sigma_vm"].mean()),
                    "u_max": float(np.max(np.sqrt(r["u"][:, 0]**2 + r["u"][:, 1]**2))),
                }

        # Save summary
        with (_OUTDIR / f"summary_{e_model}.json").open("w") as f:
            json.dump(summaries, f, indent=2)

        # Comparison figures
        if results:
            plot_all_conditions(results, _OUTDIR, e_model)

        # Print table
        print(f"\n{'='*70}")
        print(f"{'Condition':<22} {'E [Pa]':>8} {'ε_g max':>10} "
              f"{'σ_vm max':>10} {'|u|_max':>10}")
        print(f"{'='*70}")
        for cond, s in summaries.items():
            print(f"{cond:<22} {s['E_mean_pa']:8.1f} {s['eps_growth_max']:10.6f} "
                  f"{s['sigma_vm_max_pa']:10.4f} {s['u_max']:10.6f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
