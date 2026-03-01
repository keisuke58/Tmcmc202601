#!/usr/bin/env python3
"""
run_2d_stress_all_conditions.py
================================
Run 2D FEM stress analysis for all conditions and generate comparison figure.

Usage:
    python run_2d_stress_all_conditions.py
    python run_2d_stress_all_conditions.py --quick   # 10x10 grid
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "JAXFEM"))

from JAXFEM.solve_stress_2d import run_2d_stress_pipeline, plot_2d_stress

_RUNS_ROOT = _HERE.parent / "data_5species" / "_runs"
_MAIN_RUNS = _HERE.parent / "data_5species" / "main" / "_runs"
_OUTDIR = _HERE / "_stress_2d_results"

CONDITION_THETA_PATHS = {
    "dh_baseline": _RUNS_ROOT / "dh_baseline" / "theta_MAP.json",
    "commensal_static": _RUNS_ROOT / "commensal_static" / "theta_MAP.json",
    "commensal_hobic": _RUNS_ROOT / "commensal_hobic" / "theta_MAP.json",
    "dysbiotic_static": _RUNS_ROOT / "dysbiotic_static" / "theta_MAP.json",
}

# Demo theta (mild-weight MAP) as fallback
THETA_DEMO = np.array(
    [
        1.34,
        -0.18,
        1.79,
        1.17,
        2.58,
        3.51,
        2.73,
        0.71,
        2.1,
        0.37,
        2.05,
        -0.15,
        3.56,
        0.16,
        0.12,
        0.32,
        1.49,
        2.1,
        2.41,
        2.5,
    ]
)

PARAM_NAMES = [
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


def load_theta(condition):
    """Load theta_MAP for a condition."""
    path = CONDITION_THETA_PATHS.get(condition)
    if path and path.exists():
        with open(path) as f:
            d = json.load(f)
        if "theta_full" in d:
            return np.array(d["theta_full"], dtype=np.float64)
        if "theta_sub" in d:
            return np.array(d["theta_sub"], dtype=np.float64)
        return np.array([d[k] for k in PARAM_NAMES], dtype=np.float64)
    return THETA_DEMO.copy()


def plot_condition_comparison(results, outdir):
    """Generate cross-condition comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    n = len(conds)
    if n < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    # (a) DI mean comparison
    ax = axes[0, 0]
    di_means = [results[c]["DI"].mean() for c in conds]
    bars = ax.bar(
        range(n),
        di_means,
        color=[colors[i] for i in range(n)],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("DI mean")
    ax.set_title("(a) Dysbiosis Index (mean)")
    ax.grid(True, alpha=0.3, axis="y")

    # (b) σ_vm max comparison
    ax = axes[0, 1]
    svm_max = [results[c]["sigma_vm"].max() for c in conds]
    ax.bar(
        range(n),
        svm_max,
        color=[colors[i] for i in range(n)],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("σ_vm max [Pa]")
    ax.set_title("(b) Max von Mises Stress")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) E_min comparison
    ax = axes[0, 2]
    e_mins = [results[c]["E_field"].min() for c in conds]
    ax.bar(
        range(n),
        e_mins,
        color=[colors[i] for i in range(n)],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("E_min [Pa]")
    ax.set_title("(c) Minimum Young's Modulus")
    ax.grid(True, alpha=0.3, axis="y")

    # (d-f) Stress maps for top 3 conditions
    for idx, cond in enumerate(conds[:3]):
        ax = axes[1, idx]
        r = results[cond]
        Nx, Ny = r["Nx"], r["Ny"]
        svm = r["sigma_vm"].reshape(Nx - 1, Ny - 1)
        im = ax.imshow(svm.T, origin="lower", cmap="jet", aspect="equal", extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
        ax.set_title(f"(d-{idx}) σ_vm: {cond}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Cross-Condition 2D Stress Comparison", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(outdir) / "cross_condition_stress_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nComparison figure: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"],
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--e-model", default="phi_pg")
    args = ap.parse_args()

    nx = 10 if args.quick else 20
    ny = 10 if args.quick else 20
    n_macro = 15 if args.quick else 60

    _OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("2D FEM Stress: All Conditions")
    print(f"  Grid: {nx}×{ny}, n_macro={n_macro}")
    print(f"  E model: {args.e_model}")
    print(f"  Conditions: {args.conditions}")
    print("=" * 60)

    results = {}
    summaries = {}

    for cond in args.conditions:
        theta = load_theta(cond)
        print(f"\n--- {cond} ---")
        t0 = time.perf_counter()

        try:
            result = run_2d_stress_pipeline(
                theta,
                Nx=nx,
                Ny=ny,
                n_macro=n_macro,
                dt_h=1e-5,
                n_react_sub=20,
                save_every=n_macro,
                K_hill=0.05,
                n_hill=4.0,
                nu=0.30,
                alpha_coeff=0.05,
                e_model=args.e_model,
            )
            results[cond] = result

            svm = result["sigma_vm"]
            summary = {
                "condition": cond,
                "DI_mean": float(result["DI"].mean()),
                "DI_max": float(result["DI"].max()),
                "E_min_pa": float(result["E_field"].min()),
                "sigma_vm_max_pa": float(svm.max()),
                "sigma_vm_mean_pa": float(svm.mean()),
                "eps_growth_max": float(result["eps_growth"].max()),
                "timing_s": round(time.perf_counter() - t0, 1),
            }
            summaries[cond] = summary

            print(f"  DI mean={result['DI'].mean():.4f}")
            print(f"  σ_vm max={svm.max():.3f} Pa, mean={svm.mean():.3f}")
            print(f"  E min={result['E_field'].min():.1f} Pa")
            print(f"  Time: {summary['timing_s']}s")

            plot_2d_stress(result, str(_OUTDIR), cond)

            # GC after each condition
            try:
                import jax

                jax.clear_caches()
            except Exception:
                pass
            import gc

            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Save summary
    with (_OUTDIR / "all_conditions_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    # Comparison figure
    if len(results) >= 2:
        plot_condition_comparison(results, str(_OUTDIR))

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Condition':<25} {'DI mean':>8} {'σ_vm max':>10} {'E_min':>8} {'Time':>6}")
    print(f"{'='*70}")
    for cond, s in summaries.items():
        print(
            f"{cond:<25} {s['DI_mean']:8.4f} {s['sigma_vm_max_pa']:10.3f} "
            f"{s['E_min_pa']:8.1f} {s['timing_s']:6.1f}s"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
