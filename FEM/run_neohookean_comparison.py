#!/usr/bin/env python3
"""
Generate Neo-Hookean vs Linear Elastic INP files for all 4 conditions.

Comparison of large-deformation response under biofilm-mode material (Pa-scale).
Dysbiotic conditions (E ≈ 32 Pa) show ~12mm displacement in linear analysis,
which exceeds geometric validity → Neo-Hookean hyperelastic is needed.

Usage:
  python run_neohookean_comparison.py             # Generate all INPs
  python run_neohookean_comparison.py --plot       # Plot results after Abaqus run
  python run_neohookean_comparison.py --condition dysbiotic_static  # Single condition
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FEM_DIR = Path(__file__).parent
PROJECT_ROOT = FEM_DIR.parent

# DI field CSV paths (from export_for_abaqus.py / run_hamilton_2d_nutrient.py)
COND_DI_CSV = {
    "commensal_static":  "_abaqus_fields/standard_3d/abaqus_field_Commensal_Static_snap20.csv",
    "commensal_hobic":   "_abaqus_fields/standard_3d/abaqus_field_Commensal_HOBIC_snap20.csv",
    "dysbiotic_static":  "_abaqus_fields/standard_3d/abaqus_field_Dysbiotic_Static_snap20.csv",
    "dh_baseline":       "_abaqus_fields/standard_3d/abaqus_field_dh_3d.csv",
}

STL_PATH = "external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_23.stl"

OUTPUT_DIR = FEM_DIR / "_neohookean_comparison"


def generate_inp(condition: str, neo_hookean: bool):
    """Generate a single INP file."""
    di_csv = COND_DI_CSV.get(condition)
    if di_csv is None:
        print(f"WARNING: No DI CSV for {condition}, skipping")
        return None

    di_csv_path = FEM_DIR / di_csv
    if not di_csv_path.exists():
        print(f"WARNING: {di_csv_path} not found, skipping")
        return None

    stl_path = FEM_DIR / STL_PATH
    if not stl_path.exists():
        print(f"ERROR: STL not found at {stl_path}")
        return None

    tag = "nh" if neo_hookean else "linear"
    out_dir = OUTPUT_DIR / f"{condition}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"p23_{condition}_{tag}.inp"

    cmd = [
        sys.executable,
        str(FEM_DIR / "biofilm_conformal_tet.py"),
        "--stl", str(stl_path),
        "--di-csv", str(di_csv_path),
        "--out", str(out_file),
        "--mode", "biofilm",
    ]
    if neo_hookean:
        cmd.append("--neo-hookean")

    print(f"\n{'='*60}")
    print(f"  {condition} [{tag}]")
    print(f"  → {out_file}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
        return None

    print(result.stdout[-200:])
    return str(out_file)


def generate_all(conditions=None):
    """Generate linear + Neo-Hookean INPs for all conditions."""
    if conditions is None:
        conditions = list(COND_DI_CSV.keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for cond in conditions:
        for nh in [False, True]:
            tag = "nh" if nh else "linear"
            inp = generate_inp(cond, neo_hookean=nh)
            results[f"{cond}_{tag}"] = inp

    # Write summary
    summary = {
        "conditions": conditions,
        "files": results,
        "notes": [
            "Submit INP files to Abaqus (abaqus job=p23_xxx_linear/nh cpus=4)",
            "After completion, run: python run_neohookean_comparison.py --plot",
        ],
    }
    with open(OUTPUT_DIR / "comparison_config.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len([v for v in results.values() if v])} INP files in {OUTPUT_DIR}")
    print(f"Next: submit to Abaqus, then run with --plot")
    print(f"{'='*60}")

    return results


def plot_comparison():
    """Compare results after Abaqus has been run (reads ODB/DAT)."""
    import matplotlib.pyplot as plt

    config_path = OUTPUT_DIR / "comparison_config.json"
    if not config_path.exists():
        print("ERROR: Run without --plot first to generate INP files")
        return

    with open(config_path) as f:
        config = json.load(f)

    conditions = config["conditions"]

    # Try to load results from _extract.py outputs or JSON
    results = {}
    for cond in conditions:
        for tag in ["linear", "nh"]:
            result_file = OUTPUT_DIR / f"{cond}_{tag}" / "stress_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    results[f"{cond}_{tag}"] = json.load(f)

    if not results:
        print("No results found. Run Abaqus first, then extract with:")
        print("  python odb_extract.py <job_name>")
        print("Or manually create stress_results.json in each subdirectory with keys:")
        print('  {"sigma_mises_max": ..., "u_max": ..., "sigma_mises_mean": ...}')
        return

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["u_max", "sigma_mises_max", "sigma_mises_mean"]
    labels = ["U_max (mm)", "σ_Mises max (Pa)", "σ_Mises mean (Pa)"]

    for ax, metric, label in zip(axes, metrics, labels):
        lin_vals = []
        nh_vals = []
        cond_labels = []

        for cond in conditions:
            lin_key = f"{cond}_linear"
            nh_key = f"{cond}_nh"
            if lin_key in results and nh_key in results:
                lin_vals.append(results[lin_key].get(metric, 0))
                nh_vals.append(results[nh_key].get(metric, 0))
                cond_labels.append(cond.replace("_", "\n"))

        x = np.arange(len(cond_labels))
        w = 0.35
        ax.bar(x - w/2, lin_vals, w, label="Linear Elastic", color="#60a5fa")
        ax.bar(x + w/2, nh_vals, w, label="Neo-Hookean", color="#f97316")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=8)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.set_title(label)

    fig.suptitle("Linear Elastic vs Neo-Hookean Hyperelastic Comparison\n(Biofilm mode, Pa-scale)", fontweight="bold")
    plt.tight_layout()

    out_fig = OUTPUT_DIR / "neohookean_comparison.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_fig}")
    plt.close(fig)

    # Print table
    print(f"\n{'Condition':<25} {'Metric':<20} {'Linear':>12} {'Neo-Hookean':>12} {'Ratio':>8}")
    print("-" * 80)
    for cond in conditions:
        for metric, label in zip(metrics, labels):
            lin_key = f"{cond}_linear"
            nh_key = f"{cond}_nh"
            if lin_key in results and nh_key in results:
                lv = results[lin_key].get(metric, 0)
                nv = results[nh_key].get(metric, 0)
                ratio = nv / lv if lv > 0 else float('inf')
                print(f"{cond:<25} {label:<20} {lv:>12.4f} {nv:>12.4f} {ratio:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Neo-Hookean vs Linear Elastic comparison")
    parser.add_argument("--plot", action="store_true", help="Plot results after Abaqus run")
    parser.add_argument("--condition", type=str, nargs="*", default=None,
                        help="Specific conditions (default: all 4)")
    args = parser.parse_args()

    if args.plot:
        plot_comparison()
    else:
        generate_all(conditions=args.condition)


if __name__ == "__main__":
    main()
