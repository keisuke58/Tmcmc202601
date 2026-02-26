#!/usr/bin/env python3
"""
Compare two TMCMC runs to assess improvement in parameter estimation.

Usage:
    python compare_runs.py --run1 _runs/Commensal_Static_20260204_062733 --run2 _runs/improved_v1_*
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_5SPECIES_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))


PARAM_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",  # M1
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",  # M2
    "a13",
    "a14",
    "a23",
    "a24",  # M3
    "a55",
    "b5",  # M4
    "a15",
    "a25",
    "a35",
    "a45",  # M5
]

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]


def load_run(run_dir: Path) -> Dict[str, Any]:
    """Load results from a run directory."""
    results = {}

    # Load numpy arrays
    for npy_file in ["samples.npy", "logL.npy", "data.npy", "idx_sparse.npy"]:
        path = run_dir / npy_file
        if path.exists():
            results[npy_file.replace(".npy", "")] = np.load(path)

    # Load JSON files
    for json_file in [
        "config.json",
        "theta_MAP.json",
        "theta_mean.json",
        "results_summary.json",
        "fit_metrics.json",
    ]:
        path = run_dir / json_file
        if path.exists():
            with open(path, "r") as f:
                results[json_file.replace(".json", "")] = json.load(f)

    results["name"] = run_dir.name
    return results


def compare_parameters(run1: Dict, run2: Dict) -> None:
    """Compare parameter estimates between runs."""
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON")
    print("=" * 80)

    theta1_map = np.array(run1["theta_MAP"]["theta_full"])
    theta2_map = np.array(run2["theta_MAP"]["theta_full"])
    theta1_mean = np.array(run1["theta_mean"]["theta_full"])
    theta2_mean = np.array(run2["theta_mean"]["theta_full"])

    samples1 = run1.get("samples")
    samples2 = run2.get("samples")

    print(
        f"\n{'Parameter':<8} {'Run1 MAP':>10} {'Run2 MAP':>10} {'Diff':>10} | {'Run1 Mean':>10} {'Run2 Mean':>10} {'Diff':>10}"
    )
    print("-" * 80)

    for i, name in enumerate(PARAM_NAMES):
        map_diff = theta2_map[i] - theta1_map[i]
        mean_diff = theta2_mean[i] - theta1_mean[i]
        print(
            f"{name:<8} {theta1_map[i]:>10.4f} {theta2_map[i]:>10.4f} {map_diff:>+10.4f} | "
            f"{theta1_mean[i]:>10.4f} {theta2_mean[i]:>10.4f} {mean_diff:>+10.4f}"
        )

    # Overall difference
    map_l2_diff = np.linalg.norm(theta2_map - theta1_map)
    mean_l2_diff = np.linalg.norm(theta2_mean - theta1_mean)
    print("-" * 80)
    print(
        f"{'L2 norm':<8} {'':<10} {'':<10} {map_l2_diff:>10.4f} | {'':<10} {'':<10} {mean_l2_diff:>10.4f}"
    )

    # Posterior width comparison
    if samples1 is not None and samples2 is not None:
        print("\n" + "-" * 80)
        print("POSTERIOR WIDTH (Standard Deviation)")
        print("-" * 80)
        print(f"{'Parameter':<8} {'Run1 std':>12} {'Run2 std':>12} {'Ratio':>10}")
        print("-" * 50)

        std1 = np.std(samples1, axis=0)
        std2 = np.std(samples2, axis=0)

        for i, name in enumerate(PARAM_NAMES):
            ratio = std2[i] / std1[i] if std1[i] > 0 else np.inf
            print(f"{name:<8} {std1[i]:>12.4f} {std2[i]:>12.4f} {ratio:>10.2f}x")

        avg_ratio = np.mean(std2 / np.where(std1 > 0, std1, 1))
        print("-" * 50)
        print(f"{'Average':<8} {'':<12} {'':<12} {avg_ratio:>10.2f}x")


def compare_fit_metrics(run1: Dict, run2: Dict) -> None:
    """Compare fit metrics between runs."""
    print("\n" + "=" * 80)
    print("FIT METRICS COMPARISON")
    print("=" * 80)

    metrics1 = run1.get("fit_metrics")
    metrics2 = run2.get("fit_metrics")

    if metrics1 is None or metrics2 is None:
        print("Fit metrics not available for one or both runs.")
        return

    for est_type in ["MAP", "Mean"]:
        if est_type not in metrics1 or est_type not in metrics2:
            continue

        m1 = metrics1[est_type]
        m2 = metrics2[est_type]

        print(f"\n{est_type} Estimate:")
        print("-" * 60)

        # Total metrics
        rmse_diff = m2["rmse_total"] - m1["rmse_total"]
        mae_diff = m2["mae_total"] - m1["mae_total"]
        rmse_pct = (rmse_diff / m1["rmse_total"]) * 100 if m1["rmse_total"] > 0 else 0

        print(f"  Total RMSE: {m1['rmse_total']:.6f} → {m2['rmse_total']:.6f} ({rmse_pct:+.1f}%)")
        print(f"  Total MAE:  {m1['mae_total']:.6f} → {m2['mae_total']:.6f}")

        # Per-species RMSE
        print("\n  Per-species RMSE:")
        rmse1 = np.array(m1["rmse_per_species"])
        rmse2 = np.array(m2["rmse_per_species"])

        for i, name in enumerate(SPECIES_NAMES):
            diff = rmse2[i] - rmse1[i]
            pct = (diff / rmse1[i]) * 100 if rmse1[i] > 0 else 0
            arrow = "↓" if diff < 0 else "↑" if diff > 0 else "="
            print(f"    {name:<15}: {rmse1[i]:.4f} → {rmse2[i]:.4f} ({pct:+.1f}%) {arrow}")


def compare_convergence(run1: Dict, run2: Dict) -> None:
    """Compare convergence statistics."""
    print("\n" + "=" * 80)
    print("CONVERGENCE COMPARISON")
    print("=" * 80)

    summary1 = run1.get("results_summary", {})
    summary2 = run2.get("results_summary", {})
    config1 = run1.get("config", {})
    config2 = run2.get("config", {})

    print(f"\n{'Metric':<25} {'Run1':>15} {'Run2':>15}")
    print("-" * 60)
    print(
        f"{'Particles':<25} {config1.get('n_particles', 'N/A'):>15} {config2.get('n_particles', 'N/A'):>15}"
    )
    print(
        f"{'Stages':<25} {config1.get('n_stages', 'N/A'):>15} {config2.get('n_stages', 'N/A'):>15}"
    )
    print(
        f"{'Chains':<25} {config1.get('n_chains', 'N/A'):>15} {config2.get('n_chains', 'N/A'):>15}"
    )

    time1 = summary1.get("elapsed_time", 0)
    time2 = summary2.get("elapsed_time", 0)
    print(f"{'Elapsed Time (s)':<25} {time1:>15.1f} {time2:>15.1f}")
    print(f"{'Elapsed Time (h)':<25} {time1/3600:>15.2f} {time2/3600:>15.2f}")

    conv1 = summary1.get("converged", [])
    conv2 = summary2.get("converged", [])
    print(f"{'Chains Converged':<25} {str(conv1):>15} {str(conv2):>15}")


def plot_comparison(run1: Dict, run2: Dict, output_dir: Path) -> None:
    """Generate comparison plots."""
    output_dir.mkdir(exist_ok=True)

    # 1. Parameter comparison bar chart
    theta1_map = np.array(run1["theta_MAP"]["theta_full"])
    theta2_map = np.array(run2["theta_MAP"]["theta_full"])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(PARAM_NAMES))
    width = 0.35

    ax.bar(x - width / 2, theta1_map, width, label=f'Run1 ({run1["name"][:20]})', alpha=0.8)
    ax.bar(x + width / 2, theta2_map, width, label=f'Run2 ({run2["name"][:20]})', alpha=0.8)

    ax.set_xlabel("Parameter")
    ax.set_ylabel("MAP Value")
    ax.set_title("Parameter Comparison: MAP Estimates")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_parameters.png", dpi=150)
    plt.close()

    # 2. Posterior width comparison
    if "samples" in run1 and "samples" in run2:
        std1 = np.std(run1["samples"], axis=0)
        std2 = np.std(run2["samples"], axis=0)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width / 2, std1, width, label="Run1 std", alpha=0.8)
        ax.bar(x + width / 2, std2, width, label="Run2 std", alpha=0.8)

        ax.set_xlabel("Parameter")
        ax.set_ylabel("Standard Deviation")
        ax.set_title("Posterior Width Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_posterior_width.png", dpi=150)
        plt.close()

    print(f"\nComparison plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare two TMCMC runs")
    parser.add_argument("--run1", type=str, required=True, help="First run directory (baseline)")
    parser.add_argument("--run2", type=str, required=True, help="Second run directory (improved)")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for comparison plots"
    )

    args = parser.parse_args()

    # Resolve paths
    run1_dir = Path(args.run1)
    run2_dir = Path(args.run2)

    if not run1_dir.is_absolute():
        run1_dir = DATA_5SPECIES_ROOT / run1_dir
    if not run2_dir.is_absolute():
        run2_dir = DATA_5SPECIES_ROOT / run2_dir

    # Handle glob patterns
    if "*" in str(run2_dir):
        import glob

        matches = sorted(glob.glob(str(run2_dir)))
        if matches:
            run2_dir = Path(matches[-1])  # Use most recent
            print(f"Using most recent match: {run2_dir}")
        else:
            print(f"No matches found for {args.run2}")
            sys.exit(1)

    if not run1_dir.exists():
        print(f"Run1 directory not found: {run1_dir}")
        sys.exit(1)
    if not run2_dir.exists():
        print(f"Run2 directory not found: {run2_dir}")
        sys.exit(1)

    print("=" * 80)
    print("TMCMC RUN COMPARISON")
    print("=" * 80)
    print(f"\nRun1 (baseline): {run1_dir.name}")
    print(f"Run2 (improved): {run2_dir.name}")

    # Load runs
    run1 = load_run(run1_dir)
    run2 = load_run(run2_dir)

    # Compare
    compare_convergence(run1, run2)
    compare_parameters(run1, run2)
    compare_fit_metrics(run1, run2)

    # Generate plots
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = DATA_5SPECIES_ROOT / "_comparisons" / f"{run1_dir.name}_vs_{run2_dir.name}"

    plot_comparison(run1, run2, output_dir)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
