#!/usr/bin/env python3
"""Compare B1-B4 improvement run vs baseline dh_baseline results."""

import json
import numpy as np
from pathlib import Path


def main():
    base_dir = Path(__file__).parent / "_runs"

    # Old baseline
    old_dir = base_dir / "dh_baseline"
    old_MAP = json.load(open(old_dir / "theta_MAP.json"))
    old_samples = np.load(old_dir / "samples.npy")

    # New B1-B4 run (find most recent DH_150p)
    new_dirs = sorted(base_dir.glob("DH_150p_expIC_repSigma_*"))
    if not new_dirs:
        print("No B1-B4 run found yet.")
        return
    new_dir = new_dirs[-1]
    print(f"Old: {old_dir.name}")
    print(f"New: {new_dir.name}")

    if not (new_dir / "samples.npy").exists():
        print("New run not finished yet (no samples.npy).")
        # Check if log exists
        logs = list(Path(__file__).parent.glob("tmcmc-B1B4*.log"))
        if logs:
            print(f"\nLog tail ({logs[-1].name}):")
            with open(logs[-1]) as f:
                lines = f.readlines()
            for line in lines[-30:]:
                print(f"  {line.rstrip()}")
        return

    new_samples = np.load(new_dir / "samples.npy")

    # MAP comparison
    new_MAP_file = new_dir / "theta_MAP.json"
    if new_MAP_file.exists():
        new_MAP = json.load(open(new_MAP_file))
    else:
        # Compute from samples
        new_MAP = {"theta_MAP": new_samples.mean(axis=0).tolist()}

    print(f"\n{'='*60}")
    print("MAP θ Comparison (20 params)")
    print(f"{'='*60}")
    param_names = [
        "a11",
        "a21",
        "a31",
        "a41",
        "a51",
        "a12",
        "a22",
        "a32",
        "a42",
        "a52",
        "a13",
        "a23",
        "a33",
        "a43",
        "a53",
        "a14",
        "a24",
        "a34",
        "a44",
        "a54",
    ]

    old_theta = np.array(old_MAP.get("theta_MAP", old_MAP.get("map_theta", [])))
    new_theta = np.array(new_MAP.get("theta_MAP", new_MAP.get("map_theta", [])))

    if len(old_theta) > 0 and len(new_theta) > 0:
        n = min(len(old_theta), len(new_theta), 20)
        print(f"{'Param':<8} {'Old MAP':>10} {'New MAP':>10} {'Diff':>10} {'Rel%':>8}")
        print("-" * 50)
        for i in range(n):
            name = param_names[i] if i < len(param_names) else f"θ[{i}]"
            diff = new_theta[i] - old_theta[i]
            rel = 100 * abs(diff) / (abs(old_theta[i]) + 1e-10)
            print(
                f"{name:<8} {old_theta[i]:>10.4f} {new_theta[i]:>10.4f} {diff:>+10.4f} {rel:>7.1f}%"
            )

    # Posterior statistics
    print(f"\n{'='*60}")
    print("Posterior Statistics")
    print(f"{'='*60}")
    print(f"  Old samples shape: {old_samples.shape}")
    print(f"  New samples shape: {new_samples.shape}")

    # Per-param overlap (Bhattacharyya-style)
    n_params = min(old_samples.shape[1], new_samples.shape[1], 20)
    overlaps = []
    for i in range(n_params):
        # Simple histogram overlap
        lo = min(old_samples[:, i].min(), new_samples[:, i].min())
        hi = max(old_samples[:, i].max(), new_samples[:, i].max())
        bins = np.linspace(lo, hi, 50)
        h1, _ = np.histogram(old_samples[:, i], bins=bins, density=True)
        h2, _ = np.histogram(new_samples[:, i], bins=bins, density=True)
        dx = bins[1] - bins[0]
        ol = np.sum(np.minimum(h1, h2)) * dx
        overlaps.append(ol)

    mean_ol = np.mean(overlaps)
    min_ol = np.min(overlaps)
    min_idx = np.argmin(overlaps)
    print(
        f"  Posterior overlap: mean={mean_ol:.3f}, min={min_ol:.3f} (param {param_names[min_idx]})"
    )
    print(f"  Overlaps > 0.8: {sum(1 for o in overlaps if o > 0.8)}/{n_params}")

    # Log file analysis for B1-B4 metrics
    logs = sorted(Path(__file__).parent.glob("tmcmc-B1B4*.log"))
    if logs:
        log_file = logs[-1]
        print(f"\n{'='*60}")
        print(f"B1-B4 Diagnostics from {log_file.name}")
        print(f"{'='*60}")
        with open(log_file) as f:
            content = f.read()

        # Extract unique ratios
        import re

        unique_ratios = re.findall(r"unique_ratio=([\d.]+)", content)
        if unique_ratios:
            ratios = [float(r) for r in unique_ratios]
            print(
                f"  Unique ratio (B1): mean={np.mean(ratios):.3f}, "
                f"min={np.min(ratios):.3f}, max={np.max(ratios):.3f}"
            )

        # Extract acceptance rates
        acc_rates = re.findall(r"acc[_-]rate.*?([\d.]+)", content, re.IGNORECASE)
        if acc_rates:
            rates = [float(r) for r in acc_rates if 0 < float(r) < 1]
            if rates:
                print(
                    f"  Acceptance rate (B4): mean={np.mean(rates):.3f}, "
                    f"min={np.min(rates):.3f}, max={np.max(rates):.3f}"
                )

        # Extract scalem values
        scalem_vals = re.findall(r"scalem=([\d.]+)", content)
        if scalem_vals:
            svals = [float(s) for s in scalem_vals]
            print(
                f"  Scalem (B4): mean={np.mean(svals):.4f}, "
                f"range=[{np.min(svals):.4f}, {np.max(svals):.4f}]"
            )

        # Wall time
        times = re.findall(r"wall_time.*?([\d.]+)s", content, re.IGNORECASE)
        if times:
            print(f"  Wall time: {times[-1]}s")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
