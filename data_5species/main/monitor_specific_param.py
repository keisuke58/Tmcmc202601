#!/usr/bin/env python3
"""
Monitor specific parameters from TMCMC checkpoints.
Useful for checking the convergence/distribution of key parameters (e.g., Theta 18)
during a long running simulation.

Usage:
    python monitor_specific_param.py --dir /path/to/run --idx 18
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_hdi(samples, credibility=0.95):
    """Compute Highest Density Interval."""
    sorted_samples = np.sort(samples)
    n_samples = len(sorted_samples)
    interval_idx_inc = int(np.floor(credibility * n_samples))
    n_intervals = n_samples - interval_idx_inc
    width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(width)
    hdi_min = sorted_samples[min_idx]
    hdi_max = sorted_samples[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


def main():
    parser = argparse.ArgumentParser(description="Monitor Specific Parameter from Checkpoint")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to run directory containing tmcmc_checkpoint.pkl",
    )
    parser.add_argument(
        "--idx", type=int, default=18, help="Parameter index to monitor (default: 18 for V->Pg)"
    )
    parser.add_argument("--name", type=str, default="Theta", help="Parameter name for plot")
    parser.add_argument(
        "--output", type=str, default="param_monitor.png", help="Output plot filename"
    )

    args = parser.parse_args()

    run_dir = Path(args.dir)
    ckpt_path = run_dir / "tmcmc_checkpoint.pkl"

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return

    try:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        stage = ckpt.get("stage", -1)
        chains = ckpt.get("chains", [])

        if not chains:
            logger.error("No chains found in checkpoint")
            return

        # Concatenate chains to get all samples
        samples = np.concatenate(chains, axis=0)

        if args.idx >= samples.shape[1]:
            logger.error(f"Index {args.idx} out of bounds (n_params={samples.shape[1]})")
            return

        param_samples = samples[:, args.idx]

        # Statistics
        mean_val = np.mean(param_samples)
        median_val = np.median(param_samples)
        std_val = np.std(param_samples)
        hdi_95 = compute_hdi(param_samples, 0.95)

        print(f"--- Stage {stage} Statistics for Parameter {args.idx} ({args.name}) ---")
        print(f"Mean:   {mean_val:.4f}")
        print(f"Median: {median_val:.4f}")
        print(f"Std:    {std_val:.4f}")
        print(f"95% HDI: [{hdi_95[0]:.4f}, {hdi_95[1]:.4f}]")
        print("---------------------------------------------------")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(param_samples, kde=True, color="skyblue", bins=30)

        # Add vertical lines for stats
        plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        plt.axvline(hdi_95[0], color="green", linestyle=":", label="95% HDI")
        plt.axvline(hdi_95[1], color="green", linestyle=":")

        plt.title(f"Parameter {args.idx} ({args.name}) Distribution at Stage {stage}")
        plt.xlabel("Parameter Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(args.output, dpi=150)
        logger.info(f"Plot saved to {args.output}")

    except Exception as e:
        logger.error(f"Error processing checkpoint: {e}")


if __name__ == "__main__":
    main()
