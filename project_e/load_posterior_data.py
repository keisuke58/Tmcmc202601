#!/usr/bin/env python3
"""
Load TMCMC posterior data for Project E (VAE × TMCMC).

Pairs (y_obs, theta_samples) from 4 conditions:
  - y_obs: (6, 5) normalized species composition (6 days × 5 species)
  - theta: (n_samples, 20) posterior samples

Usage:
    python load_posterior_data.py --check
    python load_posterior_data.py --save data/project_e_posterior.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

# Condition -> (data_dir, samples_dir)
# data_dir has data.npy (y_obs), samples_dir has samples.npy (theta)
CONDITION_PATHS = {
    "Commensal_Static": ("commensal_static_posterior", "commensal_static_posterior"),
    "Commensal_HOBIC": ("commensal_hobic_posterior", "commensal_hobic_posterior"),
    "Dysbiotic_Static": ("dysbiotic_static_posterior", "dysbiotic_static_posterior"),
    "Dysbiotic_HOBIC": ("dysbiotic_hobic_1000p", "dh_baseline"),
}


def load_condition(cond_name: str):
    """Load (y_obs, theta_samples, metadata) for one condition."""
    data_dirname, samples_dirname = CONDITION_PATHS.get(cond_name, (None, None))
    if data_dirname is None:
        return None

    data_path = RUNS_DIR / data_dirname / "data.npy"
    samples_path = RUNS_DIR / samples_dirname / "samples.npy"
    config_path = RUNS_DIR / data_dirname / "config.json"

    if not data_path.exists():
        print(f"  [WARN] data.npy not found: {data_path}")
        return None
    if not samples_path.exists():
        print(f"  [WARN] samples.npy not found: {samples_path}")
        return None

    y_obs = np.load(data_path).astype(np.float32)  # (6, 5)
    theta = np.load(samples_path).astype(np.float32)  # (n, 20)

    metadata = {}
    if config_path.exists():
        with open(config_path) as f:
            metadata = json.load(f)

    return y_obs, theta, metadata


def load_all() -> dict:
    """
    Load all 4 conditions.

    Returns
    -------
    dict
        y_obs: (4, 6, 5) or flattened (4*300, 30) for training
        theta: (4*300, 20)
        condition_labels: list of condition names (repeated per sample)
    """
    y_list, theta_list, labels = [], [], []

    for cond_name in CONDITION_PATHS:
        result = load_condition(cond_name)
        if result is None:
            continue
        y_obs, theta, _ = result
        n_samples = theta.shape[0]
        # Repeat y_obs for each theta sample (same y per condition)
        y_repeated = np.tile(y_obs.flatten(), (n_samples, 1))  # (n, 30)
        y_list.append(y_repeated)
        theta_list.append(theta)
        labels.extend([cond_name] * n_samples)

    return {
        "y_obs": np.concatenate(y_list, axis=0),
        "theta": np.concatenate(theta_list, axis=0),
        "condition_labels": labels,
        "n_conditions": len(y_list),
        "y_shape": (6, 5),
        "theta_dim": 20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check data availability")
    parser.add_argument("--save", type=str, default=None, help="Save path for .npz")
    args = parser.parse_args()

    if args.check:
        print("Checking TMCMC posterior data...")
        for cond_name, (d, s) in CONDITION_PATHS.items():
            dp = RUNS_DIR / d / "data.npy"
            sp = RUNS_DIR / s / "samples.npy"
            ok = "OK" if (dp.exists() and sp.exists()) else "MISSING"
            print(f"  {cond_name}: data={dp.name} {ok}, samples={s}/samples.npy {ok}")
        return

    data = load_all()
    print(f"Loaded: y_obs {data['y_obs'].shape}, theta {data['theta'].shape}")
    print(f"  Conditions: {data['n_conditions']}")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            y_obs=data["y_obs"],
            theta=data["theta"],
        )
        # Save labels as text (one per line)
        labels_path = out_path.with_suffix(".labels.txt")
        with open(labels_path, "w") as f:
            f.write("\n".join(data["condition_labels"]))
        print(f"Saved to {out_path}, {labels_path}")


if __name__ == "__main__":
    main()
