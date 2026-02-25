#!/usr/bin/env python3
"""
Diagnose θ distribution mismatch: synthetic vs TMCMC posterior.

Identifies why Dysbiotic_HOBIC MAE degrades in Phase 2 (synthetic augmentation).
Compares: prior, synthetic (prior+MAP), TMCMC posterior per condition.

Usage:
    python diagnose_theta_distribution.py
    python diagnose_theta_distribution.py --synthetic data/synthetic_all_N8000.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_posterior_data import load_all, CONDITION_PATHS
from generate_synthetic_data import (
    load_prior_bounds,
    load_theta_map,
    CONDITIONS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        type=str,
        default="data/synthetic_all_N8000.npz",
        help="Path to synthetic .npz",
    )
    args = parser.parse_args()

    syn_path = Path(__file__).parent / args.synthetic
    if not syn_path.exists():
        print(f"Synthetic file not found: {syn_path}")
        print("Run: python generate_synthetic_data.py --n-samples 2000 --all-conditions")
        return 1

    # Load TMCMC posterior
    data = load_all()
    theta_post = data["theta"]
    labels = data["condition_labels"]

    # Load synthetic
    npz = np.load(syn_path)
    y_syn = npz["y_obs"]
    theta_syn = npz["theta"]
    if "condition_labels" in npz.files:
        syn_labels = list(npz["condition_labels"])
    else:
        n_total = len(theta_syn)
        n_per = n_total // 4
        syn_labels = []
        for c in CONDITIONS:
            syn_labels.extend([c] * n_per)
        syn_labels = (syn_labels + [CONDITIONS[-1]] * (n_total - len(syn_labels)))[:n_total]

    runs_dir = PROJECT_ROOT / "data_5species" / "_runs"

    print("=" * 70)
    print("θ distribution diagnosis: Synthetic vs TMCMC posterior")
    print("=" * 70)

    for cond in CONDITIONS:
        print(f"\n--- {cond} ---")

        idx_post = [i for i, l in enumerate(labels) if l == cond]
        idx_syn = [i for i, l in enumerate(syn_labels) if l == cond]

        theta_p = theta_post[idx_post] if idx_post else np.zeros((0, 20))
        theta_s = theta_syn[idx_syn] if idx_syn else np.zeros((0, 20))

        if theta_p.size == 0:
            print("  [WARN] No TMCMC posterior for this condition")
            continue
        if theta_s.size == 0:
            print("  [WARN] No synthetic samples for this condition")
            continue

        bounds = load_prior_bounds(cond)
        theta_map = load_theta_map(cond)

        # Prior mean (midpoint of bounds)
        prior_mean = np.zeros(20)
        for i in range(20):
            lo, hi = bounds[i]
            if abs(hi - lo) < 1e-12:
                prior_mean[i] = lo
            else:
                prior_mean[i] = 0.5 * (lo + hi)

        post_mean = theta_p.mean(axis=0)
        syn_mean = theta_s.mean(axis=0)
        post_std = theta_p.std(axis=0)
        syn_std = theta_s.std(axis=0)

        # Key metrics
        mae_post_syn = np.abs(post_mean - syn_mean).mean()
        mae_post_map = (
            np.abs(post_mean - theta_map).mean() if theta_map is not None else np.nan
        )
        mae_syn_map = (
            np.abs(syn_mean - theta_map).mean() if theta_map is not None else np.nan
        )

        # Prior width vs posterior width (first 5 free params)
        free_idx = [i for i in range(20) if abs(bounds[i, 1] - bounds[i, 0]) > 1e-12]
        prior_width = (bounds[free_idx, 1] - bounds[free_idx, 0]).mean()
        post_width = post_std[free_idx].mean() * 4  # ~95% CI approx
        syn_width = syn_std[free_idx].mean() * 4

        print(f"  TMCMC posterior: n={len(theta_p)}, mean[0:5]={post_mean[:5].round(3)}")
        print(f"  Synthetic:       n={len(theta_s)}, mean[0:5]={syn_mean[:5].round(3)}")
        if theta_map is not None:
            print(f"  θ_MAP:           mean[0:5]={theta_map[:5].round(3)}")
        print(f"  Prior mean:      mean[0:5]={prior_mean[:5].round(3)}")

        print(f"\n  MAE(post_mean, syn_mean) = {mae_post_syn:.4f}")
        if theta_map is not None:
            print(f"  MAE(post_mean, θ_MAP)     = {mae_post_map:.4f}")
            print(f"  MAE(syn_mean, θ_MAP)     = {mae_syn_map:.4f}")

        print(f"\n  Prior width (avg)  = {prior_width:.3f}")
        print(f"  Posterior ~95% CI  = {post_width:.3f}")
        print(f"  Synthetic ~95% CI  = {syn_width:.3f}")
        print(
            f"  Ratio syn/post     = {syn_width / post_width:.2f}x"
            if post_width > 1e-6
            else "  (posterior very narrow)"
        )

        # Diagnosis
        if mae_post_syn > 0.5:
            print(
                f"\n  >>> CAUSE: Synthetic θ mean is far from TMCMC posterior mean (MAE={mae_post_syn:.2f})"
            )
            print(
                "      Fix: Increase map_frac, decrease map_std_frac, or sample from posterior"
            )
        if post_width > 0 and syn_width / post_width > 2:
            print(
                f"\n  >>> CAUSE: Synthetic θ is much wider than posterior ({syn_width/post_width:.1f}x)"
            )
            print(
                "      Fix: Reduce prior fraction, tighten map_std_frac, or use posterior-informed sampling"
            )

    # Summary: data source check for Dysbiotic_HOBIC
    print("\n" + "=" * 70)
    print("Data source check (Dysbiotic_HOBIC)")
    print("=" * 70)
    d, s = CONDITION_PATHS["Dysbiotic_HOBIC"]
    print(f"  y_obs: {runs_dir / d / 'data.npy'}")
    print(f"  theta: {runs_dir / s / 'samples.npy'}")
    if not (runs_dir / s / "data.npy").exists():
        print(
            f"  [NOTE] dh_baseline has no data.npy — y_obs from dysbiotic_hobic_1000p, theta from dh_baseline"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
