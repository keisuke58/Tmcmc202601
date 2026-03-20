#!/usr/bin/env python3
"""Merge multi-chain production TMCMC results.

Computes:
  - Gelman-Rubin R̂ per parameter (convergence diagnostic)
  - Best MAP across chains
  - Merged posterior (all chains)
  - Posterior statistics (mean, std, CI)

Usage:
    python merge_production_chains.py --pattern "*_2000p_prod_*" --condition DH
    python merge_production_chains.py --dirs dir1 dir2 dir3 --condition DH
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def gelman_rubin(chains):
    """Compute R̂ (Gelman-Rubin) for each parameter.

    Args:
        chains: list of (n_samples, n_params) arrays

    Returns:
        R_hat: (n_params,) array
    """
    m = len(chains)
    n = min(c.shape[0] for c in chains)
    chains = [c[:n] for c in chains]
    d = chains[0].shape[1]

    chain_means = np.array([c.mean(axis=0) for c in chains])  # (m, d)
    overall_mean = chain_means.mean(axis=0)  # (d,)

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)

    # Within-chain variance
    W = np.zeros(d)
    for c in chains:
        W += np.var(c, axis=0, ddof=1)
    W /= m

    # Pooled variance estimate
    var_hat = (n - 1) / n * W + B / n

    R_hat = np.sqrt(var_hat / np.maximum(W, 1e-12))
    return R_hat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", help="Run directories to merge")
    parser.add_argument("--pattern", type=str, help="Glob pattern for run dirs")
    parser.add_argument("--condition", type=str, required=True, help="DH/DS/CS/CH")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    COND_MAP = {
        "DH": "Dysbiotic_HOBIC",
        "DS": "Dysbiotic_Static",
        "CS": "Commensal_Static",
        "CH": "Commensal_HOBIC",
    }
    cond_str = COND_MAP.get(args.condition, args.condition)

    # Find run directories
    if args.dirs:
        dirs = args.dirs
    elif args.pattern:
        base = os.path.expanduser("~/IKM_Hiwi/Tmcmc202601/data_5species/main/_runs")
        all_dirs = sorted(glob.glob(os.path.join(base, args.pattern)))
        dirs = [
            d
            for d in all_dirs
            if cond_str.replace("_", "_") in os.path.basename(d)
            or args.condition.lower() in os.path.basename(d).lower()
        ]
        if not dirs:
            # Try matching by config
            dirs = []
            for d in all_dirs:
                cfg_path = os.path.join(d, "config.json")
                if os.path.exists(cfg_path):
                    cfg = json.load(open(cfg_path))
                    if (cfg.get("condition", "") + "_" + cfg.get("cultivation", "")) == cond_str:
                        dirs.append(d)
    else:
        print("Specify --dirs or --pattern")
        sys.exit(1)

    print(f"Condition: {args.condition} ({cond_str})")
    print(f"Found {len(dirs)} chains:")

    chains = []
    configs = []
    for d in dirs:
        samples_path = os.path.join(d, "samples.npy")
        config_path = os.path.join(d, "config.json")
        if not os.path.exists(samples_path):
            print(f"  SKIP {os.path.basename(d)}: no samples.npy")
            continue
        samples = np.load(samples_path)
        cfg = json.load(open(config_path)) if os.path.exists(config_path) else {}
        chains.append(samples)
        configs.append(cfg)
        print(
            f"  {os.path.basename(d)}: "
            f"{samples.shape[0]} samples, "
            f"logL={cfg.get('max_logL', '?'):.2f}, "
            f"accept={cfg.get('mean_accept', '?')}"
        )

    if len(chains) < 2:
        print("\nNeed at least 2 chains for R̂ diagnostic.")
        if len(chains) == 1:
            print("Single chain results:")
            s = chains[0]
            print(f"  Samples: {s.shape}")
            print(f"  MAP logL: {configs[0].get('max_logL', '?')}")
        return

    # Gelman-Rubin R̂
    free_mask = np.abs(chains[0].std(axis=0)) > 1e-10
    free_dims = np.where(free_mask)[0]
    free_chains = [c[:, free_dims] for c in chains]
    R_hat = gelman_rubin(free_chains)

    PARAM_NAMES = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",  # 0-4
        "a33",
        "a23_a32",
        "a44",
        "b3",
        "b4",  # 5-9
        "a13_a31",
        "a14_a41",
        "a12_a21",
        "a13_a31b",
        "a55",  # 10-14
        "b5",
        "a04_a40",
        "a14_a41b",
        "a35",
        "a45",  # 15-19
    ]

    print(f"\n{'='*60}")
    print("Gelman-Rubin R̂ (target < 1.1):")
    print(f"{'='*60}")
    converged = 0
    for j, dim in enumerate(free_dims):
        name = PARAM_NAMES[dim] if dim < len(PARAM_NAMES) else f"θ[{dim}]"
        status = "✓" if R_hat[j] < 1.1 else "✗"
        if R_hat[j] < 1.1:
            converged += 1
        print(f"  {name:>12s} (idx {dim:2d}): R̂ = {R_hat[j]:.4f} {status}")
    print(f"\nConverged: {converged}/{len(free_dims)} parameters (R̂ < 1.1)")

    # Best MAP across chains
    best_logL = -np.inf
    best_idx = 0
    for i, cfg in enumerate(configs):
        ll = cfg.get("max_logL", -np.inf)
        if ll > best_logL:
            best_logL = ll
            best_idx = i
    print(f"\nBest chain: {os.path.basename(dirs[best_idx])} (logL={best_logL:.2f})")

    # Merge posterior
    merged = np.vstack(chains)
    print(f"Merged posterior: {merged.shape[0]} total samples")

    # Posterior statistics
    print(f"\n{'='*60}")
    print("Posterior statistics (merged):")
    print(f"{'='*60}")
    print(f"  {'Param':>12s}  {'Mean':>8s}  {'Std':>8s}  {'5%':>8s}  {'95%':>8s}")
    for dim in free_dims:
        name = PARAM_NAMES[dim] if dim < len(PARAM_NAMES) else f"θ[{dim}]"
        vals = merged[:, dim]
        print(
            f"  {name:>12s}  {vals.mean():>8.4f}  {vals.std():>8.4f}  "
            f"{np.percentile(vals, 5):>8.4f}  {np.percentile(vals, 95):>8.4f}"
        )

    # Save merged results
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.expanduser(
            f"~/IKM_Hiwi/Tmcmc202601/data_5species/main/_runs/"
            f"merged_{args.condition}_production"
        )
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "samples_merged.npy"), merged)
    np.save(os.path.join(out_dir, "R_hat.npy"), R_hat)

    # Save best MAP
    best_map_path = os.path.join(dirs[best_idx], "theta_MAP.json")
    if os.path.exists(best_map_path):
        import shutil

        shutil.copy(best_map_path, os.path.join(out_dir, "theta_MAP.json"))

    summary = {
        "condition": args.condition,
        "n_chains": len(chains),
        "n_samples_per_chain": [c.shape[0] for c in chains],
        "n_total_samples": int(merged.shape[0]),
        "best_logL": float(best_logL),
        "best_chain": os.path.basename(dirs[best_idx]),
        "R_hat_max": float(R_hat.max()),
        "R_hat_mean": float(R_hat.mean()),
        "n_converged": int(converged),
        "n_free_params": int(len(free_dims)),
    }
    with open(os.path.join(out_dir, "merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {out_dir}")
    print(f"  samples_merged.npy: {merged.shape}")
    print("  R_hat.npy, theta_MAP.json, merge_summary.json")


if __name__ == "__main__":
    main()
