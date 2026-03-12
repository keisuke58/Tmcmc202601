#!/usr/bin/env python3
"""
A4: Generate unified training data by merging all 4 condition-specific datasets.

Appends a 4-dim condition one-hot encoding to each theta vector:
  [1,0,0,0] = Commensal_Static
  [0,1,0,0] = Commensal_HOBIC
  [0,0,1,0] = Dysbiotic_Static
  [0,0,0,1] = Dysbiotic_HOBIC

Output: theta (N_total, 24), phi (N_total, T, 5), t (T,), bounds (24, 2)

Usage:
  python generate_unified_data.py [--max-per-condition 10000]
"""

import argparse
import numpy as np
from pathlib import Path

CONDITION_ORDER = [
    "Commensal_Static",
    "Commensal_HOBIC",
    "Dysbiotic_Static",
    "Dysbiotic_HOBIC",
]

CONDITION_ONEHOT = {
    "Commensal_Static": [1, 0, 0, 0],
    "Commensal_HOBIC": [0, 1, 0, 0],
    "Dysbiotic_Static": [0, 0, 1, 0],
    "Dysbiotic_HOBIC": [0, 0, 0, 1],
}


def find_best_data(data_dir: Path, condition: str) -> Path:
    """Find largest available dataset for a condition."""
    candidates = sorted(
        data_dir.glob(f"train_{condition}_N*.npz"),
        key=lambda p: int(p.stem.split("_N")[-1]),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No data found for {condition} in {data_dir}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="deeponet/data", help="Data directory")
    parser.add_argument(
        "--max-per-condition", type=int, default=0, help="Max samples per condition (0 = use all)"
    )
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / args.data_dir

    all_theta, all_phi = [], []
    ref_t = None
    ref_bounds = None
    total_n = 0

    for cond in CONDITION_ORDER:
        path = find_best_data(data_dir, cond)
        data = np.load(path)
        theta = data["theta"]  # (N, 20)
        phi = data["phi"]  # (N, T, 5)
        t = data["t"]  # (T,)
        bounds = data["bounds"]  # (20, 2)

        N = theta.shape[0]
        if args.max_per_condition > 0:
            N = min(N, args.max_per_condition)
            theta = theta[:N]
            phi = phi[:N]

        # Verify time grids match
        if ref_t is None:
            ref_t = t
            ref_bounds = bounds
        else:
            assert len(t) == len(ref_t), f"Time grid mismatch: {len(t)} vs {len(ref_t)}"

        # Append condition one-hot to theta: (N, 20) → (N, 24)
        onehot = np.tile(CONDITION_ONEHOT[cond], (N, 1))  # (N, 4)
        theta_ext = np.hstack([theta, onehot])  # (N, 24)

        all_theta.append(theta_ext)
        all_phi.append(phi)
        total_n += N

        print(f"  {cond}: {path.name}, N={N}, theta→(N,24)")

    # Concatenate
    theta_merged = np.concatenate(all_theta, axis=0)  # (N_total, 24)
    phi_merged = np.concatenate(all_phi, axis=0)  # (N_total, T, 5)

    # Extended bounds: original 20 params + 4 one-hot (bounds [0, 1])
    onehot_bounds = np.array([[0, 1]] * 4)  # (4, 2)
    bounds_ext = np.vstack([ref_bounds, onehot_bounds])  # (24, 2)

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(total_n)
    theta_merged = theta_merged[perm]
    phi_merged = phi_merged[perm]

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = data_dir / f"train_Unified_N{total_n}.npz"

    np.savez(
        out_path,
        theta=theta_merged,
        phi=phi_merged,
        t=ref_t,
        bounds=bounds_ext,
    )
    print(f"\nSaved unified data: {out_path}")
    print(f"  Total: N={total_n}, theta_dim=24, T={len(ref_t)}, 5 species")
    print(f"  Shape: theta={theta_merged.shape}, phi={phi_merged.shape}")


if __name__ == "__main__":
    main()
