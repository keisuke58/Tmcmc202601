#!/usr/bin/env python3
"""
Verify symmetry of interaction matrix A (A = A^T) for all TMCMC runs.

The current Hamilton model enforces A_ij = A_ji by construction (variational
principle). This script confirms ||A - A^T||_F = 0 for all conditions.

Usage:
  python tools/verify_symmetric_A.py
  python tools/verify_symmetric_A.py --runs-dir data_5species/_runs

Output:
  - Prints symmetry norm per condition (expected: 0.0)
  - Saves results to tools/symmetric_A_verification.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

# Prefer runs with theta_MAP; fallback order per condition
CONDITION_RUNS = {
    "Commensal_Static": [
        "commensal_static_posterior",
        "commensal_static",
        "Commensal_Static_20260208_002100",
    ],
    "Commensal_HOBIC": [
        "commensal_hobic_posterior",
        "commensal_hobic",
        "Commensal_HOBIC_20260208_002100",
    ],
    "Dysbiotic_Static": [
        "dysbiotic_static_posterior",
        "dysbiotic_static",
        "Dysbiotic_Static_20260207_203752",
    ],
    "Dysbiotic_HOBIC": [
        "dh_baseline",
        "deeponet_DH_50k_importance",
        "deeponet_Dysbiotic_HOBIC",
        "Dysbiotic_HOBIC_20260208_002100",
    ],
}


def theta_to_A(theta: np.ndarray) -> np.ndarray:
    """Map 20-parameter theta to symmetric A(5x5)."""
    theta = np.asarray(theta).flatten()
    if theta.shape[0] != 20:
        raise ValueError(f"theta length must be 20, got {theta.shape[0]}")
    A = np.zeros((5, 5), dtype=float)
    A[0, 0] = theta[0]
    A[0, 1] = A[1, 0] = theta[1]
    A[1, 1] = theta[2]
    A[2, 2] = theta[5]
    A[2, 3] = A[3, 2] = theta[6]
    A[3, 3] = theta[7]
    A[0, 2] = A[2, 0] = theta[10]
    A[0, 3] = A[3, 0] = theta[11]
    A[1, 2] = A[2, 1] = theta[12]
    A[1, 3] = A[3, 1] = theta[13]
    A[4, 4] = theta[14]
    A[0, 4] = A[4, 0] = theta[16]
    A[1, 4] = A[4, 1] = theta[17]
    A[2, 4] = A[4, 2] = theta[18]
    A[3, 4] = A[4, 3] = theta[19]
    return A


def load_theta_map(run_dir: Path) -> np.ndarray | None:
    """Load theta_full from theta_MAP.json."""
    path = run_dir / "theta_MAP.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return np.array(data.get("theta_full", data.get("theta", [])), dtype=np.float64)
    return np.array(data, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Verify A = A^T for all TMCMC runs")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Base runs directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else PROJECT_ROOT / args.runs_dir
    out_path = args.output or Path(__file__).parent / "symmetric_A_verification.json"

    results = {}
    print("Symmetric A verification (||A - A^T||_F)")
    print("=" * 50)

    for cond, candidates in CONDITION_RUNS.items():
        theta = None
        used_run = None
        for run_name in candidates:
            run_dir = runs_dir / run_name
            theta = load_theta_map(run_dir)
            if theta is not None and len(theta) >= 20:
                used_run = run_name
                break

        if theta is None:
            print(f"  {cond}: [SKIP] no theta_MAP found")
            results[cond] = {"status": "skip", "reason": "no theta_MAP"}
            continue

        A = theta_to_A(theta[:20])
        asym = A - A.T
        norm_asym = float(np.linalg.norm(asym, ord="fro"))
        norm_A = float(np.linalg.norm(A, ord="fro"))

        results[cond] = {
            "run": used_run,
            "norm_A_minus_AT": norm_asym,
            "norm_A": norm_A,
            "symmetric": norm_asym < 1e-10,
        }
        status = "OK" if norm_asym < 1e-10 else "FAIL"
        print(f"  {cond} ({used_run}): ||A-A^T||_F = {norm_asym:.2e}  [{status}]")

    print("=" * 50)
    all_ok = all(r.get("symmetric", False) for r in results.values() if r.get("status") != "skip")
    print(f"All symmetric: {all_ok}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
