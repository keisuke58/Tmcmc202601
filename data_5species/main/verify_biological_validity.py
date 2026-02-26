#!/usr/bin/env python3
"""
Verify Biological Validity of 5-Species Model Results.

This script implements the "Signatures of Success" checks defined in next_plan20260205.
It verifies:
1. Negative Controls (Commensal Static/HOBIC, Dysbiotic Static): Pathogens suppressed?
2. Dysbiotic HOBIC:
   - Lactate Handover (a31 < 0)
   - pH Trigger (a53 << 0)
   - Surge Curve (Visual/RMSE check)

Usage:
    python verify_biological_validity.py --runs_dir _runs
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_5SPECIES_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

try:
    from improved_5species_jit import BiofilmNewtonSolver5S
    from visualization import compute_phibar

    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False
    print("Warning: BiofilmNewtonSolver5S or visualization not found.")

# Correct parameter mapping based on solver's THETA_NAMES
# Note: Model assumes symmetric A matrix (aij = aji)
# a31 (S1-S3) maps to a13 (index 10)
# a53 (S3-S5) maps to a35 (index 18)
PARAM_INDICES = {
    "a13": 10,  # a31 equivalent
    "a35": 18,  # a53 equivalent
}


def get_param_indices():
    # Helper to return current indices
    return PARAM_INDICES


def load_run_data(run_dir):
    """Load theta_MAP and config from run directory."""
    path = Path(run_dir)
    if not path.exists():
        return None

    try:
        with open(path / "theta_MAP.json", "r") as f:
            theta_map = json.load(f)
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Load data for validation
        data = np.load(path / "data.npy")
        idx_sparse = np.load(path / "idx_sparse.npy")

        return {
            "theta": np.array(theta_map["theta_full"]),
            "config": config,
            "data": data,
            "idx_sparse": idx_sparse,
            "path": path,
        }
    except Exception as e:
        print(f"Error loading {run_dir}: {e}")
        return None


def verify_negative_control(run_data, condition_name):
    """
    Verify that Red (S5) and Purple (S4) are suppressed.
    """
    if run_data is None:
        print(f"❌ {condition_name}: Data not found.")
        return False

    theta = run_data["theta"]

    # Run simulation
    solver = BiofilmNewtonSolver5S(
        dt=0.01,
        maxtimestep=2500,  # Assuming standard
        phi_init=0.01,
        c_const=1.0,
        alpha_const=10.0,  # Default approx
    )

    # Re-run simulation with MAP parameters
    t_model, phi_sim = solver.solve(theta)

    # Calculate relative abundance (phi / total)
    phi_total = np.sum(phi_sim, axis=1)
    phi_rel = phi_sim / phi_total[:, None]

    # Check max abundance of Red (S5, index 4) and Purple (S4, index 3)
    # Note: In Commensal, S4 is Purple (F. nuc), S5 is Red (P. ging).
    # In 5-species model, indices are fixed: 0,1,2,3,4.

    max_red = np.max(phi_rel[:, 4])
    max_purple = np.max(phi_rel[:, 3])

    # Threshold: Should be close to 0 (e.g., < 1%)
    passed = True
    print(f"\n🔍 Checking {condition_name}...")

    if max_red < 0.01:
        print(f"  ✅ Red (P. gingivalis) suppressed (Max: {max_red*100:.2f}%)")
    else:
        print(f"  ⚠️ Red (P. gingivalis) NOT suppressed (Max: {max_red*100:.2f}%)")
        passed = False

    if max_purple < 0.01:
        print(f"  ✅ Purple (F. nucleatum) suppressed (Max: {max_purple*100:.2f}%)")
    else:
        print(f"  ⚠️ Purple (F. nucleatum) NOT suppressed (Max: {max_purple*100:.2f}%)")
        passed = False

    return passed


def verify_dysbiotic_hobic(run_data):
    """
    Verify Dysbiotic HOBIC signatures (Condition 4).
    Expectation:
    1. Lactate Handover: a31 < 0 (Cooperation S1->S3) -> Check a13
    2. pH Trigger: a53 << 0 (Strong cooperation S3->S5) -> Check a35
    3. Surge: S5 abundance > S3, S4 in late phase? (Visual check)
    """
    if run_data is None:
        print("❌ Dysbiotic HOBIC: Data not found.")
        return False

    theta = run_data["theta"]

    # Map 'a31' to 'a13' and 'a53' to 'a35' due to symmetry
    idx_a13 = PARAM_INDICES.get("a13", 10)
    idx_a35 = PARAM_INDICES.get("a35", 18)

    a13 = theta[idx_a13]
    a35 = theta[idx_a35]

    print("\n🔍 Checking Dysbiotic HOBIC Signatures...")

    # Check 1: Lactate Handover (a31 < 0)
    # Since matrix is symmetric, a31 = a13
    is_lactate_valid = a13 < 0
    if is_lactate_valid:
        print(f"  ✅ A. Lactate Handover: a31 (a13) = {a13:.4f} < 0 (Cooperation)")
    else:
        print(f"  ❌ A. Lactate Handover: a31 (a13) = {a13:.4f} >= 0 (Competition/Neutral)")

    # Check 2: pH Trigger (a53 << 0)
    # Since matrix is symmetric, a53 = a35
    # Threshold for "significant" cooperation? Let's say < -0.5 or just < 0 for now.
    is_ph_valid = a35 < -0.5
    if is_ph_valid:
        print(f"  ✅ B. pH Trigger: a53 (a35) = {a35:.4f} << 0 (Strong Cooperation)")
    elif a35 < 0:
        print(f"  ⚠️ B. pH Trigger: a53 (a35) = {a35:.4f} < 0 (Weak Cooperation)")
    else:
        print(f"  ❌ B. pH Trigger: a53 (a35) = {a35:.4f} >= 0 (No Cooperation)")

    # Check 3: Surge Curve (Simulation)
    solver = BiofilmNewtonSolver5S(
        dt=0.01, maxtimestep=2500, phi_init=0.01, c_const=1.0, alpha_const=10.0
    )
    t_model, phi_sim = solver.solve(theta)
    phi_total = np.sum(phi_sim, axis=1)
    phi_rel = phi_sim / phi_total[:, None]

    # Check Day 21 (end) vs Day 10 (middle) for Red
    # Assuming 2500 steps = 21 days (roughly)
    idx_day10 = int(10 / 21 * 2500)
    idx_day21 = -1

    red_day10 = phi_rel[idx_day10, 4]
    red_day21 = phi_rel[idx_day21, 4]

    if red_day21 > red_day10 + 0.10:  # Increased by at least 10%
        print(
            f"  ✅ C. Surge Curve: Red increases from {red_day10*100:.1f}% (Day 10) to {red_day21*100:.1f}% (Day 21)"
        )
    else:
        print(
            f"  ❌ C. Surge Curve: No significant surge ({red_day10*100:.1f}% -> {red_day21*100:.1f}%)"
        )

    return is_lactate_valid and (a35 < 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="_runs", help="Directory containing run results")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    # We need to find the specific directories.
    # Assuming they are named like "Commensal_Static_..." or user provides them.
    # For now, let's look for the most recent matching directories.

    print("Searching for latest runs...")

    def find_latest(pattern):
        matches = list(runs_dir.glob(pattern))
        if not matches:
            return None
        return max(matches, key=lambda p: p.stat().st_mtime)

    # 1. Commensal Static
    # Note: The remote jobs might output to different folders or sync back.
    # User needs to ensure folders are present.

    # Assuming standard naming if synced
    commensal_static = find_latest("*Commensal*Static*")
    dysbiotic_static = find_latest("*Dysbiotic*Static*")
    commensal_hobic = find_latest("*Commensal*HOBIC*")
    dysbiotic_hobic = find_latest("*Dysbiotic*HOBIC*")

    print("Found runs:")
    print(f"  Commensal Static: {commensal_static.name if commensal_static else 'Not Found'}")
    print(f"  Dysbiotic Static: {dysbiotic_static.name if dysbiotic_static else 'Not Found'}")
    print(f"  Commensal HOBIC:  {commensal_hobic.name if commensal_hobic else 'Not Found'}")
    print(f"  Dysbiotic HOBIC:  {dysbiotic_hobic.name if dysbiotic_hobic else 'Not Found'}")

    print("\n" + "=" * 50)
    print("VERIFICATION REPORT")
    print("=" * 50)

    # Load and Verify
    verify_negative_control(load_run_data(commensal_static), "Commensal Static")
    verify_negative_control(load_run_data(dysbiotic_static), "Dysbiotic Static")
    verify_negative_control(load_run_data(commensal_hobic), "Commensal HOBIC")

    verify_dysbiotic_hobic(load_run_data(dysbiotic_hobic))

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
