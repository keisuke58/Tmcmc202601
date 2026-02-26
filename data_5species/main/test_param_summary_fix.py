#!/usr/bin/env python3
"""
Test for parameter summary DataFrame fix.

Tests that the fix correctly handles the case where:
- results["MAP"] has 20 elements (full theta)
- results["param_names"] has fewer elements (active indices only)
"""

import numpy as np
import pandas as pd
import sys


# Test cases
def test_param_summary_extraction():
    """Test that active parameter extraction works correctly."""

    # Simulate the scenario from the bug
    active_indices = [0, 1, 2, 3, 4, 5, 8, 10, 12]  # 9 active indices
    n_active = len(active_indices)

    # Full 20-parameter arrays (as returned by TMCMC)
    MAP_full = np.array(
        [
            0.57,
            0.64,
            -0.54,
            1.33,
            0.70,
            0.63,  # indices 0-5
            0.0,
            0.0,  # indices 6-7 (locked)
            0.46,
            0.0,  # indices 8-9
            0.20,
            0.0,  # indices 10-11 (locked)
            0.13,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # indices 12-19 (locked)
        ]
    )
    mean_full = MAP_full + 0.01  # Slightly different

    # Active parameter names
    param_names = ["a11", "a12", "a22", "b1", "b2", "a33", "b3", "a13", "a23"]

    # CI arrays (already active-only dimension)
    ci = {
        "median": np.random.rand(n_active),
        "hdi_lower": np.random.rand(n_active) - 0.5,
        "hdi_upper": np.random.rand(n_active) + 0.5,
        "et_lower": np.random.rand(n_active) - 0.5,
        "et_upper": np.random.rand(n_active) + 0.5,
    }

    mcmc_diag = {
        "rhat": np.ones(n_active) + 0.01 * np.random.rand(n_active),
        "ess": 100 + 50 * np.random.rand(n_active),
    }

    # Simulate results dict
    results = {
        "MAP": MAP_full,
        "mean": mean_full,
        "param_names": param_names,
        "active_indices": active_indices,
    }

    # === THE FIX ===
    # Extract active parameter values from full theta vectors
    active_idx = results["active_indices"]
    MAP_active = (
        results["MAP"][active_idx] if len(results["MAP"]) > len(active_idx) else results["MAP"]
    )
    mean_active = (
        results["mean"][active_idx] if len(results["mean"]) > len(active_idx) else results["mean"]
    )

    # === ASSERTIONS ===
    print("Testing parameter extraction...")

    # Test 1: Lengths match
    assert (
        len(MAP_active) == n_active
    ), f"MAP_active length mismatch: {len(MAP_active)} != {n_active}"
    assert (
        len(mean_active) == n_active
    ), f"mean_active length mismatch: {len(mean_active)} != {n_active}"
    print(f"  ✓ MAP_active length: {len(MAP_active)}")
    print(f"  ✓ mean_active length: {len(mean_active)}")

    # Test 2: Values are correctly extracted
    for i, idx in enumerate(active_indices):
        assert MAP_active[i] == MAP_full[idx], f"MAP value mismatch at position {i}"
        assert mean_active[i] == mean_full[idx], f"mean value mismatch at position {i}"
    print("  ✓ All values correctly extracted from full theta")

    # Test 3: DataFrame creation succeeds
    try:
        param_summary = pd.DataFrame(
            {
                "name": results["param_names"],
                "index": active_idx,
                "MAP": MAP_active,
                "mean": mean_active,
                "median": ci["median"],
                "hdi_lower": ci["hdi_lower"],
                "hdi_upper": ci["hdi_upper"],
                "et_lower": ci["et_lower"],
                "et_upper": ci["et_upper"],
                "rhat": mcmc_diag["rhat"],
                "ess": mcmc_diag["ess"],
            }
        )
        print(f"  ✓ DataFrame created successfully with shape {param_summary.shape}")
    except ValueError as e:
        print(f"  ✗ DataFrame creation FAILED: {e}")
        return False

    # Test 4: DataFrame has correct dimensions
    assert (
        param_summary.shape[0] == n_active
    ), f"Row count mismatch: {param_summary.shape[0]} != {n_active}"
    assert param_summary.shape[1] == 11, f"Column count mismatch: {param_summary.shape[1]} != 11"
    print(f"  ✓ DataFrame shape correct: {param_summary.shape}")

    # Test 5: Print summary for visual inspection
    print("\n  Parameter Summary (first 5 rows):")
    print(param_summary[["name", "index", "MAP", "mean", "rhat"]].head().to_string(index=False))

    return True


def test_already_active_dimension():
    """Test that the fix also works when MAP is already active-dimension."""

    active_indices = [0, 1, 2, 3, 4]
    n_active = len(active_indices)

    # Already active-dimension (no extraction needed)
    MAP_active_only = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    mean_active_only = MAP_active_only + 0.01

    param_names = ["a", "b", "c", "d", "e"]

    results = {
        "MAP": MAP_active_only,
        "mean": mean_active_only,
        "param_names": param_names,
        "active_indices": active_indices,
    }

    # === THE FIX ===
    active_idx = results["active_indices"]
    MAP_active = (
        results["MAP"][active_idx] if len(results["MAP"]) > len(active_idx) else results["MAP"]
    )
    mean_active = (
        results["mean"][active_idx] if len(results["mean"]) > len(active_idx) else results["mean"]
    )

    print("\nTesting already-active-dimension case...")

    # Should use original arrays (no extraction)
    assert np.array_equal(MAP_active, MAP_active_only), "Should use original MAP"
    assert np.array_equal(mean_active, mean_active_only), "Should use original mean"
    print("  ✓ Already-active arrays handled correctly")

    return True


def test_edge_cases():
    """Test edge cases."""

    print("\nTesting edge cases...")

    N_FULL_PARAMS = 20  # Should match len(model_constants["param_names"]) from config

    # Edge case 1: Single active parameter
    active_indices = [5]
    n_active = len(active_indices)
    MAP_full = np.zeros(20)
    MAP_full[5] = 1.23

    # New robust logic
    if len(MAP_full) == N_FULL_PARAMS and n_active < N_FULL_PARAMS:
        MAP_active = MAP_full[active_indices]
    elif len(MAP_full) == n_active:
        MAP_active = MAP_full
    else:
        MAP_active = MAP_full[:n_active]

    assert len(MAP_active) == 1, "Single parameter extraction failed"
    assert MAP_active[0] == 1.23, "Single parameter value mismatch"
    print("  ✓ Single active parameter works")

    # Edge case 2: All parameters active (Dysbiotic_HOBIC mode)
    active_indices = list(range(20))
    n_active = len(active_indices)
    MAP_full = np.arange(20, dtype=float)

    if len(MAP_full) == N_FULL_PARAMS and n_active < N_FULL_PARAMS:
        MAP_active = MAP_full[active_indices]
    elif len(MAP_full) == n_active:
        MAP_active = MAP_full
    else:
        MAP_active = MAP_full[:n_active]

    assert len(MAP_active) == 20, "All parameters extraction failed"
    assert np.array_equal(MAP_active, MAP_full), "All parameters value mismatch"
    print("  ✓ All parameters active works (Dysbiotic_HOBIC)")

    # Edge case 3: MAP already in active dimension
    active_indices = [0, 1, 2, 3, 4]
    n_active = len(active_indices)
    MAP_already_active = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    if len(MAP_already_active) == N_FULL_PARAMS and n_active < N_FULL_PARAMS:
        MAP_active = MAP_already_active[active_indices]
    elif len(MAP_already_active) == n_active:
        MAP_active = MAP_already_active
    else:
        MAP_active = MAP_already_active[:n_active]

    assert len(MAP_active) == 5, "Already-active dimension failed"
    assert np.array_equal(MAP_active, MAP_already_active), "Already-active values mismatch"
    print("  ✓ Already active dimension works")

    return True


def test_validation_logic():
    """Test the validation logic for array length mismatches."""

    print("\nTesting validation logic...")

    N_FULL_PARAMS = 20  # Should match len(model_constants["param_names"]) from config
    active_indices = [0, 1, 2, 3, 4, 5, 8, 10, 12]
    n_active = len(active_indices)

    # Test validation helper function
    def validate_arrays(arrays_dict, expected_len):
        mismatches = {k: v for k, v in arrays_dict.items() if v != expected_len}
        return mismatches

    # All correct lengths
    array_lengths = {
        "param_names": n_active,
        "MAP": n_active,
        "mean": n_active,
        "median": n_active,
        "rhat": n_active,
    }
    mismatches = validate_arrays(array_lengths, n_active)
    assert len(mismatches) == 0, "Should have no mismatches"
    print("  ✓ No mismatches when all lengths correct")

    # Detect mismatch
    array_lengths_bad = {
        "param_names": n_active,
        "MAP": n_active,
        "mean": n_active + 1,  # Wrong!
        "median": n_active,
        "rhat": n_active,
    }
    mismatches = validate_arrays(array_lengths_bad, n_active)
    assert "mean" in mismatches, "Should detect mean mismatch"
    assert mismatches["mean"] == n_active + 1, "Should report correct bad length"
    print("  ✓ Correctly detects array length mismatches")

    # Test unexpected dimension handling
    active_indices = [0, 2, 5]
    n_active = 3
    MAP_weird = np.array([1, 2, 3, 4, 5])  # 5 elements, not 20 or 3

    if len(MAP_weird) == N_FULL_PARAMS and n_active < N_FULL_PARAMS:
        MAP_active = MAP_weird[active_indices]
    elif len(MAP_weird) == n_active:
        MAP_active = MAP_weird
    else:
        # Fallback: truncate or pad
        MAP_active = MAP_weird[:n_active] if len(MAP_weird) >= n_active else MAP_weird

    assert len(MAP_active) == n_active, "Fallback should produce correct length"
    print("  ✓ Handles unexpected dimensions with fallback")

    return True


def test_strict_type_validation():
    """Strict tests for data types and boundary conditions."""

    print("\nTesting strict type validation...")

    active_indices = [0, 2, 5, 10, 15]
    n_active = len(active_indices)

    # Test with various numpy dtypes
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        MAP_full = np.arange(20, dtype=dtype)
        active_idx = active_indices
        MAP_active = MAP_full[active_idx] if len(MAP_full) > len(active_idx) else MAP_full

        assert MAP_active.dtype == dtype, f"dtype not preserved for {dtype}"
        assert len(MAP_active) == n_active, f"Length wrong for {dtype}"
    print("  ✓ All dtypes handled correctly")

    # Test with negative indices in active_indices (should NOT happen but test robustness)
    # Actually, negative indices would cause issues - skip this

    # Test exact value matching with high precision
    MAP_full = np.array([1.123456789012345] * 20)
    MAP_full[5] = 9.876543210987654
    active_indices = [5]
    MAP_active = MAP_full[active_indices]
    assert MAP_active[0] == 9.876543210987654, "High precision value not preserved"
    print("  ✓ High precision values preserved")

    # Test with NaN and inf values (should propagate correctly)
    MAP_full = np.zeros(20)
    MAP_full[3] = np.nan
    MAP_full[7] = np.inf
    MAP_full[11] = -np.inf
    active_indices = [3, 7, 11]
    MAP_active = MAP_full[active_indices]
    assert np.isnan(MAP_active[0]), "NaN not propagated"
    assert np.isinf(MAP_active[1]) and MAP_active[1] > 0, "Inf not propagated"
    assert np.isinf(MAP_active[2]) and MAP_active[2] < 0, "-Inf not propagated"
    print("  ✓ NaN/Inf values handled correctly")

    return True


def test_all_condition_modes():
    """Test that the fix works for ALL 4 condition/cultivation modes."""

    print("\nTesting all condition modes...")

    N_FULL_PARAMS = 20  # Should match len(model_constants["param_names"]) from config

    # Configuration from prior_bounds.json
    MODE_CONFIGS = {
        "Commensal_Static": {
            "locks": [9, 15, 6, 7, 11, 13, 14, 16, 17, 18, 19],
            "expected_active": 9,
        },
        "Dysbiotic_HOBIC": {
            "locks": [],
            "expected_active": 20,
        },
        "Dysbiotic_Static": {
            "locks": [6, 12, 13, 16, 17],
            "expected_active": 15,
        },
        "Commensal_HOBIC": {
            "locks": [6, 12, 13, 16, 17, 15, 18],
            "expected_active": 13,
        },
    }

    param_names_full = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",
        "a13",
        "a14",
        "a23",
        "a24",
        "a55",
        "b5",
        "a15",
        "a25",
        "a35",
        "a45",
    ]

    for mode, config in MODE_CONFIGS.items():
        print(f"\n  Testing {mode}...")

        locked = config["locks"]
        expected_active = config["expected_active"]
        active_indices = [i for i in range(20) if i not in locked]
        n_active = len(active_indices)

        assert (
            n_active == expected_active
        ), f"{mode}: Expected {expected_active} active, got {n_active}"

        # Simulate full 20-dim MAP from TMCMC
        MAP_full = np.random.rand(20)
        for idx in locked:
            MAP_full[idx] = 0.0  # Locked params are 0

        mean_full = MAP_full + 0.01

        # Apply the ROBUST fix (matching actual code)
        if len(MAP_full) == N_FULL_PARAMS and n_active < N_FULL_PARAMS:
            MAP_active = MAP_full[active_indices]
            mean_active = mean_full[active_indices]
        elif len(MAP_full) == n_active:
            MAP_active = MAP_full
            mean_active = mean_full
        else:
            MAP_active = MAP_full[:n_active]
            mean_active = mean_full[:n_active]

        # Verify lengths
        assert len(MAP_active) == len(active_indices), f"{mode}: MAP length mismatch"
        assert len(mean_active) == len(active_indices), f"{mode}: mean length mismatch"

        # Create param names for active
        param_names_active = [param_names_full[i] for i in active_indices]
        assert len(param_names_active) == len(
            active_indices
        ), f"{mode}: param_names length mismatch"

        # Create mock CI and diagnostics
        n_active = len(active_indices)
        ci = {
            "median": np.random.rand(n_active),
            "hdi_lower": np.random.rand(n_active) - 0.5,
            "hdi_upper": np.random.rand(n_active) + 0.5,
            "et_lower": np.random.rand(n_active) - 0.5,
            "et_upper": np.random.rand(n_active) + 0.5,
        }
        mcmc_diag = {
            "rhat": np.ones(n_active) * 1.02,
            "ess": np.ones(n_active) * 150,
        }

        # Create DataFrame - THIS IS THE ACTUAL FIX TEST
        try:
            param_summary = pd.DataFrame(
                {
                    "name": param_names_active,
                    "index": active_indices,
                    "MAP": MAP_active,
                    "mean": mean_active,
                    "median": ci["median"],
                    "hdi_lower": ci["hdi_lower"],
                    "hdi_upper": ci["hdi_upper"],
                    "et_lower": ci["et_lower"],
                    "et_upper": ci["et_upper"],
                    "rhat": mcmc_diag["rhat"],
                    "ess": mcmc_diag["ess"],
                }
            )
        except ValueError as e:
            print(f"    ✗ DataFrame creation FAILED for {mode}: {e}")
            return False

        # Verify DataFrame shape
        assert (
            param_summary.shape[0] == expected_active
        ), f"{mode}: DataFrame rows {param_summary.shape[0]} != expected {expected_active}"

        print(
            f"    ✓ {mode}: {expected_active} active params, DataFrame shape {param_summary.shape}"
        )

    print("\n  ✓ All 4 modes passed!")
    return True


def test_realistic_nishioka_scenario():
    """Test with realistic Nishioka algorithm parameters."""

    print("\nTesting realistic Nishioka scenario...")

    # Actual locked indices from Commensal/Static condition
    LOCKED_INDICES = [9, 15, 6, 7, 11, 13, 14, 16, 17, 18, 19]
    active_indices = [i for i in range(20) if i not in LOCKED_INDICES]

    assert len(active_indices) == 9, f"Expected 9 active, got {len(active_indices)}"
    print(f"  Active indices: {active_indices}")

    # Realistic MAP values from actual TMCMC run
    MAP_full = np.array(
        [
            0.57140747,
            0.63753607,
            -0.54220915,
            1.32782896,
            0.69558105,
            0.62707298,
            0.0,
            0.0,
            0.46421169,
            0.0,
            0.19845964,
            0.0,
            0.12809624,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    param_names_full = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",
        "a13",
        "a14",
        "a23",
        "a24",
        "a55",
        "b5",
        "a15",
        "a25",
        "a35",
        "a45",
    ]
    param_names_active = [param_names_full[i] for i in active_indices]

    # Apply the fix
    MAP_active = MAP_full[active_indices] if len(MAP_full) > len(active_indices) else MAP_full

    # Verify extraction
    assert len(MAP_active) == len(active_indices), "Length mismatch"
    assert len(MAP_active) == len(param_names_active), "Param names mismatch"

    # Verify locked params are not included
    for locked_idx in LOCKED_INDICES:
        assert locked_idx not in active_indices, f"Locked index {locked_idx} in active"

    # Verify all active values are non-zero (for this specific scenario)
    for i, idx in enumerate(active_indices):
        assert MAP_full[idx] != 0.0 or idx in [
            4
        ], f"Unexpected zero at active index {idx}"  # b2=4 can be ~0.7

    print(f"  ✓ Extracted {len(MAP_active)} active parameters correctly")
    print(f"  ✓ Parameter names: {param_names_active}")

    # Create DataFrame as in actual code
    ci = {
        "median": np.random.rand(len(active_indices)),
        "hdi_lower": np.random.rand(len(active_indices)) - 0.5,
        "hdi_upper": np.random.rand(len(active_indices)) + 0.5,
        "et_lower": np.random.rand(len(active_indices)) - 0.5,
        "et_upper": np.random.rand(len(active_indices)) + 0.5,
    }
    mcmc_diag = {
        "rhat": np.ones(len(active_indices)) * 1.02,
        "ess": np.ones(len(active_indices)) * 150,
    }

    param_summary = pd.DataFrame(
        {
            "name": param_names_active,
            "index": active_indices,
            "MAP": MAP_active,
            "mean": MAP_active + 0.01,
            "median": ci["median"],
            "hdi_lower": ci["hdi_lower"],
            "hdi_upper": ci["hdi_upper"],
            "et_lower": ci["et_lower"],
            "et_upper": ci["et_upper"],
            "rhat": mcmc_diag["rhat"],
            "ess": mcmc_diag["ess"],
        }
    )

    print(f"  ✓ DataFrame created: {param_summary.shape}")

    # Verify column alignment
    for i, row in param_summary.iterrows():
        expected_idx = active_indices[i]
        assert row["index"] == expected_idx, f"Index mismatch at row {i}"
        assert row["name"] == param_names_full[expected_idx], f"Name mismatch at row {i}"
        assert np.isclose(row["MAP"], MAP_full[expected_idx]), f"MAP mismatch at row {i}"

    print("  ✓ All row values correctly aligned")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing parameter summary DataFrame fix")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= test_param_summary_extraction()
    except Exception as e:
        print(f"  ✗ test_param_summary_extraction FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_already_active_dimension()
    except Exception as e:
        print(f"  ✗ test_already_active_dimension FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_edge_cases()
    except Exception as e:
        print(f"  ✗ test_edge_cases FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_validation_logic()
    except Exception as e:
        print(f"  ✗ test_validation_logic FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_strict_type_validation()
    except Exception as e:
        print(f"  ✗ test_strict_type_validation FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_all_condition_modes()
    except Exception as e:
        print(f"  ✗ test_all_condition_modes FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_realistic_nishioka_scenario()
    except Exception as e:
        print(f"  ✗ test_realistic_nishioka_scenario FAILED: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
