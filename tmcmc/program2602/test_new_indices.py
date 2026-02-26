"""Test new implementation of data point selection."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main.case2_main import select_sparse_data_indices

# Test with typical values
n_total = 2501
n_obs = 20

# Create a sample time array (uniform spacing for testing)
t_arr = np.linspace(0.0, 0.025, n_total)

# New implementation with t_arr
indices = select_sparse_data_indices(n_total, n_obs, t_arr=t_arr)

print("=" * 80)
print("New Implementation Test")
print("=" * 80)
print(f"n_total: {n_total}")
print(f"n_obs: {n_obs}")
print(f"\nSelected indices: {indices}")
print(f"\nFirst 5 indices: {indices[:5]}")
print(f"Last 5 indices: {indices[-5:]}")

# Calculate normalized time positions from actual t_arr
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized_actual = (t_arr - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_arr)
t_normalized_positions = t_normalized_actual[indices]
print("\nNormalized time positions (new):")
print(f"All positions: {t_normalized_positions}")
print("\nExpected normalized times (20 equal divisions):")
expected_normalized_times = np.linspace(0.0, 1.0, 20)
print(f"All expected: {expected_normalized_times}")

# Check if they match
if np.allclose(t_normalized_positions, expected_normalized_times, atol=1e-6):
    print("\n[OK] Normalized time positions match expected values!")
else:
    print("\n[ERROR] Normalized time positions do NOT match expected values!")
    print(
        f"Max difference: {np.max(np.abs(t_normalized_positions - expected_normalized_times)):.10f}"
    )

# Verify indices are valid
print(f"\nIndex range: [{indices.min()}, {indices.max()}]")
print(f"Valid range: [0, {n_total - 1}]")
if indices.min() >= 0 and indices.max() < n_total:
    print("[OK] All indices are within valid range!")
else:
    print("[ERROR] Some indices are out of range!")
