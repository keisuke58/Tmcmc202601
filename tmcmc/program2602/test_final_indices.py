"""Final test of data point selection with t_arr."""
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
print("Final Implementation Test (with t_arr)")
print("=" * 80)
print(f"n_total: {n_total}")
print(f"n_obs: {n_obs}")
print(f"\nSelected indices: {indices}")

# Calculate normalized time positions from actual t_arr
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized_actual = (t_arr - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_arr)
t_normalized_positions = t_normalized_actual[indices]

print(f"\nNormalized time positions (actual from t_arr):")
print(f"All positions: {t_normalized_positions}")

print(f"\nExpected normalized times (20 equal divisions):")
expected_normalized_times = np.linspace(0.0, 1.0, 20)
print(f"All expected: {expected_normalized_times}")

# Check if they match
max_diff = np.max(np.abs(t_normalized_positions - expected_normalized_times))
print(f"\nMax difference: {max_diff:.10f}")

if max_diff < 0.001:  # Allow small tolerance for integer index rounding
    print("[OK] Normalized time positions are very close to expected values!")
    print(f"    (Difference < 0.001, which is acceptable for integer indices)")
else:
    print("[WARNING] Normalized time positions differ from expected values")
    print(f"    Max difference: {max_diff:.6f}")

# Verify indices are valid
print(f"\nIndex range: [{indices.min()}, {indices.max()}]")
print(f"Valid range: [0, {n_total - 1}]")
if indices.min() >= 0 and indices.max() < n_total:
    print("[OK] All indices are within valid range!")
else:
    print("[ERROR] Some indices are out of range!")

# Show first and last indices
print(f"\nFirst index: {indices[0]} (normalized: {t_normalized_positions[0]:.6f}, expected: {expected_normalized_times[0]:.6f})")
print(f"Last index: {indices[-1]} (normalized: {t_normalized_positions[-1]:.6f}, expected: {expected_normalized_times[-1]:.6f})")
