"""Final test of 0.05 interval implementation."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main.case2_main import select_sparse_data_indices

# Test with n_obs = 20
n_total = 2501
t_arr = np.linspace(0.0, 0.025, n_total)

print("=" * 80)
print("Final Test: 0.05 Interval Implementation")
print("=" * 80)

# Test with n_obs = 20
print("\nTest: n_obs = 20")
indices = select_sparse_data_indices(n_total, 20, t_arr=t_arr)
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized = (t_arr - t_min) / (t_max - t_min)
t_positions = t_normalized[indices]

print(f"Indices (first 5): {indices[:5]}")
print(f"Indices (last 5): {indices[-5:]}")
print(f"\nNormalized time positions (first 5): {t_positions[:5]}")
print(f"Normalized time positions (last 5): {t_positions[-5:]}")

# Check if they match expected 0.05 interval
expected_times = np.arange(0.05, 1.0 + 0.001, 0.05)
print(f"\nExpected normalized times (first 5): {expected_times[:5]}")
print(f"Expected normalized times (last 5): {expected_times[-5:]}")

# Check differences
diffs = np.abs(t_positions - expected_times)
max_diff = np.max(diffs)
print(f"\nMax difference from expected: {max_diff:.6f}")

if max_diff < 0.01:
    print("[OK] Normalized time positions match expected 0.05 interval!")
else:
    print("[WARNING] Some differences found")

# Check intervals
intervals = np.diff(t_positions)
print(f"\nIntervals (first 5): {intervals[:5]}")
print(f"All intervals are approximately 0.05: {np.allclose(intervals, 0.05, atol=0.01)}")

# Graph xlim check
print(f"\nGraph xlim should be: [-0.05, 1.05]")
print("(This will be set in plot_TSM_simulation function)")
