"""Test 0.25 interval implementation."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main.case2_main import select_sparse_data_indices

# Test with different n_obs values
n_total = 2501
t_arr = np.linspace(0.0, 0.025, n_total)

print("=" * 80)
print("Test 0.25 Interval Implementation")
print("=" * 80)

# Test with n_obs = 5 (should give 0.025, 0.25, 0.50, 0.75, 0.975)
print("\nTest 1: n_obs = 5")
indices_5 = select_sparse_data_indices(n_total, 5, t_arr=t_arr)
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized = (t_arr - t_min) / (t_max - t_min)
t_positions_5 = t_normalized[indices_5]
print(f"Indices: {indices_5}")
print(f"Normalized time positions: {t_positions_5}")
print("Expected: [0.025, 0.25, 0.50, 0.75, 0.975]")

# Test with n_obs = 20 (should use 0.25 interval pattern)
print("\nTest 2: n_obs = 20")
indices_20 = select_sparse_data_indices(n_total, 20, t_arr=t_arr)
t_positions_20 = t_normalized[indices_20]
print(f"First 5 indices: {indices_20[:5]}")
print(f"Last 5 indices: {indices_20[-5:]}")
print(f"First 5 normalized times: {t_positions_20[:5]}")
print(f"Last 5 normalized times: {t_positions_20[-5:]}")
print(f"Intervals (first 5): {np.diff(t_positions_20[:6])}")

# Check if 0.25 interval points are included
print("\nChecking for 0.25 interval points:")
target_times = np.array([0.025, 0.25, 0.50, 0.75, 0.975])
for target in target_times:
    closest_idx = np.argmin(np.abs(t_positions_20 - target))
    closest_time = t_positions_20[closest_idx]
    diff = abs(closest_time - target)
    print(f"  Target {target:.3f}: closest is {closest_time:.6f} (diff: {diff:.6f})")
