"""Test 0.05 interval data point selection."""
import numpy as np

# Test 0.05 interval: 0.05, 0.10, 0.15, ..., 0.95, 1.00
print("=" * 80)
print("Test 0.05 Interval (0.05, 0.10, ..., 0.95, 1.00)")
print("=" * 80)

# Calculate normalized times with 0.05 interval
normalized_times = np.arange(0.05, 1.0 + 0.001, 0.05)  # 0.05 to 1.00 inclusive
print(f"Normalized times: {normalized_times}")
print(f"Number of points: {len(normalized_times)}")
print(f"First 5: {normalized_times[:5]}")
print(f"Last 5: {normalized_times[-5:]}")

# Check intervals
intervals = np.diff(normalized_times)
print(f"\nIntervals: {intervals}")
print(f"All intervals are 0.05: {np.allclose(intervals, 0.05)}")

# For graph visualization, add margin on both sides
graph_xlim = [-0.05, 1.05]  # Left margin: 0.05, Right margin: 0.05
print(f"\nGraph xlim (with margins): {graph_xlim}")
