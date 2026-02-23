"""Test 0.25 interval data point selection."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test different approaches
n_total = 2501

# Approach 1: 0.25 interval with margin (e.g., 0.05 to 0.95)
print("=" * 80)
print("Approach 1: 0.25 interval with margin (0.05 to 0.95)")
print("=" * 80)
normalized_times_1 = np.arange(0.05, 1.0, 0.25)  # 0.05, 0.30, 0.55, 0.80
normalized_times_1 = np.append(normalized_times_1, 0.95)  # Add last point
print(f"Normalized times: {normalized_times_1}")
print(f"Number of points: {len(normalized_times_1)}")

# Approach 2: 0.25 interval from 0.0 to 1.0 (exact)
print("\n" + "=" * 80)
print("Approach 2: 0.25 interval from 0.0 to 1.0 (exact)")
print("=" * 80)
normalized_times_2 = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
print(f"Normalized times: {normalized_times_2}")
print(f"Number of points: {len(normalized_times_2)}")

# Approach 3: 0.25 interval with small margin (e.g., 0.025 to 0.975)
print("\n" + "=" * 80)
print("Approach 3: 0.25 interval with small margin (0.025 to 0.975)")
print("=" * 80)
normalized_times_3 = np.arange(0.025, 1.0, 0.25)  # 0.025, 0.275, 0.525, 0.775
normalized_times_3 = np.append(normalized_times_3, 0.975)  # Add last point
print(f"Normalized times: {normalized_times_3}")
print(f"Number of points: {len(normalized_times_3)}")

# Approach 4: 0.25 interval, but if user wants 20 points, maybe they mean 0.05 interval?
print("\n" + "=" * 80)
print("Approach 4: 0.05 interval (20 points from 0.0 to 1.0)")
print("=" * 80)
normalized_times_4 = np.linspace(0.0, 1.0, 20)
print(f"Normalized times (first 5): {normalized_times_4[:5]}")
print(f"Normalized times (last 5): {normalized_times_4[-5:]}")
print(f"Number of points: {len(normalized_times_4)}")
print(f"Interval: {normalized_times_4[1] - normalized_times_4[0]:.6f}")

# Approach 5: 0.25 interval with margin, but ensure we get exactly 5 points
print("\n" + "=" * 80)
print("Approach 5: 0.25 interval with margin (0.0 to 1.0, but with visual margin)")
print("=" * 80)
# Use 0.0, 0.25, 0.50, 0.75, 1.0 but plot with margin
normalized_times_5 = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
print(f"Normalized times: {normalized_times_5}")
print(f"Number of points: {len(normalized_times_5)}")
print("Note: Visual margin can be added in plotting, not in data selection")
