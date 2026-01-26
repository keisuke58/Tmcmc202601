"""Check current implementation of data point selection."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main.case2_main import select_sparse_data_indices

# Test with typical values
n_total = 2501
n_obs = 20

# Current implementation
indices = select_sparse_data_indices(n_total, n_obs)

print("=" * 80)
print("Current Implementation Check")
print("=" * 80)
print(f"n_total: {n_total}")
print(f"n_obs: {n_obs}")
print(f"\nSelected indices: {indices}")
print(f"\nFirst 5 indices: {indices[:5]}")
print(f"Last 5 indices: {indices[-5:]}")

# Calculate normalized time positions
t_normalized_positions = indices / (n_total - 1)
print(f"\nNormalized time positions (current):")
print(f"First 5: {t_normalized_positions[:5]}")
print(f"Last 5: {t_normalized_positions[-5:]}")

# What user wants: 20 equal divisions of normalized time [0.0, 1.0]
desired_normalized_times = np.linspace(0.0, 1.0, 20)
print(f"\nDesired normalized times (20 equal divisions):")
print(f"First 5: {desired_normalized_times[:5]}")
print(f"Last 5: {desired_normalized_times[-5:]}")

# Convert to indices
desired_indices = np.round(desired_normalized_times * (n_total - 1)).astype(int)
print(f"\nDesired indices (from normalized time):")
print(f"First 5: {desired_indices[:5]}")
print(f"Last 5: {desired_indices[-5:]}")

print("\n" + "=" * 80)
print("Comparison:")
print("=" * 80)
print(f"Current first index: {indices[0]} (normalized: {t_normalized_positions[0]:.4f})")
print(f"Desired first index: {desired_indices[0]} (normalized: {desired_normalized_times[0]:.4f})")
print(f"\nCurrent last index: {indices[-1]} (normalized: {t_normalized_positions[-1]:.4f})")
print(f"Desired last index: {desired_indices[-1]} (normalized: {desired_normalized_times[-1]:.4f})")
