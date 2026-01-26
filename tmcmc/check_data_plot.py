"""Check if data points match the plot."""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar

run_dir = Path("tmcmc/_runs/20260122_151859_debug_seed42")

# Load data
data_M1 = np.load(run_dir / "data_M1.npy")
idx_M1 = np.load(run_dir / "idx_M1.npy")
t_M1 = np.load(run_dir / "t_M1.npy")
x0_M1 = np.load(run_dir / "x0_M1.npy") if (run_dir / "x0_M1.npy").exists() else None

print("=" * 80)
print("Data Check for M1")
print("=" * 80)
print(f"Data shape: {data_M1.shape}")
print(f"Index shape: {idx_M1.shape}")
print(f"Time shape: {t_M1.shape}")
print(f"\nFirst 10 indices: {idx_M1[:10]}")
print(f"\nFirst 10 data points:\n{data_M1[:10]}")

# Normalize time
t_min = t_M1.min()
t_max = t_M1.max()
t_normalized = (t_M1 - t_min) / (t_max - t_min)
t_obs_normalized = t_normalized[idx_M1]

print(f"\nTime range: [{t_min:.6f}, {t_max:.6f}]")
print(f"\nFirst 10 normalized observation times:\n{t_obs_normalized[:10]}")

# Compute phibar if x0 is available
if x0_M1 is not None:
    active_species = [0, 1]  # M1 active species
    phibar = compute_phibar(x0_M1, active_species)
    print(f"\nphibar shape: {phibar.shape}")
    print(f"\nphibar at observation indices (first 10):\n{phibar[idx_M1[:10], :]}")
    print(f"\nData points (first 10):\n{data_M1[:10]}")
    print(f"\nDifference (data - phibar at indices, first 10):\n{data_M1[:10] - phibar[idx_M1[:10], :]}")
    
    # Check if data matches phibar + noise
    print("\n" + "=" * 80)
    print("Verification:")
    print("=" * 80)
    print("Expected: data[i] ≈ phibar[idx_sparse[i]] + noise")
    print("Checking first 5 points:")
    for i in range(min(5, len(idx_M1))):
        idx = idx_M1[i]
        phibar_val = phibar[idx, :]
        data_val = data_M1[i, :]
        diff = data_val - phibar_val
        print(f"\nPoint {i}:")
        print(f"  Index: {idx}")
        print(f"  Normalized time: {t_obs_normalized[i]:.6f}")
        print(f"  phibar: {phibar_val}")
        print(f"  data: {data_val}")
        print(f"  diff: {diff}")
else:
    print("\nNote: x0_M1.npy not found. Cannot verify phibar values.")
