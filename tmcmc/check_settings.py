"""Check current settings for particles and noise."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import TMCMC_DEFAULTS

print("=" * 80)
print("Current Default Settings")
print("=" * 80)

print("\nTMCMC Settings (defaults):")
print(f"  n_particles: {TMCMC_DEFAULTS.n_particles}")
print(f"  n_stages: {TMCMC_DEFAULTS.n_stages}")
print(f"  n_mutation_steps: {TMCMC_DEFAULTS.n_mutation_steps}")

print("\nData Generation Settings (defaults):")
print(f"  sigma_obs: 0.01 (observation noise)")
print(f"  n_data: 20 (number of observations)")

print("\n" + "=" * 80)
print("How to Change Settings")
print("=" * 80)

print("\n1. Change number of particles:")
print("   python -m main.case2_main --mode debug --n-particles 500")

print("\n2. Change noise level (sigma_obs):")
print("   python -m main.case2_main --mode debug --sigma-obs 0.001")
print("   (smaller value = less noise)")

print("\n3. No noise (training data):")
print("   python -m main.case2_main --mode debug --no-noise")

print("\n4. Combine options:")
print("   python -m main.case2_main --mode debug --n-particles 500 --sigma-obs 0.001")

print("\n" + "=" * 80)
print("Recommended Settings for Small Noise (not zero)")
print("=" * 80)
print("  --sigma-obs 0.001  (10x smaller than default 0.01)")
print("  --sigma-obs 0.0001 (100x smaller than default 0.01)")
print("  --sigma-obs 0.005  (2x smaller than default 0.01)")
