"""Check vertical (value) mismatch between data points and model line."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices

# Load latest run data
run_dir = Path("tmcmc/_runs/20260122_154237_debug_seed42")

if not run_dir.exists():
    print(f"Run directory not found: {run_dir}")
    sys.exit(1)

# Load saved data
data_M1 = np.load(run_dir / "data_M1.npy")
idx_M1 = np.load(run_dir / "idx_M1.npy")
t_M1 = np.load(run_dir / "t_M1.npy")

print("=" * 80)
print("Vertical Mismatch Check")
print("=" * 80)
print(f"Loaded data shape: {data_M1.shape}")
print(f"Loaded index shape: {idx_M1.shape}")
print(f"Loaded time shape: {t_M1.shape}")

# Re-generate to get x0 and phibar
config = MODEL_CONFIGS["M1"]
theta_true = get_theta_true()
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_random_seed = 42
name = "M1"

solver_kwargs = {
    k: v for k, v in config.items() if k not in ["active_species", "active_indices", "param_names"]
}

solver = BiofilmNewtonSolver(
    **solver_kwargs,
    active_species=config["active_species"],
    use_numba=HAS_NUMBA,
)

tsm = BiofilmTSM_Analytical(
    solver,
    active_theta_indices=config["active_indices"],
    cov_rel=exp_config_cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

t_arr, x0, sig2 = tsm.solve_tsm(theta_true)

# Compute phibar (same as data generation)
phibar = compute_phibar(x0, config["active_species"])

print(f"\nphibar shape: {phibar.shape}")
print(f"phibar at first 5 observation indices:\n{phibar[idx_M1[:5], :]}")

# Check if data matches phibar + noise
print("\n" + "=" * 80)
print("Data vs Model Comparison:")
print("=" * 80)

for i, sp in enumerate(config["active_species"]):
    phibar_at_obs = phibar[idx_M1, i]
    data_at_obs = data_M1[:, i]
    residuals = data_at_obs - phibar_at_obs

    print(f"\nSpecies {sp+1}:")
    print(f"  First 5 phibar values: {phibar_at_obs[:5]}")
    print(f"  First 5 data values: {data_at_obs[:5]}")
    print(f"  First 5 residuals: {residuals[:5]}")
    print(f"  Residual std: {np.std(residuals):.6f}")
    print(f"  Expected noise std: {exp_config_sigma_obs:.6f}")
    print(f"  Max |residual|: {np.max(np.abs(residuals)):.6f}")

    # Check if residuals are within expected noise range
    outliers = np.abs(residuals) > 3 * exp_config_sigma_obs
    if np.any(outliers):
        print(f"  [WARNING] {np.sum(outliers)} outliers (|residual| > 3*sigma)")
        print(f"  Outlier indices: {np.where(outliers)[0]}")
        print(f"  Outlier residuals: {residuals[outliers]}")
    else:
        print("  [OK] All residuals within 3*sigma")

# Check if t_arr matches saved t_M1
if np.allclose(t_arr, t_M1):
    print("\n[OK] t_arr matches saved t_M1")
else:
    print("\n[ERROR] t_arr does NOT match saved t_M1")
    print(f"  Max difference: {np.max(np.abs(t_arr - t_M1)):.10f}")

# Check if indices are correct
idx_regen = select_sparse_data_indices(len(t_arr), len(idx_M1), t_arr=t_arr)
if np.array_equal(idx_regen, idx_M1):
    print("[OK] Regenerated indices match saved idx_M1")
else:
    print("[WARNING] Regenerated indices do NOT match saved idx_M1")
    print(f"  First 5 saved: {idx_M1[:5]}")
    print(f"  First 5 regen: {idx_regen[:5]}")
