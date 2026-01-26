"""Verify data generation and plot consistency."""
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS, PRIOR_BOUNDS_DEFAULT
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices

# Configuration
run_dir = Path("tmcmc/_runs/20260122_151859_debug_seed42")
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42
name = "M1"

# Load saved data
data_M1 = np.load(run_dir / "data_M1.npy")
idx_M1 = np.load(run_dir / "idx_M1.npy")
t_M1 = np.load(run_dir / "t_M1.npy")

print("=" * 80)
print("Data Verification for M1")
print("=" * 80)
print(f"Loaded data shape: {data_M1.shape}")
print(f"Loaded index shape: {idx_M1.shape}")
print(f"Loaded time shape: {t_M1.shape}")
print(f"First 5 indices: {idx_M1[:5]}")
print(f"First 5 data points:\n{data_M1[:5]}")

# Re-generate data to verify
print("\n" + "=" * 80)
print("Re-generating data to verify...")
print("=" * 80)

config = MODEL_CONFIGS["M1"]
theta_true = get_theta_true()

solver_kwargs = {
    k: v for k, v in config.items()
    if k not in ["active_species", "active_indices", "param_names"]
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

print(f"Generated t_arr shape: {t_arr.shape}")
print(f"Generated x0 shape: {x0.shape}")
print(f"Generated sig2 shape: {sig2.shape}")
print(f"t_arr range: [{t_arr.min():.6f}, {t_arr.max():.6f}]")

# Check if t_arr matches saved t_M1
if np.allclose(t_arr, t_M1):
    print("[OK] t_arr matches saved t_M1")
else:
    print("[ERROR] t_arr does NOT match saved t_M1")
    print(f"  Max difference: {np.max(np.abs(t_arr - t_M1)):.10f}")

# Select sparse indices
idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data)
print(f"\nSelected indices shape: {idx_sparse.shape}")
print(f"First 5 selected indices: {idx_sparse[:5]}")

# Check if indices match saved idx_M1
if np.array_equal(idx_sparse, idx_M1):
    print("[OK] Selected indices match saved idx_M1")
else:
    print("[ERROR] Selected indices do NOT match saved idx_M1")
    print(f"  First 5 saved: {idx_M1[:5]}")
    print(f"  First 5 selected: {idx_sparse[:5]}")

# Compute phibar
phibar = compute_phibar(x0, config["active_species"])
print(f"\nphibar shape: {phibar.shape}")
print(f"phibar at first 5 observation indices:\n{phibar[idx_sparse[:5], :]}")

# Generate data with same random seed
from core.tmcmc import _stable_hash_int
rng = np.random.default_rng(exp_config_random_seed + (_stable_hash_int(name) % 1000))

data_regen = np.zeros((exp_config_n_data, len(config["active_species"])))
for i, sp in enumerate(config["active_species"]):
    data_regen[:, i] = phibar[idx_sparse, i] + rng.standard_normal(exp_config_n_data) * exp_config_sigma_obs

print(f"\nRegenerated data shape: {data_regen.shape}")
print(f"First 5 regenerated data points:\n{data_regen[:5]}")

# Check if data matches saved data
if np.allclose(data_regen, data_M1):
    print("[OK] Regenerated data matches saved data_M1")
else:
    print("[ERROR] Regenerated data does NOT match saved data_M1")
    print(f"  Max difference: {np.max(np.abs(data_regen - data_M1)):.10f}")
    print(f"  First 5 differences:\n{data_regen[:5] - data_M1[:5]}")

# Normalize time for plotting
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized = (t_arr - t_min) / (t_max - t_min)
t_obs_normalized = t_normalized[idx_sparse]

print(f"\nNormalized time range: [{t_normalized.min():.6f}, {t_normalized.max():.6f}]")
print(f"First 5 normalized observation times: {t_obs_normalized[:5]}")

# Create verification plot
plt.figure(figsize=(12, 8))

# Plot model
for i, sp in enumerate(config["active_species"]):
    plt.plot(t_normalized, phibar[:, i], label=f"φ̄{sp+1} (model)", linewidth=2, alpha=0.7)

# Plot data points
for i, sp in enumerate(config["active_species"]):
    plt.scatter(
        t_obs_normalized, data_M1[:, i], s=60, edgecolor="k", linewidth=1.5,
        label=f"Data φ̄{sp+1} (saved)", alpha=0.8, zorder=10, marker="o"
    )
    plt.scatter(
        t_obs_normalized, data_regen[:, i], s=40, edgecolor="r", linewidth=1,
        label=f"Data φ̄{sp+1} (regen)", alpha=0.6, zorder=11, marker="x"
    )

plt.xlabel("Normalized Time [0.0, 1.0]", fontsize=12)
plt.ylabel("φ̄ = φ * ψ", fontsize=12)
plt.title(f"Data Verification Plot - {name}", fontsize=14)
plt.xlim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.tight_layout()

output_path = run_dir / "figures" / "data_verification_M1.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[OK] Verification plot saved to: {output_path}")

# Check if data points are on the model line (within noise)
print("\n" + "=" * 80)
print("Checking if data points match model (within noise):")
print("=" * 80)
for i, sp in enumerate(config["active_species"]):
    phibar_at_obs = phibar[idx_sparse, i]
    data_at_obs = data_M1[:, i]
    residuals = data_at_obs - phibar_at_obs
    noise_std = exp_config_sigma_obs
    
    print(f"\nSpecies {sp+1}:")
    print(f"  phibar at observations: {phibar_at_obs[:5]}")
    print(f"  data at observations: {data_at_obs[:5]}")
    print(f"  residuals: {residuals[:5]}")
    print(f"  Expected noise std: {noise_std:.6f}")
    print(f"  Actual residual std: {np.std(residuals):.6f}")
    print(f"  Max |residual|: {np.max(np.abs(residuals)):.6f}")
    
    # Check if residuals are within reasonable bounds (3 sigma)
    outliers = np.abs(residuals) > 3 * noise_std
    if np.any(outliers):
        print(f"  [WARNING] {np.sum(outliers)} outliers (|residual| > 3*sigma)")
    else:
        print(f"  [OK] All residuals within 3*sigma")

print("\n" + "=" * 80)
print("Verification complete!")
print("=" * 80)
