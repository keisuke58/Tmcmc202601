"""Verify that plot uses the exact same phibar as data generation."""
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices

# Load latest run
run_dir = Path("tmcmc/_runs/20260122_154237_debug_seed42")
data_M1 = np.load(run_dir / "data_M1.npy")
idx_M1 = np.load(run_dir / "idx_M1.npy")
t_M1 = np.load(run_dir / "t_M1.npy")

# Re-generate to get x0 and phibar
config = MODEL_CONFIGS["M1"]
theta_true = get_theta_true()
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_random_seed = 42
name = "M1"

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

# Compute phibar (same as data generation)
phibar = compute_phibar(x0, config["active_species"])

# Normalize time
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized = (t_arr - t_min) / (t_max - t_min)
t_obs_normalized = t_normalized[idx_M1]

# Create plot exactly as plot_TSM_simulation does
plt.figure(figsize=(10, 6))
for i, sp in enumerate(config["active_species"]):
    plt.plot(t_normalized, phibar[:, i], label=f"φ̄{sp+1} (model)", linewidth=2)

# Plot data points
for i, sp in enumerate(config["active_species"]):
    plt.scatter(
        t_obs_normalized, data_M1[:, i], s=40, edgecolor="k",
        label=f"Data φ̄{sp+1}", alpha=0.8, zorder=10,
    )

plt.xlabel("Normalized Time [0.0, 1.0]", fontsize=12)
plt.ylabel("φ̄ = φ * ψ", fontsize=12)
plt.title(f"TSM Simulation (φ̄) - M1 (Verified)", fontsize=14)
plt.xlim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

output_path = run_dir / "figures" / "TSM_simulation_M1_with_data_verified.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Verified plot saved to: {output_path}")

# Verify data points match model (within noise)
print("\nVerification:")
for i, sp in enumerate(config["active_species"]):
    phibar_at_obs = phibar[idx_M1, i]
    data_at_obs = data_M1[:, i]
    residuals = data_at_obs - phibar_at_obs
    
    print(f"Species {sp+1}:")
    print(f"  Residual std: {np.std(residuals):.6f}")
    print(f"  Mean residual: {np.mean(residuals):.6f}")
    print(f"  Max |residual|: {np.max(np.abs(residuals)):.6f}")
    
    # Check if model line passes through data points (within noise)
    if np.all(np.abs(residuals) < 3 * exp_config_sigma_obs):
        print(f"  [OK] All data points within 3*sigma of model line")
    else:
        outliers = np.abs(residuals) >= 3 * exp_config_sigma_obs
        print(f"  [WARNING] {np.sum(outliers)} outliers (|residual| >= 3*sigma)")
        print(f"  Outlier indices: {np.where(outliers)[0]}")
        print(f"  Outlier residuals: {residuals[outliers]}")
