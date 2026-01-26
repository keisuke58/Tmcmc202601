"""Create debug plot showing data points vs model line with verification."""
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

# Create debug plot with model values at observation points
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for i, sp in enumerate(config["active_species"]):
    ax = axes[i]
    
    # Plot model line
    ax.plot(t_normalized, phibar[:, i], label=f"φ̄{sp+1} (model)", linewidth=2, color=f'C{i}')
    
    # Plot data points
    ax.scatter(
        t_obs_normalized, data_M1[:, i], s=60, edgecolor="k", linewidth=1.5,
        label=f"Data φ̄{sp+1}", alpha=0.8, zorder=10, color=f'C{i}', marker='o'
    )
    
    # Plot model values at observation points (for verification)
    model_at_obs = phibar[idx_M1, i]
    ax.scatter(
        t_obs_normalized, model_at_obs, s=40, edgecolor="red", linewidth=1,
        label=f"Model at obs φ̄{sp+1}", alpha=0.6, zorder=11, color='red', marker='x'
    )
    
    # Draw lines connecting data points to model values
    for j in range(len(t_obs_normalized)):
        ax.plot(
            [t_obs_normalized[j], t_obs_normalized[j]],
            [data_M1[j, i], model_at_obs[j]],
            'r--', alpha=0.3, linewidth=1
        )
    
    ax.set_xlabel("Normalized Time [0.0, 1.0]", fontsize=12)
    ax.set_ylabel("φ̄ = φ * ψ", fontsize=12)
    ax.set_title(f"Species {sp+1}: Data vs Model (with residuals)", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add residual text
    residuals = data_M1[:, i] - model_at_obs
    ax.text(0.02, 0.98, f"Residual std: {np.std(residuals):.6f}\nMean residual: {np.mean(residuals):.6f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

output_path = run_dir / "figures" / "TSM_simulation_M1_with_data_debug.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Debug plot saved to: {output_path}")
print("\nThis plot shows:")
print("  - Blue/Orange line: Model curve")
print("  - Blue/Orange circles: Data points")
print("  - Red X markers: Model values at observation points")
print("  - Red dashed lines: Residuals (data - model)")

# Print detailed comparison
print("\n" + "=" * 80)
print("Detailed Comparison:")
print("=" * 80)
for i, sp in enumerate(config["active_species"]):
    model_at_obs = phibar[idx_M1, i]
    data_at_obs = data_M1[:, i]
    residuals = data_at_obs - model_at_obs
    
    print(f"\nSpecies {sp+1}:")
    print(f"  Observation indices: {idx_M1[:5]} ... {idx_M1[-5:]}")
    print(f"  Normalized times: {t_obs_normalized[:5]} ... {t_obs_normalized[-5:]}")
    print(f"  Model values (first 5): {model_at_obs[:5]}")
    print(f"  Data values (first 5): {data_at_obs[:5]}")
    print(f"  Residuals (first 5): {residuals[:5]}")
    print(f"  Residual std: {np.std(residuals):.6f}")
    print(f"  Mean residual: {np.mean(residuals):.6f}")
    print(f"  Max |residual|: {np.max(np.abs(residuals)):.6f}")
