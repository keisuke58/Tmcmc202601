"""Debug phibar mismatch between data generation and plotting."""
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
from visualization import PlotManager

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42
name = "M1"

# Setup
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

# Generate data (same as generate_synthetic_data)
t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data, t_arr=t_arr)
phibar_data_gen = compute_phibar(x0, config["active_species"])

print("=" * 80)
print("phibar Comparison: Data Generation vs Plotting")
print("=" * 80)
print(f"phibar_data_gen shape: {phibar_data_gen.shape}")
print(f"phibar_data_gen at first 5 observation indices:\n{phibar_data_gen[idx_sparse[:5], :]}")

# Generate data
from core.tmcmc import _stable_hash_int
rng = np.random.default_rng(exp_config_random_seed + (_stable_hash_int(name) % 1000))

data = np.zeros((exp_config_n_data, len(config["active_species"])))
for i, sp in enumerate(config["active_species"]):
    data[:, i] = phibar_data_gen[idx_sparse, i] + rng.standard_normal(exp_config_n_data) * exp_config_sigma_obs

print(f"\nData shape: {data.shape}")
print(f"Data at first 5 observations:\n{data[:5, :]}")

# Simulate what happens in plot_TSM_simulation
# Check if phibar is recomputed or used as-is
phibar_plot = phibar_data_gen.copy()  # This is what should happen when phibar is passed

print(f"\nphibar_plot shape: {phibar_plot.shape}")
print(f"phibar_plot at first 5 observation indices:\n{phibar_plot[idx_sparse[:5], :]}")

# Check if they are the same
if np.allclose(phibar_data_gen, phibar_plot):
    print("\n[OK] phibar_data_gen and phibar_plot are identical")
else:
    print("\n[ERROR] phibar_data_gen and phibar_plot are NOT identical")
    print(f"Max difference: {np.max(np.abs(phibar_data_gen - phibar_plot)):.10f}")

# Check normalized time calculation
t_min = t_arr.min()
t_max = t_arr.max()
t_normalized = (t_arr - t_min) / (t_max - t_min)
t_obs_normalized = t_normalized[idx_sparse]

print(f"\nNormalized time at observations (first 5): {t_obs_normalized[:5]}")
print(f"Expected (0.05 interval): {np.arange(0.05, 1.0 + 0.001, 0.05)[:5]}")

# Check if model values at observation points match data (within noise)
print("\n" + "=" * 80)
print("Model vs Data at Observation Points:")
print("=" * 80)

for i, sp in enumerate(config["active_species"]):
    model_at_obs = phibar_plot[idx_sparse, i]
    data_at_obs = data[:, i]
    residuals = data_at_obs - model_at_obs
    
    print(f"\nSpecies {sp+1}:")
    print(f"  Model at obs (first 5): {model_at_obs[:5]}")
    print(f"  Data at obs (first 5): {data_at_obs[:5]}")
    print(f"  Residuals (first 5): {residuals[:5]}")
    print(f"  Residual std: {np.std(residuals):.6f}")
    print(f"  Expected noise std: {exp_config_sigma_obs:.6f}")
    
    # Check if residuals are systematic or random
    if np.abs(np.mean(residuals)) > 0.001:
        print(f"  [WARNING] Systematic bias detected! Mean residual: {np.mean(residuals):.6f}")
    else:
        print(f"  [OK] No systematic bias (mean residual: {np.mean(residuals):.6f})")

# Test actual plotting
print("\n" + "=" * 80)
print("Testing actual plot generation...")
print("=" * 80)

plot_mgr = PlotManager("tmcmc/_runs/debug_phibar_test")
plot_mgr.plot_TSM_simulation(t_arr, x0, config["active_species"], name, data, idx_sparse, phibar=phibar_data_gen)

print("[OK] Plot generated successfully")
print(f"Plot saved to: {plot_mgr.output_dir / 'figures' / 'TSM_simulation_M1_with_data.png'}")
