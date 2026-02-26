"""Test data generation with and without noise."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from visualization import PlotManager
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig

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

# Generate data
t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data, t_arr=t_arr)
phibar = compute_phibar(x0, config["active_species"])

print("=" * 80)
print("Test: Data Generation With and Without Noise")
print("=" * 80)

# Test 1: With noise (default)
print("\nTest 1: With noise (sigma_obs = 0.01)")
exp_config = ExperimentConfig()
exp_config.sigma_obs = exp_config_sigma_obs
exp_config.cov_rel = exp_config_cov_rel
exp_config.n_data = exp_config_n_data
exp_config.random_seed = exp_config_random_seed
exp_config.no_noise = False

from core.tmcmc import _stable_hash_int

rng = np.random.default_rng(exp_config.random_seed + (_stable_hash_int(name) % 1000))

data_with_noise = np.zeros((exp_config_n_data, len(config["active_species"])))
for i, sp in enumerate(config["active_species"]):
    data_with_noise[:, i] = (
        phibar[idx_sparse, i] + rng.standard_normal(exp_config_n_data) * exp_config.sigma_obs
    )

print(f"Data with noise (first 5):\n{data_with_noise[:5, :]}")
print(f"phibar at obs (first 5):\n{phibar[idx_sparse[:5], :]}")
residuals_with_noise = data_with_noise - phibar[idx_sparse, :]
print(f"Residuals (first 5):\n{residuals_with_noise[:5, :]}")
print(f"Residual std: {np.std(residuals_with_noise):.6f}")

# Test 2: Without noise
print("\nTest 2: Without noise (training data mode)")
exp_config.no_noise = True

data_no_noise = np.zeros((exp_config_n_data, len(config["active_species"])))
for i, sp in enumerate(config["active_species"]):
    if exp_config.no_noise:
        data_no_noise[:, i] = phibar[idx_sparse, i]
    else:
        data_no_noise[:, i] = (
            phibar[idx_sparse, i] + rng.standard_normal(exp_config_n_data) * exp_config.sigma_obs
        )

print(f"Data without noise (first 5):\n{data_no_noise[:5, :]}")
print(f"phibar at obs (first 5):\n{phibar[idx_sparse[:5], :]}")
residuals_no_noise = data_no_noise - phibar[idx_sparse, :]
print(f"Residuals (first 5):\n{residuals_no_noise[:5, :]}")
print(f"Residual std: {np.std(residuals_no_noise):.6f}")

# Verify no_noise data matches phibar exactly
if np.allclose(data_no_noise, phibar[idx_sparse, :]):
    print("\n[OK] Data without noise matches phibar exactly!")
else:
    print("\n[ERROR] Data without noise does NOT match phibar!")
    print(f"Max difference: {np.max(np.abs(data_no_noise - phibar[idx_sparse, :])):.10f}")

# Create comparison plots
plot_mgr = PlotManager("tmcmc/_runs/test_no_noise")
plot_mgr.plot_TSM_simulation(
    t_arr, x0, config["active_species"], "M1_with_noise", data_with_noise, idx_sparse, phibar=phibar
)
plot_mgr.plot_TSM_simulation(
    t_arr, x0, config["active_species"], "M1_no_noise", data_no_noise, idx_sparse, phibar=phibar
)

print(f"\nPlots saved to: {plot_mgr.output_dir / 'figures'}")
