"""Test that plot uses the same phibar as data generation."""
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
from main.case2_main import select_sparse_data_indices

# Configuration
exp_config_sigma_obs = 0.001
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

# Generate data
t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data)
phibar = compute_phibar(x0, config["active_species"])

# Generate data with same random seed
from core.tmcmc import _stable_hash_int
rng = np.random.default_rng(exp_config_random_seed + (_stable_hash_int(name) % 1000))

data = np.zeros((exp_config_n_data, len(config["active_species"])))
for i, sp in enumerate(config["active_species"]):
    data[:, i] = phibar[idx_sparse, i] + rng.standard_normal(exp_config_n_data) * exp_config_sigma_obs

# Test plot with phibar parameter
plot_mgr = PlotManager("tmcmc/_runs/test_plot_fix")
plot_mgr.plot_TSM_simulation(t_arr, x0, config["active_species"], name, data, idx_sparse, phibar=phibar)

print("Test completed successfully!")
print(f"Plot saved to: {plot_mgr.output_dir / 'figures' / 'TSM_simulation_M1_with_data.png'}")

# Verify that data points match model
for i, sp in enumerate(config["active_species"]):
    phibar_at_obs = phibar[idx_sparse, i]
    data_at_obs = data[:, i]
    residuals = data_at_obs - phibar_at_obs
    print(f"Species {sp+1}: residual std = {np.std(residuals):.6f}, max |residual| = {np.max(np.abs(residuals)):.6f}")
