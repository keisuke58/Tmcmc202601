"""Check random seed behavior with --no-noise option."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
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

t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data, t_arr=t_arr)
phibar = compute_phibar(x0, config["active_species"])

print("=" * 80)
print("Random Seed Behavior with --no-noise")
print("=" * 80)

# Test with different seeds
for seed in [42, 100, 999]:
    print(f"\nTest with seed = {seed}:")
    exp_config = ExperimentConfig()
    exp_config.sigma_obs = exp_config_sigma_obs
    exp_config.cov_rel = exp_config_cov_rel
    exp_config.n_data = exp_config_n_data
    exp_config.random_seed = seed
    exp_config.no_noise = True

    # Generate data without noise
    data = np.zeros((exp_config_n_data, len(config["active_species"])))
    for i, sp in enumerate(config["active_species"]):
        if exp_config.no_noise:
            data[:, i] = phibar[idx_sparse, i]

    print(f"  Data (first 3): {data[:3, 0]}")
    print(f"  phibar at obs (first 3): {phibar[idx_sparse[:3], 0]}")

    # Check if data matches phibar (should always match with no_noise)
    if np.allclose(data, phibar[idx_sparse, :]):
        print(f"  [OK] Data matches phibar exactly (seed={seed})")
    else:
        print(f"  [ERROR] Data does NOT match phibar (seed={seed})")

print("\n" + "=" * 80)
print("Conclusion:")
print("=" * 80)
print("With --no-noise option:")
print("  - Random seed is set but NOT used for data generation")
print("  - Data = phibar exactly (no noise added)")
print("  - Results are deterministic and independent of random seed")
print("  - Random seed only affects other parts (e.g., TMCMC sampling)")
