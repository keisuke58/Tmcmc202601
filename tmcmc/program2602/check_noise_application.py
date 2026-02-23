"""
ノイズ適用方法の詳細確認スクリプト

sigma_obs = 0.01 がどのように適用されているかを詳細に表示します。
"""
import numpy as np
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig
from core.tmcmc import _stable_hash_int

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42
name = "M1"

print("=" * 80)
print("Noise Application Details (sigma_obs = 0.01)")
print("=" * 80)

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
idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data, t_arr=t_arr)
phibar = compute_phibar(x0, config["active_species"])

# Setup random number generator (same as in main code)
rng = np.random.default_rng(exp_config_random_seed + (_stable_hash_int(name) % 1000))

print(f"\nConfiguration:")
print(f"  sigma_obs = {exp_config_sigma_obs}")
print(f"  n_data = {exp_config_n_data}")
print(f"  random_seed = {exp_config_random_seed}")
print(f"  active_species = {config['active_species']}")

print(f"\n{'='*80}")
print("Noise Generation Process Details")
print(f"{'='*80}")

# Generate noise for first species
sp = config["active_species"][0]
print(f"\nNoise generation for species {sp}:")

# Step 1: Generate standard normal random numbers
standard_normal = rng.standard_normal(exp_config_n_data)
print(f"\n1. Random numbers from standard normal N(0,1) (first 5):")
print(f"   {standard_normal[:5]}")
print(f"   Mean: {np.mean(standard_normal):.6f} (theoretical: 0.0)")
print(f"   Std: {np.std(standard_normal):.6f} (theoretical: 1.0)")

# Step 2: Scale by sigma_obs
noise = standard_normal * exp_config_sigma_obs
print(f"\n2. Scaling by sigma_obs = {exp_config_sigma_obs}:")
print(f"   Noise values (first 5): {noise[:5]}")
print(f"   Mean: {np.mean(noise):.6f} (theoretical: 0.0)")
print(f"   Std: {np.std(noise):.6f} (theoretical: {exp_config_sigma_obs})")
print(f"   Variance: {np.var(noise):.8f} (theoretical: {exp_config_sigma_obs**2})")

# Step 3: Add to phibar
phibar_values = phibar[idx_sparse, 0]
data = phibar_values + noise

print(f"\n3. Adding noise to phibar:")
print(f"   phibar values (first 5): {phibar_values[:5]}")
print(f"   Noise values (first 5): {noise[:5]}")
print(f"   Data values (first 5): {data[:5]}")
print(f"   Residuals (data - phibar): {data[:5] - phibar_values[:5]}")

print(f"\n{'='*80}")
print("Statistical Verification")
print(f"{'='*80}")

# Calculate residuals for all species
all_residuals = []
for i, sp in enumerate(config["active_species"]):
    noise_sp = rng.standard_normal(exp_config_n_data) * exp_config_sigma_obs
    data_sp = phibar[idx_sparse, i] + noise_sp
    residuals_sp = data_sp - phibar[idx_sparse, i]
    all_residuals.append(residuals_sp)
    
    print(f"\nSpecies {sp}:")
    print(f"  Residual mean: {np.mean(residuals_sp):.8f} (theoretical: 0.0)")
    print(f"  Residual std: {np.std(residuals_sp):.6f} (theoretical: {exp_config_sigma_obs})")
    print(f"  Residual variance: {np.var(residuals_sp):.8f} (theoretical: {exp_config_sigma_obs**2})")
    print(f"  Residual range: [{np.min(residuals_sp):.6f}, {np.max(residuals_sp):.6f}]")
    print(f"  Theoretical range (±3σ): [{-3*exp_config_sigma_obs:.6f}, {3*exp_config_sigma_obs:.6f}]")

all_residuals = np.array(all_residuals)
print(f"\nStatistics for all species:")
print(f"  Residual mean: {np.mean(all_residuals):.8f}")
print(f"  Residual std: {np.std(all_residuals):.6f}")
print(f"  Residual variance: {np.var(all_residuals):.8f}")

print(f"\n{'='*80}")
print("Conclusion")
print(f"{'='*80}")
print(f"""
Noise application method:
  data = phibar + N(0, sigma_obs²)
       = phibar + N(0, {exp_config_sigma_obs}²)
       = phibar + N(0, {exp_config_sigma_obs**2})

Each data point has:
  - Mean: phibar (true value)
  - Standard deviation: {exp_config_sigma_obs}
  - Variance: {exp_config_sigma_obs**2}

Important points:
  1. Individual data points always deviate from true values (this is normal)
  2. Noise mean is close to 0 (cancels out over many data points)
  3. Noise standard deviation should be approximately {exp_config_sigma_obs}
  4. ~68% of data points are within ±{exp_config_sigma_obs} of true value
  5. ~95% of data points are within ±{2*exp_config_sigma_obs} of true value

The fact that there is always some offset/error is NORMAL behavior. 
This is the nature of random noise.
""")
