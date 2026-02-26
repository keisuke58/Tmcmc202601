"""
テストB: 尤度の重みをspeciesで正規化

speciesごとに残差を正規化して、値の大きいspeciesに引っ張られる問題を消す
"""

import numpy as np
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig
from core.tmcmc import _stable_hash_int

print("=" * 80)
print("テストB: 尤度の重みをspeciesで正規化")
print("=" * 80)

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42

# Generate M3 data
print("\n1. M3データ生成")
print("-" * 80)

config_M3 = MODEL_CONFIGS["M3"]
theta_true = get_theta_true()

solver_kwargs_M3 = {
    k: v
    for k, v in config_M3.items()
    if k not in ["active_species", "active_indices", "param_names"]
}

solver_M3 = BiofilmNewtonSolver(
    **solver_kwargs_M3,
    active_species=config_M3["active_species"],
    use_numba=HAS_NUMBA,
)

tsm_M3 = BiofilmTSM_Analytical(
    solver_M3,
    active_theta_indices=config_M3["active_indices"],
    cov_rel=exp_config_cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

t_arr_M3, x0_M3, sig2_M3 = tsm_M3.solve_tsm(theta_true)
phibar_M3 = compute_phibar(x0_M3, config_M3["active_species"])

idx_sparse_M3 = select_sparse_data_indices(len(t_arr_M3), exp_config_n_data, t_arr=t_arr_M3)

rng_M3 = np.random.default_rng(exp_config_random_seed + (_stable_hash_int("M3") % 1000))
data_M3 = np.zeros((exp_config_n_data, len(config_M3["active_species"])))
for i, sp in enumerate(config_M3["active_species"]):
    data_M3[:, i] = (
        phibar_M3[idx_sparse_M3, i]
        + rng_M3.standard_normal(exp_config_n_data) * exp_config_sigma_obs
    )

print(f"  Data shape: {data_M3.shape}")
print(f"  Active species: {config_M3['active_species']}")

# Check species statistics
print("\n2. Species統計の確認")
print("-" * 80)

phibar_at_obs = phibar_M3[idx_sparse_M3, :]

for i, sp in enumerate(config_M3["active_species"]):
    data_sp = data_M3[:, i]
    phibar_sp = phibar_at_obs[:, i]
    residuals_sp = data_sp - phibar_sp

    print(f"\n  Species {sp}:")
    print(f"    Data range: [{data_sp.min():.6f}, {data_sp.max():.6f}]")
    print(f"    Data mean: {np.mean(data_sp):.6f}, std: {np.std(data_sp):.6f}")
    print(f"    phibar range: [{phibar_sp.min():.6f}, {phibar_sp.max():.6f}]")
    print(f"    phibar mean: {np.mean(phibar_sp):.6f}, std: {np.std(phibar_sp):.6f}")
    print(f"    Residual mean: {np.mean(residuals_sp):.8f}, std: {np.std(residuals_sp):.6f}")
    print(f"    Residual RMSE: {np.sqrt(np.mean(residuals_sp**2)):.8f}")

# Standard likelihood (current implementation)
print("\n3. 標準尤度（現在の実装）")
print("-" * 80)


def log_likelihood_standard(mu, data, sigma_obs):
    """Standard Gaussian likelihood"""
    residuals = data - mu
    var_total = sigma_obs**2
    logL = -0.5 * np.sum(residuals**2 / var_total) - 0.5 * len(residuals) * np.log(
        2 * np.pi * var_total
    )
    return logL


logL_standard = 0.0
for i, sp in enumerate(config_M3["active_species"]):
    mu_sp = phibar_at_obs[:, i]
    data_sp = data_M3[:, i]
    logL_sp = log_likelihood_standard(mu_sp, data_sp, exp_config_sigma_obs)
    logL_standard += logL_sp
    print(f"  Species {sp} logL: {logL_sp:.4f}")

print(f"  Total logL (standard): {logL_standard:.4f}")

# Normalized likelihood (z-score normalization)
print("\n4. 正規化尤度（z-score正規化）")
print("-" * 80)


def log_likelihood_normalized(mu, data, sigma_obs, normalize_by_std=False):
    """Normalized Gaussian likelihood"""
    residuals = data - mu

    if normalize_by_std:
        # Normalize by data standard deviation
        data_std = np.std(data)
        if data_std > 0:
            residuals_norm = residuals / data_std
            var_total_norm = (sigma_obs / data_std) ** 2
        else:
            residuals_norm = residuals
            var_total_norm = sigma_obs**2
    else:
        # Normalize by sigma_obs (already done, but for consistency)
        residuals_norm = residuals / sigma_obs
        var_total_norm = 1.0

    logL = -0.5 * np.sum(residuals_norm**2 / var_total_norm) - 0.5 * len(residuals_norm) * np.log(
        2 * np.pi * var_total_norm
    )
    return logL


logL_normalized = 0.0
for i, sp in enumerate(config_M3["active_species"]):
    mu_sp = phibar_at_obs[:, i]
    data_sp = data_M3[:, i]
    logL_sp = log_likelihood_normalized(mu_sp, data_sp, exp_config_sigma_obs, normalize_by_std=True)
    logL_normalized += logL_sp
    print(f"  Species {sp} logL (normalized): {logL_sp:.4f}")

print(f"  Total logL (normalized): {logL_normalized:.4f}")

# Species-specific sigma_obs
print("\n5. Species別sigma_obs")
print("-" * 80)

# Calculate species-specific sigma_obs based on data scale
sigma_obs_per_species = []
for i, sp in enumerate(config_M3["active_species"]):
    data_sp = data_M3[:, i]
    # Use relative scale: sigma_obs scaled by data std
    data_std = np.std(data_sp)
    sigma_obs_sp = (
        exp_config_sigma_obs * (data_std / np.mean(np.abs(data_sp)))
        if np.mean(np.abs(data_sp)) > 0
        else exp_config_sigma_obs
    )
    sigma_obs_per_species.append(sigma_obs_sp)
    print(f"  Species {sp} sigma_obs: {sigma_obs_sp:.8f} (data std: {data_std:.6f})")

logL_species_specific = 0.0
for i, sp in enumerate(config_M3["active_species"]):
    mu_sp = phibar_at_obs[:, i]
    data_sp = data_M3[:, i]
    logL_sp = log_likelihood_standard(mu_sp, data_sp, sigma_obs_per_species[i])
    logL_species_specific += logL_sp
    print(f"  Species {sp} logL (species-specific sigma): {logL_sp:.4f}")

print(f"  Total logL (species-specific sigma): {logL_species_specific:.4f}")

# Check which species dominates
print("\n6. Species支配度の確認")
print("-" * 80)

contributions_standard = []
contributions_normalized = []

for i, sp in enumerate(config_M3["active_species"]):
    mu_sp = phibar_at_obs[:, i]
    data_sp = data_M3[:, i]

    logL_sp_std = log_likelihood_standard(mu_sp, data_sp, exp_config_sigma_obs)
    logL_sp_norm = log_likelihood_normalized(
        mu_sp, data_sp, exp_config_sigma_obs, normalize_by_std=True
    )

    contributions_standard.append(logL_sp_std)
    contributions_normalized.append(logL_sp_norm)

    print(f"  Species {sp}:")
    print(f"    Standard contribution: {logL_sp_std:.4f} ({100*logL_sp_std/logL_standard:.1f}%)")
    print(
        f"    Normalized contribution: {logL_sp_norm:.4f} ({100*logL_sp_norm/logL_normalized:.1f}%)"
    )

# Check if one species dominates
max_contrib_std = max(contributions_standard)
min_contrib_std = min(contributions_standard)
ratio_std = abs(max_contrib_std / min_contrib_std) if min_contrib_std != 0 else float("inf")

max_contrib_norm = max(contributions_normalized)
min_contrib_norm = min(contributions_normalized)
ratio_norm = abs(max_contrib_norm / min_contrib_norm) if min_contrib_norm != 0 else float("inf")

print(f"\n  Standard likelihood ratio (max/min): {ratio_std:.4f}")
print(f"  Normalized likelihood ratio (max/min): {ratio_norm:.4f}")

if ratio_std > 2.0:
    print("  [WARNING] 標準尤度でspecies間の差が大きい（一部speciesが支配）")
if ratio_norm < ratio_std * 0.5:
    print("  [OK] 正規化によりspecies間の差が縮小")

print(f"\n{'='*80}")
print("テストB完了")
print(f"{'='*80}")
print(
    """
推奨される対策:
1. speciesごとに残差を /(sigma_obs) で正規化（既に実装済みならOK）
2. speciesごとに /(std(data_species)) で正規化（スケール差を吸収）
3. species別 sigma_obs を導入（φ4だけ合わない等を吸収）
"""
)
