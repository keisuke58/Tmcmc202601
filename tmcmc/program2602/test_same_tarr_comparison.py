"""
テストA: 同じt_arr上で比較

データ生成時のt_arrを固定して、M3側も同じ時刻で出力
これでズレが減るなら時間軸が主因
"""
import numpy as np
import sys
import io
from pathlib import Path
import matplotlib.pyplot as plt

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

print("=" * 80)
print("テストA: 同じt_arr上で比較")
print("=" * 80)

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42

# Step 1: Generate M3 data with M3's own t_arr (original)
print("\n1. M3データ生成（M3のt_arr使用）")
print("-" * 80)

config_M3 = MODEL_CONFIGS["M3"]
theta_true = get_theta_true()

solver_kwargs_M3 = {
    k: v for k, v in config_M3.items()
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

t_arr_M3_orig, x0_M3_orig, sig2_M3_orig = tsm_M3.solve_tsm(theta_true)
phibar_M3_orig = compute_phibar(x0_M3_orig, config_M3["active_species"])

idx_sparse_M3_orig = select_sparse_data_indices(len(t_arr_M3_orig), exp_config_n_data, t_arr=t_arr_M3_orig)

rng_M3 = np.random.default_rng(exp_config_random_seed + (_stable_hash_int("M3") % 1000))
data_M3_orig = np.zeros((exp_config_n_data, len(config_M3["active_species"])))
for i, sp in enumerate(config_M3["active_species"]):
    data_M3_orig[:, i] = phibar_M3_orig[idx_sparse_M3_orig, i] + rng_M3.standard_normal(exp_config_n_data) * exp_config_sigma_obs

print(f"  M3 t_arr range: [{t_arr_M3_orig.min():.8f}, {t_arr_M3_orig.max():.8f}]")
print(f"  M3 t_arr length: {len(t_arr_M3_orig)}")
print(f"  Data shape: {data_M3_orig.shape}")

# Step 2: Generate M1 data with M1's own t_arr
print("\n2. M1データ生成（M1のt_arr使用）")
print("-" * 80)

config_M1 = MODEL_CONFIGS["M1"]
solver_kwargs_M1 = {
    k: v for k, v in config_M1.items()
    if k not in ["active_species", "active_indices", "param_names"]
}

solver_M1 = BiofilmNewtonSolver(
    **solver_kwargs_M1,
    active_species=config_M1["active_species"],
    use_numba=HAS_NUMBA,
)

tsm_M1 = BiofilmTSM_Analytical(
    solver_M1,
    active_theta_indices=config_M1["active_indices"],
    cov_rel=exp_config_cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

t_arr_M1, x0_M1, sig2_M1 = tsm_M1.solve_tsm(theta_true)
phibar_M1 = compute_phibar(x0_M1, config_M1["active_species"])

idx_sparse_M1 = select_sparse_data_indices(len(t_arr_M1), exp_config_n_data, t_arr=t_arr_M1)

rng_M1 = np.random.default_rng(exp_config_random_seed + (_stable_hash_int("M1") % 1000))
data_M1 = np.zeros((exp_config_n_data, len(config_M1["active_species"])))
for i, sp in enumerate(config_M1["active_species"]):
    data_M1[:, i] = phibar_M1[idx_sparse_M1, i] + rng_M1.standard_normal(exp_config_n_data) * exp_config_sigma_obs

print(f"  M1 t_arr range: [{t_arr_M1.min():.8f}, {t_arr_M1.max():.8f}]")
print(f"  M1 t_arr length: {len(t_arr_M1)}")

# Step 3: Generate M3 data using M1's t_arr (interpolation)
print("\n3. M3データ生成（M1のt_arr使用 - 補間）")
print("-" * 80)

# Use M1's t_arr for M3
t_arr_fixed = t_arr_M1.copy()

# Solve M3 on M1's time grid (need to interpolate or solve directly)
# For now, interpolate M3 solution to M1's time grid
from scipy.interpolate import interp1d

x0_M3_interp = np.zeros((len(t_arr_fixed), x0_M3_orig.shape[1]))
for j in range(x0_M3_orig.shape[1]):
    interp_func = interp1d(t_arr_M3_orig, x0_M3_orig[:, j], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    x0_M3_interp[:, j] = interp_func(t_arr_fixed)

phibar_M3_interp = compute_phibar(x0_M3_interp, config_M3["active_species"])

# Select indices on fixed t_arr
idx_sparse_fixed = select_sparse_data_indices(len(t_arr_fixed), exp_config_n_data, t_arr=t_arr_fixed)

# Generate data using interpolated phibar
data_M3_fixed = np.zeros((exp_config_n_data, len(config_M3["active_species"])))
for i, sp in enumerate(config_M3["active_species"]):
    data_M3_fixed[:, i] = phibar_M3_interp[idx_sparse_fixed, i] + rng_M3.standard_normal(exp_config_n_data) * exp_config_sigma_obs

print(f"  Fixed t_arr range: [{t_arr_fixed.min():.8f}, {t_arr_fixed.max():.8f}]")
print(f"  Fixed t_arr length: {len(t_arr_fixed)}")
print(f"  Data shape: {data_M3_fixed.shape}")

# Step 4: Compare residuals
print("\n4. 残差の比較")
print("-" * 80)

# Original M3 (M3's t_arr)
residuals_M3_orig = data_M3_orig - phibar_M3_orig[idx_sparse_M3_orig, :]
rmse_M3_orig = np.sqrt(np.mean(residuals_M3_orig**2))
print(f"  M3 (M3's t_arr) RMSE: {rmse_M3_orig:.8f}")
print(f"  M3 (M3's t_arr) mean abs residual: {np.mean(np.abs(residuals_M3_orig)):.8f}")

# Fixed M3 (M1's t_arr)
residuals_M3_fixed = data_M3_fixed - phibar_M3_interp[idx_sparse_fixed, :]
rmse_M3_fixed = np.sqrt(np.mean(residuals_M3_fixed**2))
print(f"  M3 (M1's t_arr) RMSE: {rmse_M3_fixed:.8f}")
print(f"  M3 (M1's t_arr) mean abs residual: {np.mean(np.abs(residuals_M3_fixed)):.8f}")

# M1 for reference
residuals_M1 = data_M1 - phibar_M1[idx_sparse_M1, :]
rmse_M1 = np.sqrt(np.mean(residuals_M1**2))
print(f"  M1 (M1's t_arr) RMSE: {rmse_M1:.8f}")
print(f"  M1 (M1's t_arr) mean abs residual: {np.mean(np.abs(residuals_M1)):.8f}")

# Step 5: Check if fixed t_arr reduces error
print("\n5. 判定")
print("-" * 80)

if rmse_M3_fixed < rmse_M3_orig * 0.9:
    print(f"  [YES] 同じt_arrでズレが減った → 時間軸が主因の可能性が高い")
    print(f"  RMSE改善: {rmse_M3_orig:.8f} → {rmse_M3_fixed:.8f} ({100*(1-rmse_M3_fixed/rmse_M3_orig):.1f}%改善)")
else:
    print(f"  [NO] 同じt_arrでもズレは変わらない → 時間軸以外が主因")
    print(f"  RMSE変化: {rmse_M3_orig:.8f} → {rmse_M3_fixed:.8f}")

# Step 6: Visualize comparison
print("\n6. 可視化")
print("-" * 80)

output_dir = Path("_runs/test_same_tarr")
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normalize time for plotting
t_norm_M3_orig = (t_arr_M3_orig - t_arr_M3_orig.min()) / (t_arr_M3_orig.max() - t_arr_M3_orig.min())
t_norm_fixed = (t_arr_fixed - t_arr_fixed.min()) / (t_arr_fixed.max() - t_arr_fixed.min())

# Plot 1: M3 original (M3's t_arr)
ax = axes[0, 0]
for i, sp in enumerate(config_M3["active_species"]):
    ax.plot(t_norm_M3_orig, phibar_M3_orig[:, i], label=f"φ̄{sp+1} (model)", linewidth=2, alpha=0.7)
    t_obs_norm = t_norm_M3_orig[idx_sparse_M3_orig]
    ax.scatter(t_obs_norm, data_M3_orig[:, i], s=40, edgecolor="k", 
              label=f"Data φ̄{sp+1}", alpha=0.8, zorder=10)
ax.set_xlabel("Normalized Time [0.0, 1.0]")
ax.set_ylabel("φ̄ = φ * ψ")
ax.set_title("M3 (M3's t_arr)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: M3 fixed (M1's t_arr)
ax = axes[0, 1]
for i, sp in enumerate(config_M3["active_species"]):
    ax.plot(t_norm_fixed, phibar_M3_interp[:, i], label=f"φ̄{sp+1} (model)", linewidth=2, alpha=0.7)
    t_obs_norm = t_norm_fixed[idx_sparse_fixed]
    ax.scatter(t_obs_norm, data_M3_fixed[:, i], s=40, edgecolor="k", 
              label=f"Data φ̄{sp+1}", alpha=0.8, zorder=10)
ax.set_xlabel("Normalized Time [0.0, 1.0]")
ax.set_ylabel("φ̄ = φ * ψ")
ax.set_title("M3 (M1's t_arr - interpolated)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Residuals comparison
ax = axes[1, 0]
for i, sp in enumerate(config_M3["active_species"]):
    t_obs_norm_orig = t_norm_M3_orig[idx_sparse_M3_orig]
    ax.scatter(t_obs_norm_orig, residuals_M3_orig[:, i], s=40, alpha=0.6, 
              label=f"Residual φ̄{sp+1} (M3's t_arr)")
    t_obs_norm_fixed = t_norm_fixed[idx_sparse_fixed]
    ax.scatter(t_obs_norm_fixed, residuals_M3_fixed[:, i], s=40, marker='x', alpha=0.6, 
              label=f"Residual φ̄{sp+1} (M1's t_arr)")
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel("Normalized Time [0.0, 1.0]")
ax.set_ylabel("Residual")
ax.set_title("Residuals Comparison")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: RMSE comparison
ax = axes[1, 1]
models = ["M3 (M3's t_arr)", "M3 (M1's t_arr)", "M1 (M1's t_arr)"]
rmses = [rmse_M3_orig, rmse_M3_fixed, rmse_M1]
colors = ['red', 'blue', 'green']
ax.bar(models, rmses, color=colors, alpha=0.7)
ax.set_ylabel("RMSE")
ax.set_title("RMSE Comparison")
ax.grid(True, alpha=0.3, axis='y')
for i, (model, rmse) in enumerate(zip(models, rmses)):
    ax.text(i, rmse, f'{rmse:.6f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output_path = output_dir / "same_tarr_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  可視化結果を保存: {output_path}")

print(f"\n{'='*80}")
print("テストA完了")
print(f"{'='*80}")
