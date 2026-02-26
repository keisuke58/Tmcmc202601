"""
M3モデルの詳細残差分析

実際の実行結果を読み込んで、どのspeciesがどのように外れているかを分析
"""

import numpy as np
import sys
import io
from pathlib import Path
import matplotlib.pyplot as plt

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
print("M3モデルの詳細残差分析")
print("=" * 80)

# Load actual run data
run_dir = Path("_runs/20260122_162052_debug_seed42")
print(f"\n実行結果ディレクトリ: {run_dir}")

try:
    data_M3 = np.load(run_dir / "data_M3.npy")
    idx_M3 = np.load(run_dir / "idx_M3.npy")
    t_M3 = np.load(run_dir / "t_M3.npy")
    print("  [OK] データ読み込み成功")
except Exception as e:
    print(f"  [ERROR] データ読み込み失敗: {e}")
    sys.exit(1)

# Re-generate M3 model at theta_true
print(f"\n{'='*80}")
print("M3モデル再生成（theta_true使用）")
print(f"{'='*80}")

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
    cov_rel=0.005,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

t_arr_M3, x0_M3, sig2_M3 = tsm_M3.solve_tsm(theta_true)
phibar_M3 = compute_phibar(x0_M3, config_M3["active_species"])

print(f"  t_arr shape: {t_arr_M3.shape}, range: [{t_arr_M3.min():.8f}, {t_arr_M3.max():.8f}]")
print(f"  phibar shape: {phibar_M3.shape}")
print(f"  Active species: {config_M3['active_species']}")

# Verify indices match
if not np.array_equal(
    idx_M3, select_sparse_data_indices(len(t_arr_M3), len(data_M3), t_arr=t_arr_M3)
):
    print("  [WARNING] インデックスが一致しない可能性")
    idx_sparse = select_sparse_data_indices(len(t_arr_M3), len(data_M3), t_arr=t_arr_M3)
else:
    idx_sparse = idx_M3

# Calculate residuals
phibar_at_obs = phibar_M3[idx_sparse, :]
residuals = data_M3 - phibar_at_obs

print(f"\n{'='*80}")
print("残差分析（各species別）")
print(f"{'='*80}")

for i, sp in enumerate(config_M3["active_species"]):
    residuals_sp = residuals[:, i]
    data_sp = data_M3[:, i]
    phibar_sp = phibar_at_obs[:, i]

    print(f"\nSpecies {sp} (φ̄{sp+1}):")
    print(f"  データ範囲: [{data_sp.min():.6f}, {data_sp.max():.6f}]")
    print(f"  データ平均: {np.mean(data_sp):.6f}, std: {np.std(data_sp):.6f}")
    print(f"  モデル範囲: [{phibar_sp.min():.6f}, {phibar_sp.max():.6f}]")
    print(f"  モデル平均: {np.mean(phibar_sp):.6f}, std: {np.std(phibar_sp):.6f}")
    print(f"  残差平均: {np.mean(residuals_sp):.8f}")
    print(f"  残差std: {np.std(residuals_sp):.6f} (expected: {0.01:.6f})")
    print(f"  残差RMSE: {np.sqrt(np.mean(residuals_sp**2)):.8f}")
    print(f"  残差範囲: [{np.min(residuals_sp):.6f}, {np.max(residuals_sp):.6f}]")
    print(f"  最大絶対残差: {np.max(np.abs(residuals_sp)):.6f}")

    # Check if residuals are within expected noise
    sigma_obs = 0.01
    n_within_1sigma = np.sum(np.abs(residuals_sp) < sigma_obs)
    n_within_2sigma = np.sum(np.abs(residuals_sp) < 2 * sigma_obs)
    n_within_3sigma = np.sum(np.abs(residuals_sp) < 3 * sigma_obs)

    print(
        f"  1σ以内: {n_within_1sigma}/{len(residuals_sp)} ({100*n_within_1sigma/len(residuals_sp):.1f}%)"
    )
    print(
        f"  2σ以内: {n_within_2sigma}/{len(residuals_sp)} ({100*n_within_2sigma/len(residuals_sp):.1f}%)"
    )
    print(
        f"  3σ以内: {n_within_3sigma}/{len(residuals_sp)} ({100*n_within_3sigma/len(residuals_sp):.1f}%)"
    )

    # Check for systematic bias
    if abs(np.mean(residuals_sp)) > 2 * sigma_obs / np.sqrt(len(residuals_sp)):
        print("  [WARNING] 系統的なバイアスが検出されました（平均残差が大きい）")

    # Check for time-dependent bias
    t_normalized = (t_arr_M3 - t_arr_M3.min()) / (t_arr_M3.max() - t_arr_M3.min())
    t_obs_norm = t_normalized[idx_sparse]

    # Split into early and late
    n_half = len(residuals_sp) // 2
    early_residuals = residuals_sp[:n_half]
    late_residuals = residuals_sp[n_half:]

    early_mean = np.mean(np.abs(early_residuals))
    late_mean = np.mean(np.abs(late_residuals))

    print(f"  前半平均絶対残差: {early_mean:.8f}")
    print(f"  後半平均絶対残差: {late_mean:.8f}")
    print(f"  後半/前半比: {late_mean/early_mean:.4f}")

    if late_mean > 1.5 * early_mean:
        print("  [WARNING] 後半の残差が大きい（時間依存のバイアス）")

# Overall statistics
print(f"\n{'='*80}")
print("全体統計")
print(f"{'='*80}")

print("\n全speciesの残差:")
print(f"  平均: {np.mean(residuals):.8f}")
print(f"  std: {np.std(residuals):.6f}")
print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.8f}")
print(f"  最大絶対残差: {np.max(np.abs(residuals)):.6f}")

# Species comparison
print("\nSpecies間の比較:")
rmse_per_species = [
    np.sqrt(np.mean(residuals[:, i] ** 2)) for i in range(len(config_M3["active_species"]))
]
max_rmse = max(rmse_per_species)
min_rmse = min(rmse_per_species)
ratio = max_rmse / min_rmse if min_rmse > 0 else float("inf")

print(f"  RMSE範囲: [{min_rmse:.8f}, {max_rmse:.8f}]")
print(f"  RMSE比率 (max/min): {ratio:.4f}")

if ratio > 2.0:
    worst_species_idx = np.argmax(rmse_per_species)
    best_species_idx = np.argmin(rmse_per_species)
    print("  [WARNING] Species間の差が大きい")
    print(
        f"    最悪: Species {config_M3['active_species'][worst_species_idx]} (RMSE: {max_rmse:.8f})"
    )
    print(
        f"    最良: Species {config_M3['active_species'][best_species_idx]} (RMSE: {min_rmse:.8f})"
    )

# Create detailed visualization
print(f"\n{'='*80}")
print("可視化生成")
print(f"{'='*80}")

output_dir = run_dir / "figures" / "residual_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normalize time
t_normalized = (t_arr_M3 - t_arr_M3.min()) / (t_arr_M3.max() - t_arr_M3.min())
t_obs_norm = t_normalized[idx_sparse]

# Plot 1: Residuals vs time (all species)
ax = axes[0, 0]
for i, sp in enumerate(config_M3["active_species"]):
    ax.scatter(t_obs_norm, residuals[:, i], s=40, alpha=0.6, label=f"Species {sp}")
ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax.axhline(y=0.01, color="r", linestyle="--", alpha=0.3, label="±1σ")
ax.axhline(y=-0.01, color="r", linestyle="--", alpha=0.3)
ax.set_xlabel("Normalized Time [0.0, 1.0]")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs Time (All Species)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Residuals distribution (histogram)
ax = axes[0, 1]
for i, sp in enumerate(config_M3["active_species"]):
    ax.hist(residuals[:, i], bins=10, alpha=0.5, label=f"Species {sp}")
ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
ax.set_title("Residual Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: RMSE per species
ax = axes[1, 0]
species_labels = [f"Species {sp}" for sp in config_M3["active_species"]]
ax.bar(species_labels, rmse_per_species, alpha=0.7)
ax.set_ylabel("RMSE")
ax.set_title("RMSE per Species")
ax.grid(True, alpha=0.3, axis="y")
for i, rmse in enumerate(rmse_per_species):
    ax.text(i, rmse, f"{rmse:.6f}", ha="center", va="bottom", fontsize=9)

# Plot 4: Data vs Model (scatter)
ax = axes[1, 1]
for i, sp in enumerate(config_M3["active_species"]):
    ax.scatter(phibar_at_obs[:, i], data_M3[:, i], s=40, alpha=0.6, label=f"Species {sp}")
# Perfect fit line
all_phibar = phibar_at_obs.flatten()
all_data = data_M3.flatten()
min_val = min(all_phibar.min(), all_data.min())
max_val = max(all_phibar.max(), all_data.max())
ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="Perfect fit")
ax.set_xlabel("Model (phibar)")
ax.set_ylabel("Data")
ax.set_title("Data vs Model")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = output_dir / "m3_residual_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  可視化結果を保存: {output_path}")

# Save detailed statistics
stats = {
    "species": [int(sp) for sp in config_M3["active_species"]],
    "rmse_per_species": rmse_per_species,
    "mean_residuals": [
        float(np.mean(residuals[:, i])) for i in range(len(config_M3["active_species"]))
    ],
    "std_residuals": [
        float(np.std(residuals[:, i])) for i in range(len(config_M3["active_species"]))
    ],
    "max_abs_residuals": [
        float(np.max(np.abs(residuals[:, i]))) for i in range(len(config_M3["active_species"]))
    ],
    "overall_rmse": float(np.sqrt(np.mean(residuals**2))),
    "overall_mean": float(np.mean(residuals)),
    "overall_std": float(np.std(residuals)),
}

import json

with open(output_dir / "m3_residual_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"  統計情報を保存: {output_dir / 'm3_residual_stats.json'}")

print(f"\n{'='*80}")
print("分析完了")
print(f"{'='*80}")
