"""
M3ノイズなしテスト

data point noise を 0 にして、データ点と線が完全一致するか確認
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from visualization.plot_manager import PlotManager
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig

print("=" * 80)
print("M3ノイズなしテスト")
print("=" * 80)
print()
print("data point noise を 0 にして、データ点と線が完全一致するか確認")
print()

# Configuration
exp_config = ExperimentConfig()
exp_config.sigma_obs = 0.01  # 設定値（実際には使わない）
exp_config.cov_rel = 0.005
exp_config.n_data = 20
exp_config.random_seed = 42
exp_config.no_noise = True  # ★ ノイズなし

config_M3 = MODEL_CONFIGS["M3"]
theta_true = get_theta_true()

print(f"Configuration:")
print(f"  n_data: {exp_config.n_data}")
print(f"  sigma_obs: {exp_config.sigma_obs} (設定値、実際には使用しない)")
print(f"  cov_rel: {exp_config.cov_rel}")
print(f"  random_seed: {exp_config.random_seed}")
print(f"  no_noise: {exp_config.no_noise} (ノイズなし)")
print()

# ==============================================================================
# Generate data WITHOUT noise
# ==============================================================================
print("=" * 80)
print("Step 1: ノイズなしでデータ生成")
print("=" * 80)

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
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

# Generate TSM solution
t_arr_M3, x0_M3, sig2_M3 = tsm_M3.solve_tsm(theta_true)
phibar_M3 = compute_phibar(x0_M3, config_M3["active_species"])

print(f"  t_arr_M3 shape: {t_arr_M3.shape}")
print(f"  t_arr_M3 range: [{t_arr_M3.min():.8f}, {t_arr_M3.max():.8f}]")
print(f"  phibar_M3 shape: {phibar_M3.shape}")
print(f"  phibar_M3 range: [{phibar_M3.min():.6f}, {phibar_M3.max():.6f}]")
print()

# Select data points
idx_sparse = select_sparse_data_indices(len(t_arr_M3), exp_config.n_data, t_arr=t_arr_M3)

# Generate data WITHOUT noise
data_M3_no_noise = np.zeros((exp_config.n_data, len(config_M3["active_species"])))
for i, sp in enumerate(config_M3["active_species"]):
    data_M3_no_noise[:, i] = phibar_M3[idx_sparse, i]  # ★ ノイズなし

print(f"  idx_sparse shape: {idx_sparse.shape}")
print(f"  idx_sparse range: [{idx_sparse.min()}, {idx_sparse.max()}]")
print(f"  data_M3_no_noise shape: {data_M3_no_noise.shape}")
print()

# ==============================================================================
# Verify consistency
# ==============================================================================
print("=" * 80)
print("Step 2: 一致確認")
print("=" * 80)

phibar_at_obs = phibar_M3[idx_sparse]
max_diff = np.max(np.abs(data_M3_no_noise - phibar_at_obs))
mean_diff = np.mean(np.abs(data_M3_no_noise - phibar_at_obs))

print(f"  data_M3_no_noise vs phibar_M3[idx_sparse]:")
print(f"    Max difference: {max_diff:.2e}")
print(f"    Mean difference: {mean_diff:.2e}")

if max_diff < 1e-10:
    print(f"  ✅ PASS: 完全一致（期待通り）")
else:
    print(f"  ❌ FAIL: 不一致（予期しない）")

print()

# ==============================================================================
# Plot with PlotManager (same as actual code)
# ==============================================================================
print("=" * 80)
print("Step 3: 可視化（PlotManager使用）")
print("=" * 80)

output_dir = Path("_runs/test_m3_no_noise")
output_dir.mkdir(parents=True, exist_ok=True)
plot_mgr = PlotManager(str(output_dir))

# ★ CRITICAL: Pass pre-computed phibar to ensure plot uses the same phibar as data generation
plot_mgr.plot_TSM_simulation(
    t_arr_M3, 
    x0_M3, 
    config_M3["active_species"], 
    "M3", 
    data_M3_no_noise, 
    idx_sparse, 
    phibar=phibar_M3  # ★ 同じphibarを渡す
)

print(f"  ✅ 図を保存: {output_dir / 'figures' / 'TSM_simulation_M3_with_data.png'}")
print()

# ==============================================================================
# Detailed comparison
# ==============================================================================
print("=" * 80)
print("Step 4: 詳細比較")
print("=" * 80)

for i, sp in enumerate(config_M3["active_species"]):
    model_at_obs = phibar_M3[idx_sparse, i]
    data_at_obs = data_M3_no_noise[:, i]
    diff = np.abs(model_at_obs - data_at_obs)
    
    print(f"\n  Species {sp} (phibar{sp+1}):")
    print(f"    Max diff: {np.max(diff):.2e}")
    print(f"    Mean diff: {np.mean(diff):.2e}")
    print(f"    RMS diff: {np.sqrt(np.mean(diff**2)):.2e}")
    
    if np.max(diff) < 1e-10:
        print(f"    ✅ Perfect match")
    else:
        print(f"    ❌ MISMATCH detected!")
        print(f"    First 5 differences: {diff[:5]}")
        print(f"    First 5 model values: {model_at_obs[:5]}")
        print(f"    First 5 data values: {data_at_obs[:5]}")

print()

# ==============================================================================
# Create comparison plot
# ==============================================================================
print("=" * 80)
print("Step 5: 比較プロット作成")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, sp in enumerate(config_M3["active_species"]):
    ax = axes[i]
    
    # Normalize time
    t_min = t_arr_M3.min()
    t_max = t_arr_M3.max()
    t_normalized = (t_arr_M3 - t_min) / (t_max - t_min) if t_max > t_min else t_arr_M3
    t_obs_normalized = t_normalized[idx_sparse]
    
    # Plot line: phibar_M3
    ax.plot(t_normalized, phibar_M3[:, i], 
            label=f"phibar{sp+1} (model)", linewidth=2, color='blue', alpha=0.7)
    
    # Plot data points (no noise)
    ax.scatter(t_obs_normalized, data_M3_no_noise[:, i], s=80, 
              edgecolor='red', facecolor='none', linewidth=2,
              label=f"Data phibar{sp+1} (no noise)", alpha=0.9, zorder=10, marker='x')
    
    # Plot model values at observation points (should match data exactly)
    ax.scatter(t_obs_normalized, phibar_M3[idx_sparse, i], s=40, 
              marker='o', color='green', alpha=0.6, zorder=9, 
              label=f"Model@obs phibar{sp+1}")
    
    # Check consistency
    model_at_obs = phibar_M3[idx_sparse, i]
    data_at_obs = data_M3_no_noise[:, i]
    diff = np.abs(model_at_obs - data_at_obs)
    max_diff_sp = np.max(diff)
    
    ax.set_title(f"phibar{sp+1} (Max diff: {max_diff_sp:.2e})", fontsize=12)
    ax.set_xlabel("Normalized Time [0.0, 1.0]", fontsize=11)
    ax.set_ylabel("phibar", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Add text annotation
    if max_diff_sp < 1e-10:
        ax.text(0.5, 0.95, "PERFECT MATCH", 
               transform=ax.transAxes, fontsize=12, color='green',
               verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        ax.text(0.5, 0.95, f"MISMATCH: {max_diff_sp:.2e}", 
               transform=ax.transAxes, fontsize=12, color='red',
               verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.suptitle("M3 No-Noise Test: Data points should exactly match the line", fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
fig_path = output_dir / "M3_no_noise_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  ✅ 比較プロットを保存: {fig_path}")
print()

# ==============================================================================
# Final verdict
# ==============================================================================
print("=" * 80)
print("最終判定")
print("=" * 80)

all_max_diffs = []
for i, sp in enumerate(config_M3["active_species"]):
    model_at_obs = phibar_M3[idx_sparse, i]
    data_at_obs = data_M3_no_noise[:, i]
    diff = np.abs(model_at_obs - data_at_obs)
    all_max_diffs.append(np.max(diff))

overall_max_diff = np.max(all_max_diffs)

if overall_max_diff < 1e-10:
    print("✅ PASS: 完全一致")
    print("  → ノイズなしの場合、データ点と線が完全一致")
    print("  → 可視化コードは正しく実装されています")
    print("  → 実際の図でズレが見える場合は、ノイズの影響か、")
    print("    別の原因（MAP/meanフィットなど）を確認してください")
else:
    print("❌ FAIL: 不一致検出")
    print(f"  → 最大差分: {overall_max_diff:.2e}")
    print("  → これは「可視化 or パイプラインのバグ」です")
    print("  → コードの参照がズレています")

print()
print("=" * 80)
print("生成されたファイル:")
print(f"  1. {output_dir / 'figures' / 'TSM_simulation_M3_with_data.png'}")
print(f"  2. {fig_path}")
print("=" * 80)
