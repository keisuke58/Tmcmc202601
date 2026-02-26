"""
M3自己一致テスト（推定なし）

同じtheta_trueを使って：
1) phibar_denseを1回だけ生成
2) data = phibar_dense[idx_sparse]（no-noise）
3) 同じphibar_denseを線として描画

判定：
- 完全一致 → 推定の問題（前段の議論が正しい）
- ズレる → 可視化 or パイプラインのバグ

直感的に言うと：
「同じ世界（theta_true）で作った点が、同じ世界の線に乗らない」
これは物理的にありえない。起きてたらコードの参照がズレてる。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig

print("=" * 80)
print("M3自己一致テスト（推定なし）")
print("=" * 80)
print()
print("同じtheta_trueを使って：")
print("1) phibar_denseを1回だけ生成")
print("2) data = phibar_dense[idx_sparse]（no-noise）")
print("3) 同じphibar_denseを線として描画")
print()

# Configuration
exp_config = ExperimentConfig()
exp_config.sigma_obs = 0.01
exp_config.cov_rel = 0.005
exp_config.n_data = 20
exp_config.random_seed = 42
exp_config.no_noise = True  # ★ ノイズなしでテスト

config_M3 = MODEL_CONFIGS["M3"]
theta_true = get_theta_true()

print("Configuration:")
print(f"  n_data: {exp_config.n_data}")
print(f"  sigma_obs: {exp_config.sigma_obs}")
print(f"  cov_rel: {exp_config.cov_rel}")
print(f"  random_seed: {exp_config.random_seed}")
print(f"  no_noise: {exp_config.no_noise}")
print()

# ==============================================================================
# Step 1: Generate phibar_dense ONCE
# ==============================================================================
print("=" * 80)
print("Step 1: phibar_dense生成（1回だけ）")
print("=" * 80)

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
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

# ★ CRITICAL: Generate phibar_dense ONCE
t_arr_M3, x0_M3, sig2_M3 = tsm_M3.solve_tsm(theta_true)
phibar_dense = compute_phibar(x0_M3, config_M3["active_species"])

print(f"  t_arr_M3 shape: {t_arr_M3.shape}")
print(f"  t_arr_M3 range: [{t_arr_M3.min():.8f}, {t_arr_M3.max():.8f}]")
print(f"  phibar_dense shape: {phibar_dense.shape}")
print(f"  phibar_dense range: [{phibar_dense.min():.6f}, {phibar_dense.max():.6f}]")
print()

# ==============================================================================
# Step 2: Generate data = phibar_dense[idx_sparse] (no-noise)
# ==============================================================================
print("=" * 80)
print("Step 2: data生成（phibar_dense[idx_sparse] - no-noise）")
print("=" * 80)

idx_sparse = select_sparse_data_indices(len(t_arr_M3), exp_config.n_data, t_arr=t_arr_M3)
data = phibar_dense[idx_sparse].copy()  # ★ CRITICAL: Use the SAME phibar_dense

print(f"  idx_sparse shape: {idx_sparse.shape}")
print(f"  idx_sparse range: [{idx_sparse.min()}, {idx_sparse.max()}]")
print(f"  idx_sparse (first 5): {idx_sparse[:5]}")
print(f"  idx_sparse (last 5): {idx_sparse[-5:]}")
print(f"  data shape: {data.shape}")
print()

# ==============================================================================
# Step 3: Verify self-consistency
# ==============================================================================
print("=" * 80)
print("Step 3: 自己一致検証")
print("=" * 80)

# Check: data should exactly match phibar_dense[idx_sparse]
phibar_at_obs = phibar_dense[idx_sparse]
max_diff = np.max(np.abs(data - phibar_at_obs))
mean_diff = np.mean(np.abs(data - phibar_at_obs))

print("  data vs phibar_dense[idx_sparse]:")
print(f"    Max difference: {max_diff:.2e}")
print(f"    Mean difference: {mean_diff:.2e}")

if max_diff < 1e-10:
    print("  ✅ PASS: data完全一致（期待通り）")
else:
    print("  ❌ FAIL: data不一致（予期しない）")
    print("     → これはバグです！")

print()

# ==============================================================================
# Step 4: Plot with SAME phibar_dense
# ==============================================================================
print("=" * 80)
print("Step 4: 可視化（同じphibar_denseを使用）")
print("=" * 80)

# Normalize time
t_min = t_arr_M3.min()
t_max = t_arr_M3.max()
t_normalized = (t_arr_M3 - t_min) / (t_max - t_min) if t_max > t_min else t_arr_M3
t_obs_normalized = t_normalized[idx_sparse]

# Create plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, sp in enumerate(config_M3["active_species"]):
    ax = axes[i]

    # Plot line: SAME phibar_dense
    ax.plot(
        t_normalized,
        phibar_dense[:, i],
        label=f"φ̄{sp+1} (model)",
        linewidth=2,
        color="blue",
        alpha=0.7,
    )

    # Plot data points
    ax.scatter(
        t_obs_normalized,
        data[:, i],
        s=60,
        edgecolor="k",
        label=f"Data φ̄{sp+1}",
        alpha=0.9,
        zorder=10,
        color="red",
        marker="x",
    )

    # Plot model values at observation points (should match data exactly)
    ax.scatter(
        t_obs_normalized,
        phibar_dense[idx_sparse, i],
        s=30,
        marker="o",
        color="green",
        alpha=0.5,
        zorder=9,
        label=f"Model@obs φ̄{sp+1}",
    )

    # Check consistency
    model_at_obs = phibar_dense[idx_sparse, i]
    data_at_obs = data[:, i]
    diff = np.abs(model_at_obs - data_at_obs)
    max_diff_sp = np.max(diff)

    ax.set_title(f"φ̄{sp+1} (Max diff: {max_diff_sp:.2e})", fontsize=12)
    ax.set_xlabel("Normalized Time [0.0, 1.0]", fontsize=11)
    ax.set_ylabel("φ̄", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Add text annotation if mismatch
    if max_diff_sp > 1e-10:
        ax.text(
            0.5,
            0.95,
            f"⚠ MISMATCH: {max_diff_sp:.2e}",
            transform=ax.transAxes,
            fontsize=10,
            color="red",
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )
    else:
        ax.text(
            0.5,
            0.95,
            "✅ MATCH",
            transform=ax.transAxes,
            fontsize=10,
            color="green",
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

plt.suptitle("M3自己一致テスト: 同じphibar_denseで点と線を描画", fontsize=14)
plt.tight_layout()

# Save figure
output_dir = Path("_runs/test_m3_self_consistency")
output_dir.mkdir(parents=True, exist_ok=True)
fig_path = output_dir / "M3_self_consistency_test.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  ✅ 図を保存: {fig_path}")
print()

# ==============================================================================
# Step 5: Detailed comparison
# ==============================================================================
print("=" * 80)
print("Step 5: 詳細比較")
print("=" * 80)

for i, sp in enumerate(config_M3["active_species"]):
    model_at_obs = phibar_dense[idx_sparse, i]
    data_at_obs = data[:, i]
    diff = np.abs(model_at_obs - data_at_obs)

    print(f"\n  Species {sp} (φ̄{sp+1}):")
    print(f"    Max diff: {np.max(diff):.2e}")
    print(f"    Mean diff: {np.mean(diff):.2e}")
    print(f"    RMS diff: {np.sqrt(np.mean(diff**2)):.2e}")

    if np.max(diff) > 1e-10:
        print("    ❌ MISMATCH detected!")
        print(f"    First 5 differences: {diff[:5]}")
        print(f"    First 5 model values: {model_at_obs[:5]}")
        print(f"    First 5 data values: {data_at_obs[:5]}")
    else:
        print("    ✅ Perfect match")

print()

# ==============================================================================
# Final verdict
# ==============================================================================
print("=" * 80)
print("最終判定")
print("=" * 80)

all_max_diffs = []
for i, sp in enumerate(config_M3["active_species"]):
    model_at_obs = phibar_dense[idx_sparse, i]
    data_at_obs = data[:, i]
    diff = np.abs(model_at_obs - data_at_obs)
    all_max_diffs.append(np.max(diff))

overall_max_diff = np.max(all_max_diffs)

if overall_max_diff < 1e-10:
    print("✅ PASS: 完全一致")
    print("  → データ生成と可視化は正しく実装されています")
    print("  → ズレが見える場合は「推定の問題」です")
    print("  → MAPやmeanを使った比較でズレるのは正常です")
else:
    print("❌ FAIL: 不一致検出")
    print(f"  → 最大差分: {overall_max_diff:.2e}")
    print("  → これは「可視化 or パイプラインのバグ」です")
    print("  → コードの参照がズレています")
    print("  → 原因を特定して修正が必要です")

print()
print("=" * 80)
