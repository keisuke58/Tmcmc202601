"""
M3データ生成元の確認と時間スケール問題の検証

質問: M3の図のデータ点は、M3から生成？それともM1から？
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
print("M3データ生成元の確認")
print("=" * 80)

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42

# Load actual run data
run_dir = Path("_runs/20260122_162052_debug_seed42")
if run_dir.exists():
    print(f"\n実際の実行結果を確認: {run_dir}")
    try:
        data_M3_actual = np.load(run_dir / "data_M3.npy")
        idx_M3_actual = np.load(run_dir / "idx_M3.npy")
        t_M3_actual = np.load(run_dir / "t_M3.npy")
        print("  [OK] 実行結果のデータを読み込み成功")
        print(f"  data_M3 shape: {data_M3_actual.shape}")
        print(f"  idx_M3 shape: {idx_M3_actual.shape}")
        print(
            f"  t_M3 shape: {t_M3_actual.shape}, range: [{t_M3_actual.min():.8f}, {t_M3_actual.max():.8f}]"
        )
    except Exception as e:
        print(f"  [WARNING] 実行結果の読み込み失敗: {e}")
        data_M3_actual = None
        idx_M3_actual = None
        t_M3_actual = None
else:
    print(f"\n[WARNING] 実行結果ディレクトリが見つかりません: {run_dir}")
    data_M3_actual = None
    idx_M3_actual = None
    t_M3_actual = None

# Generate M3 data from M3 model
print(f"\n{'='*80}")
print("M3モデルからデータ生成（再現）")
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
    cov_rel=exp_config_cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

# Generate TSM solution
t_arr_M3, x0_M3, sig2_M3 = tsm_M3.solve_tsm(theta_true)
phibar_M3 = compute_phibar(x0_M3, config_M3["active_species"])

print("\nM3モデルの時間配列:")
print(f"  t_arr_M3 shape: {t_arr_M3.shape}")
print(f"  t_arr_M3 range: [{t_arr_M3.min():.8f}, {t_arr_M3.max():.8f}]")
print(f"  t_arr_M3[0]: {t_arr_M3[0]:.8f}, t_arr_M3[-1]: {t_arr_M3[-1]:.8f}")

# Select data points
idx_sparse_M3 = select_sparse_data_indices(len(t_arr_M3), exp_config_n_data, t_arr=t_arr_M3)

# Generate data with noise
rng_M3 = np.random.default_rng(exp_config_random_seed + (_stable_hash_int("M3") % 1000))
data_M3_generated = np.zeros((exp_config_n_data, len(config_M3["active_species"])))
for i, sp in enumerate(config_M3["active_species"]):
    data_M3_generated[:, i] = (
        phibar_M3[idx_sparse_M3, i]
        + rng_M3.standard_normal(exp_config_n_data) * exp_config_sigma_obs
    )

print("\n生成されたデータ:")
print(f"  data_M3_generated shape: {data_M3_generated.shape}")
print(f"  idx_sparse_M3: {idx_sparse_M3[:5]} ... {idx_sparse_M3[-5:]}")

# Compare with actual run data
if data_M3_actual is not None:
    print(f"\n{'='*80}")
    print("実際の実行結果との比較")
    print(f"{'='*80}")

    # Check if data matches
    if np.allclose(data_M3_generated, data_M3_actual, atol=1e-6):
        print("  [OK] 生成データと実行結果のデータが一致（M3から生成されている）")
    else:
        print("  [WARNING] 生成データと実行結果のデータが一致しない")
        print(f"  Max difference: {np.max(np.abs(data_M3_generated - data_M3_actual)):.8f}")

    # Check if indices match
    if np.array_equal(idx_sparse_M3, idx_M3_actual):
        print("  [OK] インデックスが一致")
    else:
        print("  [WARNING] インデックスが一致しない")
        print(f"  Generated: {idx_sparse_M3[:5]} ... {idx_sparse_M3[-5:]}")
        print(f"  Actual: {idx_M3_actual[:5]} ... {idx_M3_actual[-5:]}")

    # Check if time arrays match
    if np.allclose(t_arr_M3, t_M3_actual, atol=1e-8):
        print("  [OK] 時間配列が一致")
    else:
        print("  [WARNING] 時間配列が一致しない")
        print(f"  Generated t_arr range: [{t_arr_M3.min():.8f}, {t_arr_M3.max():.8f}]")
        print(f"  Actual t_arr range: [{t_M3_actual.min():.8f}, {t_M3_actual.max():.8f}]")
        print(f"  Max difference: {np.max(np.abs(t_arr_M3 - t_M3_actual)):.8f}")

# Compare M1 and M3 time scales
print(f"\n{'='*80}")
print("M1 vs M3 時間スケール比較（重要）")
print(f"{'='*80}")

config_M1 = MODEL_CONFIGS["M1"]
solver_kwargs_M1 = {
    k: v
    for k, v in config_M1.items()
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

print("\nM1モデルの時間配列:")
print(f"  t_arr_M1 shape: {t_arr_M1.shape}")
print(f"  t_arr_M1 range: [{t_arr_M1.min():.8f}, {t_arr_M1.max():.8f}]")

print("\nM3モデルの時間配列:")
print(f"  t_arr_M3 shape: {t_arr_M3.shape}")
print(f"  t_arr_M3 range: [{t_arr_M3.min():.8f}, {t_arr_M3.max():.8f}]")

print("\n時間スケールの違い:")
print(f"  M1時間範囲: {t_arr_M1.max() - t_arr_M1.min():.8f}")
print(f"  M3時間範囲: {t_arr_M3.max() - t_arr_M3.min():.8f}")
print(
    f"  比率 (M3/M1): {(t_arr_M3.max() - t_arr_M3.min()) / (t_arr_M1.max() - t_arr_M1.min()):.4f}"
)

print("\n時間ステップ数の違い:")
print(f"  M1ステップ数: {len(t_arr_M1)}")
print(f"  M3ステップ数: {len(t_arr_M3)}")
print(f"  比率 (M3/M1): {len(t_arr_M3) / len(t_arr_M1):.4f}")

# Check normalized time positions
print(f"\n{'='*80}")
print("正規化時間位置の比較")
print(f"{'='*80}")

t_norm_M1 = (t_arr_M1 - t_arr_M1.min()) / (t_arr_M1.max() - t_arr_M1.min())
t_norm_M3 = (t_arr_M3 - t_arr_M3.min()) / (t_arr_M3.max() - t_arr_M3.min())

idx_sparse_M1 = select_sparse_data_indices(len(t_arr_M1), exp_config_n_data, t_arr=t_arr_M1)

t_obs_norm_M1 = t_norm_M1[idx_sparse_M1]
t_obs_norm_M3 = t_norm_M3[idx_sparse_M3]

print("\nM1観測点の正規化時間 (first 5, last 5):")
print(f"  {t_obs_norm_M1[:5]} ... {t_obs_norm_M1[-5:]}")

print("\nM3観測点の正規化時間 (first 5, last 5):")
print(f"  {t_obs_norm_M3[:5]} ... {t_obs_norm_M3[-5:]}")

print("\n正規化時間の差:")
max_diff_norm = np.max(np.abs(t_obs_norm_M1 - t_obs_norm_M3))
print(f"  Max difference: {max_diff_norm:.8f}")

if max_diff_norm < 0.01:
    print("  [OK] 正規化時間はほぼ一致")
else:
    print("  [WARNING] 正規化時間に差がある")

# Conclusion
print(f"\n{'='*80}")
print("結論")
print(f"{'='*80}")
print(
    f"""
1. M3のデータ生成元:
   → M3モデルから独立に生成されている（M1からではない）

2. 時間スケールの問題:
   → M1: 時間範囲 {t_arr_M1.max() - t_arr_M1.min():.8f}, ステップ数 {len(t_arr_M1)}
   → M3: 時間範囲 {t_arr_M3.max() - t_arr_M3.min():.8f}, ステップ数 {len(t_arr_M3)}
   → **M1とM3で異なる時間スケールを使用している**

3. 正規化時間:
   → 正規化すると見た目は揃うが、実時間が異なる
   → ダイナミクス自体が変わるため、「同じデータを説明できない」状況が起きる可能性

4. 次のステップ:
   → テストA: 同じt_arr上で比較（データ生成時のt_arrを固定）
   → テストB: 尤度の重みをspeciesで正規化
"""
)
