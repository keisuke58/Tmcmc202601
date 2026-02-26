"""
M3モデルの問題診断スクリプト

原因候補の確認:
1. 時間正規化の不整合（t/T の定義違い）
2. データ生成とプロットで同じインデックス/時間を使っているか
3. no-noiseでM3を再生成して確認
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
from visualization import PlotManager
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig
from core.tmcmc import _stable_hash_int

print("=" * 80)
print("M3モデル問題診断")
print("=" * 80)

# Configuration
exp_config_sigma_obs = 0.01
exp_config_cov_rel = 0.005
exp_config_n_data = 20
exp_config_random_seed = 42

# Test both M1 and M3 for comparison
models_to_test = ["M1", "M3"]

for model_name in models_to_test:
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    config = MODEL_CONFIGS[model_name]
    theta_true = get_theta_true()

    solver_kwargs = {
        k: v
        for k, v in config.items()
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

    # Generate TSM solution
    t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
    phibar = compute_phibar(x0, config["active_species"])

    print("\n1. 時間配列の確認:")
    print(f"   t_arr shape: {t_arr.shape}")
    print(f"   t_arr range: [{t_arr.min():.8f}, {t_arr.max():.8f}]")
    print(f"   t_arr[0]: {t_arr[0]:.8f}, t_arr[-1]: {t_arr[-1]:.8f}")

    # Check 1: 時間正規化の定義確認
    print("\n2. 時間正規化の定義確認:")
    t_min = t_arr.min()
    t_max = t_arr.max()
    t_normalized = (t_arr - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_arr)
    print("   正規化方法: (t_arr - t_min) / (t_max - t_min)")
    print(f"   t_min: {t_min:.8f}, t_max: {t_max:.8f}")
    print(f"   t_normalized range: [{t_normalized.min():.6f}, {t_normalized.max():.6f}]")
    print(f"   t_normalized[0]: {t_normalized[0]:.6f}, t_normalized[-1]: {t_normalized[-1]:.6f}")

    # Check 2: データポイント選択の確認
    print("\n3. データポイント選択の確認:")
    idx_sparse = select_sparse_data_indices(len(t_arr), exp_config_n_data, t_arr=t_arr)
    print(f"   idx_sparse shape: {idx_sparse.shape}")
    print(f"   idx_sparse range: [{idx_sparse.min()}, {idx_sparse.max()}]")
    print(f"   idx_sparse (first 5): {idx_sparse[:5]}")
    print(f"   idx_sparse (last 5): {idx_sparse[-5:]}")

    # Check normalized time at observation points
    t_obs_normalized = t_normalized[idx_sparse]
    print("\n   観測点の正規化時間:")
    print(f"   t_obs_normalized (first 5): {t_obs_normalized[:5]}")
    print(f"   t_obs_normalized (last 5): {t_obs_normalized[-5:]}")

    # Expected normalized times (0.05 interval)
    expected_times = np.arange(0.05, 1.0 + 0.001, 0.05)[:exp_config_n_data]
    print("\n   期待される正規化時間 (0.05間隔):")
    print(f"   expected (first 5): {expected_times[:5]}")
    print(f"   expected (last 5): {expected_times[-5:]}")

    # Check difference
    time_diff = np.abs(t_obs_normalized - expected_times)
    max_time_diff = np.max(time_diff)
    print("\n   時間のずれ:")
    print(f"   max difference: {max_time_diff:.8f}")
    if max_time_diff < 0.01:
        print("   [OK] 時間のずれは許容範囲内")
    else:
        print("   [WARNING] 時間のずれが大きい！")
        print(f"   Differences: {time_diff}")

    # Check 3: データ生成（with noise）
    print(f"\n4. データ生成（with noise, sigma_obs={exp_config_sigma_obs}）:")
    rng = np.random.default_rng(exp_config_random_seed + (_stable_hash_int(model_name) % 1000))

    data_with_noise = np.zeros((exp_config_n_data, len(config["active_species"])))
    for i, sp in enumerate(config["active_species"]):
        data_with_noise[:, i] = (
            phibar[idx_sparse, i] + rng.standard_normal(exp_config_n_data) * exp_config_sigma_obs
        )

    # Check residuals
    residuals_with_noise = data_with_noise - phibar[idx_sparse, :]
    print(f"   Residuals mean: {np.mean(residuals_with_noise):.8f}")
    print(
        f"   Residuals std: {np.std(residuals_with_noise):.6f} (expected: {exp_config_sigma_obs})"
    )
    print(
        f"   Residuals range: [{np.min(residuals_with_noise):.6f}, {np.max(residuals_with_noise):.6f}]"
    )

    # Check 4: データ生成（no noise）
    print("\n5. データ生成（no noise）:")
    data_no_noise = phibar[idx_sparse, :].copy()

    # Check if no-noise data matches phibar exactly
    max_diff = np.max(np.abs(data_no_noise - phibar[idx_sparse, :]))
    print(f"   Max difference from phibar: {max_diff:.10f}")
    if max_diff < 1e-10:
        print("   [OK] no-noiseデータはphibarと完全一致")
    else:
        print("   [ERROR] no-noiseデータがphibarと一致しない！")

    # Check 5: 各speciesの乖離度
    print("\n6. 各speciesの乖離度（no-noiseデータ）:")
    for i, sp in enumerate(config["active_species"]):
        phibar_at_obs = phibar[idx_sparse, i]
        data_at_obs = data_no_noise[:, i]
        diff = np.abs(data_at_obs - phibar_at_obs)
        max_diff_sp = np.max(diff)
        mean_diff_sp = np.mean(diff)
        print(f"   Species {sp}: max_diff={max_diff_sp:.10f}, mean_diff={mean_diff_sp:.10f}")

    # Check 6: 後半の乖離（時間正規化の問題を検出）
    print("\n7. 後半の乖離チェック（時間正規化問題の検出）:")
    n_half = exp_config_n_data // 2
    first_half_residuals = np.abs(residuals_with_noise[:n_half, :])
    second_half_residuals = np.abs(residuals_with_noise[n_half:, :])

    print(f"   前半の平均残差: {np.mean(first_half_residuals):.6f}")
    print(f"   後半の平均残差: {np.mean(second_half_residuals):.6f}")
    print(f"   後半/前半比: {np.mean(second_half_residuals) / np.mean(first_half_residuals):.4f}")

    if np.mean(second_half_residuals) > 1.5 * np.mean(first_half_residuals):
        print("   [WARNING] 後半の乖離が大きい！時間正規化の問題の可能性")
    else:
        print("   [OK] 前半と後半の乖離に大きな差なし")

    # Check 7: M1 vs M3 の比較
    if model_name == "M1":
        m1_phibar = phibar.copy()
        m1_idx_sparse = idx_sparse.copy()
        m1_t_normalized = t_normalized.copy()
    elif model_name == "M3":
        print("\n8. M1 vs M3 の比較:")
        print(f"   M1 active_species: {MODEL_CONFIGS['M1']['active_species']}")
        print(f"   M3 active_species: {config['active_species']}")
        print(f"   M1 n_species: {len(MODEL_CONFIGS['M1']['active_species'])}")
        print(f"   M3 n_species: {len(config['active_species'])}")

        # Check if indices are the same
        if np.array_equal(m1_idx_sparse, idx_sparse):
            print("   [OK] データポイントインデックスはM1とM3で同じ")
        else:
            print("   [WARNING] データポイントインデックスがM1とM3で異なる")
            print(f"   M1 indices: {m1_idx_sparse}")
            print(f"   M3 indices: {idx_sparse}")

        # Check if normalized times are the same
        if np.allclose(m1_t_normalized[idx_sparse], t_normalized[idx_sparse], atol=1e-6):
            print("   [OK] 正規化時間はM1とM3で同じ")
        else:
            print("   [WARNING] 正規化時間がM1とM3で異なる")
            print(
                f"   Max difference: {np.max(np.abs(m1_t_normalized[idx_sparse] - t_normalized[idx_sparse])):.8f}"
            )

print(f"\n{'='*80}")
print("診断完了")
print(f"{'='*80}")
print(
    """
判定フロー:
1. no-noiseでもズレる？
   → YES → 時間正規化 or モデル構造の問題
   → NO → ノイズ＋尤度設計の問題

2. M1はOKでM3だけNG？
   → YES → M3の結合項/パラメータ不足
   → NO → 共通の問題（時間正規化など）

3. 後半ほど乖離が拡大？
   → YES → 時間正規化の不整合が確定的
   → NO → モデル構造の問題
"""
)
