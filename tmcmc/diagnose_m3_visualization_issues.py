"""
M3可視化問題の診断スクリプト

ユーザーが指摘した3つの原因をチェック：

原因①（最有力）
比較している"線"がM3の真のphibarじゃない
- 線：M3(MAP) や M3(mean) → 点（theta_true）と合わない
- 線：再計算したphibar → 内部入力が違えばズレる
- 線：φ や ψ → phibarとは別物

原因②
観測点インデックスと描画インデックスが違う
- data: phibar[idx_sparse] → 正しい
- line: phibar_dense → OK
- でもidxがズレてる → 点が線から離れる

原因③
M3内部でM1/M2を「真値でなく再計算」している
- 生成時: M1/M2 = theta_true
- 描画時: M1/M2 = 再生成 or 初期値
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
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from main.case2_main import select_sparse_data_indices, ExperimentConfig

print("=" * 80)
print("M3可視化問題の診断")
print("=" * 80)
print()

# Configuration
exp_config = ExperimentConfig()
exp_config.sigma_obs = 0.01
exp_config.cov_rel = 0.005
exp_config.n_data = 20
exp_config.random_seed = 42

config_M3 = MODEL_CONFIGS["M3"]
theta_true = get_theta_true()

# ==============================================================================
# Setup: Generate reference data
# ==============================================================================
print("=" * 80)
print("参照データ生成（theta_true使用）")
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

tsm_M3_ref = BiofilmTSM_Analytical(
    solver_M3,
    active_theta_indices=config_M3["active_indices"],
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_true,
)

t_arr_M3_ref, x0_M3_ref, sig2_M3_ref = tsm_M3_ref.solve_tsm(theta_true)
phibar_M3_ref = compute_phibar(x0_M3_ref, config_M3["active_species"])
idx_sparse_ref = select_sparse_data_indices(len(t_arr_M3_ref), exp_config.n_data, t_arr=t_arr_M3_ref)

# Generate data with noise
from core.tmcmc import _stable_hash_int
rng_M3 = np.random.default_rng(exp_config.random_seed + (_stable_hash_int("M3") % 1000))
data_M3_ref = np.zeros((exp_config.n_data, len(config_M3["active_species"])))
for i, sp in enumerate(config_M3["active_species"]):
    data_M3_ref[:, i] = phibar_M3_ref[idx_sparse_ref, i] + rng_M3.standard_normal(exp_config.n_data) * exp_config.sigma_obs

print(f"  Reference phibar shape: {phibar_M3_ref.shape}")
print(f"  Reference data shape: {data_M3_ref.shape}")
print(f"  Reference idx_sparse: {idx_sparse_ref[:5]} ... {idx_sparse_ref[-5:]}")
print()

# ==============================================================================
# Check ①: What line is being plotted?
# ==============================================================================
print("=" * 80)
print("原因①チェック: 描画している「線」は何か？")
print("=" * 80)
print()

# Scenario 1: Plot with theta_true (should match)
print("Scenario 1: theta_trueで生成したphibar（期待される線）")
t_arr_1, x0_1, sig2_1 = tsm_M3_ref.solve_tsm(theta_true)
phibar_1 = compute_phibar(x0_1, config_M3["active_species"])
diff_1 = np.max(np.abs(phibar_1 - phibar_M3_ref))
print(f"  phibar vs reference: max diff = {diff_1:.2e}")
if diff_1 < 1e-10:
    print("  ✅ 完全一致（期待通り）")
else:
    print("  ❌ 不一致（予期しない）")
print()

# Scenario 2: Plot with MAP (will differ)
print("Scenario 2: MAPで生成したphibar（ズレるのは正常）")
# Simulate MAP estimate (slightly different from true)
theta_MAP = theta_true.copy()
theta_MAP[10:14] += 0.1 * np.random.randn(4)  # Add small perturbation
tsm_M3_MAP = BiofilmTSM_Analytical(
    solver_M3,
    active_theta_indices=config_M3["active_indices"],
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_MAP,
)
t_arr_2, x0_2, sig2_2 = tsm_M3_MAP.solve_tsm(theta_MAP)
phibar_2 = compute_phibar(x0_2, config_M3["active_species"])
diff_2 = np.max(np.abs(phibar_2[idx_sparse_ref] - phibar_M3_ref[idx_sparse_ref]))
print(f"  MAP phibar vs reference at obs points: max diff = {diff_2:.6f}")
print("  ⚠ ズレるのは正常（MAP ≠ theta_true）")
print()

# Scenario 3: Plot with recomputed phibar (different internal inputs)
print("Scenario 3: 再計算したphibar（内部入力が違う可能性）")
# This simulates the case where M1/M2 are recomputed
theta_recomputed = theta_true.copy()
# Simulate M1/M2 being recomputed with different values
theta_recomputed[0:5] += 0.05 * np.random.randn(5)
theta_recomputed[5:10] += 0.05 * np.random.randn(5)
tsm_M3_recomp = BiofilmTSM_Analytical(
    solver_M3,
    active_theta_indices=config_M3["active_indices"],
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_recomputed,
)
t_arr_3, x0_3, sig2_3 = tsm_M3_recomp.solve_tsm(theta_recomputed)
phibar_3 = compute_phibar(x0_3, config_M3["active_species"])
diff_3 = np.max(np.abs(phibar_3[idx_sparse_ref] - phibar_M3_ref[idx_sparse_ref]))
print(f"  Recomputed phibar vs reference at obs points: max diff = {diff_3:.6f}")
print("  ⚠ ズレるのは正常（内部入力が違う）")
print()

# Scenario 4: Plot with phi or psi instead of phibar
print("Scenario 4: φ や ψ を描画（phibarではない）")
phi_only = x0_M3_ref[:, config_M3["active_species"][0]]
psi_only = x0_M3_ref[:, 5 + config_M3["active_species"][0]]
phibar_correct = phibar_M3_ref[:, 0]
diff_phi = np.max(np.abs(phi_only[idx_sparse_ref] - phibar_correct[idx_sparse_ref]))
diff_psi = np.max(np.abs(psi_only[idx_sparse_ref] - phibar_correct[idx_sparse_ref]))
print(f"  φ vs phibar at obs points: max diff = {diff_phi:.6f}")
print(f"  ψ vs phibar at obs points: max diff = {diff_psi:.6f}")
print("  ⚠ ズレるのは正常（φやψはphibarではない）")
print()

# ==============================================================================
# Check ②: Index mismatch
# ==============================================================================
print("=" * 80)
print("原因②チェック: 観測点インデックスと描画インデックスの不一致")
print("=" * 80)
print()

# Check if idx_sparse is correctly used
print("観測点インデックスの確認:")
print(f"  idx_sparse shape: {idx_sparse_ref.shape}")
print(f"  idx_sparse range: [{idx_sparse_ref.min()}, {idx_sparse_ref.max()}]")
print(f"  phibar_dense shape: {phibar_M3_ref.shape}")
print(f"  phibar_dense length: {len(phibar_M3_ref)}")

# Check if all indices are valid
if np.all((idx_sparse_ref >= 0) & (idx_sparse_ref < len(phibar_M3_ref))):
    print("  ✅ すべてのインデックスが有効範囲内")
else:
    print("  ❌ 無効なインデックスが存在")
    invalid = idx_sparse_ref[(idx_sparse_ref < 0) | (idx_sparse_ref >= len(phibar_M3_ref))]
    print(f"     無効なインデックス: {invalid}")

# Check if data points match model at observation points
print()
print("データ点とモデル値の一致確認:")
for i, sp in enumerate(config_M3["active_species"]):
    model_at_obs = phibar_M3_ref[idx_sparse_ref, i]
    data_at_obs = data_M3_ref[:, i]
    # Account for noise
    noise_level = exp_config.sigma_obs
    diff = np.abs(model_at_obs - data_at_obs)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Species {sp}:")
    print(f"    Max diff: {max_diff:.6f} (noise level: {noise_level:.6f})")
    print(f"    Mean diff: {mean_diff:.6f}")
    
    if max_diff < 3 * noise_level:  # Within 3 sigma
        print(f"    ✅ ノイズ範囲内（正常）")
    else:
        print(f"    ⚠ ノイズ範囲外（要確認）")
print()

# ==============================================================================
# Check ③: M1/M2 recomputation
# ==============================================================================
print("=" * 80)
print("原因③チェック: M3内部でM1/M2を再計算しているか？")
print("=" * 80)
print()

# M3 uses M1/M2 parameters as base
# Check if theta_base changes between generation and plotting
print("M3のtheta_base確認:")
print(f"  theta_true[0:5] (M1): {theta_true[0:5]}")
print(f"  theta_true[5:10] (M2): {theta_true[5:10]}")
print(f"  theta_true[10:14] (M3): {theta_true[10:14]}")

# Simulate the case where M1/M2 are recomputed
theta_base_gen = theta_true.copy()  # Generation time
theta_base_plot = theta_true.copy()  # Plotting time
theta_base_plot[0:5] += 0.1 * np.random.randn(5)  # M1 recomputed
theta_base_plot[5:10] += 0.1 * np.random.randn(5)  # M2 recomputed

print()
print("シミュレーション: M1/M2が再計算された場合")
print(f"  Generation theta_base[0:5]: {theta_base_gen[0:5]}")
print(f"  Plotting theta_base[0:5]:   {theta_base_plot[0:5]}")
print(f"  Generation theta_base[5:10]: {theta_base_gen[5:10]}")
print(f"  Plotting theta_base[5:10]:   {theta_base_plot[5:10]}")

# Generate phibar with different theta_base
tsm_M3_gen = BiofilmTSM_Analytical(
    solver_M3,
    active_theta_indices=config_M3["active_indices"],
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_base_gen,
)
tsm_M3_plot = BiofilmTSM_Analytical(
    solver_M3,
    active_theta_indices=config_M3["active_indices"],
    cov_rel=exp_config.cov_rel,
    use_complex_step=True,
    use_analytical=True,
    theta_linearization=theta_base_plot,
)

t_arr_gen, x0_gen, sig2_gen = tsm_M3_gen.solve_tsm(theta_base_gen)
t_arr_plot, x0_plot, sig2_plot = tsm_M3_plot.solve_tsm(theta_base_plot)
phibar_gen = compute_phibar(x0_gen, config_M3["active_species"])
phibar_plot = compute_phibar(x0_plot, config_M3["active_species"])

diff_base = np.max(np.abs(phibar_gen[idx_sparse_ref] - phibar_plot[idx_sparse_ref]))
print(f"  phibar difference at obs points: max diff = {diff_base:.6f}")
print("  ⚠ ズレるのは正常（theta_baseが違う）")
print()

# ==============================================================================
# Summary and recommendations
# ==============================================================================
print("=" * 80)
print("診断結果サマリー")
print("=" * 80)
print()
print("原因①: 描画している「線」の確認")
print("  → theta_trueで生成したphibarと比較")
print("  → MAP/meanで生成したphibarはズレる（正常）")
print("  → φやψを描画していないか確認")
print()
print("原因②: インデックスの確認")
print("  → idx_sparseが有効範囲内か確認")
print("  → data = phibar[idx_sparse]が正しいか確認")
print("  → ノイズ範囲内のズレは正常")
print()
print("原因③: M1/M2の再計算確認")
print("  → データ生成時と描画時でtheta_baseが同じか確認")
print("  → M1/M2が再計算されていないか確認")
print()
print("推奨される次のステップ:")
print("  1. test_m3_self_consistency.pyを実行して自己一致を確認")
print("  2. 実際の可視化コードで何を描画しているか確認")
print("  3. データ生成時のphibarを保存して、描画時に使用")
print("=" * 80)
