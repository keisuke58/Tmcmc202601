"""
完了済みの実行結果からM1/M2推定誤差を確認

別の実行結果を使用してM1/M2の推定誤差を確認
"""

import numpy as np
import sys
import io
from pathlib import Path
import json

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIGS
from improved1207_paper_jit import get_theta_true

print("=" * 80)
print("M1とM2の推定誤差がM3に伝播しているかを確認")
print("（完了済みの実行結果を使用）")
print("=" * 80)

theta_true = get_theta_true()

print("\ntheta_true (全14パラメータ):")
print(f"  {theta_true}")

# Try to find completed runs
completed_runs = [
    Path("_runs/m1_1000_20260118_083726_good"),
    Path("_runs/m1_check_np500_ns20_sig002_rerun_full"),
]

found_run = None
MAP_M1 = None
MEAN_M1 = None
MAP_M2 = None
MEAN_M2 = None

for run_dir in completed_runs:
    if run_dir.exists():
        try:
            with open(run_dir / "theta_MAP_M1.json", "r") as f:
                theta_MAP_M1_data = json.load(f)
            MAP_M1 = np.array(theta_MAP_M1_data["theta_sub"])

            with open(run_dir / "theta_MEAN_M1.json", "r") as f:
                theta_MEAN_M1_data = json.load(f)
            MEAN_M1 = np.array(theta_MEAN_M1_data["theta_sub"])

            print(f"\n[OK] 実行結果を読み込み: {run_dir}")
            found_run = run_dir
            break
        except Exception as e:
            continue

if MAP_M1 is None:
    print("\n[WARNING] 完了済みの実行結果が見つかりませんでした")
    print("  実行が完了するまで待つか、別の実行結果を使用してください")
    sys.exit(1)

# For M2, we need to check if it exists in the same run or use theta_true as reference
# Since M2 might not be in single-model runs, we'll use theta_true for comparison
print(f"\n{'='*80}")
print("M1推定誤差の分析")
print(f"{'='*80}")

theta_true_M1 = theta_true[MODEL_CONFIGS["M1"]["active_indices"]]

print(f"\ntheta_true (M1): {theta_true_M1}")
print(f"MAP_M1: {MAP_M1}")
print(f"MEAN_M1: {MEAN_M1}")

error_MAP_M1 = MAP_M1 - theta_true_M1
error_MEAN_M1 = MEAN_M1 - theta_true_M1

print(f"\n推定誤差 (MAP): {error_MAP_M1}")
print(f"推定誤差 (MEAN): {error_MEAN_M1}")

rmse_MAP_M1 = np.sqrt(np.mean(error_MAP_M1**2))
rmse_MEAN_M1 = np.sqrt(np.mean(error_MEAN_M1**2))

print(f"\nRMSE (MAP): {rmse_MAP_M1:.8f}")
print(f"RMSE (MEAN): {rmse_MEAN_M1:.8f}")

max_error_MAP_M1 = np.max(np.abs(error_MAP_M1))
max_error_MEAN_M1 = np.max(np.abs(error_MEAN_M1))

print(f"最大絶対誤差 (MAP): {max_error_MAP_M1:.8f}")
print(f"最大絶対誤差 (MEAN): {max_error_MEAN_M1:.8f}")

# Relative error
relative_error_MAP_M1 = np.abs(error_MAP_M1) / (np.abs(theta_true_M1) + 1e-10)
relative_error_MEAN_M1 = np.abs(error_MEAN_M1) / (np.abs(theta_true_M1) + 1e-10)

print(f"\n相対誤差 (MAP): {relative_error_MAP_M1}")
print(f"相対誤差 (MEAN): {relative_error_MEAN_M1}")
print(f"最大相対誤差 (MAP): {np.max(relative_error_MAP_M1):.4f}")
print(f"最大相対誤差 (MEAN): {np.max(relative_error_MEAN_M1):.4f}")

# For M2, use theta_true as reference (since we don't have M2 results)
print(f"\n{'='*80}")
print("M2推定誤差の分析（参考）")
print(f"{'='*80}")

theta_true_M2 = theta_true[MODEL_CONFIGS["M2"]["active_indices"]]

print(f"\ntheta_true (M2): {theta_true_M2}")
print("  [NOTE] M2の推定結果は利用できませんでした")
print("  実行が完了したら、M2の推定誤差も確認してください")

# Simulate M3 base error assuming M2 has similar error to M1
print(f"\n{'='*80}")
print("M3への影響分析（M1誤差ベースのシミュレーション）")
print(f"{'='*80}")

# Assume M2 has similar error pattern to M1 (worst case)
# In reality, M2 error might be different
MAP_M2_simulated = theta_true_M2 + error_MAP_M1  # Use M1 error pattern
MEAN_M2_simulated = theta_true_M2 + error_MEAN_M1

prior_mean = 1.5  # (0.0 + 3.0) / 2.0
theta_base_M3_MAP = np.full(14, prior_mean)
theta_base_M3_MAP[0:5] = MAP_M1
theta_base_M3_MAP[5:10] = MAP_M2_simulated  # Simulated

theta_base_M3_MEAN = np.full(14, prior_mean)
theta_base_M3_MEAN[0:5] = MEAN_M1
theta_base_M3_MEAN[5:10] = MEAN_M2_simulated  # Simulated

print("\ntheta_base_M3 (MAP使用、M2はシミュレーション):")
print(f"  M1部分: {theta_base_M3_MAP[0:5]}")
print(f"  M2部分: {theta_base_M3_MAP[5:10]} (simulated)")
print(f"  M3部分: {theta_base_M3_MAP[10:14]} (prior mean)")

print("\ntheta_true:")
print(f"  M1部分: {theta_true[0:5]}")
print(f"  M2部分: {theta_true[5:10]}")
print(f"  M3部分: {theta_true[10:14]}")

error_base_M3_MAP = theta_base_M3_MAP - theta_true
error_base_M3_MEAN = theta_base_M3_MEAN - theta_true

print("\nM3推定時のベース誤差 (MAP使用、M2はシミュレーション):")
print(f"  M1部分誤差: {error_base_M3_MAP[0:5]}")
print(f"  M2部分誤差: {error_base_M3_MAP[5:10]} (simulated)")
print(f"  M3部分誤差: {error_base_M3_MAP[10:14]} (prior mean - true)")

rmse_base_M3_MAP = np.sqrt(np.mean(error_base_M3_MAP**2))
rmse_base_M3_MEAN = np.sqrt(np.mean(error_base_M3_MEAN**2))

print(f"\nベースRMSE (MAP使用): {rmse_base_M3_MAP:.8f}")
print(f"ベースRMSE (MEAN使用): {rmse_base_M3_MEAN:.8f}")

# Check if M1 errors are significant
sigma_obs = 0.01
print(f"\n{'='*80}")
print("判定")
print(f"{'='*80}")

if max_error_MAP_M1 > 0.1 or max_error_MEAN_M1 > 0.1:
    print("  [WARNING] M1の推定誤差が大きい（>0.1）")
    print("    これがM3の適合性低下の原因の可能性が高い")
    print(f"    最大絶対誤差: MAP={max_error_MAP_M1:.6f}, MEAN={max_error_MEAN_M1:.6f}")
else:
    print("  [OK] M1の推定誤差は小さい（<0.1）")
    print(f"    最大絶対誤差: MAP={max_error_MAP_M1:.6f}, MEAN={max_error_MEAN_M1:.6f}")

if rmse_base_M3_MAP > 0.05:
    print("  [WARNING] M3推定時のベース誤差が大きい（RMSE > 0.05）")
    print("    M1/M2の推定誤差がM3に伝播している可能性が高い")
    print(f"    ベースRMSE: {rmse_base_M3_MAP:.8f}")
else:
    print("  [OK] M3推定時のベース誤差は小さい（RMSE < 0.05）")
    print(f"    ベースRMSE: {rmse_base_M3_MAP:.8f}")

# Check relative error
if np.max(relative_error_MAP_M1) > 0.1:
    print("  [WARNING] M1の相対誤差が大きい（>10%）")
    print(f"    最大相対誤差: {np.max(relative_error_MAP_M1):.2%}")
else:
    print("  [OK] M1の相対誤差は小さい（<10%）")
    print(f"    最大相対誤差: {np.max(relative_error_MAP_M1):.2%}")

print("\n推奨される対策:")
print("  1. M1とM2の推定精度を向上させる（より多くのパーティクル、より多くのステージ）")
print("  2. M3推定時にM1/M2の不確実性を考慮する（階層的ベイズ）")
print("  3. M3データ生成時にM1/M2の推定結果を使用する（一貫性の確保）")
print("  4. 実行が完了したら、実際のM2推定誤差も確認する")

print(f"\n{'='*80}")
print("確認完了")
print(f"{'='*80}")
print("\n[NOTE] M2の推定結果は利用できませんでした")
print("  実行が完了したら、実際のM2推定誤差も確認してください")
