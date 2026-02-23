"""
M1とM2の推定誤差がM3に伝播しているかを確認

M1とM2のMAP/Mean推定値がtheta_trueにどれだけ近いかを確認
"""
import numpy as np
import sys
import io
from pathlib import Path
import json

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIGS
from improved1207_paper_jit import get_theta_true

print("=" * 80)
print("M1とM2の推定誤差がM3に伝播しているかを確認")
print("=" * 80)

# Load actual run data
run_dir = Path("_runs/20260122_162052_debug_seed42")
print(f"\n実行結果ディレクトリ: {run_dir}")

theta_true = get_theta_true()

print(f"\ntheta_true (全14パラメータ):")
print(f"  {theta_true}")

# Load M1 and M2 estimation results
try:
    with open(run_dir / "theta_MAP_M1.json", "r") as f:
        theta_MAP_M1_data = json.load(f)
    MAP_M1 = np.array(theta_MAP_M1_data["theta_sub"])
    
    with open(run_dir / "theta_MEAN_M1.json", "r") as f:
        theta_MEAN_M1_data = json.load(f)
    MEAN_M1 = np.array(theta_MEAN_M1_data["theta_sub"])
    
    print(f"\n[OK] M1推定結果を読み込み")
except Exception as e:
    print(f"\n[ERROR] M1推定結果の読み込み失敗: {e}")
    MAP_M1 = None
    MEAN_M1 = None

try:
    with open(run_dir / "theta_MAP_M2.json", "r") as f:
        theta_MAP_M2_data = json.load(f)
    MAP_M2 = np.array(theta_MAP_M2_data["theta_sub"])
    
    with open(run_dir / "theta_MEAN_M2.json", "r") as f:
        theta_MEAN_M2_data = json.load(f)
    MEAN_M2 = np.array(theta_MEAN_M2_data["theta_sub"])
    
    print(f"  [OK] M2推定結果を読み込み")
except Exception as e:
    print(f"  [ERROR] M2推定結果の読み込み失敗: {e}")
    MAP_M2 = None
    MEAN_M2 = None

if MAP_M1 is not None and MAP_M2 is not None:
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
    
    print(f"\n{'='*80}")
    print("M2推定誤差の分析")
    print(f"{'='*80}")
    
    theta_true_M2 = theta_true[MODEL_CONFIGS["M2"]["active_indices"]]
    
    print(f"\ntheta_true (M2): {theta_true_M2}")
    print(f"MAP_M2: {MAP_M2}")
    print(f"MEAN_M2: {MEAN_M2}")
    
    error_MAP_M2 = MAP_M2 - theta_true_M2
    error_MEAN_M2 = MEAN_M2 - theta_true_M2
    
    print(f"\n推定誤差 (MAP): {error_MAP_M2}")
    print(f"推定誤差 (MEAN): {error_MEAN_M2}")
    
    rmse_MAP_M2 = np.sqrt(np.mean(error_MAP_M2**2))
    rmse_MEAN_M2 = np.sqrt(np.mean(error_MEAN_M2**2))
    
    print(f"\nRMSE (MAP): {rmse_MAP_M2:.8f}")
    print(f"RMSE (MEAN): {rmse_MEAN_M2:.8f}")
    
    max_error_MAP_M2 = np.max(np.abs(error_MAP_M2))
    max_error_MEAN_M2 = np.max(np.abs(error_MEAN_M2))
    
    print(f"最大絶対誤差 (MAP): {max_error_MAP_M2:.8f}")
    print(f"最大絶対誤差 (MEAN): {max_error_MEAN_M2:.8f}")
    
    print(f"\n{'='*80}")
    print("M3への影響分析")
    print(f"{'='*80}")
    
    # Reconstruct theta_base_M3 (as used in M3 estimation)
    prior_mean = 1.5  # (0.0 + 3.0) / 2.0
    theta_base_M3_MAP = np.full(14, prior_mean)
    theta_base_M3_MAP[0:5] = MAP_M1
    theta_base_M3_MAP[5:10] = MAP_M2
    
    theta_base_M3_MEAN = np.full(14, prior_mean)
    theta_base_M3_MEAN[0:5] = MEAN_M1
    theta_base_M3_MEAN[5:10] = MEAN_M2
    
    print(f"\ntheta_base_M3 (MAP使用):")
    print(f"  M1部分: {theta_base_M3_MAP[0:5]}")
    print(f"  M2部分: {theta_base_M3_MAP[5:10]}")
    print(f"  M3部分: {theta_base_M3_MAP[10:14]} (prior mean)")
    
    print(f"\ntheta_true:")
    print(f"  M1部分: {theta_true[0:5]}")
    print(f"  M2部分: {theta_true[5:10]}")
    print(f"  M3部分: {theta_true[10:14]}")
    
    error_base_M3_MAP = theta_base_M3_MAP - theta_true
    error_base_M3_MEAN = theta_base_M3_MEAN - theta_true
    
    print(f"\nM3推定時のベース誤差 (MAP使用):")
    print(f"  M1部分誤差: {error_base_M3_MAP[0:5]}")
    print(f"  M2部分誤差: {error_base_M3_MAP[5:10]}")
    print(f"  M3部分誤差: {error_base_M3_MAP[10:14]} (prior mean - true)")
    
    rmse_base_M3_MAP = np.sqrt(np.mean(error_base_M3_MAP**2))
    rmse_base_M3_MEAN = np.sqrt(np.mean(error_base_M3_MEAN**2))
    
    print(f"\nベースRMSE (MAP使用): {rmse_base_M3_MAP:.8f}")
    print(f"ベースRMSE (MEAN使用): {rmse_base_M3_MEAN:.8f}")
    
    # Check if M1/M2 errors are significant
    sigma_obs = 0.01
    print(f"\n{'='*80}")
    print("判定")
    print(f"{'='*80}")
    
    if max_error_MAP_M1 > 0.1 or max_error_MAP_M2 > 0.1:
        print(f"  [WARNING] M1/M2の推定誤差が大きい（>0.1）")
        print(f"    これがM3の適合性低下の原因の可能性が高い")
    else:
        print(f"  [OK] M1/M2の推定誤差は小さい（<0.1）")
    
    if rmse_base_M3_MAP > 0.05:
        print(f"  [WARNING] M3推定時のベース誤差が大きい（RMSE > 0.05）")
        print(f"    M1/M2の推定誤差がM3に伝播している可能性が高い")
    else:
        print(f"  [OK] M3推定時のベース誤差は小さい（RMSE < 0.05）")
    
    print(f"\n推奨される対策:")
    print(f"  1. M1とM2の推定精度を向上させる（より多くのパーティクル、より多くのステージ）")
    print(f"  2. M3推定時にM1/M2の不確実性を考慮する（階層的ベイズ）")
    print(f"  3. M3データ生成時にM1/M2の推定結果を使用する（一貫性の確保）")

else:
    print(f"\n[WARNING] M1またはM2の推定結果が読み込めませんでした")
    print(f"  実行が完了していない可能性があります")

print(f"\n{'='*80}")
print("確認完了")
print(f"{'='*80}")
