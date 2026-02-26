#!/usr/bin/env python3
"""
TMCMC並列化の統合テスト

実際のTMCMCコードで並列化が正しく動作するか確認します。
小さな設定で実行し、並列化のログを確認します。
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tmcmc import run_TMCMC
from config import DebugConfig, DebugLevel, TMCMC_DEFAULTS
from debug import DebugLogger


def simple_log_likelihood(theta: np.ndarray) -> float:
    """
    簡単な尤度関数（テスト用）
    計算時間をシミュレート
    """
    # 計算時間をシミュレート（0.001秒）
    time.sleep(0.001)
    # 簡単な尤度計算（多峰性のある関数）
    return -0.5 * np.sum((theta - 1.0) ** 2) - 0.1 * np.sum(theta**4)


def test_tmcmc_parallelization():
    """TMCMC並列化の統合テスト"""
    print("=" * 80)
    print("TMCMC並列化統合テスト開始")
    print("=" * 80)

    # テスト設定（小さな設定で高速に実行）
    n_particles = 50
    n_stages = 3
    n_params = 3
    prior_bounds = [(0.0, 3.0)] * n_params

    print("\nテスト設定:")
    print(f"  パーティクル数: {n_particles}")
    print(f"  ステージ数: {n_stages}")
    print(f"  パラメータ数: {n_params}")
    print()

    # 並列化設定
    n_jobs_list = [1, 6]
    results = {}

    for n_jobs in n_jobs_list:
        print(f"\n{'='*80}")
        print(f"テスト: n_jobs={n_jobs}")
        print(f"{'='*80}")

        # Debug logger設定
        debug_config = DebugConfig(level=DebugLevel.MINIMAL)
        debug_logger = DebugLogger(debug_config)

        # TMCMC実行
        start_time = time.perf_counter()
        result = run_TMCMC(
            log_likelihood=simple_log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=n_particles,
            n_stages=n_stages,
            target_ess_ratio=0.5,
            min_delta_beta=0.1,
            max_delta_beta=0.3,
            seed=42,
            model_name=f"Test_njobs{n_jobs}",
            n_jobs=n_jobs,
            use_threads=False,
            debug_logger=debug_logger,
        )
        elapsed_time = time.perf_counter() - start_time

        results[n_jobs] = {
            "time": elapsed_time,
            "samples": result.samples,
            "logL": result.logL_values,
            "beta_schedule": result.beta_schedule,
            "converged": result.converged,
        }

        print(f"\n実行時間: {elapsed_time:.2f}秒")
        print(f"収束: {result.converged}")
        print(f"βスケジュール: {result.beta_schedule}")
        print(f"サンプル数: {len(result.samples)}")
        print(f"平均logL: {np.mean(result.logL_values):.2f}")

    # 比較
    print(f"\n{'='*80}")
    print("結果比較")
    print(f"{'='*80}")

    if len(results) >= 2:
        time_1 = results[1]["time"]
        time_6 = results[6]["time"]
        speedup = time_1 / time_6 if time_6 > 0 else 0

        print(f"n_jobs=1 の実行時間: {time_1:.2f}秒")
        print(f"n_jobs=6 の実行時間: {time_6:.2f}秒")
        print(f"高速化率: {speedup:.2f}x")

        if speedup > 1.5:
            print("✅ 並列化が効果的に動作しています")
        elif speedup > 1.1:
            print("⚠️  並列化は動作していますが、効果が限定的です")
        else:
            print("❌ 並列化が効果的に動作していません")

    # 結果の一貫性確認
    print(f"\n{'='*80}")
    print("結果の一貫性確認")
    print(f"{'='*80}")

    if len(results) >= 2:
        samples_1 = results[1]["samples"]
        samples_6 = results[6]["samples"]

        # 統計量の比較
        mean_1 = np.mean(samples_1, axis=0)
        mean_6 = np.mean(samples_6, axis=0)
        std_1 = np.std(samples_1, axis=0)
        std_6 = np.std(samples_6, axis=0)

        mean_diff = np.max(np.abs(mean_1 - mean_6))
        std_diff = np.max(np.abs(std_1 - std_6))

        print(f"平均値の最大差分: {mean_diff:.4f}")
        print(f"標準偏差の最大差分: {std_diff:.4f}")

        if mean_diff < 0.1 and std_diff < 0.1:
            print("✅ 結果は一貫しています（並列化による影響は小さい）")
        else:
            print("⚠️  結果に差があります（並列化による影響の可能性）")

    print("\nテスト完了")
    return results


if __name__ == "__main__":
    try:
        results = test_tmcmc_parallelization()
        print("\n✅ すべてのテストが完了しました")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
