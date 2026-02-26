#!/usr/bin/env python3
"""
並列化の動作確認テスト

このスクリプトは、並列化が正しく動作しているかを確認します。
- evaluate_particles_parallel()の動作確認
- 並列化による高速化の確認
- ログ出力の確認
"""

import sys
import time
import numpy as np
import multiprocessing as mp
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tmcmc import evaluate_particles_parallel


def simple_log_likelihood(theta: np.ndarray) -> float:
    """
    簡単な尤度関数（テスト用）
    計算時間をシミュレートするために少し待機
    """
    # 計算時間をシミュレート（0.01秒）
    time.sleep(0.01)
    # 簡単な尤度計算
    return -0.5 * np.sum(theta**2)


def test_parallelization():
    """並列化の動作確認テスト"""
    print("=" * 80)
    print("並列化テスト開始")
    print("=" * 80)

    # テストパラメータ
    n_particles = 100
    n_params = 5
    n_jobs_list = [1, 4, 6]

    # テストデータ生成
    rng = np.random.default_rng(42)
    theta = rng.normal(0, 1, (n_particles, n_params))

    print("\nテスト設定:")
    print(f"  パーティクル数: {n_particles}")
    print(f"  パラメータ数: {n_params}")
    print(f"  CPUコア数: {mp.cpu_count()}")
    print()

    results = {}

    for n_jobs in n_jobs_list:
        print(f"\n{'='*80}")
        print(f"テスト: n_jobs={n_jobs}")
        print(f"{'='*80}")

        # 並列評価
        start_time = time.perf_counter()
        logL_parallel = evaluate_particles_parallel(
            simple_log_likelihood, theta, n_jobs=n_jobs, use_threads=False
        )
        parallel_time = time.perf_counter() - start_time

        # 逐次評価（比較用）
        start_time = time.perf_counter()
        logL_sequential = np.array([simple_log_likelihood(t) for t in theta])
        sequential_time = time.perf_counter() - start_time

        # 結果の比較
        max_diff = np.max(np.abs(logL_parallel - logL_sequential))
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        results[n_jobs] = {
            "parallel_time": parallel_time,
            "sequential_time": sequential_time,
            "speedup": speedup,
            "max_diff": max_diff,
        }

        print(f"  並列評価時間: {parallel_time:.2f}秒")
        print(f"  逐次評価時間: {sequential_time:.2f}秒")
        print(f"  高速化率: {speedup:.2f}x")
        print(f"  結果の最大差分: {max_diff:.2e}")

        if max_diff > 1e-10:
            print("  ⚠️  警告: 結果に差分があります（並列化の問題の可能性）")
        else:
            print("  ✅ 結果は一致しています")

    # まとめ
    print(f"\n{'='*80}")
    print("テスト結果まとめ")
    print(f"{'='*80}")
    print(f"{'n_jobs':<10} {'並列時間':<12} {'逐次時間':<12} {'高速化率':<12} {'状態':<10}")
    print("-" * 80)

    for n_jobs in n_jobs_list:
        r = results[n_jobs]
        status = "✅ OK" if r["max_diff"] < 1e-10 else "⚠️ 差分あり"
        print(
            f"{n_jobs:<10} {r['parallel_time']:<12.2f} {r['sequential_time']:<12.2f} {r['speedup']:<12.2f} {status:<10}"
        )

    # 推奨事項
    print(f"\n{'='*80}")
    print("推奨事項")
    print(f"{'='*80}")

    best_n_jobs = max(n_jobs_list, key=lambda x: results[x]["speedup"])
    best_speedup = results[best_n_jobs]["speedup"]

    if best_speedup > 1.5:
        print("✅ 並列化が効果的に動作しています")
        print(f"   推奨 n_jobs: {best_n_jobs} (高速化率: {best_speedup:.2f}x)")
    elif best_speedup > 1.1:
        print("⚠️  並列化は動作していますが、効果が限定的です")
        print(f"   推奨 n_jobs: {best_n_jobs} (高速化率: {best_speedup:.2f}x)")
        print("   原因: 計算時間が短すぎる、またはオーバーヘッドが大きい可能性")
    else:
        print("❌ 並列化が効果的に動作していません")
        print("   原因を調査してください:")
        print("   1. ProcessPoolExecutorが正しく動作しているか")
        print("   2. 関数がpickle可能か")
        print("   3. 計算時間が十分に長いか")

    print("\nテスト完了")
    return results


if __name__ == "__main__":
    try:
        results = test_parallelization()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
