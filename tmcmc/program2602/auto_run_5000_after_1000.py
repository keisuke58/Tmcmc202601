#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1000パーティクル実行完了後、自動的に5000パーティクル実行を開始するスクリプト

使用方法:
    python auto_run_5000_after_1000.py --wait-for-run-id m1_1000_20260118_083726
    python auto_run_5000_after_1000.py --wait-for-latest
"""

from __future__ import annotations

import sys
import io
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Windowsでの文字エンコーディング問題を回避
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def check_run_complete(run_id: str, runs_root: Path) -> bool:
    """
    実行が完了したかチェック

    Returns:
        True: 完了（REPORT.mdが存在）
        False: 実行中または未完了
    """
    run_dir = runs_root / run_id
    report_file = run_dir / "REPORT.md"

    if report_file.exists():
        return True
    return False


def wait_for_completion(run_id: str, runs_root: Path, check_interval: int = 60) -> bool:
    """
    実行完了を待つ

    Args:
        run_id: 待つ実行ID
        runs_root: 実行ディレクトリのルート
        check_interval: チェック間隔（秒）

    Returns:
        True: 完了
        False: タイムアウトまたはエラー
    """
    print(f"⏳ 実行完了を待機中: {run_id}")
    print(f"   チェック間隔: {check_interval}秒")
    print(f"   実行ディレクトリ: {runs_root / run_id}")
    print()

    start_time = time.time()
    last_check_time = 0

    while True:
        current_time = time.time()
        elapsed_min = (current_time - start_time) / 60

        # 定期的にチェック（check_interval秒ごと）
        if current_time - last_check_time >= check_interval:
            if check_run_complete(run_id, runs_root):
                elapsed_total = (current_time - start_time) / 60
                print("✅ 実行完了を検知しました！")
                print(f"   待機時間: {elapsed_total:.1f}分")
                return True

            # 進捗表示
            run_dir = runs_root / run_id
            run_log = run_dir / "run.log"
            if run_log.exists():
                try:
                    # ログファイルの最終更新時刻を確認
                    mtime = run_log.stat().st_mtime
                    last_update_min = (current_time - mtime) / 60

                    if last_update_min < 5:
                        print(
                            f"   [{datetime.now().strftime('%H:%M:%S')}] 実行中... (経過: {elapsed_min:.1f}分, 最終更新: {last_update_min:.1f}分前)"
                        )
                    else:
                        print(
                            f"   [{datetime.now().strftime('%H:%M:%S')}] 実行中... (経過: {elapsed_min:.1f}分, 最終更新: {last_update_min:.1f}分前) ⚠️ 更新が遅い"
                        )
                except Exception:
                    pass

            last_check_time = current_time

        time.sleep(5)  # 5秒ごとにチェック


def review_1000_results(run_id: str, runs_root: Path) -> dict:
    """
    1000パーティクル実行の結果をレビュー

    Returns:
        レビュー結果の辞書
    """
    run_dir = runs_root / run_id
    report_file = run_dir / "REPORT.md"
    metrics_file = run_dir / "metrics.json"
    config_file = run_dir / "config.json"

    review = {
        "run_id": run_id,
        "report_exists": report_file.exists(),
        "metrics_exists": metrics_file.exists(),
        "status": "UNKNOWN",
        "ess_min": None,
        "ess_mean": None,
        "rmse": None,
        "mae": None,
        "max_abs": None,
        "converged": None,
        "accept_rate_mean": None,
        "beta_final": None,
        "beta_stages": None,
        "recommend_5000": True,
        "issues": [],
        "improvements": [],
    }

    if not report_file.exists():
        review["issues"].append("REPORT.mdが見つかりません")
        return review

    # REPORT.mdを読み込んで解析
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        import re

        # ステータスを抽出
        if "status**: **FAIL" in content:
            review["status"] = "FAIL"
        elif "status**: **WARN" in content:
            review["status"] = "WARN"
        elif "status**: **PASS" in content:
            review["status"] = "PASS"

        # ESS_minを抽出
        ess_match = re.search(r"ESS_min.*?(\d+\.?\d*)", content)
        if ess_match:
            review["ess_min"] = float(ess_match.group(1))

        # RMSEを抽出
        rmse_match = re.search(r"RMSE_total.*?\(MAP\).*?(\d+\.?\d*)", content)
        if rmse_match:
            review["rmse"] = float(rmse_match.group(1))

        # MAEを抽出
        mae_match = re.search(r"MAE_total.*?\(MAP\).*?(\d+\.?\d*)", content)
        if mae_match:
            review["mae"] = float(mae_match.group(1))

        # max_absを抽出
        max_abs_match = re.search(r"max_abs.*?\(MAP\).*?(\d+\.?\d*)", content)
        if max_abs_match:
            review["max_abs"] = float(max_abs_match.group(1))

        # accept_rate_meanを抽出
        acc_match = re.search(r"accept_rate_mean.*?(\d+\.?\d*)", content)
        if acc_match:
            review["accept_rate_mean"] = float(acc_match.group(1))

        # beta_finalを抽出
        beta_match = re.search(r"beta_final.*?(\d+\.?\d*)", content)
        if beta_match:
            review["beta_final"] = float(beta_match.group(1))

        # beta_stagesを抽出
        stages_match = re.search(r"beta_stages.*?(\d+)", content)
        if stages_match:
            review["beta_stages"] = int(stages_match.group(1))

        # 収束を確認
        if "converged_chains" in content:
            review["converged"] = True

    except Exception as e:
        review["issues"].append(f"REPORT.mdの解析エラー: {e}")

    # metrics.jsonも読み込む
    if metrics_file.exists():
        try:
            import json

            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            if "timing" in metrics:
                review["execution_time_min"] = metrics["timing"].get("total_time_min")

            if "convergence" in metrics:
                conv = metrics["convergence"]
                if "M1" in conv:
                    review["converged_chains"] = conv["M1"].get("converged_chains", 0)
                    review["n_chains"] = conv["M1"].get("n_chains", 1)

        except Exception as e:
            review["issues"].append(f"metrics.jsonの解析エラー: {e}")

    # 推奨判定と改善提案
    if review["ess_min"] is not None:
        if review["ess_min"] < 100:
            review["recommend_5000"] = True
            review["issues"].append(
                f"ESS_min ({review['ess_min']:.1f}) < 100 - 5000パーティクルで改善が期待されます"
            )
            review["improvements"].append(
                "ESS_minを100以上にするため、パーティクル数を5000に増やす"
            )
        elif review["ess_min"] < 300:
            review["recommend_5000"] = True
            review["improvements"].append(
                f"ESS_min ({review['ess_min']:.1f}) < 300 - 5000パーティクルでESS_min ≥ 300を目標"
            )
        else:
            review["recommend_5000"] = False
            review["improvements"].append(f"ESS_min ({review['ess_min']:.1f}) ≥ 300 - 十分な品質")

    if review["rmse"] is not None and review["rmse"] > 0.05:
        review["improvements"].append(
            f"RMSE ({review['rmse']:.6f}) > 0.05 - 5000パーティクルで精度向上が期待されます"
        )

    if review["accept_rate_mean"] is not None:
        if review["accept_rate_mean"] < 0.3:
            review["issues"].append(
                f"受容率 ({review['accept_rate_mean']:.3f}) < 0.3 - 低すぎる可能性"
            )
        elif review["accept_rate_mean"] > 0.8:
            review["issues"].append(
                f"受容率 ({review['accept_rate_mean']:.3f}) > 0.8 - 高すぎる可能性（探索不足）"
            )

    return review


def save_review_report(review: dict, run_dir: Path) -> Path:
    """
    レビューレポートを保存

    Returns:
        保存されたファイルのパス
    """
    review_file = run_dir / "REVIEW_1000_particles.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# M1 1000パーティクル実行レビュー

**実行ID**: `{review['run_id']}`
**レビュー日時**: {timestamp}

---

## 📊 実行サマリー

### 設定
- **パーティクル数**: 1000
- **ステージ数**: {review.get('beta_stages', 'N/A')}
- **Mutation steps**: 5
- **チェーン数**: 1

### 結果
- **ステータス**: {review['status']}
- **実行時間**: {review.get('execution_time_min', 'N/A'):.1f}分（該当する場合）

---

## ✅ 良好な点

"""

    if review.get("ess_min") is not None and review["ess_min"] >= 100:
        content += f"- ✅ ESS_min: {review['ess_min']:.2f} ≥ 100（基準クリア）\n"

    if review.get("rmse") is not None and review["rmse"] < 0.05:
        content += f"- ✅ RMSE: {review['rmse']:.6f} < 0.05（良好な精度）\n"

    if review.get("accept_rate_mean") is not None and 0.3 <= review["accept_rate_mean"] <= 0.8:
        content += f"- ✅ 受容率: {review['accept_rate_mean']:.3f}（適切な範囲）\n"

    if review.get("beta_final") == 1.0:
        content += "- ✅ β=1.0に到達（完了）\n"

    content += "\n---\n\n## ⚠️ 問題点\n\n"

    if review["issues"]:
        for issue in review["issues"]:
            content += f"- ⚠️ {issue}\n"
    else:
        content += "- 特に問題なし\n"

    content += "\n---\n\n## 📈 詳細メトリクス\n\n"
    content += "| 指標 | 値 | 評価 |\n"
    content += "|------|-----|------|\n"

    if review.get("ess_min") is not None:
        ess_eval = (
            "✅ 良好"
            if review["ess_min"] >= 300
            else ("⚠️ やや低い" if review["ess_min"] >= 100 else "❌ 不足")
        )
        content += f"| ESS_min | {review['ess_min']:.2f} | {ess_eval} |\n"

    if review.get("rmse") is not None:
        rmse_eval = (
            "✅ 良好"
            if review["rmse"] < 0.03
            else ("⚠️ 中程度" if review["rmse"] < 0.05 else "❌ 大きい")
        )
        content += f"| RMSE_total (MAP) | {review['rmse']:.6f} | {rmse_eval} |\n"

    if review.get("mae") is not None:
        content += f"| MAE_total (MAP) | {review['mae']:.6f} | - |\n"

    if review.get("max_abs") is not None:
        content += f"| max_abs (MAP) | {review['max_abs']:.6f} | - |\n"

    if review.get("accept_rate_mean") is not None:
        acc_eval = "✅ 適切" if 0.3 <= review["accept_rate_mean"] <= 0.8 else "⚠️ 範囲外"
        content += f"| accept_rate_mean | {review['accept_rate_mean']:.3f} | {acc_eval} |\n"

    if review.get("beta_final") is not None:
        content += f"| beta_final | {review['beta_final']:.1f} | ✅ 完了 |\n"

    if review.get("beta_stages") is not None:
        content += f"| beta_stages | {review['beta_stages']} | - |\n"

    content += "\n---\n\n## 🎯 改善提案\n\n"

    if review["improvements"]:
        for improvement in review["improvements"]:
            content += f"- {improvement}\n"
    else:
        content += "- 特に改善点なし\n"

    content += "\n---\n\n## 📋 5000パーティクル実行への推奨\n\n"

    if review["recommend_5000"]:
        content += "✅ **5000パーティクル実行を推奨します**\n\n"
        content += "**理由**:\n"
        if review.get("ess_min") is not None and review["ess_min"] < 100:
            content += f"- ESS_min ({review['ess_min']:.1f}) < 100 - 改善が必要\n"
        if review.get("ess_min") is not None and review["ess_min"] < 300:
            content += f"- ESS_min ({review['ess_min']:.1f}) < 300 - より高い品質を目指す\n"
        if review.get("rmse") is not None and review["rmse"] > 0.03:
            content += f"- RMSE ({review['rmse']:.6f}) > 0.03 - 精度向上が期待される\n"
    else:
        content += "⚠️ **5000パーティクル実行は任意です**\n\n"
        content += "**理由**:\n"
        content += "- 現在の結果で十分な品質が得られています\n"
        content += "- より高い精度が必要な場合のみ5000パーティクル実行を検討してください\n"

    content += "\n---\n\n"
    content += "**レビュー作成者**: 自動レビューシステム\n"
    content += f"**レビュー日時**: {timestamp}\n"

    try:
        with open(review_file, "w", encoding="utf-8") as f:
            f.write(content)
        return review_file
    except Exception as e:
        print(f"⚠️ レビューレポートの保存エラー: {e}")
        return None


def check_target_achieved(review: Dict, target_criteria: Dict) -> Tuple[bool, List[str]]:
    """
    目標精度が達成されたかチェック

    Args:
        review: レビュー結果
        target_criteria: 目標基準

    Returns:
        (達成したか, 未達成項目のリスト)
    """
    achieved = True
    unmet = []

    # ESS_minのチェック
    if "ess_min" in target_criteria:
        target_ess = target_criteria["ess_min"]
        if review.get("ess_min") is None:
            achieved = False
            unmet.append("ESS_min: データなし")
        elif review["ess_min"] < target_ess:
            achieved = False
            unmet.append(f"ESS_min: {review['ess_min']:.2f} < {target_ess} (目標)")

    # RMSEのチェック
    if "rmse_max" in target_criteria:
        target_rmse = target_criteria["rmse_max"]
        if review.get("rmse") is None:
            achieved = False
            unmet.append("RMSE: データなし")
        elif review["rmse"] > target_rmse:
            achieved = False
            unmet.append(f"RMSE: {review['rmse']:.6f} > {target_rmse} (目標)")

    # ステータスのチェック
    if "status" in target_criteria:
        target_status = target_criteria["status"]
        if review.get("status") != target_status:
            achieved = False
            unmet.append(
                f"ステータス: {review.get('status', 'UNKNOWN')} != {target_status} (目標: {target_status})"
            )

    # 収束のチェック
    if target_criteria.get("require_converged", False):
        if not review.get("converged", False):
            achieved = False
            unmet.append("収束: 未収束")

    return achieved, unmet


def determine_next_config(review: Dict, iteration: int, max_iterations: int) -> Dict:
    """
    次の実行設定を決定（設定の微調整）

    微調整のロジック:
    1. 反復回数に応じた基本設定の段階的増加
    2. 現在の結果（ESS_min, RMSE）に基づく適応的調整
    3. 問題点に応じた個別調整

    Args:
        review: 現在のレビュー結果
        iteration: 現在の反復回数
        max_iterations: 最大反復回数

    Returns:
        次の実行設定の辞書
    """
    # 基本設定（反復回数に応じて段階的に増加）
    base_configs = {
        1: {"n_particles": 5000, "n_stages": 30, "n_mutation_steps": 5, "n_chains": 1},
        2: {"n_particles": 5000, "n_stages": 40, "n_mutation_steps": 7, "n_chains": 1},
        3: {"n_particles": 8000, "n_stages": 50, "n_mutation_steps": 10, "n_chains": 3},
    }

    # 基本設定を取得（3回目以降は3回目の設定を使用）
    base_config = base_configs.get(min(iteration, 3), base_configs[3]).copy()

    config = {
        "n_particles": base_config["n_particles"],
        "n_stages": base_config["n_stages"],
        "n_mutation_steps": base_config["n_mutation_steps"],
        "n_chains": base_config["n_chains"],
        "max_delta_beta": 0.05,  # 固定（最適化済み）
        "target_ess_ratio": 0.5,  # 固定（適切な値）
    }

    # ===== 微調整1: ESS_minが低い場合 =====
    if review.get("ess_min") is not None:
        if review["ess_min"] < 100:
            # ESS_minが非常に低い → パーティクル数を大幅に増やす
            config["n_particles"] = max(config["n_particles"], 10000)
            config["n_stages"] = max(config["n_stages"], 50)
            print(
                f"  🔧 微調整: ESS_min ({review['ess_min']:.1f}) < 100 → パーティクル数{config['n_particles']}, ステージ数{config['n_stages']}"
            )
        elif review["ess_min"] < 200:
            # ESS_minがやや低い → パーティクル数を増やす
            config["n_particles"] = max(config["n_particles"], 8000)
            config["n_stages"] = max(config["n_stages"], 40)
            print(
                f"  🔧 微調整: ESS_min ({review['ess_min']:.1f}) < 200 → パーティクル数{config['n_particles']}, ステージ数{config['n_stages']}"
            )
        elif review["ess_min"] < 250:
            # ESS_minが目標に近い → ステージ数を少し増やす
            config["n_stages"] = max(config["n_stages"], 35)
            print(
                f"  🔧 微調整: ESS_min ({review['ess_min']:.1f}) < 250 → ステージ数{config['n_stages']}"
            )

    # ===== 微調整2: RMSEが高い場合 =====
    if review.get("rmse") is not None:
        if review["rmse"] > 0.05:
            # RMSEが非常に高い → ステージ数とMutation stepsを大幅に増やす
            config["n_stages"] = max(config["n_stages"], 50)
            config["n_mutation_steps"] = max(config["n_mutation_steps"], 10)
            print(
                f"  🔧 微調整: RMSE ({review['rmse']:.6f}) > 0.05 → ステージ数{config['n_stages']}, Mutation steps {config['n_mutation_steps']}"
            )
        elif review["rmse"] > 0.03:
            # RMSEがやや高い → ステージ数とMutation stepsを増やす
            config["n_stages"] = max(config["n_stages"], 40)
            config["n_mutation_steps"] = max(config["n_mutation_steps"], 7)
            print(
                f"  🔧 微調整: RMSE ({review['rmse']:.6f}) > 0.03 → ステージ数{config['n_stages']}, Mutation steps {config['n_mutation_steps']}"
            )
        elif review["rmse"] > 0.025:
            # RMSEが目標に近い → ステージ数を少し増やす
            config["n_stages"] = max(config["n_stages"], 35)
            print(
                f"  🔧 微調整: RMSE ({review['rmse']:.6f}) > 0.025 → ステージ数{config['n_stages']}"
            )

    # ===== 微調整3: 受容率が不適切な場合 =====
    if review.get("accept_rate_mean") is not None:
        if review["accept_rate_mean"] < 0.2:
            # 受容率が非常に低い → Mutation stepsを減らす（より保守的に）
            config["n_mutation_steps"] = max(3, config["n_mutation_steps"] - 2)
            print(
                f"  🔧 微調整: 受容率 ({review['accept_rate_mean']:.3f}) < 0.2 → Mutation steps {config['n_mutation_steps']}"
            )
        elif review["accept_rate_mean"] > 0.9:
            # 受容率が非常に高い → Mutation stepsを増やす（より積極的に探索）
            config["n_mutation_steps"] = min(15, config["n_mutation_steps"] + 3)
            print(
                f"  🔧 微調整: 受容率 ({review['accept_rate_mean']:.3f}) > 0.9 → Mutation steps {config['n_mutation_steps']}"
            )

    # ===== 微調整4: 収束していない場合 =====
    if not review.get("converged", False):
        # 収束していない → チェーン数を増やす（収束診断のため）
        config["n_chains"] = max(config["n_chains"], 3)
        config["n_stages"] = max(config["n_stages"], 40)
        print(
            f"  🔧 微調整: 未収束 → チェーン数{config['n_chains']}, ステージ数{config['n_stages']}"
        )

    return config


def run_5000_particles(
    runs_root: Path, base_dir: Path, config: Optional[Dict] = None
) -> Optional[str]:
    """
    5000パーティクル実行を開始

    Args:
        config: 実行設定（Noneの場合はデフォルト設定）

    Returns:
        実行ID（成功時）、None（失敗時）
    """
    if config is None:
        config = {
            "n_particles": 5000,
            "n_stages": 30,
            "n_mutation_steps": 5,
            "n_chains": 1,
            "max_delta_beta": 0.05,
            "target_ess_ratio": 0.5,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"m1_{config['n_particles']}_{timestamp}"

    print(f"\n{'='*80}")
    print("🚀 高精度実行を開始します")
    print(f"{'='*80}")
    print(f"実行ID: {run_id}")
    print("設定:")
    print(f"  - パーティクル数: {config['n_particles']}")
    print(f"  - ステージ数: {config['n_stages']}")
    print(f"  - Mutation steps: {config['n_mutation_steps']}")
    print(f"  - チェーン数: {config['n_chains']}")
    print(f"  - max_delta_beta: {config['max_delta_beta']}")
    print(f"  - target_ess_ratio: {config['target_ess_ratio']}")
    print(f"{'='*80}\n")

    # 実行コマンド
    cmd = [
        sys.executable,
        str(base_dir / "tmcmc" / "run_pipeline.py"),
        "--mode",
        "debug",
        "--models",
        "M1",
        "--n-particles",
        str(config["n_particles"]),
        "--n-stages",
        str(config["n_stages"]),
        "--n-mutation-steps",
        str(config["n_mutation_steps"]),
        "--n-chains",
        str(config["n_chains"]),
        "--max-delta-beta",
        str(config["max_delta_beta"]),
        "--target-ess-ratio",
        str(config["target_ess_ratio"]),
        "--seed",
        "42",
        "--run-id",
        run_id,
    ]

    try:
        print(f"実行コマンド: {' '.join(cmd)}")
        print()

        # バックグラウンドで実行
        process = subprocess.Popen(
            cmd,
            cwd=str(base_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        print(f"✅ 5000パーティクル実行を開始しました（PID: {process.pid}）")
        print(f"   実行ID: {run_id}")
        print(f"   ログ: {runs_root / run_id / 'run.log'}")
        print()
        print("進捗確認コマンド:")
        print(f'  cd "{base_dir}"')
        print("  python tmcmc\\check_running.py")
        print()

        return run_id

    except Exception as e:
        print(f"❌ 実行開始エラー: {e}")
        return None


def find_latest_run(runs_root: Path) -> Optional[str]:
    """最新の実行IDを取得"""
    if not runs_root.exists():
        return None

    runs = []
    for run_dir in runs_root.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("_"):
            log_file = run_dir / "run.log"
            if log_file.exists():
                mtime = log_file.stat().st_mtime
                runs.append((run_dir.name, mtime))

    if not runs:
        return None

    runs.sort(key=lambda x: x[1], reverse=True)
    return runs[0][0]


def main():
    parser = argparse.ArgumentParser(
        description="1000パーティクル実行完了後、自動的に5000パーティクル実行を開始"
    )
    parser.add_argument(
        "--wait-for-run-id",
        type=str,
        default=None,
        help="待つ実行ID（指定しない場合は最新の実行を待つ）",
    )
    parser.add_argument(
        "--wait-for-latest",
        action="store_true",
        help="最新の実行を待つ（--wait-for-run-idの代わり）",
    )
    parser.add_argument(
        "--check-interval", type=int, default=60, help="チェック間隔（秒、デフォルト: 60）"
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="レビューをスキップして即座に5000パーティクル実行を開始",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help="実行ディレクトリのルート（デフォルト: tmcmc/_runs）",
    )
    parser.add_argument(
        "--iterative", action="store_true", help="目標精度に達するまで繰り返し実行（推奨）"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3, help="最大反復回数（デフォルト: 3）"
    )
    parser.add_argument(
        "--target-ess-min", type=float, default=300.0, help="目標ESS_min（デフォルト: 300）"
    )
    parser.add_argument(
        "--target-rmse-max", type=float, default=0.02, help="目標RMSE最大値（デフォルト: 0.02）"
    )
    parser.add_argument(
        "--target-status",
        type=str,
        default="PASS",
        choices=["PASS", "WARN", "FAIL"],
        help="目標ステータス（デフォルト: PASS）",
    )

    args = parser.parse_args()

    # パスの設定
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # tmcmc_docs

    if args.runs_root:
        runs_root = Path(args.runs_root)
    else:
        runs_root = script_dir / "_runs"

    # 待つ実行IDを決定
    if args.wait_for_run_id:
        wait_run_id = args.wait_for_run_id
    elif args.wait_for_latest:
        wait_run_id = find_latest_run(runs_root)
        if wait_run_id is None:
            print("❌ 実行中の実行が見つかりません")
            return 1
        print(f"📌 最新の実行ID: {wait_run_id}")
    else:
        # デフォルト: 最新の実行を待つ
        wait_run_id = find_latest_run(runs_root)
        if wait_run_id is None:
            print("❌ 実行中の実行が見つかりません")
            return 1
        print(f"📌 最新の実行ID: {wait_run_id}")

    # 実行完了を待つ
    if not wait_for_completion(wait_run_id, runs_root, args.check_interval):
        print("❌ 実行完了の待機がタイムアウトまたはエラーが発生しました")
        return 1

    # レビュー（スキップしない場合）
    if not args.skip_review:
        print(f"\n{'='*80}")
        print("📊 1000パーティクル実行結果の自動レビュー")
        print(f"{'='*80}")

        review = review_1000_results(wait_run_id, runs_root)

        # レビューレポートを保存
        run_dir = runs_root / wait_run_id
        review_file = save_review_report(review, run_dir)

        if review_file:
            print(f"✅ レビューレポートを保存しました: {review_file.name}")

        print(f"\n実行ID: {review['run_id']}")
        print(f"ステータス: {review['status']}")

        if review.get("execution_time_min"):
            print(f"実行時間: {review['execution_time_min']:.1f}分")

        if review["ess_min"] is not None:
            print(f"ESS_min: {review['ess_min']:.2f}")
            if review["ess_min"] < 100:
                print("  ⚠️ ESS_min < 100 - 5000パーティクルで改善が期待されます")
            elif review["ess_min"] < 300:
                print("  ⚠️ ESS_min < 300 - 5000パーティクルでESS_min ≥ 300を目標")
            else:
                print("  ✅ ESS_min ≥ 300 - 十分な品質")

        if review["rmse"] is not None:
            print(f"RMSE: {review['rmse']:.6f}")
            if review["rmse"] < 0.03:
                print("  ✅ RMSE < 0.03 - 良好な精度")
            elif review["rmse"] < 0.05:
                print("  ⚠️ RMSE < 0.05 - 中程度の精度")
            else:
                print("  ⚠️ RMSE ≥ 0.05 - 改善の余地あり")

        if review.get("accept_rate_mean") is not None:
            print(f"受容率: {review['accept_rate_mean']:.3f}")
            if 0.3 <= review["accept_rate_mean"] <= 0.8:
                print("  ✅ 適切な範囲")
            else:
                print("  ⚠️ 範囲外（0.3-0.8が推奨）")

        if review.get("beta_final") == 1.0:
            print("β=1.0: ✅ 到達")

        if review["issues"]:
            print("\n⚠️ 問題点:")
            for issue in review["issues"]:
                print(f"  - {issue}")

        if review["improvements"]:
            print("\n💡 改善提案:")
            for improvement in review["improvements"]:
                print(f"  - {improvement}")

        print(f"\n{'='*80}")
        if review["recommend_5000"]:
            print("✅ 5000パーティクル実行を推奨します")
        else:
            print("⚠️ 5000パーティクル実行は任意です（現在の結果で十分な品質）")
        print(f"{'='*80}")

        if not review["recommend_5000"]:
            print("\n⚠️ 5000パーティクル実行は推奨されませんが、実行を続行しますか？")
            print("   5秒後に自動的に続行します...")
            time.sleep(5)
            print("   続行します\n")

        print()

    # 目標精度の設定
    target_criteria = {
        "ess_min": args.target_ess_min,
        "rmse_max": args.target_rmse_max,
        "status": args.target_status,
        "require_converged": True,
    }

    # 反復実行モード
    if args.iterative:
        print(f"\n{'='*80}")
        print("🔄 反復実行モード: 目標精度に達するまで繰り返します")
        print(f"{'='*80}")
        print("目標基準:")
        print(f"  - ESS_min ≥ {target_criteria['ess_min']}")
        print(f"  - RMSE ≤ {target_criteria['rmse_max']}")
        print(f"  - ステータス: {target_criteria['status']}")
        print("  - 収束: 必須")
        print(f"最大反復回数: {args.max_iterations}")
        print(f"{'='*80}\n")

        current_run_id = wait_run_id
        iteration = 0

        while iteration < args.max_iterations:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"🔄 反復 {iteration}/{args.max_iterations}")
            print(f"{'='*80}")

            # 現在の実行のレビュー
            print(f"📊 実行結果のレビュー: {current_run_id}")
            review = review_1000_results(current_run_id, runs_root)

            # レビューレポートを保存
            run_dir = runs_root / current_run_id
            review_file = save_review_report(review, run_dir)
            if review_file:
                print(f"✅ レビューレポートを保存: {review_file.name}")

            # 目標達成チェック
            achieved, unmet = check_target_achieved(review, target_criteria)

            if achieved:
                print(f"\n{'='*80}")
                print("🎉 目標精度を達成しました！")
                print(f"{'='*80}")
                print("達成項目:")
                if review.get("ess_min") is not None:
                    print(f"  ✅ ESS_min: {review['ess_min']:.2f} ≥ {target_criteria['ess_min']}")
                if review.get("rmse") is not None:
                    print(f"  ✅ RMSE: {review['rmse']:.6f} ≤ {target_criteria['rmse_max']}")
                if review.get("status") == target_criteria["status"]:
                    print(f"  ✅ ステータス: {review['status']}")
                print(f"\n反復回数: {iteration}/{args.max_iterations}")
                print(f"最終実行ID: {current_run_id}")
                return 0

            # 目標未達成
            print("\n⚠️ 目標精度に未達です")
            print("未達成項目:")
            for item in unmet:
                print(f"  - {item}")

            if iteration >= args.max_iterations:
                print(f"\n❌ 最大反復回数 ({args.max_iterations}) に達しました")
                print(f"最終実行ID: {current_run_id}")
                print("目標精度に達していませんが、実行を終了します")
                return 1

            # 次の実行設定を決定（微調整を含む）
            print("\n🔧 設定の微調整を実行中...")
            next_config = determine_next_config(review, iteration, args.max_iterations)
            print("\n📋 次の実行設定（微調整後）:")
            print(f"  - パーティクル数: {next_config['n_particles']}")
            print(f"  - ステージ数: {next_config['n_stages']}")
            print(f"  - Mutation steps: {next_config['n_mutation_steps']}")
            print(f"  - チェーン数: {next_config['n_chains']}")
            print(f"  - max_delta_beta: {next_config['max_delta_beta']}")
            print(f"  - target_ess_ratio: {next_config['target_ess_ratio']}")

            # データ保存の確認（前回の実行）
            print(f"\n💾 データ保存の確認: {current_run_id}")
            run_dir_prev = runs_root / current_run_id
            saved_files = []

            if (run_dir_prev / "REPORT.md").exists():
                saved_files.append("✅ REPORT.md")
            if (run_dir_prev / "metrics.json").exists():
                saved_files.append("✅ metrics.json")
            if (run_dir_prev / "results_MAP_linearization.npz").exists():
                saved_files.append("✅ results_MAP_linearization.npz")
            if (run_dir_prev / "REVIEW_1000_particles.md").exists():
                saved_files.append("✅ REVIEW_1000_particles.md")
            if (run_dir_prev / "figures").exists():
                fig_count = len(list((run_dir_prev / "figures").glob("*.png")))
                saved_files.append(f"✅ figures/ ({fig_count}図)")

            if saved_files:
                print("  保存済みファイル:")
                for f in saved_files:
                    print(f"    {f}")
            else:
                print("  ⚠️ 保存されたファイルが見つかりません")

            # 設定変更履歴を保存
            config_history_file = run_dir_prev / "config_history.json"
            try:
                import json

                history = {
                    "iteration": iteration,
                    "previous_config": {
                        "n_particles": review.get("n_particles", "unknown"),
                        "n_stages": review.get("beta_stages", "unknown"),
                    },
                    "next_config": next_config,
                    "review_summary": {
                        "ess_min": review.get("ess_min"),
                        "rmse": review.get("rmse"),
                        "status": review.get("status"),
                    },
                    "adjustment_reason": "自動微調整",
                }
                with open(config_history_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                print("  ✅ 設定変更履歴を保存: config_history.json")
            except Exception as e:
                print(f"  ⚠️ 設定変更履歴の保存エラー: {e}")

            # 次の実行を開始
            print("\n⏳ 次の実行を開始します...")
            next_run_id = run_5000_particles(runs_root, base_dir, next_config)

            if not next_run_id:
                print("❌ 実行の開始に失敗しました")
                return 1

            # 完了を待つ
            print(f"\n⏳ 実行完了を待機中: {next_run_id}")
            if not wait_for_completion(next_run_id, runs_root, args.check_interval):
                print("❌ 実行完了の待機がタイムアウトまたはエラーが発生しました")
                return 1

            # 完了後のデータ保存確認
            print(f"\n💾 実行完了後のデータ保存確認: {next_run_id}")
            run_dir_next = runs_root / next_run_id
            time.sleep(5)  # ファイル保存の完了を待つ

            saved_files_next = []
            if (run_dir_next / "REPORT.md").exists():
                saved_files_next.append("✅ REPORT.md")
            if (run_dir_next / "metrics.json").exists():
                saved_files_next.append("✅ metrics.json")
            if (run_dir_next / "results_MAP_linearization.npz").exists():
                saved_files_next.append("✅ results_MAP_linearization.npz")

            if saved_files_next:
                print("  保存済みファイル:")
                for f in saved_files_next:
                    print(f"    {f}")
            else:
                print("  ⚠️ 主要ファイルがまだ保存されていません（数秒後に再確認）")

            current_run_id = next_run_id

        # 最大反復回数に達したが目標未達成
        print(f"\n❌ 最大反復回数 ({args.max_iterations}) に達しましたが、目標精度に達していません")
        return 1

    else:
        # 通常モード: 1回だけ実行
        run_id_5000 = run_5000_particles(runs_root, base_dir)

        if run_id_5000:
            print("✅ 5000パーティクル実行を開始しました")
            print(f"   実行ID: {run_id_5000}")
            return 0
        else:
            print("❌ 5000パーティクル実行の開始に失敗しました")
            return 1


if __name__ == "__main__":
    sys.exit(main())
