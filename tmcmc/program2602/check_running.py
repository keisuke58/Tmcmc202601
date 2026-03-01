#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実行中の計算を確認するスクリプト
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from datetime import datetime, timedelta

# Windowsでの文字エンコーディング問題を回避
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def check_running():
    """実行中の計算を確認"""
    base_dir = Path(__file__).parent
    runs_dir = base_dir / "_runs"

    if not runs_dir.exists():
        print("実行ディレクトリが見つかりません")
        return

    print("=" * 80)
    print("実行状況の確認")
    print("=" * 80)
    print()

    now = datetime.now()

    # 最近2時間以内に更新された実行を探す
    recent_runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("_"):
            log_file = run_dir / "run.log"
            if log_file.exists():
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                time_diff = now - mtime

                # 2時間以内に更新されたもの
                if time_diff < timedelta(hours=2):
                    recent_runs.append((run_dir.name, mtime, time_diff, log_file))

    if not recent_runs:
        print("最近の実行が見つかりません（2時間以内に更新されたログなし）")
        print()
        print("最新の実行を確認:")
        latest_runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("_"):
                log_file = run_dir / "run.log"
                if log_file.exists():
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    latest_runs.append((run_dir.name, mtime, log_file))

        if latest_runs:
            latest_runs.sort(key=lambda x: x[1], reverse=True)
            run_id, mtime, log_file = latest_runs[0]
            time_diff = now - mtime
            print(f"  Run ID: {run_id}")
            print(
                f"  最終更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({time_diff.total_seconds()/3600:.1f}時間前)"
            )
            print(f"  ログファイル: {log_file}")

            # ログの最後の行を確認
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"  最新ログ: {last_line[:100]}...")
            except OSError:
                pass
        return

    # 最近の実行を表示
    recent_runs.sort(key=lambda x: x[1], reverse=True)

    for run_id, mtime, time_diff, log_file in recent_runs:
        print(f"Run ID: {run_id}")
        print(f"  最終更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  経過時間: {time_diff.total_seconds()/60:.1f}分前")

        # ログの最後の行を確認
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print(f"  最新ログ: {last_line[:120]}")

                    # Stage情報を確認
                    stage_lines = [l for l in lines[-50:] if "Stage" in l]
                    if stage_lines:
                        last_stage = stage_lines[-1]
                        print(f"  最新ステージ: {last_stage.strip()[:100]}")
        except Exception as e:
            print(f"  ログ読み込みエラー: {e}")

        print()

    # 実行中かどうかの判断
    if recent_runs:
        latest_run_id, latest_mtime, latest_time_diff, latest_log = recent_runs[0]

        # 5分以内に更新されていれば実行中と判断
        if latest_time_diff < timedelta(minutes=5):
            print("=" * 80)
            print("✓ 実行中と判断されます（5分以内にログが更新されています）")
            print(f"  Run ID: {latest_run_id}")
            print("=" * 80)
        elif latest_time_diff < timedelta(minutes=30):
            print("=" * 80)
            print("⚠ 実行中かもしれません（30分以内にログが更新されています）")
            print(f"  Run ID: {latest_run_id}")
            print("=" * 80)
        else:
            print("=" * 80)
            print("✗ 実行が停止している可能性があります（30分以上ログが更新されていません）")
            print(f"  Run ID: {latest_run_id}")
            print("=" * 80)


if __name__ == "__main__":
    check_running()
