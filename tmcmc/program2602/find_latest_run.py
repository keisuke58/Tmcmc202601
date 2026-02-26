#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最新の実行IDとログファイルの場所を表示
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from datetime import datetime

# Windowsでの文字エンコーディング問題を回避
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def find_latest_runs(n=5):
    """最新の実行ディレクトリを探す"""
    base_dir = Path(__file__).parent
    runs_dir = base_dir / "_runs"

    if not runs_dir.exists():
        print("実行ディレクトリが見つかりません")
        return

    # _で始まらないディレクトリを探す
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("_"):
            log_file = run_dir / "run.log"
            if log_file.exists():
                mtime = log_file.stat().st_mtime
                runs.append((run_dir.name, mtime, log_file))

    if not runs:
        print("実行ログが見つかりません")
        return

    # 更新日時でソート
    runs.sort(key=lambda x: x[1], reverse=True)

    print("=" * 80)
    print("最新の実行IDとログファイル")
    print("=" * 80)
    print()

    for i, (run_id, mtime, log_file) in enumerate(runs[:n], 1):
        dt = datetime.fromtimestamp(mtime)
        print(f"{i}. Run ID: {run_id}")
        print(f"   更新日時: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   ログファイル: {log_file}")
        print()

    # 最新の実行のログファイルのパスを表示
    latest_run_id, latest_mtime, latest_log = runs[0]
    print("=" * 80)
    print(f"最新の実行: {latest_run_id}")
    print("ログファイルのパス:")
    print(f"  {latest_log}")
    print("=" * 80)
    print()
    print("ログファイルを確認するコマンド:")
    print(f'  type "{latest_log}"')
    print("  または")
    print(f'  notepad "{latest_log}"')


if __name__ == "__main__":
    find_latest_runs()
