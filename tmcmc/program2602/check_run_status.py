#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実行状況を確認するスクリプト
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from datetime import datetime

# Windowsでの文字エンコーディング問題を回避
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_run_status(run_id: str):
    """実行状況を確認"""
    base_dir = Path(__file__).parent
    runs_dir = base_dir / "_runs"
    run_dir = runs_dir / run_id
    
    if not run_dir.exists():
        print(f"実行ディレクトリが見つかりません: {run_dir}")
        return False
    
    # REPORT.mdが存在すれば完了
    report_file = run_dir / "REPORT.md"
    if report_file.exists():
        print(f"✓ 実行完了: {run_id}")
        print(f"  レポート: {report_file}")
        return True
    
    # run.logの最後の行を確認
    log_file = run_dir / "run.log"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print(f"実行中: {run_id}")
                    print(f"  最新ログ: {last_line[:100]}...")
                    print(f"  ログファイル: {log_file}")
        except Exception as e:
            print(f"ログファイルの読み込みエラー: {e}")
    
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = "m1_test_100"
    
    is_complete = check_run_status(run_id)
    sys.exit(0 if is_complete else 1)


