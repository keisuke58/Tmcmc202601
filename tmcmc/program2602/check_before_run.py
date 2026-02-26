#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実行前チェックスクリプト
M1計算を実行する前に、必要なファイルや依存関係を確認します。
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

# Windowsでの文字エンコーディング問題を回避
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def check_imports():
    """必要なモジュールのインポートを確認"""
    print("=" * 80)
    print("依存関係のチェック")
    print("=" * 80)

    required_modules = [
        "numpy",
        "scipy",
        "matplotlib",
    ]

    optional_modules = [
        "numba",
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}: OK")
        except ImportError:
            print(f"✗ {module}: 見つかりません")
            missing.append(module)

    for module in optional_modules:
        try:
            __import__(module)
            print(f"✓ {module}: OK (オプション)")
        except ImportError:
            print(f"⚠ {module}: 見つかりません (オプション、パフォーマンスに影響)")

    if missing:
        print(f"\nエラー: 以下のモジュールが不足しています: {', '.join(missing)}")
        return False

    return True


def check_files():
    """必要なファイルの存在を確認"""
    print("\n" + "=" * 80)
    print("ファイルのチェック")
    print("=" * 80)

    base_dir = Path(__file__).parent
    required_files = [
        "case2_tmcmc_linearization.py",
        "run_pipeline.py",
        "improved1207_paper_jit.py",
        "config.py",
        "main/case2_main.py",
    ]

    missing = []
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}: 存在します")
        else:
            print(f"✗ {file_path}: 見つかりません")
            missing.append(file_path)

    if missing:
        print(f"\nエラー: 以下のファイルが不足しています: {', '.join(missing)}")
        return False

    return True


def check_output_directory():
    """出力ディレクトリの確認"""
    print("\n" + "=" * 80)
    print("出力ディレクトリのチェック")
    print("=" * 80)

    base_dir = Path(__file__).parent
    runs_dir = base_dir / "_runs"

    if runs_dir.exists():
        print(f"✓ 出力ディレクトリ: {runs_dir} (存在します)")
    else:
        print(f"⚠ 出力ディレクトリ: {runs_dir} (存在しませんが、実行時に作成されます)")

    return True


def check_recent_logs():
    """最近のログファイルを確認"""
    print("\n" + "=" * 80)
    print("最近の実行ログの確認")
    print("=" * 80)

    base_dir = Path(__file__).parent
    runs_dir = base_dir / "_runs"

    if not runs_dir.exists():
        print("⚠ 実行ログが見つかりません（初回実行の可能性があります）")
        return True

    # 最新の成功した実行を探す
    success_dirs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("_"):
            report_file = run_dir / "REPORT.md"
            if report_file.exists():
                success_dirs.append((run_dir.name, report_file.stat().st_mtime))

    if success_dirs:
        success_dirs.sort(key=lambda x: x[1], reverse=True)
        print(f"✓ 最近の成功した実行: {success_dirs[0][0]}")
        if len(success_dirs) > 1:
            print(f"  過去の成功実行: {len(success_dirs) - 1}件")
    else:
        print("⚠ 成功した実行のログが見つかりません")

    # 失敗した実行を確認
    failed_dir = runs_dir / "_failed"
    if failed_dir.exists():
        failed_count = len(list(failed_dir.iterdir()))
        if failed_count > 0:
            print(f"⚠ 失敗した実行: {failed_count}件 ({failed_dir})")

    return True


def main():
    """メイン関数"""
    print("\n" + "=" * 80)
    print("M1計算実行前チェック")
    print("=" * 80)
    print()

    checks = [
        ("依存関係", check_imports),
        ("ファイル", check_files),
        ("出力ディレクトリ", check_output_directory),
        ("最近のログ", check_recent_logs),
    ]

    all_ok = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_ok = False
        except Exception as e:
            print(f"\nエラー: {name}のチェック中にエラーが発生しました: {e}")
            all_ok = False

    print("\n" + "=" * 80)
    if all_ok:
        print("✓ すべてのチェックが完了しました。実行可能です。")
        print("\n実行方法:")
        print("  python tmcmc/run_pipeline.py --mode debug --models M1")
        print("  または")
        print("  run_m1.bat (Windows)")
        return 0
    else:
        print("✗ いくつかのチェックで問題が見つかりました。")
        print("  上記のエラーを修正してから実行してください。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
