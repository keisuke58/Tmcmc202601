"""
バックグラウンド実行状況の確認
"""

import os
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

try:
    import psutil
except ImportError:
    print("psutil is not installed. Installing...")
    os.system("pip install psutil")
    import psutil

print("=" * 80)
print("Background Execution Status Check (sigma=0.001)")
print("=" * 80)
print()

# PIDファイルの確認
pid_file = Path("run_bg_sigma001.pid")
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        print(f"PID: {pid}")

        # プロセスの確認
        try:
            proc = psutil.Process(pid)
            print("Status: RUNNING")
            print(f"Process Name: {proc.name()}")
            print(f"CPU Percent: {proc.cpu_percent(interval=0.1):.1f}%")
            print(f"Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"Start Time: {proc.create_time()}")
            print()

            # 実行時間の計算
            import time

            runtime = time.time() - proc.create_time()
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            seconds = int(runtime % 60)
            print(f"Runtime: {hours}h {minutes}m {seconds}s")
            print()

        except psutil.NoSuchProcess:
            print("Status: COMPLETED or TERMINATED")
            print("プロセスが見つかりません（完了または終了）")
            print()
        except Exception as e:
            print(f"プロセス確認エラー: {e}")
            print()
    except Exception as e:
        print(f"PIDファイルの読み込みエラー: {e}")
        print()
else:
    print("PIDファイルが見つかりません")
    print("バックグラウンドプロセスが開始されていない可能性があります")
    print()

# 最新の実行ディレクトリの確認
print("=" * 80)
print("最新の実行ディレクトリ")
print("=" * 80)
print()

runs_dir = Path("_runs")
if runs_dir.exists():
    run_dirs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("2026"):
            run_dirs.append(d)

    if run_dirs:
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print("最新の5つの実行ディレクトリ:")
        print()
        for i, d in enumerate(run_dirs[:5], 1):
            import datetime

            mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime)
            print(f"{i}. {d.name}")
            print(f"   更新時刻: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

            # config.jsonの確認
            config_file = d / "config.json"
            if config_file.exists():
                import json

                try:
                    with open(config_file) as f:
                        config = json.load(f)
                        if "sigma_obs" in config:
                            print(f"   sigma_obs: {config['sigma_obs']}")
                except (json.JSONDecodeError, OSError):
                    pass

            # run.logの確認
            log_file = d / "run.log"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if len(last_line) > 100:
                                last_line = last_line[:100] + "..."
                            print(f"   最新ログ: {last_line}")
                except OSError:
                    pass

            print()
    else:
        print("実行ディレクトリが見つかりません")
else:
    print("_runsディレクトリが見つかりません")

print("=" * 80)
