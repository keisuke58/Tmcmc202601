"""
バックグラウンド実行状況の詳細確認
"""
import os
import sys
import io
import time
from pathlib import Path
import json
import datetime

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
        
        try:
            proc = psutil.Process(pid)
            print(f"Status: RUNNING")
            print(f"Process Name: {proc.name()}")
            print(f"CPU Percent: {proc.cpu_percent(interval=0.1):.1f}%")
            print(f"Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
            
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
        print(f"PIDファイルの読み込みエラー: {e}")
        print()
else:
    print("PIDファイルが見つかりません")
    print()

# 最新の実行ディレクトリの確認（過去10分以内に作成されたもの）
print("=" * 80)
print("Recent Run Directories (last 10 minutes)")
print("=" * 80)
print()

runs_dir = Path("_runs")
if runs_dir.exists():
    recent_runs = []
    current_time = time.time()
    
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("2026"):
            try:
                # ディレクトリの作成時刻を確認
                creation_time = d.stat().st_ctime
                if current_time - creation_time < 600:  # 10分以内
                    recent_runs.append((d, creation_time))
            except:
                pass
    
    if recent_runs:
        recent_runs.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {len(recent_runs)} recent run(s):")
        print()
        for d, ctime in recent_runs:
            ctime_str = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Directory: {d.name}")
            print(f"  Created: {ctime_str}")
            
            # config.jsonの確認
            config_file = d / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        sigma_obs = config.get("sigma_obs") or config.get("experiment", {}).get("sigma_obs", "N/A")
                        models = config.get("models") or config.get("requested_models", "N/A")
                        print(f"  sigma_obs: {sigma_obs}")
                        print(f"  models: {models}")
                except Exception as e:
                    print(f"  Config read error: {e}")
            
            # run.logの確認
            log_file = d / "run.log"
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"  Log lines: {len(lines)}")
                            # 最後の5行を表示
                            print("  Last 3 log lines:")
                            for line in lines[-3:]:
                                line = line.strip()
                                if len(line) > 100:
                                    line = line[:100] + "..."
                                print(f"    {line}")
                except Exception as e:
                    print(f"  Log read error: {e}")
            
            print()
    else:
        print("No recent runs found (within last 10 minutes)")
        print("実行がまだ開始されていないか、エラーが発生している可能性があります")
        print()

# すべての実行ディレクトリを時系列で確認
print("=" * 80)
print("All Run Directories (sorted by creation time)")
print("=" * 80)
print()

all_runs = []
for d in runs_dir.iterdir():
    if d.is_dir() and d.name.startswith("2026"):
        try:
            ctime = d.stat().st_ctime
            all_runs.append((d, ctime))
        except:
            pass

if all_runs:
    all_runs.sort(key=lambda x: x[1], reverse=True)
    print("Latest 5 runs:")
    for i, (d, ctime) in enumerate(all_runs[:5], 1):
        ctime_str = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {d.name} (created: {ctime_str})")
        
        config_file = d / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    sigma_obs = config.get("sigma_obs") or config.get("experiment", {}).get("sigma_obs", "N/A")
                    print(f"   sigma_obs: {sigma_obs}")
            except:
                pass
    print()

print("=" * 80)
print("Summary")
print("=" * 80)
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        try:
            proc = psutil.Process(pid)
            print(f"✅ Background process is RUNNING (PID: {pid})")
            print(f"   CPU: {proc.cpu_percent(interval=0.1):.1f}%")
            print(f"   Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
            print()
            print("実行は正常に進行中です。")
            print("新しい実行ディレクトリが作成されるまで、もう少しお待ちください。")
        except psutil.NoSuchProcess:
            print(f"⚠️ Process completed or terminated (PID: {pid})")
            print("実行が完了したか、エラーで終了した可能性があります。")
    except:
        pass
else:
    print("❌ PID file not found")
    print("バックグラウンドプロセスが開始されていない可能性があります。")

print("=" * 80)
