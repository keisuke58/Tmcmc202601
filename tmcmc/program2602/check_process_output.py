"""
プロセスの標準出力/エラー出力を確認
"""
import sys
import io
from pathlib import Path
import psutil
import time

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pid_file = Path("run_bg_sigma001.pid")
if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    print(f"Checking process PID: {pid}")
    print()
    
    try:
        proc = psutil.Process(pid)
        print(f"Status: RUNNING")
        print(f"CPU: {proc.cpu_percent(interval=0.1):.1f}%")
        print(f"Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
        
        runtime = time.time() - proc.create_time()
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        print(f"Runtime: {hours}h {minutes}m {seconds}s")
        print()
        
        # 実行ディレクトリの確認（過去1時間以内）
        runs_dir = Path("_runs")
        if runs_dir.exists():
            current_time = time.time()
            recent_dirs = []
            for d in runs_dir.iterdir():
                if d.is_dir() and d.name.startswith("2026"):
                    try:
                        ctime = d.stat().st_ctime
                        if current_time - ctime < 3600:  # 1時間以内
                            recent_dirs.append((d, ctime))
                    except:
                        pass
            
            if recent_dirs:
                recent_dirs.sort(key=lambda x: x[1], reverse=True)
                print(f"Found {len(recent_dirs)} recent directory(ies):")
                for d, ctime in recent_dirs:
                    ctime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctime))
                    print(f"  {d.name} (created: {ctime_str})")
            else:
                print("No recent directories found (within last hour)")
                print("⚠️ 実行ディレクトリが作成されていません")
                print("   実行が初期化段階で止まっている可能性があります")
        
    except psutil.NoSuchProcess:
        print("Process not found (completed or terminated)")
    except Exception as e:
        print(f"Error: {e}")
