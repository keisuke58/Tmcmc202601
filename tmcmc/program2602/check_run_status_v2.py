#!/usr/bin/env python3
"""
Audit TMCMC runs in tmcmc/_runs and report status, focusing on missing outputs.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime

RUNS_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs")

def check_run(run_path: Path):
    run_id = run_path.name
    
    # Define critical files
    files = {
        "theta_map": list(run_path.glob("theta_MAP*.json")),
        "refined_json": list(run_path.glob("refined/theta_MAP*.json")),
        "refined_png": list(run_path.glob("refined/*.png")),
        "logs": list(run_path.glob("*.log")),
    }
    
    # Determine Status
    status = "UNKNOWN"
    reason = []
    
    has_map = len(files["theta_map"]) > 0 or len(files["refined_json"]) > 0
    has_png = len(files["refined_png"]) > 0
    has_logs = len(files["logs"]) > 0
    
    # Check logs for errors
    error_found = False
    log_tail = ""
    if has_logs:
        # Sort logs by modification time
        latest_log = sorted(files["logs"], key=lambda p: p.stat().st_mtime)[-1]
        try:
            with open(latest_log, 'r', errors='ignore') as f:
                content = f.read()
                if "Traceback" in content or "Error:" in content or "CRITICAL" in content:
                    error_found = True
                    # Extract last error line
                    lines = content.splitlines()
                    for line in reversed(lines):
                        if "Error" in line or "Traceback" in line:
                            log_tail = line.strip()
                            break
        except Exception:
            pass

    if error_found:
        status = "FAILED"
        reason.append(f"Error in log: {log_tail[:50]}...")
    elif has_map and has_png:
        status = "SUCCESS"
    elif has_map and not has_png:
        status = "PARTIAL"
        reason.append("Missing plots (PNG)")
    elif not has_map and has_logs:
        status = "INCOMPLETE" # or FAILED without explicit error
        reason.append("No MAP JSON found")
    elif not has_logs:
        status = "EMPTY"
        reason.append("No logs found")
        
    return {
        "id": run_id,
        "path": str(run_path),
        "status": status,
        "reason": ", ".join(reason),
        "last_modified": datetime.fromtimestamp(run_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }

def main():
    results = []
    if not RUNS_DIR.exists():
        print(f"Directory not found: {RUNS_DIR}")
        return

    # Scan directories
    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    
    print(f"{'RUN ID':<50} | {'STATUS':<10} | {'REASON/DETAILS'}")
    print("-" * 100)
    
    for run_dir in run_dirs:
        res = check_run(run_dir)
        results.append(res)
        
        # Color output based on status
        status_str = res["status"]
        if status_str == "SUCCESS":
            status_disp = f"\033[92m{status_str}\033[0m"
        elif status_str == "FAILED":
            status_disp = f"\033[91m{status_str}\033[0m"
        else:
            status_disp = f"\033[93m{status_str}\033[0m"
            
        print(f"{res['id']:<50} | {status_disp:<10} | {res['reason']}")

if __name__ == "__main__":
    main()
