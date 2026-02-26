"""
sigma=0.001で実行されたディレクトリを探す
"""

import json
from pathlib import Path
import sys
import io

# Fix encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Search in both _runs and tmcmc\_runs directories
runs_dirs = [Path("_runs"), Path("tmcmc/_runs"), Path("tmcmc\\_runs")]
print("=" * 80)
print("Searching for runs with sigma_obs = 0.001")
print("=" * 80)
print()

found_runs = []

for runs_dir in runs_dirs:
    if not runs_dir.exists():
        continue
    for config_file in runs_dir.rglob("config.json"):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Check both direct and nested paths
                sigma_obs = config.get("sigma_obs") or config.get("experiment", {}).get("sigma_obs")
                if sigma_obs == 0.001:
                    run_dir = config_file.parent
                    found_runs.append((run_dir, config))
        except:
            pass

if found_runs:
    print(f"Found {len(found_runs)} run(s) with sigma_obs = 0.001:")
    print()
    for run_dir, config in sorted(
        found_runs, key=lambda x: x[0].stat().st_mtime if x[0].exists() else 0, reverse=True
    ):
        print(f"Directory: {run_dir}")
        sigma_obs = config.get("sigma_obs") or config.get("experiment", {}).get("sigma_obs", "N/A")
        print(f"  sigma_obs: {sigma_obs}")
        models = config.get("requested_models") or config.get("models", "N/A")
        print(f"  models: {models}")
        if run_dir.exists():
            import datetime

            mtime = datetime.datetime.fromtimestamp(run_dir.stat().st_mtime)
            print(f"  Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
else:
    print("No runs found with sigma_obs = 0.001")
    print()
    print("Checking recent runs...")
    print()

    # Check recent runs
    all_runs = []
    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            continue
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("2026"):
                config_file = d / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            all_runs.append((d, config))
                    except:
                        pass

    all_runs.sort(key=lambda x: x[0].stat().st_mtime if x[0].exists() else 0, reverse=True)

    print("Recent 5 runs:")
    for run_dir, config in all_runs[:5]:
        sigma_obs = config.get("sigma_obs") or config.get("experiment", {}).get("sigma_obs", "N/A")
        print(f"  {run_dir.name}: sigma_obs = {sigma_obs}")

print("=" * 80)
