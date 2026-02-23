#!/usr/bin/env python3
"""
Hyperparameter sweep runner for M1 (Python version for Windows).

What it does:
- Runs tmcmc/run_pipeline.py for a grid of (sigma_obs, cov_rel, n_particles)
- Creates run_ids like: sig0020_cov0005_np0500_ns20
- Stores ALL run outputs under: tmcmc/_runs/<sweep_prefix>/<run_id>/
- Appends a compact summary to: tmcmc/_runs/<sweep_prefix>/sweep_summary.csv

Notes:
- stdout/stderr of each run is already persisted by run_pipeline.py into each run directory.
- This script only orchestrates + summarizes.
- Runs are executed in parallel (configurable) and best run is auto-selected at the end.
"""

import os
import sys
import subprocess
import json
import csv
import re
import math
import shutil
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import argparse


def get_root_dir() -> Path:
    """Get the root directory (parent of tmcmc/)"""
    script_dir = Path(__file__).parent
    return script_dir.parent


def validate_environment() -> None:
    """Validate environment before running sweep"""
    print("Validating environment...")
    
    # Python version check
    if sys.version_info < (3, 7):
        raise RuntimeError(f"Python 3.7+ required, got {sys.version}")
    print(f"  [OK] Python version: {sys.version.split()[0]}")
    
    # Check root directory
    root_dir = get_root_dir()
    if not root_dir.exists():
        raise RuntimeError(f"Root directory not found: {root_dir}")
    print(f"  [OK] Root directory: {root_dir}")
    
    # Check run_pipeline.py exists
    pipeline_script = root_dir / "tmcmc" / "run_pipeline.py"
    if not pipeline_script.exists():
        raise RuntimeError(f"run_pipeline.py not found: {pipeline_script}")
    print(f"  [OK] run_pipeline.py found")
    
    # Check if run_pipeline.py supports --runs-root
    try:
        result = subprocess.run(
            [sys.executable, str(pipeline_script), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "--runs-root" not in result.stdout:
            raise RuntimeError("run_pipeline.py does not support --runs-root flag")
        print(f"  [OK] --runs-root flag supported")
    except subprocess.TimeoutExpired:
        raise RuntimeError("run_pipeline.py --help timed out")
    except Exception as e:
        raise RuntimeError(f"Failed to check run_pipeline.py: {e}")
    
    # Check disk space (at least 10GB free)
    runs_root = root_dir / "tmcmc" / "_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(runs_root)
    free_gb = stat.free / (1024**3)
    if free_gb < 10:
        print(f"  [WARN] Low disk space: {free_gb:.1f} GB free")
    else:
        print(f"  [OK] Disk space: {free_gb:.1f} GB free")
    
    print("Environment validation complete\n")


def normalize_tag(value: float) -> str:
    """Turn 0.02 -> "0020", 0.005 -> "0005" """
    return f"{int(round(value * 1000)):04d}"


def extract_report_field(report_path: Path, regex: str) -> Optional[str]:
    """Best-effort extraction from REPORT.md"""
    if not report_path.exists():
        return None
    try:
        text = report_path.read_text(encoding="utf-8")
        match = re.search(regex, text, flags=re.MULTILINE)
        return match.group(1) if match else None
    except Exception:
        return None


def parse_key_metrics_row(report_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse the table row starting with "| M1 |" from REPORT.md and return:
    (rmse_total_map, rom_error_final, ess_min)
    """
    if not report_path.exists():
        return (None, None, None)
    try:
        text = report_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("| M1 |"):
                cols = [c.strip() for c in line.strip().strip("|").split("|")]
                # cols: Model, RMSE_total(MAP), MAE_total(MAP), max_abs(MAP), rom_error_final, ESS_min, ...
                rmse = cols[1] if len(cols) > 1 else None
                rom = cols[4] if len(cols) > 4 else None
                ess = cols[5] if len(cols) > 5 else None
                return (rmse, rom, ess)
        return (None, None, None)
    except Exception:
        return (None, None, None)


def python_json(json_path: Path, expr: str) -> Optional[str]:
    """Extract value from JSON using Python expression"""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Safe eval with limited builtins
        v = eval(expr, {"__builtins__": {}}, {"d": d})
        return str(v) if v is not None else None
    except Exception:
        return None


def run_one(
    sigma: float,
    cov: float,
    np: int,
    n_stages: int,
    mode: str,
    debug_level: str,
    models: str,
    sweep_dir: Path,
    root_dir: Path,
    max_retries: int = 2,
) -> Dict:
    """Run a single job with retry logic"""
    sig_tag = normalize_tag(sigma)
    cov_tag = normalize_tag(cov)
    run_id = f"sig{sig_tag}_cov{cov_tag}_np{np:04d}_ns{n_stages:02d}"
    run_dir = sweep_dir / run_id
    report = run_dir / "REPORT.md"
    metrics = run_dir / "metrics.json"
    
    result = {
        "run_id": run_id,
        "mode": mode,
        "models": models,
        "n_particles": np,
        "n_stages": n_stages,
        "sigma_obs": sigma,
        "cov_rel": cov,
        "exit_code": 0,
        "status": "missing",
        "ess_min": "missing",
        "rmse_total_map": "missing",
        "map_error": "missing",
        "rom_error_final": "missing",
        "run_dir": str(run_dir),
        "report_path": str(report),
    }
    
    # Skip if already completed
    if report.exists():
        print(f"[skip] {run_id} (already has REPORT.md)")
    else:
        print(f"[run ] {run_id}")
        cmd = [
            sys.executable,
            str(root_dir / "tmcmc" / "run_pipeline.py"),
            "--runs-root", str(sweep_dir),
            "--mode", mode,
            "--debug-level", debug_level,
            "--run-id", run_id,
            "--models", models,
            "--n-particles", str(np),
            "--n-stages", str(n_stages),
            "--sigma-obs", str(sigma),
            "--cov-rel", str(cov),
        ]
        
        # Retry logic
        exit_code = 1
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Ensure run directory exists
                run_dir.mkdir(parents=True, exist_ok=True)
                
                # Run with timeout (estimate: 1 hour per 1000 particles per stage)
                timeout_seconds = int(np * n_stages * 3.6)  # ~1 hour per 1000 particles per stage
                if timeout_seconds > 86400:  # Cap at 24 hours
                    timeout_seconds = 86400
                
                exit_code = subprocess.call(
                    cmd,
                    cwd=str(root_dir),
                    timeout=timeout_seconds
                )
                
                if exit_code == 0:
                    break
                else:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 60  # 1min, 2min, ...
                        print(f"[retry] {run_id} attempt {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"[fail] {run_id} (exit_code={exit_code} after {max_retries + 1} attempts)")
                        
            except subprocess.TimeoutExpired:
                print(f"[timeout] {run_id} (exceeded {timeout_seconds}s)")
                exit_code = 124  # Standard timeout exit code
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 60
                    print(f"[retry] {run_id} after timeout, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    break
            except Exception as e:
                last_error = str(e)
                print(f"[error] {run_id} (exception: {e})")
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 60
                    print(f"[retry] {run_id} after error, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    exit_code = 1
                    break
        
        result["exit_code"] = exit_code
        if exit_code != 0:
            result["status"] = "FAIL"
            if last_error:
                print(f"[fail] {run_id} final error: {last_error}")
            return result
    
    # Parse outputs (best-effort)
    status = extract_report_field(report, r'^- \*\*status\*\*: \*\*(PASS|WARN|FAIL)\*\*')
    if result["exit_code"] != 0:
        status = "FAIL"
    result["status"] = status if status else "missing"
    
    rmse, rom, ess = parse_key_metrics_row(report)
    result["rmse_total_map"] = rmse if rmse else "missing"
    result["rom_error_final"] = rom if rom else "missing"
    result["ess_min"] = ess if ess else "missing"
    
    map_error = python_json(metrics, '((d.get("errors") or {}).get("m1_map_error"))')
    result["map_error"] = map_error if map_error else "missing"
    
    return result


def select_best(summary_csv: Path) -> None:
    """Select the best run based on status, RMSE, MAP error, and ESS"""
    if not summary_csv.exists():
        print("No summary CSV found.")
        return
    
    rows = []
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    
    if not rows:
        print("No rows to select from.")
        return
    
    def fnum(x: Optional[str]) -> float:
        try:
            if x is None:
                return math.inf
            x = x.strip()
            if x == "" or x.lower() == "missing":
                return math.inf
            return float(x)
        except Exception:
            return math.inf
    
    status_rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    
    def key(row: Dict) -> Tuple[int, float, float, float]:
        # If the process failed, force FAIL regardless of parsed status.
        try:
            exit_code = int((row.get("exit_code") or "0").strip() or "0")
            if exit_code != 0:
                status = "FAIL"
            else:
                status = (row.get("status") or "missing").strip()
        except Exception:
            status = (row.get("status") or "missing").strip()
        
        rank = status_rank.get(status, 3)
        rmse = fnum(row.get("rmse_total_map") or "missing")
        map_err = fnum(row.get("map_error") or "missing")
        ess = fnum(row.get("ess_min") or "missing")
        # Prefer: PASS > WARN > FAIL, then lowest RMSE, then lowest MAP error, then highest ESS
        return (rank, rmse, map_err, -ess)
    
    best = min(rows, key=key)
    out_dir = summary_csv.parent
    
    (out_dir / "best_run_id.txt").write_text(best.get("run_id", "missing") + "\n", encoding="utf-8")
    
    header = fieldnames if fieldnames else list(best.keys())
    line = ",".join(best.get(h, "") for h in header)
    (out_dir / "best_row.csv").write_text(",".join(header) + "\n" + line + "\n", encoding="utf-8")
    
    print(f"BEST run_id={best.get('run_id')} status={best.get('status')} "
          f"rmse={best.get('rmse_total_map')} map_error={best.get('map_error')} "
          f"ess={best.get('ess_min')}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for M1")
    parser.add_argument("--sigma-list", type=str, default="0.01",
                        help="Space-separated list of sigma_obs values (default: 0.01)")
    parser.add_argument("--cov-rel-list", type=str, default="0.02",
                        help="Space-separated list of cov_rel values (default: 0.02)")
    parser.add_argument("--np-list", type=str, default="2000 5000 10000",
                        help="Space-separated list of n_particles values (default: 2000 5000 10000)")
    parser.add_argument("--n-stages", type=int, default=30,
                        help="Number of stages (default: 30)")
    parser.add_argument("--max-jobs", type=int, default=8,
                        help="Maximum parallel jobs (default: 8)")
    parser.add_argument("--mode", type=str, default="debug",
                        help="Mode (default: debug)")
    parser.add_argument("--debug-level", type=str, default="MINIMAL",
                        help="Debug level (default: MINIMAL)")
    parser.add_argument("--models", type=str, default="M1",
                        help="Models (default: M1)")
    parser.add_argument("--sweep-prefix", type=str, default=None,
                        help="Sweep prefix (default: auto-generated)")
    parser.add_argument("--runs-root", type=str, default=None,
                        help="Runs root directory (default: tmcmc/_runs)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Maximum retries per job (default: 2)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip environment validation")
    
    args = parser.parse_args()
    
    # Validate environment first
    if not args.skip_validation:
        try:
            validate_environment()
        except Exception as e:
            print(f"[ERROR] Environment validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    root_dir = get_root_dir()
    runs_root = Path(args.runs_root) if args.runs_root else root_dir / "tmcmc" / "_runs"
    
    # Validate parameters
    try:
        sigma_list = [float(x) for x in args.sigma_list.split()]
        cov_rel_list = [float(x) for x in args.cov_rel_list.split()]
        np_list = [int(x) for x in args.np_list.split()]
        
        if not all(0 < s < 1 for s in sigma_list):
            raise ValueError("sigma_obs values must be between 0 and 1")
        if not all(0 < c < 1 for c in cov_rel_list):
            raise ValueError("cov_rel values must be between 0 and 1")
        if not all(n > 0 for n in np_list):
            raise ValueError("n_particles values must be positive")
        if args.n_stages <= 0:
            raise ValueError("n_stages must be positive")
        if args.max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
    except ValueError as e:
        print(f"[ERROR] Parameter validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Lists already parsed and validated above
    
    # Create sweep directory
    if args.sweep_prefix:
        sweep_prefix = args.sweep_prefix
    else:
        sweep_prefix = f"sweep_m1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    sweep_dir = runs_root / sweep_prefix
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = sweep_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    
    summary_csv = sweep_dir / "sweep_summary.csv"
    summary_header = [
        "run_id", "mode", "models", "n_particles", "n_stages",
        "sigma_obs", "cov_rel", "exit_code", "status", "ess_min",
        "rmse_total_map", "map_error", "rom_error_final", "run_dir", "report_path"
    ]
    
    print(f"Sweep dir: {sweep_dir}")
    print(f"Grid:")
    print(f"  sigma_obs: {sigma_list}")
    print(f"  cov_rel:   {cov_rel_list}")
    print(f"  particles: {np_list}")
    print(f"Parallel:")
    print(f"  max_jobs:  {args.max_jobs}")
    print()
    
    # Generate all combinations
    jobs = []
    for sigma in sigma_list:
        for cov in cov_rel_list:
            for np in np_list:
                jobs.append((sigma, cov, np))
    
    # Run jobs in parallel
    results = []
    completed_count = 0
    total_jobs = len(jobs)
    
    print(f"Total jobs: {total_jobs}")
    print(f"Parallel workers: {args.max_jobs}")
    print()
    
    try:
        with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
            futures = {
                executor.submit(
                    run_one,
                    sigma, cov, np, args.n_stages,
                    args.mode, args.debug_level, args.models,
                    sweep_dir, root_dir,
                    args.max_retries
                ): (sigma, cov, np)
                for sigma, cov, np in jobs
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # Write row immediately (with error handling)
                    try:
                        row_file = rows_dir / f"{result['run_id']}.csv"
                        with open(row_file, "w", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=summary_header)
                            writer.writerow(result)
                    except Exception as e:
                        print(f"[warn] Failed to write row file for {result['run_id']}: {e}")
                    
                    print(f"[progress] {completed_count}/{total_jobs} completed")
                except Exception as e:
                    sigma, cov, np = futures[future]
                    print(f"[fail] sigma={sigma} cov={cov} np={np} (exception: {e})")
                    # Create a failure result
                    sig_tag = normalize_tag(sigma)
                    cov_tag = normalize_tag(cov)
                    run_id = f"sig{sig_tag}_cov{cov_tag}_np{np:04d}_ns{args.n_stages:02d}"
                    failure_result = {
                        "run_id": run_id,
                        "mode": args.mode,
                        "models": args.models,
                        "n_particles": np,
                        "n_stages": args.n_stages,
                        "sigma_obs": sigma,
                        "cov_rel": cov,
                        "exit_code": 1,
                        "status": "FAIL",
                        "ess_min": "missing",
                        "rmse_total_map": "missing",
                        "map_error": "missing",
                        "rom_error_final": "missing",
                        "run_dir": str(sweep_dir / run_id),
                        "report_path": str(sweep_dir / run_id / "REPORT.md"),
                    }
                    results.append(failure_result)
                    completed_count += 1
    except KeyboardInterrupt:
        print("\n[interrupt] Sweep interrupted by user")
        print("Saving partial results...")
    except Exception as e:
        print(f"\n[error] Fatal error in sweep execution: {e}")
        print("Saving partial results...")
    
    # Assemble summary.csv deterministically (with error handling)
    try:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_header)
            writer.writeheader()
            # Sort by run_id for reproducibility
            for result in sorted(results, key=lambda x: x["run_id"]):
                writer.writerow(result)
        print(f"[OK] Summary saved: {summary_csv}")
    except Exception as e:
        print(f"[ERROR] Failed to write summary CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Select best run (with error handling)
    try:
        if len(results) > 0:
            select_best(summary_csv)
        else:
            print("[WARN] No results to select best from")
    except Exception as e:
        print(f"[WARN] Failed to select best run: {e}")
    
    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Completed jobs: {completed_count}/{total_jobs}")
    print(f"Summary: {summary_csv}")
    print(f"Sweep directory: {sweep_dir}")
    print()
    
    # Print summary statistics
    if results:
        passed = sum(1 for r in results if r.get("status") == "PASS")
        warned = sum(1 for r in results if r.get("status") == "WARN")
        failed = sum(1 for r in results if r.get("status") == "FAIL")
        print(f"Results summary:")
        print(f"  PASS: {passed}")
        print(f"  WARN: {warned}")
        print(f"  FAIL: {failed}")


if __name__ == "__main__":
    main()
