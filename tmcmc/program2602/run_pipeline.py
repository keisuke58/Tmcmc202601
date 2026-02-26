#!/usr/bin/env python3
"""
One-command pipeline: run experiment -> generate REPORT.md.

Usage:
  python tmcmc/run_pipeline.py --mode sanity
  python tmcmc/run_pipeline.py --mode paper --seed 123 --run-id myrun

This script is intentionally minimal and uses subprocess to avoid touching
the existing experiment code paths.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import RUNS_ROOT_DEFAULT, setup_logging

logger = logging.getLogger(__name__)


def _run_and_tee(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> None:
    """
    Run a subprocess and tee combined stdout/stderr to:
    - the current terminal (so users can watch progress)
    - a log file under the run directory (so output is persisted)

    Parameters
    ----------
    cmd : List[str]
        Command to execute
    log_path : Path
        Log file path
    env : Optional[Dict[str, str]]
        Environment variables (if None, uses current environment)
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        if env and "PYTHONPATH" in env:
            f.write(f"PYTHONPATH: {env['PYTHONPATH']}\n")
        f.write("=" * 80 + "\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            # Show to terminal
            sys.stdout.write(line)
            sys.stdout.flush()
            # Persist
            f.write(line)
        rc = p.wait()

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} rc={rc}\n")
        f.write("-" * 80 + "\n")
        f.flush()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run tmcmc case2 and generate REPORT.md",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["sanity", "debug", "paper"], default="sanity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-root", type=str, default=str(RUNS_ROOT_DEFAULT))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run (e.g. M1 or M1,M3)",
    )

    # Pass-through overrides for case2 (optional)
    p.add_argument("--sigma-obs", type=float, default=None)
    p.add_argument(
        "--no-noise", action="store_true", default=False, help="Generate data without noise"
    )
    p.add_argument("--cov-rel", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument(
        "--aleatory-samples",
        type=int,
        default=None,
        help="Reporting-only: paper Nsamples for double-loop cost conversion",
    )
    p.add_argument("--n-particles", type=int, default=None)
    p.add_argument("--n-stages", type=int, default=None)
    p.add_argument("--n-mutation-steps", type=int, default=None)
    p.add_argument("--n-chains", type=int, default=None)
    p.add_argument("--target-ess-ratio", type=float, default=None)
    p.add_argument("--min-delta-beta", type=float, default=None)
    p.add_argument("--max-delta-beta", type=float, default=None)
    p.add_argument("--update-linearization-interval", type=int, default=None)
    p.add_argument("--linearization-threshold", type=float, default=None)
    p.add_argument("--linearization-enable-rom-threshold", type=float, default=None)
    p.add_argument("--force-beta-one", action="store_true", default=False)
    p.add_argument("--lock-paper-conditions", action="store_true", default=False)
    p.add_argument("--debug-level", type=str, default=None)
    p.add_argument("--use-paper-analytical", action="store_true", default=None)
    p.add_argument("--no-paper-analytical", dest="use_paper_analytical", action="store_false")
    p.add_argument("--self-check", action="store_true", default=False)
    p.add_argument(
        "--n-jobs", type=int, default=None, help="Number of parallel jobs for particle evaluation"
    )
    p.add_argument(
        "--use-threads", action="store_true", default=False, help="Use threads instead of processes"
    )

    return p.parse_args(argv)


def main() -> int:
    setup_logging("INFO")
    args = parse_args()

    runs_root = Path(args.runs_root)
    run_id = args.run_id or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.mode}_seed{args.seed}"
    )
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist pipeline logs under the run directory.
    setup_logging("INFO", log_path=run_dir / "pipeline.log")

    # Determine script directory and find case2 script
    script_dir = Path(__file__).parent
    # Use main/case2_main.py (refactored, modular version)
    # Fallback to case2_tmcmc_linearization.py if main version doesn't exist
    case2_main = script_dir / "main" / "case2_main.py"
    case2_legacy = script_dir / "case2_tmcmc_linearization.py"

    if case2_main.exists():
        case2 = case2_main
        # Add script_dir to PYTHONPATH so imports work correctly
        # Also add parent for 'tmcmc.xxx' imports
        pythonpath = f"{script_dir}:{script_dir.parent}"
    elif case2_legacy.exists():
        case2 = case2_legacy
        pythonpath = None
    else:
        raise FileNotFoundError(f"Neither {case2_main} nor {case2_legacy} found")

    report = script_dir / "make_report.py"

    # Build command with PYTHONPATH if needed
    cmd = [sys.executable]
    if pythonpath:
        # Set PYTHONPATH environment variable
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = pythonpath + (
            ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )
    else:
        env = None

    cmd += [
        str(case2),
        "--mode",
        args.mode,
        "--seed",
        str(args.seed),
        "--output-root",
        str(runs_root),
        "--run-id",
        str(run_id),
    ]
    if args.models:
        cmd += ["--models", str(args.models)]
    if args.sigma_obs is not None:
        cmd += ["--sigma-obs", str(args.sigma_obs)]
    if args.no_noise:
        cmd += ["--no-noise"]
    if args.cov_rel is not None:
        cmd += ["--cov-rel", str(args.cov_rel)]
    if args.rho is not None:
        cmd += ["--rho", str(args.rho)]
    if args.aleatory_samples is not None:
        cmd += ["--aleatory-samples", str(args.aleatory_samples)]
    if args.n_particles is not None:
        cmd += ["--n-particles", str(args.n_particles)]
    if args.n_stages is not None:
        cmd += ["--n-stages", str(args.n_stages)]
    if args.n_mutation_steps is not None:
        cmd += ["--n-mutation-steps", str(args.n_mutation_steps)]
    if args.n_chains is not None:
        cmd += ["--n-chains", str(args.n_chains)]
    if args.target_ess_ratio is not None:
        cmd += ["--target-ess-ratio", str(args.target_ess_ratio)]
    if args.min_delta_beta is not None:
        cmd += ["--min-delta-beta", str(args.min_delta_beta)]
    if args.max_delta_beta is not None:
        cmd += ["--max-delta-beta", str(args.max_delta_beta)]
    if args.update_linearization_interval is not None:
        cmd += ["--update-linearization-interval", str(args.update_linearization_interval)]
    if args.linearization_threshold is not None:
        cmd += ["--linearization-threshold", str(args.linearization_threshold)]
    if args.linearization_enable_rom_threshold is not None:
        cmd += [
            "--linearization-enable-rom-threshold",
            str(args.linearization_enable_rom_threshold),
        ]
    if args.force_beta_one:
        cmd += ["--force-beta-one"]
    if args.lock_paper_conditions:
        cmd += ["--lock-paper-conditions"]
    if args.n_jobs is not None:
        cmd += ["--n-jobs", str(args.n_jobs)]
    if args.use_threads:
        cmd += ["--use-threads"]
    if args.debug_level is not None:
        cmd += ["--debug-level", str(args.debug_level)]
    if args.use_paper_analytical is True:
        cmd += ["--use-paper-analytical"]
    if args.use_paper_analytical is False:
        cmd += ["--no-paper-analytical"]
    if args.self_check:
        cmd += ["--self-check"]

    logger.info("Running experiment: %s", run_id)
    _run_and_tee(cmd, log_path=run_dir / "subprocess.log", env=env)

    logger.info("Generating report...")
    _run_and_tee(
        [sys.executable, str(report), "--run-dir", str(run_dir)],
        log_path=run_dir / "subprocess.log",
        env=env,
    )

    logger.info("Pipeline complete: %s/REPORT.md", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
