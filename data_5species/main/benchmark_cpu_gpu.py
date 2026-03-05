#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_cpu_gpu.py — CPU vs GPU 速度・精度比較

同じ条件で CPU と GPU を実行し、時間・logL・theta_MAP を比較する。
GPU が無い環境では CPU のみ実行し、精度の再現性を確認。

Usage:
    cd data_5species/main
    python benchmark_cpu_gpu.py [--quick]
    python benchmark_cpu_gpu.py --quick --cpu-only   # GPU 無し環境で CPU のみ
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_DIR = SCRIPT_DIR
PROJECT_ROOT = MAIN_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = os.environ.get(
    "PYTHON",
    f"{os.environ.get('HOME', '')}/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python",
)


def run_tmcmc(device: str, quick: bool, output_dir: Path) -> dict:
    """Run TMCMC with given device, return result summary."""
    cmd = [
        PYTHON,
        str(SCRIPT_DIR / "estimate_reduced_nishioka_jax.py"),
        "--condition",
        "Dysbiotic",
        "--cultivation",
        "HOBIC",
        "--device",
        device,
        "--use-exp-init",
        "--output-dir",
        str(output_dir),
    ]
    if quick:
        cmd.append("--quick")

    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    logger.info("Running: %s", " ".join(cmd))
    t0 = __import__("time").time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    elapsed = __import__("time").time() - t0

    if result.returncode != 0:
        logger.error("STDERR: %s", result.stderr[:500] if result.stderr else "")
        raise RuntimeError(f"Command failed: {result.returncode}")

    out = {
        "device": device,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "output_dir": str(output_dir),
    }

    theta_map_path = output_dir / "theta_MAP.json"
    config_path = output_dir / "config.json"
    if theta_map_path.exists():
        with open(theta_map_path) as f:
            out["theta_MAP"] = json.load(f)
    if config_path.exists():
        with open(config_path) as f:
            out["config"] = json.load(f)

    samples_path = output_dir / "samples.npy"
    logL_path = output_dir / "logL.npy"
    if samples_path.exists():
        import numpy as np

        samples = np.load(samples_path)
        logL = np.load(logL_path)
        out["n_samples"] = len(samples)
        out["logL_max"] = float(np.max(logL))
        out["logL_mean"] = float(np.mean(logL))
        out["theta_MAP_arr"] = samples[np.argmax(logL)].tolist()

    return out


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU benchmark")
    parser.add_argument("--quick", action="store_true", help="50p, 500 steps")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU run (GPU 無し環境で使用)")
    args = parser.parse_args()

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = MAIN_DIR / "_runs" / f"benchmark_cpu_gpu_{ts}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    results = []
    devices = ["cpu"]
    if not args.cpu_only:
        # Check if GPU available (subprocess to avoid polluting JAX init)
        try:
            r = subprocess.run(
                [
                    PYTHON,
                    "-c",
                    "import jax; d=jax.devices(); print('gpu' if any('gpu' in str(x).lower() or 'cuda' in str(x).lower() for x in d) else 'cpu')",
                ],
                capture_output=True,
                text=True,
                cwd=str(SCRIPT_DIR),
            )
            if r.returncode == 0 and "gpu" in (r.stdout or "").strip():
                devices.append("gpu")
            else:
                logger.info("No GPU devices found, CPU only")
        except Exception as e:
            logger.warning("Could not check JAX devices: %s", e)

    for device in devices:
        out_dir = bench_dir / device
        out_dir.mkdir(exist_ok=True)
        try:
            r = run_tmcmc(device, args.quick, out_dir)
            results.append(r)
        except Exception as e:
            logger.error("Failed for %s: %s", device, e)
            results.append({"device": device, "error": str(e)})

    # Summary
    report_path = bench_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK REPORT")
    logger.info("=" * 60)
    for r in results:
        if "error" in r:
            logger.info("%s: %s", r["device"], r["error"])
        else:
            logger.info(
                "%s: %.1fs, logL_max=%.2f, n_samples=%s",
                r["device"],
                r.get("elapsed", 0),
                r.get("logL_max", 0),
                r.get("n_samples", 0),
            )
    if len(results) >= 2 and "elapsed" in results[0] and "elapsed" in results[1]:
        cpu_t = results[0]["elapsed"]
        gpu_t = results[1]["elapsed"]
        if gpu_t > 0:
            speedup = cpu_t / gpu_t
            logger.info("Speedup (GPU/CPU): %.2fx", speedup)
    if len(results) >= 2 and "theta_MAP_arr" in results[0] and "theta_MAP_arr" in results[1]:
        import numpy as np

        a = np.array(results[0]["theta_MAP_arr"])
        b = np.array(results[1]["theta_MAP_arr"])
        mae = np.mean(np.abs(a - b))
        logger.info("theta_MAP MAE (CPU vs GPU): %.6f", mae)
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
