#!/usr/bin/env python3
"""
Train DeepONet for all 4 biofilm conditions.

Usage:
  python train_all_conditions.py [--epochs 500] [--batch-size 256]

Expects data files in data/ directory:
  data/train_{condition}_N10000.npz
"""

import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = [
    "Commensal_Static",
    "Commensal_HOBIC",
    "Dysbiotic_Static",
    "Dysbiotic_HOBIC",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    results = {}

    for cond in CONDITIONS:
        data_file = data_dir / f"train_{cond}_N{args.n_samples}.npz"
        ckpt_dir = Path(__file__).parent / f"checkpoints_{cond}"

        if not data_file.exists():
            print(f"\n[SKIP] {cond}: data file not found ({data_file})")
            continue

        if args.skip_existing and (ckpt_dir / "best.eqx").exists():
            print(f"\n[SKIP] {cond}: checkpoint already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Training: {cond}")
        print(f"{'='*60}")

        t0 = time.time()
        cmd = [
            sys.executable, "deeponet_hamilton.py", "train",
            "--data", str(data_file),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--checkpoint-dir", str(ckpt_dir),
        ]
        ret = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        elapsed = time.time() - t0

        results[cond] = {
            "time": elapsed,
            "returncode": ret.returncode,
        }

        # Run eval + benchmark
        if ret.returncode == 0:
            print(f"\n--- Eval: {cond} ---")
            subprocess.run([
                sys.executable, "deeponet_hamilton.py", "eval",
                "--checkpoint", str(ckpt_dir / "best.eqx"),
                "--data", str(data_file),
            ], cwd=str(Path(__file__).parent))

            print(f"\n--- Benchmark: {cond} ---")
            subprocess.run([
                sys.executable, "deeponet_hamilton.py", "benchmark",
                "--checkpoint", str(ckpt_dir / "best.eqx"),
                "--data", str(data_file),
                "--n-bench", "1000",
            ], cwd=str(Path(__file__).parent))

            print(f"\n--- Plot: {cond} ---")
            subprocess.run([
                sys.executable, "deeponet_hamilton.py", "plot",
                "--checkpoint", str(ckpt_dir / "best.eqx"),
                "--data", str(data_file),
            ], cwd=str(Path(__file__).parent))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond, res in results.items():
        status = "OK" if res["returncode"] == 0 else "FAIL"
        print(f"  {cond:<25} {status}  {res['time']:.0f}s")


if __name__ == "__main__":
    main()
