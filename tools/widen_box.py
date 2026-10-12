#!/usr/bin/env python3
"""箱への依存の確認（2026-10-13）: run の箱（prior_bounds_final）を中心のまわりに FACTOR 倍に広げた OVERRIDE を出す。

A の自由な成分だけ広げる（固定次元・b は触らない）。出力は paper_gateoff_job.sh の OVERRIDE にそのまま渡せる
"idx:lo:hi;idx:lo:hi;..." の形。

    python3 tools/widen_box.py data_5species/main/_runs/paper_gateoff/CH_p2_2c_seed42 --factor 1.5
"""
import argparse
import json
from pathlib import Path

import numpy as np

A_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--factor", type=float, default=1.5)
args = ap.parse_args()
rec = json.load(open(Path(args.run) / "run_record.json"))
pb = np.array(rec["prior_bounds_final"], dtype=float)
free = set(rec["free_dims"])
parts = []
for i in A_IDX:
    if i not in free or pb[i, 1] <= pb[i, 0]:
        continue
    c, h = pb[i].mean(), (pb[i, 1] - pb[i, 0]) / 2 * args.factor
    parts.append(f"{i}:{c - h:.3f}:{c + h:.3f}")
print(";".join(parts))
