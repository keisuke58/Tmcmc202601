#!/usr/bin/env python3
"""a33 (index 5) の二峰のうち「下の山」(a33 < THRESH) の重みを run ごとに出す。

使い方: python3 a33_mode_weight.py <run_dir> [...]
TMCMC の最終段の粒子は等重みなので、単純な粒子割合でよい。
"""

import json
import sys

import numpy as np

A33 = 5
THRESH = -5.0

print(
    f"{'run':52s} {'n':>6s} {'w(a33<-5)':>10s} {'median':>8s} {'5%':>8s} {'95%':>8s} {'maxlogL':>10s}"
)
for d in sys.argv[1:]:
    s = np.load(f"{d}/samples.npy")
    a33 = s[:, A33] if s.ndim == 2 else s.reshape(s.shape[0], -1)[:, A33]
    w = float(np.mean(a33 < THRESH))
    try:
        mx = float(np.max(np.load(f"{d}/logL.npy")))
    except OSError:
        mx = float("nan")
    q5, q50, q95 = np.percentile(a33, [5, 50, 95])
    name = d.rstrip("/").split("/")[-1]
    print(
        f"{name:52s} {len(a33):6d} {w:10.3f} {q50:8.2f} {q5:8.2f} {q95:8.2f} {mx:10.2f}"
    )
