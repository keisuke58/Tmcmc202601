#!/usr/bin/env python3
"""a33 (index 5) の二峰のうち「下の山」(a33 < THRESH) の重みを run ごとに出す。

使い方: python3 a33_mode_weight.py <run_dir> [...]
TMCMC の最終段の粒子は等重みなので、単純な粒子割合でよい。

山ごとの a45 (index 19) の分布も出す（2026-10-08n §2）。DS で a45 > 0 と a33 の
上の山がセットかどうかを、粒子を山で分けた a45 の 5/50/95% で読むため。
"""

import json
import sys

import numpy as np

A33 = 5
A45 = 19
THRESH = -5.0


def _samples(d):
    s = np.load(f"{d}/samples.npy")
    return s if s.ndim == 2 else s.reshape(s.shape[0], -1)


def _logL(d):
    try:
        return np.load(f"{d}/logL.npy")
    except OSError:
        return None

print(
    f"{'run':52s} {'n':>6s} {'w(a33<-5)':>10s} {'median':>8s} {'5%':>8s} {'95%':>8s} {'maxlogL':>10s}"
)
for d in sys.argv[1:]:
    a33 = _samples(d)[:, A33]
    w = float(np.mean(a33 < THRESH))
    ll = _logL(d)
    mx = float("nan") if ll is None else float(np.max(ll))
    q5, q50, q95 = np.percentile(a33, [5, 50, 95])
    name = d.rstrip("/").split("/")[-1]
    print(f"{name:52s} {len(a33):6d} {w:10.3f} {q50:8.2f} {q5:8.2f} {q95:8.2f} {mx:10.2f}")

# --- 山ごとの a45（2026-10-08n §2）---
print(
    f"\n{'run':52s} {'山':12s} {'n':>6s} {'a45 5%':>8s} {'a45 50%':>8s} "
    f"{'a45 95%':>8s} {'maxlogL':>10s}"
)
for d in sys.argv[1:]:
    s = _samples(d)
    ll = _logL(d)
    name = d.rstrip("/").split("/")[-1]
    for label, m in (("a33>=-5", s[:, A33] >= THRESH), ("a33<-5", s[:, A33] < THRESH)):
        n = int(m.sum())
        if n == 0:
            print(f"{name:52s} {label:12s} {0:6d} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>10s}")
            continue
        q5, q50, q95 = np.percentile(s[m, A45], [5, 50, 95])
        mx = float("nan") if ll is None else float(np.max(ll[m]))
        print(f"{name:52s} {label:12s} {n:6d} {q5:+8.2f} {q50:+8.2f} {q95:+8.2f} {mx:10.2f}")
