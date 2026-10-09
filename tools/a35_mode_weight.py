#!/usr/bin/env python3
"""a35 (index 18) の二峰のうち「下の山」(a35 < THRESH) の重みを run ごとに出す（2026-10-09h §1）。

使い方: python3 a35_mode_weight.py [--thresh -10 [--thresh -7 ...]] <run_dir> [...]
TMCMC の最終段の粒子は等重みなので、単純な粒子割合でよい。

a33_mode_weight.py の a35 版。DH p2 で a35 の中央値が seed ごとに大きく違い（判定 4 FAIL）、
山の配分の違いかを見る。出すもの:
  - seed ごとの山の重み w(a35 < しきい値) と a35 の分位、max logL
  - 山ごとの粒子数・max logL・a45 の 5/50/95%（山ごとの max logL の差も）
  - 全 run 合算の a35・a45・a25・a22 のヒストグラム（固定幅 30 bin）
"""

import argparse

import numpy as np

A22 = 2
A25 = 17
A35 = 18
A45 = 19
HIST_PARAMS = (("a35", A35), ("a45", A45), ("a25", A25), ("a22", A22))


def _samples(d):
    s = np.load(f"{d}/samples.npy")
    return s if s.ndim == 2 else s.reshape(s.shape[0], -1)


def _logL(d):
    try:
        return np.load(f"{d}/logL.npy")
    except OSError:
        return None


def _name(d):
    return d.rstrip("/").split("/")[-1]


ap = argparse.ArgumentParser()
ap.add_argument("--thresh", type=float, action="append")
ap.add_argument("runs", nargs="+")
args = ap.parse_args()
args.thresh = args.thresh or [-10.0]

for th in args.thresh:
    print(f"\n### しきい値 a35 = {th:g}")
    print(
        f"{'run':52s} {'n':>6s} {'w(a35<th)':>10s} {'median':>8s} {'5%':>8s} {'95%':>8s} "
        f"{'maxlogL':>10s}"
    )
    for d in args.runs:
        a35 = _samples(d)[:, A35]
        ll = _logL(d)
        mx = float("nan") if ll is None else float(np.max(ll))
        q5, q50, q95 = np.percentile(a35, [5, 50, 95])
        w = float(np.mean(a35 < th))
        print(f"{_name(d):52s} {len(a35):6d} {w:10.3f} {q50:8.2f} {q5:8.2f} {q95:8.2f} {mx:10.2f}")

    print(
        f"\n{'run':52s} {'山':12s} {'n':>6s} {'a45 5%':>8s} {'a45 50%':>8s} "
        f"{'a45 95%':>8s} {'maxlogL':>10s}"
    )
    for d in args.runs:
        s = _samples(d)
        ll = _logL(d)
        mxs = {}
        for label, m in ((f"a35>={th:g}", s[:, A35] >= th), (f"a35<{th:g}", s[:, A35] < th)):
            n = int(m.sum())
            if n == 0:
                print(f"{_name(d):52s} {label:12s} {0:6d} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>10s}")
                continue
            q5, q50, q95 = np.percentile(s[m, A45], [5, 50, 95])
            mx = float("nan") if ll is None else float(np.max(ll[m]))
            mxs[label] = mx
            print(
                f"{_name(d):52s} {label:12s} {n:6d} {q5:+8.2f} {q50:+8.2f} {q95:+8.2f} {mx:10.2f}"
            )
        if len(mxs) == 2:
            hi, lo = mxs.values()
            print(f"{'':52s} {'maxlogL 差（上−下）':12s} {hi - lo:+.2f}")

# --- 全 run 合算のヒストグラム（固定幅 30 bin）---
S = np.concatenate([_samples(d) for d in args.runs])
print(f"\n### 全 run 合算のヒストグラム（{len(S)} 粒子、固定幅 30 bin）")
for pname, idx in HIST_PARAMS:
    h, e = np.histogram(S[:, idx], bins=30)
    print(f"\n{pname}")
    for c, lo, hi in zip(h, e[:-1], e[1:]):
        print(f"  [{lo:7.2f}, {hi:7.2f})  {c:6d}")
