#!/usr/bin/env python3
"""判定 4 の FAIL が「山の配分」か「山の中の位置」かを成分ごとに切り分ける（2026-10-10a §2）。

使い方:
  python3 mode_split_check.py --param a12 --param a34:-3.5 [--param a55:0:4 ...] <run_dir> [...]

--param NAME[:T1[:T2...]] で成分としきい値（谷）を渡す。しきい値なしならヒストグラムと
seed ごとの分位だけ出す（しきい値はヒストグラムを見て決める）。
a35_mode_weight.py を成分名・しきい値の引数で一般化したもの。TMCMC の最終段の粒子は
等重みなので、単純な粒子割合でよい。出すもの:
  1. 全 run 合算のヒストグラム（固定幅 30 bin）
  2. 山ごとに、seed ごとの重み・中央値・sd・max logL
  3. 山ごとの「seed 間の中央値の幅 / 山の中でプールした sd」（判定 4 と同じ式を山の中で）
"""

import argparse

import numpy as np

NAMES = [
    "a11", "a12", "a22", "b1", "b2", "a33", "a34", "a44", "b3", "b4",
    "a13", "a14", "a23", "a24", "a55", "b5", "a15", "a25", "a35", "a45",
]  # tools/check_paper_runs.py の NAMES と同じ並び


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
ap.add_argument("--param", action="append", required=True)
ap.add_argument("runs", nargs="+")
args = ap.parse_args()

S = {d: _samples(d) for d in args.runs}
L = {d: _logL(d) for d in args.runs}

for spec in args.param:
    pname, *ths = spec.split(":")
    idx = NAMES.index(pname)
    ths = sorted(float(t) for t in ths)

    allx = np.concatenate([S[d][:, idx] for d in args.runs])
    print(f"\n{'=' * 70}\n### {pname}（index {idx}）  しきい値: {ths or 'なし'}")
    print(f"\n全 run 合算のヒストグラム（{len(allx)} 粒子、固定幅 30 bin）")
    h, e = np.histogram(allx, bins=30)
    for c, lo, hi in zip(h, e[:-1], e[1:]):
        print(f"  [{lo:7.2f}, {hi:7.2f})  {c:6d}  {'#' * int(round(60 * c / h.max()))}")

    print(f"\n{'run':48s} {'5%':>7s} {'50%':>7s} {'95%':>7s} {'sd':>6s}")
    for d in args.runs:
        x = S[d][:, idx]
        q5, q50, q95 = np.percentile(x, [5, 50, 95])
        print(f"{_name(d):48s} {q5:+7.2f} {q50:+7.2f} {q95:+7.2f} {np.std(x):6.2f}")
    sd_all = np.std(allx)
    meds = [np.median(S[d][:, idx]) for d in args.runs]
    print(f"{'全体: 中央値の幅 / プール sd':48s} {np.ptp(meds) / max(sd_all, 1e-12):.2f}")

    if not ths:
        continue
    edges = [-np.inf, *ths, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"[{lo:g}, {hi:g})"
        print(f"\n山 {pname} ∈ {label}")
        print(f"{'run':48s} {'n':>6s} {'重み':>6s} {'50%':>7s} {'sd':>6s} {'maxlogL':>9s}")
        pooled, meds = [], []
        for d in args.runs:
            x = S[d][:, idx]
            m = (x >= lo) & (x < hi)
            n = int(m.sum())
            if n == 0:
                print(f"{_name(d):48s} {0:6d} {0:6.3f} {'-':>7s} {'-':>6s} {'-':>9s}")
                continue
            ll = L[d]
            mx = float("nan") if ll is None else float(np.max(ll[m]))
            med = float(np.median(x[m]))
            meds.append(med)
            pooled.append(x[m])
            print(f"{_name(d):48s} {n:6d} {n / len(x):6.3f} {med:+7.2f} {np.std(x[m]):6.2f} {mx:9.2f}")
        if len(meds) >= 2:
            sd_in = np.std(np.concatenate(pooled))
            r = np.ptp(meds) / max(sd_in, 1e-12)
            print(f"{'山の中: 中央値の幅 / プール sd':48s} {r:.2f}  {'PASS' if r <= 0.5 else 'FAIL'}（<=0.5）")
