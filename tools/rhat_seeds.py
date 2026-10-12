#!/usr/bin/env python3
"""seed を「鎖」とみなした Gelman–Rubin の R̂（EML 補足、2026-10-12）。

判定 (e)（seed 間の中央値の差 ≤ 0.5 事後 SD）を、よく使われる指標と並べるためのもの。
各 seed の最終粒子を 1 本の鎖の標本とみなし、成分ごとに R̂ = sqrt(((n−1)/n W + B/n) / W)。
同定された成分（事後 sd < 0.8 × 箱の一様分布の sd）の最大と、全 15 成分の最大を出す。
run の群は supp/spec_provisional.json（p1・p2・ult）。

    python3 tools/rhat_seeds.py docs/revision/generated/final_runs \
        docs/revision/EML_submission/figures/provisional_2026-10-11/supp/spec_provisional.json \
        docs/revision/EML_submission/figures/provisional_2026-10-11/rhat_seeds.json
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

A_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]  # θ の中の A の 15 成分


def group_stats(root, pattern):
    ds = sorted(glob.glob(str(Path(root) / pattern)))
    if len(ds) < 2:
        return None
    Ss = [np.load(Path(d) / "samples.npy") for d in ds]
    rec = json.load(open(Path(ds[0]) / "run_record.json"))
    pb = np.array(rec.get("prior_bounds_final") or rec["prior_bounds"], dtype=float)
    P = np.concatenate(Ss)
    n = min(len(S) for S in Ss)
    r_id, r_all, e_id = [], [], []
    for k in A_IDX:
        w = pb[k, 1] - pb[k, 0]
        if w <= 0:
            continue
        m = np.array([S[:, k].mean() for S in Ss])
        W = np.mean([S[:, k].var(ddof=1) for S in Ss])
        B = n * m.var(ddof=1)
        R = float(np.sqrt(((n - 1) / n * W + B / n) / W))
        r_all.append(R)
        if P[:, k].std() < 0.8 * w / np.sqrt(12.0):
            med = [np.median(S[:, k]) for S in Ss]
            r_id.append(R)
            e_id.append(float((max(med) - min(med)) / P[:, k].std()))
    return {"n_seeds": len(ds), "rhat_max_identified": max(r_id), "rhat_max_all": max(r_all),
            "median_spread_max_identified": max(e_id)}


def main(root, spec, out):
    spec = json.load(open(spec))
    res = {}
    for t, d in spec.items():
        for st in ("p1", "p2", "ult"):
            if st in d:
                s = group_stats(root, d[st])
                if s:
                    res[f"{t}_{st}"] = s
                    print(t, st, {k: round(v, 3) for k, v in s.items()})
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
