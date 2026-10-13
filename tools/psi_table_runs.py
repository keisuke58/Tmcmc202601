#!/usr/bin/env python3
"""run の ψ_i とデータから出した ψ_i^data を並べる（13b §4 の報告用、2026-10-13）。

ψ_i^data は tools/psi_from_data.py と同じ式（生きた菌の組成 / 全菌の組成 × 群集の生きた割合、
反復の中央値、全菌の割合が 2% 未満の日は外す）で、観測日（Day 3–21）の中央値。
モデルの ψ_i は 2 通路の尤度と同じく状態 g[:, 6:11] を観測日で読み、日の中央値をとる。
  - MAP: seed ごとの theta_MAP
  - 事後: 3 seed をまとめた粒子から N_POST 本、日の中央値の 5–95% 区間
pilot・p1（--fix-psi）は ψ を固定しているので、モデルの列は参考にならない（表に印を付ける）。

    python3 tools/psi_table_runs.py data_5species/main/_runs/paper_gateoff \
        --pattern '{tag}_p2_2c_seed*' --out docs/handoff/gpu_2026-10-13_psi_p2_2c
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as MF  # noqa: E402  (estimator を cpu で読み込む)
from psi_from_data import NAMES, data_psi  # noqa: E402

N_POST = 200


def model_psi_days(thetas, rec):
    """θ（n, 20）→ 観測日（Day > 1）での ψ（n, n_day, 5）と、その日。"""
    _data, t_days, idx = MF.exp_data(*MF.CK_of(rec), rec["args"])
    keep = t_days > 1.0 + 1e-9
    g = MF.simulate_full(thetas, rec)
    return g[:, np.asarray(idx)[keep], 6:11], t_days[keep]


def summarize(root, pattern, tag):
    g = MF.group(root, pattern, tag)
    if g is None:
        return None
    rec0 = g["runs"][0]["rec"]
    cond, cult = MF.CK_of(rec0)
    dp = data_psi(cond, cult, MF.EP.data_dir_for(rec0["args"]) / "experiment_data")  # run が読んだデータ
    A = np.array([dp[d] for d in sorted(dp)])
    data = [None if np.all(np.isnan(A[:, i])) else float(np.nanmedian(A[:, i])) for i in range(5)]
    n_days = [int(np.sum(~np.isnan(A[:, i]))) for i in range(5)]

    seeds = {}
    for r in g["runs"]:
        psi, days = model_psi_days(r["theta"][None, :], r["rec"])
        seeds[r["dir"].name] = {
            "median_over_days": np.median(psi[0], axis=0).tolist(),
            "per_day": {str(int(round(d))): psi[0, k].tolist() for k, d in enumerate(days)},
            "max_logL": float(r["rec"]["max_logL"]),
        }
    rng = np.random.default_rng(0)
    S = g["pooled"]
    sel = rng.choice(len(S), size=min(N_POST, len(S)), replace=False)
    psi_post, _ = model_psi_days(S[sel], rec0)
    med = np.median(psi_post, axis=1)  # (n, 5) 日の中央値
    best = g["best"]["dir"].name
    return {
        "tag": tag,
        "runs": [r["dir"].name for r in g["runs"]],
        "psi_fixed": any(r["rec"].get("psi_fixed") is not None for r in g["runs"]),
        "two_channel": all(r["rec"]["args"].get("two_channel", False) for r in g["runs"]),
        "data": data,
        "n_days": n_days,
        "map_best": seeds[best]["median_over_days"],
        "best_run": best,
        "seeds": seeds,
        "post_median": np.median(med, axis=0).tolist(),
        "post_q05": np.quantile(med, 0.05, axis=0).tolist(),
        "post_q95": np.quantile(med, 0.95, axis=0).tolist(),
    }


def to_md(res, pattern):
    f = lambda x: "--" if x is None else f"{x:.2f}"  # noqa: E731
    L = [f"run: `{pattern}`。各セルは ψ の Day 3–21 の中央値。"
         "data = ψ^data（括弧は使えた日数）、MAP = max logL の seed、"
         "事後 = 3 seed の粒子 200 本の中央値 [5%, 95%]。", ""]
    L.append("| 条件 | " + " | ".join(NAMES) + " |")
    L.append("|---|" + "---|" * 5)
    for tag, r in res.items():
        mark = "（ψ 固定の段）" if r["psi_fixed"] else ""
        L.append(f"| {tag} data | " + " | ".join(f"{f(r['data'][i])} ({r['n_days'][i]})" for i in range(5)) + " |")
        L.append(f"| {tag} MAP{mark} | " + " | ".join(f(x) for x in r["map_best"]) + " |")
        L.append(f"| {tag} 事後 | " + " | ".join(
            f"{f(r['post_median'][i])} [{f(r['post_q05'][i])}, {f(r['post_q95'][i])}]" for i in range(5)) + " |")
    L.append("")
    L.append("seed ごとの MAP（日の中央値）:")
    L.append("")
    L.append("| run | max logL | " + " | ".join(NAMES) + " |")
    L.append("|---|---|" + "---|" * 5)
    for r in res.values():
        for name, s in r["seeds"].items():
            L.append(f"| {name} | {s['max_logL']:.2f} | " + " | ".join(f(x) for x in s["median_over_days"]) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pattern", default="{tag}_p2_2c_seed*")
    ap.add_argument("--tags", default="CS,CH,DS,DH")
    ap.add_argument("--out", default=None, help="拡張子なし。.md と .json を書く")
    a = ap.parse_args()
    res = {}
    for tag in a.tags.split(","):
        r = summarize(a.root, a.pattern, tag)
        if r is None:
            print(f"{tag}: run なし（{a.pattern.format(tag=tag)}）")
            continue
        res[tag] = r
    md = to_md(res, a.pattern)
    print(md)
    if a.out:
        Path(a.out + ".md").write_text(md)
        json.dump(res, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
