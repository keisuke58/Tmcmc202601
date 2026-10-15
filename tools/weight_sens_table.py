#!/usr/bin/env python3
"""補足 sec:S_lambda の表: 尤度の重みの感度（生データ・2 通路、決定論的な比較、2026-10-14）。

tools/two_channel_pilot_v2.py の出力（条件 × A/B/C）を読み、−logL 最小の点での
生きた菌の組成（φψ/Σφψ）・全菌の組成（φ/Σφ）・生きた割合の RMSE、a45、Pg の Day 21/15 比（生きた菌の組成 φψ/Σφψ、本文と同じ）を表にする。
  A: 今の重み（λ_Pg 5、終盤 ×3、珍しい種 ×0.1、λψ 条件ごと）
  B: 重みなし（λ_Pg = λ_late = λ_rare = 1、λψ そのまま）
  C: λψ を全条件 2.0（ほかは A）
重みが違うので −logL は比べない。

    python3 tools/weight_sens_table.py docs/revision/generated/weight_sensitivity/raw \
        --out docs/revision/EML_submission/figures/raw/supp/supp_S_lambda2c.tex
"""
import argparse
import json
from pathlib import Path

A45 = 19
VARIANTS = (("A", "current"), ("B", "unweighted"), ("C", r"$\lambda_\psi=2$"))
TAGS = ("CS", "CH", "DS", "DH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rows, missing = [], []
    for t in TAGS:
        for k, (v, name) in enumerate(VARIANTS):
            f = d / f"{t}_{v}.json"
            if not f.exists():
                missing.append(f.name)
                continue
            r = json.load(open(f))
            o = r["two_channel_opt"]
            cond = rf"\multirow{{3}}{{*}}{{{t}}}" if k == 0 else ""
            rows.append(f"{cond} & {name} & {o['viable_vs_phipsi']:.3f} & {o['total_vs_phi']:.3f} & {o['live']:.3f}"
                        f" & ${r['theta_opt'][A45]:+.2f}$ & {r.get('pg_ratio_21_15_viable', float('nan')):.2f}\\\\")
        if t != TAGS[-1]:
            rows.append(r"\addlinespace")
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r" & Weights & RMSE viable & RMSE total & RMSE live & $a_{45}$ & Pg D21/D15\\", r"\midrule",
             *rows, r"\bottomrule", r"\end{tabular}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"-> {a.out}  ({len(TAGS) * len(VARIANTS) - len(missing)} / {len(TAGS) * len(VARIANTS)})")
    if missing:
        print("まだ無い:", " ".join(missing))


if __name__ == "__main__":
    main()
