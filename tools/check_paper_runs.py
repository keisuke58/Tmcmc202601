#!/usr/bin/env python3
"""論文パイプライン（paper_gateoff_job.sh）の run が「読んでよい事後」かを判定する。

なぜ要るか: 2026-09-30 の 1000 粒子 run では、seed によって a35 の MAP が −0.96 / −14.11 /
−2.88 とばらばらで、事前分布なしの run の max logL が事前分布ありの run より 10 nats も
悪かった。事前分布なしは目標が広い（箱全体）ので、収束していれば max logL は事前分布ありの
run 以上になるはずで、これは「探索が主要なモードに届いていない」ことを示す。
にもかかわらず「端に張り付く＝非同定」と解釈されかけた。収束していない run を読まないために、
解釈の前にここで機械的に判定する。

使い方:
    python3 tools/check_paper_runs.py data_5species/main/_runs/paper_gateoff

判定（同じ TAG・STAGE・事前分布の seed 違いを 1 群とする）:
  0. 保存した logL が保存した粒子の再計算と一致している（estimator が run の最後に照合して
     run_record.json の logL_consistent に記録。記録が無い run も FAIL）
  1. 全 run が beta=1 に到達している
  2. 1 粒子が 1 ステージで平均 2 回以上動く（平均受理率 × mutation 回数 >= 2、かつ受理率 <= 0.60）
  3. seed 間で max logL の幅が 1 nat 以内
  4. seed 間で各自由次元の中央値の幅が、プールした事後 sd の 0.5 倍以内
  5. (ident のみ) 事前分布なしの max logL >= 事前分布ありの max logL − 0.5
     （尤度だけの比較。下回れば事前分布なしは探索不足）
1〜4 のどれかが FAIL の群は、粒子数・mutation 数を増やして回し直す。解釈しない。
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]
WATCH = {"a35": 18, "a45": 19}


def load(d):
    with open(d / "run_record.json") as f:
        rec = json.load(f)
    s, ll = np.load(d / "samples.npy"), np.load(d / "logL.npy")
    cfg = {}
    if (d / "config.json").exists():
        with open(d / "config.json") as f:
            cfg = json.load(f)
    return rec, s, ll, cfg


def main(root):
    root = Path(root)
    if not root.is_dir():
        print(f"判定できない: {root} が無い")
        return 1
    groups = defaultdict(list)
    for d in sorted(root.glob("*_seed*")):
        if not (d / "run_record.json").exists():
            print(f"skip（run_record.json 無し）: {d.name}")
            continue
        key = re.sub(r"_seed\d+$", "", d.name)
        groups[key].append(d)

    if not groups:
        print(f"判定できない: {root} に run が無い（空の PASS を返さない）")
        return 1

    maxll_by_group = {}
    any_fail = False
    for key, dirs in sorted(groups.items()):
        print(f"\n=== {key}  ({len(dirs)} seeds)")
        recs, samples, maxll, med = [], [], [], []
        for d in dirs:
            seed = re.search(r"_seed(\d+)$", d.name).group(1)
            rec, s, _ll, cfg = load(d)
            pb = np.array(rec["prior_bounds_final"])
            free = rec["free_dims"]
            width = pb[:, 1] - pb[:, 0]
            edge = {
                NAMES[i]: float(
                    np.mean(
                        (s[:, i] - pb[i, 0] < 0.05 * width[i])
                        | (pb[i, 1] - s[:, i] < 0.05 * width[i])
                    )
                )
                for i in free
            }
            edgy = {k: round(v, 2) for k, v in edge.items() if v > 0.10}
            w = "  ".join(
                f"{n}={np.percentile(s[:, i], 50):+.2f}[{np.percentile(s[:, i], 5):+.2f},"
                f"{np.percentile(s[:, i], 95):+.2f}]"
                for n, i in WATCH.items()
            )
            bf = rec.get("beta_final")
            rec["beta_final"] = bf  # 古い記録（beta_final 無し）は不明 = FAIL 扱い
            print(
                f"  seed{seed:>3} beta={'?' if bf is None else f'{bf:.3f}'} st={rec['n_stages']:2d} "
                f"acc={rec['mean_accept']:.2f} maxlogL={rec['max_logL']:9.2f} "
                f"lnZ={rec['log_evidence']:9.2f} rmse={cfg.get('rmse', float('nan')):.4f}  {w}"
            )
            if edgy:
                print(f"          箱の端 5% に 10% 超: {edgy}")
            recs.append(rec)
            samples.append(s)
            maxll.append(rec["max_logL"])
            med.append(np.median(s[:, free], axis=0))

        free = recs[0]["free_dims"]
        pooled_sd = np.concatenate(samples)[:, free].std(axis=0)
        med_spread = np.ptp(np.array(med), axis=0) / np.maximum(pooled_sd, 1e-12)
        checks = {
            "0 logL と粒子の対応": all(r.get("logL_consistent") is True for r in recs),
            "1 beta=1": all(
                r["beta_final"] is not None and r["beta_final"] > 1 - 1e-9 for r in recs
            ),
            # 受理率そのものではなく「1 粒子が 1 ステージで平均何回動いたか」を見る。
            # DE-MC と RW を交互に使い、提案の幅は γ/√(2d)・2.38²/d で正しく設定されている。
            # 細く曲がった事後や箱の外への提案で受理率は下がるが、mutation 回数が多ければ粒子は動く。
            "2 粒子の移動(受理率×mutation>=2)": all(
                r["mean_accept"] * r["args"]["n_mutation_steps"] >= 2.0 and r["mean_accept"] <= 0.60
                for r in recs
            ),
            "3 maxlogL 幅<=1": len(maxll) < 2 or float(np.ptp(maxll)) <= 1.0,
            "4 中央値幅<=0.5sd": len(med) < 2 or float(med_spread.max()) <= 0.5,
        }
        worst = [f"{NAMES[free[k]]}({med_spread[k]:.1f})" for k in np.argsort(-med_spread)[:3]]
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'} {k}")
        print(f"       maxlogL 幅 = {np.ptp(maxll):.2f}、中央値幅/sd 上位: {', '.join(worst)}")
        any_fail |= not all(checks.values())
        maxll_by_group[key] = max(maxll)

    # 5. ident: 事前分布なし >= 事前分布あり（尤度のみ）
    for key, m in maxll_by_group.items():
        if "_ident_prior0" in key:
            for other, m2 in maxll_by_group.items():
                if other.startswith(key.replace("_prior0", "_prior")) and other != key:
                    ok = m >= m2 - 0.5
                    any_fail |= not ok
                    print(
                        f"\n{'PASS' if ok else 'FAIL'} 5 {key} の max logL {m:.2f} >= {other} の {m2:.2f} − 0.5"
                        + ("" if ok else "  → 事前分布なしは探索不足。解釈しない")
                    )

    print("\n" + ("FAIL を含む群は回し直すまで解釈しない" if any_fail else "全群 PASS"))
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data_5species/main/_runs/paper_gateoff"))
