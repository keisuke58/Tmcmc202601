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
    python3 tools/check_paper_runs.py data_5species/main/_runs/paper_gateoff --glob '*mut80*'

判定（同じ TAG・STAGE・事前分布の seed 違いを 1 群とする）:
  0. 保存した logL が保存した粒子の再計算と一致している（estimator が run の最後に照合して
     run_record.json の logL_consistent に記録。記録が無い run も FAIL）
  1. 全 run が beta=1 に到達している
  2. 1 粒子が 1 ステージで平均 2 回以上動く（run_record の moves_per_particle_mean。
     記録が無い古い run は 平均受理率 × mutation 回数 で代用する）
  2b. 1 粒子が run 全体で各自由次元あたり 5 回以上動く（移動回数 × ステージ数 / 自由次元数）
     — 判定 2 を通っても、15 次元を 23 回の移動で埋めることはできない
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


def _moves(rec):
    """1 粒子が 1 ステージで動いた回数。実測があればそれを使う。

    古い run は mean_accept が nominal な n_mutation_steps で割られているので、
    掛け戻すと移動回数になる（2026-10-05 にエンジン側で実測を残すよう修正）。
    """
    m = rec.get("moves_per_particle_mean")
    if m is not None:
        return float(m)
    return float(rec["mean_accept"]) * float(rec["args"]["n_mutation_steps"])


def _moves_per_dim(rec):
    """run 全体での 1 粒子あたりの移動回数を、自由次元数で割った値。"""
    n_free = max(len(rec.get("free_dims") or []), 1)
    return _moves(rec) * float(rec["n_stages"]) / n_free


def load(d):
    with open(d / "run_record.json") as f:
        rec = json.load(f)
    s, ll = np.load(d / "samples.npy"), np.load(d / "logL.npy")
    cfg = {}
    if (d / "config.json").exists():
        with open(d / "config.json") as f:
            cfg = json.load(f)
    return rec, s, ll, cfg


def _side_edges(rec, s):
    """自由次元ごとに、箱の下側 5% / 上側 5% に入る粒子の割合を別々に返す。

    `check_paper_runs.py` は長らく両端の合計だけを出していたので、08d の停止基準
    （片側 5% に 0.20 以上）をそのまま当てられなかった（2026-10-08p §4）。
    """
    pb = np.array(rec["prior_bounds_final"])
    width = pb[:, 1] - pb[:, 0]
    out = {}
    for i in rec["free_dims"]:
        lo = float(np.mean(s[:, i] - pb[i, 0] < 0.05 * width[i]))
        hi = float(np.mean(pb[i, 1] - s[:, i] < 0.05 * width[i]))
        out[NAMES[i]] = (lo, hi, float(pb[i, 0]), float(pb[i, 1]))
    return out


def _prev_side_edges(cfg):
    """前段（--init-from-dir）の片側の割合。前段が無い・読めないときは None。"""
    argv = cfg.get("argv") or []
    if "--init-from-dir" not in argv:
        return None
    prev = Path(argv[argv.index("--init-from-dir") + 1])
    for cand in (prev, Path.cwd() / prev, Path("data_5species/main") / prev):
        if (cand / "run_record.json").exists() and (cand / "samples.npy").exists():
            try:
                prec, ps, _ll, _cfg = load(cand)
                return _side_edges(prec, ps), cand.name
            except Exception:
                return None
    return None


def _stage_of(name):
    """run の名前（`DS_p2_wide80_..._seed42`）から段（pilot|p1|p2|ult|ident）を取る。"""
    m = re.match(r"[A-Z]+_(pilot|p1|p2|ult|ident)(?:_|$)", name)
    return m.group(1) if m else None


# 尤度が同じ段の組（2026-10-08q §4）。p1 → p2 は尤度が変わるので増加の基準に使わない。
_SAME_LIKELIHOOD = {("pilot", "p1"), ("p2", "ult")}


def _comparable(prev_name, cur_name):
    """前段との「+0.10 以上」を当てていい組か。同じ段の回し直し、または尤度が同じ段の組だけ。"""
    ps, cs = _stage_of(prev_name), _stage_of(cur_name)
    if ps is None or cs is None:
        return False
    return ps == cs or (ps, cs) in _SAME_LIKELIHOOD


def _edge_flags(sides, prev, comparable=True):
    """切られている成分に印を付ける: 片側 0.20 以上、または前段から +0.10 以上。

    「+0.10 以上」は尤度が同じ段どうしでだけ比べる（`comparable`、2026-10-08q §4）。
    """
    rows = []
    for nm, (lo, hi, blo, bhi) in sides.items():
        if max(lo, hi) <= 0.10:
            continue
        plo = phi = None
        if prev and nm in prev[0]:
            plo, phi = prev[0][nm][0], prev[0][nm][1]
        cut = max(lo, hi) >= 0.20 or (
            comparable and plo is not None and max(lo - plo, hi - phi) >= 0.10
        )
        rows.append((nm, lo, hi, blo, bhi, plo, phi, cut))
    rows.sort(key=lambda r: -max(r[1], r[2]))
    return rows


def _prior_free_key(key):
    """ident の群名から事前分布の部分を伏せる（判定 5 で同じ設定の prior0 と priorX を組にする）。"""
    return re.sub(r"_ident_prior[0-9.]+", "_ident_prior*", key)


def main(root, pattern="*"):
    root = Path(root)
    if not root.is_dir():
        print(f"判定できない: {root} が無い")
        return 1
    groups = defaultdict(list)
    for d in sorted(root.glob(pattern)):
        if not re.search(r"_seed\d+$", d.name):
            continue
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
                # 両端それぞれ 5% の幅なので、事後が箱の中で一様なら 0.10 になる。
                # 0.10〜0.15 は「一様に近い（その成分をデータがほとんど決めていない）」で、
                # 箱で切られているわけではない。片側 5% に 0.20 以上、または前段より
                # 0.1 以上増えた成分が「切られている」（2026-10-08d）。表示だけで判定はしない。
                print(f"          箱の端 5% に 10% 超（一様なら 0.10）: {edgy}")
                # 片側に分けて出す（2026-10-08p §4）。「←」が 08d の停止基準に該当する成分。
                prev = _prev_side_edges(cfg)
                cmpable = bool(prev) and _comparable(prev[1], d.name)
                rows = _edge_flags(_side_edges(rec, s), prev, cmpable)
                if rows:
                    head = "          片側 5%: 成分 箱 下側/上側"
                    if prev:
                        head += f"（前段 {prev[1]} の 下側/上側"
                        head += "" if cmpable else "・尤度が違う段なので増加は見ない"
                        head += "）"
                    print(head)
                    for nm, lo, hi, blo, bhi, plo, phi, cut in rows:
                        line = (
                            f"            {nm:4s} [{blo:g},{bhi:g}] "
                            f"{lo:.2f}/{hi:.2f}"
                        )
                        if plo is not None:
                            line += f" (前段 {plo:.2f}/{phi:.2f})"
                        print(line + ("  ← 箱で切られている" if cut else ""))
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
            "2 粒子の移動(>=2/stage)": all(
                _moves(r) >= 2.0 and r["mean_accept"] <= 0.60 for r in recs
            ),
            "2b 次元あたりの移動(>=5)": all(_moves_per_dim(r) >= 5.0 for r in recs),
            "3 maxlogL 幅<=1": len(maxll) < 2 or float(np.ptp(maxll)) <= 1.0,
            "4 中央値幅<=0.5sd": len(med) < 2 or float(med_spread.max()) <= 0.5,
        }
        worst = [f"{NAMES[free[k]]}({med_spread[k]:.1f})" for k in np.argsort(-med_spread)[:3]]
        print(
            "       1 粒子の移動: "
            + ", ".join(
                f"seed{r['args']['seed']} {_moves(r):.1f}/stage " f"({_moves_per_dim(r):.1f}/次元)"
                for r in recs
            )
        )
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'} {k}")
        print(f"       maxlogL 幅 = {np.ptp(maxll):.2f}、中央値幅/sd 上位: {', '.join(worst)}")
        any_fail |= not all(checks.values())
        maxll_by_group[key] = max(maxll)

    # 5. ident: 事前分布なし >= 事前分布あり（尤度のみ）
    for key, m in maxll_by_group.items():
        if "_ident_prior0" in key:
            for other, m2 in maxll_by_group.items():
                # 事前分布以外（ゲート・RUNTAG など）が同じ群とだけ比べる。
                # startswith で比べると mut80 の群が 10/2 の群と組になっていた
                if other != key and _prior_free_key(other) == _prior_free_key(key):
                    ok = m >= m2 - 0.5
                    any_fail |= not ok
                    print(
                        f"\n{'PASS' if ok else 'FAIL'} 5 {key} の max logL {m:.2f} >= {other} の {m2:.2f} − 0.5"
                        + ("" if ok else "  → 事前分布なしは探索不足。解釈しない")
                    )

    print("\n" + ("FAIL を含む群は回し直すまで解釈しない" if any_fail else "全群 PASS"))
    return 1 if any_fail else 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data_5species/main/_runs/paper_gateoff")
    ap.add_argument(
        "--glob",
        default="*",
        help="判定する run の glob（例 '*mut80*'）。古い run を混ぜて全体を FAIL にしないため",
    )
    a = ap.parse_args()
    sys.exit(main(a.root, a.glob))
