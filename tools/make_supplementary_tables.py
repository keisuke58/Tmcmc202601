#!/usr/bin/env python3
"""補足資料（BMB_supplementary.tex）の表 S1・S2・S3 を run から作る。

  S1  事前分布の箱（条件 × 段、自由次元だけ）と ln V = Σ ln(u−l)。pilot の箱から広げた成分は太字
  S2  run の一覧（粒子数・mutation・段数・受理率・移動/次元・max logL・seed 間の幅・時間）
  S3  Bayes 因子（pilot の full と a45=0、ln Z の平均 ± 幅、ln B、a45 の箱の幅）

run は config.json（run_record の上位集合）を読む。glob は --spec の JSON で上書きできる:
  {"DH": {"pilot": "DH_pilot_mut80_seed*", "p1": ..., "p2": ..., "ult": ..., "a45zero": "DH_pilot_a45zero_seed*"}, ...}

使い方（GPU サーバー、ult が終わったあと）:
    python3 tools/make_supplementary_tables.py data_5species/main/_runs/paper_gateoff \\
        --out-dir docs/revision/generated
出力: supp_S1_boxes.tex / supp_S2_runs.tex / supp_S3_bayes.tex（tabular 本体。補足資料から \\input する）
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# θ の並び（tools/check_paper_runs.py の NAMES と同じ）
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
A45 = 19
CONDS = ["CS", "CH", "DS", "DH"]
STAGES = ["pilot", "p1", "p2", "ult"]
STAGE_LABEL = {"pilot": "(i)", "p1": "(ii)", "p2": "(iii)", "ult": "(iv)"}

# 2026-10-09 時点の計画（09d / 09f）。変わったら --spec で上書き
DEFAULT_SPEC = {
    "CS": {
        "pilot": "CS_pilot_mut80_seed*",
        "p1": "CS_p1_mut120_seed*",
        "p2": "CS_p2_mut160_wide2_seed*",
        "ult": "CS_ult_wide2_sd4_seed*",
        "a45zero": "CS_pilot_a45zero_m120_seed*",
    },
    "CH": {
        "pilot": "CH_pilot_mut80_seed*",
        "p1": "CH_p1_mut150_seed*",
        "p2": "CH_p2_mut150_wide2_noph_seed*",
        "ult": "CH_ult_mut150_wide2_noph_sd4_seed*",
        "a45zero": "CH_pilot_a45zero_m120_seed*",
    },
    "DS": {
        "pilot": "DS_pilot_wide80_seed*",
        "p1": "DS_p1_wide80_p3k_seed*",
        "p2": "DS_p2_wide80_p12k_wide2_seed*",
        "ult": "DS_ult_wide80_p12k_wide2_sd4_seed*",
        "a45zero": "DS_pilot_a45zero_seed*",
    },
    "DH": {
        "pilot": "DH_pilot_mut80_seed*",
        "p1": "DH_p1_mut80_seed*",
        "p2": "DH_p2_nonarrow_w2_noph_p10k_seed*",
        "ult": "DH_ult_nonarrow_w2_noph_p10k_sd4_seed*",
        "a45zero": "DH_pilot_a45zero_seed*",
    },
}


def load(run_dir):
    for name in ("config.json", "run_record.json"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def runs(root, pattern):
    out = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        rec = load(d)
        if rec is not None and rec.get("prior_bounds_final"):
            out.append((os.path.basename(d), rec))
    return out


def tex_name(n):
    return "$a_{%s}$" % n[1:] if n.startswith("a") else "$b_{%s}$" % n[1:]


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def box_table(groups):
    """S1: 条件ごとに段 × 自由次元の箱。pilot と違う箱は太字。"""
    lines = []
    for tag in CONDS:
        stages = [(st, groups[tag].get(st)) for st in STAGES]
        stages = [(st, g) for st, g in stages if g]
        if not stages:
            continue
        rec0 = stages[0][1][0][1]
        free = list(rec0["free_dims"])
        base = np.array(rec0["prior_bounds_final"], dtype=float)
        lines.append(r"\multicolumn{%d}{@{}l}{\textbf{%s}}\\" % (len(stages) + 1, tag))
        lines.append(
            " & ".join(["Entry"] + ["Stage " + STAGE_LABEL[st] for st, _ in stages]) + r"\\"
        )
        lines.append(r"\midrule")
        for i in free:
            cells = [tex_name(NAMES[i])]
            for st, g in stages:
                pb = np.array(g[0][1]["prior_bounds_final"], dtype=float)
                s = "[%g, %g]" % (pb[i, 0], pb[i, 1])
                if st != "ult" and not np.allclose(pb[i], base[i]):
                    s = r"\textbf{%s}" % s
                if st == "ult":
                    s = r"$\pm4$\,SD"  # 段 (iv) は事後の ±4SD（seed ごとに違う）
                cells.append(s)
            lines.append(" & ".join(cells) + r"\\")
        cells = [r"$\ln V$"]
        for st, g in stages:
            pb = np.array(g[0][1]["prior_bounds_final"], dtype=float)
            w = pb[free, 1] - pb[free, 0]
            cells.append(
                fmt(float(np.sum(np.log(w))), 1) if np.all(w > 0) and st != "ult" else "--"
            )
        lines.append(" & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    return "\n".join(lines)


def moves_per_dim(rec):
    m = rec.get("moves_per_particle_mean")
    nf = max(len(rec.get("free_dims") or []), 1)
    return None if m is None else float(m) * rec["n_stages"] / nf


def run_table(groups):
    """S2: 条件 × 段ごとに 1 行（3 seed をまとめる）。"""
    lines = [
        r"Cond. & Stage & $N_p$ & $K$ & Stages & Accept. & Moves/dim & $\max\log L$ (spread) & Hours\\",
        r"\midrule",
    ]
    for tag in CONDS:
        for st in STAGES:
            g = groups[tag].get(st)
            if not g:
                continue
            recs = [r for _, r in g]
            npart = sorted({r["args"]["n_particles"] for r in recs})
            nmut = sorted({r["args"]["n_mutation_steps"] for r in recs})
            nst = [r["n_stages"] for r in recs]
            acc = [r.get("mean_accept") for r in recs if r.get("mean_accept") is not None]
            mpd = [moves_per_dim(r) for r in recs if moves_per_dim(r) is not None]
            ml = [r["max_logL"] for r in recs]
            hrs = [r["total_time_s"] / 3600 for r in recs if r.get("total_time_s")]
            lines.append(
                " & ".join(
                    [
                        tag,
                        STAGE_LABEL[st],
                        "/".join(str(n) for n in npart),
                        "/".join(str(n) for n in nmut),
                        "%d--%d" % (min(nst), max(nst)) if min(nst) != max(nst) else str(nst[0]),
                        "%.2f--%.2f" % (min(acc), max(acc)) if acc else "--",
                        "%.1f--%.1f" % (min(mpd), max(mpd)) if mpd else "--",
                        "%.2f (%.2f)" % (max(ml), max(ml) - min(ml)),
                        "%.1f--%.1f" % (min(hrs), max(hrs)) if hrs else "--",
                    ]
                )
                + r"\\"
            )
    return "\n".join(lines)


def bayes_table(groups):
    """S3: pilot の full と a45=0 の ln Z（平均 ± 幅）、ln B、a45 の箱の幅。"""
    lines = [
        r"Cond. & $\ln\hat Z_{\mathrm{full}}$ & $\ln\hat Z_{a_{45}=0}$ & $\ln B$ & $a_{45}$ box & $\ln(\text{width})$\\",
        r"\midrule",
    ]
    for tag in CONDS:
        full = [r.get("log_evidence") for _, r in groups[tag].get("pilot", [])]
        zero = [r.get("log_evidence") for _, r in groups[tag].get("a45zero", [])]
        full = [x for x in full if x is not None]
        zero = [x for x in zero if x is not None]
        if not full or not zero:
            lines.append("%s & \\multicolumn{5}{l}{(pending)}\\\\" % tag)
            continue
        pb = np.array(groups[tag]["pilot"][0][1]["prior_bounds_final"], dtype=float)
        lo, hi = pb[A45]
        lines.append(
            " & ".join(
                [
                    tag,
                    "$%.2f \\pm %.2f$" % (np.mean(full), (max(full) - min(full)) / 2),
                    "$%.2f \\pm %.2f$" % (np.mean(zero), (max(zero) - min(zero)) / 2),
                    "$%+.2f$" % (np.mean(full) - np.mean(zero)),
                    "[%g, %g]" % (lo, hi),
                    fmt(float(np.log(hi - lo)), 2),
                ]
            )
            + r"\\"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root")
    ap.add_argument("--spec", help="JSON file: 条件 → 段 → glob")
    ap.add_argument("--out-dir", default="docs/revision/generated")
    a = ap.parse_args()
    spec = DEFAULT_SPEC
    if a.spec:
        with open(a.spec, encoding="utf-8") as fh:
            spec = json.load(fh)
    groups = {
        tag: {st: runs(a.root, pat) for st, pat in spec.get(tag, {}).items()} for tag in CONDS
    }
    for tag in CONDS:
        for st, g in groups[tag].items():
            n = len(g)
            if 0 < n < 3:
                print(f"[warn] {tag} {st}: {n} run しか無い（{spec[tag][st]}）", file=sys.stderr)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, body in [
        ("supp_S1_boxes", box_table(groups)),
        ("supp_S2_runs", run_table(groups)),
        ("supp_S3_bayes", bayes_table(groups)),
    ]:
        p = os.path.join(a.out_dir, name + ".tex")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("% generated by tools/make_supplementary_tables.py\n" + body + "\n")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
