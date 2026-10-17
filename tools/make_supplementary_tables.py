#!/usr/bin/env python3
"""補足資料（BMB_supplementary.tex）の表 S1・S2・S3 を run から作る。

  S1  事前分布の箱（条件 × 段、自由次元だけ）と ln V = Σ ln(u−l)。pilot の箱から広げた成分は太字
  S2  run の一覧（粒子数・mutation・段数・受理率・移動/次元・max logL・seed 間の幅・中央値の seed 間の幅 / sd（判定 4）・時間）
  S3  Bayes 因子（pilot の full と a45=0、ln Z の平均 ± 幅、ln B、a45 の箱の幅）
  S4  同定性（最終段 (iv) の 15 成分 × 4 条件: 3 seed 合算の中央値 [5%, 95%]。
      事後 sd が箱の一様分布の sd の 0.8 倍以上の成分は灰色。そのうち 90% 区間が 0 を含むものは n.i.（同定されていない）、
      含まないものは s.o.（sign only: 大きさは決まらないが符号は決まる）。
      判定 4 の規則（check_paper_runs.py の UNIDENT_RATIO）と同じ）

run は config.json（run_record の上位集合）を読む。glob は --spec の JSON で上書きできる:
  {"DH": {"pilot": "DH_pilot_mut80_seed*", "p1": ..., "p2": ..., "ult": ..., "a45zero": "DH_pilot_a45zero_seed*"}, ...}

使い方（GPU サーバー、ult が終わったあと）:
    python3 tools/make_supplementary_tables.py data_5species/main/_runs/paper_gateoff \\
        --out-dir docs/revision/generated
出力: supp_S1_boxes.tex / supp_S2_runs.tex / supp_S3_bayes.tex / supp_S4_ident.tex（補足資料から \\input する）
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

# 2026-10-16: 生データ・2 通路の群（figures/raw/spec_raw.json と同じ）。読み取りデータの run は
# docs/revision/generated/final_runs_digitised/ に分けた（その群は 10-10 の spec を --spec で渡す）。
# a45zero（ベイズ因子）は生データでの再計算が済んだら足す。変わったら --spec で上書き
DEFAULT_SPEC = {
    "CS": {"pilot": "CS_pilot_2c_raw_seed*", "p1": "CS_p1_2c_raw_seed*", "p2": "CS_p2_2c_raw_seed*", "ult": "CS_ult_2c_raw_seed*"},
    "CH": {"pilot": "CH_pilot_2c_raw_seed*", "p1": "CH_p1_2c_raw_seed*", "p2": "CH_p2_2c_raw_a33w_seed*", "ult": "CH_ult_2c_raw_a33w_seed*"},
    "DS": {"pilot": "DS_pilot_2c_raw_seed*", "p1": "DS_p1_2c_raw_seed*", "p2": "DS_p2_2c_raw_seed*", "ult": "DS_ult_2c_raw_seed*"},
    "DH": {"pilot": "DH_pilot_2c_raw_seed*", "p1": "DH_p1_2c_raw_seed*", "p2": "DH_p2_2c_raw_seed*", "ult": "DH_ult_2c_raw_seed*"},
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
            rec["_root"] = root
            out.append((os.path.basename(d), rec))
    return out


def tex_name(n):
    return "$a_{%s}$" % n[1:] if n.startswith("a") else "$b_{%s}$" % n[1:]


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def _apply_override(base, spec):
    """estimator の --override-bounds（"idx:lo:hi,..."）を箱に当てる。"""
    pb = base.copy()
    for part in (spec or "").replace(";", ",").split(","):
        if part.strip():
            k, lo, hi = part.split(":")
            pb[int(k)] = [float(lo), float(hi)]
    return pb


def _fmt_box(lo, hi):
    f = lambda x: ("%.2f" % x).rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else "%d" % round(x)
    return "[%s, %s]" % (f(lo), f(hi))


def box_table(groups):
    """S4（補足の表番号）: 条件ごとに 4 段の箱を 2×2 に並べる。

    (i)  pilot の箱（初期の箱）
    (ii) (i) の事後の ±4SD に絞った箱（p1 の run の箱）
    (iii) (i) の箱に戻し、端に質量が積んだ成分だけ広げた箱（最終群の override_bounds を (i) に当てる）。広げた成分は太字
    (iv) (iii) の事後の ±4SD（最終群のうち max logL 最大の seed の箱）。最終群が p2 なら --（仮）
    """
    blocks = []
    for tag in CONDS:
        g = groups[tag]
        if not g.get("pilot") or not g.get("ult"):
            continue
        rec_i = g["pilot"][0][1]
        free = list(rec_i["free_dims"])
        b1 = np.array(rec_i["prior_bounds_final"], dtype=float)
        b2 = np.array(g["p1"][0][1]["prior_bounds_final"], dtype=float) if g.get("p1") else None
        fin = max(g["ult"], key=lambda t: t[1].get("max_logL", -np.inf))
        fin_is_ult = "_ult_" in fin[0]
        b3 = _apply_override(b1, fin[1]["args"].get("override_bounds"))
        b4 = np.array(fin[1]["prior_bounds_final"], dtype=float) if fin_is_ult else None
        rows = [r"\multicolumn{5}{@{}l}{\textbf{%s}%s}\\" % (tag, "" if fin_is_ult else r" \textit{(stage (iii) is final, provisional)}"),
                r"Entry & (i) & (ii) & (iii) & (iv)\\", r"\midrule"]
        for i in free:
            c3 = _fmt_box(*b3[i])
            if not np.allclose(b3[i], b1[i]):
                c3 = r"\textbf{%s}" % c3
            rows.append(" & ".join([tex_name(NAMES[i]), _fmt_box(*b1[i]),
                                    _fmt_box(*b2[i]) if b2 is not None else "--", c3,
                                    _fmt_box(*b4[i]) if b4 is not None else "--"]) + r"\\")
        lnv = lambda b: fmt(float(np.sum(np.log(b[free, 1] - b[free, 0]))), 1) if b is not None else "--"
        rows.append(r"\midrule")
        rows.append(" & ".join([r"$\ln V$", lnv(b1), lnv(b2), lnv(b3), lnv(b4)]) + r"\\")
        blocks.append("\\begin{tabular}{@{}lllll@{}}\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    out = []
    for k, blk in enumerate(blocks):
        out.append(r"\begin{minipage}[t]{0.49\textwidth}\centering\scriptsize\setlength{\tabcolsep}{2pt}" + "\n" + blk + "\n" + r"\end{minipage}" + ("\\hfill" if k % 2 == 0 else "\\par\\medskip"))
    return "\n".join(out)


def moves_per_dim(rec):
    m = rec.get("moves_per_particle_mean")
    nf = max(len(rec.get("free_dims") or []), 1)
    return None if m is None else float(m) * rec["n_stages"] / nf


def median_spread(g):
    """判定 4 の量: seed 間の中央値の幅 / プールした事後 sd の最大値（同定された自由次元だけ、山分けなし）。
    seed が 1 つしか無い・samples.npy が無いときは None。"""
    S = []
    for d, rec in g:
        f = os.path.join(rec.get("_root", ""), d, "samples.npy")
        if os.path.isfile(f):
            S.append(np.load(f))
    if len(S) < 2:
        return None
    rec = g[0][1]
    free = list(rec["free_dims"])
    pb = np.array(rec["prior_bounds_final"], dtype=float)
    pooled = np.concatenate(S)
    out = 0.0
    for i in free:
        sd = pooled[:, i].std()
        unif = (pb[i, 1] - pb[i, 0]) / np.sqrt(12.0)
        if sd <= 0 or sd >= UNIDENT_RATIO * unif:
            continue
        med = [np.median(x[:, i]) for x in S]
        out = max(out, (max(med) - min(med)) / sd)
    return out


def run_table(groups):
    """S2: 条件 × 段ごとに 1 行（3 seed をまとめる）。"""
    lines = [
        r"Cond. & Stage & $N_p$ & $n_{\mathrm{mut}}$ & $K$ & Accept. & Moves/dim & $\max\log L$ (spread) & Med.\ spread & Hours\\",
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
            msp = median_spread(g)
            lines.append(
                " & ".join(
                    [
                        tag,
                        # 仮の段: 最終群が p2 の run（ult がまだ）のときは (iii) と表示する
                        "(iii)" if st == "ult" and "_p2_" in g[0][0] else STAGE_LABEL[st],
                        "/".join(str(n) for n in npart),
                        "/".join(str(n) for n in nmut),
                        "%d--%d" % (min(nst), max(nst)) if min(nst) != max(nst) else str(nst[0]),
                        "%.2f--%.2f" % (min(acc), max(acc)) if acc else "--",
                        "%.1f--%.1f" % (min(mpd), max(mpd)) if mpd else "--",
                        "%.2f (%.2f)" % (max(ml), max(ml) - min(ml)),
                        "--" if msp is None else "%.2f" % msp,
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
        fk = "bayes_full" if groups[tag].get("bayes_full") else "pilot"  # 生データ: ベイズ因子用の n_mut 120 の pilot（16k）
        full = [r.get("log_evidence") for _, r in groups[tag].get(fk, [])]
        zero = [r.get("log_evidence") for _, r in groups[tag].get("a45zero", [])]
        full = [x for x in full if x is not None]
        zero = [x for x in zero if x is not None]
        if not full or not zero:
            lines.append("%s & \\multicolumn{5}{l}{(pending)}\\\\" % tag)
            continue
        pb = np.array(groups[tag][fk][0][1]["prior_bounds_final"], dtype=float)
        lo, hi = pb[A45]
        pm = lambda v: ("$%.2f$ (one seed)" % v[0]) if len(v) == 1 else "$%.2f \\pm %.2f$" % (np.mean(v), (max(v) - min(v)) / 2)
        lines.append(
            " & ".join(
                [
                    tag,
                    pm(full),
                    pm(zero),
                    "$%+.2f$" % (np.mean(full) - np.mean(zero)),
                    "[%g, %g]" % (lo, hi),
                    fmt(float(np.log(hi - lo)), 2),
                ]
            )
            + r"\\"
        )
    return "\n".join(lines)


UNIDENT_RATIO = 0.8  # tools/check_paper_runs.py と同じ
# 本文の並び（式の blocks と同じ）
ORDER = [
    "a11",
    "a12",
    "a22",
    "a33",
    "a34",
    "a44",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "a15",
    "a25",
    "a35",
    "a45",
]


def ident_table(root, spec):
    """S4: 最終段の 15 成分 × 4 条件。3 seed 合算の中央値 [5%, 95%]、同定されていない成分は n.i."""
    cols = {}
    for tag in CONDS:
        pat = spec.get(tag, {}).get("final") or spec.get(tag, {}).get("ult")  # "final": 最終群が ult でない条件（DH は p2、10-16）
        dirs = sorted(glob.glob(os.path.join(root, pat))) if pat else []
        S, pb = [], None
        for d in dirs:
            f = os.path.join(d, "samples.npy")
            rec = load(d)
            if os.path.isfile(f) and rec:
                S.append(np.load(f))
                pb = np.array(rec["prior_bounds_final"], dtype=float)
        cols[tag] = (np.concatenate(S), pb) if S else None
    lines = [" & ".join(["Entry"] + CONDS) + r"\\", r"\midrule"]
    for nm in ORDER:
        i = NAMES.index(nm)
        cells = [tex_name(nm)]
        for tag in CONDS:
            c = cols[tag]
            if c is None:
                cells.append("--")
                continue
            x, pb = c
            q05, q50, q95 = np.percentile(x[:, i], [5, 50, 95])
            unif = (pb[i, 1] - pb[i, 0]) / np.sqrt(12.0)
            txt = "$%+.2f$ [$%+.1f$, $%+.1f$]" % (q50, q05, q95)
            if x[:, i].std() >= UNIDENT_RATIO * unif:
                # 広がりは箱並みでも 90% 区間が 0 を含まなければ「符号だけ決まる」（s.o.）
                tag_ = "s.o." if (q05 > 0 or q95 < 0) else "n.i."
                txt = r"\textcolor{gray}{%s %s}" % (txt, tag_)
            cells.append(txt)
        lines.append(" & ".join(cells) + r"\\")
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
    COLS = {
        "supp_S1_boxes": "lllll",
        "supp_S2_runs": "llrrrllllr",
        "supp_S3_bayes": "lccccc",
        "supp_S4_ident": "lcccc",
    }
    for name, body in [
        ("supp_S1_boxes", box_table(groups)),
        ("supp_S2_runs", run_table(groups)),
        ("supp_S3_bayes", bayes_table(groups)),
        ("supp_S4_ident", ident_table(a.root, spec)),
    ]:
        p = os.path.join(a.out_dir, name + ".tex")
        with open(p, "w", encoding="utf-8") as fh:
            if name == "supp_S1_boxes":  # 4 条件を 2×2 に並べた完成品（tabular を中に持つ）
                fh.write("% generated by tools/make_supplementary_tables.py\n" + body + "\n")
                print("wrote", p)
                continue
            fh.write(
                "% generated by tools/make_supplementary_tables.py\n"
                + "\\begin{tabular}{%s}\\toprule\n" % COLS[name]
                + body
                + "\n\\bottomrule\\end{tabular}\n"
            )
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
