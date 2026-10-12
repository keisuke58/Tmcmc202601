#!/usr/bin/env python3
"""補足資料の計算コスト表（Table S3 / tab:timing の最終段の行）の材料を run_record.json から作る。

条件 (CS/CH/DS/DH) × 段 (pilot/p1/p2/ult) ごとに
  N_p            args.n_particles（seed 間で違えば a/b と並べる）
  K              tempering 段数 n_stages（len(stage_history.beta)）の seed 間中央値
  n_mut          1 段あたりの mutation 回数 args.n_mutation_steps
  wall / seed    total_time_s の seed 間中央値（1 h 未満は min、以上は h）
  N_lik          尤度評価回数 ≈ N_p × (1 + Σ_stages n_mut)（初期化 1 回 + 各 mutation で全粒子 1 回）の中央値
  GPU            run_record に機種の記録は無い（args.device = "gpu" だけ）。--gpu-model で与える。無ければ "--"
欠けている値は "--"。

群の glob は tools/make_supplementary_tables.py の DEFAULT_SPEC を使う（--spec の JSON で上書き可）。
DS/DH の ult は、--ult-glob-DS / --ult-glob-DH（既定 DS_ult_* / DH_ult_*）に一致する run があれば自動で使う。
無ければ p2 が仮の最終段で、表では "(iii)$^\\ast$ prov." と印を付ける。

使い方:
    python3 tools/make_cost_table.py docs/revision/generated/final_runs \\
        --out docs/revision/EML_submission/figures/provisional_2026-10-11/supp/supp_S3_cost.tex
"""

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_supplementary_tables import CONDS, DEFAULT_SPEC, STAGE_LABEL, STAGES  # noqa: E402

DEFAULT_OUT = "docs/revision/EML_submission/figures/provisional_2026-10-11/supp/supp_S3_cost.tex"


def load_records(root, pattern):
    out = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        p = os.path.join(d, "run_record.json")
        if not os.path.isfile(p):
            p = os.path.join(d, "config.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                out.append((os.path.basename(d), json.load(fh)))
    return out


def n_stages(rec):
    sh = rec.get("stage_history") or {}
    if sh.get("beta"):
        return len(sh["beta"])
    return rec.get("n_stages")


def n_lik(rec):
    a = rec.get("args", {})
    npart = a.get("n_particles")
    if npart is None:
        return None
    sh = rec.get("stage_history") or {}
    if sh.get("n_mut"):
        tot = sum(sh["n_mut"])
    elif a.get("n_mutation_steps") and n_stages(rec):
        tot = a["n_mutation_steps"] * n_stages(rec)
    else:
        return None
    return npart * (1 + tot)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def uniq(xs):
    xs = sorted({x for x in xs if x is not None})
    return "/".join(str(x) for x in xs) if xs else None


def fmt_time(s, tex=True):
    if s is None:
        return "--"
    if s < 3600:
        v, u = s / 60, "min"
        txt = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    else:
        v, u = s / 3600, "h"
        txt = f"{v:.1f}"
    return f"{txt}\\,{u}" if tex else f"{txt} {u}"


def fmt_big(n, tex=True):
    if n is None:
        return "--"
    e = len(str(int(n))) - 1
    m = n / 10**e
    return f"${m:.1f}\\times10^{{{e}}}$" if tex else f"{m:.1f}e{e}"


def collect(root, spec, ult_globs, gpu_model):
    rows = []
    for c in CONDS:
        sp = dict(spec.get(c, {}))
        if c in ult_globs:
            sp["ult"] = ult_globs[c]
        has_ult = bool(load_records(root, sp.get("ult", "__none__")))
        for st in STAGES:
            recs = load_records(root, sp.get(st, "__none__"))
            prov = st == "p2" and not has_ult
            if st == "ult" and not recs:
                continue
            row = {
                "cond": c,
                "stage": st,
                "label": STAGE_LABEL[st] + (r"$^\ast$" if prov else ""),
                "prov": prov,
                "glob": sp.get(st),
                "n_seeds": len(recs),
                "N_p": uniq(r.get("args", {}).get("n_particles") for _, r in recs),
                "n_mut": uniq(r.get("args", {}).get("n_mutation_steps") for _, r in recs),
                "K": med(n_stages(r) for _, r in recs),
                "time_s": med(r.get("total_time_s") for _, r in recs),
                "n_lik": med(n_lik(r) for _, r in recs),
                "device": uniq(r.get("args", {}).get("device") for _, r in recs),
                "gpu": gpu_model if recs else None,
            }
            rows.append(row)
    return rows


def k_str(k):
    if k is None:
        return "--"
    return f"{k:g}"


def to_tex(rows):
    # GPU 列は機種が分かるとき（--gpu-model）だけ出す
    gpu_col = any(r["gpu"] for r in rows)
    L = [
        r"\begin{tabular}{@{}llrrrrr" + ("l" if gpu_col else "") + "@{}}",
        r"\toprule",
        r"Cond. & Stage & $N_p$ & $n_{\mathrm{mut}}$ & $K$ & Wall time / seed & $N_{\mathrm{lik}}$" + (" & GPU" if gpu_col else "") + r"\\",
        r"\midrule",
    ]
    prev = None
    for r in rows:
        if prev is not None and r["cond"] != prev:
            L.append(r"\addlinespace")
        cells = [
            r["cond"] if r["cond"] != prev else "",
            r["label"],
            r["N_p"] or "--",
            r["n_mut"] or "--",
            k_str(r["K"]),
            fmt_time(r["time_s"]),
            fmt_big(r["n_lik"]),
        ] + ([r["gpu"] or "--"] if gpu_col else [])
        L.append(" & ".join(cells) + r"\\")
        prev = r["cond"]
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def to_text(rows):
    hdr = f"{'cond':4} {'stage':6} {'seeds':>5} {'N_p':>7} {'n_mut':>6} {'K':>4} {'wall/seed':>10} {'N_lik':>9} {'dev':>4} GPU   glob"
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        out.append(
            f"{r['cond']:4} {r['stage'] + ('*' if r['prov'] else ''):6} {r['n_seeds']:>5} "
            f"{r['N_p'] or '--':>7} {r['n_mut'] or '--':>6} {k_str(r['K']):>4} "
            f"{fmt_time(r['time_s'], tex=False):>10} {fmt_big(r['n_lik'], tex=False):>9} "
            f"{r['device'] or '--':>4} {r['gpu'] or '--':5} {r['glob']}"
        )
    out.append("* = p2 used as provisional final stage (no ult run found)")
    out.append("N_lik = N_p * (1 + sum of n_mut over tempering stages); median over seeds")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="docs/revision/generated/final_runs")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--spec", help="JSON file overriding DEFAULT_SPEC (same format as make_supplementary_tables.py)")
    ap.add_argument("--ult-glob-DS", default="DS_ult_*")
    ap.add_argument("--ult-glob-DH", default="DH_ult_*")
    ap.add_argument("--gpu-model", default=None, help='e.g. "RTX\\,4090"; not recorded in run_record.json')
    a = ap.parse_args()

    spec = json.loads(json.dumps(DEFAULT_SPEC))
    if a.spec:
        with open(a.spec, encoding="utf-8") as fh:
            for c, d in json.load(fh).items():
                spec.setdefault(c, {}).update(d)
    rows = collect(a.root, spec, {"DS": a.ult_glob_DS, "DH": a.ult_glob_DH}, a.gpu_model)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("% generated by tools/make_cost_table.py -- do not edit by hand\n")
        fh.write(to_tex(rows))
    print(to_text(rows))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
