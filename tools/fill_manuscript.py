#!/usr/bin/env python3
"""paper_numbers.json（make_paper_figures.py の出力）から、原稿が \\input する数値マクロと表を作る。

出力（--out-dir、既定 docs/revision/generated/）:
  numbers.tex      \\newcommand の集まり。原稿の本文はこのマクロを使う（\\numRmseDH など）
  tab_rmse.tex     Table 3（RMSE・MAE・受理率・max logL・ln Z、Phase 1 | Phase 2）の tabular
  tab_r2.tex       Table 4（種ごとの R²）
  tab_pairwise.tex pairwise（ρ・Δ_F・d_15D）
  tab_cross.tex    cross-condition prediction RMSE
  (numbers.tex の \\numTimingRows) Table 2 の最終段の行（GPU 時間、CPU は 500 粒子・K=10 の benchmark から N_p と K に比例で外挿）

json が無い成分はマクロを \\TBD{...} のまま出すので、原稿は常にコンパイルできる。
pH の答え合わせ・gingipain の相関など json に無い数は --extra の JSON で渡す:
  {"ph_r2": 0.78, "ph_rmse": 0.13, "ph_n_samples": 500, "ph_n_points": 12, "gingipain_r": 0.90}

使い方:
    python3 tools/fill_manuscript.py docs/revision/generated/paper_numbers.json [--extra extra.json]
"""

import argparse
import json
import os
import sys

CONDS = ["CS", "CH", "DS", "DH"]
SPECIES = ["So", "An", "Vei", "Fn", "Pg"]
# CPU benchmark（原稿 Table 2）: DH, N_p=500, K=10, 2400 s / stage
CPU_S_PER_STAGE, CPU_NP, CPU_K = 2400.0, 500, 10


def tbd(x):
    return "\\TBD{%s}" % x


def f(x, nd=2, sign=False):
    if x is None:
        return tbd("")
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def rng(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return tbd("")
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if abs(hi - lo) < 10 ** (-nd) else f"{lo:.{nd}f}--{hi:.{nd}f}"


def macros(num, extra):
    M = {}
    p2 = num.get("phase2", {})
    post = num.get("posterior_phase2", {})
    ko = num.get("knockout", {})
    pc = num.get("phase_consistency", {})
    cf = num.get("channel_fit_phase2", {})
    # 当てはまり
    M["RmseRangePhaseTwo"] = rng([p2.get(t, {}).get("rmse") for t in CONDS], 2)
    for t in CONDS:
        M["Rmse" + t] = f(p2.get(t, {}).get("rmse"), 3)
        for k, kn in (
            ("a55", "PgPg"),
            ("a45", "FnPg"),
            ("a44", "FnFn"),
            ("a12", "SoAn"),
            ("a13", "SoVei"),
        ):
            v = post.get(t, {}).get(k, {})
            M["Map" + kn + t] = f(v.get("map"), 2, sign=True) if v else tbd("")
            M["Qlo" + kn + t] = f(v.get("q05"), 1, sign=True) if v else tbd("")
            M["Qhi" + kn + t] = f(v.get("q95"), 1, sign=True) if v else tbd("")
            M["Qmed" + kn + t] = f(v.get("q50"), 1, sign=True) if v else tbd("")
        M["PhaseR" + t] = f(pc.get(t, {}).get("r"), 2)
    M["PhaseRCommensalRange"] = rng([pc.get(t, {}).get("r") for t in ("CS", "CH")], 2)
    M["ViabRmseRange"] = rng([cf.get(t, {}).get("viability_rmse") for t in CONDS], 2)
    M["AcceptRange"] = rng(
        [
            100 * p2[t]["mean_accept"]
            for t in CONDS
            if t in p2 and p2[t].get("mean_accept") is not None
        ],
        0,
    )
    # ノックアウト（DH）
    k = ko.get("DH")
    if k:
        M["SurgeBase"] = rng([100 * x for x in k["frac_surge_base"]], 0)
        M["SurgeNoFn"] = rng([100 * x for x in k["frac_surge_noFn"]], 0)
        M["SurgeAzero"] = rng([100 * x for x in k["frac_surge_a45_0"]], 0)
        M["ProbSuppressed"] = rng(k["prob_suppressed"], 2)
    else:
        for n, old in [
            ("SurgeBase", "99--100"),
            ("SurgeNoFn", "10--13"),
            ("SurgeAzero", "2--6"),
            ("ProbSuppressed", "0.87--0.90"),
        ]:
            M[n] = tbd(old)
    # 計算時間（最終段）
    tm = num.get("timing", {})
    hrs = [
        tm[f"phase2_{t}"]["hours_mean"]
        for t in CONDS
        if f"phase2_{t}" in tm and tm[f"phase2_{t}"]["hours_mean"]
    ]
    M["UltHoursPerCond"] = rng(hrs, 1)
    M["UltHoursAll"] = f(sum(hrs), 1) if len(hrs) == 4 else tbd("h")
    devs = sorted({d for t in CONDS for d in tm.get(f"phase2_{t}", {}).get("devices", [])})
    M["UltGpu"] = ", ".join(devs) if devs else tbd("model")
    # 外部の数（pH の答え合わせなど）
    for key, name, old in [
        ("ph_r2", "PhRtwo", "0.78"),
        ("ph_rmse", "PhRmse", "0.13"),
        ("ph_n_samples", "PhNsamples", "500"),
        ("ph_n_points", "PhNpoints", "12"),
        ("gingipain_r", "GingipainR", "0.90"),
        ("zenodo_doi", "ZenodoDoi", "DOI"),
    ]:
        M[name] = str(extra[key]) if key in extra else tbd(old)
    return M


def tab_rmse(num):
    L = [
        r"\begin{tabular}{@{}l ccccc ccccc@{}}",
        r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{Phase 1: composition only}} & \multicolumn{5}{c}{\textbf{Phase 2: with viability --- final}}\\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"\textbf{Cond.} & \textbf{RMSE} & \textbf{MAE} & $\bar{\alpha}$ & $\max\ln\mathcal{L}$ & $\ln\hat{Z}$ & \textbf{RMSE} & \textbf{MAE} & $\bar{\alpha}$ & $\max\ln\mathcal{L}$ & $\ln\hat{Z}$\\",
        r"\midrule",
    ]
    for t in CONDS:
        cells = [t]
        for ph in ("phase1", "phase2"):
            p = num.get(ph, {}).get(t)
            cells += (
                [tbd("")] * 5
                if p is None
                else [
                    f(p["rmse"], 3),
                    f(p["mae"], 3),
                    f(p["mean_accept"], 2),
                    "$%s$" % f(p["max_logL"], 1),
                    "$%s$" % f(p.get("log_evidence"), 1),
                ]
            )
        L.append(" & ".join(cells) + r"\\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"])


def tab_r2(num):
    L = [
        r"\begin{tabular}{@{}l ccccc ccccc@{}}",
        r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{Phase 1 (composition only)}} & \multicolumn{5}{c}{\textbf{Phase 2 (with viability) --- final}}\\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"\textbf{Cond.} & So & An & Vei & Fn & Pg & So & An & Vei & Fn & Pg\\",
        r"\midrule",
    ]
    for t in CONDS:
        cells = [t]
        for ph in ("phase1", "phase2"):
            p = num.get(ph, {}).get(t)
            if p is None:
                cells += [tbd("")] * 5
            else:
                for v in p["r2"]:
                    cells.append(
                        "$-$"
                        if v is None
                        else (r"$\mathbf{%.2f}$" % v if v > 0.8 else "$%.2f$" % v)
                    )
        L.append(" & ".join(cells) + r"\\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"])


def tab_pairwise(num):
    L = [
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{C1} & \textbf{C2} & $\rho$ & $\Delta_F$ & $d_{15\mathrm{D}}$\\",
        r"\midrule",
    ]
    for r in num.get("pairwise", []):
        L.append(
            f"{r['c1']} & {r['c2']} & ${r['rho']:+.2f}$ & ${r['delta_F']:.2f}$ & ${r['d15']:.2f}$\\\\"
        )
    if not num.get("pairwise"):
        L.append(r"\multicolumn{5}{l}{\TBD{pairwise}}\\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"])


def tab_cross(num):
    cp = num.get("cross_prediction", {})
    L = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Source $\backslash$ Target & CS & CH & DS & DH \\",
        r"\midrule",
    ]
    for s in CONDS:
        cells = [s]
        for t in CONDS:
            v = cp.get(s, {}).get(t)
            cells.append(tbd("") if v is None else (r"\textbf{%.3f}" % v if s == t else "%.3f" % v))
        L.append(" & ".join(cells) + r"\\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"])


def tab_timing(num):
    """Table 2 の最終段の 2 行。CPU は benchmark から N_p・K・段数に比例で外挿。"""
    tm = num.get("timing", {})
    runs = [r for r in num.get("runs", []) if r["stage"] == "phase2"]
    rows = []
    gpu_h, cpu_h = [], []
    for t in CONDS:
        rs = [r for r in runs if r["cond"] == t]
        x = tm.get(f"phase2_{t}")
        if not rs or not x or not x.get("hours_mean"):
            continue
        r0 = rs[0]
        cpu = (
            CPU_S_PER_STAGE
            * (r0["n_particles"] / CPU_NP)
            * (r0["n_mutation_steps"] / CPU_K)
            * r0["n_stages"]
            / 3600
        )
        gpu_h.append(x["hours_mean"])
        cpu_h.append(cpu)
    if len(gpu_h) != 4:
        return (
            r"\quad Per condition & \TBD{h}$^\dagger$ & \TBD{h} & \TBD{x}\\"
            + "\n"
            + r"\quad All 4 cond.   & \TBD{h}$^\dagger$ & \TBD{h} & \TBD{x}\\"
        )
    per = r"\quad Per condition & %s\,h$^\dagger$ & %s\,h & $%d\times$\\" % (
        rng(cpu_h, 0),
        rng(gpu_h, 1),
        round(sum(cpu_h) / sum(gpu_h)),
    )
    allc = r"\quad All 4 cond.   & %.0f\,h$^\dagger$ & %.1f\,h & $%d\times$\\" % (
        sum(cpu_h),
        sum(gpu_h),
        round(sum(cpu_h) / sum(gpu_h)),
    )
    return per + "\n" + allc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("numbers", nargs="?", default="docs/revision/generated/paper_numbers.json")
    ap.add_argument("--extra")
    ap.add_argument("--out-dir", default="docs/revision/generated")
    a = ap.parse_args()
    num = {}
    if os.path.isfile(a.numbers):
        with open(a.numbers, encoding="utf-8") as fh:
            num = json.load(fh)
    else:
        print(f"[warn] {a.numbers} が無いので仮置き（\\TBD）で出す", file=sys.stderr)
    extra = {}
    if a.extra:
        with open(a.extra, encoding="utf-8") as fh:
            extra = json.load(fh)
    os.makedirs(a.out_dir, exist_ok=True)
    M = macros(num, extra)
    M["TimingRows"] = tab_timing(num)
    with open(os.path.join(a.out_dir, "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by tools/fill_manuscript.py\n")
        for k, v in M.items():
    for name, body in [
        ("tab_rmse", tab_rmse(num)),
        ("tab_r2", tab_r2(num)),
        ("tab_pairwise", tab_pairwise(num)),
        ("tab_cross", tab_cross(num)),
        ("tab_timing", tab_timing(num)),
    ]:
        with open(os.path.join(a.out_dir, name + ".tex"), "w", encoding="utf-8") as fh:
            fh.write("% generated by tools/fill_manuscript.py\n" + body + "\n")
    print("wrote numbers.tex + 5 tables in", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
