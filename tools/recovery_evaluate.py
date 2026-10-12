#!/usr/bin/env python3
"""回復テストの評価（共著者レビュー B2、2026-10-12h）。

条件ごとに「合成データ（真値つき）」と「その合成データで回した推定の出力（3 seed）」を受け取り、
15 自由成分それぞれについて:
  真値・事後中央値・90% 区間（5〜95%）・真値を含むか・誤差（|中央値 − 真値| / 箱の幅）・
  事後 sd / 箱の一様分布の sd・合成データでの分類・実データ（最終 run 群）での分類
を出す。分類は補足 S4 と同じ規則（sd 比 < 0.8 → 同定、それ以外で 90% 区間が 0 を含まない → 符号のみ、他は n.i.）。

まとめ:
  (a) 実データで「同定」の成分: 真値を 90% 区間に含む数、誤差の最大
  (b) 実データで「n.i.」の成分: 合成データでも n.i.（または符号のみ）になる数
  (c) 全成分の被覆率、分類の一致数（3×3 の対応表）

出力: <out>.json、<out>.tex（補足に入れる表の本体。booktabs）、<out>.pdf / .png（箱の中の区間と真値）

  python3 tools/recovery_evaluate.py \\
      --case CH=data_5species/main/_runs/recovery/CH_syn_n1_r0/synthetic.json='data_5species/main/_runs/recovery/CH_syn_n1_r0_seed*' \\
      --case CS=...=... --out docs/revision/EML_submission/figures/provisional_2026-10-11/recovery

  注意: 真値が固定の 1 回の合成データでは、90% 区間の被覆は成分ごとに独立な 0.9 の試行ではない（成分間に相関があり、
  ベイズの被覆は事前分布で平均したときの性質）。被覆率は目安として読み、主張は (a)(b) の分類の一致で行う。
"""
import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_common as RC  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--case", action="append", required=True,
                help="TAG=<synthetic.json>=<回復 run の glob>（条件ごとに 1 つ。複数可）")
ap.add_argument("--real-root", default=None, help="実データの最終 run 群の置き場所（既定は合成データに記録された場所）")
ap.add_argument("--out", required=True, help="出力の接頭辞（.json/.tex/.pdf/.png を付ける）")
ap.add_argument("--small-err", type=float, default=0.10, help="「誤差が小さい」の閾値（|中央値−真値| / 箱の幅）")
args = ap.parse_args()

try:
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{lmodern}", "font.size": 9})
    plt.figure(); plt.text(0, 0, r"$a_{11}$"); plt.savefig(os.devnull, format="pdf"); plt.close()
    USETEX = True
except Exception:  # noqa: BLE001  LaTeX が無ければ通常の描画
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 9})
    USETEX = False
plt.close("all")

SP = {"1": "So", "2": "An", "3": "Vei", "4": "Fn", "5": "Pg"}
COND_COL = {"CS": "#1f77b4", "CH": "#2ca02c", "DS": "#ff7f0e", "DH": "#d62728"}
CLS = ["identified", "sign-only", "n.i."]


def label(nm):
    return f"{SP[nm[1]]}--{SP[nm[2]]}" if nm[1] != nm[2] else f"{SP[nm[1]]} (self)"


results = {}
for spec in args.case:
    tag, syn_path, run_glob = spec.split("=", 2)
    syn = json.load(open(syn_path))
    assert syn["tag"] == tag, (syn["tag"], tag)
    runs = sorted(d for d in _glob.glob(run_glob) if os.path.isfile(os.path.join(d, "samples.npy")))
    if not runs:
        raise SystemExit(f"回復 run が無い: {run_glob}")
    S = np.concatenate([np.load(os.path.join(d, "samples.npy")) for d in runs])
    seeds = []
    for d in runs:
        rr = json.load(open(os.path.join(d, "run_record.json")))
        r2p = os.path.join(d, "recovery_record.json")
        r2 = json.load(open(r2p)) if os.path.isfile(r2p) else {}
        seeds.append({"run": os.path.basename(d), "max_logL": float(np.load(os.path.join(d, "logL.npy")).max()),
                      "logL_truth": r2.get("logL_truth"), "beta_final": rr.get("beta_final"),
                      "n_stages": rr.get("n_stages"), "hours": (rr.get("total_time_s") or 0) / 3600,
                      "checks": r2.get("checks"), "is_recovery_run": bool(r2)})
    # 実データの最終 run 群（分類の比較相手）
    real_root = args.real_root or str(RC.ROOT / syn["source_root"])
    real_runs = RC.load_group(real_root, syn["source_glob"])
    S_real = np.concatenate([np.load(r / "samples.npy") for r in real_runs])
    pb = np.array(syn["prior_bounds_final"], dtype=float)
    th = np.array(syn["theta_true"], dtype=float)

    rows = []
    for nm in RC.ORDER:
        i = RC.NAMES.index(nm)
        lo, hi = pb[i]
        c_syn, r_syn, (q05, q50, q95) = RC.classify(S[:, i], lo, hi)
        c_real, r_real, qr = RC.classify(S_real[:, i], lo, hi)
        rows.append({
            "entry": nm, "index": i, "box": [float(lo), float(hi)], "truth": float(th[i]),
            "median": q50, "q05": q05, "q95": q95, "q25": float(np.percentile(S[:, i], 25)),
            "q75": float(np.percentile(S[:, i], 75)),
            "covered90": bool(q05 <= th[i] <= q95),
            "err_rel_box": float(abs(q50 - th[i]) / (hi - lo)),
            "sd_ratio_synthetic": r_syn, "class_synthetic": c_syn,
            "sd_ratio_real": r_real, "class_real": c_real, "real_median_q05_q95": list(qr),
        })
    n = len(rows)
    ident = [r for r in rows if r["class_real"] == "identified"]
    noid = [r for r in rows if r["class_real"] == "n.i."]
    conf = {a: {b: sum(1 for r in rows if r["class_real"] == a and r["class_synthetic"] == b) for b in CLS}
            for a in CLS}
    summ = {
        "n_entries": n,
        "coverage90_all": sum(r["covered90"] for r in rows) / n,
        "n_identified_real": len(ident),
        "a_identified_real_covered": sum(r["covered90"] for r in ident),
        "a_identified_real_small_error": sum(r["err_rel_box"] < args.small_err for r in ident),
        "a_identified_real_max_err_rel_box": max((r["err_rel_box"] for r in ident), default=None),
        "a_identified_real_also_identified_synthetic": sum(r["class_synthetic"] == "identified" for r in ident),
        "n_ni_real": len(noid),
        "b_ni_real_ni_synthetic": sum(r["class_synthetic"] == "n.i." for r in noid),
        "b_ni_real_not_identified_synthetic": sum(r["class_synthetic"] != "identified" for r in noid),
        "n_identified_synthetic": sum(r["class_synthetic"] == "identified" for r in rows),
        "class_agreement": sum(r["class_real"] == r["class_synthetic"] for r in rows),
        "confusion_real_x_synthetic": conf,
        "small_err_threshold": args.small_err,
    }
    results[tag] = {"synthetic": syn_path, "label": syn["label"], "noise": syn["noise"],
                    "source_run": syn["source_run"], "real_runs": [r.name for r in real_runs],
                    "recovery_runs": seeds, "entries": rows, "summary": summ}
    s = summ
    print(f"{tag} ({syn['label']}, 回復 run {len(runs)} 本・{len(S)} 粒子): 被覆 {s['coverage90_all']:.2f}（{n} 成分）"
          f" | (a) 実データで同定 {s['n_identified_real']} → 区間に真値 {s['a_identified_real_covered']}、"
          f"誤差<{args.small_err:g}×箱 {s['a_identified_real_small_error']}、合成でも同定 "
          f"{s['a_identified_real_also_identified_synthetic']}（最大誤差 {s['a_identified_real_max_err_rel_box'] or 0:.3f}×箱）"
          f" | (b) 実データで n.i. {s['n_ni_real']} → 合成でも n.i. {s['b_ni_real_ni_synthetic']}"
          f"（同定にならない {s['b_ni_real_not_identified_synthetic']}） | 分類一致 {s['class_agreement']}/{n}")
    for sd in seeds:
        if not sd["is_recovery_run"]:
            print(f"   注意: {sd['run']} に recovery_record.json が無い（回復 run ではない／模擬の入力）")
        elif sd["logL_truth"] is not None:
            print(f"   {sd['run']}: max logL {sd['max_logL']:.3f}  vs log L(θ_true) {sd['logL_truth']:.3f}"
                  f"  β {sd['beta_final']}  {sd['hours']:.1f} h")

json.dump(results, open(args.out + ".json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- LaTeX（補足の表の本体）
CLS_TEX = {"identified": "id.", "sign-only": "s.o.", "n.i.": "n.i."}
tex = ["% tools/recovery_evaluate.py が生成（手で直さない）。列: 真値、合成データでの事後中央値 [5%, 95%]、"
       "真値を含むか、sd 比（事後 sd / 箱の一様 sd）、分類（合成 / 実データ）",
       r"\begin{tabular}{@{}l" + "rlcc" * len(results) + "@{}}", r"\toprule"]
tex.append(" & ".join(["Entry"] + [r"\multicolumn{4}{c}{%s (%s)}" % (t, "noise-free" if results[t]["noise"] == 0
                                                                        else r"$\sigma\times%g$" % results[t]["noise"])
                                    for t in results]) + r"\\")
tex.append(" & ".join([""] + [r"truth & median [5\%, 95\%] & sd ratio & synth./real" for _ in results]) + r"\\")
tex.append(r"\midrule")
for k, nm in enumerate(RC.ORDER):
    cells = [RC.tex_name(nm)]
    for t in results:
        r = results[t]["entries"][k]
        iv = "$%+.2f$ [$%+.2f$, $%+.2f$]" % (r["median"], r["q05"], r["q95"])
        if not r["covered90"]:
            iv = r"\textbf{%s}$^{\dagger}$" % iv
        cl = "%s/%s" % (CLS_TEX[r["class_synthetic"]], CLS_TEX[r["class_real"]])
        if r["class_synthetic"] != r["class_real"]:
            cl = r"\textit{%s}" % cl
        cells += ["$%+.2f$" % r["truth"], iv, "%.2f" % r["sd_ratio_synthetic"], cl]
    tex.append(" & ".join(cells) + r"\\")
tex.append(r"\midrule")
cells = [r"\multicolumn{1}{@{}l}{summary}"]
for t in results:
    s = results[t]["summary"]
    cells.append(r"\multicolumn{4}{l}{\footnotesize coverage %d/%d; id.\ real: %d/%d covered, %d/%d id.\ synth.; "
                 r"n.i.\ real: %d/%d n.i.\ synth.}" % (
                     round(s["coverage90_all"] * s["n_entries"]), s["n_entries"],
                     s["a_identified_real_covered"], s["n_identified_real"],
                     s["a_identified_real_also_identified_synthetic"], s["n_identified_real"],
                     s["b_ni_real_ni_synthetic"], s["n_ni_real"]))
tex.append(" & ".join(cells) + r"\\")
tex += [r"\bottomrule", r"\end{tabular}",
        r"% $^{\dagger}$: 90% interval does not contain the true value. Italic: class differs between synthetic and real data."]
open(args.out + ".tex", "w").write("\n".join(tex) + "\n")

# ---------------------------------------------------------------- 図（make_eml_figs56.py の図 4 と同じ書式＋真値）
tags = list(results)
fig, axes = plt.subplots(1, len(tags), figsize=(1.9 * len(tags) + 1.3, 4.0), sharey=True, squeeze=False)
axes = axes[0]
for c, tag in enumerate(tags):
    ax = axes[c]
    col = COND_COL.get(tag, "k")
    for r_, row in enumerate(results[tag]["entries"]):
        lo, hi = row["box"]
        u = lambda x: (x - lo) / (hi - lo)  # noqa: E731
        alpha = {"identified": 1.0, "sign-only": 0.65, "n.i.": 0.3}[row["class_synthetic"]]
        ax.add_patch(plt.Rectangle((0, r_ - 0.32), 1, 0.64, color="0.93", lw=0, zorder=0))
        if lo < 0 < hi:
            ax.plot([u(0), u(0)], [r_ - 0.32, r_ + 0.32], color="0.55", lw=0.7, ls=(0, (1.5, 1.2)), zorder=1)
        ax.plot([u(row["q05"]), u(row["q95"])], [r_, r_], color=col, lw=1.0, alpha=alpha, zorder=2)
        ax.plot([u(row["q25"]), u(row["q75"])], [r_, r_], color=col, lw=3.2, alpha=alpha, zorder=3,
                solid_capstyle="butt")
        ax.plot(u(row["median"]), r_, "o", color="white", mec=col, ms=3.4, mew=1.0, alpha=max(alpha, 0.6), zorder=4)
        ax.plot([u(row["truth"])] * 2, [r_ - 0.36, r_ + 0.36], color="k", lw=1.3, zorder=5)
        if not row["covered90"]:
            ax.text(1.04, r_, r"$\dagger$" if USETEX else "†", va="center", fontsize=7)
    s = results[tag]["summary"]
    ttl = f"{tag} (synthetic)"
    ax.set_title(ttl, color=col, fontsize=9)
    ax.text(0.5, -0.17, "covered %d/%d" % (round(s["coverage90_all"] * s["n_entries"]), s["n_entries"]),
            transform=ax.transAxes, ha="center", fontsize=7)
    ax.set_xlim(-0.03, 1.08)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["lower", "", "upper"])
    ax.set_xlabel("position in prior box")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(range(len(RC.ORDER)))
axes[0].set_yticklabels([label(n) for n in RC.ORDER])
axes[0].set_ylim(len(RC.ORDER) - 0.5, -0.5)
axes[0].tick_params(axis="y", length=0)
from matplotlib.lines import Line2D  # noqa: E402

fig.legend(handles=[Line2D([], [], color="k", lw=1.3, label="true value"),
                    Line2D([], [], color="0.4", lw=3.2, label="50\\% / 90\\% interval" if USETEX
                           else "50% / 90% interval"),
                    Line2D([], [], color="0.55", lw=0.7, ls=(0, (1.5, 1.2)), label="$a_{ij}=0$")],
           loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(w_pad=0.8, rect=(0, 0.05, 1, 1))
fig.savefig(args.out + ".pdf")
fig.savefig(args.out + ".png", dpi=150)
print(f"-> {args.out}.json / .tex / .pdf / .png  （usetex={USETEX}）")
