#!/usr/bin/env python3
"""EML の図 5（同定性マップ）と図 6（ノックアウトと pH の答え合わせ）を作る（2026-10-12）。

図 5: 15 成分 × 4 条件。最終段の 3 seed 合算の事後で、補足 S4 と同じ分類
      （tools/make_supplementary_tables.py の ident_table・UNIDENT_RATIO と同じ規則）:
        同定された成分     → 色 = 事後中央値（発散カラーマップ）、数字 = 中央値
        符号だけ決まる（s.o.） → 薄い色＋符号
        同定されていない（n.i.） → 灰色
図 6: (a) DH で Fn を除いたとき・a45=0 のときの Pg の Day21/Day15 比（tools/knockout_fn.py の json）
      (b, c) pH の答え合わせ（tools/ph_check.py の *_bands.npz）

  python3 tools/make_eml_figs56.py [--root docs/revision/generated/final_runs] [--spec <json>] [--fig-dir <dir>]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "docs/revision/EML_submission/figures/provisional_2026-10-11"
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "docs/revision/generated/final_runs"))
ap.add_argument("--spec", default=str(PROV / "supp/spec_provisional.json"))
ap.add_argument("--fig-dir", default=str(PROV))
args = ap.parse_args()

try:
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{lmodern}", "font.size": 9})
    plt.figure(); plt.text(0, 0, r"$a_{11}$"); plt.savefig(os.devnull, format="pdf"); plt.close()
except Exception:  # noqa: BLE001  LaTeX が無ければ通常の描画
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 9})

NAMES = ["a11", "a12", "a22", "b1", "b2", "a33", "a34", "a44", "b3", "b4",
         "a13", "a14", "a23", "a24", "a55", "b5", "a15", "a25", "a35", "a45"]
ORDER = ["a11", "a12", "a22", "a33", "a34", "a44", "a13", "a14", "a23", "a24", "a55", "a15", "a25", "a35", "a45"]
CONDS = ["CS", "CH", "DS", "DH"]
UNIDENT_RATIO = 0.8  # tools/check_paper_runs.py・make_supplementary_tables.py と同じ
SP = {"1": "So", "2": "An", "3": "Vei", "4": "Fn", "5": "Pg"}

# ---------------------------------------------------------------- 図 5
spec = json.load(open(args.spec))
M = np.full((len(ORDER), 4), np.nan)
cls = np.zeros((len(ORDER), 4), dtype=int)  # 0 n.i., 1 s.o., 2 identified
for c, tag in enumerate(CONDS):
    dirs = sorted(glob.glob(os.path.join(args.root, spec[tag]["ult"])))
    S = np.concatenate([np.load(os.path.join(d, "samples.npy")) for d in dirs])
    pb = np.array(json.load(open(os.path.join(dirs[0], "run_record.json")))["prior_bounds_final"], dtype=float)
    for r, nm in enumerate(ORDER):
        i = NAMES.index(nm)
        q05, q50, q95 = np.percentile(S[:, i], [5, 50, 95])
        unif = (pb[i, 1] - pb[i, 0]) / np.sqrt(12.0)
        M[r, c] = q50
        if S[:, i].std() < UNIDENT_RATIO * unif:
            cls[r, c] = 2
        elif q05 > 0 or q95 < 0:
            cls[r, c] = 1

fig, ax = plt.subplots(figsize=(3.4, 4.6))
vmax = 6.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")
for r in range(len(ORDER)):
    for c in range(4):
        v = np.clip(M[r, c], -vmax, vmax)
        if cls[r, c] == 2:
            col, txt, tc = cmap(norm(v)), f"{M[r, c]:+.1f}", "k" if abs(v) < 4 else "w"
        elif cls[r, c] == 1:
            col, txt, tc = (*cmap(norm(np.sign(v) * 1.5))[:3], 0.45), ("$+$" if v > 0 else "$-$"), "k"
        else:
            col, txt, tc = (0.9, 0.9, 0.9, 1.0), "", "k"
        ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=col, edgecolor="white", lw=1.0))
        if txt:
            ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center", fontsize=7, color=tc)
ax.set_xlim(0, 4)
ax.set_ylim(len(ORDER), 0)
ax.set_xticks(np.arange(4) + 0.5)
ax.set_xticklabels(CONDS)
ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(ORDER)) + 0.5)
ax.set_yticklabels([f"{SP[n[1]]}--{SP[n[2]]}" if n[1] != n[2] else f"{SP[n[1]]} (self)" for n in ORDER])
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("posterior median of $a_{ij}$ (identified entries)")
fig.tight_layout()
fig.savefig(os.path.join(args.fig_dir, "fig5_identifiability.pdf"))
fig.savefig(os.path.join(args.fig_dir, "fig5_identifiability.png"), dpi=150)
print("図 5: 同定 " + ", ".join(f"{t} {int((cls[:, c] == 2).sum())}" for c, t in enumerate(CONDS))
      + " / 符号のみ " + ", ".join(f"{t} {int((cls[:, c] == 1).sum())}" for c, t in enumerate(CONDS)))

# ---------------------------------------------------------------- 図 6
ko = json.load(open(os.path.join(args.fig_dir, "knockout_DH.json")))
B = np.load(os.path.join(args.fig_dir, "ph_check_bands.npz"))
fig = plt.figure(figsize=(7.0, 2.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1], wspace=0.35)
ax = fig.add_subplot(gs[0])
labels = ["full", "no Fn", "$a_{45}=0$"]
keys = ["ratio_base", "ratio_noFn", "ratio_a45_0"]
for j, (lab, k) in enumerate(zip(labels, keys)):
    for s, row in enumerate(ko):
        lo, md, hi = row[k]
        x = j + (s - 1) * 0.18
        ax.plot([x, x], [lo, hi], color=f"C{j}", lw=1.2)
        ax.plot(x, md, "o", color=f"C{j}", ms=3.5)
ax.axhline(2.74, color="k", lw=0.8, ls="--")
ax.text(0.5, 2.82, "measured", fontsize=7, ha="center", va="bottom")
ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlim(-0.6, 2.6)
ax.set_ylabel("Pg ratio day 21 / day 15")
ax.set_title("(a) DH: in-silico knockout", fontsize=9)
for p, tag in enumerate(["CH", "DH"]):
    ax = fig.add_subplot(gs[p + 1])
    d = B[f"{tag}_days"]
    ax.fill_between(d, B[f"{tag}_lo"], B[f"{tag}_hi"], color="0.8", lw=0, label="model, 90\\%")
    ax.plot(d, B[f"{tag}_med"], color="0.2", lw=1.0, label="model, median")
    ax.plot(B[f"{tag}_t_meas"], B[f"{tag}_ph_meas"], "o", ms=2.2, color="C3", label="measured (not fitted)")
    ax.set_xlabel("Day")
    ax.set_title(f"({'bc'[p]}) pH, {tag}", fontsize=9)
    ax.set_ylim(6.15, 7.4)
    if p == 0:
        ax.set_ylabel("pH")
        ax.legend(fontsize=6.5, loc="upper left", frameon=False)
fig.savefig(os.path.join(args.fig_dir, "fig6_checks.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(args.fig_dir, "fig6_checks.png"), dpi=150, bbox_inches="tight")
print("図 6: 書き出し済み")
