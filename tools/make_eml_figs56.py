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
ap.add_argument("--style", choices=["v1", "v2", "both"], default="both",
                help="v1: 旧デザイン（fig5_identifiability・fig4_posteriors・fig6_checks）、"
                     "v2: 新デザイン（fig3_ident_matrix・fig4_posteriors_v2・fig6_checks_v2、tools/eml_figs_v2.py）")
args = ap.parse_args()
V1, V2 = args.style in ("v1", "both"), args.style in ("v2", "both")

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


def ult_dirs(tag):
    """最終段の run（spec に無い・まだ無い条件は []。図では pending の列になる、2026-10-16）。"""
    g = spec.get(tag, {}).get("final") or spec.get(tag, {}).get("ult")  # "final": 最終群が ult でない条件（DH は p2、10-16）
    return sorted(glob.glob(os.path.join(args.root, g))) if g else []


M = np.full((len(ORDER), 4), np.nan)
cls = np.zeros((len(ORDER), 4), dtype=int)  # -1 pending, 0 n.i., 1 s.o., 2 identified
for c, tag in enumerate(CONDS):
    dirs = ult_dirs(tag)
    if not dirs:
        cls[:, c] = -1
        print(f"{tag}: 最終段の run なし → pending")
        continue
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
if V1:
    fig.savefig(os.path.join(args.fig_dir, "fig5_identifiability.pdf"))
    fig.savefig(os.path.join(args.fig_dir, "fig5_identifiability.png"), dpi=150)
plt.close(fig)
print("図 5: 同定 " + ", ".join(f"{t} {int((cls[:, c] == 2).sum())}" for c, t in enumerate(CONDS))
      + " / 符号のみ " + ", ".join(f"{t} {int((cls[:, c] == 1).sum())}" for c, t in enumerate(CONDS)))

# ---------------------------------------------------------------- 図 6
_ko = os.path.join(args.fig_dir, "knockout_DH.json")
ko = json.load(open(_ko)) if os.path.exists(_ko) else None  # DH の最終段のあと（knockout_fn.py）
_phb = os.path.join(args.fig_dir, "ph_check_bands.npz")
B = np.load(_phb) if os.path.exists(_phb) else None  # pH は推定に使わない検証。無ければ図 5 を飛ばす（生データの pH 待ち）
if ko is None:
    print("図 6: knockout_DH.json が無いので飛ばした（DH の最終段のあと）")
else:
    OBS_RATIO = ko[0].get("obs_ratio", 2.74)  # 実測の Pg D21/D15（run が読んだデータ、knockout_fn.py）
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
    ax.axhline(OBS_RATIO, color="k", lw=0.8, ls="--")
    ax.text(0.5, OBS_RATIO + 0.08, "measured", fontsize=7, ha="center", va="bottom")
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Pg ratio day 21 / day 15")
    ax.set_title("(a) DH: in-silico knockout", fontsize=9)
    for p, tag in enumerate(["CH", "DH"] if B is not None else []):
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
    if V1:
        fig.savefig(os.path.join(args.fig_dir, "fig6_checks.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(args.fig_dir, "fig6_checks.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("図 6: 書き出し済み")

# ---------------------------------------------------------------- 図 4（区間の図、バイオリン図の置き換え、2026-10-12）
# 横軸は段 (iii) の箱（初期の箱に、端に積んだ成分だけ広げた箱）の中での位置 0..1。
# 棒は事後の 50%（太）と 90%（細）区間、丸は中央値（3 seed 合算）。灰色の縦の刻みは a_ij = 0 の位置。
# 色の濃さは図 3 と同じ分類（同定・符号のみ・同定されない）。
COND_COL = {"CS": "#1f77b4", "CH": "#2ca02c", "DS": "#ff7f0e", "DH": "#d62728"}


def _override(base, spec_str):
    pb = base.copy()
    for part in (spec_str or "").replace(";", ",").split(","):
        if part.strip():
            k, lo, hi = part.split(":")
            pb[int(k)] = [float(lo), float(hi)]
    return pb


fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.0), sharey=True)
Q4 = np.full((len(ORDER), 4, 5), np.nan)  # v2 用: 段 (iii) の箱の中での位置
U0 = np.full((len(ORDER), 4), np.nan)  # a_ij = 0 の位置（箱の中にあるときだけ）
for c, tag in enumerate(CONDS):
    ax = axes[c]
    dirs = ult_dirs(tag)
    if not dirs:
        ax.text(0.5, len(ORDER) / 2, "pending", ha="center", va="center", color="0.5")
        ax.set_title(tag, color=COND_COL[tag])
        continue
    S = np.concatenate([np.load(os.path.join(d, "samples.npy")) for d in dirs])
    rec_f = json.load(open(os.path.join(dirs[0], "run_record.json")))
    if "_ult_" in os.path.basename(dirs[0]):
        pil = sorted(glob.glob(os.path.join(args.root, spec[tag]["pilot"])))[0]
        b1 = np.array(json.load(open(os.path.join(pil, "run_record.json")))["prior_bounds_final"], dtype=float)
        b3 = _override(b1, rec_f["args"].get("override_bounds"))
    else:
        b3 = np.array(rec_f["prior_bounds_final"], dtype=float)
    for r, nm in enumerate(ORDER):
        i = NAMES.index(nm)
        lo, hi = b3[i]
        u = lambda x: (x - lo) / (hi - lo)
        q = u(np.percentile(S[:, i], [5, 25, 50, 75, 95]))
        Q4[r, c] = q
        if lo < 0 < hi:
            U0[r, c] = u(0)
        alpha = 1.0 if cls[r, c] == 2 else (0.65 if cls[r, c] == 1 else 0.3)
        col = COND_COL[tag]
        ax.add_patch(plt.Rectangle((0, r - 0.32), 1, 0.64, color="0.93", lw=0, zorder=0))
        if lo < 0 < hi:
            ax.plot([u(0), u(0)], [r - 0.32, r + 0.32], color="0.45", lw=0.8, zorder=1)
        ax.plot([q[0], q[4]], [r, r], color=col, lw=1.0, alpha=alpha, zorder=2)
        ax.plot([q[1], q[3]], [r, r], color=col, lw=3.2, alpha=alpha, zorder=3, solid_capstyle="butt")
        ax.plot(q[2], r, "o", color="white", mec=col, ms=3.4, mew=1.0, alpha=max(alpha, 0.6), zorder=4)
    ax.set_title(tag, color=COND_COL[tag])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["lower", "", "upper"])
    ax.set_xlabel("position in prior box")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(range(len(ORDER)))
axes[0].set_yticklabels([f"{SP[n[1]]}--{SP[n[2]]}" if n[1] != n[2] else f"{SP[n[1]]} (self)" for n in ORDER])
axes[0].set_ylim(len(ORDER) - 0.5, -0.5)
axes[0].tick_params(axis="y", length=0)
fig.tight_layout(w_pad=0.8)
if V1:
    fig.savefig(os.path.join(args.fig_dir, "fig4_posteriors.pdf"))
    fig.savefig(os.path.join(args.fig_dir, "fig4_posteriors.png"), dpi=150)
    print("図 4: 書き出し済み")
plt.close(fig)

# ---------------------------------------------------------------- v2（2026-10-12、tools/eml_figs_v2.py）
# 分類（cls）・中央値（M）・区間（Q4）は上と同じものを使う。出力は新しいファイル名。
if V2:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eml_style  # noqa: E402
    import eml_figs_v2 as V2F  # noqa: E402

    eml_style.apply()
    vmax = V2F.fig3_matrix(M, cls, ORDER, CONDS, os.path.join(args.fig_dir, "fig3_ident_matrix"))
    print(f"図 3 (v2): fig3_ident_matrix、色の範囲 ±{vmax:g}")
    V2F.fig4_v2(Q4, U0, cls, M, ORDER, CONDS, SP, os.path.join(args.fig_dir, "fig4_posteriors_v2"))
    if B is not None and ko is not None:
        V2F.fig6_v2(ko, B, os.path.join(args.fig_dir, "fig6_checks_v2"), measured_ratio=OBS_RATIO)
    else:
        print("図 5 (v2): pH の帯（ph_check_bands.npz）か knockout_DH.json が無いので飛ばした")
