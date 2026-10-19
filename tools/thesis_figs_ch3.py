#!/usr/bin/env python3
"""修論の第 3 章用に EML の図を描き直す（2026-10-19）。

幅 160 mm（本文の幅）、Times、印刷時に目盛り 8 pt・軸ラベル 9 pt・見出し 10 pt 以上、上のタイトルなし、凡例は図に 1 つ。
中身（分類・中央値・区間・ノックアウト・pH の帯）は EML の図と同じ。入力は make_eml_figs56.py が書く _thesis_inputs.npz。

    python3 tools/make_eml_figs56.py --root <final_runs> --spec <spec_raw.json> --fig-dir <dir> --style v2
    python3 tools/thesis_figs_ch3.py <dir>/_thesis_inputs.npz --out-dir <thesis/figures>

出力: eml_ident.pdf（同定性の行列とネットワーク）、eml_posteriors.pdf（事後の区間）、eml_checks.pdf（ノックアウトと pH）、
eml_recovery.pdf（回収テスト、--recovery）、eml_cs_tradeoff.pdf（CS の λψ）。パイプラインの図は docs/revision/EML_submission/figures/fig1_pipeline_thesis.tex。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

import eml_style as st  # noqa: E402

SPN = ["So", "An", "Vei", "Fn", "Pg"]
NI_GREY = "#E4E4E4"
CONDS = ["CS", "CH", "DS", "DH"]
ORDER = ["a11", "a12", "a22", "a33", "a34", "a44", "a13", "a14", "a23", "a24", "a55", "a15", "a25", "a35", "a45"]
STAGE = {"CS": "stage iv", "CH": "stage iv", "DS": "stage iv", "DH": "stage iii"}
FS_TICK, FS_LAB, FS_HEAD = 8, 9, 10
W = st.THESIS_MM


def header(ax, tag, text, dy_mm=1.0, x0=0.0, x1=1.0):
    """列の見出し: 細い色の帯＋「CS (stage iv)」など（10 pt）。"""
    fig = ax.figure
    tr = ax.transAxes + ScaledTranslation(0, dy_mm * st.MM, fig.dpi_scale_trans)
    ax.text(0.5 * (x0 + x1), 1.0, text, transform=tr, ha="center", va="bottom", fontsize=FS_HEAD)
    tr2 = ax.transAxes + ScaledTranslation(0, (dy_mm + 5.2) * st.MM, fig.dpi_scale_trans)
    ax.plot([x0, x1], [1.0, 1.0], transform=tr2, color=st.COND_COL[tag], lw=2.2,
            solid_capstyle="butt", clip_on=False)


def label(ax, s, dx_mm=-1.0, dy_mm=1.0):
    fig = ax.figure
    tr = ax.transAxes + ScaledTranslation(dx_mm * st.MM, dy_mm * st.MM, fig.dpi_scale_trans)
    ax.text(0, 1, rf"\textbf{{({s})}}", transform=tr, ha="right", va="bottom", fontsize=FS_HEAD)


# ============================================================================ 同定性
def fig_ident(M, cls, out_noext):
    ij = [(int(n[1]) - 1, int(n[2]) - 1) for n in ORDER]
    ident = cls == 2
    vmax = float(np.ceil(np.nanmax(np.abs(M[ident])))) if ident.any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap(st.DIVERGING)
    col_pos, col_neg = cmap(0.86), cmap(0.14)
    H = 125.0
    fig = plt.figure(figsize=(W * st.MM, H * st.MM))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    lw_of = lambda v: 0.6 + 2.6 * min(abs(v), vmax) / vmax  # noqa: E731
    cell = 7.4
    mw = 5 * cell
    bw = 7.0 + mw + 2.0 + 33.0  # 行の名前・行列・すき間・ネットワーク
    gapx = W - 2 * bw
    for c, tag in enumerate(CONDS):
        bx = (c % 2) * (bw + gapx)
        by = 75.0 if c < 2 else 22.0  # ブロックの下端
        ax = ax_mm(bx + 7.0, by, mw, mw + 5.0)  # 上に列の名前の分
        ax.set_xlim(0, 5)
        ax.set_ylim(5, -(5.0 / cell))
        ax.axis("off")
        for r, (i, j) in enumerate(ij):
            v, k = M[r, c], cls[r, c]
            if k == 2:
                fc, txt = cmap(norm(np.clip(v, -vmax, vmax))), rf"${v:.1f}$"
                tc = "white" if abs(v) > 0.55 * vmax else st.INK
            elif k == 1:
                s = np.sign(v)
                fc, txt, tc = cmap(norm(s * 0.22 * vmax)), (r"$+$" if s > 0 else r"$-$"), st.INK
            else:
                fc, txt, tc = NI_GREY, "", st.INK
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="none"))
            if k == 1:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", lw=0,
                                       edgecolor=cmap(norm(np.sign(v) * 0.45 * vmax)), hatch="////"))
                ax.add_patch(Rectangle((j + 0.27, i + 0.27), 0.46, 0.46, facecolor=fc, lw=0))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor="white", lw=0.9))
            if txt:
                ax.text(j + 0.5, i + 0.53, txt, ha="center", va="center", fontsize=FS_TICK, color=tc)
        for k_, s_ in enumerate(SPN):
            ax.text(k_ + 0.5, -0.12, s_, ha="center", va="bottom", fontsize=FS_TICK)
            ax.text(-0.12, k_ + 0.53, s_, ha="right", va="center", fontsize=FS_TICK)
        n_id, n_so = int((cls[:, c] == 2).sum()), int((cls[:, c] == 1).sum())
        hdr = ax_mm(bx, by, bw, mw + 5.0)
        hdr.axis("off")
        header(hdr, tag, f"{tag} ({STAGE[tag]})", dy_mm=1.5)
        # ネットワーク
        an = ax_mm(bx + 7.0 + mw + 2.0, by - 1.0, 33.0, mw + 1.0)
        an.set_xlim(-1.45, 1.45)
        an.set_ylim(-1.55, 1.35)
        an.set_aspect("equal")
        an.axis("off")
        ang = np.pi / 2 - 2 * np.pi * np.arange(5) / 5
        P = np.c_[np.cos(ang), np.sin(ang)]
        rn = 0.27
        for r, (i, j) in enumerate(ij):
            k, v = cls[r, c], M[r, c]
            if k == 0:
                continue
            colr = col_pos if v > 0 else col_neg
            if i == j:
                if k == 2:
                    an.add_patch(Circle(P[i], rn + 0.09, facecolor="none", edgecolor=colr,
                                        lw=lw_of(v), zorder=2, clip_on=False))
                continue
            d = P[j] - P[i]
            u = d / np.linalg.norm(d)
            a, b = P[i] + u * (rn + 0.03), P[j] - u * (rn + 0.03)
            if k == 2:
                an.plot(*np.c_[a, b], color=colr, lw=lw_of(v), solid_capstyle="round", zorder=1)
            else:
                an.plot(*np.c_[a, b], color=colr, lw=0.8, ls=(0, (2.5, 1.6)), zorder=1)
        for k_ in range(5):
            an.add_patch(Circle(P[k_], rn, facecolor="white", edgecolor=st.INK, lw=0.5, zorder=3))
            an.text(*P[k_], SPN[k_], ha="center", va="center", fontsize=FS_TICK, zorder=4)
        if c == 0:  # (a) 行列、(b) ネットワーク（各ブロックの左と右）
            label(ax, "a", dx_mm=-1.0, dy_mm=-4.0)
            label(an, "b", dx_mm=4.0, dy_mm=1.0)
        an.text(0, -1.5, f"{n_id} identified" + (f", {n_so} sign only" if n_so else ""),
                ha="center", va="bottom", fontsize=FS_TICK, color=st.MUTE)
    # 色の目盛り（下、横向き）
    cax = ax_mm(18.0, 12.0, 52.0, 2.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    step = max(1, int(np.ceil(vmax / 3)))
    cb.set_ticks(np.arange(-(vmax // step) * step, vmax + 1e-9, step))
    cb.ax.tick_params(labelsize=FS_TICK, width=0.5, length=2)
    fig.text(16.0 / W, 13.3 / H, r"$a_{ij}$", ha="right", va="center", fontsize=FS_LAB)
    hatch_c = cmap(norm(-0.45 * vmax))
    H_ = [(Patch(facecolor=cmap(norm(-0.22 * vmax)), edgecolor=hatch_c, hatch="////", lw=0),
           Line2D([], [], color=st.MUTE, lw=0.8, ls=(0, (2.5, 1.6)))),
          Patch(facecolor=NI_GREY, edgecolor="none"),
          Line2D([], [], color=col_pos, lw=2.0), Line2D([], [], color=col_neg, lw=2.0),
          Line2D([], [], color=st.MUTE, marker="o", ms=6, mfc="none", mew=1.2, lw=0)]
    L_ = ["sign only", "not identified", r"cooperative ($a_{ij}>0$)", r"competitive ($a_{ij}<0$)",
          "self-interaction"]
    fig.legend(H_, L_, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.0),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
               handlelength=2.4, handletextpad=0.5, columnspacing=1.4, fontsize=FS_TICK, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)


# ============================================================================ 事後の区間
def fig_post(Q, U0, cls, M, out_noext):
    sp = {"1": "So", "2": "An", "3": "Vei", "4": "Fn", "5": "Pg"}
    n = len(ORDER)
    H = 110.0
    fig, axes = plt.subplots(1, 4, figsize=(W * st.MM, H * st.MM), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.865, bottom=0.215, wspace=0.14)
    for c, tag in enumerate(CONDS):
        ax = axes[c]
        col = st.COND_COL[tag]
        for r in range(n):
            q = Q[r, c]
            ax.plot([0, 1], [r, r], color="#EDEDED", lw=0.5, zorder=0, solid_capstyle="butt")
            if np.isfinite(U0[r, c]):
                ax.plot([U0[r, c]] * 2, [r - 0.32, r + 0.32], color=st.INK, lw=0.7, zorder=5)
            if cls[r, c] == 2:
                ax.plot([q[0], q[4]], [r, r], color=col, lw=1.0, zorder=2, solid_capstyle="butt")
                ax.plot([q[1], q[3]], [r, r], color=col, lw=3.2, zorder=3, solid_capstyle="butt")
                ax.plot(q[2], r, "o", mfc="white", mec=col, ms=3.6, mew=1.0, zorder=4)
            else:
                g = "#B5B5B5"
                ax.plot([q[0], q[4]], [r, r], color=g, lw=0.8, zorder=2, solid_capstyle="butt")
                ax.plot([q[1], q[3]], [r, r], color=g, lw=2.4, zorder=3, solid_capstyle="butt")
                ax.plot(q[2], r, "o", mfc="white", mec=g, ms=3.0, mew=0.8, zorder=4)
                if cls[r, c] == 1:
                    ax.text(1.05, r, r"$+$" if M[r, c] > 0 else r"$-$", ha="left", va="center",
                            fontsize=FS_TICK, color=st.INK)
        ax.set_xlim(-0.03, 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["lower", "upper"])
        ax.tick_params(axis="y", length=0, labelsize=FS_TICK)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        ax.spines["left"].set_visible(False)
        n_id = int((cls[:, c] == 2).sum())
        header(ax, tag, f"{tag} ({STAGE[tag]})", dy_mm=4.2, x1=0.95)
        ax.text(0.47, 1.0, f"{n_id} identified", transform=ax.transAxes + ScaledTranslation(
            0, 0.3 * st.MM, fig.dpi_scale_trans), ha="center", va="bottom", fontsize=FS_TICK, color=st.MUTE)
    lab = [f"{sp[s[1]]}--{sp[s[2]]}" if s[1] != s[2] else f"{sp[s[1]]} (self)" for s in ORDER]
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(lab)
    axes[0].set_ylim(n - 0.5, -0.6)
    fig.text(0.55, 0.135, "position of the posterior in the prior box of each entry", ha="center",
             va="center", fontsize=FS_LAB)
    H_ = [Line2D([], [], color=st.INK, lw=3.2, label=r"50\% interval"),
          Line2D([], [], color=st.INK, lw=1.0, label=r"90\% interval"),
          Line2D([], [], color=st.INK, marker="o", mfc="white", ms=3.6, lw=0, label="median"),
          Line2D([], [], color="#B5B5B5", lw=2.4, label="not identified"),
          Line2D([], [], color=st.INK, marker="|", ms=7, lw=0, label=r"$a_{ij}=0$"),
          Line2D([], [], color="none", label=r"$\pm$: sign only")]
    fig.legend(handles=H_, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0),
               handlelength=1.8, columnspacing=1.6, fontsize=FS_TICK, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)


# ============================================================================ ノックアウトと pH
def fig_checks(ko, B, measured_ratio, out_noext):
    H = 70.0
    fig = plt.figure(figsize=(W * st.MM, H * st.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.36,
                          left=0.085, right=0.99, top=0.85, bottom=0.30)
    ax = fig.add_subplot(gs[0])
    col = st.COND_COL["DH"]
    labels = ["full", "no Fn", r"$a_{45}=0$"]
    keys = ["ratio_base", "ratio_noFn", "ratio_a45_0"]
    ax.axhline(1.0, color=st.FAINT, lw=0.6, ls=":", zorder=0)
    ax.axhline(measured_ratio, color=st.INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
    for j, k in enumerate(keys):
        for s, row in enumerate(ko):
            lo, md, hi = row[k]
            x = j + (s - 1) * 0.22
            ax.plot([x, x], [lo, hi], color=col, lw=1.2, solid_capstyle="butt")
            ax.plot(x, md, "o", color=col, ms=4.0, mec="white", mew=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0, labelsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, max(8.0, max(r[k][2] for r in ko for k in keys) * 1.05, measured_ratio * 1.15))
    ax.set_ylabel("Pg ratio, day 21 / day 15", fontsize=FS_LAB)
    header(ax, "DH", "DH: knockout", dy_mm=1.5)
    label(ax, "a", dx_mm=-9.0, dy_mm=1.5)
    for p, tag in enumerate(["CH", "DH"]):
        ax = fig.add_subplot(gs[p + 1])
        c = st.COND_COL[tag]
        d = B[f"{tag}_days"]
        ax.fill_between(d, B[f"{tag}_lo"], B[f"{tag}_hi"], color=c, alpha=0.22, lw=0)
        ax.plot(d, B[f"{tag}_med"], color=c, lw=1.2)
        ax.plot(B[f"{tag}_t_meas"], B[f"{tag}_ph_meas"], "o", ms=1.0, color=st.INK, mew=0)
        ax.set_xlabel("Day", fontsize=FS_LAB)
        ax.set_xticks([0, 3, 6, 10, 15, 21])
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(6.15, 7.4)
        header(ax, tag, f"{tag}: pH ({STAGE[tag]})", dy_mm=1.5)
        label(ax, "bc"[p], dx_mm=-9.0, dy_mm=1.5)
        if p == 0:
            ax.set_ylabel("pH", fontsize=FS_LAB)
    g = st.MUTE
    H_ = [Line2D([], [], color=g, marker="o", ms=4.0, mec="white", lw=1.2),
          Line2D([], [], color=st.INK, lw=0.8, ls=(0, (4, 2))),
          Patch(facecolor=g, alpha=0.3, lw=0), Line2D([], [], color=g, lw=1.2),
          Line2D([], [], color=st.INK, marker="o", ms=2.4, lw=0, mew=0)]
    L_ = [r"(a) model: median and 90\% (one per seed)", "(a) measured",
          r"(b, c) model, 90\%", "(b, c) model, median", "(b, c) measured, not fitted"]
    fig.legend(H_, L_, loc="lower center", ncol=3, bbox_to_anchor=(0.53, 0.0), fontsize=FS_TICK,
               handlelength=1.8, columnspacing=1.4, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)


# ============================================================================ 回収テスト
def fig_recovery(results, out_noext):
    sp = {"1": "So", "2": "An", "3": "Vei", "4": "Fn", "5": "Pg"}
    n = len(ORDER)
    H = 112.0
    fig, axes = plt.subplots(1, 4, figsize=(W * st.MM, H * st.MM), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.895, bottom=0.25, wspace=0.14)
    for c, tag in enumerate(CONDS):
        ax = axes[c]
        col = st.COND_COL[tag]
        ent = {e["entry"]: e for e in results[tag]["entries"]}
        for r_, nm in enumerate(ORDER):
            row = ent[nm]
            lo, hi = row["box"]
            u = lambda x: (x - lo) / (hi - lo)  # noqa: E731
            alpha = {"identified": 1.0, "sign-only": 0.65, "n.i.": 0.3}[row["class_synthetic"]]
            ax.plot([0, 1], [r_, r_], color="#EDEDED", lw=0.5, zorder=0, solid_capstyle="butt")
            if lo < 0 < hi:
                ax.plot([u(0)] * 2, [r_ - 0.32, r_ + 0.32], color="0.55", lw=0.8, ls=(0, (1.5, 1.2)), zorder=1)
            ax.plot([u(row["q05"]), u(row["q95"])], [r_, r_], color=col, lw=1.0, alpha=alpha, zorder=2,
                    solid_capstyle="butt")
            ax.plot([u(row["q25"]), u(row["q75"])], [r_, r_], color=col, lw=3.2, alpha=alpha, zorder=3,
                    solid_capstyle="butt")
            ax.plot(u(row["median"]), r_, "o", color="white", mec=col, ms=3.6, mew=1.0,
                    alpha=max(alpha, 0.6), zorder=4)
            ax.plot([u(row["truth"])] * 2, [r_ - 0.38, r_ + 0.38], color=st.INK, lw=1.4, zorder=5)
            if not row["covered90"]:
                ax.text(1.05, r_, r"$\dagger$", va="center", ha="left", fontsize=FS_TICK)
        s = results[tag]["summary"]
        header(ax, tag, f"{tag} (synthetic)", dy_mm=1.5, x1=0.95)
        ax.text(0.47, -0.105, "covered %d/%d" % (round(s["coverage90_all"] * s["n_entries"]), s["n_entries"]),
                transform=ax.transAxes, ha="center", va="top", fontsize=FS_TICK)
        ax.set_xlim(-0.03, 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["lower", "upper"])
        ax.tick_params(axis="y", length=0, labelsize=FS_TICK)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"{sp[m[1]]}--{sp[m[2]]}" if m[1] != m[2] else f"{sp[m[1]]} (self)" for m in ORDER])
    axes[0].set_ylim(n - 0.5, -0.6)
    fig.text(0.55, 0.115, "position in the prior box of each entry", ha="center", va="center", fontsize=FS_LAB)
    H_ = [Line2D([], [], color=st.INK, lw=1.4), Line2D([], [], color="0.4", lw=3.2),
          Line2D([], [], color="0.4", lw=1.0), Line2D([], [], color="0.55", lw=0.8, ls=(0, (1.5, 1.2))),
          Line2D([], [], color="none")]
    L_ = ["true value", r"50\% interval", r"90\% interval", r"$a_{ij}=0$", r"$\dagger$: not covered"]
    fig.legend(H_, L_, loc="lower center", ncol=5, bbox_to_anchor=(0.55, 0.0), fontsize=FS_TICK,
               handlelength=1.8, columnspacing=1.3, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)


# ============================================================================ CS の重みのトレードオフ
# λψ, 生きた菌の組成, 全菌の組成, 生きた割合（RMSE）。tools/plot_cs_tradeoff.py と同じ表（handoff 13p）
CS_ROWS = [(1.5, 0.092, 0.039, 0.231), (3, 0.103, 0.100, 0.174), (6, 0.120, 0.120, 0.132),
           (12, 0.144, 0.141, 0.074), (24, 0.158, 0.149, 0.051)]


def fig_tradeoff(out_noext):
    H = 68.0
    fig, ax = plt.subplots(figsize=(W * st.MM, H * st.MM))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.27)
    lam = [r[0] for r in CS_ROWS]
    via, tot, live = [r[1] for r in CS_ROWS], [r[2] for r in CS_ROWS], [r[3] for r in CS_ROWS]
    c = st.COND_COL["CS"]
    ax.plot(live, via, "o-", color=c, ms=4.5, lw=1.2)
    ax.plot(live, tot, "s--", color="0.4", ms=4.0, lw=1.0)
    for lv, v, l in zip(live, via, lam):
        ax.annotate(rf"$\lambda_\psi={l:g}$", (lv, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_TICK)
    ax.set_xlim(0.03, 0.255)
    ax.set_ylim(0.02, 0.185)
    ax.set_xlabel("RMSE of the live fraction", fontsize=FS_LAB)
    ax.set_ylabel("RMSE of the composition", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, lw=0.5)
    H_ = [Line2D([], [], color=c, marker="o", ms=4.5, lw=1.2), Line2D([], [], color="0.4", marker="s", ms=4.0,
                                                                         lw=1.0, ls="--")]
    fig.legend(H_, ["viable composition", "total-count composition"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.54, 0.0), fontsize=FS_TICK, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recovery", default=None, help="recovery_evaluate.py の JSON（figures/raw/recovery.json）")
    a = ap.parse_args()
    z = np.load(a.inputs, allow_pickle=False)
    st.apply_thesis()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 600})  # 図の中のビットマップ（色の目盛り）を 600 dpi に
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_ident(z["M"], z["cls"], str(out / "eml_ident"))
    fig_post(z["Q4"], z["U0"], z["cls"], z["M"], str(out / "eml_posteriors"))
    ko = json.loads(str(z["ko"])) if str(z["ko"]) else None
    B = {k[2:]: z[k] for k in z.files if k.startswith("B_")}
    if ko and B:
        fig_checks(ko, B, float(z["OBS_RATIO"]), str(out / "eml_checks"))
    if a.recovery:
        fig_recovery(json.load(open(a.recovery)), str(out / "eml_recovery"))
    fig_tradeoff(str(out / "eml_cs_tradeoff"))


if __name__ == "__main__":
    main()
