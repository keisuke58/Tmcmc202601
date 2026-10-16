"""EML 図の v2 デザイン（2026-10-12）。tools/make_eml_figs56.py から呼ぶ（--style v2 / both）。

  fig3_matrix   図 3: 条件ごとの 5×5 上三角行列（同定・符号のみ・同定されない）＋ 同定された成分だけのネットワーク
  fig4_v2       図 4（補足へ移す）: 事後の 50/90% 区間、2 段（同定 = 条件の色、それ以外 = 灰色）
  fig6_v2       図 5: ノックアウトと pH の答え合わせ

データの分類は make_eml_figs56.py と同じもの（cls: 0 同定されない、1 符号のみ、2 同定）をそのまま受け取る。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

import eml_style as st

SPN = ["So", "An", "Vei", "Fn", "Pg"]
NI_GREY = "#E4E4E4"


def _tex(s):
    import matplotlib as mpl
    return s if mpl.rcParams["text.usetex"] else s.replace(r"\textbf{", "").replace("}", "")


def _cond_header(ax, tag, x0=0.0, x1=1.0, y=1.0, dy_mm=1.0, sub=True):
    """条件の見出し: 細い色の帯 + 「CS commensal static」（文字はインク色、色は帯だけ）。"""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    fig = ax.figure
    tr = ax.transAxes + ScaledTranslation(0, dy_mm * st.MM, fig.dpi_scale_trans)
    extra = st.COND_NAME[tag] if sub is True else (sub or "")
    txt = rf"\textbf{{{tag}}}" + (rf"\ \ {extra}" if extra else "")
    ax.text(0.5 * (x0 + x1), y, _tex(txt), transform=tr, ha="center", va="bottom", fontsize=8)
    tr2 = ax.transAxes + ScaledTranslation(0, (dy_mm + 4.4) * st.MM, fig.dpi_scale_trans)
    ax.plot([x0, x1], [y, y], transform=tr2, color=st.COND_COL[tag], lw=2.2,
            solid_capstyle="butt", clip_on=False)


# ============================================================================ 図 3
def fig3_matrix(M, cls, order, conds, out_noext, vmax=None):
    """M, cls: (15, 4)。order: ['a11', ...]（i, j は 1 始まり）。"""
    ij = [(int(n[1]) - 1, int(n[2]) - 1) for n in order]
    ident = cls == 2
    if vmax is None:
        vmax = float(np.ceil(np.nanmax(np.abs(M[ident])))) if ident.any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap(st.DIVERGING)
    col_pos, col_neg = cmap(0.86), cmap(0.14)  # ネットワークの辺（協力 / 競合）

    W, H = st.DOUBLE_MM, 86.0
    fig = plt.figure(figsize=(W * st.MM, H * st.MM))
    pw, gap, x0 = 39.25, 3.0, 10.0

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    lw_of = lambda v: 0.5 + 2.6 * min(abs(v), vmax) / vmax  # noqa: E731
    mat_axes, net_axes = [], []
    for c, tag in enumerate(conds):
        x = x0 + c * (pw + gap)
        # ---------------------------------------------------------------- 行列
        ax = ax_mm(x, 36.5, pw, 40.0)
        mat_axes.append(ax)
        ax.set_xlim(-0.05, 5.05)
        ax.set_ylim(5.05, -0.75)
        ax.set_aspect("equal")
        ax.axis("off")
        if (cls[:, c] == -1).all():  # 最終段の run がまだ無い条件（2026-10-16）
            _cond_header(ax, tag, x0=0.0, x1=1.0, dy_mm=0.2)
            ax.text(2.5, 2.5, "pending", ha="center", va="center", fontsize=8, color=st.MUTE)
            an = ax_mm(x, 9.0, pw, 26.0)
            net_axes.append(an)
            an.axis("off")
            continue
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
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor="white", lw=0.9))
            if txt:
                ax.add_patch(Rectangle((j + 0.27, i + 0.27), 0.46, 0.46, facecolor=fc, lw=0)) if k == 1 else None
                ax.text(j + 0.5, i + 0.53, txt, ha="center", va="center",
                        fontsize=(6 if abs(v) >= 10 else 7) if k == 2 else 9, color=tc)
        for k_, s_ in enumerate(SPN):
            ax.text(k_ + 0.5, -0.12, s_, ha="center", va="bottom", fontsize=7)
            if c == 0:  # 行の名前は左端の行列だけ（4 つの行列は同じ高さに並ぶ）
                ax.text(-0.15, k_ + 0.53, s_, ha="right", va="center", fontsize=7)
        _cond_header(ax, tag, x0=0.0, x1=1.0, dy_mm=0.2)
        # ---------------------------------------------------------------- ネットワーク
        an = ax_mm(x, 9.0, pw, 26.0)
        net_axes.append(an)
        an.set_xlim(-1.45, 1.45)
        an.set_ylim(-1.22, 1.38)
        an.set_aspect("equal")
        an.axis("off")
        ang = np.pi / 2 - 2 * np.pi * np.arange(5) / 5
        P = np.c_[np.cos(ang), np.sin(ang)]
        rn = 0.25
        n_id = int((cls[:, c] == 2).sum())
        n_so = int((cls[:, c] == 1).sum())
        for r, (i, j) in enumerate(ij):
            k, v = cls[r, c], M[r, c]
            if k == 0:
                continue
            colr = col_pos if v > 0 else col_neg
            if i == j:
                if k == 2:  # 自己相互作用: 節点の周りの輪
                    an.add_patch(Circle(P[i], rn + 0.085, facecolor="none", edgecolor=colr,
                                        lw=lw_of(v), zorder=2, clip_on=False))
                continue
            d = P[j] - P[i]
            u = d / np.linalg.norm(d)
            a, b = P[i] + u * (rn + 0.03), P[j] - u * (rn + 0.03)
            if k == 2:
                an.plot(*np.c_[a, b], color=colr, lw=lw_of(v), solid_capstyle="round", zorder=1)
            else:
                an.plot(*np.c_[a, b], color=colr, lw=0.7, ls=(0, (2.5, 1.6)), zorder=1)
        for k_ in range(5):
            an.add_patch(Circle(P[k_], rn, facecolor="white", edgecolor=st.INK, lw=0.5, zorder=3))
            an.text(*P[k_], SPN[k_], ha="center", va="center", fontsize=7, zorder=4)
        an.text(0, -1.2, f"{n_id} identified, {n_so} sign only", ha="center", va="top",
                fontsize=7, color=st.MUTE)

    st.panel_label(mat_axes[0], "a", dx_mm=-1.0, dy_mm=0.2)
    st.panel_label(net_axes[0], "b", dx_mm=-1.0, dy_mm=-1.5)
    # 色の目盛り（行列と同じ高さ）
    cax = ax_mm(x0 + 4 * (pw + gap) + 0.5, 40.0, 2.4, 28.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_linewidth(0.4)
    step = max(1, int(np.ceil(vmax / 3)))
    cb.set_ticks(np.arange(-(vmax // step) * step, vmax + 1e-9, step))
    cb.ax.tick_params(labelsize=7, width=0.4, length=2)
    cax.set_title(r"$a_{ij}$", fontsize=7, pad=3)
    # 凡例（下に 1 行）
    hatch_c = cmap(norm(-0.45 * vmax))
    from matplotlib.legend_handler import HandlerTuple
    H_ = [
        (Patch(facecolor=cmap(norm(-0.22 * vmax)), edgecolor=hatch_c, hatch="////", lw=0),
         Line2D([], [], color=st.MUTE, lw=0.7, ls=(0, (2.5, 1.6)))),
        Patch(facecolor=NI_GREY, edgecolor="none"),
        Line2D([], [], color=col_pos, lw=2.0),
        Line2D([], [], color=col_neg, lw=2.0),
        Line2D([], [], color=st.MUTE, marker="o", ms=6, mfc="none", mew=1.2, lw=0),
    ]
    L_ = ["sign only", "not identified", r"cooperative ($a_{ij}>0$)",
          r"competitive ($a_{ij}<0$)", "self-interaction"]
    fig.legend(H_, L_, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
               handlelength=2.6, handletextpad=0.5, columnspacing=2.2, fontsize=7,
               borderaxespad=0.3)
    st.save(fig, out_noext)
    plt.close(fig)
    return vmax


# ============================================================================ 図 4 v2
def fig4_v2(Q, U0, cls, M, order, conds, sp, out_noext):
    """Q: (15, 4, 5) 段 (iii) の箱の中での位置（5/25/50/75/95%）。U0: (15, 4) a=0 の位置（箱の外なら nan）。"""
    n = len(order)
    W, H = st.DOUBLE_MM, 98.0
    fig, axes = plt.subplots(1, 4, figsize=(W * st.MM, H * st.MM), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.17, wspace=0.12)
    for c, tag in enumerate(conds):
        ax = axes[c]
        col = st.COND_COL[tag]
        if (cls[:, c] == -1).all():  # 最終段の run がまだ無い条件（2026-10-16）
            ax.text(0.5, (n - 1) / 2, "pending", ha="center", va="center", fontsize=8, color=st.MUTE)
            ax.set_xlim(-0.03, 1.1)
            ax.set_xticks([])
            ax.spines["left"].set_visible(c == 0)
            _cond_header(ax, tag, x0=0.0, x1=0.92, dy_mm=1.2, sub="")
            continue
        for r in range(n):
            q = Q[r, c]
            ax.plot([0, 1], [r, r], color="#EDEDED", lw=0.5, zorder=0, solid_capstyle="butt")
            if np.isfinite(U0[r, c]):
                ax.plot([U0[r, c]] * 2, [r - 0.32, r + 0.32], color=st.INK, lw=0.6, zorder=5)
            if cls[r, c] == 2:
                ax.plot([q[0], q[4]], [r, r], color=col, lw=0.9, zorder=2, solid_capstyle="butt")
                ax.plot([q[1], q[3]], [r, r], color=col, lw=3.0, zorder=3, solid_capstyle="butt")
                ax.plot(q[2], r, "o", mfc="white", mec=col, ms=3.2, mew=0.9, zorder=4)
            else:
                g = "#B5B5B5"
                ax.plot([q[0], q[4]], [r, r], color=g, lw=0.7, zorder=2, solid_capstyle="butt")
                ax.plot([q[1], q[3]], [r, r], color=g, lw=2.2, zorder=3, solid_capstyle="butt")
                ax.plot(q[2], r, "o", mfc="white", mec=g, ms=2.6, mew=0.7, zorder=4)
                if cls[r, c] == 1:
                    ax.text(1.045, r, r"$+$" if M[r, c] > 0 else r"$-$", ha="left",
                            va="center", fontsize=8, color=st.INK)
        ax.set_xlim(-0.03, 1.1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["lower", "", "upper"])
        ax.spines["left"].set_visible(c == 0)
        ax.tick_params(axis="y", length=0)
        n_id = int((cls[:, c] == 2).sum())
        _cond_header(ax, tag, x0=0.0, x1=0.92, dy_mm=1.2, sub=f"{n_id} identified")
    lab = [f"{sp[s[1]]}--{sp[s[2]]}" if s[1] != s[2] else f"{sp[s[1]]} (self)" for s in order]
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(lab)
    axes[0].set_ylim(n - 0.5, -0.6)
    axes[0].spines["left"].set_visible(False)
    H_ = [
        Line2D([], [], color=st.INK, lw=3.0, label=r"50\% interval"),
        Line2D([], [], color=st.INK, lw=0.9, label=r"90\% interval"),
        Line2D([], [], color=st.INK, marker="o", mfc="white", ms=3.2, lw=0, label="median"),
        Line2D([], [], color="#B5B5B5", lw=2.2, label="not identified"),
        Line2D([], [], color=st.INK, marker="|", ms=6, lw=0, label=r"$a_{ij}=0$"),
        Line2D([], [], color="none", label=r"$\pm$: sign only"),
    ]
    fig.text(0.54, 0.085, "position of the posterior in the prior box of each entry", ha="center",
             va="center", fontsize=8)
    fig.legend(handles=H_, loc="lower center", ncol=6, bbox_to_anchor=(0.54, 0.0),
               handlelength=1.8, columnspacing=1.6, fontsize=7)
    st.save(fig, out_noext)
    plt.close(fig)


# ============================================================================ 図 5（旧 fig6）v2
def fig6_v2(ko, B, out_noext, measured_ratio=2.74):
    W, H = st.DOUBLE_MM, 60.0
    fig = plt.figure(figsize=(W * st.MM, H * st.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.32,
                          left=0.075, right=0.99, top=0.84, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    col = st.COND_COL["DH"]
    labels = ["full", r"no Fn", r"$a_{45}=0$"]
    keys = ["ratio_base", "ratio_noFn", "ratio_a45_0"]
    ax.axhline(1.0, color=st.FAINT, lw=0.5, ls=":", zorder=0)
    ax.axhline(measured_ratio, color=st.INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
    for j, k in enumerate(keys):
        for s, row in enumerate(ko):
            lo, md, hi = row[k]
            x = j + (s - 1) * 0.2
            ax.plot([x, x], [lo, hi], color=col, lw=1.0, solid_capstyle="butt")
            ax.plot(x, md, "o", color=col, ms=3.0, mec="white", mew=0.4)
    ax.text(1.0, measured_ratio + 0.12, f"measured {measured_ratio:.2f}", fontsize=7,
            ha="center", va="bottom", color=st.INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0, max(8.0, max(r[k][2] for r in ko for k in keys) * 1.05, measured_ratio * 1.15))
    ax.set_ylabel(r"Pg ratio, day 21 / day 15")
    _cond_header(ax, "DH", sub="in-silico knockout", dy_mm=1.2)
    st.panel_label(ax, "a", dx_mm=-8.0, dy_mm=1.2)
    for p, tag in enumerate(["CH", "DH"]):
        ax = fig.add_subplot(gs[p + 1])
        c = st.COND_COL[tag]
        d = B[f"{tag}_days"]
        ax.fill_between(d, B[f"{tag}_lo"], B[f"{tag}_hi"], color=c, alpha=0.22, lw=0,
                        label=r"model, 90\%")
        ax.plot(d, B[f"{tag}_med"], color=c, lw=1.2, label="model, median")
        ax.plot(B[f"{tag}_t_meas"], B[f"{tag}_ph_meas"], "o", ms=2.0, color=st.INK, mew=0,
                label="measured (not fitted)")
        ax.set_xlabel("Day")
        ax.set_xticks([0, 3, 6, 10, 15, 21])
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(6.15, 7.4)
        _cond_header(ax, tag, sub="pH, not used in the inference", dy_mm=1.2)
        st.panel_label(ax, "bc"[p], dx_mm=-8.0, dy_mm=1.2)
        if p == 0:
            ax.set_ylabel("pH")
            ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)
