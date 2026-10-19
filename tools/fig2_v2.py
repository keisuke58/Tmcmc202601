"""Fig. 2 v2（2026-10-12）: 種の割合の事後予測、5 行（種）× 4 列（条件）、190 mm 幅で最終サイズ。

tools/make_paper_figures.py の write_fig2_cache が書くキャッシュ（fig2_phi_{tag}.npz）をそのまま読む。
旧版（docs/generate_fig2_phi_version.py）からの変更:
  - 凡例は 1 行、上に。色は中立（インク色）、各列の色は条件の色（tools/eml_style.py、全図共通）
  - 列見出しに条件の色の帯、RMSE は 2 行目に小さく
  - 株の名前（V. dispar / V. parvula、Pg DSM 20709 / W83）は枠の右上に小さく
  - グリッドは y だけ、薄く。線幅をそろえる。日の目盛りは 1, 3, 6, 10, 15, 21
  color_by="species" で旧版の種の色に戻せる。
"""
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

import eml_style as st

CONDS = ["CS", "CH", "DS", "DH"]
CK = {"CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
      "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC")}
ROW_LABEL = [r"\textit{S.~oralis}", r"\textit{A.~naeslundii}", r"\textit{Veillonella}",
             r"\textit{F.~nucleatum}", r"\textit{P.~gingivalis}"]
STRAIN = {2: {"C": r"\textit{V.~dispar}", "D": r"\textit{V.~parvula}"},
          4: {"C": r"DSM\,20709", "D": r"W83"}}
SP_COLORS = {"C": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#E07070"],
             "D": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#8B0000"]}
REP_SP_MAP = {"S. oralis": 0, "A. naeslundii": 1, "V. dispar": 2, "V. parvula": 2,
              "F. nucleatum": 3, "P. gingivalis_20709": 4, "P. gingivalis_W83": 4}
DAYS = [1, 3, 6, 10, 15, 21]
DAY_GRID = np.linspace(1, 21, 2501)


def make(cache_dir, out_noext, rmse, rep_csv, color_by="condition", height_mm=188.0,
         group_gap=0.035, show_phibar=False, thesis=False, stages=None, keep_layout=False):
    """thesis=True: 修論用（2026-10-19）。幅 160 mm・Times・目盛り 8 pt・軸ラベル 9 pt・見出し 10 pt、
    列見出しは「CS (stage iv)」だけ（RMSE はキャプションへ）、凡例は図の下に 1 つ。stages = {tag: "iv"}。"""
    (st.apply_thesis if thesis else st.apply)()
    fs_lab, fs_head, fs_small = (9, 10, 8) if thesis else (8, 8, 7)
    ms_s = 16 if thesis else 9  # 点の面積（pt^2）: 16 ≈ 直径 4 pt
    rep = pd.read_csv(rep_csv)
    W, H = (st.THESIS_MM, height_mm) if thesis else (st.DOUBLE_MM, height_mm)
    fig, axes = plt.subplots(5, 4, figsize=(W * st.MM, H * st.MM), sharex=True, sharey=True)
    avail = [c for c in CONDS if Path(f"{cache_dir}/fig2_phi_{c}.npz").exists()]  # 未確定の条件は空の列（2026-10-15）
    if not avail:
        raise FileNotFoundError(f"{cache_dir}/fig2_phi_*.npz が無い")
    _z0 = np.load(f"{cache_dir}/fig2_phi_{avail[0]}.npz")
    _two0 = bool(_z0["two_channel"]) if "two_channel" in _z0.files else False  # 2 通路なら凡例が 2 行
    if thesis and keep_layout:  # 前の版の配置（見出しに条件名と RMSE、凡例は上）のまま、幅 160 mm・Times・8 pt 以上（2026-10-19）
        fig.subplots_adjust(left=0.105, right=0.99, top=0.87, bottom=0.055, hspace=0.14, wspace=0.08)
    elif thesis:
        fig.subplots_adjust(left=0.105, right=0.99, top=0.945, bottom=0.12, hspace=0.14, wspace=0.08)
    else:
        fig.subplots_adjust(left=0.085, right=0.995, top=0.875 if _two0 else 0.893, bottom=0.05, hspace=0.12, wspace=0.07)
    # commensal（CS, CH）と dysbiotic（DS, DH）の間を広げる: Veillonella 種と Pg 株が組で違うため
    if group_gap > 0:
        p0, p3 = axes[0, 0].get_position(), axes[0, 3].get_position()
        sp = axes[0, 1].get_position().x0 - p0.x1
        w = (p3.x1 - p0.x0 - 3 * sp - group_gap) / 4
        for c in range(4):
            x0 = p0.x0 + c * (w + sp) + (group_gap if c >= 2 else 0)
            for i in range(5):
                q = axes[i, c].get_position()
                axes[i, c].set_position([x0, q.y0, w, q.height])
    for c, tag in enumerate(CONDS):
        if tag not in avail:
            for i in range(5):
                axes[i, c].text(0.5, 0.5, "pending" if i == 2 else "", transform=axes[i, c].transAxes,
                                ha="center", va="center", fontsize=7, color=st.MUTE)
            axes[0, c].set_title(rf"\textbf{{{tag}}}", fontsize=8, color=st.COND_COL[tag])
            continue
        z = np.load(f"{cache_dir}/fig2_phi_{tag}.npz")
        trajs = z["trajs"]
        q10, q25, q75, q90 = np.percentile(trajs, [10, 25, 75, 90], axis=0)
        two = bool(z["two_channel"]) if "two_channel" in z.files else False  # 2 通路: 線は φψ/Σφψ、全菌は別の印
        main_map = z["viable_map"] if two else z["phi_map"]
        days = np.concatenate([[1], z["t_days"]])
        dat = np.vstack([z["d1n"][None, :], z["data_norm"]])
        cond, cult = CK[tag]
        cr = rep[(rep.condition == cond) & (rep.cultivation == cult)]
        for i in range(5):
            ax = axes[i, c]
            col = st.COND_COL[tag] if color_by == "condition" else SP_COLORS[tag[0]][i]
            names = [k for k, v in REP_SP_MAP.items() if v == i]
            bd, bp = [], []
            for d in DAYS:
                v = cr[(cr.day == d) & (cr.species.isin(names))]["distribution_pct"].values / 100.0
                if len(v):
                    bd.append(v)
                    bp.append(d)
            if bd:
                bx = ax.boxplot(bd, positions=bp, widths=1.5, patch_artist=True, showfliers=False,
                                manage_ticks=False, zorder=1,
                                medianprops=dict(color=st.INK, lw=0.6),
                                whiskerprops=dict(color="#9A9A9A", lw=0.45),
                                capprops=dict(color="#9A9A9A", lw=0.45),
                                boxprops=dict(lw=0.4))
                for p in bx["boxes"]:
                    p.set_facecolor(mpl.colors.to_rgba(col, 0.13))
                    p.set_edgecolor(mpl.colors.to_rgba(col, 0.6))
            ax.fill_between(DAY_GRID, q10[:, i], q90[:, i], color=col, alpha=0.12, lw=0, zorder=2)
            ax.fill_between(DAY_GRID, q25[:, i], q75[:, i], color=col, alpha=0.28, lw=0, zorder=2)
            ax.plot(DAY_GRID, main_map[:, i], color=col, lw=1.2, zorder=4)
            if two:  # 全菌の組成（PMA なし）と φ/Σφ
                ax.plot(DAY_GRID, z["total_map"][:, i], color=col, lw=0.7, ls=(0, (2, 1.4)), alpha=0.9, zorder=3)
                ax.scatter(z["t_days"], z["data_total"][:, i], s=ms_s, facecolors="white", edgecolors=col,
                           linewidths=0.6, zorder=5)
            if show_phibar:  # 種ごとの φψ はデータで決まらない（2026-10-13、既定で出さない）
                ax.plot(DAY_GRID, z["phibar_map"][:, i], color=col, lw=0.8, ls=(0, (3, 1.6)),
                        alpha=0.8, zorder=3)
            ax.scatter(days, dat[:, i], s=ms_s, color=col, edgecolors=st.INK, linewidths=0.35,
                       zorder=5)
            ax.set_xlim(-0.2, 22.2)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(DAYS)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.grid(axis="y", color="#ECECEC", lw=0.35)
            ax.set_axisbelow(True)
            ax.tick_params(length=2)
            if i in STRAIN:
                ax.text(0.97, 0.95, STRAIN[i][tag[0]], transform=ax.transAxes, ha="right",
                        va="top", fontsize=fs_small, color=st.MUTE)
            if c == 0:
                ax.set_ylabel(ROW_LABEL[i], fontsize=fs_lab, labelpad=3)
            if i == 4:
                ax.set_xlabel("Day", labelpad=1.5)
        # 列見出し: 色の帯 + 条件名 + RMSE
        top = axes[0, c]
        tr = lambda dy: top.transAxes + ScaledTranslation(0, dy * st.MM, fig.dpi_scale_trans)  # noqa: E731
        if thesis and keep_layout:
            top.plot([0, 1], [1, 1], transform=tr(11.5), color=st.COND_COL[tag], lw=2.2,
                     solid_capstyle="butt", clip_on=False)
            top.text(0.5, 1, rf"\textbf{{{tag}}}\ \ {st.COND_NAME[tag]}", transform=tr(5.6),
                     ha="center", va="bottom", fontsize=fs_lab)
            if tag in rmse and rmse[tag] is not None:
                top.text(0.5, 1, rf"RMSE ${rmse[tag]:.3f}$", transform=tr(1.0), ha="center",
                         va="bottom", fontsize=fs_small, color=st.MUTE)
            continue
        if thesis:
            top.plot([0, 1], [1, 1], transform=tr(6.0), color=st.COND_COL[tag], lw=2.2,
                     solid_capstyle="butt", clip_on=False)
            _st = (stages or {}).get(tag)
            top.text(0.5, 1, tag + (f" (stage {_st})" if _st else ""), transform=tr(1.2),
                     ha="center", va="bottom", fontsize=fs_head)
            continue
        top.plot([0, 1], [1, 1], transform=tr(9.6), color=st.COND_COL[tag], lw=2.2,
                 solid_capstyle="butt", clip_on=False)
        top.text(0.5, 1, rf"\textbf{{{tag}}}\ \ {st.COND_NAME[tag]}", transform=tr(4.6),
                 ha="center", va="bottom", fontsize=8)
        if tag in rmse and rmse[tag] is not None:
            top.text(0.5, 1, rf"RMSE ${rmse[tag]:.3f}$", transform=tr(1.0), ha="center",
                     va="bottom", fontsize=7, color=st.MUTE)
    fig.text(0.008, 0.46, r"species fraction" if two else r"volume fraction $\varphi_i$", rotation=90,
             ha="left", va="center", fontsize=fs_lab)
    g = st.MUTE
    H_ = [Line2D([], [], color=g, lw=1.2), Line2D([], [], color=g, lw=0.8, ls=(0, (3, 1.6))),
          Patch(facecolor=g, alpha=0.40, lw=0), Patch(facecolor=g, alpha=0.16, lw=0),
          Line2D([], [], marker="o", color="none", mfc=g, mec=st.INK, mew=0.35, ms=3.6),
          Patch(facecolor=mpl.colors.to_rgba(g, 0.13), edgecolor=mpl.colors.to_rgba(g, 0.6), lw=0.4)]
    L_ = [r"MAP $\varphi_i$", r"MAP $\bar\varphi_i=\varphi_i\psi_i$", r"50\% band", r"80\% band",
          "data (median)", "replicates (IQR)"]
    if two:  # 2 通路: 実線 = 生きた菌の予測、破線 = 全菌の予測、白抜き = 全菌のデータ
        H_ = [Line2D([], [], color=g, lw=1.2), Line2D([], [], color=g, lw=0.7, ls=(0, (2, 1.4))),
              Patch(facecolor=g, alpha=0.40, lw=0), Patch(facecolor=g, alpha=0.16, lw=0),
              Line2D([], [], marker="o", color="none", mfc=g, mec=st.INK, mew=0.35, ms=3.6),
              Line2D([], [], marker="o", color="none", mfc="white", mec=g, mew=0.6, ms=3.6),
              Patch(facecolor=mpl.colors.to_rgba(g, 0.13), edgecolor=mpl.colors.to_rgba(g, 0.6), lw=0.4)]
        L_ = [r"MAP viable $\varphi_i\psi_i/\sum_j\varphi_j\psi_j$", r"MAP total $\varphi_i/\sum_j\varphi_j$",
              r"50\% band", r"80\% band", "viable data (median)", "total data (median)", "viable replicates"]
    elif not show_phibar:
        del H_[1], L_[1]
    if thesis and keep_layout:
        fig.legend(H_, L_, loc="upper center", ncol=4 if two else len(L_), bbox_to_anchor=(0.55, 1.0), fontsize=8,
                   handlelength=2.0, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.2)
    elif thesis:
        if two:
            L_[0], L_[1] = "MAP, viable", "MAP, total"
        fig.legend(H_, L_, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.0), fontsize=8,
                   handlelength=2.0, handletextpad=0.5, columnspacing=1.5, borderaxespad=0.2)
    else:
        fig.legend(H_, L_, loc="upper center", ncol=4 if two else len(L_), bbox_to_anchor=(0.54, 1.0), fontsize=7,
                   handlelength=2.0, handletextpad=0.5, columnspacing=1.8, borderaxespad=0.2)
    st.save(fig, out_noext)
    plt.close(fig)
