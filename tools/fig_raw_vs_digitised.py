#!/usr/bin/env python3
"""補足 sec:S_raw の図: 論文の図から読み取った値と実測値（Heine 提供）の比較（2026-10-13）。

上段: 種の割合（日 × 種の中央値）。横 = 読み取り、縦 = 実測。塗り = 生きた菌（PMA あり）、白抜き = 全菌（PMA なし）。
下段: 群集の生きた割合（バイオフィルムの平均 ± SD、5 枚 = 1 バイオフィルム）と読み取り値（×）。
出すのは中央値・平均だけ（反復の値そのものは出さない。データは請求に応じて提供、N. Heine）。

    python3 tools/fig_raw_vs_digitised.py --out docs/revision/EML_submission/figures/raw/supp_raw_vs_digitised
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eml_style as st  # noqa: E402
import raw_data_report as R  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SP_COL = ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#E07070"]  # fig2_v2 と同じ種の色


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    q, ld, _v = R.load()
    memb = pd.read_csv(R.DIG / "fig2_membrane_distribution.csv")
    st.apply()
    fig, axs = plt.subplots(2, 4, figsize=(st.DOUBLE_MM * st.MM, 92 * st.MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.9, wspace=0.32, hspace=0.62)
    for j, (tag, (c, cu)) in enumerate(R.CONDS.items()):
        ax = axs[0, j]
        for pool, fn, filled in (("viable", "fig3_species_distribution_summary.csv", True),
                                 ("total", "figS1_total_species_distribution_summary.csv", False)):
            r, dg = R.raw_comp_all(q, c, cu, pool), R.dig_comp(fn, c, cu)
            for d in R.DAYS:
                if d not in r or d not in dg:
                    continue
                for i in range(5):
                    ax.scatter(dg[d][i], r[d][i], s=9, marker="o",
                               facecolors=SP_COL[i] if filled else "white", edgecolors=SP_COL[i],
                               linewidths=0.6, zorder=3)
        ax.plot([0, 1], [0, 1], color=st.MUTE, lw=0.6, zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(rf"\textbf{{{tag}}}", fontsize=8, color=st.COND_COL[tag])
        ax.set_xlabel("read off", fontsize=7)
        if j == 0:
            ax.set_ylabel("measured", fontsize=7)
        ax = axs[1, j]
        x = ld[(ld.condition == c) & (ld.cultivation == cu)].assign(biofilm=lambda z: (z.image - 1) // 5)
        bm = x.groupby(["day", "biofilm"]).live_pct.mean().reset_index().groupby("day").live_pct
        ax.errorbar(bm.mean().index, bm.mean() / 100, bm.std() / 100, fmt="o", ms=3, color=st.COND_COL[tag],
                    capsize=1.5, lw=0.7, label="measured (biofilm mean $\\pm$ SD)")
        m = memb[(memb.condition == c) & (memb.cultivation == cu)]
        ax.scatter(m.day, m.intact_membrane_pct / 100, marker="x", s=14, color=st.INK, lw=0.7, zorder=4,
                   label="read off")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(R.DAYS)
        ax.set_xlabel("Day", fontsize=7)
        if j == 0:
            ax.set_ylabel("live fraction", fontsize=7)
    names = ["So", "An", "Vei", "Fn", "Pg"]
    from matplotlib.lines import Line2D
    H = [Line2D([], [], marker="o", ls="", color=SP_COL[i], ms=3.5) for i in range(5)]
    H += [Line2D([], [], marker="o", ls="", mfc=st.MUTE, mec=st.MUTE, ms=3.5),
          Line2D([], [], marker="o", ls="", mfc="white", mec=st.MUTE, ms=3.5)]
    fig.legend(H, names + ["viable (PMA)", "total (no PMA)"], loc="upper center", ncol=7, fontsize=7,
               frameon=False, bbox_to_anchor=(0.53, 1.0), handletextpad=0.3, columnspacing=1.2)
    h2, l2 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h2, l2, loc="center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.53, 0.475))
    st.save(fig, args.out)
    print("->", args.out)


if __name__ == "__main__":
    main()
