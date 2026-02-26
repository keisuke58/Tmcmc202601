#!/usr/bin/env python3
"""
regenerate_improved_figures.py
================================
改善版図の一括再生成。4つの問題を修正：
  1. A3: Y軸ズーム（基材 0.55-0.65 MPa）+ 差をパーセント表示
  2. DI cross-condition: 上段=全体, 下段=dh-baseline を除く拡大
  3. E posterior violin: 外れ値クリップ + サマリーパネル改善
  4. B3 Gc scatter: メインクラスターにズーム + 外れ値数を注記

出力先: 各元フォルダに _improved サフィックスで保存
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

_HERE = Path(__file__).resolve().parent

CONDITIONS = ["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"]
LABELS = {
    "dh_baseline": "dh-baseline",
    "commensal_static": "Comm.\nStatic",
    "commensal_hobic": "Comm.\nHOBIC",
    "dysbiotic_static": "Dysb.\nStatic",
}
LABELS_FLAT = {k: v.replace("\n", " ") for k, v in LABELS.items()}
COLORS = {
    "dh_baseline": "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. A3 図改善：Y軸ズーム + パーセント差 + 有意差矢印
# ─────────────────────────────────────────────────────────────────────────────
def fix_A3():
    import csv

    csv_path = _HERE / "_material_sweep" / "results.csv"
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["sweep"] == "A2" and abs(float(r["n_exp"]) - 2.0) < 0.1:
                rows.append(r)

    # variant -> (sub, surf)
    variants = {}
    for r in rows:
        v = r["variant"]
        variants[v] = (float(r["substrate_smises"]) / 1e6, float(r["surface_smises"]) / 1e6)

    order = ["dh_old", "mild_weight", "nolambda"]
    vlabel = {
        "dh_old": "dh-old\n(a₃₅=21.4)",
        "mild_weight": "mild-weight\n(a₃₅=3.56)",
        "nolambda": "no-lambda\n(a₃₅=20.9)",
    }
    vcolors = {"dh_old": "#d62728", "mild_weight": "#2ca02c", "nolambda": "#1f77b4"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for si, (col_idx, slabel, ylim, ref_key) in enumerate(
        [
            (0, "Substrate", (0.565, 0.635), "dh_old"),
            (1, "Surface", (0.960, 1.010), "dh_old"),
        ]
    ):
        ax = axes[si]
        subs = [variants[v][col_idx] for v in order]
        ref = variants[ref_key][col_idx]
        cols = [vcolors[v] for v in order]
        x = np.arange(len(order))

        bars = ax.bar(
            x, subs, width=0.55, color=cols, alpha=0.85, edgecolor="k", linewidth=0.7, zorder=3
        )

        # 数値ラベル + ref からの差
        for xi, (v, s) in enumerate(zip(order, subs)):
            delta = (s - ref) / ref * 100
            sign = "+" if delta >= 0 else ""
            ax.text(
                xi,
                s + (ylim[1] - ylim[0]) * 0.015,
                f"{s:.3f} MPa\n({sign}{delta:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                fontweight="bold" if v == "mild_weight" else "normal",
            )

        # mild_weight に注目矢印
        mw_idx = order.index("mild_weight")
        mw_val = variants["mild_weight"][col_idx]
        dh_val = variants["dh_old"][col_idx]
        ax.annotate(
            "",
            xy=(mw_idx, mw_val),
            xytext=(0, mw_val),
            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.8),
        )
        ax.text(
            mw_idx / 2,
            (mw_val + dh_val) / 2,
            f"Δ={(mw_val-dh_val)/dh_val*100:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#2ca02c",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([vlabel[v] for v in order], fontsize=10)
        ax.set_ylabel("$S_\\mathrm{Mises}$ (MPa)", fontsize=11)
        ax.set_ylim(ylim)
        ax.set_title(f"A3: θ Variant Comparison — {slabel}", fontsize=12)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

        # 参照線（dh_old）
        ax.axhline(ref, color="#d62728", lw=0.8, ls=":", alpha=0.6)

    fig.suptitle(
        "A3: TMCMC θ Variant → $S_\\mathrm{Mises}$ (Y-axis zoomed for clarity)\n"
        "mild-weight (a₃₅=3.56) reduces substrate stress vs unconstrained dh-old (a₃₅=21.4)",
        fontsize=12,
        fontweight="bold",
    )
    out = _HERE / "_material_sweep" / "figures" / "fig_A3_theta_comparison_improved.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DI cross-condition 改善：上段=全体（dh 支配）、下段=dh なしで拡大
# ─────────────────────────────────────────────────────────────────────────────
def fix_DI():
    di_base = _HERE / "_di_credible"

    # 各条件の di_quantiles.npy: (3, 3375) → (p05, p50, p95) × nodes
    # ノードを x 座標でソートして深さプロファイルを作る
    data = {}
    for cond in CONDITIONS:
        q = np.load(di_base / cond / "di_quantiles.npy")  # (3, 3375)
        coords = np.load(di_base / cond / "coords.npy")  # (3375, 3)
        x_vals = coords[:, 0]  # depth axis
        # 各 x スライスの中央値
        x_unique = np.unique(np.round(x_vals, 6))
        profile = {qt: [] for qt in ["p05", "p50", "p95"]}
        for xi in x_unique:
            mask = np.abs(x_vals - xi) < 1e-5
            for qi, qt in enumerate(["p05", "p50", "p95"]):
                profile[qt].append(np.median(q[qi][mask]))
        data[cond] = {"x": x_unique, **profile}

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_all_med = fig.add_subplot(gs[0, 0])  # 全体・中央値
    ax_all_band = fig.add_subplot(gs[0, 1])  # 全体・帯
    ax_zoom_med = fig.add_subplot(gs[1, 0])  # dh なし・中央値
    ax_zoom_ci = fig.add_subplot(gs[1, 1])  # 不確かさ幅

    # ── 上段: 全体 ──────────────────────────────────────────────
    for cond in CONDITIONS:
        d = data[cond]
        c = COLORS[cond]
        ax_all_med.plot(d["x"], d["p50"], color=c, lw=1.8, label=LABELS_FLAT[cond])
        ax_all_band.fill_between(
            d["x"], d["p05"], d["p95"], color=c, alpha=0.35, label=LABELS_FLAT[cond]
        )
        ax_all_band.plot(d["x"], d["p50"], color=c, lw=1.2)

    for ax, title in [
        (ax_all_med, "Median DI (all cond.)"),
        (ax_all_band, "p05–p95 band (all cond.)"),
    ]:
        ax.set_xlabel("Depth x (0=substrate, 1=surface)", fontsize=9)
        ax.set_ylabel("Dysbiotic Index (DI)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, ls="--")

    # ── 下段: dh-baseline を除いた拡大 ──────────────────────────
    conds_no_dh = [c for c in CONDITIONS if c != "dh_baseline"]

    for cond in conds_no_dh:
        d = data[cond]
        c = COLORS[cond]
        ax_zoom_med.plot(d["x"], d["p50"], color=c, lw=2.0, label=LABELS_FLAT[cond])
        ax_zoom_ci.fill_between(
            d["x"], d["p05"], d["p95"], color=c, alpha=0.4, label=LABELS_FLAT[cond]
        )
        ax_zoom_ci.plot(d["x"], d["p50"], color=c, lw=1.5)

    # dh-baseline の信頼帯を薄く追加（参照用）
    d_dh = data["dh_baseline"]
    ax_zoom_med.plot(
        d_dh["x"],
        d_dh["p50"],
        color=COLORS["dh_baseline"],
        lw=1.0,
        ls="--",
        alpha=0.4,
        label="dh-baseline (ref)",
    )
    ax_zoom_ci.plot(d_dh["x"], d_dh["p50"], color=COLORS["dh_baseline"], lw=0.8, ls="--", alpha=0.3)

    for ax, title in [
        (ax_zoom_med, "Zoomed: Median DI (excl. dh-baseline)"),
        (ax_zoom_ci, "Zoomed: p05–p95 band (excl. dh-baseline)"),
    ]:
        ax.set_xlabel("Depth x (0=substrate, 1=surface)", fontsize=9)
        ax.set_ylabel("DI (zoomed)", fontsize=9)
        ax.set_ylim(-0.001, 0.025)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    fig.suptitle(
        "B1: DI Depth Profiles — Cross-Condition Comparison\n"
        "Top: all 4 conditions (dh-baseline dominates scale).  "
        "Bottom: commensal/dysbiotic conditions zoomed (DI ≈ 0.001–0.020).",
        fontsize=12,
        fontweight="bold",
    )
    out = _HERE / "_di_credible" / "fig_di_cross_condition_improved.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. E 改善：violin ズーム + サマリーパネル outlier 対策
# ─────────────────────────────────────────────────────────────────────────────
def fix_E():
    pa_base = _HERE / "_posterior_abaqus"
    pu_base = _HERE / "_posterior_uncertainty"

    data = {}
    for cond in CONDITIONS:
        data[cond] = np.load(pa_base / cond / "stress_all.npy")  # (20,2)

    # ── Fig E-improved: violin + CI + 外れ値注記 ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    YLIMS = [(0.35, 0.95), (0.88, 1.10)]  # substrate, surface (クリップ)

    for si, (col, slabel, ylim) in enumerate(
        [
            (0, "Substrate (歯面)", YLIMS[0]),
            (1, "Surface (バイオフィルム表面)", YLIMS[1]),
        ]
    ):
        ax = axes[si]
        vals_list = [data[c][:, col] / 1e6 for c in CONDITIONS]
        colors = [COLORS[c] for c in CONDITIONS]
        xlabs = [LABELS[c] for c in CONDITIONS]

        # violinplot
        parts = ax.violinplot(
            vals_list,
            positions=range(len(CONDITIONS)),
            showmedians=False,
            showextrema=False,
            widths=0.6,
        )
        for body, col_ in zip(parts["bodies"], colors):
            body.set_facecolor(col_)
            body.set_alpha(0.38)

        # boxplot (clipped)
        bp = ax.boxplot(
            vals_list,
            positions=range(len(CONDITIONS)),
            widths=0.30,
            patch_artist=True,
            medianprops=dict(color="black", lw=2.2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker="o", ms=5, alpha=0.3, markeredgewidth=0.5),
        )
        for patch, col_ in zip(bp["boxes"], colors):
            patch.set_facecolor(col_)
            patch.set_alpha(0.75)

        # 個別サンプル散布
        for xi, (vals, col_) in enumerate(zip(vals_list, colors)):
            jitter = np.random.default_rng(42).uniform(-0.09, 0.09, len(vals))
            # ylim 外の外れ値は記号で表示
            in_range = (vals >= ylim[0]) & (vals <= ylim[1])
            out_range = ~in_range
            ax.scatter(
                xi + jitter[in_range], vals[in_range], color=col_, s=22, zorder=5, alpha=0.85
            )
            if out_range.any():
                n_out = out_range.sum()
                ax.text(
                    xi,
                    ylim[1] - (ylim[1] - ylim[0]) * 0.05,
                    f"▲{n_out}外",
                    ha="center",
                    fontsize=7.5,
                    color=col_,
                    fontweight="bold",
                )

        # p05/p50/p95 アノテーション
        for xi, vals in enumerate(vals_list):
            p50 = np.percentile(vals, 50)
            ax.text(xi + 0.22, p50, f"{p50:.3f}", va="center", fontsize=8, color="black")

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(xlabs, fontsize=10)
        ax.set_ylabel("$S_\\mathrm{Mises}$ (MPa)", fontsize=11)
        ax.set_ylim(ylim)
        ax.set_title(f"Posterior $S_M$ — {slabel}", fontsize=11)
        ax.grid(axis="y", alpha=0.3, ls="--")
        # p05/p95 横線
        for xi, vals in enumerate(vals_list):
            for pct, ls_ in [(5, ":"), (95, ":")]:
                v = np.percentile(vals, pct)
                if ylim[0] <= v <= ylim[1]:
                    ax.plot(
                        [xi - 0.22, xi + 0.22], [v, v], color=colors[xi], lw=0.9, ls=ls_, alpha=0.7
                    )

    fig.suptitle(
        "E: Posterior $S_\\mathrm{Mises}$ — 20 TMCMC Samples × 4 Conditions\n"
        "(Y-axis clipped; ▲n = n samples outside range shown separately)",
        fontsize=12,
        fontweight="bold",
    )
    out = pu_base / "Fig1_stress_violin_improved.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")

    # ── Fig E-improved summary panel ─────────────────────────────
    import json as json_mod

    spear = {}
    for cond in CONDITIONS:
        with (pa_base / cond / "sensitivity_spearman.json").open() as f:
            s = json_mod.load(f)
        param_names = s["param_names"]
        spear[cond] = {
            "names": param_names,
            "substrate": np.array([s["spearman_substrate"][p] for p in param_names]),
        }

    PARAM_NICE = {
        "a11": "a₁₁(So·So)",
        "a12": "a₁₂(So·An)",
        "a22": "a₂₂(An·An)",
        "b1": "b₁(So grow)",
        "b2": "b₂(An grow)",
        "a33": "a₃₃(Vd·Vd)",
        "a34": "a₃₄(Vd·Fn)",
        "a44": "a₄₄(Fn·Fn)",
        "b3": "b₃(Vd grow)",
        "b4": "b₄(Fn grow)",
        "a13": "a₁₃(So·Vd)",
        "a14": "a₁₄(So·Fn)",
        "a23": "a₂₃(An·Vd)",
        "a24": "a₂₄(An·Fn)",
        "a55": "a₅₅(Pg·Pg)",
        "b5": "b₅(Pg grow)",
        "a15": "a₁₅(So·Pg)",
        "a25": "a₂₅(An·Pg)",
        "a35": "a₃₅(Vd→Pg)",
        "a45": "a₄₅(Fn→Pg)",
    }
    param_names = spear["dh_baseline"]["names"]

    fig = plt.figure(figsize=(15, 9))
    gs_main = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.42, wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.08
    )
    ax_vln = fig.add_subplot(gs_main[0, :2])
    ax_sens = fig.add_subplot(gs_main[0, 2])
    ax_ci_s = fig.add_subplot(gs_main[1, 0])
    ax_ci_u = fig.add_subplot(gs_main[1, 1])
    ax_dlt = fig.add_subplot(gs_main[1, 2])

    # violin (substrate only, zoomed)
    vals_list = [data[c][:, 0] / 1e6 for c in CONDITIONS]
    colors_l = [COLORS[c] for c in CONDITIONS]
    parts = ax_vln.violinplot(
        vals_list,
        positions=range(len(CONDITIONS)),
        showmedians=False,
        showextrema=False,
        widths=0.55,
    )
    for body, col_ in zip(parts["bodies"], colors_l):
        body.set_facecolor(col_)
        body.set_alpha(0.38)
    bp = ax_vln.boxplot(
        vals_list,
        positions=range(len(CONDITIONS)),
        widths=0.28,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.1),
        capprops=dict(lw=1.1),
        flierprops=dict(marker="o", ms=4, alpha=0.3),
    )
    for patch, col_ in zip(bp["boxes"], colors_l):
        patch.set_facecolor(col_)
        patch.set_alpha(0.75)
    for xi, (vals, col_) in enumerate(zip(vals_list, colors_l)):
        jit = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        in_r = vals <= 0.92
        ax_vln.scatter(xi + jit[in_r], vals[in_r], color=col_, s=18, zorder=5, alpha=0.8)
        n_out = (~in_r).sum()
        if n_out:
            ax_vln.text(
                xi, 0.91, f"▲{n_out}", ha="center", fontsize=7.5, color=col_, fontweight="bold"
            )

    ax_vln.set_xticks(range(len(CONDITIONS)))
    ax_vln.set_xticklabels([LABELS_FLAT[c] for c in CONDITIONS], fontsize=10)
    ax_vln.set_ylabel("$S_M$ Substrate (MPa)", fontsize=10)
    ax_vln.set_ylim(0.35, 0.95)
    ax_vln.set_title("Posterior Distribution — Substrate $S_\\mathrm{Mises}$ (zoomed)", fontsize=10)
    ax_vln.grid(axis="y", alpha=0.3, ls="--")
    ax_vln.legend(
        handles=[
            mpatches.Patch(color=COLORS[c], label=LABELS_FLAT[c], alpha=0.75) for c in CONDITIONS
        ],
        fontsize=8,
        loc="upper right",
    )

    # sensitivity top-8
    mean_abs_rho = np.mean(np.abs(np.array([spear[c]["substrate"] for c in CONDITIONS])), axis=0)
    top_idx = np.argsort(mean_abs_rho)[::-1][:8]
    top_rhos = mean_abs_rho[top_idx]
    top_labs = [PARAM_NICE.get(param_names[i], param_names[i]) for i in top_idx]
    ypos = np.arange(len(top_idx))
    ax_sens.barh(ypos, top_rhos, color="steelblue", alpha=0.8, edgecolor="k", lw=0.5)
    ax_sens.set_yticks(ypos)
    ax_sens.set_yticklabels(top_labs, fontsize=8.5)
    ax_sens.set_xlabel("Mean |Spearman ρ|", fontsize=9)
    ax_sens.set_title("Param Sensitivity (substrate)", fontsize=10)
    ax_sens.grid(axis="x", alpha=0.3, ls="--")
    ax_sens.set_xlim(0, 0.65)

    # CI bars substrate + surface
    x = np.arange(len(CONDITIONS))
    for ax, col, slabel in [(ax_ci_s, 0, "Substrate"), (ax_ci_u, 1, "Surface")]:
        ylim_ci = (0.45, 0.85) if col == 0 else (0.88, 1.06)
        p50 = np.array([np.percentile(data[c][:, col], 50) / 1e6 for c in CONDITIONS])
        p05 = np.array([np.percentile(data[c][:, col], 5) / 1e6 for c in CONDITIONS])
        p95 = np.array([np.percentile(data[c][:, col], 95) / 1e6 for c in CONDITIONS])
        ax.bar(x, p50, color=colors_l, alpha=0.82, edgecolor="k", lw=0.5)
        ax.errorbar(
            x, p50, yerr=[p50 - p05, p95 - p50], fmt="none", color="black", capsize=4, lw=1.3
        )
        for xi, (v, lo, hi) in enumerate(zip(p50, p05, p95)):
            ax.text(
                xi,
                hi + (ylim_ci[1] - ylim_ci[0]) * 0.012,
                f"{v:.3f}\n[{lo:.3f}–{hi:.3f}]",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS_FLAT[c] for c in CONDITIONS], fontsize=8.5, rotation=10, ha="right"
        )
        ax.set_ylabel("$S_M$ (MPa)", fontsize=9)
        ax.set_title(f"CI p5–p95 — {slabel}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_ylim(ylim_ci)

    # delta vs commensal_static
    ref_sub = np.median(data["commensal_static"][:, 0])
    ref_surf = np.median(data["commensal_static"][:, 1])
    d_sub = [(np.median(data[c][:, 0]) - ref_sub) / ref_sub * 100 for c in CONDITIONS]
    d_surf = [(np.median(data[c][:, 1]) - ref_surf) / ref_surf * 100 for c in CONDITIONS]
    xd = np.arange(len(CONDITIONS))
    wd = 0.35
    ax_dlt.bar(xd - wd / 2, d_sub, wd, color=colors_l, alpha=0.85, edgecolor="k", lw=0.5)
    ax_dlt.bar(
        xd + wd / 2, d_surf, wd, color=colors_l, alpha=0.4, edgecolor="k", lw=0.5, hatch="//"
    )
    ax_dlt.axhline(0, color="k", lw=0.8, ls="--")
    # 数値ラベル
    for xi, (ds, du) in enumerate(zip(d_sub, d_surf)):
        ax_dlt.text(
            xi - wd / 2, ds + (0.15 if ds >= 0 else -0.4), f"{ds:+.1f}%", ha="center", fontsize=7.5
        )
    ax_dlt.set_xticks(xd)
    ax_dlt.set_xticklabels(
        [LABELS_FLAT[c] for c in CONDITIONS], fontsize=8.5, rotation=10, ha="right"
    )
    ax_dlt.set_ylabel("Δ vs Comm. Static (%)", fontsize=9)
    ax_dlt.set_title("Relative Change vs Reference", fontsize=10)
    ax_dlt.legend(
        handles=[
            mpatches.Patch(color="gray", alpha=0.85, label="Substrate"),
            mpatches.Patch(color="gray", alpha=0.4, hatch="//", label="Surface"),
        ],
        fontsize=8,
    )
    ax_dlt.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle(
        "E: Posterior $S_\\mathrm{Mises}$ Uncertainty — 20 TMCMC Samples × 4 Conditions\n"
        "dh-baseline: wide spread (p95/p05 = 1.58×) due to unconstrained $a_{35}$  |  "
        "Commensal: tight (1.05–1.17×)",
        fontsize=12,
        fontweight="bold",
    )
    out2 = pu_base / "Fig5_stress_summary_panel_improved.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"[done] {out2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. B3 改善：Gc scatter ズーム + 外れ値注記 + 反直感説明
# ─────────────────────────────────────────────────────────────────────────────
def fix_B3():
    czm_base = _HERE / "_czm3d"

    # 解析結果再読み込み
    import csv as csv_mod

    results_raw = {}
    with (czm_base / "czm_analytical.csv").open() as f:
        for r in csv_mod.DictReader(f):
            c = r["condition"]
            qt = r["di_qtag"]
            if c not in results_raw:
                results_raw[c] = {}
            results_raw[c][qt] = {
                k: float(v) if k not in ("condition", "di_qtag") else v
                for k, v in r.items()
                if k not in ("condition", "di_qtag")
            }

    # posterior CZM
    DI_SCALE = 0.025778
    DI_EXP = 2.0
    T0 = 1.0e6
    GC0 = 10.0
    post_czm = {}
    for cond in CONDITIONS:
        stack = np.load(_HERE / "_di_credible" / cond / "di_stack.npy")
        if stack.ndim == 2 and stack.shape[0] != 20:
            stack = stack.T
        di_bot = np.array([float(stack[i].reshape(15, 15, 15)[:, :, :3].mean()) for i in range(20)])
        r = np.clip(di_bot / DI_SCALE, 0, 1)
        f = (1 - r) ** DI_EXP
        post_czm[cond] = {"di_mean": di_bot, "gc": GC0 * f, "rf_peak": T0 * f}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # ── 左: RF_peak bar + posterior dots ──────────────────────────
    ax = axes[0]
    x = np.arange(len(CONDITIONS))
    rf_p50 = [results_raw[c]["p50"]["rf_peak"] for c in CONDITIONS]
    rf_p05 = [results_raw[c]["p05"]["rf_peak"] for c in CONDITIONS]
    rf_p95 = [results_raw[c]["p95"]["rf_peak"] for c in CONDITIONS]
    colors = [COLORS[c] for c in CONDITIONS]

    bars = ax.bar(x, rf_p50, width=0.52, color=colors, alpha=0.85, edgecolor="k", lw=0.6, zorder=3)
    rf_lo = np.maximum(0, np.array(rf_p50) - np.array(rf_p95))
    rf_hi = np.maximum(0, np.array(rf_p05) - np.array(rf_p50))
    ax.errorbar(
        x, rf_p50, yerr=[rf_lo, rf_hi], fmt="none", color="black", capsize=5, lw=1.5, zorder=4
    )

    for xi, cond in enumerate(CONDITIONS):
        rf_post = post_czm[cond]["rf_peak"]
        jit = np.random.default_rng(42).uniform(-0.11, 0.11, len(rf_post))
        # clip to axis range for visibility
        ylim_rf = (2.5e5, 7.5e5)
        in_r = (rf_post >= ylim_rf[0]) & (rf_post <= ylim_rf[1])
        ax.scatter(xi + jit[in_r], rf_post[in_r], color=COLORS[cond], s=22, alpha=0.55, zorder=5)
        n_out = (~in_r).sum()
        if n_out:
            ax.text(
                xi,
                ylim_rf[1] * 0.97,
                f"▲{n_out}",
                ha="center",
                fontsize=7.5,
                color=COLORS[cond],
                fontweight="bold",
            )

    # 数値ラベル + Δ
    ref_rf = rf_p50[CONDITIONS.index("commensal_static")]
    for xi, (v, lo, hi) in enumerate(zip(rf_p50, rf_p05, rf_p95)):
        delta = (v - ref_rf) / ref_rf * 100
        ax.text(
            xi,
            max(hi, v) + 1.5e4,
            f"{v/1e5:.2f}×10⁵ N\n({delta:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS_FLAT[c] for c in CONDITIONS], fontsize=10)
    ax.set_ylabel("RF$_\\mathrm{peak}$ (N)", fontsize=11)
    ax.set_ylim(2.5e5, 7.5e5)
    ax.set_title(
        "B3: Peak Pull Force per Condition\n(bars=p50, errors=p05–p95 DI CI, dots=20 samples)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e5:.1f}×10⁵"))

    # ── 右: Gc vs DI scatter, ZOOMED ────────────────────────────
    ax2 = axes[1]
    DI_zoom = 0.03  # zoom range
    di_curve = np.linspace(0, DI_zoom, 200)
    r_curve = np.clip(di_curve / DI_SCALE, 0, 1)
    gc_curve = GC0 * (1 - r_curve) ** DI_EXP
    ax2.plot(di_curve, gc_curve, "k--", lw=1.5, alpha=0.7, label=r"$G_c=G_{c,0}(1-DI/s)^n$")

    total_out = 0
    for cond in CONDITIONS:
        di_post = post_czm[cond]["di_mean"]
        gc_post = post_czm[cond]["gc"]
        in_r = di_post <= DI_zoom
        n_out_local = (~in_r).sum()
        total_out += n_out_local
        ax2.scatter(di_post[in_r], gc_post[in_r], color=COLORS[cond], s=22, alpha=0.45, zorder=3)
        for qtag, mk, ms in [("p05", "v", 9), ("p50", "o", 12), ("p95", "^", 9)]:
            r_ = results_raw[cond][qtag]
            if r_["di_mean"] <= DI_zoom:
                ax2.scatter(
                    r_["di_mean"],
                    r_["gc_eff"],
                    color=COLORS[cond],
                    s=ms**2,
                    marker=mk,
                    edgecolors="k",
                    lw=0.7,
                    zorder=5,
                )
        r50 = results_raw[cond]["p50"]
        ax2.annotate(
            LABELS_FLAT[cond],
            (r50["di_mean"], r50["gc_eff"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8.5,
        )

    if total_out > 0:
        ax2.text(
            0.98,
            0.05,
            f"注: {total_out}サンプルが表示範囲外\n(DI > {DI_zoom:.3f})",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax2.set_xlim(-0.001, DI_zoom)
    ax2.set_xlabel("Mean DI (bottom interface layers)", fontsize=11)
    ax2.set_ylabel("$G_c$ (J/m²) — zoomed", fontsize=11)
    ax2.set_title("B3: Fracture Energy vs Interface DI\n(zoomed to main cluster)", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3, ls="--")

    # 反直感説明テキスト
    ax2.text(
        0.02,
        0.97,
        "dh-baseline: lowest DI → strongest interface\n"
        "(high a₃₅ → Pg in mid-layers, not at substrate)",
        transform=ax2.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3cd", alpha=0.9),
    )

    fig.suptitle(
        "B3: Cohesive Zone Model — Interface Strength vs Biofilm Condition\n"
        r"$t_\mathrm{max}(DI)=t_0(1-r)^n$,  $G_c(DI)=G_{c,0}(1-r)^n$  (analytical from B1 DI fields)",
        fontsize=12,
        fontweight="bold",
    )
    out = czm_base / "figures" / "fig_B3_czm_summary_improved.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print("=== A3: Y軸ズーム ===")
    fix_A3()
    print("=== DI: 双パネル ===")
    fix_DI()
    print("=== E: violin + summary 改善 ===")
    fix_E()
    print("=== B3: Gc zoom ===")
    fix_B3()
    print("\n全図生成完了")


if __name__ == "__main__":
    main()
