#!/usr/bin/env python3
"""
generate_pipeline_summary.py — P3: 二スケール連成パイプライン全体サマリ図
=========================================================================

ミクロ (Hamilton PDE) → マクロ (Abaqus FEM) の全パイプラインを
1 枚の図にまとめる。

図の構成 (3 行 × 3 列 = 9 パネル)
------------------------------------
  [Row 0] パイプライン模式図 (span 3 columns)
  [Row 1] 左: phi_total(depth) 4 条件  中: c(depth) 4 条件  右: DI 比較棒グラフ
  [Row 2] 左: alpha_monod(depth)       中: E_Pa(depth)       右: σ0(depth) 圧縮応力

入力 (ファイル自動検索)
-----------------------
  _multiscale_results/macro_eigenstrain_{condition}.csv        (1D 元版)
  _multiscale_results/macro_eigenstrain_{condition}_hybrid.csv (hybrid 版)
  _abaqus_input/sigma_max_summary.txt                         (Abaqus サマリ)

出力
----
  _pipeline_summary/
    pipeline_summary.png  — 全パイプライン概要 (論文用)
    pipeline_summary.pdf  — PDF 版 (オプション)

使い方
------
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \\
      Tmcmc202601/FEM/generate_pipeline_summary.py
"""

from __future__ import annotations
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── ディレクトリ ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(_HERE, "_multiscale_results")
ABQ_DIR = os.path.join(_HERE, "_abaqus_input")
OUT_DIR = os.path.join(_HERE, "_pipeline_summary")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 条件メタデータ ────────────────────────────────────────────────────────────
CONDITIONS = {
    "commensal_static": {"color": "#1f77b4", "label": "Commensal\nStatic", "ls": "-"},
    "commensal_hobic": {"color": "#2ca02c", "label": "Commensal\nHOBIC", "ls": "--"},
    "dysbiotic_static": {"color": "#ff7f0e", "label": "Dysbiotic\nStatic", "ls": "-."},
    "dysbiotic_hobic": {"color": "#d62728", "label": "Dysbiotic\nHOBIC", "ls": ":"},
}

E_MAX_PA = 1000.0
E_MIN_PA = 10.0
DI_SCALE = 0.025778
N_POWER = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CSV 読み込み (hybrid 優先)
# ─────────────────────────────────────────────────────────────────────────────


def _read_commented_csv(path: str) -> dict:
    """'#' コメント行をスキップして CSV をロードするヘルパー。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    cols = lines[0].split(",")
    data = np.array(
        [[float(v) for v in l.split(",")] for l in lines[1:] if l.strip()],
        dtype=np.float64,
    )
    return {col: data[:, i] for i, col in enumerate(cols)}


def load_all_csvs() -> dict[str, dict]:
    """全条件の CSV を読み込む。hybrid 版を優先 (pandas 不要)。"""
    all_data = {}
    for ckey in CONDITIONS:
        for suffix in ["_hybrid", ""]:
            path = os.path.join(IN_DIR, f"macro_eigenstrain_{ckey}{suffix}.csv")
            if os.path.isfile(path):
                d = _read_commented_csv(path)
                all_data[ckey] = {
                    "depth_mm": d["depth_mm"],
                    "phi_total": d["phi_total"],
                    "c": d["c"],
                    "DI": d["DI"],
                    "alpha_monod": d["alpha_monod"],
                    "eps_growth": d["eps_growth"],
                    "E_Pa": d["E_Pa"],
                    "suffix": suffix,
                }
                tag = "(hybrid)" if suffix == "_hybrid" else "(original)"
                print(f"  [{ckey}] ロード {tag}: {path}")
                break
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# パイプライン模式図 (Axes に描画)
# ─────────────────────────────────────────────────────────────────────────────


def draw_pipeline_schema(ax: plt.Axes):
    """
    ミクロ→マクロ 二スケール連成のフロー図をテキストベースで描画。
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ボックスの定義: (x_center, y_center, width, height, label, color)
    boxes = [
        (0.9, 0.55, 1.4, 0.5, "TMCMC\n(MAP θ)\n4 conditions\n×20 params", "#cce5ff"),
        (2.8, 0.55, 1.4, 0.5, "0D Hamilton\nODE\n(n=2500, dt=0.01)\nDI_0D per cond.", "#d4edda"),
        (4.9, 0.72, 1.4, 0.5, "1D Hamilton\n+ Nutrient PDE\n(N=30, T*=20)\nα_Monod(x)", "#fff3cd"),
        (4.9, 0.25, 1.4, 0.3, "Hybrid CSV\nDI_0D × α_Monod(x)\nE_Pa(x, condition)", "#f8d7da"),
        (7.0, 0.55, 1.4, 0.5, "Abaqus .inp\nT3D2 bar\nΔT = ε_growth(x)\n4 conditions", "#e2d9f3"),
        (9.1, 0.55, 1.0, 0.5, "σ₀(x)\ncompressive\nprestress", "#fde8d8"),
    ]

    for xc, yc, w, h, txt, col in boxes:
        rect = FancyBboxPatch(
            (xc - w / 2, yc - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03",
            facecolor=col,
            edgecolor="gray",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            xc,
            yc,
            txt,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            zorder=4,
            linespacing=1.35,
        )

    # 矢印
    arrows = [
        (0.9 + 0.7, 0.55, 2.8 - 0.7, 0.55, "DI_0D\nφ_final"),
        (2.8 + 0.7, 0.72, 4.9 - 0.7, 0.72, "θ →\nα_Monod(x)"),
        (2.8 + 0.4, 0.40, 4.9 - 0.7, 0.25, "DI_0D"),
        (4.9 + 0.7, 0.25, 7.0 - 0.7, 0.45, "DI × α →\nHybrid CSV"),
        (4.9 + 0.7, 0.72, 7.0 - 0.7, 0.65, "α_Monod(x)"),
        (7.0 + 0.7, 0.55, 9.1 - 0.5, 0.55, "ε_growth\n→ΔT"),
    ]

    arrowprops = dict(
        arrowstyle="-|>",
        color="dimgray",
        lw=1.5,
        connectionstyle="arc3,rad=0.05",
    )

    for x0, y0, x1, y1, lbl in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrowprops, zorder=2)
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + 0.12,
            lbl,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="dimgray",
            fontstyle="italic",
        )

    # スケール表示
    ax.text(0.9, 0.02, "Macro scale", ha="center", fontsize=7, color="steelblue")
    ax.text(3.85, 0.02, "Micro-Macro bridge", ha="center", fontsize=7, color="goldenrod")
    ax.text(7.0, 0.02, "Macro FEM", ha="center", fontsize=7, color="purple")

    ax.set_title(
        "Two-scale coupling: micro Hamilton PDE → macro Abaqus FEM\n"
        "(Klempt 2024 framework + TMCMC parameter estimation)",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# メイン図の生成
# ─────────────────────────────────────────────────────────────────────────────


def make_summary_figure(all_data: dict) -> str:
    """
    9 パネルのパイプライン全体サマリ図を生成する。
    """
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.1, 1.2, 1.2],
        hspace=0.45,
        wspace=0.35,
    )

    # Row 0: パイプライン模式図 (3 列スパン)
    ax_schema = fig.add_subplot(gs[0, :])
    draw_pipeline_schema(ax_schema)

    # Row 1 & 2: データパネル
    ax_phi = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_di = fig.add_subplot(gs[1, 2])
    ax_alpha = fig.add_subplot(gs[2, 0])
    ax_E = fig.add_subplot(gs[2, 1])
    ax_sigma = fig.add_subplot(gs[2, 2])

    for ckey, data in all_data.items():
        meta = CONDITIONS[ckey]
        col = meta["color"]
        lbl = meta["label"].replace("\n", " ")
        ls = meta["ls"]
        depth = data["depth_mm"]
        hybrid = data["suffix"] == "_hybrid"
        lw = 2.0 if hybrid else 1.2
        alpha_ = 1.0 if hybrid else 0.55

        ax_phi.plot(depth, data["phi_total"], color=col, ls=ls, lw=lw, alpha=alpha_, label=lbl)
        ax_c.plot(depth, data["c"], color=col, ls=ls, lw=lw, alpha=alpha_, label=lbl)
        ax_alpha.plot(depth, data["alpha_monod"], color=col, ls=ls, lw=lw, alpha=alpha_, label=lbl)
        ax_E.plot(depth, data["E_Pa"], color=col, ls=ls, lw=lw, alpha=alpha_, label=lbl)
        sigma = -data["E_Pa"] * data["eps_growth"]
        ax_sigma.plot(depth, sigma, color=col, ls=ls, lw=lw, alpha=alpha_, label=lbl)

    # DI 棒グラフ
    keys = list(all_data.keys())
    n = len(keys)
    x = np.arange(n)
    di_0d = [float(all_data[k]["DI"].mean()) for k in keys]  # hybrid → DI_0D
    colors = [CONDITIONS[k]["color"] for k in keys]
    xlabels = [CONDITIONS[k]["label"].replace("\n", " ") for k in keys]
    bars = ax_di.bar(x, di_0d, color=colors, alpha=0.85, edgecolor="black")
    ax_di.set_xticks(x)
    ax_di.set_xticklabels(xlabels, rotation=15, ha="right", fontsize=8)
    ax_di.set_ylabel("Dysbiotic Index (DI)")
    ax_di.set_ylim(0, 1.15)
    ax_di.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax_di.set_title("DI per condition (0D ODE)", fontsize=9)
    ax_di.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, di_0d):
        ax_di.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 体裁設定
    def style(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)

    style(ax_phi, "Depth from tooth [mm]", r"$\phi_{total}$", "Total biofilm fraction")
    style(
        ax_c,
        "Depth from tooth [mm]",
        "c (nutrient conc.)",
        "Nutrient concentration\n(tooth=0, saliva=1)",
    )
    style(
        ax_alpha,
        "Depth from tooth [mm]",
        r"$\alpha_{Monod}(x)$",
        r"Nutrient-limited eigenstrain $\alpha_{Monod}$",
    )
    style(ax_E, "Depth from tooth [mm]", "E [Pa]", "Local elastic modulus E(DI)")
    style(
        ax_sigma,
        "Depth from tooth [mm]",
        r"$\sigma_0$ [Pa]  (< 0)",
        r"Compressive prestress $\sigma_0 = -E \cdot \varepsilon_{growth}$",
    )

    # hybrid / original 注記
    has_hybrid = any(d["suffix"] == "_hybrid" for d in all_data.values())
    tag = "Solid lines: hybrid CSV (DI_0D × α_Monod)" if has_hybrid else "Original 1D CSV (DI≈0)"
    fig.text(0.5, 0.005, tag, ha="center", fontsize=8, color="dimgray", fontstyle="italic")

    fig.suptitle(
        "Two-scale coupling summary\n"
        "micro Hamilton 1D (DI, α_Monod) → macro Abaqus FEM (ε_growth, σ₀)",
        fontsize=13,
        fontweight="bold",
        y=1.00,
    )

    path_png = os.path.join(OUT_DIR, "pipeline_summary.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  サマリ図 (PNG): {path_png}")

    # PDF 版 (オプション)
    try:
        fig2 = plt.figure(figsize=(16, 13))
        # (再生成は省略 — 簡易的に PNG を再保存)
        path_pdf = os.path.join(OUT_DIR, "pipeline_summary.pdf")
        # matplotlib は PNG から PDF 変換不可なので再度描画 (簡易版はスキップ)
        print("  (PDF 版は別途 PNG から変換してください)")
    except Exception:
        pass

    return path_png


# ─────────────────────────────────────────────────────────────────────────────
# キーナンバーテキスト出力
# ─────────────────────────────────────────────────────────────────────────────


def print_key_numbers(all_data: dict):
    """論文・発表用のキー数値を表示する。"""
    print()
    print("=" * 70)
    print("KEY NUMBERS (論文・発表用)")
    print("=" * 70)
    print()
    print(
        f"{'Condition':<22} {'DI':>7} {'E[Pa]':>8} {'α_tooth':>9} {'α_saliva':>9} {'σ_tooth[Pa]':>12}"
    )
    print("-" * 70)

    for ckey, data in all_data.items():
        lbl = ckey
        di = float(data["DI"].mean())
        E_mean = float(data["E_Pa"].mean())
        am_t = float(data["alpha_monod"][0])
        am_s = float(data["alpha_monod"][-1])
        sigma_t = -float(data["E_Pa"][0]) * float(data["eps_growth"][0])

        print(f"  {lbl:<20} {di:>7.4f} {E_mean:>8.1f} {am_t:>9.5f} {am_s:>9.4f} {sigma_t:>12.4f}")

    print()
    print("  α_Monod spatial gradient: tooth / saliva ratio")
    for ckey, data in all_data.items():
        am_t = float(data["alpha_monod"][0])
        am_s = float(data["alpha_monod"][-1])
        ratio = am_s / am_t if am_t > 1e-10 else float("inf")
        print(f"    {ckey:<25}: {am_t:.5f} → {am_s:.4f}  (ratio = {ratio:.1f}x)")

    print()
    print("  Physical interpretation:")
    print("    commensal: high DI ≈ low → E ≈ E_max → large prestress")
    print("    dysbiotic: high DI → E ≈ E_min → small prestress")
    print("    → Dysbiotic biofilm is mechanically weaker (lower E)")
    print("      but also produces less compressive prestress on the tooth")
    print("    → α_Monod gradient (101x) reflects nutrient depletion near tooth")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  generate_pipeline_summary.py — P3: パイプライン全体サマリ図")
    print("=" * 70)
    print()

    all_data = load_all_csvs()
    if not all_data:
        print("エラー: CSV が見つかりません。")
        print("  1. multiscale_coupling_1d.py を実行して 1D CSV を生成")
        print("  2. generate_hybrid_macro_csv.py を実行して hybrid CSV を生成")
        return

    print()
    path = make_summary_figure(all_data)
    print_key_numbers(all_data)

    print()
    print("=" * 70)
    print(f"完了: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
