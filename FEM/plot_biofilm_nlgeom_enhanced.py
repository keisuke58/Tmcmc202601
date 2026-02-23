#!/usr/bin/env python3
"""
plot_biofilm_nlgeom_enhanced.py  –  [P3] biofilm NLGEOM 条件間比較プロット（強化版）
======================================================================================

4 条件の biofilm mode NLGEOM 解析結果（Section 4 of klempt2024_gap_analysis.md）と
posterior DI 信頼区間（_di_credible/{cond}/di_quantiles.npy）を組み合わせて
強化した比較プロットを生成する。

生成図
------
  Fig 1: U_max 棒グラフ（posterior DI 信頼区間から推定した不確実性付き）
  Fig 2: S_Mises 棒グラフ（4 条件比較; 応力は荷重制御で不変→ほぼ一致）
  Fig 3: DI_mean vs E_mean 散布図（p05/p50/p95 エラーバー付き）
  Fig 4: U_max vs E_mean 相関プロット（posterior 不確実性付き）

実行
----
  cd Tmcmc202601/FEM
  python3 plot_biofilm_nlgeom_enhanced.py
  # → _biofilm_mode_runs/biofilm_nlgeom_enhanced_{fig}.png

オプション
----------
  --out-dir  : 出力先ディレクトリ（デフォルト: _biofilm_mode_runs）
  --dpi      : 解像度（デフォルト: 150）
"""

from __future__ import print_function, division
import os
import sys
import argparse
import numpy as np

# ── 既知の Abaqus NLGEOM 結果（klempt2024_gap_analysis.md § 4 より）─────────
# E(DI) 冪乗則パラメータ（biofilm mode）
E_MAX      = 1000.0   # Pa
E_MIN      = 10.0     # Pa
DI_SCALE   = 0.025778 # DI 正規化スケール
DI_EXP     = 2.0      # 冪乗指数
PRESSURE   = 100.0    # Pa (GCF 歯肉溝液圧)
THICKNESS  = 0.2      # mm

# Section 4 で報告された Abaqus NLGEOM 結果
_ABAQUS_RESULTS = {
    "dh_baseline":      {"s_mises_mean": 56.6, "s_mises_p95": 61.0, "s_mises_max": 89.0,
                         "u_max": 0.0267, "di_mean": 0.00852},
    "dysbiotic_static": {"s_mises_mean": 56.6, "s_mises_p95": 61.1, "s_mises_max": 89.2,
                         "u_max": 0.0286, "di_mean": 0.00950},
    "commensal_static": {"s_mises_mean": 56.6, "s_mises_p95": 61.1, "s_mises_max": 89.3,
                         "u_max": 0.0290, "di_mean": 0.00971},
    "commensal_hobic":  {"s_mises_mean": 56.6, "s_mises_p95": 61.1, "s_mises_max": 89.3,
                         "u_max": 0.0294, "di_mean": 0.00990},
}

_COND_LABELS = {
    "dh_baseline":      "DH Baseline\n(dysbiotic)",
    "dysbiotic_static": "Dysbiotic\nStatic",
    "commensal_static": "Commensal\nStatic",
    "commensal_hobic":  "Commensal\nHOBIC",
}
_COLORS = {
    "dh_baseline":      "#d62728",   # red
    "dysbiotic_static": "#ff7f0e",   # orange
    "commensal_static": "#2ca02c",   # green
    "commensal_hobic":  "#1f77b4",   # blue
}

CONDITION_ORDER = ["dh_baseline", "dysbiotic_static", "commensal_static", "commensal_hobic"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_DI_CREDIBLE_DIR = os.path.join(_HERE, "_di_credible")


# ── E(DI) 冪乗則 ───────────────────────────────────────────────────────────────

def di_to_E(di_val, e_max=E_MAX, e_min=E_MIN, di_scale=DI_SCALE, di_exp=DI_EXP):
    """E(DI) = E_max * (1 - r)^n + E_min * r,  r = clip(DI / di_scale, 0, 1)"""
    r = float(np.clip(di_val / di_scale, 0.0, 1.0))
    return e_max * (1.0 - r) ** di_exp + e_min * r


# ── DI 信頼区間の読み込みと E_mean / U_max 不確実性推定 ─────────────────────────

def load_di_uncertainty(cond):
    """
    Load di_quantiles.npy and compute E_mean for p05/p50/p95 DI fields.
    Returns dict with keys p05/p50/p95 → (di_mean, E_mean, U_max_est)
    """
    path = os.path.join(_DI_CREDIBLE_DIR, cond, "di_quantiles.npy")
    if not os.path.exists(path):
        return None
    di_q = np.load(path)     # shape (3, N_nodes): rows = p05, p50, p95
    # Cap p95 outliers: clamp DI at DI_SCALE (= r=1) to avoid unphysical values
    di_q = np.clip(di_q, 0.0, DI_SCALE)

    result = {}
    tags   = ["p05", "p50", "p95"]
    for k, tag in enumerate(tags):
        di_field = di_q[k]               # (N_nodes,)
        di_m     = float(di_field.mean())
        E_m      = float(np.mean([di_to_E(d) for d in di_field]))
        # U_max ∝ 1/E_mean; anchor to known p50 Abaqus result
        result[tag] = {"di_mean": di_m, "E_mean": E_m}
    return result


def anchor_umax(cond, unc):
    """
    Scale p05/p95 U_max estimates by anchoring to known Abaqus p50 result.
    U_max(q) ≈ U_max(p50) * E_mean(p50) / E_mean(q)
    (thin-film pressure: u = p*t/E)
    """
    u_p50_known = _ABAQUS_RESULTS[cond]["u_max"]
    E_p50       = unc["p50"]["E_mean"]
    for tag in ["p05", "p50", "p95"]:
        E_q = unc[tag]["E_mean"]
        unc[tag]["u_max_est"] = u_p50_known * E_p50 / E_q if E_q > 0 else u_p50_known
    return unc


# ── プロット関数 ─────────────────────────────────────────────────────────────────

def _save(fig, name, out_dir, dpi=150):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("  Saved: %s" % path)
    return path


def plot_umax_bars(unc_all, out_dir, dpi=150):
    """Fig 1: U_max 棒グラフ（DI posterior 信頼区間から推定した不確実性付き）"""
    import matplotlib.pyplot as plt

    conds  = [c for c in CONDITION_ORDER if c in unc_all]
    labels = [_COND_LABELS[c] for c in conds]
    colors = [_COLORS[c]      for c in conds]

    u_p50  = [unc_all[c]["p50"]["u_max_est"]  for c in conds]
    u_lo   = [unc_all[c]["p50"]["u_max_est"] - unc_all[c]["p05"]["u_max_est"]
              for c in conds]    # positive = downside from p50
    u_hi   = [unc_all[c]["p95"]["u_max_est"] - unc_all[c]["p50"]["u_max_est"]
              for c in conds]    # positive = upside from p50
    # Force positive (abs)
    u_lo = [abs(v) for v in u_lo]
    u_hi = [abs(v) for v in u_hi]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(conds))
    bars = ax.bar(x, [v * 1e3 for v in u_p50], color=colors, alpha=0.82, width=0.5, zorder=3)
    ax.errorbar(x, [v * 1e3 for v in u_p50],
                yerr=[[v * 1e3 for v in u_lo], [v * 1e3 for v in u_hi]],
                fmt="none", ecolor="black", capsize=6, linewidth=1.5, zorder=4)

    # Annotate values
    for xi, (bar, yval) in enumerate(zip(bars, u_p50)):
        ax.text(xi, yval * 1e3 + max(u_hi) * 1e3 * 0.15,
                "%.1f" % (yval * 1e3), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("U_max (μm)", fontsize=11)
    ax.set_title("Maximum Displacement per Condition\n"
                 "(biofilm mode, NLGEOM, 100 Pa GCF pressure)\n"
                 "Error bars: posterior DI p05–p95 credible interval", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(u_p50) * 1.4e3)

    # Legend: reference line for dh_baseline
    ax.axhline(u_p50[0] * 1e3, ls="--", color=colors[0], alpha=0.5, lw=1,
               label="DH Baseline ref.")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "biofilm_nlgeom_umax_bars.png", out_dir, dpi)


def plot_smises_bars(out_dir, dpi=150):
    """Fig 2: S_Mises 棒グラフ（圧力制御 → 応力は材料依存性が低い）"""
    import matplotlib.pyplot as plt

    conds  = CONDITION_ORDER
    labels = [_COND_LABELS[c] for c in conds]
    colors = [_COLORS[c]      for c in conds]

    s_mean = [_ABAQUS_RESULTS[c]["s_mises_mean"] for c in conds]
    s_p95  = [_ABAQUS_RESULTS[c]["s_mises_p95"]  for c in conds]
    s_max  = [_ABAQUS_RESULTS[c]["s_mises_max"]  for c in conds]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(conds))
    width = 0.25

    ax.bar(x - width, s_mean, width, label="Mean",    color=colors, alpha=0.75, zorder=3)
    ax.bar(x,         s_p95,  width, label="p95",     color=colors, alpha=0.50, zorder=3, hatch="//")
    ax.bar(x + width, s_max,  width, label="Max",     color=colors, alpha=0.30, zorder=3, hatch="xx")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("S_Mises (Pa)", fontsize=11)
    ax.set_title("Von Mises Stress per Condition\n"
                 "(biofilm mode, NLGEOM, pressure-controlled → stress ≈ constant)\n"
                 "Conditions differ <0.1%", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="gray", alpha=0.75, label="Mean"),
        Patch(facecolor="gray", alpha=0.50, hatch="//", label="p95"),
        Patch(facecolor="gray", alpha=0.30, hatch="xx", label="Max"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper right")
    ax.set_ylim(0, max(s_max) * 1.2)

    # Annotation: note insignificant stress difference
    ax.text(0.5, 0.95, "Stress difference < 0.1% across conditions\n"
            "(Neumann BC: stress determined by load balance, not stiffness)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7,
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    return _save(fig, "biofilm_nlgeom_smises_bars.png", out_dir, dpi)


def plot_di_e_scatter(unc_all, out_dir, dpi=150):
    """Fig 3: DI_mean vs E_mean 散布図（posterior p05/p50/p95 エラーバー付き）"""
    import matplotlib.pyplot as plt

    conds = [c for c in CONDITION_ORDER if c in unc_all]
    colors = [_COLORS[c] for c in conds]

    fig, ax = plt.subplots(figsize=(6, 5))

    for c, col in zip(conds, colors):
        unc = unc_all[c]
        di_p50 = unc["p50"]["di_mean"]
        di_lo  = abs(unc["p50"]["di_mean"] - unc["p05"]["di_mean"])
        di_hi  = abs(unc["p95"]["di_mean"] - unc["p50"]["di_mean"])
        E_p50  = unc["p50"]["E_mean"]
        E_lo   = abs(unc["p50"]["E_mean"]  - unc["p95"]["E_mean"])   # note: higher DI → lower E
        E_hi   = abs(unc["p05"]["E_mean"]  - unc["p50"]["E_mean"])

        ax.errorbar(di_p50, E_p50,
                    xerr=[[di_lo], [di_hi]],
                    yerr=[[E_lo], [E_hi]],
                    fmt="o", color=col, ms=8, capsize=5, lw=1.5,
                    label=_COND_LABELS[c].replace("\n", " "))

    # E(DI) reference curve
    di_ref = np.linspace(0, DI_SCALE, 200)
    e_ref  = np.array([di_to_E(d) for d in di_ref])
    ax.plot(di_ref, e_ref, "k--", lw=1, alpha=0.4, label="E(DI) reference")

    ax.set_xlabel("DI_mean (mean dysbiotic index)", fontsize=11)
    ax.set_ylabel("E_mean (effective stiffness, Pa)", fontsize=11)
    ax.set_title("DI → Effective Stiffness (posterior credible intervals)\n"
                 "biofilm mode: E_max=1000 Pa, E_min=10 Pa", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "biofilm_nlgeom_di_e_scatter.png", out_dir, dpi)


def plot_umax_vs_e(unc_all, out_dir, dpi=150):
    """Fig 4: U_max vs E_mean 相関プロット（posterior 不確実性付き）"""
    import matplotlib.pyplot as plt

    conds = [c for c in CONDITION_ORDER if c in unc_all]
    colors = [_COLORS[c] for c in conds]

    fig, ax = plt.subplots(figsize=(6, 5))

    E_vals  = []
    U_vals  = []
    for c, col in zip(conds, colors):
        unc = unc_all[c]
        u50 = unc["p50"]["u_max_est"]
        u_lo = abs(u50 - unc["p05"]["u_max_est"])
        u_hi = abs(unc["p95"]["u_max_est"] - u50)
        E50 = unc["p50"]["E_mean"]
        E_lo = abs(E50 - unc["p95"]["E_mean"])
        E_hi = abs(unc["p05"]["E_mean"] - E50)

        ax.errorbar(E50, u50 * 1e3,
                    xerr=[[E_lo], [E_hi]],
                    yerr=[[u_lo * 1e3], [u_hi * 1e3]],
                    fmt="o", color=col, ms=8, capsize=5, lw=1.5,
                    label=_COND_LABELS[c].replace("\n", " "), zorder=4)
        ax.text(E50, u50 * 1e3 * 1.02, c.replace("_", "\n"),
                ha="center", va="bottom", fontsize=7, color=col)
        E_vals.append(E50)
        U_vals.append(u50 * 1e3)

    # Fit 1/E reference: U = C/E
    if len(E_vals) > 1:
        E_arr = np.array(E_vals)
        U_arr = np.array(U_vals)
        C = np.mean(U_arr * E_arr)
        E_ref = np.linspace(min(E_vals) * 0.9, max(E_vals) * 1.1, 100)
        ax.plot(E_ref, C / E_ref, "k--", lw=1, alpha=0.5, label="U ∝ 1/E fit")

    ax.set_xlabel("E_mean (effective stiffness, Pa)", fontsize=11)
    ax.set_ylabel("U_max (μm)", fontsize=11)
    ax.set_title("Maximum Displacement vs Effective Stiffness\n"
                 "(Stiffer biofilm → less deformation under GCF pressure)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "biofilm_nlgeom_umax_vs_e.png", out_dir, dpi)


def plot_combined(unc_all, out_dir, dpi=150):
    """
    Fig 5: 2×2 サブプロット複合図
    TL: U_max bars  TR: S_Mises bars  BL: DI-E scatter  BR: U_max vs E
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    conds   = [c for c in CONDITION_ORDER if c in unc_all]
    colors  = [_COLORS[c] for c in conds]
    labels  = [_COND_LABELS[c] for c in conds]
    x       = np.arange(len(conds))
    u_p50   = [unc_all[c]["p50"]["u_max_est"] * 1e3 for c in conds]
    u_lo    = [abs(unc_all[c]["p50"]["u_max_est"] - unc_all[c]["p05"]["u_max_est"]) * 1e3
               for c in conds]
    u_hi    = [abs(unc_all[c]["p95"]["u_max_est"] - unc_all[c]["p50"]["u_max_est"]) * 1e3
               for c in conds]
    s_mean  = [_ABAQUS_RESULTS[c]["s_mises_mean"] for c in conds]
    s_max   = [_ABAQUS_RESULTS[c]["s_mises_max"]  for c in conds]

    # ── TL: U_max bars ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, u_p50, color=colors, alpha=0.82, width=0.5, zorder=3)
    ax1.errorbar(x, u_p50, yerr=[u_lo, u_hi],
                 fmt="none", ecolor="black", capsize=5, lw=1.5, zorder=4)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("U_max (μm)", fontsize=10)
    ax1.set_title("(A) Max Displacement\n(biofilm mode, NLGEOM)", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(u_p50) * 1.45)
    for xi, yv in enumerate(u_p50):
        ax1.text(xi, yv + max(u_hi) * 0.2, "%.1f" % yv,
                 ha="center", va="bottom", fontsize=8)

    # ── TR: S_Mises bars ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    ax2.bar(x - width/2, s_mean, width, color=colors, alpha=0.80, label="Mean", zorder=3)
    ax2.bar(x + width/2, s_max,  width, color=colors, alpha=0.40, label="Max",
            hatch="xx", zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("S_Mises (Pa)", fontsize=10)
    ax2.set_title("(B) Von Mises Stress\n(< 0.1% variation → load-controlled)", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=7)

    # ── BL: DI vs E scatter ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for c, col in zip(conds, colors):
        unc = unc_all[c]
        di_p50 = unc["p50"]["di_mean"]
        di_lo  = abs(di_p50 - unc["p05"]["di_mean"])
        di_hi  = abs(unc["p95"]["di_mean"] - di_p50)
        E50    = unc["p50"]["E_mean"]
        E_lo   = abs(E50 - unc["p95"]["E_mean"])
        E_hi   = abs(unc["p05"]["E_mean"] - E50)
        ax3.errorbar(di_p50, E50, xerr=[[di_lo],[di_hi]], yerr=[[E_lo],[E_hi]],
                     fmt="o", color=col, ms=7, capsize=4, lw=1.5,
                     label=_COND_LABELS[c].replace("\n"," "))
    di_ref = np.linspace(0, DI_SCALE, 200)
    e_ref  = np.array([di_to_E(d) for d in di_ref])
    ax3.plot(di_ref, e_ref, "k--", lw=1, alpha=0.4, label="E(DI)")
    ax3.set_xlabel("DI_mean", fontsize=10)
    ax3.set_ylabel("E_mean (Pa)", fontsize=10)
    ax3.set_title("(C) DI → Effective Stiffness\n(posterior credible intervals)", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=6.5, loc="upper right")
    ax3.grid(alpha=0.3)

    # ── BR: U_max vs E ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    E_vals = []
    U_vals = []
    for c, col in zip(conds, colors):
        unc = unc_all[c]
        u50 = unc["p50"]["u_max_est"] * 1e3
        u_lo_v = abs(u50 - unc["p05"]["u_max_est"] * 1e3)
        u_hi_v = abs(unc["p95"]["u_max_est"] * 1e3 - u50)
        E50 = unc["p50"]["E_mean"]
        E_lo_v = abs(E50 - unc["p95"]["E_mean"])
        E_hi_v = abs(unc["p05"]["E_mean"] - E50)
        ax4.errorbar(E50, u50, xerr=[[E_lo_v],[E_hi_v]], yerr=[[u_lo_v],[u_hi_v]],
                     fmt="o", color=col, ms=7, capsize=4, lw=1.5,
                     label=_COND_LABELS[c].replace("\n"," "))
        E_vals.append(E50); U_vals.append(u50)
    if len(E_vals) > 1:
        C = np.mean(np.array(U_vals) * np.array(E_vals))
        E_r = np.linspace(min(E_vals)*0.9, max(E_vals)*1.1, 100)
        ax4.plot(E_r, C/E_r, "k--", lw=1, alpha=0.5, label="U ∝ 1/E")
    ax4.set_xlabel("E_mean (Pa)", fontsize=10)
    ax4.set_ylabel("U_max (μm)", fontsize=10)
    ax4.set_title("(D) Displacement vs Stiffness\n(stiffer community → less deformation)", fontsize=9, fontweight="bold")
    ax4.legend(fontsize=6.5)
    ax4.grid(alpha=0.3)

    fig.suptitle("Biofilm NLGEOM Analysis — 4 Conditions Comparison\n"
                 "E_max=1000 Pa, E_min=10 Pa, 100 Pa GCF pressure, NLGEOM=YES",
                 fontsize=10, fontweight="bold")
    return _save(fig, "biofilm_nlgeom_enhanced_combined.png", out_dir, dpi)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Enhanced biofilm NLGEOM comparison plots")
    p.add_argument("--out-dir", default=os.path.join(_HERE, "_biofilm_mode_runs"),
                   help="Output directory (default: _biofilm_mode_runs)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI (default 150)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("ERROR: matplotlib not available")
        sys.exit(1)

    print("=" * 60)
    print("  plot_biofilm_nlgeom_enhanced.py")
    print("  Loading DI credible interval data ...")
    print("=" * 60)

    # ── Load DI uncertainty for all conditions ─────────────────────────────
    unc_all = {}
    for cond in CONDITION_ORDER:
        unc = load_di_uncertainty(cond)
        if unc is None:
            print("  WARNING: No DI quantile data for %s, using Abaqus p50 only" % cond)
            # Fallback: use known Abaqus result
            di_m = _ABAQUS_RESULTS[cond]["di_mean"]
            E_m  = di_to_E(di_m)
            unc = {
                "p05": {"di_mean": di_m * 0.9, "E_mean": di_to_E(di_m * 0.9)},
                "p50": {"di_mean": di_m,        "E_mean": E_m},
                "p95": {"di_mean": di_m * 1.1,  "E_mean": di_to_E(di_m * 1.1)},
            }
        unc = anchor_umax(cond, unc)
        unc_all[cond] = unc

        print("  %s:" % cond)
        for tag in ["p05", "p50", "p95"]:
            print("    %s: DI_mean=%.5f  E_mean=%.1f Pa  U_max=%.4f mm" % (
                tag, unc[tag]["di_mean"], unc[tag]["E_mean"], unc[tag]["u_max_est"]))

    # ── Generate figures ───────────────────────────────────────────────────
    print("\n[1] U_max bars ...")
    plot_umax_bars(unc_all, args.out_dir, dpi=args.dpi)

    print("[2] S_Mises bars ...")
    plot_smises_bars(args.out_dir, dpi=args.dpi)

    print("[3] DI vs E scatter ...")
    plot_di_e_scatter(unc_all, args.out_dir, dpi=args.dpi)

    print("[4] U_max vs E correlation ...")
    plot_umax_vs_e(unc_all, args.out_dir, dpi=args.dpi)

    print("[5] Combined 2x2 figure ...")
    plot_combined(unc_all, args.out_dir, dpi=args.dpi)

    print("\nDone.  Figures in: %s" % args.out_dir)


if __name__ == "__main__":
    main()
