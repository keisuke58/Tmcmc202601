#!/usr/bin/env python3
"""
plot_posterior_uncertainty.py
==============================
Publication-quality figures for posterior S_Mises uncertainty.

Figures produced (in _posterior_uncertainty/):
  Fig1_stress_violin.png         — Violin + box per condition (sub & surf)
  Fig2_stress_ci_bars.png        — CI bars: p05/p50/p95 across conditions
  Fig3_sensitivity_heatmap.png   — Spearman rho heatmap (20 params × 4 conds)
  Fig4_top_params_scatter.png    — Top-5 param scatter vs substrate stress
  Fig5_stress_summary_panel.png  — Combined summary panel (paper figure)
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_BASE = _HERE / "_posterior_abaqus"
_OUT = _HERE / "_posterior_uncertainty"
_OUT.mkdir(exist_ok=True)

# ── constants ───────────────────────────────────────────────────────────────
CONDITIONS = ["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"]
LABELS = {
    "dh_baseline": "dh-baseline",
    "commensal_static": "Comm.\nStatic",
    "commensal_hobic": "Comm.\nHOBIC",
    "dysbiotic_static": "Dysb.\nStatic",
}
COLORS = {
    "dh_baseline": "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
}
PARAM_NICE = {
    "a11": "a₁₁ (So·So)",
    "a12": "a₁₂ (So·An)",
    "a22": "a₂₂ (An·An)",
    "b1": "b₁ (So grow)",
    "b2": "b₂ (An grow)",
    "a33": "a₃₃ (Vd·Vd)",
    "a34": "a₃₄ (Vd·Fn)",
    "a44": "a₄₄ (Fn·Fn)",
    "b3": "b₃ (Vd grow)",
    "b4": "b₄ (Fn grow)",
    "a13": "a₁₃ (So·Vd)",
    "a14": "a₁₄ (So·Fn)",
    "a23": "a₂₃ (An·Vd)",
    "a24": "a₂₄ (An·Fn)",
    "a55": "a₅₅ (Pg·Pg)",
    "b5": "b₅ (Pg grow)",
    "a15": "a₁₅ (So·Pg)",
    "a25": "a₂₅ (An·Pg)",
    "a35": "a₃₅ (Vd→Pg)",
    "a45": "a₄₅ (Fn→Pg)",
}


# ── load data ───────────────────────────────────────────────────────────────
def load_all():
    data = {}  # cond -> (20,2) array  [substrate, surface]  in Pa
    spear = {}  # cond -> {param -> {substrate, surface}}
    thetas = {}  # cond -> (20,20) array

    for cond in CONDITIONS:
        d = _BASE / cond
        arr = np.load(d / "stress_all.npy")  # (20,2)
        data[cond] = arr

        with (d / "sensitivity_spearman.json").open() as f:
            s = json.load(f)
        param_names = s["param_names"]
        spear[cond] = {
            "names": param_names,
            "substrate": np.array([s["spearman_substrate"][p] for p in param_names]),
            "surface": np.array([s["spearman_surface"][p] for p in param_names]),
        }

        # load per-sample theta
        th_list = []
        for i in range(20):
            th_list.append(np.load(d / f"sample_{i:04d}" / "theta.npy"))
        thetas[cond] = np.vstack(th_list)  # (20,20)

    return data, spear, thetas


# ── Fig 1: Violin + box ──────────────────────────────────────────────────────
def fig1_violin(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for si, (col, slabel) in enumerate([(0, "Substrate"), (1, "Surface")]):
        ax = axes[si]
        vals_list = [data[c][:, col] / 1e6 for c in CONDITIONS]
        xlabs = [LABELS[c] for c in CONDITIONS]
        colors = [COLORS[c] for c in CONDITIONS]

        parts = ax.violinplot(
            vals_list,
            positions=range(len(CONDITIONS)),
            showmedians=False,
            showextrema=False,
            widths=0.6,
        )
        for body, col_ in zip(parts["bodies"], colors):
            body.set_facecolor(col_)
            body.set_alpha(0.45)

        # box
        bp = ax.boxplot(
            vals_list,
            positions=range(len(CONDITIONS)),
            widths=0.28,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker="o", ms=4, alpha=0.5),
        )
        for patch, col_ in zip(bp["boxes"], colors):
            patch.set_facecolor(col_)
            patch.set_alpha(0.75)

        # individual sample dots
        for xi, (vals, col_) in enumerate(zip(vals_list, colors)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(xi + jitter, vals, color=col_, s=18, zorder=5, alpha=0.8)

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(xlabs, fontsize=10)
        ax.set_ylabel("$S_\\mathrm{Mises}$ (MPa)", fontsize=11)
        ax.set_title(f"Posterior uncertainty — {slabel}", fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Posterior $S_\\mathrm{Mises}$ Distribution (20 TMCMC Samples × 4 Conditions)",
        fontsize=13,
        fontweight="bold",
    )
    out = _OUT / "Fig1_stress_violin.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── Fig 2: CI bars p05/p50/p95 ──────────────────────────────────────────────
def fig2_ci_bars(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    w = 0.55

    for si, (col, slabel) in enumerate([(0, "Substrate"), (1, "Surface")]):
        ax = axes[si]
        p50 = np.array([np.percentile(data[c][:, col], 50) / 1e6 for c in CONDITIONS])
        p05 = np.array([np.percentile(data[c][:, col], 5) / 1e6 for c in CONDITIONS])
        p95 = np.array([np.percentile(data[c][:, col], 95) / 1e6 for c in CONDITIONS])
        colors = [COLORS[c] for c in CONDITIONS]

        bars = ax.bar(
            x, p50, width=w, color=colors, alpha=0.8, edgecolor="k", linewidth=0.6, zorder=3
        )
        ax.errorbar(
            x,
            p50,
            yerr=[p50 - p05, p95 - p50],
            fmt="none",
            color="black",
            capsize=5,
            lw=1.5,
            zorder=4,
        )

        # annotate p50 values
        for xi, v in enumerate(p50):
            ax.text(
                xi,
                v + (p95[xi] - p50[xi]) + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=10)
        ax.set_ylabel("$S_\\mathrm{Mises}$ (MPa)", fontsize=11)
        ax.set_title(f"Posterior CI (p05–p95) — {slabel}", fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0)

    fig.suptitle(
        "Posterior $S_\\mathrm{Mises}$ Credible Intervals (5th–95th percentile)",
        fontsize=13,
        fontweight="bold",
    )
    out = _OUT / "Fig2_stress_ci_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── Fig 3: Spearman heatmap ──────────────────────────────────────────────────
def fig3_heatmap(spear):
    param_names = spear["dh_baseline"]["names"]
    nice_labels = [PARAM_NICE.get(p, p) for p in param_names]
    n_params = len(param_names)
    cond_labels = [LABELS[c].replace("\n", " ") for c in CONDITIONS]

    # matrix: rows=params, cols=conditions, values=substrate rho
    mat_sub = np.array([spear[c]["substrate"] for c in CONDITIONS]).T  # (20,4)
    mat_surf = np.array([spear[c]["surface"] for c in CONDITIONS]).T

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    for si, (mat, slabel) in enumerate([(mat_sub, "Substrate"), (mat_surf, "Surface")]):
        ax = axes[si]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(cond_labels, fontsize=10, rotation=15, ha="right")
        ax.set_yticks(range(n_params))
        ax.set_yticklabels(nice_labels, fontsize=8.5)
        ax.set_title(f"Spearman ρ — {slabel}", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.035, label="ρ")

        # annotate cells
        for i in range(n_params):
            for j in range(len(CONDITIONS)):
                v = mat[i, j]
                txt_col = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=txt_col)

    fig.suptitle(
        "Spearman Rank Correlation: TMCMC Parameters vs $S_\\mathrm{Mises}$",
        fontsize=13,
        fontweight="bold",
    )
    out = _OUT / "Fig3_sensitivity_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── Fig 4: Top-param scatter ─────────────────────────────────────────────────
def fig4_scatter(data, spear, thetas):
    # find top 5 params by mean |rho| across all conditions (substrate)
    param_names = spear["dh_baseline"]["names"]
    mean_abs_rho = np.mean(np.abs(np.array([spear[c]["substrate"] for c in CONDITIONS])), axis=0)
    top_idx = np.argsort(mean_abs_rho)[::-1][:5]
    top_params = [param_names[i] for i in top_idx]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for row, cond in enumerate(["dh_baseline", "commensal_static"]):
        stress = data[cond][:, 0] / 1e6  # substrate MPa
        theta = thetas[cond]  # (20,20)
        for col, param in enumerate(top_params):
            ax = axes[row, col]
            pidx = param_names.index(param)
            x = theta[:, pidx]
            ax.scatter(x, stress, color=COLORS[cond], s=40, alpha=0.8, zorder=3)
            # trend line
            if np.std(x) > 1e-12:
                z = np.polyfit(x, stress, 1)
                xr = np.linspace(x.min(), x.max(), 50)
                ax.plot(xr, np.polyval(z, xr), color="black", lw=1.2, ls="--")
            rho = spear[cond]["substrate"][pidx]
            ax.set_title(f"{PARAM_NICE.get(param, param)}\n(ρ={rho:+.2f})", fontsize=8)
            ax.set_xlabel("θ value", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{LABELS[cond].replace(chr(10),' ')}\n$S_M$ (MPa)", fontsize=8)
            ax.grid(alpha=0.3, ls="--")
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Top-5 Sensitivity Parameters: θ vs Substrate $S_\\mathrm{Mises}$",
        fontsize=12,
        fontweight="bold",
    )
    out = _OUT / "Fig4_top_params_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── Fig 5: Summary panel ─────────────────────────────────────────────────────
def fig5_summary(data, spear):
    param_names = spear["dh_baseline"]["names"]
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_box = fig.add_subplot(gs[0, :2])  # violin/box — full width top-left
    ax_sens = fig.add_subplot(gs[0, 2])  # top sensitivity bar — top-right
    ax_ci_s = fig.add_subplot(gs[1, 0])  # CI bars substrate
    ax_ci_su = fig.add_subplot(gs[1, 1])  # CI bars surface
    ax_delta = fig.add_subplot(gs[1, 2])  # condition delta vs commensal

    # ── boxplot ──
    vals_sub = [data[c][:, 0] / 1e6 for c in CONDITIONS]
    vals_surf = [data[c][:, 1] / 1e6 for c in CONDITIONS]
    pos_s = np.arange(len(CONDITIONS)) * 2
    pos_su = pos_s + 0.75
    colors = [COLORS[c] for c in CONDITIONS]

    for pos, vals_list, marker in [(pos_s, vals_sub, "s"), (pos_su, vals_surf, "^")]:
        bp = ax_box.boxplot(
            vals_list,
            positions=pos,
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1),
            capprops=dict(lw=1),
            flierprops=dict(marker=marker, ms=4, alpha=0.4),
        )
        for patch, col_ in zip(bp["boxes"], colors):
            patch.set_facecolor(col_)
            patch.set_alpha(0.75)

    ax_box.set_xticks((pos_s + pos_su) / 2)
    ax_box.set_xticklabels([LABELS[c].replace("\n", " ") for c in CONDITIONS], fontsize=9)
    ax_box.set_ylabel("$S_\\mathrm{Mises}$ (MPa)", fontsize=10)
    ax_box.set_title("Posterior Distribution (■=substrate  ▲=surface)", fontsize=10)
    ax_box.grid(axis="y", alpha=0.3, ls="--")
    ax_box.legend(
        handles=[
            mpatches.Patch(color=COLORS[c], label=LABELS[c].replace("\n", " "), alpha=0.75)
            for c in CONDITIONS
        ],
        fontsize=8,
        loc="upper right",
    )

    # ── top-5 sensitivity bars (mean |rho| substrate) ──
    mean_abs_rho = np.mean(np.abs(np.array([spear[c]["substrate"] for c in CONDITIONS])), axis=0)
    top_idx = np.argsort(mean_abs_rho)[::-1][:8]
    top_rhos = mean_abs_rho[top_idx]
    top_pnames = [PARAM_NICE.get(param_names[i], param_names[i]) for i in top_idx]
    ypos = np.arange(len(top_idx))
    ax_sens.barh(ypos, top_rhos, color="steelblue", alpha=0.8, edgecolor="k", lw=0.5)
    ax_sens.set_yticks(ypos)
    ax_sens.set_yticklabels(top_pnames, fontsize=8)
    ax_sens.set_xlabel("Mean |Spearman ρ| (substrate)", fontsize=9)
    ax_sens.set_title("Parameter Sensitivity", fontsize=10)
    ax_sens.grid(axis="x", alpha=0.3, ls="--")
    ax_sens.set_xlim(0, 0.65)

    # ── CI bars substrate / surface ──
    x = np.arange(len(CONDITIONS))
    for ax, col, slabel in [(ax_ci_s, 0, "Substrate"), (ax_ci_su, 1, "Surface")]:
        p50 = np.array([np.percentile(data[c][:, col], 50) / 1e6 for c in CONDITIONS])
        p05 = np.array([np.percentile(data[c][:, col], 5) / 1e6 for c in CONDITIONS])
        p95 = np.array([np.percentile(data[c][:, col], 95) / 1e6 for c in CONDITIONS])
        ax.bar(x, p50, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
        ax.errorbar(
            x, p50, yerr=[p50 - p05, p95 - p50], fmt="none", color="black", capsize=4, lw=1.3
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[c].replace("\n", " ") for c in CONDITIONS], fontsize=8, rotation=12, ha="right"
        )
        ax.set_ylabel("$S_M$ (MPa)", fontsize=9)
        ax.set_title(f"CI (p5–p95) — {slabel}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_ylim(0)

    # ── delta vs commensal_static (reference) ──
    ref_sub = np.median(data["commensal_static"][:, 0])
    ref_surf = np.median(data["commensal_static"][:, 1])
    delta_sub = [(np.median(data[c][:, 0]) - ref_sub) / ref_sub * 100 for c in CONDITIONS]
    delta_surf = [(np.median(data[c][:, 1]) - ref_surf) / ref_surf * 100 for c in CONDITIONS]
    xd = np.arange(len(CONDITIONS))
    wd = 0.35
    ax_delta.bar(
        xd - wd / 2,
        delta_sub,
        wd,
        label="Substrate",
        color=[COLORS[c] for c in CONDITIONS],
        alpha=0.8,
        edgecolor="k",
        lw=0.5,
    )
    ax_delta.bar(
        xd + wd / 2,
        delta_surf,
        wd,
        label="Surface",
        color=[COLORS[c] for c in CONDITIONS],
        alpha=0.4,
        edgecolor="k",
        lw=0.5,
        hatch="//",
    )
    ax_delta.axhline(0, color="black", lw=0.8, ls="--")
    ax_delta.set_xticks(xd)
    ax_delta.set_xticklabels(
        [LABELS[c].replace("\n", " ") for c in CONDITIONS], fontsize=8, rotation=12, ha="right"
    )
    ax_delta.set_ylabel("Δ vs Comm. Static (%)", fontsize=9)
    ax_delta.set_title("Relative Change vs Reference", fontsize=10)
    ax_delta.legend(fontsize=8)
    ax_delta.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle(
        "Posterior $S_\\mathrm{Mises}$ Uncertainty — 20 TMCMC Samples × 4 Conditions",
        fontsize=13,
        fontweight="bold",
    )
    out = _OUT / "Fig5_stress_summary_panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading posterior stress data...")
    data, spear, thetas = load_all()

    print("\nSummary (substrate, MPa):")
    for cond in CONDITIONS:
        arr = data[cond][:, 0] / 1e6
        print(
            f"  {cond:20s}  p05={np.percentile(arr, 5):.3f}"
            f"  p50={np.percentile(arr,50):.3f}"
            f"  p95={np.percentile(arr,95):.3f}"
        )

    print("\nGenerating figures...")
    np.random.seed(42)
    fig1_violin(data)
    fig2_ci_bars(data)
    fig3_heatmap(spear)
    fig4_scatter(data, spear, thetas)
    fig5_summary(data, spear)

    print(f"\nAll figures → {_OUT}/")


if __name__ == "__main__":
    main()
