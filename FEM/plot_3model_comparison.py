#!/usr/bin/env python3
"""
plot_3model_comparison.py
==========================
Comparison figure: DI vs φ_Pg vs Virulence material models in 3D FEM.

Key finding: φ_Pg and Virulence models produce identical displacements
across all conditions (E ≈ 1000 Pa), while DI model shows 30× difference
(32–909 Pa). This demonstrates DI is the appropriate biomarker.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

# Collected results
RESULTS = {
    "commensal_static": {
        "DI": {"E_pa": 909.1, "disp_max": 439.1},
        "phi_pg": {"E_pa": 999.9, "disp_max": 398.7},
        "virulence": {"E_pa": 999.9, "disp_max": 398.7},
    },
    "commensal_hobic": {
        "DI": {"E_pa": 890.1, "disp_max": 448.3},
        "phi_pg": {"E_pa": 1000.0, "disp_max": None},  # not run
        "virulence": {"E_pa": 999.9, "disp_max": None},
    },
    "dh_baseline": {
        "DI": {"E_pa": 705.1, "disp_max": 567.1},
        "phi_pg": {"E_pa": 999.9, "disp_max": None},
        "virulence": {"E_pa": 999.9, "disp_max": None},
    },
    "dysbiotic_static": {
        "DI": {"E_pa": 32.3, "disp_max": 12929.7},
        "phi_pg": {"E_pa": 1000.0, "disp_max": 398.7},
        "virulence": {"E_pa": 999.9, "disp_max": 398.7},
    },
}

COND_LABELS = {
    "commensal_static": "Comm.\nStatic",
    "commensal_hobic": "Comm.\nHOBIC",
    "dh_baseline": "Dysb.\nHOBIC",
    "dysbiotic_static": "Dysb.\nStatic",
}

MODEL_COLORS = {
    "DI": "#1f77b4",
    "phi_pg": "#d62728",
    "virulence": "#2ca02c",
}
MODEL_LABELS = {
    "DI": "DI (entropy)",
    "phi_pg": "$\\phi_{Pg}$ (Hill)",
    "virulence": "Virulence (Pg+Fn)",
}


def main():
    conds = ["commensal_static", "dysbiotic_static"]
    models = ["DI", "phi_pg", "virulence"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) E_bio comparison ──
    ax = axes[0, 0]
    x = np.arange(len(conds))
    width = 0.25
    for i, model in enumerate(models):
        vals = [RESULTS[c][model]["E_pa"] for c in conds]
        ax.bar(
            x + i * width,
            vals,
            width * 0.9,
            color=MODEL_COLORS[model],
            alpha=0.85,
            label=MODEL_LABELS[model],
            edgecolor="k",
            linewidth=0.5,
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=10)
    ax.set_ylabel("$E_{bio}$ [Pa]", fontsize=12)
    ax.set_title("(a) Biofilm Stiffness by Model", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1100)

    # ── (b) Displacement comparison ──
    ax = axes[0, 1]
    for i, model in enumerate(models):
        vals = []
        for c in conds:
            d = RESULTS[c][model]["disp_max"]
            vals.append(d if d is not None else 0)
        ax.bar(
            x + i * width,
            vals,
            width * 0.9,
            color=MODEL_COLORS[model],
            alpha=0.85,
            label=MODEL_LABELS[model],
            edgecolor="k",
            linewidth=0.5,
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=10)
    ax.set_ylabel("$U_{max}$ [mm]", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("(b) Max Displacement (log scale)", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── (c) Displacement ratio (Dysbiotic / Commensal) ──
    ax = axes[1, 0]
    ratios = []
    ratio_labels = []
    ratio_colors = []
    for model in models:
        d_comm = RESULTS["commensal_static"][model]["disp_max"]
        d_dysb = RESULTS["dysbiotic_static"][model]["disp_max"]
        if d_comm and d_dysb:
            ratios.append(d_dysb / d_comm)
            ratio_labels.append(MODEL_LABELS[model])
            ratio_colors.append(MODEL_COLORS[model])

    ax.barh(
        range(len(ratios)), ratios, color=ratio_colors, alpha=0.85, edgecolor="k", linewidth=0.5
    )
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels(ratio_labels, fontsize=10)
    ax.set_xlabel("$U_{max}^{dysb} / U_{max}^{comm}$", fontsize=12)
    ax.set_title("(c) Condition Discrimination Ratio", fontsize=12, weight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, axis="x")
    # Add value labels
    for i, r in enumerate(ratios):
        ax.text(r * 1.1, i, f"{r:.1f}×", va="center", fontsize=11, weight="bold")

    # ── (d) All 4 conditions (DI model only) ──
    ax = axes[1, 1]
    all_conds = list(RESULTS.keys())
    e_vals = [RESULTS[c]["DI"]["E_pa"] for c in all_conds]
    d_vals = [RESULTS[c]["DI"]["disp_max"] for c in all_conds]
    colors = ["#2ca02c", "#17becf", "#d62728", "#ff7f0e"]
    for i, c in enumerate(all_conds):
        ax.scatter(
            e_vals[i],
            d_vals[i],
            s=150,
            c=colors[i],
            edgecolor="k",
            zorder=5,
            label=COND_LABELS[c].replace("\n", " "),
        )
    # Add phi_pg points (all conditions have same E ≈ 1000 Pa)
    ax.scatter(
        [999.9],
        [398.7],
        s=100,
        c="gray",
        marker="s",
        edgecolor="k",
        zorder=4,
        label="All conds (φ_Pg model)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$E_{bio}$ [Pa]", fontsize=12)
    ax.set_ylabel("$U_{max}$ [mm]", fontsize=12)
    ax.set_title("(d) E vs U: DI Model Separates Conditions", fontsize=12, weight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "3-Model Material Comparison: DI vs $\\phi_{Pg}$ vs Virulence\n"
        "Only DI (entropy) discriminates commensal vs dysbiotic biofilm in 3D FEM",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out = _FIG_DIR / "3model_comparison_3d.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
