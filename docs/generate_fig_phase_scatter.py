#!/usr/bin/env python3
"""Phase 1 vs Phase 2 MAP scatter — validates two-phase strategy."""
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

RUNS = Path.home() / "IKM_Hiwi/Tmcmc202601/data_5species/main/_runs"
FIG_DIR = Path.home() / "IKM_Hiwi/Tmcmc202601/docs/figures"
FIG_DIR.mkdir(exist_ok=True)

# Phase 1 (fix-psi, mc=False) best runs
P1 = {
    "CS": "jax_ode_nuts_Commensal_Static_20260320_015113",
    "CH": "jax_ode_nuts_Commensal_HOBIC_20260320_015554",
    "DS": "jax_ode_nuts_Dysbiotic_Static_20260320_015831",
    "DH": "jax_ode_nuts_Dysbiotic_HOBIC_20260320_021506",
}
# Phase 2 (free-psi, mc=True, 10000p)
P2 = {
    "CS": "jax_ode_nuts_Commensal_Static_20260320_043505",
    "CH": "jax_ode_nuts_Commensal_HOBIC_20260320_043812",
    "DS": "jax_ode_nuts_Dysbiotic_Static_20260320_044847",
    "DH": "jax_ode_nuts_Dysbiotic_HOBIC_20260320_052844",
}

PARAM_LABELS = [
    r"$a_{11}$",
    r"$a_{12}$",
    r"$a_{22}$",
    r"$\mu_1$",
    r"$\mu_2$",
    r"$a_{33}$",
    r"$a_{34}$",
    r"$a_{44}$",
    r"$\mu_3$",
    r"$\mu_4$",
    r"$a_{13}$",
    r"$a_{14}$",
    r"$a_{23}$",
    r"$a_{24}$",
    r"$a_{55}$",
    r"$\mu_5$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}$",
    r"$a_{45}$",
]

COND_COLORS = {"CS": "#1f77b4", "CH": "#2ca02c", "DS": "#ff7f0e", "DH": "#d62728"}
COND_LABELS = {"CS": "CS", "CH": "CH", "DS": "DS", "DH": "DH"}
COND_FULL = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}


def load_map(run_name):
    p = RUNS / run_name / "theta_MAP.json"
    if not p.exists():
        return None
    mj = json.load(open(p))
    return np.array([mj[str(i)] for i in range(20)])


def main():
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=False)

    all_corrs = {}
    for ax_idx, cond in enumerate(["CS", "CH", "DS", "DH"]):
        ax = axes[ax_idx]
        m1 = load_map(P1[cond])
        m2 = load_map(P2[cond])

        if m1 is None or m2 is None:
            ax.set_title(f"{COND_FULL[cond]}\n(pending)")
            continue

        color = COND_COLORS[cond]

        # Plot all 20 params
        ax.scatter(m1, m2, s=20, c=color, edgecolors="k", linewidths=0.3, zorder=3, alpha=0.85)

        # Annotate outliers (|diff| > 1.0)
        for i in range(20):
            if abs(m1[i] - m2[i]) > 1.5:
                ax.annotate(
                    PARAM_LABELS[i],
                    (m1[i], m2[i]),
                    fontsize=5.5,
                    ha="left",
                    va="bottom",
                    xytext=(3, 3),
                    textcoords="offset points",
                    color="gray",
                )

        # Diagonal line
        lo = min(m1.min(), m2.min()) - 0.5
        hi = max(m1.max(), m2.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.4, zorder=1)

        # Correlation and RMSE
        r = np.corrcoef(m1, m2)[0, 1]
        rmse = np.sqrt(np.mean((m1 - m2) ** 2))
        all_corrs[cond] = (r, rmse)

        ax.text(
            0.05,
            0.92,
            f"$r = {r:.3f}$\nRMSD$= {rmse:.2f}$",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )

        ax.set_title(COND_FULL[cond], fontsize=9, fontweight="bold")
        ax.set_xlabel(r"Phase 1 (fix-$\psi$)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, lw=0.5)

        if ax_idx == 0:
            ax.set_ylabel(r"Phase 2 (free $\psi$)")

    fig.tight_layout(w_pad=0.5)
    out = FIG_DIR / "phase1_vs_phase2_map.pdf"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")

    for cond, (r, rmse) in all_corrs.items():
        print(f"  {cond}: r={r:.4f}, RMSD={rmse:.3f}")

    plt.close(fig)


if __name__ == "__main__":
    main()
