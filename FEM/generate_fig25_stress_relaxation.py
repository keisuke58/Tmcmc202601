#!/usr/bin/env python3
"""
generate_fig25_stress_relaxation.py — Paper Fig 25: 4-condition stress relaxation
=================================================================================

Step eigenstrain at t=0 → σ_vm(t) evolution for all 4 biofilm conditions.
Shows how commensal (high E, slow τ) retains stress while dysbiotic relaxes quickly.

Output: FEM/figures/paper_final/Fig25_stress_relaxation.{png,pdf}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from material_models import (
    compute_viscoelastic_params_di,
    sls_stress_relaxation,
)
from solve_viscoelastic_1d import solve_1d_viscoelastic

OUT_DIR = _HERE / "figures" / "paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 conditions: (label, DI_0D, color)
CONDITIONS = [
    ("Commensal Static", 0.04, "#1f77b4"),
    ("Commensal HOBIC", 0.08, "#2ca02c"),
    ("Dysbiotic HOBIC", 0.55, "#d62728"),
    ("Dysbiotic Static", 0.92, "#ff7f0e"),
]

EPS_GROWTH = 0.01  # step eigenstrain
N_ELEM = 100
L = 1.0
T_ARRAY = np.concatenate(
    [
        np.array([0.0]),
        np.logspace(-1, np.log10(300), 80),
    ]
)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Analytical stress relaxation curves ──
    ax = axes[0]
    t_smooth = np.linspace(0, 300, 500)
    for label, di, color in CONDITIONS:
        ve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        E_inf = ve["E_inf"][0]
        E_1 = ve["E_1"][0]
        tau = ve["tau"][0]

        sigma = sls_stress_relaxation(E_inf, E_1, tau, EPS_GROWTH, t_smooth)
        ax.plot(t_smooth, sigma, color=color, lw=2.5, label=f"{label} (τ={tau:.1f}s)")
        # Dashed: elastic reference E_0
        E_0 = E_inf + E_1
        ax.axhline(E_0 * EPS_GROWTH, color=color, ls=":", lw=1, alpha=0.4)
        # Dotted: long-term E_inf
        ax.axhline(E_inf * EPS_GROWTH, color=color, ls="--", lw=1, alpha=0.4)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("σ [Pa]", fontsize=12)
    ax.set_title("(a) Stress relaxation: σ(t) = [E∞ + E₁·exp(-t/τ)]·ε₀", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 300)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "— elastic E₀ (dashed)\n-- long-term E∞ (dotted)",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8),
    )

    # ── (b) Normalized relaxation σ(t)/σ(0) ──
    ax = axes[1]
    for label, di, color in CONDITIONS:
        ve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        E_inf = ve["E_inf"][0]
        E_1 = ve["E_1"][0]
        tau = ve["tau"][0]
        E_0 = E_inf + E_1
        sigma = sls_stress_relaxation(E_inf, E_1, tau, EPS_GROWTH, t_smooth)
        ax.plot(t_smooth, sigma / (E_0 * EPS_GROWTH), color=color, lw=2.5, label=f"{label}")
        # Mark t/τ = 1
        ax.axvline(tau, color=color, ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("σ(t) / σ(0)", fontsize=12)
    ax.set_title("(b) Normalized relaxation (fraction of instantaneous stress)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", ls="-", lw=0.5, alpha=0.3)

    fig.suptitle(
        "Fig 25: SLS Viscoelastic Stress Relaxation — 4 Biofilm Conditions\n"
        f"Step eigenstrain ε₀ = {EPS_GROWTH}, E∞(DI) power-law, τ(DI) Eq. (X)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ["png", "pdf"]:
        path = OUT_DIR / f"Fig25_stress_relaxation.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 25 saved: {OUT_DIR / 'Fig25_stress_relaxation.png'}")


if __name__ == "__main__":
    main()
