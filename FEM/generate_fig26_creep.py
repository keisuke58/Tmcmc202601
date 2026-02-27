#!/usr/bin/env python3
"""
generate_fig26_creep.py — Paper Fig 26: Creep response under constant GCF pressure
===================================================================================

100 Pa GCF (gingival crevicular fluid) pressure applied at t=0.
Shows displacement evolution u(t) = σ₀·J(t) for 4 conditions.
DS shows large creep, CS shows minimal.

Output: FEM/figures/paper_final/Fig26_creep.{png,pdf}
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
    sls_creep_compliance,
)

OUT_DIR = _HERE / "figures" / "paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("Commensal Static", 0.04, "#1f77b4"),
    ("Commensal HOBIC", 0.08, "#2ca02c"),
    ("Dysbiotic HOBIC", 0.55, "#d62728"),
    ("Dysbiotic Static", 0.92, "#ff7f0e"),
]

SIGMA_0 = 100.0  # GCF pressure [Pa]
L_BIOFILM = 0.2e-3  # biofilm thickness [m] = 0.2 mm


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    t_smooth = np.linspace(0, 300, 500)

    # ── (a) Creep compliance J(t) ──
    ax = axes[0]
    for label, di, color in CONDITIONS:
        ve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        E_inf = ve["E_inf"][0]
        E_1 = ve["E_1"][0]
        tau = ve["tau"][0]

        J = sls_creep_compliance(E_inf, E_1, tau, t_smooth)
        ax.plot(
            t_smooth, J * 1e3, color=color, lw=2.5, label=f"{label} (E∞={E_inf:.0f} Pa)"  # mPa^-1
        )

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("J(t) [10⁻³ Pa⁻¹]", fontsize=12)
    ax.set_title("(a) Creep compliance J(t)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 300)
    ax.grid(True, alpha=0.3)

    # ── (b) Displacement under GCF pressure ──
    ax = axes[1]
    for label, di, color in CONDITIONS:
        ve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        E_inf = ve["E_inf"][0]
        E_1 = ve["E_1"][0]
        tau = ve["tau"][0]

        J = sls_creep_compliance(E_inf, E_1, tau, t_smooth)
        u = SIGMA_0 * J * L_BIOFILM * 1e6  # µm

        ax.plot(t_smooth, u, color=color, lw=2.5, label=f"{label}")
        # Dashed: instantaneous u_0
        u_0 = SIGMA_0 / (E_inf + E_1) * L_BIOFILM * 1e6
        ax.axhline(u_0, color=color, ls=":", lw=0.8, alpha=0.4)
        # Dotted: long-term u_inf
        u_inf = SIGMA_0 / E_inf * L_BIOFILM * 1e6
        ax.axhline(u_inf, color=color, ls="--", lw=0.8, alpha=0.4)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Displacement u [µm]", fontsize=12)
    ax.set_title(
        f"(b) Creep displacement (σ₀ = {SIGMA_0} Pa, L = {L_BIOFILM*1e3:.1f} mm)", fontsize=11
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 300)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "— instant u₀ (dotted)\n-- long-term u∞ (dashed)",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=7,
        va="top",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8),
    )

    # Print key numbers
    for label, di, _ in CONDITIONS:
        ve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        u_0 = SIGMA_0 / ve["E_0"][0] * L_BIOFILM * 1e6
        u_inf = SIGMA_0 / ve["E_inf"][0] * L_BIOFILM * 1e6
        ratio = u_inf / u_0
        print(f"  {label}: u_0={u_0:.3f} µm, u_∞={u_inf:.3f} µm, ratio={ratio:.2f}×")

    fig.suptitle(
        "Fig 26: SLS Viscoelastic Creep — GCF Pressure Loading\n"
        "Dysbiotic: large creep (soft + fast τ), Commensal: minimal creep",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ["png", "pdf"]:
        path = OUT_DIR / f"Fig26_creep.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 26 saved: {OUT_DIR / 'Fig26_creep.png'}")


if __name__ == "__main__":
    main()
