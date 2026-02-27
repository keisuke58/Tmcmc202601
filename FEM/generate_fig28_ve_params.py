#!/usr/bin/env python3
"""
generate_fig28_ve_params.py — Paper Fig 28: SLS viscoelastic parameter maps
============================================================================

3-panel figure showing DI-dependent SLS parameters:
  (a) E_inf(DI) — long-term equilibrium modulus
  (b) τ(DI)     — relaxation time
  (c) η(DI)     — dashpot viscosity

Literature scatter overlay: Shaw 2004, Towler 2003, Peterson 2015, Gloag 2019.
Style follows Fig 11 (plot_material_model_literature.py).

Output: FEM/figures/paper_final/Fig28_ve_params.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from material_models import compute_viscoelastic_params_di, DI_SCALE

OUT_DIR = _HERE / "figures" / "paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 biofilm conditions: (label, DI, marker, color)
CONDITIONS = [
    ("Commensal Static", 0.04, "*", "#1f77b4"),
    ("Commensal HOBIC", 0.08, "*", "#2ca02c"),
    ("Dysbiotic HOBIC", 0.55, "*", "#d62728"),
    ("Dysbiotic Static", 0.92, "*", "#ff7f0e"),
]

# Literature data for VE properties
# (label, param, value, error, DI_est, marker, color, ref)
# param: "E_inf", "tau", "eta"
LITERATURE_VE = [
    # Gloag 2019 — dual-species rheology, G'=160 Pa ≈ E_inf
    {
        "label": "Dual-species G'",
        "param": "E_inf",
        "val": 160,
        "err": 100,
        "DI": 0.40,
        "marker": "o",
        "color": "#9467bd",
        "ref": "Gloag 2019",
    },
    # Shaw 2004 — mixed biofilm, τ = 18 min ≈ 1080 s (large deformation)
    # small-deformation τ ~ 10-50 s
    {
        "label": "Mixed biofilm",
        "param": "tau",
        "val": 30,
        "err": 20,
        "DI": 0.35,
        "marker": "s",
        "color": "#17becf",
        "ref": "Shaw 2004",
    },
    # Peterson & Stoodley 2015 — P. aeruginosa, τ ~ 5-20 s
    {
        "label": "P. aeruginosa",
        "param": "tau",
        "val": 12,
        "err": 7,
        "DI": 0.60,
        "marker": "D",
        "color": "#bcbd22",
        "ref": "Peterson 2015",
    },
    # Towler 2003 — mixed oral biofilm, E_0/E_inf = 2-5
    # η = E_1·τ ≈ 500·15 = 7500 Pa·s (mixed condition estimate)
    {
        "label": "Oral mixed",
        "param": "eta",
        "val": 7500,
        "err": 5000,
        "DI": 0.30,
        "marker": "^",
        "color": "#e377c2",
        "ref": "Towler 2003",
    },
    # Stoodley 2002 — S. mutans mono, η ~ 100-1000 Pa·s
    {
        "label": "S. mutans mono",
        "param": "eta",
        "val": 500,
        "err": 400,
        "DI": 0.85,
        "marker": "v",
        "color": "#8c564b",
        "ref": "Stoodley 2002",
    },
]


def main():
    di_arr = np.linspace(0, 1, 300)
    ve = compute_viscoelastic_params_di(di_arr, di_scale=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel (a): E_inf(DI) ──
    ax = axes[0]
    ax.plot(di_arr, ve["E_inf"], "k-", lw=2.5, label="$E_\\infty(DI)$", zorder=2)
    ax.plot(
        di_arr,
        ve["E_0"],
        "k--",
        lw=1.5,
        alpha=0.5,
        label="$E_0(DI) = E_\\infty \\times$ ratio",
        zorder=2,
    )
    ax.fill_between(
        di_arr,
        ve["E_inf"],
        ve["E_0"],
        alpha=0.08,
        color="gray",
        label="$E_1 = E_0 - E_\\infty$ (spring)",
    )

    # Literature scatter
    for lit in LITERATURE_VE:
        if lit["param"] == "E_inf":
            ax.errorbar(
                lit["DI"],
                lit["val"],
                yerr=lit["err"],
                fmt=lit["marker"],
                color=lit["color"],
                markersize=9,
                markeredgecolor="k",
                markeredgewidth=0.5,
                capsize=4,
                label=f'{lit["ref"]}: {lit["label"]}',
                zorder=4,
            )

    # Our conditions
    for label, di, mk, col in CONDITIONS:
        cve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        ax.scatter(
            di,
            cve["E_inf"][0],
            marker=mk,
            s=200,
            color=col,
            edgecolor="navy",
            linewidth=1.5,
            zorder=6,
            label=f"{label}",
        )

    ax.set_xlabel("Dysbiosis Index (DI)", fontsize=12)
    ax.set_ylabel("$E$ [Pa]", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(5, 5000)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("(a) $E_\\infty(DI)$ and $E_0(DI)$", fontsize=11, weight="bold")
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.2, which="both")
    ax.annotate("Dense EPS\n(stiff)", xy=(0.05, 800), fontsize=8, color="green", ha="center")
    ax.annotate("Degraded EPS\n(soft)", xy=(0.90, 15), fontsize=8, color="red", ha="center")

    # ── Panel (b): τ(DI) ──
    ax = axes[1]
    ax.plot(di_arr, ve["tau"], "k-", lw=2.5, label="$\\tau(DI)$", zorder=2)

    # Literature scatter
    for lit in LITERATURE_VE:
        if lit["param"] == "tau":
            ax.errorbar(
                lit["DI"],
                lit["val"],
                yerr=lit["err"],
                fmt=lit["marker"],
                color=lit["color"],
                markersize=9,
                markeredgecolor="k",
                markeredgewidth=0.5,
                capsize=4,
                label=f'{lit["ref"]}: {lit["label"]}',
                zorder=4,
            )

    for label, di, mk, col in CONDITIONS:
        cve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        ax.scatter(
            di,
            cve["tau"][0],
            marker=mk,
            s=200,
            color=col,
            edgecolor="navy",
            linewidth=1.5,
            zorder=6,
            label=f"{label}",
        )

    ax.set_xlabel("Dysbiosis Index (DI)", fontsize=12)
    ax.set_ylabel("$\\tau$ [s]", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 70)
    ax.set_title("(b) Relaxation time $\\tau(DI)$", fontsize=11, weight="bold")
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.2)
    ax.annotate(
        "Slow relaxation\n(cross-linked EPS)", xy=(0.08, 55), fontsize=8, color="green", ha="center"
    )
    ax.annotate("Fast relaxation\n(weak EPS)", xy=(0.85, 8), fontsize=8, color="red", ha="center")

    # ── Panel (c): η(DI) ──
    ax = axes[2]
    ax.plot(di_arr, ve["eta"], "k-", lw=2.5, label="$\\eta(DI) = E_1 \\cdot \\tau$", zorder=2)

    # Literature scatter
    for lit in LITERATURE_VE:
        if lit["param"] == "eta":
            ax.errorbar(
                lit["DI"],
                lit["val"],
                yerr=lit["err"],
                fmt=lit["marker"],
                color=lit["color"],
                markersize=9,
                markeredgecolor="k",
                markeredgewidth=0.5,
                capsize=4,
                label=f'{lit["ref"]}: {lit["label"]}',
                zorder=4,
            )

    for label, di, mk, col in CONDITIONS:
        cve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        ax.scatter(
            di,
            cve["eta"][0],
            marker=mk,
            s=200,
            color=col,
            edgecolor="navy",
            linewidth=1.5,
            zorder=6,
            label=f"{label}",
        )

    ax.set_xlabel("Dysbiosis Index (DI)", fontsize=12)
    ax.set_ylabel("$\\eta$ [Pa$\\cdot$s]", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("(c) Dashpot viscosity $\\eta(DI)$", fontsize=11, weight="bold")
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.2, which="both")
    ax.annotate(
        "High viscosity\n(load-bearing)", xy=(0.08, 3e5), fontsize=8, color="green", ha="center"
    )
    ax.annotate("Low viscosity\n(flows easily)", xy=(0.88, 5), fontsize=8, color="red", ha="center")

    # Print key numbers
    print("  VE parameter summary:")
    for label, di, _, _ in CONDITIONS:
        cve = compute_viscoelastic_params_di(np.array([di]), di_scale=1.0)
        print(
            f"    {label:22s}: E_inf={cve['E_inf'][0]:7.1f} Pa, "
            f"E_0={cve['E_0'][0]:7.1f} Pa, "
            f"tau={cve['tau'][0]:5.1f} s, "
            f"eta={cve['eta'][0]:9.1f} Pa·s"
        )

    fig.suptitle(
        "Fig 28: SLS Viscoelastic Parameters — DI Dependence\n"
        "$E_0 = E_\\infty \\times r(DI)$,  "
        "$\\tau = \\tau_{max}(1-r)^{1.5} + \\tau_{min} r$,  "
        "$\\eta = E_1 \\cdot \\tau$",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    for ext in ["png", "pdf"]:
        path = OUT_DIR / f"Fig28_ve_params.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 28 saved: {OUT_DIR / 'Fig28_ve_params.png'}")


if __name__ == "__main__":
    main()
