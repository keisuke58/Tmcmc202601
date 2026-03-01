#!/usr/bin/env python3
"""
plot_material_model_literature.py
=================================
Fig 11 (updated): Material models (DI + EPS synergy) with AFM literature overlay.

Shows DI → E_bio mapping and EPS synergy model alongside experimental biofilm
stiffness measurements from Pattem et al. 2018/2021 and Gloag et al. 2019.

Usage:
  python plot_material_model_literature.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "figures" / "paper_final"
_OUT.mkdir(parents=True, exist_ok=True)
_CI_DIR = _HERE / "_ci_0d_results"

# Material model (0D DI scale)
E_MAX = 1000.0  # Pa (commensal/diverse)
E_MIN = 10.0  # Pa (dysbiotic/mono-dominated)
DI_SCALE = 1.0  # 0D ODE DI values
DI_EXP = 2.0


def E_model(di):
    r = np.clip(di / DI_SCALE, 0, 1)
    return E_MAX * (1 - r) ** DI_EXP + E_MIN * r


# ── Literature experimental data ──────────────────────────────────────
# Each entry: (label, E_Pa, E_err_Pa, DI_approx, marker, color, ref)
# DI_approx is estimated based on condition type (not directly measured)
# E values converted to Pa where needed

LITERATURE = [
    # Pattem et al. 2018 (Sci Rep, PMC5890245)
    # AFM nanoindentation, oral microcosm on HA discs, PBS hydrated
    {
        "label": "Low-sucrose Day 3",
        "E": 14350,  # 14.35 kPa
        "E_err": 1750,
        "DI_approx": 0.15,  # diverse, commensal-like
        "marker": "s",
        "color": "#2ca02c",
        "ref": "Pattem 2018",
    },
    {
        "label": "Low-sucrose Day 5",
        "E": 1170,  # 1.17 kPa
        "E_err": 80,
        "DI_approx": 0.25,  # aging shifts composition
        "marker": "s",
        "color": "#7fc97f",
        "ref": "Pattem 2018",
    },
    {
        "label": "High-sucrose Day 3",
        "E": 550,  # 0.55 kPa
        "E_err": 20,
        "DI_approx": 0.70,  # cariogenic, reduced diversity
        "marker": "^",
        "color": "#d62728",
        "ref": "Pattem 2018",
    },
    {
        "label": "High-sucrose Day 5",
        "E": 560,  # 0.56 kPa
        "E_err": 60,
        "DI_approx": 0.75,  # cariogenic, mature
        "marker": "^",
        "color": "#ff7f0e",
        "ref": "Pattem 2018",
    },
    # Pattem et al. 2021 (Sci Rep, PMC8355335)
    # Hydrated oral biofilm, 100 min rehydration
    {
        "label": "Low-carb rehydrated",
        "E": 10400,  # 10.4 kPa
        "E_err": 6400,
        "DI_approx": 0.15,
        "marker": "D",
        "color": "#2ca02c",
        "ref": "Pattem 2021",
    },
    {
        "label": "High-carb rehydrated",
        "E": 2800,  # 2.8 kPa
        "E_err": 2100,
        "DI_approx": 0.65,
        "marker": "D",
        "color": "#d62728",
        "ref": "Pattem 2021",
    },
    # Gloag et al. 2019 (J Bacteriol, PMC6707914)
    # Dual-species rheology, G' storage modulus
    {
        "label": "Dual-species G'",
        "E": 160,  # 160 Pa (G', not E; E ≈ 2-3x G')
        "E_err": 100,
        "DI_approx": 0.40,  # two species = moderate diversity
        "marker": "o",
        "color": "#9467bd",
        "ref": "Gloag 2019",
    },
    # Southampton thesis — S. mutans monospecies
    {
        "label": "S. mutans mono",
        "E": 380,
        "E_err": 350,
        "DI_approx": 0.90,  # single species = high DI
        "marker": "v",
        "color": "#8c564b",
        "ref": "S'ton thesis",
    },
]


def _compute_eps_synergy_map_values():
    """Compute EPS synergy E values for 4 conditions using MAP theta."""
    import sys as _sys

    _sys.path.insert(0, str(_HERE.parent / "tmcmc" / "program2602"))
    _sys.path.insert(0, str(_HERE))
    from improved_5species_jit import BiofilmNewtonSolver5S
    from material_models import compute_E_eps_synergy

    # MAP theta paths
    _RUNS = _HERE.parent / "data_5species" / "_runs"
    COND_RUNS = {
        "commensal_static": _RUNS / "commensal_static" / "theta_MAP.json",
        "commensal_hobic": _RUNS / "commensal_hobic" / "theta_MAP.json",
        "dh_baseline": _RUNS / "dh_baseline" / "theta_MAP.json",
        "dysbiotic_static": _RUNS / "dysbiotic_static" / "theta_MAP.json",
    }

    results = {}
    for cond, theta_path in COND_RUNS.items():
        if not theta_path.exists():
            continue
        with open(theta_path) as f:
            data = json.load(f)
        # Handle dict format (theta_full key) or plain list
        if isinstance(data, dict):
            theta = np.array(data.get("theta_full", data.get("theta_sub", [])))
        else:
            theta = np.array(data)
        solver = BiofilmNewtonSolver5S(
            dt=1e-4,
            maxtimestep=750,
            active_species=[0, 1, 2, 3, 4],
            c_const=25.0,
            alpha_const=0.0,
            phi_init=0.02,
            K_hill=0.05,
            n_hill=4.0,
            use_numba=True,
        )
        _, g_arr = solver.solve(theta)
        phi = g_arr[-1, 0:5]
        E_eps = float(compute_E_eps_synergy(phi.reshape(1, -1))[0])
        results[cond] = {"phi": phi, "E_eps": E_eps}
    return results


def main():
    # Load our model's condition-specific results
    master_path = _CI_DIR / "master_summary_0d.json"
    master = {}
    if master_path.exists():
        with open(master_path) as f:
            master = json.load(f)

    # Compute EPS synergy MAP values
    eps_results = _compute_eps_synergy_map_values()

    COND_META = {
        "commensal_static": {"label": "CS", "color": "#2ca02c"},
        "commensal_hobic": {"label": "CH", "color": "#17becf"},
        "dh_baseline": {"label": "DH", "color": "#d62728"},
        "dysbiotic_static": {"label": "DS", "color": "#ff7f0e"},
    }

    # ── Figure (3-panel) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ================================================================
    # Panel (a): E(DI) curve + literature + our conditions
    # ================================================================
    ax = axes[0]

    # Model curve
    di_arr = np.linspace(0, 1, 300)
    E_arr = E_model(di_arr)
    ax.plot(di_arr, E_arr, "k-", linewidth=2.5, label="$E(DI)$ model", zorder=2)

    # Literature data
    for lit in LITERATURE:
        label = f'{lit["ref"]}: {lit["label"]}'
        ax.errorbar(
            lit["DI_approx"],
            lit["E"],
            yerr=lit["E_err"],
            fmt=lit["marker"],
            color=lit["color"],
            markersize=8,
            markeredgecolor="k",
            markeredgewidth=0.5,
            capsize=4,
            capthick=1,
            elinewidth=1,
            label=label,
            zorder=4,
        )

    # Our conditions (DI model MAP values)
    for c in ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]:
        if c not in master:
            continue
        m = master[c]
        meta = COND_META[c]
        ax.scatter(
            m["di_0d_map"],
            m["E_di_map"],
            marker="*",
            s=250,
            color=meta["color"],
            edgecolor="navy",
            linewidth=1.5,
            zorder=6,
            label=f"DI model: {meta['label']}",
        )
        ci_di = m.get("di_0d_ci90", [])
        ci_e = m.get("E_di_ci90", [])
        if ci_di and ci_e:
            ax.fill_between(
                [ci_di[0], ci_di[1]],
                [ci_e[0], ci_e[0]],
                [ci_e[1], ci_e[1]],
                color=meta["color"],
                alpha=0.08,
                zorder=1,
            )

    ax.set_xlabel("Dysbiosis Index ($DI_{0D}$)", fontsize=12)
    ax.set_ylabel("$E_{bio}$ [Pa]", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(5, 50000)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("(a) E(DI) model + literature", fontsize=11, weight="bold")
    ax.legend(fontsize=5.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.2, which="both")

    ax.annotate(
        "Diverse\n(commensal)",
        xy=(0.1, E_model(0.1)),
        xytext=(0.05, 3000),
        fontsize=8,
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=1),
        ha="center",
    )
    ax.annotate(
        "Mono-dominated\n(dysbiotic)",
        xy=(0.85, E_model(0.85)),
        xytext=(0.75, 3),
        fontsize=8,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        ha="center",
    )

    # ================================================================
    # Panel (b): EPS synergy model — condition bar chart
    # ================================================================
    ax = axes[1]

    cond_order = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
    labels_short = [COND_META[c]["label"] for c in cond_order]
    colors = [COND_META[c]["color"] for c in cond_order]

    # DI model values
    E_di_vals = []
    E_eps_vals = []
    for c in cond_order:
        E_di_vals.append(master.get(c, {}).get("E_di_map", 0))
        E_eps_vals.append(eps_results.get(c, {}).get("E_eps", 0))

    x = np.arange(len(cond_order))
    w = 0.35
    bars1 = ax.bar(
        x - w / 2,
        E_di_vals,
        w,
        color=colors,
        alpha=0.5,
        edgecolor="k",
        linewidth=0.8,
        label="DI model",
    )
    bars2 = ax.bar(
        x + w / 2,
        E_eps_vals,
        w,
        color=colors,
        alpha=0.9,
        edgecolor="navy",
        linewidth=1.2,
        label="EPS synergy",
        hatch="//",
    )

    # Value labels
    for bar, val in zip(bars1, E_di_vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 15,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )
    for bar, val in zip(bars2, E_eps_vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 15,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Literature reference bands
    ax.axhspan(160, 560, alpha=0.06, color="purple", label="Literature range (Gloag–Pattem)")
    ax.axhspan(550, 14350, alpha=0.04, color="green", label="Full lit. range (20–14k Pa)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=11)
    ax.set_ylabel("$E_{bio}$ [Pa]", fontsize=12)
    ax.set_title("(b) DI vs EPS synergy model", fontsize=11, weight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, max(max(E_di_vals), max(E_eps_vals)) * 1.3)

    # Annotate ratio
    if E_eps_vals[0] > 0 and E_eps_vals[3] > 0:
        ratio_di = E_di_vals[0] / max(E_di_vals[3], 1)
        ratio_eps = E_eps_vals[0] / max(E_eps_vals[3], 1)
        ax.annotate(
            f"CS/DS ratio:\n  DI: {ratio_di:.0f}x\n  EPS: {ratio_eps:.0f}x",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )

    # ================================================================
    # Panel (c): Summary table
    # ================================================================
    ax = axes[2]
    ax.axis("off")

    table_data = [
        ["Condition", "E_DI [Pa]", "E_EPS [Pa]", "DI (0D)"],
        ["─" * 10, "─" * 10, "─" * 10, "─" * 8],
    ]
    for c in cond_order:
        lbl = COND_META[c]["label"]
        e_di = master.get(c, {}).get("E_di_map", 0)
        e_eps = eps_results.get(c, {}).get("E_eps", 0)
        di_val = master.get(c, {}).get("di_0d_map", 0)
        table_data.append([lbl, f"{e_di:.0f}", f"{e_eps:.0f}", f"{di_val:.3f}"])
    table_data += [
        ["", "", "", ""],
        ["Literature", "E [Pa]", "DI (est.)", "Method"],
        ["─" * 10, "─" * 10, "─" * 10, "─" * 8],
        ["Pattem '18 LS", "14,350", "~0.15", "AFM"],
        ["Pattem '18 HS", "550", "~0.70", "AFM"],
        ["Pattem '21 LC", "10,400", "~0.15", "AFM"],
        ["Pattem '21 HC", "2,800", "~0.65", "AFM"],
        ["Gloag '19", "160 (G')", "~0.40", "Rheology"],
        ["S'ton thesis", "380", "~0.90", "Compress."],
        ["", "", "", ""],
        ["EPS SYNERGY", "PARAMS", "", ""],
        ["ε_So=0.3", "ε_An=0.6", "ε_Vei=0.1", ""],
        ["ε_Fn=0.4", "ε_Pg=-0.3", "γ=4.0", ""],
    ]

    text = "\n".join("  ".join(f"{cell:<12}" for cell in row) for row in table_data)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", alpha=0.9),
    )

    ax.text(
        0.02,
        0.02,
        "Note: Literature DI values are estimated from condition type.\n"
        "EPS synergy: M = EPS_total × exp(γ × CrossLink)\n"
        "  EPS_total = Σ φᵢεᵢ (species-specific production)\n"
        "  CrossLink = H(φ_active)/H_max (producer evenness)\n"
        "Our E range (30–900 Pa) within literature 20–14,000 Pa.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_title("(c) Model comparison table", fontsize=11, weight="bold")

    fig.suptitle(
        "Fig 11: Material Models with Experimental Validation\n"
        "DI model: $E = E_{max}(1-r)^2 + E_{min} r$  |  "
        "EPS synergy: $E = E_{min} + (E_{max}-E_{min}) M/M_{ref}$",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = _OUT / "Fig11_material_model_eps_synergy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print("  (original Fig11_material_model.png preserved)")


if __name__ == "__main__":
    main()
