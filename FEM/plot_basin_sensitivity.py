#!/usr/bin/env python3
"""
plot_basin_sensitivity.py
=========================
Fig 16: Multi-attractor basin sensitivity of 0D Hamilton ODE.

Shows how posterior parameter uncertainty maps to DI uncertainty
through ODE basin structure (multiple stable equilibria).

Panels:
  (a) DI histogram for dh_baseline (real posterior) — bimodal
  (b) DI vs species composition — which species dominates in each basin
  (c) Cross-condition DI swarm + basin regions
  (d) E(DI) uncertainty propagation through basins

Usage:
  python plot_basin_sensitivity.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_CI_DIR = _HERE / "_ci_0d_results"
_OUT = _HERE / "figures" / "paper_final"
_OUT.mkdir(parents=True, exist_ok=True)

SPECIES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SP_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
SP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

COND_META = {
    "commensal_static": {"label": "Commensal Static", "color": "#2ca02c", "short": "CS"},
    "commensal_hobic": {"label": "Commensal HOBIC", "color": "#17becf", "short": "CH"},
    "dh_baseline": {"label": "Dysbiotic HOBIC", "color": "#d62728", "short": "DH"},
    "dysbiotic_static": {"label": "Dysbiotic Static", "color": "#ff7f0e", "short": "DS"},
}

ALL_CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]


def load_samples(condition):
    """Load per-sample 0D results."""
    path = _CI_DIR / condition / "samples_0d.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_summary(condition):
    """Load summary statistics."""
    path = _CI_DIR / condition / "summary_0d.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def di_to_E(di, E_max=1000.0, E_min=10.0, di_scale=1.0):
    """DI → E material model (same as material_models.py with 0D scale)."""
    r = np.clip(di / di_scale, 0, 1)
    return E_max * (1 - r) ** 2 + E_min * r


def phi_to_E_eps_synergy(phi_arr):
    """EPS synergy material model: φ → E (delegated to material_models)."""
    from material_models import compute_E_eps_synergy

    return compute_E_eps_synergy(np.atleast_2d(phi_arr))


def main():
    # Load all data
    all_samples = {}
    all_summaries = {}
    for c in ALL_CONDITIONS:
        s = load_samples(c)
        if s is not None:
            all_samples[c] = s
        sm = load_summary(c)
        if sm is not None:
            all_summaries[c] = sm

    # Also load master summary for filtered counts
    master_path = _CI_DIR / "master_summary_0d.json"
    with open(master_path) as f:
        master = json.load(f)

    # ----------------------------------------------------------------
    # Create figure
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ================================================================
    # (a) DI histogram for dh_baseline — bimodal distribution
    # ================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    if "dh_baseline" in all_samples:
        dh_data = all_samples["dh_baseline"]
        di_vals = np.array([r["di_0d"] for r in dh_data])
        di_map = master["dh_baseline"]["di_0d_map"]

        # Histogram
        bins = np.linspace(0, 1, 30)
        ax_a.hist(
            di_vals,
            bins=bins,
            color="#d62728",
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
            density=True,
            label="Posterior samples",
        )

        # MAP line
        ax_a.axvline(
            di_map,
            color="navy",
            linewidth=2.5,
            linestyle="--",
            label=f"MAP = {di_map:.3f}",
            zorder=5,
        )

        # Basin regions
        ax_a.axvspan(0, 0.3, alpha=0.08, color="green", label="Diverse basin")
        ax_a.axvspan(0.7, 1.0, alpha=0.08, color="red", label="Monodom. basin")

        # Mean & CI
        ci = master["dh_baseline"]["di_0d_ci90"]
        ax_a.axvspan(ci[0], ci[1], alpha=0.15, color="#d62728", zorder=0)
        ax_a.text(
            0.95,
            0.92,
            f"90% CI: [{ci[0]:.2f}, {ci[1]:.2f}]\n" f"n = {len(dh_data)} (real posterior)",
            transform=ax_a.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax_a.set_xlabel("Dysbiosis Index ($DI_{0D}$)", fontsize=11)
    ax_a.set_ylabel("Density", fontsize=11)
    ax_a.set_title("(a) DH posterior: bimodal DI distribution", fontsize=12, weight="bold")
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.set_xlim(0, 1)

    # ================================================================
    # (b) DI vs species composition — stacked bar for each sample
    # ================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    if "dh_baseline" in all_samples:
        dh_data = all_samples["dh_baseline"]
        # Sort by DI
        sorted_data = sorted(dh_data, key=lambda r: r["di_0d"])
        n = len(sorted_data)
        x_idx = np.arange(n)

        # Stacked bar
        bottom = np.zeros(n)
        phi_keys = ["phi_So", "phi_An", "phi_Vd", "phi_Fn", "phi_Pg"]
        for j, (key, color, name) in enumerate(zip(phi_keys, SP_COLORS, SP_SHORT)):
            vals = np.array([r[key] for r in sorted_data])
            ax_b.bar(
                x_idx, vals, bottom=bottom, color=color, width=1.0, label=name, edgecolor="none"
            )
            bottom += vals

        # DI overlay line (twin axis)
        ax_b2 = ax_b.twinx()
        di_sorted = [r["di_0d"] for r in sorted_data]
        ax_b2.plot(x_idx, di_sorted, "k-", linewidth=2, label="$DI_{0D}$")
        ax_b2.set_ylabel("$DI_{0D}$", fontsize=11, color="k")
        ax_b2.set_ylim(0, 1)

        ax_b.set_xlabel("Posterior samples (sorted by DI)", fontsize=11)
        ax_b.set_ylabel("Species volume fraction $\\phi_i$", fontsize=11)
        ax_b.set_xlim(-0.5, n - 0.5)
        ax_b.set_ylim(0, 1)
        ax_b.legend(fontsize=8, loc="upper left", ncol=5, bbox_to_anchor=(0, 1.02))

        # Annotate basins
        n_low = sum(1 for d in di_sorted if d < 0.5)
        if n_low > 0 and n_low < n:
            ax_b.axvline(n_low - 0.5, color="k", linewidth=1, linestyle=":")
            ax_b.text(
                n_low / 2,
                -0.08,
                "Diverse\nbasin",
                ha="center",
                fontsize=8,
                transform=ax_b.get_xaxis_transform(),
            )
            ax_b.text(
                (n_low + n) / 2,
                -0.08,
                "Monodom.\nbasin",
                ha="center",
                fontsize=8,
                transform=ax_b.get_xaxis_transform(),
            )

    ax_b.set_title("(b) Species composition by basin (DH posterior)", fontsize=12, weight="bold")

    # ================================================================
    # (c) Cross-condition DI swarm + basin filter info
    # ================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    # Basin shading
    ax_c.axhspan(0, 0.3, alpha=0.06, color="green")
    ax_c.axhspan(0.7, 1.0, alpha=0.06, color="red")
    ax_c.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

    x_positions = np.arange(len(ALL_CONDITIONS))
    for i, c in enumerate(ALL_CONDITIONS):
        meta = COND_META[c]
        if c in all_samples:
            di_vals = [r["di_0d"] for r in all_samples[c]]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(di_vals))
            ax_c.scatter(
                i + jitter,
                di_vals,
                color=meta["color"],
                alpha=0.6,
                s=25,
                edgecolor="k",
                linewidth=0.3,
                zorder=3,
            )

        # MAP marker
        if c in master:
            di_map = master[c]["di_0d_map"]
            ax_c.scatter(
                i,
                di_map,
                marker="D",
                color="navy",
                s=80,
                zorder=5,
                edgecolor="white",
                linewidth=1.5,
            )

            # Filter info
            n_kept = master[c]["n_samples_kept"]
            n_total = master[c]["n_samples_total"]
            n_filt = master[c]["n_filtered"]
            is_real = master[c].get("is_real_posterior", False)
            src = "real" if is_real else "perturb"
            label = f"{n_kept}/{n_total}\n({src})"
            if n_filt > 0:
                label += f"\n{n_filt} filtered"
            ax_c.text(
                i,
                -0.08,
                label,
                ha="center",
                fontsize=7.5,
                transform=ax_c.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8),
            )

    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels([COND_META[c]["label"] for c in ALL_CONDITIONS], fontsize=9)
    ax_c.set_ylabel("$DI_{0D}$", fontsize=12)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_title("(c) Cross-condition DI: basin sensitivity", fontsize=12, weight="bold")

    # Legend annotations
    ax_c.text(
        0.02,
        0.97,
        "Diverse basin (DI < 0.3)",
        transform=ax_c.transAxes,
        fontsize=8,
        va="top",
        color="green",
        weight="bold",
    )
    ax_c.text(
        0.02,
        0.07,
        "Mono-dominated basin (DI > 0.7)",
        transform=ax_c.transAxes,
        fontsize=8,
        va="bottom",
        color="red",
        weight="bold",
    )
    ax_c.scatter([], [], marker="D", color="navy", s=60, label="MAP estimate")
    ax_c.legend(fontsize=8, loc="center right")

    # ================================================================
    # (d) E(DI) with uncertainty propagation bands
    # ================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    # E(DI) curve
    di_curve = np.linspace(0, 1, 200)
    E_curve = di_to_E(di_curve)
    ax_d.plot(di_curve, E_curve, "k-", linewidth=2, label="$E(DI)$ model", zorder=2)

    # Per-condition: scatter + CI band
    for c in ALL_CONDITIONS:
        meta = COND_META[c]
        if c in all_samples:
            di_vals = np.array([r["di_0d"] for r in all_samples[c]])
            E_vals = np.array([r["E_di"] for r in all_samples[c]])
            ax_d.scatter(
                di_vals,
                E_vals,
                color=meta["color"],
                alpha=0.5,
                s=20,
                edgecolor="k",
                linewidth=0.3,
                label=meta["label"],
                zorder=3,
            )

        if c in master:
            # MAP point
            di_map = master[c]["di_0d_map"]
            E_map = master[c]["E_di_map"]
            ax_d.scatter(
                di_map,
                E_map,
                marker="D",
                color=meta["color"],
                s=100,
                edgecolor="navy",
                linewidth=2,
                zorder=5,
            )

            # CI horizontal band
            ci = master[c]["di_0d_ci90"]
            E_ci = master[c]["E_di_ci90"]
            ax_d.fill_between(
                [ci[0], ci[1]],
                [E_ci[0], E_ci[0]],
                [E_ci[1], E_ci[1]],
                color=meta["color"],
                alpha=0.1,
                zorder=1,
            )

    # Literature reference zone
    ax_d.axhspan(550, 14000, alpha=0.04, color="blue")
    ax_d.axhspan(20, 380, alpha=0.04, color="orange")
    ax_d.text(
        0.05,
        900,
        "Commensal range\n(Pattem 2018)",
        fontsize=7,
        color="blue",
        alpha=0.7,
        style="italic",
    )
    ax_d.text(
        0.7,
        50,
        "Dysbiotic range\n(Pattem 2018)",
        fontsize=7,
        color="orange",
        alpha=0.7,
        style="italic",
    )

    ax_d.set_xlabel("$DI_{0D}$", fontsize=12)
    ax_d.set_ylabel("$E_{bio}$ [Pa]", fontsize=12)
    ax_d.set_yscale("log")
    ax_d.set_ylim(5, 2000)
    ax_d.set_xlim(0, 1)
    ax_d.set_title("(d) DI → E uncertainty propagation", fontsize=12, weight="bold")
    ax_d.legend(fontsize=7, loc="upper right")
    ax_d.grid(True, alpha=0.2)

    # ================================================================
    # Suptitle
    # ================================================================
    fig.suptitle(
        "Fig 16: Multi-Attractor Basin Sensitivity of Hamilton ODE\n"
        "Parameter uncertainty → basin switching → bimodal DI → wide E uncertainty",
        fontsize=14,
        weight="bold",
        y=0.98,
    )

    out_path = _OUT / "Fig16_basin_sensitivity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save to _ci_0d_results for wiki
    wiki_path = _CI_DIR / "basin_sensitivity.png"
    fig2 = plt.figure(figsize=(16, 12))
    # Re-render would be needed; just copy
    import shutil

    shutil.copy2(out_path, wiki_path)
    print(f"Copied: {wiki_path}")


if __name__ == "__main__":
    main()
