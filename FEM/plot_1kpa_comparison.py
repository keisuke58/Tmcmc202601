#!/usr/bin/env python3
"""
plot_1kpa_comparison.py
========================
Comparison figure: 1 kPa (brushing) vs 1 MPa (occlusal) loading.
Shows physical displacement values under clinically relevant load.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)
_JOBS_1KPA = _HERE / "_abaqus_1kpa_jobs"
_JOBS_1MPA = _HERE / "_abaqus_auto_jobs"

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
COND_LABELS = {
    "commensal_static": "Comm. Static",
    "commensal_hobic": "Comm. HOBIC",
    "dh_baseline": "Dysb. HOBIC",
    "dysbiotic_static": "Dysb. Static",
}
COND_COLORS = {
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#17becf",
    "dh_baseline": "#d62728",
    "dysbiotic_static": "#ff7f0e",
}

# Results
R_1KPA = {
    "commensal_static": {"E_pa": 909.1, "U_max": 0.4391, "U_mean": 0.1907, "mises_max": 0.0452},
    "commensal_hobic": {"E_pa": 890.1, "U_max": 0.4483, "U_mean": 0.1947, "mises_max": 0.0452},
    "dh_baseline": {"E_pa": 705.1, "U_max": 0.5671, "U_mean": 0.2463, "mises_max": 0.0452},
    "dysbiotic_static": {"E_pa": 32.3, "U_max": 12.9297, "U_mean": 5.6161, "mises_max": 0.0452},
}
R_1MPA = {
    "commensal_static": {"E_pa": 909.1, "U_max": 439.1, "mises_max": 45.2},
    "commensal_hobic": {"E_pa": 890.1, "U_max": 448.3, "mises_max": 45.2},
    "dh_baseline": {"E_pa": 705.1, "U_max": 567.1, "mises_max": 45.2},
    "dysbiotic_static": {"E_pa": 32.3, "U_max": 12929.7, "mises_max": 45.2},
}


def load_nodes(csv_path):
    xs, ys, zs, umags = [], [], [], []
    with open(csv_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9:
                xs.append(float(parts[2]))
                ys.append(float(parts[3]))
                zs.append(float(parts[4]))
                umags.append(float(parts[8]))
    return np.array(xs), np.array(ys), np.array(zs), np.array(umags)


def fig_bar_comparison():
    """4-panel bar chart for 1 kPa results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(CONDITIONS))
    colors = [COND_COLORS[c] for c in CONDITIONS]
    labels = [COND_LABELS[c] for c in CONDITIONS]

    # (a) E_bio
    ax = axes[0, 0]
    vals = [R_1KPA[c]["E_pa"] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$E_{bio}$ [Pa]", fontsize=11)
    ax.set_title("(a) Biofilm Stiffness (DI model)", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 20, f"{v:.0f}", ha="center", fontsize=9)

    # (b) U_max [mm] — 1 kPa
    ax = axes[0, 1]
    vals = [R_1KPA[c]["U_max"] for c in CONDITIONS]
    bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$U_{max}$ [mm]", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("(b) Max Displacement (1 kPa brushing)", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.3, f"{v:.2f}", ha="center", fontsize=9)

    # (c) Comparison 1 kPa vs 1 MPa (dysb_static only)
    ax = axes[1, 0]
    load_labels = ["1 kPa\n(brushing)", "1 MPa\n(occlusal)"]
    for i, cond in enumerate(CONDITIONS):
        u_1k = R_1KPA[cond]["U_max"]
        u_1m = R_1MPA[cond]["U_max"]
        ax.plot(
            [0, 1],
            [u_1k, u_1m],
            "o-",
            color=COND_COLORS[cond],
            label=COND_LABELS[cond],
            markersize=8,
            linewidth=2,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(load_labels, fontsize=10)
    ax.set_ylabel("$U_{max}$ [mm]", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("(c) Load Comparison: Brushing vs Occlusal", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) E vs U (log-log) with physical regime annotation
    ax = axes[1, 1]
    for cond in CONDITIONS:
        e = R_1KPA[cond]["E_pa"]
        u = R_1KPA[cond]["U_max"]
        ax.scatter(
            e, u, s=150, c=COND_COLORS[cond], edgecolor="k", zorder=5, label=COND_LABELS[cond]
        )
    # Fit line
    es = np.array([R_1KPA[c]["E_pa"] for c in CONDITIONS])
    us = np.array([R_1KPA[c]["U_max"] for c in CONDITIONS])
    e_fit = np.logspace(np.log10(20), np.log10(1200), 50)
    # U ∝ 1/E → log(U) = -log(E) + const
    c_fit = np.mean(np.log10(us) + np.log10(es))
    ax.plot(e_fit, 10 ** (c_fit - np.log10(e_fit)), "k--", alpha=0.5, label="slope = -1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$E_{bio}$ [Pa]", fontsize=11)
    ax.set_ylabel("$U_{max}$ [mm]", fontsize=11)
    ax.set_title("(d) Stiffness vs Displacement (1 kPa)", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Physical regime annotation
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.text(500, 1.2, "biofilm thickness ~1 mm", fontsize=8, color="gray", ha="center")

    fig.suptitle(
        "3D Abaqus: Brushing Load (1 kPa) — Physically Realistic Displacement\n"
        "Dysbiotic biofilm deforms 29x more than commensal under identical load",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = _FIG_DIR / "stress_1kpa_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_3d_1kpa():
    """4-panel 3D scatter with unified colorbar for 1 kPa results."""
    fig = plt.figure(figsize=(18, 16))

    # Load all data
    all_umag = []
    data = {}
    for cond in CONDITIONS:
        csv_path = _JOBS_1KPA / f"{cond}_T23_1kpa" / "nodes_3d.csv"
        if csv_path.exists():
            xs, ys, zs, umags = load_nodes(csv_path)
            data[cond] = (xs, ys, zs, umags)
            all_umag.append(umags)

    if not data:
        print("No 1kPa data found")
        return

    all_umag = np.concatenate(all_umag)
    vmin = max(all_umag.min(), 0.001)
    vmax = all_umag.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    max_pts = 6000
    rng = np.random.default_rng(42)

    conds = [c for c in CONDITIONS if c in data]
    for idx, cond in enumerate(conds):
        xs, ys, zs, umags = data[cond]
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        n_nodes = len(xs)
        if n_nodes > max_pts:
            sel = rng.choice(n_nodes, max_pts, replace=False)
        else:
            sel = np.arange(n_nodes)

        ax.scatter(
            xs[sel],
            ys[sel],
            zs[sel],
            c=np.clip(umags[sel], vmin, vmax),
            cmap="inferno",
            norm=norm,
            s=3,
            alpha=0.6,
            rasterized=True,
        )

        e_pa = R_1KPA[cond]["E_pa"]
        u_max = R_1KPA[cond]["U_max"]
        label = COND_LABELS[cond]
        ax.set_title(
            f"{label}\n$E_{{bio}}$={e_pa:.0f} Pa, $U_{{max}}$={u_max:.2f} mm",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("X [mm]", fontsize=7)
        ax.set_ylabel("Y [mm]", fontsize=7)
        ax.set_zlabel("Z [mm]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-60)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm = cm.ScalarMappable(norm=norm, cmap="inferno")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Displacement [mm] (log scale)", fontsize=11)

    fig.suptitle(
        "3D Biofilm Displacement — 1 kPa Brushing Load (Unified Log Scale)\n"
        "Dysb. Static: 12.9 mm  |  Comm. Static: 0.44 mm  |  Ratio: 29.4x",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05, wspace=0.15, hspace=0.25)

    out = _FIG_DIR / "stress_3d_1kpa_unified.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    fig_bar_comparison()
    fig_3d_1kpa()


if __name__ == "__main__":
    main()
