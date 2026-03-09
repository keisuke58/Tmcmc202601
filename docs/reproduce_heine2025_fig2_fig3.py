#!/usr/bin/env python3
"""
Reproduce Heine et al. (2025) Fig 2 and Fig 3 from extracted CSV data.

Fig 2: Biofilm volume (Tukey box plots) + membrane viability (stacked bar)
Fig 3: Viable species distribution (Tukey box plots, 5 species × 4 conditions)

Data source: Digitized from published figures in
  Heine N et al., Front. Oral Health 6:1649419 (2025)
  doi: 10.3389/froh.2025.1649419

CSV files: data_5species/experiment_data/fig2_*.csv, fig3_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = os.path.join(os.path.dirname(__file__), "..", "data_5species", "experiment_data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS = [1, 3, 6, 10, 15, 21]

CONDITIONS = [
    ("Commensal", "Static"),
    ("Commensal", "HOBIC"),
    ("Dysbiotic", "Static"),
    ("Dysbiotic", "HOBIC"),
]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# ── Fig 2: Biofilm volume + viability ──


def make_fig2():
    df_vol = pd.read_csv(os.path.join(BASE, "fig2_biofilm_volume_replicates.csv"))
    df_mem = pd.read_csv(os.path.join(BASE, "fig2_membrane_distribution.csv"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    for col, (cond, cult) in enumerate(CONDITIONS):
        # ── Volume box plot (top row) ──
        ax_vol = axes[0, col]
        mask_v = (df_vol["condition"] == cond) & (df_vol["cultivation"] == cult)
        dv = df_vol[mask_v]

        box_data, box_pos = [], []
        for di, day in enumerate(DAYS):
            vals = dv[dv["day"] == day]["biofilm_volume_x1e6"].values
            if len(vals) > 0:
                box_data.append(vals)
                box_pos.append(di)

        if box_data:
            bp = ax_vol.boxplot(
                box_data,
                positions=box_pos,
                widths=0.5,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker="o",
                    markersize=3,
                    alpha=0.5,
                    markerfacecolor="gray",
                    markeredgecolor="none",
                ),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.2),
                boxprops=dict(linewidth=0.8),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("white")
                patch.set_edgecolor("black")

        ax_vol.set_xlim(-0.6, 5.6)
        ymax = dv["biofilm_volume_x1e6"].max() * 1.2 if len(dv) > 0 else 1.5
        ax_vol.set_ylim(0, max(ymax, 1.5))
        ax_vol.set_xticks(range(len(DAYS)))
        ax_vol.set_xticklabels([str(d) for d in DAYS], fontsize=8)
        ax_vol.tick_params(axis="both", labelsize=8)
        ax_vol.spines["top"].set_visible(False)
        ax_vol.spines["right"].set_visible(False)
        ax_vol.set_title(f"{cond} {cult}", fontsize=10, fontweight="bold", pad=8)

        if col == 0:
            ax_vol.set_ylabel(
                r"Biofilm Volume $\times 10^6$" + "\n" + r"[$\mu m^3$/image]", fontsize=8.5
            )
        ax_vol.set_xlabel("Time [days]", fontsize=8)

        # ── Viability stacked bar (bottom row) ──
        ax_mem = axes[1, col]
        mask_m = (df_mem["condition"] == cond) & (df_mem["cultivation"] == cult)
        dm = df_mem[mask_m].sort_values("day")

        if len(dm) > 0:
            x_pos = np.arange(len(dm))
            intact = dm["intact_membrane_pct"].values
            damaged = dm["damaged_membrane_pct"].values

            ax_mem.bar(x_pos, intact, color="#2ca02c", label="Intact Membrane", width=0.6)
            ax_mem.bar(
                x_pos, damaged, bottom=intact, color="#d62728", label="Damaged Membrane", width=0.6
            )

            # Error bars
            if "error_pct" in dm.columns:
                err = dm["error_pct"].values
                ax_mem.errorbar(
                    x_pos,
                    intact,
                    yerr=err,
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    linewidth=0.8,
                )

            ax_mem.set_xticks(x_pos)
            ax_mem.set_xticklabels([str(d) for d in dm["day"].values], fontsize=8)

        ax_mem.set_ylim(0, 105)
        ax_mem.tick_params(axis="both", labelsize=8)
        ax_mem.spines["top"].set_visible(False)
        ax_mem.spines["right"].set_visible(False)
        ax_mem.set_xlabel("Time [days]", fontsize=8)

        if col == 0:
            ax_mem.set_ylabel("Distribution [%]", fontsize=8.5)

    # Legend for viability
    legend_elems = [
        Patch(facecolor="#2ca02c", label="Intact Membrane"),
        Patch(facecolor="#d62728", label="Damaged Membrane"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper right",
        fontsize=8,
        frameon=True,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.98, 0.98),
    )

    fig.suptitle(
        "Fig. 2 — Biofilm volume and viability (reproduced from Heine et al. 2025)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    for ext in ["png", "pdf"]:
        out = os.path.join(OUT_DIR, f"fig2_heine2025_reproduced.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close()


# ── Fig 3: Species distribution ──

SPECIES_MAP = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
    "P. gingivalis_W83": 4,
}

SPECIES_LABELS_COMM = [
    r"$\it{S.\;oralis}$",
    r"$\it{A.\;naeslundii}$",
    r"$\it{V.\;dispar}$",
    r"$\it{F.\;nucleatum}$",
    r"$\it{P.\;gingivalis}$$_{20709}$",
]
SPECIES_LABELS_DYSB = [
    r"$\it{S.\;oralis}$",
    r"$\it{A.\;naeslundii}$",
    r"$\it{V.\;parvula}$",
    r"$\it{F.\;nucleatum}$",
    r"$\it{P.\;gingivalis}$$_{W83}$",
]

# Match paper colors: blue, green, orange/yellow, purple, red
SPECIES_COLORS = ["#0055D4", "#2ca02c", "#E8A000", "#9467bd", "#d62728"]


def make_fig3():
    df = pd.read_csv(os.path.join(BASE, "fig3_species_distribution_replicates.csv"))
    df["sp"] = df["species"].map(SPECIES_MAP)

    # Note: species are independent per replicate (data digitized from published box plots)
    # No need to aggregate — each (condition, cultivation, day, species, replicate) is unique

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    panel_map = {
        (0, 0): ("Commensal", "Static"),
        (0, 1): ("Commensal", "HOBIC"),
        (1, 0): ("Dysbiotic", "Static"),
        (1, 1): ("Dysbiotic", "HOBIC"),
    }

    for (ri, ci), (cond, cult) in panel_map.items():
        ax = axes[ri, ci]
        mask = (df["condition"] == cond) & (df["cultivation"] == cult)
        dc = df[mask]

        for sp in range(5):
            color = SPECIES_COLORS[sp]
            offset = (sp - 2) * 0.13

            box_data, box_pos = [], []
            for di, day in enumerate(DAYS):
                vals = dc[(dc["sp"] == sp) & (dc["day"] == day)]["distribution_pct"].values
                if len(vals) > 0:
                    box_data.append(vals)
                    box_pos.append(di + offset)

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    positions=box_pos,
                    widths=0.10,
                    patch_artist=True,
                    zorder=3,
                    showfliers=True,
                    flierprops=dict(
                        marker=".",
                        markersize=3,
                        alpha=0.6,
                        markerfacecolor=color,
                        markeredgecolor="none",
                    ),
                    whiskerprops=dict(color=color, alpha=0.7, linewidth=0.8),
                    capprops=dict(color=color, alpha=0.7, linewidth=0.8),
                    medianprops=dict(color="black", linewidth=1.0),
                    boxprops=dict(linewidth=0.8),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.35)
                    patch.set_edgecolor(color)

        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(-2, 105)
        ax.set_xticks(range(len(DAYS)))
        ax.set_xticklabels([str(d) for d in DAYS], fontsize=9)
        ax.set_xlabel("Time [days]", fontsize=9.5)
        ax.set_ylabel("Distribution [%]", fontsize=9.5)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.08, linewidth=0.3, zorder=0)

        model_label = "Commensal Model" if cond == "Commensal" else "Dysbiotic Model"
        cult_label = "Static Cultivation" if cult == "Static" else "HOBIC Cultivation"
        ax.set_title(f"{model_label} — {cult_label}", fontsize=10, fontweight="bold", pad=8)

    # Legends (different species labels for commensal vs dysbiotic)
    from matplotlib.lines import Line2D

    legend_comm = [
        Patch(
            facecolor=SPECIES_COLORS[i],
            alpha=0.5,
            edgecolor=SPECIES_COLORS[i],
            lw=0.8,
            label=SPECIES_LABELS_COMM[i],
        )
        for i in range(5)
    ]
    legend_dysb = [
        Patch(
            facecolor=SPECIES_COLORS[i],
            alpha=0.5,
            edgecolor=SPECIES_COLORS[i],
            lw=0.8,
            label=SPECIES_LABELS_DYSB[i],
        )
        for i in range(5)
    ]

    axes[0, 1].legend(
        handles=legend_comm,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        edgecolor="#cccccc",
        handlelength=1.5,
    )
    axes[1, 1].legend(
        handles=legend_dysb,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        edgecolor="#cccccc",
        handlelength=1.5,
    )

    fig.suptitle(
        "Fig. 3 — Viable species distribution (reproduced from Heine et al. 2025, $N$=8 per box)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    for ext in ["png", "pdf"]:
        out = os.path.join(OUT_DIR, f"fig3_heine2025_reproduced.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("=== Reproducing Heine et al. (2025) Fig 2 ===")
    make_fig2()
    print("\n=== Reproducing Heine et al. (2025) Fig 3 ===")
    make_fig3()
    print("\nDone.")
