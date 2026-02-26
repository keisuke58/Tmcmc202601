"""
Extract data from Figure 3 (froh-06-1649419-g003) and reproduce the 4-panel boxplot.

The figure shows species distribution (%) over time (days 1,3,6,10,15,21) for:
- Commensal Model: S. oralis, A. naeslundii, V. dispar, F. nucleatum, P. gingivalis_20709
- Dysbiotic Model: S. oralis, A. naeslundii, V. parvula, F. nucleatum, P. gingivalis_W83

Under two cultivation methods: Static and HOBIC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

DAYS = [1, 3, 6, 10, 15, 21]

COMMENSAL_SPECIES = [
    "S. oralis",
    "A. naeslundii",
    "V. dispar",
    "F. nucleatum",
    "P. gingivalis_20709",
]
DYSBIOTIC_SPECIES = [
    "S. oralis",
    "A. naeslundii",
    "V. parvula",
    "F. nucleatum",
    "P. gingivalis_W83",
]

# Colors matching the original figure
COMMENSAL_COLORS = {
    "S. oralis": "#1f77b4",  # blue
    "A. naeslundii": "#2ca02c",  # green
    "V. dispar": "#d4a017",  # dark gold/yellow
    "F. nucleatum": "#9467bd",  # purple
    "P. gingivalis_20709": "#d62728",  # red
}
DYSBIOTIC_COLORS = {
    "S. oralis": "#1f77b4",  # blue
    "A. naeslundii": "#2ca02c",  # green
    "V. parvula": "#ff7f0e",  # orange
    "F. nucleatum": "#9467bd",  # purple
    "P. gingivalis_W83": "#d62728",  # red
}

# ──────────────────────────────────────────────────────────────
# Data read from figure: (median, Q1, Q3, whisker_low, whisker_high)
# Format: {day: {species: (median, q1, q3, wlo, whi)}}
# ──────────────────────────────────────────────────────────────

# CORRECTED using paper text (page 5):
# Commensal Static: "Initially, V. dispar was the dominant species,
#   but was then gradually replaced by S. oralis up until day 6."
# "V. dispar's distribution remained on average stable at 30%"
# "A. naeslundii established itself ... to approx. 30%"
# "F. nucleatum and P. gingivalis were almost undetectable"
commensal_static = {
    1: {
        "V. dispar": (83, 72, 90, 55, 95),
        "S. oralis": (18, 10, 25, 3, 35),
        "A. naeslundii": (0.5, 0, 1, 0, 2),
        "F. nucleatum": (0.5, 0, 0.5, 0, 1),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    3: {
        "S. oralis": (70, 55, 85, 35, 95),
        "V. dispar": (20, 10, 35, 3, 50),
        "A. naeslundii": (8, 3, 12, 1, 18),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    6: {
        "S. oralis": (75, 60, 85, 40, 92),
        "V. dispar": (20, 10, 30, 3, 42),
        "A. naeslundii": (5, 2, 8, 0.5, 12),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    10: {
        "S. oralis": (70, 55, 80, 35, 90),
        "V. dispar": (25, 15, 35, 5, 48),
        "A. naeslundii": (10, 5, 15, 1, 22),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    15: {
        "S. oralis": (38, 25, 50, 10, 65),
        "V. dispar": (40, 28, 52, 15, 62),
        "A. naeslundii": (25, 12, 35, 5, 48),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    21: {
        "S. oralis": (35, 22, 48, 8, 60),
        "V. dispar": (30, 18, 42, 8, 55),
        "A. naeslundii": (35, 20, 48, 10, 62),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
}

# CORRECTED using paper text (page 5):
# Commensal HOBIC: "viable S. oralis was the initially dominant species,
#   although its amount significantly decreased over time from approx. 70% to merely 35%"
# "V. dispar's distribution remained on average stable at 30%"
# "A. naeslundii established itself ... to approx. 30%"
# "F. nucleatum and P. gingivalis were almost undetectable"
commensal_hobic = {
    1: {
        "S. oralis": (70, 55, 82, 35, 95),
        "V. dispar": (10, 3, 20, 1, 30),
        "A. naeslundii": (5, 1, 10, 0.5, 18),
        "F. nucleatum": (10, 2, 18, 0, 25),
        "P. gingivalis_20709": (0.5, 0, 1, 0, 2),
    },
    3: {
        "S. oralis": (95, 85, 98, 70, 99),
        "V. dispar": (10, 3, 18, 1, 25),
        "A. naeslundii": (0.5, 0, 1, 0, 2),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    6: {
        "S. oralis": (75, 58, 85, 35, 92),
        "V. dispar": (25, 12, 35, 5, 48),
        "A. naeslundii": (0.5, 0, 1, 0, 2),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    10: {
        "S. oralis": (55, 38, 68, 20, 82),
        "V. dispar": (30, 18, 40, 8, 52),
        "A. naeslundii": (12, 5, 20, 1, 28),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    15: {
        "S. oralis": (35, 22, 48, 10, 62),
        "V. dispar": (32, 20, 42, 8, 55),
        "A. naeslundii": (28, 15, 38, 5, 52),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 0.5, 0, 1),
    },
    21: {
        "S. oralis": (45, 30, 58, 15, 72),
        "V. dispar": (25, 12, 35, 5, 48),
        "A. naeslundii": (25, 12, 38, 5, 50),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_20709": (0.5, 0, 1, 0, 3),
    },
}

dysbiotic_static = {
    1: {
        "S. oralis": (15, 8, 22, 3, 30),
        "A. naeslundii": (8, 4, 12, 1, 18),
        "V. parvula": (62, 50, 70, 35, 82),
        "F. nucleatum": (5, 2, 8, 0.5, 12),
        "P. gingivalis_W83": (8, 3, 15, 1, 22),
    },
    3: {
        "S. oralis": (12, 5, 18, 2, 25),
        "A. naeslundii": (5, 2, 8, 0.5, 12),
        "V. parvula": (62, 55, 70, 40, 80),
        "F. nucleatum": (5, 2, 8, 0.5, 12),
        "P. gingivalis_W83": (10, 3, 18, 1, 25),
    },
    6: {
        "S. oralis": (5, 2, 10, 0.5, 15),
        "A. naeslundii": (5, 2, 8, 0.5, 12),
        "V. parvula": (58, 45, 68, 30, 80),
        "F. nucleatum": (10, 5, 15, 2, 22),
        "P. gingivalis_W83": (18, 8, 28, 2, 38),
    },
    10: {
        "S. oralis": (3, 1, 5, 0.5, 8),
        "A. naeslundii": (5, 2, 8, 0.5, 12),
        "V. parvula": (55, 40, 65, 25, 78),
        "F. nucleatum": (12, 5, 18, 2, 25),
        "P. gingivalis_W83": (20, 10, 30, 3, 40),
    },
    15: {
        "S. oralis": (3, 1, 5, 0.5, 8),
        "A. naeslundii": (8, 3, 12, 1, 18),
        "V. parvula": (45, 30, 58, 15, 72),
        "F. nucleatum": (18, 10, 25, 3, 35),
        "P. gingivalis_W83": (20, 8, 35, 2, 48),
    },
    21: {
        "S. oralis": (3, 1, 5, 0.5, 8),
        "A. naeslundii": (12, 5, 18, 2, 25),
        "V. parvula": (35, 28, 42, 15, 55),
        "F. nucleatum": (20, 12, 25, 5, 35),
        "P. gingivalis_W83": (25, 18, 32, 8, 42),
    },
}

dysbiotic_hobic = {
    1: {
        "S. oralis": (3, 1, 5, 0.5, 8),
        "A. naeslundii": (1, 0.5, 2, 0.3, 3),
        "V. parvula": (95, 90, 97, 82, 99),
        "F. nucleatum": (0.5, 0, 1, 0, 2),
        "P. gingivalis_W83": (0.5, 0, 1, 0, 2),
    },
    3: {
        "S. oralis": (3, 1, 5, 0.5, 8),
        "A. naeslundii": (2, 1, 3, 0.5, 5),
        "V. parvula": (90, 82, 95, 70, 98),
        "F. nucleatum": (1, 0, 2, 0, 3),
        "P. gingivalis_W83": (1, 0, 2, 0, 3),
    },
    6: {
        "S. oralis": (8, 3, 12, 1, 18),
        "A. naeslundii": (18, 8, 28, 2, 38),
        "V. parvula": (50, 35, 65, 20, 78),
        "F. nucleatum": (5, 2, 10, 0.5, 15),
        "P. gingivalis_W83": (3, 1, 5, 0.5, 8),
    },
    10: {
        "S. oralis": (12, 5, 18, 2, 25),
        "A. naeslundii": (25, 15, 35, 5, 45),
        "V. parvula": (35, 25, 45, 12, 58),
        "F. nucleatum": (15, 8, 22, 2, 30),
        "P. gingivalis_W83": (5, 1, 8, 0.5, 12),
    },
    15: {
        "S. oralis": (3, 1, 5, 0.5, 8),
        "A. naeslundii": (35, 22, 48, 8, 60),
        "V. parvula": (30, 22, 38, 12, 50),
        "F. nucleatum": (20, 12, 28, 5, 38),
        "P. gingivalis_W83": (5, 2, 8, 0.5, 12),
    },
    21: {
        "S. oralis": (2, 0.5, 3, 0.3, 5),
        "A. naeslundii": (38, 25, 55, 8, 68),
        "V. parvula": (22, 12, 30, 5, 42),
        "F. nucleatum": (28, 15, 40, 5, 52),
        "P. gingivalis_W83": (15, 5, 22, 1, 32),
    },
}


def generate_replicates(median, q1, q3, wlo, whi, n=8):
    """Generate n synthetic replicate values from box plot statistics."""
    iqr = q3 - q1
    if iqr < 0.5:
        iqr = 0.5
    # Use a skewed approach: generate from quartile information
    samples = []
    # Place values to approximate the box plot statistics
    # 2 near whisker_low, 2 near Q1, 2 near median, 2 near Q3, 2 near whisker_high
    targets = [
        wlo,
        q1 - 0.2 * iqr,
        q1,
        median - 0.1 * iqr,
        median + 0.1 * iqr,
        q3,
        q3 + 0.2 * iqr,
        whi,
    ]
    for t in targets[:n]:
        noise = np.random.normal(0, iqr * 0.05)
        val = max(0, t + noise)
        samples.append(round(val, 1))
    return samples


def build_dataframe():
    """Build a DataFrame with replicate-level data for all conditions."""
    rows = []

    datasets = [
        ("Commensal", "Static", commensal_static, COMMENSAL_SPECIES),
        ("Commensal", "HOBIC", commensal_hobic, COMMENSAL_SPECIES),
        ("Dysbiotic", "Static", dysbiotic_static, DYSBIOTIC_SPECIES),
        ("Dysbiotic", "HOBIC", dysbiotic_hobic, DYSBIOTIC_SPECIES),
    ]

    for condition, cultivation, data, species_list in datasets:
        for day in DAYS:
            for species in species_list:
                stats = data[day][species]
                median, q1, q3, wlo, whi = stats
                replicates = generate_replicates(median, q1, q3, wlo, whi, n=8)
                for rep_i, val in enumerate(replicates, 1):
                    rows.append(
                        {
                            "condition": condition,
                            "cultivation": cultivation,
                            "day": day,
                            "species": species,
                            "replicate": rep_i,
                            "distribution_pct": val,
                        }
                    )

    return pd.DataFrame(rows)


def plot_panel(ax, df_panel, species_list, color_map, title):
    """Plot one panel (one condition + cultivation) as grouped boxplots."""
    n_species = len(species_list)
    positions_all = []
    colors_all = []
    data_all = []
    tick_positions = []
    tick_labels = []

    # Spacing: each species box is 0.6 wide, spaced 0.75 apart within a group
    # Groups (days) are separated by a gap of 2.0
    box_width = 0.6
    species_spacing = 0.75
    group_gap = 2.5

    for di, day in enumerate(DAYS):
        group_start = di * (n_species * species_spacing + group_gap)
        group_center = group_start + (n_species - 1) * species_spacing / 2
        tick_positions.append(group_center)
        tick_labels.append(str(day))

        for si, species in enumerate(species_list):
            pos = group_start + si * species_spacing
            subset = df_panel[(df_panel["day"] == day) & (df_panel["species"] == species)]
            data_all.append(subset["distribution_pct"].values)
            positions_all.append(pos)
            colors_all.append(color_map[species])

    bp = ax.boxplot(
        data_all,
        positions=positions_all,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=4, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    # Color fliers to match their box
    for i, flier in enumerate(bp["fliers"]):
        flier.set(markerfacecolor=colors_all[i], markeredgecolor=colors_all[i])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_xlim(positions_all[0] - 1.5, positions_all[-1] + 1.5)
    ax.set_ylim(-3, 105)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Distribution [%]", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def make_legend(species_list, color_map):
    """Create legend patches."""
    patches = []
    for sp in species_list:
        # Italicize species name
        label = f"$\\it{{{sp.split('_')[0].replace('. ', '.~')}}}$"
        # Simpler: just use the name directly
        patches.append(
            mpatches.Patch(facecolor=color_map[sp], edgecolor="black", linewidth=0.5, label=sp)
        )
    return patches


def main():
    # Build data
    df = build_dataframe()

    # Save CSV
    csv_path = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_replicates.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Conditions: {df['condition'].unique()}")
    print(f"  Cultivations: {df['cultivation'].unique()}")
    print(f"  Species: {df['species'].unique()}")
    print(f"  Days: {sorted(df['day'].unique())}")

    # Also save a summary (median per group)
    summary = (
        df.groupby(["condition", "cultivation", "day", "species"])["distribution_pct"]
        .agg(
            [
                "median",
                "mean",
                "std",
                "min",
                "max",
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
            ]
        )
        .reset_index()
    )
    summary.columns = [
        "condition",
        "cultivation",
        "day",
        "species",
        "median",
        "mean",
        "std",
        "min",
        "max",
        "q1",
        "q3",
    ]
    summary_path = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))

    # Panel titles
    panel_configs = [
        (
            axes[0, 0],
            "Commensal",
            "Static",
            COMMENSAL_SPECIES,
            COMMENSAL_COLORS,
            "Static Cultivation",
        ),
        (
            axes[0, 1],
            "Commensal",
            "HOBIC",
            COMMENSAL_SPECIES,
            COMMENSAL_COLORS,
            "HOBIC Cultivation",
        ),
        (
            axes[1, 0],
            "Dysbiotic",
            "Static",
            DYSBIOTIC_SPECIES,
            DYSBIOTIC_COLORS,
            "Static Cultivation",
        ),
        (
            axes[1, 1],
            "Dysbiotic",
            "HOBIC",
            DYSBIOTIC_SPECIES,
            DYSBIOTIC_COLORS,
            "HOBIC Cultivation",
        ),
    ]

    for ax, condition, cultivation, species_list, color_map, title in panel_configs:
        df_panel = df[(df["condition"] == condition) & (df["cultivation"] == cultivation)]
        plot_panel(ax, df_panel, species_list, color_map, title)

    # Add row labels on the left side
    axes[0, 0].annotate(
        "Commensal\nModel",
        xy=(-0.18, 0.5),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )
    axes[1, 0].annotate(
        "Dysbiotic\nModel",
        xy=(-0.18, 0.5),
        xycoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )

    # Add "A" label
    fig.text(0.02, 0.97, "A", fontsize=20, fontweight="bold", va="top")

    # Legends - placed outside the plots to the right
    commensal_patches = make_legend(COMMENSAL_SPECIES, COMMENSAL_COLORS)
    axes[0, 1].legend(
        handles=commensal_patches,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        edgecolor="gray",
        title="Commensal",
        title_fontsize=10,
    )

    dysbiotic_patches = make_legend(DYSBIOTIC_SPECIES, DYSBIOTIC_COLORS)
    axes[1, 1].legend(
        handles=dysbiotic_patches,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        edgecolor="gray",
        title="Dysbiotic",
        title_fontsize=10,
    )

    plt.tight_layout(rect=[0.06, 0.02, 1, 0.98])
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    fig_path = (
        "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_fig/fig3_reproduced.png"
    )
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
