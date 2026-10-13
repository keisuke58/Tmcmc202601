"""
Extract data from Figure 2 (froh-06-1649419-g002) and reproduce the 8-panel figure.

Part A shows for each condition (Commensal/Dysbiotic) × cultivation (Static/HOBIC):
  - Left: Biofilm Volume boxplot (×10⁶ μm³/image) over time
  - Right: Stacked bar chart of Intact/Damaged Membrane Distribution (%)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

DAYS = [1, 3, 6, 10, 15, 21]

# ──────────────────────────────────────────────────────────────
# Biofilm Volume data read from figure boxplots
# Format: {day: (median, q1, q3, whisker_low, whisker_high, [fliers])}
# Units: ×10⁶ μm³/image
# ──────────────────────────────────────────────────────────────

volume_commensal_static = {
    1: (0.42, 0.32, 0.48, 0.18, 0.58, [0.90]),
    3: (0.30, 0.25, 0.36, 0.18, 0.50, [0.57]),
    6: (0.34, 0.28, 0.40, 0.20, 0.57, [0.64]),
    10: (0.30, 0.20, 0.41, 0.15, 0.58, []),
    15: (0.29, 0.22, 0.36, 0.10, 0.56, []),
    21: (0.28, 0.20, 0.36, 0.12, 0.52, []),
}

volume_commensal_hobic = {
    1: (0.50, 0.40, 0.60, 0.20, 0.80, [0.92]),
    3: (0.15, 0.10, 0.30, 0.05, 0.65, []),
    6: (0.20, 0.15, 0.30, 0.05, 0.45, []),
    10: (0.35, 0.25, 0.48, 0.15, 0.55, []),
    15: (0.38, 0.25, 0.52, 0.20, 0.70, []),
    21: (0.36, 0.28, 0.50, 0.18, 0.60, []),
}

volume_dysbiotic_static = {
    1: (0.40, 0.35, 0.45, 0.20, 0.55, []),
    3: (0.35, 0.30, 0.40, 0.25, 0.50, [0.75]),
    6: (0.38, 0.30, 0.55, 0.20, 0.78, []),
    10: (0.30, 0.10, 0.55, 0.05, 0.60, []),
    15: (0.60, 0.30, 0.80, 0.10, 0.98, []),
    21: (0.55, 0.10, 0.70, 0.05, 0.90, []),
}

volume_dysbiotic_hobic = {
    1: (0.45, 0.40, 0.55, 0.25, 0.60, []),
    3: (0.40, 0.30, 0.55, 0.20, 0.80, []),
    6: (0.75, 0.50, 1.00, 0.20, 1.30, []),
    10: (0.62, 0.52, 0.70, 0.40, 1.15, [1.16]),
    15: (0.60, 0.45, 0.75, 0.20, 0.85, []),
    21: (0.60, 0.40, 0.90, 0.05, 1.25, []),
}

# ──────────────────────────────────────────────────────────────
# Membrane distribution data (stacked bar)
# Format: {day: (intact_pct, damaged_pct, error)}
# ──────────────────────────────────────────────────────────────

membrane_commensal_static = {
    1: (66, 34, 8),
    3: (77, 23, 6),
    6: (65, 35, 7),
    10: (65, 35, 18),
    15: (69, 31, 19),
    21: (66, 34, 15),
}

membrane_commensal_hobic = {
    1: (92, 8, 5),
    3: (56, 44, 10),
    6: (60, 40, 12),
    10: (37, 63, 12),
    15: (33, 67, 15),
    21: (35, 65, 15),
}

membrane_dysbiotic_static = {
    1: (97, 3, 2),
    3: (85, 15, 10),
    6: (52, 48, 10),
    10: (57, 43, 8),
    15: (51, 49, 10),
    21: (47, 53, 8),
}

membrane_dysbiotic_hobic = {
    1: (81, 19, 10),
    3: (72, 28, 12),
    6: (39, 61, 15),
    10: (44, 56, 12),
    15: (48, 52, 12),
    21: (47, 53, 12),
}


def generate_replicates(median, q1, q3, wlo, whi, fliers, n=8):
    """Generate n synthetic replicate values from box plot statistics."""
    iqr = q3 - q1
    if iqr < 0.01:
        iqr = 0.01
    targets = [
        wlo,
        q1 - 0.15 * iqr,
        q1,
        median - 0.05 * iqr,
        median + 0.05 * iqr,
        q3,
        q3 + 0.15 * iqr,
        whi,
    ]
    samples = []
    for t in targets[:n]:
        noise = np.random.normal(0, iqr * 0.03)
        val = max(0, t + noise)
        samples.append(round(val, 4))
    # Add fliers as extra replicates
    for f in fliers:
        samples.append(round(f, 4))
    return samples


def build_dataframes():
    """Build DataFrames for biofilm volume and membrane distribution."""
    # ── Biofilm Volume (replicate-level) ──
    vol_rows = []
    vol_datasets = [
        ("Commensal", "Static", volume_commensal_static),
        ("Commensal", "HOBIC", volume_commensal_hobic),
        ("Dysbiotic", "Static", volume_dysbiotic_static),
        ("Dysbiotic", "HOBIC", volume_dysbiotic_hobic),
    ]
    for condition, cultivation, data in vol_datasets:
        for day in DAYS:
            median, q1, q3, wlo, whi, fliers = data[day]
            reps = generate_replicates(median, q1, q3, wlo, whi, fliers, n=8)
            for ri, val in enumerate(reps, 1):
                vol_rows.append(
                    {
                        "condition": condition,
                        "cultivation": cultivation,
                        "day": day,
                        "replicate": ri,
                        "biofilm_volume_x1e6": val,
                    }
                )
    df_vol = pd.DataFrame(vol_rows)

    # ── Membrane Distribution (mean + error) ──
    mem_rows = []
    mem_datasets = [
        ("Commensal", "Static", membrane_commensal_static),
        ("Commensal", "HOBIC", membrane_commensal_hobic),
        ("Dysbiotic", "Static", membrane_dysbiotic_static),
        ("Dysbiotic", "HOBIC", membrane_dysbiotic_hobic),
    ]
    for condition, cultivation, data in mem_datasets:
        for day in DAYS:
            intact, damaged, error = data[day]
            mem_rows.append(
                {
                    "condition": condition,
                    "cultivation": cultivation,
                    "day": day,
                    "intact_membrane_pct": intact,
                    "damaged_membrane_pct": damaged,
                    "error_pct": error,
                }
            )
    df_mem = pd.DataFrame(mem_rows)

    return df_vol, df_mem


def plot_volume_boxplot(ax, df_panel, title_suffix=""):
    """Plot biofilm volume boxplot for one condition+cultivation."""
    data_all = []
    positions = []
    # Gray gradient: lighter for early days, darker for later
    gray_shades = ["#ffffff", "#d9d9d9", "#bfbfbf", "#a0a0a0", "#808080", "#606060"]

    for di, day in enumerate(DAYS):
        subset = df_panel[df_panel["day"] == day]["biofilm_volume_x1e6"].values
        data_all.append(subset)
        positions.append(di)

    bp = ax.boxplot(
        data_all,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o", markersize=4, markerfacecolor="black", markeredgecolor="black", alpha=0.7
        ),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    for i, (patch, shade) in enumerate(zip(bp["boxes"], gray_shades)):
        patch.set_facecolor(shade)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(d) for d in DAYS], fontsize=10)
    ax.set_xlim(-0.6, len(DAYS) - 0.4)
    ax.set_ylim(-0.02, 1.55)
    ax.set_ylabel(r"Biofilm Volume" "\n" r"$\times 10^6$ [$\mu m^3$/image]", fontsize=10)
    ax.set_xlabel("Time [days]", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def plot_membrane_barplot(ax, df_panel):
    """Plot stacked bar chart for membrane distribution."""
    x = np.arange(len(DAYS))
    bar_width = 0.6

    intact_vals = []
    damaged_vals = []
    error_vals = []

    for day in DAYS:
        row = df_panel[df_panel["day"] == day].iloc[0]
        intact_vals.append(row["intact_membrane_pct"])
        damaged_vals.append(row["damaged_membrane_pct"])
        error_vals.append(row["error_pct"])

    intact_vals = np.array(intact_vals)
    damaged_vals = np.array(damaged_vals)
    error_vals = np.array(error_vals)

    # Intact (green) at bottom
    bars_intact = ax.bar(
        x,
        intact_vals,
        bar_width,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
        label="Intact Membrane",
    )
    # Damaged (red) on top
    bars_damaged = ax.bar(
        x,
        damaged_vals,
        bar_width,
        bottom=intact_vals,
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        label="Damaged Membrane",
    )

    # Error bars at the boundary (intact/damaged interface)
    ax.errorbar(
        x,
        intact_vals,
        yerr=error_vals,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DAYS], fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Distribution [%]", fontsize=10)
    ax.set_xlabel("Time [days]", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)


def main():
    df_vol, df_mem = build_dataframes()

    # ── Save CSVs ──
    base_path = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"

    vol_csv = f"{base_path}/fig2_biofilm_volume_replicates.csv"
    df_vol.to_csv(vol_csv, index=False)
    print(f"Saved volume CSV: {vol_csv}  ({df_vol.shape})")

    mem_csv = f"{base_path}/fig2_membrane_distribution.csv"
    df_mem.to_csv(mem_csv, index=False)
    print(f"Saved membrane CSV: {mem_csv}  ({df_mem.shape})")

    # Combined CSV
    combined_csv = f"{base_path}/fig2_combined_data.csv"
    # Merge volume summary with membrane data
    vol_summary = (
        df_vol.groupby(["condition", "cultivation", "day"])["biofilm_volume_x1e6"]
        .agg(["median", "mean", "std", "min", "max"])
        .reset_index()
    )
    vol_summary.columns = [
        "condition",
        "cultivation",
        "day",
        "volume_median",
        "volume_mean",
        "volume_std",
        "volume_min",
        "volume_max",
    ]
    combined = pd.merge(vol_summary, df_mem, on=["condition", "cultivation", "day"])
    combined.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV: {combined_csv}  ({combined.shape})")

    # ── Plot ──
    fig = plt.figure(figsize=(20, 11))

    # Use GridSpec: 2 rows × 4 columns
    # Each row: [boxplot | barplot | boxplot | barplot]
    # Boxplots slightly wider than barplots
    gs = gridspec.GridSpec(
        2, 4, figure=fig, width_ratios=[1.1, 0.9, 1.1, 0.9], hspace=0.38, wspace=0.35
    )

    # Panel definitions: (row, col, type, condition, cultivation)
    panels = [
        # Row 0: Commensal
        (0, 0, "boxplot", "Commensal", "Static"),
        (0, 1, "barplot", "Commensal", "Static"),
        (0, 2, "boxplot", "Commensal", "HOBIC"),
        (0, 3, "barplot", "Commensal", "HOBIC"),
        # Row 1: Dysbiotic
        (1, 0, "boxplot", "Dysbiotic", "Static"),
        (1, 1, "barplot", "Dysbiotic", "Static"),
        (1, 2, "boxplot", "Dysbiotic", "HOBIC"),
        (1, 3, "barplot", "Dysbiotic", "HOBIC"),
    ]

    for row, col, ptype, condition, cultivation in panels:
        ax = fig.add_subplot(gs[row, col])

        if ptype == "boxplot":
            df_panel = df_vol[
                (df_vol["condition"] == condition) & (df_vol["cultivation"] == cultivation)
            ]
            plot_volume_boxplot(ax, df_panel)
        else:
            df_panel = df_mem[
                (df_mem["condition"] == condition) & (df_mem["cultivation"] == cultivation)
            ]
            plot_membrane_barplot(ax, df_panel)

            # Add legend to rightmost barplots
            if col == 3:
                ax.legend(loc="center right", fontsize=7, framealpha=0.9, edgecolor="gray")

    # Column titles
    fig.text(
        0.27, 0.97, "Static Cultivation", fontsize=15, fontweight="bold", ha="center", va="top"
    )
    fig.text(0.75, 0.97, "HOBIC Cultivation", fontsize=15, fontweight="bold", ha="center", va="top")

    # Row labels
    fig.text(
        0.02,
        0.73,
        "Commensal\nModel",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )
    fig.text(
        0.02,
        0.30,
        "Dysbiotic\nModel",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )

    # Panel label
    fig.text(0.03, 0.97, "A", fontsize=20, fontweight="bold", va="top")

    fig_path = (
        "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_fig/fig2_reproduced.png"
    )
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
