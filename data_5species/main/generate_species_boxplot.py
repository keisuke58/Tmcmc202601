#!/usr/bin/env python3
"""
Generate Grouped Box Plots for Experimental Data.
Visualizes the species distribution side-by-side for each day.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path("../experiment_data")
OUTPUT_DIR = Path("../experiment_fig")
CSV_FILE = DATA_DIR / "species_distribution_data.csv"

# Colors
COLORS = {
    "Blue": "#1f77b4",  # S0
    "Green": "#2ca02c",  # S1
    "Yellow": "#bcbd22",  # S2 (Commensal)
    "Orange": "#ff7f0e",  # S2 (Dysbiotic)
    "Purple": "#9467bd",  # S3
    "Red": "#d62728",  # S4
}

SPECIES_LABELS = {
    "Blue": "S. oralis",
    "Green": "A. naeslundii",
    "Yellow": "V. dispar",
    "Orange": "V. parvula",
    "Purple": "F. nucleatum",
    "Red": "P. gingivalis",
}


def get_species_order(condition):
    if condition == "Dysbiotic":
        return ["Blue", "Green", "Orange", "Purple", "Red"]
    else:
        return ["Blue", "Green", "Yellow", "Purple", "Red"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found.")
        return

    df = pd.read_csv(CSV_FILE)

    # Get unique conditions
    conditions = df[["condition", "cultivation"]].drop_duplicates()

    sns.set_context("talk")
    sns.set_style("whitegrid")

    for _, row in conditions.iterrows():
        cond = row["condition"]
        cult = row["cultivation"]

        subset = df[(df["condition"] == cond) & (df["cultivation"] == cult)]

        fig, ax = plt.subplots(figsize=(12, 7))

        days = sorted(subset["day"].unique())
        species_order = get_species_order(cond)

        # Prepare data for bxp
        # We need to map Days to X-positions (0, 1, 2...) or use actual Day values?
        # Using evenly spaced X positions is better for readability if days are 1, 3, 6, 10...
        x_positions = range(len(days))

        # Width of each box
        box_width = 0.15
        total_width = box_width * 5
        start_offset = -total_width / 2 + box_width / 2

        for i, species_color in enumerate(species_order):
            stats_list = []

            for day_idx, day in enumerate(days):
                row_data = subset[(subset["day"] == day) & (subset["species"] == species_color)]

                if row_data.empty:
                    # Create dummy empty stats
                    stats = {
                        "med": 0,
                        "q1": 0,
                        "q3": 0,
                        "whislo": 0,
                        "whishi": 0,
                        "label": str(day) if i == 2 else "",  # Label only center box
                    }
                else:
                    # Extract stats (values are in %)
                    median = row_data["median"].values[0] / 100.0
                    iqr = row_data["iqr"].values[0] / 100.0
                    rng = row_data["range"].values[0] / 100.0

                    # Estimate box stats from Median, IQR, Range
                    # Since we don't have Q1/Q3 explicitly, we approximate:
                    # Q1 = Median - IQR/2
                    # Q3 = Median + IQR/2
                    # Whiskers = Median +/- Range/2 (Range represents max-min spread)

                    q1 = max(0, median - iqr / 2)
                    q3 = min(1, median + iqr / 2)
                    whislo = max(0, median - rng / 2)
                    whishi = min(1, median + rng / 2)

                    # Ensure logical consistency
                    if q1 < whislo:
                        whislo = q1
                    if q3 > whishi:
                        whishi = q3

                    stats = {
                        "med": median,
                        "q1": q1,
                        "q3": q3,
                        "whislo": whislo,
                        "whishi": whishi,
                        "label": str(day) if i == 2 else "",  # Only label the x-axis tick
                    }

                stats_list.append(stats)

            # Draw boxes for this species across all days
            # Calculate positions
            positions = [x + start_offset + i * box_width for x in x_positions]

            # Use ax.bxp
            # Note: ax.bxp does not support 'color' kwarg directly for whiskers/caps in the same way as boxplot
            # We need to set properties via the props dictionaries.

            box_props = dict(
                facecolor=COLORS[species_color], alpha=0.7, linewidth=1.5, edgecolor="black"
            )
            median_props = dict(color="black", linewidth=1.5)
            whisker_props = dict(color="black", linestyle="-", linewidth=1.0)
            capprops = dict(color="black", linewidth=1.0)

            # ax.bxp returns a dictionary mapping component names to lists of artists
            artists = ax.bxp(
                stats_list,
                positions=positions,
                widths=box_width,
                showfliers=False,
                patch_artist=True,
                boxprops=box_props,
                medianprops=median_props,
                whiskerprops=whisker_props,
                capprops=capprops,
            )

        # Customize X-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(days)
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Relative Abundance")
        ax.set_title(f"Target Distribution: {cond} - {cult}", fontsize=16, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

        # Legend
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[s], edgecolor="black", alpha=0.7)
            for s in species_order
        ]
        legend_labels = [SPECIES_LABELS[s] for s in species_order]
        ax.legend(
            legend_patches, legend_labels, loc="upper left", bbox_to_anchor=(1, 1), title="Species"
        )

        plt.tight_layout()

        filename = f"Boxplot_{cond}_{cult}.png"
        save_path = OUTPUT_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()


if __name__ == "__main__":
    main()
