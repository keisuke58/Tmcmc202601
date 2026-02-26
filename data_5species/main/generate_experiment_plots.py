#!/usr/bin/env python3
"""
Generate experimental data plots from species_distribution_data.csv.
Saves figures to ../experiment_fig/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path("../experiment_data")
OUTPUT_DIR = Path("../experiment_fig")
CSV_FILE = DATA_DIR / "species_distribution_data.csv"

SPECIES_MAP = {
    "Blue": "S. oralis",
    "Green": "A. naeslundii",
    "Yellow": "V. dispar",
    "Orange": "V. parvula",
    "Purple": "F. nucleatum",
    "Red": "P. gingivalis",
}

COLOR_MAP = {
    "Blue": "#1f77b4",
    "Green": "#2ca02c",
    "Yellow": "#bcbd22",
    "Orange": "#ff7f0e",
    "Purple": "#9467bd",
    "Red": "#d62728",
}


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found.")
        return

    # Load data
    df = pd.read_csv(CSV_FILE)

    # Get unique conditions
    conditions = df[["condition", "cultivation"]].drop_duplicates()

    sns.set_context("talk")
    sns.set_style("whitegrid")

    for _, row in conditions.iterrows():
        cond = row["condition"]
        cult = row["cultivation"]

        subset = df[(df["condition"] == cond) & (df["cultivation"] == cult)]

        plt.figure(figsize=(10, 6))

        # Determine active species for this condition
        # Commensal has Yellow, Dysbiotic has Orange
        # But let's just plot whatever is in the subset

        available_colors = subset["species"].unique()

        for color in available_colors:
            species_data = subset[subset["species"] == color]

            # Sort by day
            species_data = species_data.sort_values("day")

            days = species_data["day"]
            medians = species_data["median"] / 100.0  # Data is in %, convert to fraction

            # Approximate error bars from IQR or Range if needed,
            # but let's just plot lines/scatter for now as "Target"

            label = SPECIES_MAP.get(color, color)
            plot_color = COLOR_MAP.get(color, "gray")

            plt.plot(
                days, medians, marker="o", label=label, color=plot_color, linewidth=2, markersize=8
            )

        plt.title(f"Experimental Target: {cond} - {cult}")
        plt.xlabel("Time (Days)")
        plt.ylabel("Relative Abundance")
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        filename = f"Target_{cond}_{cult}.png"
        save_path = OUTPUT_DIR / filename
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")
        plt.close()

    print("All plots generated.")


if __name__ == "__main__":
    main()
