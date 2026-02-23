"""
Compute expected species-specific volumes from Fig 2 × Fig 3.

Expected species volume [×10⁶ μm³] = total biofilm volume (Fig 2) × species fraction (Fig 3)

Produces:
  - expected_species_volumes.csv  (all 4 conditions, median + IQR)
  - expected_species_volumes_figure.png  (4-panel stacked area + line plot)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"
FIG_DIR = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_fig"

# ══════════════════════════════════════════════════════════════
# Species display config
# ══════════════════════════════════════════════════════════════

SPECIES_ORDER = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
# Map fig3 names to standard names
SPECIES_MAP = {
    "S. oralis": "S. oralis",
    "A. naeslundii": "A. naeslundii",
    "V. dispar": "V. dispar/parvula",
    "V. parvula": "V. dispar/parvula",
    "F. nucleatum": "F. nucleatum",
    "P. gingivalis_20709": "P. gingivalis",
    "P. gingivalis_W83": "P. gingivalis",
}

SPECIES_COLORS = {
    "S. oralis":        "#2196F3",
    "A. naeslundii":    "#43A047",
    "V. dispar/parvula": "#FF9800",
    "F. nucleatum":     "#7B1FA2",
    "P. gingivalis":    "#E53935",
}

SPECIES_DISPLAY = [
    "S. oralis", "A. naeslundii", "V. dispar/parvula", "F. nucleatum", "P. gingivalis"
]

CONDITIONS = [
    ("Commensal", "Static"),
    ("Commensal", "HOBIC"),
    ("Dysbiotic", "Static"),
    ("Dysbiotic", "HOBIC"),
]

DAYS = [1, 3, 6, 10, 15, 21]


def load_data():
    """Load Fig 2 (biofilm volume) and Fig 3 (species distribution) data."""
    df_vol = pd.read_csv(f"{BASE}/fig2_biofilm_volume_replicates.csv")
    df_sp = pd.read_csv(f"{BASE}/fig3_species_distribution_summary.csv")
    return df_vol, df_sp


def compute_species_volumes(df_vol, df_sp):
    """Compute expected species volumes = total volume × species fraction."""
    rows = []

    for cond, cult in CONDITIONS:
        for day in DAYS:
            # Get median total biofilm volume for this condition/day
            vol_subset = df_vol[
                (df_vol["condition"] == cond) &
                (df_vol["cultivation"] == cult) &
                (df_vol["day"] == day)
            ]["biofilm_volume_x1e6"]

            if vol_subset.empty:
                continue

            vol_median = vol_subset.median()
            vol_q1 = vol_subset.quantile(0.25)
            vol_q3 = vol_subset.quantile(0.75)

            # Get species fractions for this condition/day
            sp_subset = df_sp[
                (df_sp["condition"] == cond) &
                (df_sp["cultivation"] == cult) &
                (df_sp["day"] == day)
            ]

            total_fraction = 0.0
            species_data = {}

            for _, sp_row in sp_subset.iterrows():
                raw_name = sp_row["species"]
                std_name = SPECIES_MAP.get(raw_name, raw_name)
                frac_median = sp_row["median"] / 100.0
                frac_q1 = sp_row["q1"] / 100.0
                frac_q3 = sp_row["q3"] / 100.0

                # Accumulate for species that map to same standard name
                if std_name not in species_data:
                    species_data[std_name] = {"frac": 0.0, "frac_q1": 0.0, "frac_q3": 0.0}
                species_data[std_name]["frac"] += frac_median
                species_data[std_name]["frac_q1"] += frac_q1
                species_data[std_name]["frac_q3"] += frac_q3

            for sp_name in SPECIES_DISPLAY:
                sd = species_data.get(sp_name, {"frac": 0, "frac_q1": 0, "frac_q3": 0})
                vol_sp = vol_median * sd["frac"]
                vol_sp_lo = vol_q1 * sd["frac_q1"]
                vol_sp_hi = vol_q3 * sd["frac_q3"]

                rows.append({
                    "condition": cond,
                    "cultivation": cult,
                    "day": day,
                    "species": sp_name,
                    "total_volume_median_x1e6": round(vol_median, 4),
                    "species_fraction_median": round(sd["frac"], 4),
                    "species_volume_x1e6": round(vol_sp, 6),
                    "species_volume_lo_x1e6": round(vol_sp_lo, 6),
                    "species_volume_hi_x1e6": round(vol_sp_hi, 6),
                })

    return pd.DataFrame(rows)


def plot_expected_results(df):
    """Create 4-panel figure showing expected species volumes over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
    fig.suptitle(
        "Expected Species Volumes  =  Fig 2 (Total Biofilm Volume)  ×  Fig 3 (Species Distribution)",
        fontsize=14, fontweight='bold', y=0.98
    )

    panel_labels = ["A", "B", "C", "D"]

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx // 2][idx % 2]
        df_sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]

        # Stacked area plot
        days = sorted(df_sub["day"].unique())

        # Build arrays for stacked area
        volumes = {}
        for sp in SPECIES_DISPLAY:
            sp_data = df_sub[df_sub["species"] == sp].set_index("day")
            volumes[sp] = [sp_data.loc[d, "species_volume_x1e6"] if d in sp_data.index else 0
                           for d in days]

        # Stacked area
        y_stack = np.zeros(len(days))
        for sp in SPECIES_DISPLAY:
            y_vals = np.array(volumes[sp])
            ax.fill_between(days, y_stack, y_stack + y_vals,
                            color=SPECIES_COLORS[sp], alpha=0.4, linewidth=0)
            ax.plot(days, y_stack + y_vals, color=SPECIES_COLORS[sp],
                    linewidth=1.5, alpha=0.8)
            y_stack += y_vals

        # Individual species lines with markers
        for sp in SPECIES_DISPLAY:
            y_vals = np.array(volumes[sp])
            ax.plot(days, y_vals, 'o-', color=SPECIES_COLORS[sp],
                    linewidth=2.0, markersize=6, label=sp, alpha=0.9)

        # Total volume line (dashed)
        total_per_day = []
        for d in days:
            dd = df_sub[df_sub["day"] == d]
            total_per_day.append(dd["total_volume_median_x1e6"].iloc[0] if len(dd) > 0 else 0)
        ax.plot(days, total_per_day, 'k--', linewidth=2.0, alpha=0.5, label="Total biofilm vol.")

        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=11)
        ax.set_title(f"{cond} — {cult}", fontsize=13, fontweight='bold')
        ax.set_xticks(days)
        ax.set_xlim(0.5, 21.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)

        ax.text(-0.08, 1.05, panel_labels[idx], fontsize=16, fontweight='bold',
                transform=ax.transAxes, va='top')

        if idx == 0:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{FIG_DIR}/expected_species_volumes.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {path}")
    plt.close()


def plot_detailed_panels(df):
    """Create detailed 4-panel figure: each condition gets stacked bars + line overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Expected Species Volumes (Stacked Bars)  —  Data = Fig 2 × Fig 3",
        fontsize=14, fontweight='bold', y=0.98
    )

    panel_labels = ["A", "B", "C", "D"]
    bar_width = 1.8

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx // 2][idx % 2]
        df_sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]
        days = sorted(df_sub["day"].unique())

        # Build arrays
        volumes = {}
        for sp in SPECIES_DISPLAY:
            sp_data = df_sub[df_sub["species"] == sp].set_index("day")
            volumes[sp] = [sp_data.loc[d, "species_volume_x1e6"] if d in sp_data.index else 0
                           for d in days]

        # Stacked bars
        bottom = np.zeros(len(days))
        for sp in SPECIES_DISPLAY:
            y_vals = np.array(volumes[sp])
            ax.bar(days, y_vals, bottom=bottom, width=bar_width,
                   color=SPECIES_COLORS[sp], edgecolor='white', linewidth=0.5,
                   alpha=0.85, label=sp)
            # Add value labels for significant contributions
            for j, (d, v) in enumerate(zip(days, y_vals)):
                if v > 0.02:
                    ax.text(d, bottom[j] + v/2, f"{v:.3f}",
                            ha='center', va='center', fontsize=5.5,
                            fontweight='bold', color='white')
            bottom += y_vals

        # Total line
        total_per_day = []
        for d in days:
            dd = df_sub[df_sub["day"] == d]
            total_per_day.append(dd["total_volume_median_x1e6"].iloc[0] if len(dd) > 0 else 0)

        ax.plot(days, total_per_day, 'k^-', linewidth=1.5, markersize=7,
                alpha=0.7, label="Total volume (Fig 2)", zorder=5)

        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=11)
        ax.set_title(f"{cond} — {cult}", fontsize=13, fontweight='bold')
        ax.set_xticks(days)
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)

        ax.text(-0.08, 1.05, panel_labels[idx], fontsize=16, fontweight='bold',
                transform=ax.transAxes, va='top')

        if idx == 0:
            ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{FIG_DIR}/expected_species_volumes_stacked.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {path}")
    plt.close()


def main():
    df_vol, df_sp = load_data()

    # Compute expected species volumes
    df_expected = compute_species_volumes(df_vol, df_sp)

    # Save CSV
    csv_path = f"{BASE}/expected_species_volumes.csv"
    df_expected.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}  ({df_expected.shape})")

    # Print summary table
    print("\n=== Expected Species Volumes [×10⁶ μm³] ===\n")
    for cond, cult in CONDITIONS:
        print(f"--- {cond} / {cult} ---")
        sub = df_expected[(df_expected["condition"] == cond) & (df_expected["cultivation"] == cult)]
        pivot = sub.pivot_table(index="day", columns="species",
                                values="species_volume_x1e6", aggfunc="first")
        pivot = pivot[SPECIES_DISPLAY]
        pivot["TOTAL"] = pivot.sum(axis=1)
        # Also show the original total volume
        total_vol = sub.drop_duplicates(["day"])["total_volume_median_x1e6"].values
        pivot["Fig2_total"] = total_vol
        print(pivot.round(4).to_string())
        print()

    # Compare with current TMCMC target data (fractions)
    print("=== Verification: Fractions sum to ~1.0 ===")
    for cond, cult in CONDITIONS:
        sub = df_expected[(df_expected["condition"] == cond) & (df_expected["cultivation"] == cult)]
        pivot = sub.pivot_table(index="day", columns="species",
                                values="species_fraction_median", aggfunc="first")
        sums = pivot.sum(axis=1)
        print(f"  {cond}/{cult}: fraction sums = {sums.values.round(4)}")

    # Plot
    plot_expected_results(df_expected)
    plot_detailed_panels(df_expected)

    print("\nDone!")


if __name__ == "__main__":
    main()
