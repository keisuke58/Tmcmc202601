"""
Multiple visualization styles for expected species volumes (Fig2 × Fig3).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

BASE = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"
FIG_DIR = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"

SPECIES_DISPLAY = [
    "S. oralis", "A. naeslundii", "V. dispar/parvula", "F. nucleatum", "P. gingivalis"
]
SPECIES_SHORT = ["S.o.", "A.n.", "V.d/p.", "F.n.", "P.g."]

SPECIES_COLORS = {
    "S. oralis":        "#2196F3",
    "A. naeslundii":    "#43A047",
    "V. dispar/parvula": "#FF9800",
    "F. nucleatum":     "#7B1FA2",
    "P. gingivalis":    "#E53935",
}

CONDITIONS = [
    ("Commensal", "Static"),
    ("Commensal", "HOBIC"),
    ("Dysbiotic", "Static"),
    ("Dysbiotic", "HOBIC"),
]
COND_LABELS = ["Com. Static", "Com. HOBIC", "Dys. Static", "Dys. HOBIC"]
DAYS = [1, 3, 6, 10, 15, 21]


def load_data():
    return pd.read_csv(f"{BASE}/expected_species_volumes.csv")


# ══════════════════════════════════════════════════════════════
# 1. Heatmap: species × day for each condition
# ══════════════════════════════════════════════════════════════

def plot_heatmaps(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Species Volume Heatmap [×10⁶ μm³]", fontsize=15, fontweight='bold', y=0.99)

    vmax = df["species_volume_x1e6"].max() * 1.05

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx // 2][idx % 2]
        sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]
        pivot = sub.pivot_table(index="species", columns="day",
                                values="species_volume_x1e6", aggfunc="first")
        pivot = pivot.reindex(SPECIES_DISPLAY)

        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(DAYS)))
        ax.set_xticklabels([f"Day {d}" for d in DAYS], fontsize=9)
        ax.set_yticks(range(len(SPECIES_DISPLAY)))
        ax.set_yticklabels([f"  {s}" for s in SPECIES_SHORT], fontsize=10, fontstyle='italic')
        ax.set_title(f"{cond} — {cult}", fontsize=12, fontweight='bold')

        # Annotate cells
        for i in range(len(SPECIES_DISPLAY)):
            for j in range(len(DAYS)):
                val = pivot.values[i, j]
                color = 'white' if val > vmax * 0.5 else 'black'
                ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Volume [×10⁶ μm³]", pad=0.02)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    path = f"{FIG_DIR}/expected_heatmap.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 2. Small multiples: one row per species, columns = conditions
# ══════════════════════════════════════════════════════════════

def plot_small_multiples(df):
    fig, axes = plt.subplots(5, 4, figsize=(18, 16), sharex=True)
    fig.suptitle("Species Volume Trajectories — Each Species × Each Condition",
                 fontsize=14, fontweight='bold', y=0.99)

    for i, sp in enumerate(SPECIES_DISPLAY):
        for j, (cond, cult) in enumerate(CONDITIONS):
            ax = axes[i][j]
            sub = df[(df["condition"] == cond) & (df["cultivation"] == cult) &
                     (df["species"] == sp)].sort_values("day")

            color = SPECIES_COLORS[sp]
            ax.fill_between(sub["day"], 0, sub["species_volume_x1e6"],
                            color=color, alpha=0.25)
            ax.plot(sub["day"], sub["species_volume_x1e6"], 'o-',
                    color=color, linewidth=2, markersize=5)

            # Also plot total volume (gray dashed)
            ax.plot(sub["day"], sub["total_volume_median_x1e6"], '--',
                    color='gray', linewidth=1, alpha=0.5)

            ax.set_ylim(bottom=0)
            ax.set_xlim(0, 22)
            ax.set_xticks(DAYS)
            ax.grid(True, alpha=0.15, linestyle='--')
            ax.set_axisbelow(True)

            if i == 0:
                ax.set_title(COND_LABELS[j], fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(SPECIES_SHORT[i], fontsize=11, fontstyle='italic',
                              fontweight='bold', rotation=0, labelpad=35, va='center')
            if i == 4:
                ax.set_xlabel("Day", fontsize=10)

    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    path = f"{FIG_DIR}/expected_small_multiples.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 3. 100% Stacked bars: composition shift over time
# ══════════════════════════════════════════════════════════════

def plot_normalized_stacked(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Species Composition Shift (100% Stacked)  —  Fraction of Total Volume",
                 fontsize=14, fontweight='bold', y=0.99)

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx // 2][idx % 2]
        sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]

        days = sorted(sub["day"].unique())
        bar_w = 1.6

        bottom = np.zeros(len(days))
        for sp in SPECIES_DISPLAY:
            sp_data = sub[sub["species"] == sp].set_index("day")
            fracs = [sp_data.loc[d, "species_fraction_median"] if d in sp_data.index else 0
                     for d in days]
            # Normalize to sum=1
            fracs = np.array(fracs)

            ax.bar(days, fracs * 100, bottom=bottom * 100, width=bar_w,
                   color=SPECIES_COLORS[sp], edgecolor='white', linewidth=0.5,
                   alpha=0.85, label=sp)

            # Labels for fractions > 8%
            for j, (d, f) in enumerate(zip(days, fracs)):
                if f > 0.08:
                    ax.text(d, (bottom[j] + f / 2) * 100, f"{f*100:.0f}%",
                            ha='center', va='center', fontsize=7,
                            fontweight='bold', color='white')
            bottom += fracs

        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Species Composition [%]", fontsize=11)
        ax.set_title(f"{cond} — {cult}", fontsize=12, fontweight='bold')
        ax.set_xticks(days)
        ax.set_xlim(-1, 23)
        ax.set_ylim(0, 105)

        if idx == 0:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{FIG_DIR}/expected_composition_100pct.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 4. Radar / Spider chart: species profile per condition at key days
# ══════════════════════════════════════════════════════════════

def plot_radar(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                              subplot_kw=dict(polar=True))
    fig.suptitle("Species Volume Radar — Day 1 vs Day 21",
                 fontsize=14, fontweight='bold', y=1.0)

    n_sp = len(SPECIES_DISPLAY)
    angles = np.linspace(0, 2 * np.pi, n_sp, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    day_styles = {1: ('--', 'o', 0.5), 21: ('-', 's', 0.9)}
    day_colors = {1: '#90CAF9', 21: '#1565C0'}

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx // 2][idx % 2]
        sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]

        for day, (ls, marker, alpha) in day_styles.items():
            sp_data = sub[sub["day"] == day].set_index("species")
            values = [sp_data.loc[sp, "species_volume_x1e6"] if sp in sp_data.index else 0
                      for sp in SPECIES_DISPLAY]
            values += values[:1]

            ax.fill(angles, values, alpha=0.15, color=day_colors[day])
            ax.plot(angles, values, ls, marker=marker, color=day_colors[day],
                    linewidth=2, markersize=6, alpha=alpha, label=f"Day {day}")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(SPECIES_SHORT, fontsize=10, fontweight='bold')
        ax.set_title(f"{cond} — {cult}", fontsize=12, fontweight='bold', pad=20)
        ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{FIG_DIR}/expected_radar.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 5. Bubble chart: day × species, size = volume, color = condition
# ══════════════════════════════════════════════════════════════

def plot_bubble(df):
    fig, ax = plt.subplots(figsize=(18, 8))

    cond_colors = {
        ("Commensal", "Static"): "#1f77b4",
        ("Commensal", "HOBIC"):  "#aec7e8",
        ("Dysbiotic", "Static"): "#ff7f0e",
        ("Dysbiotic", "HOBIC"):  "#ffbb78",
    }
    cond_offsets = {
        ("Commensal", "Static"): -0.3,
        ("Commensal", "HOBIC"):  -0.1,
        ("Dysbiotic", "Static"):  0.1,
        ("Dysbiotic", "HOBIC"):   0.3,
    }

    max_vol = df["species_volume_x1e6"].max()

    for (cond, cult), color in cond_colors.items():
        sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]
        offset = cond_offsets[(cond, cult)]

        for _, row in sub.iterrows():
            sp_idx = SPECIES_DISPLAY.index(row["species"])
            x = row["day"] + offset
            y = sp_idx
            size = (row["species_volume_x1e6"] / max_vol) * 1500 + 20

            ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black',
                       linewidth=0.5, zorder=3)

    ax.set_yticks(range(len(SPECIES_DISPLAY)))
    ax.set_yticklabels([f"  {s}" for s in SPECIES_DISPLAY], fontsize=11, fontstyle='italic')
    ax.set_xticks(DAYS)
    ax.set_xticklabels([f"Day {d}" for d in DAYS], fontsize=11)
    ax.set_xlim(-1, 23)
    ax.set_ylim(-0.6, len(SPECIES_DISPLAY) - 0.4)
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_title("Species Volume Bubble Chart  —  Bubble Size ∝ Volume",
                 fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [plt.scatter([], [], s=80, c=c, edgecolors='black', linewidth=0.5,
                                    label=f"{cond} {cult}")
                       for (cond, cult), c in cond_colors.items()]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right',
              framealpha=0.9, title="Condition", title_fontsize=11)

    # Size legend
    for vol_label, vol_val in [(0.05, 0.05), (0.15, 0.15), (0.35, 0.35)]:
        size = (vol_val / max_vol) * 1500 + 20
        ax.scatter([], [], s=size, c='gray', alpha=0.4, edgecolors='black',
                   linewidth=0.5, label=f"{vol_val}")
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9,
              title="Condition / Size [×10⁶ μm³]", title_fontsize=9, ncol=2)

    fig.tight_layout()
    path = f"{FIG_DIR}/expected_bubble.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 6. Streamgraph: smooth stacked area per condition
# ══════════════════════════════════════════════════════════════

def plot_streamgraph(df):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Species Volume Streams Over Time",
                 fontsize=15, fontweight='bold', y=0.99)

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx]
        sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]
        days = sorted(sub["day"].unique())

        # Interpolate for smooth curve
        from scipy.interpolate import PchipInterpolator
        days_fine = np.linspace(min(days), max(days), 200)

        volumes_fine = {}
        for sp in SPECIES_DISPLAY:
            sp_data = sub[sub["species"] == sp].sort_values("day")
            y = sp_data["species_volume_x1e6"].values
            if len(y) == len(days):
                interp = PchipInterpolator(days, y)
                volumes_fine[sp] = np.maximum(0, interp(days_fine))
            else:
                volumes_fine[sp] = np.zeros_like(days_fine)

        # Stacked area (baseline = 0)
        y_stack = np.zeros_like(days_fine)
        for sp in SPECIES_DISPLAY:
            y_vals = volumes_fine[sp]
            ax.fill_between(days_fine, y_stack, y_stack + y_vals,
                            color=SPECIES_COLORS[sp], alpha=0.6, linewidth=0)
            ax.plot(days_fine, y_stack + y_vals, color=SPECIES_COLORS[sp],
                    linewidth=0.8, alpha=0.8)
            y_stack += y_vals

        # Overlay markers at actual data points
        for sp in SPECIES_DISPLAY:
            sp_data = sub[sub["species"] == sp].sort_values("day")
            ax.plot(sp_data["day"], sp_data["species_volume_x1e6"].cumsum() * 0
                    + sp_data["species_volume_x1e6"],  # dummy for markers
                    'o', color=SPECIES_COLORS[sp], markersize=4, alpha=0)

        ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=10)
        ax.set_title(f"{cond} — {cult}", fontsize=12, fontweight='bold', loc='left')
        ax.set_xlim(1, 21)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.15, linestyle='--')
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Day", fontsize=12)
    axes[-1].set_xticks(DAYS)

    # Shared legend
    legend_elements = [plt.Line2D([0], [0], color=SPECIES_COLORS[sp], lw=8, alpha=0.6, label=sp)
                       for sp in SPECIES_DISPLAY]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = f"{FIG_DIR}/expected_stream.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 7. Condition comparison: grouped bars per day (side-by-side)
# ══════════════════════════════════════════════════════════════

def plot_condition_comparison(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Condition Comparison at Each Timepoint  —  Grouped Bars",
                 fontsize=14, fontweight='bold', y=0.99)

    cond_colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    bar_w = 0.18

    for d_idx, day in enumerate(DAYS):
        ax = axes[d_idx // 3][d_idx % 3]
        sub_day = df[df["day"] == day]

        x = np.arange(len(SPECIES_DISPLAY))

        for c_idx, (cond, cult) in enumerate(CONDITIONS):
            sub = sub_day[(sub_day["condition"] == cond) & (sub_day["cultivation"] == cult)]
            vals = []
            for sp in SPECIES_DISPLAY:
                sp_row = sub[sub["species"] == sp]
                vals.append(sp_row["species_volume_x1e6"].values[0] if len(sp_row) > 0 else 0)

            offset = (c_idx - 1.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w * 0.9, color=cond_colors[c_idx],
                          edgecolor='white', linewidth=0.5, alpha=0.85,
                          label=COND_LABELS[c_idx])

        ax.set_xticks(x)
        ax.set_xticklabels(SPECIES_SHORT, fontsize=9, fontstyle='italic')
        ax.set_title(f"Day {day}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.15, linestyle='--')
        ax.set_axisbelow(True)

        if d_idx == 0:
            ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{FIG_DIR}/expected_condition_comparison.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 8. Slope chart: Day 1 → Day 21 change per species
# ══════════════════════════════════════════════════════════════

def plot_slope(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)
    fig.suptitle("Species Volume Change:  Day 1 → Day 21  (Slope Chart)",
                 fontsize=14, fontweight='bold', y=0.99)

    for idx, (cond, cult) in enumerate(CONDITIONS):
        ax = axes[idx]
        sub = df[(df["condition"] == cond) & (df["cultivation"] == cult)]

        for sp in SPECIES_DISPLAY:
            sp_data = sub[sub["species"] == sp]
            d1 = sp_data[sp_data["day"] == 1]["species_volume_x1e6"].values
            d21 = sp_data[sp_data["day"] == 21]["species_volume_x1e6"].values

            if len(d1) > 0 and len(d21) > 0:
                v1, v21 = d1[0], d21[0]
                color = SPECIES_COLORS[sp]
                ax.plot([0, 1], [v1, v21], 'o-', color=color, linewidth=2.5,
                        markersize=8, alpha=0.8)
                # Labels
                ax.text(-0.08, v1, f"{v1:.3f}", ha='right', va='center',
                        fontsize=8, color=color, fontweight='bold')
                ax.text(1.08, v21, f"{v21:.3f}", ha='left', va='center',
                        fontsize=8, color=color, fontweight='bold')

                # Arrow indicator
                change = v21 - v1
                symbol = "↑" if change > 0.005 else ("↓" if change < -0.005 else "→")
                ax.text(0.5, (v1 + v21) / 2 + 0.008, f"{symbol} {change:+.3f}",
                        ha='center', va='bottom', fontsize=7, color=color, alpha=0.7)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Day 1", "Day 21"], fontsize=11, fontweight='bold')
        ax.set_title(f"{cond}\n{cult}", fontsize=11, fontweight='bold')
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, axis='y', alpha=0.15, linestyle='--')
        ax.set_axisbelow(True)
        if idx == 0:
            ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=11)

    # Legend
    legend_elements = [Line2D([0], [0], color=SPECIES_COLORS[sp], lw=2.5,
                              marker='o', markersize=6, label=sp)
                       for sp in SPECIES_DISPLAY]
    fig.legend(handles=legend_elements, fontsize=9, loc='lower center',
               ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = f"{FIG_DIR}/expected_slope.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")

    plot_heatmaps(df)
    plot_small_multiples(df)
    plot_normalized_stacked(df)
    plot_radar(df)
    plot_bubble(df)
    plot_streamgraph(df)
    plot_condition_comparison(df)
    plot_slope(df)

    print("\nAll 8 visualizations saved!")


if __name__ == "__main__":
    main()
