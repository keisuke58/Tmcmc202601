"""
High-quality reproduction of Figure 4C: Metabolic Interaction Network
from Heine et al. (2025) Front. Oral Health 6:1649419

Species arranged in pentagon layout with manually positioned metabolites,
color-coded arrows per species, culture medium box, and legend.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import math

FIG_DIR = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_fig"

# ══════════════════════════════════════════════════════════════
# Node positions (manually arranged to match paper layout)
# ══════════════════════════════════════════════════════════════

# Species positions (pentagon, matching paper arrangement)
SPECIES_POS = {
    "Streptococcus\noralis":       (0.50, 0.88),  # top center
    "Porphyromonas\ngingivalis":   (0.82, 0.58),  # right
    "Actinomyces\nnaeslundii":     (0.70, 0.15),  # bottom right
    "Fusobacterium\nnucleatum":    (0.28, 0.15),  # bottom left
    "Veillonella dispar /\nparvula": (0.15, 0.58),  # left
}

# Species colors matching paper
SPECIES_COLORS = {
    "Streptococcus\noralis":       "#2196F3",  # blue
    "Porphyromonas\ngingivalis":   "#E53935",  # red
    "Actinomyces\nnaeslundii":     "#43A047",  # green
    "Fusobacterium\nnucleatum":    "#7B1FA2",  # purple
    "Veillonella dispar /\nparvula": "#FF9800",  # orange
}

# Arrow colors per species (for metabolic arrows)
ARROW_COLORS = {
    "So": "#29B6F6",   # S. oralis - cyan/light blue
    "An": "#66BB6A",   # A. naeslundii - green
    "Fn": "#AB47BC",   # F. nucleatum - purple
    "Pg": "#EF5350",   # P. gingivalis - red/pink
    "Vd": "#FFA726",   # V. dispar/parvula - orange
}

# Metabolite positions (manually placed in center area, spread out)
METABOLITE_POS = {
    "Lactate":          (0.47, 0.44),
    "Succinate":        (0.41, 0.37),
    "Acetate":          (0.37, 0.30),
    "Formate":          (0.32, 0.25),
    "Propionate":       (0.36, 0.21),
    "Butyrate":         (0.31, 0.18),
    "Carbon\ndioxide":  (0.38, 0.65),
    "Hydrogen":         (0.27, 0.73),
    "Hydrogen\nsulfide": (0.21, 0.46),
    "Indole":           (0.25, 0.36),
    "Glucose":          (0.43, 0.55),
    "Carbohydrates":    (0.50, 0.35),
    "(Glyco)\nProteins": (0.56, 0.71),
    "Amino\nacids":     (0.62, 0.29),
    "Peptides":         (0.60, 0.45),
    "Vitamins":         (0.28, 0.54),
    "Other growth\nfactors": (0.19, 0.30),
}

# Enzyme positions (near producing species)
ENZYME_POS = {
    "Phosphatase":  (0.36, 0.78),   # S. oralis enzyme
    "DNase":        (0.53, 0.11),   # A. naeslundii enzyme
    "Protease":     (0.76, 0.47),   # P. gingivalis enzymes
    "Peptidases":   (0.70, 0.36),
    "Glycosidases": (0.73, 0.67),
}

# ══════════════════════════════════════════════════════════════
# Interaction definitions
# ══════════════════════════════════════════════════════════════

# Format: (species_key, metabolite/enzyme, direction)
# direction: "produces" = species -> metabolite, "consumes" = metabolite -> species

INTERACTIONS = [
    # S. oralis (cyan)
    ("So", "Streptococcus\noralis", "Lactate", "produces"),
    ("So", "Streptococcus\noralis", "Carbon\ndioxide", "produces"),
    ("So", "Streptococcus\noralis", "Hydrogen", "produces"),
    ("So", "Streptococcus\noralis", "Acetate", "produces"),
    ("So", "Streptococcus\noralis", "Formate", "produces"),
    ("So", "Streptococcus\noralis", "Succinate", "produces"),
    ("So", "Streptococcus\noralis", "Phosphatase", "produces"),
    ("So", "Glucose", "Streptococcus\noralis", "consumes"),
    ("So", "Carbohydrates", "Streptococcus\noralis", "consumes"),
    ("So", "(Glyco)\nProteins", "Streptococcus\noralis", "consumes"),

    # A. naeslundii (green)
    ("An", "Actinomyces\nnaeslundii", "Lactate", "produces"),
    ("An", "Actinomyces\nnaeslundii", "Succinate", "produces"),
    ("An", "Actinomyces\nnaeslundii", "Acetate", "produces"),
    ("An", "Actinomyces\nnaeslundii", "Formate", "produces"),
    ("An", "Actinomyces\nnaeslundii", "Carbon\ndioxide", "produces"),
    ("An", "Actinomyces\nnaeslundii", "DNase", "produces"),
    ("An", "Glucose", "Actinomyces\nnaeslundii", "consumes"),
    ("An", "Amino\nacids", "Actinomyces\nnaeslundii", "consumes"),
    ("An", "Vitamins", "Actinomyces\nnaeslundii", "consumes"),

    # F. nucleatum (purple)
    ("Fn", "Fusobacterium\nnucleatum", "Butyrate", "produces"),
    ("Fn", "Fusobacterium\nnucleatum", "Propionate", "produces"),
    ("Fn", "Fusobacterium\nnucleatum", "Acetate", "produces"),
    ("Fn", "Fusobacterium\nnucleatum", "Hydrogen\nsulfide", "produces"),
    ("Fn", "Fusobacterium\nnucleatum", "Indole", "produces"),
    ("Fn", "Fusobacterium\nnucleatum", "Carbon\ndioxide", "produces"),
    ("Fn", "Amino\nacids", "Fusobacterium\nnucleatum", "consumes"),
    ("Fn", "Peptides", "Fusobacterium\nnucleatum", "consumes"),
    ("Fn", "Glucose", "Fusobacterium\nnucleatum", "consumes"),
    ("Fn", "Other growth\nfactors", "Fusobacterium\nnucleatum", "consumes"),

    # P. gingivalis (red)
    ("Pg", "Porphyromonas\ngingivalis", "Butyrate", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Propionate", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Acetate", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Hydrogen\nsulfide", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Indole", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Protease", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Peptidases", "produces"),
    ("Pg", "Porphyromonas\ngingivalis", "Glycosidases", "produces"),
    ("Pg", "Peptides", "Porphyromonas\ngingivalis", "consumes"),
    ("Pg", "Amino\nacids", "Porphyromonas\ngingivalis", "consumes"),
    ("Pg", "Carbohydrates", "Porphyromonas\ngingivalis", "consumes"),
    ("Pg", "(Glyco)\nProteins", "Porphyromonas\ngingivalis", "consumes"),

    # V. dispar/parvula (orange)
    ("Vd", "Veillonella dispar /\nparvula", "Propionate", "produces"),
    ("Vd", "Veillonella dispar /\nparvula", "Acetate", "produces"),
    ("Vd", "Veillonella dispar /\nparvula", "Carbon\ndioxide", "produces"),
    ("Vd", "Veillonella dispar /\nparvula", "Hydrogen", "produces"),
    ("Vd", "Lactate", "Veillonella dispar /\nparvula", "consumes"),
    ("Vd", "Succinate", "Veillonella dispar /\nparvula", "consumes"),
    ("Vd", "Other growth\nfactors", "Veillonella dispar /\nparvula", "consumes"),
    ("Vd", "Vitamins", "Veillonella dispar /\nparvula", "consumes"),
]


def draw_curved_arrow(ax, start, end, color, lw=1.0, alpha=0.75,
                      connectionstyle="arc3,rad=0.08", shrinkA=18, shrinkB=10):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        connectionstyle=connectionstyle,
        color=color,
        lw=lw,
        alpha=alpha,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        mutation_scale=12,
    )
    ax.add_patch(arrow)


def plot_fig4c():
    """Create high-quality reproduction of Figure 4C."""
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── 1. Draw arrows (connections) first (behind nodes) ──
    # Group arrows by source-target to calculate offset for parallel edges
    edge_count = {}
    for sp_key, src, tgt, direction in INTERACTIONS:
        pair = tuple(sorted([src, tgt]))
        edge_count[pair] = edge_count.get(pair, 0) + 1

    edge_drawn = {}
    for sp_key, src, tgt, direction in INTERACTIONS:
        color = ARROW_COLORS[sp_key]

        if direction == "produces":
            start_node, end_node = src, tgt
        else:
            start_node, end_node = src, tgt

        # Get positions
        all_pos = {**SPECIES_POS, **METABOLITE_POS, **ENZYME_POS}
        if start_node not in all_pos or end_node not in all_pos:
            continue

        start_pos = all_pos[start_node]
        end_pos = all_pos[end_node]

        # Determine if species or metabolite for shrink values
        is_species_start = start_node in SPECIES_POS
        is_species_end = end_node in SPECIES_POS
        shrinkA = 20 if is_species_start else 8
        shrinkB = 20 if is_species_end else 8

        # Vary curvature slightly for parallel edges
        pair = tuple(sorted([start_node, end_node]))
        idx = edge_drawn.get(pair, 0)
        edge_drawn[pair] = idx + 1
        rad = 0.06 + idx * 0.04
        if idx % 2 == 1:
            rad = -rad

        draw_curved_arrow(
            ax, start_pos, end_pos, color,
            lw=1.2, alpha=0.6,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrinkA, shrinkB=shrinkB,
        )

    # ── 2. Draw species nodes (large colored circles) ──
    # Per-species label offsets to avoid collisions
    label_config = {
        "Streptococcus\noralis":       {"dx": 0, "dy": 0.05, "ha": "center", "va": "bottom"},
        "Porphyromonas\ngingivalis":   {"dx": 0.05, "dy": -0.02, "ha": "left", "va": "top"},
        "Actinomyces\nnaeslundii":     {"dx": 0.04, "dy": -0.05, "ha": "center", "va": "top"},
        "Fusobacterium\nnucleatum":    {"dx": -0.04, "dy": -0.05, "ha": "center", "va": "top"},
        "Veillonella dispar /\nparvula": {"dx": -0.05, "dy": -0.04, "ha": "right", "va": "top"},
    }
    for name, pos in SPECIES_POS.items():
        color = SPECIES_COLORS[name]
        circle = plt.Circle(pos, 0.036, color=color, ec='black', lw=1.5, zorder=10)
        ax.add_patch(circle)
        cfg = label_config[name]
        ax.text(pos[0] + cfg["dx"], pos[1] + cfg["dy"], name,
                ha=cfg["ha"], va=cfg["va"],
                fontsize=9, fontstyle='italic', fontweight='bold', zorder=11)

    # ── 3. Draw metabolite nodes (small open squares) ──
    sq_size = 0.013
    for name, pos in METABOLITE_POS.items():
        rect = plt.Rectangle(
            (pos[0] - sq_size/2, pos[1] - sq_size/2),
            sq_size, sq_size,
            fill=True, facecolor='white', edgecolor='black', lw=0.8, zorder=10
        )
        ax.add_patch(rect)
        # Label
        ax.text(pos[0], pos[1] - 0.02, name, ha='center', va='top',
                fontsize=7, zorder=11)

    # ── 4. Draw enzyme nodes (small filled diamonds) ──
    diamond_size = 0.008
    for name, pos in ENZYME_POS.items():
        # Diamond = rotated square
        diamond = plt.Polygon([
            (pos[0], pos[1] + diamond_size),
            (pos[0] + diamond_size, pos[1]),
            (pos[0], pos[1] - diamond_size),
            (pos[0] - diamond_size, pos[1]),
        ], closed=True, facecolor='black', edgecolor='black', lw=0.8, zorder=10)
        ax.add_patch(diamond)
        ax.text(pos[0], pos[1] - 0.016, name, ha='center', va='top',
                fontsize=7, fontstyle='italic', zorder=11)

    # ── 5. V. dispar / V. parvula inset (top-left) ──
    # Show separate Veillonella species with Thiamine connection
    inset_x, inset_y = 0.04, 0.92
    # V. dispar circle
    c1 = plt.Circle((inset_x + 0.02, inset_y), 0.018, color='#FF9800',
                     ec='black', lw=1.0, zorder=10)
    ax.add_patch(c1)
    ax.text(inset_x + 0.045, inset_y, "Veillonella\ndispar", ha='left', va='center',
            fontsize=7, fontstyle='italic', zorder=11)

    # V. parvula circle (below)
    c2 = plt.Circle((inset_x + 0.02, inset_y - 0.05), 0.018, color='#FF9800',
                     ec='black', lw=1.0, zorder=10)
    ax.add_patch(c2)
    ax.text(inset_x + 0.045, inset_y - 0.05, "Veillonella\nparvula", ha='left', va='center',
            fontsize=7, fontstyle='italic', zorder=11)

    # Thiamine box
    thiamine_pos = (inset_x + 0.16, inset_y - 0.025)
    rect_t = plt.Rectangle(
        (thiamine_pos[0] - 0.008, thiamine_pos[1] - 0.008),
        0.016, 0.016,
        fill=True, facecolor='white', edgecolor='black', lw=0.8, zorder=10
    )
    ax.add_patch(rect_t)
    ax.text(thiamine_pos[0], thiamine_pos[1] - 0.02, "Thiamine\n(Vitamin B1)",
            ha='center', va='top', fontsize=7, zorder=11)

    # Arrow from V. parvula to Thiamine
    draw_curved_arrow(ax,
                      (inset_x + 0.038, inset_y - 0.05),
                      (thiamine_pos[0] - 0.01, thiamine_pos[1]),
                      '#FFA726', lw=1.0, alpha=0.8,
                      connectionstyle="arc3,rad=0.05",
                      shrinkA=5, shrinkB=5)

    # ── 6. Culture medium box (right side) ──
    medium_components = [
        "Albumin", r"$\alpha$-Amylase", "Beef heart infusion solids",
        "Brain infusion solids", "Carbon dioxide", "Di-sodium phosphate",
        "Glucose", "Hemin", "Hydrogen", "Lysozyme", "Proteose peptone",
        "Sodium chloride", "Mucin", "Nitrogen", "Vitamin K1",
    ]
    box_x, box_y = 0.88, 0.92
    box_w, box_h = 0.15, 0.32

    # Box background
    culture_box = FancyBboxPatch(
        (box_x, box_y - box_h), box_w, box_h,
        boxstyle="round,pad=0.008",
        facecolor='white', edgecolor='black', lw=1.2, zorder=12
    )
    ax.add_patch(culture_box)

    # Title
    ax.text(box_x + box_w/2, box_y - 0.01, "Culture medium",
            ha='center', va='top', fontsize=8.5, fontweight='bold', zorder=13)

    # Components list
    for i, comp in enumerate(medium_components):
        ax.text(box_x + 0.01, box_y - 0.035 - i * 0.019, comp,
                ha='left', va='top', fontsize=6.5, zorder=13)

    # ── 7. Legend box (bottom-right) ──
    leg_x, leg_y = 0.85, 0.18
    leg_w, leg_h = 0.18, 0.16

    legend_box = FancyBboxPatch(
        (leg_x, leg_y - leg_h), leg_w, leg_h,
        boxstyle="round,pad=0.008",
        facecolor='white', edgecolor='black', lw=1.2, zorder=12
    )
    ax.add_patch(legend_box)

    ax.text(leg_x + leg_w/2, leg_y - 0.01, "Legend",
            ha='center', va='top', fontsize=9, fontweight='bold', zorder=13)

    # Species circle
    c_leg = plt.Circle((leg_x + 0.025, leg_y - 0.045), 0.012,
                        color='gray', ec='black', lw=0.8, zorder=13)
    ax.add_patch(c_leg)
    ax.text(leg_x + 0.05, leg_y - 0.045, "Species", ha='left', va='center',
            fontsize=7, zorder=13)

    # Metabolite square
    rect_leg = plt.Rectangle(
        (leg_x + 0.095, leg_y - 0.052), 0.012, 0.012,
        fill=True, facecolor='white', edgecolor='black', lw=0.8, zorder=13
    )
    ax.add_patch(rect_leg)
    ax.text(leg_x + 0.115, leg_y - 0.045, "Metabolite", ha='left', va='center',
            fontsize=7, zorder=13)

    # Enzyme diamond
    ey = leg_y - 0.08
    diamond_leg = plt.Polygon([
        (leg_x + 0.025, ey + 0.008),
        (leg_x + 0.033, ey),
        (leg_x + 0.025, ey - 0.008),
        (leg_x + 0.017, ey),
    ], closed=True, facecolor='black', edgecolor='black', lw=0.8, zorder=13)
    ax.add_patch(diamond_leg)
    ax.text(leg_x + 0.05, ey, "Enzyme", ha='left', va='center',
            fontsize=7, zorder=13)

    # Production/utilization arrows
    arrow_y = leg_y - 0.115
    # Production arrow (species -> metabolite)
    c_p = plt.Circle((leg_x + 0.02, arrow_y), 0.008,
                      color='gray', ec='black', lw=0.6, zorder=13)
    ax.add_patch(c_p)
    draw_curved_arrow(ax, (leg_x + 0.03, arrow_y), (leg_x + 0.065, arrow_y),
                      'black', lw=1.0, alpha=0.9,
                      connectionstyle="arc3,rad=0.0", shrinkA=2, shrinkB=2)
    rect_p = plt.Rectangle(
        (leg_x + 0.065, arrow_y - 0.006), 0.012, 0.012,
        fill=True, facecolor='white', edgecolor='black', lw=0.6, zorder=13
    )
    ax.add_patch(rect_p)
    ax.text(leg_x + 0.09, arrow_y, "Production/\nutilization",
            ha='left', va='center', fontsize=6.5, zorder=13)

    # Panel label
    ax.text(0.01, 0.99, "C", fontsize=20, fontweight='bold', va='top',
            transform=ax.transAxes, zorder=15)

    fig.tight_layout()
    path = f"{FIG_DIR}/fig4C_reproduced_v2.png"
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    plot_fig4c()
