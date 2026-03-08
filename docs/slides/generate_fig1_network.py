#!/usr/bin/env python3
"""
Figure 1: 5-species interaction network derived from Heine et al. Figure 4C.
Solid arrows = active interactions, dashed gray = locked (A_ij = 0).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")

# Node positions: ecological succession layout
# Top: early colonizers (So, An), Middle: bridge (Vei, Fn), Bottom: pathogen (Pg)
positions = {
    "So": (-0.8, 1.0),
    "An": (0.8, 1.0),
    "Vd": (-1.0, -0.15),
    "Fn": (1.0, -0.15),
    "Pg": (0.0, -1.15),
}

# Full species names
full_names = {
    "So": "S. oralis",
    "An": "A. naeslundii",
    "Vd": "V. dispar",
    "Fn": "F. nucleatum",
    "Pg": "P. gingivalis",
}

# Node colors (colorblind-friendly)
node_colors = {
    "So": "#4393C3",  # blue
    "An": "#5AAE61",  # green
    "Vd": "#FDB863",  # orange
    "Fn": "#9970AB",  # purple
    "Pg": "#D6604D",  # red
}

# Ecological role labels
role_labels = {
    "So": "Pioneer",
    "An": "Commensal",
    "Vd": "Bridge",
    "Fn": "Bridge",
    "Pg": "Pathogen",
}

node_r = 0.22

# Draw nodes
for name, (x, y) in positions.items():
    circle = plt.Circle((x, y), node_r, color=node_colors[name], ec="black", lw=1.5, zorder=10)
    ax.add_patch(circle)
    # Species abbreviation (italic)
    ax.text(
        x,
        y + 0.03,
        name,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        fontstyle="italic",
        color="white",
        zorder=11,
    )
    # Full name below node
    ax.text(
        x,
        y - node_r - 0.12,
        full_names[name],
        ha="center",
        va="top",
        fontsize=8.5,
        fontstyle="italic",
        color="#333333",
    )
    # Role label above node
    ax.text(
        x,
        y + node_r + 0.08,
        role_labels[name],
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#666666",
        fontstyle="italic",
    )


def draw_edge(
    ax,
    p1,
    p2,
    color,
    ls,
    lw,
    label,
    label_pos=0.5,
    label_offset=(0, 0),
    arrow=False,
    arrow_dir="forward",
    curvature=0.0,
):
    """Draw edge between two nodes with optional curvature and arrow."""
    x1, y1 = positions[p1]
    x2, y2 = positions[p2]

    # Shorten to node boundary
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist

    sx = x1 + ux * node_r
    sy = y1 + uy * node_r
    ex = x2 - ux * node_r
    ey = y2 - uy * node_r

    if curvature == 0:
        if arrow:
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=lw, ls=ls, connectionstyle="arc3,rad=0"
                ),
            )
        else:
            ax.plot([sx, ex], [sy, ey], color=color, ls=ls, lw=lw, zorder=2)
    else:
        style = f"arc3,rad={curvature}"
        if arrow:
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls, connectionstyle=style),
            )
        else:
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, ls=ls, connectionstyle=style),
            )

    # Label
    if label:
        # Mid-point with offset
        mx = (x1 + x2) * label_pos + label_offset[0]
        my = (y1 + y2) * label_pos + label_offset[1]
        # Perpendicular offset for curvature
        if curvature != 0:
            perp_x = -uy * curvature * 1.5
            perp_y = ux * curvature * 1.5
            mx += perp_x
            my += perp_y

        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7,
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )


# Active interactions (solid blue arrows, bidirectional = double-headed line)
active_color = "#2166AC"
locked_color = "#BBBBBB"

# So -- An: co-aggregation (bidirectional)
draw_edge(
    ax, "So", "An", active_color, "-", 2.2, "Co-aggregation", label_pos=0.5, label_offset=(0, 0.15)
)

# So -- Vd: lactate exchange (bidirectional)
draw_edge(
    ax,
    "So",
    "Vd",
    active_color,
    "-",
    2.2,
    "Lactate\nexchange",
    label_pos=0.5,
    label_offset=(-0.2, 0.1),
)

# So -- Fn: formate symbiosis (bidirectional)
draw_edge(
    ax,
    "So",
    "Fn",
    active_color,
    "-",
    2.2,
    "Formate\nsymbiosis",
    label_pos=0.5,
    label_offset=(0.25, 0.1),
)

# Vd → Pg: pH rise (directed, with arrow)
draw_edge(
    ax,
    "Vd",
    "Pg",
    active_color,
    "-",
    2.2,
    "pH rise\n(Hill gate)",
    label_pos=0.5,
    label_offset=(-0.25, -0.05),
    arrow=True,
)

# Fn -- Pg: peptide supply (bidirectional)
draw_edge(
    ax,
    "Fn",
    "Pg",
    active_color,
    "-",
    2.2,
    "Peptide\nsupply",
    label_pos=0.5,
    label_offset=(0.25, -0.05),
)


# Locked interactions (dashed gray)
# An -- Vd
draw_edge(ax, "An", "Vd", locked_color, "--", 1.0, None)
# An -- Fn
draw_edge(ax, "An", "Fn", locked_color, "--", 1.0, None)
# Vd -- Fn
draw_edge(ax, "Vd", "Fn", locked_color, "--", 1.0, None)
# So -- Pg
draw_edge(ax, "So", "Pg", locked_color, "--", 1.0, None)
# An -- Pg
draw_edge(ax, "An", "Pg", locked_color, "--", 1.0, None)


# Succession arrow at the bottom
ax.annotate(
    "",
    xy=(0.7, -1.55),
    xytext=(-0.7, -1.55),
    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2),
)
ax.text(
    0.0,
    -1.55,
    "Ecological succession",
    ha="center",
    va="top",
    fontsize=8,
    color="#888888",
    fontstyle="italic",
)

# Legend
legend_elements = [
    mpatches.Patch(
        facecolor="white",
        edgecolor=active_color,
        linewidth=2,
        label="Active interaction ($A_{ij} \\neq 0$)",
    ),
    mpatches.Patch(
        facecolor="white",
        edgecolor=locked_color,
        linewidth=1,
        linestyle="--",
        label="Locked ($A_{ij} = 0$, no metabolic pathway)",
    ),
]
# Manual legend with lines
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=active_color, lw=2.2, label="Active ($A_{ij} \\neq 0$)"),
    Line2D([0], [0], color=locked_color, lw=1.0, ls="--", label="Locked ($A_{ij} = 0$)"),
    Line2D(
        [0], [0], color=active_color, lw=2.2, marker=">", markersize=6, label="Directed (Hill gate)"
    ),
]
ax.legend(
    handles=legend_elements, loc="upper right", fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC"
)

fig.suptitle(
    "Species interaction network\n(derived from Heine et al. 2025, Fig. 4C)",
    fontsize=11,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout()
plt.savefig(
    "/home/nishioka/IKM_Hiwi/Tmcmc202601/docs/slides/public/img/fig1_network.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.savefig(
    "/home/nishioka/IKM_Hiwi/Tmcmc202601/docs/slides/public/img/fig1_network.pdf",
    bbox_inches="tight",
    facecolor="white",
)
print("Saved fig1_network.png/pdf")
