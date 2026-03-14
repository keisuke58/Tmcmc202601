"""
Publication-quality style utilities for biofilm TMCMC paper.

Wong (2011) colorblind-safe palette + Nature/PNAS formatting.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette
# ---------------------------------------------------------------------------
WONG = {
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "black": "#000000",
}

# 5 species mapped to colorblind-safe colors (ordered by Socransky complex)
SPECIES_COLORS = [
    WONG["blue"],  # S. oralis (early coloniser)
    WONG["green"],  # A. naeslundii (early coloniser)
    WONG["sky_blue"],  # V. dispar (bridge)
    WONG["orange"],  # F. nucleatum (bridge)
    WONG["vermillion"],  # P. gingivalis (pathogen, red complex)
]

SPECIES_NAMES = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{V.\ dispar}$",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]

SPECIES_NAMES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]

# Condition colors and labels
CONDITION_COLORS = {
    "CS": WONG["blue"],
    "CH": WONG["sky_blue"],
    "DS": WONG["orange"],
    "DH": WONG["vermillion"],
}

CONDITION_LABELS = {
    "CS": "Commensal / Static",
    "CH": "Commensal / HOBIC",
    "DS": "Dysbiotic / Static",
    "DH": "Dysbiotic / HOBIC",
}

# Figure widths for journals (inches)
SINGLE_COL = 3.5  # Nature single column
DOUBLE_COL = 7.5  # Nature double column
FULL_PAGE = 7.5  # Full page width


def apply_paper_style():
    """Apply the paper matplotlib style sheet."""
    style_path = Path(__file__).parent / "paper.mplstyle"
    plt.style.use(str(style_path))


def add_panel_labels(
    axes,
    labels: Optional[List[str]] = None,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = 11,
    fontweight: str = "bold",
):
    """Add (a), (b), (c), ... panel labels to axes."""
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va="bottom",
            ha="right",
        )


def format_axis_sci(ax, axis="y", scilimits=(-2, 2)):
    """Apply scientific notation to axis."""
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits(scilimits)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


def despine(ax, top=True, right=True, bottom=False, left=False):
    """Remove specified spines from axes."""
    for spine, remove in [("top", top), ("right", right), ("bottom", bottom), ("left", left)]:
        ax.spines[spine].set_visible(not remove)


def savefig_paper(fig, path, formats=("png", "pdf"), dpi=300):
    """Save figure in multiple formats for publication."""
    path = Path(path)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
