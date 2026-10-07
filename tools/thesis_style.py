"""thesis_style.py — shared matplotlib style for the master-thesis figures.

Goal: every figure in 30_Masterarbeit looks identical in font and size. We render
text with LaTeX (usetex) using the thesis body font (lmodern), at a fixed 9 pt, and
generate each figure at its on-page display width so `\\includegraphics[width=...]`
needs NO rescaling (→ uniform physical font size across all figures).

Requirements: run the figure script with TeX Live 2025 on PATH so matplotlib finds
`latex`/`dvipng`/`dvips`/`gs`, e.g.

    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python scripts/.../foo.py

Usage in a figure script:
    from thesis_style import use
    fig, ax = plt.subplots(figsize=use(width_frac=1.0, aspect=0.45))
    ...
    fig.savefig(path)            # emit .pdf (vector); .png only for imaging
"""

import matplotlib as mpl

# Thesis \textwidth, read from main.log ("\textwidth=455.24411pt"); 72.27 pt/inch.
TEXTWIDTH_PT = 455.24411
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27  # = 6.299 in

# usetex preamble. newunicodechar maps the unicode glyphs that pepper the figure
# scripts (β, γ, Δ, →, ✓, − …) onto LaTeX so individual strings need no manual
# sanitising; only a literal '%' still has to be escaped per call.
_PREAMBLE = "".join(
    [
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{textcomp}",
        r"\usepackage{newunicodechar}",
        r"\newunicodechar{−}{-}",
        r"\newunicodechar{–}{--}",
        r"\newunicodechar{—}{---}",
        r"\newunicodechar{α}{\ensuremath{\alpha}}",
        r"\newunicodechar{β}{\ensuremath{\beta}}",
        r"\newunicodechar{γ}{\ensuremath{\gamma}}",
        r"\newunicodechar{Δ}{\ensuremath{\Delta}}",
        r"\newunicodechar{μ}{\ensuremath{\mu}}",
        r"\newunicodechar{ρ}{\ensuremath{\rho}}",
        r"\newunicodechar{σ}{\ensuremath{\sigma}}",
        r"\newunicodechar{τ}{\ensuremath{\tau}}",
        r"\newunicodechar{φ}{\ensuremath{\phi}}",
        r"\newunicodechar{ψ}{\ensuremath{\psi}}",
        r"\newunicodechar{χ}{\ensuremath{\chi}}",
        r"\newunicodechar{×}{\ensuremath{\times}}",
        r"\newunicodechar{±}{\ensuremath{\pm}}",
        r"\newunicodechar{≥}{\ensuremath{\geq}}",
        r"\newunicodechar{≤}{\ensuremath{\leq}}",
        r"\newunicodechar{°}{\ensuremath{^\circ}}",
        r"\newunicodechar{→}{\ensuremath{\rightarrow}}",
        r"\newunicodechar{←}{\ensuremath{\leftarrow}}",
        r"\newunicodechar{↓}{\ensuremath{\downarrow}}",
        r"\newunicodechar{↑}{\ensuremath{\uparrow}}",
        r"\newunicodechar{✓}{\ensuremath{\checkmark}}",
        r"\newunicodechar{✗}{\ensuremath{\times}}",
        r"\newunicodechar{★}{\ensuremath{\bigstar}}",
        r"\newunicodechar{☆}{\ensuremath{\star}}",
        r"\newunicodechar{●}{\ensuremath{\bullet}}",
        r"\newunicodechar{○}{\ensuremath{\circ}}",
        r"\newunicodechar{≈}{\ensuremath{\approx}}",
        r"\newunicodechar{≠}{\ensuremath{\neq}}",
        r"\newunicodechar{∞}{\ensuremath{\infty}}",
        r"\newunicodechar{∑}{\ensuremath{\sum}}",
        r"\newunicodechar{√}{\ensuremath{\surd}}",
        r"\newunicodechar{Σ}{\ensuremath{\Sigma}}",
        r"\newunicodechar{Φ}{\ensuremath{\Phi}}",
        r"\newunicodechar{Ψ}{\ensuremath{\Psi}}",
        r"\newunicodechar{Ω}{\ensuremath{\Omega}}",
        r"\newunicodechar{λ}{\ensuremath{\lambda}}",
        r"\newunicodechar{θ}{\ensuremath{\theta}}",
        r"\newunicodechar{η}{\ensuremath{\eta}}",
        r"\newunicodechar{δ}{\ensuremath{\delta}}",
        r"\newunicodechar{ε}{\ensuremath{\epsilon}}",
        r"\newunicodechar{π}{\ensuremath{\pi}}",
        r"\newunicodechar{▲}{\ensuremath{\blacktriangle}}",
        r"\newunicodechar{▼}{\ensuremath{\blacktriangledown}}",
        r"\newunicodechar{△}{\ensuremath{\triangle}}",
        r"\newunicodechar{▽}{\ensuremath{\triangledown}}",
        r"\newunicodechar{■}{\ensuremath{\blacksquare}}",
        r"\newunicodechar{□}{\ensuremath{\square}}",
        r"\newunicodechar{◆}{\ensuremath{\blacklozenge}}",
        # subscript / superscript digits (e.g. log₁₀, 10⁻³)
        r"\newunicodechar{₀}{\textsubscript{0}}",
        r"\newunicodechar{₁}{\textsubscript{1}}",
        r"\newunicodechar{₂}{\textsubscript{2}}",
        r"\newunicodechar{₃}{\textsubscript{3}}",
        r"\newunicodechar{₄}{\textsubscript{4}}",
        r"\newunicodechar{₅}{\textsubscript{5}}",
        r"\newunicodechar{₆}{\textsubscript{6}}",
        r"\newunicodechar{₇}{\textsubscript{7}}",
        r"\newunicodechar{₈}{\textsubscript{8}}",
        r"\newunicodechar{₉}{\textsubscript{9}}",
        r"\newunicodechar{⁰}{\textsuperscript{0}}",
        r"\newunicodechar{¹}{\textsuperscript{1}}",
        r"\newunicodechar{²}{\textsuperscript{2}}",
        r"\newunicodechar{³}{\textsuperscript{3}}",
        r"\newunicodechar{⁻}{\textsuperscript{-}}",
    ]
)


def use(width_frac=1.0, aspect=0.62):
    """Apply the thesis rcParams and return a figsize (w, h) in inches.

    width_frac : fraction of \\textwidth the figure will be \\includegraphics'd at
                 (use the SAME fraction in LaTeX → 1:1, no font rescaling).
    aspect     : height / width ratio of the figure.
    """
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [],  # LaTeX (lmodern) chooses
            "text.latex.preamble": _PREAMBLE,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 9,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",  # prevent transparent background on export
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)


# ── Shared palette ────────────────────────────────────────────────────────────
# Use these in all figure scripts for consistency across the thesis.
PALETTE = {
    "health": "#2196A6",  # teal   — commensal/health conditions
    "dysbiosis": "#D95F02",  # orange — dysbiotic conditions
    "prior_on": "#1a7431",  # dark green — with metabolic prior
    "prior_off": "#BDBDBD",  # grey       — without prior / baseline
    "ch": "#2196A6",  # commensal HOBIC
    "dh": "#D95F02",  # dysbiotic HOBIC
    "cs": "#5ab4ac",  # commensal static (lighter teal)
    "ds": "#f1a340",  # dysbiotic static (lighter orange)
}

# Species colours (5-species Heine model, consistent with plot_glv_heine_paper.py)
SP_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
SP_SHORT = ["So", "An", "Vd/Vp", "Fn", "Pg"]


def clean_ax(ax):
    """Remove top and right spines — call once per Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def tight_ylim(ax, data, margin=0.12):
    """Set y-limits to data range ± margin*range, avoiding zero-anchoring."""
    lo, hi = float(min(data)), float(max(data))
    span = hi - lo if hi != lo else abs(hi) * 0.1 or 0.1
    ax.set_ylim(lo - margin * span, hi + margin * span)
