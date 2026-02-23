"""
Create a new high-quality poster from overview_figs (PNG).
Composes individual figures into a single poster. Does not use existing fig6/fig7.
Run from Tmcmc202601/FEM/:  python make_poster_new.py
"""

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mimg

# Paths
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "overview_figs"
OUT = FIG_DIR
# Poster size (inches): landscape A1-style
W_INCH, H_INCH = 28.0, 18.0
# PDF = print-ready. PNG preview: set PREVIEW_DPI > 0 (e.g. 150); 0 = skip PNG.
PREVIEW_DPI = 0

# Which figures to use (order and layout)
FIGURES = [
    ("fig1_pipeline", "A. End-to-end pipeline"),
    ("fig2_network", "B. Species network & interaction matrix"),
    ("fig4_data_model", "C. Data structure & ODE model"),
    ("fig5_positioning", "D. Method positioning"),
    ("fig8_tmcmc_fem_flow", "E. TMCMC → FEM coupling flow"),
    ("fig9_coupling_DI_Eeff", "F. DI → E_eff mapping"),
    ("fig10_coupling_four_conditions", "G. Four conditions → same FEM"),
]


def ensure_pngs() -> None:
    """Generate PNG (and PDF, SVG) if missing."""
    missing = []
    for name, _ in FIGURES:
        if not (FIG_DIR / f"{name}.png").exists():
            missing.append(name)
    if not missing:
        return
    print("Generating overview figures (PNG)…")
    ret = subprocess.run(
        [sys.executable, str(BASE / "visualize_overview.py")],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    if ret.returncode != 0:
        print(ret.stderr or ret.stdout, file=sys.stderr)
        raise RuntimeError("visualize_overview.py failed; run it manually to create PNGs.")
    print("  Done.")


def load_image(name: str):
    """Load PNG from FIG_DIR."""
    p = FIG_DIR / f"{name}.png"
    if not p.exists():
        raise FileNotFoundError(f"No PNG found for {name} in {FIG_DIR}")
    return mimg.imread(p)


def build_poster() -> None:
    """Compose poster figure and save."""
    ensure_pngs()

    fig = plt.figure(figsize=(W_INCH, H_INCH), facecolor="#FFFFFF")
    # Grid: title, then rows of panels
    gs = GridSpec(
        5, 3,
        figure=fig,
        left=0.04, right=0.97, top=0.96, bottom=0.04,
        hspace=0.30, wspace=0.26,
        height_ratios=[0.75, 1.0, 1.0, 1.0, 1.0],
    )

    # Title block
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.set_facecolor("#FFFFFF")
    fig.text(0.5, 0.935, "Oral Biofilm Parameter Estimation & FEM Coupling", ha="center", fontsize=26, fontweight="bold", color="#1A1A2E")
    fig.text(0.5, 0.895, "TMCMC Bayesian inference → DI field → E_eff(DI) → 3-tooth Abaqus FEM", ha="center", fontsize=14, color="#444455")
    fig.text(0.5, 0.858, "5 species · 20 parameters · Dysbiotic × HOBIC · ~19.7× displacement ratio", ha="center", fontsize=11, color="#666677")

    # Row 1: Pipeline full width
    ax_a = fig.add_subplot(gs[1, :])
    ax_a.axis("off")
    im = load_image("fig1_pipeline")
    ax_a.imshow(im, aspect="auto", interpolation="lanczos")
    ax_a.set_title("A. End-to-end pipeline", fontsize=12, fontweight="bold", color="#1A1A2E", pad=4)

    # Row 2: B, C, D
    for col, (name, label) in enumerate([("fig2_network", "B. Species network"), ("fig4_data_model", "C. Data & model"), ("fig5_positioning", "D. Positioning")]):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        im = load_image(name)
        ax.imshow(im, aspect="auto", interpolation="lanczos")
        ax.set_title(label, fontsize=11, fontweight="bold", color="#1A1A2E", pad=4)

    # Row 3: E full width
    ax_e = fig.add_subplot(gs[3, :])
    ax_e.axis("off")
    im = load_image("fig8_tmcmc_fem_flow")
    ax_e.imshow(im, aspect="auto", interpolation="lanczos")
    ax_e.set_title("E. TMCMC → FEM coupling flow", fontsize=12, fontweight="bold", color="#1A1A2E", pad=4)

    # Row 4: F, G
    for col, (name, label) in enumerate([("fig9_coupling_DI_Eeff", "F. DI → E_eff mapping"), ("fig10_coupling_four_conditions", "G. Four conditions → FEM")]):
        ax = fig.add_subplot(gs[4, col])
        ax.axis("off")
        im = load_image(name)
        ax.imshow(im, aspect="auto", interpolation="lanczos")
        ax.set_title(label, fontsize=11, fontweight="bold", color="#1A1A2E", pad=4)

    # Footer
    fig.text(0.5, 0.018, "References: biofilm_3tooth_report.pdf · overview_tmcmc_fem_en.md", ha="center", fontsize=9, color="#888899")

    out_name = "poster_new"
    path_pdf = OUT / f"{out_name}.pdf"
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"  Saved: {path_pdf}")
    if PREVIEW_DPI > 0:
        path_png = OUT / f"{out_name}.png"
        fig.savefig(path_png, dpi=PREVIEW_DPI, bbox_inches="tight", facecolor="#FFFFFF")
        print(f"  Saved: {path_png}")
    plt.close()


if __name__ == "__main__":
    print("Building new poster from overview_figs…")
    build_poster()
    print("Done.")
