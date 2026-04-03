#!/usr/bin/env python3
"""
plot_dfba_results.py
====================
Generate publication-quality figures for NIFE/SIIRI dFBA results.

Produces two figures:
  1. nife_dfba_main.png   — 5-panel main figure (species + DI + media)
  2. nife_dfba_siiri.png  — SIIRI infection risk indicator (1-panel, presentation format)

Usage:
  python nife/comets/plot_dfba_results.py [--cycles 500] [--outdir nife/comets/]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# Suppress COMETS/cometspy warnings
warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from nife.comets.oral_biofilm import (
    OralBiofilmComets, SPECIES, MONOD_PARAMS, MEDIA_HEALTHY, MEDIA_DISEASED
)

# ── aesthetics ────────────────────────────────────────────────────────────────
COLORS = {
    "So": "#2196F3",   # blue
    "An": "#4CAF50",   # green
    "Vp": "#FF9800",   # orange
    "Fn": "#9C27B0",   # purple
    "Pg": "#F44336",   # red
}
SPECIES_NAMES = {
    "So": "S. oralis",
    "An": "A. naeslundii",
    "Vp": "V. parvula",
    "Fn": "F. nucleatum",
    "Pg": "P. gingivalis",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── simulation ───────────────────────────────────────────────────────────────

def run_both(max_cycles: int = 500):
    model = OralBiofilmComets()
    print(f"Running healthy  ({max_cycles} cycles)...", flush=True)
    rh = model.run("healthy",  max_cycles=max_cycles)
    print(f"Running diseased ({max_cycles} cycles)...", flush=True)
    rd = model.run("diseased", max_cycles=max_cycles)
    mode = "Monod dFBA" if getattr(rh, "_is_cobra", False) else "mock"
    print(f"Simulation mode: {mode}")
    dih = model.compute_di(rh.total_biomass)
    did = model.compute_di(rd.total_biomass)
    return model, rh, rd, dih, did


def get_media_df(result, condition: str):
    """Extract media time series from result or recompute from biomass."""
    if result.media is not None and len(result.media) > 0:
        return result.media
    # Reconstruct approximate media from reference dicts
    import pandas as pd
    ref = MEDIA_HEALTHY if condition == "healthy" else MEDIA_DISEASED
    cycles = result.total_biomass["cycle"].values
    rows = []
    for c in cycles[::10]:
        for k, v in ref.items():
            rows.append({"cycle": c, "metabolite": k, "conc_mmol": v})
    return pd.DataFrame(rows)


# ── plotting helpers ──────────────────────────────────────────────────────────

def _time_hours(cycles: np.ndarray, time_step: float = 0.01) -> np.ndarray:
    """Convert COMETS cycles to hours (default 0.01 h/cycle)."""
    return cycles * time_step


def plot_species_abundance(ax, bm_df, condition_label: str, time_step: float = 0.01):
    sp_cols = [c for c in SPECIES if c in bm_df.columns]
    totals = bm_df[sp_cols].sum(axis=1).replace(0, np.nan)
    t = _time_hours(bm_df["cycle"].values, time_step)

    for sp in sp_cols:
        frac = bm_df[sp] / totals
        ax.plot(t, frac, color=COLORS[sp], lw=1.8, label=SPECIES_NAMES[sp])

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative abundance")
    ax.set_title(condition_label, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Final fractions annotation
    final_frac = (bm_df[sp_cols].iloc[-1] / totals.iloc[-1]).sort_values(ascending=False)
    top2 = final_frac.index[:2]
    for sp in top2:
        frac = bm_df[sp].iloc[-1] / totals.iloc[-1]
        ax.annotate(
            f"{SPECIES_NAMES[sp]}\n{frac:.0%}",
            xy=(t[-1], frac),
            xytext=(-55, 0), textcoords="offset points",
            fontsize=7, color=COLORS[sp],
            arrowprops=dict(arrowstyle="-", color=COLORS[sp], lw=0.8),
            ha="left",
        )


def plot_di_trajectory(ax, dih, did, time_step: float = 0.01):
    th = _time_hours(dih["cycle"].values, time_step)
    td = _time_hours(did["cycle"].values, time_step)

    # Risk zones
    ax.axhspan(0.00, 0.30, alpha=0.07, color="#4CAF50", zorder=0)
    ax.axhspan(0.30, 0.55, alpha=0.07, color="#FF9800", zorder=0)
    ax.axhspan(0.55, 1.00, alpha=0.07, color="#F44336", zorder=0)
    ax.axhline(0.30, color="#4CAF50", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0.55, color="#FF9800", lw=0.8, ls="--", alpha=0.6)

    ax.plot(th, dih["DI"], color="#2196F3", lw=2.2, label="Healthy (commensal)")
    ax.plot(td, did["DI"], color="#F44336", lw=2.2, label="Peri-implantitis (diseased)")

    # Zone labels (right side)
    t_end = max(th[-1], td[-1])
    ax.text(t_end * 1.01, 0.15, "Healthy\nzone", fontsize=7, color="#4CAF50", va="center")
    ax.text(t_end * 1.01, 0.42, "Early\ndysbiosis", fontsize=7, color="#FF9800", va="center")
    ax.text(t_end * 1.01, 0.77, "Peri-implantitis\nrisk", fontsize=7, color="#F44336", va="center")

    ax.set_xlim(0, t_end * 1.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Dysbiosis Index (DI)")
    ax.set_title("DI trajectory: SIIRI infection indicator", fontweight="bold")
    ax.legend(loc="center right")

    # Delta DI annotation
    di_h_final = float(dih["DI"].iloc[-1])
    di_d_final = float(did["DI"].iloc[-1])
    ax.annotate(
        f"ΔDI = {di_d_final - di_h_final:.2f}",
        xy=(th[-1] * 0.5, (di_h_final + di_d_final) / 2),
        fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_media_dynamics(ax, media_df, condition: str):
    met_colors = {
        "glc_D[e]": "#F44336",   # red = glucose depletes
        "lac_L[e]": "#2196F3",   # blue = lactate accumulates
        "succ[e]":  "#FF9800",   # orange = succinate
        "pheme[e]": "#9C27B0",   # purple = hemin
        "o2[e]":    "#4CAF50",   # green = oxygen
    }
    met_labels = {
        "glc_D[e]": "Glucose",
        "lac_L[e]": "Lactate",
        "succ[e]":  "Succinate",
        "pheme[e]": "Hemin",
        "o2[e]":    "O₂",
    }
    plotted = set()
    for met, color in met_colors.items():
        sub = media_df[media_df["metabolite"] == met]
        if len(sub) < 2:
            continue
        t = _time_hours(sub["cycle"].values)
        ax.plot(t, sub["conc_mmol"].values, color=color, lw=1.6,
                label=met_labels.get(met, met))
        plotted.add(met)

    if not plotted:
        ax.text(0.5, 0.5, "No media data\n(use run_dfba_cobra)", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mM)")
    title = "Healthy media" if condition == "healthy" else "Diseased media"
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(bottom=0)
    if plotted:
        ax.legend(fontsize=7)


def plot_cross_feed_diagram(ax):
    """Schematic of cross-feeding network (So→Vp via lactate, etc.)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Cross-feeding network\n(Monod dFBA, Dukovski 2021 framework)",
                 fontweight="bold")

    # Species positions
    pos = {
        "So": (2, 8), "An": (2, 5), "Vp": (5, 6.5),
        "Fn": (7.5, 5), "Pg": (7.5, 2),
    }
    for sp, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.8, color=COLORS[sp], alpha=0.85, zorder=3))
        ax.text(x, y, sp, ha="center", va="center",
                fontweight="bold", fontsize=9, color="white", zorder=4)
        ax.text(x, y - 1.2, SPECIES_NAMES[sp], ha="center",
                fontsize=6.5, color=COLORS[sp], style="italic")

    # Arrows: cross-feeding interactions
    arrows = [
        ("So", "Vp", "lac 1.8×", "#2196F3"),
        ("An", "Vp", "lac 1.2×", "#4CAF50"),
        ("An", "Fn", "succ", "#9C27B0"),
        ("Vp", "Fn", "(shared lactate)", "#FF9800"),
        ("Fn", "Pg", "succ→hemin?", "#F44336"),
    ]
    for src, tgt, label, color in arrows:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx ** 2 + dy ** 2) ** 0.5
        # Offset to circle edge
        sx = x1 + 0.85 * dx / norm
        sy = y1 + 0.85 * dy / norm
        ex = x2 - 0.85 * dx / norm
        ey = y2 - 0.85 * dy / norm
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.4, mutation_scale=12))
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx + 0.1, my + 0.2, label, fontsize=6.5, color=color)

    # AGORA GEM badge
    ax.text(5, 0.7, "AGORA GEM validation\n(exchange reaction existence)",
            ha="center", va="bottom", fontsize=7, color="gray",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="gray"))


# ── main figure ───────────────────────────────────────────────────────────────

def make_main_figure(model, rh, rd, dih, did, outpath: Path):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45, wspace=0.38,
        left=0.07, right=0.92, top=0.93, bottom=0.07,
    )

    # Row 0: species abundances (left=healthy, center=diseased) + cross-feed diagram (right)
    ax_h   = fig.add_subplot(gs[0, 0])
    ax_d   = fig.add_subplot(gs[0, 1])
    ax_net = fig.add_subplot(gs[0, 2])

    # Row 1: DI (full width)
    ax_di  = fig.add_subplot(gs[1, :])

    # Row 2: media dynamics
    ax_mh  = fig.add_subplot(gs[2, 0])
    ax_md  = fig.add_subplot(gs[2, 1])

    # Shared legend for species (bottom right)
    ax_leg = fig.add_subplot(gs[2, 2])
    ax_leg.axis("off")

    # --- Plot ---
    plot_species_abundance(ax_h, rh.total_biomass, "Healthy (commensal)")
    plot_species_abundance(ax_d, rd.total_biomass, "Peri-implantitis (diseased)")
    plot_cross_feed_diagram(ax_net)
    plot_di_trajectory(ax_di, dih, did)
    plot_media_dynamics(ax_mh, rh.media, "healthy")
    plot_media_dynamics(ax_md, rd.media, "diseased")

    # Legend
    handles = [
        Line2D([0], [0], color=COLORS[sp], lw=2.5, label=SPECIES_NAMES[sp])
        for sp in SPECIES
    ]
    ax_leg.legend(handles=handles, loc="center", title="Species", title_fontsize=9,
                  frameon=True, edgecolor="lightgray")

    # Panel labels
    for ax, lbl in zip([ax_h, ax_d, ax_net, ax_di, ax_mh, ax_md],
                       ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.08, 1.08, lbl, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    fig.suptitle(
        "AGORA-calibrated Monod dFBA — Oral Biofilm Community Dynamics (NIFE/SIIRI)",
        fontsize=11, fontweight="bold", y=0.97,
    )

    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"Saved: {outpath}")


def make_siiri_figure(dih, did, outpath: Path):
    """Single-panel SIIRI infection risk indicator for presentations."""
    fig, ax = plt.subplots(figsize=(8, 4))

    th = _time_hours(dih["cycle"].values)
    td = _time_hours(did["cycle"].values)

    # Background zones
    ax.axhspan(0.00, 0.30, alpha=0.12, color="#4CAF50", zorder=0, label="__nolabel__")
    ax.axhspan(0.30, 0.55, alpha=0.12, color="#FF9800", zorder=0, label="__nolabel__")
    ax.axhspan(0.55, 1.00, alpha=0.12, color="#F44336", zorder=0, label="__nolabel__")

    ax.plot(th, dih["DI"], color="#2196F3", lw=2.5, label="Healthy implant")
    ax.plot(td, did["DI"], color="#F44336", lw=2.5, label="Peri-implantitis")

    # Zone labels
    ax.text(th[-1] * 0.85, 0.15, "Safe", fontsize=9, color="#2E7D32", fontweight="bold")
    ax.text(th[-1] * 0.85, 0.42, "Warning", fontsize=9, color="#E65100", fontweight="bold")
    ax.text(th[-1] * 0.85, 0.77, "Alert", fontsize=9, color="#C62828", fontweight="bold")

    # DI threshold lines
    ax.axhline(0.30, color="#4CAF50", lw=1, ls="--", alpha=0.7)
    ax.axhline(0.55, color="#FF9800", lw=1, ls="--", alpha=0.7)

    ax.set_xlabel("Time (h)", fontsize=11)
    ax.set_ylabel("Dysbiosis Index (DI)", fontsize=11)
    ax.set_title(
        "SIIRI Infection Indicator: DI separates healthy vs peri-implantitis biofilm\n"
        "DI = H(φ) / log N   (normalized Shannon entropy, Nishioka et al. 2026)",
        fontsize=10,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, th[-1])
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"Saved: {outpath}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate NIFE dFBA figures")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).parent)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    model, rh, rd, dih, did = run_both(args.cycles)

    make_main_figure(model, rh, rd, dih, did,
                     args.outdir / "nife_dfba_main.png")
    make_siiri_figure(dih, did,
                      args.outdir / "nife_dfba_siiri.png")

    # Summary stats
    sp = ["So", "An", "Vp", "Fn", "Pg"]
    print("\n=== Final species fractions ===")
    for cond, bm in [("Healthy", rh.total_biomass), ("Diseased", rd.total_biomass)]:
        totals = bm[sp].iloc[-1].sum()
        fracs = (bm[sp].iloc[-1] / totals * 100).round(1)
        print(f"  {cond}: " + ", ".join(f"{s}={fracs[s]}%" for s in sp))

    print("\n=== DI summary ===")
    for cond, di in [("Healthy", dih), ("Diseased", did)]:
        print(f"  {cond}: {di['DI'].iloc[0]:.3f} → {di['DI'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
