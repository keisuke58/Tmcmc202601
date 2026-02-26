#!/usr/bin/env python3
"""
fem_visualize.py  (v2 – fixed & improved)

Enhanced visualization from saved FEM simulation results.
Loads snapshots_G.npy, snapshots_t.npy, mesh_x.npy, theta_MAP.npy
and generates 8 publication-quality figures.

Fixes over v1
-------------
- fig1 : unified colorbar per species group; contour lines added
- fig3 : single shared legend (not repeated 4×)
- fig4 : void fraction φ₀ shown as gray band + relative-composition inset
- fig6 : dysbiotic index (1 - H/H_max) instead of raw entropy
- fig7 : initialization transient clipped (t < t_skip)
- fig8C : θ bar colored by sign (blue = positive, red = negative)

Usage
-----
  cd Tmcmc202601/FEM
  python fem_visualize.py --results-dir _results/dh_baseline \\
                          --condition "Dysbiotic HOBIC"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# ── Species / parameter metadata ─────────────────────────────────────────────
SPECIES_NAMES = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["S.o", "A.n", "Vei", "F.n", "P.g"]
SPECIES_COLORS = ["royalblue", "forestgreen", "goldenrod", "mediumpurple", "crimson"]

THETA_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]
# Block labels for θ colour-coding in fig8C
THETA_BLOCKS = ["M1"] * 5 + ["M2"] * 5 + ["M3"] * 4 + ["M4"] * 2 + ["M5"] * 4
BLOCK_COLORS = {
    "M1": "royalblue",
    "M2": "forestgreen",
    "M3": "goldenrod",
    "M4": "crimson",
    "M5": "mediumpurple",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def shannon_entropy(phi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    total = phi.sum(axis=-1, keepdims=True)
    p = np.where(total > eps, phi / (total + eps), 0.0)
    return -np.sum(np.where(p > eps, p * np.log(p), 0.0), axis=-1)


def load_results(rdir: Path):
    G = np.load(rdir / "snapshots_G.npy")  # (n_snap, n_nodes, 12)
    t = np.load(rdir / "snapshots_t.npy")  # (n_snap,)
    x = np.load(rdir / "mesh_x.npy")  # (n_nodes,)
    theta = np.load(rdir / "theta_MAP.npy")  # (20,)
    phi = G[:, :, 0:5]  # (n_snap, n_nodes, 5)
    phi0 = G[:, :, 5]  # (n_snap, n_nodes)  void
    return G, t, x, theta, phi, phi0


def _sfx(condition: str) -> str:
    return f"  —  {condition}" if condition else ""


# ── Fig 1: Heatmaps with unified scale & contours ────────────────────────────
def fig1_heatmaps(phi, t, x, out_path: Path, condition: str = ""):
    """φᵢ(x,t) heatmaps. Commensal species share one vmax; P.g has its own."""
    comm_vmax = np.percentile(phi[:, :, :4], 97)  # 97th pct of commensal
    pg_vmax = np.percentile(phi[:, :, 4], 97)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    fig.suptitle(f"Space-Time φᵢ(x, t){_sfx(condition)}", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes):
        vmax = pg_vmax if i == 4 else comm_vmax
        z = phi[:, :, i].T  # (n_nodes, n_snap)
        im = ax.pcolormesh(t, x, z, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        # Contour lines at 25%, 50%, 75% of vmax
        try:
            cs = ax.contour(
                t,
                x,
                z,
                levels=[0.25 * vmax, 0.5 * vmax, 0.75 * vmax],
                colors="white",
                linewidths=0.6,
                alpha=0.5,
            )
            ax.clabel(cs, fmt="%.2f", fontsize=6, colors="white")
        except Exception:
            pass
        cbar = fig.colorbar(im, ax=ax, label="φᵢ", pad=0.02)
        ax.set_title(SPECIES_NAMES[i], color=SPECIES_COLORS[i], fontweight="bold", fontsize=10)
        ax.set_xlabel("Hamilton time t", fontsize=9)
        if i == 0:
            ax.set_ylabel("Spatial position x\n(x=0: surface, x=L: bulk)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 2: Time series at 3 positions ────────────────────────────────────────
def fig2_time_series(phi, t, x, out_path: Path, condition: str = "", smooth_sigma: float = 0.8):
    """φᵢ(t) at surface, mid, bulk. Legends on every panel. Light smoothing."""
    nodes = {
        "x=0\n(implant surface)": 0,
        "x=L/2\n(mid)": len(x) // 2,
        "x=L\n(bulk medium)": len(x) - 1,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(
        f"Time Series at Key Spatial Locations{_sfx(condition)}", fontsize=13, fontweight="bold"
    )

    for ax, (label, k) in zip(axes, nodes.items()):
        for i in range(5):
            y = gaussian_filter1d(phi[:, k, i], sigma=smooth_sigma)
            ax.plot(t, y, color=SPECIES_COLORS[i], label=SPECIES_SHORT[i], lw=2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Hamilton time t")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

    axes[0].set_ylabel("φᵢ (volume fraction)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 3: Spatial profiles at 4 times ───────────────────────────────────────
def fig3_spatial_profiles(phi, t, x, out_path: Path, condition: str = ""):
    """φᵢ(x) at 4 time snapshots. Single shared legend at top-right."""
    n_snap = len(t)
    idxs = sorted({0, n_snap // 4, n_snap // 2, n_snap - 1})
    while len(idxs) < 4:
        idxs.append(idxs[-1])
    idxs = idxs[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Spatial Profiles at Key Times{_sfx(condition)}", fontsize=13, fontweight="bold")

    for ax, si in zip(axes.flat, idxs):
        for i in range(5):
            ax.plot(x, phi[si, :, i], color=SPECIES_COLORS[i], lw=2)
        ax.set_title(f"t = {t[si]:.4f}", fontsize=10)
        ax.set_xlabel("Position x (← implant surface | bulk →)")
        ax.set_ylabel("φᵢ")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Single shared legend on last axis
    handles = [
        Line2D([0], [0], color=c, lw=2, label=n) for c, n in zip(SPECIES_COLORS, SPECIES_NAMES)
    ]
    axes.flat[-1].legend(handles=handles, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 4: Final composition with void fraction ───────────────────────────────
def fig4_final_composition(phi, phi0, t, x, out_path: Path, condition: str = ""):
    """
    Two panels:
      Left  – absolute φᵢ stacked area + φ₀ (void) as gray band → sums to 1.0
      Right – relative composition φᵢ/Σφᵢ (%) excluding void
    """
    final_phi = phi[-1]  # (n_nodes, 5)
    final_phi0 = phi0[-1]  # (n_nodes,)
    total_occ = final_phi.sum(axis=1)  # Σφᵢ per node
    rel_phi = final_phi / np.maximum(total_occ[:, None], 1e-12) * 100  # %

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"Final Spatial Composition (t={t[-1]:.4f}){_sfx(condition)}",
        fontsize=13,
        fontweight="bold",
    )

    # ── Left: absolute with void ─────────────────────────────────────────────
    ax = axes[0]
    colors_all = SPECIES_COLORS + ["#b0b0b0"]  # gray for void
    stacks = [final_phi[:, i] for i in range(5)] + [final_phi0]
    labels_all = SPECIES_NAMES + ["φ₀ (void)"]
    ax.stackplot(x, stacks, labels=labels_all, colors=colors_all, alpha=0.80)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Position x  (← implant surface | bulk →)", fontsize=10)
    ax.set_ylabel("Volume fraction φᵢ", fontsize=10)
    ax.set_title("Absolute (includes void φ₀)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.grid(True, alpha=0.25)

    # ── Right: relative composition (%) ──────────────────────────────────────
    ax = axes[1]
    ax.stackplot(
        x,
        [rel_phi[:, i] for i in range(5)],
        labels=SPECIES_NAMES,
        colors=SPECIES_COLORS,
        alpha=0.80,
    )
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Position x  (← implant surface | bulk →)", fontsize=10)
    ax.set_ylabel("Relative composition (%)", fontsize=10)
    ax.set_title("Relative composition φᵢ / Σφᵢ  (void excluded)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 5: Pathogen invasion front ───────────────────────────────────────────
def fig5_pathogen_front(phi, t, x, out_path: Path, condition: str = ""):
    """P.g spatial profiles + front position over time."""
    pg = phi[:, :, 4]
    thresholds = [0.01, 0.03, 0.05]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"P. gingivalis Spatial Invasion{_sfx(condition)}", fontsize=13, fontweight="bold")

    # Left: φ_pg(x) per snapshot coloured by time
    ax = axes[0]
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=t[0], vmax=t[-1])
    for s in range(len(t)):
        ax.plot(x, gaussian_filter1d(pg[s], sigma=1), color=cmap(norm(t[s])), lw=1.2, alpha=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Hamilton time t")
    for thr in thresholds:
        ax.axhline(thr, ls="--", lw=0.8, color="gray", alpha=0.5)
        ax.text(x[-1] * 0.97, thr + 0.001, f"φ={thr}", ha="right", fontsize=7, color="gray")
    ax.set_xlabel("Position x")
    ax.set_ylabel("φ_Pg")
    ax.set_title("Spatial profile over time (purple→yellow = early→late)")
    ax.set_xlim(0, x[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Right: front position vs time
    ax = axes[1]
    for thr, ls, alp in zip(thresholds, ["-", "--", ":"], [0.9, 0.75, 0.6]):
        front = []
        for s in range(len(t)):
            prof = gaussian_filter1d(pg[s], sigma=1)
            exc = np.where(prof > thr)[0]
            front.append(x[exc[-1]] if len(exc) else 0.0)
        ax.plot(t, front, lw=2, ls=ls, color="crimson", alpha=alp, label=f"φ_Pg > {thr}")
    ax.set_xlabel("Hamilton time t")
    ax.set_ylabel("Invasion front x")
    ax.set_title("Front propagation (rightmost position exceeding threshold)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, x[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 6: Dysbiotic index (inverted Shannon) ─────────────────────────────────
def fig6_dysbiotic_index(phi, t, x, out_path: Path, condition: str = ""):
    """
    Dysbiotic index DI = 1 − H/H_max  ∈ [0, 1].
    DI → 0: maximally diverse (healthy)
    DI → 1: one species dominates (dysbiotic)
    """
    H = shannon_entropy(phi)
    H_max = np.log(5)
    DI = 1.0 - H / H_max  # (n_snap, n_nodes)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Dysbiotic Index DI(x,t) = 1 − H/H_max{_sfx(condition)}", fontsize=13, fontweight="bold"
    )

    # Left: heatmap
    ax = axes[0]
    im = ax.pcolormesh(t, x, DI.T, cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto")
    try:
        ax.contour(t, x, DI.T, levels=[0.2, 0.4, 0.6], colors="k", linewidths=0.6, alpha=0.4)
    except Exception:
        pass
    fig.colorbar(im, ax=ax, label="DI  (0=diverse, 1=dominated)")
    ax.set_xlabel("Hamilton time t")
    ax.set_ylabel("Position x")
    ax.set_title("DI(x,t)  [red = dysbiotic, green = healthy]")

    # Right: time series at key nodes
    ax = axes[1]
    k0 = 0
    km = len(x) // 2
    kL = len(x) - 1
    ax.plot(t, DI[:, k0], lw=2, color="darkorange", label="x=0 (surface)")
    ax.plot(t, DI[:, km], lw=2, color="steelblue", label="x=L/2 (mid)")
    ax.plot(t, DI[:, kL], lw=2, color="seagreen", label="x=L (bulk)")
    ax.plot(t, DI.mean(axis=1), lw=2, ls="--", color="black", label="spatial mean")
    ax.axhline(0.3, ls=":", color="red", lw=1, alpha=0.6, label="threshold (DI=0.3)")
    ax.set_xlabel("Hamilton time t")
    ax.set_ylabel("Dysbiotic Index DI")
    ax.set_title("DI over time at key locations  (↑ = more dysbiotic)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 7: Surface vs bulk (with transient clipping) ─────────────────────────
def fig7_surface_vs_bulk(phi, t, x, out_path: Path, condition: str = "", t_skip_frac: float = 0.05):
    """
    Surface (x=0) vs bulk (x=L) dynamics.
    Clips early transient (first t_skip_frac of total time) in title annotation.
    """
    k0 = 0
    kL = len(x) - 1
    t_skip = t[-1] * t_skip_frac
    mask = t >= t_skip  # skip very early transient for annotation

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"Ecology: Implant Surface vs Bulk{_sfx(condition)}", fontsize=13, fontweight="bold"
    )

    for ax, k, loc in zip(
        axes, [k0, kL], ["x = 0  (implant surface)", f"x = {x[-1]:.1f}  (bulk medium)"]
    ):
        for i in range(5):
            y = gaussian_filter1d(phi[:, k, i], sigma=0.5)
            ax.plot(t, y, color=SPECIES_COLORS[i], label=SPECIES_SHORT[i], lw=2)
        ax.fill_between(t, phi[:, k, 4], alpha=0.12, color="crimson")
        ax.axvspan(t[0], t_skip, color="gray", alpha=0.08, label="init transient")
        ax.set_title(loc, fontsize=11)
        ax.set_xlabel("Hamilton time t")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

    axes[0].set_ylabel("φᵢ (volume fraction)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Fig 8: Summary panel ──────────────────────────────────────────────────────
def fig8_summary_panel(phi, phi0, t, x, theta, out_path: Path, condition: str = ""):
    """
    2×3 publication panel.
    (A) P.g heatmap  (B) Dysbiotic index DI(x,t)  (C) θ bar (signed, block-coloured)
    (D) Surface dynamics  (E) Final composition (absolute+void)  (F) P.g front
    """
    DI = 1.0 - shannon_entropy(phi) / np.log(5)
    pg = phi[:, :, 4]
    k0 = 0
    kL = len(x) - 1

    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
    fig.suptitle(
        f"FEM Spatial Extension — Summary{_sfx(condition)}", fontsize=14, fontweight="bold"
    )

    # ── A: P.g heatmap ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    vmax_pg = np.percentile(pg, 97)
    im = ax.pcolormesh(t, x, pg.T, cmap="Reds", vmin=0, vmax=vmax_pg, shading="auto")
    try:
        ax.contour(t, x, pg.T, levels=[0.02, 0.05, 0.08], colors="white", linewidths=0.6, alpha=0.6)
    except Exception:
        pass
    fig.colorbar(im, ax=ax, label="φ_Pg")
    ax.set_title("(A)  P. gingivalis φ(x,t)", fontweight="bold")
    ax.set_xlabel("Hamilton time")
    ax.set_ylabel("Position x")

    # ── B: Dysbiotic index heatmap ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(t, x, DI.T, cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, label="DI (0=healthy, 1=dysbiotic)")
    ax.set_title("(B)  Dysbiotic Index DI(x,t)", fontweight="bold")
    ax.set_xlabel("Hamilton time")
    ax.set_ylabel("Position x")

    # ── C: θ bar (signed + block-coloured) ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    bar_colors = []
    for idx, name in enumerate(THETA_NAMES):
        block_c = BLOCK_COLORS[THETA_BLOCKS[idx]]
        sign_c = "#cc3333" if theta[idx] < 0 else block_c
        bar_colors.append(sign_c)
    bars = ax.barh(range(20), theta, color=bar_colors, alpha=0.80, edgecolor="k", lw=0.3)
    ax.set_yticks(range(20))
    ax.set_yticklabels(THETA_NAMES, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("θᵢ (MAP estimate, signed)")
    ax.set_title("(C)  Estimated θ  [red=negative]", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Block legend
    block_handles = [Patch(color=c, label=f"Block {b}") for b, c in BLOCK_COLORS.items()]
    ax.legend(handles=block_handles, fontsize=7, loc="lower right")

    # ── D: Surface time-series ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for i in range(5):
        ax.plot(
            t,
            gaussian_filter1d(phi[:, k0, i], 0.5),
            color=SPECIES_COLORS[i],
            label=SPECIES_SHORT[i],
            lw=1.8,
        )
    ax.set_title("(D)  Surface (x=0) dynamics", fontweight="bold")
    ax.set_xlabel("Hamilton time")
    ax.set_ylabel("φᵢ")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)

    # ── E: Final composition (absolute, with void) ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    stacks = [phi[-1, :, i] for i in range(5)] + [phi0[-1]]
    labels = SPECIES_NAMES + ["φ₀ (void)"]
    colors = SPECIES_COLORS + ["#b0b0b0"]
    ax.stackplot(x, stacks, labels=labels, colors=colors, alpha=0.78)
    ax.set_title(f"(E)  Final Composition (t={t[-1]:.4f})", fontweight="bold")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Volume fraction")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 1.0)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.grid(True, alpha=0.3)

    # ── F: P.g invasion front ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for thr, ls, alp in zip([0.01, 0.03, 0.05], ["-", "--", ":"], [0.9, 0.75, 0.6]):
        front = []
        for s in range(len(t)):
            prof = gaussian_filter1d(pg[s], sigma=1)
            exc = np.where(prof > thr)[0]
            front.append(x[exc[-1]] if len(exc) else 0.0)
        ax.plot(t, front, lw=2, ls=ls, color="crimson", alpha=alp, label=f"φ_Pg>{thr}")
    ax.set_title("(F)  P.g Invasion Front", fontweight="bold")
    ax.set_xlabel("Hamilton time")
    ax.set_ylabel("Front position x")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, x[-1])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    _FEM_DIR = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_FEM_DIR / "_results" / "dh_baseline",
        help="Directory containing snapshots_G.npy etc.",
    )
    p.add_argument("--condition", default="", help="Condition label for plot titles")
    return p.parse_args()


def main():
    args = parse_args()
    rdir = Path(args.results_dir)
    if not rdir.exists():
        print(f"Results directory not found: {rdir}")
        sys.exit(1)

    print(f"Loading results from: {rdir}")
    G, t, x, theta, phi, phi0 = load_results(rdir)
    print(f"  snapshots : {len(t)}  |  nodes : {len(x)}")
    print(f"  t range   : [{t[0]:.5f}, {t[-1]:.5f}]")
    print(f"  phi range : [{phi.min():.4f}, {phi.max():.4f}]")
    print(f"  phi0 range: [{phi0.min():.4f}, {phi0.max():.4f}]")
    print()

    c = args.condition
    fig1_heatmaps(phi, t, x, rdir / "fig1_spacetime_heatmaps.png", c)
    fig2_time_series(phi, t, x, rdir / "fig2_time_series.png", c)
    fig3_spatial_profiles(phi, t, x, rdir / "fig3_spatial_profiles.png", c)
    fig4_final_composition(phi, phi0, t, x, rdir / "fig4_final_composition.png", c)
    fig5_pathogen_front(phi, t, x, rdir / "fig5_pathogen_front.png", c)
    fig6_dysbiotic_index(phi, t, x, rdir / "fig6_dysbiotic_index.png", c)
    fig7_surface_vs_bulk(phi, t, x, rdir / "fig7_surface_vs_bulk.png", c)
    fig8_summary_panel(phi, phi0, t, x, theta, rdir / "fig8_summary_panel.png", c)

    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
