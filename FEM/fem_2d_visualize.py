#!/usr/bin/env python3
"""
fem_2d_visualize.py  –  Visualisation for 2D FEM biofilm results

Loads from --results-dir:
  snapshots_phi.npy   (n_snap, 5, Nx, Ny)
  snapshots_t.npy     (n_snap,)
  mesh_x.npy          (Nx,)          x = depth  (0 = substratum)
  mesh_y.npy          (Ny,)          y = lateral
  theta_MAP.npy       (20,)
  snapshots_c.npy     (n_snap, Nx, Ny)   [optional, nutrient coupling]
  di_field.npy        (n_snap, Nx, Ny)   [optional]
  alpha_monod.npy     (Nx, Ny)           [optional]

Figures
  fig1_2d_heatmaps.png    5 species × 3 time points (2D imshow)
  fig2_hovmoller.png      Hovmöller  φ(x,t) averaged over y
  fig3_lateral.png        Lateral (y) profiles at 3 depths, t_final
  fig4_dysbiotic_2d.png   2D Dysbiotic Index at 3 time points
  fig5_summary.png        6-panel summary
  fig6_nutrient_2d.png    Nutrient c(x,y) at 3 time points  [if c available]
  fig7_alpha_monod.png    alpha_Monod(x,y) + eigenstrain     [if alpha available]
  fig8_nutrient_summary.png  Combined nutrient coupling summary [if c available]

Convention
  Image orientation: x = depth on vertical axis, 0 (substratum) at BOTTOM;
                     y = lateral on horizontal axis.

Usage
-----
  python fem_2d_visualize.py --results-dir _results_2d_nutrient/quick_test \\
                              --condition "quick_test"
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── visual constants ─────────────────────────────────────────────────────────
SPECIES = ["S.oralis", "A.naeslundii", "Veillonella", "F.nucleatum", "P.gingivalis"]
COLORS  = ["#4477AA", "#66AADD", "#228833", "#CCBB44", "#EE6677"]
CMAPS   = ["Blues",   "Blues",   "Greens",  "YlOrBr",  "Reds"]

_PARAM_KEYS = [
    "a11","a12","a22","b1","b2",
    "a33","a34","a44","b3","b4",
    "a13","a14","a23","a24",
    "a55","b5",
    "a15","a25","a35","a45",
]
_BLOCK_COLORS = (["#4477AA"]*5 + ["#228833"]*5 + ["#CCBB44"]*4
                 + ["#EE6677"]*2 + ["#AA3377"]*4)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_results(d: Path):
    phi   = np.load(d / "snapshots_phi.npy")   # (n_snap, 5, Nx, Ny)
    t     = np.load(d / "snapshots_t.npy")
    x     = np.load(d / "mesh_x.npy")
    y     = np.load(d / "mesh_y.npy")
    theta = np.load(d / "theta_MAP.npy")
    n_snap, _, Nx, Ny = phi.shape
    print(f"Loading results from: {d}")
    print(f"  snapshots : {n_snap}  |  grid : {Nx}×{Ny}")
    print(f"  t range   : [{t[0]:.5f}, {t[-1]:.5f}]")
    print(f"  phi range : [{phi.min():.4f}, {phi.max():.4f}]")

    # Optional nutrient coupling outputs
    c_path = d / "snapshots_c.npy"
    c = np.load(c_path) if c_path.exists() else None
    alpha_path = d / "alpha_monod.npy"
    alpha = np.load(alpha_path) if alpha_path.exists() else None
    if c is not None:
        print(f"  c range   : [{c.min():.4f}, {c.max():.4f}]")
    if alpha is not None:
        print(f"  alpha range: [{alpha.min():.6f}, {alpha.max():.6f}]")

    return phi, t, x, y, theta, c, alpha


def _pick3(n):
    """Return [first, mid, last] snapshot indices."""
    return [0, max(1, n // 2), n - 1]


def _di(phi_snap):
    """Dysbiotic index (n_snap, Nx, Ny) → (n_snap, Nx, Ny).
    DI = 1 - H/H_max,  H = Shannon entropy over 5 species, H_max = ln(5)
    """
    phi_sum = phi_snap.sum(axis=1, keepdims=True).clip(1e-12)
    p = phi_snap / phi_sum
    H = -np.sum(p * np.log(p + 1e-12), axis=1)   # (n_snap, Nx, Ny)
    return 1.0 - H / np.log(5)


def _extent(x, y):
    """imshow extent: [left, right, bottom, top]  (substratum at bottom)."""
    return [y[0], y[-1], x[-1], x[0]]


# ── Fig 1: 2D heatmaps ────────────────────────────────────────────────────────
def fig1_2d_heatmaps(phi, t, x, y, out_dir, cond):
    n_snap = len(t)
    ti3    = _pick3(n_snap)
    ext    = _extent(x, y)

    fig, axes = plt.subplots(5, 3, figsize=(12, 16))
    fig.suptitle(f"2D Species Distribution  |  {cond}", fontsize=13, fontweight="bold")

    for row in range(5):
        vmax = max(phi[ti, row].max() for ti in ti3)
        vmax = max(vmax, 1e-4)
        for col, ti in enumerate(ti3):
            ax = axes[row, col]
            im = ax.imshow(
                phi[ti, row],
                origin="lower",
                extent=ext,
                aspect="auto",
                cmap=CMAPS[row],
                vmin=0, vmax=vmax,
                interpolation="bilinear",
            )
            if row == 0:
                ax.set_title(f"t = {t[ti]:.4f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{SPECIES[row]}\nDepth x", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == 4:
                ax.set_xlabel("Lateral y", fontsize=9)
            else:
                ax.set_xticklabels([])
            plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "fig1_2d_heatmaps.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 2: Hovmöller (depth × time, y-averaged) ───────────────────────────────
def fig2_hovmoller(phi, t, x, y, out_dir, cond):
    phi_xt = phi.mean(axis=3)       # (n_snap, 5, Nx) – average over y

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    fig.suptitle(f"Hovmöller Diagram (y-averaged)  |  {cond}",
                 fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes):
        data = phi_xt[:, i, :].T    # (Nx, n_snap) for pcolormesh
        vmax = max(data.max(), 1e-4)
        pm = ax.pcolormesh(t, x, data, cmap=CMAPS[i], vmin=0, vmax=vmax,
                           shading="auto")
        plt.colorbar(pm, ax=ax, label="φ", pad=0.02)
        ax.set_xlabel("Time t", fontsize=9)
        if i == 0:
            ax.set_ylabel("Depth x  (0 = substratum)", fontsize=9)
        ax.set_title(SPECIES[i], fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "fig2_hovmoller.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 3: Lateral (y) profiles at 3 depths, t_final ─────────────────────────
def fig3_lateral(phi, t, x, y, out_dir, cond):
    phi_f = phi[-1]             # (5, Nx, Ny)
    Nx    = len(x)
    depth_ids    = [0, Nx // 2, Nx - 1]
    depth_labels = [f"x={x[i]:.2f}  ({'surface' if i==0 else 'mid' if i==Nx//2 else 'deep'})"
                    for i in depth_ids]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(f"Lateral Profiles at t_final={t[-1]:.4f}  |  {cond}",
                 fontsize=13, fontweight="bold")

    for col, (ix, dlabel) in enumerate(zip(depth_ids, depth_labels)):
        ax = axes[col]
        for i, sp in enumerate(SPECIES):
            ax.plot(y, phi_f[i, ix, :], color=COLORS[i], label=sp, lw=1.8)
        ax.set_xlabel("Lateral position y", fontsize=10)
        if col == 0:
            ax.set_ylabel("Volume fraction φ", fontsize=10)
        ax.set_title(dlabel, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "fig3_lateral_profiles.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 4: 2D Dysbiotic Index at 3 time points ───────────────────────────────
def fig4_dysbiotic_2d(phi, t, x, y, out_dir, cond):
    n_snap = len(t)
    ti3    = _pick3(n_snap)
    DI_all = _di(phi)             # (n_snap, Nx, Ny)
    ext    = _extent(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"2D Dysbiotic Index  |  {cond}", fontsize=13, fontweight="bold")

    for col, ti in enumerate(ti3):
        ax = axes[col]
        im = ax.imshow(
            DI_all[ti],
            origin="lower",
            extent=ext,
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=0, vmax=1,
            interpolation="bilinear",
        )
        ax.set_title(f"t = {t[ti]:.4f}", fontsize=10)
        ax.set_xlabel("Lateral y", fontsize=9)
        if col == 0:
            ax.set_ylabel("Depth x  (0 = substratum)", fontsize=9)
        plt.colorbar(im, ax=ax, label="DI  (0=healthy, 1=dysbiotic)", pad=0.02)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "fig4_dysbiotic_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 5: 6-panel summary ────────────────────────────────────────────────────
def fig5_summary(phi, t, x, y, theta, out_dir, cond):
    ext     = _extent(x, y)
    DI_all  = _di(phi)                        # (n_snap, Nx, Ny)
    DI_mean = DI_all.mean(axis=(1, 2))        # (n_snap,)
    phi_mean = phi.mean(axis=(2, 3))          # (n_snap, 5)
    phi_f_xt = phi[-1].mean(axis=2)           # (5, Nx) – y-averaged depth profile

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Summary  |  {cond}  |  2D FEM Biofilm",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38)

    # (A) domain-averaged φᵢ(t)
    ax = fig.add_subplot(gs[0, 0])
    for i, sp in enumerate(SPECIES):
        ax.plot(t, phi_mean[:, i], color=COLORS[i], label=sp, lw=1.8)
    ax.set_xlabel("Time t"); ax.set_ylabel("Mean φ (domain)")
    ax.set_title("(A) Domain-averaged φᵢ(t)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) final y-averaged depth profile
    ax = fig.add_subplot(gs[0, 1])
    for i, sp in enumerate(SPECIES):
        ax.plot(phi_f_xt[i], x, color=COLORS[i], label=sp, lw=1.8)
    ax.set_xlabel("φ  (y-averaged)"); ax.set_ylabel("Depth x")
    ax.set_title(f"(B) Depth profile  t={t[-1]:.4f}")
    ax.invert_yaxis()       # 0 = substratum at top
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (C) θ_MAP bar chart (signed, block colours)
    ax = fig.add_subplot(gs[0, 2])
    bar_c = ["#cc3333" if v < 0 else bc for bc, v in zip(_BLOCK_COLORS, theta)]
    ax.barh(range(20), theta, color=bar_c, edgecolor="none", height=0.7)
    ax.set_yticks(range(20))
    ax.set_yticklabels(_PARAM_KEYS, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("θ value"); ax.set_title("(C) θ_MAP")
    ax.invert_yaxis()

    # (D) P.g 2D heatmap at t_final
    ax = fig.add_subplot(gs[1, 0])
    vmax_pg = max(phi[-1, 4].max(), 1e-4)
    im = ax.imshow(phi[-1, 4], origin="lower", extent=ext, aspect="auto",
                   cmap="Reds", vmin=0, vmax=vmax_pg, interpolation="bilinear")
    plt.colorbar(im, ax=ax, pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(D) P.gingivalis  t={t[-1]:.4f}")

    # (E) 2D Dysbiotic Index at t_final
    ax = fig.add_subplot(gs[1, 1])
    im2 = ax.imshow(DI_all[-1], origin="lower", extent=ext, aspect="auto",
                    cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(im2, ax=ax, label="DI", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(E) Dysbiotic Index  t={t[-1]:.4f}")

    # (F) mean DI over time
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, DI_mean, color="#cc3333", lw=2)
    ax.axhline(0.3, color="gray", lw=1, ls="--", label="DI = 0.3 threshold")
    ax.set_xlabel("Time t"); ax.set_ylabel("Mean DI")
    ax.set_title("(F) Domain-mean Dysbiotic Index")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    p = out_dir / "fig5_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 6: Nutrient field c(x,y) at 3 time points ───────────────────────────
def fig6_nutrient_2d(c, t, x, y, out_dir, cond):
    n_snap = len(t)
    ti3    = _pick3(n_snap)
    ext    = _extent(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"2D Nutrient Concentration  |  {cond}",
                 fontsize=13, fontweight="bold")

    for col, ti in enumerate(ti3):
        ax = axes[col]
        im = ax.imshow(
            c[ti],
            origin="lower",
            extent=ext,
            aspect="auto",
            cmap="viridis",
            vmin=0, vmax=c.max(),
            interpolation="bilinear",
        )
        ax.set_title(f"t = {t[ti]:.4f}", fontsize=10)
        ax.set_xlabel("Lateral y", fontsize=9)
        if col == 0:
            ax.set_ylabel("Depth x  (0 = substratum)", fontsize=9)
        plt.colorbar(im, ax=ax, label="c  (nutrient)", pad=0.02)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "fig6_nutrient_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 7: alpha_Monod + eigenstrain ────────────────────────────────────────
def fig7_alpha_monod(alpha, x, y, out_dir, cond):
    ext = _extent(x, y)
    eps_growth = alpha / 3.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Monod Growth Activity & Eigenstrain  |  {cond}",
                 fontsize=13, fontweight="bold")

    # (A) alpha_Monod
    ax = axes[0]
    im = ax.imshow(alpha, origin="lower", extent=ext, aspect="auto",
                   cmap="inferno", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="α_Monod", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(A) α_Monod  [{alpha.min():.2e}, {alpha.max():.2e}]")

    # (B) Eigenstrain eps_growth = alpha/3
    ax = axes[1]
    im2 = ax.imshow(eps_growth, origin="lower", extent=ext, aspect="auto",
                    cmap="magma", interpolation="bilinear")
    plt.colorbar(im2, ax=ax, label="ε_growth", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(B) ε_growth = α/3  [{eps_growth.min():.2e}, {eps_growth.max():.2e}]")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "fig7_alpha_monod.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 8: Combined nutrient coupling summary ───────────────────────────────
def fig8_nutrient_summary(phi, c, alpha, t, x, y, out_dir, cond):
    ext = _extent(x, y)
    DI_all = _di(phi)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Nutrient Coupling Summary  |  {cond}",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38)

    # (A) Final nutrient c
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(c[-1], origin="lower", extent=ext, aspect="auto",
                   cmap="viridis", vmin=0, vmax=c.max(), interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="c", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(A) Nutrient  t={t[-1]:.4f}")

    # (B) Final P.gingivalis
    ax = fig.add_subplot(gs[0, 1])
    vmax_pg = max(phi[-1, 4].max(), 1e-4)
    im2 = ax.imshow(phi[-1, 4], origin="lower", extent=ext, aspect="auto",
                    cmap="Reds", vmin=0, vmax=vmax_pg, interpolation="bilinear")
    plt.colorbar(im2, ax=ax, label="φ_Pg", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(B) P.gingivalis  t={t[-1]:.4f}")

    # (C) Final DI
    ax = fig.add_subplot(gs[0, 2])
    im3 = ax.imshow(DI_all[-1], origin="lower", extent=ext, aspect="auto",
                    cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(im3, ax=ax, label="DI", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(C) Dysbiosis Index  t={t[-1]:.4f}")

    # (D) alpha_Monod
    ax = fig.add_subplot(gs[1, 0])
    im4 = ax.imshow(alpha, origin="lower", extent=ext, aspect="auto",
                    cmap="inferno", interpolation="bilinear")
    plt.colorbar(im4, ax=ax, label="α_Monod", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title("(D) α_Monod (Monod growth activity)")

    # (E) c mean + phi_total mean over time
    ax = fig.add_subplot(gs[1, 1])
    c_mean = c.mean(axis=(1, 2))
    phi_total = phi.sum(axis=1).mean(axis=(1, 2))
    ax.plot(t, c_mean, color="#1b9e77", lw=2, label="c (mean)")
    ax.set_xlabel("Time t"); ax.set_ylabel("Mean c", color="#1b9e77")
    ax.tick_params(axis="y", labelcolor="#1b9e77")
    ax2 = ax.twinx()
    ax2.plot(t, phi_total, color="#d95f02", lw=2, label="φ_total (mean)")
    ax2.set_ylabel("Mean φ_total", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax.set_title("(E) Nutrient depletion & growth")
    ax.grid(True, alpha=0.3)

    # (F) Depth profile: c and phi_total (y-averaged) at t_final
    ax = fig.add_subplot(gs[1, 2])
    c_depth = c[-1].mean(axis=1)
    phi_depth = phi[-1].sum(axis=0).mean(axis=1)
    ax.plot(c_depth, x, color="#1b9e77", lw=2, label="c (nutrient)")
    ax.plot(phi_depth, x, color="#d95f02", lw=2, label="φ_total")
    ax.set_xlabel("Value (y-averaged)"); ax.set_ylabel("Depth x")
    ax.set_title(f"(F) Depth profiles  t={t[-1]:.4f}")
    ax.invert_yaxis()
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    p = out_dir / "fig8_nutrient_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Visualise 2D FEM biofilm results")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--condition",   default="unknown")
    args = ap.parse_args()

    d = Path(args.results_dir)
    phi, t, x, y, theta, c, alpha = load_results(d)

    fig1_2d_heatmaps  (phi, t, x, y,        d, args.condition)
    fig2_hovmoller    (phi, t, x, y,        d, args.condition)
    fig3_lateral      (phi, t, x, y,        d, args.condition)
    fig4_dysbiotic_2d (phi, t, x, y,        d, args.condition)
    fig5_summary      (phi, t, x, y, theta, d, args.condition)

    if c is not None:
        fig6_nutrient_2d(c, t, x, y, d, args.condition)
    if alpha is not None:
        fig7_alpha_monod(alpha, x, y, d, args.condition)
    if c is not None and alpha is not None:
        fig8_nutrient_summary(phi, c, alpha, t, x, y, d, args.condition)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
