#!/usr/bin/env python3
"""
fem_3d_visualize.py  –  Visualisation for 3D FEM biofilm results

Loads from --results-dir:
  snapshots_phi.npy   (n_snap, 5, Nx, Ny, Nz)
  snapshots_t.npy     (n_snap,)
  mesh_{x,y,z}.npy
  theta_MAP.npy       (20,)

Figures
  fig1_3d_slices.png       Cross-section slices at mid-domain (XY, XZ, YZ) for each species
  fig2_hovmoller_3d.png    Depth×time Hovmöller (yz-averaged)
  fig3_depth_profiles.png  Final depth profile (yz-averaged) for all species
  fig4_dysbiotic_3d.png    Dysbiotic Index: 3 cross-sections at t_final
  fig5_summary_3d.png      6-panel summary

Convention: x = depth (0=substratum), y/z = lateral
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SPECIES = ["S.oralis", "A.naeslundii", "Veillonella", "F.nucleatum", "P.gingivalis"]
COLORS  = ["#4477AA", "#66AADD", "#228833", "#CCBB44", "#EE6677"]
CMAPS   = ["Blues", "Blues", "Greens", "YlOrBr", "Reds"]
_PARAM_KEYS = [
    "a11","a12","a22","b1","b2","a33","a34","a44","b3","b4",
    "a13","a14","a23","a24","a55","b5","a15","a25","a35","a45",
]
_BLOCK_COLORS = ["#4477AA"]*5 + ["#228833"]*5 + ["#CCBB44"]*4 + ["#EE6677"]*2 + ["#AA3377"]*4


def load(d: Path):
    phi   = np.load(d / "snapshots_phi.npy")   # (n_snap, 5, Nx, Ny, Nz)
    t     = np.load(d / "snapshots_t.npy")
    x     = np.load(d / "mesh_x.npy")
    y     = np.load(d / "mesh_y.npy")
    z     = np.load(d / "mesh_z.npy")
    theta = np.load(d / "theta_MAP.npy")
    _, _, Nx, Ny, Nz = phi.shape
    print(f"Loading: {d}")
    print(f"  snapshots={len(t)}  grid={Nx}×{Ny}×{Nz}  t=[{t[0]:.4f},{t[-1]:.4f}]")
    print(f"  phi range: [{phi.min():.4f}, {phi.max():.4f}]")
    return phi, t, x, y, z, theta


def _pick3(n):
    return [0, max(1, n // 2), n - 1]


def _di_3d(phi_snap):
    """phi_snap: (5, Nx, Ny, Nz) → DI (Nx, Ny, Nz)."""
    phi_sum = phi_snap.sum(axis=0).clip(1e-12)
    p = phi_snap / phi_sum
    H = -np.sum(p * np.log(p + 1e-12), axis=0)
    return 1.0 - H / np.log(5)


# ── Fig 1: 3D cross-section slices ───────────────────────────────────────────
def fig1_3d_slices(phi, t, x, y, z, out_dir, cond):
    """For each species: XY, XZ, YZ slices at mid-domain at t_final."""
    n_snap, n_sp, Nx, Ny, Nz = phi.shape
    phi_f  = phi[-1]    # (5, Nx, Ny, Nz)
    ix_mid = Nx // 2
    iy_mid = Ny // 2
    iz_mid = Nz // 2

    # 3 slice types × 5 species = 15 subplots
    fig, axes = plt.subplots(5, 3, figsize=(13, 17))
    fig.suptitle(f"3D Cross-Section Slices at t_final={t[-1]:.4f}  |  {cond}",
                 fontsize=13, fontweight="bold")

    slice_labels = [
        f"XY (z={z[iz_mid]:.2f})",
        f"XZ (y={y[iy_mid]:.2f})",
        f"YZ (x={x[ix_mid]:.2f})",
    ]

    for row in range(n_sp):
        vmax = max(phi_f[row].max(), 1e-4)
        for col in range(3):
            ax = axes[row, col]
            if col == 0:   # XY slice: axes=(x=depth, y=lateral1), fixed z
                data = phi_f[row, :, :, iz_mid]       # (Nx, Ny)
                ext  = [y[0], y[-1], x[-1], x[0]]
                ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
            elif col == 1:  # XZ slice: axes=(x=depth, z=lateral2), fixed y
                data = phi_f[row, :, iy_mid, :]       # (Nx, Nz)
                ext  = [z[0], z[-1], x[-1], x[0]]
                ax.set_xlabel("Lateral z"); ax.set_ylabel("Depth x")
            else:           # YZ slice: axes=(y, z), fixed x
                data = phi_f[row, ix_mid, :, :]       # (Ny, Nz)
                ext  = [z[0], z[-1], y[-1], y[0]]
                ax.set_xlabel("Lateral z"); ax.set_ylabel("Lateral y")

            im = ax.imshow(data, origin="upper", extent=ext, aspect="auto",
                           cmap=CMAPS[row], vmin=0, vmax=vmax,
                           interpolation="bilinear")
            plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)

            if row == 0:
                ax.set_title(slice_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{SPECIES[row]}\n" + ax.get_ylabel(), fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "fig1_3d_slices.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 2: Hovmöller (depth × time, yz-averaged) ────────────────────────────
def fig2_hovmoller_3d(phi, t, x, y, z, out_dir, cond):
    phi_xt = phi.mean(axis=(3, 4))   # (n_snap, 5, Nx)

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    fig.suptitle(f"Hovmöller Diagram (yz-averaged)  |  {cond}",
                 fontsize=13, fontweight="bold")
    for i, ax in enumerate(axes):
        data = phi_xt[:, i, :].T    # (Nx, n_snap)
        vmax = max(data.max(), 1e-4)
        pm   = ax.pcolormesh(t, x, data, cmap=CMAPS[i], vmin=0, vmax=vmax,
                             shading="auto")
        plt.colorbar(pm, ax=ax, label="φ", pad=0.02)
        ax.set_xlabel("Time t"); ax.set_title(SPECIES[i], fontsize=10)
        if i == 0:
            ax.set_ylabel("Depth x  (0=substratum)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "fig2_hovmoller_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 3: Final depth profiles (yz-averaged) ────────────────────────────────
def fig3_depth_profiles(phi, t, x, y, z, out_dir, cond):
    phi_f  = phi[-1]                    # (5, Nx, Ny, Nz)
    prof   = phi_f.mean(axis=(2, 3))    # (5, Nx) yz-average

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, sp in enumerate(SPECIES):
        ax.plot(prof[i], x, color=COLORS[i], label=sp, lw=2)
    ax.set_xlabel("φ (yz-averaged)"); ax.set_ylabel("Depth x  (0=substratum)")
    ax.set_title(f"Final depth profile  t={t[-1]:.4f}  |  {cond}")
    ax.invert_yaxis(); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig3_depth_profiles.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 4: Dysbiotic Index slices ────────────────────────────────────────────
def fig4_dysbiotic_3d(phi, t, x, y, z, out_dir, cond):
    n_snap, n_sp, Nx, Ny, Nz = phi.shape
    ti3 = _pick3(n_snap)

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    fig.suptitle(f"3D Dysbiotic Index Slices  |  {cond}", fontsize=13, fontweight="bold")

    for row, ti in enumerate(ti3):
        DI = _di_3d(phi[ti])   # (Nx, Ny, Nz)
        ix_mid, iy_mid, iz_mid = Nx//2, Ny//2, Nz//2
        slices = [
            (DI[:, :, iz_mid], [y[0],y[-1],x[-1],x[0]], "upper", "Lateral y", "Depth x",
             f"XY  z={z[iz_mid]:.2f}"),
            (DI[:, iy_mid, :], [z[0],z[-1],x[-1],x[0]], "upper", "Lateral z", "Depth x",
             f"XZ  y={y[iy_mid]:.2f}"),
            (DI[ix_mid, :, :], [z[0],z[-1],y[-1],y[0]], "upper", "Lateral z", "Lateral y",
             f"YZ  x={x[ix_mid]:.2f}"),
        ]
        for col, (data, ext, orig, xlabel, ylabel, ttl) in enumerate(slices):
            ax = axes[row, col]
            im = ax.imshow(data, origin=orig, extent=ext, aspect="auto",
                           cmap="RdYlGn_r", vmin=0, vmax=1,
                           interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="DI", pad=0.02)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ttl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"t={t[ti]:.4f}\n{ylabel}", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "fig4_dysbiotic_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


# ── Fig 5: 6-panel summary ────────────────────────────────────────────────────
def fig5_summary_3d(phi, t, x, y, z, theta, out_dir, cond):
    n_snap, n_sp, Nx, Ny, Nz = phi.shape
    phi_mean = phi.mean(axis=(2, 3, 4))    # (n_snap, 5)
    phi_f    = phi[-1]
    prof_f   = phi_f.mean(axis=(2, 3))     # (5, Nx)
    DI_all   = np.array([_di_3d(phi[k]).mean() for k in range(n_snap)])
    DI_f     = _di_3d(phi_f)              # (Nx, Ny, Nz)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Summary  |  {cond}  |  3D FEM Biofilm",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38)

    # (A) mean φᵢ(t)
    ax = fig.add_subplot(gs[0, 0])
    for i, sp in enumerate(SPECIES):
        ax.plot(t, phi_mean[:, i], color=COLORS[i], label=sp, lw=1.8)
    ax.set_xlabel("t"); ax.set_ylabel("Mean φ (domain)")
    ax.set_title("(A) Domain-averaged φᵢ(t)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (B) final depth profile (yz-avg)
    ax = fig.add_subplot(gs[0, 1])
    for i, sp in enumerate(SPECIES):
        ax.plot(prof_f[i], x, color=COLORS[i], label=sp, lw=1.8)
    ax.set_xlabel("φ (yz-avg)"); ax.set_ylabel("Depth x")
    ax.set_title(f"(B) Depth profile  t={t[-1]:.4f}")
    ax.invert_yaxis(); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (C) θ_MAP signed bar chart
    ax = fig.add_subplot(gs[0, 2])
    bar_c = ["#cc3333" if v < 0 else bc for bc, v in zip(_BLOCK_COLORS, theta)]
    ax.barh(range(20), theta, color=bar_c, edgecolor="none", height=0.7)
    ax.set_yticks(range(20)); ax.set_yticklabels(_PARAM_KEYS, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("θ value"); ax.set_title("(C) θ_MAP"); ax.invert_yaxis()

    # (D) P.g XY mid-slice at t_final
    iz_mid = Nz // 2
    ax = fig.add_subplot(gs[1, 0])
    vmax_pg = max(phi_f[4].max(), 1e-4)
    im = ax.imshow(phi_f[4, :, :, iz_mid], origin="upper",
                   extent=[y[0],y[-1],x[-1],x[0]], aspect="auto",
                   cmap="Reds", vmin=0, vmax=vmax_pg, interpolation="bilinear")
    plt.colorbar(im, ax=ax, pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(D) P.g XY slice  t={t[-1]:.4f}")

    # (E) DI XY mid-slice at t_final
    ax = fig.add_subplot(gs[1, 1])
    im2 = ax.imshow(DI_f[:, :, iz_mid], origin="upper",
                    extent=[y[0],y[-1],x[-1],x[0]], aspect="auto",
                    cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(im2, ax=ax, label="DI", pad=0.02)
    ax.set_xlabel("Lateral y"); ax.set_ylabel("Depth x")
    ax.set_title(f"(E) DI XY slice  t={t[-1]:.4f}")

    # (F) mean DI over time
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, DI_all, color="#cc3333", lw=2)
    ax.axhline(0.3, color="gray", lw=1, ls="--", label="DI=0.3 threshold")
    ax.set_xlabel("t"); ax.set_ylabel("Mean DI")
    ax.set_title("(F) Domain-mean Dysbiotic Index")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    p = out_dir / "fig5_summary_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p.name}")


def fig6_overview_3d(phi, t, x, y, z, out_dir, cond):
    phi_f = phi[-1]
    pg = phi_f[4]
    Nx, Ny, Nz = pg.shape
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vals = pg.ravel()
    Xv = X.ravel()
    Yv = Y.ravel()
    Zv = Z.ravel()
    vmax = max(vals.max(), 1e-6)
    thr = 0.1 * vmax
    mask = vals > thr
    if not np.any(mask):
        mask = vals >= 0.0
    Xs = Xv[mask]
    Ys = Yv[mask]
    Zs = Zv[mask]
    Cs = vals[mask]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Ys, Zs, Xs, c=Cs, cmap="Reds", s=20, alpha=0.7)
    ax.set_xlabel("Lateral y")
    ax.set_ylabel("Lateral z")
    ax.set_zlabel("Depth x")
    ax.set_title(f"3D overview P.g  t={t[-1]:.4f}  |  {cond}")
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    p = out_dir / "fig6_overview_pg_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


def fig7_overview_all_species_3d(phi, t, x, y, z, out_dir, cond):
    phi_f = phi[-1]
    Nx, Ny, Nz = phi_f.shape[1:]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Xv = X.ravel()
    Yv = Y.ravel()
    Zv = Z.ravel()
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3)
    for i in range(5):
        vals = phi_f[i].ravel()
        vmax = max(vals.max(), 1e-6)
        thr = 0.1 * vmax
        mask = vals > thr
        if not np.any(mask):
            mask = vals >= 0.0
        Xs = Xv[mask]
        Ys = Yv[mask]
        Zs = Zv[mask]
        Cs = vals[mask]
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col], projection="3d")
        sc = ax.scatter(Ys, Zs, Xs, c=Cs, cmap=CMAPS[i], s=15, alpha=0.7)
        ax.set_title(SPECIES[i], fontsize=10)
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_zlabel("x")
        ax.view_init(elev=25, azim=-60)
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
    ax = fig.add_subplot(gs[1, 2], projection="3d")
    ax.axis("off")
    fig.suptitle(f"3D overview of all species  t={t[-1]:.4f}  |  {cond}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / "fig7_overview_all_species_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--condition",   default="unknown")
    args = ap.parse_args()

    d = Path(args.results_dir)
    phi, t, x, y, z, theta = load(d)

    fig1_3d_slices              (phi, t, x, y, z,        d, args.condition)
    fig2_hovmoller_3d           (phi, t, x, y, z,        d, args.condition)
    fig3_depth_profiles         (phi, t, x, y, z,        d, args.condition)
    fig4_dysbiotic_3d           (phi, t, x, y, z,        d, args.condition)
    fig5_summary_3d             (phi, t, x, y, z, theta, d, args.condition)
    fig6_overview_3d            (phi, t, x, y, z,        d, args.condition)
    fig7_overview_all_species_3d(phi, t, x, y, z,        d, args.condition)

    print("\nAll 3D figures generated.")


if __name__ == "__main__":
    main()
