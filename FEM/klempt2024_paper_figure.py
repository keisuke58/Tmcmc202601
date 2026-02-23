"""Klempt 2024: publication-quality comparison figure.

Solves the steady-state nutrient equation for three consumption regimes
(reaction-limited / intermediate / diffusion-limited) and produces a
paper-ready 5-panel comparison figure.

PDE (Klempt 2024):
    -D_c * Delta(c) + g * phi0(x) * c / (k + c) = 0    in [0,1]^2
    c = c_inf = 1                                        on boundary

Thiele modulus:  Th = sqrt(g * R^2 / D_c)  (R = 0.35 = biofilm semi-axis)
  Case A  (g=5,   Th~0.8): reaction-limited  -> c uniform inside biofilm
  Case B  (g=50,  Th~2.5): intermediate      -> moderate depletion
  Case C  (g=500, Th~7.9): diffusion-limited -> severe depletion at centre

Figure layout (double-column, 7 x 4 inches):
  [phi0 | c (A) | c (B) | c (C) | 1D profiles along y=0.5]

Usage:
    /path/to/klempt_fem/bin/python klempt2024_paper_figure.py
"""

import os
import sys
import numpy as onp
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------- import demo helpers ----------
sys.path.insert(0, os.path.dirname(__file__))
from jax_fem_reaction_diffusion_demo import (
    build_problem, phi0_fn,
    AX, AY, SKEW, EPS_PHI, D_C, K_MONOD, C_INF,
)
from jax_fem.solver import solver

jax.config.update("jax_enable_x64", True)

# ============================================================
# Matplotlib style: journal-ready
# ============================================================
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        200,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":   1.2,
    "pdf.fonttype":      42,   # TrueType in PDF
})

# ============================================================
# Cases to compare
# ============================================================
CASES = [
    {"label": "A",  "g_eff":   5, "title": r"$g=5$  (Th$\approx$0.8, reaction-limited)"},
    {"label": "B",  "g_eff":  50, "title": r"$g=50$  (Th$\approx$2.5, intermediate)"},
    {"label": "C",  "g_eff": 500, "title": r"$g=500$ (Th$\approx$7.9, diffusion-limited)"},
]

NX = 60   # mesh resolution (61x61 nodes -> 3600 QUAD4 elements)


# ============================================================
# Solve all cases
# ============================================================
def solve_case(g_eff, nx=NX):
    prob = build_problem(D_c=D_C, k_monod=K_MONOD, g_eff=g_eff, nx=nx, ny=nx)
    sl = solver(prob, solver_options={"umfpack_solver": {}})
    c_sol = onp.array(sl[0][:, 0])    # (num_nodes,)
    coords = onp.array(prob.fes[0].mesh.points)   # (num_nodes, 2)
    return coords, c_sol


def coords_to_grid(coords, values, nx=NX):
    """Reshape flat node arrays to 2D grids (X, Y, Z) for contourf.

    rectangle_mesh(nx, nx, 1, 1) places nodes with x varying along axis-0:
        points[i*(nx+1)+j] = (x[i], y[j])
    """
    n = nx + 1
    X = coords[:, 0].reshape(n, n)   # x varies along axis 0
    Y = coords[:, 1].reshape(n, n)
    Z = values.reshape(n, n)
    return X, Y, Z


def phi0_grid(coords, nx=NX):
    phi_vals = onp.array(jax.vmap(phi0_fn)(jnp.array(coords)))
    return coords_to_grid(coords, phi_vals, nx)


# ============================================================
# Main: solve + plot
# ============================================================
def main():
    print("Solving 3 cases ...")
    results = []
    for case in CASES:
        g = case["g_eff"]
        Th = onp.sqrt(g * AX**2 / D_C)
        print(f"  g={g:4d}  Thiele={Th:.2f} ...")
        coords, c_sol = solve_case(g)
        X, Y, C = coords_to_grid(coords, c_sol)
        c_min = float(c_sol.min())
        print(f"         c_min={c_min:.3f}, c_max={c_sol.max():.3f}")
        results.append({"X": X, "Y": Y, "C": C,
                        "coords": coords, "c_sol": c_sol, **case})

    # phi0 grid (same mesh for all cases)
    Xp, Yp, PHI = phi0_grid(results[0]["coords"])

    # ============================================================
    # Figure: 5 panels (1 row)
    # ============================================================
    fig = plt.figure(figsize=(7.2, 2.6))
    # widths: 4 map panels + 1 profile panel
    gs = fig.add_gridspec(1, 5, wspace=0.38, left=0.05, right=0.97,
                          top=0.88, bottom=0.18)

    axes_maps = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_prof   = fig.add_subplot(gs[0, 4])

    cmap_phi  = plt.cm.Greens
    cmap_c    = plt.cm.RdYlGn
    levels_c  = onp.linspace(0, 1, 21)
    levels_phi = onp.linspace(0, 1, 11)

    def add_colorbar(fig, ax, im, label, ticks):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.set_label(label, labelpad=2)
        cb.ax.tick_params(labelsize=7)
        return cb

    # --- Panel 0: phi0 ---
    ax = axes_maps[0]
    im0 = ax.contourf(Xp, Yp, PHI, levels=levels_phi,
                      cmap=cmap_phi, extend="neither")
    ax.contour(Xp, Yp, PHI, levels=[0.5], colors="k",
               linewidths=0.8, linestyles="--")
    add_colorbar(fig, ax, im0, r"$\phi_0$ [-]", [0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(r"Biofilm indicator $\phi_0$", pad=3)
    ax.set_xlabel(r"$x$", labelpad=1)
    ax.set_ylabel(r"$y$", labelpad=1)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.text(0.03, 0.93, "(a)", transform=ax.transAxes, fontsize=8, va="top")

    # --- Panels 1-3: c for each case ---
    panel_labels = ["(b)", "(c)", "(d)"]
    for k, res in enumerate(results):
        ax = axes_maps[k + 1]
        im = ax.contourf(res["X"], res["Y"], res["C"],
                         levels=levels_c, cmap=cmap_c, extend="neither")
        # Biofilm boundary overlay
        ax.contour(Xp, Yp, PHI, levels=[0.5], colors="k",
                   linewidths=0.8, linestyles="--")
        add_colorbar(fig, ax, im, r"$c$ [-]", [0, 0.25, 0.5, 0.75, 1.0])
        g    = res["g_eff"]
        Th   = onp.sqrt(g * AX**2 / D_C)
        ax.set_title(f"$g={g}$,  Th$\\approx${Th:.1f}", pad=3)
        ax.set_xlabel(r"$x$", labelpad=1)
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.text(0.03, 0.93, panel_labels[k], transform=ax.transAxes,
                fontsize=8, va="top")

    # --- Panel 4: 1D profiles along y=0.5 ---
    ax_prof.axvline(0.5 - AX, color="gray", lw=0.7, ls=":", alpha=0.8)
    ax_prof.axvline(0.5 + AX, color="gray", lw=0.7, ls=":", alpha=0.8,
                    label="Biofilm edge")

    colors = ["#1a9641", "#fdae61", "#d7191c"]
    styles = ["-", "--", ":"]
    # y=0.5 slice index: y[j]=0.5 → j = NX//2
    j_mid = NX // 2
    x_lin = onp.linspace(0, 1, NX + 1)

    for k, res in enumerate(results):
        g  = res["g_eff"]
        Th = onp.sqrt(g * AX**2 / D_C)
        c_slice = res["C"][:, j_mid]   # C[i, j_mid]: x varies along axis 0
        ax_prof.plot(x_lin, c_slice,
                     color=colors[k], ls=styles[k],
                     label=f"$g={g}$ (Th$\\approx${Th:.1f})")

    # phi0 cross-section (scaled for overlay)
    phi_slice = PHI[:, j_mid]
    ax_prof.fill_between(x_lin, 0, phi_slice * 0.3,
                         color="#99d594", alpha=0.3,
                         label=r"$\phi_0 \times 0.3$")

    ax_prof.set_xlim(0, 1)
    ax_prof.set_ylim(-0.02, 1.05)
    ax_prof.set_xlabel(r"$x$  (at $y=0.5$)", labelpad=1)
    ax_prof.set_ylabel(r"$c(x,\,0.5)$", labelpad=1)
    ax_prof.set_title("Cross-section profiles", pad=3)
    ax_prof.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_prof.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_prof.legend(loc="lower center", framealpha=0.85, fontsize=7,
                   borderpad=0.4, handlelength=1.4)
    ax_prof.text(0.03, 0.93, "(e)", transform=ax_prof.transAxes,
                fontsize=8, va="top")

    # Common figure title
    fig.suptitle(
        r"Nutrient transport in egg-shaped biofilm — Klempt et al. (2024)"
        "\n"
        r"$-D_c\,\Delta c + g\,\phi_0(x)\,c/(k+c)=0,\quad"
        r"c|_{\partial\Omega}=1,\quad D_c=1,\;k=1$",
        fontsize=8, y=1.0,
    )

    # ============================================================
    # Save
    # ============================================================
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "klempt2024_results")
    os.makedirs(out_dir, exist_ok=True)

    for ext in ["png", "pdf"]:
        fpath = os.path.join(out_dir, f"klempt2024_comparison.{ext}")
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print(f"Saved: {fpath}")

    # --- Extra: Thiele modulus sensitivity curve ---
    _plot_thiele_curve(out_dir)

    print("Done.")


# ============================================================
# Bonus: Thiele modulus sensitivity (c_min vs Th)
# ============================================================
def _plot_thiele_curve(out_dir):
    """Compute c_min inside biofilm vs Thiele modulus and plot."""
    print("Computing Thiele sensitivity curve (7 points) ...")
    g_vals = [1, 3, 8, 20, 50, 150, 500]
    th_vals = [onp.sqrt(g * AX**2 / D_C) for g in g_vals]

    c_mins = []
    for g in g_vals:
        coords, c_sol = solve_case(g, nx=30)
        phi_vals = onp.array(jax.vmap(phi0_fn)(jnp.array(coords)))
        mask = phi_vals > 0.5
        c_min = float(c_sol[mask].min()) if mask.any() else 0.0
        print(f"  g={g:4d}  Th={onp.sqrt(g*AX**2/D_C):.2f}  c_min={c_min:.3f}")
        c_mins.append(c_min)

    fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))
    ax2.semilogx(th_vals, c_mins, "o-", color="#2166ac",
                 markersize=5, markeredgewidth=0.5, markeredgecolor="k")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    ax2.axvline(1.0, color="gray", lw=0.6, ls=":", alpha=0.7,
                label="Th = 1")
    ax2.set_xlabel(r"Thiele modulus  Th $= \sqrt{g\,R^2/D_c}$")
    ax2.set_ylabel(r"$c_{\min}$ inside biofilm  [-]")
    ax2.set_title(
        r"Diffusion-limitation onset" "\n"
        r"($R=0.35$, $D_c=1$, $k=1$, $c_\infty=1$)",
        fontsize=8
    )
    ax2.set_xlim(min(th_vals)*0.5, max(th_vals)*2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", lw=0.3, alpha=0.5)
    fig2.tight_layout()

    for ext in ["png", "pdf"]:
        fpath = os.path.join(out_dir, f"klempt2024_thiele_curve.{ext}")
        fig2.savefig(fpath, dpi=200, bbox_inches="tight")
        print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
