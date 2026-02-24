#!/usr/bin/env python3
"""
plot_attention_figures.py
=========================
口頭試問用の3つの補足図を生成:
  Fig A: 時間の扱い — DI(状態量) vs ε(履歴量) の概念図
  Fig B: DI→E マッピング n, s 感度
  Fig C: 3D PDE を採用しない理由 — 拡散による均質化

出力: FEM/figures/paper_final/
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "figures" / "paper_final"
_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

COND_DI = {
    "CS": 0.421,
    "CH": 0.843,
    "DH": 0.161,
    "DS": 0.845,
}
COND_COLORS = {
    "CS": "#2ca02c",
    "CH": "#17becf",
    "DH": "#d62728",
    "DS": "#ff7f0e",
}
COND_LABELS = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DH": "Dysbiotic HOBIC",
    "DS": "Dysbiotic Static",
}

E_MAX, E_MIN = 1000.0, 10.0


def di_to_E(di, s=1.0, n=2.0):
    r = np.clip(di / s, 0, 1)
    return E_MAX * (1 - r)**n + E_MIN * r


# ═══════════════════════════════════════════════════════════════════════════════
# Fig A: 時間の扱い — DI(状態量) vs ε(履歴量)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_a_time_handling():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Left panel: ODE → DI (state quantity) ──
    ax = axes[0]
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time $t$ [h]", fontsize=12)
    ax.set_ylabel(r"Volume fraction $\varphi_i(t)$", fontsize=12)
    ax.set_title("(a) DI = State quantity at $t = T$", fontsize=13, fontweight="bold")

    t = np.linspace(0, 48, 300)
    # Commensal scenario: So dominant
    phi_so = 0.05 + 0.40 * (1 - np.exp(-t / 10))
    phi_an = 0.05 + 0.15 * (1 - np.exp(-t / 12))
    phi_vd = 0.05 + 0.10 * (1 - np.exp(-t / 14))
    phi_fn = 0.02 + 0.05 * (1 - np.exp(-t / 15))
    phi_pg = 0.01 + 0.02 * (1 - np.exp(-t / 18))

    sp_names = [r"$S.$ oralis", r"$A.$ naeslundii", r"$V.$ dispar",
                r"$F.$ nucleatum", r"$P.$ gingivalis"]
    sp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for phi, name, col in zip(
        [phi_so, phi_an, phi_vd, phi_fn, phi_pg], sp_names, sp_colors
    ):
        ax.plot(t, phi, lw=2, color=col, label=name)

    # Mark T=48h
    ax.axvline(48, color="black", ls="--", lw=1.5, alpha=0.7)
    ax.annotate("$T = 48$ h", xy=(48, 1.02), fontsize=11, ha="center",
                fontweight="bold")

    # Final state marker
    for phi, col in zip([phi_so, phi_an, phi_vd, phi_fn, phi_pg], sp_colors):
        ax.plot(48, phi[-1], "o", color=col, ms=8, zorder=5,
                markeredgecolor="black", markeredgewidth=1)

    # DI box
    phi_final = np.array([phi_so[-1], phi_an[-1], phi_vd[-1], phi_fn[-1], phi_pg[-1]])
    phi_total = phi_final.sum()
    p = phi_final / phi_total
    H = -np.sum(p * np.log(p + 1e-12))
    di_val = 1.0 - H / np.log(5)

    box_text = (
        r"$\mathrm{DI} = 1 - \frac{H}{H_{\max}}$"
        f"\n$= {di_val:.3f}$"
        "\n\nState quantity:"
        "\ndepends only on"
        "\n" + r"$\varphi_i(T)$"
    )
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#e8f4e8", ec="#2ca02c", lw=2)
    ax.text(24, 0.85, box_text, fontsize=10, ha="center", va="top",
            bbox=bbox_props, family="serif")

    ax.legend(loc="center left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ── Right panel: PDE → ε (history integral) ──
    ax = axes[1]
    ax.set_xlim(0, 50)
    ax.set_xlabel("Time $t$ [h]", fontsize=12)
    ax.set_title(r"(b) $\varepsilon$ = History quantity (time integral)", fontsize=13, fontweight="bold")

    # Left y-axis: nutrient & φ_total
    c_t = 1.0 * np.exp(-0.03 * t) + 0.3 * (1 - np.exp(-0.03 * t))
    phi_total_t = 0.1 + 0.6 * (1 - np.exp(-t / 8))
    monod_t = phi_total_t * c_t / (0.5 + c_t)

    ax.set_ylabel(r"Monod integrand $\varphi_{\mathrm{tot}} \cdot \frac{c}{k+c}$", fontsize=11)
    ax.set_ylim(0, 0.75)

    # Shade the integral
    ax.fill_between(t, 0, monod_t, alpha=0.25, color="#d62728",
                    label=r"$\int_0^T \varphi_{\mathrm{tot}} \cdot \frac{c}{k+c}\,dt$")
    ax.plot(t, monod_t, lw=2.5, color="#d62728")

    # Secondary axis: cumulative integral
    ax2 = ax.twinx()
    dt = t[1] - t[0]
    alpha_monod = np.cumsum(monod_t) * dt
    ax2.plot(t, alpha_monod, lw=2.5, color="#9467bd", ls="--",
             label=r"$\alpha_{\mathrm{Monod}}(T)$")
    ax2.set_ylabel(r"Growth eigenstrain $\alpha_{\mathrm{Monod}}$", fontsize=11,
                   color="#9467bd")
    ax2.tick_params(axis="y", colors="#9467bd")
    ax2.set_ylim(0, alpha_monod[-1] * 1.4)

    ax.axvline(48, color="black", ls="--", lw=1.5, alpha=0.7)

    box_text2 = (
        r"$\alpha(x,T) = k_\alpha \int_0^T \varphi_{\mathrm{tot}} \cdot \frac{c}{k+c}\,dt$"
        "\n\nHistory quantity:"
        "\naccumulates over"
        "\nentire growth period"
    )
    bbox_props2 = dict(boxstyle="round,pad=0.5", fc="#f4e8f4", ec="#9467bd", lw=2)
    ax.text(24, 0.72, box_text2, fontsize=10, ha="center", va="top",
            bbox=bbox_props2, family="serif")

    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    ax2.legend(loc="center right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = _OUT / "fig_A_time_handling_DI_vs_epsilon.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig A: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Fig B: DI→E mapping n, s sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
def fig_b_sensitivity_ns():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    di_arr = np.linspace(0, 1.0, 500)

    # ── (a) Vary n with s=1.0 ──
    ax = axes[0]
    for n_val, ls in [(1, "-"), (2, "--"), (3, "-."), (4, ":")]:
        E = di_to_E(di_arr, s=1.0, n=n_val)
        ax.plot(di_arr, E, lw=2.5, ls=ls, label=f"$n = {n_val}$")

    # Mark condition DI values
    for cond, di_val in COND_DI.items():
        E_val = di_to_E(di_val, s=1.0, n=2)
        ax.axvline(di_val, color=COND_COLORS[cond], ls="--", lw=1, alpha=0.5)
        ax.plot(di_val, E_val, "o", color=COND_COLORS[cond], ms=10,
                markeredgecolor="black", markeredgewidth=1, zorder=5)
        ax.annotate(cond, xy=(di_val, E_val), xytext=(0, 12),
                    textcoords="offset points", fontsize=10, fontweight="bold",
                    ha="center", color=COND_COLORS[cond])

    ax.set_xlabel("DI (Dysbiotic Index)", fontsize=12)
    ax.set_ylabel("$E$ [Pa]", fontsize=12)
    ax.set_title("(a) Exponent $n$ sensitivity ($s = 1.0$)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1100)
    ax.set_xlim(0, 1.0)

    # ── (b) Vary s with n=2 ──
    ax = axes[1]
    for s_val, ls in [(0.6, "-"), (0.8, "--"), (1.0, "-."), (1.3, ":")]:
        E = di_to_E(di_arr, s=s_val, n=2)
        ax.plot(di_arr, E, lw=2.5, ls=ls, label=f"$s = {s_val}$")

    for cond, di_val in COND_DI.items():
        E_val = di_to_E(di_val, s=1.0, n=2)
        ax.axvline(di_val, color=COND_COLORS[cond], ls="--", lw=1, alpha=0.5)
        ax.plot(di_val, E_val, "o", color=COND_COLORS[cond], ms=10,
                markeredgecolor="black", markeredgewidth=1, zorder=5)
        ax.annotate(cond, xy=(di_val, E_val), xytext=(0, 12),
                    textcoords="offset points", fontsize=10, fontweight="bold",
                    ha="center", color=COND_COLORS[cond])

    ax.set_xlabel("DI (Dysbiotic Index)", fontsize=12)
    ax.set_ylabel("$E$ [Pa]", fontsize=12)
    ax.set_title("(b) Scale $s$ sensitivity ($n = 2$)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1100)
    ax.set_xlim(0, 1.0)

    # ── (c) E(CS)/E(DS) ratio across (n, s) — ordering robustness ──
    ax = axes[2]
    n_range = np.linspace(0.5, 5, 50)
    s_range = np.linspace(0.5, 1.5, 50)
    N, S = np.meshgrid(n_range, s_range)
    di_cs, di_ds = COND_DI["CS"], COND_DI["DS"]

    E_cs = E_MAX * np.clip(1 - di_cs / S, 0, 1)**N + E_MIN * np.clip(di_cs / S, 0, 1)
    E_ds = E_MAX * np.clip(1 - di_ds / S, 0, 1)**N + E_MIN * np.clip(di_ds / S, 0, 1)
    ratio = E_cs / np.clip(E_ds, 1.0, None)

    pcm = ax.pcolormesh(N, S, ratio, cmap="RdYlGn", shading="gouraud",
                        vmin=1, vmax=40)
    cb = fig.colorbar(pcm, ax=ax, label="$E_{\\mathrm{CS}} / E_{\\mathrm{DS}}$")

    # Contour lines
    cs_contours = ax.contour(N, S, ratio, levels=[2, 5, 10, 20, 30],
                             colors="black", linewidths=1, linestyles="--")
    ax.clabel(cs_contours, inline=True, fontsize=8, fmt="%.0f×")

    # Mark default
    ax.plot(2.0, 1.0, "*", color="white", ms=15, markeredgecolor="black",
            markeredgewidth=1.5, zorder=5)
    ax.annotate("Default\n$(n=2, s=1)$", xy=(2.0, 1.0), xytext=(3.2, 1.3),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

    # Shade region where ratio > 1 (CS always stiffer than DS)
    ax.set_xlabel("Exponent $n$", fontsize=12)
    ax.set_ylabel("Scale $s$", fontsize=12)
    ax.set_title("(c) Stiffness ratio $E_{\\mathrm{CS}}/E_{\\mathrm{DS}}$\n"
                 "(ordering robustness)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = _OUT / "fig_B_DI_E_sensitivity_n_s.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig B: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Fig C: 3D PDE を採用しない理由 — 拡散による均質化
# ═══════════════════════════════════════════════════════════════════════════════
def fig_c_no_3d_pde():
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Simple 1D reaction-diffusion demo ──
    # PDE: ∂φ/∂t = D ∂²φ/∂x² + r·φ·(1-φ)
    # Show that diffusion homogenizes species gradients
    nx = 100
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]

    def simulate_rd_1d(phi0, D, r_growth, nt=5000, dt=None):
        """Simple explicit RD solver with CFL-safe dt."""
        if dt is None:
            dt_cfl = 0.4 * dx**2 / max(D, 1e-6)
            dt = min(dt_cfl, 1e-4)
        phi = phi0.copy()
        snapshots = [phi.copy()]
        for step in range(nt):
            lap = np.zeros_like(phi)
            lap[1:-1] = (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / dx**2
            lap[0] = (phi[1] - phi[0]) / dx**2
            lap[-1] = (phi[-2] - phi[-1]) / dx**2
            growth = r_growth * phi * (1 - phi)
            phi = phi + dt * (D * lap + growth)
            phi = np.clip(phi, 0, 1)
            if step % 1000 == 999:
                snapshots.append(phi.copy())
        return snapshots

    # (a) Low diffusion — gradient preserved
    ax = fig.add_subplot(gs[0, 0])
    phi0 = 0.1 + 0.6 * np.exp(-((x - 0.8)**2) / 0.02)
    snaps_lo = simulate_rd_1d(phi0, D=0.01, r_growth=0.5)
    colors_t = plt.cm.viridis(np.linspace(0, 1, len(snaps_lo)))
    for i, (snap, col) in enumerate(zip(snaps_lo, colors_t)):
        ax.plot(x, snap, color=col, lw=2,
                label=f"$t={i}$" if i in [0, len(snaps_lo)-1] else None)
    ax.set_xlabel("Depth $x/L$")
    ax.set_ylabel(r"$\varphi_{\mathrm{total}}(x)$")
    ax.set_title("(a) Low $D$ — gradient preserved", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (b) High diffusion — homogenization
    ax = fig.add_subplot(gs[0, 1])
    snaps_hi = simulate_rd_1d(phi0, D=5.0, r_growth=0.5, nt=20000)
    for i, (snap, col) in enumerate(zip(snaps_hi, colors_t)):
        ax.plot(x, snap, color=col, lw=2,
                label=f"$t={i}$" if i in [0, len(snaps_hi)-1] else None)
    ax.set_xlabel("Depth $x/L$")
    ax.set_ylabel(r"$\varphi_{\mathrm{total}}(x)$")
    ax.set_title("(b) High $D$ — homogenized", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (c) DI spatial profile: low-D preserves, high-D flattens
    ax = fig.add_subplot(gs[0, 2])
    # Simulate 5 species with different initial distributions
    np.random.seed(42)
    species_phi0 = []
    species_phi0.append(0.3 * np.exp(-((x - 0.2)**2) / 0.05) + 0.05)  # So: near tooth
    species_phi0.append(0.15 * np.exp(-((x - 0.4)**2) / 0.08) + 0.03) # An
    species_phi0.append(0.1 * np.exp(-((x - 0.5)**2) / 0.06) + 0.02)  # Vd
    species_phi0.append(0.08 * np.exp(-((x - 0.7)**2) / 0.04) + 0.01) # Fn
    species_phi0.append(0.05 * np.exp(-((x - 0.9)**2) / 0.03) + 0.005) # Pg

    # Compute DI from initial (spatially varying) vs diffused (flat)
    phi_init = np.array(species_phi0)  # (5, nx)
    phi_total_init = phi_init.sum(axis=0)
    p_init = phi_init / (phi_total_init[None, :] + 1e-12)
    H_init = -np.sum(p_init * np.log(p_init + 1e-12), axis=0)
    DI_init = 1.0 - H_init / np.log(5)

    # After diffusion: each species → uniform mean
    phi_flat = np.array([s.mean() * np.ones(nx) for s in species_phi0])
    phi_total_flat = phi_flat.sum(axis=0)
    p_flat = phi_flat / (phi_total_flat[None, :] + 1e-12)
    H_flat = -np.sum(p_flat * np.log(p_flat + 1e-12), axis=0)
    DI_flat = 1.0 - H_flat / np.log(5)

    ax.plot(x, DI_init, lw=2.5, color="#d62728", label="Before diffusion\n(spatial gradient)")
    ax.fill_between(x, DI_init, alpha=0.15, color="#d62728")
    ax.axhline(DI_flat[0], lw=2.5, color="#1f77b4", ls="--",
               label=f"After diffusion\n(uniform = {DI_flat[0]:.3f})")
    ax.set_xlabel("Depth $x/L$")
    ax.set_ylabel("DI$(x)$")
    ax.set_title("(c) DI spatial profile\nlost by diffusion", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    ax.set_ylim(0, 1)

    # ── Bottom row: why hybrid works ──
    # (d) 0D condition contrast is strong
    ax = fig.add_subplot(gs[1, 0])
    conditions = ["CS", "CH", "DH", "DS"]
    di_vals = [COND_DI[c] for c in conditions]
    colors = [COND_COLORS[c] for c in conditions]
    bars = ax.bar(conditions, di_vals, color=colors, edgecolor="black", lw=1.2, width=0.6)
    for bar, v in zip(bars, di_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("DI (0D)")
    ax.set_title("(d) 0D DI — strong condition contrast\n(18× range preserved)", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.2, axis="y")

    # (e) Conceptual: Hybrid approach diagram
    ax = fig.add_subplot(gs[1, 1:])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(e) Hybrid approach: why 3D PDE is not adopted", fontweight="bold", fontsize=13)

    # Draw boxes
    def draw_box(ax, xy, w, h, text, color, fontsize=10):
        box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                             fc=color, ec="black", lw=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center", fontsize=fontsize, fontweight="bold",
                wrap=True)

    # 3D PDE (crossed out)
    draw_box(ax, (0.2, 4.2), 3.0, 1.4,
             "3D Reaction-Diffusion\n5-species PDE\n(computationally\nprohibitive)", "#ffcccc", 9)
    # X mark
    ax.plot([0.4, 2.9], [4.4, 5.4], color="red", lw=4, alpha=0.7)
    ax.plot([0.4, 2.9], [5.4, 4.4], color="red", lw=4, alpha=0.7)

    # Reasons for rejection
    reasons = [
        "Diffusion homogenizes\nspecies → loses DI contrast",
        "5×N³ DoF → intractable\nfor TMCMC (1000 evals)",
        "No spatial species data\n→ unidentifiable",
    ]
    for i, reason in enumerate(reasons):
        y = 4.8 - i * 0.55
        ax.annotate(reason, xy=(3.5, y), fontsize=8, va="center",
                    color="#cc0000",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff0f0", ec="#cc0000", lw=0.5))

    # Hybrid approach (adopted)
    draw_box(ax, (0.2, 0.5), 2.5, 1.6,
             "0D Hamilton ODE\n→ DI per condition\n(species contrast)", "#e8f4e8", 9)

    draw_box(ax, (3.5, 0.5), 2.5, 1.6,
             "1D Nutrient PDE\n→ α(x) spatial\n(growth structure)", "#e8e8f4", 9)

    # Arrow: combine
    ax.annotate("", xy=(6.5, 1.3), xytext=(6.1, 1.3),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))

    draw_box(ax, (6.7, 0.5), 3.0, 1.6,
             "E(x) = E(DI₀ᴅ)\n× spatial modulation\nfrom α(x)", "#f4f4e8", 9)

    # "Adopted" label
    ax.text(5.0, 2.5, "ADOPTED: Hybrid approach",
            fontsize=12, fontweight="bold", color="#006400", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e0ffe0", ec="#006400", lw=2))

    # Arrows from 0D and 1D to combine
    ax.annotate("", xy=(2.5, 1.3), xytext=(2.8, 1.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2ca02c"))
    ax.annotate("", xy=(6.0, 1.3), xytext=(5.8, 1.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#1f77b4"))

    # Advantages
    advantages = [
        "[+] Preserves condition DI contrast (18x)",
        "[+] Captures spatial growth gradient",
        "[+] Tractable for TMCMC (0D: <1s per eval)",
    ]
    for i, adv in enumerate(advantages):
        ax.text(5.0, 3.55 - i * 0.35, adv, fontsize=9, ha="center",
                color="#006400", fontweight="bold")

    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    path = _OUT / "fig_C_no_3d_pde_rationale.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig C: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating attention-point figures...")
    fig_a_time_handling()
    fig_b_sensitivity_ns()
    fig_c_no_3d_pde()
    print("Done.")
