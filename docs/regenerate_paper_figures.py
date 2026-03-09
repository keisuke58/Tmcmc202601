#!/usr/bin/env python3
"""
regenerate_paper_figures.py
============================
Regenerate publication-quality Fig 2 & Fig 3 from saved TMCMC run data.

Improvements over original:
  - STIX font (LaTeX-compatible serif)
  - Larger font sizes for print readability
  - 300 dpi + PDF vector output
  - No "Fig X:" prefix in titles
  - Shared legend (not repeated per subplot)
  - Consistent style across all panels

Usage:
    python regenerate_paper_figures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
TMCMC_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "docs" else SCRIPT_DIR
RUNS = TMCMC_ROOT / "data_5species" / "_runs"
OUT_DIR = SCRIPT_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add solver to path
sys.path.insert(0, str(TMCMC_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(TMCMC_ROOT / "data_5species" / "main"))
sys.path.insert(0, str(TMCMC_ROOT / "data_5species"))

from improved_5species_jit import BiofilmNewtonSolver5S
from visualization.helpers import compute_phibar

# ============================================================
# Run directories (with full data)
# ============================================================
RUN_DIRS = {
    "CS": RUNS / "commensal_static_posterior",
    "CH": RUNS / "commensal_hobic_posterior",
    "DS": RUNS / "dysbiotic_static_posterior",
    "DH": RUNS / "dh_v6_wide_baseline",
}

COND_LABELS = {
    "CS": "Commensal Static (CS)",
    "CH": "Commensal HOBIC (CH)",
    "DS": "Dysbiotic Static (DS)",
    "DH": "Dysbiotic HOBIC (DH)",
}

# ============================================================
# Publication style
# ============================================================
SPECIES_NAMES = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{V.\ dispar}$",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]
SPECIES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
COLORS = ["#2196F3", "#4CAF50", "#9467bd", "#ff7f0e", "#d62728"]

plt.rcParams.update(
    {
        # Font — STIX matches LaTeX body
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        # Sizes — large enough for single-column (~85mm) print
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        # Output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        # Lines
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2.0,
    }
)

N_DRAWS = 100
ACTIVE_SPECIES = [0, 1, 2, 3, 4]


def theta_to_matrices(theta):
    """Convert theta(20) -> A(5x5), b(5)."""
    A = np.zeros((5, 5))
    b = np.zeros(5)
    A[0, 0], A[0, 1], A[1, 1] = theta[0], theta[1], theta[2]
    A[1, 0] = theta[1]
    b[0], b[1] = theta[3], theta[4]
    A[2, 2], A[2, 3], A[3, 3] = theta[5], theta[6], theta[7]
    A[3, 2] = theta[6]
    b[2], b[3] = theta[8], theta[9]
    A[0, 2], A[2, 0] = theta[10], theta[10]
    A[0, 3], A[3, 0] = theta[11], theta[11]
    A[1, 2], A[2, 1] = theta[12], theta[12]
    A[1, 3], A[3, 1] = theta[13], theta[13]
    A[4, 4] = theta[14]
    b[4] = theta[15]
    A[0, 4], A[4, 0] = theta[16], theta[16]
    A[1, 4], A[4, 1] = theta[17], theta[17]
    A[2, 4], A[4, 2] = theta[18], theta[18]
    A[3, 4], A[4, 3] = theta[19], theta[19]
    return A, b


def model_time_to_days(t_arr, t_days):
    t_min, t_max = t_arr.min(), t_arr.max()
    d_min, d_max = t_days.min(), t_days.max()
    if t_max > t_min:
        return d_min + (t_arr - t_min) / (t_max - t_min) * (d_max - d_min)
    return t_arr


def load_run(cond):
    """Load all data for a condition."""
    d = RUN_DIRS[cond]
    info = {
        "config": json.load(open(d / "config.json")),
        "data": np.load(d / "data.npy"),
        "samples": np.load(d / "samples.npy"),
        "idx_sparse": np.load(d / "idx_sparse.npy"),
        "t_days": np.load(d / "t_days.npy"),
        "theta_MAP": np.array(json.load(open(d / "theta_MAP.json"))["theta_full"]),
        "theta_mean": np.array(json.load(open(d / "theta_mean.json"))["theta_full"]),
    }
    return info


def run_posterior_draws(info, n_draws=N_DRAWS):
    """Run forward model for posterior draws + MAP."""
    cfg = info["config"]
    phi_init = cfg.get("phi_init", 0.02)
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init, dtype=np.float64)

    solver = BiofilmNewtonSolver5S(
        dt=cfg["dt"],
        maxtimestep=cfg["maxtimestep"],
        c_const=cfg["c_const"],
        alpha_const=cfg["alpha_const"],
        phi_init=phi_init,
    )

    # MAP
    t_fit, x_map = solver.solve(info["theta_MAP"])
    phibar_map = compute_phibar(x_map, ACTIVE_SPECIES)
    t_plot = model_time_to_days(t_fit, info["t_days"])

    # Posterior draws
    rng = np.random.default_rng(42)
    n = min(n_draws, len(info["samples"]))
    idx = rng.choice(len(info["samples"]), n, replace=False)

    phibar_draws = []
    for si in idx:
        _, x_k = solver.solve(info["samples"][si])
        phibar_draws.append(compute_phibar(x_k, ACTIVE_SPECIES))
    phibar_draws = np.array(phibar_draws)

    return t_plot, phibar_map, phibar_draws


# ============================================================
# Fig 2: Posterior predictive fits (one panel per condition)
# ============================================================
def generate_fig2():
    print("Generating Fig 2: Posterior Predictive Fits...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, cond in zip(axes.flat, ["CS", "CH", "DS", "DH"]):
        print(f"  {cond}...")
        info = load_run(cond)
        t_plot, phibar_map, phibar_draws = run_posterior_draws(info)

        q05 = np.nanpercentile(phibar_draws, 5, axis=0)
        q25 = np.nanpercentile(phibar_draws, 25, axis=0)
        q50 = np.nanpercentile(phibar_draws, 50, axis=0)
        q75 = np.nanpercentile(phibar_draws, 75, axis=0)
        q95 = np.nanpercentile(phibar_draws, 95, axis=0)

        for i in range(5):
            c = COLORS[i]
            ax.fill_between(t_plot, q05[:, i], q95[:, i], alpha=0.12, color=c)
            ax.fill_between(t_plot, q25[:, i], q75[:, i], alpha=0.25, color=c)
            ax.plot(t_plot, q50[:, i], color=c, lw=2)
            ax.plot(t_plot, phibar_map[:, i], "--", color="black", lw=1.2, alpha=0.6)
            ax.scatter(
                info["t_days"],
                info["data"][:, i],
                s=80,
                color=c,
                edgecolors="black",
                zorder=10,
                linewidth=1.2,
            )

        ax.set_title(COND_LABELS[cond], fontweight="bold")
        ax.set_xlabel("Days")
        ax.set_ylabel(r"$\bar{\varphi}_i$")
        ax.set_xticks(info["t_days"])
        ax.set_xlim(info["t_days"].min() - 1, info["t_days"].max() + 1)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    # Shared legend
    legend_handles = []
    for i in range(5):
        legend_handles.append(Line2D([0], [0], color=COLORS[i], lw=2.5, label=SPECIES_NAMES[i]))
    legend_handles.append(Line2D([0], [0], color="black", lw=1.2, ls="--", alpha=0.6, label="MAP"))
    legend_handles.append(
        plt.fill_between([], [], [], color="gray", alpha=0.25, label="25\u201375% CI")
    )
    legend_handles.append(
        plt.fill_between([], [], [], color="gray", alpha=0.12, label="5\u201395% CI")
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Data",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        fancybox=True,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"fig2_posterior_predictive.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None)
        print(f"  Saved: {out}")
    plt.close()


# ============================================================
# Fig 2-alt: Per-species panels (one figure per condition)
# ============================================================
def generate_fig2_per_condition():
    print("\nGenerating Fig 2 per-condition panels...")

    for cond in ["CS", "CH", "DS", "DH"]:
        print(f"  {cond}...")
        info = load_run(cond)
        t_plot, phibar_map, phibar_draws = run_posterior_draws(info)

        q05 = np.nanpercentile(phibar_draws, 5, axis=0)
        q25 = np.nanpercentile(phibar_draws, 25, axis=0)
        q50 = np.nanpercentile(phibar_draws, 50, axis=0)
        q75 = np.nanpercentile(phibar_draws, 75, axis=0)
        q95 = np.nanpercentile(phibar_draws, 95, axis=0)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()

        for i in range(5):
            ax = axes_flat[i]
            c = COLORS[i]
            ax.fill_between(t_plot, q05[:, i], q95[:, i], alpha=0.15, color=c, label="5\u201395%")
            ax.fill_between(t_plot, q25[:, i], q75[:, i], alpha=0.30, color=c, label="25\u201375%")
            ax.plot(t_plot, q50[:, i], color=c, lw=2.5, label="Median")
            ax.plot(t_plot, phibar_map[:, i], "--", color="black", lw=1.5, label="MAP")
            ax.scatter(
                info["t_days"],
                info["data"][:, i],
                s=100,
                color=c,
                edgecolors="black",
                zorder=10,
                linewidth=1.5,
                label="Data",
            )
            ax.set_title(SPECIES_NAMES[i], fontsize=16, fontweight="bold")
            ax.set_xlabel("Days")
            ax.set_ylabel(r"$\bar{\varphi}$", fontsize=16)
            ax.set_xticks(info["t_days"])
            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.set_xlim(info["t_days"].min() - 1, info["t_days"].max() + 1)
            if i == 0:
                ax.legend(fontsize=11, loc="best", framealpha=0.9)

        axes_flat[5].axis("off")
        fig.suptitle(
            f"{COND_LABELS[cond]}: Per-Species Posterior Predictive Fits",
            fontsize=18,
            fontweight="bold",
        )
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            out = (
                OUT_DIR
                / f"fig2{chr(97+['CS','CH','DS','DH'].index(cond))}_{cond}_posterior_predictive.{ext}"
            )
            fig.savefig(out, dpi=300 if ext == "png" else None)
            print(f"    Saved: {out}")
        plt.close()


# ============================================================
# Fig 3: MAP Interaction Matrices
# ============================================================
def generate_fig3():
    print("\nGenerating Fig 3: Interaction Matrices...")

    for cond in ["CS", "CH", "DS", "DH"]:
        print(f"  {cond}...")
        info = load_run(cond)
        A_map, b_map = theta_to_matrices(info["theta_MAP"])
        A_mean, b_mean = theta_to_matrices(info["theta_mean"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [5, 5, 1.4]})

        vmax = max(abs(A_map).max(), abs(A_mean).max())
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for ax, A, title in [(axes[0], A_map, "MAP Estimate"), (axes[1], A_mean, "Posterior Mean")]:
            im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(SPECIES_SHORT, fontsize=14)
            ax.set_yticklabels(SPECIES_SHORT, fontsize=14)
            ax.set_title(title, fontsize=16, fontweight="bold")
            for i in range(5):
                for j in range(5):
                    ax.text(
                        j,
                        i,
                        f"{A[i,j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="white" if abs(A[i, j]) > vmax * 0.55 else "black",
                    )

        # Decay bar
        ax3 = axes[2]
        y_pos = np.arange(5)
        ax3.barh(y_pos, b_map, 0.35, label="MAP", color="steelblue", alpha=0.8)
        ax3.barh(y_pos + 0.35, b_mean, 0.35, label="Mean", color="coral", alpha=0.8)
        ax3.set_yticks(y_pos + 0.175)
        ax3.set_yticklabels(SPECIES_SHORT, fontsize=14)
        ax3.set_xlabel(r"$b_i$ (decay)", fontsize=15)
        ax3.set_title("Decay", fontsize=16, fontweight="bold")
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3, axis="x")
        ax3.invert_yaxis()

        fig.suptitle(
            f"{COND_LABELS[cond]}: Interaction Matrix $A$ and Decay $b$",
            fontsize=18,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            out = (
                OUT_DIR
                / f"fig3{chr(97+['CS','CH','DS','DH'].index(cond))}_{cond}_interaction_matrix.{ext}"
            )
            fig.savefig(out, dpi=300 if ext == "png" else None)
            print(f"    Saved: {out}")
        plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    generate_fig2()
    generate_fig2_per_condition()
    generate_fig3()
    print(f"\nAll figures saved to {OUT_DIR}/")
