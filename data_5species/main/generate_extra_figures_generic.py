#!/usr/bin/env python3
"""
Generate additional high-quality figures for TMCMC runs.
Dynamically adapts to Condition/Cultivation from config.json.

New figures beyond the standard set:
  Fig_A01 - Interaction matrix heatmap (MAP)
  Fig_A02 - Per-species panel (individual subplots with posterior bands + data)
  Fig_A03 - State decomposition (phi, psi, phibar per species)
  Fig_A04 - Species composition stacked area chart
  Fig_A05 - Parameter violin plots grouped by block (M1-M5)
  Fig_A06 - Parameter correlation matrix
  Fig_A07 - Log-likelihood landscape
  Fig_A08 - Posterior predictive check (PPC) at observation times
  Fig_A09 - MAP vs Mean fit comparison (overlay)
  Fig_A10 - TMCMC convergence summary dashboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# -- Project imports --
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_5SPECIES_DIR = SCRIPT_DIR.parent
TMCMC_ROOT = DATA_5SPECIES_DIR.parent
PROGRAM_DIR = TMCMC_ROOT / "tmcmc" / "program2602"

sys.path.insert(0, str(DATA_5SPECIES_DIR))
sys.path.insert(0, str(PROGRAM_DIR))

try:
    from visualization.helpers import compute_phibar, compute_fit_metrics
except ImportError as e:
    print(f"Error importing visualization.helpers: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

from improved_5species_jit import BiofilmNewtonSolver5S

# ============================================================
# Config & Setup
# ============================================================
if len(sys.argv) > 1:
    RUN_DIR = Path(sys.argv[1]).resolve()
else:
    print("Usage: python generate_extra_figures_generic.py <run_dir>")
    sys.exit(1)

if not RUN_DIR.exists():
    print(f"Error: Run directory {RUN_DIR} does not exist.")
    sys.exit(1)

OUT_DIR = RUN_DIR
# If figures are usually in the root of run_dir or in figures/ subdir?
# The original script output to current directory. 
# But let's check if previous script used 'figures' subdir.
# The user said "run in each folder to create many images". Usually images are in the run folder directly or figures/.
# The original script had: path = OUT_DIR / "Fig_A01..."
# And earlier I saw: OUT_DIR = RUN_DIR / "figures" in the snippet I read.
# Let's keep OUT_DIR = RUN_DIR (or RUN_DIR / "figures" if that's what I saw).

# Wait, the read snippet showed:
# 45→RUN_DIR = Path(__file__).resolve().parent
# 46→OUT_DIR = RUN_DIR / "figures"
# 47→OUT_DIR.mkdir(exist_ok=True)

# So it WAS creating a "figures" subdirectory.
# I should stick to that or just output to RUN_DIR if the user wants them "in each folder".
# User said: "generate_extra_figures.py を各フォルダで行って画像をたくさん作成したうえで"
# Usually existing figures are in the run root.
# Let's check where standard figures are.
# In RESULTS_JP.md generation, I saw: ("TSM_simulation_*_MAP_Fit_with_data.png", ...)
# These are usually in the run root.
# But "extra" figures might be better in a subfolder or root?
# The original script I read (lines 45-47) put them in `figures/`.
# However, `generate_report.py` looks for files to list them.
# In `generate_report.py`, I listed them as:
# | `Fig_A01...` | ... |
# If they are in `figures/` subdir, I should link them as `figures/Fig_A01...`.
# But my `generate_report.py` just lists the filename.
# If I put them in `figures/`, I should update `generate_report.py` or put them in root.
# Let's put them in `RUN_DIR` directly to match standard figures behavior (usually).
# Or better, check where standard figures are.
# Standard figures (TSM_simulation...) are in the root of run dir.
# So I will change OUT_DIR to RUN_DIR.
OUT_DIR = RUN_DIR

# Load config first to get metadata
with open(RUN_DIR / "config.json") as f:
    config = json.load(f)

CONDITION = config["metadata"].get("condition", "Unknown")
CULTIVATION = config["metadata"].get("cultivation", "Unknown")
TITLE_PREFIX = f"{CONDITION} {CULTIVATION}"
FILE_PREFIX = f"{CONDITION}_{CULTIVATION}"

SPECIES_NAMES = [
    r"$S.\ oralis$",
    r"$A.\ naeslundii$",
    r"$V.\ dispar$",
    r"$F.\ nucleatum$",
    r"$P.\ gingivalis$",
]
SPECIES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728']

THETA_NAMES = [
    "a11","a12","a22","b1","b2",
    "a33","a34","a44","b3","b4",
    "a13","a14","a23","a24",
    "a55","b5",
    "a15","a25","a35","a45",
]

BLOCK_NAMES  = ["M1", "M1", "M1", "M1", "M1",
                "M2", "M2", "M2", "M2", "M2",
                "M3", "M3", "M3", "M3",
                "M4", "M4",
                "M5", "M5", "M5", "M5"]
BLOCK_COLORS = {"M1": "#1f77b4", "M2": "#2ca02c", "M3": "#9467bd",
                "M4": "#ff7f0e", "M5": "#d62728"}

N_POSTERIOR_DRAWS = 100  # draws for posterior bands
ACTIVE_SPECIES = [0, 1, 2, 3, 4]


def theta_to_matrices(theta):
    """Convert theta(20) -> A(5x5), b(5)."""
    A = np.zeros((5, 5))
    b = np.zeros(5)
    A[0,0]=theta[0]; A[0,1]=theta[1]; A[1,0]=theta[1]; A[1,1]=theta[2]
    b[0]=theta[3]; b[1]=theta[4]
    A[2,2]=theta[5]; A[2,3]=theta[6]; A[3,2]=theta[6]; A[3,3]=theta[7]
    b[2]=theta[8]; b[3]=theta[9]
    A[0,2]=theta[10]; A[2,0]=theta[10]; A[0,3]=theta[11]; A[3,0]=theta[11]
    A[1,2]=theta[12]; A[2,1]=theta[12]; A[1,3]=theta[13]; A[3,1]=theta[13]
    A[4,4]=theta[14]; b[4]=theta[15]
    A[0,4]=theta[16]; A[4,0]=theta[16]; A[1,4]=theta[17]; A[4,1]=theta[17]
    A[2,4]=theta[18]; A[4,2]=theta[18]; A[3,4]=theta[19]; A[4,3]=theta[19]
    return A, b


def model_time_to_days(t_arr, t_days):
    """Map model time array to experimental days."""
    t_min, t_max = t_arr.min(), t_arr.max()
    d_min, d_max = t_days.min(), t_days.max()
    if t_max > t_min:
        return d_min + (t_arr - t_min) / (t_max - t_min) * (d_max - d_min)
    return t_arr


# ============================================================
# Load data
# ============================================================
print(f"Loading data for {TITLE_PREFIX}...")
data = np.load(RUN_DIR / "data.npy")
samples = np.load(RUN_DIR / "samples.npy")
logL = np.load(RUN_DIR / "logL.npy")
idx_sparse = np.load(RUN_DIR / "idx_sparse.npy")
t_days = np.load(RUN_DIR / "t_days.npy")

with open(RUN_DIR / "theta_MAP.json") as f:
    theta_MAP = np.array(json.load(f)["theta_full"])
with open(RUN_DIR / "theta_mean.json") as f:
    theta_mean = np.array(json.load(f)["theta_full"])

phi_init = config.get("phi_init", 0.2)
if isinstance(phi_init, list):
    phi_init = np.array(phi_init, dtype=np.float64)

# ============================================================
# Initialize solver & run simulations
# ============================================================
print("Initializing solver...")
solver = BiofilmNewtonSolver5S(
    dt=config["dt"],
    maxtimestep=config["maxtimestep"],
    c_const=config["c_const"],
    alpha_const=config["alpha_const"],
    phi_init=phi_init,
)

print("Solving MAP estimate...")
t_fit, x_map = solver.solve(theta_MAP)
phibar_map = compute_phibar(x_map, ACTIVE_SPECIES)
t_plot = model_time_to_days(t_fit, t_days)

print("Solving Mean estimate...")
_, x_mean = solver.solve(theta_mean)
phibar_mean = compute_phibar(x_mean, ACTIVE_SPECIES)

print(f"Running {N_POSTERIOR_DRAWS} posterior draws...")
rng = np.random.default_rng(42)
# Handle case where samples might be fewer than N_POSTERIOR_DRAWS
n_samples = len(samples)
draw_size = min(N_POSTERIOR_DRAWS, n_samples)
draw_idx = rng.choice(n_samples, size=draw_size, replace=False)

phibar_draws = []
x_draws = []
for k, si in enumerate(draw_idx):
    _, x_k = solver.solve(samples[si])
    x_draws.append(x_k)
    phibar_draws.append(compute_phibar(x_k, ACTIVE_SPECIES))
    if (k + 1) % 25 == 0:
        print(f"  ... {k+1}/{len(draw_idx)} done")
phibar_draws = np.array(phibar_draws)  # (N, n_time, 5)
x_draws = np.array(x_draws)            # (N, n_time, 12)

print("All simulations complete. Generating figures...\n")


# ============================================================
# Fig A01 - Interaction matrix heatmap
# ============================================================
def fig_a01_interaction_heatmap():
    A_map, b_map = theta_to_matrices(theta_MAP)
    A_mean, b_mean = theta_to_matrices(theta_mean)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={"width_ratios": [5, 5, 1.2]})

    vmax = max(abs(A_map).max(), abs(A_mean).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, A, title in [(axes[0], A_map, "MAP Estimate"),
                          (axes[1], A_mean, "Posterior Mean")]:
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(SPECIES_SHORT, fontsize=12)
        ax.set_yticklabels(SPECIES_SHORT, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(A[i,j]) > vmax*0.6 else "black")

    # Decay vector bar chart
    ax3 = axes[2]
    y_pos = np.arange(5)
    ax3.barh(y_pos, b_map, 0.35, label="MAP", color="steelblue", alpha=0.8)
    ax3.barh(y_pos + 0.35, b_mean, 0.35, label="Mean", color="coral", alpha=0.8)
    ax3.set_yticks(y_pos + 0.175)
    ax3.set_yticklabels(SPECIES_SHORT, fontsize=11)
    ax3.set_xlabel("b (decay)", fontsize=12)
    ax3.set_title("Decay", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.invert_yaxis()

    fig.suptitle(f"{TITLE_PREFIX}: Species Interaction Matrix A & Decay b",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "Fig_A01_interaction_matrix_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A02 - Per-species panel
# ============================================================
def fig_a02_per_species_panel():
    q05 = np.nanpercentile(phibar_draws, 5, axis=0)
    q25 = np.nanpercentile(phibar_draws, 25, axis=0)
    q50 = np.nanpercentile(phibar_draws, 50, axis=0)
    q75 = np.nanpercentile(phibar_draws, 75, axis=0)
    q95 = np.nanpercentile(phibar_draws, 95, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, sp in enumerate(ACTIVE_SPECIES):
        ax = axes_flat[i]
        c = COLORS[sp]
        ax.fill_between(t_plot, q05[:, i], q95[:, i], alpha=0.15, color=c, label="5-95%")
        ax.fill_between(t_plot, q25[:, i], q75[:, i], alpha=0.3, color=c, label="25-75%")
        ax.plot(t_plot, q50[:, i], color=c, linewidth=2, label="Median")
        ax.plot(t_plot, phibar_map[:, i], "--", color="black", linewidth=1.5, label="MAP")
        ax.scatter(t_days, data[:, i], s=80, color=c, edgecolors="black",
                   zorder=10, linewidth=1.5, label="Data")
        ax.set_title(f"{SPECIES_NAMES[i]}", fontsize=14)
        ax.set_xlabel("Days", fontsize=12)
        ax.set_ylabel(r"$\bar{\varphi}$", fontsize=13)
        ax.set_xticks(t_days)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(t_days.min() - 1, t_days.max() + 1)

    axes_flat[5].axis("off")
    fig.suptitle(f"{TITLE_PREFIX}: Per-Species Posterior Predictive Fits",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A02_per_species_panel.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A03 - State decomposition (phi, psi, phibar)
# ============================================================
def fig_a03_state_decomposition():
    fig, axes = plt.subplots(3, 5, figsize=(25, 12), sharex=True)

    row_labels = [r"$\varphi_i$ (space occupied)",
                  r"$\psi_i$ (living fraction)",
                  r"$\bar{\varphi}_i = \varphi_i \cdot \psi_i$"]

    for i, sp in enumerate(ACTIVE_SPECIES):
        c = COLORS[sp]
        # Row 0: phi
        ax = axes[0, i]
        phi_map = x_map[:, sp]
        phi_q05 = np.nanpercentile(x_draws[:, :, sp], 5, axis=0)
        phi_q95 = np.nanpercentile(x_draws[:, :, sp], 95, axis=0)
        ax.fill_between(t_plot, phi_q05, phi_q95, alpha=0.2, color=c)
        ax.plot(t_plot, phi_map, color=c, linewidth=2)
        ax.set_title(f"{SPECIES_NAMES[i]}", fontsize=12, fontweight="bold")
        if i == 0:
            ax.set_ylabel(row_labels[0], fontsize=11)

        # Row 1: psi
        ax = axes[1, i]
        psi_idx = 6 + sp
        psi_map = x_map[:, psi_idx]
        psi_q05 = np.nanpercentile(x_draws[:, :, psi_idx], 5, axis=0)
        psi_q95 = np.nanpercentile(x_draws[:, :, psi_idx], 95, axis=0)
        ax.fill_between(t_plot, psi_q05, psi_q95, alpha=0.2, color=c)
        ax.plot(t_plot, psi_map, color=c, linewidth=2)
        if i == 0:
            ax.set_ylabel(row_labels[1], fontsize=11)

        # Row 2: phibar
        ax = axes[2, i]
        q05 = np.nanpercentile(phibar_draws[:, :, i], 5, axis=0)
        q95 = np.nanpercentile(phibar_draws[:, :, i], 95, axis=0)
        ax.fill_between(t_plot, q05, q95, alpha=0.2, color=c)
        ax.plot(t_plot, phibar_map[:, i], color=c, linewidth=2)
        ax.scatter(t_days, data[:, i], s=60, color=c, edgecolors="black", zorder=10)
        ax.set_xlabel("Days", fontsize=11)
        ax.set_xticks(t_days)
        if i == 0:
            ax.set_ylabel(row_labels[2], fontsize=11)

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.3)

    fig.suptitle(rf"{TITLE_PREFIX}: State Decomposition ($\varphi$, $\psi$, $\bar{{\varphi}}$)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A03_state_decomposition.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A04 - Species composition stacked area
# ============================================================
def fig_a04_stacked_area():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # MAP estimate
    ax = axes[0]
    phibar_stack = phibar_map.copy()
    phibar_stack = np.clip(phibar_stack, 0, None)
    total = phibar_stack.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    fractions = phibar_stack / total
    ax.stackplot(t_plot, fractions.T, labels=SPECIES_SHORT, colors=COLORS, alpha=0.8)
    for i in range(5):
        obs_frac = data[:, i] / data.sum(axis=1)
        ax.scatter(t_days, obs_frac, color=COLORS[i], edgecolors="black",
                   s=60, zorder=10, linewidth=1.5)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Relative Abundance", fontsize=12)
    ax.set_title("MAP Estimate", fontsize=14, fontweight="bold")
    ax.set_xlim(t_days.min() - 1, t_days.max() + 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(t_days)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Observed data
    ax = axes[1]
    obs_total = data.sum(axis=1, keepdims=True)
    obs_total[obs_total == 0] = 1
    obs_frac = data / obs_total
    width = 1.5
    bottom = np.zeros(len(t_days))
    for i in range(5):
        ax.bar(t_days, obs_frac[:, i], width, bottom=bottom,
               color=COLORS[i], label=SPECIES_SHORT[i], alpha=0.8, edgecolor="white")
        bottom += obs_frac[:, i]
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Relative Abundance", fontsize=12)
    ax.set_title("Observed Data", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xticks(t_days)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{TITLE_PREFIX}: Species Composition Over Time",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A04_species_composition.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A05 - Parameter violin plots by block
# ============================================================
def fig_a05_violin_plots():
    blocks = {
        "M1: So-An\nSelf + Decay": list(range(0, 5)),
        "M2: Vd-Fn\nSelf + Decay": list(range(5, 10)),
        "M3: Cross\n(So,An)-(Vd,Fn)": list(range(10, 14)),
        "M4: Pg\nSelf + Decay": list(range(14, 16)),
        "M5: Pg\nCross-interactions": list(range(16, 20)),
    }

    fig, axes = plt.subplots(1, 5, figsize=(24, 7),
                             gridspec_kw={"width_ratios": [5, 5, 4, 2, 4]})

    block_items = list(blocks.items())
    block_clr = list(BLOCK_COLORS.values())

    for bi, (bname, indices) in enumerate(block_items):
        ax = axes[bi]
        n_p = len(indices)
        parts = ax.violinplot(
            [samples[:, j] for j in indices],
            positions=range(n_p),
            showmeans=True, showmedians=True, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(block_clr[bi])
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("black")

        # MAP markers
        for k, j in enumerate(indices):
            ax.plot(k, theta_MAP[j], "D", color="green", markersize=8, zorder=10)

        ax.set_xticks(range(n_p))
        ax.set_xticklabels([THETA_NAMES[j] for j in indices], fontsize=10, rotation=45, ha="right")
        ax.set_title(bname, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if bi == 0:
            ax.set_ylabel("Parameter Value", fontsize=12)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Mean"),
        Line2D([0], [0], color="black", lw=2, label="Median"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="green",
               markersize=8, label="MAP"),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=10, loc="upper right")

    fig.suptitle(f"{TITLE_PREFIX}: Posterior Parameter Distributions by Block",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A05_parameter_violins.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A06 - Parameter correlation matrix
# ============================================================
def fig_a06_correlation_matrix():
    corr = np.corrcoef(samples.T)  # (20, 20)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=12)

    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels(THETA_NAMES, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(THETA_NAMES, fontsize=9)

    # Annotate strong correlations
    for i in range(20):
        for j in range(20):
            if abs(corr[i, j]) > 0.3 and i != j:
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr[i,j]) > 0.6 else "black")

    # Block separators
    for pos in [5, 10, 14, 16]:
        ax.axhline(pos - 0.5, color="black", linewidth=1.5)
        ax.axvline(pos - 0.5, color="black", linewidth=1.5)

    ax.set_title(f"{TITLE_PREFIX}: Posterior Parameter Correlation Matrix",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A06_correlation_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A07 - Log-likelihood landscape
# ============================================================
def fig_a07_log_likelihood():
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.5])

    # Panel 1: logL histogram
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(logL, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.axvline(logL.max(), color="red", linestyle="--", linewidth=2,
                label=f"Max logL = {logL.max():.2f}")
    ax1.axvline(logL.mean(), color="orange", linestyle="--", linewidth=2,
                label=f"Mean logL = {logL.mean():.2f}")
    ax1.set_xlabel("Log-Likelihood", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Posterior Log-Likelihood Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: logL vs top 2 PCA components
    ax2 = fig.add_subplot(gs[1])
    # Simple PCA via covariance
    samp_centered = samples - samples.mean(axis=0)
    cov = np.cov(samp_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = samp_centered @ eigvecs[:, -1]
    pc2 = samp_centered @ eigvecs[:, -2]
    sc = ax2.scatter(pc1, pc2, c=logL, cmap="viridis", s=8, alpha=0.7)
    fig.colorbar(sc, ax=ax2, label="logL", shrink=0.8)
    ax2.set_xlabel("PC1", fontsize=12)
    ax2.set_ylabel("PC2", fontsize=12)
    ax2.set_title("Posterior in PCA Space (colored by logL)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Explained variance
    ax3 = fig.add_subplot(gs[2])
    explained = eigvals[::-1] / eigvals.sum() * 100
    cum = np.cumsum(explained)
    ax3.bar(range(1, 11), explained[:10], color="steelblue", alpha=0.7, label="Individual")
    ax3.plot(range(1, 11), cum[:10], "ro-", label="Cumulative")
    ax3.set_xlabel("Principal Component", fontsize=12)
    ax3.set_ylabel("Variance Explained (%)", fontsize=12)
    ax3.set_title("PCA Variance", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"{TITLE_PREFIX}: Log-Likelihood & Posterior Structure",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "Fig_A07_loglikelihood_landscape.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A08 - Posterior predictive check at observation times
# ============================================================
def fig_a08_ppc():
    """Box plots of posterior predictions vs observed data at each time point."""
    n_obs = len(t_days)

    fig, axes = plt.subplots(1, n_obs, figsize=(4 * n_obs, 7), sharey=True)
    if n_obs == 1:
        axes = [axes]

    for t_idx in range(n_obs):
        ax = axes[t_idx]
        obs_idx = idx_sparse[t_idx]

        # Posterior predictions at this time
        pred_at_t = phibar_draws[:, obs_idx, :]  # (N_draws, 5)

        bp = ax.boxplot(
            [pred_at_t[:, sp] for sp in range(5)],
            positions=range(5),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)

        # Overlay observed data
        for sp in range(5):
            ax.plot(sp, data[t_idx, sp], "ko", markersize=10, zorder=10)
            ax.plot(sp, data[t_idx, sp], "o", color=COLORS[sp],
                    markersize=7, zorder=11)

        ax.set_xticks(range(5))
        ax.set_xticklabels(SPECIES_SHORT, fontsize=11)
        ax.set_title(f"Day {t_days[t_idx]}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(r"$\bar{\varphi}$", fontsize=14)
    fig.suptitle(f"{TITLE_PREFIX}: Posterior Predictive Check (boxes = model, dots = data)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A08_posterior_predictive_check.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A09 - MAP vs Mean fit comparison
# ============================================================
def fig_a09_map_vs_mean():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: trajectories overlaid
    ax = axes[0]
    for i, sp in enumerate(ACTIVE_SPECIES):
        c = COLORS[sp]
        ax.plot(t_plot, phibar_map[:, i], "-", color=c, linewidth=2.5,
                label=f"{SPECIES_SHORT[i]} MAP")
        ax.plot(t_plot, phibar_mean[:, i], "--", color=c, linewidth=1.5,
                alpha=0.8, label=f"{SPECIES_SHORT[i]} Mean")
        ax.scatter(t_days, data[:, i], color=c, edgecolors="black",
                   s=60, zorder=10, linewidth=1.2)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel(r"$\bar{\varphi}$", fontsize=13)
    ax.set_title("MAP (solid) vs Mean (dashed) Fits", fontsize=14, fontweight="bold")
    ax.set_xticks(t_days)
    ax.set_xlim(t_days.min() - 1, t_days.max() + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")

    # Panel 2: per-species RMSE comparison
    ax = axes[1]
    with open(RUN_DIR / "fit_metrics.json") as f:
        fm = json.load(f)
    rmse_map_sp = fm["MAP"]["rmse_per_species"]
    rmse_mean_sp = fm["Mean"]["rmse_per_species"]
    x_pos = np.arange(5)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, rmse_map_sp, width, label="MAP",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, rmse_mean_sp, width, label="Mean",
                   color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SPECIES_SHORT, fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Per-Species RMSE: MAP vs Mean", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add total RMSE text
    ax.text(0.98, 0.95,
            f"Total RMSE\nMAP:  {fm['MAP']['rmse_total']:.4f}\nMean: {fm['Mean']['rmse_total']:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle(f"{TITLE_PREFIX}: MAP vs Posterior Mean Comparison",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "Fig_A09_MAP_vs_Mean_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Fig A10 - TMCMC convergence dashboard
# ============================================================
def fig_a10_convergence_dashboard():
    diag_dir = RUN_DIR / "diagnostics_tables"

    # Load beta schedule
    beta_data = {}
    # Use dynamic filename: e.g. "Commensal_HOBIC_beta_schedule.csv"
    csv_path = diag_dir / f"{FILE_PREFIX}_beta_schedule.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                chain = int(row["chain"])
                beta_data.setdefault(chain, []).append(float(row["beta"]))

    # Load acceptance rates
    acc_data = {}
    csv_path = diag_dir / f"{FILE_PREFIX}_acceptance_rate.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                chain = int(row["chain"])
                acc_data.setdefault(chain, []).append(float(row["accept_rate"]))

    # Load stage summary
    stage_data = {}
    csv_path = diag_dir / f"{FILE_PREFIX}_stage_summary.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                chain = int(row["chain"])
                stage_data.setdefault(chain, []).append({
                    "stage": int(row["stage"]),
                    "beta": float(row["beta"]),
                    "delta_beta": float(row["delta_beta"]),
                    "ess": float(row["ess"]),
                    "acc_rate": float(row["accept_rate"]),
                    "logL_min": float(row["logL_min"]),
                    "logL_max": float(row["logL_max"]),
                })

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Beta schedule
    ax1 = fig.add_subplot(gs[0, 0])
    for chain, betas in beta_data.items():
        ax1.plot(range(len(betas)), betas, "o-", label=f"Chain {chain}", linewidth=2, markersize=6)
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.5, label=r"$\beta=1$")
    ax1.set_xlabel("Stage", fontsize=12)
    ax1.set_ylabel(r"$\beta$", fontsize=14)
    ax1.set_title(r"Tempering Schedule ($\beta$)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Acceptance rate
    ax2 = fig.add_subplot(gs[0, 1])
    for chain, acc in acc_data.items():
        ax2.plot(range(len(acc)), acc, "s-", label=f"Chain {chain}", linewidth=2, markersize=6)
    ax2.axhline(0.234, color="gray", linestyle=":", label="Optimal (0.234)")
    ax2.set_xlabel("Stage", fontsize=12)
    ax2.set_ylabel("Acceptance Rate", fontsize=12)
    ax2.set_title("MCMC Acceptance Rate", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Panel 3: ESS
    ax3 = fig.add_subplot(gs[0, 2])
    for chain, stages in stage_data.items():
        ess_vals = [s["ess"] for s in stages]
        ax3.plot(range(len(ess_vals)), ess_vals, "^-", label=f"Chain {chain}", linewidth=2, markersize=6)
    ax3.axhline(500, color="red", linestyle="--", alpha=0.5, label="Target ESS")
    ax3.set_xlabel("Stage", fontsize=12)
    ax3.set_ylabel("ESS", fontsize=12)
    ax3.set_title("Effective Sample Size", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: LogL range per stage
    ax4 = fig.add_subplot(gs[1, 0])
    for chain, stages in stage_data.items():
        ll_min = [s["logL_min"] for s in stages]
        ll_max = [s["logL_max"] for s in stages]
        x = list(range(len(stages)))
        ax4.fill_between(x, ll_min, ll_max, alpha=0.3, label=f"Chain {chain}")
        ax4.plot(x, ll_max, "o-", markersize=4, linewidth=1.5)
    ax4.set_xlabel("Stage", fontsize=12)
    ax4.set_ylabel("logL", fontsize=12)
    ax4.set_title("Log-Likelihood Range per Stage", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Summary info card
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    # Get elapsed from results_summary
    with open(RUN_DIR / "results_summary.json") as f:
        rs = json.load(f)
    elapsed = rs.get("elapsed_time", 0)

    summary_text = (
        f"{TITLE_PREFIX} Run Summary\n"
        f"{'='*35}\n"
        f"Particles:     {config['n_particles']}\n"
        f"Max Stages:    {config['n_stages']}\n"
        f"Chains:        {config['n_chains']}\n"
        f"Parameters:    20\n"
        f"Observations:  {len(t_days)} time points\n"
        f"{'='*35}\n"
        f"Elapsed Time:  {elapsed/3600:.1f} hours\n"
        f"MAP RMSE:      {fm['MAP']['rmse_total']:.5f}\n"
        f"Mean RMSE:     {fm['Mean']['rmse_total']:.5f}\n"
        f"Max |logL|:    {logL.max():.3f}\n"
        f"Converged:     {rs.get('converged', 'N/A')}\n"
    )
    ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=12, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Panel 6: delta_beta per stage
    ax6 = fig.add_subplot(gs[1, 2])
    for chain, stages in stage_data.items():
        db = [s["delta_beta"] for s in stages]
        ax6.plot(range(len(db)), db, "D-", label=f"Chain {chain}", linewidth=2, markersize=6)
    ax6.set_xlabel("Stage", fontsize=12)
    ax6.set_ylabel(r"$\Delta\beta$", fontsize=14)
    ax6.set_title(r"Step Size $\Delta\beta$ per Stage", fontsize=13, fontweight="bold")
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f"{TITLE_PREFIX}: TMCMC Convergence Dashboard",
                 fontsize=16, fontweight="bold")
    path = OUT_DIR / "Fig_A10_convergence_dashboard.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    with open(RUN_DIR / "fit_metrics.json") as f:
        fm = json.load(f)

    fig_a01_interaction_heatmap()
    fig_a02_per_species_panel()
    fig_a03_state_decomposition()
    fig_a04_stacked_area()
    fig_a05_violin_plots()
    fig_a06_correlation_matrix()
    fig_a07_log_likelihood()
    fig_a08_ppc()
    fig_a09_map_vs_mean()
    fig_a10_convergence_dashboard()

    print(f"\nAll 10 extra figures saved to {OUT_DIR}/")
