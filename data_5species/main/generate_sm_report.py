#!/usr/bin/env python3
"""
generate_sm_report.py — Sanz-Martin 2022 TMCMC report.
Same visual style as nishioka_paper_publish.pdf (regenerate_all_figures.py).
"""
import sys, os, json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))

from improved_5species_jit import BiofilmNewtonSolver5S

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_dist

# ═══════════════════════════════════════════════════════════════
# UNIFIED PUBLICATION STYLE — matches 11pt lmodern body text
# ═══════════════════════════════════════════════════════════════
STYLE = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}
plt.rcParams.update(STYLE)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_ITALIC = [
    r"$\it{S.\ oralis}$", r"$\it{A.\ naeslundii}$", r"$\it{V.\ dispar}$",
    r"$\it{F.\ nucleatum}$", r"$\it{P.\ gingivalis}$",
]
SM_DAYS = np.array([0.25, 1, 3, 7, 14, 21])
SM_PLANKTONIC = np.array([
    [0.9868, 0.0132, 0.0000, 0.0000, 0.0000],
    [0.4737, 0.0526, 0.1053, 0.3684, 0.0000],
    [0.1053, 0.0526, 0.1053, 0.7368, 0.0000],
    [0.1579, 0.0526, 0.1053, 0.6842, 0.0000],
    [0.1053, 0.0526, 0.1053, 0.3895, 0.3474],
    [0.0526, 0.0526, 0.1053, 0.2105, 0.5789],
])
SM_ADHERENT = np.array([
    [0.9783, 0.0217, 0.0000, 0.0000, 0.0000],
    [0.6842, 0.0526, 0.0526, 0.2105, 0.0000],
    [0.4211, 0.0526, 0.1053, 0.4211, 0.0000],
    [0.7368, 0.0526, 0.0526, 0.1579, 0.0000],
    [0.6316, 0.0526, 0.0526, 0.1579, 0.1053],
    [0.3478, 0.0843, 0.0527, 0.2002, 0.3150],
])

COND_LABELS = {"plan": "Planktonic", "adh": "Adherent"}
COND_COLORS_SM = {"plan": "#2166AC", "adh": "#B2182B"}
SP_COLORS = ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"]

PARAM_LABELS_20 = [
    r"$a_{11}$", r"$a_{12}$", r"$a_{22}$", r"$\mu_1$", r"$\mu_2$",
    r"$a_{33}$", r"$a_{34}$", r"$a_{44}$", r"$\mu_3$", r"$\mu_4$",
    r"$a_{13}$", r"$a_{14}$", r"$a_{23}$", r"$a_{24}$",
    r"$a_{55}$", r"$\mu_5$",
    r"$a_{15}$", r"$a_{25}$", r"$a_{35}$", r"$a_{45}$",
]
MU_IDX = [3, 4, 8, 9, 15]
MU_LABELS = [r"$\mu_1$ (So)", r"$\mu_2$ (An)", r"$\mu_3$ (Vd)", r"$\mu_4$ (Fn)", r"$\mu_5$ (Pg)"]
INTERACTION_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]

# E(DI) model
E_MAX, E_MIN, DI_EXP = 1000.0, 10.0, 2.0


def run_ode(theta, ic, n_steps=2500, dt=1e-4):
    phi_init = np.clip(ic, 0.001, 0.99).astype(np.float64)
    phi_init = phi_init / phi_init.sum() * 0.999
    solver = BiofilmNewtonSolver5S(
        dt=dt, maxtimestep=n_steps, phi_init=phi_init,
        K_hill=0.05, n_hill=2.0, c_const=25.0, alpha_const=100.0,
        active_species=[0, 1, 2, 3, 4],
    )
    g_arr = solver.run_deterministic(theta)
    if isinstance(g_arr, tuple):
        g_arr = g_arr[1]
    return g_arr[:, 0:5]


def theta_to_A(theta):
    A = np.zeros((5, 5))
    A[0,0]=theta[0]; A[0,1]=theta[1]; A[1,1]=theta[2]
    A[2,2]=theta[5]; A[2,3]=theta[6]; A[3,3]=theta[7]
    A[0,2]=theta[10]; A[0,3]=theta[11]; A[1,2]=theta[12]; A[1,3]=theta[13]
    A[4,4]=theta[14]
    A[0,4]=theta[16]; A[1,4]=theta[17]; A[2,4]=theta[18]; A[3,4]=theta[19]
    for i in range(5):
        for j in range(i+1, 5):
            A[j,i] = A[i,j]
    return A


def compute_di(phi):
    eps = 1e-12
    p = np.clip(phi / (phi.sum() + eps), eps, 1.0)
    return 1.0 - (-np.sum(p * np.log(p))) / np.log(5)


def compute_metrics(obs, pred):
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    cos_sim = 1.0 - cosine_dist(obs.flatten(), pred.flatten())
    rho, _ = spearmanr(obs.flatten(), pred.flatten())
    return {"rmse": rmse, "cosine": cos_sim, "spearman": rho}


def load_run(run_dir):
    with open(run_dir / "theta_MAP.json") as f:
        raw = json.load(f)
    theta = np.array([raw[str(i)] for i in range(20)])
    samples = np.load(run_dir / "samples.npy")
    logL = np.load(run_dir / "logL.npy")
    return theta, samples, logL


# ═══════════════════════════════════════════════════════════════
# FIG 1: 5-panel Posterior Predictive (paper style)
# ═══════════════════════════════════════════════════════════════
def generate_fig1(theta, samples, obs, ic, label, colors, fig_dir):
    """2 rows × 5 cols: MAP + CI bands + data scatter."""
    model_days = np.linspace(0, 22, 2501)

    # MAP trajectory
    traj_map = run_ode(theta, ic)

    # Posterior predictive (subsample 100)
    n_sub = min(100, len(samples))
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(samples), n_sub, replace=False)
    trajs = np.zeros((n_sub, 2501, 5))
    for k, si in enumerate(idx_sub):
        trajs[k] = run_ode(samples[si], ic)

    q10 = np.percentile(trajs, 10, axis=0)
    q25 = np.percentile(trajs, 25, axis=0)
    q75 = np.percentile(trajs, 75, axis=0)
    q90 = np.percentile(trajs, 90, axis=0)

    # Metrics
    idx_sparse = np.round(SM_DAYS / 22 * 2500).astype(int)
    idx_sparse = np.clip(idx_sparse, 0, 2500)
    pred = traj_map[idx_sparse]
    pred = pred / pred.sum(axis=1, keepdims=True)
    m = compute_metrics(obs, pred)

    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.8), sharex=True, sharey=True)

    for col_idx in range(5):
        ax = axes[col_idx]
        ax.fill_between(model_days, q10[:, col_idx], q90[:, col_idx],
                        color=colors[col_idx], alpha=0.12)
        ax.fill_between(model_days, q25[:, col_idx], q75[:, col_idx],
                        color=colors[col_idx], alpha=0.25)
        ax.plot(model_days, traj_map[:, col_idx], color=colors[col_idx], lw=1.3, zorder=3)
        ax.scatter(SM_DAYS, obs[:, col_idx], s=18, color=colors[col_idx],
                   edgecolors="k", linewidths=0.4, zorder=5)
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.set_title(SPECIES_ITALIC[col_idx], fontsize=8)
        ax.set_xlabel("Day")
        if col_idx == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$")

    axes[4].text(0.95, 0.92,
                 f"RMSE {m['rmse']:.3f}\nCos {m['cosine']:.3f}",
                 transform=axes[4].transAxes, fontsize=6.5, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    # Legend
    handles = [
        Line2D([0], [0], color="gray", lw=1.3, label="MAP"),
        Patch(facecolor="gray", alpha=0.25, label="50% CI"),
        Patch(facecolor="gray", alpha=0.12, label="80% CI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=4, label="SM data"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=5.5,
                   handlelength=1.0, handletextpad=0.3, borderpad=0.3)

    fig.suptitle(f"{label}", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=0.2)

    tag = label.lower().replace(" ", "_")
    fig.savefig(fig_dir / f"fig1_{tag}_5panel.pdf")
    fig.savefig(fig_dir / f"fig1_{tag}_5panel.png")
    plt.close(fig)
    return m, pred


# ═══════════════════════════════════════════════════════════════
# FIG 2: Interaction Matrix Heatmaps (A)
# ═══════════════════════════════════════════════════════════════
def generate_fig2_matrix(theta_plan, theta_adh, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))

    A_plan = theta_to_A(theta_plan)
    A_adh = theta_to_A(theta_adh)
    vmax = max(np.abs(A_plan).max(), np.abs(A_adh).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, A, title, color in zip(axes, [A_plan, A_adh],
                                    ["Planktonic", "Adherent"],
                                    ["#2166AC", "#B2182B"]):
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
        for i in range(5):
            for j in range(5):
                val = A[i, j]
                c = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color=c)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(SPECIES, fontsize=7)
        ax.set_yticklabels(SPECIES, fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20, pad=0.03)
    cbar.set_label(r"$A_{ij}$", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_interaction_matrix.pdf")
    fig.savefig(fig_dir / "fig2_interaction_matrix.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 3: Growth rate (μ) violin comparison
# ═══════════════════════════════════════════════════════════════
def generate_fig3_mu(samples_plan, samples_adh, theta_plan, theta_adh, fig_dir):
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.0), sharey=True)

    for sp_idx in range(5):
        ax = axes[sp_idx]
        pidx = MU_IDX[sp_idx]
        data = [samples_plan[:, pidx], samples_adh[:, pidx]]

        parts = ax.violinplot(data, positions=[0, 1], showmeans=False,
                              showextrema=False, widths=0.7)
        for i, pc in enumerate(parts["bodies"]):
            c = COND_COLORS_SM["plan"] if i == 0 else COND_COLORS_SM["adh"]
            pc.set_facecolor(c)
            pc.set_edgecolor("k")
            pc.set_linewidth(0.4)
            pc.set_alpha(0.7)

        # MAP stars
        ax.scatter(0, theta_plan[pidx], marker="*", s=40,
                   color=COND_COLORS_SM["plan"], edgecolors="k", linewidths=0.3, zorder=5)
        ax.scatter(1, theta_adh[pidx], marker="*", s=40,
                   color=COND_COLORS_SM["adh"], edgecolors="k", linewidths=0.3, zorder=5)

        ax.set_title(MU_LABELS[sp_idx], fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Plan", "Adh"], fontsize=7)
        ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
        ax.grid(True, axis="y")
        if sp_idx == 0:
            ax.set_ylabel("Growth rate")

    handles = [
        Patch(facecolor=COND_COLORS_SM["plan"], edgecolor="k", alpha=0.7, label="Planktonic"),
        Patch(facecolor=COND_COLORS_SM["adh"], edgecolor="k", alpha=0.7, label="Adherent"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=8, label="MAP"),
    ]
    axes[4].legend(handles=handles, loc="upper right", fontsize=6, borderpad=0.3)

    fig.tight_layout(w_pad=0.3)
    fig.savefig(fig_dir / "fig3_growth_rates.pdf")
    fig.savefig(fig_dir / "fig3_growth_rates.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 4: DI + E trajectory
# ═══════════════════════════════════════════════════════════════
def generate_fig4_di_E(obs_plan, pred_plan, obs_adh, pred_adh, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    for obs, pred, label, color in [
        (obs_plan, pred_plan, "Planktonic", COND_COLORS_SM["plan"]),
        (obs_adh, pred_adh, "Adherent", COND_COLORS_SM["adh"]),
    ]:
        di_obs = np.array([compute_di(obs[i]) for i in range(6)])
        di_pred = np.array([compute_di(pred[i]) for i in range(6)])
        E_obs = E_MIN + (E_MAX - E_MIN) * (1 - di_obs) ** DI_EXP
        E_pred = E_MIN + (E_MAX - E_MIN) * (1 - di_pred) ** DI_EXP

        axes[0].plot(SM_DAYS, di_obs, "o-", color=color, lw=1.2, ms=5,
                     markeredgecolor="k", markeredgewidth=0.3, label=f"{label} obs")
        axes[0].plot(SM_DAYS, di_pred, "s--", color=color, lw=1.2, ms=4,
                     markeredgecolor="k", markeredgewidth=0.3, alpha=0.7, label=f"{label} pred")

        axes[1].plot(SM_DAYS, E_obs, "o-", color=color, lw=1.2, ms=5,
                     markeredgecolor="k", markeredgewidth=0.3, label=f"{label} obs")
        axes[1].plot(SM_DAYS, E_pred, "s--", color=color, lw=1.2, ms=4,
                     markeredgecolor="k", markeredgewidth=0.3, alpha=0.7, label=f"{label} pred")

    axes[0].set_xlabel("Day"); axes[0].set_ylabel("DI")
    axes[0].set_title("Dysbiosis Index", fontweight="bold")
    axes[0].legend(fontsize=6, frameon=True, edgecolor="0.8")
    axes[0].set_ylim(0, 1.05); axes[0].grid(True)

    axes[1].set_xlabel("Day"); axes[1].set_ylabel("$E$ (Pa)")
    axes[1].set_title("Young's modulus $E(\\mathrm{DI})$", fontweight="bold")
    axes[1].legend(fontsize=6, frameon=True, edgecolor="0.8")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_di_E_trajectory.pdf")
    fig.savefig(fig_dir / "fig4_di_E_trajectory.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 5: Scatter + Day 21
# ═══════════════════════════════════════════════════════════════
def generate_fig5_scatter(obs_plan, pred_plan, obs_adh, pred_adh,
                          m_plan, m_adh, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    for ax, obs, pred, label, m in [
        (axes[0], obs_plan, pred_plan, "Planktonic", m_plan),
        (axes[1], obs_adh, pred_adh, "Adherent", m_adh),
    ]:
        for j in range(5):
            ax.scatter(obs[:, j], pred[:, j], c=SP_COLORS[j], s=25, zorder=3,
                       edgecolors="k", linewidths=0.3, label=SPECIES[j])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.text(0.05, 0.92, f"RMSE={m['rmse']:.3f}\nCos={m['cosine']:.3f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.7))
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")

    # Day 21 bar chart
    ax = axes[2]
    x = np.arange(5)
    w = 0.2
    ax.bar(x - 1.5*w, obs_plan[-1], w, color=COND_COLORS_SM["plan"], alpha=0.4, edgecolor=COND_COLORS_SM["plan"], label="Plan obs")
    ax.bar(x - 0.5*w, pred_plan[-1], w, color=COND_COLORS_SM["plan"], alpha=0.9, label="Plan pred")
    ax.bar(x + 0.5*w, obs_adh[-1], w, color=COND_COLORS_SM["adh"], alpha=0.4, edgecolor=COND_COLORS_SM["adh"], label="Adh obs")
    ax.bar(x + 1.5*w, pred_adh[-1], w, color=COND_COLORS_SM["adh"], alpha=0.9, label="Adh pred")
    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES, fontsize=7)
    ax.set_ylabel("Fraction")
    ax.set_title("Day 21 equilibrium", fontweight="bold")
    ax.legend(fontsize=5.5, frameon=True, edgecolor="0.8", ncol=2)

    fig.tight_layout(w_pad=0.4)
    fig.savefig(fig_dir / "fig5_scatter_day21.pdf")
    fig.savefig(fig_dir / "fig5_scatter_day21.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 6: Posterior violin (all 20 params)
# ═══════════════════════════════════════════════════════════════
def generate_fig6_violin20(samples_plan, samples_adh, theta_plan, theta_adh, fig_dir):
    groups = [
        ("So--An block", [0, 1, 2, 3, 4]),
        ("Vd--Fn block", [5, 6, 7, 8, 9]),
        ("Cross-block", [10, 11, 12, 13, 14]),
        ("Pg + cross", [15, 16, 17, 18, 19]),
    ]
    fig, axes = plt.subplots(4, 5, figsize=(7.2, 7.0))

    for grp_idx, (grp_label, param_indices) in enumerate(groups):
        for col_idx, pidx in enumerate(param_indices):
            ax = axes[grp_idx, col_idx]
            data = [samples_plan[:, pidx], samples_adh[:, pidx]]

            parts = ax.violinplot(data, positions=[0, 1], showmeans=False,
                                  showextrema=False, widths=0.7)
            for i, pc in enumerate(parts["bodies"]):
                c = COND_COLORS_SM["plan"] if i == 0 else COND_COLORS_SM["adh"]
                pc.set_facecolor(c)
                pc.set_edgecolor("k")
                pc.set_linewidth(0.4)
                pc.set_alpha(0.7)

            ax.scatter(0, theta_plan[pidx], marker="*", s=30,
                       color=COND_COLORS_SM["plan"], edgecolors="k", linewidths=0.3, zorder=5)
            ax.scatter(1, theta_adh[pidx], marker="*", s=30,
                       color=COND_COLORS_SM["adh"], edgecolors="k", linewidths=0.3, zorder=5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["P", "A"], fontsize=6)
            ax.set_title(PARAM_LABELS_20[pidx], fontsize=8)
            ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
            ax.grid(True, axis="y")
            if col_idx == 0:
                ax.set_ylabel(grp_label, fontsize=7)

    handles = [
        Patch(facecolor=COND_COLORS_SM["plan"], edgecolor="k", alpha=0.7, label="Planktonic"),
        Patch(facecolor=COND_COLORS_SM["adh"], edgecolor="k", alpha=0.7, label="Adherent"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=8, label="MAP"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor="0.8",
               fontsize=7, borderpad=0.3)

    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    fig.savefig(fig_dir / "fig6_posterior_violin_20.pdf")
    fig.savefig(fig_dir / "fig6_posterior_violin_20.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 7: Stacked area succession
# ═══════════════════════════════════════════════════════════════
def generate_fig7_stacked(obs_plan, pred_plan, obs_adh, pred_adh, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    for row, (obs, pred, label) in enumerate([
        (obs_plan, pred_plan, "Planktonic"),
        (obs_adh, pred_adh, "Adherent"),
    ]):
        # Observed stacked area
        axes[row, 0].stackplot(SM_DAYS, obs.T, labels=SPECIES if row == 0 else None,
                               colors=SP_COLORS, alpha=0.7)
        axes[row, 0].set_title(f"{label} — Observed", fontweight="bold")
        axes[row, 0].set_ylabel(r"$\bar{\varphi}_i$")
        axes[row, 0].set_ylim(0, 1.05); axes[row, 0].set_xlim(0, 22)
        axes[row, 0].grid(True)

        # Predicted stacked area
        axes[row, 1].stackplot(SM_DAYS, pred.T, colors=SP_COLORS, alpha=0.7)
        axes[row, 1].set_title(f"{label} — MAP Predicted", fontweight="bold")
        axes[row, 1].set_ylim(0, 1.05); axes[row, 1].set_xlim(0, 22)
        axes[row, 1].grid(True)

    for ax in axes[1]:
        ax.set_xlabel("Day")
    axes[0, 0].legend(fontsize=6, loc="center right", frameon=False)

    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    fig.savefig(fig_dir / "fig7_stacked_area.pdf")
    fig.savefig(fig_dir / "fig7_stacked_area.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 8: Residuals
# ═══════════════════════════════════════════════════════════════
def generate_fig8_residuals(obs_plan, pred_plan, obs_adh, pred_adh, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    for ax, obs, pred, title in [
        (axes[0], obs_plan, pred_plan, "Planktonic"),
        (axes[1], obs_adh, pred_adh, "Adherent"),
    ]:
        for j in range(5):
            residual = pred[:, j] - obs[:, j]
            ax.plot(SM_DAYS, residual, color=SP_COLORS[j], marker="s", ms=4,
                    label=SPECIES[j], lw=1.0)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Day"); ax.set_ylabel("Residual (pred $-$ obs)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True)
    axes[0].legend(ncol=5, loc="upper center", frameon=False, fontsize=6)

    fig.tight_layout(w_pad=0.4)
    fig.savefig(fig_dir / "fig8_residuals.pdf")
    fig.savefig(fig_dir / "fig8_residuals.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 9: Continuous ODE trajectory
# ═══════════════════════════════════════════════════════════════
def generate_fig9_ode_continuous(traj_plan, traj_adh, fig_dir):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0))
    n_pts = traj_plan.shape[0]
    t_days = np.linspace(0, 22, n_pts)

    for ax, traj, obs, title in [
        (axes[0], traj_plan, SM_PLANKTONIC, "Planktonic"),
        (axes[1], traj_adh, SM_ADHERENT, "Adherent"),
    ]:
        traj_norm = traj / np.maximum(traj.sum(axis=1, keepdims=True), 1e-12)
        for j in range(5):
            ax.plot(t_days, traj_norm[:, j], color=SP_COLORS[j], lw=1.2,
                    label=SPECIES_ITALIC[j])
            ax.scatter(SM_DAYS, obs[:, j], c=SP_COLORS[j], s=50, zorder=5,
                       edgecolors="black", linewidths=0.6)
        ax.set_xlabel("Day"); ax.set_ylabel(r"$\bar{\varphi}_i$")
        ax.set_title(f"{title} — continuous ODE vs data", fontweight="bold")
        ax.set_xlim(0, 22); ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)
        ax.grid(True)

    fig.tight_layout(h_pad=0.5)
    fig.savefig(fig_dir / "fig9_ode_continuous.pdf")
    fig.savefig(fig_dir / "fig9_ode_continuous.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIG 10: Metrics comparison bar
# ═══════════════════════════════════════════════════════════════
def generate_fig10_metrics_bar(m_plan, m_adh, logL_plan, logL_adh, fig_dir):
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))
    labels = ["RMSE", "Cosine", r"Spearman $\rho$"]
    plan_vals = [m_plan["rmse"], m_plan["cosine"], m_plan["spearman"]]
    adh_vals = [m_adh["rmse"], m_adh["cosine"], m_adh["spearman"]]
    x = np.arange(len(labels))
    w = 0.3
    b1 = ax.bar(x - w/2, plan_vals, w, label="Planktonic", color=COND_COLORS_SM["plan"], alpha=0.8)
    b2 = ax.bar(x + w/2, adh_vals, w, label="Adherent", color=COND_COLORS_SM["adh"], alpha=0.8)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Fit quality comparison", fontweight="bold")
    ax.legend(frameon=True, edgecolor="0.8", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_metrics_bar.pdf")
    fig.savefig(fig_dir / "fig10_metrics_bar.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# LaTeX report
# ═══════════════════════════════════════════════════════════════
def write_latex(out_dir, m_plan, m_adh, theta_plan, theta_adh,
                logL_plan, logL_adh, pred_plan, pred_adh):
    di_plan = [compute_di(pred_plan[i]) for i in range(6)]
    di_adh = [compute_di(pred_adh[i]) for i in range(6)]
    E_plan = [E_MIN + (E_MAX - E_MIN) * (1 - d) ** DI_EXP for d in di_plan]
    E_adh = [E_MIN + (E_MAX - E_MIN) * (1 - d) ** DI_EXP for d in di_adh]

    theta_rows = ""
    for i in range(20):
        theta_rows += f"    {i} & {PARAM_LABELS_20[i]} & {theta_plan[i]:+.3f} & {theta_adh[i]:+.3f} \\\\\n"

    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{float}
\usepackage{lmodern}

\title{Siddiqui et al.\ 2022 --- TMCMC Re-calibration Report\\[4pt]
\large 10,000-particle GPU-accelerated Bayesian Inference\\[2pt]
\normalsize on Independent External Dataset (Siddiqui et al.\ 2022, PMC8828709)}
\author{Keisuke Nishioka}
\date{\today}

\begin{document}
\maketitle

\section{Summary}

TMCMC re-calibration of the 5-species Hamilton ODE model on
Siddiqui et al.\ (2022, \textit{Dental Materials}, DOI: 10.1016/j.dental.2021.12.021, PMC8828709)
in vitro biofilm data.
Six species (\textit{S.\ oralis}, \textit{A.\ naeslundii}, \textit{A.\ actinomycetemcomitans},
\textit{V.\ parvula}, \textit{F.\ nucleatum}, \textit{P.\ gingivalis}) were co-cultured on
polished cpTi disks and monitored over 21 days (6h, 1, 3, 7, 14, 21d).
\textit{A.\ actinomycetemcomitans} was excluded and data renormalized to 5 species.
Two fractions reported: planktonic (bulk medium) and adherent (biofilm on cpTi surface).

\textbf{Key differences from Heine 2025 (our calibration data)}:
\begin{itemize}
\item No antibiotic or antimicrobial treatment---pure succession
\item Single culture condition (no commensal/dysbiotic manipulation)
\item Substrate: polished cpTi (dental implant) vs hydroxyapatite (Heine)
\item Species: \textit{V.\ parvula} (SM) vs \textit{V.\ dispar} (Heine)---same genus, interchangeable in oral models
\item Inoculum: So monoculture (Day 0.25 $\approx$ 99\% So) vs mixed inoculum (Heine)
\end{itemize}

\begin{table}[H]
\centering
\caption{Fit quality (10,000 particles, RW $\times$ 10 mutation steps, RTX 4090 GPU)}
\label{tab:metrics}
\begin{tabular}{lcccc}
\toprule
Dataset & RMSE & Cosine sim & Spearman $\rho$ & max $\log L$ \\
\midrule
Planktonic & """ + f"{m_plan['rmse']:.4f}" + r""" & """ + f"{m_plan['cosine']:.4f}" + r""" & """ + f"{m_plan['spearman']:.4f}" + r""" & """ + f"{logL_plan.max():.1f}" + r""" \\
Adherent   & """ + f"{m_adh['rmse']:.4f}" + r""" & """ + f"{m_adh['cosine']:.4f}" + r""" & """ + f"{m_adh['spearman']:.4f}" + r""" & """ + f"{logL_adh.max():.1f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

Key findings:
\begin{itemize}
\item Adherent dataset shows excellent fit (Cos.\ sim """ + f"{m_adh['cosine']:.3f}" + r""", RMSE """ + f"{m_adh['rmse']:.3f}" + r""")
\item Planktonic has rapid Fn surge (Day 3: 74\%) that is harder to capture
\item Day 21 planktonic: $E=""" + f"{E_plan[-1]:.0f}" + r"""$\,Pa; adherent: $E=""" + f"{E_adh[-1]:.0f}" + r"""$\,Pa
\item Both within literature range [20, 14000] Pa
\end{itemize}

\section{Posterior Predictive: Species Succession (5-panel)}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig1_planktonic_5panel.pdf}
\caption{Planktonic: MAP trajectory (line), 50\%/80\% CI (shaded), SM data (dots).}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig1_adherent_5panel.pdf}
\caption{Adherent on polished cpTi.}
\end{figure}

\section{Interaction Matrix $A$}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{figures/fig2_interaction_matrix.pdf}
\caption{MAP interaction matrices. Red = mutualistic, blue = competitive.
Note planktonic has strong Fn--Pg coupling ($a_{45}=""" + f"{theta_to_A(theta_plan)[3,4]:.1f}" + r"""$)
vs adherent ($a_{45}=""" + f"{theta_to_A(theta_adh)[3,4]:.1f}" + r"""$).}
\end{figure}

\section{Growth Rates $\mu_i$}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig3_growth_rates.pdf}
\caption{Posterior distribution of growth rates. Star = MAP. Pg growth
highest in planktonic ($\mu_5=""" + f"{theta_plan[15]:.2f}" + r"""$) driving Day 21 dominance.}
\end{figure}

\section{Dysbiosis Index and Young's Modulus}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig4_di_E_trajectory.pdf}
\caption{DI and $E(\text{DI})$ trajectories.}
\end{figure}

\section{Observed vs Predicted}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig5_scatter_day21.pdf}
\caption{Left/center: scatter. Right: Day 21 species composition.}
\end{figure}

\section{Stacked Area Composition}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig7_stacked_area.pdf}
\caption{Stacked area plots: observed (left) vs MAP predicted (right).}
\end{figure}

\section{Residual Analysis}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig8_residuals.pdf}
\caption{Per-species residuals (pred $-$ obs). Planktonic shows large Fn residual at Day 3.}
\end{figure}

\section{Continuous ODE Trajectory}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig9_ode_continuous.pdf}
\caption{Full ODE solution with observed data overlaid (filled circles).}
\end{figure}

\section{Metrics Comparison}

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{figures/fig10_metrics_bar.pdf}
\caption{Summary metrics: adherent outperforms planktonic across all measures.}
\end{figure}

\section{Full Posterior (20 parameters)}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig6_posterior_violin_20.pdf}
\caption{All 20 parameters: interaction ($a_{ij}$) and growth ($\mu_i$).
P = Planktonic, A = Adherent.}
\end{figure}

\section{MAP Parameters}

\begin{table}[H]
\centering
\caption{MAP $\theta$ (10,000 particles)}
\label{tab:map}
\begin{tabular}{clcc}
\toprule
Idx & Param & Planktonic & Adherent \\
\midrule
""" + theta_rows + r"""\bottomrule
\end{tabular}
\end{table}

\section{Planktonic Accuracy Improvement Ideas}

Current planktonic RMSE (""" + f"{m_plan['rmse']:.3f}" + r""") is higher than adherent
(""" + f"{m_adh['rmse']:.3f}" + r"""). Root causes and potential fixes:

\begin{enumerate}
\item \textbf{Rapid Fn surge (Day 1--3)}: Fn jumps 0$\to$74\% in 3 days.
The Hamilton ODE with uniform $\Delta t$ struggles to resolve this.
\textit{Fix}: Adaptive time-stepping or finer $\Delta t$ near early phase.

\item \textbf{Hill gate delay}: Current $K=0.05$, $n=2$ may be too restrictive
for SM where Pg appears by Day 14. \textit{Fix}: Include $K_\text{hill}$ and $n_\text{hill}$
as free parameters in TMCMC (2 extra dims, 22 total).

\item \textbf{So monoculture IC}: Day 0.25 is 99\% So. This sharp IC amplifies
early-phase sensitivity. \textit{Fix}: Use Day 1 as IC (skip Day 0.25) to
avoid the transient.

\item \textbf{Nutrient coupling}: Planktonic cells compete for nutrients in
bulk media (well-mixed). Adding Monod nutrient ODE (\texttt{simulate\_0d\_nutrient})
could explain the Fn--So switch.

\item \textbf{Prior tightening}: Wide prior [-1, 3] may allow multimodal
posterior. \textit{Fix}: Use Heine 2025 posterior as informative prior
(hierarchical Bayes).

\item \textbf{$\sigma_\text{obs}$ schedule}: Currently fixed at 0.10.
Early timepoints (digitized from bar charts) have higher uncertainty.
\textit{Fix}: Per-timepoint $\sigma_i$ = [0.15, 0.12, 0.10, 0.08, 0.06, 0.05].
\end{enumerate}

\section{Configuration}
\begin{itemize}
\item \textbf{Particles}: 10,000 (vmap GPU parallel)
\item \textbf{Mutation}: RW $\times$ 10 steps/particle/stage
\item \textbf{$\sigma_\text{obs}$}: 0.10
\item \textbf{ODE}: 2,500 steps ($\Delta t = 10^{-4}$), $K=0.05$, $n=2$
\item \textbf{Prior}: $a_{ij} \in [-1, 3]$, $\mu_i \in [0, 5]$
\item \textbf{GPU}: NVIDIA RTX 4090 (vancouver01)
\end{itemize}

\end{document}
"""
    with open(out_dir / "sm_report.tex", "w") as f:
        f.write(latex)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    OUT = SCRIPT_DIR / "_runs" / "sm_report"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR = OUT / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    plan_dir = SCRIPT_DIR / "_runs" / "sm_planktonic_vmap_10000p"
    adh_dir = SCRIPT_DIR / "_runs" / "sm_adherent_vmap_10000p"
    theta_plan, samples_plan, logL_plan = load_run(plan_dir)
    theta_adh, samples_adh, logL_adh = load_run(adh_dir)

    # JIT warmup
    print("JIT warmup...")
    _ = run_ode(np.zeros(20), np.full(5, 0.2))

    print("Fig 1: 5-panel planktonic...")
    m_plan, pred_plan = generate_fig1(theta_plan, samples_plan, SM_PLANKTONIC,
                                       SM_PLANKTONIC[0], "Planktonic", SP_COLORS, FIG_DIR)

    print("Fig 1: 5-panel adherent...")
    m_adh, pred_adh = generate_fig1(theta_adh, samples_adh, SM_ADHERENT,
                                     SM_ADHERENT[0], "Adherent", SP_COLORS, FIG_DIR)

    print(f"  Planktonic: RMSE={m_plan['rmse']:.4f}, Cos={m_plan['cosine']:.4f}")
    print(f"  Adherent:   RMSE={m_adh['rmse']:.4f}, Cos={m_adh['cosine']:.4f}")

    print("Fig 2: interaction matrix...")
    generate_fig2_matrix(theta_plan, theta_adh, FIG_DIR)

    print("Fig 3: growth rates...")
    generate_fig3_mu(samples_plan, samples_adh, theta_plan, theta_adh, FIG_DIR)

    print("Fig 4: DI + E...")
    generate_fig4_di_E(SM_PLANKTONIC, pred_plan, SM_ADHERENT, pred_adh, FIG_DIR)

    print("Fig 5: scatter + day21...")
    generate_fig5_scatter(SM_PLANKTONIC, pred_plan, SM_ADHERENT, pred_adh,
                          m_plan, m_adh, FIG_DIR)

    print("Fig 6: posterior violin 20 params...")
    generate_fig6_violin20(samples_plan, samples_adh, theta_plan, theta_adh, FIG_DIR)

    print("Fig 7: stacked area...")
    generate_fig7_stacked(SM_PLANKTONIC, pred_plan, SM_ADHERENT, pred_adh, FIG_DIR)

    print("Fig 8: residuals...")
    generate_fig8_residuals(SM_PLANKTONIC, pred_plan, SM_ADHERENT, pred_adh, FIG_DIR)

    # Full ODE trajectories for continuous plot
    traj_plan_full = run_ode(theta_plan, SM_PLANKTONIC[0])
    traj_adh_full = run_ode(theta_adh, SM_ADHERENT[0])
    print("Fig 9: continuous ODE trajectory...")
    generate_fig9_ode_continuous(traj_plan_full, traj_adh_full, FIG_DIR)

    print("Fig 10: metrics bar...")
    generate_fig10_metrics_bar(m_plan, m_adh, logL_plan, logL_adh, FIG_DIR)

    print("Writing LaTeX...")
    write_latex(OUT, m_plan, m_adh, theta_plan, theta_adh,
                logL_plan, logL_adh, pred_plan, pred_adh)

    print(f"\nDone! Compile: cd {OUT} && pdflatex sm_report.tex")


if __name__ == "__main__":
    main()
