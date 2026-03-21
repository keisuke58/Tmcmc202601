#!/usr/bin/env python3
"""
generate_siddiqui_6sp_report.py — 6-species TMCMC report for Siddiqui 2021.

Generates publication-quality figures + LaTeX report.
Same visual style as regenerate_all_figures.py (Latin Modern, 300 dpi).

Run after 6sp TMCMC completes:
    python generate_siddiqui_6sp_report.py \
        --plan-dir _runs/siddiqui_6sp_planktonic_10000p_best \
        --adh-dir _runs/siddiqui_6sp_adherent_10000p_best
"""
import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_dist

# ═══════════════════════════════════════════════════════════════
# STYLE
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
# CONSTANTS (6 species)
# ═══════════════════════════════════════════════════════════════
N_SP = 6
N_PARAMS = 27
SPECIES = ["So", "An", "Aa", "Vp", "Fn", "Pg"]
SPECIES_ITALIC = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{A.\ actinom.}$",
    r"$\it{V.\ parvula}$",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]
SP_COLORS = ["#2166AC", "#1B7837", "#66c2a5", "#FF8C00", "#7B3294", "#B2182B"]
COND_COLORS = {"plan": "#2166AC", "adh": "#B2182B"}

SM_DAYS = np.array([0.25, 1, 3, 7, 14, 21])

PARAM_NAMES = [
    "a(So-So)",
    "a(So-An)",
    "a(An-An)",
    "a(So-Aa)",
    "a(An-Aa)",
    "a(Aa-Aa)",
    "a(So-Vp)",
    "a(An-Vp)",
    "a(Aa-Vp)",
    "a(Vp-Vp)",
    "a(So-Fn)",
    "a(An-Fn)",
    "a(Aa-Fn)",
    "a(Vp-Fn)",
    "a(Fn-Fn)",
    "a(So-Pg)",
    "a(An-Pg)",
    "a(Aa-Pg)",
    "a(Vp-Pg)",
    "a(Fn-Pg)",
    "a(Pg-Pg)",
    "b(So)",
    "b(An)",
    "b(Aa)",
    "b(Vp)",
    "b(Fn)",
    "b(Pg)",
]
PARAM_TEX = [
    r"$a_{11}$",
    r"$a_{12}$",
    r"$a_{22}$",
    r"$a_{13}$",
    r"$a_{23}$",
    r"$a_{33}$",
    r"$a_{14}$",
    r"$a_{24}$",
    r"$a_{34}$",
    r"$a_{44}$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}$",
    r"$a_{45}$",
    r"$a_{55}$",
    r"$a_{16}$",
    r"$a_{26}$",
    r"$a_{36}$",
    r"$a_{46}$",
    r"$a_{56}$",
    r"$a_{66}$",
    r"$\mu_1$",
    r"$\mu_2$",
    r"$\mu_3$",
    r"$\mu_4$",
    r"$\mu_5$",
    r"$\mu_6$",
]
MU_IDX = [21, 22, 23, 24, 25, 26]
MU_LABELS = [
    r"$\mu_1$ So",
    r"$\mu_2$ An",
    r"$\mu_3$ Aa",
    r"$\mu_4$ Vp",
    r"$\mu_5$ Fn",
    r"$\mu_6$ Pg",
]

E_MAX, E_MIN, DI_EXP = 1000.0, 10.0, 2.0

# Use Numba solver (avoids JAX LLVM OOM on CPU nodes)
try:
    from improved_5species_jit import BiofilmNewtonSolver5S

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Fallback: JAX CPU
if not HAS_NUMBA:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from hamilton_ode_jax_6sp import simulate_0d_6sp


def run_ode_6sp(theta, ic, n_steps=5000, dt=1e-4):
    """Run 6sp ODE. Try JAX first, fail gracefully."""
    ic_safe = np.clip(ic, 0.005, 0.99).astype(np.float64)
    ic_safe = ic_safe / ic_safe.sum() * 0.999
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        from hamilton_ode_jax_6sp import simulate_0d_6sp

        traj = simulate_0d_6sp(
            jnp.array(theta, dtype=jnp.float64),
            n_steps=n_steps,
            dt=dt,
            phi_init=jnp.array(ic_safe, dtype=jnp.float64),
            K_hill=0.05,
            n_hill=2.0,
        )
        return np.array(traj)
    except Exception as e:
        print(f"  JAX ODE failed: {e}, using zero trajectory")
        return np.tile(ic_safe, (n_steps + 1, 1))


def theta_to_A_6sp(theta):
    A = np.zeros((N_SP, N_SP))
    A[0, 0] = theta[0]
    A[0, 1] = theta[1]
    A[1, 0] = theta[1]
    A[1, 1] = theta[2]
    A[0, 2] = theta[3]
    A[2, 0] = theta[3]
    A[1, 2] = theta[4]
    A[2, 1] = theta[4]
    A[2, 2] = theta[5]
    A[0, 3] = theta[6]
    A[3, 0] = theta[6]
    A[1, 3] = theta[7]
    A[3, 1] = theta[7]
    A[2, 3] = theta[8]
    A[3, 2] = theta[8]
    A[3, 3] = theta[9]
    A[0, 4] = theta[10]
    A[4, 0] = theta[10]
    A[1, 4] = theta[11]
    A[4, 1] = theta[11]
    A[2, 4] = theta[12]
    A[4, 2] = theta[12]
    A[3, 4] = theta[13]
    A[4, 3] = theta[13]
    A[4, 4] = theta[14]
    A[0, 5] = theta[15]
    A[5, 0] = theta[15]
    A[1, 5] = theta[16]
    A[5, 1] = theta[16]
    A[2, 5] = theta[17]
    A[5, 2] = theta[17]
    A[3, 5] = theta[18]
    A[5, 3] = theta[18]
    A[4, 5] = theta[19]
    A[5, 4] = theta[19]
    A[5, 5] = theta[20]
    return A


def compute_di(phi):
    eps = 1e-12
    p = np.clip(phi / (phi.sum() + eps), eps, 1.0)
    return 1.0 - (-np.sum(p * np.log(p))) / np.log(N_SP)


def compute_metrics(obs, pred):
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    cos_sim = 1.0 - cosine_dist(obs.flatten(), pred.flatten())
    rho, _ = spearmanr(obs.flatten(), pred.flatten())
    return {"rmse": rmse, "cosine": cos_sim, "spearman": rho}


def load_data(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return np.array(d["data_frac"])


def load_run(run_dir):
    with open(run_dir / "theta_MAP.json") as f:
        raw = json.load(f)
    # Handle both "0" and "theta_0" key formats
    if "0" in raw:
        theta = np.array([raw[str(i)] for i in range(N_PARAMS)])
    else:
        theta = np.array([raw[f"theta_{i}"] for i in range(N_PARAMS)])
    samples = np.load(run_dir / "samples.npy")
    logL = np.load(run_dir / "logL.npy")
    return theta, samples, logL


def get_idx_sparse(n_steps=5000):
    t_max = n_steps * 1e-4
    scale = (t_max * 0.95) / SM_DAYS.max()
    idx = np.round(SM_DAYS * scale / 1e-4).astype(int)
    return np.clip(idx, 0, n_steps)


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════


def fig_6panel(theta, samples, obs, ic, label, fig_dir, n_steps=5000):
    """6-panel posterior predictive (one per species)."""
    model_days = np.linspace(0, 22, n_steps + 1)
    traj_map = run_ode_6sp(theta, ic, n_steps=n_steps)

    n_sub = min(50, len(samples))
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(samples), n_sub, replace=False)
    trajs = np.zeros((n_sub, n_steps + 1, N_SP))
    for k, si in enumerate(idx_sub):
        trajs[k] = run_ode_6sp(samples[si], ic, n_steps=n_steps)
        if (k + 1) % 10 == 0:
            print(f"    trajectory {k+1}/{n_sub}")

    q10 = np.percentile(trajs, 10, axis=0)
    q25 = np.percentile(trajs, 25, axis=0)
    q75 = np.percentile(trajs, 75, axis=0)
    q90 = np.percentile(trajs, 90, axis=0)

    idx_sparse = get_idx_sparse(n_steps)
    pred = traj_map[idx_sparse]
    pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)
    m = compute_metrics(obs, pred)

    fig, axes = plt.subplots(1, N_SP, figsize=(9.0, 1.8), sharex=True, sharey=True)
    for j in range(N_SP):
        ax = axes[j]
        ax.fill_between(model_days, q10[:, j], q90[:, j], color=SP_COLORS[j], alpha=0.12)
        ax.fill_between(model_days, q25[:, j], q75[:, j], color=SP_COLORS[j], alpha=0.25)
        ax.plot(model_days, traj_map[:, j], color=SP_COLORS[j], lw=1.3, zorder=3)
        ax.scatter(
            SM_DAYS, obs[:, j], s=18, color=SP_COLORS[j], edgecolors="k", linewidths=0.4, zorder=5
        )
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.set_title(SPECIES_ITALIC[j], fontsize=7)
        ax.set_xlabel("Day")
        if j == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$")

    axes[5].text(
        0.95,
        0.92,
        f"RMSE {m['rmse']:.3f}\nCos {m['cosine']:.3f}",
        transform=axes[5].transAxes,
        fontsize=6,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
    )

    handles = [
        Line2D([0], [0], color="gray", lw=1.3, label="MAP"),
        Patch(facecolor="gray", alpha=0.25, label="50% CI"),
        Patch(facecolor="gray", alpha=0.12, label="80% CI"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=4,
            label="Data",
        ),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=5, handlelength=1.0, borderpad=0.3)
    fig.suptitle(f"{label} (6 species, {N_PARAMS} params)", fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout(w_pad=0.2)
    tag = label.lower().replace(" ", "_")
    fig.savefig(fig_dir / f"fig_6panel_{tag}.pdf")
    fig.savefig(fig_dir / f"fig_6panel_{tag}.png")
    plt.close(fig)
    return m, pred


def fig_A_matrix(theta_plan, theta_adh, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
    A_plan = theta_to_A_6sp(theta_plan)
    A_adh = theta_to_A_6sp(theta_adh)
    vmax = max(np.abs(A_plan).max(), np.abs(A_adh).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, A, title, color in zip(
        axes, [A_plan, A_adh], ["Planktonic", "Adherent"], [COND_COLORS["plan"], COND_COLORS["adh"]]
    ):
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
        for i in range(N_SP):
            for j in range(N_SP):
                val = A[i, j]
                c = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5, color=c)
        ax.set_xticks(range(N_SP))
        ax.set_yticks(range(N_SP))
        ax.set_xticklabels(SPECIES, fontsize=6)
        ax.set_yticklabels(SPECIES, fontsize=6)
        ax.set_title(title, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20, pad=0.03)
    cbar.set_label(r"$A_{ij}$", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_A_matrix_6sp.pdf")
    fig.savefig(fig_dir / "fig_A_matrix_6sp.png")
    plt.close(fig)


def fig_mu_violin(samples_plan, samples_adh, theta_plan, theta_adh, fig_dir):
    fig, axes = plt.subplots(1, N_SP, figsize=(9.0, 2.0), sharey=True)
    for sp_idx in range(N_SP):
        ax = axes[sp_idx]
        pidx = MU_IDX[sp_idx]
        data = [samples_plan[:, pidx], samples_adh[:, pidx]]
        parts = ax.violinplot(
            data, positions=[0, 1], showmeans=False, showextrema=False, widths=0.7
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(COND_COLORS["plan"] if i == 0 else COND_COLORS["adh"])
            pc.set_edgecolor("k")
            pc.set_linewidth(0.4)
            pc.set_alpha(0.7)
        ax.scatter(
            0,
            theta_plan[pidx],
            marker="*",
            s=40,
            color=COND_COLORS["plan"],
            edgecolors="k",
            linewidths=0.3,
            zorder=5,
        )
        ax.scatter(
            1,
            theta_adh[pidx],
            marker="*",
            s=40,
            color=COND_COLORS["adh"],
            edgecolors="k",
            linewidths=0.3,
            zorder=5,
        )
        ax.set_title(MU_LABELS[sp_idx], fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["P", "A"], fontsize=6)
        ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
        ax.grid(True, axis="y")
        if sp_idx == 0:
            ax.set_ylabel("Growth rate")

    handles = [
        Patch(facecolor=COND_COLORS["plan"], edgecolor="k", alpha=0.7, label="Planktonic"),
        Patch(facecolor=COND_COLORS["adh"], edgecolor="k", alpha=0.7, label="Adherent"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=8,
            label="MAP",
        ),
    ]
    axes[5].legend(handles=handles, loc="upper right", fontsize=5.5, borderpad=0.3)
    fig.tight_layout(w_pad=0.3)
    fig.savefig(fig_dir / "fig_mu_violin_6sp.pdf")
    fig.savefig(fig_dir / "fig_mu_violin_6sp.png")
    plt.close(fig)


def fig_scatter_day21(obs_plan, pred_plan, obs_adh, pred_adh, m_plan, m_adh, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for ax, obs, pred, label, m in [
        (axes[0], obs_plan, pred_plan, "Planktonic", m_plan),
        (axes[1], obs_adh, pred_adh, "Adherent", m_adh),
    ]:
        for j in range(N_SP):
            ax.scatter(
                obs[:, j],
                pred[:, j],
                c=SP_COLORS[j],
                s=25,
                zorder=3,
                edgecolors="k",
                linewidths=0.3,
                label=SPECIES[j],
            )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.text(
            0.05,
            0.92,
            f"RMSE={m['rmse']:.3f}\nCos={m['cosine']:.3f}",
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.7),
        )
    axes[0].legend(fontsize=5, frameon=False, loc="lower right")

    ax = axes[2]
    x = np.arange(N_SP)
    w = 0.2
    ax.bar(x - 1.5 * w, obs_plan[-1], w, color=COND_COLORS["plan"], alpha=0.4, label="Plan obs")
    ax.bar(x - 0.5 * w, pred_plan[-1], w, color=COND_COLORS["plan"], alpha=0.9, label="Plan pred")
    ax.bar(x + 0.5 * w, obs_adh[-1], w, color=COND_COLORS["adh"], alpha=0.4, label="Adh obs")
    ax.bar(x + 1.5 * w, pred_adh[-1], w, color=COND_COLORS["adh"], alpha=0.9, label="Adh pred")
    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES, fontsize=6)
    ax.set_ylabel("Fraction")
    ax.set_title("Day 21", fontweight="bold")
    ax.legend(fontsize=5, ncol=2, frameon=True, edgecolor="0.8")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(fig_dir / "fig_scatter_day21_6sp.pdf")
    fig.savefig(fig_dir / "fig_scatter_day21_6sp.png")
    plt.close(fig)


def fig_di_E(obs_plan, pred_plan, obs_adh, pred_adh, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for obs, pred, label, color in [
        (obs_plan, pred_plan, "Planktonic", COND_COLORS["plan"]),
        (obs_adh, pred_adh, "Adherent", COND_COLORS["adh"]),
    ]:
        di_obs = np.array([compute_di(obs[i]) for i in range(6)])
        di_pred = np.array([compute_di(pred[i]) for i in range(6)])
        E_obs = E_MIN + (E_MAX - E_MIN) * (1 - di_obs) ** DI_EXP
        E_pred = E_MIN + (E_MAX - E_MIN) * (1 - di_pred) ** DI_EXP
        axes[0].plot(
            SM_DAYS,
            di_obs,
            "o-",
            color=color,
            lw=1.2,
            ms=5,
            markeredgecolor="k",
            markeredgewidth=0.3,
            label=f"{label} obs",
        )
        axes[0].plot(
            SM_DAYS, di_pred, "s--", color=color, lw=1.2, ms=4, alpha=0.7, label=f"{label} pred"
        )
        axes[1].plot(
            SM_DAYS,
            E_obs,
            "o-",
            color=color,
            lw=1.2,
            ms=5,
            markeredgecolor="k",
            markeredgewidth=0.3,
            label=f"{label} obs",
        )
        axes[1].plot(
            SM_DAYS, E_pred, "s--", color=color, lw=1.2, ms=4, alpha=0.7, label=f"{label} pred"
        )
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("DI")
    axes[0].set_title("Dysbiosis Index", fontweight="bold")
    axes[0].legend(fontsize=5.5)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("$E$ (Pa)")
    axes[1].set_title("$E(\\mathrm{DI})$", fontweight="bold")
    axes[1].legend(fontsize=5.5)
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_di_E_6sp.pdf")
    fig.savefig(fig_dir / "fig_di_E_6sp.png")
    plt.close(fig)


def fig_stacked(obs_plan, pred_plan, obs_adh, pred_adh, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for row, (obs, pred, label) in enumerate(
        [(obs_plan, pred_plan, "Planktonic"), (obs_adh, pred_adh, "Adherent")]
    ):
        axes[row, 0].stackplot(
            SM_DAYS, obs.T, labels=SPECIES if row == 0 else None, colors=SP_COLORS, alpha=0.7
        )
        axes[row, 0].set_title(f"{label} — Observed", fontweight="bold")
        axes[row, 0].set_ylabel(r"$\bar{\varphi}_i$")
        axes[row, 0].set_ylim(0, 1.05)
        axes[row, 0].grid(True)
        axes[row, 1].stackplot(SM_DAYS, pred.T, colors=SP_COLORS, alpha=0.7)
        axes[row, 1].set_title(f"{label} — MAP Predicted", fontweight="bold")
        axes[row, 1].set_ylim(0, 1.05)
        axes[row, 1].grid(True)
    for ax in axes[1]:
        ax.set_xlabel("Day")
    axes[0, 0].legend(fontsize=5, loc="center right", frameon=False)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(fig_dir / "fig_stacked_6sp.pdf")
    fig.savefig(fig_dir / "fig_stacked_6sp.png")
    plt.close(fig)


def fig_residuals(obs_plan, pred_plan, obs_adh, pred_adh, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, obs, pred, title in [
        (axes[0], obs_plan, pred_plan, "Planktonic"),
        (axes[1], obs_adh, pred_adh, "Adherent"),
    ]:
        for j in range(N_SP):
            ax.plot(
                SM_DAYS,
                pred[:, j] - obs[:, j],
                color=SP_COLORS[j],
                marker="s",
                ms=4,
                label=SPECIES[j],
                lw=1.0,
            )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Day")
        ax.set_ylabel("Pred $-$ Obs")
        ax.set_title(title, fontweight="bold")
        ax.grid(True)
    axes[0].legend(ncol=6, loc="upper center", frameon=False, fontsize=5)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(fig_dir / "fig_residuals_6sp.pdf")
    fig.savefig(fig_dir / "fig_residuals_6sp.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# LaTeX (template — fills in values at runtime)
# ═══════════════════════════════════════════════════════════════
def write_latex(out_dir, m_plan, m_adh, theta_plan, theta_adh, logL_plan, logL_adh, config_plan):
    theta_rows = ""
    for i in range(N_PARAMS):
        theta_rows += f"    {i} & {PARAM_TEX[i]} & {PARAM_NAMES[i]:12s} & {theta_plan[i]:+.3f} & {theta_adh[i]:+.3f} \\\\\n"

    latex = (
        r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,hyperref,xcolor,float,lmodern}

\title{Siddiqui et al.\ 2021 --- 6-Species TMCMC Re-calibration\\[4pt]
\large 10,000-particle GPU-accelerated Bayesian inference (27 parameters)}
\author{Keisuke Nishioka}
\date{\today}

\begin{document}
\maketitle

\section{Dataset}

Siddiqui DA, Fidai AB, Natarajan SG, Rodrigues DC (2022).
``Succession of Oral Bacterial Colonizers on Dental Implant Materials:
An In Vitro Biofilm Model.'' \textit{Dental Materials} 38(2):384--396.
DOI: 10.1016/j.dental.2021.12.021. PMC8828709.

\textbf{6 species}: \textit{S.\ oralis}, \textit{A.\ naeslundii},
\textit{A.\ actinomycetemcomitans}, \textit{V.\ parvula},
\textit{F.\ nucleatum}, \textit{P.\ gingivalis}.
Planktonic + adherent on polished cpTi, 6 timepoints over 21 days.
No antibiotics. Pure in vitro colonization succession.

\subsection{Species and Socransky Classification}

\begin{table}[H]\centering\small
\caption{Bacterial species in Siddiqui et al.\ model}
\begin{tabular}{llll}
\toprule
Species & Abbrev & Socransky complex & Role \\
\midrule
\textit{Streptococcus oralis} & So & Yellow & Early colonizer \\
\textit{Actinomyces naeslundii} & An & Blue & Early colonizer \\
\textit{Aggregatibacter actinomycetemcomitans} & Aa & Green & Early colonizer \\
\textit{Veillonella parvula} & Vp & Purple & Secondary (metabolic bridge) \\
\textit{Fusobacterium nucleatum} & Fn & Orange & Secondary (physical bridge) \\
\textit{Porphyromonas gingivalis} & Pg & Red & Late colonizer (pathogen) \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Experimental Design}

Siddiqui et al.\ prepared commercially pure titanium (cpTi) and yttria-stabilized
zirconia (ZrO$_2$) disks with three surface treatments---polished, acid-etched, and
sandblasted---yielding six substrate conditions in total. In the present re-calibration
study, we focus on the \textbf{planktonic fraction} and the \textbf{adherent biofilm on
polished cpTi}, as these represent the two extremes of bacterial habitat (bulk medium
vs.\ surface-attached community).

The bacteria were cultured anaerobically (10\% CO$_2$, 5\% H$_2$, 85\% N$_2$) at
37$^\circ$C in Brain Heart Infusion (BHI) broth supplemented with hemin (5\,mg/mL) and
vitamin K1 (0.1\,mg/mL). Each species was inoculated at $1.67 \times 10^6$\,CFU/mL
(total: $10^7$\,CFU/mL). Culture medium was replaced daily by withdrawing 2.5\,mL
and adding 2.7\,mL of fresh media, simulating nutrient replenishment analogous to
gingival crevicular fluid (GCF) flow.

Bacterial composition was quantified by species-specific quantitative PCR (qPCR) targeting
single-copy genes: GDH1 (\textit{S.\ oralis}), rpoB (\textit{A.\ naeslundii},
\textit{V.\ parvula}), waaA (\textit{A.\ actinomycetemcomitans}, \textit{P.\ gingivalis}),
and GGT (\textit{F.\ nucleatum}). Amplification efficiencies ranged from 91.1\% to 100.7\%
with off-target amplification below 0.1\%, ensuring reliable species-level resolution.
Sampling was performed at six timepoints: 6\,h, 1, 3, 7, 14, and 21 days post-inoculation.

\subsection{Published Succession Pattern}

The succession dynamics reported by Siddiqui et al.\ are summarized in Tables~\ref{tab:plan_succession}
and~\ref{tab:adh_succession}. The planktonic fraction showed a classic early-to-late colonizer
transition: \textit{S.\ oralis} dominated at 6\,h ($\sim$75\%), was overtaken by the bridging
species \textit{F.\ nucleatum} by Day~3 ($\sim$74\%), and was ultimately supplanted by the
red-complex pathogen \textit{P.\ gingivalis} at Day~21 ($\sim$60\%). The transition from
Day~7 to Day~14 was described as ``drastic,'' with \textit{P.\ gingivalis} increasing 100-fold
as it exploited the ecological niche prepared by \textit{F.\ nucleatum}---consistent with the
Hill-gated Fn$\to$Pg mechanism in our Hamilton ODE model.

The adherent fraction on polished cpTi exhibited a slower and more complex dynamics.
A notable observation was the stagnation of \textit{S.\ oralis} at Day~3 on polished cpTi
(statistically significant vs.\ sandblasted ZrO$_2$), followed by a 10-fold recovery at Day~7.
By Day~21 the adherent composition reached a balanced state with approximately equal representation
of early colonizers ($\sim$33\%), secondary colonizers ($\sim$33\%), and \textit{P.\ gingivalis}
($\sim$33\%). This non-monotonic behavior presents a particular challenge for ODE-based
modeling, as the Day~7 reversal requires interaction dynamics beyond simple competitive exclusion.

\begin{table}[H]\centering\small
\caption{Planktonic succession (digitized from Siddiqui et al.\ Figs.\ 1--2 and text)}
\label{tab:plan_succession}
\begin{tabular}{rcccccc}
\toprule
Day & So & An & Aa & Vp & Fn & Pg \\
\midrule
0.25 & 0.75 & 0.01 & 0.23 & 0.00 & 0.01 & 0.00 \\
1    & 0.42 & 0.03 & 0.05 & 0.12 & 0.38 & 0.00 \\
3    & 0.08 & 0.03 & 0.02 & 0.12 & 0.74 & 0.01 \\
7    & 0.12 & 0.05 & 0.03 & 0.12 & 0.65 & 0.03 \\
14   & 0.08 & 0.03 & 0.02 & 0.10 & 0.44 & 0.33 \\
21   & 0.03 & 0.05 & 0.02 & 0.05 & 0.25 & 0.60 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering\small
\caption{Adherent succession on polished cpTi (digitized)}
\label{tab:adh_succession}
\begin{tabular}{rcccccc}
\toprule
Day & So & An & Aa & Vp & Fn & Pg \\
\midrule
0.25 & 0.88 & 0.01 & 0.09 & 0.00 & 0.02 & 0.00 \\
1    & 0.58 & 0.04 & 0.08 & 0.08 & 0.22 & 0.00 \\
3    & 0.28 & 0.04 & 0.08 & 0.15 & 0.45 & 0.00 \\
7    & 0.58 & 0.04 & 0.08 & 0.10 & 0.20 & 0.00 \\
14   & 0.48 & 0.04 & 0.08 & 0.10 & 0.15 & 0.15 \\
21   & 0.22 & 0.05 & 0.06 & 0.10 & 0.24 & 0.33 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Comparison with Heine 2025}

The Siddiqui 2021 dataset differs from our primary calibration data (Heine et al.\ 2025) in
several important respects (Table~\ref{tab:design_comparison}). Most critically, Siddiqui uses
six species including \textit{A.\ actinomycetemcomitans} (green complex), which is absent in
the Heine 5-species model. There is no commensal/dysbiotic manipulation---only a single culture
condition representing natural succession. The substrate (polished cpTi) and \textit{Veillonella}
species (\textit{V.\ parvula} vs.\ \textit{V.\ dispar}) also differ, though these are considered
interchangeable in oral biofilm models. Both studies share the absence of antibiotics and a
21-day observation window with comparable timepoint spacing.

\begin{table}[H]\centering\small
\caption{Experimental design comparison}
\label{tab:design_comparison}
\begin{tabular}{lll}
\toprule
 & Heine 2025 & Siddiqui 2021 \\
\midrule
Species & 5 (So, An, Vd, Fn, Pg) & 6 (+Aa) \\
Conditions & 4 (CS/CH/DS/DH) & 1 (single culture) \\
Manipulation & Nutrient media & None (pure succession) \\
Substrate & Hydroxyapatite & Polished cpTi \\
\textit{Veillonella} & \textit{V.\ dispar} & \textit{V.\ parvula} \\
Inoculum & Mixed (equal) & Mixed ($1.67 \times 10^6$/sp) \\
Quantification & 16S rRNA (FISH) & qPCR (species-specific) \\
Antibiotics & No & No \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Limitations of the Source Data}

The authors noted several limitations that also affect our re-calibration.
The simplified 6-species model excludes \textit{Corynebacterium} spp., recently
identified as a key architect of oral biofilm spatial structure in vivo.
Standard qPCR amplifies genomic DNA from both live and dead cells, potentially
overestimating viable counts. The BHI-based culture system yields bacterial
densities of $10^{8\text{--}9}$\,CFU/mL, two to three orders of magnitude
higher than typical oral cavity counts ($10^{5\text{--}6}$\,CFU/mL), which
may compress inter-species competitive dynamics. Furthermore, the model lacks
salivary pellicle formation, host cell interactions, and oxygen gradients
characteristic of the supragingival environment. These factors should be
considered when interpreting the inferred interaction parameters.

\section{Results}

\begin{table}[H]
\centering
\caption{Fit quality (10,000 particles, RW $\times$ 20 steps, 5,000 ODE steps)}
\begin{tabular}{lcccc}
\toprule
Dataset & RMSE & Cosine & Spearman $\rho$ & max $\log L$ \\
\midrule
Planktonic & """
        + f"{m_plan['rmse']:.4f}"
        + r""" & """
        + f"{m_plan['cosine']:.4f}"
        + r""" & """
        + f"{m_plan['spearman']:.4f}"
        + r""" & """
        + f"{logL_plan.max():.1f}"
        + r""" \\
Adherent   & """
        + f"{m_adh['rmse']:.4f}"
        + r""" & """
        + f"{m_adh['cosine']:.4f}"
        + r""" & """
        + f"{m_adh['spearman']:.4f}"
        + r""" & """
        + f"{logL_adh.max():.1f}"
        + r""" \\
\bottomrule
\end{tabular}
\end{table}

\section{6-Panel Posterior Predictive}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_6panel_planktonic.pdf}
\caption{Planktonic: MAP (line), 50\%/80\% CI (shaded), data (dots). 6 species.}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_6panel_adherent.pdf}
\caption{Adherent on polished cpTi.}
\end{figure}

\section{Interaction Matrix $A$ (6$\times$6)}
\begin{figure}[H]\centering
\includegraphics[width=0.8\textwidth]{figures/fig_A_matrix_6sp.pdf}
\caption{MAP interaction matrices. Red = mutualistic, blue = competitive.}
\end{figure}

\section{Growth Rates $\mu_i$}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_mu_violin_6sp.pdf}
\caption{Posterior growth rates. Star = MAP.}
\end{figure}

\section{Stacked Composition}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_stacked_6sp.pdf}
\caption{Stacked area: observed (left) vs predicted (right).}
\end{figure}

\section{DI and Young's Modulus}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_di_E_6sp.pdf}
\caption{DI and $E(\text{DI})$ trajectories.}
\end{figure}

\section{Scatter + Day 21}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_scatter_day21_6sp.pdf}
\caption{Obs vs pred scatter + Day 21 composition bar chart.}
\end{figure}

\section{Residuals}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_residuals_6sp.pdf}
\caption{Per-species residuals.}
\end{figure}

\section{MAP Parameters (27)}
\begin{table}[H]\centering\small
\caption{MAP $\theta$}
\begin{tabular}{clccc}
\toprule
Idx & TeX & Name & Planktonic & Adherent \\
\midrule
"""
        + theta_rows
        + r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""
    )
    with open(out_dir / "siddiqui_6sp_report.tex", "w") as f:
        f.write(latex)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=str, default="_runs/siddiqui_6sp_planktonic_10000p_best")
    parser.add_argument("--adh-dir", type=str, default="_runs/siddiqui_6sp_adherent_10000p_best")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--n-posterior", type=int, default=50, help="Posterior trajectories for CI")
    args = parser.parse_args()

    plan_dir = (
        Path(args.plan_dir) if Path(args.plan_dir).is_absolute() else SCRIPT_DIR / args.plan_dir
    )
    adh_dir = Path(args.adh_dir) if Path(args.adh_dir).is_absolute() else SCRIPT_DIR / args.adh_dir

    OUT = SCRIPT_DIR / "_runs" / "siddiqui_6sp_report"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR = OUT / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    obs_plan = load_data(SCRIPT_DIR / "siddiqui2021_planktonic_6sp.json")
    obs_adh = load_data(SCRIPT_DIR / "siddiqui2021_adherent_6sp.json")

    theta_plan, samples_plan, logL_plan = load_run(plan_dir)
    theta_adh, samples_adh, logL_adh = load_run(adh_dir)

    config_plan = {}
    cfg_path = plan_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            config_plan = json.load(f)

    print("Fig: 6-panel planktonic...")
    m_plan, pred_plan = fig_6panel(
        theta_plan, samples_plan, obs_plan, obs_plan[0], "Planktonic", FIG_DIR, args.n_steps
    )
    print("Fig: 6-panel adherent...")
    m_adh, pred_adh = fig_6panel(
        theta_adh, samples_adh, obs_adh, obs_adh[0], "Adherent", FIG_DIR, args.n_steps
    )

    print(f"  Planktonic: RMSE={m_plan['rmse']:.4f}, Cos={m_plan['cosine']:.4f}")
    print(f"  Adherent:   RMSE={m_adh['rmse']:.4f}, Cos={m_adh['cosine']:.4f}")

    print("Fig: A matrix...")
    fig_A_matrix(theta_plan, theta_adh, FIG_DIR)
    print("Fig: μ violin...")
    fig_mu_violin(samples_plan, samples_adh, theta_plan, theta_adh, FIG_DIR)
    print("Fig: scatter + day21...")
    fig_scatter_day21(obs_plan, pred_plan, obs_adh, pred_adh, m_plan, m_adh, FIG_DIR)
    print("Fig: DI + E...")
    fig_di_E(obs_plan, pred_plan, obs_adh, pred_adh, FIG_DIR)
    print("Fig: stacked...")
    fig_stacked(obs_plan, pred_plan, obs_adh, pred_adh, FIG_DIR)
    print("Fig: residuals...")
    fig_residuals(obs_plan, pred_plan, obs_adh, pred_adh, FIG_DIR)

    print("Writing LaTeX...")
    write_latex(OUT, m_plan, m_adh, theta_plan, theta_adh, logL_plan, logL_adh, config_plan)

    print(f"\nDone! Compile: cd {OUT} && pdflatex siddiqui_6sp_report.tex")


if __name__ == "__main__":
    main()
