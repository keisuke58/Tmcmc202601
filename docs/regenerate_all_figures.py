#!/usr/bin/env python3
"""Regenerate ALL paper figures with unified publication style.

CPU only (JAX_PLATFORMS=cpu). No GPU needed:
  ~/miniforge3/envs/klempt_fem2/bin/python3 ~/Tmcmc202601/docs/regenerate_all_figures.py \
      --p1-glob '{cond}_p1_*' --p2-glob '{cond}_p2_nonarrow_*'

Run directories are resolved from _runs/paper_gateoff by glob ("{cond}" is replaced by
CS/CH/DS/DH). When several seeds match, the one with the largest max_logL in config.json
is used, and the others are listed so the choice is visible.

The forward model and the gate (K_hill / n_hill), the initial state (phi_init) and the
solver branch (simulate_0d vs simulate_0d_full) are all taken from each run's own
config.json, so the figures are made with exactly the model that produced the posterior.
Previously this script hardcoded 2026-03 run names, imported hamilton_ode_jax (not the
paper engine) and simulated with K_hill=0.05 (gate ON) — see docs/handoff/.

Generates:
  Fig 2: paper_fig2_final.pdf  — Posterior predictive fits (4 cond × 5 species)
  Fig 3: paper_posterior_violin.pdf — Posterior distributions (15 params × 4 cond)
  Fig 4: heatmap_A_4cond.pdf — MAP interaction matrices
  Fig 5: phase1_vs_phase2_map.pdf — Phase 1 vs Phase 2 MAP scatter
"""

import sys
import os

sys.path.insert(0, os.path.expanduser("~/Tmcmc202601/data_5species/main"))

import json
import csv
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import argparse
import fnmatch

# 推定側（estimate_paper_jax.py）と同じ前進モデルを使う。
# 旧実装は hamilton_ode_jax を import していて、実質 222 行違う別系統だった。
from hamilton_ode_jax_paper import simulate_0d, simulate_0d_full

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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
RUNS = Path.home() / "Tmcmc202601/data_5species/main/_runs"
PAPER_RUNS = RUNS / "paper_gateoff"
DATA_CSV = (
    Path.home() / "Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_summary.csv"
)
FIG_DIR = Path.home() / "Tmcmc202601/docs/figures"
FIG_DIR.mkdir(exist_ok=True)

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_ITALIC = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{V.\ dispar}$",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]
SPECIES_MAP = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
}
EXP_DAYS = np.array([1, 3, 6, 10, 15, 21])

# Heine 2025 inspired palette
SP_COLORS = {
    "C": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#B2182B"],
    "D": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"],
}

CONDS = ["CS", "CH", "DS", "DH"]
COND_KEYS = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}
COND_LABELS = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}
COND_SHORT = {"CS": "CS", "CH": "CH", "DS": "DS", "DH": "DH"}

# run ディレクトリは --p1-glob / --p2-glob から resolve_runs() で埋める。
# 向き先は _runs/paper_gateoff の下（PAPER_RUNS）。
P2_DIRS: dict = {}
P1_DIRS: dict = {}

PARAM_LABELS_15 = [
    r"$a_{11}$",
    r"$a_{12}$",
    r"$a_{22}$",
    r"$a_{33}$",
    r"$a_{34}$",
    r"$a_{44}$",
    r"$a_{13}$",
    r"$a_{14}$",
    r"$a_{23}$",
    r"$a_{24}$",
    r"$a_{55}$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}$",
    r"$a_{45}$",
]
# Indices of interaction params (exclude growth rates b/μ: 3,4,8,9,15)
INTERACTION_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]

PARAM_LABELS_20 = [
    r"$a_{11}$",
    r"$a_{12}$",
    r"$a_{22}$",
    r"$\mu_1$",
    r"$\mu_2$",
    r"$a_{33}$",
    r"$a_{34}$",
    r"$a_{44}$",
    r"$\mu_3$",
    r"$\mu_4$",
    r"$a_{13}$",
    r"$a_{14}$",
    r"$a_{23}$",
    r"$a_{24}$",
    r"$a_{55}$",
    r"$\mu_5$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}$",
    r"$a_{45}$",
]

COND_COLORS = {"CS": "#2166AC", "CH": "#1B7837", "DS": "#FF8C00", "DH": "#B2182B"}


def load_exp_data():
    raw = {}
    with open(DATA_CSV) as f:
        for row in csv.DictReader(f):
            key = f"{row['condition']}_{row['cultivation']}"
            day = int(row["day"])
            si = SPECIES_MAP.get(row["species"])
            if si is None:
                continue
            if key not in raw:
                raw[key] = {}
            if day not in raw[key]:
                raw[key][day] = np.zeros(5)
            raw[key][day][si] = float(row["mean"])
    result = {}
    cond_to_key = {
        "CS": "Commensal_Static",
        "CH": "Commensal_HOBIC",
        "DS": "Dysbiotic_Static",
        "DH": "Dysbiotic_HOBIC",
    }
    for ck, rk in cond_to_key.items():
        if rk in raw:
            arr = np.array([raw[rk][d] for d in EXP_DAYS])
            sums = arr.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1
            result[ck] = arr / sums
    return result


def load_map(run_name):
    p = RUNS / run_name / "theta_MAP.json"
    if not p.exists():
        return None
    mj = json.load(open(p))
    return np.array([mj[str(i)] for i in range(20)])


def load_samples(run_name):
    p = RUNS / run_name / "samples.npy"
    if not p.exists():
        return None
    return np.load(p)


def load_config(run_name):
    p = RUNS / run_name / "config.json"
    if not p.exists():
        return {}
    return json.load(open(p))


def run_ode(theta, ic, cfg, n_steps=None, dt=None):
    """Forward solve with the same model/gate/solver the run itself used.

    cfg is that run's config.json. Everything that changes the trajectory is read
    from it: the gate (K_hill / n_hill) and the solver branch. Hardcoding K_hill=0.05
    here would simulate with the gate ON against a gate-OFF posterior.
    """
    args = cfg.get("args", {})
    if n_steps is None:
        n_steps = int(args.get("n_steps", 2500))
    if dt is None:
        dt = float(args.get("dt", 1e-4))
    K_hill = float(cfg.get("K_hill", args.get("K_hill", 0.0)))
    n_hill = float(cfg.get("n_hill", args.get("n_hill", 4.0)))

    # estimate_paper_jax.py: _need_full = (viability or pH channel) and not fix_psi
    lam = cfg.get("lambda_ch", {}) or {}
    has_viab = float(lam.get("3", 0.0)) > 0.0
    has_pH = float(lam.get("5", 0.0)) > 0.0
    need_full = (has_viab or has_pH) and not bool(args.get("fix_psi", False))

    phi_init = jnp.array(ic, dtype=jnp.float64)
    theta_jax = jnp.array(theta, dtype=jnp.float64)
    if need_full:
        g_traj = simulate_0d_full(
            theta_jax, n_steps=n_steps, dt=dt, phi_init=phi_init, K_hill=K_hill, n_hill=n_hill
        )
        return np.array(g_traj[:, 0:5])
    traj = simulate_0d(
        theta_jax, n_steps=n_steps, dt=dt, phi_init=phi_init, K_hill=K_hill, n_hill=n_hill
    )
    return np.array(traj)


def resolve_runs(pattern, label):
    """Resolve one glob ("{cond}_p2_*") into {cond: "paper_gateoff/<dir>"}.

    When several seeds match, the largest max_logL wins; the rest are printed so the
    choice is auditable. Missing conditions are reported and left out.
    """
    out = {}
    for ck in CONDS:
        pat = pattern.format(cond=ck)
        cands = sorted(
            d for d in PAPER_RUNS.glob(pat) if d.is_dir() and (d / "theta_MAP.json").exists()
        )
        if not cands:
            print(f"  [{label}] {ck}: no run matches {pat!r}")
            continue
        scored = []
        for d in cands:
            try:
                ml = json.load(open(d / "config.json")).get("max_logL")
            except (OSError, ValueError):
                ml = None
            scored.append((-(ml if ml is not None else -np.inf), d.name, ml))
        scored.sort()
        _, best, best_ml = scored[0]
        out[ck] = f"paper_gateoff/{best}"
        extra = (
            ""
            if len(scored) == 1
            else "  (others: " + ", ".join(f"{n} {m:.2f}" for _, n, m in scored[1:]) + ")"
        )
        print(f"  [{label}] {ck}: {best}  max_logL={best_ml}{extra}")
    return out


def theta_to_A(theta):
    A = np.zeros((5, 5))
    A[0, 0] = theta[0]
    A[0, 1] = theta[1]
    A[1, 1] = theta[2]
    A[2, 2] = theta[5]
    A[2, 3] = theta[6]
    A[3, 3] = theta[7]
    A[0, 2] = theta[10]
    A[0, 3] = theta[11]
    A[1, 2] = theta[12]
    A[1, 3] = theta[13]
    A[4, 4] = theta[14]
    A[0, 4] = theta[16]
    A[1, 4] = theta[17]
    A[2, 4] = theta[18]
    A[3, 4] = theta[19]
    for i in range(5):
        for j in range(i + 1, 5):
            A[j, i] = A[i, j]
    return A


# ═══════════════════════════════════════════════════════════════
# FIG 2: Posterior Predictive Fits
# ═══════════════════════════════════════════════════════════════
def generate_fig2():
    print("Generating Fig 2: Posterior predictive fits...")
    exp_data = load_exp_data()
    model_days = np.linspace(0, 21, 2501)

    # JIT warmup
    _ = run_ode(np.zeros(20), np.full(5, 0.2), load_config(P2_DIRS[CONDS[0]]))

    fig, axes = plt.subplots(4, 5, figsize=(7.2, 7.5), sharex=True)

    for row_idx, ck in enumerate(CONDS):
        theta_map = load_map(P2_DIRS[ck])
        samples = load_samples(P2_DIRS[ck])
        cfg = load_config(P2_DIRS[ck])
        obs = exp_data[ck]
        ic = obs[0].copy()
        ic = np.clip(ic, 0.001, 0.99)
        ic /= ic.sum()

        colors = SP_COLORS["C"] if ck.startswith("C") else SP_COLORS["D"]

        # MAP trajectory
        traj_map = run_ode(theta_map, ic, cfg)

        # Posterior predictive (subsample 100)
        n_sub = min(100, len(samples))
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(len(samples), n_sub, replace=False)
        trajs = np.zeros((n_sub, 2501, 5))
        for k, si in enumerate(idx_sub):
            trajs[k] = run_ode(samples[si], ic, cfg)

        q10 = np.percentile(trajs, 10, axis=0)
        q25 = np.percentile(trajs, 25, axis=0)
        q75 = np.percentile(trajs, 75, axis=0)
        q90 = np.percentile(trajs, 90, axis=0)

        rmse = cfg.get("rmse", 0)
        logL = cfg.get("max_logL", 0)

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]

            # CI bands
            ax.fill_between(
                model_days, q10[:, col_idx], q90[:, col_idx], color=colors[col_idx], alpha=0.12
            )
            ax.fill_between(
                model_days, q25[:, col_idx], q75[:, col_idx], color=colors[col_idx], alpha=0.25
            )

            # MAP line
            ax.plot(model_days, traj_map[:, col_idx], color=colors[col_idx], lw=1.3, zorder=3)

            # Data
            ax.scatter(
                EXP_DAYS,
                obs[:, col_idx],
                s=18,
                color=colors[col_idx],
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
            )

            ax.set_xlim(0, 22)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True)

            if row_idx == 0:
                ax.set_title(SPECIES_ITALIC[col_idx], fontsize=8)
            if row_idx == 3:
                ax.set_xlabel("Day")
            if col_idx == 0:
                ax.set_ylabel(r"$\bar{\varphi}_i$")

        # Condition label + metrics on first column
        axes[row_idx, 0].text(
            -0.35,
            0.5,
            f"{COND_SHORT[ck]}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=10,
            fontweight="bold",
            va="center",
            rotation=90,
        )

        # RMSE label on last column
        axes[row_idx, 4].text(
            0.95,
            0.92,
            f"RMSE {rmse:.3f}\nlogL {logL:.1f}",
            transform=axes[row_idx, 4].transAxes,
            fontsize=6.5,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    # Legend on first panel
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
            label="Exp. data",
        ),
    ]
    axes[0, 0].legend(
        handles=handles,
        loc="upper right",
        fontsize=6,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.3,
    )

    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.subplots_adjust(left=0.08)
    out = FIG_DIR / "paper_fig2_final.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# FIG 3: Posterior Violin Plots (15 interaction params)
# ═══════════════════════════════════════════════════════════════
def generate_fig3():
    print("Generating Fig 3: Posterior violin plots...")

    fig, axes = plt.subplots(3, 5, figsize=(7.2, 5.5))

    groups = [
        ("So--An / Vei--Fn", [0, 1, 2, 5, 6]),
        ("Cross-block", [7, 10, 11, 12, 13]),
        ("Pg self + cross", [14, 16, 17, 18, 19]),
    ]

    for grp_idx, (grp_label, param_indices) in enumerate(groups):
        for col_idx, pidx in enumerate(param_indices):
            ax = axes[grp_idx, col_idx]
            data_per_cond = []
            for ck in CONDS:
                samples = load_samples(P2_DIRS[ck])
                if samples is not None:
                    data_per_cond.append(samples[:, pidx])
                else:
                    data_per_cond.append(np.zeros(100))

            parts = ax.violinplot(
                data_per_cond, positions=range(4), showmeans=False, showextrema=False, widths=0.7
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(COND_COLORS[CONDS[i]])
                pc.set_edgecolor("k")
                pc.set_linewidth(0.4)
                pc.set_alpha(0.7)

            # MAP markers
            for i, ck in enumerate(CONDS):
                theta_map = load_map(P2_DIRS[ck])
                if theta_map is not None:
                    ax.scatter(
                        i,
                        theta_map[pidx],
                        marker="*",
                        s=30,
                        color=COND_COLORS[ck],
                        edgecolors="k",
                        linewidths=0.3,
                        zorder=5,
                    )

            ax.set_xticks(range(4))
            ax.set_xticklabels(CONDS, fontsize=7)
            ax.set_title(PARAM_LABELS_20[pidx], fontsize=9)
            ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
            ax.grid(True, axis="y")

            if col_idx == 0:
                ax.set_ylabel(grp_label, fontsize=8)

    # Legend
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=COND_COLORS[ck], edgecolor="k", linewidth=0.4, alpha=0.7, label=ck)
        for ck in CONDS
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=8,
            label="MAP",
        )
    )
    axes[0, 4].legend(
        handles=handles, loc="upper right", fontsize=6.5, borderpad=0.3, handletextpad=0.3
    )

    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    out = FIG_DIR / "paper_posterior_violin.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# FIG 4: Interaction Matrix Heatmaps
# ═══════════════════════════════════════════════════════════════
def generate_fig4():
    print("Generating Fig 4: Interaction matrix heatmaps...")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    sp_labels = ["So", "An", "Vd", "Fn", "Pg"]

    vmax = 0
    As = {}
    for ck in CONDS:
        theta = load_map(P2_DIRS[ck])
        if theta is not None:
            A = theta_to_A(theta)
            As[ck] = A
            vmax = max(vmax, np.abs(A).max())

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax_idx, ck in enumerate(CONDS):
        ax = axes[ax_idx]
        A = As.get(ck, np.zeros((5, 5)))
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")

        # Annotate values
        for i in range(5):
            for j in range(5):
                val = A[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=color,
                    fontweight="normal",
                )

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(sp_labels, fontsize=7)
        ax.set_yticklabels(sp_labels if ax_idx == 0 else [], fontsize=7)
        ax.set_title(COND_LABELS[ck], fontsize=9, fontweight="bold")
        ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label(r"$A_{ij}$", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=0.3)
    out = FIG_DIR / "heatmap_A_4cond.pdf"  # PDF instead of PNG!
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# FIG 5: Phase 1 vs Phase 2 MAP Scatter
# ═══════════════════════════════════════════════════════════════
def generate_fig5():
    print("Generating Fig 5: Phase 1 vs Phase 2 scatter...")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=False)

    for ax_idx, ck in enumerate(CONDS):
        ax = axes[ax_idx]
        m1 = load_map(P1_DIRS[ck])
        m2 = load_map(P2_DIRS[ck])

        if m1 is None or m2 is None:
            ax.set_title(f"{COND_LABELS[ck]}\n(pending)")
            continue

        color = COND_COLORS[ck]
        ax.scatter(m1, m2, s=18, c=color, edgecolors="k", linewidths=0.3, zorder=3, alpha=0.85)

        # Annotate outliers
        for i in range(20):
            if abs(m1[i] - m2[i]) > 1.5:
                ax.annotate(
                    PARAM_LABELS_20[i],
                    (m1[i], m2[i]),
                    fontsize=5,
                    ha="left",
                    va="bottom",
                    xytext=(3, 3),
                    textcoords="offset points",
                    color="gray",
                )

        lo = min(m1.min(), m2.min()) - 0.5
        hi = max(m1.max(), m2.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4, zorder=1)

        r = np.corrcoef(m1, m2)[0, 1]
        rmse = np.sqrt(np.mean((m1 - m2) ** 2))

        ax.text(
            0.05,
            0.92,
            f"$r = {r:.3f}$\nRMSD$= {rmse:.2f}$",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
        )

        ax.set_title(COND_LABELS[ck], fontsize=9, fontweight="bold")
        ax.set_xlabel(r"Phase 1 (fix-$\psi$)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True)

        if ax_idx == 0:
            ax.set_ylabel(r"Phase 2 (free $\psi$)")

    fig.tight_layout(w_pad=0.5)
    out = FIG_DIR / "phase1_vs_phase2_map.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--p2-glob",
        default="{cond}_ult_*",
        help="Phase 2 / ult runs under _runs/paper_gateoff ({cond} -> CS/CH/DS/DH). "
        "Default: '{cond}_ult_*'",
    )
    ap.add_argument(
        "--p1-glob",
        default="{cond}_p1_*",
        help="Phase 1 runs (used by Fig 5 only). Default: '{cond}_p1_*'",
    )
    ap.add_argument(
        "--figs",
        default="2,3,4,5",
        help="Which figures to generate, comma separated. Default: 2,3,4,5",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("Regenerating paper figures with unified style")
    print("=" * 60)
    print("Resolving runs:")
    P2_DIRS.update(resolve_runs(args.p2_glob, "p2"))
    P1_DIRS.update(resolve_runs(args.p1_glob, "p1"))

    want = {f.strip() for f in args.figs.split(",") if f.strip()}
    missing2 = [c for c in CONDS if c not in P2_DIRS]
    if missing2:
        print(f"\n  P2 runs missing for {missing2} -> skipping figures that need them")
    missing1 = [c for c in CONDS if c not in P1_DIRS]

    if "2" in want and not missing2:
        generate_fig2()
    if "3" in want and not missing2:
        generate_fig3()
    if "4" in want and not missing2:
        generate_fig4()
    if "5" in want:
        if missing1 or missing2:
            print(f"  Fig 5 skipped (p1 missing {missing1}, p2 missing {missing2})")
        else:
            generate_fig5()
    print("\n" + "=" * 60)
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
