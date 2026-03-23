#!/usr/bin/env python3
"""
generate_stein2013_report.py — Stein 2013 gut microbiome TMCMC report.
Same visual style as Siddiqui 6sp report (Latin Modern, 300 dpi, 2x3 panels).
"""
import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_dist

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

N_SP = 11
SP_NAMES = [
    "Enterobact",
    "Blautia",
    "Barnesiella",
    "Mollicutes",
    "Lachnospi",
    "Akkermansia",
    "C.difficile",
    "unc_Lachno",
    "Coprobacil",
    "Enterococc",
    "Other",
]
SP_NAMES_ITALIC = [
    r"Enterobact.",
    r"$\it{Blautia}$",
    r"$\it{Barnesiella}$",
    r"unc. Mollicutes",
    r"Lachnospi.",
    r"$\it{Akkermansia}$",
    r"$\it{C.\ difficile}$",
    r"unc. Lachno.",
    r"$\it{Coprobacillus}$",
    r"$\it{Enterococcus}$",
    r"Other",
]
SP_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
]
DATA_DIR = SCRIPT_DIR / "external_data" / "stein2013"


def load_mouse_csv(mouse_key):
    import csv as csv_mod

    with open(DATA_DIR / f"stein2013_{mouse_key}.csv") as f:
        reader = csv_mod.reader(f)
        header = next(reader)
        rows = list(reader)
    t_days = np.array([float(r[0]) for r in rows])
    data = np.array([[float(v) for v in r[1:]] for r in rows])
    return t_days, data


def load_run(run_dir):
    with open(run_dir / "theta_MAP.json") as f:
        raw = json.load(f)
    n_p = len(raw)
    theta = np.array([raw[f"theta_{i}"] for i in range(n_p)])
    samples = np.load(run_dir / "samples.npy")
    logL = np.load(run_dir / "logL.npy")
    config = {}
    if (run_dir / "config.json").exists():
        with open(run_dir / "config.json") as f:
            config = json.load(f)
    return theta, samples, logL, config


def load_glv_matrix():
    p = DATA_DIR / "stein2013_glv_A_matrix.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return np.array(d["A_glv"]), d["species"]
    return None, None


def theta_to_A(theta, n_sp):
    A = np.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A[i, j] = theta[idx]
            A[j, i] = theta[idx]
            idx += 1
    return A


def compute_metrics(obs, pred):
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    cos = 1.0 - cosine_dist(obs.flatten(), pred.flatten())
    rho, _ = spearmanr(obs.flatten(), pred.flatten())
    return {"rmse": rmse, "cosine": cos, "spearman": rho}


def fig_species_panels(t_days, obs, pred, mouse_key, fig_dir, ic=None, ic_day=None):
    """3x4 panel (11 species + 1 blank), paper style."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes_flat = axes.flatten()

    # Model time → days mapping for MAP line
    # pred is already at t_days indices, so just connect with lines
    for j in range(N_SP):
        ax = axes_flat[j]
        # Data scatter
        ax.scatter(
            t_days, obs[:, j], s=30, color=SP_COLORS[j], edgecolors="k", linewidths=0.5, zorder=5
        )
        # MAP line
        ax.plot(
            t_days,
            pred[:, j],
            color=SP_COLORS[j],
            lw=2.0,
            zorder=3,
            marker="s",
            ms=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        # IC diamond
        if ic is not None and ic_day is not None:
            ax.scatter(
                [ic_day],
                [ic[j]],
                s=50,
                color=SP_COLORS[j],
                marker="D",
                edgecolors="k",
                linewidths=0.8,
                zorder=6,
            )

        ax.set_title(SP_NAMES_ITALIC[j], fontsize=9)
        ax.grid(True)
        ymax = max(obs[:, j].max(), pred[:, j].max()) * 1.3 + 0.02
        ax.set_ylim(-0.02, 1.05)
        x_margin = max((t_days[-1] - t_days[0]) * 0.08, 1.5)
        ax.set_xlim(t_days[0] - x_margin, t_days[-1] + x_margin)
        # Day ticks at measurement points
        if ic_day is not None and ic_day not in t_days:
            all_ticks = sorted(set([ic_day] + list(t_days)))
        else:
            all_ticks = list(t_days)
        ax.set_xticks([int(d) for d in all_ticks])
        ax.set_xticklabels([str(int(d)) for d in all_ticks], fontsize=7)
        if j >= 8:
            ax.set_xlabel("Day")
        if j % 4 == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$")

    axes_flat[11].set_visible(False)

    # Legend
    handles = [
        Line2D([0], [0], marker="s", color="gray", lw=2, ms=4, label="MAP"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=6,
            label="Data",
        ),
    ]
    if ic is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor="gray",
                markeredgecolor="k",
                markersize=6,
                label="IC",
            )
        )
    axes_flat[0].legend(handles=handles, fontsize=7, loc="upper right")

    m = compute_metrics(obs, pred)
    fig.suptitle(
        f"Stein 2013 — {mouse_key} (11 species, 77 params)\n"
        f"RMSE = {m['rmse']:.3f}, Cosine = {m['cosine']:.3f}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_panels_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_panels_{mouse_key}.png")
    plt.close(fig)
    return m


def _reorder_glv_to_our_order(A_glv, glv_species):
    """Reorder gLV A matrix from Stein's species order to our order."""
    # Stein order -> our order mapping
    glv_to_ours = {
        "Barnesiella": 2,
        "Lachnospiraceae": 4,
        "unc_Lachnospiraceae": 7,
        "Other": 10,
        "Blautia": 1,
        "unc_Mollicutes": 3,
        "Akkermansia": 5,
        "Coprobacillus": 8,
        "Clostridium_difficile": 6,
        "Enterococcus": 9,
        "Enterobacteriaceae": 0,
    }
    n = len(glv_species)
    A_reordered = np.zeros((n, n))
    for i_glv in range(n):
        for j_glv in range(n):
            i_our = glv_to_ours.get(glv_species[i_glv], i_glv)
            j_our = glv_to_ours.get(glv_species[j_glv], j_glv)
            A_reordered[i_our, j_our] = A_glv[i_glv, j_glv]
    return A_reordered


def fig_A_heatmap(theta, mouse_key, fig_dir, A_glv=None, glv_species=None):
    A_ham = theta_to_A(theta, N_SP)
    # Reorder gLV to our species order
    if A_glv is not None and glv_species is not None:
        A_glv_reordered = _reorder_glv_to_our_order(A_glv, glv_species)
    else:
        A_glv_reordered = A_glv

    n_plots = 2 if A_glv is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    vmax = np.abs(A_ham).max()
    if A_glv_reordered is not None:
        vmax = max(vmax, np.abs(A_glv_reordered).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    matrices = [(axes[0], A_ham, "Hamilton ODE (this work)")]
    if A_glv_reordered is not None:
        matrices.append((axes[1], A_glv_reordered, "gLV (Stein 2013)"))

    for ax, A, title in matrices:
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
        for i in range(N_SP):
            for j in range(N_SP):
                c = "white" if abs(A[i, j]) > vmax * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{A[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=c,
                    fontweight="bold",
                )
        # Same labels for both
        ax.set_xticks(range(N_SP))
        ax.set_yticks(range(N_SP))
        ax.set_xticklabels(SP_NAMES, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(SP_NAMES, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.subplots_adjust(right=0.87, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$A_{ij}$ (interaction strength)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(fig_dir / f"fig_A_comparison_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_A_comparison_{mouse_key}.png")
    plt.close(fig)


def fig_growth_rates(theta, mouse_key, fig_dir):
    n_A = N_SP * (N_SP + 1) // 2
    mu = theta[n_A : n_A + N_SP]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    bars = ax.bar(range(N_SP), mu, color=SP_COLORS, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(N_SP))
    ax.set_xticklabels(SP_NAMES_ITALIC, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(r"$\mu_i$ (growth rate)", fontsize=10)
    ax.set_title(f"Growth rates — {mouse_key}", fontsize=11, fontweight="bold")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, axis="y")
    # Value labels
    for bar, val in zip(bars, mu):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_mu_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_mu_{mouse_key}.png")
    plt.close(fig)


def fig_residuals(t_days, obs, pred, mouse_key, fig_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for j in range(N_SP):
        residual = pred[:, j] - obs[:, j]
        ax.plot(t_days, residual, color=SP_COLORS[j], marker="s", ms=4, label=SP_NAMES[j], lw=1.0)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Residual (pred $-$ obs)")
    ax.set_title(f"Residuals — {mouse_key}", fontweight="bold")
    ax.legend(ncol=4, fontsize=6, loc="upper center", frameon=True, edgecolor="0.8")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_residuals_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_residuals_{mouse_key}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse", default="pop2_rep1_id1")
    parser.add_argument("--n-steps", type=int, default=2500)
    args = parser.parse_args()

    OUT = SCRIPT_DIR / "_runs" / "stein2013_report"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR = OUT / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from hamilton_ode_jax_nsp import simulate_0d_nsp

    A_glv, glv_species = load_glv_matrix()

    # Best runs
    best_runs = {
        "pop1_rep1_id2": ("_runs/stein2013_pop1_rep1_id2_2000p", 1),
        "pop1_rep2_id5": ("_runs/stein2013_pop1_rep2_id5_2000p", 1),
        "pop1_rep3_id8": ("_runs/stein2013_pop1_rep3_id8_2000p", 1),
        "pop3_rep1_id3": ("_runs/stein2013_pop3r1_day2ic_5000p", 2),
        "pop3_rep2_id6": ("_runs/stein2013_pop3r2_day2ic_5000p", 2),
        "pop3_rep3_id9": ("_runs/stein2013_pop3r3_day2ic_5000p", 2),
    }

    if args.mouse == "all":
        mice = list(best_runs.keys())
    elif args.mouse in best_runs:
        mice = [args.mouse]
    else:
        mice = [args.mouse]

    results = {}
    for mouse_key in mice:
        if mouse_key not in best_runs:
            print(f"  {mouse_key}: no best run configured, skip")
            continue
        run_path, sfi = best_runs[mouse_key]
        run_dir = SCRIPT_DIR / run_path
        if not (run_dir / "theta_MAP.json").exists():
            print(f"  {mouse_key}: no results at {run_path}, skip")
            continue

        print(f"\n=== {mouse_key} (sfi={sfi}) ===")
        theta, samples, logL, config = load_run(run_dir)
        t_days, data_all = load_mouse_csv(mouse_key)

        ic_idx = sfi - 1
        ic = data_all[ic_idx].copy()
        ic = np.clip(ic, 0.001, 0.99)
        ic = ic / ic.sum()
        ic_day = t_days[ic_idx]
        obs = data_all[sfi:]
        t_fit = t_days[sfi:]

        dt = config.get("dt", 1e-4)
        n_steps = config.get("n_steps", args.n_steps)
        t_max = n_steps * dt
        scale = (t_max * 0.95) / t_fit.max()
        idx = np.clip(np.round(t_fit * scale / dt).astype(int), 0, n_steps)

        traj = np.array(
            simulate_0d_nsp(
                jnp.array(theta),
                n_sp=N_SP,
                n_steps=n_steps,
                dt=dt,
                phi_init=jnp.array(ic),
            )
        )
        pred = traj[idx]
        pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)

        m = compute_metrics(obs, pred)
        print(f"  RMSE={m['rmse']:.4f}, Cos={m['cosine']:.4f}")

        print("  Fig: species panels...")
        fig_species_panels(t_fit, obs, pred, mouse_key, FIG_DIR, ic=ic, ic_day=ic_day)
        print("  Fig: A matrix...")
        fig_A_heatmap(theta, mouse_key, FIG_DIR, A_glv, glv_species)
        print("  Fig: growth rates...")
        fig_growth_rates(theta, mouse_key, FIG_DIR)
        print("  Fig: residuals...")
        fig_residuals(t_fit, obs, pred, mouse_key, FIG_DIR)

        # Save data
        np.save(FIG_DIR / f"pred_{mouse_key}.npy", pred)
        np.save(FIG_DIR / f"obs_{mouse_key}.npy", obs)

        results[mouse_key] = {
            "rmse": m["rmse"],
            "cosine": m["cosine"],
            "spearman": m["spearman"],
            "max_logL": float(logL.max()),
            "start_from_idx": sfi,
        }

    with open(OUT / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! {len(results)} mice. Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
