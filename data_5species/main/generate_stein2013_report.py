#!/usr/bin/env python3
"""
generate_stein2013_report.py — Stein 2013 gut microbiome TMCMC report.

Usage (after TMCMC completes):
    python generate_stein2013_report.py --mouse pop2_rep1_id1
    python generate_stein2013_report.py --mouse all
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
SP_COLORS = plt.cm.tab20(np.linspace(0, 1, N_SP))
DATA_DIR = SCRIPT_DIR / "external_data" / "stein2013"


def load_mouse_csv(mouse_key):
    import csv

    with open(DATA_DIR / f"stein2013_{mouse_key}.csv") as f:
        reader = csv.reader(f)
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


def fig_species_panels(t_days, obs, pred, mouse_key, fig_dir):
    """4x3 panel: one per species."""
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True)
    axes_flat = axes.flatten()
    for j in range(N_SP):
        ax = axes_flat[j]
        ax.scatter(
            t_days,
            obs[:, j],
            s=25,
            color=SP_COLORS[j],
            edgecolors="k",
            linewidths=0.4,
            zorder=5,
            label="Data",
        )
        ax.plot(t_days, pred[:, j], color=SP_COLORS[j], lw=1.5, zorder=3, label="MAP")
        ax.set_title(SP_NAMES[j], fontsize=9)
        ax.grid(True)
        ax.set_ylim(-0.02, max(obs[:, j].max(), pred[:, j].max()) * 1.2 + 0.02)
        if j >= 8:
            ax.set_xlabel("Day")
        if j % 4 == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$")
    # Hide unused
    axes_flat[11].set_visible(False)
    axes_flat[0].legend(fontsize=6, loc="upper right")

    m = compute_metrics(obs, pred)
    fig.suptitle(
        f"Stein 2013 — {mouse_key} (11 species, RMSE={m['rmse']:.3f}, Cos={m['cosine']:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_panels_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_panels_{mouse_key}.png")
    plt.close(fig)
    return m


def fig_A_heatmap(theta, mouse_key, fig_dir, A_glv=None, glv_species=None):
    """Interaction matrix heatmap, optionally side-by-side with gLV."""
    A_ham = theta_to_A(theta, N_SP)
    n_plots = 2 if A_glv is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    vmax = np.abs(A_ham).max()
    if A_glv is not None:
        vmax = max(vmax, np.abs(A_glv).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Hamilton
    im = axes[0].imshow(A_ham, cmap="RdBu_r", norm=norm, aspect="equal")
    for i in range(N_SP):
        for j in range(N_SP):
            c = "white" if abs(A_ham[i, j]) > vmax * 0.6 else "black"
            axes[0].text(j, i, f"{A_ham[i,j]:.1f}", ha="center", va="center", fontsize=6, color=c)
    axes[0].set_xticks(range(N_SP))
    axes[0].set_yticks(range(N_SP))
    axes[0].set_xticklabels(SP_NAMES, fontsize=6, rotation=45, ha="right")
    axes[0].set_yticklabels(SP_NAMES, fontsize=6)
    axes[0].set_title("Hamilton ODE (this work)", fontweight="bold")

    # gLV (Stein 2013)
    if A_glv is not None:
        im2 = axes[1].imshow(A_glv, cmap="RdBu_r", norm=norm, aspect="equal")
        for i in range(N_SP):
            for j in range(N_SP):
                c = "white" if abs(A_glv[i, j]) > vmax * 0.6 else "black"
                axes[1].text(
                    j, i, f"{A_glv[i,j]:.1f}", ha="center", va="center", fontsize=6, color=c
                )
        axes[1].set_xticks(range(N_SP))
        axes[1].set_yticks(range(N_SP))
        glv_short = [s[:10] for s in glv_species] if glv_species else SP_NAMES
        axes[1].set_xticklabels(glv_short, fontsize=6, rotation=45, ha="right")
        axes[1].set_yticklabels(glv_short, fontsize=6)
        axes[1].set_title("gLV (Stein 2013)", fontweight="bold")

    fig.subplots_adjust(right=0.88, wspace=0.4)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r"$A_{ij}$")
    fig.savefig(fig_dir / f"fig_A_comparison_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_A_comparison_{mouse_key}.png")
    plt.close(fig)


def fig_growth_rates(theta, mouse_key, fig_dir):
    n_A = N_SP * (N_SP + 1) // 2
    mu = theta[n_A : n_A + N_SP]
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    colors = SP_COLORS[:N_SP]
    bars = ax.bar(range(N_SP), mu, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(N_SP))
    ax.set_xticklabels(SP_NAMES, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel(r"$\mu_i$")
    ax.set_title(f"Growth rates — {mouse_key}", fontweight="bold")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_mu_{mouse_key}.pdf")
    fig.savefig(fig_dir / f"fig_mu_{mouse_key}.png")
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

    # Find all completed runs
    if args.mouse == "all":
        runs = sorted(SCRIPT_DIR.glob("_runs/stein2013_pop*"))
        mice = [r.name.replace("stein2013_", "").rsplit("_", 1)[0] for r in runs]
    else:
        mice = [args.mouse]

    A_glv, glv_species = load_glv_matrix()

    results = {}
    for mouse_key in mice:
        run_dir = list(SCRIPT_DIR.glob(f"_runs/stein2013_{mouse_key}_*"))
        if not run_dir:
            print(f"  {mouse_key}: no results found, skipping")
            continue
        run_dir = run_dir[0]

        print(f"\n=== {mouse_key} ===")
        theta, samples, logL, config = load_run(run_dir)
        t_days, data_all = load_mouse_csv(mouse_key)

        # Predict with MAP
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        sys.path.insert(0, str(SCRIPT_DIR))
        from hamilton_ode_jax_nsp import simulate_0d_nsp

        ic = data_all[0].copy()
        ic = np.clip(ic, 0.001, 0.99)
        ic = ic / ic.sum()
        obs = data_all[1:]
        t_fit = t_days[1:]

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
        print(f"  RMSE={m['rmse']:.4f}, Cos={m['cosine']:.4f}, logL={logL.max():.1f}")

        # Figures
        print("  Fig: species panels...")
        fig_species_panels(t_fit, obs, pred, mouse_key, FIG_DIR)
        print("  Fig: A matrix...")
        fig_A_heatmap(theta, mouse_key, FIG_DIR, A_glv, glv_species)
        print("  Fig: growth rates...")
        fig_growth_rates(theta, mouse_key, FIG_DIR)

        results[mouse_key] = {
            "rmse": m["rmse"],
            "cosine": m["cosine"],
            "spearman": m["spearman"],
            "max_logL": float(logL.max()),
            "n_stages": config.get("n_stages", "?"),
            "time_s": config.get("total_time_s", "?"),
        }

    # Save summary
    with open(OUT / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {OUT / 'results_summary.json'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
