#!/usr/bin/env python3
"""Generate So IC sensitivity figure using the CORRECT Hamilton ODE solver.
Must run on GPU server with klempt_fem env:
  ssh vancouver01 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate klempt_fem && \
    cd ~/Tmcmc202601/data_5species/main && python3 ~/Tmcmc202601/docs/generate_fig_so_sensitivity_gpu.py"
"""
import sys
import os

sys.path.insert(0, os.path.expanduser("~/Tmcmc202601/data_5species/main"))

import json
import csv
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only for plotting
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hamilton_ode_jax import simulate_0d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

RUNS = Path.home() / "Tmcmc202601/data_5species/main/_runs"
DATA_CSV = (
    Path.home() / "Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_summary.csv"
)
FIG_DIR = Path.home() / "Tmcmc202601/docs/figures"
FIG_DIR.mkdir(exist_ok=True)

SPECIES_MAP = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
}
EXP_DAYS = np.array([1, 3, 6, 10, 15, 21])
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
SIGMA_SO = {"CS": 0.19, "CH": 0.19, "DS": 0.08, "DH": 0.08}

# Phase 1 fix-psi best runs (mc=False)
PHASE1_DIRS = {
    "CS": "jax_ode_nuts_Commensal_Static_20260320_015113",
    "CH": "jax_ode_nuts_Commensal_HOBIC_20260320_015554",
    "DS": "jax_ode_nuts_Dysbiotic_Static_20260320_015831",
    "DH": "jax_ode_nuts_Dysbiotic_HOBIC_20260320_021506",
}


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


def run_hamilton_ode(theta, ic, n_steps=2500, dt=1e-4):
    """Run the Hamilton ODE solver (same as TMCMC uses)."""
    phi_init = jnp.array(ic, dtype=jnp.float64)
    theta_jax = jnp.array(theta, dtype=jnp.float64)
    traj = simulate_0d(
        theta_jax, n_steps=n_steps, dt=dt, phi_init=phi_init, K_hill=0.05, n_hill=4.0
    )
    return np.array(traj)


def main():
    exp_data = load_exp_data()
    model_days = np.linspace(0, 21, 2501)

    # JIT warmup
    print("JIT warmup...")
    _ = run_hamilton_ode(np.zeros(20), np.full(5, 0.2))
    print("OK")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)

    for ax_idx, cond_key in enumerate(["CS", "CH", "DS", "DH"]):
        ax = axes[ax_idx]
        run_dir = RUNS / PHASE1_DIRS[cond_key]

        # Check if dir exists, fallback to search
        if not run_dir.exists():
            condition, cultivation = COND_KEYS[cond_key]
            prefix = f"jax_ode_nuts_{condition}_{cultivation}_"
            candidates = sorted(RUNS.glob(prefix + "*"))
            # Find best fix-psi (mc=False)
            for c in reversed(candidates):
                if (c / "theta_MAP.json").exists() and (c / "config.json").exists():
                    cfg = json.load(open(c / "config.json"))
                    if not cfg.get("multichannel", False):
                        run_dir = c
                        break

        mj = json.load(open(run_dir / "theta_MAP.json"))
        theta_map = np.array([mj[str(i)] for i in range(20)])
        obs = exp_data[cond_key]
        ic_nom = obs[0].copy()
        # Normalize IC
        ic_nom = np.clip(ic_nom, 0.001, 0.99)
        ic_nom /= ic_nom.sum()
        sig = SIGMA_SO[cond_key]

        print(f"  {cond_key}: {run_dir.name}, IC So={ic_nom[0]:.4f}")

        # Simulate perturbed trajectories
        deltas = [-2 * sig, -1 * sig, 0, +1 * sig, +2 * sig]
        trajs = []
        for delta in deltas:
            ic = ic_nom.copy()
            ic[0] += delta
            ic = np.clip(ic, 1e-4, None)
            ic /= ic.sum()
            traj = run_hamilton_ode(theta_map, ic)
            trajs.append(traj)

        # Fill between ±2σ and ±1σ envelopes
        lo2 = np.minimum(trajs[0][:, 0], trajs[4][:, 0])
        hi2 = np.maximum(trajs[0][:, 0], trajs[4][:, 0])
        lo1 = np.minimum(trajs[1][:, 0], trajs[3][:, 0])
        hi1 = np.maximum(trajs[1][:, 0], trajs[3][:, 0])

        ax.fill_between(
            model_days,
            lo2,
            hi2,
            color="#1f77b4",
            alpha=0.12,
            label=r"$\pm 2\sigma$" if ax_idx == 0 else None,
        )
        ax.fill_between(
            model_days,
            lo1,
            hi1,
            color="#1f77b4",
            alpha=0.25,
            label=r"$\pm 1\sigma$" if ax_idx == 0 else None,
        )

        # Nominal MAP trajectory
        ax.plot(
            model_days,
            trajs[2][:, 0],
            color="#1f77b4",
            lw=1.5,
            zorder=3,
            label="MAP" if ax_idx == 0 else None,
        )

        # Experimental So data with error bars
        so_std = sig * np.ones(len(EXP_DAYS))
        ax.errorbar(
            EXP_DAYS,
            obs[:, 0],
            yerr=so_std,
            fmt="o",
            color="#1f77b4",
            ms=5,
            capsize=3,
            capthick=0.7,
            lw=0.8,
            ecolor="gray",
            markeredgecolor="k",
            markeredgewidth=0.5,
            zorder=5,
            label="data" if ax_idx == 0 else None,
        )

        # Annotate range at Day 21
        range_21 = hi2[-1] - lo2[-1]
        ax.annotate(
            f"$\\Delta = {range_21:.2f}$",
            xy=(21, (hi2[-1] + lo2[-1]) / 2),
            xytext=(-5, 0),
            textcoords="offset points",
            fontsize=7,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8),
        )

        ax.set_title(COND_LABELS[cond_key], fontsize=9, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.15, lw=0.5)

        if ax_idx == 0:
            ax.set_ylabel(r"Species fraction $\phi_i$")

    axes[0].legend(
        loc="upper right", framealpha=0.9, handlelength=1.5, borderpad=0.3, handletextpad=0.4
    )

    fig.tight_layout(w_pad=0.3)
    out = FIG_DIR / "so_ic_sensitivity.pdf"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
