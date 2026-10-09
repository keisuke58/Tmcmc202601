#!/usr/bin/env python3
"""
plot_gpu_tmcmc_results.py — GPU TMCMC 結果の可視化 (5-panel species + multichannel)

Usage:
    python plot_gpu_tmcmc_results.py [--run-dirs DIR1 DIR2 ...]
    python plot_gpu_tmcmc_results.py --latest   # auto-detect latest runs per condition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# --- Constants ---
SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_FULL = [
    "S. oralis",
    "A. naeslundii",
    "V. dispar",
    "F. nucleatum",
    "P. gingivalis",
]
SPECIES_COLORS = ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"]
CONDITIONS = [
    ("Dysbiotic", "HOBIC", "DH"),
    ("Commensal", "Static", "CS"),
    ("Commensal", "HOBIC", "CH"),
    ("Dysbiotic", "Static", "DS"),
]

SPECIES_MAP = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_W83": 4,
    "P. gingivalis_20709": 4,
}

# Paper style
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)


def find_latest_runs(base_dir: Path) -> dict:
    """Find most recent JAX run for each condition."""
    runs = {}
    for cond, cult, label in CONDITIONS:
        pattern = f"jax_ode_nuts_{cond}_{cult}_*"
        candidates = sorted(base_dir.glob(pattern))
        # Filter: must have samples.npy
        candidates = [c for c in candidates if (c / "samples.npy").exists()]
        if candidates:
            runs[label] = candidates[-1]  # latest by name (timestamp)
    return runs


def load_run(run_dir: Path) -> dict:
    """Load all data from a run directory."""
    result = {"dir": run_dir}
    result["samples"] = np.load(run_dir / "samples.npy")
    result["logL"] = np.load(run_dir / "logL.npy")
    with open(run_dir / "theta_MAP.json") as f:
        theta_map = json.load(f)
    result["theta_MAP"] = np.array([theta_map[str(i)] for i in range(len(theta_map))])
    with open(run_dir / "config.json") as f:
        result["config"] = json.load(f)
    return result


def load_experiment_data(data_dir: Path, condition: str, cultivation: str):
    """Load experimental species fraction data."""
    # Species distribution
    species_file = data_dir / "fig3_species_distribution_summary.csv"
    if not species_file.exists():
        species_file = data_dir / "experiment_data" / "fig3_species_distribution_summary.csv"
    species_df = pd.read_csv(species_file)
    mask = (species_df["condition"] == condition) & (
        species_df["cultivation"] == cultivation
    )
    species_df = species_df[mask]

    # Boxplot (total volume)
    box_file = data_dir / f"boxplot_{condition}_{cultivation}.csv"
    if not box_file.exists():
        box_file = data_dir / "experiment_data" / f"boxplot_{condition}_{cultivation}.csv"
    box_df = pd.read_csv(box_file)
    if "condition" in box_df.columns:
        mask_b = (box_df["condition"] == condition) & (
            box_df["cultivation"] == cultivation
        )
        box_df = box_df[mask_b]

    days = sorted(box_df["day"].unique())
    n_tp = len(days)
    data = np.zeros((n_tp, 5))
    total_vol = np.zeros(n_tp)

    smap = SPECIES_MAP
    for i, day in enumerate(days):
        dv = box_df[box_df["day"] == day]
        if len(dv) > 0:
            tv = dv["median"].values[0]
            total_vol[i] = tv
        for _, row in species_df[species_df["day"] == day].iterrows():
            sp = row["species"]
            if sp in smap:
                data[i, smap[sp]] = tv * row["median"] / 100.0

    # Day 1 initial condition (absolute volumes)
    phi_init_abs = data[0, :].copy()

    # Normalize to fractions
    row_sums = data.sum(axis=1, keepdims=True)
    fractions = data / np.maximum(row_sums, 1e-12)

    return np.array(days), fractions, total_vol, data, phi_init_abs


def simulate_ode_map(theta, condition, cultivation, phi_init=None, n_steps=2500, dt=0.0001):
    """Simulate Hamilton ODE with MAP theta using Numba solver."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

    if phi_init is not None:
        phi0_frac = np.array(phi_init, dtype=np.float64)
        phi0_frac = phi0_frac / max(phi0_frac.sum(), 1e-12)
        phi0_frac = np.clip(phi0_frac, 0.001, 0.99)
    else:
        phi0_frac = np.full(5, 0.02)

    solver = BiofilmNewtonSolver5S(
        dt=dt, maxtimestep=n_steps, c_const=25.0, alpha_const=0.0,
        K_hill=0.05, n_hill=4.0, phi_init=phi0_frac,
    )

    try:
        t_arr, g_arr = solver.run_deterministic(theta)
        return g_arr  # (n_steps+1, 12) — [:, 0:5] = phi, [:, 6:11] = psi
    except Exception:
        return np.full((n_steps + 1, 12), np.nan)


def convert_days_to_idx(days, dt=0.0001, n_steps=2500):
    """Convert experiment days to ODE time indices."""
    t_max = n_steps * dt
    day_max = max(days)
    day_scale = (t_max * 0.95) / day_max
    indices = np.array([int(d * day_scale / dt) for d in days])
    indices = np.clip(indices, 0, n_steps)
    return indices


def compute_posterior_predictive(
    samples, phi_init, condition, cultivation, days, n_max=100,
    return_all=False,
):
    """Run ODE for posterior samples.

    Returns φ̄ = φ×ψ (normalized) by default.
    If return_all=True, returns dict with 'phibar', 'phi', 'psi' arrays.
    """
    n_use = min(len(samples), n_max)
    idx_sub = np.random.choice(len(samples), n_use, replace=False)
    idx_days = convert_days_to_idx(days)

    phibar_list, phi_list, psi_list = [], [], []
    for i, si in enumerate(idx_sub):
        theta = samples[si]
        traj = simulate_ode_map(theta, condition, cultivation, phi_init=phi_init)
        phi = traj[idx_days, 0:5]
        psi = traj[idx_days, 6:11]
        phibar = phi * psi
        if not np.all(np.isfinite(phibar)):
            continue
        # Normalize φ̄
        pb = np.clip(phibar, 1e-10, 1.0)
        pb = pb / np.maximum(pb.sum(axis=1, keepdims=True), 1e-12)
        # Normalize φ
        ph = np.clip(phi, 1e-10, 1.0)
        ph = ph / np.maximum(ph.sum(axis=1, keepdims=True), 1e-12)

        phibar_list.append(pb)
        phi_list.append(ph)
        psi_list.append(np.clip(psi, 0, 1))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n_use} ({len(phibar_list)} ok)")

    print(f"    {len(phibar_list)}/{n_use} valid")

    if return_all:
        return {
            "phibar": np.array(phibar_list),
            "phi": np.array(phi_list),
            "psi": np.array(psi_list),
        }
    return np.array(phibar_list)


def plot_5panel_species(
    ax_list, days, data_frac, pred_frac, condition_label,
    r2_per_sp=None, posterior_pred=None, rep_data=None,
):
    """Plot 5-panel species fraction fit with CI bands + replicate bars."""
    sp_name_to_idx = {
        "S. oralis": 0, "A. naeslundii": 1,
        "V. dispar": 2, "V. parvula": 2,
        "F. nucleatum": 3,
        "P. gingivalis_W83": 4, "P. gingivalis_20709": 4,
    }

    for sp_idx in range(5):
        ax = ax_list[sp_idx]

        # Replicate boxplots (IQR)
        if rep_data is not None:
            sp_names = [k for k, v in sp_name_to_idx.items() if v == sp_idx]
            box_data, box_pos = [], []
            for day in days:
                vals = rep_data[
                    (rep_data["day"] == day) & (rep_data["species"].isin(sp_names))
                ]["distribution_pct"].values / 100.0
                if len(vals) > 0:
                    box_data.append(vals)
                    box_pos.append(day)
            if box_data:
                bp_rep = ax.boxplot(
                    box_data, positions=box_pos, widths=1.5,
                    patch_artist=True, showfliers=False, manage_ticks=False,
                    zorder=1,
                    medianprops=dict(color="k", linewidth=1),
                    whiskerprops=dict(color="gray", linewidth=0.8),
                    capprops=dict(color="gray", linewidth=0.8),
                )
                for patch in bp_rep["boxes"]:
                    patch.set_facecolor(SPECIES_COLORS[sp_idx])
                    patch.set_alpha(0.25)
                    patch.set_edgecolor(SPECIES_COLORS[sp_idx])
                    patch.set_linewidth(0.8)

        # Posterior CI band
        if posterior_pred is not None:
            sp_samples = posterior_pred[:, :, sp_idx]
            q05 = np.percentile(sp_samples, 5, axis=0)
            q25 = np.percentile(sp_samples, 25, axis=0)
            q75 = np.percentile(sp_samples, 75, axis=0)
            q95 = np.percentile(sp_samples, 95, axis=0)
            ax.fill_between(days, q05, q95, alpha=0.15, color=SPECIES_COLORS[sp_idx],
                            label="90% CI")
            ax.fill_between(days, q25, q75, alpha=0.3, color=SPECIES_COLORS[sp_idx],
                            label="50% CI")

        # MAP prediction
        ax.plot(days, pred_frac[:, sp_idx], "-", color=SPECIES_COLORS[sp_idx],
                linewidth=1.5, label="MAP")

        # Experimental data points
        ax.plot(days, data_frac[:, sp_idx], "o", color=SPECIES_COLORS[sp_idx],
                markersize=5, markeredgecolor="k", markeredgewidth=0.5,
                label="Exp", zorder=5)

        title = f"{SPECIES[sp_idx]}"
        if r2_per_sp is not None:
            r2 = r2_per_sp[sp_idx]
            if r2 > -10:
                title += f"  R²={r2:.2f}"
            else:
                title += f"  (~0%)"  # near-zero species, R² meaningless
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if sp_idx == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$")
            if posterior_pred is not None:
                ax.legend(fontsize=5, loc="upper right")
        if sp_idx >= 3:
            ax.set_xlabel("Day")
        ax.grid(True, alpha=0.3)


def plot_all_conditions(runs: dict, data_dir: Path, output_path: Path):
    """Main figure: 4 conditions × 5 species = 20 panels."""
    n_cond = len(runs)
    fig, axes = plt.subplots(n_cond, 5, figsize=(14, 3 * n_cond))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    # Load replicate data once
    rep_df = load_replicate_data(data_dir)

    for row_idx, (label, run_dir) in enumerate(runs.items()):
        # Find condition/cultivation
        run = load_run(run_dir)
        cfg = run["config"]
        cond = cfg["condition"]
        cult = cfg["cultivation"]
        theta_map = run["theta_MAP"]

        # Load experiment data
        days, data_frac, total_vol, data_abs, phi_init_abs = load_experiment_data(
            data_dir, cond, cult
        )

        # Replicate data for this condition
        cond_rep = rep_df[
            (rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)
        ]

        # Simulate ODE with experimental Day 1 fraction as initial condition
        phi_init_frac = data_frac[0, :]  # Day 1 fraction from experiment
        traj = simulate_ode_map(theta_map, cond, cult, phi_init=phi_init_frac)
        idx = convert_days_to_idx(days)

        # φ̄ = φ × ψ (viable bacteria volume fraction)
        phi_raw = traj[idx, 0:5]
        psi_raw = traj[idx, 6:11]
        phi_pred = phi_raw * psi_raw
        phi_pred = np.clip(phi_pred, 1e-10, 1.0)
        phi_pred = phi_pred / np.maximum(phi_pred.sum(axis=1, keepdims=True), 1e-12)

        # Posterior predictive (100 samples)
        print(f"  Computing posterior predictive for {label}...")
        posterior_pred = compute_posterior_predictive(
            run["samples"], phi_init_frac, cond, cult, days, n_max=100
        )

        # R² per species
        r2_per_sp = cfg.get("r2_per_species", None)

        # RMSE
        rmse = cfg.get("rmse", 0)
        accept = cfg.get("mean_accept", 0)
        n_part = cfg.get("n_particles", 0)
        n_mut = cfg.get("n_mutation_steps", 0)

        # Plot
        plot_5panel_species(
            axes[row_idx], days, data_frac, phi_pred, label, r2_per_sp,
            posterior_pred=posterior_pred, rep_data=cond_rep,
        )

        # Row label (outside left margin)
        fig.text(
            0.01, 1.0 - (row_idx + 0.5) / n_cond * 0.95 - 0.03,
            f"{label}\nRMSE={rmse:.3f}\nacc={accept:.2f}\n{n_part}p×{n_mut}m",
            fontsize=8, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    fig.suptitle(r"GPU TMCMC: $\bar{\varphi}_i = \varphi_i \cdot \psi_i$ (Viable Fraction)", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_phi_psi_phibar(runs: dict, data_dir: Path, output_path: Path):
    """3-row panel per condition: φ, ψ, φ̄ × 5 species."""
    rep_df = load_replicate_data(data_dir)
    sp_name_to_idx = {
        "S. oralis": 0, "A. naeslundii": 1,
        "V. dispar": 2, "V. parvula": 2,
        "F. nucleatum": 3,
        "P. gingivalis_W83": 4, "P. gingivalis_20709": 4,
    }

    for label, run_dir in runs.items():
        run = load_run(run_dir)
        cfg = run["config"]
        cond = cfg["condition"]
        cult = cfg["cultivation"]

        days, data_frac, total_vol, data_abs, phi_init_abs = load_experiment_data(
            data_dir, cond, cult
        )
        phi_init_frac = data_frac[0, :]

        print(f"  φ/ψ/φ̄ for {label}...")
        pred_all = compute_posterior_predictive(
            run["samples"], phi_init_frac, cond, cult, days, n_max=100,
            return_all=True,
        )

        # MAP trajectory
        theta_map = run["theta_MAP"]
        traj = simulate_ode_map(theta_map, cond, cult, phi_init=phi_init_frac)
        idx = convert_days_to_idx(days)
        phi_map = traj[idx, 0:5]
        psi_map = traj[idx, 6:11]
        phibar_map = phi_map * psi_map

        # Normalize
        phi_map_n = phi_map / np.maximum(phi_map.sum(axis=1, keepdims=True), 1e-12)
        phibar_map_n = phibar_map / np.maximum(phibar_map.sum(axis=1, keepdims=True), 1e-12)
        psi_map_c = np.clip(psi_map, 0, 1)

        # Replicate data
        cond_rep = rep_df[
            (rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)
        ]

        # Load viability experimental data
        viab_file = data_dir / "experiment_data" / "fig2_membrane_distribution.csv"
        viab_df = pd.read_csv(viab_file)
        viab_cond = viab_df[
            (viab_df["condition"] == cond) & (viab_df["cultivation"] == cult)
        ].sort_values("day")
        viab_days = viab_cond["day"].values
        viab_intact = viab_cond["intact_membrane_pct"].values / 100.0
        viab_error = viab_cond["error_pct"].values / 100.0

        fig, axes = plt.subplots(3, 5, figsize=(15, 8))
        row_labels = [r"$\varphi_i$ (total)", r"$\psi_i$ (viability)",
                      r"$\bar{\varphi}_i = \varphi_i \psi_i$"]
        row_data = [
            (pred_all["phi"], phi_map_n, data_frac, True),
            (pred_all["psi"], psi_map_c, None, False),
            (pred_all["phibar"], phibar_map_n, data_frac, True),
        ]

        for row_idx, (post_samples, map_pred, exp_data, show_rep) in enumerate(row_data):
            for sp_idx in range(5):
                ax = axes[row_idx, sp_idx]

                # Replicate boxplots (only for φ and φ̄ rows)
                if show_rep and rep_df is not None:
                    sp_names = [k for k, v in sp_name_to_idx.items() if v == sp_idx]
                    box_data, box_pos = [], []
                    for day in days:
                        vals = cond_rep[
                            (cond_rep["day"] == day) & (cond_rep["species"].isin(sp_names))
                        ]["distribution_pct"].values / 100.0
                        if len(vals) > 0:
                            box_data.append(vals)
                            box_pos.append(day)
                    if box_data:
                        bp = ax.boxplot(
                            box_data, positions=box_pos, widths=1.5,
                            patch_artist=True, showfliers=False, manage_ticks=False,
                            zorder=1,
                            medianprops=dict(color="k", linewidth=1),
                            whiskerprops=dict(color="gray", linewidth=0.8),
                            capprops=dict(color="gray", linewidth=0.8),
                        )
                        for patch in bp["boxes"]:
                            patch.set_facecolor(SPECIES_COLORS[sp_idx])
                            patch.set_alpha(0.2)
                            patch.set_edgecolor(SPECIES_COLORS[sp_idx])

                # Posterior CI
                sp_samples = post_samples[:, :, sp_idx]
                q05 = np.percentile(sp_samples, 5, axis=0)
                q25 = np.percentile(sp_samples, 25, axis=0)
                q75 = np.percentile(sp_samples, 75, axis=0)
                q95 = np.percentile(sp_samples, 95, axis=0)
                ax.fill_between(days, q05, q95, alpha=0.15, color=SPECIES_COLORS[sp_idx])
                ax.fill_between(days, q25, q75, alpha=0.3, color=SPECIES_COLORS[sp_idx])

                # MAP line
                if map_pred.ndim == 2:
                    ax.plot(days, map_pred[:, sp_idx], "-", color=SPECIES_COLORS[sp_idx],
                            linewidth=1.5)
                else:
                    ax.plot(days, map_pred[:, sp_idx], "-", color=SPECIES_COLORS[sp_idx],
                            linewidth=1.5)

                # Exp data points (only for φ and φ̄)
                if exp_data is not None:
                    ax.plot(days, exp_data[:, sp_idx], "o", color=SPECIES_COLORS[sp_idx],
                            markersize=4, markeredgecolor="k", markeredgewidth=0.5, zorder=5)

                # Viability experimental data on ψ row
                if row_idx == 1:
                    ax.errorbar(
                        viab_days, viab_intact, yerr=viab_error,
                        fmt="^", color="k", markersize=5,
                        capsize=3, linewidth=1.2, alpha=0.9, zorder=5,
                        label="Exp viab" if sp_idx == 0 else None,
                    )

                if row_idx == 0:
                    ax.set_title(SPECIES[sp_idx], fontsize=9)
                if sp_idx == 0:
                    ax.set_ylabel(row_labels[row_idx], fontsize=9)
                    if row_idx == 1:
                        ax.legend(fontsize=6, loc="lower left")
                if row_idx == 2:
                    ax.set_xlabel("Day")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)

        rmse = cfg.get("rmse", 0)
        fig.suptitle(f"{label} ({cond} {cult}) — RMSE={rmse:.4f}", fontsize=13, y=1.01)
        fig.tight_layout()
        out = output_path.parent / f"gpu_tmcmc_phi_psi_{label}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close(fig)


def plot_posterior_comparison(runs: dict, output_path: Path):
    """Violin/box plot comparing posterior distributions across conditions."""
    n_params = 20
    fig, axes = plt.subplots(4, 5, figsize=(14, 10))
    axes = axes.flatten()

    param_labels = [
        "a11", "a12", "a13", "a14", "a15",
        "a21", "a22", "a23", "a24", "a25",
        "a31", "a32", "a33", "a34", "a35",
        "a41", "a42", "a43", "a44", "a45",
    ]

    colors_cond = {"DH": "#e41a1c", "CS": "#377eb8", "CH": "#4daf4a", "DS": "#984ea3"}

    for p_idx in range(n_params):
        ax = axes[p_idx]
        positions = []
        bp_data = []
        labels_cond = []

        for i, (label, run_dir) in enumerate(runs.items()):
            run = load_run(run_dir)
            samples = run["samples"][:, p_idx]
            positions.append(i)
            bp_data.append(samples)
            labels_cond.append(label)

        bp = ax.boxplot(
            bp_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )
        for i, (patch, label) in enumerate(zip(bp["boxes"], labels_cond)):
            patch.set_facecolor(colors_cond.get(label, "gray"))
            patch.set_alpha(0.6)

        ax.set_title(param_labels[p_idx], fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_cond, fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Posterior Parameter Distributions (4 Conditions)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def load_replicate_data(data_dir: Path):
    """Load replicate-level species distribution data."""
    rep_file = data_dir / "experiment_data" / "fig3_species_distribution_replicates.csv"
    if not rep_file.exists():
        rep_file = data_dir / "fig3_species_distribution_replicates.csv"
    return pd.read_csv(rep_file)


def plot_boxplot_posterior(runs: dict, data_dir: Path, output_path: Path):
    """Boxplot: posterior predictive + replicate experimental bars side by side."""
    n_cond = len(runs)
    fig, axes = plt.subplots(n_cond, 5, figsize=(16, 3.2 * n_cond))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    rep_df = load_replicate_data(data_dir)

    # Map species names to index (handle V. parvula / P. gingivalis variants)
    sp_name_to_idx = {
        "S. oralis": 0, "A. naeslundii": 1,
        "V. dispar": 2, "V. parvula": 2,
        "F. nucleatum": 3,
        "P. gingivalis_W83": 4, "P. gingivalis_20709": 4,
    }

    for row_idx, (label, run_dir) in enumerate(runs.items()):
        run = load_run(run_dir)
        cfg = run["config"]
        cond = cfg["condition"]
        cult = cfg["cultivation"]

        days, data_frac, total_vol, data_abs, phi_init_abs = load_experiment_data(
            data_dir, cond, cult
        )

        phi_init_frac = data_frac[0, :]
        print(f"  Boxplot posterior for {label}...")
        posterior_pred = compute_posterior_predictive(
            run["samples"], phi_init_frac, cond, cult, days, n_max=100
        )

        # Get replicates for this condition
        cond_rep = rep_df[
            (rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)
        ]

        bar_w = 1.0

        for sp_idx in range(5):
            ax = axes[row_idx, sp_idx]
            sp_data = posterior_pred[:, :, sp_idx]  # (n_samples, n_days)

            for t_idx, day in enumerate(days):
                # Replicate data bar (left, colored)
                sp_names = [k for k, v in sp_name_to_idx.items() if v == sp_idx]
                rep_vals = cond_rep[
                    (cond_rep["day"] == day) & (cond_rep["species"].isin(sp_names))
                ]["distribution_pct"].values / 100.0

                if len(rep_vals) > 0:
                    rep_mean = np.mean(rep_vals)
                    rep_std = np.std(rep_vals)
                    ax.bar(
                        day - bar_w * 0.28, rep_mean, width=bar_w * 0.45,
                        color=SPECIES_COLORS[sp_idx], alpha=0.7,
                        edgecolor="k", linewidth=0.5, zorder=3,
                    )
                    ax.errorbar(
                        day - bar_w * 0.28, rep_mean, yerr=rep_std,
                        fmt="none", ecolor="k", capsize=2, linewidth=1, zorder=4,
                    )

                # Posterior boxplot (right, gray)
                bp = ax.boxplot(
                    [sp_data[:, t_idx]],
                    positions=[day + bar_w * 0.28],
                    widths=bar_w * 0.45,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor("#CCCCCC")
                    patch.set_alpha(0.7)
                    patch.set_edgecolor("k")
                    patch.set_linewidth(0.5)
                for median in bp["medians"]:
                    median.set_color("red")
                    median.set_linewidth(1.5)

            ax.set_title(f"{SPECIES[sp_idx]}", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-1, max(days) + 3)
            ax.set_xticks(days)
            if sp_idx == 0:
                ax.set_ylabel(r"$\bar{\varphi}_i$")
            if row_idx == n_cond - 1:
                ax.set_xlabel("Day")
            ax.grid(True, alpha=0.3, axis="y")

        # Compute RMSE and R² from posterior mean vs experimental
        pred_mean = np.mean(posterior_pred, axis=0)  # (n_days, 5)
        rmse = np.sqrt(np.mean((pred_mean - data_frac) ** 2))
        ss_res = np.sum((data_frac - pred_mean) ** 2)
        ss_tot = np.sum((data_frac - np.mean(data_frac)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        # Row label (outside left margin)
        fig.text(
            0.01, 1.0 - (row_idx + 0.5) / n_cond * 0.93 - 0.04,
            f"{label}\nRMSE={rmse:.3f}\nR²={r2:.2f}",
            fontsize=9, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.7, edgecolor="k", linewidth=0.5,
              label="Exp (mean±std)"),
        Patch(facecolor="#CCCCCC", alpha=0.7, edgecolor="k", linewidth=0.5,
              label="Posterior (model)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(r"Posterior Predictive $\bar{\varphi}_i$ vs Experimental Replicates",
                 fontsize=13, y=1.04)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_summary_table(runs: dict, output_path: Path):
    """Summary metrics table as a figure."""
    rows = []
    for label, run_dir in runs.items():
        run = load_run(run_dir)
        cfg = run["config"]
        mc_rmse = cfg.get("multichannel_rmse", {})
        r2_per = cfg.get("r2_per_species", [0] * 5)

        # R² for dominant species only (>5% mean fraction, exclude Fn/Pg in commensal)
        r2_dominant = [r for r in r2_per if r > -10]
        r2_dom_mean = np.mean(r2_dominant) if r2_dominant else float("nan")

        row = {
            "Cond": label,
            "N_p": cfg.get("n_particles", "?"),
            "Stages": cfg.get("n_stages", "?"),
            "Time": f"{cfg.get('total_time_s', 0):.0f}s",
            "Accept": f"{cfg.get('mean_accept', 0):.2f}",
            "RMSE": f"{cfg.get('rmse', 0):.4f}",
            "MAE": f"{cfg.get('mae', 0):.4f}",
            "R²(dom)": f"{r2_dom_mean:.3f}" if not np.isnan(r2_dom_mean) else "-",
            "Ch2": f"{mc_rmse.get('ch2_total', {}).get('rmse', '-'):.4f}"
            if "ch2_total" in mc_rmse
            else "-",
            "logZ": f"{cfg.get('log_evidence', 0):.2f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.4 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Color header
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best RMSE
    rmse_vals = [float(r["RMSE"]) for r in rows]
    best_idx = np.argmin(rmse_vals)
    for j in range(len(df.columns)):
        tbl[best_idx + 1, j].set_facecolor("#E2EFDA")

    fig.suptitle("GPU TMCMC Summary — R²(dom) = dominant species only", fontsize=12, y=0.95)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GPU TMCMC results")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="Run directories to plot",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Auto-detect latest runs per condition",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    runs_dir = base / "_runs"
    data_dir = base.parent  # data_5species/

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = base / "_runs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dirs:
        runs = {}
        for d in args.run_dirs:
            p = Path(d)
            run = load_run(p)
            c = run["config"]["condition"]
            v = run["config"]["cultivation"]
            label = ("DH" if c == "Dysbiotic" and v == "HOBIC"
                     else "CS" if c == "Commensal" and v == "Static"
                     else "CH" if c == "Commensal" and v == "HOBIC"
                     else "DS")
            runs[label] = p
    else:
        runs = find_latest_runs(runs_dir)

    if not runs:
        print("No runs found!")
        return

    print(f"Found {len(runs)} runs: {list(runs.keys())}")
    for label, d in runs.items():
        print(f"  {label}: {d.name}")

    # Plot all figures
    plot_all_conditions(runs, data_dir, out_dir / "gpu_tmcmc_5panel.png")
    plot_phi_psi_phibar(runs, data_dir, out_dir / "gpu_tmcmc_phi_psi.png")
    plot_boxplot_posterior(runs, data_dir, out_dir / "gpu_tmcmc_boxplot.png")
    plot_posterior_comparison(runs, out_dir / "gpu_tmcmc_posterior.png")
    plot_summary_table(runs, out_dir / "gpu_tmcmc_summary.png")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
