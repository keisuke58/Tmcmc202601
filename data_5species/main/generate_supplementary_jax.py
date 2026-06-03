#!/usr/bin/env python3
"""
generate_supplementary_jax.py — Publication-quality supplementary figures
from JAX ODE TMCMC results.

Usage:
    python generate_supplementary_jax.py --runs-dir _runs --output-dir figures_supp
    python generate_supplementary_jax.py --best-per-condition  # auto-select best RMSE per condition
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ham = _load_module(SCRIPT_DIR / "hamilton_ode_jax.py", "hamilton_ode_jax")
_est = _load_module(SCRIPT_DIR / "estimate_reduced_nishioka.py", "estimate_reduced_nishioka")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

simulate_0d = _ham.simulate_0d
load_experimental_data = _est.load_experimental_data
convert_days_to_model_time = _est.convert_days_to_model_time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

SPECIES_NAMES = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
CONDITION_LABELS = {
    "Commensal_Static": "Commensal Static",
    "Commensal_HOBIC": "Commensal HOBIC",
    "Dysbiotic_HOBIC": "Dysbiotic HOBIC",
    "Dysbiotic_Static": "Dysbiotic Static",
}
CONDITION_ORDER = ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_HOBIC", "Dysbiotic_Static"]

DT = 1e-4
N_STEPS = 2500
K_HILL = 0.05
N_HILL = 4.0


# ──────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────


def load_run(run_dir: Path) -> dict | None:
    """Load config, theta_MAP, samples, logL from a run directory."""
    try:
        with open(run_dir / "config.json") as f:
            cfg = json.load(f)
        with open(run_dir / "theta_MAP.json") as f:
            theta_d = json.load(f)
    except Exception:
        return None

    theta = np.array([theta_d[str(i)] for i in range(20)], dtype=np.float64)
    logL = np.load(run_dir / "logL.npy")
    samples = np.load(run_dir / "samples.npy")

    return {
        "dir": run_dir,
        "cond_key": f"{cfg['condition']}_{cfg['cultivation']}",
        "condition": cfg["condition"],
        "cultivation": cfg["cultivation"],
        "theta_MAP": theta,
        "samples": samples,
        "logL": logL,
        "logL_max": float(logL.max()),
        "n_particles": cfg.get("n_particles", len(samples)),
    }


def simulate_map(theta: np.ndarray, phi_init: np.ndarray, idx_sparse: np.ndarray) -> np.ndarray:
    """Run forward ODE and return predicted fractions at observation times."""
    phi_traj = simulate_0d(
        jnp.array(theta),
        n_steps=N_STEPS,
        dt=DT,
        phi_init=jnp.array(phi_init),
        K_hill=K_HILL,
        n_hill=N_HILL,
    )
    phi_pred = np.array(phi_traj[idx_sparse, :])
    phi_pred = np.clip(phi_pred, 1e-10, 1.0)
    phi_pred = phi_pred / phi_pred.sum(axis=1, keepdims=True)
    return phi_pred


def compute_rmse(data: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (total_rmse, per_species_rmse)."""
    residuals = data - pred
    per_species = np.sqrt(np.mean(residuals**2, axis=0))
    total = float(np.sqrt(np.mean(residuals**2)))
    return total, per_species


def get_best_runs(runs_dir: Path) -> dict[str, dict]:
    """Return the best run (highest logL) per condition."""
    best: dict[str, dict] = {}
    for d in sorted(runs_dir.glob("jax_ode_nuts_*")):
        if not d.is_dir():
            continue
        run = load_run(d)
        if run is None:
            continue
        key = run["cond_key"]
        if key not in best or run["logL_max"] > best[key]["logL_max"]:
            best[key] = run
    return best


def get_data_for_condition(cond: str, cult: str) -> tuple:
    """Returns (data, t_days, phi_init, idx_sparse)."""
    data, t_days, _, phi_init_exp, _ = load_experimental_data(
        DATA_DIR, cond, cult, 1, normalize=True
    )
    phi_init = phi_init_exp / phi_init_exp.sum()
    phi_init = np.clip(phi_init, 0.01, 0.99)
    _, idx_sparse = convert_days_to_model_time(t_days, DT, N_STEPS, day_scale=None)
    idx_sparse = np.clip(idx_sparse, 0, N_STEPS)
    return data, t_days, phi_init, idx_sparse


# ──────────────────────────────────────────────
# Figure S-A: 4-condition overview (5-species overlay)
# ──────────────────────────────────────────────


def fig_sa_overview(best_runs: dict, out_dir: Path) -> None:
    """One panel per condition, all 5 species overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, cond_key in zip(axes_flat, CONDITION_ORDER):
        run = best_runs.get(cond_key)
        if run is None:
            ax.set_visible(False)
            continue

        cond, cult = run["condition"], run["cultivation"]
        data, t_days, phi_init, idx_sparse = get_data_for_condition(cond, cult)
        phi_pred = simulate_map(run["theta_MAP"], phi_init, idx_sparse)
        rmse_total, rmse_per_sp = compute_rmse(data, phi_pred)

        for i, (name, color) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
            ax.plot(t_days, phi_pred[:, i], "-", color=color, lw=2, label=name if i == 0 else None)
            ax.scatter(t_days, data[:, i], color=color, s=40, zorder=5)

        ax.set_title(
            f"{CONDITION_LABELS.get(cond_key, cond_key)}\nRMSE = {rmse_total:.4f}",
            fontsize=11,
        )
        ax.set_xlabel("Day", fontsize=10)
        ax.set_ylabel("Species fraction", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, t_days.max() * 1.05)

    legend_handles = [
        Line2D([0], [0], color=c, lw=2, label=n) for n, c in zip(SPECIES_NAMES, SPECIES_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("MAP Fit — All 4 Conditions (JAX ODE TMCMC)", fontsize=13, fontweight="bold")

    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"figS_A_overview.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[S-A] Saved overview figure.")


# ──────────────────────────────────────────────
# Figure S-B: 4×5 per-species grid
# ──────────────────────────────────────────────


def fig_sb_per_species(best_runs: dict, out_dir: Path) -> None:
    """4 conditions × 5 species = 20 panels."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.45, wspace=0.3)

    rmse_table = {}

    for row, cond_key in enumerate(CONDITION_ORDER):
        run = best_runs.get(cond_key)
        if run is None:
            continue

        cond, cult = run["condition"], run["cultivation"]
        data, t_days, phi_init, idx_sparse = get_data_for_condition(cond, cult)
        phi_pred = simulate_map(run["theta_MAP"], phi_init, idx_sparse)
        rmse_total, rmse_per_sp = compute_rmse(data, phi_pred)
        rmse_table[cond_key] = {"total": rmse_total, "per_species": rmse_per_sp.tolist()}

        for col, (name, color) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t_days, phi_pred[:, col], "-", color=color, lw=2)
            ax.scatter(t_days, data[:, col], color=color, s=35, zorder=5, edgecolors="k", lw=0.5)

            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, t_days.max() * 1.05)

            if row == 0:
                ax.set_title(name, fontsize=11, fontweight="bold", color=color)
            if col == 0:
                short = CONDITION_LABELS.get(cond_key, cond_key)
                ax.set_ylabel(short.replace(" ", "\n"), fontsize=8.5)
            if row == 3:
                ax.set_xlabel("Day", fontsize=9)

            ax.text(
                0.97,
                0.97,
                f"RMSE\n{rmse_per_sp[col]:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    fig.suptitle(
        "Per-Species MAP Fit — All 4 Conditions (JAX ODE TMCMC)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"figS_B_per_species.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "rmse_table.json", "w") as f:
        json.dump(rmse_table, f, indent=2)
    print(f"[S-B] Saved per-species grid. RMSE table → {out_dir / 'rmse_table.json'}")
    return rmse_table


# ──────────────────────────────────────────────
# Figure S-C: Posterior predictive bands
# ──────────────────────────────────────────────


def fig_sc_posterior_band(best_runs: dict, out_dir: Path, n_draws: int = 100) -> None:
    """Posterior predictive 95% CI bands from posterior samples."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, cond_key in zip(axes_flat, CONDITION_ORDER):
        run = best_runs.get(cond_key)
        if run is None:
            ax.set_visible(False)
            continue

        cond, cult = run["condition"], run["cultivation"]
        data, t_days, phi_init, idx_sparse = get_data_for_condition(cond, cult)

        samples = run["samples"]
        logL = run["logL"]
        # Draw top-n_draws particles weighted by logL
        n_avail = min(n_draws, len(samples))
        top_idx = np.argsort(logL)[-n_avail:]
        draw_idx = top_idx

        preds = []
        for idx in draw_idx:
            try:
                pred = simulate_map(samples[idx], phi_init, idx_sparse)
                preds.append(pred)
            except Exception:
                continue

        if len(preds) == 0:
            ax.set_title(f"{CONDITION_LABELS.get(cond_key, cond_key)}\n(no valid draws)")
            continue

        preds = np.array(preds)  # (n_draws, n_obs, 5)
        p025 = np.percentile(preds, 2.5, axis=0)
        p975 = np.percentile(preds, 97.5, axis=0)
        p500 = np.percentile(preds, 50.0, axis=0)

        phi_map = simulate_map(run["theta_MAP"], phi_init, idx_sparse)

        for i, (name, color) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
            ax.fill_between(t_days, p025[:, i], p975[:, i], color=color, alpha=0.18)
            ax.plot(t_days, p500[:, i], "--", color=color, lw=1.2, alpha=0.7)
            ax.plot(t_days, phi_map[:, i], "-", color=color, lw=2.0)
            ax.scatter(t_days, data[:, i], color=color, s=40, zorder=6, edgecolors="k", lw=0.4)

        rmse_total, _ = compute_rmse(data, phi_map)
        ax.set_title(
            f"{CONDITION_LABELS.get(cond_key, cond_key)}\nMAP RMSE = {rmse_total:.4f}",
            fontsize=11,
        )
        ax.set_xlabel("Day", fontsize=10)
        ax.set_ylabel("Species fraction", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, t_days.max() * 1.05)

    legend_handles = [
        Line2D([0], [0], color=c, lw=2, label=n) for n, c in zip(SPECIES_NAMES, SPECIES_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Posterior Predictive Bands (top-{n_draws} particles, 95% CI)",
        fontsize=13,
        fontweight="bold",
    )

    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"figS_C_posterior_band.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[S-C] Saved posterior band figure ({n_draws} draws).")


# ──────────────────────────────────────────────
# Figure S-D: RMSE heatmap table
# ──────────────────────────────────────────────


def fig_sd_rmse_heatmap(best_runs: dict, out_dir: Path) -> None:
    """Heatmap of per-species RMSE across all conditions."""
    conds_present = [c for c in CONDITION_ORDER if c in best_runs]
    rmse_matrix = np.zeros((len(conds_present), 5))

    for row, cond_key in enumerate(conds_present):
        run = best_runs[cond_key]
        cond, cult = run["condition"], run["cultivation"]
        data, t_days, phi_init, idx_sparse = get_data_for_condition(cond, cult)
        phi_pred = simulate_map(run["theta_MAP"], phi_init, idx_sparse)
        _, per_sp = compute_rmse(data, phi_pred)
        rmse_matrix[row] = per_sp

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = ax.imshow(rmse_matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=0.20)
    fig.colorbar(im, ax=ax, label="RMSE", shrink=0.8)

    ax.set_xticks(range(5))
    ax.set_xticklabels(SPECIES_NAMES, fontsize=11)
    ax.set_yticks(range(len(conds_present)))
    ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in conds_present], fontsize=10)

    for row in range(len(conds_present)):
        for col in range(5):
            val = rmse_matrix[row, col]
            txt_color = "white" if val > 0.12 else "black"
            ax.text(
                col,
                row,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
                fontweight="bold",
            )

    ax.set_title("Per-Species RMSE (MAP Estimate)", fontsize=12, fontweight="bold")

    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"figS_D_rmse_heatmap.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[S-D] Saved RMSE heatmap.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate supplementary figures from JAX TMCMC runs"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(SCRIPT_DIR / "_runs"),
        help="Directory containing jax_ode_nuts_* run directories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "_runs" / "supplementary_figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=100,
        help="Number of posterior draws for CI bands (S-C figure)",
    )
    parser.add_argument(
        "--condition-dirs",
        nargs="*",
        help="Explicit run dirs: CS=<path> CH=<path> DH=<path> DS=<path>",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning runs in {runs_dir}...")
    best_runs = get_best_runs(runs_dir)

    if not best_runs:
        print("No valid runs found. Check --runs-dir.")
        return

    print("Best runs found:")
    for cond_key in CONDITION_ORDER:
        run = best_runs.get(cond_key)
        if run:
            cond, cult = run["condition"], run["cultivation"]
            data, t_days, phi_init, idx_sparse = get_data_for_condition(cond, cult)
            phi_pred = simulate_map(run["theta_MAP"], phi_init, idx_sparse)
            rmse, _ = compute_rmse(data, phi_pred)
            print(
                f"  {cond_key:25s}  logL={run['logL_max']:8.2f}  RMSE={rmse:.4f}  {run['dir'].name}"
            )
        else:
            print(f"  {cond_key:25s}  (not found)")

    print(f"\nGenerating figures → {out_dir}")
    fig_sa_overview(best_runs, out_dir)
    rmse_table = fig_sb_per_species(best_runs, out_dir)
    fig_sc_posterior_band(best_runs, out_dir, n_draws=args.n_draws)
    fig_sd_rmse_heatmap(best_runs, out_dir)

    # Print RMSE summary
    print("\n=== RMSE Summary ===")
    print(f"{'Condition':25s}  {'Total':>7s}  Per-species (So, An, Vd, Fn, Pg)")
    for cond_key in CONDITION_ORDER:
        if cond_key in rmse_table:
            total = rmse_table[cond_key]["total"]
            per_sp = rmse_table[cond_key]["per_species"]
            sp_str = "  ".join(f"{v:.3f}" for v in per_sp)
            print(f"{cond_key:25s}  {total:.4f}   {sp_str}")
    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
