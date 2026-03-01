#!/usr/bin/env python3
"""
Diagnostic: RMSE gap analysis between old (20-free) and new (locked) runs.

Quantifies per-species, per-timepoint residuals and identifies improvement opportunities.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).parent.parent / "_runs"
SPECIES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]

# Experimental data (normalized fractions) — from processed_data/target_data_*_normalized.csv
EXP_DATA = {
    "CS": {
        "days": [1, 3, 6, 10, 15, 21],
        "fracs": np.array(
            [
                [0.176, 0.005, 0.810, 0.005, 0.005],
                [0.707, 0.081, 0.202, 0.005, 0.005],
                [0.743, 0.050, 0.198, 0.005, 0.005],
                [0.660, 0.094, 0.236, 0.005, 0.005],
                [0.365, 0.240, 0.385, 0.005, 0.005],
                [0.347, 0.347, 0.297, 0.005, 0.005],
            ]
        ),
    },
    "CH": {
        "days": [1, 3, 6, 10, 15, 21],
        "fracs": np.array(
            [
                [0.746, 0.050, 0.100, 0.100, 0.005],
                [0.892, 0.005, 0.094, 0.005, 0.005],
                [0.739, 0.005, 0.246, 0.005, 0.005],
                [0.561, 0.122, 0.306, 0.005, 0.005],
                [0.365, 0.292, 0.333, 0.005, 0.005],
                [0.469, 0.260, 0.260, 0.005, 0.005],
            ]
        ),
    },
    "DS": {
        "days": [1, 3, 6, 10, 15, 21],
        "fracs": np.array(
            [
                [0.153, 0.082, 0.612, 0.051, 0.102],
                [0.024, 0.060, 0.747, 0.072, 0.096],
                [0.019, 0.049, 0.602, 0.117, 0.214],
                [0.010, 0.010, 0.577, 0.135, 0.269],
                [0.011, 0.055, 0.495, 0.220, 0.220],
                [0.010, 0.133, 0.362, 0.210, 0.286],
            ]
        ),
    },
    "DH": {
        "days": [1, 3, 6, 10, 15, 21],
        "fracs": np.array(
            [
                [0.040, 0.010, 0.940, 0.005, 0.005],
                [0.029, 0.010, 0.951, 0.005, 0.005],
                [0.067, 0.114, 0.810, 0.005, 0.005],
                [0.109, 0.136, 0.679, 0.072, 0.005],
                [0.021, 0.372, 0.372, 0.213, 0.021],
                [0.009, 0.321, 0.229, 0.275, 0.165],
            ]
        ),
    },
}


def load_fit_metrics(run_dir):
    """Load fit_metrics.json from a run directory."""
    p = run_dir / "fit_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_theta_map(run_dir):
    """Load theta_MAP.json."""
    p = run_dir / "theta_MAP.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return np.array(d.get("theta_full", d.get("theta_sub", [])))
    return None


def load_config(run_dir):
    """Load config.json."""
    p = run_dir / "config.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 80)
    logger.info("RMSE GAP DIAGNOSTIC: Old (20-free) vs New (biologically-locked) runs")
    logger.info("=" * 80)

    # ── Collect metrics ──────────────────────────────────────────────
    runs_info = {
        "CS_old": ("commensal_static", "CS (20-free)"),
        "CS_new": ("commensal_static_posterior", "CS (9-free)"),
        "CH_new": ("commensal_hobic_posterior", "CH (13-free)"),
        "DS_old": ("dysbiotic_static", "DS (20-free)"),
        "DS_new": ("dysbiotic_static_posterior", "DS (15-free)"),
        "DH_old": ("dh_baseline", "DH (20-free)"),
    }

    logger.info("\n── Per-Species RMSE Comparison ──")
    for key, (dirname, label) in runs_info.items():
        run_dir = RUNS / dirname
        metrics = load_fit_metrics(run_dir)
        theta = load_theta_map(run_dir)
        config = load_config(run_dir)

        if metrics is None:
            n_free = "?"
            if theta is not None:
                n_free = len([v for v in theta if abs(v) > 1e-10])
            logger.info(
                f"\n  {label} ({dirname}): NO fit_metrics.json [theta has {n_free} non-zero]"
            )
            continue

        m = metrics["MAP"]
        rmse_sp = m["rmse_per_species"]
        n_active = len(config.get("metadata", {}).get("idx_sparse", [])) if config else "?"
        use_init = config.get("use_exp_init", False) if config else "?"
        n_part = config.get("n_particles", "?") if config else "?"

        logger.info(f"\n  {label} ({dirname})")
        logger.info(f"    Particles={n_part}, use_exp_init={use_init}")
        logger.info(f"    Total RMSE = {m['rmse_total']:.4f}")
        logger.info(f"    {'Species':<16} {'RMSE':>8} {'MAE':>8}")
        logger.info(f"    {'─'*32}")
        for i, sp in enumerate(SPECIES):
            flag = " <<<" if rmse_sp[i] > 0.10 else ""
            logger.info(f"    {sp:<16} {rmse_sp[i]:8.4f} {m['mae_per_species'][i]:8.4f}{flag}")

    # ── Experimental data dynamics analysis ──────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENTAL DATA DYNAMICS (normalized fractions)")
    logger.info("=" * 80)

    for cond, info in EXP_DATA.items():
        fracs = info["fracs"]
        days = info["days"]
        logger.info(f"\n  ── {cond} ──")
        logger.info(f"    {'Day':>4}  " + "  ".join(f"{s:>6}" for s in SPECIES_SHORT))
        for i, d in enumerate(days):
            vals = "  ".join(f"{fracs[i,j]:6.3f}" for j in range(5))
            logger.info(f"    {d:>4}  {vals}")

        # Key dynamics metrics
        vd_ratio = fracs[0, 2] / max(fracs[-1, 2], 0.001)
        so_peak = np.max(fracs[:, 0])
        so_final = fracs[-1, 0]
        pg_change = fracs[-1, 4] - fracs[0, 4]

        logger.info(f"    V.dispar Day1/Day21 ratio = {vd_ratio:.1f}×")
        logger.info(f"    S.oralis peak = {so_peak:.3f}, final = {so_final:.3f}")
        logger.info(f"    P.gingivalis Δ = {pg_change:+.3f} (Day 1→21)")

    # ── Identify improvement opportunities ───────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT OPPORTUNITIES")
    logger.info("=" * 80)

    improvements = []

    # Check if use_exp_init is being used
    for key in ["CS_new", "DS_new", "CH_new"]:
        dirname, label = runs_info[key]
        config = load_config(RUNS / dirname)
        if config and not config.get("use_exp_init", False):
            improvements.append(
                f"  1. [HIGH] {label}: use_exp_init=False → uniform φ=0.02 start.\n"
                f"     Experimental init: {config.get('metadata', {}).get('phi_init_exp', [])}\n"
                f"     Using --use-exp-init should reduce Day 1 residual significantly."
            )

    # Check particle count
    for key in ["CS_new", "DS_new", "CH_new"]:
        dirname, label = runs_info[key]
        config = load_config(RUNS / dirname)
        if config and config.get("n_particles", 0) < 300:
            improvements.append(
                f"  2. [MED]  {label}: n_particles={config.get('n_particles')} is low.\n"
                f"     Recommend 500+ for better posterior exploration."
            )

    # V. dispar structural issue
    improvements.append(
        "  3. [STRUCTURAL] V. dispar RMSE > 0.20 in CS/DS — model limitation.\n"
        "     Hamilton ODE has no nutrient depletion mechanism.\n"
        "     V.d is a lactate specialist (peaks Day 1-3, then declines).\n"
        "     Options: (a) species-specific σ weighting, (b) add decay term,\n"
        "     (c) accept as model inadequacy and document in paper."
    )

    for imp in improvements:
        logger.info(imp)

    # ── Summary figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RMSE Gap Diagnostic: Per-Species Breakdown", fontsize=14)

    plot_data = []
    for key in ["CS_new", "CH_new", "DS_new"]:
        dirname, label = runs_info[key]
        metrics = load_fit_metrics(RUNS / dirname)
        if metrics:
            plot_data.append((label, metrics["MAP"]["rmse_per_species"]))

    # Also add old runs if they have metrics
    for key in ["CS_old", "DS_old"]:
        dirname, label = runs_info[key]
        # These don't have fit_metrics, skip
        pass

    colors = ["#2196F3", "#4CAF50", "#FFC107", "#9C27B0", "#F44336"]

    for idx, (label, rmse_sp) in enumerate(plot_data):
        ax = axes.flat[idx]
        bars = ax.bar(SPECIES_SHORT, rmse_sp, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.set_ylim(0, 0.25)
        ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="target 0.05")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, rmse_sp):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 4th panel: experimental data dynamics (V. dispar across conditions)
    ax = axes.flat[3]
    for cond, info in EXP_DATA.items():
        ax.plot(info["days"], info["fracs"][:, 2], "o-", label=f"Vd ({cond})", linewidth=1.5)
    ax.set_title("V. dispar dynamics (data)", fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RUNS / "rmse_gap_diagnostic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"\n  Figure saved: {out_path}")
    plt.close()

    # ── Concrete next steps ──────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED NEXT STEPS (priority order)")
    logger.info("=" * 80)
    logger.info(
        """
  1. Re-run with --use-exp-init + more particles:
     python estimate_reduced_nishioka.py \\
       --condition Commensal --cultivation Static \\
       --n-particles 500 --n-stages 15 \\
       --use-exp-init --start-from-day 1

  2. Add species-specific σ_obs (reduce V.dispar weight):
     σ_Vd = 2× σ_global (acknowledges model inadequacy)

  3. For paper: present old 20-free RMSE as "unconstrained baseline",
     posterior RMSE as "biologically constrained", and discuss trade-off.

  4. Consider start-from-day 3 with exp-init:
     → Removes Day 1 transient, focuses on Day 3-21 dynamics
     → May improve V. dispar fit (avoids need to model Day 1→3 crash)
"""
    )


if __name__ == "__main__":
    main()
