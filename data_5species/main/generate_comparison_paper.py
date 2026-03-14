#!/usr/bin/env python3
"""
Generate publication-quality cross-condition comparison figures.

Produces:
  Fig A: Parameter heatmap (4 conditions × 20 params, MAP + HDI)
  Fig B: RMSE comparison bar chart (per-species × per-condition)
  Fig C: Posterior overlap matrix (cross-condition parameter similarity)

Usage:
    python generate_comparison_paper.py
    python generate_comparison_paper.py --output-dir _runs/comparison_paper
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    SPECIES_COLORS,
    SPECIES_NAMES,
    SPECIES_NAMES_SHORT,
    add_panel_labels,
    apply_paper_style,
    savefig_paper,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = [
    "$a_{11}$",
    "$a_{12}$",
    "$a_{22}$",
    "$b_1$",
    "$b_2$",
    "$a_{33}$",
    "$a_{34}$",
    "$a_{44}$",
    "$b_3$",
    "$b_4$",
    "$a_{13}$",
    "$a_{14}$",
    "$a_{23}$",
    "$a_{24}$",
    "$a_{55}$",
    "$b_5$",
    "$a_{15}$",
    "$a_{25}$",
    "$a_{35}$",
    "$a_{45}$",
]


def find_latest_runs(runs_base: Path):
    """Find latest run for each condition."""
    found = {}
    for prefix in ["CS", "CH", "DS", "DH"]:
        candidates = sorted(runs_base.glob(f"{prefix}_*p_expIC_repSigma_*"))
        if not candidates:
            candidates = sorted(runs_base.glob(f"{prefix}_*p_expIC_*"))
        if not candidates:
            candidates = sorted(runs_base.glob(f"{prefix}_*"))
        for c in reversed(candidates):
            if (c / "samples.npy").exists():
                found[prefix] = c
                break
    return found


def load_run(run_dir: Path):
    """Load essential data from a run directory."""
    result = {"dir": run_dir}
    result["samples"] = np.load(run_dir / "samples.npy")

    for fname in [
        "config.json",
        "theta_MAP.json",
        "theta_mean.json",
        "fit_metrics.json",
        "mcmc_diagnostics.json",
        "credible_intervals.json",
        "parameter_summary.csv",
    ]:
        fpath = run_dir / fname
        if fpath.exists():
            if fname.endswith(".json"):
                with open(fpath) as f:
                    result[fname.replace(".json", "")] = json.load(f)
            elif fname.endswith(".csv"):
                import pandas as pd

                result[fname.replace(".csv", "")] = pd.read_csv(fpath)
    return result


def fig_param_heatmap(runs: dict, output_dir: Path):
    """Fig A: Normalized parameter heatmap across 4 conditions."""
    apply_paper_style()

    n_params = 20
    conditions = ["CS", "CH", "DS", "DH"]
    available = [c for c in conditions if c in runs]

    # Collect MAP values
    map_values = np.full((len(available), n_params), np.nan)
    for i, cond in enumerate(available):
        run = runs[cond]
        if "theta_MAP" in run and "theta_full" in run["theta_MAP"]:
            vals = run["theta_MAP"]["theta_full"]
            map_values[i, : len(vals)] = vals[:n_params]

    # Normalize per parameter (min-max across conditions)
    vmin = np.nanmin(map_values, axis=0)
    vmax = np.nanmax(map_values, axis=0)
    denom = vmax - vmin
    denom[denom == 0] = 1.0
    normalized = (map_values - vmin) / denom

    fig, ax = plt.subplots(figsize=(7.5, 2.5))
    im = ax.imshow(normalized, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in available])
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=7)

    # Annotate cells with actual values
    for i in range(len(available)):
        for j in range(n_params):
            val = map_values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if normalized[i, j] > 0.6 else "black",
                )

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Normalized MAP value", fontsize=8)
    ax.set_title("MAP Parameter Estimates Across Conditions", fontsize=10)

    savefig_paper(fig, output_dir / "fig_param_heatmap")
    plt.close(fig)
    logger.info("Saved fig_param_heatmap")


def fig_rmse_comparison(runs: dict, output_dir: Path):
    """Fig B: Per-species RMSE comparison across conditions."""
    apply_paper_style()

    conditions = ["CS", "CH", "DS", "DH"]
    available = [c for c in conditions if c in runs and "fit_metrics" in runs[c]]

    if not available:
        logger.warning("No fit_metrics available for RMSE comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # Panel (a): Per-species RMSE grouped by condition
    ax = axes[0]
    n_sp = 5
    n_cond = len(available)
    bar_width = 0.8 / n_cond
    x = np.arange(n_sp)

    for ci, cond in enumerate(available):
        fm = runs[cond]["fit_metrics"]
        # Use Mean estimate RMSE (more representative of posterior)
        rmse = fm.get("Mean", fm.get("MAP", {})).get("rmse_per_species", [0] * n_sp)
        offset = (ci - (n_cond - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            rmse,
            bar_width * 0.9,
            color=CONDITION_COLORS.get(cond, f"C{ci}"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=CONDITION_LABELS.get(cond, cond),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES_NAMES_SHORT)
    ax.set_ylabel("RMSE")
    ax.set_title("Per-species fit quality", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    # Panel (b): Total RMSE + MAE comparison
    ax2 = axes[1]
    metrics = {"RMSE": [], "MAE": []}
    for cond in available:
        fm = runs[cond]["fit_metrics"]
        est = fm.get("Mean", fm.get("MAP", {}))
        metrics["RMSE"].append(est.get("rmse_total", 0))
        metrics["MAE"].append(est.get("mae_total", 0))

    x2 = np.arange(len(available))
    bar_w = 0.35
    ax2.bar(
        x2 - bar_w / 2,
        metrics["RMSE"],
        bar_w,
        label="RMSE",
        color=[CONDITION_COLORS.get(c, "gray") for c in available],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )
    ax2.bar(
        x2 + bar_w / 2,
        metrics["MAE"],
        bar_w,
        label="MAE",
        color=[CONDITION_COLORS.get(c, "gray") for c in available],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.4,
        hatch="//",
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([CONDITION_LABELS.get(c, c) for c in available], rotation=15, ha="right")
    ax2.set_ylabel("Error")
    ax2.set_title("Overall fit quality (Mean estimate)", fontsize=9)
    ax2.legend(fontsize=7)

    add_panel_labels(axes)
    savefig_paper(fig, output_dir / "fig_rmse_comparison")
    plt.close(fig)
    logger.info("Saved fig_rmse_comparison")


def fig_convergence_summary(runs: dict, output_dir: Path):
    """Fig C: Convergence diagnostics summary (R-hat, ESS, beta schedule)."""
    apply_paper_style()

    conditions = ["CS", "CH", "DS", "DH"]
    available = [c for c in conditions if c in runs]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    # (a) R-hat per parameter
    ax = axes[0]
    for ci, cond in enumerate(available):
        run = runs[cond]
        if "mcmc_diagnostics" in run:
            rhat = run["mcmc_diagnostics"]["rhat"]
            ax.plot(
                range(len(rhat)),
                rhat,
                "o-",
                markersize=3,
                linewidth=0.8,
                color=CONDITION_COLORS.get(cond, f"C{ci}"),
                label=CONDITION_LABELS.get(cond, cond),
            )
    ax.axhline(1.1, color="red", linestyle="--", linewidth=0.6, alpha=0.5, label=r"$\hat{R}$ = 1.1")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("Parameter index")
    ax.set_ylabel(r"$\hat{R}$")
    ax.set_title("Gelman-Rubin", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # (b) ESS per parameter
    ax2 = axes[1]
    for ci, cond in enumerate(available):
        run = runs[cond]
        if "mcmc_diagnostics" in run:
            ess = run["mcmc_diagnostics"]["ess"]
            ax2.plot(
                range(len(ess)),
                ess,
                "s-",
                markersize=3,
                linewidth=0.8,
                color=CONDITION_COLORS.get(cond, f"C{ci}"),
                label=CONDITION_LABELS.get(cond, cond),
            )
    ax2.axhline(100, color="red", linestyle="--", linewidth=0.6, alpha=0.5, label="ESS = 100")
    ax2.set_xlabel("Parameter index")
    ax2.set_ylabel("ESS")
    ax2.set_title("Effective Sample Size", fontsize=9)
    ax2.legend(fontsize=6, ncol=2)

    # (c) Beta schedule
    ax3 = axes[2]
    for ci, cond in enumerate(available):
        run = runs[cond]
        diag_dir = run["dir"] / "tmcmc_diagnostics"
        if (diag_dir / "beta_schedule.json").exists():
            with open(diag_dir / "beta_schedule.json") as f:
                beta_data = json.load(f)
            for chain_key, betas in beta_data.items():
                ax3.plot(
                    range(len(betas)),
                    betas,
                    "-",
                    linewidth=0.8,
                    color=CONDITION_COLORS.get(cond, f"C{ci}"),
                    label=CONDITION_LABELS.get(cond, cond) if chain_key == "chain_0" else None,
                )
    ax3.set_xlabel("Stage")
    ax3.set_ylabel(r"$\beta$")
    ax3.set_title("Tempering schedule", fontsize=9)
    ax3.legend(fontsize=6, ncol=2)
    ax3.set_ylim(-0.05, 1.05)

    add_panel_labels(axes)
    savefig_paper(fig, output_dir / "fig_convergence_summary")
    plt.close(fig)
    logger.info("Saved fig_convergence_summary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="_runs/comparison_paper")
    parser.add_argument("--runs-base", default="_runs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_base = Path(args.runs_base)
    found = find_latest_runs(runs_base)
    logger.info(f"Found {len(found)}/4 conditions: {list(found.keys())}")
    for k, v in found.items():
        logger.info(f"  {k}: {v.name}")

    if not found:
        logger.error("No runs found!")
        sys.exit(1)

    runs = {k: load_run(v) for k, v in found.items()}

    fig_param_heatmap(runs, output_dir)
    fig_rmse_comparison(runs, output_dir)
    fig_convergence_summary(runs, output_dir)

    logger.info(f"All comparison figures saved to {output_dir}")


if __name__ == "__main__":
    main()
