#!/usr/bin/env python3
"""
Regenerate publication-quality plots from existing TMCMC run results.

Usage:
    python regenerate_paper_plots.py _runs/CS_1000p_expIC_repSigma_*
    python regenerate_paper_plots.py _runs/CS_* _runs/DH_*  # multiple runs
    python regenerate_paper_plots.py --all  # all runs with samples.npy
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.helpers import compute_phibar, load_exp_boxplot
from visualization.plot_manager import PlotManager
from visualization.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    SPECIES_COLORS,
    SPECIES_NAMES,
    SPECIES_NAMES_SHORT,
    apply_paper_style,
    savefig_paper,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_run(run_dir: Path):
    """Load TMCMC run results from directory."""
    samples = np.load(run_dir / "samples.npy")
    logL = np.load(run_dir / "logL.npy")

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    theta_map_data = None
    theta_mean_data = None
    if (run_dir / "theta_MAP.json").exists():
        with open(run_dir / "theta_MAP.json") as f:
            theta_map_data = json.load(f)
    if (run_dir / "theta_mean.json").exists():
        with open(run_dir / "theta_mean.json") as f:
            theta_mean_data = json.load(f)

    fit_metrics = None
    if (run_dir / "fit_metrics.json").exists():
        with open(run_dir / "fit_metrics.json") as f:
            fit_metrics = json.load(f)

    return {
        "samples": samples,
        "logL": logL,
        "config": config,
        "theta_MAP": theta_map_data,
        "theta_mean": theta_mean_data,
        "fit_metrics": fit_metrics,
        "dir": run_dir,
    }


def regenerate_5panel(run_dir: Path):
    """Regenerate publication 5-panel plot for a single run."""
    import matplotlib.pyplot as plt

    run = load_run(run_dir)
    config = run["config"]
    condition = config["condition"]
    cultivation = config["cultivation"]
    name_tag = f"{condition}_{cultivation}"

    logger.info(f"Regenerating: {name_tag} from {run_dir.name}")

    # Check for saved posterior trajectories
    traj_file = run_dir / "posterior_trajectories.npz"
    if traj_file.exists():
        logger.info("  Using saved posterior trajectories")
        traj = np.load(traj_file)
        phibar_samples = traj["phibar_samples"]
        t_fit = traj["t_fit"]
        phibar_map = traj.get("phibar_map")
        phibar_mean = traj.get("phibar_mean")
        idx_sparse = traj["idx_sparse"]
        data = traj["data"]
    else:
        logger.info("  No saved trajectories — re-simulating from samples")
        # Re-simulate from samples
        from core.solver import BiofilmNewtonSolver5S

        phi_init = config.get("phi_init", [0.2, 0.2, 0.2, 0.2, 0.2])
        solver = BiofilmNewtonSolver5S(
            dt=config["dt"],
            maxtimestep=config["maxtimestep"],
            c_const=config["c_const"],
            alpha_const=config.get("alpha_const", 0.0),
            phi_init=phi_init,
            Kp1=config.get("Kp1", 0.0001),
        )
        meta = config.get("metadata", {})
        idx_sparse = np.array(meta.get("idx_sparse", []))

        # Load data
        data_file = run_dir / "data.npy"
        if data_file.exists():
            data = np.load(data_file)
        else:
            data = None

        active_species = [0, 1, 2, 3, 4]

        # MAP trajectory
        phibar_map = None
        if run["theta_MAP"] is not None:
            theta_full = np.array(run["theta_MAP"]["theta_full"])
            t_fit, x_fit = solver.solve(theta_full)
            phibar_map = compute_phibar(x_fit, active_species)

        # Mean trajectory
        phibar_mean = None
        if run["theta_mean"] is not None:
            theta_full = np.array(run["theta_mean"]["theta_full"])
            _, x_fit = solver.solve(theta_full)
            phibar_mean = compute_phibar(x_fit, active_species)

        # Posterior samples
        n_draw = min(50, run["samples"].shape[0])
        indices = np.random.choice(run["samples"].shape[0], n_draw, replace=False)
        phibar_list = []
        theta_base = np.array(run["theta_MAP"]["theta_full"]) if run["theta_MAP"] else np.zeros(20)
        for idx in indices:
            theta = theta_base.copy()
            theta[:20] = run["samples"][idx]
            _, x_sim = solver.solve(theta)
            phibar_list.append(compute_phibar(x_sim, active_species))
        phibar_samples = np.array(phibar_list)

    # Time in days
    meta = config.get("metadata", {})
    t_days = np.array(meta.get("days", []))

    # Load boxplot
    try:
        exp_boxplot = load_exp_boxplot(condition, cultivation)
    except Exception:
        exp_boxplot = None

    # Phi init
    phi_init_arr = None
    if config.get("use_exp_init") and "phi_init_exp" in meta:
        phi_init_arr = np.array(meta["phi_init_exp"])
    elif config.get("phi_init"):
        phi_init_arr = np.array(config["phi_init"])

    # Generate plot
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_mgr = PlotManager(str(figures_dir))

    plot_mgr.plot_5panel_paper(
        t_fit,
        phibar_samples,
        [0, 1, 2, 3, 4],
        name_tag,
        data,
        idx_sparse.astype(int),
        t_days=t_days,
        exp_boxplot=exp_boxplot,
        phibar_map=phibar_map,
        phibar_mean=phibar_mean,
        phi_init=phi_init_arr,
        condition_label=f"{condition} / {cultivation}",
    )
    plt.close("all")
    logger.info(f"  Done: {figures_dir}/5panel_paper_{name_tag}.png/.pdf")


def main():
    parser = argparse.ArgumentParser(description="Regenerate publication plots from TMCMC results")
    parser.add_argument("dirs", nargs="*", help="Run directories")
    parser.add_argument("--all", action="store_true", help="Process all runs with samples.npy")
    args = parser.parse_args()

    if args.all:
        runs_base = Path("_runs")
        dirs = sorted(d for d in runs_base.iterdir() if (d / "samples.npy").exists())
    else:
        dirs = [Path(d) for d in args.dirs]

    if not dirs:
        logger.error("No run directories specified. Use --all or provide paths.")
        sys.exit(1)

    logger.info(f"Processing {len(dirs)} run(s)")
    for d in dirs:
        try:
            regenerate_5panel(d)
        except Exception as e:
            logger.error(f"Failed for {d}: {e}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
