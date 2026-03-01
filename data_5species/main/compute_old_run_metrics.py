#!/usr/bin/env python3
"""
Compute fit_metrics for old 20-free runs that don't have them.
Uses the Hamilton ODE solver to simulate with saved theta_MAP.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DATA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

from improved_5species_jit import BiofilmNewtonSolver5S
from visualization.helpers import compute_phibar

RUNS = DATA_ROOT / "_runs"
SPECIES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]

# Conditions to analyze
RUNS_TO_ANALYZE = [
    ("commensal_static", "CS old (20-free)", "Commensal", "Static"),
    ("commensal_static_posterior", "CS new (9-free)", "Commensal", "Static"),
    ("dysbiotic_static", "DS old (20-free)", "Dysbiotic", "Static"),
    ("dysbiotic_static_posterior", "DS new (15-free)", "Dysbiotic", "Static"),
    ("commensal_hobic", "CH old (20-free)", "Commensal", "HOBIC"),
    ("commensal_hobic_posterior", "CH new (13-free)", "Commensal", "HOBIC"),
    ("dh_baseline", "DH old (20-free)", "Dysbiotic", "HOBIC"),
]


def get_solver_kwargs(condition, cultivation):
    """Build solver kwargs matching the runner config."""
    return dict(
        dt=0.0001,
        maxtimestep=2500,
        c_const=25.0,
        Kp1=0.0001,
        K_hill=0.05,
        n_hill=4.0,
        alpha_const=0.0,
        phi_init=0.02,
    )


def load_data_for_condition(condition, cultivation):
    """Load experimental target data (normalized fractions)."""
    cond_name = f"{condition}_{cultivation}"
    csv_path = DATA_ROOT / "processed_data" / f"target_data_{cond_name}_normalized.csv"
    if not csv_path.exists():
        logger.warning("%s not found", csv_path)
        return None, None, None

    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_obs = len(rows)
    data = np.zeros((n_obs, 5))
    days = []
    for i, row in enumerate(rows):
        days.append(int(row["Day"]))
        data[i, 0] = float(row["Frac_S.oralis"])
        data[i, 1] = float(row["Frac_A.naeslundii"])
        data[i, 2] = float(row["Frac_Veillonella"])
        data[i, 3] = float(row["Frac_F.nucleatum"])
        data[i, 4] = float(row["Frac_P.gingivalis"])

    # Compute idx_sparse from days (same mapping as config)
    # Day → model timestep: idx = int(day * 84 / 21 * 2500 / 2500 * ...)
    # Actually, use the config's idx_sparse if available
    idx_sparse = np.array([113, 339, 679, 1131, 1696, 2375])
    return data, days, idx_sparse


def simulate_and_compute_rmse(run_dir, label, condition, cultivation):
    """Simulate with theta_MAP and compute RMSE."""
    theta_map_path = run_dir / "theta_MAP.json"
    if not theta_map_path.exists():
        return None

    with open(theta_map_path) as f:
        theta_data = json.load(f)
    theta_full = np.array(theta_data.get("theta_full", theta_data.get("theta_sub", [])))

    if len(theta_full) != 20:
        logger.info("  {label}: theta has {len(theta_full)} values, expected 20")
        return None

    # Load experimental data
    data, days, idx_sparse = load_data_for_condition(condition, cultivation)
    if data is None:
        return None

    # Setup solver
    active_species = list(range(5))
    solver_kwargs = get_solver_kwargs(condition, cultivation)

    try:
        solver = BiofilmNewtonSolver5S(
            **solver_kwargs,
            active_species=active_species,
            use_numba=True,
        )
        t_arr, x0 = solver.run_deterministic(theta_full)
    except Exception as e:
        logger.warning("  %s: Solver failed: %s", label, e)
        return None

    # Compute φ̄ = φ × ψ at observation times
    phibar = compute_phibar(x0, active_species)  # (n_time, 5)
    phibar_obs = phibar[idx_sparse]  # (n_obs, 5)

    # Normalize to fractions (sum=1 per timepoint), same as data
    row_sums = phibar_obs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)
    phibar_norm = phibar_obs / row_sums

    # Compute RMSE per species
    residuals = phibar_norm - data
    rmse_per_sp = np.sqrt(np.mean(residuals**2, axis=0))
    mae_per_sp = np.mean(np.abs(residuals), axis=0)
    rmse_total = np.sqrt(np.mean(residuals**2))
    mae_total = np.mean(np.abs(residuals))

    return {
        "rmse_total": float(rmse_total),
        "rmse_per_species": rmse_per_sp.tolist(),
        "mae_total": float(mae_total),
        "mae_per_species": mae_per_sp.tolist(),
        "residuals": residuals,
        "phibar_norm": phibar_norm,
        "data": data,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 80)
    logger.info("FIT METRICS RECOMPUTATION: Old vs New runs")
    logger.info("=" * 80)

    results = {}
    for dirname, label, condition, cultivation in RUNS_TO_ANALYZE:
        run_dir = RUNS / dirname
        if not run_dir.exists():
            logger.info("  %s: directory not found", label)
            continue

        existing = run_dir / "fit_metrics.json"
        if existing.exists():
            with open(existing) as f:
                fm = json.load(f)
            m = fm["MAP"]
            logger.info("  %s (from existing fit_metrics.json)", label)
        else:
            result = simulate_and_compute_rmse(run_dir, label, condition, cultivation)
            if result is None:
                continue
            m = result
            logger.info("  %s (recomputed)", label)

        logger.info("    Total RMSE = %.4f", m["rmse_total"])
        logger.info("    %-16s %8s", "Species", "RMSE")
        logger.info("    %s", "─" * 24)
        for i, sp in enumerate(SPECIES):
            flag = " <<<" if m["rmse_per_species"][i] > 0.10 else ""
            logger.info("    %-16s %8.4f%s", sp, m["rmse_per_species"][i], flag)
        results[label] = m

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY: RMSE COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info("  %-25s %8s %8s %8s %8s %8s %8s", "Run", "Total", "So", "An", "Vd", "Fn", "Pg")
    logger.info("  %s", "─" * 73)
    for label, m in results.items():
        r = m["rmse_per_species"]
        logger.info(
            "  %-25s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f",
            label,
            m["rmse_total"],
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
        )


if __name__ == "__main__":
    main()
