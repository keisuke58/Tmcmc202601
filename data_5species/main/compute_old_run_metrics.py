#!/usr/bin/env python3
"""
Compute fit_metrics for old 20-free runs that don't have them.
Uses the Hamilton ODE solver to simulate with saved theta_MAP.
"""
import sys
import json
import numpy as np
from pathlib import Path

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
        print(f"  WARNING: {csv_path} not found")
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
        print(f"  {label}: theta has {len(theta_full)} values, expected 20")
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
        print(f"  {label}: Solver failed: {e}")
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
    print("=" * 80)
    print("FIT METRICS RECOMPUTATION: Old vs New runs")
    print("=" * 80)

    results = {}
    for dirname, label, condition, cultivation in RUNS_TO_ANALYZE:
        run_dir = RUNS / dirname
        if not run_dir.exists():
            print(f"\n  {label}: directory not found")
            continue

        # Check if fit_metrics already exists
        existing = run_dir / "fit_metrics.json"
        if existing.exists():
            with open(existing) as f:
                fm = json.load(f)
            m = fm["MAP"]
            print(f"\n  {label} (from existing fit_metrics.json)")
        else:
            result = simulate_and_compute_rmse(run_dir, label, condition, cultivation)
            if result is None:
                continue
            m = result
            print(f"\n  {label} (recomputed)")

        print(f"    Total RMSE = {m['rmse_total']:.4f}")
        print(f"    {'Species':<16} {'RMSE':>8}")
        print(f"    {'─'*24}")
        for i, sp in enumerate(SPECIES):
            flag = " <<<" if m["rmse_per_species"][i] > 0.10 else ""
            print(f"    {sp:<16} {m['rmse_per_species'][i]:8.4f}{flag}")
        results[label] = m

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: RMSE COMPARISON TABLE")
    print("=" * 80)
    print(f"\n  {'Run':<25} {'Total':>8} {'So':>8} {'An':>8} {'Vd':>8} {'Fn':>8} {'Pg':>8}")
    print(f"  {'─'*73}")
    for label, m in results.items():
        r = m["rmse_per_species"]
        print(
            f"  {label:<25} {m['rmse_total']:8.4f} {r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.4f}"
        )


if __name__ == "__main__":
    main()
