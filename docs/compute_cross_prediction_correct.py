#!/usr/bin/env python3
"""
Cross-condition prediction RMSE matrix — correct computation.

Uses the colab_package JAX ODE (20-param, b terms active), same mapping
as the 10000p GPU-TMCMC runs (jax_ode_nuts_*_20260320_*).
Diagonal should match config.json rmse from each 10000p run.
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# ── paths ──────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent / "Tmcmc202601"
RUNS = PROJ / "data_5species" / "main" / "_runs"
DATA_DIR = PROJ / "data_5species" / "experiment_data"
MAIN_DIR = PROJ / "data_5species" / "main"
COLAB_DIR = PROJ / "colab_package"

# Add colab_package ONLY (20-param version with b decay terms).
# Do NOT add MAIN_DIR first — main/hamilton_ode_jax.py uses a 15-param layout
# (b=zeros) incompatible with the 10000p theta_MAP values.
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))

from hamilton_ode_jax import simulate_0d_full  # noqa: E402 — colab 20-param version (with b decay)

# ── 10000p GPU-TMCMC run dirs ───────────────────────────────────────────────
NUTS_RUNS = {
    "CS": "jax_ode_nuts_Commensal_Static_20260320_043505",
    "CH": "jax_ode_nuts_Commensal_HOBIC_20260320_043812",
    "DS": "jax_ode_nuts_Dysbiotic_Static_20260320_044847",
    "DH": "jax_ode_nuts_Dysbiotic_HOBIC_20260320_052844",
}
COND_META = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}
SPECIES_MAP_COMMENSAL = {
    "S. oralis": 0, "A. naeslundii": 1, "V. dispar": 2,
    "F. nucleatum": 3, "P. gingivalis_20709": 4,
}
SPECIES_MAP_DYSBIOTIC = {
    "S. oralis": 0, "A. naeslundii": 1, "V. parvula": 2,
    "F. nucleatum": 3, "P. gingivalis_W83": 4,
}

# ── model params matching 10000p GPU runs ──────────────────────────────────
DT          = 1e-4
N_STEPS     = 2500
K_HILL      = 0.05
N_HILL      = 4.0      # from config.json
C_CONST     = 25.0     # default in estimate_reduced_nishioka_jax.py
ALPHA_CONST = 0.0
OBS_DAYS    = np.array([3, 6, 10, 15, 21])   # Day 1 = IC


def load_theta(cond_key: str) -> np.ndarray:
    f = RUNS / NUTS_RUNS[cond_key] / "theta_MAP.json"
    d = json.load(open(f))
    return np.array([d[str(i)] for i in range(20)], dtype=float)


def load_exp_data(cond_key: str):
    """Return (phi_ic_exp, phi_ic_uniform, data_obs_norm, idx_sparse).

    phi_ic_exp     : (5,) Day-1 fractions normalized to sum=1, clipped [0.01, 0.99]
    phi_ic_uniform : (5,) uniform 0.2 (default if --use-exp-init not set)
    data_obs_norm  : (n_obs, 5) species fractions normalized to row-sum=1
    idx_sparse     : (n_obs,) model time indices for OBS_DAYS
    """
    condition, cultivation = COND_META[cond_key]
    sp_map = SPECIES_MAP_COMMENSAL if condition == "Commensal" else SPECIES_MAP_DYSBIOTIC

    bp = pd.read_csv(DATA_DIR / f"boxplot_{condition}_{cultivation}.csv")
    bp = bp[(bp["condition"] == condition) & (bp["cultivation"] == cultivation)]

    sp = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
    sp = sp[(sp["condition"] == condition) & (sp["cultivation"] == cultivation)]

    all_days = sorted(bp["day"].unique())
    data_abs = np.zeros((len(all_days), 5))

    for i, day in enumerate(all_days):
        tv = float(bp[bp["day"] == day]["median"].iloc[0])
        for _, row in sp[sp["day"] == day].iterrows():
            name = row["species"]
            if name in sp_map:
                data_abs[i, sp_map[name]] = tv * row["median"] / 100.0

    # Day 1 IC: normalize to sum=1, clip to [0.01, 0.99]
    phi_ic_exp = data_abs[0].copy()
    total = phi_ic_exp.sum()
    if total > 1e-10:
        phi_ic_exp = phi_ic_exp / total
    phi_ic_exp = np.clip(phi_ic_exp, 0.01, 0.99)

    # Uniform IC (default in estimate_reduced_nishioka_jax.py)
    phi_ic_uniform = np.full(5, 0.2)

    # Observations = days > 1, normalize rows to sum=1
    obs_mask = np.array(all_days) > 1
    data_obs = data_abs[obs_mask]
    row_sums = data_obs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    data_obs_norm = data_obs / row_sums

    # Time indices (same formula as estimate_reduced_nishioka.py convert_days_to_model_time)
    t_max = N_STEPS * DT
    day_scale = (t_max * 0.95) / float(max(all_days))
    idx_sparse = np.round(OBS_DAYS * day_scale / DT).astype(int)
    idx_sparse = np.clip(idx_sparse, 0, N_STEPS - 1)

    return phi_ic_exp, phi_ic_uniform, data_obs_norm, idx_sparse


import jax.numpy as jnp

def run_model(theta: np.ndarray, phi_ic: np.ndarray) -> np.ndarray:
    """Run JAX simulate_0d_full (colab 20-param) → return phi trajectory (N_STEPS+1, 5).

    Uses phi (volume fraction) only — NOT phi*psi — because the 10000p GPU-TMCMC
    calibration RMSE in config.json is computed against normalized phi (same as the
    likelihood which normalizes phi_pred before comparing to normalized data).
    """
    phi_init_jax = jnp.array(phi_ic, dtype=jnp.float64)
    theta_jax = jnp.array(theta, dtype=jnp.float64)
    g_traj = simulate_0d_full(
        theta_jax,
        n_steps=N_STEPS,
        dt=DT,
        phi_init=phi_init_jax,
        K_hill=K_HILL,
        n_hill=N_HILL,
        c_const=C_CONST,
        alpha_const=ALPHA_CONST,
    )
    g = np.array(g_traj)
    return g[:, 0:5]  # phi only (rows: time steps, cols: 5 species)


def compute_rmse(phi_traj: np.ndarray, idx_sparse: np.ndarray,
                 data_obs_norm: np.ndarray) -> float:
    """Normalize phi at observation times, compare vs normalized data."""
    pred = phi_traj[idx_sparse, :]          # (n_obs, 5)
    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) > 1e-12, row_sums, 1.0)
    pred_norm = pred / row_sums
    return float(np.sqrt(np.mean((pred_norm - data_obs_norm) ** 2)))


def main():
    thetas = {k: load_theta(k) for k in COND_META}
    exp    = {k: load_exp_data(k) for k in COND_META}

    conds = ["CS", "CH", "DS", "DH"]

    # Try both IC variants to find which matches config.json diagonal
    print("=== Checking IC variants for diagonal match ===")
    print(f"{'':4s}  {'exp IC':>8s}  {'uniform IC':>10s}  {'config.json':>11s}")
    for k in conds:
        phi_ic_exp, phi_ic_uni, data_obs, idx_sparse = exp[k]
        phi_traj_exp = run_model(thetas[k], phi_ic_exp)
        phi_traj_uni = run_model(thetas[k], phi_ic_uni)
        rmse_exp = compute_rmse(phi_traj_exp, idx_sparse, data_obs)
        rmse_uni = compute_rmse(phi_traj_uni, idx_sparse, data_obs)
        cfg = json.load(open(RUNS / NUTS_RUNS[k] / "config.json"))
        print(f"  {k}:  {rmse_exp:.4f}      {rmse_uni:.4f}       {cfg['rmse']:.4f}")

    # Build 4×4 matrix with experimental IC (closer to visualization convention)
    print("\n=== Computing full 4×4 matrix with experimental Day-1 IC ===")
    rmse_mat = np.full((4, 4), np.nan)
    for ri, src in enumerate(conds):
        for ci, tgt in enumerate(conds):
            phi_ic_exp, _, data_obs_norm, idx_sparse = exp[tgt]
            phi_traj = run_model(thetas[src], phi_ic_exp)
            rmse_mat[ri, ci] = compute_rmse(phi_traj, idx_sparse, data_obs_norm)
            print(f"  {src}→{tgt}: {rmse_mat[ri, ci]:.4f}")

    print("\n=== RMSE matrix (exp IC) ===")
    print(f"{'':4s}  {'CS':>6s} {'CH':>6s} {'DS':>6s} {'DH':>6s}")
    for ri, src in enumerate(conds):
        vals = "  ".join(
            f"*{rmse_mat[ri,ci]:.3f}*" if ri == ci else f" {rmse_mat[ri,ci]:.3f} "
            for ci in range(4)
        )
        print(f"{src}    {vals}")

    print("\n=== diagonal vs config.json rmse ===")
    for i, k in enumerate(conds):
        cfg = json.load(open(RUNS / NUTS_RUNS[k] / "config.json"))
        print(f"  {k}: computed={rmse_mat[i,i]:.4f}  config={cfg['rmse']:.4f}")

    print("\n% --- LaTeX table ---")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r"Source $\backslash$ Target & CS & CH & DS & DH \\")
    print(r"\midrule")
    for ri, src in enumerate(conds):
        cells = []
        for ci, tgt in enumerate(conds):
            v = rmse_mat[ri, ci]
            if ri == ci:
                cells.append(rf"\textbf{{{v:.3f}}}")
            elif src[0] != tgt[0]:
                cells.append(rf"\textit{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        print(f"{src} & {' & '.join(cells)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
