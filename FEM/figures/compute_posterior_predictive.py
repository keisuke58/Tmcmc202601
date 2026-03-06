#!/usr/bin/env python3
"""
compute_posterior_predictive.py
================================
Posterior predictive check using Numba solver.

Loads TMCMC posterior samples and run configs, runs forward model with
the exact same parameters used during estimation.

Usage:
    python compute_posterior_predictive.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tmcmc", "program2602"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_5species", "main"))

import numpy as np
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from improved_5species_jit import BiofilmNewtonSolver5S

# Condition run directories
RUN_DIRS = {
    "CS": "data_5species/_runs/commensal_static_posterior",
    "CH": "data_5species/_runs/commensal_hobic_posterior",
    "DH": "data_5species/_runs/dh_baseline",
    "DS": "data_5species/_runs/dysbiotic_static_posterior",
}

# Default solver params (from run configs)
DEFAULT_PARAMS = {
    "dt": 1e-4,
    "maxtimestep": 2500,
    "c_const": 25.0,
    "alpha_const": 0.0,
    "K_hill": 0.05,
    "n_hill": 4.0,
    "phi_init": 0.02,
}

# DH uses n_hill=2 (from Dysbiotic_HOBIC configs)
DH_OVERRIDES = {"n_hill": 2.0}

DAYS = np.array([1, 3, 6, 10, 15, 21])
SPECIES = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]
SP_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


def load_condition(cond):
    """Load samples, data, idx_sparse, and config for a condition."""
    run_dir = RUN_DIRS[cond]

    samples_path = os.path.join(run_dir, "samples.npy")
    if not os.path.exists(samples_path):
        return None

    result = {
        "samples": np.load(samples_path),
    }

    # Load data.npy (actual fitting data)
    data_path = os.path.join(run_dir, "data.npy")
    if os.path.exists(data_path):
        result["data"] = np.load(data_path)

    # Load idx_sparse (observation indices)
    idx_path = os.path.join(run_dir, "idx_sparse.npy")
    if os.path.exists(idx_path):
        result["idx_sparse"] = np.load(idx_path)

    # Load config
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            result["config"] = json.load(f)

    return result


def get_solver_params(cond, info):
    """Get solver parameters matching the TMCMC estimation."""
    params = dict(DEFAULT_PARAMS)

    # Override from config if available
    if "config" in info:
        cfg = info["config"]
        for key in ["dt", "maxtimestep", "c_const", "alpha_const", "K_hill", "n_hill", "phi_init"]:
            if key in cfg:
                params[key] = cfg[key]

    # DH override if no config
    if cond == "DH" and "config" not in info:
        params.update(DH_OVERRIDES)

    return params


def run_forward(samples, solver_params, idx_obs, max_samples=100):
    """Run forward model on posterior samples."""
    solver = BiofilmNewtonSolver5S(
        dt=solver_params["dt"],
        maxtimestep=solver_params["maxtimestep"],
        c_const=solver_params["c_const"],
        alpha_const=solver_params["alpha_const"],
        K_hill=solver_params["K_hill"],
        n_hill=solver_params["n_hill"],
        phi_init=solver_params["phi_init"],
    )

    n_samp = min(len(samples), max_samples)
    idx = np.random.choice(len(samples), n_samp, replace=False)

    phi_preds = []
    for i, si in enumerate(idx):
        theta = samples[si]
        try:
            t_arr, g_arr = solver.run_deterministic(theta)
            # φ̄ = φ × ψ (living bacteria volume fraction)
            phi = g_arr[idx_obs, :5]  # (n_obs, 5)
            psi = g_arr[idx_obs, 6:11]  # (n_obs, 5)
            phibar = phi * psi
            if np.all(np.isfinite(phibar)):
                phi_preds.append(phibar)
        except Exception:
            continue

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_samp} ({len(phi_preds)} ok)")

    return np.array(phi_preds)


def compute_metrics(phi_preds, y_obs):
    """Compute posterior predictive metrics (skip day 1 = IC)."""
    pred_mean = np.mean(phi_preds, axis=0)
    pred_q025 = np.percentile(phi_preds, 2.5, axis=0)
    pred_q975 = np.percentile(phi_preds, 97.5, axis=0)

    rmse = np.sqrt(np.mean((pred_mean[1:] - y_obs[1:]) ** 2))

    # 95% CI coverage (skip day 1)
    in_ci = (y_obs[1:] >= pred_q025[1:]) & (y_obs[1:] <= pred_q975[1:])
    coverage = np.mean(in_ci)
    sp_coverage = np.mean(in_ci, axis=0)

    return {
        "pred_mean": pred_mean,
        "pred_q025": pred_q025,
        "pred_q975": pred_q975,
        "rmse": rmse,
        "coverage": coverage,
        "sp_coverage": sp_coverage,
    }


def main():
    np.random.seed(42)
    results = {}

    for cond in ["CS", "CH", "DS", "DH"]:
        print(f"\n{'='*50}")
        print(f"Condition: {cond}")

        info = load_condition(cond)
        if info is None:
            print("  No samples found, skipping")
            continue

        samples = info["samples"]
        print(f"  Loaded {samples.shape[0]} samples")

        solver_params = get_solver_params(cond, info)
        print(
            f"  Solver: alpha={solver_params['alpha_const']}, n_hill={solver_params['n_hill']}, phi_init={solver_params['phi_init']}"
        )

        # Observation indices
        if "idx_sparse" in info:
            idx_obs = info["idx_sparse"].astype(int)
        else:
            N = solver_params["maxtimestep"]
            dt = solver_params["dt"]
            T_MAX = N * dt
            DAY_SCALE = T_MAX * 0.95 / DAYS.max()
            idx_obs = np.round(DAYS * DAY_SCALE / dt).astype(int)

        # Observation data
        if "data" in info:
            y_obs = info["data"]
        else:
            print("  No data.npy, skipping")
            continue

        print(f"  idx_obs = {idx_obs}")
        print(f"  y_obs[0] = {y_obs[0]}")

        phi_preds = run_forward(samples, solver_params, idx_obs, max_samples=100)
        print(f"  {len(phi_preds)} successful predictions")

        if len(phi_preds) < 5:
            print("  Too few predictions, skipping")
            continue

        m = compute_metrics(phi_preds, y_obs)
        results[cond] = {**m, "phi_preds": phi_preds, "y_obs": y_obs, "idx_obs": idx_obs}

        print(f"  RMSE = {m['rmse']:.4f}")
        print(f"  95% CI coverage = {m['coverage']:.0%}")
        print(f"  Per-species: {[f'{c:.0%}' for c in m['sp_coverage']]}")

    plot_predictive(results)
    print_summary(results)


def plot_predictive(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Posterior Predictive Check (100 samples)", fontsize=14, y=0.98)

    for ax, cond in zip(axes.flat, ["CS", "CH", "DS", "DH"]):
        if cond not in results:
            ax.set_title(f"{cond}: no data")
            continue
        r = results[cond]
        y_obs = r["y_obs"]

        for s in range(5):
            ax.fill_between(
                DAYS, r["pred_q025"][:, s], r["pred_q975"][:, s], color=SP_COLORS[s], alpha=0.2
            )
            ax.plot(DAYS, r["pred_mean"][:, s], "-", color=SP_COLORS[s], lw=1.5)
            ax.plot(DAYS, y_obs[:, s], "o", color=SP_COLORS[s], ms=6, label=SPECIES[s])

        ax.set_title(f"{cond} (RMSE={r['rmse']:.3f}, 95% cov={r['coverage']:.0%})")
        ax.set_xlabel("Day")
        ax.set_ylabel(r"$\varphi_i$")
        if cond == "CS":
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), "paper_final", "posterior_predictive_check.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


def print_summary(results):
    print("\n" + "=" * 60)
    print("LaTeX Summary")
    print("=" * 60)
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r" & RMSE & 95\% Coverage & Status \\")
    print(r"\midrule")
    for cond in ["CS", "CH", "DS", "DH"]:
        if cond in results:
            r = results[cond]
            status = (
                "calibrated"
                if abs(r["coverage"] - 0.95) < 0.15
                else ("over-confident" if r["coverage"] < 0.95 else "conservative")
            )
            print(f"{cond} & {r['rmse']:.3f} & {r['coverage']:.0%} & {status} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
