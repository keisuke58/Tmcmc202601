#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_sanz_martin_jax.py — TMCMC re-calibration on Sanz-Martin 2022 data.

Uses JAX ODE + GPU for fast TMCMC on the 6-species (→5) in vitro biofilm
data from Sanz-Martin et al. (2022) PMC8828709.

Two datasets: planktonic and adherent on polished cpTi.
Timepoints: 0.25, 1, 3, 7, 14, 21 days.

Usage:
    python estimate_sanz_martin_jax.py --dataset planktonic --mutation rw --n-particles 500
    python estimate_sanz_martin_jax.py --dataset adherent --mutation rw --n-particles 500
    python estimate_sanz_martin_jax.py --dataset both --mutation rw --n-particles 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Device selection MUST be before jax import
def _parse_device_early():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "auto"


_device_early = _parse_device_early()
if _device_early == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif _device_early in ("auto", "gpu"):
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda"
    try:
        import jax_cuda12_plugin  # noqa: F401
    except ImportError:
        pass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from hamilton_ode_jax import simulate_0d
from tmcmc_nuts_engine import tmcmc_engine

# ======================================================================
# Sanz-Martin 2022 Digitized Data (PMC8828709)
# ======================================================================
# Species order after renormalization: [So, An, Vd, Fn, Pg]
# Original 6 species: [So, An, Aa, Vp, Fn, Pg]
# A. actinomycetemcomitans (Aa) excluded, renormalized to 5 species.

SM_DAYS = np.array([0.25, 1, 3, 7, 14, 21])

# Planktonic composition (%) — 6 species [So, An, Aa, Vp, Fn, Pg]
SM_PLANKTONIC_6SP = {
    0.25: [75, 1, 24, 0, 0, 0],
    1: [45, 5, 5, 10, 35, 0],
    3: [10, 5, 5, 10, 70, 0],
    7: [15, 5, 5, 10, 65, 0],
    14: [10, 5, 5, 10, 37, 33],
    21: [5, 5, 5, 10, 20, 55],
}

# Adherent biofilm on polished cpTi (%) — 6 species
SM_ADHERENT_6SP = {
    0.25: [90, 2, 8, 0, 0, 0],
    1: [65, 5, 5, 5, 20, 0],
    3: [40, 5, 5, 10, 40, 0],
    7: [70, 5, 5, 5, 15, 0],
    14: [60, 5, 5, 5, 15, 10],
    21: [33, 8, 5, 5, 19, 30],
}

SP_NAMES = ["So", "An", "Vd", "Fn", "Pg"]


def renormalize_to_5species(data_6sp):
    """Remove Aa (index 2) and renormalize to 5 species fractions."""
    result = {}
    for day, vals in data_6sp.items():
        v = np.array(vals, dtype=np.float64)
        # Remove Aa (index 2): keep [So, An, Vp, Fn, Pg]
        v5 = np.array([v[0], v[1], v[3], v[4], v[5]])
        total = v5.sum()
        if total > 0:
            v5 = v5 / total
        result[day] = v5
    return result


def build_sm_data(dataset="planktonic"):
    """Build observation matrix and metadata for SM dataset."""
    if dataset == "planktonic":
        raw = SM_PLANKTONIC_6SP
    elif dataset == "adherent":
        raw = SM_ADHERENT_6SP
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    data_5sp = renormalize_to_5species(raw)
    days = sorted(data_5sp.keys())
    t_days = np.array(days)
    data = np.array([data_5sp[d] for d in days])  # (n_obs, 5)

    # Initial composition from first timepoint
    phi_init = data[0].copy()
    phi_init = np.clip(phi_init, 0.01, 0.99)
    phi_init = phi_init / phi_init.sum()

    # Estimate sigma from data variability (use 0.05 as default for digitized data)
    sigma_obs = 0.05

    return data, t_days, sigma_obs, phi_init


def convert_days_to_idx(t_days, dt, n_steps):
    """Map experimental days to ODE step indices."""
    t_max_model = n_steps * dt
    day_scale = (t_max_model * 0.95) / t_days.max()
    t_model = t_days * day_scale
    idx = np.round(t_model / dt).astype(int)
    idx = np.clip(idx, 0, n_steps)
    return idx


def make_log_likelihood_sm(
    data,
    idx_sparse,
    sigma_obs,
    phi_init,
    dt=1e-4,
    n_steps=2500,
    K_hill=0.05,
    n_hill=2.0,
    c_const=25.0,
    lambda_pg=5.0,
    lambda_late=3.0,
    n_late=2,
):
    """Build JAX log-likelihood for SM data."""
    obs = jnp.array(data, dtype=jnp.float64)
    phi_init_jax = jnp.array(phi_init, dtype=jnp.float64)
    idx = jnp.array(idx_sparse, dtype=jnp.int32)
    n_obs, n_species = data.shape

    # Weights: emphasize Pg and late timepoints
    weights = jnp.ones((n_obs, n_species))
    weights = weights.at[:, 4].set(lambda_pg)
    if n_late > 0 and n_obs > n_late:
        weights = weights.at[-n_late:, :].set(weights[-n_late:, :] * lambda_late)

    def log_likelihood(theta):
        phi_traj = simulate_0d(
            theta,
            n_steps=n_steps,
            dt=dt,
            phi_init=phi_init_jax,
            K_hill=K_hill,
            n_hill=n_hill,
            c_const=c_const,
        )
        phi_pred = phi_traj[idx, :]
        phi_pred = jnp.clip(phi_pred, 1e-10, 1.0 - 1e-10)
        phi_sum = jnp.sum(phi_pred, axis=1, keepdims=True)
        phi_pred = phi_pred / jnp.maximum(phi_sum, 1e-12)
        residual = obs - phi_pred
        logL = -0.5 * jnp.sum(weights * (residual / sigma_obs) ** 2)
        return logL

    return log_likelihood


def get_wide_prior_bounds():
    """Wide prior bounds for SM re-calibration (no condition-specific narrowing)."""
    bounds = np.zeros((20, 2))
    # Default: [-1, 3] for all interaction params (wider than Heine-calibrated)
    for i in range(20):
        bounds[i] = [-1.0, 3.0]
    # Growth rates b_i: positive bias
    for i in [3, 4, 8, 9, 15]:  # b indices
        bounds[i] = [0.0, 5.0]
    return bounds


def main():
    parser = argparse.ArgumentParser(description="TMCMC on Sanz-Martin 2022 data (JAX GPU)")
    parser.add_argument(
        "--dataset",
        default="planktonic",
        choices=["planktonic", "adherent", "both"],
    )
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--max-stages", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-pg", type=float, default=5.0)
    parser.add_argument("--lambda-late", type=float, default=3.0)
    parser.add_argument("--sigma-obs", type=float, default=0.05)
    parser.add_argument("--K-hill", type=float, default=0.05)
    parser.add_argument("--n-hill", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mutation", default="rw", choices=["rw", "hmc", "nuts"])
    parser.add_argument("--n-mutation-steps", type=int, default=10,
                        help="Mutation steps per particle per stage (default: 10)")
    parser.add_argument("--quick", action="store_true", help="Test: 50p, 500 steps")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "gpu"], default="auto",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_particles = 50
        args.n_steps = 500
        logger.info("Quick mode: n_particles=50, n_steps=500")

    devs = jax.devices()
    has_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs)
    logger.info(f"JAX devices: {[str(d) for d in devs]}")
    if args.device == "gpu" and not has_gpu:
        raise RuntimeError("GPU not available")

    datasets = ["planktonic", "adherent"] if args.dataset == "both" else [args.dataset]

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: Sanz-Martin 2022 — {ds_name}")
        logger.info(f"{'='*60}")

        data, t_days, sigma_default, phi_init = build_sm_data(ds_name)
        sigma_obs = args.sigma_obs if args.sigma_obs else sigma_default
        logger.info(f"Data shape: {data.shape}, timepoints: {t_days}")
        logger.info(f"phi_init: {phi_init}")
        logger.info(f"sigma_obs: {sigma_obs}")

        for i, d in enumerate(t_days):
            logger.info(f"  Day {d:5.2f}: {' '.join(f'{SP_NAMES[j]}={data[i,j]:.3f}' for j in range(5))}")

        idx_sparse = convert_days_to_idx(t_days, args.dt, args.n_steps)
        logger.info(f"idx_sparse: {idx_sparse}")

        log_likelihood = make_log_likelihood_sm(
            data=data,
            idx_sparse=idx_sparse,
            sigma_obs=sigma_obs,
            phi_init=phi_init,
            dt=args.dt,
            n_steps=args.n_steps,
            K_hill=args.K_hill,
            n_hill=args.n_hill,
            lambda_pg=args.lambda_pg,
            lambda_late=args.lambda_late,
        )

        prior_bounds = get_wide_prior_bounds()
        prior_bounds = np.array(prior_bounds, dtype=np.float32)

        logger.info("JIT warmup...")
        _ = jax.jit(log_likelihood)(jnp.zeros(20, dtype=jnp.float64))
        if args.mutation in ("hmc", "nuts"):
            _ = jax.jit(jax.value_and_grad(log_likelihood))(jnp.zeros(20, dtype=jnp.float64))
        logger.info("Warmup OK")

        logger.info(f"Running {args.mutation.upper()}-TMCMC ({args.n_particles} particles)...")
        result = tmcmc_engine(
            log_likelihood,
            prior_bounds,
            mutation=args.mutation,
            n_particles=args.n_particles,
            max_stages=args.max_stages,
            seed=args.seed,
            nuts_max_depth=6,
            n_mutation_steps=args.n_mutation_steps,
        )

        theta_MAP = result["theta_MAP"]
        logger.info(
            f"Done: {result['n_stages']} stages, "
            f"total_time={result['total_time']:.1f}s, "
            f"accept={np.mean(result['accept_rates']):.2f}, "
            f"max logL={result['log_likelihoods'].max():.1f}"
        )

        # Save results
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = SCRIPT_DIR / "_runs" / f"sanz_martin_{ds_name}_{args.mutation}_{ts}"
        out_dir = Path(args.output_dir) if args.output_dir else default_out
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "samples.npy", result["samples"])
        np.save(out_dir / "logL.npy", result["log_likelihoods"])
        np.save(out_dir / "betas.npy", result["betas"])
        with open(out_dir / "theta_MAP.json", "w") as f:
            json.dump(
                {f"theta_{i}": float(v) for i, v in enumerate(theta_MAP)},
                f, indent=2,
            )
        config = {
            "paper": "Sanz-Martin et al. 2022 (PMC8828709)",
            "dataset": ds_name,
            "n_particles": args.n_particles,
            "mutation": args.mutation,
            "sigma_obs": sigma_obs,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "K_hill": args.K_hill,
            "n_hill": args.n_hill,
            "lambda_pg": args.lambda_pg,
            "lambda_late": args.lambda_late,
            "prior_bounds": "wide [-1,3] / b:[0,5]",
            "n_stages": result["n_stages"],
            "total_time_s": result["total_time"],
            "mean_accept": float(np.mean(result["accept_rates"])),
            "max_logL": float(result["log_likelihoods"].max()),
            "phi_init": phi_init.tolist(),
            "t_days": t_days.tolist(),
            "data": data.tolist(),
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Print MAP theta summary
        logger.info("\n--- MAP theta ---")
        param_names = [
            "a00(So-So)", "a01(So-An)", "a11(An-An)", "b0(So)", "b1(An)",
            "a22(Vd-Vd)", "a23(Vd-Fn)", "a33(Fn-Fn)", "b2(Vd)", "b3(Fn)",
            "a02(So-Vd)", "a03(So-Fn)", "a12(An-Vd)", "a13(An-Fn)",
            "a44(Pg-Pg)", "b4(Pg)",
            "a04(So-Pg)", "a14(An-Pg)", "a24(Vd-Pg)", "a34(Fn-Pg)",
        ]
        for i, (name, val) in enumerate(zip(param_names, theta_MAP)):
            logger.info(f"  {i:2d} {name:15s} = {val:+.4f}")

        # Compute predicted trajectory at MAP
        phi_traj = simulate_0d(
            jnp.array(theta_MAP, dtype=jnp.float64),
            n_steps=args.n_steps,
            dt=args.dt,
            phi_init=jnp.array(phi_init, dtype=jnp.float64),
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )
        phi_pred = np.array(phi_traj[idx_sparse, :])
        phi_pred = phi_pred / phi_pred.sum(axis=1, keepdims=True)

        logger.info("\n--- Fit quality ---")
        for i, d in enumerate(t_days):
            obs_str = " ".join(f"{data[i,j]:.3f}" for j in range(5))
            pred_str = " ".join(f"{phi_pred[i,j]:.3f}" for j in range(5))
            logger.info(f"  Day {d:5.2f}  obs=[{obs_str}]  pred=[{pred_str}]")
        rmse = np.sqrt(np.mean((data - phi_pred) ** 2))
        cosine = np.dot(data.flatten(), phi_pred.flatten()) / (
            np.linalg.norm(data.flatten()) * np.linalg.norm(phi_pred.flatten())
        )
        logger.info(f"  RMSE={rmse:.4f}, Cosine={cosine:.4f}")

        # Save fit comparison
        fit_summary = {
            "rmse": float(rmse),
            "cosine_sim": float(cosine),
            "observed": data.tolist(),
            "predicted": phi_pred.tolist(),
            "t_days": t_days.tolist(),
        }
        with open(out_dir / "fit_summary.json", "w") as f:
            json.dump(fit_summary, f, indent=2)

        logger.info(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
