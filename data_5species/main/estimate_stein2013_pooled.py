#!/usr/bin/env python3
"""
estimate_stein2013_pooled.py — Pooled TMCMC on all pop2+pop3 mice.

Single A matrix + μ estimated from all 6 mice simultaneously.
Same approach as Stein's Ridge regression (but Bayesian).

Usage:
    python estimate_stein2013_pooled.py --device gpu
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

def _parse_device_early():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "auto"

_dev = _parse_device_early()
if _dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif _dev in ("auto", "gpu"):
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda"
    try:
        import jax_cuda12_plugin
    except ImportError:
        pass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(SCRIPT_DIR))
from hamilton_ode_jax_nsp import simulate_0d_nsp, count_params
from tmcmc_nuts_engine import tmcmc_engine
from estimate_stein2013_gut import load_mouse_data, convert_days_to_idx, SP_NAMES

N_SP = 11
N_PARAMS = count_params(N_SP)  # 77
DATA_DIR = SCRIPT_DIR / "external_data" / "stein2013"

MICE_POP23 = [
    "pop2_rep1_id1", "pop2_rep2_id4", "pop2_rep3_id7",
    "pop3_rep1_id3", "pop3_rep2_id6", "pop3_rep3_id9",
]


def make_pooled_likelihood(mice, start_from_idx, sigma_obs, dt, n_steps):
    """Build likelihood that sums logL over all mice with shared theta."""
    mouse_data = []
    for mouse_key in mice:
        obs, t_fit, phi_init, t_all, data_all = load_mouse_data(
            mouse_key, start_from_idx=start_from_idx
        )
        idx_sparse = convert_days_to_idx(t_fit, dt, n_steps)
        mouse_data.append({
            "obs": jnp.array(obs, dtype=jnp.float64),
            "phi_init": jnp.array(phi_init, dtype=jnp.float64),
            "idx": jnp.array(idx_sparse, dtype=jnp.int32),
            "n_obs": len(obs),
        })
        logger.info(
            f"  {mouse_key}: IC Day{t_all[start_from_idx-1]:.0f}, "
            f"fit {len(t_fit)} pts (Day{t_fit[0]:.0f}-{t_fit[-1]:.0f})"
        )

    sigma = jnp.float64(sigma_obs)

    def log_likelihood(theta):
        total_logL = jnp.float64(0.0)
        for md in mouse_data:
            traj = simulate_0d_nsp(
                theta, n_sp=N_SP, n_steps=n_steps, dt=dt,
                phi_init=md["phi_init"],
            )
            pred = traj[md["idx"], :]
            pred = jnp.clip(pred, 1e-10, 1.0 - 1e-10)
            pred = pred / jnp.maximum(jnp.sum(pred, axis=1, keepdims=True), 1e-12)
            residual = md["obs"] - pred
            total_logL += -0.5 * jnp.sum((residual / sigma) ** 2)
        return total_logL

    total_pts = sum(md["n_obs"] for md in mouse_data)
    logger.info(f"Pooled: {len(mice)} mice, {total_pts} timepoints, {N_PARAMS} params")
    return log_likelihood


def get_prior_bounds(use_glv_prior=True):
    n_A = N_SP * (N_SP + 1) // 2
    bounds = np.zeros((N_PARAMS, 2))
    if use_glv_prior:
        prior_path = DATA_DIR / "stein2013_glv_informative_prior.json"
        with open(prior_path) as f:
            prior_data = json.load(f)
        center = np.array(prior_data["theta_prior_center"])
        margin = 2.0
        for i in range(N_PARAMS):
            lo = center[i] - margin
            hi = center[i] + margin
            if i >= n_A:
                lo = max(lo, 0.0)
            bounds[i] = [lo, hi]
    else:
        for i in range(n_A):
            bounds[i] = [-2.0, 3.0]
        for i in range(n_A, N_PARAMS):
            bounds[i] = [0.0, 5.0]
    return bounds


def main():
    parser = argparse.ArgumentParser(description="Pooled TMCMC on Stein 2013")
    parser.add_argument("--n-particles", type=int, default=5000)
    parser.add_argument("--max-stages", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mutation-steps", type=int, default=30)
    parser.add_argument("--sigma-obs", type=float, default=0.12)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--max-delta-beta", type=float, default=0.1)
    parser.add_argument("--start-from-idx", type=int, default=2)
    parser.add_argument("--glv-prior", action="store_true", default=True)
    parser.add_argument("--no-glv-prior", dest="glv_prior", action="store_false")
    parser.add_argument("--mutation", default="rw", choices=["rw", "hmc", "nuts"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    devs = jax.devices()
    logger.info(f"JAX devices: {[str(d) for d in devs]}")
    logger.info(f"Pooled TMCMC: {len(MICE_POP23)} mice, {N_SP} species, {N_PARAMS} params")

    log_likelihood = make_pooled_likelihood(
        MICE_POP23, args.start_from_idx, args.sigma_obs, args.dt, args.n_steps
    )

    prior_bounds = get_prior_bounds(use_glv_prior=args.glv_prior)
    if args.glv_prior:
        logger.info("Using gLV informative prior")
    prior_bounds = np.array(prior_bounds, dtype=np.float32)

    logger.info("JIT warmup...")
    _ = jax.jit(log_likelihood)(jnp.zeros(N_PARAMS, dtype=jnp.float64))
    logger.info("Warmup OK")

    logger.info(f"Running {args.mutation.upper()}-TMCMC ({args.n_particles}p, pooled)...")
    result = tmcmc_engine(
        log_likelihood, prior_bounds,
        mutation=args.mutation,
        n_particles=args.n_particles,
        max_stages=args.max_stages,
        seed=args.seed,
        n_mutation_steps=args.n_mutation_steps,
        max_delta_beta=args.max_delta_beta,
    )

    theta_MAP = result["theta_MAP"]
    logger.info(
        f"Done: {result['n_stages']} stages, "
        f"time={result['total_time']:.1f}s, "
        f"accept={np.mean(result['accept_rates']):.2f}, "
        f"max logL={result['log_likelihoods'].max():.1f}"
    )

    # Save
    default_out = SCRIPT_DIR / "_runs" / f"stein2013_pooled_{args.n_particles}p"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "samples.npy", result["samples"])
    np.save(out_dir / "logL.npy", result["log_likelihoods"])
    np.save(out_dir / "betas.npy", result["betas"])
    with open(out_dir / "theta_MAP.json", "w") as f:
        json.dump({f"theta_{i}": float(v) for i, v in enumerate(theta_MAP)}, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "mice": MICE_POP23,
            "n_mice": len(MICE_POP23),
            "n_species": N_SP,
            "n_params": N_PARAMS,
            "n_particles": args.n_particles,
            "sigma_obs": args.sigma_obs,
            "start_from_idx": args.start_from_idx,
            "n_stages": result["n_stages"],
            "total_time_s": result["total_time"],
            "max_logL": float(result["log_likelihoods"].max()),
        }, f, indent=2)

    # Per-mouse RMSE with pooled theta
    logger.info("\n--- Per-mouse RMSE (pooled theta) ---")
    from hamilton_ode_jax_nsp import simulate_0d_nsp as sim
    for mouse_key in MICE_POP23:
        obs, t_fit, phi_init, t_all, data_all = load_mouse_data(
            mouse_key, start_from_idx=args.start_from_idx
        )
        idx = convert_days_to_idx(t_fit, args.dt, args.n_steps)
        traj = np.array(sim(
            jnp.array(theta_MAP), n_sp=N_SP, n_steps=args.n_steps, dt=args.dt,
            phi_init=jnp.array(phi_init),
        ))
        pred = traj[idx]
        pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        logger.info(f"  {mouse_key}: RMSE={rmse:.4f}")

    logger.info(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
