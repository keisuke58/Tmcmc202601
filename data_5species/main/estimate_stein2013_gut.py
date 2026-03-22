#!/usr/bin/env python3
"""
estimate_stein2013_gut.py — N-species TMCMC on Stein 2013 gut microbiome data.

Gnotobiotic mice, 11 genera, clindamycin perturbation, 23 days.
Uses hamilton_ode_jax_nsp (generalized N-species solver).

Usage:
    python estimate_stein2013_gut.py --mouse pop2_rep1_id1 --device gpu
    python estimate_stein2013_gut.py --mouse all --device gpu
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


# Device selection before JAX import
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
        import jax_cuda12_plugin  # noqa: F401
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

N_SP = 11
N_PARAMS = count_params(N_SP)  # 77
SP_NAMES = [
    "Enterobact",
    "Blautia",
    "Barnesiella",
    "Mollicutes",
    "Lachnospi",
    "Akkermansia",
    "C.difficile",
    "unc_Lachno",
    "Coprobacil",
    "Enterococc",
    "Other",
]

DATA_DIR = SCRIPT_DIR / "external_data" / "stein2013"


def load_mouse_data(mouse_key, start_from_idx=1):
    """Load CSV for one mouse. Use first timepoint as IC, fit rest."""
    csv_path = DATA_DIR / f"stein2013_{mouse_key}.csv"
    import csv as csv_mod

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    t_days = np.array([float(r["day"]) for r in rows])
    data = np.zeros((len(rows), N_SP))
    for i, r in enumerate(rows):
        for j, sp in enumerate(SP_NAMES):
            # Match CSV header
            for key in r:
                if sp in key or key in sp:
                    data[i, j] = float(r[key])
                    break

    # Re-read properly with column order from header
    with open(csv_path) as f:
        reader = csv_mod.reader(f)
        header = next(reader)
        data = []
        t_days = []
        for row in reader:
            t_days.append(float(row[0]))
            data.append([float(v) for v in row[1:]])
    t_days = np.array(t_days)
    data = np.array(data)

    # IC from first timepoint
    phi_init = data[0].copy()
    phi_init = np.clip(phi_init, 0.001, 0.99)
    phi_init = phi_init / phi_init.sum()

    # Fit from start_from_idx onward
    obs = data[start_from_idx:]
    t_fit = t_days[start_from_idx:]

    return obs, t_fit, phi_init, t_days, data


def convert_days_to_idx(t_days, dt, n_steps):
    t_max = n_steps * dt
    t_data_max = t_days.max()
    scale = (t_max * 0.95) / t_data_max
    idx = np.round(t_days * scale / dt).astype(int)
    return np.clip(idx, 0, n_steps)


def make_log_likelihood(data, idx_sparse, sigma_obs, phi_init, dt=1e-4, n_steps=2500, c_const=25.0):
    obs = jnp.array(data, dtype=jnp.float64)
    phi_init_jax = jnp.array(phi_init, dtype=jnp.float64)
    idx = jnp.array(idx_sparse, dtype=jnp.int32)
    n_obs = data.shape[0]
    sigma_arr = jnp.full((n_obs, 1), sigma_obs, dtype=jnp.float64)

    def log_likelihood(theta):
        phibar_traj = simulate_0d_nsp(
            theta,
            n_sp=N_SP,
            n_steps=n_steps,
            dt=dt,
            phi_init=phi_init_jax,
            c_const=c_const,
        )
        pred = phibar_traj[idx, :]
        pred = jnp.clip(pred, 1e-10, 1.0 - 1e-10)
        pred = pred / jnp.maximum(jnp.sum(pred, axis=1, keepdims=True), 1e-12)
        residual = obs - pred
        return -0.5 * jnp.sum((residual / sigma_arr) ** 2)

    return log_likelihood


def get_prior_bounds(use_glv_prior=False):
    """Prior bounds for 77 params.

    If use_glv_prior=True, uses Stein's gLV MAP (symmetrized) as center ± margin.
    """
    n_A = N_SP * (N_SP + 1) // 2  # 66
    bounds = np.zeros((N_PARAMS, 2))

    if use_glv_prior:
        prior_path = DATA_DIR / "stein2013_glv_informative_prior.json"
        with open(prior_path) as f:
            prior_data = json.load(f)
        center = np.array(prior_data["theta_prior_center"])
        margin = prior_data.get("margin", 2.0)
        for i in range(N_PARAMS):
            lo = center[i] - margin
            hi = center[i] + margin
            if i >= n_A:  # growth rates >= 0
                lo = max(lo, 0.0)
            bounds[i] = [lo, hi]
    else:
        for i in range(n_A):
            bounds[i] = [-2.0, 3.0]
        for i in range(n_A, N_PARAMS):
            bounds[i] = [0.0, 5.0]
    return bounds


def list_available_mice():
    csvs = sorted(DATA_DIR.glob("stein2013_pop*.csv"))
    return [p.stem.replace("stein2013_", "") for p in csvs]


def main():
    parser = argparse.ArgumentParser(description="N-species TMCMC on Stein 2013 gut data")
    parser.add_argument(
        "--mouse", default="pop2_rep1_id1", help="Mouse ID (e.g. pop2_rep1_id1) or 'all' or 'best'"
    )
    parser.add_argument("--n-particles", type=int, default=5000)
    parser.add_argument("--max-stages", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mutation-steps", type=int, default=20)
    parser.add_argument("--sigma-obs", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--max-delta-beta", type=float, default=0.1)
    parser.add_argument(
        "--glv-prior",
        action="store_true",
        default=True,
        help="Use Stein gLV MAP as informative prior (default: on)",
    )
    parser.add_argument(
        "--no-glv-prior",
        dest="glv_prior",
        action="store_false",
        help="Disable gLV prior, use wide bounds",
    )
    parser.add_argument("--mutation", default="rw", choices=["rw", "hmc", "nuts"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    devs = jax.devices()
    logger.info(f"JAX devices: {[str(d) for d in devs]}")
    logger.info(f"N_SP={N_SP}, N_PARAMS={N_PARAMS}")

    # Select mice
    available = list_available_mice()
    if args.mouse == "all":
        mice = available
    elif args.mouse == "best":
        # Use mice with >=10 timepoints
        mice = [m for m in available if "pop2" in m or "pop3" in m]
    else:
        mice = [args.mouse]

    logger.info(f"Mice to process: {mice}")

    for mouse_key in mice:
        logger.info(f"\n{'='*60}")
        logger.info(f"Stein 2013 gut — {mouse_key} ({N_SP} species, {N_PARAMS} params)")
        logger.info(f"{'='*60}")

        obs, t_fit, phi_init, t_all, data_all = load_mouse_data(mouse_key)
        logger.info(f"Data: {obs.shape} (fit), IC from Day {t_all[0]}")
        logger.info(f"Timepoints (fit): {t_fit}")
        logger.info(f"phi_init: {' '.join(f'{v:.3f}' for v in phi_init)}")

        idx_sparse = convert_days_to_idx(t_fit, args.dt, args.n_steps)
        logger.info(f"idx_sparse: {idx_sparse}")

        log_likelihood = make_log_likelihood(
            data=obs,
            idx_sparse=idx_sparse,
            sigma_obs=args.sigma_obs,
            phi_init=phi_init,
            dt=args.dt,
            n_steps=args.n_steps,
        )

        prior_bounds = get_prior_bounds(use_glv_prior=args.glv_prior)
        if args.glv_prior:
            logger.info("Using Stein gLV MAP as informative prior (center ± 2.0)")
        prior_bounds = np.array(prior_bounds, dtype=np.float32)

        logger.info(f"JIT warmup ({N_SP}sp, {N_PARAMS}p)...")
        _ = jax.jit(log_likelihood)(jnp.zeros(N_PARAMS, dtype=jnp.float64))
        logger.info("Warmup OK")

        logger.info(f"Running {args.mutation.upper()}-TMCMC ({args.n_particles}p)...")
        result = tmcmc_engine(
            log_likelihood,
            prior_bounds,
            mutation=args.mutation,
            n_particles=args.n_particles,
            max_stages=args.max_stages,
            seed=args.seed,
            nuts_max_depth=6,
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
        default_out = SCRIPT_DIR / "_runs" / f"stein2013_{mouse_key}_{args.n_particles}p"
        out_dir = Path(args.output_dir) if args.output_dir else default_out
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "samples.npy", result["samples"])
        np.save(out_dir / "logL.npy", result["log_likelihoods"])
        np.save(out_dir / "betas.npy", result["betas"])
        with open(out_dir / "theta_MAP.json", "w") as f:
            json.dump(
                {f"theta_{i}": float(v) for i, v in enumerate(theta_MAP)},
                f,
                indent=2,
            )

        config = {
            "source": "Stein et al. 2013 (PLoS Comput Biol 9:e1003388)",
            "mouse": mouse_key,
            "n_species": N_SP,
            "n_params": N_PARAMS,
            "species": SP_NAMES,
            "n_particles": args.n_particles,
            "mutation": args.mutation,
            "n_mutation_steps": args.n_mutation_steps,
            "sigma_obs": args.sigma_obs,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "max_delta_beta": args.max_delta_beta,
            "n_stages": result["n_stages"],
            "total_time_s": result["total_time"],
            "max_logL": float(result["log_likelihoods"].max()),
            "t_fit": t_fit.tolist(),
            "phi_init": phi_init.tolist(),
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Print MAP summary
        n_A = N_SP * (N_SP + 1) // 2
        logger.info("\n--- MAP growth rates ---")
        for i in range(N_SP):
            logger.info(f"  μ({SP_NAMES[i]:12s}) = {theta_MAP[n_A + i]:+.4f}")

        # Compute fit
        pred_traj = simulate_0d_nsp(
            jnp.array(theta_MAP),
            n_sp=N_SP,
            n_steps=args.n_steps,
            dt=args.dt,
            phi_init=jnp.array(phi_init),
        )
        pred = np.array(pred_traj[idx_sparse])
        pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        cos_sim = np.dot(obs.flatten(), pred.flatten()) / (
            np.linalg.norm(obs.flatten()) * np.linalg.norm(pred.flatten())
        )
        logger.info(f"\nRMSE={rmse:.4f}, Cosine={cos_sim:.4f}")
        logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
