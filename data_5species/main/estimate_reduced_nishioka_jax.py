#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_reduced_nishioka_jax.py — TMCMC with JAX ODE + NUTS.

Uses pure JAX Hamilton ODE (hamilton_ode_jax) instead of DeepONet.
Enables NUTS mutation with exact gradients, no surrogate approximation.

Usage:
    cd data_5species/main
    python estimate_reduced_nishioka_jax.py --condition Dysbiotic --cultivation HOBIC \\
        --n-particles 200 --use-exp-init

Requires: jax, jaxlib (e.g. conda env klempt_fem2)

  PYTHON=$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem2/bin/python
  $PYTHON estimate_reduced_nishioka_jax.py --condition Dysbiotic --cultivation HOBIC ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_DIR = SCRIPT_DIR
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
    # CUDA を優先: JAX_PLATFORMS が未設定なら cuda を明示（ROCM 誤検出回避）
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda"
    # jax-cuda12-plugin を JAX より先にロード（CUDA 初期化を確実に）
    try:
        import jax_cuda12_plugin  # noqa: F401
    except ImportError:
        pass  # CPU-only jaxlib の場合は無視

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports
from hamilton_ode_jax import simulate_0d

# Import data loading and bounds from main estimator
from estimate_reduced_nishioka import (
    convert_days_to_model_time,
    load_experimental_data,
    load_multichannel_data,
)
from core.nishioka_model import get_condition_bounds

from tmcmc_nuts_engine import tmcmc_engine


def make_log_likelihood_jax_ode(
    data: np.ndarray,
    t_days: np.ndarray,
    idx_sparse: np.ndarray,
    sigma_obs: float,
    phi_init: np.ndarray,
    dt: float = 1e-4,
    n_steps: int = 2500,
    K_hill: float = 0.05,
    n_hill: float = 2.0,
    c_const: float = 25.0,
    lambda_pg: float = 1.0,
    lambda_late: float = 1.0,
    n_late: int = 2,
    # Multi-channel arguments
    data_total: np.ndarray | None = None,
    sigma_obs_total: float | np.ndarray | None = None,
    data_viability: np.ndarray | None = None,
    sigma_obs_viability: float = 0.10,
    data_pH: np.ndarray | None = None,
    idx_pH: np.ndarray | None = None,
    sigma_obs_pH: float = 0.15,
    lambda_ch: dict | None = None,
    # Robust likelihood
    use_student_t: bool = False,
    student_t_nu: float = 5.0,
    # Rare species downweighting
    lambda_rare: float = 1.0,
    rare_threshold: float = 0.05,
):
    """
    Build JAX-differentiable log-likelihood using Hamilton ODE.

    Supports multi-channel likelihood:
      Ch1: viable species (φ·ψ) — primary
      Ch2: total species (φ) — qRT-PCR without PMA
      Ch3: membrane integrity (viability) — LIVE/DEAD
      Ch5: pH time series — HOBIC only

    Returns
    -------
    log_likelihood : callable(theta) -> scalar
    """
    if lambda_ch is None:
        lambda_ch = {1: 1.0, 2: 0.5, 3: 2.0, 5: 0.3}

    obs = jnp.array(data, dtype=jnp.float64)
    phi_init_jax = jnp.array(phi_init, dtype=jnp.float64)
    idx = jnp.array(idx_sparse, dtype=jnp.int32)
    n_obs, n_species = data.shape

    # Weights: lambda_pg for Pg (species 4), lambda_late for last n_late timepoints
    weights = jnp.ones((n_obs, n_species))
    weights = weights.at[:, 4].set(lambda_pg)
    if n_late > 0:
        late_slice = slice(-n_late, None)
        weights = weights.at[late_slice, :].set(weights[late_slice, :] * lambda_late)
    # Downweight rare species: if mean abundance < rare_threshold, apply lambda_rare
    if lambda_rare < 1.0:
        mean_abundance = jnp.mean(obs, axis=0)  # (n_species,)
        rare_mask = (mean_abundance < rare_threshold).astype(jnp.float64)
        rare_weight = rare_mask * lambda_rare + (1.0 - rare_mask) * 1.0
        weights = weights * rare_weight[jnp.newaxis, :]

    # Pre-convert multichannel data to JAX arrays
    _has_total = data_total is not None
    _has_viab = data_viability is not None
    _has_pH = data_pH is not None and idx_pH is not None

    if _has_total:
        obs_total = jnp.array(data_total, dtype=jnp.float64)
        if sigma_obs_total is None:
            sigma_obs_total = 0.08
        sig_total = (
            jnp.array(sigma_obs_total, dtype=jnp.float64)
            if hasattr(sigma_obs_total, "__len__")
            else jnp.float64(sigma_obs_total)
        )
    if _has_viab:
        obs_viab = jnp.array(data_viability, dtype=jnp.float64)
        sig_viab = jnp.float64(sigma_obs_viability)
    if _has_pH:
        obs_pH = jnp.array(data_pH, dtype=jnp.float64)
        idx_ph = jnp.array(idx_pH, dtype=jnp.int32)
        sig_pH = jnp.float64(sigma_obs_pH)

    lw1 = lambda_ch.get(1, 1.0)
    lw2 = lambda_ch.get(2, 0.5)
    lw3 = lambda_ch.get(3, 2.0)
    lw5 = lambda_ch.get(5, 0.3)

    # Student-t log-density: log p(x | mu, sigma, nu)
    # More robust to outliers than Gaussian (heavier tails)
    _use_t = use_student_t
    _nu = jnp.float64(student_t_nu)

    def _log_student_t(residual_scaled_sq):
        """Log-density of Student-t (up to constant), given (r/sigma)^2."""
        return -0.5 * (_nu + 1) * jnp.log(1.0 + residual_scaled_sq / _nu)

    # Use full state solver when viability/pH channels are needed
    _need_full = _has_viab or _has_pH

    def log_likelihood(theta):
        if _need_full:
            from hamilton_ode_jax import simulate_0d_full

            g_traj = simulate_0d_full(
                theta,
                n_steps=n_steps,
                dt=dt,
                phi_init=phi_init_jax,
                K_hill=K_hill,
                n_hill=n_hill,
                c_const=c_const,
            )
            phi_traj = g_traj[:, 0:5]
        else:
            phi_traj = simulate_0d(
                theta,
                n_steps=n_steps,
                dt=dt,
                phi_init=phi_init_jax,
                K_hill=K_hill,
                n_hill=n_hill,
                c_const=c_const,
            )

        # Sample at observation indices
        phi_pred = phi_traj[idx, :]  # (n_obs, 5)
        phi_pred = jnp.clip(phi_pred, 1e-10, 1.0 - 1e-10)
        # Normalize to fractions
        phi_sum = jnp.sum(phi_pred, axis=1, keepdims=True)
        phi_pred = phi_pred / jnp.maximum(phi_sum, 1e-12)

        # Ch1: viable species (primary)
        residual = obs - phi_pred
        r_sq = (residual / sigma_obs) ** 2
        if _use_t:
            # Student-t: weights multiply the log-density, NOT the argument
            # (multiplying the argument changes effective nu per observation)
            logL = lw1 * jnp.sum(weights * _log_student_t(r_sq))
        else:
            logL = lw1 * (-0.5 * jnp.sum(weights * r_sq))

        # Ch2: total species distribution (φ without ψ weighting)
        if _has_total:
            residual_total = obs_total - phi_pred
            logL_total = -0.5 * jnp.sum((residual_total / sig_total) ** 2)
            logL = logL + lw2 * logL_total

        # Ch3: membrane integrity / viability
        if _has_viab:
            phi_at_obs = g_traj[idx, 0:5]
            psi_at_obs = g_traj[idx, 6:11]
            phi_psi = phi_at_obs * psi_at_obs
            viab_pred = jnp.sum(phi_psi, axis=1) / jnp.maximum(jnp.sum(phi_at_obs, axis=1), 1e-12)
            residual_viab = obs_viab - viab_pred
            logL_viab = -0.5 * jnp.sum((residual_viab / sig_viab) ** 2)
            logL = logL + lw3 * logL_viab

        # Ch5: pH (HOBIC only)
        if _has_pH:
            phi_at_pH = g_traj[idx_ph, 0:5]
            psi_at_pH = g_traj[idx_ph, 6:11]
            phibar = phi_at_pH * psi_at_pH
            phibar_sum = jnp.maximum(jnp.sum(phibar, axis=1), 1e-12)
            phi_bar_So = phibar[:, 0] / phibar_sum
            phi_bar_Vd = phibar[:, 2] / phibar_sum
            pH_pred = 7.5 - 0.74 * phi_bar_So - 0.48 * phi_bar_Vd
            residual_pH = obs_pH - pH_pred
            logL_pH = -0.5 * jnp.sum((residual_pH / sig_pH) ** 2)
            logL = logL + lw5 * logL_pH

        return logL

    return log_likelihood


def load_prior_bounds(condition: str, cultivation: str) -> np.ndarray:
    """Get prior bounds as (20, 2) array."""
    bounds, _ = get_condition_bounds(condition, cultivation)
    return np.array(bounds[:20], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="TMCMC with JAX ODE + NUTS (no DeepONet)")
    parser.add_argument("--condition", default="Dysbiotic")
    parser.add_argument("--cultivation", default="HOBIC")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--max-stages", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-exp-init", action="store_true")
    parser.add_argument("--start-from-day", type=int, default=1)
    parser.add_argument("--lambda-pg", type=float, default=5.0)
    parser.add_argument("--lambda-late", type=float, default=3.0)
    parser.add_argument(
        "--lambda-rare",
        type=float,
        default=0.1,
        help="Weight for rare species (mean < 5%%). Default 0.1 = downweight 10x",
    )
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument(
        "--replicate-sigma",
        action="store_true",
        help="Use per-timepoint×per-species replicate sigma matrix (matches CPU)",
    )
    parser.add_argument("--K-hill", type=float, default=0.05)
    parser.add_argument("--n-hill", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mutation", default="nuts", choices=["rw", "hmc", "nuts"])
    parser.add_argument("--quick", action="store_true", help="Test run: 50p, 500 steps")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ultra-short benchmark: 20p, 500 steps, 3 stages (~1min each)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device: auto (use GPU if available), cpu, gpu",
    )
    # Multi-channel likelihood
    parser.add_argument(
        "--multichannel",
        action="store_true",
        help="Enable multi-channel likelihood: viable + total + viability + pH",
    )
    parser.add_argument("--lambda-ch1", type=float, default=1.0, help="Weight for viable channel")
    parser.add_argument("--lambda-ch2", type=float, default=0.5, help="Weight for total species")
    parser.add_argument("--lambda-ch3", type=float, default=2.0, help="Weight for viability")
    parser.add_argument("--lambda-ch5", type=float, default=0.3, help="Weight for pH (HOBIC only)")
    parser.add_argument(
        "--n-mutation-steps", type=int, default=1, help="RW mutation steps per stage"
    )
    # Accuracy improvements
    parser.add_argument(
        "--use-de-mc", action="store_true", help="Enable DE-MC proposals (alternating with RW)"
    )
    parser.add_argument(
        "--de-mc-gamma",
        type=float,
        default=2.38,
        help="DE-MC scale factor (default 2.38, ter Braak 2006)",
    )
    parser.add_argument(
        "--use-student-t", action="store_true", help="Use Student-t likelihood (robust to outliers)"
    )
    parser.add_argument(
        "--student-t-nu",
        type=float,
        default=5.0,
        help="Student-t degrees of freedom (lower = heavier tails, default 5)",
    )
    parser.add_argument(
        "--resample-method",
        default="systematic",
        choices=["systematic", "multinomial"],
        help="Resampling method (systematic = lower variance)",
    )
    parser.add_argument(
        "--max-info-ratio",
        type=float,
        default=10.0,
        help="Max information ratio for replicate sigma (caps An dominance, default 10)",
    )
    # Flow-enhanced SMC (Gabrie et al. 2022)
    parser.add_argument("--use-flow", action="store_true", help="Enable NF proposal (RealNVP)")
    parser.add_argument("--flow-n-layers", type=int, default=6)
    parser.add_argument("--flow-hidden-dim", type=int, default=64)
    parser.add_argument("--flow-n-epochs", type=int, default=200)
    parser.add_argument("--flow-lr", type=float, default=1e-3)
    parser.add_argument("--flow-start-beta", type=float, default=0.1)
    parser.add_argument("--flow-mix-ratio", type=float, default=0.8)
    # Waste-free SMC (Dau & Chopin 2022)
    parser.add_argument("--waste-free", action="store_true", help="Enable waste-free SMC")
    args = parser.parse_args()

    if args.quick:
        args.n_particles = 50
        args.n_steps = 500
        logger.info("Quick mode: n_particles=50, n_steps=500")
    if args.benchmark:
        args.n_particles = 20
        args.n_steps = 500
        args.max_stages = 3
        logger.info("Benchmark mode: n_particles=20, n_steps=500, max_stages=3")

    devs = jax.devices()
    has_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs)
    logger.info(f"JAX devices: {[str(d) for d in devs]} (--device={args.device})")
    if args.device == "gpu" and not has_gpu:
        raise RuntimeError(
            "GPU が利用できません。以下を確認してください:\n"
            "  1. pip install jax[cuda12] または jax-cuda12-plugin が入っているか\n"
            "  2. nvidia-smi で GPU が認識されているか\n"
            '  3. python -c "import jax; print(jax.devices())" でデバイス確認\n'
            "  4. LD_LIBRARY_PATH がシステム CUDA を指しており pip の nvidia-* と競合していないか"
        )

    logger.info("Loading experimental data...")
    data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
        DATA_DIR,
        args.condition,
        args.cultivation,
        args.start_from_day,
        normalize=True,
    )
    if args.replicate_sigma:
        from estimate_reduced_nishioka import build_replicate_sigma

        replicate_csv = DATA_DIR / "fig3_species_distribution_replicates.csv"
        if not replicate_csv.exists():
            replicate_csv = (
                DATA_DIR.parent / "experiment_data" / "fig3_species_distribution_replicates.csv"
            )
        _rep_species_map = {
            "S. oralis": 0,
            "A. naeslundii": 1,
            "V. dispar": 2,
            "V. parvula": 2,
            "F. nucleatum": 3,
            "P. gingivalis_20709": 4,
            "P. gingivalis_W83": 4,
        }
        sigma_obs = (
            build_replicate_sigma(
                replicate_csv=str(replicate_csv),
                condition=args.condition,
                cultivation=args.cultivation,
                days=metadata["days"],
                species_map=_rep_species_map,
                n_species=data.shape[1],
                min_sigma=0.05,
                max_info_ratio=getattr(args, "max_info_ratio", 10.0),
            )
            * args.sigma_scale
        )
        logger.info(f"Replicate sigma: shape={sigma_obs.shape}, mean={np.mean(sigma_obs):.4f}")
    else:
        sigma_obs = sigma_obs_est * args.sigma_scale
    logger.info(f"Data: {data.shape}, sigma_obs mean={np.mean(sigma_obs):.4f}")

    t_model, idx_sparse = convert_days_to_model_time(t_days, args.dt, args.n_steps, day_scale=None)
    idx_sparse = np.clip(idx_sparse, 0, args.n_steps)
    logger.info(f"idx_sparse: {idx_sparse}")

    phi_init = phi_init_exp if args.use_exp_init else np.full(5, 0.2)
    if args.use_exp_init:
        total = phi_init.sum()
        if total > 0:
            phi_init = phi_init / total
        phi_init = np.clip(phi_init, 0.001, 0.99)

    # Multi-channel data loading
    mc_kwargs = {}
    if args.multichannel:
        mc_data = load_multichannel_data(
            data_dir=DATA_DIR / "experiment_data",
            condition=args.condition,
            cultivation=args.cultivation,
            days_filter=t_days.tolist(),
        )
        # Adaptive channel weights: DH needs stronger viability signal for Pg
        # HOBIC uses pH channel; Static doesn't have pH data
        lambda_ch = {
            1: args.lambda_ch1,
            2: args.lambda_ch2,
            3: args.lambda_ch3,
            5: args.lambda_ch5,
        }
        # Auto-tune: Dysbiotic HOBIC benefits from higher viability weight
        # (Pg viability drops significantly, key diagnostic signal)
        if args.condition == "Dysbiotic" and args.cultivation == "HOBIC":
            if args.lambda_ch3 == 2.0:  # only if user didn't override
                lambda_ch[3] = 3.0
                logger.info("Auto-tuned: lambda_ch3=3.0 for DH (Pg viability signal)")
        # Commensal conditions: rare Pg/Fn have minimal viability signal,
        # reduce Ch3 weight to avoid noise amplification
        if args.condition == "Commensal":
            if args.lambda_ch3 == 2.0:  # only if user didn't override
                lambda_ch[3] = 1.5
                logger.info(
                    f"Auto-tuned: lambda_ch3=1.5 for {args.condition} (less Pg/Fn viability info)"
                )
        mc_kwargs = {
            "data_total": mc_data.get("data_total"),
            "sigma_obs_total": mc_data.get("sigma_obs_total"),
            "data_viability": mc_data.get("data_viability"),
            "sigma_obs_viability": mc_data.get("sigma_obs_viability", 0.10),
            "lambda_ch": lambda_ch,
        }
        # pH channel (HOBIC only)
        if mc_data.get("data_pH") is not None:
            t_pH_days = mc_data["t_pH_days"]
            _, idx_pH = convert_days_to_model_time(t_pH_days, args.dt, args.n_steps, day_scale=None)
            idx_pH = np.clip(idx_pH, 0, args.n_steps)
            mc_kwargs["data_pH"] = mc_data["data_pH"]
            mc_kwargs["idx_pH"] = idx_pH
            mc_kwargs["sigma_obs_pH"] = mc_data.get("sigma_obs_pH", 0.15)
        logger.info(
            f"Multi-channel: total={'yes' if mc_data.get('data_total') is not None else 'no'}, "
            f"viability={'yes' if mc_data.get('data_viability') is not None else 'no'}, "
            f"pH={'yes' if mc_data.get('data_pH') is not None else 'no'}"
        )

    log_likelihood = make_log_likelihood_jax_ode(
        data=data,
        t_days=t_days,
        idx_sparse=idx_sparse,
        sigma_obs=sigma_obs,
        phi_init=phi_init,
        dt=args.dt,
        n_steps=args.n_steps,
        K_hill=args.K_hill,
        n_hill=args.n_hill,
        lambda_pg=args.lambda_pg,
        lambda_late=args.lambda_late,
        use_student_t=args.use_student_t,
        student_t_nu=args.student_t_nu,
        lambda_rare=args.lambda_rare,
        **mc_kwargs,
    )

    prior_bounds = load_prior_bounds(args.condition, args.cultivation)
    prior_bounds = np.array(prior_bounds, dtype=np.float64)  # float64 for consistency

    logger.info("JIT warmup...")
    _ = jax.jit(log_likelihood)(jnp.zeros(20, dtype=jnp.float64))
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
        use_de_mc=args.use_de_mc,
        de_mc_gamma=args.de_mc_gamma,
        resample_method=args.resample_method,
        use_flow=args.use_flow,
        flow_n_layers=args.flow_n_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_n_epochs=args.flow_n_epochs,
        flow_lr=args.flow_lr,
        flow_start_beta=args.flow_start_beta,
        flow_mix_ratio=args.flow_mix_ratio,
        waste_free=args.waste_free,
    )

    theta_MAP = result["theta_MAP"]
    log_evidence = result.get("log_evidence", None)
    logger.info(
        f"Done: {result['n_stages']} stages, "
        f"total_time={result['total_time']:.1f}s, "
        f"accept={np.mean(result['accept_rates']):.2f}, "
        f"max logL={result['log_likelihoods'].max():.1f}"
        + (f", log_evidence={log_evidence:.2f}" if log_evidence is not None else "")
    )

    # Compute RMSE for MAP estimate (multichannel-aware)
    theta_MAP_jax = jnp.array(theta_MAP, dtype=jnp.float64)
    mc_rmse = {}
    if args.multichannel:
        from hamilton_ode_jax import simulate_0d_full

        g_traj_map = simulate_0d_full(
            theta_MAP_jax,
            n_steps=args.n_steps,
            dt=args.dt,
            phi_init=jnp.array(phi_init, dtype=jnp.float64),
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )
        phi_traj_map = g_traj_map[:, 0:5]
    else:
        phi_traj_map = simulate_0d(
            theta_MAP_jax,
            n_steps=args.n_steps,
            dt=args.dt,
            phi_init=jnp.array(phi_init, dtype=jnp.float64),
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )
        g_traj_map = None
    phi_pred_map = np.array(phi_traj_map[idx_sparse, :])
    phi_pred_map = np.clip(phi_pred_map, 1e-10, 1.0 - 1e-10)
    phi_pred_map = phi_pred_map / phi_pred_map.sum(axis=1, keepdims=True)
    rmse = np.sqrt(np.mean((data - phi_pred_map) ** 2))
    mae = np.mean(np.abs(data - phi_pred_map))
    r2_vals = []
    for sp in range(data.shape[1]):
        ss_res = np.sum((data[:, sp] - phi_pred_map[:, sp]) ** 2)
        ss_tot = np.sum((data[:, sp] - np.mean(data[:, sp])) ** 2)
        r2_vals.append(1.0 - ss_res / max(ss_tot, 1e-12))
    mc_rmse["ch1_species"] = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": [float(v) for v in r2_vals],
    }
    logger.info(
        f"Ch1 (species): RMSE={rmse:.4f}, MAE={mae:.4f}, R²={np.mean(r2_vals):.3f} "
        f"(per-sp: {[f'{v:.3f}' for v in r2_vals]})"
    )
    # Ch2: total species (φ without viability weighting)
    if args.multichannel and mc_kwargs.get("data_total") is not None and g_traj_map is not None:
        phi_at_obs = np.array(g_traj_map[idx_sparse, 0:5])
        phi_total_pred = phi_at_obs / np.maximum(phi_at_obs.sum(axis=1, keepdims=True), 1e-12)
        data_total = mc_kwargs["data_total"]
        rmse_total = np.sqrt(np.mean((data_total - phi_total_pred) ** 2))
        mc_rmse["ch2_total"] = {"rmse": float(rmse_total)}
        logger.info(f"Ch2 (total species): RMSE={rmse_total:.4f}")
    # Ch3: viability
    if args.multichannel and mc_kwargs.get("data_viability") is not None and g_traj_map is not None:
        g_at_obs = np.array(g_traj_map[idx_sparse, :])
        phi_at = g_at_obs[:, 0:5]
        psi_at = g_at_obs[:, 6:11]
        viab_pred = np.sum(phi_at * psi_at, axis=1) / np.maximum(np.sum(phi_at, axis=1), 1e-12)
        data_viab = mc_kwargs["data_viability"]
        rmse_viab = np.sqrt(np.mean((data_viab - viab_pred) ** 2))
        r2_viab = 1.0 - np.sum((data_viab - viab_pred) ** 2) / max(
            np.sum((data_viab - np.mean(data_viab)) ** 2), 1e-12
        )
        mc_rmse["ch3_viability"] = {"rmse": float(rmse_viab), "r2": float(r2_viab)}
        logger.info(f"Ch3 (viability): RMSE={rmse_viab:.4f}, R²={r2_viab:.3f}")
    # Ch5: pH
    if args.multichannel and mc_kwargs.get("data_pH") is not None and g_traj_map is not None:
        idx_ph = mc_kwargs["idx_pH"]
        g_pH = np.array(g_traj_map[idx_ph, :])
        phi_pH = g_pH[:, 0:5]
        psi_pH = g_pH[:, 6:11]
        phibar = phi_pH * psi_pH
        phibar_sum = np.maximum(np.sum(phibar, axis=1), 1e-12)
        phi_So = phibar[:, 0] / phibar_sum
        phi_Vd = phibar[:, 2] / phibar_sum
        pH_pred = 7.5 - 0.74 * phi_So - 0.48 * phi_Vd
        data_pH = mc_kwargs["data_pH"]
        rmse_pH = np.sqrt(np.mean((data_pH - pH_pred) ** 2))
        r2_pH = 1.0 - np.sum((data_pH - pH_pred) ** 2) / max(
            np.sum((data_pH - np.mean(data_pH)) ** 2), 1e-12
        )
        mc_rmse["ch5_pH"] = {"rmse": float(rmse_pH), "r2": float(r2_pH)}
        logger.info(f"Ch5 (pH): RMSE={rmse_pH:.4f}, R²={r2_pH:.3f}")

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = MAIN_DIR / "_runs" / f"jax_ode_nuts_{args.condition}_{args.cultivation}_{ts}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "samples.npy", result["samples"])
    np.save(out_dir / "logL.npy", result["log_likelihoods"])
    with open(out_dir / "theta_MAP.json", "w") as f:
        json.dump({str(i): float(v) for i, v in enumerate(theta_MAP)}, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "cultivation": args.cultivation,
                "n_particles": args.n_particles,
                "n_mutation_steps": args.n_mutation_steps,
                "mutation": args.mutation,
                "sigma_obs": (
                    float(np.mean(sigma_obs)) if hasattr(sigma_obs, "__len__") else float(sigma_obs)
                ),
                "device": str(jax.devices()[0]),
                "multichannel": args.multichannel,
                "use_de_mc": args.use_de_mc,
                "use_student_t": args.use_student_t,
                "student_t_nu": args.student_t_nu,
                "resample_method": args.resample_method,
                "n_hill": args.n_hill,
                "K_hill": args.K_hill,
                "replicate_sigma": args.replicate_sigma,
                "max_info_ratio": args.max_info_ratio,
                "lambda_rare": args.lambda_rare,
                "lambda_ch": {
                    "1": args.lambda_ch1,
                    "2": args.lambda_ch2,
                    "3": args.lambda_ch3,
                    "5": args.lambda_ch5,
                },
                "n_stages": result["n_stages"],
                "total_time_s": result["total_time"],
                "mean_accept": float(np.mean(result["accept_rates"])),
                "max_logL": float(result["log_likelihoods"].max()),
                "log_evidence": float(log_evidence) if log_evidence is not None else None,
                "cv_weights_final": (
                    float(result["cv_weights"][-1]) if result.get("cv_weights") else None
                ),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_mean": float(np.mean(r2_vals)),
                "r2_per_species": [float(v) for v in r2_vals],
                "multichannel_rmse": mc_rmse,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
