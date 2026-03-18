#!/usr/bin/env python3
"""
Joint 4-Condition TMCMC Parameter Estimation.

Estimates a shared θ_joint (15 A_ij parameters) across all 4 experimental
conditions simultaneously. Each condition's evaluator uses only its
unlocked subset of θ_joint.

logL(θ_joint) = Σ_c w_c · logL_c(θ_c)

Usage:
    python estimate_joint_nishioka.py --n-particles 200 --n-stages 30

    # With multichannel (total species + viability + pH)
    python estimate_joint_nishioka.py --multichannel --n-particles 200

    # Custom condition weights
    python estimate_joint_nishioka.py --weight-cs 1.0 --weight-ch 1.0 --weight-ds 1.0 --weight-dh 1.0
"""

import argparse
import sys
import json
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Tmcmc202601
DATA_5SPECIES_ROOT = Path(__file__).parent.parent  # data_5species

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import setup_logging, DebugConfig, DebugLevel, PRIOR_BOUNDS_DEFAULT
from core import (
    LogLikelihoodEvaluator,
    run_multi_chain_TMCMC,
    compute_MAP_with_uncertainty,
    build_replicate_sigma,
)
from core.joint_evaluator import (
    JointLogLikelihoodEvaluator,
    ALL_AIJ_INDICES,
    PARAM_NAMES,
    get_condition_free_aij,
)
from debug import DebugLogger
from utils import save_json, save_npy

# These functions are in the main runner — import directly to avoid __init__.py issues
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "estimate_reduced_nishioka",
    str(Path(__file__).parent / "estimate_reduced_nishioka.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_mcmc_diagnostics = _mod.compute_mcmc_diagnostics
compute_credible_intervals = _mod.compute_credible_intervals
compute_model_evidence = _mod.compute_model_evidence
load_experimental_data = _mod.load_experimental_data
load_multichannel_data = _mod.load_multichannel_data
convert_days_to_model_time = _mod.convert_days_to_model_time

try:
    from data_5species.core.nishioka_model import get_condition_bounds, get_model_constants
except ImportError:
    from core.nishioka_model import get_condition_bounds, get_model_constants

import logging

logger = logging.getLogger(__name__)

# All 4 conditions
CONDITIONS = [
    ("Commensal", "Static"),
    ("Commensal", "HOBIC"),
    ("Dysbiotic", "Static"),
    ("Dysbiotic", "HOBIC"),
]

CONDITION_KEYS = [f"{c}_{v}" for c, v in CONDITIONS]


def load_condition_data(
    data_dir: Path,
    condition: str,
    cultivation: str,
    dt: float,
    maxtimestep: int,
    day_scale: float = None,
    use_exp_init: bool = True,
    multichannel: bool = False,
) -> Dict[str, Any]:
    """Load experimental data for one condition."""
    data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
        data_dir,
        condition,
        cultivation,
        start_from_day=1,
        normalize=False,
        use_exp_init=use_exp_init,
    )

    t_model, idx_sparse = convert_days_to_model_time(t_days, dt, maxtimestep, day_scale)
    metadata["t_model"] = t_model.tolist()
    metadata["idx_sparse"] = idx_sparse.tolist()

    result = {
        "data": data,
        "t_days": t_days,
        "idx_sparse": idx_sparse,
        "sigma_obs": sigma_obs_est,
        "phi_init": phi_init_exp if use_exp_init else None,
        "metadata": metadata,
    }

    if multichannel:
        mc_data = load_multichannel_data(
            data_dir=data_dir / "experiment_data",
            condition=condition,
            cultivation=cultivation,
            days_filter=metadata["days"],
            normalize=True,
        )
        # Convert pH time indices
        if mc_data.get("data_pH") is not None and mc_data.get("t_pH_days") is not None:
            t_pH_days = mc_data["t_pH_days"]
            t_max_model = maxtimestep * dt
            t_max_days = float(metadata["days"][-1])
            day_scale_pH = (t_max_model * 0.95) / t_max_days
            idx_pH = np.round(t_pH_days * day_scale_pH / dt).astype(int)
            idx_pH = np.clip(idx_pH, 0, maxtimestep - 1)
            mc_data["idx_pH"] = idx_pH
        result["multichannel"] = mc_data

    return result


def build_joint_prior_bounds(condition_bounds_map: Dict[str, Tuple]) -> List[Tuple[float, float]]:
    """
    Build joint prior bounds from per-condition bounds.

    For each A_ij parameter, take the union (widest) of all conditions'
    bounds where that parameter is free.
    """
    joint_bounds = []
    for aij_idx in ALL_AIJ_INDICES:
        lo_min, hi_max = np.inf, -np.inf
        any_free = False
        for cond_key, (bounds, locks) in condition_bounds_map.items():
            if aij_idx not in locks:
                lo, hi = bounds[aij_idx]
                lo_min = min(lo_min, lo)
                hi_max = max(hi_max, hi)
                any_free = True
        if any_free:
            joint_bounds.append((lo_min, hi_max))
        else:
            joint_bounds.append((0.0, 0.0))  # fully locked across all conditions
    return joint_bounds


def parse_args():
    parser = argparse.ArgumentParser(description="Joint 4-Condition TMCMC Parameter Estimation")

    # Data
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_5SPECIES_ROOT), help="Data directory"
    )
    parser.add_argument("--output-dir", type=str, default=None)

    # Model
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--maxtimestep", type=int, default=2500)
    parser.add_argument("--day-scale", type=float, default=None)
    parser.add_argument("--c-const", type=float, default=1.0)
    parser.add_argument("--alpha-const", type=float, default=0.0)
    parser.add_argument("--kp1", type=float, default=1.0)
    parser.add_argument("--K-hill", type=float, default=0.05)
    parser.add_argument("--n-hill", type=float, default=4.0)
    parser.add_argument("--phi-init", type=float, default=0.02)
    parser.add_argument("--use-exp-init", action="store_true", default=True)

    # TMCMC
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--n-stages", type=int, default=30)
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--n-mutation-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-threads",
        action="store_true",
        default=True,
        help="Use threads (Numba nogil=True enables real parallelism)",
    )
    parser.add_argument(
        "--no-parallel-cond",
        action="store_true",
        default=False,
        help="Disable parallel condition evaluation",
    )

    # Sigma
    parser.add_argument("--sigma-obs", type=float, default=None)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--replicate-sigma", action="store_true", default=False)

    # Multi-channel
    parser.add_argument("--multichannel", action="store_true", default=False)
    parser.add_argument("--lambda-ch1", type=float, default=1.0)
    parser.add_argument("--lambda-ch2", type=float, default=0.5)
    parser.add_argument("--lambda-ch3", type=float, default=2.0)
    parser.add_argument("--lambda-ch5", type=float, default=0.3)

    # Condition weights
    parser.add_argument("--weight-cs", type=float, default=1.0)
    parser.add_argument("--weight-ch", type=float, default=1.0)
    parser.add_argument("--weight-ds", type=float, default=1.0)
    parser.add_argument("--weight-dh", type=float, default=1.0)

    # DeepONet surrogate
    parser.add_argument(
        "--use-deeponet",
        action="store_true",
        default=False,
        help="Use DeepONet surrogate (~100x speedup)",
    )

    # Diagnostics
    parser.add_argument("--hdi-credibility", type=float, default=0.95)
    parser.add_argument(
        "--debug-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug_level)

    data_dir = Path(args.data_dir)

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = data_dir / "_runs" / f"joint_4cond_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("JOINT 4-CONDITION TMCMC ESTIMATION")
    logger.info("=" * 70)

    # ── Load data for all 4 conditions ──
    condition_data = {}
    for condition, cultivation in CONDITIONS:
        cond_key = f"{condition}_{cultivation}"
        logger.info(f"Loading data: {cond_key}")
        condition_data[cond_key] = load_condition_data(
            data_dir=data_dir,
            condition=condition,
            cultivation=cultivation,
            dt=args.dt,
            maxtimestep=args.maxtimestep,
            day_scale=args.day_scale,
            use_exp_init=args.use_exp_init,
            multichannel=args.multichannel,
        )

    # ── Get per-condition bounds and locks ──
    condition_bounds_map = {}
    for condition, cultivation in CONDITIONS:
        cond_key = f"{condition}_{cultivation}"
        bounds, locks = get_condition_bounds(condition, cultivation)
        condition_bounds_map[cond_key] = (bounds, locks)
        free_aij = get_condition_free_aij(cond_key)
        names = [PARAM_NAMES[i] for i in free_aij]
        logger.info(f"  {cond_key}: {len(free_aij)} free A_ij params: {names}")

    # ── Build joint prior bounds (15 A_ij) ──
    joint_prior_bounds = build_joint_prior_bounds(condition_bounds_map)
    joint_active_indices = list(range(len(ALL_AIJ_INDICES)))  # 0..14 in joint space
    n_joint = len(joint_active_indices)

    logger.info(f"Joint parameter space: {n_joint} A_ij parameters")
    for i, aij_idx in enumerate(ALL_AIJ_INDICES):
        lo, hi = joint_prior_bounds[i]
        logger.info(f"  [{i}] {PARAM_NAMES[aij_idx]}: [{lo:.2f}, {hi:.2f}]")

    # ── Model constants ──
    model_constants = get_model_constants()
    active_species = model_constants["active_species"]

    # ── Condition weights ──
    condition_weights = {
        "Commensal_Static": args.weight_cs,
        "Commensal_HOBIC": args.weight_ch,
        "Dysbiotic_Static": args.weight_ds,
        "Dysbiotic_HOBIC": args.weight_dh,
    }
    logger.info(f"Condition weights: {condition_weights}")

    # ── Channel weights ──
    lambda_ch = {
        1: args.lambda_ch1,
        2: args.lambda_ch2,
        3: args.lambda_ch3,
        5: args.lambda_ch5,
    }

    # ── Debug config ──
    debug_level_map = {
        "DEBUG": DebugLevel.VERBOSE,
        "INFO": DebugLevel.MINIMAL,
        "WARNING": DebugLevel.ERROR,
        "ERROR": DebugLevel.ERROR,
    }
    debug_config = DebugConfig(level=debug_level_map.get(args.debug_level, DebugLevel.MINIMAL))
    debug_logger = DebugLogger(debug_config)

    # ── DeepONet surrogate setup (optional) ──
    deeponet_surrogates = {}  # {cond_key: DeepONetSurrogate}
    if args.use_deeponet:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "deeponet"))
            from surrogate_tmcmc import DeepONetSurrogate

            deeponet_dir = PROJECT_ROOT / "deeponet"
            # Priority order per condition: 50k > v2 > original
            candidates = {
                "Commensal_Static": [
                    "checkpoints_CS_50k",
                    "checkpoints_CS_v2",
                    "checkpoints_Commensal_Static",
                ],
                "Commensal_HOBIC": [
                    "checkpoints_CH_50k",
                    "checkpoints_CH_v2",
                    "checkpoints_Commensal_HOBIC",
                ],
                "Dysbiotic_Static": ["checkpoints_DS_v2", "checkpoints_Dysbiotic_Static"],
                "Dysbiotic_HOBIC": [
                    "checkpoints_DH_50k_importance",
                    "checkpoints_Dysbiotic_HOBIC_50k",
                    "checkpoints_Dysbiotic_HOBIC",
                ],
            }
            for cond_key in CONDITION_KEYS:
                ckpt_name = None
                for name in candidates.get(cond_key, []):
                    if (deeponet_dir / name / "best.eqx").exists():
                        ckpt_name = name
                        break
                if ckpt_name is not None:
                    ckpt_path = deeponet_dir / ckpt_name
                    surrogate = DeepONetSurrogate(
                        str(ckpt_path / "best.eqx"),
                        str(ckpt_path / "norm_stats.npz"),
                    )
                    deeponet_surrogates[cond_key] = surrogate
                    logger.info(f"DeepONet loaded: {cond_key} <- {ckpt_name}")
                else:
                    logger.warning(f"No DeepONet checkpoint for {cond_key}, will use ODE solver")

            if deeponet_surrogates:
                # JAX objects can't be pickled → force threads
                args.use_threads = True
                logger.info(
                    f"DeepONet: {len(deeponet_surrogates)}/4 conditions loaded, threading enabled"
                )
        except Exception as e:
            logger.error(f"Failed to load DeepONet: {e}. Falling back to ODE solver.")
            deeponet_surrogates = {}

    # ── Build per-condition evaluators ──
    def make_joint_evaluator(theta_linearization=None):
        evaluators = {}
        for cond_key in CONDITION_KEYS:
            condition, cultivation = cond_key.split("_", 1)
            cd = condition_data[cond_key]
            bounds, locks = condition_bounds_map[cond_key]

            # Build condition theta_base (20-dim)
            theta_base_cond = np.zeros(20)
            for i, (lo, hi) in enumerate(bounds):
                theta_base_cond[i] = (lo + hi) / 2.0
            for idx in locks:
                theta_base_cond[idx] = 0.0

            active_indices_cond = [i for i in range(20) if i not in locks]

            phi_init = cd["phi_init"] if cd["phi_init"] is not None else args.phi_init

            solver_kwargs = {
                "dt": args.dt,
                "maxtimestep": args.maxtimestep,
                "c_const": args.c_const,
                "alpha_const": args.alpha_const,
                "phi_init": phi_init,
                "Kp1": args.kp1,
                "K_hill": args.K_hill,
                "n_hill": args.n_hill,
            }

            # Sigma
            sigma_obs = cd["sigma_obs"]
            if args.sigma_obs is not None:
                sigma_obs = args.sigma_obs * args.sigma_scale
            elif isinstance(sigma_obs, np.ndarray):
                sigma_obs = sigma_obs * args.sigma_scale
            else:
                sigma_obs = float(sigma_obs) * args.sigma_scale

            # Multi-channel data
            mc = cd.get("multichannel", {})

            # Convert theta_linearization from joint space (15) to full 20-dim
            if theta_linearization is not None:
                if len(theta_linearization) == n_joint:
                    # Joint 15-dim → expand to 20-dim
                    theta_lin = theta_base_cond.copy()
                    for j, aij_idx in enumerate(ALL_AIJ_INDICES):
                        theta_lin[aij_idx] = theta_linearization[j]
                elif len(theta_linearization) >= 20:
                    theta_lin = theta_linearization[:20]
                else:
                    theta_lin = theta_base_cond.copy()
            else:
                theta_lin = theta_base_cond.copy()

            # Use DeepONet if available for this condition
            if cond_key in deeponet_surrogates:
                from core.evaluator import DeepONetEvaluator

                # Map ODE idx_sparse (0..maxtimestep-1) to DeepONet 100-point grid (0..99)
                idx_sparse_don = np.round(
                    cd["idx_sparse"] * 99.0 / max(args.maxtimestep - 1, 1)
                ).astype(int)
                idx_sparse_don = np.clip(idx_sparse_don, 0, 99)
                evaluator = DeepONetEvaluator(
                    surrogate=deeponet_surrogates[cond_key],
                    active_species=active_species,
                    active_indices=active_indices_cond,
                    theta_base=theta_base_cond,
                    data=cd["data"],
                    idx_sparse=idx_sparse_don,
                    sigma_obs=sigma_obs,
                    rho=0.0,
                    weights=None,
                )
                logger.info(f"  {cond_key}: DeepONet evaluator (idx_sparse mapped to 0..99)")
            else:
                evaluator = LogLikelihoodEvaluator(
                    solver_kwargs=solver_kwargs,
                    active_species=active_species,
                    active_indices=active_indices_cond,
                    theta_base=theta_base_cond,
                    data=cd["data"],
                    idx_sparse=cd["idx_sparse"],
                    sigma_obs=sigma_obs,
                    cov_rel=0.0,
                    rho=0.0,
                    theta_linearization=theta_lin,
                    paper_mode=False,
                    debug_logger=debug_logger,
                    use_absolute_volume=False,
                    weights=None,
                    data_total=mc.get("data_total") if args.multichannel else None,
                    sigma_obs_total=mc.get("sigma_obs_total"),
                    data_viability=mc.get("data_viability") if args.multichannel else None,
                    sigma_obs_viability=mc.get("sigma_obs_viability", 0.10),
                    data_pH=mc.get("data_pH") if args.multichannel else None,
                    idx_pH=mc.get("idx_pH"),
                    sigma_obs_pH=mc.get("sigma_obs_pH", 0.15),
                    lambda_ch=lambda_ch,
                )
                logger.info(f"  {cond_key}: ODE evaluator")

            evaluators[cond_key] = evaluator

        # Wrap in JointLogLikelihoodEvaluator
        joint_eval = JointLogLikelihoodEvaluator(
            evaluators=evaluators,
            joint_active_indices=ALL_AIJ_INDICES,
            condition_weights=condition_weights,
            parallel=not args.no_parallel_cond,
        )

        return joint_eval

    # ── Build joint theta_base (15-dim for TMCMC sampling) ──
    # theta_base in joint space: midpoint of joint bounds
    theta_base_joint = np.zeros(n_joint)
    for i in range(n_joint):
        lo, hi = joint_prior_bounds[i]
        theta_base_joint[i] = (lo + hi) / 2.0

    # ── Prior bounds for TMCMC: need full-size (15) array ──
    # TMCMC expects prior_bounds indexed by active_indices
    # For joint model: active_indices = [0..14], prior_bounds = joint_prior_bounds
    prior_bounds_full = list(joint_prior_bounds)

    # ── Save config ──
    config = {
        "mode": "joint_4condition",
        "conditions": CONDITION_KEYS,
        "condition_weights": condition_weights,
        "n_joint_params": n_joint,
        "joint_param_names": [PARAM_NAMES[i] for i in ALL_AIJ_INDICES],
        "joint_prior_bounds": [(lo, hi) for lo, hi in joint_prior_bounds],
        "dt": args.dt,
        "maxtimestep": args.maxtimestep,
        "n_particles": args.n_particles,
        "n_stages": args.n_stages,
        "n_chains": args.n_chains,
        "multichannel": args.multichannel,
        "lambda_ch": lambda_ch,
        "seed": args.seed,
    }
    save_json(output_dir / "config.json", config)

    # ── Run TMCMC ──
    logger.info("Starting Joint TMCMC estimation...")
    logger.info(
        f"  n_particles={args.n_particles}, n_stages={args.n_stages}, n_chains={args.n_chains}"
    )
    start_time = time.time()

    mutation_kwargs = {}
    if args.n_mutation_steps is not None:
        mutation_kwargs["n_mutation_steps"] = args.n_mutation_steps

    chains, logL, MAP, converged, diag = run_multi_chain_TMCMC(
        model_tag="joint_4cond",
        make_evaluator=make_joint_evaluator,
        prior_bounds=prior_bounds_full,
        theta_base_full=theta_base_joint,
        active_indices=joint_active_indices,
        theta_linearization_init=theta_base_joint,
        n_particles=args.n_particles,
        n_stages=args.n_stages,
        n_chains=args.n_chains,
        min_delta_beta=0.02,
        max_delta_beta=0.5,
        logL_scale=1.0,
        seed=args.seed,
        n_jobs=args.n_jobs,
        use_threads=args.use_threads,
        debug_config=debug_config,
        **mutation_kwargs,
    )

    elapsed = time.time() - start_time
    logger.info(f"Joint TMCMC completed in {elapsed:.1f} seconds")

    # ── Process results ──
    samples = np.concatenate(chains, axis=0)
    logL_all = np.concatenate(logL, axis=0)

    results = compute_MAP_with_uncertainty(samples, logL_all)
    MAP_estimate = results["MAP"]
    mean_estimate = results["mean"]

    # Active param names in joint space
    joint_param_names = [PARAM_NAMES[i] for i in ALL_AIJ_INDICES]

    # ── Diagnostics ──
    mcmc_diagnostics = compute_mcmc_diagnostics(chains, logL, joint_param_names)
    credible_intervals = compute_credible_intervals(samples, args.hdi_credibility)

    logger.info(f"R-hat max: {mcmc_diagnostics['rhat_max']:.4f} (target < 1.1)")
    logger.info(f"ESS min: {mcmc_diagnostics['ess_min']:.1f} (target > 100)")

    # ── Model Evidence ──
    beta_schedule = None
    mean_logL_per_stage = None
    if diag and "beta_schedules" in diag:
        beta_schedule = diag["beta_schedules"][0] if diag["beta_schedules"] else None
    if diag and "mean_logL_per_stage" in diag:
        mean_logL_per_stage = (
            diag["mean_logL_per_stage"][0] if diag["mean_logL_per_stage"] else None
        )

    evidence_results = compute_model_evidence(
        chains=chains,
        logL=logL,
        prior_bounds=prior_bounds_full,
        active_indices=joint_active_indices,
        beta_schedule=beta_schedule,
        mean_logL_per_stage=mean_logL_per_stage,
    )
    logger.info(f"Model Evidence: log(p(D)) ≈ {evidence_results['log_evidence']:.2f}")

    # ── Per-condition logL at MAP ──
    joint_eval = make_joint_evaluator()
    _ = joint_eval(MAP_estimate)
    map_per_cond = joint_eval.get_MAP_per_condition()
    logger.info("Per-condition logL at MAP:")
    for cond_key, ll in map_per_cond.items():
        logger.info(f"  {cond_key}: {ll:.2f}")

    # ── Save results ──
    save_npy(output_dir / "samples.npy", samples)
    save_npy(output_dir / "logL.npy", logL_all)

    # Per-chain samples
    chains_dir = output_dir / "chains"
    chains_dir.mkdir(exist_ok=True)
    for ci, chain_data in enumerate(chains):
        save_npy(chains_dir / f"chain_{ci}.npy", np.asarray(chain_data))

    # MAP theta (joint → full 20-dim for each condition)
    theta_joint_MAP = MAP_estimate if len(MAP_estimate) == n_joint else MAP_estimate[:n_joint]

    # Expand to per-condition full theta
    theta_full_per_cond = {}
    for cond_key in CONDITION_KEYS:
        bounds, locks = condition_bounds_map[cond_key]
        theta_full = np.zeros(20)
        active_cond = [i for i in range(20) if i not in locks]
        # Map joint → condition
        for j, aij_idx in enumerate(ALL_AIJ_INDICES):
            if aij_idx in active_cond:
                theta_full[aij_idx] = theta_joint_MAP[j]
        theta_full_per_cond[cond_key] = theta_full.tolist()

    save_json(
        output_dir / "theta_MAP.json",
        {
            "theta_joint": theta_joint_MAP.tolist(),
            "joint_param_names": joint_param_names,
            "theta_full_per_condition": theta_full_per_cond,
            "logL_per_condition": {k: float(v) for k, v in map_per_cond.items()},
        },
    )

    # Diagnostics
    save_json(
        output_dir / "mcmc_diagnostics.json",
        {
            "rhat": mcmc_diagnostics["rhat"].tolist(),
            "ess": mcmc_diagnostics["ess"].tolist(),
            "rhat_max": mcmc_diagnostics["rhat_max"],
            "ess_min": mcmc_diagnostics["ess_min"],
            "converged": mcmc_diagnostics["converged"],
            "per_parameter": mcmc_diagnostics["per_parameter"],
        },
    )

    # Credible intervals
    save_json(
        output_dir / "credible_intervals.json",
        {
            "credibility": credible_intervals["credibility"],
            "hdi_lower": credible_intervals["hdi_lower"].tolist(),
            "hdi_upper": credible_intervals["hdi_upper"].tolist(),
            "et_lower": credible_intervals["et_lower"].tolist(),
            "et_upper": credible_intervals["et_upper"].tolist(),
            "median": credible_intervals["median"].tolist(),
            "param_names": joint_param_names,
        },
    )

    # Evidence
    save_json(output_dir / "evidence.json", evidence_results)

    # Prior bounds
    save_json(
        output_dir / "prior_bounds.json",
        {
            "joint_active_indices": ALL_AIJ_INDICES,
            "param_names": joint_param_names,
            "bounds": [
                {
                    "name": joint_param_names[i],
                    "index": ALL_AIJ_INDICES[i],
                    "min": float(joint_prior_bounds[i][0]),
                    "max": float(joint_prior_bounds[i][1]),
                }
                for i in range(n_joint)
            ],
        },
    )

    # ── Summary ──
    logger.info("=" * 70)
    logger.info("JOINT ESTIMATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info(f"Samples: {samples.shape}")
    logger.info(f"Converged: {mcmc_diagnostics['converged']}")

    logger.info("\nMAP estimates (15 A_ij):")
    for i, aij_idx in enumerate(ALL_AIJ_INDICES):
        lo, hi = joint_prior_bounds[i]
        ci_lo = credible_intervals["hdi_lower"][i]
        ci_hi = credible_intervals["hdi_upper"][i]
        logger.info(
            f"  {PARAM_NAMES[aij_idx]:5s} = {theta_joint_MAP[i]:+.4f} "
            f"[{ci_lo:+.4f}, {ci_hi:+.4f}] (prior [{lo:.2f}, {hi:.2f}])"
        )

    logger.info(f"\nTotal logL(MAP) = {logL_all.max():.2f}")
    for cond_key, ll in map_per_cond.items():
        logger.info(f"  {cond_key}: {ll:.2f}")

    logger.info(f"Evidence: log(p(D)) = {evidence_results['log_evidence']:.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
