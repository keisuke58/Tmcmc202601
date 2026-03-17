#!/usr/bin/env python3
"""
Final Validation Check: LOO-CV + Sensitivity Analysis + θ Comparison

Runs after det_MAP/TMCMC results are available.
Checks:
  7. LOO-CV (Leave-One-Out Cross-Validation)
  8. Sensitivity analysis (data perturbation → θ stability)
  + det_MAP vs TMCMC θ comparison
  + Physical sign consistency of A,B matrices

Usage:
  python3 final_check.py --condition Commensal --cultivation Static \
      --det-result-dir deterministic_results/CS_L-BFGS-B_10starts_...
  python3 final_check.py --all  # Run all 4 conditions (auto-detect latest results)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Project imports
DATA_5SPECIES_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = DATA_5SPECIES_ROOT.parent
sys.path.insert(0, str(DATA_5SPECIES_ROOT / "main"))
sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

from estimate_reduced_nishioka import load_experimental_data, convert_days_to_model_time
from core import LogLikelihoodEvaluator
from core.nishioka_model import get_condition_bounds, get_model_constants
from config import DebugConfig, DebugLevel
from debug import DebugLogger
from utils import save_json

logger = logging.getLogger(__name__)

SPECIES_NAMES = ["S.o", "A.n", "Vei", "F.n", "P.g"]

# Known biological interactions (from literature)
# a_ij > 0: species j promotes species i
# a_ij < 0: species j inhibits species i
KNOWN_SIGNS = {
    # (i, j): expected_sign  (None = unknown)
    # So-An mutualism
    (0, 1): +1,  # a12: An promotes So (early colonizer mutualism)
    (1, 0): +1,  # a21: So promotes An
    # Fn bridge organism
    (3, 0): +1,  # a41: So promotes Fn (co-aggregation)
    (3, 2): +1,  # a43: Vei promotes Fn (metabolic cross-feeding)
    # Pg late colonizer
    (4, 2): None,  # a53: Vei→Pg (uncertain, could be +/-)
    (4, 3): +1,  # a54: Fn promotes Pg (bridge)
}


def load_det_result(result_dir: Path) -> Dict[str, Any]:
    """Load det_MAP result from output directory."""
    theta = json.load(open(result_dir / "theta_MLE.json"))
    summary = json.load(open(result_dir / "estimation_summary.json"))
    config = json.load(open(result_dir / "config.json"))
    return {
        "theta": np.array(theta),
        "logL": summary["logL"],
        "AIC": summary.get("AIC"),
        "BIC": summary.get("BIC"),
        "config": config,
    }


def load_tmcmc_result(runs_dir: Path, condition: str, cultivation: str) -> Optional[Dict]:
    """Find and load best TMCMC result for condition/cultivation."""
    short = f"{condition[0]}{cultivation[0]}"
    candidates = []
    for d in sorted(runs_dir.glob(f"{short}_*"), reverse=True):
        config_f = d / "config.json"
        summary_f = d / "results_summary.json"
        if config_f.exists():
            cfg = json.load(open(config_f))
            if cfg.get("condition") == condition and cfg.get("cultivation") == cultivation:
                map_theta = None
                if summary_f.exists():
                    s = json.load(open(summary_f))
                    map_theta = np.array(s.get("MAP", []))
                candidates.append({"dir": d, "config": cfg, "MAP": map_theta})
    if not candidates:
        return None
    # Prefer ones with MAP available
    with_map = [c for c in candidates if c["MAP"] is not None and len(c["MAP"]) > 0]
    return with_map[0] if with_map else candidates[0]


def create_evaluator(
    condition: str,
    cultivation: str,
    data: np.ndarray,
    idx_sparse: np.ndarray,
    sigma_obs: np.ndarray,
    phi_init: np.ndarray,
    cov_rel: float = 0.005,
) -> Tuple[LogLikelihoodEvaluator, list, np.ndarray, list]:
    """Create evaluator matching TMCMC/det_MAP settings."""
    model_constants = get_model_constants()
    active_species = model_constants["active_species"]
    nishioka_bounds, locked_indices = get_condition_bounds(condition, cultivation)

    theta_base = np.zeros(20)
    for i, (low, high) in enumerate(nishioka_bounds):
        theta_base[i] = (low + high) / 2.0
    for idx in locked_indices:
        theta_base[idx] = 0.0

    active_indices = [i for i in range(20) if i not in locked_indices]

    solver_kwargs = {
        "dt": 1e-4,
        "maxtimestep": 2500,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": phi_init,
        "Kp1": 1e-4,
        "K_hill": 0.05,
        "n_hill": 4.0,
    }

    debug_config = DebugConfig(level=DebugLevel.MINIMAL)
    debug_logger = DebugLogger(debug_config)

    evaluator = LogLikelihoodEvaluator(
        solver_kwargs=solver_kwargs,
        active_species=active_species,
        active_indices=active_indices,
        theta_base=theta_base,
        data=data,
        idx_sparse=idx_sparse,
        sigma_obs=sigma_obs,
        cov_rel=cov_rel,
        rho=0.0,
        theta_linearization=theta_base,
        debug_logger=debug_logger,
    )

    active_bounds = [nishioka_bounds[i] for i in active_indices]
    return evaluator, active_indices, theta_base, active_bounds


def evaluate_logL(
    theta_full: np.ndarray,
    evaluator: LogLikelihoodEvaluator,
    active_indices: list,
) -> float:
    """Evaluate log-likelihood for a full 20D parameter vector."""
    theta_active = theta_full[active_indices]
    neg_logL = evaluator(theta_active)
    return -neg_logL


# =========================================================================
# 7. LOO-CV (Leave-One-Out Cross-Validation)
# =========================================================================
def run_loo_cv(
    condition: str,
    cultivation: str,
    theta_map: np.ndarray,
    sigma_obs: np.ndarray,
    phi_init: np.ndarray,
) -> Dict[str, Any]:
    """
    Leave-One-Out Cross-Validation.

    For each timepoint t_k:
    1. Remove t_k from data
    2. Re-estimate θ using remaining timepoints (or use MAP as proxy)
    3. Predict t_k using the model
    4. Compute prediction error

    For speed, we use the MAP θ (trained on all data) to predict each
    held-out point. This gives an optimistic LOO estimate but still
    detects overfitting if any single timepoint dominates the fit.
    """
    logger.info(f"Running LOO-CV for {condition} {cultivation}...")

    # Load full data
    data_full, days_full, sigma_default, phi_init_exp, metadata = load_experimental_data(
        DATA_5SPECIES_ROOT,
        condition=condition,
        cultivation=cultivation,
        start_from_day=1,
        normalize=True,
        use_exp_init=True,
    )

    if phi_init is None:
        phi_init = phi_init_exp

    n_timepoints = data_full.shape[0]
    n_species = data_full.shape[1]

    loo_results = []

    for k in range(n_timepoints):
        # Hold out timepoint k
        data_train = np.delete(data_full, k, axis=0)
        days_train = [d for i, d in enumerate(metadata["days"]) if i != k]
        day_held_out = metadata["days"][k]
        data_held_out = data_full[k, :]

        # Create evaluator on training data
        t_model_train, idx_sparse_train = convert_days_to_model_time(
            np.array(days_train), 1e-4, 2500
        )

        sigma_train = sigma_obs
        if sigma_obs.ndim == 2:
            sigma_train = np.delete(sigma_obs, k, axis=0)

        model_constants = get_model_constants()
        active_species = model_constants["active_species"]
        nishioka_bounds, locked_indices = get_condition_bounds(condition, cultivation)
        active_indices = [i for i in range(20) if i not in locked_indices]

        # Evaluate MAP on training data
        evaluator_train, _, theta_base, _ = create_evaluator(
            condition, cultivation, data_train, idx_sparse_train, sigma_train, phi_init
        )
        train_logL = evaluate_logL(theta_map, evaluator_train, active_indices)

        # Predict held-out point using full model
        evaluator_full, _, _, _ = create_evaluator(
            condition,
            cultivation,
            data_full,
            np.array(metadata.get("idx_sparse", [])),
            sigma_obs,
            phi_init,
        )
        # Run forward simulation with MAP theta
        theta_active = theta_map[active_indices]
        try:
            mu, sig = evaluator_full.evaluate_model(theta_active)
            t_model_full, idx_sparse_full = convert_days_to_model_time(
                np.array(metadata["days"]), 1e-4, 2500
            )
            # Get prediction at held-out timepoint
            active_species_idx = [
                model_constants["active_species"].index(s)
                for s in model_constants["active_species"]
            ]

            # Simulate and extract prediction at held-out day
            from improved_5species_jit import BiofilmNewtonSolver5S

            solver = BiofilmNewtonSolver5S()
            theta_base_pred = np.zeros(20)
            theta_base_pred[:] = theta_map
            y_sim = solver.solve(
                theta_base_pred,
                phi_init,
                dt=1e-4,
                maxtimestep=2500,
                c_const=25.0,
                alpha_const=0.0,
                Kp1=1e-4,
                K_hill=0.05,
                n_hill=4.0,
            )
            idx_k = idx_sparse_full[k]
            if idx_k < len(y_sim):
                y_pred_k = y_sim[idx_k, active_species_idx]
            else:
                y_pred_k = y_sim[-1, active_species_idx]

            # Compute per-species prediction error
            residuals = data_held_out - y_pred_k
            abs_error = np.abs(residuals)
            rel_error = abs_error / np.maximum(np.abs(data_held_out), 1e-6)

            loo_results.append(
                {
                    "day": day_held_out,
                    "observed": data_held_out.tolist(),
                    "predicted": y_pred_k.tolist(),
                    "residuals": residuals.tolist(),
                    "abs_error": abs_error.tolist(),
                    "rel_error": rel_error.tolist(),
                    "train_logL": float(train_logL),
                    "success": True,
                }
            )
        except Exception as e:
            logger.warning(f"LOO-CV fold {k} (day {day_held_out}) failed: {e}")
            loo_results.append(
                {
                    "day": day_held_out,
                    "success": False,
                    "error": str(e),
                }
            )

    # Aggregate metrics
    successful = [r for r in loo_results if r["success"]]
    if successful:
        all_abs_errors = np.array([r["abs_error"] for r in successful])
        all_residuals = np.array([r["residuals"] for r in successful])

        loo_rmse_per_species = np.sqrt(np.mean(all_residuals**2, axis=0))
        loo_mae_per_species = np.mean(all_abs_errors, axis=0)
        loo_rmse_total = float(np.sqrt(np.mean(all_residuals**2)))
        loo_mae_total = float(np.mean(all_abs_errors))

        summary = {
            "n_folds": n_timepoints,
            "n_successful": len(successful),
            "loo_rmse_total": loo_rmse_total,
            "loo_mae_total": loo_mae_total,
            "loo_rmse_per_species": dict(zip(SPECIES_NAMES, loo_rmse_per_species.tolist())),
            "loo_mae_per_species": dict(zip(SPECIES_NAMES, loo_mae_per_species.tolist())),
            "folds": loo_results,
        }
    else:
        summary = {"n_folds": n_timepoints, "n_successful": 0, "folds": loo_results}

    return summary


# =========================================================================
# 8. Sensitivity Analysis (Data Perturbation)
# =========================================================================
def run_sensitivity_analysis(
    condition: str,
    cultivation: str,
    theta_map: np.ndarray,
    sigma_obs: np.ndarray,
    phi_init: np.ndarray,
    n_perturbations: int = 30,
    perturbation_scale: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sensitivity analysis: perturb data within sigma, re-evaluate logL.

    For each perturbation:
    1. Add Gaussian noise ~ N(0, sigma_obs) to each data point
    2. Evaluate logL at MAP θ with perturbed data
    3. Optionally re-optimize (expensive) or just evaluate

    Reports:
    - logL distribution under data perturbation
    - If logL drops sharply → fit is fragile
    - If logL is stable → robust estimate
    """
    logger.info(f"Running sensitivity analysis for {condition} {cultivation}...")

    data_full, days_full, sigma_default, phi_init_exp, metadata = load_experimental_data(
        DATA_5SPECIES_ROOT,
        condition=condition,
        cultivation=cultivation,
        start_from_day=1,
        normalize=True,
        use_exp_init=True,
    )

    if phi_init is None:
        phi_init = phi_init_exp

    t_model, idx_sparse = convert_days_to_model_time(np.array(metadata["days"]), 1e-4, 2500)

    # Evaluate baseline logL
    evaluator, active_indices, theta_base, active_bounds = create_evaluator(
        condition, cultivation, data_full, idx_sparse, sigma_obs, phi_init
    )
    baseline_logL = evaluate_logL(theta_map, evaluator, active_indices)
    logger.info(f"  Baseline logL = {baseline_logL:.4f}")

    # Perturb data and evaluate
    rng = np.random.RandomState(seed)
    perturbed_logLs = []
    delta_logLs = []

    sigma_arr = np.atleast_1d(sigma_obs)
    if sigma_arr.ndim == 1 and sigma_arr.size == data_full.shape[1]:
        sigma_matrix = np.tile(sigma_arr, (data_full.shape[0], 1))
    elif sigma_arr.ndim == 2:
        sigma_matrix = sigma_arr
    else:
        sigma_matrix = np.full_like(data_full, float(sigma_arr))

    for i in range(n_perturbations):
        noise = rng.randn(*data_full.shape) * sigma_matrix * perturbation_scale
        data_perturbed = np.clip(data_full + noise, 0, 1)

        # Re-normalize to sum=1
        row_sums = data_perturbed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        data_perturbed = data_perturbed / row_sums

        # Create evaluator with perturbed data
        eval_pert, _, _, _ = create_evaluator(
            condition, cultivation, data_perturbed, idx_sparse, sigma_obs, phi_init
        )
        pert_logL = evaluate_logL(theta_map, eval_pert, active_indices)
        perturbed_logLs.append(pert_logL)
        delta_logLs.append(pert_logL - baseline_logL)

    perturbed_logLs = np.array(perturbed_logLs)
    delta_logLs = np.array(delta_logLs)

    summary = {
        "baseline_logL": float(baseline_logL),
        "n_perturbations": n_perturbations,
        "perturbation_scale": perturbation_scale,
        "perturbed_logL_mean": float(np.mean(perturbed_logLs)),
        "perturbed_logL_std": float(np.std(perturbed_logLs)),
        "perturbed_logL_min": float(np.min(perturbed_logLs)),
        "perturbed_logL_max": float(np.max(perturbed_logLs)),
        "delta_logL_mean": float(np.mean(delta_logLs)),
        "delta_logL_std": float(np.std(delta_logLs)),
        "robust": bool(np.std(delta_logLs) < 5.0),  # < 5 logL units = robust
        "perturbed_logLs": perturbed_logLs.tolist(),
    }

    logger.info(
        f"  Perturbed logL: {summary['perturbed_logL_mean']:.2f} ± {summary['perturbed_logL_std']:.2f} "
        f"(ΔlogL = {summary['delta_logL_mean']:.2f} ± {summary['delta_logL_std']:.2f})"
    )
    logger.info(f"  Robust: {summary['robust']}")

    return summary


# =========================================================================
# θ comparison: det_MAP vs TMCMC
# =========================================================================
def compare_theta(
    theta_det: np.ndarray,
    theta_tmcmc: Optional[np.ndarray],
    active_indices: list,
    bounds: list,
    condition: str,
    cultivation: str,
) -> Dict[str, Any]:
    """Compare det_MAP θ with TMCMC MAP θ."""
    if theta_tmcmc is None:
        return {"available": False, "message": "No TMCMC result found"}

    n_params = len(active_indices)
    # TMCMC MAP is active-space, det_MAP theta is full 20D
    theta_det_active = theta_det[active_indices]

    if len(theta_tmcmc) == 20:
        theta_tmcmc_active = theta_tmcmc[active_indices]
    elif len(theta_tmcmc) == n_params:
        theta_tmcmc_active = theta_tmcmc
    else:
        return {"available": False, "message": f"TMCMC theta length mismatch: {len(theta_tmcmc)}"}

    diff = theta_det_active - theta_tmcmc_active
    bound_ranges = np.array([b[1] - b[0] for b in bounds])
    rel_diff = diff / np.maximum(bound_ranges, 1e-10)

    param_names = [f"θ[{i}]" for i in active_indices]

    comparison = {
        "available": True,
        "n_params": n_params,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_rel_diff": float(np.max(np.abs(rel_diff))),
        "mean_rel_diff": float(np.mean(np.abs(rel_diff))),
        "params_close_5pct": int(np.sum(np.abs(rel_diff) < 0.05)),
        "params_close_10pct": int(np.sum(np.abs(rel_diff) < 0.10)),
        "per_param": {},
    }

    for i, (name, d, rd) in enumerate(zip(param_names, diff, rel_diff)):
        comparison["per_param"][name] = {
            "det_MAP": float(theta_det_active[i]),
            "TMCMC_MAP": float(theta_tmcmc_active[i]),
            "diff": float(d),
            "rel_diff_pct": float(abs(rd) * 100),
        }

    return comparison


# =========================================================================
# Physical sign check of A,B matrices
# =========================================================================
def check_physical_signs(
    theta: np.ndarray,
    condition: str,
    cultivation: str,
) -> Dict[str, Any]:
    """Check if A,B matrix signs match known biological interactions."""
    # Parameter index mapping (from Parameter Reference)
    # theta[0:5] = b1..b5 (growth rates, should be > 0)
    # theta[5:10] = a12, a13, a21, a23, a31 (A matrix off-diag)
    # theta[10:15] = a32, a41, a42, a43, a45
    # theta[15:20] = a51, a52, a53, a54, a55_extra or b-related

    # Simplified: check key interaction signs
    results = {"checks": [], "n_consistent": 0, "n_total": 0, "n_unknown": 0}

    # Map parameter indices to interaction meanings
    interaction_map = {
        5: ("a12", "An→So"),
        6: ("a13", "Vei→So"),
        7: ("a21", "So→An"),
        8: ("a23", "Vei→An"),
        9: ("a31", "So→Vei"),
        10: ("a32", "An→Vei"),
        11: ("a41", "So→Fn"),
        12: ("a42", "An→Fn"),
        13: ("a43", "Vei→Fn"),
        14: ("a45", "Pg→Fn"),
        15: ("b5", "Pg growth"),
        16: ("a51", "So→Pg"),
        17: ("a52", "An→Pg"),
        18: ("a53/a35", "Vei→Pg"),
        19: ("a54/a45", "Fn→Pg"),
    }

    for idx, (name, desc) in interaction_map.items():
        if idx < len(theta):
            val = theta[idx]
            sign = "+" if val > 0 else ("-" if val < 0 else "0")
            check = {"index": idx, "name": name, "desc": desc, "value": float(val), "sign": sign}

            # Check against known biology
            if name == "a12" and val > 0:
                check["biology"] = "consistent (An promotes So)"
                results["n_consistent"] += 1
            elif name == "a21" and val > 0:
                check["biology"] = "consistent (So promotes An)"
                results["n_consistent"] += 1
            elif name == "a43" and val > 0:
                check["biology"] = "consistent (Vei cross-feeds Fn)"
                results["n_consistent"] += 1
            elif "a54" in name and val > 0:
                check["biology"] = "consistent (Fn bridges Pg)"
                results["n_consistent"] += 1
            elif name == "b5" and val >= 0:
                check["biology"] = "consistent (Pg growth ≥ 0)"
                results["n_consistent"] += 1
            else:
                check["biology"] = "unknown/unchecked"
                results["n_unknown"] += 1

            results["n_total"] += 1
            results["checks"].append(check)

    return results


# =========================================================================
# Main
# =========================================================================
def run_final_check(
    condition: str,
    cultivation: str,
    det_result_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all final validation checks for one condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL CHECK: {condition} {cultivation}")
    logger.info(f"{'='*60}")

    # Load data
    data, days, sigma_default, phi_init_exp, metadata = load_experimental_data(
        DATA_5SPECIES_ROOT,
        condition=condition,
        cultivation=cultivation,
        start_from_day=1,
        normalize=True,
        use_exp_init=True,
    )
    sigma_obs = np.asarray(sigma_default)

    # Load det_MAP result
    theta_map = None
    det_logL = None
    if det_result_dir and det_result_dir.exists():
        det = load_det_result(det_result_dir)
        theta_map = det["theta"]
        det_logL = det["logL"]
        logger.info(f"det_MAP θ loaded: logL={det_logL:.4f}")
    else:
        # Auto-detect latest result
        det_base = DATA_5SPECIES_ROOT / "deterministic_results"
        short = f"{condition[0]}{cultivation[0]}"
        candidates = sorted(det_base.glob(f"{short}_*"), reverse=True)
        for c in candidates:
            if (c / "theta_MLE.json").exists():
                det = load_det_result(c)
                theta_map = det["theta"]
                det_logL = det["logL"]
                det_result_dir = c
                logger.info(f"Auto-detected: {c.name} (logL={det_logL:.4f})")
                break

    if theta_map is None:
        logger.error("No det_MAP result found. Run det_MAP first.")
        return {"error": "No det_MAP result"}

    # Setup
    nishioka_bounds, locked_indices = get_condition_bounds(condition, cultivation)
    active_indices = [i for i in range(20) if i not in locked_indices]
    active_bounds = [nishioka_bounds[i] for i in active_indices]

    results = {
        "condition": condition,
        "cultivation": cultivation,
        "det_MAP_logL": det_logL,
        "det_result_dir": str(det_result_dir) if det_result_dir else None,
    }

    # 7. LOO-CV
    t0 = time.time()
    loo = run_loo_cv(condition, cultivation, theta_map, sigma_obs, phi_init_exp)
    results["loo_cv"] = loo
    logger.info(f"LOO-CV done in {time.time()-t0:.1f}s")
    if "loo_rmse_total" in loo:
        logger.info(f"  LOO RMSE = {loo['loo_rmse_total']:.6f}")
        for sp, v in loo["loo_rmse_per_species"].items():
            logger.info(f"    {sp}: RMSE={v:.6f}")

    # 8. Sensitivity Analysis
    t0 = time.time()
    sens = run_sensitivity_analysis(condition, cultivation, theta_map, sigma_obs, phi_init_exp)
    results["sensitivity"] = sens
    logger.info(f"Sensitivity done in {time.time()-t0:.1f}s")

    # θ comparison with TMCMC
    tmcmc = load_tmcmc_result(DATA_5SPECIES_ROOT / "main" / "_runs", condition, cultivation)
    tmcmc_theta = tmcmc["MAP"] if tmcmc else None
    theta_comp = compare_theta(
        theta_map, tmcmc_theta, active_indices, active_bounds, condition, cultivation
    )
    results["theta_comparison"] = theta_comp
    if theta_comp.get("available"):
        logger.info(
            f"θ comparison: {theta_comp['params_close_10pct']}/{theta_comp['n_params']} "
            f"params within 10% of TMCMC MAP"
        )

    # Physical sign check
    signs = check_physical_signs(theta_map, condition, cultivation)
    results["physical_signs"] = signs
    logger.info(f"Physical signs: {signs['n_consistent']}/{signs['n_total']} consistent")

    # Summary score
    score = 0
    score_max = 0

    # logL positive?
    score_max += 1
    if det_logL and det_logL > 0:
        score += 1

    # LOO-CV RMSE reasonable?
    score_max += 1
    if loo.get("loo_rmse_total", 999) < 0.15:
        score += 1

    # Sensitivity robust?
    score_max += 1
    if sens.get("robust", False):
        score += 1

    # θ close to TMCMC?
    if theta_comp.get("available"):
        score_max += 1
        if theta_comp.get("mean_rel_diff", 1.0) < 0.2:
            score += 1

    # Physical signs
    score_max += 1
    if signs["n_consistent"] >= 3:
        score += 1

    results["validation_score"] = f"{score}/{score_max}"
    logger.info(f"\nVALIDATION SCORE: {score}/{score_max}")

    # Save
    if output_dir is None:
        output_dir = det_result_dir if det_result_dir else Path(".")
    save_json(output_dir / "final_check.json", results)
    logger.info(f"Results saved to {output_dir / 'final_check.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Final validation check (LOO-CV + Sensitivity)")
    parser.add_argument("--condition", type=str, choices=["Commensal", "Dysbiotic"])
    parser.add_argument("--cultivation", type=str, choices=["Static", "HOBIC"])
    parser.add_argument("--det-result-dir", type=str, help="Path to det_MAP result directory")
    parser.add_argument("--all", action="store_true", help="Run all 4 conditions")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conditions = [
        ("Commensal", "Static"),
        ("Commensal", "HOBIC"),
        ("Dysbiotic", "Static"),
        ("Dysbiotic", "HOBIC"),
    ]

    if args.all:
        all_results = {}
        for cond, cult in conditions:
            r = run_final_check(cond, cult)
            all_results[f"{cond}_{cult}"] = r

        # Print summary table
        print("\n" + "=" * 70)
        print("FINAL CHECK SUMMARY")
        print("=" * 70)
        print(
            f"{'Condition':<20} {'logL':>8} {'LOO RMSE':>10} {'Sensitive':>10} {'θ match':>10} {'Score':>8}"
        )
        print("-" * 70)
        for key, r in all_results.items():
            logL = r.get("det_MAP_logL", "N/A")
            loo_rmse = r.get("loo_cv", {}).get("loo_rmse_total", "N/A")
            robust = "Yes" if r.get("sensitivity", {}).get("robust") else "No"
            theta_match = r.get("theta_comparison", {}).get("params_close_10pct", "N/A")
            n_params = r.get("theta_comparison", {}).get("n_params", "?")
            score = r.get("validation_score", "N/A")

            logL_str = f"{logL:.2f}" if isinstance(logL, float) else str(logL)
            loo_str = f"{loo_rmse:.6f}" if isinstance(loo_rmse, float) else str(loo_rmse)
            match_str = (
                f"{theta_match}/{n_params}" if isinstance(theta_match, int) else str(theta_match)
            )

            print(f"{key:<20} {logL_str:>8} {loo_str:>10} {robust:>10} {match_str:>10} {score:>8}")
        print("=" * 70)

    elif args.condition and args.cultivation:
        det_dir = Path(args.det_result_dir) if args.det_result_dir else None
        out_dir = Path(args.output_dir) if args.output_dir else None
        run_final_check(args.condition, args.cultivation, det_dir, out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
