"""
Helper functions for visualization.

Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys

# Try to import tmcmc, add to path if needed
try:
    from tmcmc.utils.io import write_csv
except ImportError:
    # Add project root to sys.path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from tmcmc.utils.io import write_csv
    except ImportError:
        # Try local utils relative to visualization
        utils_dir = current_dir.parent / "utils"
        if str(utils_dir.parent) not in sys.path:
            sys.path.insert(0, str(utils_dir.parent))
        from utils.io import write_csv

logger = logging.getLogger(__name__)


def compute_phibar(x0: np.ndarray, active_species: List[int]) -> np.ndarray:
    """
    Compute observable φ̄ = φ * ψ (living bacteria volume fraction).

    Parameters
    ----------
    x0 : np.ndarray
        State vector with shape (n_time, n_state)
        State vector: [phi_0..phi_{N-1}, phi0, psi_0..psi_{N-1}, gamma]
    active_species : List[int]
        Active species indices

    Returns
    -------
    np.ndarray
        φ̄ values with shape (n_time, n_species)
    """
    n_t = x0.shape[0]
    n_sp = len(active_species)
    phibar = np.zeros((n_t, n_sp))

    n_state = x0.shape[1]
    n_total_species = (n_state - 2) // 2
    psi_offset = n_total_species + 1

    for i, sp in enumerate(active_species):
        phibar[:, i] = x0[:, sp] * x0[:, psi_offset + sp]

    return phibar


def compute_fit_metrics(
    t_arr: np.ndarray,
    x0: np.ndarray,
    active_species: List[int],
    data: np.ndarray,
    idx_sparse: np.ndarray,
    weights: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Compute simple misfit metrics between model observable φ̄ and observed data.

    Parameters
    ----------
    t_arr : np.ndarray
        Time array
    x0 : np.ndarray
        State vector
    active_species : List[int]
        Active species indices
    data : np.ndarray
        Observed data at sparse observation times: shape (n_obs, n_species)
    idx_sparse : np.ndarray
        Sparse observation indices
    weights : np.ndarray, optional
        Per-(time, species) weights, shape (n_obs, n_species).
        When provided, a weighted RMSE is computed in addition to the
        unweighted metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing fit metrics (RMSE, MAE, etc.)

    Notes
    -----
    - data is expected to be φ̄ at sparse observation times: shape (n_obs, n_species)
    - model observable uses φ̄_i = φ_i * ψ_i, consistent with likelihood definition
    """
    phibar = compute_phibar(x0, active_species)
    pred = phibar[idx_sparse]
    resid = pred - data
    rmse_per = np.sqrt(np.mean(resid**2, axis=0))
    mae_per = np.mean(np.abs(resid), axis=0)
    metrics: Dict[str, Any] = {
        "n_obs": int(data.shape[0]),
        "n_species": int(data.shape[1]),
        "rmse_per_species": rmse_per,
        "mae_per_species": mae_per,
        "rmse_total": float(np.sqrt(np.mean(resid**2))),
        "mae_total": float(np.mean(np.abs(resid))),
        "max_abs": float(np.max(np.abs(resid))),
    }
    if weights is not None:
        # Weighted RMSE: sqrt( sum(w * r^2) / sum(w) )
        w_rmse = float(np.sqrt(np.sum(weights * resid**2) / np.sum(weights)))
        metrics["weighted_rmse_total"] = w_rmse
        # P.g. (species 4) late-stage RMSE (last n_late obs)
        if data.shape[1] > 4:
            metrics["rmse_pg_last2"] = float(np.sqrt(np.mean(resid[-2:, 4] ** 2)))
    return metrics


def export_tmcmc_diagnostics_tables(
    output_dir: Path,
    model_tag: str,
    diag: Dict[str, Any],
) -> None:
    """
    Export TMCMC diagnostics (β/acc/ROM/θ0) into simple CSV tables.

    Parameters
    ----------
    output_dir : Path
        Output directory
    model_tag : str
        Model identifier (e.g., "M1", "M2", "M3")
    diag : Dict[str, Any]
        Diagnostics dictionary containing:
        - beta_schedules: List of beta schedules per chain
        - acc_rate_histories: List of acceptance rate histories per chain
        - stage_summaries: List of stage summaries per chain
        - linearization_histories: List of linearization point histories per chain
        - rom_error_histories: List of ROM error histories per chain
    """
    tables_dir = output_dir / "diagnostics_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # β schedules
    beta_rows: List[List[Any]] = []
    for chain_id, sched in enumerate(diag.get("beta_schedules", []), start=1):
        for stage, beta in enumerate(sched):
            beta_rows.append([model_tag, chain_id, stage, float(beta)])
    if beta_rows:
        write_csv(
            tables_dir / f"{model_tag}_beta_schedule.csv",
            ["model", "chain", "stage", "beta"],
            beta_rows,
        )

    # acceptance rate histories
    acc_rows: List[List[Any]] = []
    for chain_id, hist in enumerate(diag.get("acc_rate_histories", []), start=1):
        for stage, acc in enumerate(hist):
            acc_rows.append([model_tag, chain_id, stage, float(acc)])
    if acc_rows:
        write_csv(
            tables_dir / f"{model_tag}_acceptance_rate.csv",
            ["model", "chain", "stage", "accept_rate"],
            acc_rows,
        )

    # Stage summary (per chain, per stage)
    stage_rows: List[List[Any]] = []
    for chain_id, hist in enumerate(diag.get("stage_summaries", []), start=1):
        if not isinstance(hist, list):
            continue
        for row in hist:
            if not isinstance(row, dict):
                continue
            stage_rows.append(
                [
                    model_tag,
                    chain_id,
                    int(row.get("stage", -1)),
                    float(row.get("beta", float("nan"))),
                    float(row.get("beta_next", float("nan"))),
                    float(row.get("delta_beta", float("nan"))),
                    float(row.get("ess", float("nan"))),
                    float(row.get("ess_target", float("nan"))),
                    float(row.get("acc_rate", float("nan"))),
                    float(row.get("logL_min", float("nan"))),
                    float(row.get("logL_max", float("nan"))),
                    int(row.get("linearization_enabled", 0)),
                    (
                        float(row.get("rom_error_pre", float("nan")))
                        if row.get("rom_error_pre") is not None
                        else float("nan")
                    ),
                    (
                        float(row.get("rom_error_post", float("nan")))
                        if row.get("rom_error_post") is not None
                        else float("nan")
                    ),
                    (
                        float(row.get("delta_theta0", float("nan")))
                        if row.get("delta_theta0") is not None
                        else float("nan")
                    ),
                ]
            )
    if stage_rows:
        write_csv(
            tables_dir / f"{model_tag}_stage_summary.csv",
            [
                "model",
                "chain",
                "stage",
                "beta",
                "beta_next",
                "delta_beta",
                "ess",
                "ess_target",
                "accept_rate",
                "logL_min",
                "logL_max",
                "linearization_enabled",
                "rom_error_pre",
                "rom_error_post",
                "delta_theta0",
            ],
            stage_rows,
        )

    # ROM error histories (at linearization update events)
    rom_rows: List[List[Any]] = []
    # Prefer post-update ROM error if available; keep pre-update as an extra column for debugging.
    # Backward compatibility: diag["rom_error_histories"] is treated as post-update values.
    rom_post_histories = diag.get("rom_error_histories", [])
    rom_pre_histories = diag.get("rom_error_pre_histories", None)
    for chain_id, post_hist in enumerate(rom_post_histories, start=1):
        pre_hist = None
        if isinstance(rom_pre_histories, list) and (chain_id - 1) < len(rom_pre_histories):
            pre_hist = rom_pre_histories[chain_id - 1]
        for upd, post_err in enumerate(post_hist):
            pre_err = None
            if isinstance(pre_hist, (list, tuple)) and upd < len(pre_hist):
                pre_err = pre_hist[upd]
            rom_rows.append(
                [
                    model_tag,
                    chain_id,
                    upd,
                    float(post_err) if post_err is not None else float("nan"),
                    float(pre_err) if pre_err is not None else float("nan"),
                ]
            )
    if rom_rows:
        write_csv(
            tables_dir / f"{model_tag}_rom_error.csv",
            ["model", "chain", "update", "rom_error", "rom_error_pre"],
            rom_rows,
        )

    # θ0 history + step norm
    theta0_rows: List[List[Any]] = []
    for chain_id, hist in enumerate(diag.get("theta0_history", []), start=1):
        for upd, theta0 in enumerate(hist):
            theta0 = np.asarray(theta0, dtype=float).reshape(-1)
            step_norm = None
            if upd > 0:
                prev = np.asarray(hist[upd - 1], dtype=float).reshape(-1)
                step_norm = float(np.linalg.norm(theta0 - prev))
            theta0_rows.append([model_tag, chain_id, upd, step_norm, *theta0.tolist()])
    if theta0_rows:
        header = ["model", "chain", "update", "step_norm"] + [
            f"theta0_{i}" for i in range(len(theta0_rows[0]) - 4)
        ]
        write_csv(tables_dir / f"{model_tag}_theta0_history.csv", header, theta0_rows)

    logger.info("Exported diagnostics tables for %s to %s", model_tag, tables_dir)
