#!/usr/bin/env python3
"""
experiment_design.py — 実験デザイン支援（サンプルサイズ・パワー解析）

Phase 1 で使用。目標 RMSE や検出力を満たすために必要な
サンプル数をモンテカルロで見積もる。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .fit_constitutive import fit_E_DI
from .synthetic_data import generate_synthetic_data

logger = logging.getLogger(__name__)


def estimate_required_samples(
    target_rmse: float = 50.0,
    target_r2: float = 0.90,
    noise_frac: float = 0.15,
    n_trials: int = 100,
    seed: int | None = None,
) -> dict:
    """
    目標 RMSE / R² を満たすために必要なサンプル数をモンテカルロで推定。

    Parameters
    ----------
    target_rmse : float
        目標 RMSE [Pa]
    target_r2 : float
        目標 R²
    noise_frac : float
        想定測定誤差（E に対する相対標準偏差）
    n_trials : int
        モンテカルロ試行数
    seed : int, optional
        乱数シード

    Returns
    -------
    dict
        n_samples_range, success_rate_by_n, recommended_n
    """
    rng = np.random.default_rng(seed)
    n_candidates = [15, 20, 25, 30, 40, 50, 60, 80, 100]
    success_rmse: dict[int, int] = {n: 0 for n in n_candidates}
    success_r2: dict[int, int] = {n: 0 for n in n_candidates}

    for n in n_candidates:
        for trial in range(n_trials):
            data = generate_synthetic_data(
                n_samples=n,
                noise_frac=noise_frac,
                seed=rng.integers(0, 2**31),
            )
            try:
                result, _ = fit_E_DI(
                    data["di"],
                    data["E"],
                    E_err=data["E_err"],
                )
                if result.rmse <= target_rmse:
                    success_rmse[n] += 1
                if result.r_squared >= target_r2:
                    success_r2[n] += 1
            except Exception:
                pass

    success_rate_rmse = {n: success_rmse[n] / n_trials for n in n_candidates}
    success_rate_r2 = {n: success_r2[n] / n_trials for n in n_candidates}

    # 80% 以上の成功率を満たす最小 n
    rec_rmse = next((n for n in n_candidates if success_rate_rmse[n] >= 0.80), n_candidates[-1])
    rec_r2 = next((n for n in n_candidates if success_rate_r2[n] >= 0.80), n_candidates[-1])
    recommended_n = max(rec_rmse, rec_r2)

    return {
        "n_samples_tested": n_candidates,
        "success_rate_rmse": success_rate_rmse,
        "success_rate_r2": success_rate_r2,
        "target_rmse": target_rmse,
        "target_r2": target_r2,
        "noise_frac": noise_frac,
        "n_trials": n_trials,
        "recommended_n_samples": recommended_n,
    }


def run_design_estimate(out_dir: Path | None = None) -> dict:
    """実験デザイン推定を実行し、結果を保存。"""
    logger.info("Running sample size estimation (Monte Carlo)...")
    # target_rmse: 15% noise で E~500 Pa の場合、残差 std ~ 75 Pa 程度
    result = estimate_required_samples(
        target_rmse=100.0,
        target_r2=0.90,
        noise_frac=0.15,
        n_trials=50,
    )
    logger.info("Recommended n_samples: %d", result["recommended_n_samples"])
    for n in result["n_samples_tested"]:
        logger.info(
            "  n=%d: RMSE≤%.0f success=%.0f%%, R²≥0.9 success=%.0f%%",
            n,
            result["target_rmse"],
            result["success_rate_rmse"][n] * 100,
            result["success_rate_r2"][n] * 100,
        )

    if out_dir is not None:
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sample_size_estimate.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", out_path)

    return result
