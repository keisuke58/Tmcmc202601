#!/usr/bin/env python3
"""
synthetic_data.py — 合成 (DI, E) データの生成

実験データが得られるまでの検証用。真のパラメータで E(DI) を計算し、
測定誤差を加えた合成データを生成する。
"""

from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

import numpy as np

from .fit_constitutive import predict_E_DI

logger = logging.getLogger(__name__)

# 論文値（Pattem 2018/2021, 0D DI 用）
E_MAX_TRUE = 909.0
E_MIN_TRUE = 32.0
DI_SCALE_TRUE = 1.0
EXPONENT_TRUE = 2.0


def generate_synthetic_data(
    n_samples: int = 30,
    di_range: tuple[float, float] = (0.05, 0.95),
    noise_frac: float = 0.15,
    seed: Optional[int] = None,
    e_max: float = E_MAX_TRUE,
    e_min: float = E_MIN_TRUE,
    di_scale: float = DI_SCALE_TRUE,
    exponent: float = EXPONENT_TRUE,
) -> dict:
    """
    合成 (DI, E) データを生成する。

    Parameters
    ----------
    n_samples : int
        サンプル数
    di_range : tuple
        DI の範囲 (min, max)
    noise_frac : float
        相対ノイズ（E に対する標準偏差の割合、例: 0.15 = 15%）
    seed : int, optional
        乱数シード
    e_max, e_min, di_scale, exponent : float
        真のモデルパラメータ

    Returns
    -------
    dict
        di, E, E_err, E_true, params_true
    """
    rng = np.random.default_rng(seed)
    di = rng.uniform(di_range[0], di_range[1], size=n_samples)
    di = np.sort(di)  # 可視化しやすく

    E_true = predict_E_DI(di, e_max, e_min, di_scale, exponent)
    noise_std = noise_frac * E_true
    noise_std = np.where(noise_std > 1.0, noise_std, 1.0)  # 最小 1 Pa
    E = E_true + rng.normal(0, noise_std, size=n_samples)
    E = np.clip(E, e_min * 0.5, e_max * 1.5)  # 物理的範囲内にクリップ

    return {
        "di": di,
        "E": E,
        "E_err": noise_std,
        "E_true": E_true,
        "params_true": {
            "e_max": e_max,
            "e_min": e_min,
            "di_scale": di_scale,
            "exponent": exponent,
        },
        "n_samples": n_samples,
    }


def generate_condition_aware_data(
    n_per_condition: int = 8,
    noise_frac: float = 0.15,
    seed: Optional[int] = None,
) -> dict:
    """
    4 条件（CS, CH, DS, DH）を模した合成データを生成。

    各条件の代表 DI は 0D Hamilton ODE の MAP に近い値を用いる。
    """
    # 0D MAP 付近の DI（論文・posterior_ci_0d より）
    condition_di = {
        "commensal_static": 0.16,
        "commensal_hobic": 0.25,
        "dysbiotic_hobic": 0.51,
        "dysbiotic_static": 0.85,
    }
    rng = np.random.default_rng(seed)

    di_list: list[float] = []
    E_list: list[float] = []
    E_err_list: list[float] = []
    condition_list: list[str] = []
    sample_ids: list[str] = []

    for cond, di_center in condition_di.items():
        di_spread = 0.08
        di_vals = rng.normal(di_center, di_spread, size=n_per_condition)
        di_vals = np.clip(di_vals, 0.01, 0.99)
        E_true = predict_E_DI(
            di_vals,
            E_MAX_TRUE,
            E_MIN_TRUE,
            DI_SCALE_TRUE,
            EXPONENT_TRUE,
        )
        noise_std = noise_frac * E_true
        E_vals = E_true + rng.normal(0, noise_std)
        E_vals = np.clip(E_vals, E_MIN_TRUE * 0.5, E_MAX_TRUE * 1.5)

        for i in range(n_per_condition):
            di_list.append(float(di_vals[i]))
            E_list.append(float(E_vals[i]))
            E_err_list.append(float(noise_std[i]))
            condition_list.append(cond)
            sample_ids.append(f"{cond}_{i+1:02d}")

    return {
        "di": np.array(di_list),
        "E": np.array(E_list),
        "E_err": np.array(E_err_list),
        "condition": condition_list,
        "sample_id": sample_ids,
        "n_samples": len(di_list),
    }
