#!/usr/bin/env python3
"""
fit_constitutive.py — E(DI) constitutive law のフィッティング

モデル: E = E_max * (1 - r)^n + E_min * r,  r = clip(DI / di_scale, 0, 1)

scipy.optimize.curve_fit を用いた非線形最小二乗で
e_max, e_min, di_scale, exponent を推定する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)

# デフォルト（論文値、Pattem 2018/2021 等と整合）
E_MAX_DEFAULT = 909.0  # Pa (commensal)
E_MIN_DEFAULT = 32.0  # Pa (dysbiotic)
DI_SCALE_DEFAULT = 1.0  # 0D DI 用
EXPONENT_DEFAULT = 2.0


def _model_E_DI(
    di: np.ndarray,
    e_max: float,
    e_min: float,
    di_scale: float,
    exponent: float,
) -> np.ndarray:
    """E(DI) モデル（curve_fit 用）。"""
    r = np.clip(di / di_scale, 0.0, 1.0)
    return e_max * (1.0 - r) ** exponent + e_min * r


@dataclass
class FitResult:
    """E(DI) フィット結果。"""

    e_max: float
    e_min: float
    di_scale: float
    exponent: float
    rmse: float
    r_squared: float
    n_samples: int
    cov: Optional[np.ndarray] = None  # 4x4 共分散行列


def fit_E_DI(
    di: np.ndarray,
    E: np.ndarray,
    E_err: Optional[np.ndarray] = None,
    p0: Optional[tuple[float, float, float, float]] = None,
    bounds: Optional[tuple[tuple[float, ...], tuple[float, ...]]] = None,
) -> tuple[FitResult, dict]:
    """
    (DI, E) データから E(DI) constitutive law をフィットする。

    Parameters
    ----------
    di, E : np.ndarray
        測定データ
    E_err : np.ndarray, optional
        測定誤差（重み = 1/E_err^2）。None の場合は等方性誤差
    p0 : tuple, optional
        初期値 (e_max, e_min, di_scale, exponent)
    bounds : tuple of (lb, ub), optional
        パラメータ上下限

    Returns
    -------
    FitResult
        フィット結果
    dict
        診断情報（残差、予測値等）
    """
    di = np.asarray(di, dtype=np.float64).ravel()
    E = np.asarray(E, dtype=np.float64).ravel()
    if len(di) != len(E):
        raise ValueError(f"di and E length mismatch: {len(di)} vs {len(E)}")

    if p0 is None:
        p0 = (E_MAX_DEFAULT, E_MIN_DEFAULT, DI_SCALE_DEFAULT, EXPONENT_DEFAULT)

    if bounds is None:
        bounds = (
            (1.0, 1.0, 0.01, 0.5),  # lb
            (1e6, 1e6, 10.0, 5.0),  # ub (Pa, Pa, scale, exp)
        )

    sigma = None
    if E_err is not None:
        E_err = np.asarray(E_err, dtype=np.float64).ravel()
        if len(E_err) == len(E):
            sigma = np.where(E_err > 0, E_err, np.median(E) * 0.1)

    try:
        popt, pcov = optimize.curve_fit(
            _model_E_DI,
            di,
            E,
            p0=p0,
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            bounds=bounds,
            maxfev=5000,
        )
    except (RuntimeError, optimize.OptimizeWarning) as e:
        logger.warning("curve_fit failed: %s. Using initial guess.", e)
        popt = np.array(p0)
        pcov = np.diag([1e6, 1e6, 1e2, 1e2])

    e_max, e_min, di_scale, exponent = popt
    E_pred = _model_E_DI(di, *popt)
    residuals = E - E_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((E - np.mean(E)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-20)
    rmse = np.sqrt(np.mean(residuals**2))

    result = FitResult(
        e_max=float(e_max),
        e_min=float(e_min),
        di_scale=float(di_scale),
        exponent=float(exponent),
        rmse=float(rmse),
        r_squared=float(r_squared),
        n_samples=len(di),
        cov=pcov if pcov is not None else None,
    )

    report = {
        "E_pred": E_pred,
        "residuals": residuals,
        "popt": popt,
    }
    return result, report


def predict_E_DI(
    di: np.ndarray,
    e_max: float,
    e_min: float,
    di_scale: float,
    exponent: float = 2.0,
) -> np.ndarray:
    """
    E(DI) モデルで予測する。

    Parameters
    ----------
    di : np.ndarray
        DI 値
    e_max, e_min, di_scale, exponent : float
        モデルパラメータ

    Returns
    -------
    np.ndarray
        予測弾性率 [Pa]
    """
    return _model_E_DI(di, e_max, e_min, di_scale, exponent)
