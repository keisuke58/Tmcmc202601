#!/usr/bin/env python3
"""
test_fit_validation.py — 実験検証モジュールのユニットテスト
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# パス追加
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experimental_verification_DI.fit_constitutive import (
    fit_E_DI,
    predict_E_DI,
    FitResult,
)
from experimental_verification_DI.synthetic_data import generate_synthetic_data
from experimental_verification_DI.data_loader import (
    load_experimental_data,
    save_experimental_data,
)


def test_predict_E_DI_bounds():
    """E(DI) は [E_min, E_max] の範囲内。"""
    di = np.array([0.0, 0.5, 1.0])
    E = predict_E_DI(di, e_max=1000.0, e_min=10.0, di_scale=1.0, exponent=2.0)
    assert np.all(E >= 10.0 - 1e-6)
    assert np.all(E <= 1000.0 + 1e-6)
    assert np.isclose(E[0], 1000.0)
    assert np.isclose(E[2], 10.0)


def test_fit_synthetic_recovery():
    """合成データから真のパラメータを概ね回復できる。"""
    data = generate_synthetic_data(n_samples=80, noise_frac=0.08, seed=123)
    pt = data["params_true"]
    result, _ = fit_E_DI(data["di"], data["E"], E_err=data["E_err"])

    # 相対誤差 20% 以内
    assert abs(result.e_max - pt["e_max"]) / pt["e_max"] < 0.25
    assert abs(result.e_min - pt["e_min"]) / pt["e_min"] < 0.25
    assert result.r_squared > 0.85


def test_fit_result_type():
    """FitResult が正しい型を持つ。"""
    di = np.array([0.1, 0.5, 0.9])
    E = predict_E_DI(di, 900.0, 30.0, 1.0, 2.0)
    result, _ = fit_E_DI(di, E)
    assert isinstance(result, FitResult)
    assert result.n_samples == 3
    assert result.e_max > 0
    assert result.e_min > 0


def test_data_loader_roundtrip(tmp_path):
    """save → load でデータが保持される。"""
    di = np.array([0.1, 0.5, 0.8])
    E = np.array([800.0, 200.0, 50.0])
    E_err = np.array([50.0, 20.0, 5.0])
    csv_path = tmp_path / "test.csv"
    save_experimental_data(csv_path, di, E, E_err=E_err)
    loaded = load_experimental_data(csv_path)
    np.testing.assert_array_almost_equal(loaded["di"], di)
    np.testing.assert_array_almost_equal(loaded["E"], E)
    np.testing.assert_array_almost_equal(loaded["E_err"], E_err)


def test_literature_data():
    """文献データが正しく取得できる。"""
    from experimental_verification_DI.literature_data import (
        get_literature_arrays,
        get_literature_points,
    )

    points = get_literature_points()
    assert len(points) >= 5
    di, e, labels = get_literature_arrays()
    assert len(di) == len(e) == len(labels)
    assert np.all(di >= 0) and np.all(di <= 1)
    assert np.all(e > 0)


def test_condition_aware_data():
    """4 条件データが正しく生成される。"""
    from experimental_verification_DI.synthetic_data import generate_condition_aware_data

    data = generate_condition_aware_data(n_per_condition=5, seed=0)
    assert data["n_samples"] == 20
    assert len(data["condition"]) == 20
    assert "commensal_static" in data["condition"]
    assert "dysbiotic_static" in data["condition"]
    # DI が高いほど E が低い
    idx_cs = [i for i, c in enumerate(data["condition"]) if c == "commensal_static"]
    idx_ds = [i for i, c in enumerate(data["condition"]) if c == "dysbiotic_static"]
    assert np.mean(data["E"][idx_cs]) > np.mean(data["E"][idx_ds])
