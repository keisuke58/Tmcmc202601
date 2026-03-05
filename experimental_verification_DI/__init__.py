"""
experimental_verification_DI — DI→E constitutive law の実験検証モジュール

同一サンプルでの 16S rRNA + AFM 測定データを用いた E(DI) モデルの
フィッティング・検証パイプラインを提供する。

使用例
------
    from experimental_verification_DI import load_experimental_data, fit_E_DI
    data = load_experimental_data("data/di_e_pairs.csv")
    params, report = fit_E_DI(data["di"], data["E"], data.get("E_err"))
"""

from .data_loader import (
    load_experimental_data,
    save_experimental_data,
    DATA_FORMAT_SPEC,
)
from .fit_constitutive import (
    fit_E_DI,
    predict_E_DI,
    FitResult,
)

__all__ = [
    "load_experimental_data",
    "save_experimental_data",
    "DATA_FORMAT_SPEC",
    "fit_E_DI",
    "predict_E_DI",
    "FitResult",
]
