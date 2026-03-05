#!/usr/bin/env python3
"""
data_loader.py — 実験 (DI, E) データの読み込み・保存

データフォーマット（CSV）:
  di,E,E_err,condition,sample_id
  0.05,850.2,120.5,commensal_static,S01
  0.82,45.1,8.2,dysbiotic_static,S02
  ...

必須列: di, E
オプション: E_err (測定誤差), condition, sample_id
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_FORMAT_SPEC = """
実験データ CSV フォーマット
==========================

必須列:
  - di    : Dysbiosis Index [0, 1]
  - E     : Young 弾性率 [Pa]

オプション列:
  - E_err : 測定誤差（標準偏差）[Pa]。重み付き最小二乗で使用
  - condition : 培養条件 (commensal_static, dysbiotic_static, etc.)
  - sample_id : サンプル識別子

例:
  di,E,E_err,condition,sample_id
  0.05,850.2,120.5,commensal_static,S01
  0.82,45.1,8.2,dysbiotic_static,S02
"""


def load_experimental_data(
    path: str | Path,
) -> dict[str, Any]:
    """
    実験 (DI, E) データを CSV から読み込む。

    Parameters
    ----------
    path : str or Path
        CSV ファイルパス

    Returns
    -------
    dict
        - di : np.ndarray, shape (N,)
        - E : np.ndarray, shape (N,)
        - E_err : np.ndarray or None
        - condition : list[str] or None
        - sample_id : list[str] or None
        - n_samples : int
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    di_list: list[float] = []
    e_list: list[float] = []
    e_err_list: list[float] = []
    condition_list: list[str] = []
    sample_id_list: list[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "di" not in reader.fieldnames or "E" not in reader.fieldnames:
            raise ValueError(f"CSV must have 'di' and 'E' columns. Found: {reader.fieldnames}")
        has_e_err = "E_err" in (reader.fieldnames or [])
        has_condition = "condition" in (reader.fieldnames or [])
        has_sample_id = "sample_id" in (reader.fieldnames or [])

        for row in reader:
            di_list.append(float(row["di"]))
            e_list.append(float(row["E"]))
            if has_e_err and row.get("E_err"):
                e_err_list.append(float(row["E_err"]))
            else:
                e_err_list.append(np.nan)
            if has_condition:
                condition_list.append(row.get("condition", ""))
            if has_sample_id:
                sample_id_list.append(row.get("sample_id", ""))

    di = np.array(di_list, dtype=np.float64)
    E = np.array(e_list, dtype=np.float64)
    e_err_arr = np.array(e_err_list, dtype=np.float64)
    has_valid_err = np.any(np.isfinite(e_err_arr))

    out: dict[str, Any] = {
        "di": di,
        "E": E,
        "n_samples": len(di),
    }
    if has_valid_err:
        default_err = np.nanmedian(e_err_arr[np.isfinite(e_err_arr)])
        out["E_err"] = np.where(np.isfinite(e_err_arr), e_err_arr, default_err)
    if condition_list:
        out["condition"] = condition_list
    if sample_id_list:
        out["sample_id"] = sample_id_list

    logger.info("Loaded %d (DI, E) pairs from %s", len(di), path)
    return out


def save_experimental_data(
    path: str | Path,
    di: np.ndarray,
    E: np.ndarray,
    E_err: np.ndarray | None = None,
    condition: list[str] | None = None,
    sample_id: list[str] | None = None,
) -> None:
    """
    実験データを CSV に保存する。

    Parameters
    ----------
    path : str or Path
        出力 CSV パス
    di, E : np.ndarray
        DI と弾性率
    E_err : np.ndarray, optional
        測定誤差
    condition, sample_id : list, optional
        メタデータ
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(di)
    if len(E) != n:
        raise ValueError(f"di and E length mismatch: {len(di)} vs {len(E)}")

    fieldnames = ["di", "E"]
    if E_err is not None:
        fieldnames.append("E_err")
    if condition is not None:
        fieldnames.append("condition")
    if sample_id is not None:
        fieldnames.append("sample_id")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            row: dict[str, str | float] = {"di": float(di[i]), "E": float(E[i])}
            if E_err is not None:
                row["E_err"] = float(E_err[i]) if i < len(E_err) else ""
            if condition is not None and i < len(condition):
                row["condition"] = condition[i]
            if sample_id is not None and i < len(sample_id):
                row["sample_id"] = sample_id[i]
            writer.writerow(row)

    logger.info("Saved %d rows to %s", n, path)
