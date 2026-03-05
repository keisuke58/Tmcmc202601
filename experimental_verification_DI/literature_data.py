#!/usr/bin/env python3
"""
literature_data.py — 文献の AFM/レオロジーデータ（E, DI 相当）

Pattem 2018/2021, Gloag 2019 の測定値。DI は直接測定されていないため、
diverse/cariogenic の対応で近似値を付与（論文 Fig 11 参照）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LiteraturePoint:
    """文献データ点。"""

    label: str
    E_kPa: float  # Young 弾性率 [kPa]
    di_approx: Optional[float] = None  # 近似 DI（組成未測定のため推定）
    source: str = ""
    note: str = ""


# Pattem 2018: low-sucrose (diverse) 14.35 kPa, high-sucrose (cariogenic) 0.55 kPa
# DI: diverse → low (~0.15), cariogenic → high (~0.85)
LITERATURE_PATTEM_2018 = [
    LiteraturePoint(
        "Pattem 2018 (low-suc)", 14.35, di_approx=0.15, source="Pattem2018", note="diverse"
    ),
    LiteraturePoint(
        "Pattem 2018 (high-suc)", 0.55, di_approx=0.85, source="Pattem2018", note="cariogenic"
    ),
]

# Pattem 2021: hydrated LC 10.4 kPa vs HC 2.8 kPa
LITERATURE_PATTEM_2021 = [
    LiteraturePoint(
        "Pattem 2021 (LC)", 10.4, di_approx=0.20, source="Pattem2021", note="low cariogenic"
    ),
    LiteraturePoint(
        "Pattem 2021 (HC)", 2.8, di_approx=0.75, source="Pattem2021", note="high cariogenic"
    ),
]

# Gloag 2019: G' = 160 Pa (storage modulus). E ≈ 3*G' for incompressible → ~480 Pa
# Dual-species: intermediate diversity
LITERATURE_GLOAG_2019 = [
    LiteraturePoint(
        "Gloag 2019 (dual)", 0.48, di_approx=0.50, source="Gloag2019", note="G'=160 Pa, E≈3G'"
    ),
]


def get_literature_points(include_gloag: bool = True) -> list[LiteraturePoint]:
    """文献データ点のリストを返す。"""
    points = LITERATURE_PATTEM_2018 + LITERATURE_PATTEM_2021
    if include_gloag:
        points = points + LITERATURE_GLOAG_2019
    return points


def get_literature_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    (di, E_kPa, labels) のタプルを返す。
    プロット用。di_approx が None の点は除外。
    """
    points = get_literature_points()
    di_list = []
    e_list = []
    labels = []
    for p in points:
        if p.di_approx is not None:
            di_list.append(p.di_approx)
            e_list.append(p.E_kPa)
            labels.append(p.label)
    return np.array(di_list), np.array(e_list), labels
