#!/usr/bin/env python3
"""
Unit tests for DI (Dysbiotic Index) computation and E(DI) mapping.

Run from Tmcmc202601/FEM/:
    python -m pytest tests/test_di_e_mapping.py -v
    python -m pytest tests/test_di_e_mapping.py -v -k test_compute_di
"""
import sys
from pathlib import Path

import numpy as np

# Add FEM to path for imports
_FEM_DIR = Path(__file__).resolve().parent.parent
if str(_FEM_DIR) not in sys.path:
    sys.path.insert(0, str(_FEM_DIR))

from material_models import (
    DI_SCALE,
    compute_E_di,
    compute_di_gini,
    compute_di_pielou,
    compute_di_simpson,
    compute_E_reuss,
    compute_E_voigt,
)


def compute_di(phi_all: np.ndarray) -> np.ndarray:
    """
    Compute Dysbiotic Index (mirrors tmcmc_to_fem_coupling._compute_di).
    DI = 1 - H / H_max,  H = -sum p_i ln p_i,  H_max = ln(5)
    """
    sum_phi = np.sum(phi_all, axis=-1)
    sum_phi_safe = np.where(sum_phi > 0.0, sum_phi, 1.0)
    p = phi_all / sum_phi_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log(p), 0.0)
    H = -np.sum(p * log_p, axis=-1)
    H_max = np.log(5.0)
    return 1.0 - H / H_max


# ── DI tests ────────────────────────────────────────────────────────────────


def test_compute_di_equal_fractions():
    """DI = 0 when all 5 species have equal fractions (max diversity)."""
    phi = np.ones(5) / 5.0
    di = compute_di(phi)
    assert np.isclose(di, 0.0, atol=1e-10)


def test_compute_di_single_species():
    """DI = 1 when all mass in one species (maximally dysbiotic)."""
    phi = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    di = compute_di(phi)
    assert np.isclose(di, 1.0, atol=1e-10)


def test_compute_di_two_species_half():
    """DI for two species at 50-50."""
    phi = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    di = compute_di(phi)
    # H = -2 * 0.5 * ln(0.5) = ln(2), DI = 1 - ln(2)/ln(5)
    expected = 1.0 - np.log(2.0) / np.log(5.0)
    assert np.isclose(di, expected, atol=1e-10)


def test_compute_di_batch():
    """DI works for batched (N, 5) input."""
    phi_batch = np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
        ]
    )
    di = compute_di(phi_batch)
    assert di.shape == (3,)
    assert np.isclose(di[0], 0.0, atol=1e-10)
    assert np.isclose(di[1], 1.0, atol=1e-10)
    assert np.isclose(di[2], 1.0 - np.log(2.0) / np.log(5.0), atol=1e-10)


def test_compute_di_zero_sum_handling():
    """DI handles zero sum (avoids div-by-zero)."""
    phi = np.zeros(5)
    di = compute_di(phi)
    # sum_phi_safe = 1.0, p = 0 → log_p = 0, H = 0, DI = 1
    assert np.isfinite(di)


# ── E(DI) mapping tests ──────────────────────────────────────────────────────


def test_compute_E_di_r0():
    """E(DI) = E_max when r=0 (DI=0, commensal)."""
    e_max, e_min = 10.0e9, 0.5e9
    di = np.array([0.0])
    E = compute_E_di(di, e_max=e_max, e_min=e_min, di_scale=DI_SCALE, exponent=2.0)
    assert np.isclose(E[0], e_max, atol=1e-6)


def test_compute_E_di_r1():
    """E(DI) = E_min when r=1 (DI >= di_scale, dysbiotic)."""
    e_max, e_min = 10.0e9, 0.5e9
    di = np.array([DI_SCALE, DI_SCALE * 2.0])
    E = compute_E_di(di, e_max=e_max, e_min=e_min, di_scale=DI_SCALE, exponent=2.0)
    assert np.allclose(E, e_min, atol=1e-6)


def test_compute_E_di_r_half():
    """E(DI) at r=0.5: E = E_max * 0.25 + E_min * 0.5 for n=2."""
    e_max, e_min = 10.0e9, 0.5e9
    di = np.array([DI_SCALE * 0.5])
    E = compute_E_di(di, e_max=e_max, e_min=e_min, di_scale=DI_SCALE, exponent=2.0)
    expected = e_max * 0.25 + e_min * 0.5
    assert np.isclose(E[0], expected, atol=1e-6)


def test_compute_E_di_commensal_greater_than_dysbiotic():
    """E(0) > E(DI_SCALE): commensal (DI=0) is stiffer than dysbiotic (DI>=s)."""
    E_commensal = compute_E_di(np.array([0.0]), e_max=10.0e9, e_min=0.5e9)[0]
    E_dysbiotic = compute_E_di(np.array([DI_SCALE]), e_max=10.0e9, e_min=0.5e9)[0]
    assert E_commensal > E_dysbiotic


def test_compute_E_di_bounds():
    """E(DI) stays within [E_min, E_max] for any DI."""
    di_range = np.linspace(0, 1.0, 100)
    E = compute_E_di(di_range, e_max=10.0e9, e_min=0.5e9)
    assert np.all(E >= 0.5e9 - 1e-6)
    assert np.all(E <= 10.0e9 + 1e-6)


# ── Alternative DI indices ───────────────────────────────────────────────────


def test_compute_di_simpson_uniform():
    """Simpson DI = 0 for uniform (max diversity)."""
    phi = np.ones(5) / 5.0
    di = compute_di_simpson(phi)
    assert np.isclose(di, 0.0, atol=1e-10)


def test_compute_di_simpson_single():
    """Simpson DI = 1 for single-species dominance."""
    phi = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    di = compute_di_simpson(phi)
    assert np.isclose(di, 1.0, atol=1e-10)


def test_compute_di_gini_bounds():
    """Gini DI in [0, 1]."""
    phi = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0]])
    di = compute_di_gini(phi)
    assert np.all(di >= 0.0)
    assert np.all(di <= 1.0)


def test_compute_di_pielou_uniform():
    """Pielou DI = 0 for uniform."""
    phi = np.ones(5) / 5.0
    di = compute_di_pielou(phi)
    assert np.isclose(di, 0.0, atol=1e-6)


def test_compute_E_voigt_uniform():
    """Voigt: uniform gives mean of E_species."""
    phi = np.ones(5) / 5.0
    E = compute_E_voigt(phi.reshape(1, 5))[0]
    expected = 522.0  # (1000+800+600+200+10)/5
    assert np.isclose(E, expected, atol=1.0)


def test_compute_E_reuss_bounds():
    """Reuss E between min and max of E_species."""
    phi = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0]])
    E = compute_E_reuss(phi)
    assert np.all(E >= 10.0 - 1e-6)
    assert np.all(E <= 1000.0 + 1e-6)
