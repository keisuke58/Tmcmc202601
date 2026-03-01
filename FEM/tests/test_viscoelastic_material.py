#!/usr/bin/env python3
"""
test_viscoelastic_material.py — Unit tests for SLS viscoelastic material model
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from material_models import (
    compute_viscoelastic_params_di,
    sls_stress_relaxation,
    sls_creep_compliance,
    compute_E_di,
    E_MAX_PA,
    E_MIN_PA,
    TAU_MAX_S,
    TAU_MIN_S,
    E0_EINF_RATIO_MIN,
    E0_EINF_RATIO_MAX,
)


class TestViscoelasticParams:
    """Test compute_viscoelastic_params_di."""

    def test_commensal_limit(self):
        """DI=0 → max E_inf, slow tau, low ratio."""
        ve = compute_viscoelastic_params_di(np.array([0.0]), di_scale=1.0)
        assert np.isclose(ve["E_inf"][0], E_MAX_PA)
        assert np.isclose(ve["tau"][0], TAU_MAX_S)
        assert np.isclose(ve["E_0"][0], E_MAX_PA * E0_EINF_RATIO_MIN)

    def test_dysbiotic_limit(self):
        """DI=1 → min E_inf, fast tau, high ratio."""
        ve = compute_viscoelastic_params_di(np.array([1.0]), di_scale=1.0)
        assert np.isclose(ve["E_inf"][0], E_MIN_PA)
        assert np.isclose(ve["tau"][0], TAU_MIN_S)
        assert np.isclose(ve["E_0"][0], E_MIN_PA * E0_EINF_RATIO_MAX)

    def test_E_inf_equals_elastic(self):
        """E_inf should match the elastic E(DI) model."""
        di_arr = np.linspace(0, 1, 50)
        ve = compute_viscoelastic_params_di(di_arr, di_scale=1.0)
        E_elastic = compute_E_di(di_arr, di_scale=1.0)
        np.testing.assert_allclose(ve["E_inf"], E_elastic, rtol=1e-12)

    def test_E0_greater_than_Einf(self):
        """E_0 > E_inf for all DI (ratio > 1)."""
        di_arr = np.linspace(0, 1, 100)
        ve = compute_viscoelastic_params_di(di_arr, di_scale=1.0)
        assert np.all(ve["E_0"] > ve["E_inf"])

    def test_E1_positive(self):
        """E_1 = E_0 - E_inf > 0."""
        di_arr = np.linspace(0, 1, 100)
        ve = compute_viscoelastic_params_di(di_arr, di_scale=1.0)
        assert np.all(ve["E_1"] > 0)

    def test_eta_equals_E1_times_tau(self):
        """η = E_1 · τ."""
        di_arr = np.array([0.0, 0.3, 0.7, 1.0])
        ve = compute_viscoelastic_params_di(di_arr, di_scale=1.0)
        np.testing.assert_allclose(ve["eta"], ve["E_1"] * ve["tau"], rtol=1e-12)

    def test_monotonicity(self):
        """E_inf and tau should decrease with increasing DI."""
        di_arr = np.linspace(0, 1, 50)
        ve = compute_viscoelastic_params_di(di_arr, di_scale=1.0)
        assert np.all(np.diff(ve["E_inf"]) <= 0)
        assert np.all(np.diff(ve["tau"]) <= 0)

    def test_array_shapes(self):
        """Output shapes match input."""
        di = np.array([0.1, 0.5, 0.9])
        ve = compute_viscoelastic_params_di(di, di_scale=1.0)
        for key in ["E_inf", "E_0", "E_1", "tau", "eta"]:
            assert ve[key].shape == (3,), f"{key} shape mismatch"

    def test_scalar_input(self):
        """Works with scalar."""
        ve = compute_viscoelastic_params_di(0.5, di_scale=1.0)
        assert np.isscalar(ve["E_inf"]) or ve["E_inf"].ndim == 0


class TestSLSAnalytical:
    """Test analytical SLS solutions."""

    def test_relaxation_t0(self):
        """At t=0: σ = E_0 · ε."""
        E_inf, E_1, tau, eps = 500.0, 500.0, 30.0, 0.01
        sigma = sls_stress_relaxation(E_inf, E_1, tau, eps, 0.0)
        assert np.isclose(sigma, (E_inf + E_1) * eps)

    def test_relaxation_t_inf(self):
        """At t→∞: σ = E_inf · ε."""
        E_inf, E_1, tau, eps = 500.0, 500.0, 30.0, 0.01
        sigma = sls_stress_relaxation(E_inf, E_1, tau, eps, 1e6)
        assert np.isclose(sigma, E_inf * eps, rtol=1e-6)

    def test_relaxation_monotone(self):
        """Stress relaxation is monotonically decreasing."""
        E_inf, E_1, tau, eps = 500.0, 500.0, 30.0, 0.01
        t = np.linspace(0, 300, 100)
        sigma = sls_stress_relaxation(E_inf, E_1, tau, eps, t)
        assert np.all(np.diff(sigma) <= 0)

    def test_creep_t0(self):
        """At t=0: J = 1/E_0."""
        E_inf, E_1, tau = 500.0, 500.0, 30.0
        J = sls_creep_compliance(E_inf, E_1, tau, 0.0)
        assert np.isclose(J, 1.0 / (E_inf + E_1), rtol=1e-6)

    def test_creep_t_inf(self):
        """At t→∞: J = 1/E_inf."""
        E_inf, E_1, tau = 500.0, 500.0, 30.0
        J = sls_creep_compliance(E_inf, E_1, tau, 1e6)
        assert np.isclose(J, 1.0 / E_inf, rtol=1e-6)

    def test_creep_monotone(self):
        """Creep compliance is monotonically increasing."""
        E_inf, E_1, tau = 500.0, 500.0, 30.0
        t = np.linspace(0, 300, 100)
        J = sls_creep_compliance(E_inf, E_1, tau, t)
        assert np.all(np.diff(J) >= 0)

    def test_creep_bounded(self):
        """Creep is bounded: 1/E_0 ≤ J(t) ≤ 1/E_inf."""
        E_inf, E_1, tau = 500.0, 500.0, 30.0
        E_0 = E_inf + E_1
        t = np.linspace(0, 300, 100)
        J = sls_creep_compliance(E_inf, E_1, tau, t)
        assert np.all(J >= 1.0 / E_0 - 1e-12)
        assert np.all(J <= 1.0 / E_inf + 1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
