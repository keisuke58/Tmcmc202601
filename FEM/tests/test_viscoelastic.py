#!/usr/bin/env python3
"""
Unit tests for viscoelastic and Mooney-Rivlin material models.

Run from Tmcmc202601/FEM/:
    python -m pytest tests/test_viscoelastic.py -v
"""

import sys
from pathlib import Path

import numpy as np

_FEM_DIR = Path(__file__).resolve().parent.parent
if str(_FEM_DIR) not in sys.path:
    sys.path.insert(0, str(_FEM_DIR))

from material_models import (
    compute_mooney_rivlin_params,
    compute_prony_params_di,
    compute_viscosity_di,
    compute_relaxation_modulus,
    compute_viscoelastic_params_di,
    sls_stress_relaxation,
    DI_SCALE,
    E_MAX_PA,
    E_MIN_PA,
    PRONY_G1_HEALTHY,
    PRONY_G1_DEGRADED,
    PRONY_TAU_HEALTHY,
    PRONY_TAU_DEGRADED,
)

# ── Mooney-Rivlin tests ──────────────────────────────────────────────


def test_mooney_rivlin_neohookean_limit():
    """c01_ratio=0 should reduce Mooney-Rivlin to Neo-Hookean (C01=0)."""
    E = np.array([500.0, 1000.0])
    params = compute_mooney_rivlin_params(E, nu=0.30, c01_ratio=0.0)
    np.testing.assert_allclose(params["C01"], 0.0)
    mu = E / (2 * 1.30)
    np.testing.assert_allclose(params["C10"], 0.5 * mu, rtol=1e-10)


def test_mooney_rivlin_c01_ratio():
    """C01/C10 should match the specified ratio."""
    E = np.array([500.0])
    for ratio in [0.1, 0.15, 0.25]:
        params = compute_mooney_rivlin_params(E, c01_ratio=ratio)
        actual_ratio = params["C01"] / params["C10"]
        np.testing.assert_allclose(actual_ratio, ratio, rtol=1e-10)


def test_mooney_rivlin_mu_consistency():
    """mu = 2*(C10 + C01) should hold."""
    E = np.array([100.0, 500.0, 1000.0])
    params = compute_mooney_rivlin_params(E, nu=0.30, c01_ratio=0.15)
    mu_expected = E / (2 * 1.30)
    mu_from_params = 2 * (params["C10"] + params["C01"])
    np.testing.assert_allclose(mu_from_params, mu_expected, rtol=1e-10)


def test_mooney_rivlin_d1_consistency():
    """K = 2/D1 should match E/(3*(1-2*nu))."""
    E = np.array([1000.0])
    nu = 0.30
    params = compute_mooney_rivlin_params(E, nu=nu)
    K_expected = E / (3.0 * (1 - 2 * nu))
    K_from_d1 = 2.0 / params["D1"]
    np.testing.assert_allclose(K_from_d1, K_expected, rtol=1e-10)


# ── Prony series tests ───────────────────────────────────────────────


def test_prony_params_healthy():
    """DI=0 (healthy) should give low relaxation."""
    di = np.array([0.0])
    p = compute_prony_params_di(di)
    np.testing.assert_allclose(p["g1"], PRONY_G1_HEALTHY)
    np.testing.assert_allclose(p["tau1"], PRONY_TAU_HEALTHY)
    np.testing.assert_allclose(p["k1"], 0.0)


def test_prony_params_dysbiotic():
    """DI=DI_SCALE (max dysbiosis) should give high relaxation."""
    di = np.array([DI_SCALE])
    p = compute_prony_params_di(di)
    np.testing.assert_allclose(p["g1"], PRONY_G1_DEGRADED)
    np.testing.assert_allclose(p["tau1"], PRONY_TAU_DEGRADED)


def test_prony_params_monotonic():
    """g1 and tau1 should increase monotonically with DI."""
    di = np.linspace(0, DI_SCALE, 50)
    p = compute_prony_params_di(di)
    assert np.all(np.diff(p["g1"]) >= 0), "g1 not monotonically increasing"
    assert np.all(np.diff(p["tau1"]) >= 0), "tau1 not monotonically increasing"


# ── Viscosity tests ──────────────────────────────────────────────────


def test_viscosity_healthy_vs_degraded():
    """Dysbiotic biofilm should be more viscous (higher eta)."""
    di_h = np.array([0.0])
    di_d = np.array([DI_SCALE])
    eta_h = compute_viscosity_di(di_h)
    eta_d = compute_viscosity_di(di_d)
    assert eta_d > eta_h, "Dysbiotic should have higher viscosity"


# ── Relaxation modulus tests ─────────────────────────────────────────


def test_relaxation_t0():
    """At t=0, G(0) = G0 (instantaneous modulus)."""
    G0 = 500.0
    g1 = 0.5
    G_t0 = compute_relaxation_modulus(np.array([0.0]), G0, g1, tau1=10.0)
    np.testing.assert_allclose(G_t0, G0, rtol=1e-10)


def test_relaxation_t_inf():
    """At t>>tau, G(inf) = G0*(1-g1)."""
    G0 = 500.0
    g1 = 0.5
    G_inf = compute_relaxation_modulus(np.array([1e6]), G0, g1, tau1=10.0)
    np.testing.assert_allclose(G_inf, G0 * (1 - g1), rtol=1e-6)


def test_relaxation_monotonic():
    """G(t) should be monotonically decreasing."""
    t = np.linspace(0, 200, 500)
    G = compute_relaxation_modulus(t, G0=500.0, g1=0.5, tau1=10.0)
    assert np.all(np.diff(G) <= 0), "G(t) not monotonically decreasing"


# ── SLS viscoelastic parameter tests ─────────────────────────────────


def test_sls_params_ranges():
    """SLS E_inf should match DI-based E, E_0 > E_inf."""
    di = np.linspace(0, DI_SCALE, 20)
    p = compute_viscoelastic_params_di(di)
    assert np.all(p["E_0"] >= p["E_inf"]), "E_0 must be >= E_inf"
    assert np.all(p["E_1"] >= 0), "E_1 must be non-negative"
    assert np.all(p["tau"] > 0), "tau must be positive"
    assert np.all(p["eta"] >= 0), "eta must be non-negative"


def test_sls_stress_relaxation_limits():
    """Stress relaxation: t=0 → E_0*eps, t=inf → E_inf*eps."""
    p = compute_viscoelastic_params_di(np.array([0.01]))
    eps_0 = 0.01
    t_arr = np.array([0.0, 1e6])
    sigma = sls_stress_relaxation(p["E_inf"], p["E_1"], p["tau"], eps_0, t_arr)
    np.testing.assert_allclose(sigma[0], (p["E_inf"] + p["E_1"]) * eps_0, rtol=1e-6)
    np.testing.assert_allclose(sigma[1], p["E_inf"] * eps_0, rtol=1e-3)


# ── 2D Viscoelastic solver tests (small grid) ────────────────────────


def test_2d_viscoelastic_relaxation():
    """Under constant eigenstrain, stress should relax over time."""
    sys.path.insert(0, str(_FEM_DIR / "JAXFEM"))
    from solve_stress_2d import solve_2d_fem, solve_2d_fem_viscoelastic

    Nx, Ny = 5, 5
    E_field = np.full((Nx, Ny), 500.0)
    eps_g = np.full((Nx, Ny), 0.01)

    # Static (elastic) solution
    res_static = solve_2d_fem(E_field, 0.30, eps_g, Nx, Ny)

    # Viscoelastic solution
    res_visco = solve_2d_fem_viscoelastic(
        E_field,
        0.30,
        eps_g,
        Nx,
        Ny,
        g1=0.5,
        tau1=5.0,
        t_total=50.0,
        dt=1.0,
    )

    # At long time, viscoelastic stress should be lower than elastic
    svm_static = res_static["sigma_vm"].mean()
    svm_visco = res_visco["sigma_vm"].mean()
    assert svm_visco < svm_static, (
        f"Viscoelastic stress ({svm_visco:.4f}) should be less than "
        f"elastic stress ({svm_static:.4f}) after relaxation"
    )


def test_2d_viscoelastic_time_history():
    """Stress should decrease monotonically in time history."""
    sys.path.insert(0, str(_FEM_DIR / "JAXFEM"))
    from solve_stress_2d import solve_2d_fem_viscoelastic

    Nx, Ny = 5, 5
    E_field = np.full((Nx, Ny), 500.0)
    eps_g = np.full((Nx, Ny), 0.01)

    res = solve_2d_fem_viscoelastic(
        E_field,
        0.30,
        eps_g,
        Nx,
        Ny,
        g1=0.5,
        tau1=5.0,
        t_total=50.0,
        dt=1.0,
    )

    svm_history = res["snap_sigma_vm_mean"]
    # After initial ramp (first few steps), stress should decrease
    # Skip first step (t=0 has eps_v=0)
    late = svm_history[2:]
    diffs = np.diff(late)
    # Allow small numerical noise
    assert np.all(diffs < 1e-6), "Stress should decrease monotonically after loading"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
