#!/usr/bin/env python3
"""
test_viscoelastic_2d.py — Tests for 2D SLS viscoelastic FEM solver
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "JAXFEM"))

from solve_stress_2d import solve_2d_fem, solve_2d_fem_viscoelastic_sls


class TestViscoelastic2D:
    """Test SLS viscoelastic 2D solver."""

    Nx, Ny = 6, 6
    nu = 0.3

    def _make_uniform_fields(self, E_inf_val, E_1_val, tau_val, eps_g_val):
        E_inf = np.full((self.Nx, self.Ny), E_inf_val)
        E_1 = np.full((self.Nx, self.Ny), E_1_val)
        tau = np.full((self.Nx, self.Ny), tau_val)
        eps_g = np.full((self.Nx, self.Ny), eps_g_val)
        return E_inf, E_1, tau, eps_g

    def test_t0_matches_E0_elastic(self):
        """At t=0, VE solver should give same result as elastic with E_0."""
        E_inf_val, E_1_val, tau_val = 500.0, 500.0, 30.0
        eps_g_val = 0.005
        E_inf, E_1, tau, eps_g = self._make_uniform_fields(E_inf_val, E_1_val, tau_val, eps_g_val)

        # VE at t=0
        t_array = np.array([0.0])
        res_ve = solve_2d_fem_viscoelastic_sls(
            E_inf,
            E_1,
            tau,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
            t_array,
        )

        # Elastic with E_0 = E_inf + E_1
        E_0 = np.full((self.Nx, self.Ny), E_inf_val + E_1_val)
        res_el = solve_2d_fem(
            E_0,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
        )

        # Compare von Mises stress
        np.testing.assert_allclose(
            res_ve["sigma_vm"],
            res_el["sigma_vm"],
            rtol=1e-6,
            err_msg="t=0 VE should match E_0 elastic",
        )

    def test_t_inf_approaches_Einf_elastic(self):
        """At t→∞, VE solver should approach elastic with E_inf."""
        E_inf_val, E_1_val, tau_val = 500.0, 500.0, 30.0
        eps_g_val = 0.005
        E_inf, E_1, tau, eps_g = self._make_uniform_fields(E_inf_val, E_1_val, tau_val, eps_g_val)

        # VE at very long time
        t_array = np.array([0.0, 1e6])
        res_ve = solve_2d_fem_viscoelastic_sls(
            E_inf,
            E_1,
            tau,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
            t_array,
        )

        # Elastic with E_inf
        res_el = solve_2d_fem(
            E_inf,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
        )

        np.testing.assert_allclose(
            res_ve["sigma_vm_history"][-1],
            res_el["sigma_vm"],
            rtol=1e-4,
            err_msg="t→∞ VE should match E_inf elastic",
        )

    def test_stress_relaxation(self):
        """Stress should decrease monotonically over time."""
        E_inf_val, E_1_val, tau_val = 500.0, 500.0, 30.0
        eps_g_val = 0.005
        E_inf, E_1, tau, eps_g = self._make_uniform_fields(E_inf_val, E_1_val, tau_val, eps_g_val)

        t_array = np.array([0.0, 5.0, 10.0, 30.0, 60.0, 120.0])
        res = solve_2d_fem_viscoelastic_sls(
            E_inf,
            E_1,
            tau,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
            t_array,
        )

        # Mean σ_vm should decrease
        mean_svm = [res["sigma_vm_history"][ti].mean() for ti in range(len(t_array))]
        for i in range(1, len(mean_svm)):
            assert (
                mean_svm[i] <= mean_svm[i - 1] + 1e-10
            ), f"Stress should relax: σ_vm[{i}]={mean_svm[i]} > σ_vm[{i-1}]={mean_svm[i-1]}"

    def test_tau_inf_gives_constant_stress(self):
        """τ→∞ should give constant stress (pure elastic E_0)."""
        E_inf_val, E_1_val = 500.0, 500.0
        eps_g_val = 0.005
        E_inf, E_1, tau, eps_g = self._make_uniform_fields(E_inf_val, E_1_val, 1e10, eps_g_val)

        t_array = np.array([0.0, 10.0, 100.0])
        res = solve_2d_fem_viscoelastic_sls(
            E_inf,
            E_1,
            tau,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
            t_array,
        )

        # All time steps should have same σ_vm
        for ti in range(1, len(t_array)):
            np.testing.assert_allclose(
                res["sigma_vm_history"][ti],
                res["sigma_vm_history"][0],
                rtol=1e-6,
                err_msg=f"τ→∞: stress at t={t_array[ti]} should equal t=0",
            )

    def test_output_shapes(self):
        """Check output array shapes."""
        E_inf, E_1, tau, eps_g = self._make_uniform_fields(500, 500, 30, 0.005)
        t_array = np.array([0.0, 10.0, 30.0])
        res = solve_2d_fem_viscoelastic_sls(
            E_inf,
            E_1,
            tau,
            self.nu,
            eps_g,
            self.Nx,
            self.Ny,
            t_array,
        )
        n_elem = (self.Nx - 1) * (self.Ny - 1)
        n_nodes = self.Nx * self.Ny

        assert res["u_history"].shape == (3, n_nodes, 2)
        assert res["sigma_vm_history"].shape == (3, n_elem)
        assert res["sigma_vm"].shape == (n_elem,)
        assert res["u"].shape == (n_nodes, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
