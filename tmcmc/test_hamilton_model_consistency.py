"""
Regression tests connecting the biofilm model equations (derived via Hamilton principle)
to the implemented residual in `improved1207_paper_jit.py`.

These tests do NOT prove the full continuum theory.
They prove that the implemented residual matches the paper's strong form
after (a) material-point reduction and (b) implicit Euler time discretization,
up to the barrier terms used to enforce 0<phi,psi<1.
"""

from __future__ import annotations

import numpy as np
import pytest

from improved1207_paper_jit import _compute_Q_vector_numpy


def _paper_residual_no_barrier(
    *,
    phi_new: np.ndarray,
    phi0_new: float,
    psi_new: np.ndarray,
    gamma_new: float,
    phi_old: np.ndarray,
    phi0_old: float,
    psi_old: np.ndarray,
    dt: float,
    eta: np.ndarray,
    eta_phi: np.ndarray,
    c_val: float,
    alpha_val: float,
    A: np.ndarray,
    b_diag: np.ndarray,
) -> np.ndarray:
    """
    Paper strong form (biofilm_simulation eqs. (16)-(18)) discretized by implicit Euler,
    expressed in the same scaling convention as the implementation:

    - phi equations are divided by eta[i] (since the code uses (1/eta[i]) * (...) terms)
    - psi equations are divided by eta[i] (same)
    - NO barrier terms here
    - IMPORTANT: gamma does NOT appear in psi-equations because the constraint depends only on phi.
    """
    phi_new = np.asarray(phi_new, dtype=float)
    psi_new = np.asarray(psi_new, dtype=float)
    phi_old = np.asarray(phi_old, dtype=float)
    psi_old = np.asarray(psi_old, dtype=float)
    eta = np.asarray(eta, dtype=float)
    eta_phi = np.asarray(eta_phi, dtype=float)
    A = np.asarray(A, dtype=float)
    b_diag = np.asarray(b_diag, dtype=float)

    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    phibar = phi_new * psi_new
    interaction = A @ phibar  # (A * phibar)_i

    Q = np.zeros(10, dtype=float)

    # phi_i (i=0..3 are species, phi0 is separate index 4 in code)
    for i in range(4):
        # paper: 0 = -c* psi_i * (A phibar)_i + eta_i*(phidot_i*psi_i^2 + phibar_i*psidot_i) + eta_phi_i*phidot_i + gamma
        # divide by eta_i to match code scaling
        visc = (eta_phi[i] + eta[i] * psi_new[i] ** 2) * phidot[i] + eta[i] * phi_new[i] * psi_new[i] * psidot[i]
        Q[i] = (gamma_new + visc - c_val * psi_new[i] * interaction[i]) / eta[i]

    # phi0 equation: code uses gamma + phi0dot + (barrier term)
    # paper: gamma + phi0dot = 0 (phi0 is enforced via constraint; no energy term for phi0)
    Q[4] = gamma_new + phi0dot

    # psi_i
    for i in range(4):
        # paper: 0 = -c* phi_i * (A phibar)_i + alpha* b_i * psi_i + eta_i*(psidot_i*phi_i^2 + phibar_i*phidot_i)
        # divide by eta_i to match code scaling
        visc = (phi_new[i] * psi_new[i] * phidot[i]) + (phi_new[i] ** 2) * psidot[i]
        Q[5 + i] = (alpha_val * b_diag[i] * psi_new[i] + visc - c_val * phi_new[i] * interaction[i]) / eta[i]

    # constraint
    Q[9] = float(np.sum(phi_new) + phi0_new - 1.0)
    return Q


class TestHamiltonModelConsistency:
    def test_residual_matches_paper_when_barrier_is_off(self):
        rng = np.random.default_rng(0)

        dt = 1e-4
        c_val = 100.0
        alpha_val = 10.0

        # symmetric interaction matrix, as in the paper
        M = rng.normal(size=(4, 4))
        A = 0.5 * (M + M.T)
        b_diag = np.abs(rng.normal(size=4)) + 0.1

        # choose states strictly inside (0,1)
        phi_old = rng.uniform(0.05, 0.3, size=4)
        phi_new = rng.uniform(0.05, 0.3, size=4)
        # enforce phi0 via constraint (as solver typically does)
        phi0_old = float(max(1e-6, 1.0 - np.sum(phi_old)))
        phi0_new = float(max(1e-6, 1.0 - np.sum(phi_new)))

        psi_old = rng.uniform(0.2, 0.9, size=4)
        psi_new = rng.uniform(0.2, 0.9, size=4)

        gamma_new = float(rng.normal())

        eta = np.ones(4)
        # match paper setting (same eta appears in both dissipation terms)
        eta_phi = eta.copy()

        # Barrier disabled
        Kp1 = 0.0

        Q_code = _compute_Q_vector_numpy(
            phi_new=phi_new,
            phi0_new=phi0_new,
            psi_new=psi_new,
            gamma_new=gamma_new,
            phi_old=phi_old,
            phi0_old=phi0_old,
            psi_old=psi_old,
            dt=dt,
            Kp1=Kp1,
            Eta_vec=eta,
            Eta_phi_vec=eta_phi,
            c_val=c_val,
            alpha_val=alpha_val,
            A=A,
            b_diag=b_diag,
        ).astype(float)

        Q_paper = _paper_residual_no_barrier(
            phi_new=phi_new,
            phi0_new=phi0_new,
            psi_new=psi_new,
            gamma_new=gamma_new,
            phi_old=phi_old,
            phi0_old=phi0_old,
            psi_old=psi_old,
            dt=dt,
            eta=eta,
            eta_phi=eta_phi,
            c_val=c_val,
            alpha_val=alpha_val,
            A=A,
            b_diag=b_diag,
        )

        np.testing.assert_allclose(Q_code, Q_paper, rtol=0.0, atol=1e-10)

    def test_psi_equations_do_not_depend_on_gamma(self):
        """
        Mathematical requirement:
        constraint f(phi)=sum(phi)-1 does not depend on psi, so delta_psi(C)=0.
        Therefore the psi equations must not depend on gamma.
        """
        rng = np.random.default_rng(1)

        dt = 1e-4
        c_val = 100.0
        alpha_val = 10.0
        M = rng.normal(size=(4, 4))
        A = 0.5 * (M + M.T)
        b_diag = np.abs(rng.normal(size=4)) + 0.1

        phi_old = rng.uniform(0.1, 0.2, size=4)
        phi_new = rng.uniform(0.1, 0.2, size=4)
        phi0_old = float(1.0 - np.sum(phi_old))
        phi0_new = float(1.0 - np.sum(phi_new))
        psi_old = rng.uniform(0.2, 0.8, size=4)
        psi_new = rng.uniform(0.2, 0.8, size=4)

        eta = np.ones(4)
        eta_phi = np.ones(4)
        Kp1 = 0.0

        gamma0 = 0.123
        eps = 1e-7
        Q0 = _compute_Q_vector_numpy(
            phi_new=phi_new,
            phi0_new=phi0_new,
            psi_new=psi_new,
            gamma_new=gamma0,
            phi_old=phi_old,
            phi0_old=phi0_old,
            psi_old=psi_old,
            dt=dt,
            Kp1=Kp1,
            Eta_vec=eta,
            Eta_phi_vec=eta_phi,
            c_val=c_val,
            alpha_val=alpha_val,
            A=A,
            b_diag=b_diag,
        ).astype(float)

        Q1 = _compute_Q_vector_numpy(
            phi_new=phi_new,
            phi0_new=phi0_new,
            psi_new=psi_new,
            gamma_new=gamma0 + eps,
            phi_old=phi_old,
            phi0_old=phi0_old,
            psi_old=psi_old,
            dt=dt,
            Kp1=Kp1,
            Eta_vec=eta,
            Eta_phi_vec=eta_phi,
            c_val=c_val,
            alpha_val=alpha_val,
            A=A,
            b_diag=b_diag,
        ).astype(float)

        dQ_dgamma = (Q1 - Q0) / eps
        # psi equations are indices 5..8
        assert np.all(np.abs(dQ_dgamma[5:9]) < 1e-10)

