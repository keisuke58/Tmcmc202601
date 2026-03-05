# -*- coding: utf-8 -*-
"""
hamilton_ode_jax.py — Pure JAX 0D Hamilton ODE for TMCMC.

Provides θ → φ(t;θ) with jax.grad support for NUTS/HMC.
Uses the same physics as improved_5species_jit.py (NumPy+Numba) but in JAX.

Based on FEM/JAXFEM/core_hamilton_1d.py, simplified for 0D (single node, no diffusion).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def theta_to_matrices(theta):
    """Map 20 params to A(5,5) and b(5). Same layout as improved_5species_jit."""
    A = jnp.zeros((5, 5))
    b = jnp.zeros(5)
    A = A.at[0, 0].set(theta[0])
    A = A.at[0, 1].set(theta[1])
    A = A.at[1, 0].set(theta[1])
    A = A.at[1, 1].set(theta[2])
    b = b.at[0].set(theta[3])
    b = b.at[1].set(theta[4])
    A = A.at[2, 2].set(theta[5])
    A = A.at[2, 3].set(theta[6])
    A = A.at[3, 2].set(theta[6])
    A = A.at[3, 3].set(theta[7])
    b = b.at[2].set(theta[8])
    b = b.at[3].set(theta[9])
    A = A.at[0, 2].set(theta[10])
    A = A.at[2, 0].set(theta[10])
    A = A.at[0, 3].set(theta[11])
    A = A.at[3, 0].set(theta[11])
    A = A.at[1, 2].set(theta[12])
    A = A.at[2, 1].set(theta[12])
    A = A.at[1, 3].set(theta[13])
    A = A.at[3, 1].set(theta[13])
    A = A.at[4, 4].set(theta[14])
    b = b.at[4].set(theta[15])
    A = A.at[0, 4].set(theta[16])
    A = A.at[4, 0].set(theta[16])
    A = A.at[1, 4].set(theta[17])
    A = A.at[4, 1].set(theta[17])
    A = A.at[2, 4].set(theta[18])
    A = A.at[4, 2].set(theta[18])
    A = A.at[3, 4].set(theta[19])
    A = A.at[4, 3].set(theta[19])
    return A, b


def clip_state(g, active_mask):
    """Clip state to valid range."""
    eps = 1e-10
    phi = jnp.clip(g[0:5], eps, 1.0 - eps)
    phi0 = jnp.clip(g[5], eps, 1.0 - eps)
    psi = jnp.clip(g[6:11], eps, 1.0 - eps)
    gamma = jnp.clip(g[11], -1e6, 1e6)
    mask = active_mask.astype(jnp.float64)
    phi = mask * phi
    psi = mask * psi
    return jnp.concatenate([phi, phi0[jnp.newaxis], psi, gamma[jnp.newaxis]])


def residual(g_new, g_prev, params):
    """Hamilton residual Q(g_new, g_prev)."""
    dt = params["dt_h"]
    Kp1 = params["Kp1"]
    Eta = params["Eta"]
    EtaPhi = params["EtaPhi"]
    c = params["c"]
    alpha = params["alpha"]
    K_hill = params["K_hill"]
    n_hill = params["n_hill"]
    A = params["A"]
    b_diag = params["b_diag"]
    active_mask = params["active_mask"]
    eps = 1e-12

    phi_new = g_new[0:5]
    phi0_new = g_new[5]
    psi_new = g_new[6:11]
    gamma_new = g_new[11]
    phi_old = g_prev[0:5]
    phi0_old = g_prev[5]
    psi_old = g_prev[6:11]

    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Ia = A @ (phi_new * psi_new)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[4] == 1).astype(jnp.float64)
    fn = jnp.maximum(phi_new[3] * psi_new[3], 0.0)
    num = fn**n_hill
    den = K_hill**n_hill + num
    factor = jnp.where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[4].set(Ia[4] * factor)

    Q = jnp.zeros(12, dtype=jnp.float64)
    for i in range(5):
        active = active_mask[i] == 1

        def active_phi():
            t1 = Kp1 * (2.0 - 4.0 * phi_new[i]) / ((phi_new[i] - 1.0) ** 3 * phi_new[i] ** 3)
            t2 = (1.0 / Eta[i]) * (
                gamma_new
                + (EtaPhi[i] + Eta[i] * psi_new[i] ** 2) * phidot[i]
                + Eta[i] * phi_new[i] * psi_new[i] * psidot[i]
            )
            t3 = (c / Eta[i]) * psi_new[i] * Ia[i]
            return t1 + t2 - t3

        def inactive_phi():
            return phi_new[i]

        val = jax.lax.cond(active, active_phi, inactive_phi)
        Q = Q.at[i].set(val)

    Q = Q.at[5].set(
        gamma_new + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new**3) + phi0dot
    )

    for i in range(5):
        active = active_mask[i] == 1

        def active_psi():
            t1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0) ** 2 * psi_new[i] ** 3) - (2.0 * Kp1) / (
                (psi_new[i] - 1.0) ** 3 * psi_new[i] ** 2
            )
            t2 = (b_diag[i] * alpha / Eta[i]) * psi_new[i]
            t3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i] ** 2 * psidot[i]
            t4 = (c / Eta[i]) * phi_new[i] * Ia[i]
            return t1 + t2 + t3 - t4

        def inactive_psi():
            return psi_new[i]

        val = jax.lax.cond(active, active_psi, inactive_psi)
        Q = Q.at[6 + i].set(val)

    Q = Q.at[11].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


def newton_step(g_prev, params):
    """One implicit Euler step."""
    active_mask = params["active_mask"]
    n_steps = 6

    def body(carry, _):
        g = clip_state(carry, active_mask)

        def F(gg):
            return residual(gg, g_prev, params)

        Q = F(g)
        J = jax.jacfwd(F)(g)
        delta = jnp.linalg.solve(J, -Q)
        g_next = clip_state(g + delta, active_mask)
        return g_next, None

    g0 = clip_state(g_prev, active_mask)
    g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_steps))
    return g_final


def make_initial_state(phi_init, active_mask):
    """Build g0 from phi_init (5,) or scalar."""
    phi = jnp.asarray(phi_init, dtype=jnp.float64)
    if phi.ndim == 0 or phi.size == 1:
        phi = jnp.full(5, float(phi.flat[0]))
    phi = jnp.where(active_mask == 1, phi, 0.0)
    phi_sum = jnp.minimum(jnp.sum(phi), 0.999999)
    phi = phi * (0.999999 / phi_sum)
    phi0 = 1.0 - jnp.sum(phi)
    psi = jnp.where(active_mask == 1, 0.999, 0.0)
    return jnp.concatenate([phi, phi0[jnp.newaxis], psi, jnp.array([0.0])])


def simulate_0d(
    theta,
    n_steps=2500,
    dt=1e-4,
    phi_init=None,
    K_hill=0.05,
    n_hill=2.0,
    c_const=25.0,
    alpha_const=100.0,
):
    """
    Run 0D Hamilton ODE. Returns phi trajectory (n_steps+1, 5).

    Parameters
    ----------
    theta : (20,) JAX array
    n_steps : int
    dt : float
    phi_init : (5,) or scalar, optional. Default: uniform 0.2
    K_hill, n_hill : Hill gate params
    c_const, alpha_const : Hamilton model constants

    Returns
    -------
    phi_traj : (n_steps+1, 5)
    """
    A, b_diag = theta_to_matrices(theta)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    if phi_init is None:
        phi_init = jnp.full(5, 0.2)
    g0 = make_initial_state(phi_init, active_mask)

    params = {
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5, dtype=jnp.float64),
        "EtaPhi": jnp.ones(5, dtype=jnp.float64),
        "c": c_const,
        "alpha": alpha_const,
        "K_hill": jnp.array(K_hill, dtype=jnp.float64),
        "n_hill": jnp.array(n_hill, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    def body(g, _):
        g_next = newton_step(g, params)
        return g_next, g_next

    _, g_traj = jax.lax.scan(body, g0, jnp.arange(n_steps))
    phi_traj = g_traj[:, 0:5]
    phi_first = g0[0:5][jnp.newaxis, :]
    phi_traj = jnp.concatenate([phi_first, phi_traj], axis=0)
    return phi_traj
