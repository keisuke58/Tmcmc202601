# -*- coding: utf-8 -*-
"""
hamilton_ode_jax_6sp.py — Pure JAX 0D Hamilton ODE for 6-species TMCMC.

Extension of hamilton_ode_jax.py from 5 to 6 species.
State vector: g = [phi(6), phi0, psi(6), gamma] = 14 dimensions.

Species order: [So, An, Aa, Vp, Fn, Pg] (indices 0-5)
Hill gate: Pg (index 5) gated by Fn (index 4).

Parameter layout (27 total):
  A matrix (6x6 symmetric, 21 unique):
    theta[0:6]   = A[0,0], A[0,1], A[1,1], A[0,2], A[1,2], A[2,2]   (So-An-Aa block)
    theta[6:10]  = A[0,3], A[1,3], A[2,3], A[3,3]                     (Vp col)
    theta[10:15] = A[0,4], A[1,4], A[2,4], A[3,4], A[4,4]             (Fn col)
    theta[15:21] = A[0,5], A[1,5], A[2,5], A[3,5], A[4,5], A[5,5]     (Pg col)
  b vector (6 growth rates):
    theta[21:27] = b[0], b[1], b[2], b[3], b[4], b[5]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N_SP = 6
N_STATE = 2 * N_SP + 2  # 14
N_PARAMS_BASE = 27  # 21 (A) + 6 (b)
N_PARAMS = 28  # 27 + K_hill (free)


def _solve_linear_pure_jax(A, b):
    """Gaussian elimination (no pivoting) for 14x14 Newton system."""
    n = A.shape[0]
    Ab = jnp.concatenate([A, b[:, jnp.newaxis]], axis=1)

    def elim_step(carry, k):
        Ab_curr = carry
        pivot = Ab_curr[k, k]
        pivot_safe = jnp.where(jnp.abs(pivot) > 1e-14, pivot, 1.0)
        row_k = Ab_curr[k, :]
        scale = Ab_curr[:, k] / pivot_safe
        scale = jnp.where(jnp.arange(n) > k, scale, 0.0)
        Ab_new = Ab_curr - scale[:, jnp.newaxis] * row_k
        return Ab_new, None

    Ab_final, _ = jax.lax.scan(elim_step, Ab, jnp.arange(n - 1))
    x = jnp.zeros(n)
    for i in range(n - 1, -1, -1):
        row = Ab_final[i, :]
        known = jnp.dot(row[i + 1 : n], x[i + 1 : n])
        diag = jnp.where(jnp.abs(row[i]) > 1e-14, row[i], 1.0)
        x_i = (row[n] - known) / diag
        x = x.at[i].set(x_i)
    return x


def theta_to_matrices_6sp(theta):
    """Map 27 params to A(6,6) symmetric and b(6)."""
    A = jnp.zeros((N_SP, N_SP))
    b = jnp.zeros(N_SP)

    # Upper triangle of A (row-major, column by column)
    # Column 0 (So): A[0,0]
    A = A.at[0, 0].set(theta[0])
    # Column 1 (An): A[0,1], A[1,1]
    A = A.at[0, 1].set(theta[1]); A = A.at[1, 0].set(theta[1])
    A = A.at[1, 1].set(theta[2])
    # Column 2 (Aa): A[0,2], A[1,2], A[2,2]
    A = A.at[0, 2].set(theta[3]); A = A.at[2, 0].set(theta[3])
    A = A.at[1, 2].set(theta[4]); A = A.at[2, 1].set(theta[4])
    A = A.at[2, 2].set(theta[5])
    # Column 3 (Vp): A[0,3], A[1,3], A[2,3], A[3,3]
    A = A.at[0, 3].set(theta[6]); A = A.at[3, 0].set(theta[6])
    A = A.at[1, 3].set(theta[7]); A = A.at[3, 1].set(theta[7])
    A = A.at[2, 3].set(theta[8]); A = A.at[3, 2].set(theta[8])
    A = A.at[3, 3].set(theta[9])
    # Column 4 (Fn): A[0,4], A[1,4], A[2,4], A[3,4], A[4,4]
    A = A.at[0, 4].set(theta[10]); A = A.at[4, 0].set(theta[10])
    A = A.at[1, 4].set(theta[11]); A = A.at[4, 1].set(theta[11])
    A = A.at[2, 4].set(theta[12]); A = A.at[4, 2].set(theta[12])
    A = A.at[3, 4].set(theta[13]); A = A.at[4, 3].set(theta[13])
    A = A.at[4, 4].set(theta[14])
    # Column 5 (Pg): A[0,5], A[1,5], A[2,5], A[3,5], A[4,5], A[5,5]
    A = A.at[0, 5].set(theta[15]); A = A.at[5, 0].set(theta[15])
    A = A.at[1, 5].set(theta[16]); A = A.at[5, 1].set(theta[16])
    A = A.at[2, 5].set(theta[17]); A = A.at[5, 2].set(theta[17])
    A = A.at[3, 5].set(theta[18]); A = A.at[5, 3].set(theta[18])
    A = A.at[4, 5].set(theta[19]); A = A.at[5, 4].set(theta[19])
    A = A.at[5, 5].set(theta[20])
    # Growth rates b
    for i in range(N_SP):
        b = b.at[i].set(theta[21 + i])
    return A, b


def clip_state_6sp(g, active_mask):
    eps = 1e-10
    phi = jnp.clip(g[0:N_SP], eps, 1.0 - eps)
    phi0 = jnp.clip(g[N_SP], eps, 1.0 - eps)
    psi = jnp.clip(g[N_SP + 1 : 2 * N_SP + 1], eps, 1.0 - eps)
    gamma = jnp.clip(g[2 * N_SP + 1], -1e6, 1e6)
    mask = active_mask.astype(jnp.float64)
    phi = mask * phi
    psi = mask * psi
    return jnp.concatenate([phi, phi0[jnp.newaxis], psi, gamma[jnp.newaxis]])


def residual_6sp(g_new, g_prev, params):
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

    phi_new = g_new[0:N_SP]
    phi0_new = g_new[N_SP]
    psi_new = g_new[N_SP + 1 : 2 * N_SP + 1]
    gamma_new = g_new[2 * N_SP + 1]
    phi_old = g_prev[0:N_SP]
    phi0_old = g_prev[N_SP]
    psi_old = g_prev[N_SP + 1 : 2 * N_SP + 1]

    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Ia = A @ (phi_new * psi_new)

    # Hill gate for Pg (index 5), gated by Fn (index 4)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[5] == 1).astype(jnp.float64)
    fn = jnp.maximum(phi_new[4] * psi_new[4], 0.0)
    num = fn**n_hill
    den = K_hill**n_hill + num
    factor = jnp.where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[5].set(Ia[5] * factor)

    Q = jnp.zeros(N_STATE, dtype=jnp.float64)
    for i in range(N_SP):
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

    Q = Q.at[N_SP].set(
        gamma_new + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new**3) + phi0dot
    )

    for i in range(N_SP):
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
        Q = Q.at[N_SP + 1 + i].set(val)

    Q = Q.at[2 * N_SP + 1].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


def newton_step_6sp(g_prev, params):
    active_mask = params["active_mask"]
    n_newton = 6

    def body(carry, _):
        g = clip_state_6sp(carry, active_mask)

        def F(gg):
            return residual_6sp(gg, g_prev, params)

        Q = F(g)
        J = jax.jacfwd(F)(g)
        delta = _solve_linear_pure_jax(J, -Q)
        g_next = clip_state_6sp(g + delta, active_mask)
        return g_next, None

    g0 = clip_state_6sp(g_prev, active_mask)
    g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_newton))
    return g_final


def make_initial_state_6sp(phi_init, active_mask):
    phi = jnp.asarray(phi_init, dtype=jnp.float64)
    if phi.ndim == 0 or phi.size == 1:
        phi = jnp.full(N_SP, float(phi.flat[0]))
    phi = jnp.where(active_mask == 1, phi, 0.0)
    phi_sum = jnp.minimum(jnp.sum(phi), 0.999999)
    phi = phi * (0.999999 / phi_sum)
    phi0 = 1.0 - jnp.sum(phi)
    psi = jnp.where(active_mask == 1, 0.999, 0.0)
    return jnp.concatenate([phi, phi0[jnp.newaxis], psi, jnp.array([0.0])])


def simulate_0d_6sp(
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
    Run 0D Hamilton ODE for 6 species. Returns phibar trajectory (n_steps+1, 6).

    Parameters
    ----------
    theta : (27,) or (28,) JAX array.
            If 28 params, theta[27] = K_hill (overrides K_hill kwarg).
    """
    A, b_diag = theta_to_matrices_6sp(theta[:N_PARAMS_BASE])
    # If theta has 28 elements, use theta[27] as K_hill
    K_hill_val = jnp.where(theta.shape[0] > N_PARAMS_BASE, theta[N_PARAMS_BASE], K_hill)
    active_mask = jnp.ones(N_SP, dtype=jnp.int64)

    if phi_init is None:
        phi_init = jnp.full(N_SP, 1.0 / N_SP)
    g0 = make_initial_state_6sp(phi_init, active_mask)

    params = {
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(N_SP, dtype=jnp.float64),
        "EtaPhi": jnp.ones(N_SP, dtype=jnp.float64),
        "c": c_const,
        "alpha": alpha_const,
        "K_hill": jnp.array(K_hill_val, dtype=jnp.float64),
        "n_hill": jnp.array(n_hill, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    def body(g, _):
        g_next = newton_step_6sp(g, params)
        return g_next, g_next

    _, g_traj = jax.lax.scan(body, g0, jnp.arange(n_steps))
    # Observable: phibar = phi * psi (species abundance fraction)
    phi = g_traj[:, 0:N_SP]
    psi = g_traj[:, N_SP + 1 : 2 * N_SP + 1]
    phibar = phi * psi
    # First timepoint
    phi0 = g0[0:N_SP]
    psi0 = g0[N_SP + 1 : 2 * N_SP + 1]
    phibar0 = (phi0 * psi0)[jnp.newaxis, :]
    phibar_traj = jnp.concatenate([phibar0, phibar], axis=0)
    return phibar_traj
