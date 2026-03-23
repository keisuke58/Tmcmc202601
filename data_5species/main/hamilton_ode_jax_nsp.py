# -*- coding: utf-8 -*-
"""
hamilton_ode_jax_nsp.py — Pure JAX 0D Hamilton ODE for N-species TMCMC.

Generalized solver: any number of species N.
State vector: g = [phi(N), phi0, psi(N), gamma] = 2N+2 dimensions.

Parameter layout:
  A matrix (NxN symmetric): N*(N+1)/2 unique entries (upper triangle, column-major)
  b vector (N growth rates)
  Optional: K_hill (if theta has n_A + N + 1 elements)

  Total: N*(N+1)/2 + N [+ 1] parameters

Usage:
    from hamilton_ode_jax_nsp import simulate_0d_nsp, count_params

    n_sp = 11
    n_params = count_params(n_sp)  # 77
    traj = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=2500)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def count_params(n_sp, k_hill_free=False):
    """Number of parameters for N species."""
    n_A = n_sp * (n_sp + 1) // 2
    return n_A + n_sp + (1 if k_hill_free else 0)


def _solve_linear(A, b):
    """Gaussian elimination for (2N+2) x (2N+2) Newton system."""
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
        x = x.at[i].set((row[n] - known) / diag)
    return x


def theta_to_matrices(theta, n_sp):
    """Map theta to symmetric A(N,N) and b(N).

    A is stored as upper triangle, column-major:
      col 0: A[0,0]
      col 1: A[0,1], A[1,1]
      col 2: A[0,2], A[1,2], A[2,2]
      ...
      col j: A[0,j], A[1,j], ..., A[j,j]
    """
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(theta[idx])
            A = A.at[j, i].set(theta[idx])
            idx += 1
    n_A = n_sp * (n_sp + 1) // 2
    b = theta[n_A : n_A + n_sp]
    return A, b


def _clip_state(g, n_sp, active_mask):
    eps = 1e-10
    phi = jnp.clip(g[0:n_sp], eps, 1.0 - eps)
    phi0 = jnp.clip(g[n_sp], eps, 1.0 - eps)
    psi = jnp.clip(g[n_sp + 1 : 2 * n_sp + 1], eps, 1.0 - eps)
    gamma = jnp.clip(g[2 * n_sp + 1], -1e6, 1e6)
    mask = active_mask.astype(jnp.float64)
    phi = mask * phi
    psi = mask * psi
    return jnp.concatenate([phi, phi0[jnp.newaxis], psi, gamma[jnp.newaxis]])


def _residual(g_new, g_prev, params):
    n_sp = params["n_sp"]
    dt = params["dt_h"]
    Kp1 = params["Kp1"]
    Eta = params["Eta"]
    EtaPhi = params["EtaPhi"]
    c = params["c"]
    alpha = params["alpha"]
    K_hill = params["K_hill"]
    n_hill = params["n_hill"]
    hill_gate_species = params["hill_gate_species"]  # (gated_sp, gating_sp) or (-1,-1)
    A = params["A"]
    b_diag = params["b_diag"]
    active_mask = params["active_mask"]
    eps = 1e-12
    n_state = 2 * n_sp + 2

    phi_new = g_new[0:n_sp]
    phi0_new = g_new[n_sp]
    psi_new = g_new[n_sp + 1 : 2 * n_sp + 1]
    gamma_new = g_new[2 * n_sp + 1]
    phi_old = g_prev[0:n_sp]
    phi0_old = g_prev[n_sp]
    psi_old = g_prev[n_sp + 1 : 2 * n_sp + 1]

    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Ia = A @ (phi_new * psi_new)

    # Hill gate (optional)
    gated = hill_gate_species[0]
    gating = hill_gate_species[1]
    hill_active = (K_hill > 1e-9) * (gated >= 0)
    fn_val = jnp.maximum(phi_new[gating] * psi_new[gating], 0.0)
    num = fn_val**n_hill
    den = K_hill**n_hill + num
    factor = jnp.where(den > eps, num / den, 0.0) * hill_active
    # Apply gate only to gated species
    Ia_gated = Ia[gated] * factor
    Ia = jnp.where(jnp.arange(n_sp) == gated, Ia_gated, Ia)

    Q = jnp.zeros(n_state, dtype=jnp.float64)
    for i in range(n_sp):
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

        Q = Q.at[i].set(jax.lax.cond(active, active_phi, inactive_phi))

    Q = Q.at[n_sp].set(
        gamma_new + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new**3) + phi0dot
    )

    for i in range(n_sp):
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

        Q = Q.at[n_sp + 1 + i].set(jax.lax.cond(active, active_psi, inactive_psi))

    Q = Q.at[2 * n_sp + 1].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


def _newton_step(g_prev, params):
    n_sp = params["n_sp"]
    active_mask = params["active_mask"]
    n_newton = 6

    def body(carry, _):
        g = _clip_state(carry, n_sp, active_mask)

        def F(gg):
            return _residual(gg, g_prev, params)

        Q = F(g)
        J = jax.jacfwd(F)(g)
        delta = _solve_linear(J, -Q)
        return _clip_state(g + delta, n_sp, active_mask), None

    g0 = _clip_state(g_prev, n_sp, active_mask)
    g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_newton))
    return g_final


def _make_initial_state(phi_init, n_sp, active_mask):
    phi = jnp.asarray(phi_init, dtype=jnp.float64)
    if phi.ndim == 0 or phi.size == 1:
        phi = jnp.full(n_sp, float(phi.flat[0]))
    phi = jnp.where(active_mask == 1, phi, 0.0)
    phi_sum = jnp.minimum(jnp.sum(phi), 0.999999)
    phi = phi * (0.999999 / phi_sum)
    phi0 = 1.0 - jnp.sum(phi)
    psi = jnp.where(active_mask == 1, 0.999, 0.0)
    return jnp.concatenate([phi, phi0[jnp.newaxis], psi, jnp.array([0.0])])


def simulate_0d_nsp(
    theta,
    n_sp,
    n_steps=2500,
    dt=1e-4,
    phi_init=None,
    K_hill=0.0,
    n_hill=2.0,
    c_const=25.0,
    alpha_const=100.0,
    hill_gate_species=None,
    perturbation=None,
):
    """
    Run 0D Hamilton ODE for N species. Returns phibar trajectory (n_steps+1, N).

    Parameters
    ----------
    theta : JAX array, length = N*(N+1)/2 + N [+ 1 for K_hill]
    n_sp : int, number of species
    hill_gate_species : tuple (gated, gating) or None.
    perturbation : dict or None.
        If provided: {"step_start": int, "step_end": int, "alpha_perturb": float}
        alpha = alpha_perturb during [step_start, step_end), alpha_const otherwise.
        Represents antibiotic effect (low alpha = growth suppression).
    """
    n_A = n_sp * (n_sp + 1) // 2
    A, b_diag = theta_to_matrices(theta[: n_A + n_sp], n_sp)

    # K_hill from theta if extra param present
    has_k_hill = theta.shape[0] > n_A + n_sp
    K_hill_val = jnp.where(has_k_hill, theta[n_A + n_sp], K_hill)

    active_mask = jnp.ones(n_sp, dtype=jnp.int64)

    if phi_init is None:
        phi_init = jnp.full(n_sp, 1.0 / n_sp)
    g0 = _make_initial_state(phi_init, n_sp, active_mask)

    if hill_gate_species is None:
        hill_gate_species = (-1, 0)  # disabled

    # Perturbation schedule
    if perturbation is not None:
        p_start = perturbation["step_start"]
        p_end = perturbation["step_end"]
        alpha_p = perturbation["alpha_perturb"]
    else:
        p_start = -1
        p_end = -1
        alpha_p = alpha_const

    base_params = {
        "n_sp": n_sp,
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(n_sp, dtype=jnp.float64),
        "EtaPhi": jnp.ones(n_sp, dtype=jnp.float64),
        "c": c_const,
        "alpha": alpha_const,
        "K_hill": jnp.array(K_hill_val, dtype=jnp.float64),
        "n_hill": jnp.array(n_hill, dtype=jnp.float64),
        "hill_gate_species": jnp.array(hill_gate_species, dtype=jnp.int32),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    def body(carry, step_idx):
        g = carry
        # Time-varying alpha for antibiotic perturbation
        alpha_t = jnp.where(
            (step_idx >= p_start) & (step_idx < p_end),
            alpha_p,
            alpha_const,
        )
        params = {**base_params, "alpha": alpha_t}
        g_next = _newton_step(g, params)
        return g_next, g_next

    _, g_traj = jax.lax.scan(body, g0, jnp.arange(n_steps))
    # Observable: phibar = phi * psi
    phi = g_traj[:, 0:n_sp]
    psi = g_traj[:, n_sp + 1 : 2 * n_sp + 1]
    phibar = phi * psi
    phi0_ic = g0[0:n_sp]
    psi0_ic = g0[n_sp + 1 : 2 * n_sp + 1]
    phibar0 = (phi0_ic * psi0_ic)[jnp.newaxis, :]
    return jnp.concatenate([phibar0, phibar], axis=0)
