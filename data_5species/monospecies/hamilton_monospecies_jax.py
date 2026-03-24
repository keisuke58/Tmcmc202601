# -*- coding: utf-8 -*-
"""
hamilton_monospecies_jax.py — JAX Hamilton ODE solver for monospecies biofilm (Liu 2019).

GPU-compatible version of hamilton_monospecies.py.
Avoids lax.scan to prevent LLVM OOM on large step counts (4000+).
Uses Python loop + JIT'd single-step function instead.

State: g = [φ₁, φ₀, ψ₁, γ]  (4D)
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


# ── Hamilton residual & Newton solver (JAX) ──────────────────────────────────

@jit
def penalty_phi(x, Kp):
    return Kp * (2.0 - 4.0 * x) / ((x - 1.0)**3 * x**3)


@jit
def penalty_psi(x, Kp):
    return Kp * (2.0 - 4.0 * x) / ((x - 1.0)**3 * x**3)


@jit
def clip4(g):
    eps = 1e-10
    return jnp.array([
        jnp.clip(g[0], eps, 1.0 - eps),
        jnp.clip(g[1], eps, 1.0 - eps),
        jnp.clip(g[2], eps, 1.0 - eps),
        jnp.clip(g[3], -1e6, 1e6),
    ])


@jit
def residual_4d(g_new, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b):
    phi, phi0, psi, gamma = g_new
    phi_old, phi0_old, psi_old, _ = g_prev

    phidot = (phi - phi_old) / dt
    phi0dot = (phi0 - phi0_old) / dt
    psidot = (psi - psi_old) / dt

    Ia = sigma * phi * psi

    Q0 = (penalty_phi(phi, Kp)
           + (1.0/eta) * (gamma + (eta_phi + eta * psi**2) * phidot
                          + eta * phi * psi * psidot)
           - (c/eta) * psi * Ia)

    Q1 = gamma + penalty_phi(phi0, Kp) + phi0dot

    Q2 = (penalty_psi(psi, Kp)
           + (b * alpha / eta) * psi
           + phi * psi * phidot + phi**2 * psidot
           - (c/eta) * phi * Ia)

    Q3 = phi + phi0 - 1.0
    return jnp.array([Q0, Q1, Q2, Q3])


@jit
def residual_4d_dynamic_c(g_new, g_prev, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b):
    phi, phi0, psi, gamma = g_new
    phi_old, phi0_old, psi_old, _ = g_prev

    phidot = (phi - phi_old) / dt
    phi0dot = (phi0 - phi0_old) / dt
    psidot = (psi - psi_old) / dt

    Ia = sigma * phi * psi
    c = c_scale * (1.0 - phi * psi)

    Q0 = (penalty_phi(phi, Kp)
           + (1.0/eta) * (gamma + (eta_phi + eta * psi**2) * phidot
                          + eta * phi * psi * psidot)
           - (c/eta) * psi * Ia)

    Q1 = gamma + penalty_phi(phi0, Kp) + phi0dot

    Q2 = (penalty_psi(psi, Kp)
           + (b * alpha / eta) * psi
           + phi * psi * phidot + phi**2 * psidot
           - (c/eta) * phi * Ia)

    Q3 = phi + phi0 - 1.0
    return jnp.array([Q0, Q1, Q2, Q3])


def _build_jacobian_col(g, j, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b, Q):
    """Numerical Jacobian column j via forward FD."""
    eps_fd = 1e-8
    gp = g.at[j].add(eps_fd)
    gp = clip4(gp)
    Qp = residual_4d(gp, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b)
    return (Qp - Q) / eps_fd


@jit
def _newton_iteration(g, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b):
    """One Newton iteration (JIT'd)."""
    g = clip4(g)
    Q = residual_4d(g, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b)

    # Build Jacobian column by column
    eps_fd = 1e-8
    cols = []
    for j in range(4):
        gp = g.at[j].add(eps_fd)
        gp = clip4(gp)
        Qp = residual_4d(gp, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b)
        cols.append((Qp - Q) / eps_fd)
    J = jnp.column_stack(cols)

    # lstsq instead of solve — robust to singular Jacobian (matches NumPy fallback)
    delta, _, _, _ = jnp.linalg.lstsq(J, -Q, rcond=None)
    g_new = clip4(g + delta)
    return g_new


@partial(jit, static_argnums=(9,))
def newton_step_4d(g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b, n_iter):
    """One implicit Euler step: n_iter Newton iterations (JIT'd)."""
    g = g_prev

    def body(_, g):
        return _newton_iteration(g, g_prev, sigma, dt, Kp, eta, eta_phi, c, alpha, b)

    g = jax.lax.fori_loop(0, n_iter, body, g)
    return g


@jit
def _newton_iteration_dynamic_c(g, g_prev, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b):
    g = clip4(g)
    Q = residual_4d_dynamic_c(g, g_prev, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b)

    eps_fd = 1e-8
    cols = []
    for j in range(4):
        gp = g.at[j].add(eps_fd)
        gp = clip4(gp)
        Qp = residual_4d_dynamic_c(gp, g_prev, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b)
        cols.append((Qp - Q) / eps_fd)
    J = jnp.column_stack(cols)

    delta, _, _, _ = jnp.linalg.lstsq(J, -Q, rcond=None)
    g_new = clip4(g + delta)
    return g_new


@partial(jit, static_argnums=(9,))
def newton_step_4d_dynamic_c(g_prev, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b, n_iter):
    g = g_prev

    def body(_, g):
        return _newton_iteration_dynamic_c(g, g_prev, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b)

    g = jax.lax.fori_loop(0, n_iter, body, g)
    return g


@partial(jit, static_argnums=(1, 10))
def simulate_monospecies_phibar_hist_dynamic_c(sigma, n_steps, dt=1e-4, phi_init=0.1,
                                               Kp=1e-4, eta=1.0, eta_phi=1.0,
                                               c_scale=10.0, alpha=0.0, b=0.0, n_newton=6):
    """
    Simulate monospecies Hamilton ODE with dynamic nutrient coupling:
      c(t) = c_scale * (1 - phi*psi)
    """
    sigma = jnp.float64(sigma)
    dt = jnp.float64(dt)
    Kp = jnp.float64(Kp)
    eta = jnp.float64(eta)
    eta_phi = jnp.float64(eta_phi)
    c_scale = jnp.float64(c_scale)
    alpha = jnp.float64(alpha)
    b = jnp.float64(b)

    phi0_init = 1.0 - phi_init
    psi_init = 0.999
    g0 = jnp.array([phi_init, phi0_init, psi_init, 0.0])

    phibar_hist0 = jnp.empty((n_steps + 1,), dtype=jnp.float64)
    phibar_hist0 = phibar_hist0.at[0].set(jnp.float64(phi_init * psi_init))

    def body(i, carry):
        g, phibar_hist = carry
        g = newton_step_4d_dynamic_c(g, sigma, dt, Kp, eta, eta_phi, c_scale, alpha, b, n_newton)
        phibar_hist = phibar_hist.at[i + 1].set(g[0] * g[2])
        return g, phibar_hist

    _, phibar_hist = jax.lax.fori_loop(0, n_steps, body, (g0, phibar_hist0))
    return phibar_hist


@partial(jit, static_argnums=(1, 10))
def simulate_monospecies_phibar_hist_dynamic_c_sigma_series(sigmas, n_steps, dt=1e-4, phi_init=0.1,
                                                            Kp=1e-4, eta=1.0, eta_phi=1.0,
                                                            c_scale=10.0, alpha=0.0, b=0.0, n_newton=6):
    sigmas = jnp.asarray(sigmas, dtype=jnp.float64)
    dt = jnp.float64(dt)
    Kp = jnp.float64(Kp)
    eta = jnp.float64(eta)
    eta_phi = jnp.float64(eta_phi)
    c_scale = jnp.float64(c_scale)
    alpha = jnp.float64(alpha)
    b = jnp.float64(b)

    phi0_init = 1.0 - phi_init
    psi_init = 0.999
    g0 = jnp.array([phi_init, phi0_init, psi_init, 0.0])

    phibar_hist0 = jnp.empty((n_steps + 1,), dtype=jnp.float64)
    phibar_hist0 = phibar_hist0.at[0].set(jnp.float64(phi_init * psi_init))

    def body(i, carry):
        g, phibar_hist = carry
        g = newton_step_4d_dynamic_c(g, sigmas[i], dt, Kp, eta, eta_phi, c_scale, alpha, b, n_newton)
        phibar_hist = phibar_hist.at[i + 1].set(g[0] * g[2])
        return g, phibar_hist

    _, phibar_hist = jax.lax.fori_loop(0, n_steps, body, (g0, phibar_hist0))
    return phibar_hist


@partial(jit, static_argnums=(1, 10))
def simulate_monospecies_phibar_hist(sigma, n_steps, dt=1e-4, phi_init=0.1,
                                     Kp=1e-4, eta=1.0, eta_phi=1.0,
                                     c=100.0, alpha=0.0, b=0.0, n_newton=6):
    sigma = jnp.float64(sigma)
    dt = jnp.float64(dt)
    Kp = jnp.float64(Kp)
    eta = jnp.float64(eta)
    eta_phi = jnp.float64(eta_phi)
    c = jnp.float64(c)
    alpha = jnp.float64(alpha)
    b = jnp.float64(b)

    phi0_init = 1.0 - phi_init
    psi_init = 0.999
    g0 = jnp.array([phi_init, phi0_init, psi_init, 0.0])

    phibar_hist0 = jnp.empty((n_steps + 1,), dtype=jnp.float64)
    phibar_hist0 = phibar_hist0.at[0].set(jnp.float64(phi_init * psi_init))

    def body(i, carry):
        g, phibar_hist = carry
        g = newton_step_4d(g, sigma, dt, Kp, eta, eta_phi, c, alpha, b, n_newton)
        phibar_hist = phibar_hist.at[i + 1].set(g[0] * g[2])
        return g, phibar_hist

    _, phibar_hist = jax.lax.fori_loop(0, n_steps, body, (g0, phibar_hist0))
    return phibar_hist


def simulate_monospecies(sigma, n_steps, dt=1e-4, phi_init=0.1,
                         Kp=1e-4, eta=1.0, eta_phi=1.0,
                         c=100.0, alpha=0.0, b=0.0, n_newton=6):
    """
    Run monospecies Hamilton ODE (JAX, GPU-compatible).

    Uses Python loop over time steps (avoids lax.scan LLVM OOM).
    Each step is JIT'd for GPU acceleration.

    Returns
    -------
    phi_hist : (n_steps+1,) jnp.array
    psi_hist : (n_steps+1,) jnp.array
    phibar_hist : (n_steps+1,) jnp.array
    """
    sigma = jnp.float64(sigma)
    dt = jnp.float64(dt)
    Kp = jnp.float64(Kp)
    eta = jnp.float64(eta)
    eta_phi = jnp.float64(eta_phi)
    c = jnp.float64(c)
    alpha = jnp.float64(alpha)
    b = jnp.float64(b)

    phi0_init = 1.0 - phi_init
    psi_init = 0.999
    g = jnp.array([phi_init, phi0_init, psi_init, 0.0])

    phi_list = [phi_init]
    psi_list = [psi_init]

    for step in range(n_steps):
        g = newton_step_4d(g, sigma, dt, Kp, eta, eta_phi, c, alpha, b, n_newton)
        phi_list.append(g[0])
        psi_list.append(g[2])

    phi_hist = jnp.array(phi_list)
    psi_hist = jnp.array(psi_list)
    phibar_hist = phi_hist * psi_hist
    return phi_hist, psi_hist, phibar_hist


# ── Batched simulation (vmap over sigma) ─────────────────────────────────────

def simulate_monospecies_batch(sigmas, n_steps, dt=1e-4, phi_init=0.1,
                               Kp=1e-4, eta=1.0, eta_phi=1.0,
                               c=100.0, alpha=0.0, b=0.0, n_newton=6):
    """
    Batch simulation over multiple sigma values using vmap.
    Useful for parameter sweeps on GPU.

    Parameters
    ----------
    sigmas : array-like, shape (B,)

    Returns
    -------
    phi_hist : (B, n_steps+1)
    psi_hist : (B, n_steps+1)
    phibar_hist : (B, n_steps+1)
    """
    dt = jnp.float64(dt)
    Kp = jnp.float64(Kp)
    eta = jnp.float64(eta)
    eta_phi = jnp.float64(eta_phi)
    c_val = jnp.float64(c)
    alpha = jnp.float64(alpha)
    b = jnp.float64(b)

    phi0_init = 1.0 - phi_init
    psi_init = 0.999

    # JIT'd single step that takes sigma as arg
    @jit
    def one_step(g, sigma):
        return newton_step_4d(g, sigma, dt, Kp, eta, eta_phi, c_val, alpha, b, n_newton)

    sigmas = jnp.asarray(sigmas)
    B = sigmas.shape[0]

    g_batch = jnp.tile(
        jnp.array([phi_init, phi0_init, psi_init, 0.0]),
        (B, 1)
    )

    phi_list = [jnp.full(B, phi_init)]
    psi_list = [jnp.full(B, psi_init)]

    # vmap over batch dim for each time step
    vmapped_step = jax.vmap(one_step, in_axes=(0, 0))

    for step in range(n_steps):
        g_batch = vmapped_step(g_batch, sigmas)
        phi_list.append(g_batch[:, 0])
        psi_list.append(g_batch[:, 2])

    phi_hist = jnp.stack(phi_list, axis=1)
    psi_hist = jnp.stack(psi_list, axis=1)
    phibar_hist = phi_hist * psi_hist
    return phi_hist, psi_hist, phibar_hist


# ── RMSE & Optimization (JAX) ───────────────────────────────────────────────

def compute_rmse(sigma, exp_time, exp_cfu, n_steps,
                 Kp=1e-4, eta=1.0, eta_phi=1.0, c=100.0,
                 time_to_step=10.0):
    """RMSE with optimal affine mapping φ̄(t) → log₁₀(CFU/mL)."""
    try:
        _, _, phibar = simulate_monospecies(
            sigma, n_steps, Kp=Kp, eta=eta, eta_phi=eta_phi, c=c
        )
    except Exception:
        return 1e10

    step_idx = jnp.clip((exp_time * time_to_step).astype(int), 0, n_steps)
    sim_vals = phibar[step_idx]

    X = jnp.column_stack([sim_vals, jnp.ones_like(sim_vals)])
    coeff, _, _, _ = jnp.linalg.lstsq(X, exp_cfu, rcond=None)
    predicted = X @ coeff
    return float(jnp.sqrt(jnp.mean((predicted - exp_cfu)**2)))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time as time_mod
    import numpy as np

    # Enable float64
    jax.config.update('jax_enable_x64', True)

    print(f"JAX devices: {jax.devices()}")

    # Sanity check: 500 steps
    print("\n[Sanity] σ=50, 500 steps...")
    t0 = time_mod.time()
    phi, psi, phibar = simulate_monospecies(50.0, 500)
    # Block until done (GPU async)
    phi.block_until_ready()
    t_first = time_mod.time() - t0
    print(f"  First call (incl. JIT compile): {t_first:.2f}s")
    print(f"  φ: {phi[0]:.4f} → {phi[250]:.4f} → {phi[500]:.4f}")
    print(f"  ψ: {psi[0]:.4f} → {psi[250]:.4f} → {psi[500]:.4f}")
    print(f"  φ̄: {phibar[0]:.4f} → {phibar[250]:.4f} → {phibar[500]:.4f}")

    # Second call (warm cache)
    t0 = time_mod.time()
    phi2, psi2, phibar2 = simulate_monospecies(50.0, 500)
    phi2.block_until_ready()
    t_warm = time_mod.time() - t0
    print(f"  Warm call: {t_warm:.2f}s")

    # 4000 steps test (the problematic case for lax.scan)
    print("\n[Test] σ=25, 4000 steps (was OOM with lax.scan)...")
    t0 = time_mod.time()
    phi4k, psi4k, phibar4k = simulate_monospecies(25.0, 4000)
    phi4k.block_until_ready()
    print(f"  Done in {time_mod.time()-t0:.2f}s")
    print(f"  φ: {phi4k[0]:.4f} → {phi4k[2000]:.4f} → {phi4k[4000]:.4f}")

    # Batch test
    print("\n[Batch] 8 sigmas × 500 steps (vmap)...")
    sigmas = np.array([25, 50, 50, 70, 70, 110, 110, 50], dtype=np.float64)
    t0 = time_mod.time()
    phi_b, psi_b, phibar_b = simulate_monospecies_batch(sigmas, 500)
    phi_b.block_until_ready()
    print(f"  Done in {time_mod.time()-t0:.2f}s")
    for i, s in enumerate(sigmas):
        print(f"  σ={s:>3.0f}: φ̄_final={phibar_b[i, -1]:.4f}")
