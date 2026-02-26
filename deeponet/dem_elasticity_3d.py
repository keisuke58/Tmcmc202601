#!/usr/bin/env python3
"""
dem_elasticity_3d.py — Deep Energy Method (DEM) for 3D elasticity.

Instead of solving the PDE (Strong form) like PINN, this minimizes the
Total Potential Energy (Variational form):
    Π(u) = ∫_Ω W(ε) dΩ - ∫_Γt t·u dΓ

Where W(ε) is the strain energy density:
    W(ε) = 1/2 σ : ε = 1/2 (λ(tr ε)² + 2μ tr(ε²))

Benefits:
    - Only requires 1st order derivatives (PINN requires 2nd order).
    - Physically more robust (Minimizing energy vs minimizing residual).
    - Naturally handles Neumann BCs (Traction) in the energy functional.

Usage:
    python dem_elasticity_3d.py train --epochs 2000
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

jax.config.update("jax_enable_x64", False)

# ============================================================
# Constants
# ============================================================
E_MAX = 1000.0  # Pa
E_MIN = 10.0  # Pa
NU = 0.30  # Poisson's ratio
W = 1.0  # width [mm]
H = 0.2  # height [mm]
D = 1.0  # depth [mm]
P_APPLIED = 1.0  # applied pressure [Pa]


# ============================================================
# Material model
# ============================================================
def di_to_E(di, di_scale=1.0, n=2.0):
    r = jnp.clip(di / di_scale, 0.0, 1.0)
    return E_MAX * (1.0 - r) ** n + E_MIN * r


# ============================================================
# Model (Same MLP as PINN)
# ============================================================
class ElasticityNetwork(eqx.Module):
    fourier_B: jnp.ndarray
    layers: list
    output_scale: jnp.ndarray

    def __init__(self, hidden=128, n_layers=5, n_fourier=32, *, key):
        k1, k2 = jr.split(key)
        self.fourier_B = jr.normal(k1, (3, n_fourier)) * 2.0
        in_dim = 2 * n_fourier + 1
        keys = jr.split(k2, n_layers + 1)
        self.layers = []
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 3
            self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))
            in_dim = hidden
        self.output_scale = jnp.array([1e-3, 1e-3, 1e-3])

    def __call__(self, x, y, z, E_norm):
        xyz = jnp.array([x, y, z])
        proj = self.fourier_B.T @ xyz
        ff = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])
        inp = jnp.concatenate([ff, jnp.array([E_norm])])
        h = inp
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        raw = self.layers[-1](h)
        # Hard BC: u = 0 at y = 0
        return raw * self.output_scale * y


# ============================================================
# Energy Loss (Variational Principle)
# ============================================================
def strain_energy_density(model, x, y, z, E_local):
    E_norm = E_local / E_MAX
    lam = E_local * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E_local / (2 * (1 + NU))

    def u_fn(xyz):
        return model(xyz[0], xyz[1], xyz[2], E_norm)

    # Gradient: F = I + ∇u (Linear elasticity: just ∇u)
    J = jax.jacobian(u_fn)(jnp.array([x, y, z]))

    # Linear Strain: ε = 0.5 (∇u + ∇u^T)
    eps = 0.5 * (J + J.T)

    # Strain Energy Density: W = 1/2 λ (tr ε)^2 + μ tr(ε^2)
    trace_eps = jnp.trace(eps)
    trace_eps2 = jnp.trace(eps @ eps)

    W_energy = 0.5 * lam * (trace_eps**2) + mu * trace_eps2
    return W_energy


@partial(jax.jit, static_argnums=(4,))
def compute_energy_loss(model, domain_pts, boundary_top_pts, volume_scale, area_scale):
    """
    Total Potential Energy Π = U_int - W_ext
    U_int = ∫ W dΩ  (Strain Energy)
    W_ext = ∫ t·u dΓ (External Work)
    """

    # 1. Internal Strain Energy (Monte Carlo Integration)
    def single_energy(pt):
        # pt: (x, y, z, E)
        return strain_energy_density(model, pt[0], pt[1], pt[2], pt[3])

    W_vals = jax.vmap(single_energy)(domain_pts)
    U_int = jnp.mean(W_vals) * volume_scale

    # 2. External Work (Traction on Top Surface)
    # Traction vector t = (0, -P, 0)
    # Work density = t · u = -P * u_y
    def single_work(pt):
        # pt: (x, H, z, E)
        u = model(pt[0], pt[1], pt[2], pt[3] / E_MAX)
        # u dot t = u_x*0 + u_y*(-P) + u_z*0
        return u[1] * (-P_APPLIED)

    Work_vals = jax.vmap(single_work)(boundary_top_pts)
    W_ext = jnp.mean(Work_vals) * area_scale

    # Total Potential Energy
    # Note: We minimize Π = U_int - W_ext (Potential of loads is -W_ext)
    # Actually W_ext definition: Work done BY external forces.
    # Potential Energy of loads V = - ∫ t·u
    # So Π = U_int + V = U_int - ∫ t·u

    loss = U_int + W_ext  # W_ext contains the negative sign from dot product?
    # Wait, Work = Force * Disp.
    # If force is down (-P) and disp is down (-δ), work is positive (-P * -δ = Pδ).
    # Potential energy decreases.
    # V = - Work.
    # single_work calculates (t · u).
    # So we want to MINIMIZE (U_int - ∫ t·u).
    # My single_work returns -P * u_y.
    # If u_y is negative (down), single_work is positive.
    # We subtract positive work -> Energy decreases. Correct.

    return U_int - jnp.mean(Work_vals) * area_scale, (U_int, W_ext)


# ============================================================
# Training
# ============================================================
def generate_E_field_3d(key, n_modes=5):
    """Same random E field generator."""
    k1, k2, k3, k4 = jr.split(key, 4)
    coeffs = jr.normal(k1, (n_modes,)) * 0.3
    fx = jr.uniform(k2, (n_modes,), minval=0.5, maxval=3.0)
    fy = jr.uniform(k3, (n_modes,), minval=0.5, maxval=5.0)
    fz = jr.uniform(k4, (n_modes,), minval=0.5, maxval=3.0)

    def E_fn(x, y, z):
        di = 0.5
        for k in range(n_modes):
            val = (
                coeffs[k]
                * jnp.sin(fx[k] * jnp.pi * x / W)
                * jnp.cos(fy[k] * jnp.pi * y / H)
                * jnp.sin(fz[k] * jnp.pi * z / D)
            )
            di = di + val
        return di_to_E(jnp.clip(di, 0.0, 1.0))

    return E_fn


def train(epochs=2000, n_interior=2000, n_bc=500):
    print("Starting Deep Energy Method (DEM) Training...")

    key = jr.PRNGKey(0)
    k_model, k_data = jr.split(key)
    model = ElasticityNetwork(key=k_model)

    # Schedule
    warmup = min(200, epochs // 5)
    decay = max(1, epochs - warmup)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-3,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=1e-6,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Volume and Area for integration
    vol_scale = W * H * D
    area_scale = W * D  # Top surface area

    @eqx.filter_jit
    def step(model, opt_state, domain, boundary):
        (loss, (U, W_work)), grads = eqx.filter_value_and_grad(compute_energy_loss, has_aux=True)(
            model, domain, boundary, vol_scale, area_scale
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, U, W_work

    E_fn = generate_E_field_3d(jr.PRNGKey(99))

    t0 = time.time()
    for epoch in range(epochs):
        # Sampling
        k_iter = jr.fold_in(k_data, epoch)
        k1, k2, k3, k4, k5, k6 = jr.split(k_iter, 6)

        # Domain points
        x = jr.uniform(k1, (n_interior,), minval=0.0, maxval=W)
        y = jr.uniform(k2, (n_interior,), minval=0.0, maxval=H)
        z = jr.uniform(k3, (n_interior,), minval=0.0, maxval=D)
        E = jax.vmap(E_fn)(x, y, z)
        domain = jnp.stack([x, y, z, E], axis=1)

        # Top boundary points
        xt = jr.uniform(k4, (n_bc,), minval=0.0, maxval=W)
        zt = jr.uniform(k5, (n_bc,), minval=0.0, maxval=D)
        Et = jax.vmap(lambda x, z: E_fn(x, H, z))(xt, zt)
        boundary = jnp.stack([xt, jnp.full(n_bc, H), zt, Et], axis=1)

        model, opt_state, loss, U, W_work = step(model, opt_state, domain, boundary)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Energy: {loss:.4e} | Strain: {U:.4e} | Work: {W_work:.4e}")

    print(f"Done in {time.time() - t0:.1f}s")
    eqx.tree_serialise_leaves("dem_3d.eqx", model)
    print("Saved to dem_3d.eqx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    if args.command == "train":
        train(epochs=args.epochs)
