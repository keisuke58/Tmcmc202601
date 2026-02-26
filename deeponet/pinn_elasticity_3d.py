#!/usr/bin/env python3
"""
pinn_elasticity_3d.py — Physics-Informed Neural Network for 3D biofilm elasticity.

Solves 3D linear elasticity with spatially varying E(x,y,z):
    ∇·σ = 0
where σ = λ tr(ε) I + 2μ ε

Domain: Box [0, W] × [0, H] × [0, D]
  - Bottom (y=0): fixed, u = 0
  - Top (y=H): traction σ·n = (0, -p, 0)
  - Sides: free (traction free, implicit)

Usage:
  python pinn_elasticity_3d.py train --epochs 5000
"""

import argparse
import time
from pathlib import Path
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
W = 1.0  # width [mm] (x)
H = 0.2  # height [mm] (y)
D = 1.0  # depth [mm] (z)
P_APPLIED = 1.0  # applied pressure [Pa]


# ============================================================
# Material model
# ============================================================
def di_to_E(di, di_scale=1.0, n=2.0):
    r = jnp.clip(di / di_scale, 0.0, 1.0)
    return E_MAX * (1.0 - r) ** n + E_MIN * r


# ============================================================
# PINN Model (3D)
# ============================================================
class ElasticityPINN3D(eqx.Module):
    """
    PINN for 3D elasticity with variable E.
    Input: (x, y, z, E_norm) → (u_x, u_y, u_z)
    """

    fourier_B: jnp.ndarray  # frozen random matrix for Fourier features
    layers: list
    output_scale: jnp.ndarray

    def __init__(self, hidden: int = 128, n_layers: int = 5, n_fourier: int = 32, *, key):
        k1, k2 = jr.split(key)
        # Fourier feature matrix: (3, n_fourier) for (x, y, z)
        self.fourier_B = jr.normal(k1, (3, n_fourier)) * 2.0

        # Input dim = 2*n_fourier (sin+cos) + 1 (E_norm)
        in_dim = 2 * n_fourier + 1
        keys = jr.split(k2, n_layers + 1)
        self.layers = []
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 3  # (u_x, u_y, u_z)
            self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))
            in_dim = hidden

        self.output_scale = jnp.array([1e-3, 1e-3, 1e-3])

    def __call__(self, x, y, z, E_norm):
        xyz = jnp.array([x, y, z])
        proj = self.fourier_B.T @ xyz  # (n_fourier,)
        ff = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])
        inp = jnp.concatenate([ff, jnp.array([E_norm])])

        h = inp
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        raw = self.layers[-1](h)

        # Hard BC: u = 0 at y = 0
        u = raw * self.output_scale * y
        return u


# ============================================================
# PDE residual (3D)
# ============================================================
def pde_residual_3d(model, x, y, z, E_local):
    E_norm = E_local / E_MAX
    lam = E_local * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E_local / (2 * (1 + NU))

    def u_fn(xyz):
        return model(xyz[0], xyz[1], xyz[2], E_norm)

    xyz = jnp.array([x, y, z])
    u = u_fn(xyz)
    J = jax.jacobian(u_fn)(xyz)  # (3, 3) -> du_i / dx_j

    # Strain: ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    eps = 0.5 * (J + J.T)
    trace_eps = jnp.trace(eps)

    # Stress: σ_ij = λ tr(ε) δ_ij + 2μ ε_ij
    stress = lam * trace_eps * jnp.eye(3) + 2 * mu * eps

    # Divergence of stress via second derivatives
    def stress_fn(xyz_):
        J_ = jax.jacobian(lambda p: model(p[0], p[1], p[2], E_norm))(xyz_)
        eps_ = 0.5 * (J_ + J_.T)
        trace_eps_ = jnp.trace(eps_)
        return lam * trace_eps_ * jnp.eye(3) + 2 * mu * eps_

    ds_dx = jax.jacobian(stress_fn)(xyz)  # (3, 3, 3) -> ∂σ_ij / ∂x_k

    # Equilibrium: ∂σ_ij / ∂x_j = 0 (sum over j)
    # i=0 (x): ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z
    # i=1 (y): ∂σ_yx/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z
    # i=2 (z): ∂σ_zx/∂x + ∂σ_zy/∂y + ∂σ_zz/∂z

    div_sigma = jnp.array(
        [
            ds_dx[0, 0, 0] + ds_dx[0, 1, 1] + ds_dx[0, 2, 2],
            ds_dx[1, 0, 0] + ds_dx[1, 1, 1] + ds_dx[1, 2, 2],
            ds_dx[2, 0, 0] + ds_dx[2, 1, 1] + ds_dx[2, 2, 2],
        ]
    )

    return div_sigma, stress, u


# ============================================================
# Loss function (3D)
# ============================================================
def generate_E_field_3d(key, n_modes=5):
    """Random smooth E(x,y,z) field."""
    k1, k2, k3, k4 = jr.split(key, 4)
    coeffs = jr.normal(k1, (n_modes,)) * 0.3
    fx = jr.uniform(k2, (n_modes,), minval=0.5, maxval=3.0)
    fy = jr.uniform(k3, (n_modes,), minval=0.5, maxval=5.0)
    fz = jr.uniform(k4, (n_modes,), minval=0.5, maxval=3.0)

    def E_fn(x, y, z):
        di = 0.5
        for k in range(n_modes):
            # Map x, y, z to normalized domain for Fourier modes if needed,
            # but here using raw coords with appropriate frequency scaling
            val = (
                coeffs[k]
                * jnp.sin(fx[k] * jnp.pi * x / W)
                * jnp.cos(fy[k] * jnp.pi * y / H)
                * jnp.sin(fz[k] * jnp.pi * z / D)
            )
            di = di + val
        return di_to_E(jnp.clip(di, 0.0, 1.0))

    return E_fn


@partial(jax.jit, static_argnums=(4,))
def compute_loss_3d(model, colloc_pts, bc_bot_pts, bc_top_pts, n_colloc):
    # PDE residual
    def single_pde(pt):
        # pt: (x, y, z, E)
        res, _, _ = pde_residual_3d(model, pt[0], pt[1], pt[2], pt[3])
        return jnp.sum(res**2)

    pde_loss = jnp.mean(jax.vmap(single_pde)(colloc_pts))

    # Bottom BC: u = 0 at y = 0
    def single_bc_bot(pt):
        # pt: (x, 0, z, E)
        u = model(pt[0], 0.0, pt[2], pt[3] / E_MAX)
        return jnp.sum(u**2)

    bc_bot_loss = jnp.mean(jax.vmap(single_bc_bot)(bc_bot_pts))

    # Top BC: σ·n = (0, -p, 0) at y = H
    # n = (0, 1, 0) -> σ_yx, σ_yy, σ_yz
    def single_bc_top(pt):
        # pt: (x, H, z, E)
        x, z, E = pt[0], pt[2], pt[3]
        E_norm = E / E_MAX

        # Calculate stress at surface
        def u_fn(xyz):
            return model(xyz[0], xyz[1], xyz[2], E_norm)

        J = jax.jacobian(u_fn)(jnp.array([x, H, z]))
        eps = 0.5 * (J + J.T)
        trace_eps = jnp.trace(eps)

        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))
        stress = lam * trace_eps * jnp.eye(3) + 2 * mu * eps

        # Traction vector t = σ · n = (σ_xy, σ_yy, σ_zy)
        # Note: stress tensor is symmetric, so row 1 is (σ_yx, σ_yy, σ_yz)
        t_vec = stress[1, :]
        target = jnp.array([0.0, -P_APPLIED, 0.0])

        return jnp.sum((t_vec - target) ** 2)

    bc_top_loss = jnp.mean(jax.vmap(single_bc_top)(bc_top_pts))

    total = pde_loss + 10.0 * bc_bot_loss + 10.0 * bc_top_loss
    return total, (pde_loss, bc_bot_loss, bc_top_loss)


# ============================================================
# Data sampling (3D)
# ============================================================
def sample_points_3d(key, n_interior, n_bc, E_fn):
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)

    # Interior
    x = jr.uniform(k1, (n_interior,), minval=0.0, maxval=W)
    y = jr.uniform(k2, (n_interior,), minval=0.01 * H, maxval=0.99 * H)
    z = jr.uniform(k3, (n_interior,), minval=0.0, maxval=D)
    E = jax.vmap(E_fn)(x, y, z)
    colloc = jnp.stack([x, y, z, E], axis=1)

    # Bottom (y=0)
    x_b = jr.uniform(k4, (n_bc,), minval=0.0, maxval=W)
    z_b = jr.uniform(k5, (n_bc,), minval=0.0, maxval=D)
    E_b = jax.vmap(lambda x, z: E_fn(x, 0.0, z))(x_b, z_b)
    # y=0 is implicit in usage
    bc_bot = jnp.stack([x_b, jnp.zeros(n_bc), z_b, E_b], axis=1)

    # Top (y=H)
    x_t = jr.uniform(k6, (n_bc,), minval=0.0, maxval=W)
    z_t = jr.uniform(k4, (n_bc,), minval=0.0, maxval=D)  # Reuse k4 dist
    E_t = jax.vmap(lambda x, z: E_fn(x, H, z))(x_t, z_t)
    bc_top = jnp.stack([x_t, jnp.full(n_bc, H), z_t, E_t], axis=1)

    return colloc, bc_bot, bc_top


# ============================================================
# Training Loop
# ============================================================
def train(n_epochs=2000, n_interior=1000, n_bc=100, lr=1e-3, checkpoint_dir="pinn_checkpoints_3d"):
    print("Starting 3D PINN Training...")
    print(f"Domain: {W}x{H}x{D} mm")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(0)
    k_model, k_data = jr.split(key)

    model = ElasticityPINN3D(key=k_model)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {n_params:,}")

    # Schedule
    warmup = min(200, n_epochs // 5)
    decay = max(1, n_epochs - warmup)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=1e-6,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, colloc, bc_bot, bc_top):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss_3d, has_aux=True)(
            model, colloc, bc_bot, bc_top, colloc.shape[0]
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    # Generate a fixed random E field for demo training
    E_fn = generate_E_field_3d(jr.PRNGKey(42))

    t0 = time.time()
    best_loss = float("inf")

    for epoch in range(n_epochs):
        colloc, bc_bot, bc_top = sample_points_3d(jr.PRNGKey(epoch), n_interior, n_bc, E_fn)
        model, opt_state, loss, (pde, bot, top) = step(model, opt_state, colloc, bc_bot, bc_top)

        loss_val = float(loss)
        if loss_val < best_loss:
            best_loss = loss_val
            eqx.tree_serialise_leaves(str(ckpt_dir / "best.eqx"), model)

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch:4d} | Loss: {loss:.2e} | PDE: {pde:.2e} | Bot: {bot:.2e} | Top: {top:.2e}"
            )

    print(f"Training complete in {time.time() - t0:.1f}s")
    print(f"Best loss: {best_loss:.2e}")
    print(f"Saved to {ckpt_dir}/best.eqx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    if args.command == "train":
        train(n_epochs=args.epochs)


if __name__ == "__main__":
    main()
