#!/usr/bin/env python3
"""
pinn_elasticity.py — Physics-Informed Neural Network for 2D biofilm elasticity.

Solves 2D plane-strain linear elasticity with spatially varying E(x,y) = E(DI):
    ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    ∂σ_xy/∂x + ∂σ_yy/∂y = 0

where σ = C(E,ν) : ε(u), E(x,y) from DI field.

Domain: Rectangular biofilm layer [0, W] × [0, H]
  - Bottom (y=0): fixed (tooth surface), u = 0
  - Top (y=H): traction σ_yy = -p (pressure)
  - Left/Right: free or periodic

The PINN is **conditioned on E(x,y)** so it can generalize across conditions:
  θ → [DeepONet] → φ(T) → DI → E(DI) → [PINN] → u(x,y), σ(x,y)

Architecture:
  Input: (x, y, E_local)  — spatial coords + local stiffness
  Output: (u_x, u_y)      — displacement components
  Loss: PDE residual + BC (no labeled data needed)

Usage:
  # Train on multiple E-field realizations
  python pinn_elasticity.py train --n-fields 50 --epochs 5000

  # Evaluate + visualize
  python pinn_elasticity.py eval --checkpoint pinn_checkpoints/best.eqx

  # End-to-end demo: θ → DeepONet → DI → PINN → σ
  python pinn_elasticity.py e2e --deeponet-checkpoint checkpoints/best.eqx
"""

import argparse
import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

jax.config.update("jax_enable_x64", False)  # float32 for speed

# ============================================================
# Material model (must match FEM/material_models.py)
# ============================================================

E_MAX = 1000.0   # Pa (commensal)
E_MIN = 10.0     # Pa (dysbiotic)
NU = 0.30        # Poisson's ratio (biofilm)

# Domain
W = 1.0          # width [mm] (normalized)
H = 0.2          # height [mm] (biofilm thickness)
P_APPLIED = 1.0  # applied pressure [Pa] (normalized)


def di_to_E(di, di_scale=1.0, n=2.0):
    """DI → E material model."""
    r = jnp.clip(di / di_scale, 0.0, 1.0)
    return E_MAX * (1.0 - r)**n + E_MIN * r


def compute_di_from_phi(phi):
    """Shannon entropy DI from species fractions. phi: (5,)."""
    eps = 1e-12
    phi_sum = jnp.sum(phi)
    phi_sum = jnp.where(phi_sum > eps, phi_sum, 1.0)
    p = phi / phi_sum
    log_p = jnp.where(p > eps, jnp.log(p), 0.0)
    H = -jnp.sum(p * log_p)
    return 1.0 - H / jnp.log(5.0)


# ============================================================
# PINN Model
# ============================================================

class ElasticityPINN(eqx.Module):
    """
    PINN for 2D plane-strain elasticity with variable E.

    Input: (x, y, E_local) → (u_x, u_y)
    """
    layers: list
    output_scale: jnp.ndarray  # learnable output scaling

    def __init__(self, hidden: int = 64, n_layers: int = 4, *, key):
        keys = jr.split(key, n_layers + 1)
        # Input: (x, y, E_normalized)
        self.layers = []
        in_dim = 3
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 2  # final: (u_x, u_y)
            self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))
            in_dim = hidden

        self.output_scale = jnp.array([1e-3, 1e-3])  # displacement scale

    def __call__(self, x, y, E_norm):
        """
        Predict displacement (u_x, u_y) at point (x, y) with local E.

        Args:
            x, y: spatial coordinates (scalars)
            E_norm: normalized Young's modulus E/E_MAX (scalar)

        Returns:
            (u_x, u_y) displacement
        """
        inp = jnp.array([x, y, E_norm])

        h = inp
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        raw = self.layers[-1](h)

        # Hard BC: u = 0 at y = 0 → multiply by y
        u = raw * self.output_scale * y
        return u


# ============================================================
# Physics residual (PDE loss)
# ============================================================

def plane_strain_C(E, nu):
    """
    Plane-strain stiffness matrix (Voigt notation).
    σ = C : ε  where σ = [σ_xx, σ_yy, σ_xy], ε = [ε_xx, ε_yy, 2ε_xy]
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = jnp.array([
        [lam + 2*mu,  lam,       0.0],
        [lam,         lam + 2*mu, 0.0],
        [0.0,         0.0,        mu],
    ])
    return C


def pde_residual(model, x, y, E_local):
    """
    Compute PDE residual for equilibrium equations.

    ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    """
    E_norm = E_local / E_MAX

    # Displacement and its derivatives via AD
    def u_fn(xy):
        return model(xy[0], xy[1], E_norm)

    xy = jnp.array([x, y])
    u = u_fn(xy)                      # (2,)
    J = jax.jacobian(u_fn)(xy)        # (2, 2): du_i/dx_j

    # Strain: ε_xx, ε_yy, 2*ε_xy (Voigt)
    du_dx = J[0, 0]   # ∂u_x/∂x
    du_dy = J[0, 1]   # ∂u_x/∂y
    dv_dx = J[1, 0]   # ∂u_y/∂x
    dv_dy = J[1, 1]   # ∂u_y/∂y

    eps_xx = du_dx
    eps_yy = dv_dy
    gamma_xy = du_dy + dv_dx  # = 2 * ε_xy

    strain = jnp.array([eps_xx, eps_yy, gamma_xy])

    # Stress
    C = plane_strain_C(E_local, NU)
    stress = C @ strain  # [σ_xx, σ_yy, σ_xy]

    # Second derivatives for equilibrium
    def stress_fn(xy):
        """Compute stress at a point."""
        u_local = model(xy[0], xy[1], E_norm)
        J_local = jax.jacobian(lambda p: model(p[0], p[1], E_norm))(xy)

        e_xx = J_local[0, 0]
        e_yy = J_local[1, 1]
        g_xy = J_local[0, 1] + J_local[1, 0]
        s = C @ jnp.array([e_xx, e_yy, g_xy])
        return s  # (3,): [σ_xx, σ_yy, σ_xy]

    dstress_dxy = jax.jacobian(stress_fn)(xy)  # (3, 2)

    # Equilibrium:
    # ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    # ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    res_x = dstress_dxy[0, 0] + dstress_dxy[2, 1]  # ∂σ_xx/∂x + ∂σ_xy/∂y
    res_y = dstress_dxy[2, 0] + dstress_dxy[1, 1]  # ∂σ_xy/∂x + ∂σ_yy/∂y

    return jnp.array([res_x, res_y]), stress, u


# ============================================================
# Loss function
# ============================================================

def generate_E_field(key, n_modes=5):
    """Generate a random E(x,y) field from Fourier modes (smooth)."""
    keys = jr.split(key, 3)
    # Base DI from random Fourier series
    coeffs = jr.normal(keys[0], (n_modes,)) * 0.3
    freqs_x = jr.uniform(keys[1], (n_modes,), minval=0.5, maxval=3.0)
    freqs_y = jr.uniform(keys[2], (n_modes,), minval=0.5, maxval=5.0)

    def E_fn(x, y):
        di_base = 0.5  # mean DI
        for k in range(n_modes):
            di_base = di_base + coeffs[k] * jnp.sin(freqs_x[k] * jnp.pi * x / W) * \
                      jnp.cos(freqs_y[k] * jnp.pi * y / H)
        di = jnp.clip(di_base, 0.0, 1.0)
        return di_to_E(di)

    return E_fn


def generate_condition_E_field(di_value):
    """Generate a uniform E field from a single DI value."""
    E_val = di_to_E(jnp.array(di_value))
    def E_fn(x, y):
        return E_val
    return E_fn


@partial(jax.jit, static_argnums=(3,))
def compute_loss(model, colloc_pts, bc_pts, n_colloc):
    """
    Compute total PINN loss.

    colloc_pts: (n_colloc, 3) — (x, y, E)
    bc_pts: dict with 'bottom' and 'top' arrays
    """
    # --- PDE residual loss ---
    def single_residual(pt):
        x, y, E = pt[0], pt[1], pt[2]
        res, _, _ = pde_residual(model, x, y, E)
        return jnp.sum(res**2)

    pde_loss = jnp.mean(jax.vmap(single_residual)(colloc_pts))

    # --- BC: bottom (y=0) fixed, u = 0 ---
    def bc_bottom_loss(pt):
        x, E_norm = pt[0], pt[2] / E_MAX
        u = model(x, 0.0, E_norm)
        return jnp.sum(u**2)

    bc_bot = jnp.mean(jax.vmap(bc_bottom_loss)(bc_pts))

    # --- BC: top (y=H) traction σ_yy = -P ---
    def bc_top_loss(pt):
        x, E = pt[0], pt[2]
        E_norm = E / E_MAX

        def u_fn(y_val):
            return model(x, y_val, E_norm)

        # ∂u/∂y at y=H
        J = jax.jacobian(u_fn)(H)  # (2,)
        dv_dy = J[1]

        # For top traction: σ_yy ≈ (λ + 2μ) * ε_yy (simplified for top BC)
        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))
        sigma_yy_approx = (lam + 2 * mu) * dv_dy
        return (sigma_yy_approx - (-P_APPLIED))**2

    bc_top = jnp.mean(jax.vmap(bc_top_loss)(bc_pts))

    total = pde_loss + 10.0 * bc_bot + 1.0 * bc_top
    return total, (pde_loss, bc_bot, bc_top)


# ============================================================
# Training
# ============================================================

def sample_collocation_points(key, n_interior, n_bc, E_fn):
    """Sample collocation points with E values."""
    k1, k2, k3 = jr.split(key, 3)

    # Interior points
    x_int = jr.uniform(k1, (n_interior,), minval=0.0, maxval=W)
    y_int = jr.uniform(k2, (n_interior,), minval=0.0, maxval=H)
    E_int = jax.vmap(E_fn)(x_int, y_int)
    colloc = jnp.stack([x_int, y_int, E_int], axis=1)  # (n_interior, 3)

    # BC points (bottom + top)
    x_bc = jr.uniform(k3, (n_bc,), minval=0.0, maxval=W)
    # Bottom
    E_bot = jax.vmap(lambda x: E_fn(x, 0.0))(x_bc)
    bc_bot = jnp.stack([x_bc, jnp.zeros(n_bc), E_bot], axis=1)
    # Top
    E_top = jax.vmap(lambda x: E_fn(x, H))(x_bc)
    bc_top = jnp.stack([x_bc, jnp.full(n_bc, H), E_top], axis=1)

    return colloc, bc_bot, bc_top


def train(
    n_fields: int = 20,
    n_epochs: int = 3000,
    n_interior: int = 500,
    n_bc: int = 100,
    lr: float = 1e-3,
    hidden: int = 64,
    n_layers: int = 4,
    seed: int = 0,
    checkpoint_dir: str = "pinn_checkpoints",
):
    """Train PINN on multiple E-field realizations."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(seed)
    k_model, k_data = jr.split(key)

    model = ElasticityPINN(hidden=hidden, n_layers=n_layers, key=k_model)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"PINN params: {n_params:,}")

    # Generate E-field realizations
    print(f"Generating {n_fields} E-field realizations...")
    field_keys = jr.split(k_data, n_fields)

    # Mix of random fields + condition-specific uniform fields
    E_fns = []
    cond_dis = [0.16, 0.42, 0.84, 0.85]  # DH, CS, CH, DS
    for di_val in cond_dis:
        E_fns.append(generate_condition_E_field(di_val))
    for i in range(n_fields - len(cond_dis)):
        E_fns.append(generate_E_field(field_keys[i]))

    # Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5, peak_value=lr,
        warmup_steps=200, decay_steps=n_epochs, end_value=1e-6,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, colloc, bc_pts):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            model, colloc, bc_pts, colloc.shape[0]
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    best_loss = float("inf")
    history = {"total": [], "pde": [], "bc_bot": [], "bc_top": []}

    print(f"Training for {n_epochs} epochs...")
    t0 = time.time()

    for epoch in range(n_epochs):
        # Cycle through E-field realizations
        field_idx = epoch % len(E_fns)
        E_fn = E_fns[field_idx]

        # Fresh collocation points each epoch
        data_key = jr.PRNGKey(epoch * 1000 + seed)
        colloc, bc_bot, bc_top = sample_collocation_points(
            data_key, n_interior, n_bc, E_fn
        )

        # Use bottom points for both BC (combined)
        bc_combined = jnp.concatenate([bc_bot, bc_top], axis=0)

        model, opt_state, loss, (pde_l, bc_bot_l, bc_top_l) = step(
            model, opt_state, colloc, bc_combined
        )

        loss_val = float(loss)
        history["total"].append(loss_val)
        history["pde"].append(float(pde_l))
        history["bc_bot"].append(float(bc_bot_l))
        history["bc_top"].append(float(bc_top_l))

        if loss_val < best_loss:
            best_loss = loss_val
            eqx.tree_serialise_leaves(str(ckpt_dir / "best.eqx"), model)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:5d}/{n_epochs}  "
                  f"loss={loss_val:.6f}  pde={float(pde_l):.6f}  "
                  f"bc_bot={float(bc_bot_l):.6f}  bc_top={float(bc_top_l):.6f}  "
                  f"best={best_loss:.6f}  [{elapsed:.0f}s]")

    eqx.tree_serialise_leaves(str(ckpt_dir / "final.eqx"), model)
    np.savez(ckpt_dir / "training_history.npz",
             **{k: np.array(v) for k, v in history.items()})

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Best loss: {best_loss:.6f}")
    return model


# ============================================================
# End-to-end pipeline: θ → DeepONet → DI → PINN → σ
# ============================================================

def end_to_end_demo(deeponet_ckpt: str, pinn_ckpt: str,
                    norm_stats_path: str):
    """
    Demonstrate full differentiable pipeline:
    θ → DeepONet → φ(T) → DI → E(DI) → PINN → u, σ
    And compute ∂σ/∂θ via automatic differentiation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deeponet_hamilton import DeepONet as DeepONetModel

    # Load DeepONet
    key = jr.PRNGKey(0)
    don_model = DeepONetModel(theta_dim=20, n_species=5, p=64, hidden=128, key=key)
    don_model = eqx.tree_deserialise_leaves(deeponet_ckpt, don_model)

    stats = np.load(norm_stats_path)
    theta_lo = jnp.array(stats["theta_lo"])
    theta_width = jnp.array(stats["theta_width"])

    # Load PINN
    pinn_model = ElasticityPINN(hidden=64, n_layers=4, key=jr.PRNGKey(1))
    pinn_model = eqx.tree_deserialise_leaves(pinn_ckpt, pinn_model)

    # ── Full differentiable pipeline ──
    @jax.jit
    def pipeline(theta_raw, x, y):
        """
        θ → φ(T) → DI → E → u(x,y), σ(x,y)

        Returns: (u_x, u_y, sigma_yy, DI, E)
        """
        # Normalize θ
        theta_norm = (theta_raw - theta_lo) / theta_width

        # DeepONet: θ → φ(t=T)
        phi_T = don_model(theta_norm, jnp.array(1.0))  # species at final time

        # DI from species composition
        di = compute_di_from_phi(phi_T)

        # E from DI
        E = di_to_E(di)

        # PINN: (x, y, E) → u
        E_norm = E / E_MAX
        u = pinn_model(x, y, E_norm)

        return u[0], u[1], di, E

    # Test with a real theta
    # Load a MAP theta for demo
    import json
    map_file = Path(__file__).parent.parent / "data_5species" / "_runs" / "dh_baseline" / "theta_MAP.json"
    if map_file.exists():
        with open(map_file) as f:
            theta_map = jnp.array(json.load(f)["theta_full"], dtype=jnp.float32)
    else:
        theta_map = jnp.ones(20, dtype=jnp.float32) * 0.5

    # Evaluate on a grid
    nx, ny = 30, 10
    xs = jnp.linspace(0.01, W - 0.01, nx)
    ys = jnp.linspace(0.01, H - 0.01, ny)
    X, Y = jnp.meshgrid(xs, ys)

    print("Running end-to-end pipeline...")

    # Vectorized evaluation
    @jax.jit
    def eval_grid(theta):
        def single_point(x, y):
            return pipeline(theta, x, y)
        return jax.vmap(jax.vmap(single_point, in_axes=(None, 0)), in_axes=(0, None))(xs, ys)

    ux, uy, di_grid, E_grid = eval_grid(theta_map)

    print(f"  DI = {float(di_grid[0, 0]):.4f}")
    print(f"  E  = {float(E_grid[0, 0]):.1f} Pa")
    print(f"  u_y range: [{float(uy.min()):.6f}, {float(uy.max()):.6f}]")

    # ── Compute ∂u_y/∂θ (sensitivity) ──
    print("\nComputing ∂u_y/∂θ (sensitivity to parameters)...")

    @jax.jit
    def sensitivity(theta):
        """∂u_y(center)/∂θ"""
        _, uy_val, _, _ = pipeline(theta, W / 2, H)
        return uy_val

    grad_uy_theta = jax.grad(sensitivity)(theta_map)

    print("  Top-5 most sensitive parameters:")
    grad_abs = jnp.abs(grad_uy_theta)
    top5 = jnp.argsort(-grad_abs)[:5]
    param_names = [
        "a11", "a12", "a22", "b1", "b2",
        "a33", "a34", "a44", "b3", "b4",
        "a13", "a14", "a23", "a24",
        "a55", "b5", "a15", "a25", "a35", "a45",
    ]
    for idx in top5:
        print(f"    {param_names[int(idx)]}: ∂u_y/∂θ = {float(grad_uy_theta[int(idx)]):.6f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # (a) DI per condition
    ax = axes[0, 0]
    conditions = {"CS": 0.42, "CH": 0.84, "DH": 0.16, "DS": 0.85}
    colors = {"CS": "#2ca02c", "CH": "#17becf", "DH": "#d62728", "DS": "#ff7f0e"}
    for cond, di_val in conditions.items():
        E_val = float(di_to_E(jnp.array(di_val)))
        ax.barh(cond, E_val, color=colors[cond], edgecolor="black")
        ax.text(E_val + 10, cond, f"{E_val:.0f} Pa", va="center", fontsize=10)
    ax.set_xlabel("E [Pa]")
    ax.set_title("(a) E(DI) per condition")
    ax.set_xlim(0, 1100)

    # (b) u_y field
    ax = axes[0, 1]
    uy_np = np.array(uy)
    pcm = ax.pcolormesh(np.array(X), np.array(Y), uy_np.T, cmap="RdBu_r", shading="gouraud")
    fig.colorbar(pcm, ax=ax, label="$u_y$ [mm]")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_title(f"(b) $u_y(x,y)$ — DI={float(di_grid[0,0]):.3f}")
    ax.set_aspect("equal")

    # (c) u_x field
    ax = axes[0, 2]
    ux_np = np.array(ux)
    pcm = ax.pcolormesh(np.array(X), np.array(Y), ux_np.T, cmap="RdBu_r", shading="gouraud")
    fig.colorbar(pcm, ax=ax, label="$u_x$ [mm]")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_title("(c) $u_x(x,y)$")
    ax.set_aspect("equal")

    # (d) Pipeline diagram
    ax = axes[1, 0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    steps = [
        (1, 3, r"$\theta$ (20 params)", "#fff3e0"),
        (3, 3, "DeepONet\n" + r"$\phi(T)$", "#e3f2fd"),
        (5, 3, r"DI $\rightarrow$ E", "#e8f5e9"),
        (7, 3, "PINN\n" + r"$u, \sigma$", "#fce4ec"),
    ]
    for x_pos, y_pos, txt, col in steps:
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x_pos - 0.7, y_pos - 0.5), 1.4, 1.0,
                             boxstyle="round,pad=0.1", fc=col, ec="black", lw=1.5)
        ax.add_patch(box)
        ax.text(x_pos, y_pos, txt, ha="center", va="center", fontsize=9, fontweight="bold")
    for i in range(3):
        x_s = steps[i][0] + 0.7
        x_e = steps[i+1][0] - 0.7
        ax.annotate("", xy=(x_e, 3), xytext=(x_s, 3),
                    arrowprops=dict(arrowstyle="-|>", lw=2))
    ax.text(5, 1.5, r"$\frac{\partial \sigma}{\partial \theta}$ via autodiff",
            fontsize=12, ha="center", fontweight="bold", color="#d32f2f",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d32f2f"))
    ax.set_title("(d) End-to-end differentiable pipeline", fontweight="bold")

    # (e) Sensitivity ∂u_y/∂θ
    ax = axes[1, 1]
    grad_np = np.array(jnp.abs(grad_uy_theta))
    sorted_idx = np.argsort(-grad_np)[:10]
    names = [param_names[i] for i in sorted_idx]
    vals = grad_np[sorted_idx]
    ax.barh(names[::-1], vals[::-1], color="#e57373", edgecolor="black")
    ax.set_xlabel(r"$|\partial u_y / \partial \theta_i|$")
    ax.set_title(r"(e) Sensitivity $\partial u_y / \partial \theta$")

    # (f) E across conditions
    ax = axes[1, 2]
    di_range = np.linspace(0, 1, 200)
    E_range = np.array([float(di_to_E(jnp.array(d))) for d in di_range])
    ax.plot(di_range, E_range, "k-", lw=2)
    for cond, di_val in conditions.items():
        E_val = float(di_to_E(jnp.array(di_val)))
        ax.plot(di_val, E_val, "o", color=colors[cond], ms=12,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.annotate(cond, xy=(di_val, E_val), xytext=(5, 10),
                    textcoords="offset points", fontsize=10, fontweight="bold",
                    color=colors[cond])
    ax.set_xlabel("DI")
    ax.set_ylabel("E [Pa]")
    ax.set_title("(f) Material model $E(\\mathrm{DI})$")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(pinn_ckpt).parent / "e2e_pipeline_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


# ============================================================
# Visualization for trained PINN
# ============================================================

def visualize(checkpoint_dir: str):
    """Plot training curve and sample predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_dir = Path(checkpoint_dir)
    history = np.load(ckpt_dir / "training_history.npz")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for key in ["total", "pde", "bc_bot", "bc_top"]:
        ax.semilogy(history[key], label=key, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("PINN Training Losses")
    ax.grid(alpha=0.3)

    # Load model and predict on different DI values
    ax = axes[1]
    model = ElasticityPINN(hidden=64, n_layers=4, key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(str(ckpt_dir / "best.eqx"), model)

    y_pts = jnp.linspace(0, H, 50)
    x_mid = W / 2

    for di_val, color, label in [
        (0.16, "#d62728", "DH (DI=0.16)"),
        (0.42, "#2ca02c", "CS (DI=0.42)"),
        (0.85, "#ff7f0e", "DS (DI=0.85)"),
    ]:
        E_val = float(di_to_E(jnp.array(di_val)))
        E_norm = E_val / E_MAX

        @jax.jit
        def predict_profile(y_arr):
            return jax.vmap(lambda y: model(x_mid, y, E_norm))(y_arr)

        u = predict_profile(y_pts)
        ax.plot(np.array(u[:, 1]), np.array(y_pts), lw=2, color=color,
                label=f"{label}, E={E_val:.0f} Pa")

    ax.set_xlabel("$u_y$ [mm]")
    ax.set_ylabel("Depth $y$ [mm]")
    ax.set_title("$u_y(y)$ profile at $x = W/2$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = ckpt_dir / "pinn_results.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PINN for biofilm elasticity")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--n-fields", type=int, default=20)
    p_train.add_argument("--epochs", type=int, default=3000)
    p_train.add_argument("--n-interior", type=int, default=500)
    p_train.add_argument("--n-bc", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--hidden", type=int, default=64)
    p_train.add_argument("--n-layers", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--checkpoint-dir", default="pinn_checkpoints")

    p_viz = sub.add_parser("viz")
    p_viz.add_argument("--checkpoint-dir", default="pinn_checkpoints")

    p_e2e = sub.add_parser("e2e")
    p_e2e.add_argument("--deeponet-checkpoint", default="checkpoints/best.eqx")
    p_e2e.add_argument("--pinn-checkpoint", default="pinn_checkpoints/best.eqx")
    p_e2e.add_argument("--norm-stats", default="checkpoints/norm_stats.npz")

    args = parser.parse_args()

    if args.command == "train":
        train(
            n_fields=args.n_fields,
            n_epochs=args.epochs,
            n_interior=args.n_interior,
            n_bc=args.n_bc,
            lr=args.lr,
            hidden=args.hidden,
            n_layers=args.n_layers,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
    elif args.command == "viz":
        visualize(args.checkpoint_dir)
    elif args.command == "e2e":
        end_to_end_demo(
            args.deeponet_checkpoint,
            args.pinn_checkpoint,
            args.norm_stats,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
