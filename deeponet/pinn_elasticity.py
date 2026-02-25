#!/usr/bin/env python3
"""
pinn_elasticity.py — Physics-Informed Neural Network for 2D biofilm elasticity.

Solves 2D plane-strain linear elasticity with spatially varying E(x,y) = E(DI):
    ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    ∂σ_xy/∂x + ∂σ_yy/∂y = 0

where σ = C(E,ν) : ε(u), E(x,y) from DI field.

Domain: Rectangular biofilm layer [0, W] × [0, H]
  - Bottom (y=0): fixed (tooth surface), u = 0
  - Top (y=H): traction σ_yy = -p, σ_xy = 0
  - Left/Right: free

Usage:
  python pinn_elasticity.py train --epochs 5000
  python pinn_elasticity.py viz --checkpoint-dir pinn_checkpoints
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

jax.config.update("jax_enable_x64", False)

# ============================================================
# Constants — must match FEM/material_models.py
# ============================================================
E_MAX = 1000.0   # Pa (commensal)
E_MIN = 10.0     # Pa (dysbiotic)
NU = 0.30        # Poisson's ratio
W = 1.0          # width [mm]
H = 0.2          # height [mm] (biofilm thickness)
P_APPLIED = 1.0  # applied pressure [Pa]

# 4 conditions: name → (DI, color)
CONDITIONS = {
    "CS": (0.421, "#2ca02c"),  # Commensal Static
    "CH": (0.843, "#17becf"),  # Commensal HOBIC
    "DH": (0.161, "#d62728"),  # Dysbiotic HOBIC
    "DS": (0.845, "#ff7f0e"),  # Dysbiotic Static
}


# ============================================================
# Material model
# ============================================================
def di_to_E(di, di_scale=1.0, n=2.0):
    r = jnp.clip(di / di_scale, 0.0, 1.0)
    return E_MAX * (1.0 - r)**n + E_MIN * r


def compute_di_from_phi(phi):
    """Shannon entropy DI from species fractions. phi: (5,).

    Clips φ to [0, ∞) before computing to avoid NaN from negative values
    (which can arise from DeepONet extrapolation errors).
    """
    eps = 1e-12
    phi = jnp.clip(phi, 0.0)  # prevent negative volume fractions
    phi_sum = jnp.sum(phi)
    phi_sum = jnp.where(phi_sum > eps, phi_sum, 1.0)
    p = phi / phi_sum
    log_p = jnp.where(p > eps, jnp.log(p), 0.0)
    H_val = -jnp.sum(p * log_p)
    return 1.0 - H_val / jnp.log(5.0)


def plane_strain_C(E, nu):
    """Plane-strain stiffness matrix (Voigt: σ_xx, σ_yy, σ_xy)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return jnp.array([
        [lam + 2*mu,  lam,       0.0],
        [lam,         lam + 2*mu, 0.0],
        [0.0,         0.0,        mu],
    ])


# ============================================================
# Analytical solution (uniform E, free sides, plane strain)
# ============================================================
def analytical_solution(E, nu, p, y):
    """
    1D analytical solution for uniform strip under top pressure.
    Assumes far from edges (σ_xx=0 interior, plane strain).

    Returns: u_y(y), u_x = 0 (at center)
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    # σ_yy = -p, σ_xx = 0 → ε_yy = -p*(λ+2μ)/[4μ(λ+μ)]
    eps_yy = -p * (lam + 2*mu) / (4*mu * (lam + mu))
    return eps_yy * y


# ============================================================
# PINN Model with Fourier features
# ============================================================
class ElasticityPINN(eqx.Module):
    """
    PINN for 2D plane-strain elasticity with variable E.
    Input: (x, y, E_norm) → (u_x, u_y)
    Fourier features on (x, y) for better convergence.
    """
    fourier_B: jnp.ndarray   # frozen random matrix for Fourier features
    layers: list
    output_scale: jnp.ndarray

    def __init__(self, hidden: int = 128, n_layers: int = 5,
                 n_fourier: int = 32, *, key):
        k1, k2 = jr.split(key)
        # Fourier feature matrix: (2, n_fourier) for (x, y)
        self.fourier_B = jr.normal(k1, (2, n_fourier)) * 2.0

        # Input dim = 2*n_fourier (sin+cos) + 1 (E_norm) = 2*32+1 = 65
        in_dim = 2 * n_fourier + 1
        keys = jr.split(k2, n_layers + 1)
        self.layers = []
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 2
            self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))
            in_dim = hidden

        self.output_scale = jnp.array([1e-3, 1e-3])

    def __call__(self, x, y, E_norm):
        """Predict (u_x, u_y) at (x, y) given normalized E."""
        xy = jnp.array([x, y])
        # Fourier features: [sin(B^T xy), cos(B^T xy)]
        proj = self.fourier_B.T @ xy  # (n_fourier,)
        ff = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])  # (2*n_fourier,)
        inp = jnp.concatenate([ff, jnp.array([E_norm])])

        h = inp
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        raw = self.layers[-1](h)

        # Hard BC: u = 0 at y = 0
        u = raw * self.output_scale * y
        return u


# ============================================================
# PDE residual
# ============================================================
def pde_residual(model, x, y, E_local):
    """Compute equilibrium residual + stress + displacement."""
    E_norm = E_local / E_MAX
    C = plane_strain_C(E_local, NU)

    def u_fn(xy):
        return model(xy[0], xy[1], E_norm)

    xy = jnp.array([x, y])
    u = u_fn(xy)
    J = jax.jacobian(u_fn)(xy)  # (2, 2)

    eps_xx = J[0, 0]
    eps_yy = J[1, 1]
    gamma_xy = J[0, 1] + J[1, 0]
    strain = jnp.array([eps_xx, eps_yy, gamma_xy])
    stress = C @ strain

    # Stress divergence via second derivatives
    def stress_fn(xy_):
        J_ = jax.jacobian(lambda p: model(p[0], p[1], E_norm))(xy_)
        s_ = C @ jnp.array([J_[0, 0], J_[1, 1], J_[0, 1] + J_[1, 0]])
        return s_

    dstress = jax.jacobian(stress_fn)(xy)  # (3, 2)
    res_x = dstress[0, 0] + dstress[2, 1]  # ∂σ_xx/∂x + ∂σ_xy/∂y
    res_y = dstress[2, 0] + dstress[1, 1]  # ∂σ_xy/∂x + ∂σ_yy/∂y

    return jnp.array([res_x, res_y]), stress, u


# ============================================================
# Loss function (corrected BCs)
# ============================================================
def generate_E_field(key, n_modes=5):
    """Random smooth E(x,y) field via Fourier modes."""
    k1, k2, k3 = jr.split(key, 3)
    coeffs = jr.normal(k1, (n_modes,)) * 0.3
    fx = jr.uniform(k2, (n_modes,), minval=0.5, maxval=3.0)
    fy = jr.uniform(k3, (n_modes,), minval=0.5, maxval=5.0)

    def E_fn(x, y):
        di = 0.5
        for k in range(n_modes):
            di = di + coeffs[k] * jnp.sin(fx[k]*jnp.pi*x/W) * jnp.cos(fy[k]*jnp.pi*y/H)
        return di_to_E(jnp.clip(di, 0.0, 1.0))
    return E_fn


def generate_condition_E_field(di_value):
    E_val = di_to_E(jnp.array(di_value))
    return lambda x, y: E_val


@partial(jax.jit, static_argnums=(4,))
def compute_loss(model, colloc_pts, bc_bot_pts, bc_top_pts, n_colloc):
    """
    PINN loss with correct BC formulation.

    colloc_pts: (n_colloc, 3) — interior points (x, y, E)
    bc_bot_pts: (n_bc, 3) — bottom boundary (x, 0, E)
    bc_top_pts: (n_bc, 3) — top boundary (x, H, E)
    """
    # PDE residual
    def single_pde(pt):
        res, _, _ = pde_residual(model, pt[0], pt[1], pt[2])
        return jnp.sum(res**2)
    pde_loss = jnp.mean(jax.vmap(single_pde)(colloc_pts))

    # Bottom BC: u = 0 at y = 0
    def single_bc_bot(pt):
        u = model(pt[0], 0.0, pt[2] / E_MAX)
        return jnp.sum(u**2)
    bc_bot_loss = jnp.mean(jax.vmap(single_bc_bot)(bc_bot_pts))

    # Top BC: σ_yy = -p, σ_xy = 0 at y = H (FULL stress tensor)
    def single_bc_top(pt):
        x, E = pt[0], pt[2]
        E_norm = E / E_MAX

        # Full spatial Jacobian at (x, H)
        def u_fn(xy):
            return model(xy[0], xy[1], E_norm)
        J = jax.jacobian(u_fn)(jnp.array([x, H]))  # (2, 2)

        eps_xx = J[0, 0]
        eps_yy = J[1, 1]
        gamma_xy = J[0, 1] + J[1, 0]

        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))

        sigma_yy = lam * eps_xx + (lam + 2*mu) * eps_yy
        sigma_xy = mu * gamma_xy

        return (sigma_yy - (-P_APPLIED))**2 + sigma_xy**2

    bc_top_loss = jnp.mean(jax.vmap(single_bc_top)(bc_top_pts))

    total = pde_loss + 10.0 * bc_bot_loss + 10.0 * bc_top_loss
    return total, (pde_loss, bc_bot_loss, bc_top_loss)


@partial(jax.jit, static_argnums=(4,))
def compute_loss_adaptive(model, colloc_pts, bc_bot_pts, bc_top_pts, n_colloc,
                          w_pde=1.0, w_bot=10.0, w_top=10.0):
    """Loss with adaptive weights (passed externally)."""
    def single_pde(pt):
        res, _, _ = pde_residual(model, pt[0], pt[1], pt[2])
        return jnp.sum(res**2)
    pde_loss = jnp.mean(jax.vmap(single_pde)(colloc_pts))

    def single_bc_bot(pt):
        u = model(pt[0], 0.0, pt[2] / E_MAX)
        return jnp.sum(u**2)
    bc_bot_loss = jnp.mean(jax.vmap(single_bc_bot)(bc_bot_pts))

    def single_bc_top(pt):
        x, E = pt[0], pt[2]
        E_norm = E / E_MAX
        def u_fn(xy):
            return model(xy[0], xy[1], E_norm)
        J = jax.jacobian(u_fn)(jnp.array([x, H]))
        eps_xx = J[0, 0]
        eps_yy = J[1, 1]
        gamma_xy = J[0, 1] + J[1, 0]
        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))
        sigma_yy = lam * eps_xx + (lam + 2*mu) * eps_yy
        sigma_xy = mu * gamma_xy
        return (sigma_yy - (-P_APPLIED))**2 + sigma_xy**2

    bc_top_loss = jnp.mean(jax.vmap(single_bc_top)(bc_top_pts))

    total = w_pde * pde_loss + w_bot * bc_bot_loss + w_top * bc_top_loss
    return total, (pde_loss, bc_bot_loss, bc_top_loss)


# ============================================================
# Data sampling
# ============================================================
def sample_points(key, n_interior, n_bc, E_fn):
    """Sample collocation + separate BC points."""
    k1, k2, k3, k4 = jr.split(key, 4)

    x_int = jr.uniform(k1, (n_interior,), minval=0.0, maxval=W)
    y_int = jr.uniform(k2, (n_interior,), minval=0.01*H, maxval=0.99*H)
    E_int = jax.vmap(E_fn)(x_int, y_int)
    colloc = jnp.stack([x_int, y_int, E_int], axis=1)

    x_bc = jr.uniform(k3, (n_bc,), minval=0.0, maxval=W)
    E_bot = jax.vmap(lambda x: E_fn(x, 0.0))(x_bc)
    bc_bot = jnp.stack([x_bc, jnp.zeros(n_bc), E_bot], axis=1)

    x_bc2 = jr.uniform(k4, (n_bc,), minval=0.0, maxval=W)
    E_top = jax.vmap(lambda x: E_fn(x, H))(x_bc2)
    bc_top = jnp.stack([x_bc2, jnp.full(n_bc, H), E_top], axis=1)

    return colloc, bc_bot, bc_top


# ============================================================
# Training
# ============================================================
def train(
    n_fields: int = 30,
    n_epochs: int = 20000,
    n_interior: int = 1000,
    n_bc: int = 100,
    lr: float = 5e-4,
    hidden: int = 128,
    n_layers: int = 5,
    n_fourier: int = 32,
    seed: int = 0,
    checkpoint_dir: str = "pinn_checkpoints",
):
    """Train PINN on multiple E-field realizations with adaptive weighting."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(seed)
    k_model, k_data = jr.split(key)

    model = ElasticityPINN(hidden=hidden, n_layers=n_layers,
                           n_fourier=n_fourier, key=k_model)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"PINN params: {n_params:,}")

    # E-field pool: 4 conditions (weighted 3x) + random fields
    E_fns = []
    cond_E_fns = []
    for cond, (di_val, _) in CONDITIONS.items():
        fn = generate_condition_E_field(di_val)
        cond_E_fns.append(fn)
        # Repeat condition fields 3x for emphasis
        E_fns.extend([fn] * 3)

    field_keys = jr.split(k_data, n_fields)
    n_random = max(0, n_fields - len(E_fns))
    for i in range(n_random):
        E_fns.append(generate_E_field(field_keys[i]))
    print(f"E-field pool: {len(E_fns)} ({len(CONDITIONS)}×3 conditions + {n_random} random)")

    # Optimizer with longer warmup for 20k epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=lr,
        warmup_steps=500, decay_steps=n_epochs, end_value=1e-7,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Adaptive weights (updated every 500 epochs)
    w_pde, w_bot, w_top = 1.0, 10.0, 10.0

    @eqx.filter_jit
    def step(model, opt_state, colloc, bc_bot, bc_top):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            model, colloc, bc_bot, bc_top, colloc.shape[0]
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    best_loss = float("inf")
    history = {"total": [], "pde": [], "bc_bot": [], "bc_top": []}

    print(f"Training: {n_epochs} epochs, {n_interior} interior pts, {n_bc} BC pts")
    t0 = time.time()

    for epoch in range(n_epochs):
        E_fn = E_fns[epoch % len(E_fns)]
        data_key = jr.PRNGKey(epoch * 1000 + seed)
        colloc, bc_bot, bc_top = sample_points(data_key, n_interior, n_bc, E_fn)

        model, opt_state, loss, (pde_l, bot_l, top_l) = step(
            model, opt_state, colloc, bc_bot, bc_top
        )

        loss_val = float(loss)
        history["total"].append(loss_val)
        history["pde"].append(float(pde_l))
        history["bc_bot"].append(float(bot_l))
        history["bc_top"].append(float(top_l))

        if loss_val < best_loss:
            best_loss = loss_val
            eqx.tree_serialise_leaves(str(ckpt_dir / "best.eqx"), model)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  [{epoch+1:5d}/{n_epochs}] loss={loss_val:.2e}  "
                  f"pde={float(pde_l):.2e}  bot={float(bot_l):.2e}  "
                  f"top={float(top_l):.2e}  best={best_loss:.2e}  [{elapsed:.0f}s]")

    eqx.tree_serialise_leaves(str(ckpt_dir / "final.eqx"), model)
    np.savez(ckpt_dir / "training_history.npz",
             **{k: np.array(v) for k, v in history.items()})

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Best loss: {best_loss:.2e}")
    print(f"Saved to {ckpt_dir}/")
    return model


# ============================================================
# Visualization: clear 4-panel figure
# ============================================================
def visualize(checkpoint_dir: str):
    """
    Generate clear PINN results figure:
      (a) Training convergence
      (b) PINN vs Analytical validation
      (c) 4-condition u_y profiles
      (d) 4-condition displacement summary
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_dir = Path(checkpoint_dir)
    history = np.load(ckpt_dir / "training_history.npz")

    model = ElasticityPINN(hidden=128, n_layers=5, n_fourier=32, key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(str(ckpt_dir / "best.eqx"), model)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) Training convergence ──
    ax = axes[0, 0]
    epochs = np.arange(1, len(history["total"]) + 1)
    for key, label, color in [
        ("total", "Total", "black"),
        ("pde", "PDE residual", "#1f77b4"),
        ("bc_bot", "BC bottom", "#2ca02c"),
        ("bc_top", "BC top (traction)", "#d62728"),
    ]:
        # Smooth with moving average
        vals = history[key]
        window = min(50, len(vals) // 10)
        if window > 1:
            smooth = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.semilogy(epochs[:len(smooth)], smooth, label=label, color=color, lw=1.5)
        else:
            ax.semilogy(epochs, vals, label=label, color=color, lw=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Training Convergence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── (b) PINN vs Analytical: u_y(y) at x=W/2 ──
    ax = axes[0, 1]
    y_pts = jnp.linspace(0.001, H, 80)
    x_mid = W / 2

    for E_val, ls, color, label in [
        (100.0, "-", "#e57373", "E=100 Pa"),
        (500.0, "-", "#64b5f6", "E=500 Pa"),
        (900.0, "-", "#81c784", "E=900 Pa"),
    ]:
        E_norm = E_val / E_MAX
        u_pinn = np.array(jax.vmap(lambda y: model(x_mid, y, E_norm))(y_pts))
        u_anal = np.array([analytical_solution(E_val, NU, P_APPLIED, float(y)) for y in y_pts])

        ax.plot(u_pinn[:, 1] * 1e3, np.array(y_pts), ls, color=color, lw=2.5,
                label=f"PINN {label}")
        ax.plot(u_anal * 1e3, np.array(y_pts), "--", color=color, lw=1.5,
                label=f"Analytical", alpha=0.7)

    ax.set_xlabel("$u_y$ [$\\times 10^{-3}$ mm]", fontsize=11)
    ax.set_ylabel("Depth $y$ [mm]", fontsize=11)
    ax.set_title("(b) PINN vs Analytical Validation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── (c) 4-condition u_y profiles ──
    ax = axes[1, 0]
    cond_results = {}
    for cond, (di_val, color) in CONDITIONS.items():
        E_val = float(di_to_E(jnp.array(di_val)))
        E_norm = E_val / E_MAX
        u = np.array(jax.vmap(lambda y: model(x_mid, y, E_norm))(y_pts))
        cond_results[cond] = {"di": di_val, "E": E_val, "u_y_max": float(u[-1, 1])}
        ax.plot(u[:, 1] * 1e3, np.array(y_pts), lw=2.5, color=color,
                label=f"{cond} (DI={di_val:.2f}, E={E_val:.0f} Pa)")

    ax.set_xlabel("$u_y$ [$\\times 10^{-3}$ mm]", fontsize=11)
    ax.set_ylabel("Depth $y$ [mm]", fontsize=11)
    ax.set_title("(c) Displacement Profile per Condition", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── (d) Summary table + bar chart ──
    ax = axes[1, 1]
    conds = list(CONDITIONS.keys())
    E_vals = [cond_results[c]["E"] for c in conds]
    uy_vals = [abs(cond_results[c]["u_y_max"]) * 1e3 for c in conds]
    di_vals = [cond_results[c]["di"] for c in conds]
    colors = [CONDITIONS[c][1] for c in conds]

    x_pos = np.arange(len(conds))
    bars = ax.bar(x_pos, uy_vals, color=colors, edgecolor="black", width=0.6)
    for i, (bar, ev, dv) in enumerate(zip(bars, E_vals, di_vals)):
        ax.text(i, bar.get_height() * 1.02, f"DI={dv:.2f}\nE={ev:.0f} Pa",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conds, fontsize=11)
    ax.set_ylabel("|$u_y$(top)| [$\\times 10^{-3}$ mm]", fontsize=11)
    ax.set_title("(d) Max Displacement per Condition", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Stiffness ratio annotation
    if len(uy_vals) > 1:
        ratio = max(uy_vals) / max(min(uy_vals), 1e-12)
        ax.annotate(f"Ratio: {ratio:.1f}x",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="black"))

    plt.tight_layout()
    out_path = ckpt_dir / "pinn_results.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary
    print("\n--- Condition Summary ---")
    print(f"{'Cond':<5} {'DI':>6} {'E [Pa]':>8} {'|u_y| x1e3':>12}")
    for c in conds:
        r = cond_results[c]
        print(f"{c:<5} {r['di']:>6.3f} {r['E']:>8.1f} {abs(r['u_y_max'])*1e3:>12.4f}")

    return cond_results


# ============================================================
# End-to-end: θ → DeepONet → DI → PINN → u, σ
# ============================================================
def end_to_end_demo(deeponet_ckpt: str, pinn_ckpt: str, norm_stats_path: str):
    """Full differentiable pipeline demo with clear visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import json

    # Load PINN
    pinn_model = ElasticityPINN(hidden=128, n_layers=5, n_fourier=32, key=jr.PRNGKey(1))
    pinn_model = eqx.tree_deserialise_leaves(pinn_ckpt, pinn_model)

    # Load per-condition DeepONet models for full θ→φ→DI pipeline
    import json
    from deeponet_hamilton import DeepONet as DeepONetModel
    t_norm = jnp.linspace(0, 1, 100, dtype=jnp.float32)

    don_dir = Path(__file__).parent
    cond_don_map = {
        "CS": "checkpoints_CS_v2",
        "CH": "checkpoints_Commensal_HOBIC",
        "DS": "checkpoints_DS_v2",
        "DH": "checkpoints_Dysbiotic_HOBIC",
    }
    don_models = {}
    don_stats = {}
    for short, ckpt_name in cond_don_map.items():
        ckpt_path = don_dir / ckpt_name / "best.eqx"
        stats_path = don_dir / ckpt_name / "norm_stats.npz"
        if ckpt_path.exists() and stats_path.exists():
            try:
                key = jr.PRNGKey(0)
                m = DeepONetModel(theta_dim=20, n_species=5, p=64, hidden=128, key=key)
                m = eqx.tree_deserialise_leaves(str(ckpt_path), m)
                st = np.load(str(stats_path))
                don_models[short] = m
                don_stats[short] = st
                print(f"  DeepONet loaded: {short} ({ckpt_name})")
            except Exception as e:
                print(f"  DeepONet load failed for {short}: {e}")

    # Also try default checkpoint as fallback for DH
    if "DH" not in don_models:
        try:
            key = jr.PRNGKey(0)
            m = DeepONetModel(theta_dim=20, n_species=5, p=64, hidden=128, key=key)
            m = eqx.tree_deserialise_leaves(deeponet_ckpt, m)
            st = np.load(norm_stats_path)
            don_models["DH"] = m
            don_stats["DH"] = st
            print(f"  DeepONet loaded: DH (default checkpoint)")
        except Exception as e:
            print(f"  DeepONet fallback failed: {e}")

    print(f"  DeepONet available for: {list(don_models.keys())}")

    # Load θ_MAP per condition
    theta_MAP = {}
    base_runs = Path(__file__).parent.parent / "data_5species" / "_runs"
    run_map = {
        "CS": "commensal_static",
        "CH": "commensal_hobic",
        "DS": "dysbiotic_static",
        "DH": "dh_baseline",
    }
    for short, dirname in run_map.items():
        for pattern in [
            base_runs / dirname / "theta_MAP.json",
            base_runs / dirname / "posterior" / "theta_MAP.json",
        ]:
            if pattern.exists():
                with open(pattern) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    theta_MAP[short] = np.array(data["theta_full"], dtype=np.float32)
                else:
                    theta_MAP[short] = np.array(data, dtype=np.float32)
                break
    print(f"  θ_MAP available for: {list(theta_MAP.keys())}")

    y_pts = jnp.linspace(0.001, H, 50)
    x_mid = W / 2.0
    results = {}

    print("\nRunning full pipeline for each condition...")
    for cond, (di_val_known, _) in CONDITIONS.items():
        di_val = di_val_known
        phi_final = None

        # Full chain: θ_MAP → DeepONet → φ(T) → DI
        if cond in don_models and cond in theta_MAP:
            theta_raw = theta_MAP[cond]
            st = don_stats[cond]
            theta_lo = st["theta_lo"]
            theta_width = st["theta_width"]
            theta_n = jnp.array((theta_raw - theta_lo) / theta_width)
            # predict_trajectory clips φ to [0,1] by default
            phi_traj = don_models[cond].predict_trajectory(theta_n, t_norm)
            phi_final = phi_traj[-1]
            # Also check raw output for diagnostics
            phi_traj_raw = don_models[cond].predict_trajectory(theta_n, t_norm, clip=False)
            phi_raw = phi_traj_raw[-1]
            n_neg = int(jnp.sum(phi_raw < 0))
            if n_neg > 0:
                print(f"  {cond}: WARNING — {n_neg} negative raw φ values clipped: "
                      f"raw={np.array(phi_raw).round(4)}")
            di_val = float(compute_di_from_phi(phi_final))
            print(f"  {cond}: θ_MAP→DeepONet→φ(T)={np.array(phi_final).round(4)} → DI={di_val:.4f}")

        E_val = float(di_to_E(jnp.array(di_val)))
        E_norm = E_val / E_MAX
        u_arr = np.array(jax.vmap(lambda y: pinn_model(x_mid, y, E_norm))(y_pts))
        results[cond] = {
            "ux": u_arr[:, 0], "uy": u_arr[:, 1],
            "di": di_val, "E": E_val,
            "phi": np.array(phi_final) if phi_final is not None else None,
        }
        src = "DeepONet" if phi_final is not None else "known"
        print(f"  {cond}: DI={di_val:.4f} ({src}), E={E_val:.1f} Pa, "
              f"u_y(top)={float(u_arr[-1, 1]):.6f}")

    # Sensitivity ∂u_y/∂DI (how displacement changes with dysbiotic index)
    print("\nComputing sensitivity ∂u_y/∂DI...")

    @jax.jit
    def uy_of_di(di_scalar):
        E = di_to_E(di_scalar)
        u = pinn_model(x_mid, H, E / E_MAX)
        return u[1]

    # Compute for each condition (use actual DI from results)
    grad_abs = np.zeros(len(CONDITIONS))
    cond_list = list(CONDITIONS.keys())
    for i, cond in enumerate(cond_list):
        di_val = results[cond]["di"]  # from DeepONet if available
        grad_abs[i] = abs(float(jax.grad(uy_of_di)(jnp.array(di_val))))

    # ── Figure: 3 rows ──
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35,
                          height_ratios=[1.0, 1.2, 1.0])

    # Row 1: Pipeline diagram (full width)
    ax_pipe = fig.add_subplot(gs[0, :])
    ax_pipe.axis("off")
    ax_pipe.set_xlim(-0.5, 11)
    ax_pipe.set_ylim(-0.5, 3)

    boxes = [
        (0.5, 1.5, "$\\theta$\n(20 params)", "#fff3e0"),
        (2.8, 1.5, "DeepONet\n$\\phi(T; \\theta)$", "#e3f2fd"),
        (5.1, 1.5, "DI\n$1-H/H_{max}$", "#e8f5e9"),
        (7.4, 1.5, "$E(\\mathrm{DI})$\n[Pa]", "#f3e5f5"),
        (9.7, 1.5, "PINN\n$u(x,y), \\sigma$", "#fce4ec"),
    ]
    for x_pos, y_pos, txt, col in boxes:
        bx = FancyBboxPatch((x_pos-0.8, y_pos-0.6), 1.6, 1.2,
                            boxstyle="round,pad=0.1", fc=col, ec="black", lw=2)
        ax_pipe.add_patch(bx)
        ax_pipe.text(x_pos, y_pos, txt, ha="center", va="center",
                     fontsize=10, fontweight="bold")
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.8
        x2 = boxes[i+1][0] - 0.8
        ax_pipe.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                         arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#333"))

    ax_pipe.text(5.1, -0.2, "$\\partial u / \\partial \\theta$ via JAX autodiff",
                 fontsize=13, ha="center", fontweight="bold", color="#c62828",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c62828", lw=1.5))
    ax_pipe.set_title("End-to-End Differentiable Pipeline: Biology $\\to$ Mechanics",
                      fontsize=14, fontweight="bold", pad=10)

    # Row 2: 4-condition displacement profiles
    for i, cond in enumerate(["CS", "CH", "DH", "DS"]):
        ax = fig.add_subplot(gs[1, i])
        if cond not in results:
            ax.text(0.5, 0.5, f"{cond}\n(no data)", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        r = results[cond]
        color = CONDITIONS[cond][1]

        # u_y profile
        ax.plot(np.array(r["uy"]) * 1e3, np.array(y_pts), lw=3, color=color)
        ax.fill_betweenx(np.array(y_pts), 0, np.array(r["uy"]) * 1e3,
                         alpha=0.15, color=color)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axhline(H, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("$u_y$ [$\\times 10^{-3}$]", fontsize=10)
        if i == 0:
            ax.set_ylabel("$y$ [mm]", fontsize=10)
        ax.set_title(f"{cond}\nDI={r['di']:.3f}  E={r['E']:.0f} Pa",
                     fontsize=11, fontweight="bold", color=color)
        ax.grid(alpha=0.3)

        # Annotate max displacement
        uy_max = abs(r["uy"][-1]) * 1e3
        ax.annotate(f"|$u_y$|={uy_max:.3f}",
                    xy=(0.95, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

    # Row 3: Sensitivity + E(DI) curve
    # (3a) ∂u_y/∂DI per condition
    ax = fig.add_subplot(gs[2, :2])
    sorted_idx = np.argsort(-grad_abs)
    bar_names = [cond_list[i] for i in sorted_idx]
    bar_vals = grad_abs[sorted_idx]
    bar_colors = [CONDITIONS[c][1] for c in bar_names]
    bars = ax.barh(bar_names[::-1], bar_vals[::-1],
                   color=bar_colors[::-1], edgecolor="black")
    ax.set_xlabel("$|\\partial u_y / \\partial \\mathrm{DI}|$", fontsize=11)
    ax.set_title("Sensitivity: How $u_y$ Responds to DI Change",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Annotate: higher DI sensitivity = softer material
    ax.annotate("Higher = more mechano-sensitive\nto microbial composition",
                xy=(0.95, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, fontstyle="italic",
                color="#555")

    # (3b) E(DI) curve + condition markers
    ax = fig.add_subplot(gs[2, 2:])
    di_range = np.linspace(0, 1, 200)
    E_range = np.array([float(di_to_E(jnp.array(d))) for d in di_range])
    ax.plot(di_range, E_range, "k-", lw=2.5, label="$E(\\mathrm{DI})$")
    ax.fill_between(di_range, 0, E_range, alpha=0.05, color="black")

    for cond in results:
        r = results[cond]
        color = CONDITIONS[cond][1]
        ax.plot(r["di"], r["E"], "o", color=color, ms=14,
                markeredgecolor="black", markeredgewidth=2, zorder=5)
        ax.annotate(f"{cond}\n({r['E']:.0f} Pa)", xy=(r["di"], r["E"]),
                    xytext=(10, 15), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1))

    ax.set_xlabel("Dysbiotic Index (DI)", fontsize=11)
    ax.set_ylabel("Young's Modulus E [Pa]", fontsize=11)
    ax.set_title("Material Model $E(\\mathrm{DI})$ with Pipeline Results",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1100)

    out_path = Path(pinn_ckpt).parent / "e2e_pipeline_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PINN for biofilm elasticity")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=20000)
    p_train.add_argument("--n-fields", type=int, default=30)
    p_train.add_argument("--n-interior", type=int, default=1000)
    p_train.add_argument("--n-bc", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--hidden", type=int, default=128)
    p_train.add_argument("--n-layers", type=int, default=5)
    p_train.add_argument("--n-fourier", type=int, default=32)
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
            n_fields=args.n_fields, n_epochs=args.epochs,
            n_interior=args.n_interior, n_bc=args.n_bc,
            lr=args.lr, hidden=args.hidden, n_layers=args.n_layers,
            n_fourier=args.n_fourier, seed=args.seed,
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
