#!/usr/bin/env python3
"""
e2e_differentiable_pipeline.py — Fully differentiable bio-mechanical pipeline.

    θ (20 bio params)
        → DeepONet [JAX]      → φ(T; θ) ∈ R^5   (species fractions)
        → DI = 1 − H/H_max   → E(DI) material model
        → DEM [JAX]           → u(x,y,z), σ(x,y,z)
        → ∂u/∂θ via autodiff  (exact sensitivity)

All steps are JAX-differentiable. No finite-difference needed.

Usage:
    python e2e_differentiable_pipeline.py              # full demo
    python e2e_differentiable_pipeline.py --sensitivity # sensitivity analysis
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

jax.config.update("jax_enable_x64", False)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

# ============================================================
# Constants (shared with DEM)
# ============================================================
E_MAX = 1000.0  # Pa
E_MIN = 10.0  # Pa
NU = 0.30
W, H, D = 1.0, 0.2, 1.0  # mm
P_APPLIED = 1.0  # Pa
N_SPECIES = 5

# Species names
SPECIES = ["So", "An", "Vd", "Fn", "Pg"]

# Condition → DeepONet checkpoint mapping (best validated per condition)
CONDITION_CHECKPOINTS = {
    "Commensal_Static": "checkpoints_Commensal_Static",  # original: 62% MAP err
    "Commensal_HOBIC": "checkpoints_Commensal_HOBIC",  # original: 44% MAP err
    "Dysbiotic_Static": "checkpoints_DS_v2",  # v2 MAP-centered: 52%
    "Dysbiotic_HOBIC": "checkpoints_Dysbiotic_HOBIC_50k",  # 50k: 11% MAP err
}

# Condition → theta_MAP mapping
CONDITION_RUNS = {
    "Commensal_Static": "commensal_static",
    "Commensal_HOBIC": "commensal_hobic",
    "Dysbiotic_Static": "dysbiotic_static",
    "Dysbiotic_HOBIC": "dh_baseline",
}

# ============================================================
# Import models
# ============================================================
from deeponet_hamilton import DeepONet
from dem_elasticity_3d import ElasticityNetwork


# ============================================================
# Step 1: DeepONet → φ(T; θ)
# ============================================================
def load_deeponet(condition: str):
    """Load trained DeepONet for a condition."""
    ckpt_dir = SCRIPT_DIR / CONDITION_CHECKPOINTS.get(condition, "checkpoints")
    ckpt_path = ckpt_dir / "best.eqx"
    stats_path = ckpt_dir / "norm_stats.npz"

    # Load config
    config_path = ckpt_dir / "config.json"
    p, hidden, n_layers = 64, 128, 3
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        p = cfg.get("p", p)
        hidden = cfg.get("hidden", hidden)
        n_layers = cfg.get("n_layers", n_layers)

    key = jr.PRNGKey(0)
    model = DeepONet(theta_dim=20, n_species=5, p=p, hidden=hidden, n_layers=n_layers, key=key)
    model = eqx.tree_deserialise_leaves(str(ckpt_path), model)

    # Normalization stats
    stats = np.load(str(stats_path))
    theta_lo = jnp.array(stats["theta_lo"], dtype=jnp.float32)
    theta_width = jnp.array(stats["theta_width"], dtype=jnp.float32)
    # Avoid div by zero for locked dims
    theta_width = jnp.where(theta_width < 1e-12, 1.0, theta_width)

    return model, theta_lo, theta_width


def deeponet_predict_final(model, theta_raw, theta_lo, theta_width):
    """
    θ_raw (20,) → φ(T_final; θ) ∈ R^5.
    Fully differentiable.
    """
    theta_norm = (theta_raw - theta_lo) / theta_width
    # Predict at t=1.0 (final time, normalized)
    phi = model(theta_norm, jnp.float32(1.0))
    return jnp.clip(phi, 0.0, 1.0)


# ============================================================
# Step 2: φ → DI (Shannon entropy)
# ============================================================
def compute_di(phi):
    """
    Dysbiotic Index from Shannon entropy of species fractions.
    DI = 1 - H / H_max, where H = -sum(p_i * log(p_i)), H_max = log(5).
    Differentiable.
    """
    eps = 1e-8
    p = jnp.clip(phi, eps, None)
    p = p / jnp.sum(p)  # normalize
    H = -jnp.sum(p * jnp.log(p))
    H_max = jnp.log(jnp.float32(N_SPECIES))
    return 1.0 - H / H_max


# ============================================================
# Step 3: DI → E(DI) (material model)
# ============================================================
def di_to_E(di, di_scale=1.0, exponent=2.0):
    """
    Power-law material model. Differentiable.
    E = E_MAX * (1 - r)^n + E_MIN * r, where r = clip(di/di_scale, 0, 1).
    """
    r = jnp.clip(di / di_scale, 0.0, 1.0)
    return E_MAX * (1.0 - r) ** exponent + E_MIN * r


# ============================================================
# Step 4: E → DEM → u(x,y,z)
# ============================================================
def load_dem():
    """Load trained DEM model."""
    key = jr.PRNGKey(0)
    model = ElasticityNetwork(key=key)
    model = eqx.tree_deserialise_leaves(str(SCRIPT_DIR / "dem_3d.eqx"), model)
    return model


def dem_predict_displacement(dem_model, E_val, x, y, z):
    """
    Predict displacement at a single point.
    E_val: Young's modulus [Pa]
    Returns: u = (u_x, u_y, u_z)
    """
    E_norm = E_val / E_MAX
    return dem_model(x, y, z, E_norm)


def dem_predict_max_uy(dem_model, E_val):
    """
    Predict maximum vertical displacement at top center.
    Differentiable scalar output.
    """
    E_norm = E_val / E_MAX
    # Top center point
    u = dem_model(jnp.float32(W / 2), jnp.float32(H), jnp.float32(D / 2), E_norm)
    return u[1]  # u_y


# ============================================================
# Full Pipeline: θ → u_y_max (differentiable scalar)
# ============================================================
def make_pipeline(don_model, theta_lo, theta_width, dem_model):
    """
    Create the full differentiable pipeline function.
    Returns a function: θ_raw (20,) → u_y_max (scalar).
    """

    def pipeline(theta_raw):
        # Step 1: DeepONet → φ
        phi = deeponet_predict_final(don_model, theta_raw, theta_lo, theta_width)
        # Step 2: φ → DI
        di = compute_di(phi)
        # Step 3: DI → E
        E = di_to_E(di)
        # Step 4: E → DEM → u_y_max
        uy = dem_predict_max_uy(dem_model, E)
        return uy

    return pipeline


def make_pipeline_full(don_model, theta_lo, theta_width, dem_model):
    """
    Full pipeline returning all intermediate values.
    θ_raw → (φ, DI, E, u_y_max)
    """

    def pipeline(theta_raw):
        phi = deeponet_predict_final(don_model, theta_raw, theta_lo, theta_width)
        di = compute_di(phi)
        E = di_to_E(di)
        uy = dem_predict_max_uy(dem_model, E)
        return phi, di, E, uy

    return pipeline


# ============================================================
# Load θ_MAP from runs
# ============================================================
def load_theta_map(condition: str):
    """Load θ_MAP for a condition."""
    run_name = CONDITION_RUNS[condition]
    path = RUNS_DIR / run_name / "theta_MAP.json"
    with open(path) as f:
        data = json.load(f)
    # Use theta_full if available, else reconstruct from theta_sub
    if "theta_full" in data:
        return np.array(data["theta_full"], dtype=np.float32)
    elif "theta_sub" in data:
        return np.array(data["theta_sub"], dtype=np.float32)
    else:
        raise ValueError(f"No theta found in {path}")


# ============================================================
# Demo: 4-condition comparison
# ============================================================
def run_demo():
    print("=" * 70)
    print("End-to-End Differentiable Pipeline: θ → φ → DI → E → u")
    print("=" * 70)

    dem_model = load_dem()
    print("[OK] DEM loaded")

    results = {}

    for cond in ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_Static", "Dysbiotic_HOBIC"]:
        print(f"\n--- {cond} ---")

        # Load DeepONet
        try:
            don_model, theta_lo, theta_width = load_deeponet(cond)
        except Exception as e:
            print(f"  [SKIP] DeepONet not available: {e}")
            continue

        # Load θ_MAP
        try:
            theta_map = load_theta_map(cond)
        except Exception as e:
            print(f"  [SKIP] theta_MAP not available: {e}")
            continue

        theta_jax = jnp.array(theta_map)

        # Full pipeline
        pipeline_full = make_pipeline_full(don_model, theta_lo, theta_width, dem_model)
        phi, di, E, uy = pipeline_full(theta_jax)

        # Timing
        pipeline_scalar = make_pipeline(don_model, theta_lo, theta_width, dem_model)
        pipeline_jit = jax.jit(pipeline_scalar)
        _ = pipeline_jit(theta_jax)  # warmup

        t0 = time.time()
        for _ in range(100):
            _ = pipeline_jit(theta_jax)
            jax.block_until_ready(_)
        t_per_call = (time.time() - t0) / 100 * 1000  # ms

        # Sensitivity: ∂u_y/∂θ
        grad_fn = jax.jit(jax.grad(pipeline_scalar))
        _ = grad_fn(theta_jax)  # warmup
        grads = grad_fn(theta_jax)

        t0 = time.time()
        for _ in range(100):
            g = grad_fn(theta_jax)
            jax.block_until_ready(g)
        t_grad = (time.time() - t0) / 100 * 1000  # ms

        phi_np = np.array(phi)
        grads_np = np.array(grads)

        print(f"  φ = [{', '.join(f'{v:.3f}' for v in phi_np)}]")
        print(f"  DI = {float(di):.4f}")
        print(f"  E  = {float(E):.1f} Pa")
        print(f"  u_y(top) = {float(uy)*1000:.2f} μm")
        print(f"  Forward: {t_per_call:.2f} ms | Grad: {t_grad:.2f} ms")

        # Top 5 sensitive parameters
        abs_grad = np.abs(grads_np)
        top_idx = np.argsort(abs_grad)[::-1][:5]
        print("  Top sensitivities ∂u_y/∂θ:")
        for i in top_idx:
            print(f"    θ[{i:2d}]: {grads_np[i]:+.4e}")

        results[cond] = {
            "phi": phi_np,
            "DI": float(di),
            "E": float(E),
            "uy": float(uy),
            "grads": grads_np,
            "t_forward": t_per_call,
            "t_grad": t_grad,
            "theta_MAP": np.array(theta_map),
        }

    if not results:
        print("No conditions could be evaluated!")
        return

    # ============================================================
    # Visualization
    # ============================================================
    conds = list(results.keys())
    n = len(conds)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "End-to-End Differentiable Pipeline: θ → DeepONet → DI → DEM → u\n"
        "All steps JAX-differentiable",
        fontsize=14,
        fontweight="bold",
    )

    gs = fig.add_gridspec(
        3, n, hspace=0.4, wspace=0.35, left=0.06, right=0.96, top=0.90, bottom=0.06
    )

    colors = {
        "Commensal_Static": "#4CAF50",
        "Commensal_HOBIC": "#2196F3",
        "Dysbiotic_Static": "#F44336",
        "Dysbiotic_HOBIC": "#FF9800",
    }

    # --- Row 1: Species bar charts ---
    for col, cond in enumerate(conds):
        ax = fig.add_subplot(gs[0, col])
        r = results[cond]
        bars = ax.bar(SPECIES, r["phi"], color=colors[cond], alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{cond}\nDI={r['DI']:.3f}  E={r['E']:.0f} Pa", fontsize=10)
        ax.set_ylabel("φ (fraction)" if col == 0 else "")

    # --- Row 2: DEM displacement profiles ---
    ny_pts = 50
    y_pts = jnp.linspace(0, H, ny_pts)

    for col, cond in enumerate(conds):
        ax = fig.add_subplot(gs[1, col])
        r = results[cond]
        E_val = jnp.float32(r["E"])
        E_norm = E_val / E_MAX

        # Evaluate DEM along vertical line at center
        uy_profile = []
        for yi in y_pts:
            u = dem_model(jnp.float32(W / 2), yi, jnp.float32(D / 2), E_norm)
            uy_profile.append(float(u[1]))
        uy_arr = np.array(uy_profile) * 1000  # mm → μm

        ax.plot(uy_arr, np.array(y_pts) * 1000, "-", color=colors[cond], lw=2)
        ax.axhline(H * 1000, color="gray", ls="--", alpha=0.5, label="Top")
        ax.set_xlabel("u_y [μm]")
        ax.set_ylabel("y [μm]" if col == 0 else "")
        ax.set_title(f"u_y max = {r['uy']*1000:.2f} μm", fontsize=10)
        ax.grid(True, alpha=0.3)

    # --- Row 3: Sensitivity ∂u_y/∂θ ---
    ax_sens = fig.add_subplot(gs[2, :])
    bar_width = 0.8 / n
    x_pos = np.arange(20)

    for i, cond in enumerate(conds):
        r = results[cond]
        offset = (i - n / 2 + 0.5) * bar_width
        ax_sens.bar(
            x_pos + offset, r["grads"], bar_width, color=colors[cond], alpha=0.7, label=cond
        )

    ax_sens.set_xlabel("Parameter index θ[i]")
    ax_sens.set_ylabel("∂u_y / ∂θ[i]")
    ax_sens.set_title("Exact Sensitivity via JAX Autodiff (no finite differences)")
    ax_sens.set_xticks(range(20))
    ax_sens.legend(fontsize=8, ncol=2)
    ax_sens.grid(True, alpha=0.3, axis="y")

    out_path = str(SCRIPT_DIR / "e2e_pipeline_3d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(
        f"{'Condition':<22} {'DI':>6} {'E [Pa]':>8} {'u_y [μm]':>10} "
        f"{'Fwd [ms]':>9} {'Grad [ms]':>10}"
    )
    print("-" * 70)
    for cond in conds:
        r = results[cond]
        print(
            f"{cond:<22} {r['DI']:>6.3f} {r['E']:>8.1f} "
            f"{r['uy']*1000:>10.2f} {r['t_forward']:>9.2f} {r['t_grad']:>10.2f}"
        )

    # Speedup vs Abaqus
    if conds:
        avg_fwd = np.mean([results[c]["t_forward"] for c in conds])
        abaqus_time_ms = 120_000  # ~2 min per Abaqus run
        print(f"\nAvg forward pass: {avg_fwd:.2f} ms")
        print(f"Abaqus FEM: ~{abaqus_time_ms/1000:.0f} s")
        print(f"Speedup: ~{abaqus_time_ms/avg_fwd:.0f}x")
        print("+ exact ∂u/∂θ via autodiff (Abaqus: impossible)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensitivity", action="store_true", help="Detailed sensitivity analysis only"
    )
    args = parser.parse_args()

    run_demo()
