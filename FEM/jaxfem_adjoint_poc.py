#!/usr/bin/env python3
"""
jaxfem_adjoint_poc.py
======================
B2: JAX-FEM adjoint inverse problem proof-of-concept.

Demonstrates inverse problem capability using JAX autodiff:
  Forward: theta (TMCMC params) → Hamilton 2D ODE → DI field → E_eff(x,y)
  Inverse: Given target E_eff*, find theta that minimizes ||E_eff(theta) - E_eff*||^2

Uses JAX's autodifferentiation through the entire forward chain:
  1. theta → A,b matrices
  2. Hamilton Newton solver (vmapped over spatial grid)
  3. DI computation (Shannon entropy)
  4. DI → E_eff power-law mapping
  5. Loss = ||E_eff - E_eff_target||^2

This is a PoC — full inverse (theta → Abaqus stress) requires Abaqus adjoint
or finite-difference gradients through the FEM solver.

Usage
-----
  # Run PoC with default target (commensal E_eff):
  python jaxfem_adjoint_poc.py

  # Custom target:
  python jaxfem_adjoint_poc.py --target-condition commensal_static

  # Quick test:
  python jaxfem_adjoint_poc.py --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Import JAX only when needed (avoid slow import for --help)
_jax_loaded = False


def _ensure_jax():
    global _jax_loaded, jax, jnp
    if not _jax_loaded:
        import jax as _jax
        import jax.numpy as _jnp

        _jax.config.update("jax_enable_x64", True)
        jax = _jax
        jnp = _jnp
        _jax_loaded = True


# Material model (power-law)
E_MAX = 10.0e9  # Pa
E_MIN = 0.5e9  # Pa
DI_SCALE = 0.025778
DI_EXP = 2.0


def di_to_eeff_jax(di, e_max=E_MAX, e_min=E_MIN, di_scale=DI_SCALE, di_exp=DI_EXP):
    """DI → E_eff (JAX differentiable)."""
    r = jnp.clip(di / di_scale, 0, 1)
    return e_max * (1 - r) ** di_exp + e_min * r


def compute_di_jax(phi):
    """Compute DI from phi (5,) — JAX differentiable."""
    eps = 1e-12
    phi_sum = jnp.sum(phi)
    phi_sum_safe = jnp.where(phi_sum > eps, phi_sum, 1.0)
    p = phi / phi_sum_safe
    log_p = jnp.where(p > eps, jnp.log(p), 0.0)
    H = -jnp.sum(p * log_p)
    return 1.0 - H / jnp.log(5.0)


def phi_pg_to_eeff_jax(phi, e_max=E_MAX, e_min=E_MIN, phi_crit=0.05, hill_m=4.0):
    """φ_Pg → E(φ_Pg) Hill sigmoid (JAX differentiable, mechanism-based).

    Maps P. gingivalis abundance directly to effective stiffness via a
    Hill-type sigmoid, bypassing the entropy-based DI entirely.
    When φ_Pg >> φ_crit the biofilm is soft (E → E_MIN);
    when φ_Pg << φ_crit the biofilm is stiff (E → E_MAX).
    """
    phi_pg = phi[..., 4]  # P. gingivalis = species index 4
    xm = jnp.power(jnp.clip(phi_pg / phi_crit, 0.0, None), hill_m)
    sig = xm / (1.0 + xm)
    return e_max - (e_max - e_min) * sig


def build_forward_0d(K_hill=0.05, n_hill=4.0, n_steps=500, dt=0.01):
    """Build a simplified 0D forward model: theta → phi_final → DI → E_eff.

    Uses simplified Lotka-Volterra dynamics (not full Hamilton) for PoC:
      dphi_i/dt = phi_i * (b_i - sum_j A_ij * phi_j) * gate_hill(i)

    This is analytically differentiable by JAX.
    """
    _ensure_jax()

    def theta_to_Ab(theta):
        """Convert 20-vector theta to A(5x5), b(5)."""
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

    K_h = jnp.array(K_hill)
    n_h = jnp.array(n_hill)

    def forward(theta):
        """theta (20,) → E_eff scalar."""
        A, b = theta_to_Ab(theta)

        # Initial conditions
        phi = jnp.array([0.12, 0.12, 0.08, 0.05, 0.02])

        # Lotka-Volterra ODE integration (simplified Hamilton)
        def step(phi, _):
            interaction = A @ phi
            # Hill gate for P.gingivalis (species 4)
            fn = jnp.maximum(phi[3], 0.0)
            hill = fn**n_h / (K_h**n_h + fn**n_h + 1e-12)
            # Growth rates
            growth = phi * (b - interaction)
            # Apply Hill gate to Pg
            growth = growth.at[4].set(growth[4] * hill)
            phi_new = phi + dt * growth
            phi_new = jnp.clip(phi_new, 1e-10, 1.0)
            # Normalize if > 1
            phi_sum = jnp.sum(phi_new)
            phi_new = jnp.where(phi_sum > 0.999, phi_new * 0.999 / phi_sum, phi_new)
            return phi_new, phi_new

        phi_final, phi_traj = jax.lax.scan(step, phi, jnp.arange(n_steps))

        # DI from final state
        di = compute_di_jax(phi_final)
        e_eff = di_to_eeff_jax(di)
        return e_eff, phi_final, di

    return forward, theta_to_Ab


def run_inverse_poc(args):
    """Run the inverse problem PoC."""
    _ensure_jax()

    print("=" * 60)
    print("JAX-FEM Adjoint Inverse Problem PoC")
    print("=" * 60)

    n_steps = 100 if args.quick else 500
    forward_fn, _ = build_forward_0d(K_hill=args.k_hill, n_hill=args.n_hill, n_steps=n_steps)

    # Generate target from a known theta
    _PARAM_KEYS = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",
        "a13",
        "a14",
        "a23",
        "a24",
        "a55",
        "b5",
        "a15",
        "a25",
        "a35",
        "a45",
    ]

    # Load target theta from TMCMC
    _TMCMC_ROOT = _HERE.parent
    _RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
    COND_RUNS = {
        "dh_baseline": _RUNS_ROOT / "dh_baseline",
        "commensal_static": _RUNS_ROOT / "commensal_static",
        "commensal_hobic": _RUNS_ROOT / "commensal_hobic",
        "dysbiotic_static": _RUNS_ROOT / "dysbiotic_static",
    }

    target_cond = args.target_condition
    theta_target = None
    if target_cond in COND_RUNS:
        tp = COND_RUNS[target_cond] / "theta_MAP.json"
        if tp.exists():
            with open(tp) as f:
                d = json.load(f)
            if "theta_full" in d:
                theta_target = jnp.array(d["theta_full"], dtype=jnp.float64)
            elif "theta_sub" in d:
                theta_target = jnp.array(d["theta_sub"], dtype=jnp.float64)

    if theta_target is None:
        # Default synthetic target
        theta_target = jnp.array(
            [
                1.5,
                0.5,
                1.2,
                2.0,
                0.3,
                0.8,
                0.3,
                1.5,
                3.0,
                1.0,
                0.5,
                0.3,
                0.5,
                0.3,
                1.0,
                0.1,
                0.5,
                0.3,
                3.0,
                2.0,
            ],
            dtype=jnp.float64,
        )
        print("  Using synthetic target theta")
    else:
        print(f"  Target theta from: {target_cond}")

    # Compute target E_eff
    print("\n[1/3] Computing target E_eff...")
    e_target, phi_target, di_target = forward_fn(theta_target)
    e_target_pg = phi_pg_to_eeff_jax(phi_target)
    print(f"  Target DI: {float(di_target):.6f}")
    print(f"  Target E_eff (DI model):   {float(e_target)*1e-9:.4f} GPa")
    print(f"  Target E_eff (φ_Pg model): {float(e_target_pg)*1e-9:.4f} GPa")
    print(f"  Target φ_Pg: {float(phi_target[4]):.6f}")
    print(f"  Target phi: {[f'{float(p):.4f}' for p in phi_target]}")

    # Initialize from perturbed theta
    key = jax.random.PRNGKey(42)
    theta_init = theta_target + 0.3 * jax.random.normal(key, shape=(20,))
    # Clip to reasonable bounds
    theta_init = jnp.clip(theta_init, -1.0, 10.0)

    e_init, phi_init, di_init = forward_fn(theta_init)
    e_init_pg = phi_pg_to_eeff_jax(phi_init)
    print(f"\n  Initial DI: {float(di_init):.6f}")
    print(f"  Initial E_eff (DI model):   {float(e_init)*1e-9:.4f} GPa")
    print(f"  Initial E_eff (φ_Pg model): {float(e_init_pg)*1e-9:.4f} GPa")

    # Loss function: DI + per-species composition + Pg emphasis
    # Species weights: Pg (idx 4) gets 3x weight
    species_w = jnp.array([1.0, 1.0, 1.0, 1.0, 3.0])

    def loss_fn(theta):
        e_eff, phi, di = forward_fn(theta)
        # DI match
        loss_di = (di - di_target) ** 2 / (di_target**2 + 1e-12)
        # Per-species weighted composition match
        diff2 = (phi - phi_target) ** 2 * species_w
        loss_phi = jnp.sum(diff2) / jnp.sum(phi_target**2 * species_w + 1e-12)
        # L2 regularization (mild, prevents runaway)
        loss_reg = 1e-4 * jnp.sum(theta**2)
        return loss_di + 0.5 * loss_phi + loss_reg

    # Gradient via autodiff
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    # Adam optimizer with cosine LR schedule
    lr_base = args.lr
    n_iters = 50 if args.quick else args.n_iters

    print(
        f"\n[2/3] Running gradient descent (Adam, lr={lr_base}, {n_iters} iters, "
        f"cosine schedule)..."
    )
    t_start = time.perf_counter()

    # Manual Adam with cosine annealing
    theta = theta_init.copy()
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    lr_min = lr_base * 0.01

    history = []
    for i in range(n_iters):
        loss = float(loss_jit(theta))
        g = grad_fn(theta)

        # Cosine annealing LR
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + float(jnp.cos(jnp.pi * i / n_iters)))

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        e_cur, phi_cur, di_cur = forward_fn(theta)
        history.append(
            {
                "iter": i,
                "loss": loss,
                "di": float(di_cur),
                "e_eff_gpa": float(e_cur) * 1e-9,
                "grad_norm": float(jnp.linalg.norm(g)),
                "lr": lr,
            }
        )

        if (i + 1) % max(1, n_iters // 10) == 0 or i == 0:
            print(
                f"  iter {i+1:4d}: loss={loss:.6e}, "
                f"DI={float(di_cur):.6f}, "
                f"E={float(e_cur)*1e-9:.4f} GPa, "
                f"|grad|={float(jnp.linalg.norm(g)):.4e}, "
                f"lr={lr:.5f}"
            )

    dt = time.perf_counter() - t_start

    # Final result
    e_final, phi_final, di_final = forward_fn(theta)
    e_final_pg = phi_pg_to_eeff_jax(phi_final)
    theta_err = float(jnp.linalg.norm(theta - theta_target) / jnp.linalg.norm(theta_target))

    print("\n[3/3] Results:")
    print(f"  Time: {dt:.1f}s ({dt/n_iters*1000:.1f} ms/iter)")
    print(f"  Final DI: {float(di_final):.6f} (target: {float(di_target):.6f})")
    print(
        f"  Final E_eff (DI model):   {float(e_final)*1e-9:.4f} GPa "
        f"(target: {float(e_target)*1e-9:.4f})"
    )
    print(
        f"  Final E_eff (φ_Pg model): {float(e_final_pg)*1e-9:.4f} GPa "
        f"(target: {float(e_target_pg)*1e-9:.4f})"
    )
    print(f"  Final φ_Pg: {float(phi_final[4]):.6f} " f"(target: {float(phi_target[4]):.6f})")
    print(f"  Relative theta error: {theta_err:.4f} ({theta_err*100:.1f}%)")
    print(f"  Final loss: {history[-1]['loss']:.6e}")

    # Parameter comparison
    print("\n  Parameter comparison (target vs recovered):")
    param_names = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",
        "a13",
        "a14",
        "a23",
        "a24",
        "a55",
        "b5",
        "a15",
        "a25",
        "a35",
        "a45",
    ]
    for idx, name in enumerate(param_names):
        t_val = float(theta_target[idx])
        r_val = float(theta[idx])
        err = abs(r_val - t_val) / max(abs(t_val), 1e-6)
        marker = " *" if err > 0.3 else ""
        print(
            f"    {name:5s}: target={t_val:7.3f}  recovered={r_val:7.3f}  "
            f"err={err*100:5.1f}%{marker}"
        )

    # Save results
    out_dir = _HERE / "_adjoint_poc"
    out_dir.mkdir(exist_ok=True)
    result = {
        "target_condition": target_cond,
        "n_iters": n_iters,
        "lr": lr,
        "K_hill": args.k_hill,
        "n_hill": args.n_hill,
        "n_steps_ode": n_steps,
        "target_di": float(di_target),
        "target_e_eff_gpa": float(e_target) * 1e-9,
        "target_e_eff_phi_pg_gpa": float(e_target_pg) * 1e-9,
        "target_phi_pg": float(phi_target[4]),
        "final_di": float(di_final),
        "final_e_eff_gpa": float(e_final) * 1e-9,
        "final_e_eff_phi_pg_gpa": float(e_final_pg) * 1e-9,
        "final_phi_pg": float(phi_final[4]),
        "theta_rel_error": theta_err,
        "final_loss": history[-1]["loss"],
        "timing_s": round(dt, 1),
        "theta_target": [float(x) for x in theta_target],
        "theta_recovered": [float(x) for x in theta],
    }
    with (out_dir / "adjoint_poc_result.json").open("w") as f:
        json.dump(result, f, indent=2)

    # Plot convergence
    _plot_convergence(history, out_dir, float(di_target), float(e_target) * 1e-9)

    print(f"\n  Output: {out_dir}")
    return result


def _plot_convergence(history, out_dir, di_target, e_target_gpa):
    """Plot optimization convergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [h["iter"] for h in history]
    losses = [h["loss"] for h in history]
    dis = [h["di"] for h in history]
    effs = [h["e_eff_gpa"] for h in history]
    grads = [h["grad_norm"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.semilogy(iters, losses, "b-", lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Loss Convergence")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, dis, "g-", lw=2, label="Current")
    ax.axhline(di_target, color="r", ls="--", label=f"Target ({di_target:.4f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("DI")
    ax.set_title("(b) Dysbiotic Index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, effs, "m-", lw=2, label="Current")
    ax.axhline(e_target_gpa, color="r", ls="--", label=f"Target ({e_target_gpa:.2f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$E_{eff}$ [GPa]")
    ax.set_title("(c) Effective Stiffness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(iters, grads, "k-", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|grad|")
    ax.set_title("(d) Gradient Norm")
    ax.grid(True, alpha=0.3)

    fig.suptitle("JAX Adjoint Inverse Problem PoC", fontsize=14, weight="bold")
    fig.tight_layout()
    out = out_dir / "adjoint_poc_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot: {out}")


def main():
    ap = argparse.ArgumentParser(description="JAX-FEM adjoint inverse problem PoC")
    ap.add_argument(
        "--target-condition",
        default="dh_baseline",
        choices=["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"],
    )
    ap.add_argument("--k-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-iters", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    run_inverse_poc(args)


if __name__ == "__main__":
    main()
