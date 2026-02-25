#!/usr/bin/env python3
"""
gradient_tmcmc.py — Gradient-based TMCMC with HMC mutation kernel.

Standard TMCMC uses random-walk Metropolis for the mutation step.
This implementation replaces it with Hamiltonian Monte Carlo (HMC),
exploiting the fully differentiable pipeline:

    θ → DeepONet → φ → DI → E(DI) → DEM → u → logL(θ|data)
    ∂logL/∂θ via JAX autodiff (0.04 ms per gradient)

Benefits:
    - HMC proposes moves along probability contours → higher acceptance
    - Fewer stages needed (better mixing per stage)
    - Exact gradients (no finite differences)

Usage:
    python gradient_tmcmc.py                        # full comparison
    python gradient_tmcmc.py --condition Dysbiotic_HOBIC --n-particles 200
"""

import argparse
import json
import sys
import time
from pathlib import Path
from functools import partial

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

# Import pipeline components
from deeponet_hamilton import DeepONet
from dem_elasticity_3d import ElasticityNetwork
from e2e_differentiable_pipeline import (
    load_deeponet, load_dem, load_theta_map,
    deeponet_predict_final, compute_di, di_to_E,
    dem_predict_max_uy,
    E_MAX, E_MIN, W, H, D, N_SPECIES, SPECIES,
    CONDITION_CHECKPOINTS, CONDITION_RUNS,
)


# ============================================================
# Differentiable Log-Likelihood
# ============================================================
def make_differentiable_log_likelihood(don_model, theta_lo, theta_width,
                                       dem_model, observed_uy, sigma_obs=0.05):
    """
    Create a JAX-differentiable log-likelihood function.

    Uses the E2E pipeline: θ → φ → DI → E → u_y
    Then compares predicted u_y with observed u_y.

    Args:
        don_model: DeepONet model
        theta_lo, theta_width: normalization stats
        dem_model: DEM model
        observed_uy: scalar, observed u_y displacement
        sigma_obs: observation noise std

    Returns:
        log_likelihood(theta) function (JAX-differentiable)
    """
    def log_likelihood(theta):
        # Forward pipeline
        phi = deeponet_predict_final(don_model, theta, theta_lo, theta_width)
        di = compute_di(phi)
        E = di_to_E(di)
        uy_pred = dem_predict_max_uy(dem_model, E)

        # Gaussian log-likelihood
        residual = observed_uy - uy_pred
        logL = -0.5 * (residual / sigma_obs) ** 2
        return logL

    return log_likelihood


def make_multi_output_log_likelihood(don_model, theta_lo, theta_width,
                                      dem_model, observed_phi, sigma_phi=0.02,
                                      observed_uy=None, sigma_uy=0.05):
    """
    Multi-output likelihood: species fractions + displacement.
    """
    def log_likelihood(theta):
        phi = deeponet_predict_final(don_model, theta, theta_lo, theta_width)

        # Species fraction likelihood
        residual_phi = observed_phi - phi
        logL = -0.5 * jnp.sum((residual_phi / sigma_phi) ** 2)

        # Optional displacement likelihood
        if observed_uy is not None:
            di = compute_di(phi)
            E = di_to_E(di)
            uy_pred = dem_predict_max_uy(dem_model, E)
            logL = logL - 0.5 * ((observed_uy - uy_pred) / sigma_uy) ** 2

        return logL

    return log_likelihood


# ============================================================
# HMC Kernel (Leapfrog integrator)
# ============================================================
def hmc_step(key, theta, log_prob_and_grad, step_size, n_leapfrog,
             bounds_lo, bounds_hi):
    """
    Single HMC step with leapfrog integration and reflection at bounds.

    Args:
        key: JAX random key
        theta: current position (d,)
        log_prob_and_grad: function returning (logp, grad_logp)
        step_size: leapfrog step size
        n_leapfrog: number of leapfrog steps
        bounds_lo, bounds_hi: parameter bounds (d,)

    Returns:
        new_theta, accepted (bool), new_logp
    """
    d = theta.shape[0]

    # Sample momentum
    momentum = jr.normal(key, (d,))

    # Current energy
    logp_current, grad_current = log_prob_and_grad(theta)
    H_current = -logp_current + 0.5 * jnp.sum(momentum ** 2)

    # Leapfrog integration
    q = theta
    p = momentum

    # Half step for momentum
    p = p + 0.5 * step_size * grad_current

    # Full steps
    for _ in range(n_leapfrog - 1):
        q = q + step_size * p
        # Reflect at bounds
        q = jnp.clip(q, bounds_lo, bounds_hi)
        _, grad_q = log_prob_and_grad(q)
        p = p + step_size * grad_q

    # Last full step for position
    q = q + step_size * p
    q = jnp.clip(q, bounds_lo, bounds_hi)

    # Half step for momentum
    logp_proposed, grad_proposed = log_prob_and_grad(q)
    p = p + 0.5 * step_size * grad_proposed

    # Negate momentum (for reversibility)
    p = -p

    # Accept/reject
    H_proposed = -logp_proposed + 0.5 * jnp.sum(p ** 2)
    log_alpha = H_current - H_proposed

    # Metropolis correction
    k_accept = jr.split(key)[1]
    accepted = jnp.log(jr.uniform(k_accept)) < log_alpha

    new_theta = jnp.where(accepted, q, theta)
    new_logp = jnp.where(accepted, logp_proposed, logp_current)

    return new_theta, accepted, new_logp


# ============================================================
# TMCMC with HMC Mutation
# ============================================================
def tmcmc_hmc(
    log_likelihood,
    grad_log_likelihood,
    prior_bounds,
    n_particles=200,
    max_stages=30,
    target_ess_ratio=0.5,
    hmc_step_size=0.01,
    hmc_n_leapfrog=10,
    seed=42,
    label="HMC-TMCMC",
):
    """
    TMCMC with HMC mutation kernel.

    Args:
        log_likelihood: callable(theta) -> scalar (JAX)
        grad_log_likelihood: callable(theta) -> (scalar, grad)
        prior_bounds: (d, 2) array
        n_particles: number of particles
        hmc_step_size: leapfrog step size
        hmc_n_leapfrog: leapfrog steps per HMC proposal
    """
    rng = np.random.default_rng(seed)
    d = prior_bounds.shape[0]

    bounds_lo = jnp.array(prior_bounds[:, 0], dtype=jnp.float32)
    bounds_hi = jnp.array(prior_bounds[:, 1], dtype=jnp.float32)

    # Free dimensions
    free_mask = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
    d_free = int(np.sum(free_mask))

    # Sample from prior
    particles = np.zeros((n_particles, d), dtype=np.float32)
    for i in range(d):
        lo, hi = prior_bounds[i]
        if abs(hi - lo) < 1e-12:
            particles[:, i] = lo
        else:
            particles[:, i] = rng.uniform(lo, hi, n_particles)

    # Evaluate initial log-likelihoods
    print(f"\n{label}: {n_particles} particles, {d_free} free dims")
    t0 = time.time()

    # JIT-compile the likelihood and gradient
    logL_jit = jax.jit(log_likelihood)
    grad_jit = jax.jit(jax.value_and_grad(log_likelihood))

    logL = np.array([float(logL_jit(jnp.array(p))) for p in particles])
    t_init = time.time() - t0
    print(f"  Init: {t_init:.1f}s ({t_init/n_particles*1000:.1f} ms/particle)")

    beta = 0.0
    betas = [0.0]
    stage_times = []
    accept_rates = []
    ess_history = []
    stage = 0

    while beta < 1.0 and stage < max_stages:
        stage += 1
        t_stage = time.time()

        # Find next beta via bisection
        def compute_ess(db):
            w = db * logL
            w = w - w.max()
            w = np.exp(w)
            return (np.sum(w) ** 2) / np.sum(w ** 2)

        db_lo, db_hi = 0.0, 1.0 - beta
        for _ in range(50):
            db_mid = (db_lo + db_hi) / 2
            ess = compute_ess(db_mid)
            if ess > target_ess_ratio * n_particles:
                db_lo = db_mid
            else:
                db_hi = db_mid

        delta_beta = db_lo
        if delta_beta < 1e-6:
            delta_beta = 1.0 - beta

        beta_new = min(beta + delta_beta, 1.0)

        # Importance weights
        w = (beta_new - beta) * logL
        w = w - w.max()
        w = np.exp(w)
        ess_val = (np.sum(w) ** 2) / np.sum(w ** 2)
        w = w / w.sum()

        # Resample
        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles = particles[idx].copy()
        logL = logL[idx].copy()

        # HMC Mutation
        # Tempered log-probability: beta * logL(theta)
        def tempered_logp_and_grad(theta):
            val, grad = grad_jit(theta)
            return beta_new * val, beta_new * grad

        # Adaptive step size based on parameter scale
        param_scales = np.array(prior_bounds[:, 1] - prior_bounds[:, 0])
        param_scales = np.where(param_scales < 1e-12, 1.0, param_scales)
        adapted_step = hmc_step_size * np.mean(param_scales[free_mask])

        n_accept = 0
        key = jr.PRNGKey(seed + stage * 1000)

        for i in range(n_particles):
            k_i = jr.fold_in(key, i)
            theta_i = jnp.array(particles[i])

            new_theta, accepted, new_logp = hmc_step(
                k_i, theta_i, tempered_logp_and_grad,
                adapted_step, hmc_n_leapfrog,
                bounds_lo, bounds_hi,
            )

            if bool(accepted):
                particles[i] = np.array(new_theta)
                logL[i] = float(new_logp) / beta_new  # un-temper
                n_accept += 1

        beta = beta_new
        betas.append(beta)
        dt = time.time() - t_stage
        stage_times.append(dt)
        ar = n_accept / n_particles
        accept_rates.append(ar)
        ess_history.append(ess_val)

        print(f"  Stage {stage:2d}: beta={beta:.4f}, "
              f"accept={ar:.2f}, ESS={ess_val:.0f}, "
              f"logL=[{logL.min():.1f}, {logL.max():.1f}], "
              f"{dt:.1f}s")

    total_time = time.time() - t0
    best_idx = np.argmax(logL)

    return {
        "label": label,
        "samples": particles,
        "log_likelihoods": logL,
        "betas": np.array(betas),
        "theta_MAP": particles[best_idx],
        "total_time": total_time,
        "stage_times": stage_times,
        "accept_rates": accept_rates,
        "ess_history": ess_history,
        "n_stages": len(betas) - 1,
    }


# ============================================================
# Standard TMCMC (Random-Walk Mutation) for comparison
# ============================================================
def tmcmc_rw(
    log_likelihood,
    prior_bounds,
    n_particles=200,
    max_stages=30,
    target_ess_ratio=0.5,
    seed=42,
    label="RW-TMCMC",
):
    """Standard TMCMC with random-walk Metropolis mutation."""
    rng = np.random.default_rng(seed)
    d = prior_bounds.shape[0]
    free_dims = np.where(np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12)[0]
    d_free = len(free_dims)

    particles = np.zeros((n_particles, d), dtype=np.float32)
    for i in range(d):
        lo, hi = prior_bounds[i]
        if abs(hi - lo) < 1e-12:
            particles[:, i] = lo
        else:
            particles[:, i] = rng.uniform(lo, hi, n_particles)

    print(f"\n{label}: {n_particles} particles, {d_free} free dims")
    t0 = time.time()

    logL_jit = jax.jit(log_likelihood)
    logL = np.array([float(logL_jit(jnp.array(p))) for p in particles])
    t_init = time.time() - t0
    print(f"  Init: {t_init:.1f}s ({t_init/n_particles*1000:.1f} ms/particle)")

    beta = 0.0
    betas = [0.0]
    stage_times = []
    accept_rates = []
    ess_history = []
    stage = 0

    while beta < 1.0 and stage < max_stages:
        stage += 1
        t_stage = time.time()

        def compute_ess(db):
            w = db * logL
            w = w - w.max()
            w = np.exp(w)
            return (np.sum(w) ** 2) / np.sum(w ** 2)

        db_lo, db_hi = 0.0, 1.0 - beta
        for _ in range(50):
            db_mid = (db_lo + db_hi) / 2
            if compute_ess(db_mid) > target_ess_ratio * n_particles:
                db_lo = db_mid
            else:
                db_hi = db_mid

        delta_beta = db_lo
        if delta_beta < 1e-6:
            delta_beta = 1.0 - beta

        beta_new = min(beta + delta_beta, 1.0)

        w = (beta_new - beta) * logL
        w = w - w.max()
        w = np.exp(w)
        ess_val = (np.sum(w) ** 2) / np.sum(w ** 2)
        w = w / w.sum()

        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles = particles[idx].copy()
        logL = logL[idx].copy()

        # Adaptive covariance
        cov = np.cov(particles[:, free_dims].T)
        if d_free == 1:
            cov = np.atleast_2d(cov)
        cov *= 0.04

        n_accept = 0
        for i in range(n_particles):
            proposal = particles[i].copy()
            proposal[free_dims] += rng.multivariate_normal(np.zeros(d_free), cov)

            in_bounds = True
            for dim in free_dims:
                if proposal[dim] < prior_bounds[dim, 0] or proposal[dim] > prior_bounds[dim, 1]:
                    in_bounds = False
                    break

            if in_bounds:
                logL_new = float(logL_jit(jnp.array(proposal)))
                log_alpha = beta_new * (logL_new - logL[i])
                if np.log(rng.random()) < log_alpha:
                    particles[i] = proposal
                    logL[i] = logL_new
                    n_accept += 1

        beta = beta_new
        betas.append(beta)
        dt = time.time() - t_stage
        stage_times.append(dt)
        ar = n_accept / n_particles
        accept_rates.append(ar)
        ess_history.append(ess_val)

        print(f"  Stage {stage:2d}: beta={beta:.4f}, "
              f"accept={ar:.2f}, ESS={ess_val:.0f}, "
              f"logL=[{logL.min():.1f}, {logL.max():.1f}], "
              f"{dt:.1f}s")

    total_time = time.time() - t0
    best_idx = np.argmax(logL)

    return {
        "label": label,
        "samples": particles,
        "log_likelihoods": logL,
        "betas": np.array(betas),
        "theta_MAP": particles[best_idx],
        "total_time": total_time,
        "stage_times": stage_times,
        "accept_rates": accept_rates,
        "ess_history": ess_history,
        "n_stages": len(betas) - 1,
    }


# ============================================================
# Visualization
# ============================================================
def plot_comparison(result_rw, result_hmc, theta_true, condition, free_dims):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"TMCMC Comparison: Random-Walk vs HMC Mutation\n"
                 f"Condition: {condition}", fontsize=14, fontweight="bold")

    colors = {"RW-TMCMC": "#F44336", "HMC-TMCMC": "#2196F3"}

    # 1. Beta schedule
    ax = axes[0, 0]
    for r in [result_rw, result_hmc]:
        ax.plot(r["betas"], "o-", color=colors[r["label"]], label=r["label"], ms=4)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Beta")
    ax.set_title("Tempering Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Acceptance rates
    ax = axes[0, 1]
    for r in [result_rw, result_hmc]:
        ax.plot(r["accept_rates"], "o-", color=colors[r["label"]],
                label=f"{r['label']} (avg={np.mean(r['accept_rates']):.2f})", ms=4)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Mutation Acceptance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. Summary bar chart
    ax = axes[0, 2]
    metrics = ["Stages", "Time [s]", "Avg Accept"]
    rw_vals = [result_rw["n_stages"], result_rw["total_time"],
               np.mean(result_rw["accept_rates"])]
    hmc_vals = [result_hmc["n_stages"], result_hmc["total_time"],
                np.mean(result_hmc["accept_rates"])]

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, rw_vals, w, color=colors["RW-TMCMC"], label="RW", alpha=0.8)
    ax.bar(x + w/2, hmc_vals, w, color=colors["HMC-TMCMC"], label="HMC", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title("Performance Summary")

    for i, (rv, hv) in enumerate(zip(rw_vals, hmc_vals)):
        ax.text(i - w/2, rv + 0.02 * max(rw_vals + hmc_vals), f"{rv:.2f}",
                ha='center', va='bottom', fontsize=8)
        ax.text(i + w/2, hv + 0.02 * max(rw_vals + hmc_vals), f"{hv:.2f}",
                ha='center', va='bottom', fontsize=8)

    # 4-5. Posterior marginals (top 6 free dims)
    show_dims = free_dims[:6]
    ax_post = axes[1, 0]
    for i, dim in enumerate(show_dims):
        offset = i * 0.15
        rw_samples = result_rw["samples"][:, dim]
        hmc_samples = result_hmc["samples"][:, dim]

        ax_post.hist(rw_samples, bins=20, alpha=0.3, color=colors["RW-TMCMC"],
                     density=True, label="RW" if i == 0 else None)
        ax_post.hist(hmc_samples, bins=20, alpha=0.3, color=colors["HMC-TMCMC"],
                     density=True, label="HMC" if i == 0 else None)

    ax_post.set_title(f"Posterior Marginals (θ[{show_dims[0]}])")
    ax_post.legend()
    ax_post.grid(True, alpha=0.3)

    # 5. MAP comparison
    ax = axes[1, 1]
    n_show = min(10, len(free_dims))
    dims_show = free_dims[:n_show]
    x = np.arange(n_show)
    w = 0.25

    ax.bar(x - w, theta_true[dims_show], w, color="gray", alpha=0.6, label="True")
    ax.bar(x, result_rw["theta_MAP"][dims_show], w,
           color=colors["RW-TMCMC"], alpha=0.8, label="RW MAP")
    ax.bar(x + w, result_hmc["theta_MAP"][dims_show], w,
           color=colors["HMC-TMCMC"], alpha=0.8, label="HMC MAP")
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ[{d}]" for d in dims_show], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("MAP Estimates vs True")
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Log-likelihood convergence
    ax = axes[1, 2]
    for r in [result_rw, result_hmc]:
        # Per-stage mean logL (approximate from final samples)
        ax.plot(range(1, r["n_stages"] + 1),
                [np.mean(r["log_likelihoods"])] * r["n_stages"],
                "--", color=colors[r["label"]], alpha=0.5)
        # ESS history
        ax.plot(range(1, r["n_stages"] + 1), r["ess_history"],
                "o-", color=colors[r["label"]], label=f"{r['label']} ESS", ms=4)
    ax.set_xlabel("Stage")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = str(SCRIPT_DIR / "gradient_tmcmc_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


# ============================================================
# Main
# ============================================================
def main(condition="Dysbiotic_HOBIC", n_particles=200, seed=42):
    print("=" * 70)
    print("Gradient-Based TMCMC: HMC vs Random-Walk Comparison")
    print("=" * 70)

    # Load models
    don_model, theta_lo, theta_width = load_deeponet(condition)
    dem_model = load_dem()
    print(f"[OK] Models loaded for {condition}")

    # Load prior bounds
    bounds_file = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(bounds_file) as f:
        cfg = json.load(f)

    strategy = cfg["strategies"][condition]
    locks = set(strategy.get("locks", []))
    custom = strategy.get("bounds", {})

    prior_bounds = np.zeros((20, 2))
    for i in range(20):
        if i in locks:
            prior_bounds[i] = [0.0, 0.0]
        elif str(i) in custom:
            prior_bounds[i] = custom[str(i)]
        else:
            prior_bounds[i] = cfg["default_bounds"]

    free_dims = np.where(np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12)[0]
    print(f"[OK] Prior: {len(free_dims)} free dims out of 20")

    # Create synthetic "true" theta and observed data
    rng = np.random.default_rng(seed)
    theta_true = np.zeros(20, dtype=np.float32)
    for i in range(20):
        lo, hi = prior_bounds[i]
        if abs(hi - lo) > 1e-12:
            theta_true[i] = rng.uniform(lo, hi)

    # Generate "observed" data from true theta
    theta_true_jax = jnp.array(theta_true)
    phi_true = deeponet_predict_final(don_model, theta_true_jax, theta_lo, theta_width)
    di_true = compute_di(phi_true)
    E_true = di_to_E(di_true)
    uy_true = dem_predict_max_uy(dem_model, E_true)

    print(f"  True: φ={np.array(phi_true)}, DI={float(di_true):.3f}, "
          f"E={float(E_true):.0f} Pa, u_y={float(uy_true)*1000:.2f} μm")

    # Add noise to observations
    sigma_phi = 0.03
    observed_phi = jnp.array(np.array(phi_true) + rng.normal(0, sigma_phi, 5),
                             dtype=jnp.float32)
    observed_phi = jnp.clip(observed_phi, 0.0, 1.0)

    sigma_uy = 0.0001  # 0.1 μm noise
    observed_uy = float(uy_true) + rng.normal(0, sigma_uy)

    # Create differentiable log-likelihood
    log_likelihood = make_multi_output_log_likelihood(
        don_model, theta_lo, theta_width, dem_model,
        observed_phi, sigma_phi=sigma_phi,
        observed_uy=jnp.float32(observed_uy), sigma_uy=sigma_uy,
    )

    # Warmup JIT
    print("  JIT warmup...")
    _ = jax.jit(log_likelihood)(theta_true_jax)
    _ = jax.jit(jax.value_and_grad(log_likelihood))(theta_true_jax)
    print("  [OK] JIT ready")

    # Run standard TMCMC (Random-Walk)
    result_rw = tmcmc_rw(
        log_likelihood, prior_bounds,
        n_particles=n_particles, seed=seed,
        label="RW-TMCMC",
    )

    # Run gradient TMCMC (HMC)
    result_hmc = tmcmc_hmc(
        log_likelihood,
        jax.value_and_grad(log_likelihood),
        prior_bounds,
        n_particles=n_particles, seed=seed,
        hmc_step_size=0.005,
        hmc_n_leapfrog=5,
        label="HMC-TMCMC",
    )

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'RW-TMCMC':>12} {'HMC-TMCMC':>12} {'Ratio':>10}")
    print("-" * 65)
    print(f"{'Total time [s]':<25} {result_rw['total_time']:>12.1f} "
          f"{result_hmc['total_time']:>12.1f} "
          f"{result_rw['total_time']/result_hmc['total_time']:>10.1f}x")
    print(f"{'Stages':<25} {result_rw['n_stages']:>12d} "
          f"{result_hmc['n_stages']:>12d}")
    print(f"{'Avg acceptance':<25} {np.mean(result_rw['accept_rates']):>12.3f} "
          f"{np.mean(result_hmc['accept_rates']):>12.3f}")
    print(f"{'Final max logL':<25} {result_rw['log_likelihoods'].max():>12.1f} "
          f"{result_hmc['log_likelihoods'].max():>12.1f}")

    # MAP error
    rw_err = np.sqrt(np.mean((result_rw["theta_MAP"][free_dims] -
                               theta_true[free_dims]) ** 2))
    hmc_err = np.sqrt(np.mean((result_hmc["theta_MAP"][free_dims] -
                                theta_true[free_dims]) ** 2))
    print(f"{'MAP RMSE (free dims)':<25} {rw_err:>12.4f} {hmc_err:>12.4f}")

    # Plot
    plot_comparison(result_rw, result_hmc, theta_true, condition, free_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="Dysbiotic_HOBIC")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(condition=args.condition, n_particles=args.n_particles, seed=args.seed)
