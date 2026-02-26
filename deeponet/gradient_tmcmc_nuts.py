#!/usr/bin/env python3
"""
gradient_tmcmc_nuts.py — NUTS-TMCMC: No-U-Turn Sampler within TMCMC.

Extends gradient_tmcmc.py with:
  1. NUTS (No-U-Turn Sampler) — automatic trajectory length
  2. Dual averaging — automatic step-size adaptation
  3. Real data mode — use experimental observations from 4 conditions
  4. Paper-quality figure generation

Usage:
    # Real data, all 4 conditions
    python gradient_tmcmc_nuts.py --real --all-conditions --n-particles 200

    # Single condition, synthetic (for testing)
    python gradient_tmcmc_nuts.py --condition Dysbiotic_HOBIC --n-particles 100

    # NUTS vs HMC vs RW comparison on real data
    python gradient_tmcmc_nuts.py --real --condition Commensal_Static --compare-all
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", False)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

from e2e_differentiable_pipeline import (
    load_deeponet,
    load_dem,
    deeponet_predict_final,
    CONDITION_CHECKPOINTS,
)

# Experimental time points (days) → normalized to [0,1]
DAYS = np.array([1, 3, 6, 10, 15, 21], dtype=np.float32)
T_NORM = DAYS / DAYS[-1]  # [0.048, 0.143, 0.286, 0.476, 0.714, 1.0]

# Condition → run directory for real data
CONDITION_POSTERIOR_DIRS = {
    "Commensal_Static": "commensal_static_posterior",
    "Commensal_HOBIC": "commensal_hobic_posterior",
    "Dysbiotic_Static": "dysbiotic_static_posterior",
    "Dysbiotic_HOBIC": "dysbiotic_hobic_1000p",
}


# ============================================================
# Real Data Loading
# ============================================================
def load_real_data(condition: str):
    """Load experimental observations (6 timepoints × 5 species)."""
    run_name = CONDITION_POSTERIOR_DIRS[condition]
    data_path = RUNS_DIR / run_name / "data.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"No data.npy for {condition} at {data_path}")
    data = np.load(str(data_path))  # (6, 5)
    return data


def load_real_sigma(condition: str):
    """Load sigma_obs from config."""
    run_name = CONDITION_POSTERIOR_DIRS[condition]
    config_path = RUNS_DIR / run_name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("sigma_obs", 0.1)
    return 0.1


# ============================================================
# Multi-Timepoint Differentiable Log-Likelihood (Real Data)
# ============================================================
def make_real_log_likelihood(don_model, theta_lo, theta_width, observed_data, sigma_obs=0.1):
    """
    Log-likelihood over 6 timepoints × 5 species.

    Args:
        don_model: DeepONet model
        theta_lo, theta_width: normalization
        observed_data: (6, 5) array
        sigma_obs: observation noise

    Returns:
        log_likelihood(theta) — JAX-differentiable
    """
    obs = jnp.array(observed_data, dtype=jnp.float32)
    t_norm = jnp.array(T_NORM, dtype=jnp.float32)
    n_t = len(T_NORM)

    def log_likelihood(theta):
        theta_norm = (theta - theta_lo) / theta_width
        logL = jnp.float32(0.0)
        for i in range(n_t):
            phi_pred = don_model(theta_norm, t_norm[i])
            phi_pred = jnp.clip(phi_pred, 0.0, 1.0)
            residual = obs[i] - phi_pred
            logL = logL - 0.5 * jnp.sum((residual / sigma_obs) ** 2)
        return logL

    return log_likelihood


# ============================================================
# NUTS (No-U-Turn Sampler) with Dual Averaging
# ============================================================
def _leapfrog(q, p, grad_fn, step_size, bounds_lo, bounds_hi):
    """Single leapfrog step."""
    _, grad_q = grad_fn(q)
    p = p + 0.5 * step_size * grad_q
    q = q + step_size * p
    q = jnp.clip(q, bounds_lo, bounds_hi)
    logp_new, grad_new = grad_fn(q)
    p = p + 0.5 * step_size * grad_new
    return q, p, logp_new, grad_new


def _compute_hamiltonian(logp, p):
    """H = -logp + 0.5 * p^T p"""
    return -logp + 0.5 * jnp.sum(p**2)


def nuts_step(key, theta, log_prob_and_grad, step_size, bounds_lo, bounds_hi, max_depth=6):
    """
    NUTS (No-U-Turn Sampler) — one transition.

    Uses iterative tree-doubling with a U-turn criterion.
    Bounded version: clip at prior bounds.

    Args:
        key: JAX random key
        theta: current position (d,)
        log_prob_and_grad: fn(theta) → (logp, grad)
        step_size: leapfrog step size (adapted by dual averaging)
        bounds_lo, bounds_hi: parameter bounds
        max_depth: maximum tree depth (trajectory length = 2^depth)

    Returns:
        new_theta, accepted, new_logp, n_leapfrog
    """
    d = theta.shape[0]

    # Initial state
    logp0, grad0 = log_prob_and_grad(theta)
    key, k_mom = jr.split(key)
    p0 = jr.normal(k_mom, (d,))
    H0 = _compute_hamiltonian(logp0, p0)

    # Slice variable for multinomial sampling
    key, k_slice = jr.split(key)
    log_u = jnp.log(jr.uniform(k_slice)) + logp0 - 0.5 * jnp.sum(p0**2)

    # Initialize tree
    q_minus = theta
    q_plus = theta
    p_minus = p0
    p_plus = p0
    q_propose = theta
    logp_propose = logp0
    depth = 0
    n_valid = 1
    keep_going = True
    n_leapfrog_total = 0

    while keep_going and depth < max_depth:
        # Choose direction
        key, k_dir = jr.split(key)
        direction = 2 * int(jr.bernoulli(k_dir)) - 1  # -1 or +1

        # Build tree in chosen direction
        if direction == -1:
            q_inner, p_inner = q_minus, p_minus
        else:
            q_inner, p_inner = q_plus, p_plus

        # Take 2^depth leapfrog steps
        n_steps = 2**depth
        q_new = q_inner
        p_new = p_inner
        q_candidate = q_inner
        logp_candidate = logp0
        n_valid_subtree = 0
        diverged = False

        for _ in range(n_steps):
            q_new, p_new, logp_new, _ = _leapfrog(
                q_new, float(direction) * p_new, log_prob_and_grad, step_size, bounds_lo, bounds_hi
            )
            p_new = float(direction) * p_new
            n_leapfrog_total += 1

            # Check divergence (energy error > 1000)
            H_new = _compute_hamiltonian(logp_new, p_new)
            if float(H_new - H0) > 1000:
                diverged = True
                break

            # Count valid states (within slice)
            if float(-H_new) > float(log_u):
                n_valid_subtree += 1
                # Multinomial: accept with prob 1/n_valid_subtree
                key, k_accept = jr.split(key)
                if float(jr.uniform(k_accept)) < 1.0 / max(n_valid_subtree, 1):
                    q_candidate = q_new
                    logp_candidate = logp_new

        if diverged:
            break

        # Update tree endpoints
        if direction == -1:
            q_minus = q_new
            p_minus = p_new
        else:
            q_plus = q_new
            p_plus = p_new

        # Accept subtree proposal
        if n_valid_subtree > 0:
            key, k_sub = jr.split(key)
            accept_prob = n_valid_subtree / max(n_valid + n_valid_subtree, 1)
            if float(jr.uniform(k_sub)) < accept_prob:
                q_propose = q_candidate
                logp_propose = logp_candidate

        n_valid += n_valid_subtree

        # U-turn check
        dq = q_plus - q_minus
        uturn = (float(jnp.sum(dq * p_minus)) < 0) or (float(jnp.sum(dq * p_plus)) < 0)
        keep_going = not uturn and not diverged

        depth += 1

    accepted = not jnp.array_equal(q_propose, theta)
    return q_propose, accepted, float(logp_propose), n_leapfrog_total


def dual_averaging_init(
    step_size_init=1.0, target_accept=0.65, gamma=0.05, t0=10, kappa=0.75, eps_max_factor=5.0
):
    """Initialize dual averaging state for step size adaptation."""
    return {
        "log_eps": np.log(step_size_init),
        "log_eps_bar": np.log(step_size_init),
        "H_bar": 0.0,
        "mu": np.log(10 * step_size_init),
        "target_accept": target_accept,
        "gamma": gamma,
        "t0": t0,
        "kappa": kappa,
        "m": 0,
        "eps_max": step_size_init * eps_max_factor,
    }


def dual_averaging_update(state, accept_prob):
    """Update step size via dual averaging (Hoffman & Gelman 2014)."""
    m = state["m"] + 1
    eta = 1.0 / (m + state["t0"])

    H_bar = (1 - eta) * state["H_bar"] + eta * (state["target_accept"] - accept_prob)

    log_eps = state["mu"] - np.sqrt(m) / state["gamma"] * H_bar
    # Clamp step size to prevent explosion
    log_eps = min(log_eps, np.log(state["eps_max"]))
    m_kappa = m ** (-state["kappa"])
    log_eps_bar = m_kappa * log_eps + (1 - m_kappa) * state["log_eps_bar"]
    log_eps_bar = min(log_eps_bar, np.log(state["eps_max"]))

    return {
        **state,
        "log_eps": log_eps,
        "log_eps_bar": log_eps_bar,
        "H_bar": H_bar,
        "m": m,
    }


# ============================================================
# HMC Step (imported from gradient_tmcmc but streamlined)
# ============================================================
def hmc_step(key, theta, log_prob_and_grad, step_size, n_leapfrog, bounds_lo, bounds_hi):
    """Single HMC step with leapfrog integration."""
    d = theta.shape[0]
    momentum = jr.normal(key, (d,))
    logp_current, grad_current = log_prob_and_grad(theta)
    H_current = -logp_current + 0.5 * jnp.sum(momentum**2)

    q, p = theta, momentum
    p = p + 0.5 * step_size * grad_current

    for _ in range(n_leapfrog - 1):
        q = q + step_size * p
        q = jnp.clip(q, bounds_lo, bounds_hi)
        _, grad_q = log_prob_and_grad(q)
        p = p + step_size * grad_q

    q = q + step_size * p
    q = jnp.clip(q, bounds_lo, bounds_hi)
    logp_proposed, grad_proposed = log_prob_and_grad(q)
    p = p + 0.5 * step_size * grad_proposed
    p = -p

    H_proposed = -logp_proposed + 0.5 * jnp.sum(p**2)
    log_alpha = H_current - H_proposed

    k_accept = jr.split(key)[1]
    accepted = jnp.log(jr.uniform(k_accept)) < log_alpha
    new_theta = jnp.where(accepted, q, theta)
    new_logp = jnp.where(accepted, logp_proposed, logp_current)

    return new_theta, accepted, new_logp


# ============================================================
# TMCMC Engine (supports RW / HMC / NUTS mutation)
# ============================================================
def tmcmc_engine(
    log_likelihood,
    prior_bounds,
    mutation="nuts",  # "rw", "hmc", "nuts"
    n_particles=200,
    max_stages=30,
    target_ess_ratio=0.5,
    hmc_step_size=0.01,
    hmc_n_leapfrog=10,
    nuts_max_depth=6,
    warmup_stages=3,  # stages for dual averaging warmup
    seed=42,
    label=None,
    verbose=True,
    log_prior_fn=None,  # GNN prior: theta → scalar (JAX)
):
    """
    Unified TMCMC with selectable mutation kernel.

    Args:
        mutation: "rw" (random-walk), "hmc" (fixed leapfrog), "nuts" (auto)
        warmup_stages: number of stages for NUTS step-size adaptation
        log_prior_fn: optional JAX function theta → log_prior(theta).
            If provided, target at stage j is: log_prior(θ) + β_j * log_L(θ).
            Importance weights still use only L(θ)^{Δβ}.
            Initial particles are sampled from prior (within bounds).
    """
    if label is None:
        label = f"{mutation.upper()}-TMCMC"

    rng = np.random.default_rng(seed)
    d = prior_bounds.shape[0]

    bounds_lo = jnp.array(prior_bounds[:, 0], dtype=jnp.float32)
    bounds_hi = jnp.array(prior_bounds[:, 1], dtype=jnp.float32)
    free_mask = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
    free_dims = np.where(free_mask)[0]
    d_free = len(free_dims)

    # Sample from prior (GNN-informed or uniform)
    particles = np.zeros((n_particles, d), dtype=np.float32)
    for i in range(d):
        lo, hi = prior_bounds[i]
        if abs(hi - lo) < 1e-12:
            particles[:, i] = lo
        else:
            particles[:, i] = rng.uniform(lo, hi, n_particles)

    has_gnn_prior = log_prior_fn is not None
    if has_gnn_prior:
        # Build composite target: log_target = log_prior + beta * log_L
        log_prior_jit = jax.jit(log_prior_fn)
        # Resample initial particles from GNN prior (within bounds)
        # Rejection sampling: accept if log_prior > threshold
        _lp = np.array([float(log_prior_jit(jnp.array(p))) for p in particles])
        # Keep top 70% by prior density, resample rest
        thresh = np.percentile(_lp, 30)
        for idx in range(n_particles):
            if _lp[idx] < thresh:
                # Resample until above threshold (max 20 tries)
                for _ in range(20):
                    cand = particles[idx].copy()
                    for dim_i in free_dims:
                        lo, hi = prior_bounds[dim_i]
                        cand[dim_i] = rng.uniform(lo, hi)
                    lp_cand = float(log_prior_jit(jnp.array(cand)))
                    if lp_cand >= thresh:
                        particles[idx] = cand
                        break

    if verbose:
        prior_label = " + GNN prior" if has_gnn_prior else ""
        print(
            f"\n{label}: {n_particles} particles, {d_free} free dims, "
            f"mutation={mutation}{prior_label}"
        )
    t0 = time.time()

    logL_jit = jax.jit(log_likelihood)
    # For gradient-based methods, target = log_prior + beta * log_L
    if has_gnn_prior:

        def _composite_logL_and_grad(theta, beta_val):
            lp = log_prior_fn(theta)
            ll, gl = jax.value_and_grad(log_likelihood)(theta)
            _, gp = jax.value_and_grad(log_prior_fn)(theta)
            return lp + beta_val * ll, gp + beta_val * gl

        # Pre-JIT the prior gradient
        _prior_vg_jit = jax.jit(jax.value_and_grad(log_prior_fn))
    grad_jit = jax.jit(jax.value_and_grad(log_likelihood))

    logL = np.array([float(logL_jit(jnp.array(p))) for p in particles])
    t_init = time.time() - t0
    if verbose:
        print(f"  Init: {t_init:.1f}s ({t_init/n_particles*1000:.1f} ms/particle)")

    # NUTS dual averaging state
    param_scales = np.array(prior_bounds[:, 1] - prior_bounds[:, 0])
    param_scales = np.where(param_scales < 1e-12, 1.0, param_scales)
    init_eps = hmc_step_size * np.mean(param_scales[free_mask])
    da_state = dual_averaging_init(step_size_init=init_eps, target_accept=0.65)

    beta = 0.0
    betas = [0.0]
    stage_times = []
    accept_rates = []
    ess_history = []
    n_leapfrog_history = []
    eps_history = []
    stage = 0

    while beta < 1.0 and stage < max_stages:
        stage += 1
        t_stage = time.time()

        # --- Beta schedule via bisection ---
        def compute_ess(db):
            w = db * logL
            w = w - w.max()
            w = np.exp(w)
            return (np.sum(w) ** 2) / np.sum(w**2)

        db_lo_b, db_hi_b = 0.0, 1.0 - beta
        for _ in range(50):
            db_mid = (db_lo_b + db_hi_b) / 2
            if compute_ess(db_mid) > target_ess_ratio * n_particles:
                db_lo_b = db_mid
            else:
                db_hi_b = db_mid

        delta_beta = db_lo_b
        if delta_beta < 1e-6:
            delta_beta = 1.0 - beta
        beta_new = min(beta + delta_beta, 1.0)

        # --- Importance weights + resample ---
        w = (beta_new - beta) * logL
        w = w - w.max()
        w = np.exp(w)
        ess_val = (np.sum(w) ** 2) / np.sum(w**2)
        w = w / w.sum()

        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles = particles[idx].copy()
        logL = logL[idx].copy()

        # --- Mutation ---
        # Target at stage j: log_prior(θ) + β_j * log_L(θ)
        # For gradient methods, we need value_and_grad of the full target.
        if has_gnn_prior:
            _beta_new = jnp.float32(beta_new)

            @jax.jit
            def tempered_vg(theta):
                lp, gp = _prior_vg_jit(theta)
                ll, gl = grad_jit(theta)
                return lp + _beta_new * ll, gp + _beta_new * gl

        else:

            def tempered_vg(theta):
                val, grad = grad_jit(theta)
                return beta_new * val, beta_new * grad

        n_accept = 0
        n_leapfrog_stage = 0
        key = jr.PRNGKey(seed + stage * 1000)

        current_eps = (
            np.exp(da_state["log_eps"])
            if mutation == "nuts"
            else hmc_step_size * np.mean(param_scales[free_mask])
        )

        if mutation == "rw":
            # Random-Walk Metropolis
            cov = np.cov(particles[:, free_dims].T)
            if d_free == 1:
                cov = np.atleast_2d(cov)
            cov *= 0.04

            for i in range(n_particles):
                proposal = particles[i].copy()
                proposal[free_dims] += rng.multivariate_normal(np.zeros(d_free), cov)
                in_bounds = all(
                    prior_bounds[dim, 0] <= proposal[dim] <= prior_bounds[dim, 1]
                    for dim in free_dims
                )
                if in_bounds:
                    logL_new = float(logL_jit(jnp.array(proposal)))
                    if has_gnn_prior:
                        lp_old = float(log_prior_jit(jnp.array(particles[i])))
                        lp_new = float(log_prior_jit(jnp.array(proposal)))
                        log_alpha = (lp_new - lp_old) + beta_new * (logL_new - logL[i])
                    else:
                        log_alpha = beta_new * (logL_new - logL[i])
                    if np.log(rng.random()) < log_alpha:
                        particles[i] = proposal
                        logL[i] = logL_new
                        n_accept += 1

        elif mutation == "hmc":
            for i in range(n_particles):
                k_i = jr.fold_in(key, i)
                theta_i = jnp.array(particles[i])
                new_theta, accepted, new_logp = hmc_step(
                    k_i,
                    theta_i,
                    tempered_vg,
                    current_eps,
                    hmc_n_leapfrog,
                    bounds_lo,
                    bounds_hi,
                )
                n_leapfrog_stage += hmc_n_leapfrog
                if bool(accepted):
                    particles[i] = np.array(new_theta)
                    # Extract log_likelihood only (remove prior) for storage
                    if has_gnn_prior:
                        lp = float(log_prior_jit(new_theta))
                        logL[i] = (float(new_logp) - lp) / beta_new
                    else:
                        logL[i] = float(new_logp) / beta_new
                    n_accept += 1

        elif mutation == "nuts":
            accept_probs = []
            for i in range(n_particles):
                k_i = jr.fold_in(key, i)
                theta_i = jnp.array(particles[i])
                new_theta, accepted, new_logp, n_lf = nuts_step(
                    k_i,
                    theta_i,
                    tempered_vg,
                    current_eps,
                    bounds_lo,
                    bounds_hi,
                    max_depth=nuts_max_depth,
                )
                n_leapfrog_stage += n_lf
                ap = 1.0 if accepted else 0.0
                accept_probs.append(ap)
                if accepted:
                    particles[i] = np.array(new_theta)
                    if has_gnn_prior:
                        lp = float(log_prior_jit(new_theta))
                        logL[i] = (float(new_logp) - lp) / beta_new
                    else:
                        logL[i] = float(new_logp) / beta_new
                    n_accept += 1

            # Dual averaging during warmup stages
            if stage <= warmup_stages:
                avg_ap = np.mean(accept_probs)
                da_state = dual_averaging_update(da_state, avg_ap)

        beta = beta_new
        betas.append(beta)
        dt = time.time() - t_stage
        stage_times.append(dt)
        ar = n_accept / n_particles
        accept_rates.append(ar)
        ess_history.append(ess_val)
        n_leapfrog_history.append(n_leapfrog_stage)
        eps_history.append(current_eps)

        if verbose:
            extra = ""
            if mutation == "nuts":
                avg_lf = n_leapfrog_stage / n_particles
                extra = f", eps={current_eps:.4f}, avg_lf={avg_lf:.1f}"
            print(
                f"  Stage {stage:2d}: beta={beta:.4f}, "
                f"accept={ar:.2f}, ESS={ess_val:.0f}, "
                f"logL=[{logL.min():.1f}, {logL.max():.1f}], "
                f"{dt:.1f}s{extra}"
            )

    total_time = time.time() - t0
    best_idx = np.argmax(logL)

    # Finalize NUTS step size
    if mutation == "nuts":
        final_eps = np.exp(da_state["log_eps_bar"])
    else:
        final_eps = current_eps

    return {
        "label": label,
        "mutation": mutation,
        "samples": particles,
        "log_likelihoods": logL,
        "betas": np.array(betas),
        "theta_MAP": particles[best_idx],
        "total_time": total_time,
        "stage_times": stage_times,
        "accept_rates": accept_rates,
        "ess_history": ess_history,
        "n_stages": len(betas) - 1,
        "n_leapfrog_history": n_leapfrog_history,
        "eps_history": eps_history,
        "final_eps": final_eps,
    }


# ============================================================
# Paper Figure: 3-Method Comparison
# ============================================================
def plot_3method_comparison(results, condition, free_dims, save_dir=None):
    """
    Paper-quality figure comparing RW / HMC / NUTS TMCMC.
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"TMCMC Mutation Kernel Comparison — {condition}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(
        3, 3, hspace=0.4, wspace=0.35, left=0.07, right=0.96, top=0.92, bottom=0.06
    )

    colors = {"RW-TMCMC": "#F44336", "HMC-TMCMC": "#2196F3", "NUTS-TMCMC": "#4CAF50"}

    # (0,0) Beta schedule
    ax = fig.add_subplot(gs[0, 0])
    for r in results:
        ax.plot(r["betas"], "o-", color=colors[r["label"]], label=r["label"], ms=3)
    ax.set_xlabel("Stage")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("Tempering Schedule")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Acceptance rates
    ax = fig.add_subplot(gs[0, 1])
    for r in results:
        avg_ar = np.mean(r["accept_rates"])
        ax.plot(
            r["accept_rates"],
            "o-",
            color=colors[r["label"]],
            label=f"{r['label']} ({avg_ar:.2f})",
            ms=3,
        )
    ax.set_xlabel("Stage")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Mutation Acceptance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (0,2) Summary bar chart
    ax = fig.add_subplot(gs[0, 2])
    metrics = ["Stages", "Time [s]", "Avg Accept"]
    x = np.arange(len(metrics))
    w = 0.25
    for i, r in enumerate(results):
        vals = [r["n_stages"], r["total_time"], np.mean(r["accept_rates"])]
        offset = (i - len(results) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=colors[r["label"]], alpha=0.8, label=r["label"])
        for j, v in enumerate(vals):
            ax.text(
                x[j] + offset,
                v + 0.01 * max(vals),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=8)
    ax.set_title("Performance Summary")

    # (1,0) ESS history
    ax = fig.add_subplot(gs[1, 0])
    for r in results:
        ax.plot(
            range(1, r["n_stages"] + 1),
            r["ess_history"],
            "o-",
            color=colors[r["label"]],
            label=r["label"],
            ms=3,
        )
    ax.set_xlabel("Stage")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Leapfrog steps (HMC/NUTS only)
    ax = fig.add_subplot(gs[1, 1])
    for r in results:
        if r["mutation"] in ("hmc", "nuts") and r["n_leapfrog_history"]:
            avg_lf = [nl / 200 for nl in r["n_leapfrog_history"]]  # approx per-particle
            ax.plot(
                range(1, len(avg_lf) + 1),
                avg_lf,
                "o-",
                color=colors[r["label"]],
                label=r["label"],
                ms=3,
            )
    ax.set_xlabel("Stage")
    ax.set_ylabel("Avg Leapfrog / Particle")
    ax.set_title("Trajectory Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Step size adaptation (NUTS)
    ax = fig.add_subplot(gs[1, 2])
    for r in results:
        if r["eps_history"]:
            ax.plot(
                range(1, len(r["eps_history"]) + 1),
                r["eps_history"],
                "o-",
                color=colors[r["label"]],
                label=r["label"],
                ms=3,
            )
    ax.set_xlabel("Stage")
    ax.set_ylabel(r"$\epsilon$ (step size)")
    ax.set_title("Step Size Adaptation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,:) Posterior marginals — top 6 free dims
    ax = fig.add_subplot(gs[2, :])
    show_dims = free_dims[:6]
    n_show = len(show_dims)
    sub_gs = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[2, :], wspace=0.3)
    for j, dim in enumerate(show_dims):
        ax_sub = fig.add_subplot(sub_gs[0, j])
        for r in results:
            samples = r["samples"][:, dim]
            ax_sub.hist(
                samples,
                bins=25,
                alpha=0.4,
                color=colors[r["label"]],
                density=True,
                label=r["label"] if j == 0 else None,
            )
        ax_sub.set_title(f"θ[{dim}]", fontsize=9)
        ax_sub.tick_params(labelsize=7)
        if j == 0:
            ax_sub.legend(fontsize=6)

    out_path = str(Path(save_dir) / f"nuts_comparison_{condition}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ============================================================
# Paper Figure: 4-Condition Real Data Summary
# ============================================================
def plot_4condition_summary(all_results, save_dir=None):
    """
    Paper-quality figure: NUTS-TMCMC results across 4 conditions.
    """
    if save_dir is None:
        save_dir = SCRIPT_DIR

    conditions = list(all_results.keys())
    n_cond = len(conditions)

    fig, axes = plt.subplots(2, n_cond, figsize=(5 * n_cond, 10))
    fig.suptitle(
        "NUTS-TMCMC on Real Experimental Data (4 Conditions)", fontsize=14, fontweight="bold"
    )

    cond_colors = {
        "Commensal_Static": "#4CAF50",
        "Commensal_HOBIC": "#2196F3",
        "Dysbiotic_Static": "#F44336",
        "Dysbiotic_HOBIC": "#FF9800",
    }

    for col, cond in enumerate(conditions):
        r = all_results[cond]
        color = cond_colors.get(cond, "gray")

        # Row 0: Acceptance + beta
        ax = axes[0, col] if n_cond > 1 else axes[0]
        ax.plot(r["accept_rates"], "o-", color=color, ms=4)
        ax.set_ylim(0, 1.05)
        avg_ar = np.mean(r["accept_rates"])
        ax.set_title(f"{cond}\naccept={avg_ar:.2f}, stages={r['n_stages']}", fontsize=10)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Acceptance Rate" if col == 0 else "")
        ax.grid(True, alpha=0.3)

        # Row 1: Posterior logL distribution
        ax = axes[1, col] if n_cond > 1 else axes[1]
        ax.hist(r["log_likelihoods"], bins=30, color=color, alpha=0.7)
        ax.axvline(
            r["log_likelihoods"].max(),
            color="k",
            ls="--",
            lw=1,
            label=f"max={r['log_likelihoods'].max():.1f}",
        )
        ax.set_xlabel("log L")
        ax.set_ylabel("Count" if col == 0 else "")
        ax.set_title(f"Posterior logL\nε={r['final_eps']:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = str(Path(save_dir) / "nuts_4condition_real.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ============================================================
# Load Prior Bounds
# ============================================================
def load_prior_bounds(condition):
    """Load prior bounds from config."""
    bounds_file = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(bounds_file) as f:
        cfg = json.load(f)

    strategy = cfg["strategies"].get(condition, {})
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

    return prior_bounds


# ============================================================
# GNN Prior — Informative Gaussian prior from InteractionGNN
# ============================================================
GNN_DIR = PROJECT_ROOT / "gnn"
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]  # θ indices for 5 active edges
GNN_PREDICTIONS_FILE = GNN_DIR / "data" / "gnn_prior_predictions.json"


def load_gnn_predictions(path=None):
    """
    Load pre-computed GNN predictions from JSON.

    The JSON is generated by gnn/predict_for_tmcmc.py (runs in torch env).
    This avoids needing torch in the JAX environment.

    Returns:
        dict: condition → {"a_ij_pred": [...], "a_ij_free": [...], ...}
    """
    fpath = Path(path) if path else GNN_PREDICTIONS_FILE
    if not fpath.exists():
        raise FileNotFoundError(
            f"GNN predictions not found at {fpath}\n" f"Run: cd gnn && python predict_for_tmcmc.py"
        )
    with open(fpath) as f:
        return json.load(f)


def get_gnn_prior_for_condition(gnn_preds, condition):
    """Get a_ij predictions for a specific condition."""
    if condition not in gnn_preds:
        raise KeyError(f"No GNN prediction for {condition}")
    entry = gnn_preds[condition]
    return np.array(entry["a_ij_pred"], dtype=np.float32)


def make_gnn_log_prior(gnn_predictions, sigma_prior=1.0, prior_bounds=None):
    """
    Create a JAX-compatible Gaussian log-prior for active edges.

    log p(θ) = -0.5 * Σ_k ((θ[idx_k] - μ_k) / σ)²

    Args:
        gnn_predictions: (5,) GNN-predicted a_ij values
        sigma_prior: prior width (default 1.0)
        prior_bounds: (20, 2) bounds for clipping

    Returns:
        JAX function: theta → log_prior(theta)
    """
    mu = jnp.array(gnn_predictions, dtype=jnp.float32)
    sigma = jnp.float32(sigma_prior)
    idx = jnp.array(ACTIVE_THETA_IDX, dtype=jnp.int32)

    def log_prior(theta):
        theta_active = theta[idx]
        return -0.5 * jnp.sum(((theta_active - mu) / sigma) ** 2)

    return log_prior


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="NUTS-TMCMC: gradient-based TMCMC with NUTS/HMC/RW"
    )
    parser.add_argument("--condition", default="Dysbiotic_HOBIC")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--real", action="store_true", help="Use real experimental data (not synthetic)"
    )
    parser.add_argument("--all-conditions", action="store_true", help="Run all 4 conditions")
    parser.add_argument("--compare-all", action="store_true", help="Compare RW vs HMC vs NUTS")
    parser.add_argument(
        "--mutation",
        default="nuts",
        choices=["rw", "hmc", "nuts"],
        help="Mutation kernel (default: nuts)",
    )
    parser.add_argument("--hmc-step-size", type=float, default=0.005)
    parser.add_argument("--hmc-n-leapfrog", type=int, default=5)
    parser.add_argument("--nuts-max-depth", type=int, default=6)
    parser.add_argument("--paper-fig", action="store_true", help="Generate paper-quality figures")
    parser.add_argument(
        "--gnn-prior",
        action="store_true",
        help="Use GNN-predicted a_ij as informative Gaussian prior",
    )
    parser.add_argument(
        "--gnn-sigma", type=float, default=1.0, help="Width of GNN Gaussian prior (default: 1.0)"
    )
    parser.add_argument(
        "--gnn-predictions",
        type=str,
        default=None,
        help="Path to GNN predictions JSON (default: gnn/data/gnn_prior_predictions.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("NUTS-TMCMC: Gradient-Based TMCMC")
    if args.gnn_prior:
        print(f"  + GNN Prior (sigma={args.gnn_sigma})")
    print("=" * 70)

    dem_model = load_dem()

    # Load GNN predictions if requested
    gnn_preds = None
    if args.gnn_prior:
        print("Loading GNN predictions...")
        gnn_preds = load_gnn_predictions(args.gnn_predictions)
        print(f"  Loaded predictions for {list(gnn_preds.keys())}")

    conditions = list(CONDITION_CHECKPOINTS.keys()) if args.all_conditions else [args.condition]

    all_results = {}
    paper_fig_dir = PROJECT_ROOT / "FEM" / "figures" / "paper_final"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond}")
        print(f"{'='*60}")

        don_model, theta_lo, theta_width = load_deeponet(cond)
        prior_bounds = load_prior_bounds(cond)
        free_dims = np.where(np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12)[0]

        # Build GNN prior if requested
        log_prior_fn = None
        if args.gnn_prior and gnn_preds is not None:
            try:
                gnn_pred = get_gnn_prior_for_condition(gnn_preds, cond)
                print(f"  GNN prior: a_ij = {gnn_pred.round(3)} (sigma={args.gnn_sigma})")
                log_prior_fn = make_gnn_log_prior(
                    gnn_pred, sigma_prior=args.gnn_sigma, prior_bounds=prior_bounds
                )
            except Exception as e:
                print(f"  WARNING: GNN prior failed ({e}), falling back to uniform")
                log_prior_fn = None

        # Build log-likelihood
        if args.real:
            print("  [Real data mode]")
            obs_data = load_real_data(cond)
            sigma_obs = load_real_sigma(cond)
            print(f"  Observed data: {obs_data.shape}, sigma={sigma_obs:.4f}")

            log_likelihood = make_real_log_likelihood(
                don_model, theta_lo, theta_width, obs_data, sigma_obs
            )
        else:
            print("  [Synthetic data mode]")
            # Generate synthetic data from random true theta
            rng = np.random.default_rng(args.seed)
            theta_true = np.zeros(20, dtype=np.float32)
            for i in range(20):
                lo, hi = prior_bounds[i]
                if abs(hi - lo) > 1e-12:
                    theta_true[i] = rng.uniform(lo, hi)

            theta_jax = jnp.array(theta_true)
            from e2e_differentiable_pipeline import deeponet_predict_final as dpf

            phi_true = dpf(don_model, theta_jax, theta_lo, theta_width)
            sigma_phi = 0.03
            obs_phi = jnp.array(np.array(phi_true) + rng.normal(0, sigma_phi, 5), dtype=jnp.float32)
            obs_phi = jnp.clip(obs_phi, 0.0, 1.0)

            def log_likelihood(theta):
                phi = deeponet_predict_final(don_model, theta, theta_lo, theta_width)
                residual = obs_phi - phi
                return -0.5 * jnp.sum((residual / sigma_phi) ** 2)

        # JIT warmup
        print("  JIT warmup...")
        _ = jax.jit(log_likelihood)(jnp.zeros(20, dtype=jnp.float32))
        _ = jax.jit(jax.value_and_grad(log_likelihood))(jnp.zeros(20, dtype=jnp.float32))
        print("  [OK]")

        if args.compare_all:
            # Run all 3 methods
            results = []
            for mut in ["rw", "hmc", "nuts"]:
                r = tmcmc_engine(
                    log_likelihood,
                    prior_bounds,
                    mutation=mut,
                    n_particles=args.n_particles,
                    seed=args.seed,
                    hmc_step_size=args.hmc_step_size,
                    hmc_n_leapfrog=args.hmc_n_leapfrog,
                    nuts_max_depth=args.nuts_max_depth,
                    log_prior_fn=log_prior_fn,
                )
                results.append(r)

            # Summary
            print(f"\n{'='*70}")
            print(f"COMPARISON: {cond}")
            gnn_tag = " + GNN prior" if log_prior_fn is not None else ""
            print(f"{'='*70}")
            print(f"{'Metric':<25} {'RW':>10} {'HMC':>10} {'NUTS':>10}")
            print("-" * 60)
            for r in results:
                pass
            print(
                f"{'Total time [s]':<25} " + " ".join(f"{r['total_time']:>10.1f}" for r in results)
            )
            print(f"{'Stages':<25} " + " ".join(f"{r['n_stages']:>10d}" for r in results))
            print(
                f"{'Avg acceptance':<25} "
                + " ".join(f"{np.mean(r['accept_rates']):>10.3f}" for r in results)
            )
            print(
                f"{'Final max logL':<25} "
                + " ".join(f"{r['log_likelihoods'].max():>10.1f}" for r in results)
            )

            # Plot
            save_dir = paper_fig_dir if args.paper_fig else SCRIPT_DIR
            plot_3method_comparison(results, cond, free_dims, save_dir)

            all_results[cond] = results[-1]  # NUTS result

        else:
            # Single method
            r = tmcmc_engine(
                log_likelihood,
                prior_bounds,
                mutation=args.mutation,
                n_particles=args.n_particles,
                seed=args.seed,
                hmc_step_size=args.hmc_step_size,
                hmc_n_leapfrog=args.hmc_n_leapfrog,
                nuts_max_depth=args.nuts_max_depth,
                log_prior_fn=log_prior_fn,
            )

            print(
                f"\n  Result: {r['n_stages']} stages, "
                f"accept={np.mean(r['accept_rates']):.2f}, "
                f"max logL={r['log_likelihoods'].max():.1f}, "
                f"time={r['total_time']:.1f}s"
            )

            all_results[cond] = r

    # 4-condition summary figure
    if len(all_results) > 1:
        save_dir = paper_fig_dir if args.paper_fig else SCRIPT_DIR
        plot_4condition_summary(all_results, save_dir)

    # Save results
    out = {}
    for cond, r in all_results.items():
        out[cond] = {
            "mutation": r["mutation"],
            "n_stages": r["n_stages"],
            "total_time": r["total_time"],
            "avg_acceptance": float(np.mean(r["accept_rates"])),
            "max_logL": float(r["log_likelihoods"].max()),
            "theta_MAP": r["theta_MAP"].tolist(),
            "final_eps": float(r.get("final_eps", 0)),
        }

    out_path = SCRIPT_DIR / "nuts_tmcmc_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
