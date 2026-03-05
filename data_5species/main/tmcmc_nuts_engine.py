# -*- coding: utf-8 -*-
"""
tmcmc_nuts_engine.py — Standalone TMCMC engine with NUTS/HMC/RW.

Extracted from deeponet/gradient_tmcmc_nuts.py for use with JAX ODE.
No DeepONet dependency.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def _leapfrog(q, p, grad_fn, step_size, bounds_lo, bounds_hi):
    _, grad_q = grad_fn(q)
    p = p + 0.5 * step_size * grad_q
    q = q + step_size * p
    q = jnp.clip(q, bounds_lo, bounds_hi)
    logp_new, grad_new = grad_fn(q)
    p = p + 0.5 * step_size * grad_new
    return q, p, logp_new, grad_new


def _compute_hamiltonian(logp, p):
    return -logp + 0.5 * jnp.sum(p**2)


def nuts_step(key, theta, log_prob_and_grad, step_size, bounds_lo, bounds_hi, max_depth=6):
    d = theta.shape[0]
    logp0, grad0 = log_prob_and_grad(theta)
    key, k_mom = jr.split(key)
    p0 = jr.normal(k_mom, (d,))
    H0 = _compute_hamiltonian(logp0, p0)
    key, k_slice = jr.split(key)
    log_u = jnp.log(jr.uniform(k_slice)) + logp0 - 0.5 * jnp.sum(p0**2)

    q_minus, q_plus = theta, theta
    p_minus, p_plus = p0, p0
    q_propose, logp_propose = theta, logp0
    depth, n_valid, keep_going = 0, 1, True
    n_leapfrog_total = 0

    while keep_going and depth < max_depth:
        key, k_dir = jr.split(key)
        direction = 2 * int(jr.bernoulli(k_dir)) - 1
        q_inner, p_inner = (q_minus, p_minus) if direction == -1 else (q_plus, p_plus)

        n_steps = 2**depth
        q_new, p_new = q_inner, p_inner
        q_candidate, logp_candidate = q_inner, logp0
        n_valid_subtree, diverged = 0, False

        for _ in range(n_steps):
            q_new, p_new, logp_new, _ = _leapfrog(
                q_new, float(direction) * p_new, log_prob_and_grad, step_size, bounds_lo, bounds_hi
            )
            p_new = float(direction) * p_new
            n_leapfrog_total += 1
            H_new = _compute_hamiltonian(logp_new, p_new)
            if float(H_new - H0) > 1000:
                diverged = True
                break
            if float(-H_new) > float(log_u):
                n_valid_subtree += 1
                key, k_accept = jr.split(key)
                if float(jr.uniform(k_accept)) < 1.0 / max(n_valid_subtree, 1):
                    q_candidate, logp_candidate = q_new, logp_new

        if diverged:
            break
        if direction == -1:
            q_minus, p_minus = q_new, p_new
        else:
            q_plus, p_plus = q_new, p_new
        if n_valid_subtree > 0:
            key, k_sub = jr.split(key)
            accept_prob = n_valid_subtree / max(n_valid + n_valid_subtree, 1)
            if float(jr.uniform(k_sub)) < accept_prob:
                q_propose, logp_propose = q_candidate, logp_candidate
        n_valid += n_valid_subtree
        dq = q_plus - q_minus
        uturn = (float(jnp.sum(dq * p_minus)) < 0) or (float(jnp.sum(dq * p_plus)) < 0)
        keep_going = not uturn and not diverged
        depth += 1

    accepted = not jnp.array_equal(q_propose, theta)
    return q_propose, accepted, float(logp_propose), n_leapfrog_total


def dual_averaging_init(
    step_size_init=1.0, target_accept=0.65, gamma=0.05, t0=10, kappa=0.75, eps_max_factor=5.0
):
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
    m = state["m"] + 1
    eta = 1.0 / (m + state["t0"])
    H_bar = (1 - eta) * state["H_bar"] + eta * (state["target_accept"] - accept_prob)
    log_eps = state["mu"] - np.sqrt(m) / state["gamma"] * H_bar
    log_eps = min(log_eps, np.log(state["eps_max"]))
    m_kappa = m ** (-state["kappa"])
    log_eps_bar = m_kappa * log_eps + (1 - m_kappa) * state["log_eps_bar"]
    log_eps_bar = min(log_eps_bar, np.log(state["eps_max"]))
    return {**state, "log_eps": log_eps, "log_eps_bar": log_eps_bar, "H_bar": H_bar, "m": m}


def hmc_step(key, theta, log_prob_and_grad, step_size, n_leapfrog, bounds_lo, bounds_hi):
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


def tmcmc_engine(
    log_likelihood: Callable,
    prior_bounds: np.ndarray,
    mutation: str = "nuts",
    n_particles: int = 200,
    max_stages: int = 30,
    target_ess_ratio: float = 0.5,
    hmc_step_size: float = 0.01,
    hmc_n_leapfrog: int = 10,
    nuts_max_depth: int = 6,
    warmup_stages: int = 3,
    seed: int = 42,
    label: Optional[str] = None,
    verbose: bool = True,
    log_prior_fn: Optional[Callable] = None,
) -> dict:
    """TMCMC with RW / HMC / NUTS mutation. No DeepONet dependency."""
    if label is None:
        label = f"{mutation.upper()}-TMCMC"

    rng = np.random.default_rng(seed)
    d = prior_bounds.shape[0]
    bounds_lo = jnp.array(prior_bounds[:, 0], dtype=jnp.float32)
    bounds_hi = jnp.array(prior_bounds[:, 1], dtype=jnp.float32)
    free_mask = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
    free_dims = np.where(free_mask)[0]
    d_free = len(free_dims)

    particles = np.zeros((n_particles, d), dtype=np.float32)
    for i in range(d):
        lo, hi = prior_bounds[i]
        particles[:, i] = lo if abs(hi - lo) < 1e-12 else rng.uniform(lo, hi, n_particles)

    has_gnn_prior = log_prior_fn is not None
    if has_gnn_prior:
        log_prior_jit = jax.jit(log_prior_fn)
        _lp = np.array([float(log_prior_jit(jnp.array(p))) for p in particles])
        thresh = np.percentile(_lp, 30)
        for idx in range(n_particles):
            if _lp[idx] < thresh:
                for _ in range(20):
                    cand = particles[idx].copy()
                    for dim_i in free_dims:
                        lo, hi = prior_bounds[dim_i]
                        cand[dim_i] = rng.uniform(lo, hi)
                    if float(log_prior_jit(jnp.array(cand))) >= thresh:
                        particles[idx] = cand
                        break

    t0 = time.time()
    logL_jit = jax.jit(log_likelihood)
    grad_jit = jax.jit(jax.value_and_grad(log_likelihood))
    logL_vmap = jax.jit(jax.vmap(log_likelihood))

    particles_jax = jnp.array(particles)
    logL = np.array(logL_vmap(particles_jax))
    if verbose:
        print(f"  Init: {time.time()-t0:.1f}s")

    param_scales = np.array(prior_bounds[:, 1] - prior_bounds[:, 0])
    param_scales = np.where(param_scales < 1e-12, 1.0, param_scales)
    init_eps = hmc_step_size * np.mean(param_scales[free_mask])
    da_state = dual_averaging_init(step_size_init=init_eps, target_accept=0.65)

    beta, betas = 0.0, [0.0]
    stage_times, accept_rates, ess_history = [], [], []
    n_leapfrog_history, eps_history = [], []
    stage = 0

    while beta < 1.0 and stage < max_stages:
        stage += 1
        t_stage = time.time()

        def compute_ess(db):
            w = np.exp(np.clip(db * logL - logL.max(), -500, 500))
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
        delta_beta = min(delta_beta, 1.0 - beta)
        beta_new = min(beta + delta_beta, 1.0)

        w = np.exp(np.clip((beta_new - beta) * logL - np.max((beta_new - beta) * logL), -500, 500))
        ess_val = (np.sum(w) ** 2) / np.sum(w**2)
        w = w / w.sum()
        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles, logL = particles[idx].copy(), logL[idx].copy()

        def tempered_vg(theta):
            val, grad = grad_jit(theta)
            return beta_new * val, beta_new * grad

        n_accept, n_leapfrog_stage = 0, 0
        key = jr.PRNGKey(seed + stage * 1000)
        current_eps = (
            np.exp(da_state["log_eps"])
            if mutation == "nuts"
            else hmc_step_size * np.mean(param_scales[free_mask])
        )

        if mutation == "rw":
            cov = np.cov(particles[:, free_dims].T)
            cov = np.atleast_2d(cov) if d_free == 1 else cov
            cov = cov * 0.04
            for i in range(n_particles):
                proposal = particles[i].copy()
                proposal[free_dims] += rng.multivariate_normal(np.zeros(d_free), cov)
                if all(
                    prior_bounds[dim, 0] <= proposal[dim] <= prior_bounds[dim, 1]
                    for dim in free_dims
                ):
                    logL_new = float(logL_jit(jnp.array(proposal)))
                    log_alpha = beta_new * (logL_new - logL[i])
                    if np.log(rng.random()) < log_alpha:
                        particles[i], logL[i], n_accept = proposal, logL_new, n_accept + 1

        elif mutation == "hmc":
            for i in range(n_particles):
                k_i = jr.fold_in(key, i)
                new_theta, accepted, new_logp = hmc_step(
                    k_i,
                    jnp.array(particles[i]),
                    tempered_vg,
                    current_eps,
                    hmc_n_leapfrog,
                    bounds_lo,
                    bounds_hi,
                )
                n_leapfrog_stage += hmc_n_leapfrog
                if bool(accepted):
                    particles[i] = np.array(new_theta)
                    logL[i] = float(new_logp) / beta_new
                    n_accept += 1

        elif mutation == "nuts":
            accept_probs = []
            for i in range(n_particles):
                k_i = jr.fold_in(key, i)
                new_theta, accepted, new_logp, n_lf = nuts_step(
                    k_i,
                    jnp.array(particles[i]),
                    tempered_vg,
                    current_eps,
                    bounds_lo,
                    bounds_hi,
                    max_depth=nuts_max_depth,
                )
                n_leapfrog_stage += n_lf
                accept_probs.append(1.0 if accepted else 0.0)
                if accepted:
                    particles[i] = np.array(new_theta)
                    logL[i] = float(new_logp) / beta_new
                    n_accept += 1
            if stage <= warmup_stages:
                da_state = dual_averaging_update(da_state, np.mean(accept_probs))

        beta = beta_new
        betas.append(beta)
        stage_times.append(time.time() - t_stage)
        accept_rates.append(n_accept / n_particles)
        ess_history.append(ess_val)
        n_leapfrog_history.append(n_leapfrog_stage)
        eps_history.append(current_eps)

        if verbose:
            extra = f", eps={current_eps:.4f}" if mutation == "nuts" else ""
            print(
                f"  Stage {stage:2d}: beta={beta:.4f}, accept={n_accept/n_particles:.2f}, "
                f"ESS={ess_val:.0f}, logL=[{logL.min():.1f},{logL.max():.1f}], "
                f"{stage_times[-1]:.1f}s{extra}"
            )

    best_idx = np.argmax(logL)
    final_eps = np.exp(da_state["log_eps"]) if mutation == "nuts" else current_eps

    return {
        "label": label,
        "mutation": mutation,
        "samples": particles,
        "log_likelihoods": logL,
        "betas": np.array(betas),
        "theta_MAP": particles[best_idx],
        "total_time": time.time() - t0,
        "stage_times": stage_times,
        "accept_rates": accept_rates,
        "ess_history": ess_history,
        "n_stages": len(betas) - 1,
        "n_leapfrog_history": n_leapfrog_history,
        "eps_history": eps_history,
        "final_eps": final_eps,
    }
