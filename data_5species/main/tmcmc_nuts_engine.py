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
    """Original Python-loop NUTS (for CPU / debugging)."""
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


# ── Pure-JAX NUTS (vmappable, fixed max_depth unroll) ──


def _leapfrog_jax(q, p, grad_fn, step_size, bounds_lo, bounds_hi):
    """Single leapfrog step, pure JAX."""
    _, grad_q = grad_fn(q)
    p = p + 0.5 * step_size * grad_q
    q = q + step_size * p
    q = jnp.clip(q, bounds_lo, bounds_hi)
    logp_new, grad_new = grad_fn(q)
    p = p + 0.5 * step_size * grad_new
    return q, p, logp_new, grad_new


def _build_subtree_jax(
    q, p, direction, depth, grad_fn, step_size, bounds_lo, bounds_hi, H0, log_u, key
):
    """
    Build a NUTS binary tree of given depth using lax.fori_loop.
    Returns (q_new, p_new, q_candidate, logp_candidate, n_valid, diverged, key).
    """
    n_steps = jnp.int32(2**depth)

    # State: (q, p, q_cand, logp_cand, n_valid, diverged, key)
    init_state = (q, p, q, jnp.float64(-1e30), jnp.int32(0), jnp.bool_(False), key)

    def body_fn(i, state):
        q_cur, p_cur, q_cand, logp_cand, n_valid, diverged, key_s = state

        # Leapfrog in direction
        q_new, p_new, logp_new, _ = _leapfrog_jax(
            q_cur, direction * p_cur, grad_fn, step_size, bounds_lo, bounds_hi
        )
        p_new = direction * p_new

        H_new = -logp_new + 0.5 * jnp.sum(p_new**2)
        is_diverged = (H_new - H0) > 1000.0
        is_valid = (-H_new > log_u) & ~is_diverged & ~diverged

        n_valid_new = jnp.where(is_valid, n_valid + 1, n_valid)
        key_s, k_acc = jr.split(key_s)
        # Multinomial selection: accept with prob 1/n_valid_new
        accept = is_valid & (
            jr.uniform(k_acc) < (1.0 / jnp.maximum(n_valid_new, 1).astype(jnp.float32))
        )
        q_cand_new = jnp.where(accept, q_new, q_cand)
        logp_cand_new = jnp.where(accept, logp_new, logp_cand)
        diverged_new = diverged | is_diverged

        return (q_new, p_new, q_cand_new, logp_cand_new, n_valid_new, diverged_new, key_s)

    final = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
    q_end, p_end, q_cand, logp_cand, n_valid, diverged, key_out = final
    return q_end, p_end, q_cand, logp_cand, n_valid, diverged, key_out


def nuts_step_jax(key, theta, log_prob_and_grad, step_size, bounds_lo, bounds_hi, max_depth=6):
    """
    Pure-JAX NUTS step (vmappable). Fixed max_depth unroll with masking.
    """
    d = theta.shape[0]
    logp0, _ = log_prob_and_grad(theta)
    key, k_mom = jr.split(key)
    p0 = jr.normal(k_mom, (d,), dtype=jnp.float64)
    H0 = _compute_hamiltonian(logp0, p0)
    key, k_slice = jr.split(key)
    log_u = jnp.log(jr.uniform(k_slice)) + logp0 - 0.5 * jnp.sum(p0**2)

    # State: (q_minus, p_minus, q_plus, p_plus, q_propose, logp_propose,
    #         n_valid, keep_going, key, depth)
    init_state = (
        theta,
        p0,  # q_minus, p_minus
        theta,
        p0,  # q_plus, p_plus
        theta,
        logp0,  # q_propose, logp_propose
        jnp.int32(1),  # n_valid
        jnp.bool_(True),  # keep_going
        key,
    )

    def depth_body(depth_i, state):
        q_minus, p_minus, q_plus, p_plus, q_propose, logp_propose, n_valid, keep_going, key_s = (
            state
        )

        key_s, k_dir = jr.split(key_s)
        go_right = jr.bernoulli(k_dir)
        direction = jnp.where(go_right, 1.0, -1.0)

        # Select starting point based on direction
        q_inner = jnp.where(go_right, q_plus, q_minus)
        p_inner = jnp.where(go_right, p_plus, p_minus)

        # Build subtree
        q_end, p_end, q_cand, logp_cand, n_valid_sub, diverged, key_s = _build_subtree_jax(
            q_inner,
            p_inner,
            direction,
            depth_i,
            log_prob_and_grad,
            step_size,
            bounds_lo,
            bounds_hi,
            H0,
            log_u,
            key_s,
        )

        # Update tree endpoints
        q_minus_new = jnp.where(go_right, q_minus, q_end)
        p_minus_new = jnp.where(go_right, p_minus, p_end)
        q_plus_new = jnp.where(go_right, q_end, q_plus)
        p_plus_new = jnp.where(go_right, p_end, p_plus)

        # Accept candidate from subtree
        key_s, k_sub = jr.split(key_s)
        n_total = jnp.maximum(n_valid + n_valid_sub, 1)
        accept_sub = (n_valid_sub > 0) & (
            jr.uniform(k_sub) < (n_valid_sub / n_total).astype(jnp.float32)
        )
        q_propose_new = jnp.where(accept_sub & keep_going, q_cand, q_propose)
        logp_propose_new = jnp.where(accept_sub & keep_going, logp_cand, logp_propose)
        n_valid_new = n_valid + n_valid_sub

        # U-turn check
        dq = q_plus_new - q_minus_new
        uturn = (jnp.sum(dq * p_minus_new) < 0) | (jnp.sum(dq * p_plus_new) < 0)
        keep_going_new = keep_going & ~diverged & ~uturn

        return (
            q_minus_new,
            p_minus_new,
            q_plus_new,
            p_plus_new,
            q_propose_new,
            logp_propose_new,
            n_valid_new,
            keep_going_new,
            key_s,
        )

    final_state = jax.lax.fori_loop(0, max_depth, depth_body, init_state)
    q_propose = final_state[4]
    logp_propose = final_state[5]
    accepted = jnp.any(q_propose != theta)
    return q_propose, logp_propose, accepted


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
            # --- Batched RW mutation (vmap) ---
            cov = np.cov(particles[:, free_dims].T)
            cov = np.atleast_2d(cov) if d_free == 1 else cov
            cov = cov * 0.04
            # Generate all proposals at once
            perturbations = rng.multivariate_normal(np.zeros(d_free), cov, size=n_particles)
            proposals = particles.copy()
            proposals[:, free_dims] += perturbations
            # Bounds check (vectorized)
            in_bounds = np.all(
                (proposals[:, free_dims] >= prior_bounds[free_dims, 0])
                & (proposals[:, free_dims] <= prior_bounds[free_dims, 1]),
                axis=1,
            )
            # Batch logL evaluation for valid proposals using vmap
            valid_idx = np.where(in_bounds)[0]
            if len(valid_idx) > 0:
                proposals_jax = jnp.array(proposals[valid_idx])
                logL_proposals = np.array(logL_vmap(proposals_jax))
                log_alpha = beta_new * (logL_proposals - logL[valid_idx])
                u = np.log(rng.random(len(valid_idx)))
                accept_mask = u < log_alpha
                for j, vi in enumerate(valid_idx):
                    if accept_mask[j]:
                        particles[vi] = proposals[vi]
                        logL[vi] = logL_proposals[j]
                        n_accept += 1

        elif mutation == "hmc":
            # --- Batched HMC mutation ---
            # Generate all momenta at once
            keys = jr.split(key, n_particles)
            momenta = jr.normal(keys[0], (n_particles, d))
            particles_jax = jnp.array(particles)

            # Batch logL + grad for all particles
            logp_all, grad_all = jax.vmap(tempered_vg)(particles_jax)

            # Leapfrog integration (batched across particles)
            q_all = particles_jax
            p_all = momenta
            H_current = -logp_all + 0.5 * jnp.sum(p_all**2, axis=1)

            # Half step momentum
            p_all = p_all + 0.5 * current_eps * grad_all

            for _lf in range(hmc_n_leapfrog - 1):
                q_all = q_all + current_eps * p_all
                q_all = jnp.clip(q_all, bounds_lo, bounds_hi)
                _, grad_all = jax.vmap(tempered_vg)(q_all)
                p_all = p_all + current_eps * grad_all

            q_all = q_all + current_eps * p_all
            q_all = jnp.clip(q_all, bounds_lo, bounds_hi)
            logp_proposed, grad_proposed = jax.vmap(tempered_vg)(q_all)
            p_all = p_all + 0.5 * current_eps * grad_proposed
            p_all = -p_all

            H_proposed = -logp_proposed + 0.5 * jnp.sum(p_all**2, axis=1)
            log_alpha = H_current - H_proposed
            u = jnp.log(jr.uniform(jr.split(key, 1)[0], (n_particles,)))
            accepted_mask = u < log_alpha

            n_leapfrog_stage += n_particles * hmc_n_leapfrog
            for i in range(n_particles):
                if bool(accepted_mask[i]):
                    particles[i] = np.array(q_all[i])
                    logL[i] = float(logp_proposed[i]) / beta_new
                    n_accept += 1

        elif mutation == "nuts":
            # --- Fully vmapped NUTS ---
            particles_jax = jnp.array(particles, dtype=jnp.float64)
            keys = jr.split(key, n_particles)

            def _single_nuts(key_i, theta_i):
                return nuts_step_jax(
                    key_i,
                    theta_i,
                    tempered_vg,
                    current_eps,
                    bounds_lo.astype(jnp.float64),
                    bounds_hi.astype(jnp.float64),
                    max_depth=nuts_max_depth,
                )

            nuts_vmap = jax.vmap(_single_nuts)
            new_thetas, new_logps, accepted_arr = nuts_vmap(keys, particles_jax)

            # Update particles
            new_thetas_np = np.array(new_thetas)
            accepted_np = np.array(accepted_arr)
            for i in range(n_particles):
                if accepted_np[i]:
                    particles[i] = new_thetas_np[i]
                    n_accept += 1

            # Batch re-evaluate logL for all particles (vmap)
            particles_jax = jnp.array(particles)
            logL = np.array(logL_vmap(particles_jax))

            if stage <= warmup_stages:
                da_state = dual_averaging_update(
                    da_state, float(jnp.mean(accepted_arr.astype(jnp.float32)))
                )

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
