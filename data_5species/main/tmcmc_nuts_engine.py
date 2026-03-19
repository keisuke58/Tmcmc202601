# -*- coding: utf-8 -*-
"""
tmcmc_nuts_engine.py — Standalone TMCMC engine with NUTS/HMC/RW.

Extracted from deeponet/gradient_tmcmc_nuts.py for use with JAX ODE.
No DeepONet dependency.

References:
  - Ching & Chen 2007: TMCMC with evidence estimation
  - Betz et al. 2016: Weighted covariance for RW proposals
  - ter Braak 2006: Differential Evolution MCMC
  - Kitagawa 1996: Systematic resampling
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
    q, p, direction, n_steps, grad_fn, step_size, bounds_lo, bounds_hi, H0, log_u, key
):
    """
    Build a NUTS binary tree with STATIC n_steps using lax.fori_loop.
    n_steps must be a Python int (not jnp traced value) for fast JIT.
    Returns (q_new, p_new, q_candidate, logp_candidate, n_valid, diverged, key).
    """
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
    Pure-JAX NUTS step. Depth loop is Python-unrolled so each _build_subtree_jax
    gets a static n_steps (1, 2, 4, 8, ...), enabling fast JIT compilation.
    """
    d = theta.shape[0]
    logp0, _ = log_prob_and_grad(theta)
    key, k_mom = jr.split(key)
    p0 = jr.normal(k_mom, (d,), dtype=jnp.float64)
    H0 = _compute_hamiltonian(logp0, p0)
    key, k_slice = jr.split(key)
    log_u = jnp.log(jr.uniform(k_slice)) + logp0 - 0.5 * jnp.sum(p0**2)

    # Initialize tree state
    q_minus, p_minus = theta, p0
    q_plus, p_plus = theta, p0
    q_propose, logp_propose = theta, logp0
    n_valid = jnp.int32(1)
    keep_going = jnp.bool_(True)

    # Python-unrolled depth loop (static n_steps per depth)
    for depth_i in range(max_depth):
        n_steps_depth = 2**depth_i  # Python int, static!

        key, k_dir = jr.split(key)
        go_right = jr.bernoulli(k_dir)
        direction = jnp.where(go_right, 1.0, -1.0)

        q_inner = jnp.where(go_right, q_plus, q_minus)
        p_inner = jnp.where(go_right, p_plus, p_minus)

        q_end, p_end, q_cand, logp_cand, n_valid_sub, diverged, key = _build_subtree_jax(
            q_inner,
            p_inner,
            direction,
            n_steps_depth,
            log_prob_and_grad,
            step_size,
            bounds_lo,
            bounds_hi,
            H0,
            log_u,
            key,
        )

        # Update tree endpoints
        q_minus = jnp.where(go_right, q_minus, q_end)
        p_minus = jnp.where(go_right, p_minus, p_end)
        q_plus = jnp.where(go_right, q_end, q_plus)
        p_plus = jnp.where(go_right, p_end, p_plus)

        # Accept candidate from subtree
        key, k_sub = jr.split(key)
        n_total = jnp.maximum(n_valid + n_valid_sub, 1)
        accept_sub = (n_valid_sub > 0) & (
            jr.uniform(k_sub) < (n_valid_sub / n_total).astype(jnp.float32)
        )
        q_propose = jnp.where(accept_sub & keep_going, q_cand, q_propose)
        logp_propose = jnp.where(accept_sub & keep_going, logp_cand, logp_propose)
        n_valid = n_valid + n_valid_sub

        # U-turn check
        dq = q_plus - q_minus
        uturn = (jnp.sum(dq * p_minus) < 0) | (jnp.sum(dq * p_plus) < 0)
        keep_going = keep_going & ~diverged & ~uturn

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


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling — lower variance than multinomial (Kitagawa 1996)."""
    n = len(weights)
    cumw = np.cumsum(weights)
    cumw /= cumw[-1]  # normalize
    u0 = rng.random() / n
    positions = u0 + np.arange(n) / n
    indices = np.searchsorted(cumw, positions)
    return np.clip(indices, 0, n - 1)


def _de_mc_proposal(
    particles: np.ndarray,
    free_dims: np.ndarray,
    rng: np.random.Generator,
    gamma: float = 2.38,
    b_small: float = 1e-4,
) -> np.ndarray:
    """
    Differential Evolution MCMC proposal (ter Braak 2006).

    For each particle i, pick two distinct others r1, r2 and propose:
        theta_new = theta_i + gamma/sqrt(2*d) * (theta_r1 - theta_r2) + b*N(0,1)

    Much more efficient than independent RW for correlated posteriors.
    """
    n, d_full = particles.shape
    d_free = len(free_dims)
    proposals = particles.copy()

    # Scale factor
    scale = gamma / np.sqrt(2 * max(d_free, 1))

    # Pick pairs r1, r2 for each particle — Fisher-Yates style (correct)
    r1 = np.empty(n, dtype=int)
    r2 = np.empty(n, dtype=int)
    for i in range(n):
        # Draw r1 != i
        candidates = np.arange(n)
        candidates = candidates[candidates != i]
        r1[i] = rng.choice(candidates)
        # Draw r2 != i and != r1
        candidates2 = candidates[candidates != r1[i]]
        r2[i] = rng.choice(candidates2)

    diff = particles[r1][:, free_dims] - particles[r2][:, free_dims]
    noise = rng.normal(0, b_small, size=(n, d_free))
    proposals[:, free_dims] += scale * diff + noise

    return proposals


def _weighted_covariance(
    particles: np.ndarray,
    weights: np.ndarray,
    free_dims: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted sample covariance (Betz et al. 2016).

    Uses importance weights from the current stage instead of equal weights.
    This better captures the shape of the tempered posterior.
    """
    x = particles[:, free_dims]  # (n, d_free)
    w = weights / weights.sum()  # normalize
    # Weighted mean
    mean = np.average(x, weights=w, axis=0)
    # Weighted covariance
    dx = x - mean
    cov = (dx * w[:, np.newaxis]).T @ dx
    # Bessel-like correction for effective sample size
    w2 = np.sum(w**2)
    correction = 1.0 / (1.0 - w2 + 1e-12)
    cov *= correction
    # Regularize: add small diagonal to prevent singularity
    cov += np.eye(len(free_dims)) * 1e-8
    return cov


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
    n_mutation_steps: int = 1,
    use_de_mc: bool = False,
    de_mc_gamma: float = 2.38,
    resample_method: str = "systematic",
    # --- Flow-enhanced SMC (Gabrie et al. 2022) ---
    use_flow: bool = False,
    flow_n_layers: int = 6,
    flow_hidden_dim: int = 64,
    flow_n_epochs: int = 200,
    flow_lr: float = 1e-3,
    flow_start_beta: float = 0.1,
    flow_mix_ratio: float = 0.8,
    # --- Waste-free SMC (Dau & Chopin 2022) ---
    waste_free: bool = False,
) -> dict:
    """
    TMCMC with RW / HMC / NUTS mutation.

    Optionally enhanced with:
      - Normalizing Flow proposals (use_flow=True): learns transport map
        from particle distribution, achieving near-100% acceptance.
        Ref: Gabrie et al. 2022, Arbel et al. 2021.
      - Waste-free SMC (waste_free=True): keeps all MCMC proposals
        (not just accepted), effectively doubling ESS for same compute.
        Ref: Dau & Chopin 2022, JRSS-B.

    Returns dict with samples, log_likelihoods, theta_MAP, log_evidence, etc.
    Log evidence is computed via Ching & Chen 2007: sum(log(mean(w_j))) at each stage.
    """
    if label is None:
        label = f"{mutation.upper()}-TMCMC"

    rng = np.random.default_rng(seed)
    d = prior_bounds.shape[0]
    # Use float64 consistently (matching JAX ODE and likelihood)
    bounds_lo = jnp.array(prior_bounds[:, 0], dtype=jnp.float64)
    bounds_hi = jnp.array(prior_bounds[:, 1], dtype=jnp.float64)
    free_mask = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
    free_dims = np.where(free_mask)[0]
    d_free = len(free_dims)

    particles = np.zeros((n_particles, d), dtype=np.float64)
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
    cv_weights_history = []  # Convergence diagnostic
    log_evidence = 0.0  # Ching & Chen 2007: cumulative log evidence
    stage = 0
    _last_resampling_weights = np.ones(n_particles) / n_particles  # for weighted cov

    # --- Flow initialization (lazy: created on first use) ---
    flow_params = None
    if use_flow:
        try:
            from normalizing_flow import init_flow, train_flow, flow_proposal_mh

            flow_key = jr.PRNGKey(seed + 9999)
            flow_params = init_flow(
                flow_key, d_free, n_layers=flow_n_layers, hidden_dim=flow_hidden_dim
            )
            if verbose:
                print(f"  Flow initialized: {flow_n_layers} layers, hidden={flow_hidden_dim}")
        except ImportError:
            if verbose:
                print("  WARNING: normalizing_flow.py not found, disabling flow")
            use_flow = False

    # --- Waste-free SMC setup ---
    if waste_free:
        try:
            from waste_free_smc import waste_free_resample_and_mutate_batch
        except ImportError:
            if verbose:
                print("  WARNING: waste_free_smc.py not found, disabling waste-free")
            waste_free = False

    while beta < 1.0 and stage < max_stages:
        stage += 1
        t_stage = time.time()

        def compute_ess(db):
            lw = db * logL
            lw_shifted = lw - lw.max()
            w = np.exp(np.clip(lw_shifted, -500, 500))
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

        # Importance weights for this stage
        lw_raw = (beta_new - beta) * logL
        lw_shifted = lw_raw - np.max(lw_raw)
        w = np.exp(np.clip(lw_shifted, -500, 500))

        # Log evidence accumulation (Ching & Chen 2007, Eq. 17):
        # log Z = sum_j log(1/N * sum_i w_i^j)
        log_evidence += np.log(np.mean(w) + 1e-300)

        ess_val = (np.sum(w) ** 2) / np.sum(w**2)
        w_normalized = w / w.sum()

        # Convergence diagnostic: CV of weights (lower = better mixing)
        cv_w = np.std(w_normalized) / (np.mean(w_normalized) + 1e-12)
        cv_weights_history.append(cv_w)

        # Save weights for weighted covariance before resampling
        _last_resampling_weights = w_normalized.copy()

        # --- Waste-free SMC path (Dau & Chopin 2022) ---
        if waste_free and mutation == "rw" and n_mutation_steps > 1:

            def _batch_rw_mutation(p_batch, lL_batch, _rng):
                """Single RW mutation step for waste-free SMC."""
                n_b = p_batch.shape[0]
                cov_wf = _weighted_covariance(p_batch, np.ones(n_b) / n_b, free_dims)
                _sc = (2.38**2) / max(d_free, 1)
                cov_wf = cov_wf * _sc
                pert = _rng.multivariate_normal(np.zeros(d_free), cov_wf, size=n_b)
                props = p_batch.copy()
                props[:, free_dims] += pert
                ib = np.all(
                    (props[:, free_dims] >= prior_bounds[free_dims, 0])
                    & (props[:, free_dims] <= prior_bounds[free_dims, 1]),
                    axis=1,
                )
                vi = np.where(ib)[0]
                n_acc = 0
                if len(vi) > 0:
                    pj = jnp.array(props[vi])
                    lL_p = np.array(logL_vmap(pj))
                    la = beta_new * (lL_p - lL_batch[vi])
                    u = np.log(_rng.random(len(vi)))
                    am = u < la
                    ai = vi[am]
                    aj = np.where(am)[0]
                    p_batch[ai] = props[ai]
                    lL_batch[ai] = lL_p[aj]
                    n_acc = int(am.sum())
                return p_batch, lL_batch, n_acc

            particles, logL, _last_resampling_weights, n_accept = (
                waste_free_resample_and_mutate_batch(
                    particles,
                    logL,
                    w_normalized,
                    beta,
                    beta_new,
                    _batch_rw_mutation,
                    n_mutation_steps=n_mutation_steps,
                    rng=rng,
                )
            )
            n_leapfrog_stage = 0
        else:
            # --- Standard resample ---
            if resample_method == "systematic":
                idx = _systematic_resample(w_normalized, rng)
            else:
                idx = rng.choice(n_particles, size=n_particles, p=w_normalized)
            particles, logL = particles[idx].copy(), logL[idx].copy()

        def tempered_vg(theta):
            val, grad = grad_jit(theta)
            return beta_new * val, beta_new * grad

        # Only reset counters if waste-free didn't already handle mutation
        _skip_mutation = waste_free and mutation == "rw" and n_mutation_steps > 1
        if not _skip_mutation:
            n_accept, n_leapfrog_stage = 0, 0
        key = jr.PRNGKey(seed + stage * 1000)
        current_eps = (
            np.exp(da_state["log_eps"])
            if mutation == "nuts"
            else hmc_step_size * np.mean(param_scales[free_mask])
        )

        if mutation == "rw" and not _skip_mutation:
            # --- Batched RW + optional DE-MC mutation (vmap) with adaptive scale ---
            # Adaptive mutation steps: fewer at low beta, more at high beta
            _n_mut = max(3, int(n_mutation_steps * max(0.3, beta_new)))

            # --- Flow-enhanced proposals (Gabrie et al. 2022) ---
            _use_flow_this_stage = (
                use_flow and flow_params is not None and beta_new >= flow_start_beta
            )
            if _use_flow_this_stage:
                # Train flow on current (resampled) particles
                flow_key, key = jr.split(key)
                particles_free = jnp.array(particles[:, free_dims])
                flow_params = train_flow(
                    flow_key,
                    flow_params,
                    particles_free,
                    weights=jnp.array(_last_resampling_weights),
                    n_epochs=flow_n_epochs,
                    lr=flow_lr,
                )
                if verbose:
                    print(f"    Flow trained (beta={beta_new:.3f})")

            # Weighted covariance (Betz et al. 2016): use importance weights
            # from this stage to better capture the tempered posterior shape
            cov_base = _weighted_covariance(particles, _last_resampling_weights, free_dims)
            # Adaptive scale: start at 2.38^2/d (optimal for Gaussian targets)
            _scale = (2.38**2) / max(d_free, 1)
            _adapt_factor = 1.0
            cov = cov_base * _scale * _adapt_factor
            for _mut_step in range(_n_mut):
                # Decide proposal method: flow, DE-MC, or RW
                _use_flow_step = _use_flow_this_stage and rng.random() < flow_mix_ratio
                _use_demc = (
                    not _use_flow_step and use_de_mc and (_mut_step % 2 == 1) and n_particles >= 10
                )

                if _use_flow_step:
                    # --- Flow proposal with MH correction ---
                    flow_key, key = jr.split(key)
                    from normalizing_flow import flow_sample, flow_log_prob

                    # Sample in free-dim space
                    props_free, log_q_props = flow_sample(flow_key, flow_params, n_particles)
                    # Map back to full space
                    proposals = particles.copy()
                    proposals[:, free_dims] = np.array(props_free)
                    # Bounds check
                    in_bounds = np.all(
                        (proposals[:, free_dims] >= prior_bounds[free_dims, 0])
                        & (proposals[:, free_dims] <= prior_bounds[free_dims, 1]),
                        axis=1,
                    )
                    valid_idx = np.where(in_bounds)[0]
                    n_acc_step = 0
                    if len(valid_idx) > 0:
                        proposals_jax = jnp.array(proposals[valid_idx])
                        logL_proposals = np.array(logL_vmap(proposals_jax))
                        # MH ratio for independent proposal:
                        # log alpha = beta * (logL_new - logL_old) + (log_q_old - log_q_new)
                        log_q_old = np.array(
                            jax.vmap(lambda x: flow_log_prob(flow_params, x))(
                                jnp.array(particles[valid_idx][:, free_dims])
                            )
                        )
                        log_q_new = np.array(log_q_props)[valid_idx]
                        log_alpha = beta_new * (logL_proposals - logL[valid_idx]) + (
                            log_q_old - log_q_new
                        )
                        u = np.log(rng.random(len(valid_idx)))
                        accept_mask = u < log_alpha
                        accepted_idx = valid_idx[accept_mask]
                        accepted_j = np.where(accept_mask)[0]
                        particles[accepted_idx] = proposals[accepted_idx]
                        logL[accepted_idx] = logL_proposals[accepted_j]
                        n_acc_step = int(accept_mask.sum())
                        n_accept += n_acc_step
                elif _use_demc:
                    proposals = _de_mc_proposal(particles, free_dims, rng, gamma=de_mc_gamma)
                    in_bounds = np.all(
                        (proposals[:, free_dims] >= prior_bounds[free_dims, 0])
                        & (proposals[:, free_dims] <= prior_bounds[free_dims, 1]),
                        axis=1,
                    )
                    valid_idx = np.where(in_bounds)[0]
                    n_acc_step = 0
                    if len(valid_idx) > 0:
                        proposals_jax = jnp.array(proposals[valid_idx])
                        logL_proposals = np.array(logL_vmap(proposals_jax))
                        log_alpha = beta_new * (logL_proposals - logL[valid_idx])
                        if has_gnn_prior:
                            for j, vi in enumerate(valid_idx):
                                lp_new = float(log_prior_jit(jnp.array(proposals[vi])))
                                lp_old = float(log_prior_jit(jnp.array(particles[vi])))
                                log_alpha[j] += lp_new - lp_old
                        u = np.log(rng.random(len(valid_idx)))
                        accept_mask = u < log_alpha
                        accepted_idx = valid_idx[accept_mask]
                        accepted_j = np.where(accept_mask)[0]
                        particles[accepted_idx] = proposals[accepted_idx]
                        logL[accepted_idx] = logL_proposals[accepted_j]
                        n_acc_step = int(accept_mask.sum())
                        n_accept += n_acc_step
                else:
                    perturbations = rng.multivariate_normal(np.zeros(d_free), cov, size=n_particles)
                    proposals = particles.copy()
                    proposals[:, free_dims] += perturbations
                    in_bounds = np.all(
                        (proposals[:, free_dims] >= prior_bounds[free_dims, 0])
                        & (proposals[:, free_dims] <= prior_bounds[free_dims, 1]),
                        axis=1,
                    )
                    valid_idx = np.where(in_bounds)[0]
                    n_acc_step = 0
                    if len(valid_idx) > 0:
                        proposals_jax = jnp.array(proposals[valid_idx])
                        logL_proposals = np.array(logL_vmap(proposals_jax))
                        log_alpha = beta_new * (logL_proposals - logL[valid_idx])
                        if has_gnn_prior:
                            for j, vi in enumerate(valid_idx):
                                lp_new = float(log_prior_jit(jnp.array(proposals[vi])))
                                lp_old = float(log_prior_jit(jnp.array(particles[vi])))
                                log_alpha[j] += lp_new - lp_old
                        u = np.log(rng.random(len(valid_idx)))
                        accept_mask = u < log_alpha
                        accepted_idx = valid_idx[accept_mask]
                        accepted_j = np.where(accept_mask)[0]
                        particles[accepted_idx] = proposals[accepted_idx]
                        logL[accepted_idx] = logL_proposals[accepted_j]
                        n_acc_step = int(accept_mask.sum())
                        n_accept += n_acc_step
                # Adaptive scale: target ~23% acceptance (Roberts et al. 1997)
                acc_rate = n_acc_step / max(n_particles, 1)
                if not _use_demc and not _use_flow_step:
                    if acc_rate < 0.10:
                        _adapt_factor *= 0.5  # aggressive shrink for very low accept
                    elif acc_rate < 0.18:
                        _adapt_factor *= 0.8
                    elif acc_rate > 0.35:
                        _adapt_factor *= 1.3
                    elif acc_rate > 0.50:
                        _adapt_factor *= 1.5  # aggressive grow for very high accept
                    _adapt_factor = np.clip(_adapt_factor, 0.001, 50.0)
                # Update cov every few steps using current particles
                if _mut_step % 3 == 2 and _n_mut > 3:
                    cov_base = np.cov(particles[:, free_dims].T)
                    cov_base = np.atleast_2d(cov_base) if d_free == 1 else cov_base
                    cov_base += np.eye(d_free) * 1e-8
                cov = cov_base * _scale * _adapt_factor

        elif mutation == "hmc":
            # --- Batched HMC mutation ---
            keys = jr.split(key, n_particles + 1)
            # Fix: each particle gets its own key for momentum
            momenta = jax.vmap(lambda k: jr.normal(k, (d,)))(keys[1:])
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
            u = jnp.log(jr.uniform(keys[0], (n_particles,)))
            accepted_mask = np.array(u < log_alpha)

            n_leapfrog_stage += n_particles * hmc_n_leapfrog
            # Vectorized accept
            q_np = np.array(q_all)
            logp_np = np.array(logp_proposed)
            acc_idx = np.where(accepted_mask)[0]
            particles[acc_idx] = q_np[acc_idx]
            logL[acc_idx] = logp_np[acc_idx] / beta_new
            n_accept = int(accepted_mask.sum())

        elif mutation == "nuts":
            # --- NUTS: JIT per-particle + vmap logL ---
            # Note: vmap(nuts_step_jax) causes extremely long JIT compile
            # due to nested lax.fori_loop with dynamic trip count.
            # Instead: JIT a single nuts_step, call per particle, vmap logL.
            particles_jax = jnp.array(particles, dtype=jnp.float64)
            keys = jr.split(key, n_particles)
            _blo = jnp.array(bounds_lo, dtype=jnp.float64)
            _bhi = jnp.array(bounds_hi, dtype=jnp.float64)

            @jax.jit
            def _jit_nuts(key_i, theta_i):
                return nuts_step_jax(
                    key_i,
                    theta_i,
                    tempered_vg,
                    current_eps,
                    _blo,
                    _bhi,
                    max_depth=nuts_max_depth,
                )

            accepted_list = []
            for i in range(n_particles):
                new_th, new_lp, acc = _jit_nuts(keys[i], particles_jax[i])
                if bool(acc):
                    particles[i] = np.array(new_th)
                    n_accept += 1
                accepted_list.append(bool(acc))

            # Batch re-evaluate logL for all particles (vmap)
            particles_jax = jnp.array(particles)
            logL = np.array(logL_vmap(particles_jax))

            if stage <= warmup_stages:
                da_state = dual_averaging_update(da_state, float(np.mean(accepted_list)))

        beta = beta_new
        betas.append(beta)
        stage_times.append(time.time() - t_stage)
        _n_total_proposals = n_particles * (n_mutation_steps if mutation == "rw" else 1)
        accept_rates.append(n_accept / _n_total_proposals)
        ess_history.append(ess_val)
        n_leapfrog_history.append(n_leapfrog_stage)
        eps_history.append(current_eps)

        if verbose:
            extra = f", eps={current_eps:.4f}" if mutation == "nuts" else ""
            if use_flow and beta >= flow_start_beta:
                extra += " [flow]"
            if waste_free and mutation == "rw" and n_mutation_steps > 1:
                extra += " [waste-free]"
            acc_display = n_accept / max(_n_total_proposals, 1)
            print(
                f"  Stage {stage:2d}: beta={beta:.4f}, accept={acc_display:.2f}, "
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
        # New: evidence and diagnostics
        "log_evidence": log_evidence,
        "cv_weights": cv_weights_history,
        # Flow + waste-free flags
        "use_flow": use_flow,
        "waste_free": waste_free,
    }
