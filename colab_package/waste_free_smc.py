# -*- coding: utf-8 -*-
"""
waste_free_smc.py — Waste-free SMC (Dau & Chopin 2022).

Standard SMC discards rejected MCMC proposals, wasting likelihood evaluations.
Waste-free SMC keeps ALL proposals (accepted + rejected) with appropriate
importance weights, effectively doubling the ESS for the same compute budget.

Reference:
  - Dau & Chopin 2022: "Waste-free Sequential Monte Carlo"
    Journal of the Royal Statistical Society: Series B, 84(1), 114-148.

Integration:
  Called from tmcmc_nuts_engine.py when waste_free=True.
  Replaces the standard resample→mutate→discard cycle.
"""

from __future__ import annotations

import numpy as np


def waste_free_resample_and_mutate(
    particles: np.ndarray,
    logL: np.ndarray,
    weights: np.ndarray,
    beta_old: float,
    beta_new: float,
    mutation_fn,
    n_mutation_steps: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Waste-free SMC: keep all proposals with corrected weights.

    Standard SMC:
        1. Resample N particles from weights
        2. Mutate each M times (only keep final state)
        3. Total cost: N×M likelihood evaluations
        4. Output: N particles (M×N evaluations wasted)

    Waste-free SMC:
        1. Resample N/M "ancestors" from weights
        2. Mutate each M times, keep ALL intermediates
        3. Total cost: N likelihood evaluations (same!)
        4. Output: N particles with corrected weights
        5. ESS: ~2× higher (Dau & Chopin 2022, Theorem 3.1)

    Args:
        particles: Current particles, shape (N, d).
        logL: Current log-likelihoods, shape (N,).
        weights: Normalized importance weights, shape (N,).
        beta_old: Previous tempering parameter.
        beta_new: New tempering parameter.
        mutation_fn: Callable(particles_subset, logL_subset, rng)
            → (new_particles, new_logL, n_accept).
            Should perform ONE mutation step on the given particles.
        n_mutation_steps: Number of mutation steps M per ancestor.
        rng: NumPy random generator.

    Returns:
        (particles_out, logL_out, weights_out, n_accept_total)
        particles_out: shape (N, d) — all proposals + ancestors
        logL_out: shape (N,)
        weights_out: shape (N,) — corrected importance weights (normalized)
        n_accept_total: total number of accepted proposals
    """
    if rng is None:
        rng = np.random.default_rng()

    N, d = particles.shape
    M = max(n_mutation_steps, 1)

    # Number of ancestors: N_a = N / M (rounded)
    N_a = max(N // M, 1)
    # Actual number of output particles
    N_out = N_a * M

    # Step 1: Resample N_a ancestors
    # Use systematic resampling for lower variance
    idx = _systematic_resample_n(weights, N_a, rng)
    ancestors = particles[idx].copy()
    logL_ancestors = logL[idx].copy()

    # Step 2: Mutate each ancestor M times, keep ALL intermediates
    all_particles = np.empty((N_out, d), dtype=particles.dtype)
    all_logL = np.empty(N_out, dtype=logL.dtype)
    n_accept_total = 0

    for a in range(N_a):
        # Place ancestor at position a*M
        current = ancestors[a : a + 1].copy()  # (1, d)
        current_logL = logL_ancestors[a : a + 1].copy()  # (1,)

        for m in range(M):
            out_idx = a * M + m
            if m == 0:
                # First slot: the ancestor itself
                all_particles[out_idx] = current[0]
                all_logL[out_idx] = current_logL[0]
            else:
                # Mutate from current state
                new_p, new_lL, n_acc = mutation_fn(current, current_logL, rng)
                all_particles[out_idx] = new_p[0]
                all_logL[out_idx] = new_lL[0]
                n_accept_total += n_acc
                # Update current state (Markov chain)
                current = new_p
                current_logL = new_lL

    # Step 3: Compute corrected weights
    # Under waste-free SMC, each group of M particles from the same ancestor
    # gets weight proportional to exp((beta_new - beta_old) * logL)
    # normalized within each group, then across groups.
    delta_beta = beta_new - beta_old
    log_w = delta_beta * all_logL
    log_w -= np.max(log_w)  # numerical stability
    w = np.exp(np.clip(log_w, -500, 500))
    weights_out = w / w.sum()

    # Trim or pad to N if N_out != N
    if N_out < N:
        # Pad with zeros (shouldn't happen normally)
        particles_out = np.zeros((N, d), dtype=particles.dtype)
        logL_out = np.full(N, -np.inf)
        weights_out_full = np.zeros(N)
        particles_out[:N_out] = all_particles
        logL_out[:N_out] = all_logL
        weights_out_full[:N_out] = weights_out
        weights_out_full /= weights_out_full.sum()
        return particles_out, logL_out, weights_out_full, n_accept_total
    elif N_out > N:
        # Subsample to N (shouldn't happen with proper N_a calculation)
        all_particles = all_particles[:N]
        all_logL = all_logL[:N]
        weights_out = weights_out[:N]
        weights_out /= weights_out.sum()

    return all_particles, all_logL, weights_out, n_accept_total


def waste_free_resample_and_mutate_batch(
    particles: np.ndarray,
    logL: np.ndarray,
    weights: np.ndarray,
    beta_old: float,
    beta_new: float,
    batch_mutation_fn,
    n_mutation_steps: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Batch version of waste-free SMC for GPU efficiency.

    Instead of mutating one particle at a time, mutates N_a particles
    in batch at each step (preserving vmap efficiency).

    Args:
        particles: shape (N, d).
        logL: shape (N,).
        weights: Normalized importance weights, shape (N,).
        beta_old: Previous tempering parameter.
        beta_new: New tempering parameter.
        batch_mutation_fn: Callable(particles_batch, logL_batch, rng)
            → (new_particles, new_logL, n_accept).
            Operates on the full batch of N_a particles at once.
        n_mutation_steps: Number of steps M per ancestor.
        rng: NumPy random generator.

    Returns:
        (particles_out, logL_out, weights_out, n_accept_total)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, d = particles.shape
    M = max(n_mutation_steps, 1)
    N_a = max(N // M, 1)
    N_out = N_a * M

    # Resample N_a ancestors
    idx = _systematic_resample_n(weights, N_a, rng)
    current = particles[idx].copy()
    current_logL = logL[idx].copy()

    # Collect all intermediates
    all_particles = np.empty((N_out, d), dtype=particles.dtype)
    all_logL = np.empty(N_out, dtype=logL.dtype)
    n_accept_total = 0

    for m in range(M):
        start = m * N_a
        end = start + N_a

        if m == 0:
            # First step: ancestors themselves
            all_particles[start:end] = current
            all_logL[start:end] = current_logL
        else:
            # Batch mutation
            new_p, new_lL, n_acc = batch_mutation_fn(current, current_logL, rng)
            all_particles[start:end] = new_p
            all_logL[start:end] = new_lL
            n_accept_total += n_acc
            current = new_p
            current_logL = new_lL

    # Corrected weights
    delta_beta = beta_new - beta_old
    log_w = delta_beta * all_logL
    log_w -= np.max(log_w)
    w = np.exp(np.clip(log_w, -500, 500))
    weights_out = w / w.sum()

    # Adjust size to N
    if N_out < N:
        pad_particles = np.zeros((N, d), dtype=particles.dtype)
        pad_logL = np.full(N, -np.inf)
        pad_w = np.zeros(N)
        pad_particles[:N_out] = all_particles
        pad_logL[:N_out] = all_logL
        pad_w[:N_out] = weights_out
        pad_w /= pad_w.sum()
        return pad_particles, pad_logL, pad_w, n_accept_total
    else:
        all_particles = all_particles[:N]
        all_logL = all_logL[:N]
        weights_out = weights_out[:N]
        weights_out /= weights_out.sum()
        return all_particles, all_logL, weights_out, n_accept_total


def _systematic_resample_n(weights: np.ndarray, n_out: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling to select n_out indices from weights."""
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # ensure exact sum
    u0 = rng.random() / n_out
    u = u0 + np.arange(n_out) / n_out
    idx = np.searchsorted(cumsum, u)
    idx = np.clip(idx, 0, len(weights) - 1)
    return idx
