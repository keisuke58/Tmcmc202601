# -*- coding: utf-8 -*-
"""
normalizing_flow.py — RealNVP Normalizing Flow for SMC proposals.

Pure JAX implementation. Can be used as a learned proposal distribution
within TMCMC to achieve near-100% acceptance rates.

References:
  - Dinh et al. 2017: Density estimation using Real-valued Non-Volume Preserving (RealNVP)
  - Gabrie et al. 2022: Adaptive Monte Carlo augmented with normalizing flows
  - Arbel et al. 2021: Annealed flow transport Monte Carlo
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)


# =====================================================================
# MLP building blocks
# =====================================================================


def _init_mlp(key: jax.Array, layer_sizes: list[int]) -> list[Tuple]:
    """Initialize MLP parameters (Xavier init)."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, k = jr.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        W = jr.normal(k, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def _forward_mlp(params: list, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through MLP with LeakyReLU activations."""
    for W, b in params[:-1]:
        x = x @ W + b
        x = jnp.where(x > 0, x, 0.01 * x)  # LeakyReLU
    W, b = params[-1]
    return x @ W + b


# =====================================================================
# RealNVP Coupling Layer
# =====================================================================


def _make_mask(d: int, layer_idx: int) -> jnp.ndarray:
    """Alternating binary mask for coupling layers."""
    mask = jnp.arange(d) % 2
    if layer_idx % 2 == 1:
        mask = 1 - mask
    return mask.astype(jnp.float64)


def _init_coupling_layer(key: jax.Array, d: int, hidden_dim: int) -> dict:
    """Initialize one coupling layer (scale + shift networks)."""
    k1, k2 = jr.split(key)
    # Input: masked half of dimensions → output: other half
    # Use d as input (full x is input, mask applied inside)
    layers_s = [d, hidden_dim, hidden_dim, d]
    layers_t = [d, hidden_dim, hidden_dim, d]
    return {
        "s_params": _init_mlp(k1, layers_s),
        "t_params": _init_mlp(k2, layers_t),
    }


def _coupling_forward(params: dict, x: jnp.ndarray, mask: jnp.ndarray):
    """Forward pass: x → z, returns (z, log_det_J)."""
    x_masked = x * mask
    s = _forward_mlp(params["s_params"], x_masked)
    t = _forward_mlp(params["t_params"], x_masked)
    # Clamp s for numerical stability
    s = jnp.clip(s, -5.0, 5.0)
    # Apply to non-masked dimensions
    z = x_masked + (1 - mask) * (x * jnp.exp(s) + t)
    log_det = jnp.sum((1 - mask) * s)
    return z, log_det


def _coupling_inverse(params: dict, z: jnp.ndarray, mask: jnp.ndarray):
    """Inverse pass: z → x, returns (x, log_det_J)."""
    z_masked = z * mask
    s = _forward_mlp(params["s_params"], z_masked)
    t = _forward_mlp(params["t_params"], z_masked)
    s = jnp.clip(s, -5.0, 5.0)
    x = z_masked + (1 - mask) * (z - t) * jnp.exp(-s)
    log_det = -jnp.sum((1 - mask) * s)
    return x, log_det


# =====================================================================
# Full RealNVP Flow
# =====================================================================


def init_flow(key: jax.Array, d: int, n_layers: int = 6, hidden_dim: int = 64) -> dict:
    """Initialize a RealNVP flow.

    Args:
        key: JAX PRNG key.
        d: Dimensionality of the parameter space.
        n_layers: Number of coupling layers.
        hidden_dim: Hidden dimension of scale/shift MLPs.

    Returns:
        Flow parameters dict with 'layers', 'masks', 'd', 'n_layers'.
    """
    layers = []
    masks = []
    for i in range(n_layers):
        key, k = jr.split(key)
        layers.append(_init_coupling_layer(k, d, hidden_dim))
        masks.append(_make_mask(d, i))
    return {
        "layers": layers,
        "masks": masks,
        "d": d,
        "n_layers": n_layers,
    }


def flow_forward(flow_params: dict, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transform x (data space) → z (latent space).

    Args:
        flow_params: Flow parameters from init_flow().
        x: Input samples, shape (d,).

    Returns:
        (z, log_det_J): Transformed samples and log determinant of Jacobian.
    """
    z = x
    log_det = 0.0
    for layer, mask in zip(flow_params["layers"], flow_params["masks"]):
        z, ld = _coupling_forward(layer, z, mask)
        log_det += ld
    return z, log_det


def flow_inverse(flow_params: dict, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transform z (latent space) → x (data space).

    Args:
        flow_params: Flow parameters from init_flow().
        z: Latent samples, shape (d,).

    Returns:
        (x, log_det_J): Transformed samples and log determinant of Jacobian.
    """
    x = z
    log_det = 0.0
    # Inverse order
    for layer, mask in zip(
        reversed(flow_params["layers"]),
        reversed(flow_params["masks"]),
    ):
        x, ld = _coupling_inverse(layer, x, mask)
        log_det += ld
    return x, log_det


def flow_log_prob(flow_params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Compute log q(x) under the flow.

    The base distribution is standard normal N(0, I).

    Args:
        flow_params: Flow parameters.
        x: Sample in data space, shape (d,).

    Returns:
        log q(x) scalar.
    """
    d = flow_params["d"]
    z, log_det = flow_forward(flow_params, x)
    # Standard normal base density
    log_pz = -0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(z**2)
    return log_pz + log_det


# =====================================================================
# Training: fit flow to weighted particle set
# =====================================================================


def _flatten_params(flow_params: dict) -> Tuple[jnp.ndarray, dict]:
    """Flatten flow params into a single vector + structure info for unflatten."""
    flat_list = []
    structure = {"shapes": [], "n_layers": flow_params["n_layers"], "d": flow_params["d"]}
    for layer in flow_params["layers"]:
        for net_key in ["s_params", "t_params"]:
            for W, b in layer[net_key]:
                flat_list.append(W.ravel())
                structure["shapes"].append(("W", W.shape))
                flat_list.append(b.ravel())
                structure["shapes"].append(("b", b.shape))
    return jnp.concatenate(flat_list), structure


def _unflatten_params(flat: jnp.ndarray, structure: dict, masks: list) -> dict:
    """Unflatten a parameter vector back into flow params dict."""
    layers = []
    idx = 0
    n_layers = structure["n_layers"]
    for _ in range(n_layers):
        layer = {}
        for net_key in ["s_params", "t_params"]:
            params = []
            # Each MLP has 3 layers → 3 (W, b) pairs
            for _ in range(3):
                _, W_shape = structure["shapes"][idx]
                W_size = W_shape[0] * W_shape[1]
                W = flat[
                    idx * 0
                    + sum(s[0] * s[1] if t == "W" else s[0] for t, s in structure["shapes"][:idx]) :
                ].reshape(
                    -1
                )  # noqa
                # Simpler: just consume sequentially
                pass
            layer[net_key] = params
        layers.append(layer)
    # This is getting complex; use pytree instead
    return {"layers": layers, "masks": masks, "d": structure["d"], "n_layers": n_layers}


def train_flow(
    key: jax.Array,
    flow_params: dict,
    particles: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 0,
) -> dict:
    """Train flow to approximate weighted particle distribution.

    Uses Adam optimizer to maximize weighted log-likelihood of particles
    under the flow. This is equivalent to minimizing KL(p_target || q_flow).

    Args:
        key: JAX PRNG key.
        flow_params: Initial flow parameters.
        particles: Particle positions, shape (N, d).
        weights: Importance weights, shape (N,). If None, uniform.
        n_epochs: Number of training epochs.
        lr: Learning rate for Adam.
        batch_size: Mini-batch size. 0 = full batch.

    Returns:
        Trained flow parameters.
    """
    import optax

    n = particles.shape[0]
    if weights is None:
        weights = jnp.ones(n) / n
    else:
        weights = jnp.asarray(weights)
        weights = weights / weights.sum()

    if batch_size <= 0 or batch_size >= n:
        batch_size = n

    # Use JAX pytree for params (treat flow_params as nested dict/list)
    # We need to make all params into a proper pytree
    # Convert list-of-tuples to list-of-dicts for JAX compatibility
    def _convert_to_pytree(fp):
        """Convert flow params to JAX-friendly pytree."""
        pytree_layers = []
        for layer in fp["layers"]:
            pt_layer = {}
            for net_key in ["s_params", "t_params"]:
                pt_net = []
                for W, b in layer[net_key]:
                    pt_net.append({"W": W, "b": b})
                pt_layer[net_key] = pt_net
            pytree_layers.append(pt_layer)
        return pytree_layers

    def _convert_from_pytree(pytree_layers, fp):
        """Convert back from pytree to flow params."""
        layers = []
        for pt_layer in pytree_layers:
            layer = {}
            for net_key in ["s_params", "t_params"]:
                params = []
                for d in pt_layer[net_key]:
                    params.append((d["W"], d["b"]))
                layer[net_key] = params
            layers.append(layer)
        return {
            "layers": layers,
            "masks": fp["masks"],
            "d": fp["d"],
            "n_layers": fp["n_layers"],
        }

    def _mlp_forward_pt(pt_net, x):
        """MLP forward with pytree params."""
        for layer_d in pt_net[:-1]:
            x = x @ layer_d["W"] + layer_d["b"]
            x = jnp.where(x > 0, x, 0.01 * x)
        d_last = pt_net[-1]
        return x @ d_last["W"] + d_last["b"]

    def _coupling_fwd_pt(pt_layer, x, mask):
        s = _mlp_forward_pt(pt_layer["s_params"], x * mask)
        t = _mlp_forward_pt(pt_layer["t_params"], x * mask)
        s = jnp.clip(s, -5.0, 5.0)
        z = x * mask + (1 - mask) * (x * jnp.exp(s) + t)
        log_det = jnp.sum((1 - mask) * s)
        return z, log_det

    masks = flow_params["masks"]
    d_dim = flow_params["d"]

    def loss_fn(pytree_layers, batch_x, batch_w):
        """Weighted negative log-likelihood loss."""

        def single_log_prob(x):
            z = x
            ld = 0.0
            for pt_layer, mask in zip(pytree_layers, masks):
                z, l = _coupling_fwd_pt(pt_layer, z, mask)
                ld += l
            log_pz = -0.5 * d_dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(z**2)
            return log_pz + ld

        log_probs = jax.vmap(single_log_prob)(batch_x)
        # Weighted loss: minimize -E_p[log q(x)]
        return -jnp.sum(batch_w * log_probs) / jnp.sum(batch_w)

    pytree = _convert_to_pytree(flow_params)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(pytree)

    @jax.jit
    def step(pytree, opt_state, batch_x, batch_w):
        loss, grads = jax.value_and_grad(loss_fn)(pytree, batch_x, batch_w)
        updates, opt_state_new = optimizer.update(grads, opt_state, pytree)
        pytree_new = optax.apply_updates(pytree, updates)
        return pytree_new, opt_state_new, loss

    best_loss = jnp.inf
    best_pytree = pytree

    for epoch in range(n_epochs):
        if batch_size < n:
            key, k = jr.split(key)
            perm = jr.permutation(k, n)
            batch_x = particles[perm[:batch_size]]
            batch_w = weights[perm[:batch_size]]
            batch_w = batch_w / batch_w.sum()
        else:
            batch_x = particles
            batch_w = weights

        pytree, opt_state, loss = step(pytree, opt_state, batch_x, batch_w)

        if loss < best_loss:
            best_loss = loss
            best_pytree = jax.tree.map(lambda x: x.copy(), pytree)

    return _convert_from_pytree(best_pytree, flow_params)


# =====================================================================
# SMC proposal: sample from trained flow
# =====================================================================


def flow_sample(
    key: jax.Array,
    flow_params: dict,
    n_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw samples from the flow by inverting standard normal samples.

    Args:
        key: JAX PRNG key.
        flow_params: Trained flow parameters.
        n_samples: Number of samples to draw.

    Returns:
        (samples, log_q): Samples in data space and their log-densities.
            samples: shape (n_samples, d)
            log_q: shape (n_samples,)
    """
    d = flow_params["d"]
    z = jr.normal(key, (n_samples, d))

    def single_inverse(zi):
        return flow_inverse(flow_params, zi)

    samples, log_det = jax.vmap(single_inverse)(z)
    # log q(x) = log p(z) + log|det(dz/dx)| = log p(z) - log|det(dx/dz)|
    # flow_inverse returns log|det(dx/dz)|, so log q = log p(z) - log_det_inverse
    log_pz = -0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(z**2, axis=1)
    # log_det from inverse is log|det(dx/dz)|, but we need log|det(dz/dx)| = -log_det
    log_q = log_pz - log_det
    return samples, log_q


def flow_proposal_mh(
    key: jax.Array,
    flow_params: dict,
    particles: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
    beta: float,
    prior_bounds: jnp.ndarray,
    log_likelihood_fn=None,
    log_likelihood_vmap=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Generate flow proposals with independent MH correction.

    For each particle, sample from flow and accept/reject using:
        alpha = min(1, [p(x'|y)^beta * q(x)] / [p(x|y)^beta * q(x')])

    where q is the flow density.

    Args:
        key: JAX PRNG key.
        flow_params: Trained flow parameters.
        particles: Current particles, shape (N, d).
        log_likelihoods: Current log-likelihoods, shape (N,).
        beta: Current tempering parameter.
        prior_bounds: Prior bounds, shape (d, 2).
        log_likelihood_fn: Single-point log-likelihood function.
        log_likelihood_vmap: Batched log-likelihood function.

    Returns:
        (new_particles, new_logL, n_accept)
    """
    import numpy as np

    n = particles.shape[0]
    d = particles.shape[1]

    # Sample proposals from flow
    k1, k2 = jr.split(key)
    proposals, log_q_proposals = flow_sample(k1, flow_params, n)

    # Check bounds
    in_bounds = jnp.all(
        (proposals >= prior_bounds[:, 0]) & (proposals <= prior_bounds[:, 1]),
        axis=1,
    )

    # Compute log q for current particles
    log_q_current = jax.vmap(lambda x: flow_log_prob(flow_params, x))(particles)

    # Compute log-likelihoods for proposals
    if log_likelihood_vmap is not None:
        logL_proposals = log_likelihood_vmap(proposals)
    else:
        logL_proposals = jax.vmap(log_likelihood_fn)(proposals)

    # MH acceptance ratio (independent proposal):
    # log alpha = beta * (logL_new - logL_old) + (log_q_old - log_q_new)
    log_alpha = beta * (logL_proposals - log_likelihoods) + (log_q_current - log_q_proposals)

    # Random uniform for acceptance
    u = jnp.log(jr.uniform(k2, (n,)))

    # Accept where in bounds and alpha > u
    accept = in_bounds & (u < log_alpha) & jnp.isfinite(logL_proposals)

    new_particles = jnp.where(accept[:, None], proposals, particles)
    new_logL = jnp.where(accept, logL_proposals, log_likelihoods)
    n_accept = int(jnp.sum(accept))

    return np.array(new_particles), np.array(new_logL), n_accept
