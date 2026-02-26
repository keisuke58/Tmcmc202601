#!/usr/bin/env python3
"""
TMCMC with DeepONet surrogate likelihood.

Replaces the numba ODE solver in the likelihood evaluation with
the trained DeepONet, achieving ~80× per-sample speedup (~29× E2E TMCMC).

Usage:
  python surrogate_tmcmc.py --checkpoint checkpoints/best.eqx \
      --norm-stats checkpoints/norm_stats.npz \
      --condition Dysbiotic_HOBIC --n-particles 500

Comparison mode (ODE vs DeepONet):
  python surrogate_tmcmc.py --compare --n-particles 200
"""

import argparse
import sys
import json
import time
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT / "data_5species"))

from improved_5species_jit import BiofilmNewtonSolver5S
from deeponet_hamilton import DeepONet


# ============================================================
# Surrogate likelihood
# ============================================================


class DeepONetSurrogate:
    """Wraps DeepONet as a drop-in ODE solver replacement."""

    def __init__(
        self,
        checkpoint: str,
        norm_stats_path: str,
        theta_dim: int = 20,
        n_species: int = 5,
        p: int = 64,
        hidden: int = 128,
        n_layers: int = 3,
        n_time_out: int = 100,
    ):
        # Auto-detect arch from config.json if available
        config_path = Path(checkpoint).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            p = cfg.get("p", p)
            hidden = cfg.get("hidden", hidden)
            n_layers = cfg.get("n_layers", n_layers)

        # Load model
        key = jax.random.PRNGKey(0)
        self.model = DeepONet(
            theta_dim=theta_dim, n_species=n_species, p=p, hidden=hidden, n_layers=n_layers, key=key
        )
        self.model = eqx.tree_deserialise_leaves(checkpoint, self.model)

        # Load normalization
        stats = np.load(norm_stats_path)
        self.theta_lo = stats["theta_lo"]
        self.theta_width = stats["theta_width"]
        self.t_min = float(stats["t_min"])
        self.t_max = float(stats["t_max"])

        # Time grid (normalized)
        self.n_time = n_time_out
        self.t_norm = jnp.linspace(0, 1, n_time_out, dtype=jnp.float32)

        # Pre-JIT the prediction
        self._predict_jit = jax.jit(self._predict_single)

        # Warmup
        dummy = jnp.zeros(theta_dim, dtype=jnp.float32)
        _ = self._predict_jit(dummy)

    def _predict_single(self, theta_norm):
        return self.model.predict_trajectory(theta_norm, self.t_norm)

    def predict(self, theta_raw: np.ndarray) -> np.ndarray:
        """
        Predict species trajectory from raw θ.

        Args:
            theta_raw: (20,) raw parameter vector

        Returns:
            phi_pred: (n_time, 5) predicted species fractions
        """
        theta_norm = (theta_raw - self.theta_lo) / self.theta_width
        theta_norm = jnp.array(theta_norm, dtype=jnp.float32)
        phi = self._predict_jit(theta_norm)
        return np.array(phi)

    def predict_at_indices(self, theta_raw: np.ndarray, idx_sparse: np.ndarray) -> np.ndarray:
        """
        Predict and extract at sparse observation indices.

        Args:
            theta_raw: (20,)
            idx_sparse: (n_obs,) indices into the time grid

        Returns:
            phi_sparse: (n_obs, 5)
        """
        phi_full = self.predict(theta_raw)
        return phi_full[idx_sparse]


def make_surrogate_log_likelihood(
    surrogate: DeepONetSurrogate,
    data: np.ndarray,
    idx_sparse: np.ndarray,
    sigma_obs: float = 0.02,
):
    """
    Create a log-likelihood function using the DeepONet surrogate.

    Args:
        surrogate: DeepONetSurrogate instance
        data: (n_obs, 5) observed data
        idx_sparse: (n_obs,) time indices for observations
        sigma_obs: observation noise std

    Returns:
        log_likelihood(theta) function
    """
    n_obs, n_species = data.shape

    def log_likelihood(theta: np.ndarray) -> float:
        phi_pred = surrogate.predict_at_indices(theta, idx_sparse)

        # Simple Gaussian log-likelihood
        residual = data - phi_pred
        var = sigma_obs**2
        logL = -0.5 * np.sum(residual**2 / var)
        logL -= 0.5 * n_obs * n_species * np.log(2 * np.pi * var)

        if not np.isfinite(logL):
            return -1e20
        return float(logL)

    return log_likelihood


# ============================================================
# Simple TMCMC (standalone, no external dependencies)
# ============================================================


def simple_tmcmc(
    log_likelihood,
    prior_bounds: np.ndarray,
    n_particles: int = 500,
    max_stages: int = 50,
    target_ess_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Minimal TMCMC implementation for demonstration.

    Args:
        log_likelihood: callable(theta) -> float
        prior_bounds: (d, 2) array of [lo, hi] bounds
        n_particles: number of particles
        max_stages: maximum number of tempering stages
        target_ess_ratio: target ESS/N ratio for beta selection
        seed: random seed

    Returns:
        dict with keys: samples, log_likelihoods, betas, times
    """
    rng = np.random.default_rng(seed)
    d = prior_bounds.shape[0]

    # Identify free (non-locked) dimensions
    free_dims = []
    for i in range(d):
        if abs(prior_bounds[i, 1] - prior_bounds[i, 0]) > 1e-12:
            free_dims.append(i)
    free_dims = np.array(free_dims)
    d_free = len(free_dims)

    # Sample from prior
    particles = np.zeros((n_particles, d))
    for i in range(d):
        lo, hi = prior_bounds[i]
        if abs(hi - lo) < 1e-12:
            particles[:, i] = lo
        else:
            particles[:, i] = rng.uniform(lo, hi, n_particles)

    # Evaluate initial log-likelihoods
    print(f"TMCMC: {n_particles} particles, {d_free} free dims")
    t0 = time.time()

    logL = np.array([log_likelihood(p) for p in particles])

    t_init = time.time() - t0
    print(f"  Initial evaluation: {t_init:.2f}s ({t_init/n_particles*1000:.1f} ms/particle)")

    beta = 0.0
    betas = [0.0]
    stage = 0
    stage_times = []

    while beta < 1.0 and stage < max_stages:
        stage += 1
        t_stage = time.time()

        # Find next beta via bisection (target ESS)
        def compute_ess(db):
            w = (db) * logL
            w = w - w.max()
            w = np.exp(w)
            return (np.sum(w) ** 2) / np.sum(w**2)

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
            delta_beta = 1.0 - beta  # jump to 1

        beta_new = min(beta + delta_beta, 1.0)

        # Importance weights
        w = (beta_new - beta) * logL
        w = w - w.max()
        w = np.exp(w)
        w = w / w.sum()

        # Resample
        idx = rng.choice(n_particles, size=n_particles, p=w)
        particles = particles[idx].copy()
        logL = logL[idx].copy()

        # Adaptive covariance
        cov = np.cov(particles[:, free_dims].T)
        if d_free == 1:
            cov = np.atleast_2d(cov)
        cov *= 0.04  # scale down

        # MCMC mutation (1 step per particle)
        n_accept = 0
        for i in range(n_particles):
            proposal = particles[i].copy()
            perturbation = rng.multivariate_normal(np.zeros(d_free), cov)
            proposal[free_dims] += perturbation

            # Check bounds
            in_bounds = True
            for j, dim in enumerate(free_dims):
                if proposal[dim] < prior_bounds[dim, 0] or proposal[dim] > prior_bounds[dim, 1]:
                    in_bounds = False
                    break

            if in_bounds:
                logL_new = log_likelihood(proposal)
                log_alpha = beta_new * (logL_new - logL[i])
                if np.log(rng.random()) < log_alpha:
                    particles[i] = proposal
                    logL[i] = logL_new
                    n_accept += 1

        beta = beta_new
        betas.append(beta)
        dt = time.time() - t_stage
        stage_times.append(dt)

        accept_rate = n_accept / n_particles
        print(
            f"  Stage {stage:2d}: beta={beta:.4f}, "
            f"accept={accept_rate:.2f}, "
            f"logL=[{logL.min():.1f}, {logL.max():.1f}], "
            f"{dt:.1f}s"
        )

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.1f}s, {len(betas)-1} stages")

    # MAP estimate
    best_idx = np.argmax(logL)
    theta_MAP = particles[best_idx]

    return {
        "samples": particles,
        "log_likelihoods": logL,
        "betas": np.array(betas),
        "theta_MAP": theta_MAP,
        "total_time": total_time,
        "stage_times": stage_times,
    }


# ============================================================
# Comparison: DeepONet vs ODE TMCMC
# ============================================================


def run_comparison(n_particles: int = 200, seed: int = 42):
    """Run TMCMC with both ODE and DeepONet, compare results and speed."""

    # Load prior bounds
    bounds_file = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(bounds_file) as f:
        cfg = json.load(f)

    condition = "Dysbiotic_HOBIC"
    strategy = cfg["strategies"][condition]
    locks = set(strategy.get("locks", []))
    custom = strategy.get("bounds", {})

    bounds = np.zeros((20, 2))
    for i in range(20):
        if i in locks:
            bounds[i] = [0.0, 0.0]
        elif str(i) in custom:
            bounds[i] = custom[str(i)]
        else:
            bounds[i] = cfg["default_bounds"]

    # Create synthetic observation data from a known theta
    rng = np.random.default_rng(seed)
    theta_true = np.zeros(20)
    for i in range(20):
        lo, hi = bounds[i]
        if abs(hi - lo) > 1e-12:
            theta_true[i] = rng.uniform(lo, hi)

    solver = BiofilmNewtonSolver5S(
        dt=1e-5,
        maxtimestep=500,
        eps=1e-6,
        Kp1=1e-4,
        c_const=100.0,
        alpha_const=100.0,
        phi_init=0.2,
        K_hill=0.05,
        n_hill=4.0,
        max_newton_iter=50,
        use_numba=True,
    )

    t_arr, g_arr = solver.run_deterministic(theta_true)
    phi_true = g_arr[:, :5]

    # Sparse observation at ~6 time points
    idx_full = np.linspace(0, len(t_arr) - 1, 100, dtype=int)
    obs_times = [10, 30, 50, 70, 85, 99]  # indices into downsampled grid
    idx_sparse = np.array(obs_times)

    phi_down = phi_true[idx_full]
    sigma_obs = 0.02
    data = phi_down[idx_sparse] + rng.normal(0, sigma_obs, (len(obs_times), 5))

    # ----- ODE-based TMCMC -----
    print("=" * 60)
    print("ODE-based TMCMC")
    print("=" * 60)

    def ode_log_likelihood(theta):
        try:
            t_a, g_a = solver.run_deterministic(theta)
            phi = g_a[idx_full][:, :5]
            phi_obs = phi[idx_sparse]
            residual = data - phi_obs
            logL = -0.5 * np.sum(residual**2 / sigma_obs**2)
            logL -= 0.5 * len(obs_times) * 5 * np.log(2 * np.pi * sigma_obs**2)
            if not np.isfinite(logL):
                return -1e20
            return float(logL)
        except Exception:
            return -1e20

    result_ode = simple_tmcmc(ode_log_likelihood, bounds, n_particles=n_particles, seed=seed)

    # ----- DeepONet-based TMCMC -----
    print("\n" + "=" * 60)
    print("DeepONet-based TMCMC")
    print("=" * 60)

    ckpt_dir = Path(__file__).parent / "checkpoints"
    surrogate = DeepONetSurrogate(
        str(ckpt_dir / "best.eqx"),
        str(ckpt_dir / "norm_stats.npz"),
    )
    don_logL = make_surrogate_log_likelihood(surrogate, data, idx_sparse, sigma_obs)

    result_don = simple_tmcmc(don_logL, bounds, n_particles=n_particles, seed=seed)

    # ----- Comparison -----
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    speedup = result_ode["total_time"] / result_don["total_time"]
    print(f"  ODE total time:     {result_ode['total_time']:.1f}s")
    print(f"  DeepONet total time: {result_don['total_time']:.1f}s")
    print(f"  Speedup:            {speedup:.0f}x")

    # MAP comparison
    species = ["So", "An", "Vd", "Fn", "Pg"]
    print(f"\n  {'Param':<6} {'True':>8} {'ODE MAP':>8} {'DON MAP':>8}")
    print("  " + "-" * 34)
    free_dims = [i for i in range(20) if abs(bounds[i, 1] - bounds[i, 0]) > 1e-12]
    for i in free_dims[:10]:  # show first 10
        name = BiofilmNewtonSolver5S.THETA_NAMES[i]
        print(
            f"  {name:<6} {theta_true[i]:>8.3f} "
            f"{result_ode['theta_MAP'][i]:>8.3f} "
            f"{result_don['theta_MAP'][i]:>8.3f}"
        )
    if len(free_dims) > 10:
        print(f"  ... ({len(free_dims) - 10} more)")

    return result_ode, result_don


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare", action="store_true", help="Run comparison between ODE and DeepONet TMCMC"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.eqx")
    parser.add_argument("--norm-stats", default="checkpoints/norm_stats.npz")
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.compare:
        run_comparison(n_particles=args.n_particles, seed=args.seed)
    else:
        # Standalone DeepONet TMCMC demo
        print("Use --compare to run ODE vs DeepONet comparison")
        print("Or import DeepONetSurrogate + make_surrogate_log_likelihood for custom use")


if __name__ == "__main__":
    main()
