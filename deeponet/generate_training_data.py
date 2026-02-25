#!/usr/bin/env python3
"""
Generate training data for DeepONet surrogate of 5-species Hamilton ODE.

Samples θ (20 params) from prior → runs Hamilton ODE → saves (θ, t, φ(t)).
Uses numba-jitted solver for speed.

Output:
  data/train_N{n_samples}.npz with keys:
    theta: (N, 20) - parameter samples
    phi:   (N, T, 5) - species volume fractions φ_i(t)
    t:     (T,) - time grid
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import time

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

from improved_5species_jit import BiofilmNewtonSolver5S


def load_prior_bounds(condition: str = "Dysbiotic_HOBIC") -> np.ndarray:
    """Load prior bounds from config. Returns (20, 2) array."""
    bounds_file = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(bounds_file) as f:
        cfg = json.load(f)

    default_lo, default_hi = cfg["default_bounds"]
    strategy = cfg["strategies"].get(condition, {})
    locks = set(strategy.get("locks", []))
    custom = strategy.get("bounds", {})

    bounds = np.zeros((20, 2))
    for i in range(20):
        if i in locks:
            bounds[i] = [0.0, 0.0]  # locked = 0
        elif str(i) in custom:
            bounds[i] = custom[str(i)]
        else:
            bounds[i] = [default_lo, default_hi]
    return bounds


def sample_theta(bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a single θ from uniform prior."""
    theta = np.zeros(20)
    for i in range(20):
        lo, hi = bounds[i]
        if abs(hi - lo) < 1e-12:
            theta[i] = lo  # locked
        else:
            theta[i] = rng.uniform(lo, hi)
    return theta


def run_single(solver: BiofilmNewtonSolver5S, theta: np.ndarray):
    """Run solver, return (t, phi) or None if failed."""
    try:
        t_arr, g_arr = solver.run_deterministic(theta)
        phi = g_arr[:, :5]  # species volume fractions
        # Check for NaN/explosion
        if np.any(~np.isfinite(phi)):
            return None
        if np.any(phi < -0.5) or np.any(phi > 1.5):
            return None
        return t_arr, phi
    except Exception:
        return None


def generate_dataset(
    n_samples: int = 10000,
    condition: str = "Dysbiotic_HOBIC",
    seed: int = 42,
    maxtimestep: int = 500,
    dt: float = 1e-5,
    n_time_out: int = 100,  # downsample to this many time points
):
    """Generate training dataset."""

    bounds = load_prior_bounds(condition)
    rng = np.random.default_rng(seed)

    solver = BiofilmNewtonSolver5S(
        dt=dt,
        maxtimestep=maxtimestep,
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

    # Warmup numba JIT
    print("Warming up JIT...")
    theta_test = sample_theta(bounds, rng)
    _ = run_single(solver, theta_test)

    print(f"Generating {n_samples} samples (condition={condition})...")

    theta_list = []
    phi_list = []
    t_out = None

    n_success = 0
    n_failed = 0
    t0 = time.time()

    while n_success < n_samples:
        theta = sample_theta(bounds, rng)
        result = run_single(solver, theta)

        if result is None:
            n_failed += 1
            continue

        t_arr, phi = result

        # Downsample time dimension
        if t_out is None:
            idx = np.linspace(0, len(t_arr) - 1, n_time_out, dtype=int)
            t_out = t_arr[idx]

        phi_down = phi[idx]

        theta_list.append(theta)
        phi_list.append(phi_down)
        n_success += 1

        if n_success % 500 == 0:
            elapsed = time.time() - t0
            rate = n_success / elapsed
            print(f"  {n_success}/{n_samples} done ({rate:.0f} samples/s, "
                  f"{n_failed} failed)")

    elapsed = time.time() - t0
    print(f"Done: {n_success} samples in {elapsed:.1f}s "
          f"({n_success/elapsed:.0f}/s, {n_failed} failed)")

    theta_arr = np.array(theta_list)   # (N, 20)
    phi_arr = np.array(phi_list)       # (N, n_time_out, 5)

    return theta_arr, phi_arr, t_out, bounds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--condition", default="Dysbiotic_HOBIC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-time", type=int, default=100)
    parser.add_argument("--maxtimestep", type=int, default=500)
    args = parser.parse_args()

    theta_arr, phi_arr, t_out, bounds = generate_dataset(
        n_samples=args.n_samples,
        condition=args.condition,
        seed=args.seed,
        n_time_out=args.n_time,
        maxtimestep=args.maxtimestep,
    )

    # Save
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"train_{args.condition}_N{args.n_samples}.npz"

    np.savez_compressed(
        out_file,
        theta=theta_arr,
        phi=phi_arr,
        t=t_out,
        bounds=bounds,
    )
    print(f"Saved to {out_file}")
    print(f"  theta: {theta_arr.shape}")
    print(f"  phi:   {phi_arr.shape}")
    print(f"  t:     {t_out.shape}")


if __name__ == "__main__":
    main()
