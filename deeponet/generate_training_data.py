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


def sample_theta(bounds: np.ndarray, rng: np.random.Generator,
                 theta_map: np.ndarray = None, map_std_frac: float = 0.1) -> np.ndarray:
    """Sample a single θ from uniform prior, or Gaussian around θ_MAP.

    If theta_map is provided, samples N(θ_MAP, (σ_i)^2) where σ_i = map_std_frac * range_i,
    then clips to prior bounds.
    """
    theta = np.zeros(20)
    for i in range(20):
        lo, hi = bounds[i]
        if abs(hi - lo) < 1e-12:
            theta[i] = lo  # locked
        elif theta_map is not None:
            sigma = map_std_frac * (hi - lo)
            theta[i] = np.clip(rng.normal(theta_map[i], sigma), lo, hi)
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


def load_theta_map(condition: str) -> np.ndarray:
    """Load θ_MAP from TMCMC runs. Returns (20,) or None."""
    run_map = {
        "Commensal_Static": "commensal_static",
        "Commensal_HOBIC": "commensal_hobic",
        "Dysbiotic_Static": "dysbiotic_static",
        "Dysbiotic_HOBIC": "dh_baseline",
    }
    dirname = run_map.get(condition)
    if dirname is None:
        return None

    base = PROJECT_ROOT / "data_5species" / "_runs"
    for pattern in [
        base / dirname / "theta_MAP.json",
        base / dirname / "posterior" / "theta_MAP.json",
    ]:
        if pattern.exists():
            with open(pattern) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return np.array(data["theta_full"], dtype=np.float64)
            return np.array(data, dtype=np.float64)
    return None


def generate_dataset(
    n_samples: int = 10000,
    condition: str = "Dysbiotic_HOBIC",
    seed: int = 42,
    maxtimestep: int = 500,
    dt: float = 1e-5,
    n_time_out: int = 100,  # downsample to this many time points
    map_frac: float = 0.3,  # fraction of samples around θ_MAP
    map_std_frac: float = 0.1,  # std dev as fraction of prior range
    expand_bounds: bool = True,  # expand bounds to include θ_MAP + margin
):
    """Generate training dataset with optional importance sampling around θ_MAP.

    Args:
        map_frac: fraction of total samples to draw from Gaussian around θ_MAP
                  (remaining drawn uniformly from prior). Set 0 to disable.
        map_std_frac: Gaussian std = map_std_frac * (hi - lo) per parameter.
        expand_bounds: if True and θ_MAP exists, expand bounds to include
                       θ_MAP ± 20% margin so DeepONet is never extrapolating.
    """

    bounds = load_prior_bounds(condition)

    # Expand bounds to include θ_MAP if it falls outside
    theta_map_check = load_theta_map(condition)
    if expand_bounds and theta_map_check is not None:
        n_expanded = 0
        for i in range(20):
            lo, hi = bounds[i]
            if abs(hi - lo) < 1e-12:
                continue  # locked param
            val = theta_map_check[i]
            rng_size = max(hi - lo, abs(val) * 0.5, 0.5)
            margin = 0.2 * rng_size
            if val < lo:
                bounds[i, 0] = val - margin
                n_expanded += 1
            if val > hi:
                bounds[i, 1] = val + margin
                n_expanded += 1
        if n_expanded > 0:
            print(f"Expanded {n_expanded} bound(s) to include θ_MAP")
    rng = np.random.default_rng(seed)

    # Use θ_MAP for importance sampling
    theta_map = theta_map_check if map_frac > 0 else None
    if theta_map is not None:
        n_map = int(n_samples * map_frac)
        n_uniform = n_samples - n_map
        print(f"Importance sampling: {n_map} MAP-centered + {n_uniform} uniform "
              f"(std={map_std_frac:.0%} of range)")
    else:
        n_map = 0
        n_uniform = n_samples
        if map_frac > 0:
            print(f"Warning: θ_MAP not found for {condition}, using uniform only")

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
    idx = None  # downsample indices

    n_success = 0
    n_failed = 0
    n_map_done = 0
    t0 = time.time()

    while n_success < n_samples:
        # Decide: MAP-centered or uniform
        use_map = (theta_map is not None and n_map_done < n_map)
        theta = sample_theta(
            bounds, rng,
            theta_map=theta_map if use_map else None,
            map_std_frac=map_std_frac,
        )
        result = run_single(solver, theta)

        if result is None:
            n_failed += 1
            continue

        t_arr, phi = result

        # Downsample time dimension
        if idx is None:
            idx = np.linspace(0, len(t_arr) - 1, n_time_out, dtype=int)
            t_out = t_arr[idx]

        phi_down = phi[idx]

        theta_list.append(theta)
        phi_list.append(phi_down)
        n_success += 1
        if use_map:
            n_map_done += 1

        if n_success % 500 == 0:
            elapsed = time.time() - t0
            rate = n_success / elapsed
            map_info = f", MAP={n_map_done}" if theta_map is not None else ""
            print(f"  {n_success}/{n_samples} done ({rate:.0f} samples/s, "
                  f"{n_failed} failed{map_info})")

    elapsed = time.time() - t0
    print(f"Done: {n_success} samples in {elapsed:.1f}s "
          f"({n_success/elapsed:.0f}/s, {n_failed} failed)")
    if theta_map is not None:
        print(f"  MAP-centered: {n_map_done}, Uniform: {n_success - n_map_done}")

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
    parser.add_argument("--map-frac", type=float, default=0.3,
                        help="Fraction of MAP-centered samples (0=uniform only)")
    parser.add_argument("--map-std", type=float, default=0.1,
                        help="Gaussian std as fraction of prior range")
    args = parser.parse_args()

    theta_arr, phi_arr, t_out, bounds = generate_dataset(
        n_samples=args.n_samples,
        condition=args.condition,
        seed=args.seed,
        n_time_out=args.n_time,
        maxtimestep=args.maxtimestep,
        map_frac=args.map_frac,
        map_std_frac=args.map_std,
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
