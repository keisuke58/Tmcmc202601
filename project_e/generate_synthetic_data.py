#!/usr/bin/env python3
"""
Project E Phase 2: Generate synthetic (y_obs, theta) pairs for amortized inference.

Samples theta from prior (or around MAP), runs Hamilton ODE, extracts y_obs at 6 days.
Output format matches TMCMC data: y_obs (6, 5), theta (20).

Usage:
    python generate_synthetic_data.py --n-samples 5000 --condition Dysbiotic_HOBIC
    python generate_synthetic_data.py --n-samples 10000 --all-conditions
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

from improved_5species_jit import BiofilmNewtonSolver5S

# Observation timepoints (model time) for days [1, 3, 6, 10, 15, 21]
# From config metadata: t_model
T_OBS = np.array([0.0113, 0.0339, 0.0679, 0.113, 0.170, 0.2375], dtype=np.float64)

CONDITIONS = ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_Static", "Dysbiotic_HOBIC"]
RUN_MAP = {
    "Commensal_Static": "commensal_static_posterior",
    "Commensal_HOBIC": "commensal_hobic_posterior",
    "Dysbiotic_Static": "dysbiotic_static_posterior",
    "Dysbiotic_HOBIC": "dh_baseline",
}


def load_prior_bounds(condition: str):
    """Load prior bounds from model_config."""
    cfg_path = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    default_lo, default_hi = cfg["default_bounds"]
    strategy = cfg["strategies"].get(condition, {})
    locks = set(strategy.get("locks", []))
    custom = strategy.get("bounds", {})

    bounds = np.zeros((20, 2))
    for i in range(20):
        if i in locks:
            bounds[i] = [0.0, 0.0]
        elif str(i) in custom:
            bounds[i] = custom[str(i)]
        else:
            bounds[i] = [default_lo, default_hi]
    return bounds


def load_theta_map(condition: str):
    """Load theta_MAP for condition."""
    dirname = RUN_MAP.get(condition)
    if dirname is None:
        return None
    base = PROJECT_ROOT / "data_5species" / "_runs"
    for pattern in [base / dirname / "theta_MAP.json", base / dirname / "theta_mean.json"]:
        if pattern.exists():
            with open(pattern) as f:
                data = json.load(f)
            arr = data.get("theta_full", data.get("theta_sub"))
            return np.array(arr, dtype=np.float64)
    return None


def load_posterior_samples(condition: str):
    """Load TMCMC posterior samples for condition. Returns (n, 20) or None."""
    dirname = RUN_MAP.get(condition)
    if dirname is None:
        return None
    samples_path = PROJECT_ROOT / "data_5species" / "_runs" / dirname / "samples.npy"
    if not samples_path.exists():
        return None
    return np.load(samples_path).astype(np.float64)


# Dysbiotic_HOBIC: posterior mean ≠ θ_MAP, prior wide → use posterior-informed sampling
CONDITION_PRESETS = {
    "Dysbiotic_HOBIC": {"map_frac": 0.3, "posterior_frac": 0.5},
}


def sample_theta(bounds, rng, theta_map=None, map_std_frac=0.15):
    """Sample theta from prior or around MAP."""
    theta = np.zeros(20)
    for i in range(20):
        lo, hi = bounds[i]
        if abs(hi - lo) < 1e-12:
            theta[i] = lo
        elif theta_map is not None:
            sigma = map_std_frac * max(abs(hi - lo), 0.1)
            theta[i] = np.clip(rng.normal(theta_map[i], sigma), lo, hi)
        else:
            theta[i] = rng.uniform(lo, hi)
    return theta


def run_ode(solver, theta):
    """Run ODE, return phi at T_OBS or None if failed."""
    try:
        t_arr, g_arr = solver.run_deterministic(theta)
        phi_full = g_arr[:, :5]
        if np.any(~np.isfinite(phi_full)) or np.any(phi_full < -0.1) or np.any(phi_full > 1.1):
            return None
        # Interpolate at T_OBS
        phi_obs = np.zeros((6, 5))
        for sp in range(5):
            phi_obs[:, sp] = np.interp(T_OBS, t_arr, phi_full[:, sp])
        # Normalize to sum=1 per timepoint (species fractions)
        row_sum = phi_obs.sum(axis=1, keepdims=True)
        row_sum[row_sum < 1e-8] = 1.0
        phi_obs = phi_obs / row_sum
        return phi_obs.astype(np.float32)
    except Exception:
        return None


def generate_dataset(
    n_samples: int = 5000,
    condition: str = "Dysbiotic_HOBIC",
    seed: int = 42,
    map_frac: float = 0.3,
    posterior_frac: float = 0.0,
    sigma_noise: float = 0.0,
    maxtimestep: int = 2500,
    dt: float = 1e-4,
):
    """Generate (y_obs, theta) pairs.

    Sampling strategy:
    - posterior_frac: from TMCMC posterior (if available) — fixes Dysbiotic_HOBIC MAE
    - map_frac of remainder: around theta_MAP
    - rest: uniform prior
    """
    preset = CONDITION_PRESETS.get(condition, {})
    posterior_frac = (
        preset.get("posterior_frac", 0) if posterior_frac == 0 else posterior_frac
    )
    map_frac = preset.get("map_frac", map_frac)

    rng = np.random.default_rng(seed)
    bounds = load_prior_bounds(condition)
    theta_map = load_theta_map(condition)
    posterior = load_posterior_samples(condition)
    solver = BiofilmNewtonSolver5S(maxtimestep=maxtimestep, dt=dt)

    n_post = int(n_samples * posterior_frac) if posterior is not None else 0
    n_rest = n_samples - n_post
    n_map = int(n_rest * map_frac)
    n_prior = n_rest - n_map

    y_list, theta_list = [], []
    n_try = 0
    max_try = n_samples * 3

    # 1) Posterior samples (theta from TMCMC, y from ODE)
    if n_post > 0 and posterior is not None:
        indices = rng.integers(0, len(posterior), size=n_post)
        for i in indices:
            th = posterior[i].copy()
            phi = run_ode(solver, th)
            if phi is not None:
                if sigma_noise > 0:
                    phi = np.clip(
                        phi + rng.normal(0, sigma_noise, phi.shape), 0.01, 0.99
                    )
                    phi = phi / phi.sum(axis=1, keepdims=True)
                y_list.append(phi)
                theta_list.append(th)

    # 2) Prior (uniform)
    while len(y_list) < n_post + n_prior and n_try < max_try:
        th = sample_theta(bounds, rng, theta_map=None)
        phi = run_ode(solver, th)
        if phi is not None:
            if sigma_noise > 0:
                phi = np.clip(phi + rng.normal(0, sigma_noise, phi.shape), 0.01, 0.99)
                phi = phi / phi.sum(axis=1, keepdims=True)
            y_list.append(phi)
            theta_list.append(th)
        n_try += 1

    # 3) MAP neighborhood
    while len(y_list) < n_samples and theta_map is not None and n_try < max_try:
        th = sample_theta(bounds, rng, theta_map=theta_map, map_std_frac=0.15)
        phi = run_ode(solver, th)
        if phi is not None:
            if sigma_noise > 0:
                phi = np.clip(phi + rng.normal(0, sigma_noise, phi.shape), 0.01, 0.99)
                phi = phi / phi.sum(axis=1, keepdims=True)
            y_list.append(phi)
            theta_list.append(th)
        n_try += 1

    y_obs = np.array(y_list, dtype=np.float32)
    theta = np.array(theta_list, dtype=np.float32)
    return {"y_obs": y_obs, "theta": theta, "condition": condition, "n_valid": len(y_list)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--condition", type=str, default="Dysbiotic_HOBIC")
    parser.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--map-frac", type=float, default=0.3)
    parser.add_argument(
        "--posterior-frac",
        type=float,
        default=0.0,
        help="Fraction from TMCMC posterior (Dysbiotic_HOBIC preset: 0.5)",
    )
    parser.add_argument("--sigma-noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_conditions:
        all_y, all_theta, all_labels = [], [], []
        for cond in CONDITIONS:
            print(f"Generating {args.n_samples} for {cond}...")
            data = generate_dataset(
                n_samples=args.n_samples,
                condition=cond,
                seed=args.seed,
                map_frac=args.map_frac,
                posterior_frac=args.posterior_frac,
                sigma_noise=args.sigma_noise,
            )
            all_y.append(data["y_obs"])
            all_theta.append(data["theta"])
            all_labels.extend([cond] * data["n_valid"])
            print(f"  Valid: {data['n_valid']}")

        y_obs = np.concatenate(all_y, axis=0)
        theta = np.concatenate(all_theta, axis=0)
        out_path = out_dir / f"synthetic_all_N{len(y_obs)}.npz"
        save_kw = {"y_obs": y_obs, "theta": theta, "condition_labels": np.array(all_labels)}
    else:
        print(f"Generating {args.n_samples} for {args.condition}...")
        data = generate_dataset(
            n_samples=args.n_samples,
            condition=args.condition,
            seed=args.seed,
            map_frac=args.map_frac,
            posterior_frac=args.posterior_frac,
            sigma_noise=args.sigma_noise,
        )
        y_obs = data["y_obs"]
        theta = data["theta"]
        print(f"  Valid: {data['n_valid']}")
        out_path = out_dir / (args.output or f"synthetic_{args.condition}_N{len(y_obs)}.npz")
        save_kw = {"y_obs": y_obs, "theta": theta}

    np.savez_compressed(out_path, **save_kw)
    print(f"Saved: {out_path}")
    print(f"  y_obs: {y_obs.shape}, theta: {theta.shape}")


if __name__ == "__main__":
    main()
