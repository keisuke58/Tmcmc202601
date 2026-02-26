#!/usr/bin/env python3
"""
Generate training data for GNN a_ij prediction (Project B, Issue #39).

Samples theta from prior, runs Hamilton ODE, extracts (phi stats, a_ij).
Uses same pipeline as DeepONet for consistency.

Output: data/train_gnn_N{n}.npz
  theta, phi, phi_mean, phi_std, phi_final, a_ij_active, t
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

from improved_5species_jit import BiofilmNewtonSolver5S

ACTIVE_EDGE_THETA_IDX = [1, 10, 11, 18, 19]


def load_prior_bounds(condition="Dysbiotic_HOBIC"):
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


def sample_theta(bounds, rng, theta_map=None, map_std_frac=0.1):
    theta = np.zeros(20)
    for i in range(20):
        lo, hi = bounds[i]
        if abs(hi - lo) < 1e-12:
            theta[i] = lo
        elif theta_map is not None:
            sigma = map_std_frac * (hi - lo)
            theta[i] = np.clip(rng.normal(theta_map[i], sigma), lo, hi)
        else:
            theta[i] = rng.uniform(lo, hi)
    return theta


def run_single(solver, theta):
    try:
        t_arr, g_arr = solver.run_deterministic(theta)
        phi = g_arr[:, :5]
        if np.any(~np.isfinite(phi)) or np.any(phi < -0.5) or np.any(phi > 1.5):
            return None
        return t_arr, phi
    except Exception:
        return None


def load_theta_map(condition):
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
            arr = data["theta_full"] if isinstance(data, dict) else data
            return np.array(arr, dtype=np.float64)
    return None


def generate_dataset(
    n_samples=10000,
    condition="Dysbiotic_HOBIC",
    seed=42,
    maxtimestep=500,
    dt=1e-5,
    n_time_out=100,
    map_frac=0.3,
):
    rng = np.random.default_rng(seed)
    bounds = load_prior_bounds(condition)
    theta_map = load_theta_map(condition)
    solver = BiofilmNewtonSolver5S(maxtimestep=maxtimestep, dt=dt)

    theta_list, phi_list, t_list = [], [], []
    n_map = int(n_samples * map_frac)
    n_prior = n_samples - n_map

    for _ in range(n_prior):
        th = sample_theta(bounds, rng, theta_map=None)
        result = run_single(solver, th)
        if result is not None:
            t_arr, phi = result
            t_out = np.linspace(t_arr[0], t_arr[-1], n_time_out)
            phi_out = np.array([np.interp(t_out, t_arr, phi[:, i]) for i in range(5)]).T
            theta_list.append(th)
            phi_list.append(phi_out)
            t_list.append(t_out)

    for _ in range(n_map):
        if theta_map is None:
            break
        th = sample_theta(bounds, rng, theta_map=theta_map, map_std_frac=0.15)
        result = run_single(solver, th)
        if result is not None:
            t_arr, phi = result
            t_out = np.linspace(t_arr[0], t_arr[-1], n_time_out)
            phi_out = np.array([np.interp(t_out, t_arr, phi[:, i]) for i in range(5)]).T
            theta_list.append(th)
            phi_list.append(phi_out)
            t_list.append(t_out)

    theta = np.array(theta_list)
    phi = np.array(phi_list)
    t = t_list[0] if t_list else np.zeros(n_time_out)

    return {
        "theta": theta,
        "phi": phi,
        "t": t,
        "phi_mean": np.mean(phi, axis=1),
        "phi_std": np.std(phi, axis=1),
        "phi_final": phi[:, -1, :],
        "a_ij_active": theta[:, ACTIVE_EDGE_THETA_IDX],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--condition", type=str, default="Dysbiotic_HOBIC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = args.output or "data/train_gnn_N{}.npz".format(args.n_samples)
    out_path = Path(__file__).parent / out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating {} samples for {}...".format(args.n_samples, args.condition))
    data = generate_dataset(n_samples=args.n_samples, condition=args.condition, seed=args.seed)
    np.savez_compressed(out_path, **data)
    print("Saved to {}".format(out_path))
    print("  theta: {}".format(data["theta"].shape))
    print("  a_ij_active: {}".format(data["a_ij_active"].shape))


if __name__ == "__main__":
    main()
