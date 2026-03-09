#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_jax_ode_numerical.py — JAX ODE と NumPy ODE の数値一致検証

hamilton_ode_jax.simulate_0d と improved_5species_jit.BiofilmNewtonSolver5S が
同一 θ で同じ軌道を返すか検証する。

Usage:
    cd data_5species/main
    python verify_jax_ode_numerical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))

# NumPy solver (improved_5species_jit)
from improved_5species_jit import BiofilmNewtonSolver5S

# JAX solver
from hamilton_ode_jax import simulate_0d

# Match JAX defaults
DT = 1e-4
N_STEPS = 2500
C_CONST = 25.0
ALPHA_CONST = 100.0
K_HILL = 0.05
N_HILL = 2.0
PHI_INIT = np.full(5, 0.2)


def run_numpy(theta: np.ndarray) -> np.ndarray:
    """Run NumPy solver, return phi (n_steps+1, 5)."""
    solver = BiofilmNewtonSolver5S(
        dt=DT,
        maxtimestep=N_STEPS,
        c_const=C_CONST,
        alpha_const=ALPHA_CONST,
        phi_init=PHI_INIT,
        K_hill=K_HILL,
        n_hill=N_HILL,
    )
    t_arr, g_arr = solver.solve(theta)
    return g_arr[:, 0:5]


def run_jax(theta: np.ndarray) -> np.ndarray:
    """Run JAX solver, return phi (n_steps+1, 5)."""
    import jax.numpy as jnp

    phi_traj = simulate_0d(
        jnp.array(theta, dtype=jnp.float64),
        n_steps=N_STEPS,
        dt=DT,
        phi_init=PHI_INIT,
        K_hill=K_HILL,
        n_hill=N_HILL,
        c_const=C_CONST,
        alpha_const=ALPHA_CONST,
    )
    return np.array(phi_traj)


def main():
    print("=" * 60)
    print("JAX ODE vs NumPy ODE 数値検証")
    print("=" * 60)

    # JAX version
    import jax

    print(f"JAX version: {jax.__version__}")
    print(f"Parameters: dt={DT}, n_steps={N_STEPS}, c={C_CONST}, K_hill={K_HILL}, n_hill={N_HILL}")
    print()

    # Test theta: use prior bounds midpoint or known-good values
    theta = np.array(
        [
            0.5,
            0.3,
            0.5,
            0.1,
            0.1,  # a11, a12, a22, b1, b2
            0.5,
            0.3,
            0.5,
            0.1,
            0.1,  # a33, a34, a44, b3, b4
            0.1,
            0.1,
            0.1,
            0.1,  # cross a13, a14, a23, a24
            0.5,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,  # a55, b5, a15, a25, a35, a45
        ],
        dtype=np.float64,
    )
    theta = np.asarray(theta[:20], dtype=np.float64)

    phi_np = run_numpy(theta)
    phi_jax = run_jax(theta)

    # Compare
    diff = np.abs(phi_np - phi_jax)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = diff / (np.abs(phi_np) + 1e-12)
    max_rel = np.max(rel_diff)

    print("結果:")
    print(f"  phi_np  shape: {phi_np.shape}")
    print(f"  phi_jax shape: {phi_jax.shape}")
    print(f"  Max |diff|:    {max_diff:.2e}")
    print(f"  Mean |diff|:   {mean_diff:.2e}")
    print(f"  Max rel diff:  {max_rel:.2e}")

    # Sample at observation-like indices (e.g. 113, 339, 679, 1131, 1696, 2375)
    idx_sample = np.array([0, 113, 339, 679, 1131, 1696, 2375, N_STEPS])
    idx_sample = np.clip(idx_sample, 0, min(len(phi_np), len(phi_jax)) - 1)

    print()
    print("観測相当インデックスでの比較:")
    print("  idx    | phi_np (sum) | phi_jax (sum) | |diff|")
    for i in idx_sample:
        s_np = phi_np[i].sum()
        s_jax = phi_jax[i].sum()
        d = np.abs(phi_np[i] - phi_jax[i]).max()
        print(f"  {i:5d}  | {s_np:.6f}     | {s_jax:.6f}      | {d:.2e}")

    # Pass/fail: 観測インデックスでの差が 1e-4 程度、全体で 1e-2 以下なら実用上問題なし
    tol_abs = 5e-3
    tol_rel = 2e-2
    ok = max_diff < tol_abs and max_rel < tol_rel
    print()
    if ok:
        print("PASS: JAX ODE と NumPy ODE は数値的に一致")
    else:
        print(
            f"WARN: 差が許容範囲外 (tol_abs={tol_abs}, tol_rel={tol_rel}). "
            "パラメータ・実装の差異を確認してください。"
        )
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
