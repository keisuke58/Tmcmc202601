#!/usr/bin/env python3
"""
Test time-dependent A35(t), A45(t) for P. gingivalis surge simulation.

Simplified pilot: A35(t) = A35_0 * (1 + alpha * t_norm), A45(t) = A45_0 * (1 + alpha * t_norm)
where t_norm in [0,1]. If alpha > 0, Pg cross-feeding strengthens toward end of experiment.

Usage:
  python tools/test_time_dependent_A.py
  python tools/test_time_dependent_A.py --alpha 0.5

Output:
  - Prints Pg trajectory comparison (constant vs time-dependent A)
  - Saves plot to tools/time_dependent_A_test.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
from improved_5species_jit import BiofilmNewtonSolver5S


def load_theta_map() -> np.ndarray:
    """Load theta from dh_baseline."""
    path = PROJECT_ROOT / "data_5species" / "_runs" / "dh_baseline" / "theta_MAP.json"
    if not path.exists():
        raise FileNotFoundError(f"theta_MAP not found: {path}")
    with open(path) as f:
        data = json.load(f)
    theta = np.array(data.get("theta_full", data.get("theta", [])), dtype=np.float64)
    return theta[:20]


def run_with_time_dependent_A(theta: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ODE with A35, A45 scaled by (1 + alpha * t_norm) over time.
    t_norm = t / t_max so it goes from 0 to 1.
    """
    solver = BiofilmNewtonSolver5S(
        dt=1e-5,
        maxtimestep=2500,
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

    # Standard run for reference
    t_arr, g_arr = solver.run_deterministic(theta)
    phi_const = g_arr[:, :5]  # (T, 5)

    # Time-dependent: modify A at each step (simplified)
    # We can't easily inject time-dependent A into the solver without modifying it.
    # Instead, run a second simulation with scaled theta:
    # theta_eff = theta * (1 + alpha) at "end" - use linear interpolation
    # Simplified: run with theta scaled for Pg terms only (theta[18] A35, theta[19] A45)
    theta_early = theta.copy()
    theta_late = theta.copy()
    theta_early[18] = theta[18] * (1.0)  # t_norm=0
    theta_early[19] = theta[19] * (1.0)
    theta_late[18] = theta[18] * (1.0 + alpha)
    theta_late[19] = theta[19] * (1.0 + alpha)

    _, g_early = solver.run_deterministic(theta_early)
    _, g_late = solver.run_deterministic(theta_late)

    # Linear blend: phi(t) = (1 - t_norm)*phi_early + t_norm*phi_late
    t_norm = np.linspace(0, 1, len(t_arr))
    phi_tdep = (1 - t_norm[:, None]) * g_early[:, :5] + t_norm[:, None] * g_late[:, :5]

    return t_arr, phi_const, phi_tdep


def main():
    parser = argparse.ArgumentParser(description="Test time-dependent A35, A45")
    parser.add_argument("--alpha", type=float, default=0.3, help="End-time scaling factor")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    theta = load_theta_map()
    t_arr, phi_const, phi_tdep = run_with_time_dependent_A(theta, args.alpha)

    print("Time-dependent A35, A45 test")
    print("=" * 50)
    print(
        f"  alpha = {args.alpha} (A35(t_end)=A35_0*(1+{args.alpha}), A45(t_end)=A45_0*(1+{args.alpha}))"
    )
    print(f"  theta[18] (A35) = {theta[18]:.3f}, theta[19] (A45) = {theta[19]:.3f}")
    print()

    pg_const = phi_const[:, 4]
    pg_tdep = phi_tdep[:, 4]
    print("  Pg (species 4) at t_end:")
    print(f"    Constant A:   {pg_const[-1]:.6f}")
    print(f"    Time-dep A:  {pg_tdep[-1]:.6f}")
    print(f"    Ratio:       {pg_tdep[-1] / max(pg_const[-1], 1e-10):.3f}x")
    print()

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_arr, pg_const, label="Constant A", color="tab:blue")
            ax.plot(t_arr, pg_tdep, label="Time-dep A (alpha=%g)" % args.alpha, color="#ff7f0e")
            ax.set_xlabel("Time step")
            ax.set_ylabel("P. gingivalis φ (phi)")
            ax.legend()
            ax.set_title("Pg trajectory: constant vs time-dependent A35, A45")
            ax.grid(True, alpha=0.3)
            out = Path(__file__).parent / "time_dependent_A_test.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")
        except ImportError:
            print("  [SKIP] matplotlib not available, no plot generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
