#!/usr/bin/env python3
"""
compute_alpha_eigenstrain.py  –  [P2] alpha_final 0D 計算ヘルパー
================================================================

TMCMC MAP パラメータから生成した ODE トラジェクトリを積分し、
Klempt 2024 の α 膨張固有ひずみ (α̇ = k_α φ) の 0D 近似値を求める。

得られた alpha_final を biofilm_conformal_tet.py の --growth-eigenstrain に渡す。

物理モデル
----------
  α̇ = k_α * φ(t)
  → alpha_final = k_α * ∫₀^{t_end} φ(t) dt   [0D 近似: φ は空間平均]

ここで φ(t) = Σᵢ phiᵢ(t)  (5 菌種の volume fraction 合計)
t は TMCMC ODE の無次元時刻 T* (dt=0.01, 最大 2500 ステップ)

使い方
------
  # 4 条件すべての alpha_final を表示:
  python3 compute_alpha_eigenstrain.py \\
      --runs-base  ../data_5species/_runs \\
      --k-alpha    0.05

  # 特定の run ディレクトリを指定:
  python3 compute_alpha_eigenstrain.py \\
      --run-dir ../data_5species/_runs/Commensal_Static_20260204_062733 \\
      --k-alpha 0.05

  # 結果を biofilm_conformal_tet.py に渡す:
  python3 biofilm_conformal_tet.py \\
      --stl external_tooth_models/.../P1_Tooth_23.stl \\
      --di-csv _di_credible/commensal_static/p50_field.csv \\
      --out p23_commensal_eigenstrain.inp \\
      --mode biofilm \\
      --growth-eigenstrain 0.312   # ← compute_alpha_eigenstrain.py の出力値

パラメータガイド
----------------
  k_alpha:  Klempt 2024 では文献値なし; 合理的な範囲は 0.01~0.1 [T*^-1]
            eps_growth = alpha_final / 3.0 が O(0.01~0.3) になる値を推奨
  phi_avg:  通常 0.3~0.7 (biofilm 中の総菌体積分率)
  t_end:    ODE が収束するまでの T* 値 (通常 15~25)

出力例
------
  Run: Commensal_Static_20260204_062733
    phi_total mean  : 0.4521
    phi_total sum × dt: 11.23 [T*]
    k_alpha          : 0.0500
    alpha_final      : 0.5615
    eps_growth       : 0.1872  (= alpha/3)
    sigma_0 / E      : -18.7%  (compressive prestress ratio)
  --growth-eigenstrain 0.5615
"""

from __future__ import print_function, division
import sys
import os
import json
import argparse
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in [_PROJECT_ROOT,
           os.path.join(_PROJECT_ROOT, "tmcmc", "program2602"),
           os.path.join(_PROJECT_ROOT, "data_5species")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from tmcmc.program2602.improved_5species_jit import BiofilmNewtonSolver5S
except ImportError:
    try:
        from improved_5species_jit import BiofilmNewtonSolver5S
    except ImportError:
        BiofilmNewtonSolver5S = None

# ── Known completed run directories per condition ─────────────────────────────
_CONDITION_MAP = {
    "commensal_static" : "Commensal_Static_20260204_062733",
    "commensal_hobic"  : "Commensal_HOBIC_20260205_003113",
    "dysbiotic_static" : "Dysbiotic_Static",          # fill in if present
    "dh_baseline"      : "Dysbiotic_HOBIC_20260205_013530",
}


def find_run_dirs(runs_base, condition=None):
    """Return list of (label, run_dir) tuples for available TMCMC runs."""
    if not os.path.isdir(runs_base):
        return []
    results = []
    for name in sorted(os.listdir(runs_base)):
        d = os.path.join(runs_base, name)
        if not os.path.isdir(d):
            continue
        if not os.path.isfile(os.path.join(d, "theta_MAP.json")):
            continue
        if condition is not None and condition.lower() not in name.lower():
            continue
        results.append((name, d))
    return results


def load_theta(run_dir):
    """Load theta_MAP from a TMCMC run directory."""
    path = os.path.join(run_dir, "theta_MAP.json")
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return np.array(data, dtype=np.float64)
    elif isinstance(data, dict):
        theta = data.get("theta_full", data.get("theta_sub", None))
        if theta is None:
            raise ValueError("theta_MAP.json has no 'theta_full' or 'theta_sub' key")
        return np.array(theta, dtype=np.float64)
    else:
        raise ValueError("Unexpected theta_MAP.json format")


def compute_alpha_final(run_dir, k_alpha=0.05, dt=0.01, maxtimestep=2500, verbose=True):
    """
    Run ODE from MAP parameters and compute alpha_final.

    Parameters
    ----------
    run_dir   : str  – TMCMC run directory (must contain theta_MAP.json)
    k_alpha   : float – growth-to-eigenstrain coupling [T*^-1]
    dt        : float – ODE timestep [T*]
    maxtimestep: int  – maximum ODE timesteps

    Returns
    -------
    alpha_final : float – k_alpha * integral(phi_total, 0, t_end)
    eps_growth  : float – alpha_final / 3.0
    t_arr       : np.ndarray – ODE time array
    phi_total   : np.ndarray – total biofilm fraction at each timestep
    """
    if BiofilmNewtonSolver5S is None:
        raise ImportError(
            "BiofilmNewtonSolver5S not found. "
            "Ensure tmcmc/program2602/improved_5species_jit.py is importable."
        )

    theta = load_theta(run_dir)
    if verbose:
        print("  theta (20 params): %s ..." % str(theta[:5].round(4)))

    solver = BiofilmNewtonSolver5S(
        dt=dt, maxtimestep=maxtimestep,
        phi_init=0.01, use_numba=True
    )
    t_arr, g_arr = solver.solve(theta)

    # Total biofilm volume fraction: sum of phi_1..phi_5 (indices 0..4)
    phi_total = np.sum(g_arr[:, 0:5], axis=1)

    # 0D integral: ∫ phi_total dt  using trapezoidal rule
    integral_phi = np.trapz(phi_total, t_arr)
    alpha_final  = k_alpha * integral_phi
    eps_growth   = alpha_final / 3.0

    if verbose:
        print("  ODE: %d steps  t_end=%.2f T*" % (len(t_arr), t_arr[-1]))
        print("  phi_total: mean=%.4f  max=%.4f  sum*dt=%.4f T*" % (
            phi_total.mean(), phi_total.max(), integral_phi))
        print("  k_alpha   : %.4f" % k_alpha)
        print("  alpha_final: %.4f" % alpha_final)
        print("  eps_growth : %.4f  (= alpha/3, isotropic expansion strain)" % eps_growth)
        print("  sigma/E    : %.1f%%  (compressive prestress ratio)" % (-eps_growth * 100))

    return alpha_final, eps_growth, t_arr, phi_total


def print_eigenstrain_summary(label, alpha_final, eps_growth):
    """Print the ready-to-use flag for biofilm_conformal_tet.py."""
    print()
    print("  --> biofilm_conformal_tet.py flag:")
    print("      --growth-eigenstrain %.4f" % alpha_final)
    print("  --> Expected initial compressive stress: sigma_0 = -E * %.4g" % eps_growth)
    print("      For E_max=1000 Pa: sigma_0 = %.4g Pa" % (-1000 * eps_growth))
    print("      For E_min= 10 Pa: sigma_0 = %.4g Pa" % (-10   * eps_growth))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute alpha_final eigenstrain from TMCMC ODE trajectory (0D approximation)"
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-dir",    type=str,
                     help="Single TMCMC run directory (contains theta_MAP.json)")
    grp.add_argument("--runs-base",  type=str,
                     help="Base directory of all TMCMC runs; processes all completed runs")
    p.add_argument("--condition",    type=str, default=None,
                   help="Filter runs-base by condition name substring (e.g. commensal)")
    p.add_argument("--k-alpha",      type=float, default=0.05,
                   help="Growth-eigenstrain coupling k_alpha [T*^-1] (default 0.05)")
    p.add_argument("--dt",           type=float, default=0.01,
                   help="ODE timestep (default 0.01)")
    p.add_argument("--maxtimestep",  type=int,   default=2500,
                   help="Max ODE timesteps (default 2500)")
    p.add_argument("--plot",         action="store_true",
                   help="Save phi_total(t) plot alongside results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.run_dir:
        run_dirs = [(os.path.basename(args.run_dir), args.run_dir)]
    else:
        run_dirs = find_run_dirs(args.runs_base, condition=args.condition)
        if not run_dirs:
            print("No completed TMCMC runs found in %s" % args.runs_base)
            sys.exit(1)

    print("=" * 60)
    print("  compute_alpha_eigenstrain.py")
    print("  k_alpha = %.4f  dt=%.3f  max_steps=%d" % (
        args.k_alpha, args.dt, args.maxtimestep))
    print("=" * 60)

    summaries = []
    for label, run_dir in run_dirs:
        print("\n[Run] %s" % label)
        try:
            alpha_f, eps_g, t_arr, phi_tot = compute_alpha_final(
                run_dir, k_alpha=args.k_alpha,
                dt=args.dt, maxtimestep=args.maxtimestep,
                verbose=True
            )
            print_eigenstrain_summary(label, alpha_f, eps_g)
            summaries.append({
                "label": label,
                "alpha_final": float(alpha_f),
                "eps_growth":  float(eps_g),
                "phi_total_mean": float(phi_tot.mean()),
            })

            if args.plot:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(t_arr, phi_tot, lw=1.5, color="steelblue", label=r"$\phi_{total}(t)$")
                    ax.axhline(phi_tot.mean(), ls="--", color="gray",
                               label=r"$\bar\phi$ = %.3f" % phi_tot.mean())
                    ax.set_xlabel("T* (non-dimensional time)")
                    ax.set_ylabel(r"$\phi_{total}$ (biofilm volume fraction)")
                    ax.set_title(label + "\nalpha_final=%.4f, eps=%.4f" % (alpha_f, eps_g))
                    ax.legend()
                    ax.grid(alpha=0.3)
                    out_png = os.path.join(run_dir, "phi_total_alpha.png")
                    fig.savefig(out_png, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print("  Plot saved: %s" % out_png)
                except ImportError:
                    print("  (matplotlib not available, skipping plot)")

        except Exception as e:
            print("  ERROR: %s" % str(e))

    # Summary table
    if len(summaries) > 1:
        print()
        print("=" * 60)
        print("SUMMARY TABLE")
        print("  k_alpha = %.4f" % args.k_alpha)
        print("  %-45s  %8s  %8s" % ("Run", "alpha", "eps=a/3"))
        print("  " + "-" * 65)
        for s in summaries:
            print("  %-45s  %8.4f  %8.4f" % (s["label"][:45], s["alpha_final"], s["eps_growth"]))

    print()
    print("Done. Pass alpha_final as --growth-eigenstrain to biofilm_conformal_tet.py")


if __name__ == "__main__":
    main()
