#!/usr/bin/env python3
"""
run_condition_sweep.py
=======================
B1: Automated batch execution for parameter sweeps.

Sweeps over combinations of:
  - K_hill: Hill gate saturation constant
  - n_hill: Hill cooperativity
  - lambda_pg: P.gingivalis likelihood weight
  - lambda_late: Late-timepoint likelihood weight
  - Conditions: dh_baseline, commensal_static, etc.

For each (condition, param_set):
  1. Run 2D Hamilton+nutrient with MAP theta
  2. Compute DI field, alpha_Monod, eigenstrain
  3. Export Abaqus CSV + macro eigenstrain CSV
  4. Optionally generate INP and submit Abaqus job

Results are organized as:
  _sweep_results/{sweep_id}/{cond}_{param_tag}/
    di_field.npy, phi_snaps.npy, c_snaps.npy, abaqus_field_2d.csv

Usage
-----
  # Default sweep (K_hill x lambda_pg):
  python run_condition_sweep.py

  # Custom sweep:
  python run_condition_sweep.py \\
      --k-hill 0.01 0.05 0.10 \\
      --lambda-pg 1.0 2.0 5.0 \\
      --conditions dh_baseline commensal_static

  # Quick test:
  python run_condition_sweep.py --quick

  # Resume incomplete sweep:
  python run_condition_sweep.py --resume _sweep_results/sweep_YYYYMMDD_HHMMSS
"""
import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
_SWEEP_BASE = _HERE / "_sweep_results"

sys.path.insert(0, str(_HERE))

CONDITION_RUNS = {
    "dh_baseline": _RUNS_ROOT / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "commensal_static",
    "commensal_hobic": _RUNS_ROOT / "commensal_hobic",
    "dysbiotic_static": _RUNS_ROOT / "dysbiotic_static",
}

_PARAM_KEYS = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]


def load_theta(path):
    with open(path) as f:
        d = json.load(f)
    if "theta_full" in d:
        return np.array(d["theta_full"], dtype=np.float64)
    elif "theta_sub" in d:
        return np.array(d["theta_sub"], dtype=np.float64)
    else:
        return np.array([d[k] for k in _PARAM_KEYS], dtype=np.float64)


def compute_di(phi_final):
    """Compute DI from final phi snapshot (5, Nx, Ny)."""
    phi_t = phi_final.transpose(1, 2, 0)  # (Nx, Ny, 5)
    phi_sum = phi_t.sum(axis=-1)
    phi_sum_safe = np.where(phi_sum > 0, phi_sum, 1.0)
    p = phi_t / phi_sum_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=-1)
    return 1.0 - H / np.log(5.0)


def compute_alpha_monod(phi_snaps, c_snaps, t_snaps, k_monod=1.0, k_alpha=0.01):
    """Compute Monod growth activity integral: k_alpha * integral(phi_total * c/(k+c) dt)."""
    n_snap = len(t_snaps)
    if n_snap < 2:
        return np.zeros_like(c_snaps[0])

    # Trapezoidal integration
    alpha = np.zeros_like(c_snaps[0])
    for i in range(n_snap - 1):
        dt = t_snaps[i + 1] - t_snaps[i]
        phi_total_i = phi_snaps[i].sum(axis=0)
        phi_total_ip1 = phi_snaps[i + 1].sum(axis=0)
        monod_i = c_snaps[i] / (k_monod + c_snaps[i])
        monod_ip1 = c_snaps[i + 1] / (k_monod + c_snaps[i + 1])
        integrand_i = phi_total_i * monod_i
        integrand_ip1 = phi_total_ip1 * monod_ip1
        alpha += 0.5 * dt * (integrand_i + integrand_ip1)

    return k_alpha * alpha


def run_single(theta, condition, K_hill, n_hill, cfg_base, out_dir):
    """Run one simulation with specific parameters. Returns dict of results."""
    from JAXFEM.core_hamilton_2d_nutrient import run_simulation, Config2D

    cfg = Config2D(
        Nx=cfg_base["Nx"],
        Ny=cfg_base["Ny"],
        n_macro=cfg_base["n_macro"],
        n_react_sub=cfg_base["n_react_sub"],
        dt_h=cfg_base["dt_h"],
        save_every=cfg_base["save_every"],
        K_hill=K_hill,
        n_hill=n_hill,
    )

    result = run_simulation(theta, cfg)

    phi_snaps = np.array(result["phi_snaps"])
    c_snaps = np.array(result["c_snaps"])
    t_snaps = np.array(result["t_snaps"])

    di = compute_di(phi_snaps[-1])
    alpha_monod = compute_alpha_monod(phi_snaps, c_snaps, t_snaps)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "phi_snaps.npy", phi_snaps)
    np.save(out_dir / "c_snaps.npy", c_snaps)
    np.save(out_dir / "t_snaps.npy", t_snaps)
    np.save(out_dir / "di_field.npy", di)
    np.save(out_dir / "alpha_monod.npy", alpha_monod)

    # Export Abaqus CSV
    phi_final = phi_snaps[-1]
    c_final = c_snaps[-1]
    x = np.linspace(0, cfg.Lx, cfg.Nx)
    y = np.linspace(0, cfg.Ly, cfg.Ny)
    csv_path = out_dir / "abaqus_field_2d.csv"
    with open(csv_path, "w") as f:
        f.write("x,y,phi_pg,di,phi_total,c,alpha_monod\n")
        for ix in range(cfg.Nx):
            for iy in range(cfg.Ny):
                f.write(
                    "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n"
                    % (
                        x[ix],
                        y[iy],
                        float(phi_final[4, ix, iy]),
                        float(di[ix, iy]),
                        float(phi_final.sum(axis=0)[ix, iy]),
                        float(c_final[ix, iy]),
                        float(alpha_monod[ix, iy]),
                    )
                )

    return {
        "di_mean": float(np.mean(di)),
        "di_max": float(np.max(di)),
        "phi_pg_mean": float(np.mean(phi_final[4])),
        "phi_pg_max": float(np.max(phi_final[4])),
        "alpha_monod_mean": float(np.mean(alpha_monod)),
        "alpha_monod_max": float(np.max(alpha_monod)),
        "c_min": float(np.min(c_final)),
    }


def run_sweep(args):
    """Execute the full parameter sweep."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.resume) if args.resume else _SWEEP_BASE / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Build parameter grid
    k_hills = args.k_hill
    n_hills = args.n_hill
    conditions = args.conditions

    grid = list(itertools.product(conditions, k_hills, n_hills))
    n_total = len(grid)

    cfg_base = {
        "Nx": args.nx,
        "Ny": args.ny,
        "n_macro": args.n_macro,
        "n_react_sub": args.n_react_sub,
        "dt_h": args.dt_h,
        "save_every": args.save_every,
    }

    # Save sweep config
    config = {
        "timestamp": timestamp,
        "conditions": conditions,
        "k_hills": k_hills,
        "n_hills": n_hills,
        "n_total": n_total,
        "cfg_base": cfg_base,
    }
    with (sweep_dir / "sweep_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"{'='*60}")
    print(f"Condition Sweep: {n_total} configurations")
    print(f"  Conditions: {conditions}")
    print(f"  K_hill: {k_hills}")
    print(f"  n_hill: {n_hills}")
    print(f"  Grid: {cfg_base['Nx']}x{cfg_base['Ny']}, {cfg_base['n_macro']} macro steps")
    print(f"  Output: {sweep_dir}")
    print(f"{'='*60}")

    results = []
    t_start = time.perf_counter()

    for i, (cond, K_h, n_h) in enumerate(grid):
        param_tag = f"K{K_h:.3f}_n{n_h:.1f}"
        run_tag = f"{cond}_{param_tag}"
        run_dir = sweep_dir / run_tag

        # Check if already completed (resume support)
        done_flag = run_dir / "done.flag"
        if done_flag.exists():
            print(f"\n[{i+1}/{n_total}] {run_tag} — SKIP (already done)")
            # Load existing results
            meta_path = run_dir / "run_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    results.append(json.load(f))
            continue

        print(f"\n[{i+1}/{n_total}] {run_tag}")

        # Load theta
        run_path = CONDITION_RUNS.get(cond)
        if run_path is None:
            print(f"  [SKIP] Unknown condition: {cond}")
            continue
        theta_path = run_path / "theta_MAP.json"
        if not theta_path.exists():
            print(f"  [SKIP] theta_MAP.json not found: {theta_path}")
            continue
        theta = load_theta(str(theta_path))

        t0 = time.perf_counter()
        try:
            res = run_single(theta, cond, K_h, n_h, cfg_base, run_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")
            res = {"error": str(e)}

        dt = time.perf_counter() - t0
        meta = {
            "condition": cond,
            "K_hill": K_h,
            "n_hill": n_h,
            "tag": run_tag,
            "timing_s": round(dt, 1),
            **res,
        }
        with (run_dir / "run_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        done_flag.touch()
        results.append(meta)

        print(f"  DI: mean={res.get('di_mean',0):.4f}, max={res.get('di_max',0):.4f}")
        print(f"  Pg: max={res.get('phi_pg_max',0):.4f}")
        print(f"  Time: {dt:.1f}s")

    total_time = time.perf_counter() - t_start

    # Save sweep summary
    summary = {
        "sweep_dir": str(sweep_dir),
        "n_total": n_total,
        "n_completed": len([r for r in results if "error" not in r]),
        "total_time_s": round(total_time, 1),
        "results": results,
    }
    with (sweep_dir / "sweep_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Generate sweep comparison figure
    _plot_sweep(results, sweep_dir, conditions, k_hills, n_hills)

    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(results)}/{n_total} runs")
    print(f"Total time: {total_time:.0f}s")
    print(f"Output: {sweep_dir}")
    print(f"{'='*60}")


def _plot_sweep(results, sweep_dir, conditions, k_hills, n_hills):
    """Generate sweep result visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results if "error" not in r and "di_mean" in r]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Organize by condition
    cond_colors = {
        "dh_baseline": "#d62728",
        "commensal_static": "#2ca02c",
        "commensal_hobic": "#17becf",
        "dysbiotic_static": "#ff7f0e",
    }

    # Panel 1: DI vs K_hill
    ax = axes[0, 0]
    for cond in conditions:
        cond_runs = [r for r in valid if r["condition"] == cond]
        if cond_runs:
            ks = [r["K_hill"] for r in cond_runs]
            dis = [r["di_mean"] for r in cond_runs]
            ax.plot(ks, dis, "o-", color=cond_colors.get(cond, "gray"), label=cond, markersize=6)
    ax.set_xlabel("$K_{hill}$", fontsize=12)
    ax.set_ylabel("DI mean", fontsize=12)
    ax.set_title("(a) DI vs Hill Constant", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 2: Pg fraction vs K_hill
    ax = axes[0, 1]
    for cond in conditions:
        cond_runs = [r for r in valid if r["condition"] == cond]
        if cond_runs:
            ks = [r["K_hill"] for r in cond_runs]
            pgs = [r["phi_pg_max"] for r in cond_runs]
            ax.plot(ks, pgs, "s-", color=cond_colors.get(cond, "gray"), label=cond, markersize=6)
    ax.set_xlabel("$K_{hill}$", fontsize=12)
    ax.set_ylabel("$\\phi_{Pg}$ max", fontsize=12)
    ax.set_title("(b) P.gingivalis vs Hill Constant", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 3: Alpha Monod vs K_hill
    ax = axes[1, 0]
    for cond in conditions:
        cond_runs = [r for r in valid if r["condition"] == cond]
        if cond_runs:
            ks = [r["K_hill"] for r in cond_runs]
            alphas = [r["alpha_monod_max"] for r in cond_runs]
            ax.plot(ks, alphas, "^-", color=cond_colors.get(cond, "gray"), label=cond, markersize=6)
    ax.set_xlabel("$K_{hill}$", fontsize=12)
    ax.set_ylabel(r"$\alpha_{Monod}$ max", fontsize=12)
    ax.set_title("(c) Growth Activity vs Hill Constant", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary heatmap (DI mean for dh_baseline across K_hill x n_hill)
    ax = axes[1, 1]
    if len(k_hills) > 1 and len(n_hills) > 1:
        dh_runs = [r for r in valid if r["condition"] == "dh_baseline"]
        if dh_runs:
            # Build 2D grid
            di_grid = np.full((len(n_hills), len(k_hills)), np.nan)
            for r in dh_runs:
                ki = k_hills.index(r["K_hill"]) if r["K_hill"] in k_hills else -1
                ni = n_hills.index(r["n_hill"]) if r["n_hill"] in n_hills else -1
                if ki >= 0 and ni >= 0:
                    di_grid[ni, ki] = r["di_mean"]
            im = ax.imshow(di_grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
            ax.set_xticks(range(len(k_hills)))
            ax.set_xticklabels([f"{k:.3f}" for k in k_hills])
            ax.set_yticks(range(len(n_hills)))
            ax.set_yticklabels([f"{n:.1f}" for n in n_hills])
            ax.set_xlabel("$K_{hill}$", fontsize=12)
            ax.set_ylabel("$n_{hill}$", fontsize=12)
            plt.colorbar(im, ax=ax, label="DI mean")
            ax.set_title("(d) DI Heatmap (DH baseline)", fontsize=13)
    else:
        ax.text(
            0.5,
            0.5,
            "Need >1 K_hill and n_hill\nfor heatmap",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    fig.tight_layout()
    out = sweep_dir / "sweep_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sweep figure: {out}")


def main():
    ap = argparse.ArgumentParser(description="Automated parameter sweep")
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"],
    )
    ap.add_argument(
        "--k-hill", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    )
    ap.add_argument("--n-hill", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", default=None, help="Resume sweep from directory")
    # Simulation config
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--n-macro", type=int, default=500)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=50)

    args = ap.parse_args()

    if args.quick:
        args.nx = 10
        args.ny = 10
        args.n_macro = 10
        args.n_react_sub = 5
        args.save_every = 5
        args.k_hill = [0.01, 0.05, 0.10]
        args.n_hill = [2.0, 4.0]
        args.conditions = ["dh_baseline"]

    run_sweep(args)


if __name__ == "__main__":
    main()
