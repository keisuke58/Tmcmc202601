#!/usr/bin/env python3
"""
run_posterior_pipeline.py
=========================
End-to-end posterior FEM pipeline.

For each condition (dh_baseline, commensal_static):
  1. Run 3D posterior FEM (N samples, with per-sample save + resume).
  2. Plot phibar posterior bands.
  3. Plot Pg penetration-depth posterior band.

Then produce a side-by-side comparison of Pg depth for both conditions.
Finally (optionally) run posterior_sensitivity.py scatter plots.

Usage
-----
  # Full run (20 samples per condition, resume-safe):
  python run_posterior_pipeline.py

  # Quick test (3 samples):
  python run_posterior_pipeline.py --n-samples 3 --n-macro 20

  # Skip FEM computation (re-plot from existing outputs):
  python run_posterior_pipeline.py --plot-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

# ── canonical run-directory mapping ──────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_DATA_ROOT = _HERE.parent / "data_5species" / "_runs"

CONDITION_RUNS: dict[str, Path] = {
    "dh_baseline":      _DATA_ROOT / "sweep_pg_20260217_081459" / "dh_baseline",
    "commensal_static": _DATA_ROOT / "Commensal_Static_20260208_002100",
}

# ── output layout ─────────────────────────────────────────────────────────────
_OUT_3D   = _HERE / "_results_3d"
_OUT_PLOT = _HERE / "_posterior_plots"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    ret = subprocess.run([str(c) for c in cmd])
    if ret.returncode != 0:
        print(f"[error] command exited with code {ret.returncode}")
        sys.exit(ret.returncode)


def run_fem(
    cond: str,
    run_dir: Path,
    n_samples: int,
    n_macro: int,
    n_react_sub: int,
    dt_h: float,
    nx: int,
    ny: int,
    nz: int,
    solver: str,
    seed: int,
) -> Path:
    """Run posterior FEM for one condition; return the posterior output dir."""
    out_3d   = _OUT_3D / f"{cond}_posterior"
    post_dir = out_3d / "posterior"

    _run([
        sys.executable, _HERE / "fem_3d_extension.py",
        "--posterior-only",
        "--tmcmc-run-dir", run_dir,
        "--posterior-n-samples", n_samples,
        "--posterior-seed", seed,
        "--condition", cond,
        "--nx", nx, "--ny", ny, "--nz", nz,
        "--n-macro", n_macro,
        "--n-react-sub", n_react_sub,
        "--dt-h", dt_h,
        "--solver", solver,
        "--out-dir", out_3d,
    ])
    return post_dir


def plot_single(cond: str, post_dir: Path) -> None:
    """Plot phibar + depth for one condition."""
    _run([
        sys.executable, _HERE / "plot_posterior_fem.py",
        "single", post_dir,
        "--condition", cond,
        "--out-dir", _OUT_PLOT,
    ])


def plot_comparison(post_dirs: dict[str, Path]) -> None:
    """Overlay Pg depth posteriors for all conditions."""
    cmd = [
        sys.executable, _HERE / "plot_posterior_fem.py",
        "compare",
        "--dirs",  *[post_dirs[c] for c in post_dirs],
        "--conds", *list(post_dirs.keys()),
        "--out",   _OUT_PLOT / "comparison_depth_pg.png",
    ]
    _run(cmd)


def run_sensitivity(cond: str, post_dir: Path, n_samples: int) -> None:
    """θ vs Pg depth scatter plots via posterior_sensitivity.py."""
    out_sens = _HERE / "_posterior_sensitivity" / cond
    _run([
        sys.executable, _HERE / "posterior_sensitivity.py",
        "--run-dir",   CONDITION_RUNS[cond],
        "--fem-base",  post_dir,
        "--n-samples", n_samples,
        "--out-dir",   out_sens,
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end posterior FEM pipeline")
    ap.add_argument("--n-samples",    type=int,   default=20,       help="Posterior samples per condition")
    ap.add_argument("--n-macro",      type=int,   default=100,      help="FEM macro steps")
    ap.add_argument("--n-react-sub",  type=int,   default=50,       help="Reaction sub-steps per macro step")
    ap.add_argument("--dt-h",         type=float, default=1e-5,     help="Reaction sub-step size dt_h")
    ap.add_argument("--nx",           type=int,   default=15)
    ap.add_argument("--ny",           type=int,   default=15)
    ap.add_argument("--nz",           type=int,   default=15)
    ap.add_argument("--solver",       default="superlu", choices=["superlu", "cg"])
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--conditions",   nargs="+",  default=list(CONDITION_RUNS.keys()),
                    help="Which conditions to run (default: all)")
    ap.add_argument("--plot-only",    action="store_true",
                    help="Skip FEM computation; re-plot from existing outputs")
    ap.add_argument("--sensitivity",  action="store_true",
                    help="Also run posterior_sensitivity.py scatter plots")
    args = ap.parse_args()

    post_dirs: dict[str, Path] = {}

    for cond in args.conditions:
        if cond not in CONDITION_RUNS:
            print(f"[warn] unknown condition {cond!r}, skipping")
            continue
        run_dir  = CONDITION_RUNS[cond]
        post_dir = _OUT_3D / f"{cond}_posterior" / "posterior"

        if not args.plot_only:
            post_dir = run_fem(
                cond        = cond,
                run_dir     = run_dir,
                n_samples   = args.n_samples,
                n_macro     = args.n_macro,
                n_react_sub = args.n_react_sub,
                dt_h        = args.dt_h,
                nx=args.nx, ny=args.ny, nz=args.nz,
                solver      = args.solver,
                seed        = args.seed,
            )

        post_dirs[cond] = post_dir
        plot_single(cond, post_dir)

        if args.sensitivity:
            run_sensitivity(cond, post_dir, args.n_samples)

    if len(post_dirs) > 1:
        plot_comparison(post_dirs)

    print("\n[pipeline complete]")
    print(f"  Plots  : {_OUT_PLOT}")
    print(f"  Data   : {_OUT_3D}")


if __name__ == "__main__":
    main()
