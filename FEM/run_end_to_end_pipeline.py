#!/usr/bin/env python3
"""
run_end_to_end_pipeline.py
===========================
End-to-end biofilm FEM pipeline:

    TMCMC MAP theta -> 2D Hamilton+Nutrient -> Eigenstrain CSV -> Abaqus INP

Chains:
  1. Load theta_MAP from TMCMC run directory
  2. Run 2D Hamilton+Nutrient PDE coupling (JAX Lie splitting)
  3. Compute DI field, alpha_Monod, eigenstrain
  4. Export Abaqus CSV + macro eigenstrain CSV
  5. Generate two-layer Tie model INP (tooth S3 + biofilm C3D4)
  6. Generate visualization figures

Usage
-----
  # Full pipeline for dysbiotic HOBIC (default):
  python run_end_to_end_pipeline.py

  # Specify condition and output tag:
  python run_end_to_end_pipeline.py --condition dh_baseline --tag v1

  # Skip INP generation (no STL needed):
  python run_end_to_end_pipeline.py --no-inp

  # Quick test (small grid):
  python run_end_to_end_pipeline.py --quick

  # Custom theta:
  python run_end_to_end_pipeline.py --theta-json /path/to/theta_MAP.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
sys.path.insert(0, str(_HERE))

# ── condition -> TMCMC run directory mapping ──────────────────────────────────

CONDITION_RUNS = {
    "dh_baseline": _RUNS_ROOT / "sweep_pg_20260217_081459" / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "Commensal_Static_20260208_002100",
    "commensal_hobic": _RUNS_ROOT / "Commensal_HOBIC_20260208_002100",
    "dysbiotic_static": _RUNS_ROOT / "Dysbiotic_Static_20260207_203752",
}


def load_theta(path):
    """Load 20-element theta vector from theta_MAP.json."""
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
    with open(path) as f:
        d = json.load(f)
    if "theta_full" in d:
        return np.array(d["theta_full"], dtype=np.float64)
    elif "theta_sub" in d:
        return np.array(d["theta_sub"], dtype=np.float64)
    else:
        return np.array([d[k] for k in _PARAM_KEYS], dtype=np.float64)


# ── Step 1: 2D Hamilton + Nutrient ────────────────────────────────────────────


def step_hamilton_nutrient(theta, condition, cfg, out_dir):
    """Run 2D Hamilton+nutrient solver, export CSVs."""
    from run_hamilton_2d_nutrient import run_condition

    return run_condition(theta, condition, cfg, out_dir)


# ── Step 2: Visualization ────────────────────────────────────────────────────


def step_visualize(results_dir, condition):
    """Generate 2D visualization figures."""
    from fem_2d_visualize import main as viz_main
    import matplotlib

    matplotlib.use("Agg")

    # Simulate CLI args for the viz script
    sys.argv = [
        "fem_2d_visualize.py",
        "--results-dir",
        str(results_dir),
        "--condition",
        condition,
    ]
    try:
        viz_main()
        print(f"[VIZ] Figures saved to {results_dir}")
    except Exception as exc:
        print(f"[VIZ] Warning: visualization failed: {exc}")


# ── Step 3: Abaqus INP ──────────────────────────────────────────────────────


def step_abaqus_inp(results_dir, tooth="T23"):
    """Generate two-layer Tie model INP from DI CSV."""
    di_csv = Path(results_dir) / "abaqus_field_2d.csv"
    if not di_csv.exists():
        logger.info("Skipping: %s not found", di_csv)
        return None

    try:
        from biofilm_tooth_tie_assembly import main as tie_main

        out_inp = Path(results_dir) / f"two_layer_{tooth}.inp"
        sys.argv = [
            "biofilm_tooth_tie_assembly.py",
            "--tooth",
            tooth,
            "--di-csv",
            str(di_csv),
            "--out",
            str(out_inp),
        ]
        tie_main()
        print(f"[INP] Generated: {out_inp}")
        return out_inp
    except Exception as exc:
        print(f"[INP] Warning: INP generation failed: {exc}")
        return None


# ── Pipeline orchestrator ────────────────────────────────────────────────────


def run_pipeline(args):
    """Run the complete end-to-end pipeline."""
    from JAXFEM.core_hamilton_2d_nutrient import Config2D

    t_start = time.perf_counter()

    # ── Resolve theta ─────────────────────────────────────────────
    if args.theta_json:
        theta_path = args.theta_json
    elif args.condition in CONDITION_RUNS:
        theta_path = str(CONDITION_RUNS[args.condition] / "theta_MAP.json")
    else:
        logger.error(
            "Unknown condition: %s. Available: %s", args.condition, list(CONDITION_RUNS.keys())
        )
        sys.exit(1)

    if not Path(theta_path).exists():
        logger.error("theta_MAP.json not found: %s", theta_path)
        sys.exit(1)

    theta = load_theta(theta_path)

    # ── Output directory ──────────────────────────────────────────
    tag = args.tag or args.condition
    out_dir = _HERE / "_pipeline_runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build simulation config ───────────────────────────────────
    if args.quick:
        cfg = Config2D(
            Nx=10,
            Ny=10,
            n_macro=10,
            n_react_sub=5,
            dt_h=1e-5,
            save_every=5,
            D_c=args.D_c,
            k_monod=args.k_monod,
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )
    else:
        cfg = Config2D(
            Nx=args.nx,
            Ny=args.ny,
            Lx=args.lx,
            Ly=args.ly,
            dt_h=args.dt_h,
            n_react_sub=args.n_react_sub,
            n_macro=args.n_macro,
            save_every=args.save_every,
            D_c=args.D_c,
            k_monod=args.k_monod,
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )

    # ── Header ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("End-to-End Biofilm FEM Pipeline")
    logger.info("=" * 60)
    logger.info("  Condition : %s", args.condition)
    logger.info("  Theta from: %s", theta_path)
    logger.info("  Grid      : %dx%d", cfg.Nx, cfg.Ny)
    logger.info("  n_macro   : %d", cfg.n_macro)
    logger.info("  Output    : %s", out_dir)
    logger.info(
        "  Steps     : Hamilton+Nutrient -> CSV -> %sViz", "INP -> " if not args.no_inp else ""
    )
    logger.info("=" * 60)

    # ── STEP 1: Hamilton + Nutrient ───────────────────────────────
    logger.info("[STEP 1/3] 2D Hamilton + Nutrient PDE coupling...")
    t1 = time.perf_counter()
    step_hamilton_nutrient(theta, args.condition, cfg, out_dir)
    dt1 = time.perf_counter() - t1
    logger.info("  -> %.1fs", dt1)

    # ── STEP 2: Abaqus INP (optional) ────────────────────────────
    if not args.no_inp:
        logger.info("[STEP 2/3] Abaqus INP generation (tooth=%s)...", args.tooth)
        t2 = time.perf_counter()
        step_abaqus_inp(out_dir, tooth=args.tooth)
        dt2 = time.perf_counter() - t2
        logger.info("  -> %.1fs", dt2)
    else:
        logger.info("[STEP 2/3] Skipped (--no-inp)")
        dt2 = 0.0

    # ── STEP 3: Visualization ─────────────────────────────────────
    logger.info("[STEP 3/3] Visualization...")
    t3 = time.perf_counter()
    step_visualize(out_dir, args.condition)
    dt3 = time.perf_counter() - t3
    logger.info("  -> %.1fs", dt3)

    # ── Summary ───────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Hamilton+Nutrient : %6.1fs", dt1)
    if not args.no_inp:
        logger.info("  Abaqus INP        : %6.1fs", dt2)
    logger.info("  Visualization     : %6.1fs", dt3)
    logger.info("  Total             : %6.1fs", total)
    logger.info("  Output dir        : %s", out_dir)
    logger.info("=" * 60)

    # ── Save pipeline metadata ────────────────────────────────────
    meta = {
        "condition": args.condition,
        "theta_path": str(theta_path),
        "theta": theta.tolist(),
        "grid": f"{cfg.Nx}x{cfg.Ny}",
        "n_macro": cfg.n_macro,
        "K_hill": cfg.K_hill,
        "n_hill": cfg.n_hill,
        "timing": {
            "hamilton_s": round(dt1, 1),
            "inp_s": round(dt2, 1),
            "viz_s": round(dt3, 1),
            "total_s": round(total, 1),
        },
        "outputs": {
            "snapshots_phi": str(out_dir / "snapshots_phi.npy"),
            "abaqus_csv": str(out_dir / "abaqus_field_2d.csv"),
            "eigenstrain_csv": str(out_dir / "macro_eigenstrain_2d.csv"),
        },
    }
    with (out_dir / "pipeline_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="End-to-end biofilm FEM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input
    ap.add_argument(
        "--condition",
        default="dh_baseline",
        choices=list(CONDITION_RUNS.keys()),
        help="TMCMC condition name",
    )
    ap.add_argument("--theta-json", default=None, help="Override: path to theta_MAP.json")
    ap.add_argument("--tag", default=None, help="Output subdirectory tag (default: condition name)")

    # Simulation
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--lx", type=float, default=1.0)
    ap.add_argument("--ly", type=float, default=1.0)
    ap.add_argument("--n-macro", type=int, default=100)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--D-c", type=float, default=0.01)
    ap.add_argument("--k-monod", type=float, default=1.0)
    ap.add_argument("--K-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)

    # Pipeline control
    ap.add_argument("--no-inp", action="store_true", help="Skip Abaqus INP generation")
    ap.add_argument("--tooth", default="T23", help="Tooth ID for INP generation")
    ap.add_argument("--quick", action="store_true", help="Quick test (10x10, 10 macro steps)")

    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
