#!/usr/bin/env python3
"""
run_hamilton_2d_nutrient.py
============================
Production script: 2D Hamilton + Nutrient PDE coupling.

Connects the JAX-based 2D Hamilton-nutrient solver to the TMCMC pipeline.
Loads MAP theta from TMCMC runs, runs the 2D simulation, and exports
results compatible with the downstream Abaqus pipeline.

Outputs (saved to --out-dir):
  snapshots_phi.npy   (n_snap, 5, Nx, Ny)   species volume fractions
  snapshots_c.npy     (n_snap, Nx, Ny)       nutrient concentration
  snapshots_t.npy     (n_snap,)              time values
  di_field.npy        (n_snap, Nx, Ny)       dysbiosis index
  alpha_monod.npy     (Nx, Ny)               Monod growth activity
  mesh_x.npy          (Nx,)
  mesh_y.npy          (Ny,)
  theta_MAP.npy       (20,)
  abaqus_field_2d.csv                        DI + phi_pg for Abaqus import
  run_config.json                            run configuration

Usage
-----
  # Single condition with MAP theta:
  python run_hamilton_2d_nutrient.py \\
      --theta-json ../data_5species/_runs/.../theta_MAP.json \\
      --condition dh_baseline \\
      --nx 20 --ny 20 --n-macro 100 \\
      --out-dir _results_2d_nutrient/dh_baseline

  # All conditions (automated):
  python run_hamilton_2d_nutrient.py --all-conditions

  # Quick sanity test:
  python run_hamilton_2d_nutrient.py --quick-test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
sys.path.insert(0, str(_HERE))

# ── TMCMC run dirs for each condition ────────────────────────────────────────
CONDITION_RUNS = {
    "dh_baseline": _RUNS_ROOT / "sweep_pg_20260217_081459" / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "Commensal_Static_20260208_002100",
    "commensal_hobic": _RUNS_ROOT / "Commensal_HOBIC_20260208_002100",
    "dysbiotic_static": _RUNS_ROOT / "Dysbiotic_Static_20260207_203752",
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

_DEFAULT_THETA = str(_RUNS_ROOT / "sweep_pg_20260217_081459" / "dh_baseline" / "theta_MAP.json")


# ── theta loading (shared pattern with fem_2d_extension.py) ──────────────────


def load_theta(path: str) -> np.ndarray:
    """Load theta vector from theta_MAP.json."""
    with open(path) as f:
        d = json.load(f)
    if "theta_full" in d:
        vec = np.array(d["theta_full"], dtype=np.float64)
    elif "theta_sub" in d:
        vec = np.array(d["theta_sub"], dtype=np.float64)
    else:
        vec = np.array([d[k] for k in _PARAM_KEYS], dtype=np.float64)
    logger.info("Loaded theta from: %s", path)
    for i, (k, v) in enumerate(zip(_PARAM_KEYS, vec)):
        logger.info("  [%2d] %5s = %8.4f", i, k, v)
    return vec


# ── export for Abaqus ────────────────────────────────────────────────────────


def export_abaqus_csv(
    phi_snaps, c_snaps, di_field, mesh_x, mesh_y, out_csv: Path, snap_idx: int = -1
):
    """
    Export 2D field data to CSV for Abaqus import.

    Columns: x, y, phi_pg, di, phi_total, c, r_pg
    """
    phi = phi_snaps[snap_idx]  # (5, Nx, Ny)
    di = di_field[snap_idx]  # (Nx, Ny)
    c = c_snaps[snap_idx]  # (Nx, Ny)
    Nx, Ny = di.shape

    phi_pg = phi[4]  # P.gingivalis
    phi_total = phi.sum(axis=0)
    phi_total_safe = np.where(phi_total > 0, phi_total, 1.0)
    r_pg = phi_pg / phi_total_safe  # relative abundance of Pg

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("x,y,phi_pg,di,phi_total,c,r_pg\n")
        for ix in range(Nx):
            for iy in range(Ny):
                f.write(
                    "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n"
                    % (
                        mesh_x[ix],
                        mesh_y[iy],
                        float(phi_pg[ix, iy]),
                        float(di[ix, iy]),
                        float(phi_total[ix, iy]),
                        float(c[ix, iy]),
                        float(r_pg[ix, iy]),
                    )
                )
    logger.info("Exported Abaqus CSV -> %s", out_csv)


# ── macro eigenstrain CSV (bridge to P1 Abaqus) ─────────────────────────────


def export_eigenstrain_csv(
    alpha_monod, di_field, mesh_x, mesh_y, out_csv: Path, snap_idx: int = -1
):
    """
    Export eigenstrain field for Abaqus thermal-analogy import.

    eps_growth = alpha_monod / 3  (isotropic expansion)
    E_eff = f(DI)  (DI-dependent Young's modulus)

    Columns: x, y, alpha_monod, eps_growth, di, E_Pa
    """
    di = di_field[snap_idx]
    Nx, Ny = di.shape

    # Material model: power-law DI -> E
    E_MAX = 10.0  # GPa (healthy = high DI = dominated by commensals)
    E_MIN = 0.5  # GPa (dysbiotic = low DI = diverse community)
    DI_EXP = 2.0
    E_Pa = (E_MIN + (E_MAX - E_MIN) * np.clip(di, 0, 1) ** DI_EXP) * 1e9

    eps_growth = alpha_monod / 3.0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("# Macro eigenstrain from 2D Hamilton+Nutrient coupling\n")
        f.write("x,y,alpha_monod,eps_growth,di,E_Pa\n")
        for ix in range(Nx):
            for iy in range(Ny):
                f.write(
                    "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n"
                    % (
                        mesh_x[ix],
                        mesh_y[iy],
                        float(alpha_monod[ix, iy]),
                        float(eps_growth[ix, iy]),
                        float(di[ix, iy]),
                        float(E_Pa[ix, iy]),
                    )
                )
    logger.info("Exported eigenstrain CSV -> %s", out_csv)


# ── run for one condition ────────────────────────────────────────────────────


def run_condition(theta, condition, cfg, out_dir):
    """Run 2D Hamilton+nutrient and save all outputs."""
    from JAXFEM.core_hamilton_2d_nutrient import (
        run_simulation,
        compute_di_field,
        compute_alpha_monod,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = run_simulation(theta, cfg)
    elapsed = time.perf_counter() - t0

    phi_snaps = result["phi_snaps"]
    c_snaps = result["c_snaps"]
    t_snaps = result["t_snaps"]

    # Derived fields
    di_field = compute_di_field(phi_snaps)
    alpha_monod = compute_alpha_monod(phi_snaps, c_snaps, t_snaps)

    # Save numpy arrays
    np.save(out_dir / "snapshots_phi.npy", phi_snaps)
    np.save(out_dir / "snapshots_c.npy", c_snaps)
    np.save(out_dir / "snapshots_t.npy", t_snaps)
    np.save(out_dir / "di_field.npy", di_field)
    np.save(out_dir / "alpha_monod.npy", alpha_monod)
    mesh_x = np.linspace(0, cfg.Lx, cfg.Nx)
    mesh_y = np.linspace(0, cfg.Ly, cfg.Ny)
    np.save(out_dir / "mesh_x.npy", mesh_x)
    np.save(out_dir / "mesh_y.npy", mesh_y)
    np.save(out_dir / "theta_MAP.npy", theta)

    # Export CSVs
    export_abaqus_csv(
        phi_snaps,
        c_snaps,
        di_field,
        mesh_x,
        mesh_y,
        out_dir / "abaqus_field_2d.csv",
    )
    export_eigenstrain_csv(
        alpha_monod,
        di_field,
        mesh_x,
        mesh_y,
        out_dir / "macro_eigenstrain_2d.csv",
    )

    # Config metadata
    config = {
        "condition": condition,
        "Nx": cfg.Nx,
        "Ny": cfg.Ny,
        "Lx": cfg.Lx,
        "Ly": cfg.Ly,
        "dt_h": cfg.dt_h,
        "n_react_sub": cfg.n_react_sub,
        "n_macro": cfg.n_macro,
        "D_c": cfg.D_c,
        "k_monod": cfg.k_monod,
        "K_hill": cfg.K_hill,
        "n_hill": cfg.n_hill,
        "elapsed_s": elapsed,
        "n_snapshots": len(t_snaps),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Condition: %s", condition)
    logger.info("  phi shape: %s", phi_snaps.shape)
    logger.info("  DI range:  [%.4f, %.4f]", di_field[-1].min(), di_field[-1].max())
    logger.info("  alpha range: [%.6f, %.6f]", alpha_monod.min(), alpha_monod.max())
    logger.info("  c_min=%.4f  c_max=%.4f", c_snaps[-1].min(), c_snaps[-1].max())
    logger.info("  Elapsed: %.1fs", elapsed)
    logger.info("  Output: %s", out_dir)
    logger.info("=" * 50)

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="2D Hamilton + Nutrient PDE coupling (JAX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--theta-json", default=_DEFAULT_THETA, help="Path to theta_MAP.json")
    ap.add_argument("--condition", default="dh_baseline", help="Condition name")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--lx", type=float, default=1.0)
    ap.add_argument("--ly", type=float, default=1.0)
    ap.add_argument("--n-macro", type=int, default=100)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--D-c", type=float, default=0.01, help="Nutrient diffusion coefficient")
    ap.add_argument("--k-monod", type=float, default=1.0, help="Monod half-saturation constant")
    ap.add_argument("--K-hill", type=float, default=0.05, help="Hill gate K for Fn->Pg")
    ap.add_argument("--n-hill", type=float, default=4.0, help="Hill gate exponent")
    ap.add_argument("--out-dir", default="_results_2d_nutrient/run")
    ap.add_argument(
        "--all-conditions", action="store_true", help="Run all 4 conditions sequentially"
    )
    ap.add_argument(
        "--quick-test", action="store_true", help="Quick sanity test (small grid, few steps)"
    )
    args = ap.parse_args()

    # Lazy import to avoid JAX startup if just checking --help
    from JAXFEM.core_hamilton_2d_nutrient import Config2D

    if args.quick_test:
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
        from JAXFEM.core_hamilton_2d_nutrient import THETA_DEMO

        run_condition(THETA_DEMO, "quick_test", cfg, _HERE / "_results_2d_nutrient" / "quick_test")
        return

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

    if args.all_conditions:
        for cond, run_dir in CONDITION_RUNS.items():
            theta_json = run_dir / "theta_MAP.json"
            if not theta_json.exists():
                logger.info("[skip] %s: theta_MAP.json not found at %s", cond, theta_json)
                continue
            theta = load_theta(str(theta_json))
            out_dir = _HERE / "_results_2d_nutrient" / cond
            run_condition(theta, cond, cfg, out_dir)
    else:
        theta = load_theta(args.theta_json)
        run_condition(theta, args.condition, cfg, Path(args.out_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
