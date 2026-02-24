#!/usr/bin/env python3
"""
generate_3d_conformal_auto.py
==============================
A1: Automated 3D tooth + biofilm conformal mesh generation.

Given a TMCMC condition, this script:
  1. Loads MAP theta from TMCMC run directory
  2. Runs 2D Hamilton+nutrient PDE to get DI field
  3. Exports DI CSV for each condition
  4. Generates conformal C3D4 biofilm mesh from STL
  5. Builds two-layer Tie INP (tooth S3 + biofilm C3D4)

Supports all 4 conditions + batch mode.

Usage
-----
  # Single condition:
  python generate_3d_conformal_auto.py --condition dh_baseline

  # All conditions:
  python generate_3d_conformal_auto.py --all

  # Quick test:
  python generate_3d_conformal_auto.py --condition dh_baseline --quick
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
_STL_ROOT = _HERE / "external_tooth_models" / "OpenJaw_Dataset" / "Patient_1" / "Teeth"
_OUT_BASE = _HERE / "_3d_conformal_auto"

sys.path.insert(0, str(_HERE))

CONDITION_RUNS = {
    "dh_baseline":      _RUNS_ROOT / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "commensal_static",
    "commensal_hobic":  _RUNS_ROOT / "commensal_hobic",
    "dysbiotic_static": _RUNS_ROOT / "dysbiotic_static",
}

TEETH = {
    "T23": "P1_Tooth_23.stl",
    "T30": "P1_Tooth_30.stl",
    "T31": "P1_Tooth_31.stl",
}

_PARAM_KEYS = [
    "a11", "a12", "a22", "b1", "b2",
    "a33", "a34", "a44", "b3", "b4",
    "a13", "a14", "a23", "a24",
    "a55", "b5",
    "a15", "a25", "a35", "a45",
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


def compute_0d_di(theta_np, K_hill=0.05, n_hill=4.0, n_steps=2500, dt=0.01):
    """Run 0D Hamilton ODE to get condition-specific DI_0D.

    Returns dict with di_0d, phi_final, E_di (Pa).
    """
    from generate_hybrid_macro_csv import solve_0d_composition
    result = solve_0d_composition(theta_np, n_steps=n_steps, dt=dt)
    print(f"  0D DI = {result['di_0d']:.4f}, E_di = {result['E_di']:.1f} Pa")
    return result


def run_fem_2d(theta, condition, cfg_override=None):
    """Run 2D Hamilton+nutrient and return DI field data."""
    from JAXFEM.core_hamilton_2d_nutrient import run_simulation, Config2D

    cfg = cfg_override or Config2D(
        Nx=20, Ny=20, n_macro=100, n_react_sub=20,
        dt_h=1e-5, save_every=10, K_hill=0.05, n_hill=4.0,
    )
    result = run_simulation(theta, cfg)
    return result, cfg


def compute_hybrid_di(di_2d, di_0d):
    """Combine 0D DI (condition scale) with 2D DI (spatial pattern).

    Hybrid DI(x,y) = DI_0D * (DI_2D(x,y) / mean(DI_2D))
    This preserves:
      - condition-specific magnitude from 0D ODE
      - spatial variation pattern from 2D PDE
    """
    di_2d_mean = float(np.mean(di_2d))
    if di_2d_mean < 1e-12:
        return np.full_like(di_2d, di_0d)
    return di_0d * (di_2d / di_2d_mean)


def export_di_csv(result, cfg, out_csv, di_0d=None):
    """Export DI field to CSV from 2D simulation results.

    If di_0d is provided, apply Hybrid scaling (0D scale * 2D pattern).
    """
    phi_snaps = result["phi_snaps"]
    c_snaps = result["c_snaps"]

    phi_final = phi_snaps[-1]  # (5, Nx, Ny)
    c_final = c_snaps[-1]      # (Nx, Ny)
    Nx, Ny = cfg.Nx, cfg.Ny
    x = np.linspace(0, cfg.Lx, Nx)
    y = np.linspace(0, cfg.Ly, Ny)

    # Compute raw 2D DI
    phi_t = phi_final.transpose(1, 2, 0)  # (Nx, Ny, 5)
    phi_sum = phi_t.sum(axis=-1)
    phi_sum_safe = np.where(phi_sum > 0, phi_sum, 1.0)
    p = phi_t / phi_sum_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=-1)
    di_raw = 1.0 - H / np.log(5.0)

    # Apply Hybrid scaling if 0D DI provided
    if di_0d is not None:
        di = compute_hybrid_di(di_raw, di_0d)
        print(f"  Hybrid DI: 0D={di_0d:.4f}, 2D mean={np.mean(di_raw):.4f} "
              f"→ hybrid mean={np.mean(di):.4f}")
    else:
        di = di_raw

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("x,y,phi_pg,di,phi_total,c\n")
        for ix in range(Nx):
            for iy in range(Ny):
                f.write("%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n" % (
                    x[ix], y[iy],
                    float(phi_final[4, ix, iy]),
                    float(di[ix, iy]),
                    float(phi_sum[ix, iy]),
                    float(c_final[ix, iy]),
                ))
    print(f"  DI CSV: {out_csv} ({Nx*Ny} points)")
    return di


def generate_conformal_mesh(condition, tooth_key, di_csv, out_dir, cfg_mesh):
    """Generate conformal C3D4 mesh + two-layer Tie INP."""
    from biofilm_conformal_tet import (
        read_stl, deduplicate_vertices, compute_vertex_normals,
        laplacian_smooth_offset, build_tet_mesh,
        read_di_csv, assign_di_bins, write_abaqus_inp,
    )
    from biofilm_tooth_tie_assembly import (
        process_tooth_shell, process_biofilm,
        write_two_layer_inp, detect_tooth_base,
    )

    stl_path = str(_STL_ROOT / TEETH[tooth_key])
    if not Path(stl_path).exists():
        print(f"  [SKIP] STL not found: {stl_path}")
        return None

    # Build tooth shell
    tooth_data = process_tooth_shell(stl_path, dedup_tol=cfg_mesh["dedup_tol"])
    tooth_data["base_ids"] = detect_tooth_base(
        tooth_data["nodes"], frac=cfg_mesh["base_frac"])

    # Build biofilm conformal mesh
    bio_cfg = {
        "thickness": cfg_mesh["thickness"],
        "n_layers": cfg_mesh["n_layers"],
        "n_bins": cfg_mesh["n_bins"],
        "e_max": cfg_mesh["e_max"],
        "e_min": cfg_mesh["e_min"],
        "di_scale": cfg_mesh["di_scale"],
        "di_exp": cfg_mesh["di_exp"],
        "smooth_iter": cfg_mesh["smooth_iter"],
        "dedup_tol": cfg_mesh["dedup_tol"],
    }
    bio_data = process_biofilm(stl_path, str(di_csv), bio_cfg)

    # Write standalone C3D4 INP
    standalone_inp = out_dir / f"biofilm_{tooth_key}_{condition}.inp"
    write_abaqus_inp(
        str(standalone_inp),
        bio_data["nodes"], bio_data["tets"], bio_data["tet_bins"],
        bio_data["inner_nodes"], bio_data["outer_nodes"],
        bio_data["bin_E_stiff"], cfg_mesh["aniso_ratio"], cfg_mesh["nu"],
        cfg_mesh["n_bins"], cfg_mesh["pressure"], cfg_mesh["bc_mode"],
        bio_data["verts_outer"], bio_data["faces"], bio_data["vnorms_outer"],
        stl_path, str(di_csv), cfg_mesh["n_layers"], cfg_mesh["thickness"],
    )

    # Write two-layer Tie INP
    tie_inp = out_dir / f"two_layer_{tooth_key}_{condition}.inp"
    write_two_layer_inp(
        str(tie_inp), tooth_key, tooth_data, bio_data,
        bio_data["bin_E_stiff"], cfg_mesh["n_bins"], cfg_mesh["nu"],
        cfg_mesh["pressure"], cfg_mesh["shell_thick"],
        str(di_csv), cfg_mesh["thickness"], cfg_mesh["n_layers"],
    )

    return {
        "standalone_inp": str(standalone_inp),
        "tie_inp": str(tie_inp),
        "n_tooth_nodes": len(tooth_data["nodes"]),
        "n_bio_nodes": len(bio_data["nodes"]),
        "n_bio_tets": len(bio_data["tets"]),
    }


def run_condition(condition, args):
    """Full pipeline for one condition."""
    t0 = time.perf_counter()

    # Resolve theta
    run_dir = CONDITION_RUNS.get(condition)
    if run_dir is None:
        print(f"[ERROR] Unknown condition: {condition}")
        return None
    theta_path = run_dir / "theta_MAP.json"
    if not theta_path.exists():
        print(f"[ERROR] theta_MAP.json not found: {theta_path}")
        return None

    theta = load_theta(str(theta_path))
    out_dir = _OUT_BASE / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Condition: {condition}")
    print(f"  Theta from: {theta_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Step 1: 2D Hamilton+nutrient
    from JAXFEM.core_hamilton_2d_nutrient import Config2D
    if args.quick:
        cfg = Config2D(Nx=10, Ny=10, n_macro=10, n_react_sub=5,
                       dt_h=1e-5, save_every=5, K_hill=0.05, n_hill=4.0)
    else:
        cfg = Config2D(Nx=args.nx, Ny=args.ny,
                       n_macro=args.n_macro, n_react_sub=args.n_react_sub,
                       dt_h=args.dt_h, save_every=args.save_every,
                       K_hill=args.k_hill, n_hill=args.n_hill)

    # Step 0: 0D Hamilton ODE → condition-specific DI_0D
    print("\n[0/3] 0D Hamilton ODE → DI_0D...")
    ode_result = compute_0d_di(theta, K_hill=args.k_hill, n_hill=args.n_hill)
    di_0d = ode_result["di_0d"]

    print("\n[1/3] 2D Hamilton + Nutrient PDE...")
    result, cfg = run_fem_2d(theta, condition, cfg)
    di_csv = out_dir / "di_field_2d.csv"
    di = export_di_csv(result, cfg, di_csv, di_0d=di_0d)

    # Save numpy arrays
    np.save(out_dir / "phi_snaps.npy", np.array(result["phi_snaps"]))
    np.save(out_dir / "c_snaps.npy", np.array(result["c_snaps"]))
    np.save(out_dir / "t_snaps.npy", np.array(result["t_snaps"]))
    np.save(out_dir / "di_field.npy", np.array(di))
    np.save(out_dir / "theta.npy", theta)

    # Step 2: Generate conformal meshes for each tooth
    cfg_mesh = {
        "thickness": args.thickness,
        "n_layers": args.n_layers,
        "n_bins": args.n_bins,
        "e_max": args.e_max,
        "e_min": args.e_min,
        "di_scale": args.di_scale,
        "di_exp": args.di_exp,
        "nu": args.nu,
        "pressure": args.pressure,
        "aniso_ratio": 0.5,
        "bc_mode": "inner_fixed",
        "smooth_iter": 3,
        "dedup_tol": 1e-4,
        "base_frac": 0.10,
        "shell_thick": args.shell_thick,
    }

    print(f"\n[2/3] Conformal mesh generation...")
    mesh_results = {}
    for tooth_key in args.teeth:
        print(f"\n  --- {tooth_key} ---")
        mr = generate_conformal_mesh(condition, tooth_key, di_csv, out_dir, cfg_mesh)
        if mr is not None:
            mesh_results[tooth_key] = mr

    # Step 3: Save metadata
    print(f"\n[3/3] Saving metadata...")
    meta = {
        "condition": condition,
        "theta_path": str(theta_path),
        "theta": theta.tolist(),
        "grid": f"{cfg.Nx}x{cfg.Ny}",
        "n_macro": cfg.n_macro,
        "K_hill": cfg.K_hill,
        "n_hill": cfg.n_hill,
        "di_0d": di_0d,
        "E_di_Pa": ode_result["E_di"],
        "di_stats": {
            "mean": float(np.mean(di)),
            "max": float(np.max(di)),
            "min": float(np.min(di)),
        },
        "meshes": mesh_results,
        "timing_s": round(time.perf_counter() - t0, 1),
    }
    with (out_dir / "auto_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    dt = time.perf_counter() - t0
    print(f"\n  Done ({condition}): {dt:.1f}s")
    print(f"  Output: {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser(
        description="Automated 3D conformal mesh generation")
    ap.add_argument("--condition", default="dh_baseline",
                    choices=list(CONDITION_RUNS.keys()))
    ap.add_argument("--all", action="store_true",
                    help="Process all conditions")
    ap.add_argument("--teeth", nargs="+", default=["T23"],
                    choices=list(TEETH.keys()))
    ap.add_argument("--quick", action="store_true")
    # 2D simulation
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--n-macro", type=int, default=100)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--k-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    # Mesh config
    ap.add_argument("--thickness", type=float, default=0.5)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--e-max", type=float, default=1000.0,
                    help="E_healthy [Pa] (default: 1000 Pa)")
    ap.add_argument("--e-min", type=float, default=10.0,
                    help="E_degraded [Pa] (default: 10 Pa)")
    ap.add_argument("--di-scale", type=float, default=1.0,
                    help="DI normalization (default: 1.0 for Hybrid DI)")
    ap.add_argument("--di-exp", type=float, default=2.0)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--pressure", type=float, default=1.0e6)
    ap.add_argument("--shell-thick", type=float, default=0.5)

    args = ap.parse_args()

    if args.all:
        results = {}
        for cond in CONDITION_RUNS:
            results[cond] = run_condition(cond, args)
        # Summary
        print(f"\n{'='*60}")
        print("All conditions processed:")
        for cond, meta in results.items():
            if meta:
                print(f"  {cond}: DI mean={meta['di_stats']['mean']:.4f}, "
                      f"max={meta['di_stats']['max']:.4f}, "
                      f"{meta['timing_s']:.0f}s")
        print(f"{'='*60}")
    else:
        run_condition(args.condition, args)


if __name__ == "__main__":
    main()
