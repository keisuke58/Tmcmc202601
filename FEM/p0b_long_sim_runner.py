#!/usr/bin/env python3
"""
p0b_long_sim_runner.py  –  [P0b] Late-snapshot condition comparison
====================================================================

Runs fem_3d_extension.py for all four biological conditions with a long
time horizon (n_macro=1000, t_end ≈ 3.0) to capture the fully-developed
dysbiotic cascade, then exports DI fields at a late snapshot and (optionally)
regenerates Abaqus INPs and runs FEM solves.

Background
----------
At t=0.05 (current snapshots), the Pg cascade in dh_baseline has not yet
fully developed — DI is lowest for the most dysbiotic condition (counterintuitive).
At t >> 0.05, Pg dominates in dysbiotic conditions, driving DI upward.
This script generates the late-time data needed to make the condition
comparison biologically meaningful.

Usage
-----
  # Step 1: Run long 3D FD simulations for all conditions
  python3 p0b_long_sim_runner.py --step sim

  # Step 2: Export DI fields at late snapshot and run Abaqus
  python3 p0b_long_sim_runner.py --step fem --snapshot -1

  # Full pipeline (sim + fem):
  python3 p0b_long_sim_runner.py --step all

  # Dry-run (print commands only):
  python3 p0b_long_sim_runner.py --step all --dry-run

Output
------
  _results_3d_long/{condition}/snapshots_phi.npy   (1001 snapshots, t=0→3)
  abaqus_field_{condition}_late.csv                (DI field at late snapshot)
  biofilm_3tooth_{condition}_late.inp              (Abaqus INP)
  BioFilm3T_{condition}_late.odb                  (FEM result)
"""

import argparse
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_FEM3D = os.path.join(_HERE, "fem_3d_extension.py")
_COUPLING = os.path.join(_HERE, "tmcmc_to_fem_coupling.py")
_ASSEMBLY = os.path.join(_HERE, "biofilm_3tooth_assembly.py")
_ABAQUS = "/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"
_EXTRACT = os.path.join(_HERE, "odb_extract.py")
_RUNS_ROOT = os.path.join(
    os.path.dirname(_HERE), "data_5species", "_runs", "sweep_pg_20260217_081459"
)
_RESULTS_LONG = os.path.join(_HERE, "_results_3d_long")
_STL_ROOT = os.path.join(_HERE, "external_tooth_models", "OpenJaw_Dataset", "Patient_1")

CONDITIONS = [
    {"key": "dh_baseline", "label": "DH-baseline", "theta_subdir": "dh_baseline"},
    {"key": "commensal_static", "label": "Commensal-static", "theta_subdir": "commensal_static"},
    {"key": "Dysbiotic_Static", "label": "Dysbiotic-static", "theta_subdir": "Dysbiotic_Static"},
    {"key": "Commensal_HOBIC", "label": "Commensal-HOBIC", "theta_subdir": "Commensal_HOBIC"},
]


def run_cmd(cmd, cwd=None, dry_run=False, log_file=None):
    print("  $", " ".join(str(c) for c in cmd))
    if dry_run:
        print("  [dry-run] skipped")
        return 0
    if log_file:
        with open(log_file, "w") as lf:
            ret = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    else:
        ret = subprocess.run(cmd, cwd=cwd)
    return ret.returncode


def step_sim(args):
    """Run fem_3d_extension.py for each condition with long time horizon."""
    print("\n" + "=" * 60)
    print("  P0b Step 1: Long 3D FD simulations")
    print(f"  n_macro={args.n_macro}  save_every={args.save_every}")
    print("=" * 60)

    os.makedirs(_RESULTS_LONG, exist_ok=True)

    for cond in CONDITIONS:
        theta_path = os.path.join(_RUNS_ROOT, cond["theta_subdir"], "theta_MAP.json")
        if not os.path.exists(theta_path):
            print(f"\n[skip] {cond['key']}: theta_MAP.json not found at {theta_path}")
            continue

        out_dir = os.path.join(_RESULTS_LONG, cond["key"] + "_long")
        done_flag = os.path.join(out_dir, "done.flag")
        if os.path.exists(done_flag) and not args.force:
            print(f"\n[skip] {cond['key']}: already done ({out_dir}/done.flag exists)")
            continue

        print(f"\n── {cond['label']} ──")
        log = os.path.join(_RESULTS_LONG, f"{cond['key']}_long.log")
        cmd = [
            "python3",
            _FEM3D,
            "--theta-json",
            theta_path,
            "--condition",
            cond["key"],
            "--nx",
            str(args.nx),
            "--ny",
            str(args.ny),
            "--nz",
            str(args.nz),
            "--n-macro",
            str(args.n_macro),
            "--n-react-sub",
            str(args.n_react_sub),
            "--save-every",
            str(args.save_every),
            "--out-dir",
            out_dir,
        ]
        rc = run_cmd(cmd, dry_run=args.dry_run, log_file=log if not args.dry_run else None)
        if rc == 0 and not args.dry_run:
            open(done_flag, "w").close()
            print(f"  Done → {out_dir}")
        elif rc != 0:
            print(f"  [error] rc={rc}  See {log}")


def step_fem(args):
    """Export DI fields at late snapshot, regenerate INPs, run Abaqus."""
    print("\n" + "=" * 60)
    print(f"  P0b Step 2: Export DI + FEM  (snapshot={args.snapshot})")
    print("=" * 60)

    for cond in CONDITIONS:
        results_dir = os.path.join(_RESULTS_LONG, cond["key"] + "_long")
        snaps_path = os.path.join(results_dir, "snapshots_phi.npy")

        if not os.path.exists(snaps_path) and not args.dry_run:
            print(f"\n[skip] {cond['key']}: no long snapshots ({snaps_path})")
            print("  Run: python3 p0b_long_sim_runner.py --step sim")
            continue

        print(f"\n── {cond['label']} ──")
        tag = cond["key"]
        di_csv = os.path.join(_HERE, f"abaqus_field_{tag}_late.csv")
        inp_file = os.path.join(_HERE, f"biofilm_3tooth_{tag}_late.inp")
        job_name = f"BioFilm3T_{tag}_late"
        odb_file = os.path.join(_HERE, f"{job_name}.odb")
        elem_csv = os.path.join(_HERE, f"odb_elements_{tag}_late.csv")
        node_csv = os.path.join(_HERE, f"odb_nodes_{tag}_late.csv")

        # Step 2a: Export DI field from long simulation
        # We use tmcmc_to_fem_coupling.py but pointing to the long results dir
        # (we need to temporarily override _RESULTS_3D)
        # Instead, call export_di_field directly via a small inline script
        export_script = f"""
import sys, os
sys.path.insert(0, r"{_HERE}")
import numpy as np

results_dir = r"{results_dir}"
snaps_phi = np.load(os.path.join(results_dir, "snapshots_phi.npy"))
snaps_t   = np.load(os.path.join(results_dir, "snapshots_t.npy"))
x = np.load(os.path.join(results_dir, "mesh_x.npy"))
y = np.load(os.path.join(results_dir, "mesh_y.npy"))
z = np.load(os.path.join(results_dir, "mesh_z.npy"))

idx = {args.snapshot} if {args.snapshot} >= 0 else snaps_phi.shape[0] + {args.snapshot}
phi_at_t = snaps_phi[idx]      # (5, Nx, Ny, Nz)
t_val    = float(snaps_t[idx])
print(f"  Snapshot {{idx}}: t={{t_val:.4f}}")

phi_nodes = phi_at_t.transpose(1,2,3,0)  # (Nx,Ny,Nz,5)
Nx,Ny,Nz  = phi_nodes.shape[:3]
sum_phi   = phi_nodes.sum(axis=3)
p         = phi_nodes / np.where(sum_phi>0,sum_phi,1)[...,None]
lnp       = np.where(p>1e-15, np.log(p), 0.0)
H         = -(p*lnp).sum(axis=3)
di        = 1 - H/np.log(5)
phi_pg    = phi_nodes[:,:,:,4]
phi_tot   = phi_nodes.sum(axis=3)
r_pg      = phi_pg / np.where(phi_tot>0,phi_tot,1)
xx,yy,zz  = np.meshgrid(x[:Nx],y[:Ny],z[:Nz],indexing='ij')

out_csv = r"{di_csv}"
with open(out_csv,'w') as f:
    f.write(f"# condition={tag}, snapshot_index={{idx}}, t={{t_val:.6e}}\\n")
    f.write("x,y,z,phi_pg,di,phi_tot,r_pg,t\\n")
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                f.write(f"{{xx[ix,iy,iz]:.6f}},{{yy[ix,iy,iz]:.6f}},{{zz[ix,iy,iz]:.6f}},"
                        f"{{phi_pg[ix,iy,iz]:.8f}},{{di[ix,iy,iz]:.8f}},"
                        f"{{phi_tot[ix,iy,iz]:.8f}},{{r_pg[ix,iy,iz]:.8f}},{{t_val:.6f}}\\n")
print(f"  DI CSV written: {{out_csv}} ({{Nx*Ny*Nz}} points)")
print(f"  DI mean={{di.mean():.4f}}  DI max={{di.max():.4f}}")
"""
        tag_var = tag  # close over
        cmd_export = ["python3", "-c", export_script]
        rc = run_cmd(cmd_export, dry_run=args.dry_run)
        if rc != 0:
            print("  [error] DI export failed")
            continue

        # Step 2b: Regenerate INP with late DI field
        cmd_inp = [
            "python3",
            _ASSEMBLY,
            "--stl-root",
            _STL_ROOT,
            "--di-csv",
            di_csv,
            "--out",
            inp_file,
        ]
        rc = run_cmd(cmd_inp, dry_run=args.dry_run)
        if rc != 0:
            print("  [error] INP generation failed")
            continue

        if not args.no_abaqus:
            # Step 2c: Run Abaqus
            cmd_abq = [
                _ABAQUS,
                f"job={job_name}",
                f"input={inp_file}",
                f"cpus={args.cpus}",
                "ask=off",
                "interactive",
            ]
            rc = run_cmd(cmd_abq, cwd=_HERE, dry_run=args.dry_run)
            if rc != 0:
                print("  [error] Abaqus failed")
                continue

            # Step 2d: Extract ODB
            cmd_ext = [
                _ABAQUS,
                "python",
                _EXTRACT,
                odb_file,
                "--out-elem",
                elem_csv,
                "--out-node",
                node_csv,
            ]
            run_cmd(cmd_ext, cwd=_HERE, dry_run=args.dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["sim", "fem", "all"], default="all")
    ap.add_argument(
        "--snapshot",
        type=int,
        default=-1,
        help="Snapshot index for DI export (-1 = last) [default: -1]",
    )
    ap.add_argument(
        "--n-macro",
        type=int,
        default=1000,
        help="Number of FD macro steps [default: 1000  → t≈3.0]",
    )
    ap.add_argument("--n-react-sub", type=int, default=50)
    ap.add_argument(
        "--save-every", type=int, default=10, help="Save snapshot every N steps [default: 10]"
    )
    ap.add_argument("--nx", type=int, default=15)
    ap.add_argument("--ny", type=int, default=15)
    ap.add_argument("--nz", type=int, default=15)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run even if done.flag exists")
    ap.add_argument(
        "--no-abaqus", action="store_true", help="Skip Abaqus solve (only export DI + INP)"
    )
    args = ap.parse_args()

    print("=" * 60)
    print("  P0b Long-snapshot Condition Comparison Runner")
    print("=" * 60)
    print(f"  step:       {args.step}")
    print(f"  n_macro:    {args.n_macro}  (≈ t={args.n_macro * 3e-3:.2f})")
    print(f"  snapshot:   {args.snapshot}")
    print(f"  dry-run:    {args.dry_run}")

    if args.step in ("sim", "all"):
        step_sim(args)

    if args.step in ("fem", "all"):
        step_fem(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
