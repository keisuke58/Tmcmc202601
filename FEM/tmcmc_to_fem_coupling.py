#!/usr/bin/env python3
"""
tmcmc_to_fem_coupling.py  –  [P2] TMCMC → FEM coupling
========================================================

Reads TMCMC posterior MAP parameters for a chosen biological condition,
exports the corresponding 3-D DI (Dysbiotic Index) field as an Abaqus-ready
CSV, and (optionally) regenerates the BioFilm3T INP with the new material
distribution.

Pipeline
--------
  TMCMC run (sweep_pg_20260217_081459/{condition}/theta_MAP.json)
      ↓
  3-D FD simulation results (_results_3d/{condition}/snapshots_phi.npy)
      ↓  [ _compute_di() ]
  DI field CSV  →  biofilm_3tooth_assembly.py  →  new BioFilm3T.inp

Usage
-----
  # List available conditions and snapshots:
  python3 tmcmc_to_fem_coupling.py --list

  # Export DI field for dh_baseline at snapshot 20 (default):
  python3 tmcmc_to_fem_coupling.py --condition dh_baseline --snapshot 20

  # Export + regenerate INP:
  python3 tmcmc_to_fem_coupling.py --condition dh_baseline --snapshot 20 --regen-inp

  # Export for commensal_static and compare to current run:
  python3 tmcmc_to_fem_coupling.py --condition commensal_static --snapshot -1

Output
------
  abaqus_field_{condition}_snap{N:02d}.csv   – DI field for the condition
  (optional) biofilm_3tooth_{condition}.inp  – new INP using the new field

Notes
-----
  * The 3-D simulation results must already exist in _results_3d/{condition}/.
    Run fem_3d_extension.py first if they are missing.
  * No Abaqus solve is run by this script.  After regenerating the INP:
      abaqus job=BioFilm3T_{condition} input=biofilm_3tooth_{condition}.inp cpus=4 interactive
"""

from __future__ import print_function, division
import sys
import os
import json
import argparse
import subprocess

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_RESULTS_3D  = os.path.join(_HERE, "_results_3d")
_RUNS_ROOT   = os.path.join(os.path.dirname(_HERE),
                            "data_5species", "_runs",
                            "sweep_pg_20260217_081459")

ASSEMBLY_SCRIPT = os.path.join(_HERE, "biofilm_3tooth_assembly.py")
ABAQUS_CMD      = "/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"

# ── DI computation (mirrors export_for_abaqus.py) ─────────────────────────────

def _compute_di(phi_all):
    """
    Compute Dysbiotic Index per spatial node.

    phi_all : (..., 5) float – species volume fractions
    Returns  (...,) float – DI in [0, 1], where 1 = maximally dysbiotic
    (all mass in one species) and 0 = maximally diverse (equal fractions).

    Formula: DI = 1 - H / H_max,  H = -sum p_i ln p_i,  H_max = ln(5)
    """
    sum_phi      = np.sum(phi_all, axis=-1)
    sum_phi_safe = np.where(sum_phi > 0.0, sum_phi, 1.0)
    p            = phi_all / sum_phi_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log(p), 0.0)
    H     = -np.sum(p * log_p, axis=-1)
    H_max = np.log(5.0)
    return 1.0 - H / H_max


# ── Discover available conditions ─────────────────────────────────────────────

def list_conditions():
    """Print available conditions from TMCMC sweep and 3-D simulation directories."""
    print("\n[TMCMC sweep conditions]  (%s)" % _RUNS_ROOT)
    if os.path.isdir(_RUNS_ROOT):
        for name in sorted(os.listdir(_RUNS_ROOT)):
            theta_path = os.path.join(_RUNS_ROOT, name, "theta_MAP.json")
            marker = "✓ theta_MAP.json" if os.path.isfile(theta_path) else "(no theta_MAP)"
            print("  %-40s  %s" % (name, marker))
    else:
        print("  (directory not found: %s)" % _RUNS_ROOT)

    print("\n[3-D simulation results]  (%s)" % _RESULTS_3D)
    if os.path.isdir(_RESULTS_3D):
        for name in sorted(os.listdir(_RESULTS_3D)):
            snaps = os.path.join(_RESULTS_3D, name, "snapshots_phi.npy")
            if os.path.isfile(snaps):
                phi = np.load(snaps)
                print("  %-35s  n_snapshots=%d  shape=%s" % (
                    name, phi.shape[0], str(phi.shape)))
    else:
        print("  (directory not found: %s)" % _RESULTS_3D)


# ── Load theta_MAP ────────────────────────────────────────────────────────────

def load_theta_map(condition):
    """Return (theta_20, source_path). Searches TMCMC sweep dir."""
    candidates = [
        os.path.join(_RUNS_ROOT, condition, "theta_MAP.json"),
        os.path.join(os.path.dirname(_HERE), "data_5species", "_runs",
                     condition, "theta_MAP.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                d = json.load(f)
            # Support both {"theta_sub": [...]} and {"theta_full": [...]} and plain list
            if isinstance(d, list):
                return np.array(d[:20], dtype=float), path
            theta = d.get("theta_full") or d.get("theta_sub") or d.get("MAP") or d
            if isinstance(theta, list):
                return np.array(theta[:20], dtype=float), path
    raise FileNotFoundError(
        "theta_MAP.json not found for condition '%s'.\n"
        "Checked: %s" % (condition, "\n  ".join(candidates)))


# ── Export DI field CSV ───────────────────────────────────────────────────────

def export_di_field(condition, snapshot_index, out_csv):
    """
    Read 3-D simulation snapshots for *condition* and write a DI field CSV.

    Parameters
    ----------
    condition      : str  e.g. 'dh_baseline'
    snapshot_index : int  time snapshot index (negative = from end)
    out_csv        : str  output CSV path
    """
    results_dir = os.path.join(_RESULTS_3D, condition)
    snaps_path  = os.path.join(results_dir, "snapshots_phi.npy")
    t_path      = os.path.join(results_dir, "snapshots_t.npy")
    x_path      = os.path.join(results_dir, "mesh_x.npy")
    y_path      = os.path.join(results_dir, "mesh_y.npy")
    z_path      = os.path.join(results_dir, "mesh_z.npy")

    for p in [snaps_path, t_path, x_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                "3-D simulation result not found: %s\n"
                "Run fem_3d_extension.py --condition %s first." % (p, condition))

    snaps_phi = np.load(snaps_path)    # (n_snap, 5, Nx, Ny, Nz)
    snaps_t   = np.load(t_path)       # (n_snap,)
    x         = np.load(x_path)
    y         = np.load(y_path) if os.path.isfile(y_path) else x
    z         = np.load(z_path) if os.path.isfile(z_path) else x

    n_snap = snaps_phi.shape[0]
    if snapshot_index < 0:
        idx = n_snap + snapshot_index
    else:
        idx = snapshot_index
    if idx < 0 or idx >= n_snap:
        raise IndexError(
            "snapshot_index %d out of range (n_snap=%d)" % (snapshot_index, n_snap))

    phi_at_t = snaps_phi[idx]    # (5, Nx, Ny, Nz)
    t_val    = float(snaps_t[idx])

    ndim = phi_at_t.ndim - 1     # spatial dims: 1, 2, or 3
    print("  Shape: %s  t=%.5f  ndim=%d" % (str(phi_at_t.shape), t_val, ndim))

    if ndim == 3:
        # (5, Nx, Ny, Nz) → (Nx, Ny, Nz, 5)
        phi_nodes = np.transpose(phi_at_t, (1, 2, 3, 0))
        Nx, Ny, Nz = phi_nodes.shape[:3]
        di     = _compute_di(phi_nodes)        # (Nx, Ny, Nz)
        phi_pg = phi_nodes[:, :, :, 4]
        phi_tot = np.sum(phi_nodes, axis=3)
        phi_tot_safe = np.where(phi_tot > 0, phi_tot, 1.0)
        r_pg   = phi_pg / phi_tot_safe
        xx, yy, zz = np.meshgrid(x[:Nx], y[:Ny], z[:Nz], indexing="ij")

        print("  Writing 3D DI field: %d × %d × %d = %d points" % (
            Nx, Ny, Nz, Nx * Ny * Nz))
        with open(out_csv, "w") as f:
            f.write("# condition=%s, snapshot_index=%d, t=%.6e\n" % (
                condition, idx, t_val))
            f.write("x,y,z,phi_pg,di,phi_tot,r_pg,t\n")
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        f.write("%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n" % (
                            xx[ix, iy, iz], yy[ix, iy, iz], zz[ix, iy, iz],
                            float(phi_pg[ix, iy, iz]),
                            float(di[ix, iy, iz]),
                            float(phi_tot[ix, iy, iz]),
                            float(r_pg[ix, iy, iz]),
                            t_val))

    elif ndim == 2:
        phi_nodes = np.transpose(phi_at_t, (1, 2, 0))   # (Nx, Ny, 5)
        Nx, Ny = phi_nodes.shape[:2]
        di     = _compute_di(phi_nodes)
        phi_pg = phi_nodes[:, :, 4]
        phi_tot = np.sum(phi_nodes, axis=2)
        phi_tot_safe = np.where(phi_tot > 0, phi_tot, 1.0)
        r_pg   = phi_pg / phi_tot_safe
        xx, yy = np.meshgrid(x[:Nx], y[:Ny], indexing="ij")
        zz = np.zeros_like(xx)   # collapse z to 0 for 3D INP compatibility

        print("  Writing 2D→3D DI field (z=0): %d × %d = %d points" % (
            Nx, Ny, Nx * Ny))
        with open(out_csv, "w") as f:
            f.write("# condition=%s, snapshot_index=%d, t=%.6e (2D expanded to 3D)\n" % (
                condition, idx, t_val))
            f.write("x,y,z,phi_pg,di,phi_tot,r_pg,t\n")
            for ix in range(Nx):
                for iy in range(Ny):
                    f.write("%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n" % (
                        xx[ix, iy], yy[ix, iy], 0.0,
                        float(phi_pg[ix, iy]),
                        float(di[ix, iy]),
                        float(phi_tot[ix, iy]),
                        float(r_pg[ix, iy]),
                        t_val))

    elif ndim == 1:
        phi_nodes = phi_at_t.T   # (Nx, 5)
        Nx = phi_nodes.shape[0]
        di    = _compute_di(phi_nodes)
        phi_pg = phi_nodes[:, 4]
        phi_tot = np.sum(phi_nodes, axis=1)
        phi_tot_safe = np.where(phi_tot > 0, phi_tot, 1.0)
        r_pg  = phi_pg / phi_tot_safe
        xs    = x[:Nx]

        print("  Writing 1D→3D DI field (y=z=0): %d points" % Nx)
        with open(out_csv, "w") as f:
            f.write("# condition=%s, snapshot_index=%d, t=%.6e (1D expanded to 3D)\n" % (
                condition, idx, t_val))
            f.write("x,y,z,phi_pg,di,phi_tot,r_pg,t\n")
            for ix in range(Nx):
                f.write("%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n" % (
                    xs[ix], 0.0, 0.0,
                    float(phi_pg[ix]),
                    float(di[ix]),
                    float(phi_tot[ix]),
                    float(r_pg[ix]),
                    t_val))
    else:
        raise ValueError("Unexpected snapshot dimensionality: %s" % str(phi_at_t.shape))

    # Summary statistics
    di_flat = di.ravel()
    print("  DI stats: min=%.4f  mean=%.4f  median=%.4f  max=%.4f" % (
        di_flat.min(), di_flat.mean(), float(np.median(di_flat)), di_flat.max()))
    print("  Written: %s  (%.1f KB)" % (out_csv, os.path.getsize(out_csv) / 1024))


# ── Summarise MAP parameters ───────────────────────────────────────────────────

def print_theta_summary(theta, source_path):
    """Print a formatted summary of the 20 MAP parameters."""
    param_names = [
        "a11", "a12", "a22", "b1",  "b2",    # M1: S.o, A.n
        "a33", "a34", "a44", "b3",  "b4",    # M2: Vei, F.n
        "a13", "a14", "a23", "a24",           # M3: cross-commensal
        "a55", "b5",                           # M4: P.g self
        "a15", "a25", "a35", "a45",           # M5: support of P.g
    ]
    print("\n  MAP parameters from: %s" % source_path)
    print("  %-6s  %8s" % ("Name", "Value"))
    print("  " + "-" * 20)
    for name, val in zip(param_names, theta):
        print("  %-6s  %8.4f" % (name, val))
    print("\n  Key P.g support parameters:")
    print("    a35 (Veillonella → P.g) = %.4f  [theta[18]]" % theta[18])
    print("    a45 (F.nucleatum → P.g) = %.4f  [theta[19]]" % theta[19])
    print("    a55 (P.g self-growth)   = %.4f  [theta[14]]" % theta[14])


# ── Regenerate INP ────────────────────────────────────────────────────────────

def regen_inp(di_csv, condition, dry_run=False):
    """Call biofilm_3tooth_assembly.py with the new DI CSV."""
    inp_name = "biofilm_3tooth_%s.inp" % condition
    out_path = os.path.join(_HERE, inp_name)
    cmd = [
        sys.executable, ASSEMBLY_SCRIPT,
        "--di-csv", di_csv,
        "--out",    out_path,
        "--n-bins", "20",
    ]
    print("\n  Command: %s" % " ".join(cmd))
    if dry_run:
        print("  [DRY RUN] Not executing.")
        return out_path
    result = subprocess.run(cmd, capture_output=False, cwd=_HERE)
    if result.returncode != 0:
        print("  ERROR: assembly script returned code %d" % result.returncode)
    else:
        print("  INP written: %s" % out_path)
        print("  Next step:")
        print("    abaqus job=BioFilm3T_%s input=%s cpus=4 interactive" % (
            condition, inp_name))
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TMCMC → FEM coupling: export DI field for a TMCMC condition")
    p.add_argument("--list",       action="store_true",
                   help="List available conditions and exit")
    p.add_argument("--condition",  default="dh_baseline",
                   help="Biological condition name [dh_baseline]")
    p.add_argument("--snapshot",   type=int, default=20,
                   help="Snapshot index (-1 = last) [20]")
    p.add_argument("--out-csv",    default=None,
                   help="Output CSV path [auto: abaqus_field_{condition}_snap{N}.csv]")
    p.add_argument("--regen-inp",  action="store_true",
                   help="Regenerate biofilm_3tooth_{condition}.inp with new DI field")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print commands but don't execute assembly script")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_conditions()
        return

    condition = args.condition
    snap      = args.snapshot

    print("=" * 62)
    print("  tmcmc_to_fem_coupling.py")
    print("  condition  : %s" % condition)
    print("  snapshot   : %d" % snap)
    print("=" * 62)

    # 1. Load MAP parameters
    print("\n[1/3] Loading TMCMC MAP parameters ...")
    try:
        theta, theta_path = load_theta_map(condition)
        print_theta_summary(theta, theta_path)
    except FileNotFoundError as e:
        print("  WARNING: %s" % e)
        print("  Proceeding with DI field export only (theta not required).")
        theta = None

    # 2. Export DI field
    print("\n[2/3] Exporting DI field from 3-D simulation ...")
    if args.out_csv:
        out_csv = args.out_csv
    else:
        snap_label = ("%d" % snap) if snap >= 0 else ("last")
        out_csv = os.path.join(_HERE, "abaqus_field_%s_snap%s.csv" % (
            condition, snap_label))

    try:
        export_di_field(condition, snap, out_csv)
    except FileNotFoundError as e:
        print("  ERROR: %s" % e)
        sys.exit(1)

    # 3. Optionally regenerate INP
    if args.regen_inp:
        print("\n[3/3] Regenerating biofilm_3tooth_%s.inp ..." % condition)
        regen_inp(out_csv, condition, dry_run=args.dry_run)
    else:
        print("\n[3/3] DI field export complete.")
        print("  To regenerate INP with this field:")
        print("    python3 tmcmc_to_fem_coupling.py --condition %s --snapshot %d --regen-inp" % (
            condition, snap))
        print("  Or pass manually to assembly:")
        print("    python3 biofilm_3tooth_assembly.py --di-csv %s" % out_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
