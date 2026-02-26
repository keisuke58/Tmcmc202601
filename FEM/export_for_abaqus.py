#!/usr/bin/env python3
"""
export_for_abaqus.py  –  Export FEM biofilm fields for Abaqus coupling

This script reads 2D or 3D FEM results from:

  snapshots_phi.npy   (n_snap, 5, Nx, Ny[, Nz])
  snapshots_t.npy     (n_snap,)
  mesh_x.npy          (Nx,)
  mesh_y.npy          (Ny,)
  mesh_z.npy          (Nz,)          [3D only]

and writes a CSV file with coordinates and scalar fields that can be
consumed from an Abaqus Python script on a structured grid.

Exported scalar fields:
  - phi_pg        (P. gingivalis volume fraction)
  - di            (Dysbiotic Index)
  - phi_tot       (total bacterial volume fraction)
  - r_pg          (P. gingivalis fraction = phi_pg / phi_tot)

Usage (from Tmcmc202601/FEM/):

  python export_for_abaqus.py \
      --results-dir _results_2d/dh_baseline \
      --snapshot-index -1 \
      --out-csv abaqus_field_dh_2d.csv

  python export_for_abaqus.py \
      --results-dir _results_3d/commensal_static \
      --snapshot-index -1 \
      --out-csv abaqus_field_cs_3d.csv
"""

import argparse
from pathlib import Path

import numpy as np


def _compute_di(phi_all: np.ndarray) -> np.ndarray:
    """
    Compute Dysbiotic Index per node for a stack of 5 species.

    phi_all: (..., 5) array of volume fractions (no void included).
    """
    # sum over species axis
    sum_phi = np.sum(phi_all, axis=-1)
    # avoid division by zero
    sum_phi_safe = np.where(sum_phi > 0.0, sum_phi, 1.0)
    p = phi_all / sum_phi_safe[..., None]
    # entropy H = -sum p_i ln p_i (ignore p_i <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log(p), 0.0)
    H = -np.sum(p * log_p, axis=-1)
    H_max = np.log(5.0)
    di = 1.0 - H / H_max
    return di


def export_fields(results_dir: Path, snapshot_index: int, out_csv: Path) -> None:
    phi_path = results_dir / "snapshots_phi.npy"
    if phi_path.exists():
        snaps_phi = np.load(phi_path)
        snaps_t = np.load(results_dir / "snapshots_t.npy")
        x = np.load(results_dir / "mesh_x.npy")
        if snapshot_index < 0:
            idx = snaps_phi.shape[0] + snapshot_index
        else:
            idx = snapshot_index
        if idx < 0 or idx >= snaps_phi.shape[0]:
            raise IndexError(
                f"snapshot_index {snapshot_index} out of range for n_snap={snaps_phi.shape[0]}"
            )
        phi_all = snaps_phi[idx]
        t = snaps_t[idx]
        ndim = phi_all.ndim
        if ndim == 3:
            y = np.load(results_dir / "mesh_y.npy")
            Nx = phi_all.shape[1]
            Ny = phi_all.shape[2]
            phi_all_nodes = np.transpose(phi_all, (1, 2, 0))
            di = _compute_di(phi_all_nodes)
            phi_pg = phi_all_nodes[:, :, 4]
            phi_tot = np.sum(phi_all_nodes, axis=2)
            phi_tot_safe = np.where(phi_tot > 0.0, phi_tot, 1.0)
            r_pg = phi_pg / phi_tot_safe
            xx, yy = np.meshgrid(x, y, indexing="ij")
            coords = np.stack([xx, yy], axis=-1)
            rows = []
            for ix in range(Nx):
                for iy in range(Ny):
                    rows.append(
                        (
                            coords[ix, iy, 0],
                            coords[ix, iy, 1],
                            float(phi_pg[ix, iy]),
                            float(di[ix, iy]),
                            float(phi_tot[ix, iy]),
                            float(r_pg[ix, iy]),
                        )
                    )
            header = "x,y,phi_pg,di,phi_tot,r_pg,t"
            with out_csv.open("w", encoding="utf-8") as f:
                f.write(f"# snapshot_index={idx}, t={t:.6e}\n")
                f.write(header + "\n")
                for xv, yv, pv, dv, fv, rv in rows:
                    f.write(f"{xv:.8e},{yv:.8e},{pv:.8e},{dv:.8e},{fv:.8e},{rv:.8e},{t:.8e}\n")
        elif ndim == 4:
            y = np.load(results_dir / "mesh_y.npy")
            z = np.load(results_dir / "mesh_z.npy")
            Nx = phi_all.shape[1]
            Ny = phi_all.shape[2]
            Nz = phi_all.shape[3]
            phi_all_nodes = np.transpose(phi_all, (1, 2, 3, 0))
            di = _compute_di(phi_all_nodes)
            phi_pg = phi_all_nodes[:, :, :, 4]
            phi_tot = np.sum(phi_all_nodes, axis=3)
            phi_tot_safe = np.where(phi_tot > 0.0, phi_tot, 1.0)
            r_pg = phi_pg / phi_tot_safe
            xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
            coords = np.stack([xx, yy, zz], axis=-1)
            header = "x,y,z,phi_pg,di,phi_tot,r_pg,t"
            with out_csv.open("w", encoding="utf-8") as f:
                f.write(f"# snapshot_index={idx}, t={t:.6e}\n")
                f.write(header + "\n")
                for ix in range(Nx):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            xv = coords[ix, iy, iz, 0]
                            yv = coords[ix, iy, iz, 1]
                            zv = coords[ix, iy, iz, 2]
                            pv = float(phi_pg[ix, iy, iz])
                            dv = float(di[ix, iy, iz])
                            fv = float(phi_tot[ix, iy, iz])
                            rv = float(r_pg[ix, iy, iz])
                            f.write(
                                f"{xv:.8e},{yv:.8e},{zv:.8e},{pv:.8e},{dv:.8e},{fv:.8e},{rv:.8e},{t:.8e}\n"
                            )
        else:
            raise ValueError(f"Unexpected snapshots_phi dimensionality: {snaps_phi.shape}")
    else:
        snaps_G = np.load(results_dir / "snapshots_G.npy")
        snaps_t = np.load(results_dir / "snapshots_t.npy")
        x = np.load(results_dir / "mesh_x.npy")
        if snapshot_index < 0:
            idx = snaps_G.shape[0] + snapshot_index
        else:
            idx = snapshot_index
        if idx < 0 or idx >= snaps_G.shape[0]:
            raise IndexError(
                f"snapshot_index {snapshot_index} out of range for n_snap={snaps_G.shape[0]}"
            )
        G = snaps_G[idx]
        t = snaps_t[idx]
        phi_nodes = G[:, 0:5]
        di = _compute_di(phi_nodes)
        phi_pg = phi_nodes[:, 4]
        phi_tot = np.sum(phi_nodes, axis=1)
        phi_tot_safe = np.where(phi_tot > 0.0, phi_tot, 1.0)
        r_pg = phi_pg / phi_tot_safe
        y_vals = np.array([0.0, 1.0])
        header = "x,y,phi_pg,di,phi_tot,r_pg,t"
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(f"# snapshot_index={idx}, t={t:.6e}\n")
            f.write(header + "\n")
            for ix, xv in enumerate(x):
                for yv in y_vals:
                    pv = float(phi_pg[ix])
                    dv = float(di[ix])
                    fv = float(phi_tot[ix])
                    rv = float(r_pg[ix])
                    f.write(f"{xv:.8e},{yv:.8e},{pv:.8e},{dv:.8e},{fv:.8e},{rv:.8e},{t:.8e}\n")
        print(f"Exported 1D fields from {results_dir} (snapshot {idx}, t={t:.4f}) to {out_csv}")
        return
    print(f"Exported fields from {results_dir} (snapshot {idx}, t={t:.4f}) to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Export FEM fields (phi_pg, DI) for Abaqus coupling")
    ap.add_argument(
        "--results-dir",
        required=True,
        help="FEM results directory (_results_2d/... or _results_3d/...)",
    )
    ap.add_argument(
        "--snapshot-index", type=int, default=-1, help="Snapshot index (default: -1, last)"
    )
    ap.add_argument(
        "--out-csv", required=True, help="Output CSV path relative to current directory"
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results_dir not found: {results_dir}")

    out_csv = Path(args.out_csv)
    export_fields(results_dir, args.snapshot_index, out_csv)


if __name__ == "__main__":
    main()
