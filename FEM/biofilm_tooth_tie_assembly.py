#!/usr/bin/env python3
"""
biofilm_tooth_tie_assembly.py
─────────────────────────────────────────────────────────────────────
Two-layer biofilm + tooth model with *Tie constraint.

Layer 1 — Tooth shell : S3 elements from STL surface triangles
Layer 2 — Biofilm solid: C3D4 conformal tets (existing pipeline)
Interface             : *Tie (biofilm INNER → tooth shell surface)

Improvements over ENCASTRE-only model:
  - Tooth can deform under load (E_dentin ≈ 18.6 GPa)
  - Stress concentrations at tooth-biofilm interface are captured
  - Reaction forces at tooth base show total load transmission
  - Compatible with downstream mandible-level models (tooth ↔ PDL ↔ bone)

Convention:
  - All units in mm / N / MPa (standard Abaqus unit system)
  - Node numbering: tooth first (1..N_tooth), biofilm after (N_tooth+1..)
  - Tooth shell thickness set by --shell-thick (default 0.5 mm)

Usage
-----
  # Single tooth (T23 crown):
  python3 biofilm_tooth_tie_assembly.py --tooth T23

  # With custom DI field:
  python3 biofilm_tooth_tie_assembly.py --tooth T23 \\
      --di-csv abaqus_field_dh_3d.csv --out two_layer_T23.inp

  # All 3 teeth:
  python3 biofilm_tooth_tie_assembly.py --all-teeth
"""

from __future__ import print_function, division
import sys, os, argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from biofilm_conformal_tet import (
    read_stl, deduplicate_vertices, compute_vertex_normals,
    laplacian_smooth_offset, build_tet_mesh,
    read_di_csv, assign_di_bins,
    compute_outer_face_loads,
)

# ── Material properties ──────────────────────────────────────────────────────
# Dentin: Kinney et al. (2003), Dental Materials
E_DENTIN_MPA = 18600.0     # Young's modulus [MPa]
NU_DENTIN    = 0.31         # Poisson's ratio

# Enamel (for reference, not used by default):
# E_ENAMEL_MPA = 84100.0
# NU_ENAMEL    = 0.33

TOOTH_INFO = {
    "T23": {"stl_key": "P1_Tooth_23", "role": "crown"},
    "T30": {"stl_key": "P1_Tooth_30", "role": "slit (T30 side)"},
    "T31": {"stl_key": "P1_Tooth_31", "role": "slit (T31 side)"},
}


# ── INP helper functions ─────────────────────────────────────────────────────

def _write_nset(f, name, ids_1based):
    f.write("*Nset, nset=%s\n" % name)
    row = []
    for idx in ids_1based:
        row.append("%d" % idx)
        if len(row) == 16:
            f.write(", ".join(row) + ",\n")
            row = []
    if row:
        f.write(", ".join(row) + "\n")


def _write_elset(f, name, ids_1based):
    f.write("*Elset, elset=%s\n" % name)
    row = []
    for idx in ids_1based:
        row.append("%d" % idx)
        if len(row) == 16:
            f.write(", ".join(row) + ",\n")
            row = []
    if row:
        f.write(", ".join(row) + "\n")


# ── Tooth base detection ─────────────────────────────────────────────────────

def detect_tooth_base(nodes, frac=0.10):
    """
    Find nodes near the cervical (root) end of the tooth.

    The tooth's long axis is approximately vertical (z-axis in OpenJaw).
    The cervical end is at max z (root direction for mandibular teeth)
    or min z depending on orientation. We fix the top `frac` of z-range.

    Returns (base_ids,) 0-indexed node indices.
    """
    z = nodes[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min
    # OpenJaw mandibular: root is at higher z (closer to jaw bone)
    z_thresh = z_max - frac * z_range
    base = np.where(z >= z_thresh)[0]
    if len(base) < 3:
        # Fall back: try lower end
        z_thresh_low = z_min + frac * z_range
        base = np.where(z <= z_thresh_low)[0]
    return base


# ── Process tooth (shell) ────────────────────────────────────────────────────

def process_tooth_shell(stl_path, dedup_tol=1e-4):
    """
    Build tooth shell mesh (S3) from STL.

    Returns dict:
      nodes     : (V, 3)    deduplicated vertices
      faces     : (F, 3)    triangle connectivity (0-indexed)
      base_ids  : (B,)      cervical base node indices (0-indexed)
    """
    print("  [TOOTH-SHELL] Reading %s ..." % os.path.basename(stl_path))
    faces_verts, _ = read_stl(stl_path)
    verts, faces = deduplicate_vertices(faces_verts, tol=dedup_tol)
    print("    %d nodes, %d S3 triangles" % (len(verts), len(faces)))
    base = detect_tooth_base(verts)
    print("    Cervical base: %d nodes (%.1f%%)" % (
        len(base), 100.0 * len(base) / len(verts)))
    return {"nodes": verts, "faces": faces, "base_ids": base}


# ── Process biofilm (conformal C3D4) ────────────────────────────────────────

def process_biofilm(stl_path, di_csv_path, cfg):
    """Thin wrapper around the existing pipeline."""
    print("  [BIOFILM] Reading %s ..." % os.path.basename(stl_path))
    faces_verts, face_normals_stored = read_stl(stl_path)
    verts_inner, faces = deduplicate_vertices(faces_verts, tol=cfg["dedup_tol"])
    V, F = len(verts_inner), len(faces)
    print("    %d unique verts, %d faces" % (V, F))

    vnorms_inner = compute_vertex_normals(verts_inner, faces, face_normals_stored)
    verts_outer_raw = verts_inner + cfg["thickness"] * vnorms_inner

    if cfg["smooth_iter"] > 0:
        verts_outer = laplacian_smooth_offset(
            verts_outer_raw, verts_inner, faces,
            iterations=cfg["smooth_iter"], lam=0.5)
    else:
        verts_outer = verts_outer_raw

    vnorms_outer = compute_vertex_normals(verts_outer, faces)

    nodes, tets, tet_layers, inner_nodes, outer_nodes = build_tet_mesh(
        verts_inner, verts_outer, faces, cfg["n_layers"])

    # Fix negative volumes
    tv = nodes[tets]
    a = tv[:, 1] - tv[:, 0]; b = tv[:, 2] - tv[:, 0]; c = tv[:, 3] - tv[:, 0]
    vols = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    bad = vols < 0
    if bad.any():
        tets[bad, 2], tets[bad, 3] = tets[bad, 3].copy(), tets[bad, 2].copy()
        print("    Fixed %d negative-volume tets" % bad.sum())

    print("    %d nodes, %d C3D4, %d layers" % (len(nodes), len(tets), cfg["n_layers"]))

    # DI bins
    xs_field, di_vals_field = read_di_csv(di_csv_path)
    tet_bins, bin_E_stiff = assign_di_bins(
        tet_layers, cfg["n_layers"],
        xs_field, di_vals_field,
        cfg["n_bins"], cfg["di_scale"], cfg["di_exp"],
        cfg["e_max"], cfg["e_min"])

    return {
        "nodes": nodes, "tets": tets, "tet_bins": tet_bins,
        "inner_nodes": inner_nodes, "outer_nodes": outer_nodes,
        "verts_outer": verts_outer, "faces": faces,
        "vnorms_outer": vnorms_outer,
        "bin_E_stiff": bin_E_stiff,
    }


# ── Combined two-layer INP writer ───────────────────────────────────────────

def write_two_layer_inp(out_path, tooth_name, tooth_data, bio_data,
                        bin_E_stiff, n_bins, nu, pressure, shell_thick,
                        di_csv_path, thickness, n_layers):
    """
    Write Abaqus INP with tooth shell + biofilm solid + Tie constraint.

    Node numbering: tooth nodes 1..N_tooth, biofilm nodes N_tooth+1..N_total
    Element numbering: tooth S3 1..F_tooth, biofilm C3D4 F_tooth+1..F_total
    """
    N_tooth = len(tooth_data["nodes"])
    N_bio   = len(bio_data["nodes"])
    F_tooth = len(tooth_data["faces"])
    E_bio   = len(bio_data["tets"])
    N_total = N_tooth + N_bio
    E_total = F_tooth + E_bio

    n_off = N_tooth   # node offset for biofilm
    e_off = F_tooth   # element offset for biofilm

    with open(out_path, "w") as f:

        # ── Header ───────────────────────────────────────────────────────
        f.write("** biofilm_tooth_tie_assembly.py – Two-layer Tie model\n")
        f.write("** Tooth: %s (S3 shell, E=%.0f MPa, nu=%.2f)\n" % (
            tooth_name, E_DENTIN_MPA, NU_DENTIN))
        f.write("** Biofilm: C3D4 conformal (%d layers, t=%.3f mm)\n" % (
            n_layers, thickness))
        f.write("** DI CSV: %s\n" % os.path.basename(di_csv_path))
        f.write("** Shell thickness: %.3f mm\n" % shell_thick)
        f.write("** Tooth: %d nodes, %d S3 | Biofilm: %d nodes, %d C3D4\n" % (
            N_tooth, F_tooth, N_bio, E_bio))
        f.write("** Interface: *Tie (biofilm INNER → tooth shell)\n")
        f.write("** Pressure: %.3g Pa (= %.4g MPa) on biofilm outer\n" % (
            pressure, pressure * 1e-6))
        f.write("**\n")
        f.write("*Heading\n")
        f.write(" Two-layer biofilm+tooth: %s (Tie constraint)\n" % tooth_name)
        f.write("**\n")

        # ── Nodes ────────────────────────────────────────────────────────
        f.write("** ===== NODES =====\n")
        f.write("** Tooth nodes: 1 .. %d\n" % N_tooth)
        f.write("** Biofilm nodes: %d .. %d\n" % (n_off + 1, N_total))
        f.write("*Node\n")

        # Tooth nodes
        for i, (x, y, z) in enumerate(tooth_data["nodes"]):
            f.write(" %d, %.8g, %.8g, %.8g\n" % (i + 1, x, y, z))

        # Biofilm nodes
        for i, (x, y, z) in enumerate(bio_data["nodes"]):
            f.write(" %d, %.8g, %.8g, %.8g\n" % (n_off + i + 1, x, y, z))
        f.write("**\n")

        # ── Elements ─────────────────────────────────────────────────────
        f.write("** ===== ELEMENTS =====\n")

        # Tooth: S3 shell elements
        f.write("*Element, type=S3\n")
        for i, tri in enumerate(tooth_data["faces"]):
            n1, n2, n3 = tri + 1  # 1-based tooth numbering
            f.write(" %d, %d, %d, %d\n" % (i + 1, n1, n2, n3))
        f.write("**\n")

        # Biofilm: C3D4 solid elements
        f.write("*Element, type=C3D4\n")
        for i, tet in enumerate(bio_data["tets"]):
            n1, n2, n3, n4 = tet + n_off + 1  # 1-based + offset
            f.write(" %d, %d, %d, %d, %d\n" % (e_off + i + 1, n1, n2, n3, n4))
        f.write("**\n")

        # ── Node Sets ────────────────────────────────────────────────────
        f.write("** ===== NODE SETS =====\n")

        # Tooth sets
        _write_nset(f, "TOOTH_ALL", np.arange(1, N_tooth + 1))
        f.write("**\n")
        _write_nset(f, "TOOTH_BASE",
                    tooth_data["base_ids"] + 1)
        f.write("**\n")

        # Biofilm sets
        _write_nset(f, "BIO_INNER",
                    bio_data["inner_nodes"] + n_off + 1)
        f.write("**\n")
        _write_nset(f, "BIO_OUTER",
                    bio_data["outer_nodes"] + n_off + 1)
        f.write("**\n")
        _write_nset(f, "BIO_ALL",
                    np.arange(n_off + 1, N_total + 1))
        f.write("**\n")

        # ── Element Sets ─────────────────────────────────────────────────
        f.write("** ===== ELEMENT SETS =====\n")
        _write_elset(f, "TOOTH_ELEMS", np.arange(1, F_tooth + 1))
        f.write("**\n")

        # Biofilm DI-bin elsets
        used_bins = set()
        bin_labels = [[] for _ in range(n_bins)]
        for i, b in enumerate(bio_data["tet_bins"]):
            bin_labels[b].append(e_off + i + 1)
            used_bins.add(b)

        for b in sorted(used_bins):
            _write_elset(f, "BIO_BIN_%02d" % b, bin_labels[b])
            f.write("**\n")

        _write_elset(f, "BIO_ELEMS",
                     np.arange(e_off + 1, E_total + 1))
        f.write("**\n")

        # ── Materials ────────────────────────────────────────────────────
        f.write("** ===== MATERIALS =====\n")

        # Tooth: dentin
        f.write("*Material, name=MAT_DENTIN\n")
        f.write("*Elastic\n")
        f.write(" %.1f, %.4f\n" % (E_DENTIN_MPA, NU_DENTIN))
        f.write("**\n")

        # Biofilm: per DI-bin (same as existing pipeline)
        for b in sorted(used_bins):
            E_MPa = bin_E_stiff[b] * 1e-6  # Pa → MPa
            f.write("*Material, name=MAT_BIO_%02d\n" % b)
            f.write("*Elastic\n")
            f.write(" %.6e, %.4f\n" % (E_MPa, nu))
            f.write("**\n")

        # ── Sections ─────────────────────────────────────────────────────
        f.write("** ===== SECTIONS =====\n")

        # Tooth: shell section
        f.write("*Shell Section, elset=TOOTH_ELEMS, material=MAT_DENTIN\n")
        f.write(" %.4f, 5\n" % shell_thick)   # thickness, # integration points
        f.write("**\n")

        # Biofilm: solid sections per bin
        for b in sorted(used_bins):
            f.write("*Solid Section, elset=BIO_BIN_%02d, material=MAT_BIO_%02d\n" % (b, b))
            f.write(",\n")
            f.write("**\n")

        # ── Surfaces for Tie ─────────────────────────────────────────────
        f.write("** ===== TIE CONSTRAINT (biofilm INNER → tooth shell) =====\n")
        f.write("** Biofilm inner nodes are at the same positions as tooth shell nodes.\n")
        f.write("** position tolerance should cover the near-zero gap between them.\n")

        # Tooth surface: element-based surface (shell outer face = SPOS)
        f.write("*Surface, type=ELEMENT, name=TOOTH_SURF\n")
        f.write(" TOOTH_ELEMS, SPOS\n")
        f.write("**\n")

        # Biofilm inner surface: node-based surface
        f.write("*Surface, type=NODE, name=BIO_INNER_SURF\n")
        f.write(" BIO_INNER\n")
        f.write("**\n")

        # Tie constraint
        f.write("*Tie, name=TOOTH_BIO_TIE, position tolerance=0.1, adjust=YES\n")
        f.write(" BIO_INNER_SURF, TOOTH_SURF\n")
        f.write("**\n")

        # ── Boundary Conditions ──────────────────────────────────────────
        f.write("** ===== BOUNDARY CONDITIONS =====\n")
        f.write("** Tooth base (cervical region) fixed\n")
        f.write("*Boundary\n")
        f.write(" TOOTH_BASE, ENCASTRE\n")
        f.write("**\n")

        # ── Step: LOAD ───────────────────────────────────────────────────
        f.write("** ===== STEP =====\n")
        f.write("*Step, name=LOAD, nlgeom=NO\n")
        f.write(" Inward pressure %.4g MPa on biofilm outer face\n" % (
            pressure * 1e-6))
        f.write("*Static\n")
        f.write(" 0.1, 1.0, 1e-5, 1.0\n")
        f.write("**\n")

        # Cload on biofilm outer face
        f.write("** Pressure = %.3g Pa (= %.4g MPa)\n" % (
            pressure, pressure * 1e-6))
        f.write("*Cload\n")
        outer_forces = compute_outer_face_loads(
            bio_data["verts_outer"], bio_data["faces"],
            pressure, bio_data["vnorms_outer"])
        for vi, fvec in sorted(outer_forces.items()):
            ni = int(bio_data["outer_nodes"][vi]) + n_off + 1
            if abs(fvec[0]) > 1e-20:
                f.write(" %d, 1, %.8g\n" % (ni, fvec[0]))
            if abs(fvec[1]) > 1e-20:
                f.write(" %d, 2, %.8g\n" % (ni, fvec[1]))
            if abs(fvec[2]) > 1e-20:
                f.write(" %d, 3, %.8g\n" % (ni, fvec[2]))
        f.write("**\n")

        # ── Output requests ──────────────────────────────────────────────
        f.write("*Output, field\n")
        f.write("*Node Output\n")
        f.write(" U, RF\n")
        f.write("*Element Output\n")
        f.write(" S, E, MISES\n")
        f.write("**\n")
        f.write("*End Step\n")

    print("\n  Two-layer INP written: %s" % out_path)
    print("  Tooth: %d nodes, %d S3 (shell thick=%.3f mm)" % (
        N_tooth, F_tooth, shell_thick))
    print("  Biofilm: %d nodes, %d C3D4" % (N_bio, E_bio))
    print("  Total: %d nodes, %d elements" % (N_total, E_total))


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-layer biofilm + tooth Tie model INP generator")
    p.add_argument("--stl-root", default="external_tooth_models/OpenJaw_Dataset/Patient_1",
                   help="Root dir with Teeth/ sub-dir")
    p.add_argument("--tooth", default="T23",
                   choices=list(TOOTH_INFO.keys()),
                   help="Which tooth to process")
    p.add_argument("--all-teeth", action="store_true",
                   help="Process all 3 teeth (separate INP each)")
    p.add_argument("--di-csv", default="abaqus_field_dh_3d.csv")
    p.add_argument("--out", default=None,
                   help="Output INP path (auto-named if not given)")
    p.add_argument("--thickness", type=float, default=0.5,
                   help="Biofilm layer thickness [mm]")
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--e-max", type=float, default=10e9,
                   help="Max biofilm E [Pa]")
    p.add_argument("--e-min", type=float, default=0.5e9,
                   help="Min biofilm E [Pa]")
    p.add_argument("--di-scale", type=float, default=0.025778)
    p.add_argument("--di-exp", type=float, default=2.0)
    p.add_argument("--nu", type=float, default=0.30,
                   help="Biofilm Poisson's ratio")
    p.add_argument("--pressure", type=float, default=1.0e6,
                   help="Outer face pressure [Pa]")
    p.add_argument("--shell-thick", type=float, default=0.5,
                   help="Tooth shell element thickness [mm]")
    p.add_argument("--base-frac", type=float, default=0.10,
                   help="Fraction of tooth z-range for cervical base BC")
    p.add_argument("--smooth-iter", type=int, default=3)
    p.add_argument("--dedup-tol", type=float, default=1e-4)
    return p.parse_args()


def run_one_tooth(tooth_key, args):
    """Process one tooth and write two-layer INP."""
    info = TOOTH_INFO[tooth_key]
    stl_name = "%s.stl" % info["stl_key"]
    stl_path = os.path.join(args.stl_root, "Teeth", stl_name)
    if not os.path.exists(stl_path):
        raise FileNotFoundError("STL not found: %s" % stl_path)

    print("\n" + "=" * 62)
    print("  Two-layer Tie model: %s (%s)" % (tooth_key, info["role"]))
    print("=" * 62)

    # 1. Tooth shell mesh
    tooth_data = process_tooth_shell(stl_path, dedup_tol=args.dedup_tol)

    # Override base detection fraction
    tooth_data["base_ids"] = detect_tooth_base(
        tooth_data["nodes"], frac=args.base_frac)
    print("    Base nodes (frac=%.2f): %d" % (
        args.base_frac, len(tooth_data["base_ids"])))

    # 2. Biofilm conformal mesh
    cfg = {
        "thickness": args.thickness,
        "n_layers": args.n_layers,
        "n_bins": args.n_bins,
        "e_max": args.e_max,
        "e_min": args.e_min,
        "di_scale": args.di_scale,
        "di_exp": args.di_exp,
        "smooth_iter": args.smooth_iter,
        "dedup_tol": args.dedup_tol,
    }
    bio_data = process_biofilm(stl_path, args.di_csv, cfg)

    # 3. Write two-layer INP
    out_path = args.out or "two_layer_%s.inp" % tooth_key
    write_two_layer_inp(
        out_path, tooth_key, tooth_data, bio_data,
        bio_data["bin_E_stiff"], args.n_bins, args.nu,
        args.pressure, args.shell_thick,
        args.di_csv, args.thickness, args.n_layers,
    )
    return out_path


def main():
    args = parse_args()

    if args.all_teeth:
        for key in TOOTH_INFO:
            args.out = None  # auto-name each
            run_one_tooth(key, args)
    else:
        out = run_one_tooth(args.tooth, args)
        print("\n  To run in Abaqus:")
        job = "TwoLayer_%s" % args.tooth
        print("  abaqus job=%s input=%s cpus=4 interactive" % (
            job, os.path.abspath(out)))


if __name__ == "__main__":
    main()
