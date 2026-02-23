#!/usr/bin/env python3
"""
biofilm_3tooth_assembly.py
──────────────────────────────────────────────────────────────────────────────
Conformal tetrahedral biofilm around P1_Tooth_23, P1_Tooth_30, P1_Tooth_31
combined into ONE Abaqus INP → one .odb output.

Crown:  T23 – biofilm wraps full tooth surface (hollow, inner = STL exact)
"Slit": T30 + T31 – each tooth gets its own full conformal biofilm;
        the approximal (inter-proximal) face biofilms together represent
        the gingival pocket / slit region.

Mesh strategy
  Tooth STL edge: min≈0.27mm, mean≈0.46mm
  Biofilm in-plane: same as STL (exact conformity, no penetration)
  Biofilm layer thickness: --thickness / --n-layers = 0.5/8 = 0.0625mm
  → layer resolution 7× finer than tooth-surface element size ✓

Combined INP structure
  Node numbering : 1-based, offset per tooth
  Element numbering: 1-based, offset per tooth
  Nsets  : T23_INNER, T23_OUTER, T30_INNER, T30_OUTER, T31_INNER, T31_OUTER
  Elsets : T23_BIN_XX, T30_BIN_XX, T31_BIN_XX
  Materials: MAT_ANISO_XX (shared, one definition per bin)
  Sections : per elset
  Step    : LOAD – inner faces ENCASTRE, outer face pressure (Cload)

Usage
  python3 biofilm_3tooth_assembly.py \\
      [--stl-root  external_tooth_models/OpenJaw_Dataset/Patient_1] \\
      [--di-csv    abaqus_field_dh_3d.csv] \\
      [--out       biofilm_3tooth.inp] \\
      [--thickness 0.5] \\
      [--n-layers  8] \\
      [--n-bins    20] \\
      [--pressure  1.0e6] \\
      [--smooth-iter 3] \\
      [--run]             ← also submit Abaqus job after writing INP
      [--job-name  BioFilm3T]
"""

from __future__ import print_function, division
import sys, os, argparse
import numpy as np

# ── Import from biofilm_conformal_tet.py (same directory) ────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from biofilm_conformal_tet import (
    read_stl, deduplicate_vertices, compute_vertex_normals,
    laplacian_smooth_offset, build_tet_mesh,
    read_di_csv, assign_di_bins,
    compute_outer_face_loads,
)

# ============================================================================
# Approximal (slit / inter-proximal) face detection
# ============================================================================

def get_approximal_info(td_src, td_dst, threshold=0.30, max_dist=None):
    """
    Identify which outer faces of td_src point toward td_dst (approximal direction).

    Algorithm:
      1. Compute inter-tooth direction as unit vector between inner-surface centroids.
      2. Compute normalised outward face normals of td_src outer surface.
      3. Faces whose normal · direction > threshold are approximal (normal filter).
      4. If max_dist is set, further restrict to nodes within max_dist mm of
         the nearest outer-surface node on td_dst (distance filter).
      5. Node mask = union of vertices from all surviving approximal faces.

    Parameters
    ----------
    td_src, td_dst : tooth data dicts (with keys nodes, inner_nodes, verts_outer, faces)
    threshold      : dot-product cutoff ∈ (0,1); 0.3 ≈ 72° cone around approx direction
    max_dist       : optional float – maximum distance (mm) from any td_dst outer node;
                     nodes farther than this are excluded from APPROX.
                     Recommended: 2.0–5.0 mm for typical inter-proximal pocket geometry.

    Returns
    -------
    face_mask : (F,) bool
    node_mask : (V,) bool  – V = len(verts_outer)
    direction : (3,) float – unit vector src → dst
    """
    from scipy.spatial import cKDTree

    c_src = td_src["nodes"][td_src["inner_nodes"]].mean(axis=0)
    c_dst = td_dst["nodes"][td_dst["inner_nodes"]].mean(axis=0)
    d = c_dst - c_src
    d /= np.linalg.norm(d)

    verts = td_src["verts_outer"]
    faces = td_src["faces"]

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)                          # (F,3)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn /= np.where(fn_len < 1e-14, 1.0, fn_len)

    face_mask = (fn @ d) > threshold                          # (F,)

    node_mask = np.zeros(len(verts), dtype=bool)
    if face_mask.any():
        node_mask[faces[face_mask].ravel()] = True            # (V,)

    # ── Distance filter (P7 fix) ──────────────────────────────────────────────
    if max_dist is not None and node_mask.any():
        dst_verts = td_dst["verts_outer"]                     # (V_dst, 3)
        tree      = cKDTree(dst_verts)
        src_approx_idx = np.where(node_mask)[0]
        dists, _  = tree.query(verts[src_approx_idx], k=1)
        keep      = dists <= max_dist                         # (n_approx,)

        node_mask_new = np.zeros(len(verts), dtype=bool)
        node_mask_new[src_approx_idx[keep]] = True

        # Re-derive face_mask: keep only faces where ALL nodes pass distance filter
        face_mask_new = np.zeros(len(faces), dtype=bool)
        for fi in np.where(face_mask)[0]:
            if node_mask_new[faces[fi]].all():
                face_mask_new[fi] = True

        n_before = node_mask.sum()
        n_after  = node_mask_new.sum()
        print("  [P7-dist] max_dist=%.1f mm: %d → %d approx nodes (%.0f%% retained)" % (
            max_dist, n_before, n_after, 100.0 * n_after / max(n_before, 1)))

        node_mask = node_mask_new
        face_mask = face_mask_new

    return face_mask, node_mask, d

# ============================================================================
# Combined INP writer
# ============================================================================

def _write_nset_block(f, name, indices_1based):
    f.write("*Nset, nset=%s\n" % name)
    row = []
    for idx in indices_1based:
        row.append("%d" % idx)
        if len(row) == 16:
            f.write(", ".join(row) + ",\n")
            row = []
    if row:
        f.write(", ".join(row) + "\n")


def _write_elset_block(f, name, labels_1based):
    f.write("*Elset, elset=%s\n" % name)
    row = []
    for lab in labels_1based:
        row.append("%d" % lab)
        if len(row) == 16:
            f.write(", ".join(row) + ",\n")
            row = []
    if row:
        f.write(", ".join(row) + "\n")


def write_combined_inp(out_path, teeth_data, bin_E_stiff, n_bins,
                       aniso_ratio, nu, pressure, bc_mode,
                       di_csv_path, thickness, n_layers,
                       slit_approx=None):
    """
    teeth_data : list of dict, one per tooth:
        name         : str  e.g. 'T23'
        nodes        : (N,3) float  – node coordinates
        tets         : (E,4) int    – 0-indexed tet connectivity
        tet_bins     : (E,)  int    – DI bin per tet
        inner_nodes  : (V,)  int    – 0-indexed inner-face node indices
        outer_nodes  : (V,)  int    – 0-indexed outer-face node indices
        verts_outer  : (V,3) float  – outer surface vertices (for Cload)
        faces        : (F,3) int    – surface face connectivity
        vnorms_outer : (V,3) float  – outer surface vertex normals
    """
    G_factor = 1.0 / (2.0 * (1.0 + nu))
    Pa_to_Nmm2 = 1.0e-6

    # Pre-compute total counts for header
    total_nodes = sum(len(d["nodes"]) for d in teeth_data)
    total_tets  = sum(len(d["tets"])  for d in teeth_data)

    with open(out_path, "w") as f:

        # ── Header ────────────────────────────────────────────────────────────
        f.write("** biofilm_3tooth_assembly.py – 3-tooth conformal C3D4 biofilm\n")
        f.write("** Teeth: %s\n" % ", ".join(d["name"] for d in teeth_data))
        f.write("** DI CSV : %s\n" % os.path.basename(di_csv_path))
        f.write("** Thickness: %.3f mm  Layers: %d\n" % (thickness, n_layers))
        f.write("** Total nodes: %d  Total C3D4: %d\n" % (total_nodes, total_tets))
        f.write("** Inner face BC: ENCASTRE (tooth adhesion)\n")
        f.write("** Outer face load: %.3g Pa inward pressure\n" % pressure)
        f.write("**\n")
        f.write("*Heading\n")
        f.write(" 3-tooth conformal biofilm: %s\n" % ", ".join(d["name"] for d in teeth_data))
        f.write("**\n")

        # ── Nodes (all teeth, sequential 1-based numbering) ───────────────────
        f.write("** ===== NODES =====\n")
        f.write("*Node\n")
        node_offset = 0
        tooth_node_offsets = []
        for td in teeth_data:
            tooth_node_offsets.append(node_offset)
            for i, (x, y, z) in enumerate(td["nodes"]):
                f.write(" %d, %.8g, %.8g, %.8g\n" % (node_offset + i + 1, x, y, z))
            node_offset += len(td["nodes"])
        f.write("**\n")

        # ── Elements (all teeth, sequential 1-based) ──────────────────────────
        f.write("** ===== ELEMENTS =====\n")
        f.write("*Element, type=C3D4\n")
        elem_offset = 0
        tooth_elem_offsets = []
        for td, noff in zip(teeth_data, tooth_node_offsets):
            tooth_elem_offsets.append(elem_offset)
            for i, tet in enumerate(td["tets"]):
                n1, n2, n3, n4 = tet + noff + 1   # 1-based + node offset
                f.write(" %d, %d, %d, %d, %d\n" % (
                    elem_offset + i + 1, n1, n2, n3, n4))
            elem_offset += len(td["tets"])
        f.write("**\n")

        # ── Node sets ─────────────────────────────────────────────────────────
        f.write("** ===== NODE SETS =====\n")
        for td, noff in zip(teeth_data, tooth_node_offsets):
            name = td["name"]
            _write_nset_block(f, "%s_INNER" % name,
                              td["inner_nodes"] + noff + 1)
            f.write("**\n")
            _write_nset_block(f, "%s_OUTER" % name,
                              td["outer_nodes"] + noff + 1)
            f.write("**\n")

        # Combined inner/outer sets (for reference)
        all_inner = []
        all_outer = []
        for td, noff in zip(teeth_data, tooth_node_offsets):
            all_inner.extend((td["inner_nodes"] + noff + 1).tolist())
            all_outer.extend((td["outer_nodes"] + noff + 1).tolist())
        _write_nset_block(f, "ALL_INNER", all_inner)
        f.write("**\n")
        _write_nset_block(f, "ALL_OUTER", all_outer)
        f.write("**\n")

        # All-node generate set
        f.write("*Nset, nset=ALL_NODES, generate\n")
        f.write(" 1, %d, 1\n" % total_nodes)
        f.write("**\n")

        # ── Approximal (slit) node sets for T30 ↔ T31 tie ─────────────────────
        if slit_approx:
            f.write("** ===== SLIT (APPROXIMAL) NODE SETS =====\n")
            for td, noff in zip(teeth_data, tooth_node_offsets):
                tname = td["name"]
                if tname not in slit_approx:
                    continue
                nm = slit_approx[tname]["node_mask"]   # (V,) bool over outer verts
                # outer_nodes maps outer-vertex index → 0-based index into td["nodes"]
                approx_global = []
                for vi in range(len(nm)):
                    if nm[vi]:
                        approx_global.append(int(td["outer_nodes"][vi]) + noff + 1)
                if approx_global:
                    _write_nset_block(f, "%s_APPROX" % tname, approx_global)
                    f.write("**\n")
                    print("  [SLIT] %s_APPROX: %d nodes" % (tname, len(approx_global)))

        # ── Element sets per tooth per DI bin ─────────────────────────────────
        f.write("** ===== ELEMENT SETS =====\n")
        # Collect which bins are actually used (across all teeth)
        used_bins = set()
        tooth_bin_labels = []
        for td, eoff in zip(teeth_data, tooth_elem_offsets):
            bin_labels = [[] for _ in range(n_bins)]
            for i, b in enumerate(td["tet_bins"]):
                bin_labels[b].append(eoff + i + 1)   # 1-based global label
            tooth_bin_labels.append(bin_labels)
            for b in range(n_bins):
                if bin_labels[b]:
                    used_bins.add(b)

        for b in sorted(used_bins):
            # Write per-tooth elsets
            for td, bl in zip(teeth_data, tooth_bin_labels):
                if bl[b]:
                    _write_elset_block(f, "%s_BIN_%02d" % (td["name"], b), bl[b])
                    f.write("**\n")
            # Write combined bin elset (for material assignment)
            combined = []
            for bl in tooth_bin_labels:
                combined.extend(bl[b])
            _write_elset_block(f, "BIN_%02d" % b, combined)
            f.write("**\n")

        # All-elem set
        f.write("*Elset, elset=ALL_ELEMS, generate\n")
        f.write(" 1, %d, 1\n" % total_tets)
        f.write("**\n")

        # ── Materials (isotropic per DI bin, shared across all teeth) ───────
        # Note: isotropic avoids the need for per-element orientation systems.
        # Stiffness gradient is captured by having different E per bin.
        # aniso_ratio is kept for reference but material is treated as isotropic here.
        # Unit: Abaqus mm/N/MPa system → E must be in MPa (= N/mm²)
        #   bin_E_stiff is stored in Pa → divide by 1e6 to get MPa
        f.write("** ===== MATERIALS (isotropic per DI bin, E in MPa = N/mm²) =====\n")
        for b in sorted(used_bins):
            E_MPa = bin_E_stiff[b] * 1e-6   # Pa → MPa
            f.write("*Material, name=MAT_ANISO_%02d\n" % b)
            f.write("*Elastic\n")
            f.write(" %.4f, %.4f\n" % (E_MPa, nu))
            f.write("**\n")

        # ── Solid Sections ────────────────────────────────────────────────────
        f.write("** ===== SECTIONS =====\n")
        for b in sorted(used_bins):
            f.write("*Solid Section, elset=BIN_%02d, material=MAT_ANISO_%02d\n" % (b, b))
            f.write(",\n")
            f.write("**\n")

        # ── Boundary Conditions ───────────────────────────────────────────────
        f.write("** ===== BOUNDARY CONDITIONS =====\n")
        if bc_mode == "inner_fixed":
            f.write("** All inner faces fixed (biofilm adhered to tooth surface)\n")
            f.write("*Boundary\n")
            f.write(" ALL_INNER, ENCASTRE\n")
        elif bc_mode == "bot_fixed":
            # Fix bottom z-nodes of each tooth
            f.write("*Boundary\n")
            for td, noff in zip(teeth_data, tooth_node_offsets):
                zv = td["nodes"][:, 2]
                z_min = zv.min()
                tol = (zv.max() - z_min) * 0.02 + 0.1
                bot = np.where(zv <= z_min + tol)[0]
                bot_1based = (bot + noff + 1).tolist()
                for ni in bot_1based:
                    f.write(" %d, ENCASTRE\n" % ni)
        f.write("**\n")

        # ── Slit: surface + tie constraint ────────────────────────────────────
        if slit_approx and "T30" in slit_approx and "T31" in slit_approx:
            f.write("** ===== SLIT TIE (T30 master, T31 slave) =====\n")
            f.write("** T30_APPROX_SURF and T31_APPROX_SURF are the inter-proximal faces.\n")
            f.write("** No Cload applied to approximal nodes (not free surfaces).\n")
            f.write("*Surface, type=NODE, name=T30_APPROX_SURF\n")
            f.write(" T30_APPROX\n")
            f.write("**\n")
            f.write("*Surface, type=NODE, name=T31_APPROX_SURF\n")
            f.write(" T31_APPROX\n")
            f.write("**\n")
            f.write("*Tie, name=SLIT_TIE, position tolerance=0.5, adjust=NO\n")
            f.write(" T31_APPROX_SURF, T30_APPROX_SURF\n")
            f.write("**\n")

        # ── Step: LOAD ────────────────────────────────────────────────────────
        f.write("** ===== STEP =====\n")
        f.write("*Step, name=LOAD, nlgeom=NO\n")
        f.write(" Inward pressure %.4g MPa on outer biofilm faces\n" % (pressure * 1e-6))
        f.write("*Static\n")
        f.write(" 0.1, 1.0, 1e-5, 1.0\n")
        f.write("**\n")

        # Cload for outer face pressure
        f.write("** Pressure = %.3g Pa (= %.4g MPa), forces in N (Pa*1e-6*mm²)\n" % (
            pressure, pressure * 1e-6))
        f.write("*Cload\n")
        for td, noff in zip(teeth_data, tooth_node_offsets):
            outer_forces = compute_outer_face_loads(
                td["verts_outer"], td["faces"], pressure, td["vnorms_outer"])
            # Exclude approximal (slit) nodes – they're not free outer surfaces
            if slit_approx and td["name"] in slit_approx:
                am = slit_approx[td["name"]]["node_mask"]
                outer_forces = {vi: fvec for vi, fvec in outer_forces.items()
                                if not am[vi]}
            for vi, fvec in sorted(outer_forces.items()):
                ni = int(td["outer_nodes"][vi]) + noff + 1
                if abs(fvec[0]) > 1e-20:
                    f.write(" %d, 1, %.8g\n" % (ni, fvec[0]))
                if abs(fvec[1]) > 1e-20:
                    f.write(" %d, 2, %.8g\n" % (ni, fvec[1]))
                if abs(fvec[2]) > 1e-20:
                    f.write(" %d, 3, %.8g\n" % (ni, fvec[2]))
        f.write("**\n")

        # ── Output requests ───────────────────────────────────────────────────
        # Field output only (no history → avoids "too many requests" error)
        f.write("*Output, field\n")
        f.write("*Node Output\n")
        f.write(" U, RF\n")
        f.write("*Element Output\n")
        f.write(" S, E, MISES\n")
        f.write("**\n")
        f.write("*End Step\n")

    print("  Combined INP written: %s" % out_path)
    print("  Nodes: %d  C3D4: %d" % (total_nodes, total_tets))


# ============================================================================
# Main
# ============================================================================

TEETH = [
    {"name": "T23", "stl_key": "P1_Tooth_23",
     "role": "crown (single tooth, full-wrap biofilm)"},
    {"name": "T30", "stl_key": "P1_Tooth_30",
     "role": "slit/inter-proximal (T30 side)"},
    {"name": "T31", "stl_key": "P1_Tooth_31",
     "role": "slit/inter-proximal (T31 side)"},
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Conformal biofilm tet mesh for 3 teeth → combined INP + Abaqus run")
    p.add_argument("--stl-root",   default="external_tooth_models/OpenJaw_Dataset/Patient_1",
                   help="Root dir with Teeth/ sub-dir")
    p.add_argument("--di-csv",     default="abaqus_field_dh_3d.csv")
    p.add_argument("--out",        default="biofilm_3tooth.inp")
    p.add_argument("--job-name",   default="BioFilm3T")
    p.add_argument("--thickness",  type=float, default=0.5)
    p.add_argument("--n-layers",   type=int,   default=8)
    p.add_argument("--n-bins",     type=int,   default=20)
    p.add_argument("--e-max",      type=float, default=10e9)
    p.add_argument("--e-min",      type=float, default=0.5e9)
    p.add_argument("--di-scale",   type=float, default=0.025778)
    p.add_argument("--di-exp",     type=float, default=2.0)
    p.add_argument("--nu",         type=float, default=0.30)
    p.add_argument("--aniso-ratio",type=float, default=0.5)
    p.add_argument("--pressure",   type=float, default=1.0e6)
    p.add_argument("--bc-mode",    default="inner_fixed",
                   choices=["inner_fixed", "bot_fixed"])
    p.add_argument("--smooth-iter",type=int,   default=3)
    p.add_argument("--dedup-tol",  type=float, default=1e-4)
    p.add_argument("--run",        action="store_true",
                   help="Submit Abaqus job after writing INP")
    p.add_argument("--abaqus",     default="/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus",
                   help="Path to abaqus executable")
    p.add_argument("--slit-threshold", type=float, default=0.30,
                   help="Dot-product threshold for approximal face detection [0.3]")
    p.add_argument("--slit-max-dist", type=float, default=None,
                   help="[P7] Max distance (mm) from opposing tooth for APPROX filter. "
                        "None = normal-only (legacy). Recommended: 2.0–5.0 mm.")
    p.add_argument("--no-slit",    action="store_true",
                   help="Disable T30↔T31 tie constraint (slit treatment)")
    return p.parse_args()


def process_one_tooth(stl_path, di_csv_path, cfg):
    """Build conformal tet mesh for one tooth. Returns dict for combined INP."""
    print("  [STL] Reading %s ..." % os.path.basename(stl_path))
    faces_verts, face_normals_stored = read_stl(stl_path)
    F_raw = len(faces_verts)

    print("  [DEDUP] %d raw triangles ..." % F_raw)
    verts_inner, faces = deduplicate_vertices(faces_verts, tol=cfg["dedup_tol"])
    V, F = len(verts_inner), len(faces)
    print("         → %d unique verts  %d faces" % (V, F))

    print("  [NORMALS] area-weighted vertex normals ...")
    vnorms_inner = compute_vertex_normals(verts_inner, faces, face_normals_stored)

    print("  [OFFSET] thickness=%.3f mm ..." % cfg["thickness"])
    verts_outer_raw = verts_inner + cfg["thickness"] * vnorms_inner

    if cfg["smooth_iter"] > 0:
        print("  [SMOOTH] %d Laplacian iterations ..." % cfg["smooth_iter"])
        verts_outer = laplacian_smooth_offset(
            verts_outer_raw, verts_inner, faces,
            iterations=cfg["smooth_iter"], lam=0.5)
    else:
        verts_outer = verts_outer_raw

    vnorms_outer = compute_vertex_normals(verts_outer, faces)

    print("  [TET] %d layers → prism-to-tet meshing ..." % cfg["n_layers"])
    nodes, tets, tet_layers, inner_nodes, outer_nodes = build_tet_mesh(
        verts_inner, verts_outer, faces, cfg["n_layers"])

    # Volume sanity check
    tv  = nodes[tets]
    a   = tv[:, 1] - tv[:, 0]
    b   = tv[:, 2] - tv[:, 0]
    c   = tv[:, 3] - tv[:, 0]
    vols = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    n_neg = (vols < 0).sum()
    print("         → %d nodes  %d C3D4  neg-vol=%d" % (
        len(nodes), len(tets), n_neg))
    if n_neg > 0:
        print("  [FIX] Flipping %d negative-volume tets ..." % n_neg)
        bad = vols < 0
        tets[bad, 2], tets[bad, 3] = tets[bad, 3].copy(), tets[bad, 2].copy()

    print("  [DI] Assigning bins ...")
    xs_field, di_vals_field = read_di_csv(di_csv_path)
    tet_bins, bin_E_stiff = assign_di_bins(
        tet_layers, cfg["n_layers"],
        xs_field, di_vals_field,
        cfg["n_bins"], cfg["di_scale"], cfg["di_exp"],
        cfg["e_max"], cfg["e_min"])

    return {
        "nodes":        nodes,
        "tets":         tets,
        "tet_bins":     tet_bins,
        "inner_nodes":  inner_nodes,
        "outer_nodes":  outer_nodes,
        "verts_outer":  verts_outer,
        "faces":        faces,
        "vnorms_outer": vnorms_outer,
        "bin_E_stiff":  bin_E_stiff,
        "n_neg":        n_neg,
    }


def main():
    args = parse_args()

    print("=" * 62)
    print("  biofilm_3tooth_assembly.py")
    print("  teeth     : T23 (crown)  T30 + T31 (slit/inter-proximal)")
    print("  thickness : %.3f mm  n_layers=%d" % (args.thickness, args.n_layers))
    print("  n_bins    : %d  pressure=%.3g Pa" % (args.n_bins, args.pressure))
    print("  BC mode   : %s" % args.bc_mode)
    print("=" * 62)

    cfg = {
        "thickness":   args.thickness,
        "n_layers":    args.n_layers,
        "n_bins":      args.n_bins,
        "e_max":       args.e_max,
        "e_min":       args.e_min,
        "di_scale":    args.di_scale,
        "di_exp":      args.di_exp,
        "smooth_iter": args.smooth_iter,
        "dedup_tol":   args.dedup_tol,
    }

    # Process all teeth
    teeth_data = []
    bin_E_stiff_shared = None

    for t in TEETH:
        stl_name = "%s.stl" % t["stl_key"]
        stl_path = os.path.join(args.stl_root, "Teeth", stl_name)
        print("\n── %s (%s) ──" % (t["name"], t["role"]))

        if not os.path.exists(stl_path):
            raise FileNotFoundError("STL not found: %s" % stl_path)

        td = process_one_tooth(stl_path, args.di_csv, cfg)
        td["name"] = t["name"]
        teeth_data.append(td)

        # bin_E_stiff is the same for all teeth (same DI field, same params)
        if bin_E_stiff_shared is None:
            bin_E_stiff_shared = td["bin_E_stiff"]

    # Summary
    print("\n── Mesh Summary ──")
    total_n = sum(len(d["nodes"]) for d in teeth_data)
    total_e = sum(len(d["tets"])  for d in teeth_data)
    total_neg = sum(d["n_neg"]    for d in teeth_data)
    print("  Tooth | Nodes  | C3D4   | Neg-vol")
    print("  ------+--------+--------+--------")
    for d in teeth_data:
        print("  %-5s | %6d | %6d | %d" % (
            d["name"], len(d["nodes"]), len(d["tets"]), d["n_neg"]))
    print("  TOTAL | %6d | %6d | %d" % (total_n, total_e, total_neg))

    layer_thickness = args.thickness / args.n_layers
    print("\n  Layer thickness : %.4f mm" % layer_thickness)
    print("  STL edge mean   : ~0.46 mm  (tooth surface ref)")
    print("  Finer by        : %.1fx in thickness direction" % (0.46 / layer_thickness))

    # ── Slit (approximal) detection for T30 ↔ T31 ──────────────────────────────
    slit_approx = None
    if not args.no_slit:
        td30 = teeth_data[1]   # T30
        td31 = teeth_data[2]   # T31
        thr  = args.slit_threshold
        max_dist = getattr(args, "slit_max_dist", None)
        fm30, nm30, d30 = get_approximal_info(td30, td31, threshold=thr, max_dist=max_dist)
        fm31, nm31, d31 = get_approximal_info(td31, td30, threshold=thr, max_dist=max_dist)
        print("\n── Slit (Approximal) Detection (threshold=%.2f) ──" % thr)
        print("  T30→T31 direction : [%.3f, %.3f, %.3f]" % tuple(d30))
        print("  T30 approx faces  : %d / %d (%.1f%%)" % (
            fm30.sum(), len(fm30), 100. * fm30.mean()))
        print("  T30 approx nodes  : %d / %d" % (nm30.sum(), len(nm30)))
        print("  T31 approx faces  : %d / %d (%.1f%%)" % (
            fm31.sum(), len(fm31), 100. * fm31.mean()))
        print("  T31 approx nodes  : %d / %d" % (nm31.sum(), len(nm31)))

        # Approximate inter-surface gap (T30 approx outer → T31 approx outer)
        if nm30.any() and nm31.any():
            from scipy.spatial import cKDTree
            pts30 = td30["verts_outer"][nm30]
            pts31 = td31["verts_outer"][nm31]
            tree30 = cKDTree(pts30)
            dists, _ = tree30.query(pts31, k=1)
            print("  Gap T31→T30 approx: min=%.3f  mean=%.3f  max=%.3f mm" % (
                dists.min(), dists.mean(), dists.max()))

        slit_approx = {
            "T30": {"face_mask": fm30, "node_mask": nm30},
            "T31": {"face_mask": fm31, "node_mask": nm31},
        }

    # Write combined INP
    print("\n── Writing combined INP: %s ──" % args.out)
    write_combined_inp(
        args.out, teeth_data, bin_E_stiff_shared, args.n_bins,
        args.aniso_ratio, args.nu, args.pressure, args.bc_mode,
        args.di_csv, args.thickness, args.n_layers,
        slit_approx=slit_approx)

    # Verify element coverage
    print("\n── INP verification ──")
    import subprocess
    result = subprocess.run(
        ["python3", "-c", """
import re, sys
all_tets = %d
assigned = set()
current = None
with open('%s') as f:
    for line in f:
        if line.startswith('*Elset') and 'BIN_' in line and 'T23' not in line and 'T30' not in line and 'T31' not in line:
            current = True
        elif current and not line.startswith('*') and line.strip():
            for e in line.split(','):
                e = e.strip()
                if e.isdigit(): assigned.add(int(e))
        elif line.startswith('*') and current:
            current = None
print('Elements with section:', len(assigned))
print('Missing sections:', all_tets - len(assigned))
""" % (total_e, args.out)],
        capture_output=True, text=True)
    if result.stdout:
        print("  " + result.stdout.strip().replace("\n", "\n  "))

    # Submit Abaqus job
    if args.run:
        job = args.job_name
        inp  = os.path.abspath(args.out)
        abq  = args.abaqus

        print("\n── Submitting Abaqus job: %s ──" % job)
        print("  Input : %s" % inp)
        print("  This may take a few minutes ...")

        cmd = [abq, "job=%s" % job,
               "input=%s" % inp,
               "cpus=4",
               "ask=off",
               "interactive"]
        import subprocess
        ret = subprocess.run(cmd, capture_output=False,
                             cwd=os.path.dirname(inp))
        if ret.returncode == 0:
            odb_path = os.path.join(os.path.dirname(inp), "%s.odb" % job)
            if os.path.exists(odb_path):
                size_mb = os.path.getsize(odb_path) / 1024 / 1024
                print("\n  ODB written: %s  (%.1f MB)" % (odb_path, size_mb))
            else:
                print("\n  Abaqus completed but ODB not found at: %s" % odb_path)
        else:
            print("\n  Abaqus returned code %d – check .msg / .sta files" % ret.returncode)
    else:
        print("\n  (To run Abaqus: add --run flag, or execute:)")
        print("  %s job=%s input=%s cpus=4 ask=off interactive" % (
            args.abaqus, args.job_name, os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
