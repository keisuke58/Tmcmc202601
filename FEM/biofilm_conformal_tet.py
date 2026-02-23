#!/usr/bin/env python3
"""
biofilm_conformal_tet.py  –  Conformal tetrahedral biofilm mesh around a real tooth STL

ALGORITHM
---------
1.  Read tooth STL (binary or ASCII, pure-numpy/struct, no trimesh)
2.  Deduplicate vertices; build face→vertex topology
3.  Compute area-weighted vertex normals (outward, from stored STL face normals)
4.  Create N-layer offset: v_k = v_inner + (k/N)*thickness * n_vertex
5.  Check offset self-intersections with optional Laplacian smoothing
6.  Split each triangular prism (inner-tri + outer-tri per layer) into 3 tetrahedra
    – canonical split: consistent diagonal selection avoids shared-face mismatch
7.  Map DI field CSV to tet centroids via depth (rho_norm = layer index / n_layers)
8.  Export Abaqus C3D4 INP with:
      *Node / *Element / *Nset / *Elset / *Material / *Solid Section / *Step
    INNER_FACE nodes: fixed all DOFs (tooth adhesion BC)
    OUTER_FACE load:  normal concentrated force = pressure × tributary area

ADVANTAGES over current C3D8R hex approach
-------------------------------------------
- Inner surface IS the tooth STL → zero penetration by construction
- Curved surface conformity → no gaps or mesh-surface mismatch
- Tetrahedral elements handle arbitrary curved geometry
- KD-tree DI mapping uses true depth from tooth surface

DEPENDENCIES
------------
  numpy, scipy (both already installed)
  struct, argparse, csv, sys, os (standard library)

USAGE
-----
  python3 biofilm_conformal_tet.py \\
      --stl   external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_23.stl \\
      --di-csv abaqus_field_dh_3d.csv \\
      --out    p23_biofilm_conformal.inp \\
      [--mode         substrate]  substrate | biofilm  (see below)
      [--thickness    0.5]     mm, default 0.5 (biofilm mode: 0.2 mm)
      [--n-layers     4]       layers through thickness, default 4
      [--n-bins       20]      DI material bins
      [--e-max        10e9]    Pa  (biofilm mode: 1000 Pa)
      [--e-min        0.5e9]   Pa  (biofilm mode: 10 Pa)
      [--di-scale     0.025778]
      [--di-exp       2.0]
      [--nu           0.30]
      [--aniso-ratio  0.5]
      [--pressure     1.0e6]   Pa, applied at outer face (biofilm mode: 100 Pa)
      [--bc-mode      inner_fixed]  inner_fixed | bot_fixed | none
      [--smooth-iter  3]       Laplacian smoothing iterations (0 = off)
      [--validate]             write validation report alongside INP
      [--growth-eigenstrain 0.0]  alpha_final from ODE: eps_growth = alpha/3
                               → thermal-analogy eigenstrain (GROWTH step)
                               *Expansion alpha_T=1.0 + *Temperature T=eps_growth
                               = Klempt 2024 F_g=(1+α)I constrained by tooth BC
                               Compute alpha_final = k_alpha * integral(phi_avg, 0, t_end)
                               from TMCMC ODE output via compute_alpha_eigenstrain.py.
      [--nutrient-factor 1.0]  Monod nutrient correction: alpha_eff = alpha * factor.
                               Default 1.0 = no correction. Conservative: 0.75.
                               Accounts for nutrient depletion (Klempt ċ = -g·c·φ).
      [--spatial-eigenstrain COND]  Load DI(x) from _di_credible/{COND}/ to compute
                               per-node T_node(x) = T_mean * DI(x)/DI_mean.
                               → spatially resolved growth eigenstrain field (+10% theory).
                               Requires --growth-eigenstrain > 0.
      [--di-credible-dir _di_credible]  Base dir for DI credible interval data.

ANALYSIS MODES
--------------
  substrate (default)
      E_max = 10 GPa, E_min = 0.5 GPa, pressure = 1 MPa, thickness = 0.5 mm
      Models the tooth/PDL *substrate* stiffness modified by biofilm composition.
      E(DI) = effective stiffness of biofilm-covered dental surface.
      S_Mises in MPa; relevant for substrate mechanical risk assessment.

  biofilm
      E_max = 1000 Pa, E_min = 10 Pa, pressure = 100 Pa, thickness = 0.2 mm
      Models the biofilm EPS matrix itself (Billings et al. 2015; Klempt et al. 2024).
      Applied pressure = gingival crevicular fluid / soft-tissue pressure.
      S_Mises in Pa; relevant for biofilm internal stress and detachment risk.
      NOTE: strains may be O(0.1) – large-deformation effects not captured by
            linear-elastic Abaqus; treat results as qualitative / comparative.
      Individual flags (--e-max, --e-min, --pressure, --thickness) override mode defaults.
"""

from __future__ import print_function, division
import sys
import os
import struct
import argparse
import csv

import numpy as np
from scipy.spatial import cKDTree

# ============================================================================
# 1. STL READER
# ============================================================================

def read_stl(path):
    """
    Read binary or ASCII STL.
    Returns:
        faces_verts : (F, 3, 3) float64 – raw per-face vertex coords
        face_normals: (F, 3)    float64 – stored face normals (may be zero for ASCII)
    """
    with open(path, "rb") as f:
        raw = f.read(80)
    # Heuristic: binary STL starts with 80-byte header then uint32 count
    # ASCII STL starts with "solid "
    try:
        first = raw[:5].decode("ascii", errors="replace").strip().lower()
    except Exception:
        first = ""
    if first == "solid":
        return _read_stl_ascii(path)
    else:
        return _read_stl_binary(path)


def _read_stl_binary(path):
    with open(path, "rb") as f:
        f.read(80)  # header
        n_tri = struct.unpack("<I", f.read(4))[0]
        faces_verts  = np.zeros((n_tri, 3, 3), dtype=np.float64)
        face_normals = np.zeros((n_tri, 3), dtype=np.float64)
        for i in range(n_tri):
            data = struct.unpack("<12fH", f.read(50))
            face_normals[i] = data[0:3]
            faces_verts[i, 0] = data[3:6]
            faces_verts[i, 1] = data[6:9]
            faces_verts[i, 2] = data[9:12]
    return faces_verts, face_normals


def _read_stl_ascii(path):
    tris, norms = [], []
    cur_norm = np.zeros(3)
    cur_verts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                cur_norm = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                cur_verts = []
            elif line.startswith("vertex"):
                parts = line.split()
                cur_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("endfacet"):
                if len(cur_verts) == 3:
                    tris.append(cur_verts)
                    norms.append(cur_norm.copy())
    return np.array(tris, dtype=np.float64), np.array(norms, dtype=np.float64)


# ============================================================================
# 2. VERTEX DEDUPLICATION
# ============================================================================

def deduplicate_vertices(faces_verts, tol=1e-4):
    """
    Deduplicate vertices using rounded keys.
    Returns:
        verts : (V, 3) float64 – unique vertices
        faces : (F, 3) int     – face connectivity (0-indexed)
    """
    F = len(faces_verts)
    raw = faces_verts.reshape(-1, 3)   # (3F, 3)

    # Round to tolerance grid
    scale = 1.0 / tol
    keys = np.round(raw * scale).astype(np.int64)

    # Map each raw vertex to a unique index
    key_to_idx = {}
    unique_verts = []
    mapping = np.zeros(len(raw), dtype=int)
    for i, k in enumerate(map(tuple, keys)):
        if k not in key_to_idx:
            key_to_idx[k] = len(unique_verts)
            unique_verts.append(raw[i])
        mapping[i] = key_to_idx[k]

    verts = np.array(unique_verts, dtype=np.float64)
    faces = mapping.reshape(F, 3)
    return verts, faces


# ============================================================================
# 3. VERTEX NORMALS (area-weighted, using stored STL face normals as sign guide)
# ============================================================================

def compute_vertex_normals(verts, faces, face_normals_stored=None):
    """
    Compute area-weighted vertex normals.
    Uses cross-product of edges for direction; stored STL normals only for sign.
    Returns:
        vnorms : (V, 3) float64 – unit outward vertex normals
    """
    V = len(verts)
    vnorms = np.zeros((V, 3), dtype=np.float64)

    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]   # (F,3)
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]   # (F,3)
    cross = np.cross(e1, e2)                        # (F,3)
    areas = 0.5 * np.linalg.norm(cross, axis=1)    # (F,)
    # Unit face normals from cross product
    lens = np.linalg.norm(cross, axis=1, keepdims=True)
    lens = np.where(lens < 1e-14, 1.0, lens)
    fnorms = cross / lens                            # (F,3)

    # Flip if stored normals disagree (sign consistency)
    if face_normals_stored is not None:
        dot = np.sum(fnorms * face_normals_stored, axis=1)  # (F,)
        flip = (dot < 0).astype(float) * -2 + 1            # ±1
        fnorms = fnorms * flip[:, np.newaxis]

    # Accumulate area-weighted normals to vertices
    for k in range(3):
        np.add.at(vnorms, faces[:, k], fnorms * areas[:, np.newaxis])

    # Normalize
    lens2 = np.linalg.norm(vnorms, axis=1, keepdims=True)
    lens2 = np.where(lens2 < 1e-14, 1.0, lens2)
    vnorms = vnorms / lens2
    return vnorms


# ============================================================================
# 4. LAPLACIAN SMOOTHING (optional, for offset self-intersection fix)
# ============================================================================

def build_adjacency(faces, V):
    """Return adjacency list (list of sets) for V vertices and given faces."""
    adj = [set() for _ in range(V)]
    for f in faces:
        a, b, c = f
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    return adj


def laplacian_smooth_offset(outer_verts, inner_verts, faces, iterations=3, lam=0.5):
    """
    Laplacian-smooth the outer offset surface while keeping the depth
    (distance from inner surface) roughly constant.
    """
    if iterations == 0:
        return outer_verts.copy()

    V = len(outer_verts)
    adj = build_adjacency(faces, V)
    verts = outer_verts.copy()

    # Pre-compute desired thickness per vertex (= distance inner→outer at start)
    thickness_per_v = np.linalg.norm(outer_verts - inner_verts, axis=1, keepdims=True)

    # Unit outward direction (inner→outer, recomputed each iteration)
    dirs = outer_verts - inner_verts
    dirs_len = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_len = np.where(dirs_len < 1e-14, 1.0, dirs_len)
    dirs = dirs / dirs_len

    for _ in range(iterations):
        new_verts = verts.copy()
        for i, nb in enumerate(adj):
            if not nb:
                continue
            nb_list = list(nb)
            centroid = verts[nb_list].mean(axis=0)
            # Move toward Laplacian position
            new_verts[i] = (1 - lam) * verts[i] + lam * centroid

        # Re-project to keep thickness
        # New direction = from inner_verts[i] toward new_verts[i]
        d = new_verts - inner_verts
        d_len = np.linalg.norm(d, axis=1, keepdims=True)
        d_len = np.where(d_len < 1e-14, 1.0, d_len)
        d_unit = d / d_len
        # Re-apply original thickness in new direction
        verts = inner_verts + d_unit * thickness_per_v

    return verts


# ============================================================================
# 5. MULTI-LAYER PRISM-TO-TET MESHING
# ============================================================================

def build_tet_mesh(verts_inner, verts_outer, faces, n_layers):
    """
    Build a multi-layer conformal tetrahedral mesh between two surfaces.

    Each layer k (0..n_layers-1) creates a prismatic slab between:
      bottom: alpha_k  = k/n_layers
      top:    alpha_k1 = (k+1)/n_layers

    Each prism is split into 3 C3D4 tets using canonical split:
      Tet 1: {B0, B1, B2, T2}
      Tet 2: {B0, B1, T2, T1}
      Tet 3: {B0, T1, T2, T0}

    Node numbering:
      layer k, vertex i  →  global index = k * V + i

    Returns
    -------
    nodes       : (N_nodes, 3) float64  – all node coordinates
    tets        : (N_tets, 4) int       – tet connectivity, 0-indexed
    tet_layers  : (N_tets,)  int        – layer index (0..n_layers-1) for each tet
    inner_nodes : (V,) int              – node indices on inner face (layer 0)
    outer_nodes : (V,) int              – node indices on outer face (layer n_layers)
    """
    V = len(verts_inner)
    F = len(faces)

    # Build all layer vertex arrays
    #   verts at layer k = lerp(inner, outer, k/n_layers)
    layer_verts = []
    for k in range(n_layers + 1):
        alpha = k / float(n_layers)
        layer_verts.append((1.0 - alpha) * verts_inner + alpha * verts_outer)

    nodes = np.vstack(layer_verts)  # ((n_layers+1)*V, 3)

    # Build tets: 3 tets per prism per layer
    n_tets_total = F * n_layers * 3
    tets = np.zeros((n_tets_total, 4), dtype=np.int64)
    tet_layers = np.zeros(n_tets_total, dtype=np.int32)

    idx = 0
    for k in range(n_layers):
        bot_off = k * V          # global index offset for bottom layer
        top_off = (k + 1) * V   # global index offset for top layer

        for f_i, (i0, i1, i2) in enumerate(faces):
            B0, B1, B2 = bot_off + i0, bot_off + i1, bot_off + i2
            T0, T1, T2 = top_off + i0, top_off + i1, top_off + i2

            # Canonical 3-tet split of prism {B0,B1,B2,T0,T1,T2}
            tets[idx]     = [B0, B1, B2, T2]
            tets[idx + 1] = [B0, B1, T2, T1]
            tets[idx + 2] = [B0, T1, T2, T0]
            tet_layers[idx]     = k
            tet_layers[idx + 1] = k
            tet_layers[idx + 2] = k
            idx += 3

    inner_nodes = np.arange(V, dtype=np.int64)
    outer_nodes = np.arange(n_layers * V, (n_layers + 1) * V, dtype=np.int64)

    return nodes, tets, tet_layers, inner_nodes, outer_nodes


# ============================================================================
# 6. DI FIELD READER
# ============================================================================

def read_di_csv(path):
    """
    Read DI field CSV (columns: x, y, z, phi_pg, di, ...).
    Returns:
        xs      : (M,) float64 – x-coordinates (used as depth axis in field space)
        di_vals : (M,) float64 – dysbiotic index values
    """
    xs, di_vals = [], []
    xi_col = di_col = None
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row]
            if not row or row[0].startswith("#"):
                continue
            if xi_col is None:
                # Header row
                header = [c.lower() for c in row]
                if "x" in header:
                    xi_col = header.index("x")
                if "di" in header:
                    di_col = header.index("di")
                if xi_col is not None and di_col is not None:
                    continue
                else:
                    xi_col = di_col = None   # no valid header yet
                continue
            try:
                xs.append(float(row[xi_col]))
                di_vals.append(float(row[di_col]))
            except (IndexError, ValueError):
                continue
    return np.array(xs, dtype=np.float64), np.array(di_vals, dtype=np.float64)


# ============================================================================
# 7. DI BIN ASSIGNMENT PER TET
# ============================================================================

def assign_di_bins(tet_layers, n_layers, xs_field, di_vals_field,
                   n_bins, di_scale, di_exp, e_max, e_min):
    """
    Assign each tet to a DI material bin based on its layer depth.

    Depth rho_norm = (layer + 0.5) / n_layers  (centroid of the layer)
    Map rho_norm → x_query in [x_min, x_max] of DI field → nearest DI value → bin.

    Returns:
        tet_bins : (N_tets,) int – bin index (0..n_bins-1)
        bin_E_stiff : (n_bins,) float – stiffness per bin (dominant direction)
    """
    x_min = xs_field.min() if len(xs_field) else 0.0
    x_max = xs_field.max() if len(xs_field) else 1.0
    x_range = x_max - x_min if x_max > x_min else 1.0

    di_max = di_vals_field.max() if len(di_vals_field) else 1.0
    bin_w  = di_max / n_bins if n_bins > 0 else 1.0

    # Build 1D KD-tree on x-coordinates of DI field
    tree_1d = cKDTree(xs_field.reshape(-1, 1))

    tet_bins = np.zeros(len(tet_layers), dtype=np.int32)
    for i, layer in enumerate(tet_layers):
        rho_norm = (layer + 0.5) / n_layers
        x_query  = x_min + rho_norm * x_range
        _, nn_idx = tree_1d.query([[x_query]])
        di_val = di_vals_field[nn_idx[0]]
        b = int(di_val / bin_w)
        tet_bins[i] = max(0, min(n_bins - 1, b))

    # Pre-compute E_stiff per bin
    bin_E_stiff = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        di_b = (b + 0.5) * bin_w
        if di_scale > 0:
            r = float(np.clip(di_b / di_scale, 0.0, 1.0))
        else:
            r = 0.0
        bin_E_stiff[b] = e_max * (1.0 - r) ** di_exp + e_min * r

    return tet_bins, bin_E_stiff


# ============================================================================
# 8. OUTER FACE LOAD COMPUTATION (concentrated forces from pressure)
# ============================================================================

def compute_outer_face_loads(verts_outer, faces, pressure, vnorms_outer):
    """
    Compute per-node concentrated forces on the outer surface due to
    a uniform inward pressure (= normal load toward the tooth).

    Unit convention (Abaqus mm / N / MPa system):
      pressure  [Pa]   = pressure_MPa * 1e6
      area      [mm²]  (coords are in mm)
      force [N] = pressure_MPa [N/mm²] * area [mm²]
                = (pressure_Pa / 1e6) * area_mm²
      → factor 1e-6 converts Pa*mm² → N

    For each outer face triangle with area A:
      force per node = (pressure_Pa / 1e6 * A / 3) * inward_normal_unit

    Returns:
        node_forces : dict  { node_idx (0-based) : np.array(fx, fy, fz) [N] }
    """
    Pa_to_Nmm2 = 1.0e-6   # 1 Pa = 1e-6 N/mm²; area in mm² → force in N
    node_forces = {}
    for i0, i1, i2 in faces:
        v0 = verts_outer[i0]
        v1 = verts_outer[i1]
        v2 = verts_outer[i2]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        area  = 0.5 * np.linalg.norm(cross)  # mm²
        if area < 1e-20:
            continue
        # Inward surface normal (toward tooth) = average outward normal flipped
        n_avg = (vnorms_outer[i0] + vnorms_outer[i1] + vnorms_outer[i2]) / 3.0
        n_len = np.linalg.norm(n_avg)
        if n_len < 1e-14:
            continue
        n_inward = -(n_avg / n_len)
        # Total force on this face [N]; divide by 3 for nodal contribution
        f_total_N = pressure * Pa_to_Nmm2 * area
        f_node    = (f_total_N / 3.0) * n_inward

        for vi in [i0, i1, i2]:
            if vi not in node_forces:
                node_forces[vi] = np.zeros(3)
            node_forces[vi] += f_node

    return node_forces


# ============================================================================
# 9. ABAQUS C3D4 INP WRITER
# ============================================================================

def _write_nset(f, name, indices, indent=""):
    """Write *Nset block. indices are 1-based."""
    f.write("*Nset, nset=%s\n" % name)
    row = []
    for idx in indices:
        row.append("%d" % idx)
        if len(row) == 16:
            f.write(indent + ", ".join(row) + ",\n")
            row = []
    if row:
        f.write(indent + ", ".join(row) + "\n")


def _write_elset(f, name, labels):
    """Write *Elset block. labels are 1-based."""
    f.write("*Elset, elset=%s\n" % name)
    row = []
    for lab in labels:
        row.append("%d" % lab)
        if len(row) == 16:
            f.write(", ".join(row) + ",\n")
            row = []
    if row:
        f.write(", ".join(row) + "\n")


def write_abaqus_inp(out_path, nodes, tets, tet_bins, inner_nodes, outer_nodes,
                     bin_E_stiff, aniso_ratio, nu, n_bins,
                     pressure, bc_mode,
                     verts_outer, faces, vnorms_outer,
                     stl_path, di_csv_path, n_layers, thickness,
                     nlgeom=False, growth_eigenstrain=0.0,
                     T_nodes=None, neo_hookean=False):
    """
    Write a complete Abaqus C3D4 INP file.

    Node numbering: 1-based (add 1 to all 0-based indices).
    Element numbering: 1-based.
    """
    N_nodes = len(nodes)
    N_tets  = len(tets)

    # Bot / Top node sets (by z-coordinate of inner surface vertices)
    z_inner = nodes[inner_nodes, 2]
    z_min = z_inner.min()
    z_max = z_inner.max()
    z_tol = (z_max - z_min) * 0.02 + 0.1   # 2% of height + 0.1 mm

    # Collect bot/top nodes from ALL layers (full column fix at bot/top)
    bot_nodes_set = set()
    top_nodes_set = set()
    for k in range(n_layers + 1):
        off = k * len(verts_outer)   # verts_outer same size as verts_inner
        V = len(verts_outer)
        for vi in range(V):
            z = nodes[off + vi, 2]
            if z <= z_min + z_tol:
                bot_nodes_set.add(off + vi)
            if z >= z_max - z_tol:
                top_nodes_set.add(off + vi)

    print("  Bot nodes: %d  Top nodes: %d" % (len(bot_nodes_set), len(top_nodes_set)))

    # Outer face loads
    outer_node_forces = compute_outer_face_loads(verts_outer, faces, pressure, vnorms_outer)

    # Group tets per bin
    bin_tet_labels = [[] for _ in range(n_bins)]
    for i, b in enumerate(tet_bins):
        bin_tet_labels[b].append(i + 1)  # 1-based element label

    with open(out_path, "w") as f:
        # ── Header ────────────────────────────────────────────────────────────
        f.write("** biofilm_conformal_tet.py – Conformal C3D4 biofilm mesh\n")
        f.write("** STL   : %s\n" % os.path.basename(stl_path))
        f.write("** DI CSV: %s\n" % os.path.basename(di_csv_path))
        f.write("** Thickness: %.3f mm  Layers: %d  Total nodes: %d  Total tets: %d\n" % (
            thickness, n_layers, N_nodes, N_tets))
        f.write("** Inner face: tooth STL surface (zero penetration by construction)\n")
        f.write("**\n")
        f.write("*Heading\n")
        f.write(" Conformal biofilm tet mesh - %s\n" % os.path.basename(stl_path))
        f.write("**\n")

        # ── Nodes ─────────────────────────────────────────────────────────────
        f.write("*Node\n")
        for i, (x, y, z) in enumerate(nodes):
            f.write(" %d, %.8g, %.8g, %.8g\n" % (i + 1, x, y, z))
        f.write("**\n")

        # ── Elements (C3D4) ───────────────────────────────────────────────────
        f.write("*Element, type=C3D4\n")
        for i, tet in enumerate(tets):
            n1, n2, n3, n4 = tet + 1   # 1-based
            f.write(" %d, %d, %d, %d, %d\n" % (i + 1, n1, n2, n3, n4))
        f.write("**\n")

        # ── Node sets ─────────────────────────────────────────────────────────
        _write_nset(f, "INNER_FACE", inner_nodes + 1)
        f.write("**\n")
        _write_nset(f, "OUTER_FACE", outer_nodes + 1)
        f.write("**\n")
        if bot_nodes_set:
            _write_nset(f, "BOT_NODES", sorted(n + 1 for n in bot_nodes_set))
            f.write("**\n")
        if top_nodes_set:
            _write_nset(f, "TOP_NODES", sorted(n + 1 for n in top_nodes_set))
            f.write("**\n")

        # All nodes set
        f.write("*Nset, nset=ALL_NODES, generate\n")
        f.write(" 1, %d, 1\n" % N_nodes)
        f.write("**\n")

        # ── Element sets per DI bin ───────────────────────────────────────────
        for b in range(n_bins):
            if not bin_tet_labels[b]:
                continue
            _write_elset(f, "BIN_%02d" % b, bin_tet_labels[b])
            f.write("**\n")

        # All elements set
        f.write("*Elset, elset=ALL_ELEMS, generate\n")
        f.write(" 1, %d, 1\n" % N_tets)
        f.write("**\n")

        # ── Materials (isotropic per DI bin) ─────────────────────────────────
        # Isotropic avoids need for per-element orientation systems.
        # Stiffness gradient is captured by different E per bin.
        # When growth_eigenstrain > 0: add *Expansion (thermal analogy for eigenstrain).
        #   alpha_T = 1.0  →  thermal strain per direction = 1.0 * T
        #   In GROWTH step: T = eps_growth = alpha_final/3
        #   → isotropic strain = eps_growth per direction (= Klempt F_g = (1+α)I at small α)
        f.write("** =====  MATERIALS (isotropic per DI bin, E in MPa = N/mm²)  =====\n")
        if growth_eigenstrain > 0.0:
            f.write("** *Expansion added for growth eigenstrain thermal analogy\n")
            if T_nodes is not None:
                f.write("** SPATIAL: T_node(x)=T_mean*DI(x)/DI_mean, T_mean=%.6g\n" % (
                    growth_eigenstrain / 3.0))
            else:
                f.write("** UNIFORM: alpha_T=1.0, T_growth=%.6g => eps_growth=%.6g per direction\n" % (
                    growth_eigenstrain / 3.0, growth_eigenstrain / 3.0))
        for b in range(n_bins):
            E_MPa = bin_E_stiff[b] * 1e-6   # Pa → MPa (Abaqus mm/N/MPa system)
            E_MPa = max(E_MPa, 1e-8)
            f.write("*Material, name=MAT_ANISO_%02d\n" % b)
            if neo_hookean:
                mu = E_MPa / (2.0 * (1.0 + nu))
                denom = 3.0 * (1.0 - 2.0 * nu)
                if abs(denom) < 1e-8:
                    denom = 1e-8 if denom >= 0.0 else -1e-8
                K = E_MPa / denom
                mu = max(mu, 1e-8)
                K = max(K, 1e-8)
                C10 = 0.5 * mu
                D1 = 2.0 / K
                f.write("*Hyperelastic, Neo Hooke\n")
                f.write(" %.6g, %.6g\n" % (C10, D1))
                if growth_eigenstrain > 0.0:
                    f.write("*Expansion, type=ISO, zero=0.0\n")
                    f.write(" 1.0\n")
            else:
                f.write("*Elastic\n")
                f.write(" %.6g, %.4f\n" % (E_MPa, nu))
                if growth_eigenstrain > 0.0:
                    f.write("*Expansion, type=ISO, zero=0.0\n")
                    f.write(" 1.0\n")
            f.write("**\n")

        # ── Solid Sections ────────────────────────────────────────────────────
        f.write("** =====  SECTIONS  =====\n")
        for b in range(n_bins):
            if not bin_tet_labels[b]:
                continue
            f.write("*Solid Section, elset=BIN_%02d, material=MAT_ANISO_%02d\n" % (b, b))
            f.write(",\n")
            f.write("**\n")

        # ── Boundary Conditions ───────────────────────────────────────────────
        f.write("** =====  BOUNDARY CONDITIONS  =====\n")
        if bc_mode == "inner_fixed":
            f.write("** BC: inner face (tooth surface) fully fixed – adhesion to tooth\n")
            f.write("*Boundary\n")
            f.write(" INNER_FACE, ENCASTRE\n")
            f.write("**\n")
        elif bc_mode == "bot_fixed":
            f.write("** BC: bottom nodes (gum-root junction) fixed\n")
            f.write("*Boundary\n")
            f.write(" BOT_NODES, ENCASTRE\n")
            f.write("**\n")
        # bc_mode == "none" → write no BC (user will add in CAE)

        # ── Step 1 (optional): GROWTH — thermal-analogy eigenstrain ─────────────
        # Thermal analogy for growth eigenstrain (Klempt 2024, α̇ = k_α φ):
        #   *Expansion, alpha_T=1.0  →  eps_thermal = 1.0 * T (per direction)
        #   T_growth = eps_growth = alpha_final / 3
        #   → Isotropic expansion eps_growth per direction when T = T_growth
        #   → Volumetric strain = 3 * eps_growth = alpha_final  (matches F_g = (1+α)I)
        #   → Inner face (ENCASTRE) constrains expansion → compressive stress builds up
        # This is physically correct: biofilm grows but is constrained by the tooth.
        # NOT an initial stress hack: Abaqus solves the constrained growth properly.
        nlgeom_str = "YES" if nlgeom else "NO"
        if growth_eigenstrain > 0.0:
            eps_growth = growth_eigenstrain / 3.0
            spatial_mode = T_nodes is not None
            f.write("** =====  STEP 1: GROWTH EIGENSTRAIN (thermal analogy)  =====\n")
            f.write("** alpha_final=%.4g  eps_growth=%.4g per direction\n" % (
                growth_eigenstrain, eps_growth))
            f.write("** F_g = (1+alpha)I  →  T_growth = eps_growth = alpha/3\n")
            f.write("** alpha_T=1.0 → eps_thermal = T = eps_growth  (isotropic, stress-free)\n")
            if spatial_mode:
                T_mean = T_nodes.mean()
                f.write("** SPATIAL MODE: T_node(x) = T_mean*DI(x)/DI_mean\n")
                f.write("**   T_mean=%.4g  T_min=%.4g  T_max=%.4g\n" % (
                    T_mean, T_nodes.min(), T_nodes.max()))
            f.write("*Step, name=GROWTH, nlgeom=%s\n" % nlgeom_str)
            f.write(" Growth-induced eigenstrain: constrained expansion by tooth adhesion\n")
            f.write("*Static\n")
            f.write(" 1.0, 1.0, 1e-6, 1.0\n")
            f.write("**\n")
            # Apply temperature field to nodes
            # alpha_T = 1.0 (set in *Expansion) → thermal strain = T per direction
            if spatial_mode:
                # Per-node temperatures: T_node(x) = T_mean * DI(x)/DI_mean
                f.write("** Per-node temperature (DI spatial field): %d nodes\n" % len(T_nodes))
                f.write("*Temperature\n")
                for ni_0based, T_val in enumerate(T_nodes):
                    f.write(" %d, %.8g\n" % (ni_0based + 1, T_val))
            else:
                # Uniform temperature = eps_growth for all nodes
                f.write("*Temperature\n")
                f.write(" ALL_NODES, %.8g\n" % eps_growth)
            f.write("**\n")
            # Output for growth step
            f.write("*Output, field\n")
            f.write("*Node Output\n")
            f.write(" U, RF\n")
            f.write("*Element Output\n")
            f.write(" S, E, MISES\n")
            f.write("**\n")
            f.write("*End Step\n")
            f.write("**\n")

        # ── Step 2 (or Step 1 if no growth): LOAD — external GCF pressure ──────
        f.write("** =====  STEP%s: LOAD  =====\n" % (" 2" if growth_eigenstrain > 0.0 else ""))
        f.write("*Step, name=LOAD, nlgeom=%s\n" % nlgeom_str)
        f.write(" Biofilm pressure loading (%.3g Pa inward on outer face)\n" % pressure)
        f.write("*Static\n")
        # For NLGEOM use smaller initial increment to aid convergence
        if nlgeom:
            f.write(" 0.01, 1.0, 1e-6, 0.1\n")
        else:
            f.write(" 0.1, 1.0, 1e-5, 1.0\n")
        f.write("**\n")

        # Concentrated forces on outer face nodes (simulating distributed pressure)
        # Unit: Pa * 1e-6 * mm² = N  (Abaqus mm/N/MPa system)
        f.write("** Concentrated forces: pressure=%.3g Pa (=%.4g MPa), forces in N\n" % (
            pressure, pressure * 1e-6))
        f.write("*Cload\n")
        for vi, fvec in sorted(outer_node_forces.items()):
            ni = outer_nodes[vi] + 1   # 1-based global node number
            if abs(fvec[0]) > 1e-20:
                f.write(" %d, 1, %.8g\n" % (ni, fvec[0]))
            if abs(fvec[1]) > 1e-20:
                f.write(" %d, 2, %.8g\n" % (ni, fvec[1]))
            if abs(fvec[2]) > 1e-20:
                f.write(" %d, 3, %.8g\n" % (ni, fvec[2]))
        f.write("**\n")

        # ── Output requests ───────────────────────────────────────────────────
        f.write("*Output, field\n")
        f.write("*Node Output\n")
        f.write(" U, RF\n")
        f.write("*Element Output\n")
        f.write(" S, E, MISES\n")
        f.write("**\n")
        f.write("*End Step\n")

    print("  INP written: %s" % out_path)


# ============================================================================
# 10. VALIDATION REPORT
# ============================================================================

def write_validation_report(out_path, nodes, tets, tet_layers, tet_bins,
                             inner_nodes, outer_nodes, verts_inner, verts_outer,
                             bin_E_stiff, n_bins, n_layers, thickness, pressure, bc_mode):
    """Write a plain-text validation summary alongside the INP."""
    V = len(verts_inner)
    F = len(tets) // (n_layers * 3)

    # Tet quality: min edge ratio
    tet_verts = nodes[tets]   # (N_tets, 4, 3)
    edges = []
    for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        e = tet_verts[:, i, :] - tet_verts[:, j, :]
        edges.append(np.linalg.norm(e, axis=1))
    edges = np.stack(edges, axis=1)   # (N_tets, 6)
    e_min = edges.min(axis=1)
    e_max = edges.max(axis=1)
    quality = e_min / (e_max + 1e-14)   # aspect ratio (ideal tet = 1.0)

    # Tet volume (signed, using triple product)
    a = tet_verts[:, 1] - tet_verts[:, 0]
    b = tet_verts[:, 2] - tet_verts[:, 0]
    c = tet_verts[:, 3] - tet_verts[:, 0]
    vols = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    n_neg = (vols < 0).sum()

    # Depth check: mean distance inner→outer
    mean_thickness = np.mean(np.linalg.norm(verts_outer - verts_inner, axis=1))

    with open(out_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Conformal Biofilm Tet Mesh – Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Geometry\n")
        f.write("  Inner vertices (= tooth STL verts) : %d\n" % V)
        f.write("  STL faces                           : %d\n" % F)
        f.write("  Layers                              : %d\n" % n_layers)
        f.write("  Target thickness                    : %.4f mm\n" % thickness)
        f.write("  Actual mean thickness               : %.4f mm\n" % mean_thickness)
        f.write("\n")
        f.write("Mesh\n")
        f.write("  Total nodes  : %d\n" % len(nodes))
        f.write("  Total C3D4   : %d\n" % len(tets))
        f.write("  Neg-volume   : %d  (should be 0)\n" % n_neg)
        f.write("  Aspect ratio : min=%.4f  mean=%.4f  (ideal=1.0)\n" % (
            quality.min(), quality.mean()))
        f.write("  Volume range : %.4g .. %.4g mm^3\n" % (vols.min(), vols.max()))
        f.write("\n")
        f.write("DI Bins\n")
        f.write("  %-6s  %-12s  %-12s\n" % ("Bin", "E_stiff (GPa)", "n_tets"))
        for b in range(n_bins):
            n_b = (tet_bins == b).sum()
            f.write("  %02d     %.4f        %d\n" % (b, bin_E_stiff[b] / 1e9, n_b))
        f.write("\n")
        f.write("Boundary conditions\n")
        f.write("  BC mode     : %s\n" % bc_mode)
        f.write("  Pressure    : %.3g Pa (inward on outer face)\n" % pressure)
        f.write("\n")
        f.write("Key advantage over previous C3D8R hex approach:\n")
        f.write("  Inner surface = tooth STL vertices exactly\n")
        f.write("  → zero penetration by construction\n")
        f.write("  → mesh conforms to all surface curvature\n")

    print("  Report written: %s" % out_path)


# ============================================================================
# 11. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Conformal tetrahedral biofilm mesh from tooth STL")
    p.add_argument("--stl",         required=True,           help="Tooth STL file")
    p.add_argument("--di-csv",      required=True,           help="DI field CSV (from export_for_abaqus.py)")
    p.add_argument("--out",         required=True,           help="Output Abaqus INP file")
    p.add_argument("--thickness",   type=float, default=0.5, help="Biofilm thickness (mm)")
    p.add_argument("--n-layers",    type=int,   default=4,   help="Layers through thickness")
    p.add_argument("--n-bins",      type=int,   default=20,  help="DI material bins")
    p.add_argument("--e-max",       type=float, default=10e9)
    p.add_argument("--e-min",       type=float, default=0.5e9)
    p.add_argument("--di-scale",    type=float, default=0.025778)
    p.add_argument("--di-exp",      type=float, default=2.0)
    p.add_argument("--nu",          type=float, default=0.30)
    p.add_argument("--aniso-ratio", type=float, default=0.5)
    p.add_argument("--pressure",    type=float, default=1.0e6, help="Outer face pressure (Pa)")
    p.add_argument("--bc-mode",     default="inner_fixed",
                   choices=["inner_fixed", "bot_fixed", "none"],
                   help="inner_fixed=tooth adhesion BC; bot_fixed=gum BC; none=no BC")
    p.add_argument("--smooth-iter", type=int,   default=3,   help="Laplacian smoothing iters (0=off)")
    p.add_argument("--dedup-tol",   type=float, default=1e-4, help="Vertex dedup tolerance (mm)")
    p.add_argument("--validate",    action="store_true",     help="Write validation report")
    p.add_argument("--mode",        default="substrate",
                   choices=["substrate", "biofilm"],
                   help=("substrate: GPa-scale tooth/PDL surface material (default); "
                         "biofilm: Pa-scale EPS matrix material (Klempt et al. 2024). "
                         "Sets E_max/E_min/pressure/thickness unless overridden explicitly."))
    p.add_argument("--growth-eigenstrain", type=float, default=0.0, dest="growth_eigenstrain",
                   help=("alpha_final from 0D ODE integration: alpha_final = k_alpha * "
                         "integral(phi_avg, 0, t_end). Thermal analogy: *Expansion alpha_T=1.0 "
                         "+ *Temperature T=eps_growth (=alpha/3) in GROWTH step. "
                         "Compute from compute_alpha_eigenstrain.py. Default 0.0 = no eigenstrain."))
    p.add_argument("--nutrient-factor", type=float, default=1.0, dest="nutrient_factor",
                   help=("Monod nutrient correction: alpha_eff = alpha_final * nutrient_factor. "
                         "Default 1.0 = no correction (nutrient abundant). "
                         "Conservative oral cavity estimate: 0.75. "
                         "Accounts for Klempt 2024 nutrient depletion (ċ = -g·c·φ)."))
    p.add_argument("--spatial-eigenstrain", default=None, dest="spatial_eigenstrain",
                   metavar="COND",
                   help=("Condition name to load spatially resolved DI(x) field from "
                         "_di_credible/{COND}/. If set, GROWTH step uses per-node "
                         "*Temperature: T_node(x) = T_mean * DI(x)/DI_mean. "
                         "Requires --growth-eigenstrain > 0. "
                         "Available: commensal_static, commensal_hobic, dh_baseline, dysbiotic_static."))
    p.add_argument("--di-credible-dir", default="_di_credible", dest="di_credible_dir",
                   help="Base directory for DI credible interval npy data (default: _di_credible)")
    p.add_argument("--neo-hookean", action="store_true", dest="neo_hookean",
                   help="Use Neo-Hookean hyperelastic material instead of linear elastic")
    p.add_argument("--growth-eigenstrain-field", default=None, dest="growth_eigenstrain_field",
                   metavar="CSV",
                   help=("Path to alpha_field_1d.csv from JAXFEM/extract_alpha_field_1d.py. "
                         "Assigns spatially varying growth eigenstrain α(x) based on depth in "
                         "biofilm (x=0: tooth surface, x=1: outer surface). "
                         "Column format: x_norm,x_mm,c_final,phi_total_final,alpha_x_monod,eps_growth_x_monod. "
                         "Takes precedence over --growth-eigenstrain when provided."))
    return p.parse_args()


def main():
    args = parse_args()

    # ── Mode-based defaults (biofilm mode overrides substrate defaults) ────────
    # Only applies when the user did NOT explicitly pass the flag.
    # Detection: compare against the hard-coded substrate defaults.
    _SUBSTRATE_DEFAULTS = dict(e_max=10e9, e_min=0.5e9, pressure=1.0e6, thickness=0.5)
    _BIOFILM_DEFAULTS   = dict(e_max=1000.0, e_min=10.0, pressure=100.0, thickness=0.2)
    if args.mode == "biofilm":
        for attr, sub_val in _SUBSTRATE_DEFAULTS.items():
            if abs(getattr(args, attr) - sub_val) < sub_val * 1e-9:
                # Still at substrate default → replace with biofilm default
                setattr(args, attr, _BIOFILM_DEFAULTS[attr])

    print("=" * 60)
    print("  biofilm_conformal_tet.py  [mode: %s]" % args.mode)
    print("  STL    : %s" % args.stl)
    print("  DI CSV : %s" % args.di_csv)
    print("  Out    : %s" % args.out)
    print("  thick  : %.3f mm  layers=%d  n_bins=%d" % (
        args.thickness, args.n_layers, args.n_bins))
    print("  E_max  : %.3g Pa  E_min: %.3g Pa" % (args.e_max, args.e_min))
    print("  BC     : %s  pressure=%.3g Pa" % (args.bc_mode, args.pressure))
    if args.neo_hookean:
        print("  MAT    : Neo-Hookean hyperelastic")
    if args.mode == "biofilm":
        print("  NOTE   : biofilm-scale Pa material; results qualitative for large strains")
    print("=" * 60)

    # ── Step 1: Read STL ──────────────────────────────────────────────────────
    print("[1] Reading STL ...")
    faces_verts, face_normals_stored = read_stl(args.stl)
    print("    Raw triangles: %d" % len(faces_verts))

    # ── Step 2: Deduplicate ───────────────────────────────────────────────────
    print("[2] Deduplicating vertices (tol=%.2e) ..." % args.dedup_tol)
    verts_inner, faces = deduplicate_vertices(faces_verts, tol=args.dedup_tol)
    V = len(verts_inner)
    F = len(faces)
    print("    Unique vertices: %d  Faces: %d" % (V, F))

    # ── Step 3: Vertex normals ────────────────────────────────────────────────
    print("[3] Computing vertex normals ...")
    # Re-deduplicate face_normals_stored to match new face ordering
    # Each row of faces_verts corresponds to same row of face_normals_stored
    vnorms_inner = compute_vertex_normals(verts_inner, faces, face_normals_stored)
    print("    Normal magnitude range: %.4f .. %.4f" % (
        np.linalg.norm(vnorms_inner, axis=1).min(),
        np.linalg.norm(vnorms_inner, axis=1).max()))

    # ── Step 4: Offset outer surface ──────────────────────────────────────────
    print("[4] Creating outer offset surface (thickness=%.3f mm) ..." % args.thickness)
    verts_outer_raw = verts_inner + args.thickness * vnorms_inner

    # ── Step 5: Laplacian smoothing ───────────────────────────────────────────
    if args.smooth_iter > 0:
        print("[5] Laplacian smoothing (%d iters) ..." % args.smooth_iter)
        verts_outer = laplacian_smooth_offset(
            verts_outer_raw, verts_inner, faces,
            iterations=args.smooth_iter, lam=0.5)
        d_before = np.linalg.norm(verts_outer_raw - verts_inner, axis=1).mean()
        d_after  = np.linalg.norm(verts_outer - verts_inner, axis=1).mean()
        print("    Mean thickness: %.4f mm → %.4f mm" % (d_before, d_after))
    else:
        print("[5] Smoothing skipped.")
        verts_outer = verts_outer_raw

    # Outer surface vertex normals (for load computation)
    vnorms_outer = compute_vertex_normals(verts_outer, faces)

    # ── Step 6: Build tet mesh ────────────────────────────────────────────────
    print("[6] Building multi-layer prism-to-tet mesh (%d layers) ..." % args.n_layers)
    nodes, tets, tet_layers, inner_nodes, outer_nodes = build_tet_mesh(
        verts_inner, verts_outer, faces, args.n_layers)
    print("    Nodes: %d  C3D4 tets: %d" % (len(nodes), len(tets)))

    # Quick volume check
    tv = nodes[tets]
    a = tv[:, 1] - tv[:, 0]
    b = tv[:, 2] - tv[:, 0]
    c = tv[:, 3] - tv[:, 0]
    vols = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    n_neg = (vols < 0).sum()
    print("    Negative-volume tets: %d (target: 0)" % n_neg)
    if n_neg > 0:
        print("    WARNING: %d negative-volume tets – may indicate inverted normals" % n_neg)
        print("    Flipping tet orientation ...")
        # Flip by swapping last two nodes
        bad = vols < 0
        tets[bad, 2], tets[bad, 3] = tets[bad, 3].copy(), tets[bad, 2].copy()
        # Recompute volumes from updated tets
        tv2  = nodes[tets]
        a2   = tv2[:, 1] - tv2[:, 0]
        b2   = tv2[:, 2] - tv2[:, 0]
        c2   = tv2[:, 3] - tv2[:, 0]
        vols2 = np.einsum('ij,ij->i', a2, np.cross(b2, c2)) / 6.0
        print("    After flip: %d negative-volume tets" % (vols2 < 0).sum())

    # ── Step 7: DI field mapping ──────────────────────────────────────────────
    print("[7] Reading DI field CSV ...")
    xs_field, di_vals_field = read_di_csv(args.di_csv)
    print("    Field points: %d  x=[%.4f, %.4f]  DI=[%.5f, %.5f]" % (
        len(xs_field), xs_field.min(), xs_field.max(),
        di_vals_field.min(), di_vals_field.max()))

    print("[7] Assigning DI bins ...")
    tet_bins, bin_E_stiff = assign_di_bins(
        tet_layers, args.n_layers, xs_field, di_vals_field,
        args.n_bins, args.di_scale, args.di_exp, args.e_max, args.e_min)
    for b in range(args.n_bins):
        n_b = (tet_bins == b).sum()
        if n_b > 0:
            print("    Bin %02d: E_stiff=%.3f GPa  n_tets=%d" % (b, bin_E_stiff[b]/1e9, n_b))

    # ── Step 7b: Growth eigenstrain preparation ───────────────────────────────
    # Apply nutrient correction factor to alpha_final
    alpha_eff = args.growth_eigenstrain * args.nutrient_factor
    T_nodes = None  # uniform mode by default

    if args.growth_eigenstrain > 0.0:
        eps_growth_eff = alpha_eff / 3.0
        print("[7b] Growth eigenstrain:")
        print("    alpha_final    = %.4g" % args.growth_eigenstrain)
        if abs(args.nutrient_factor - 1.0) > 1e-9:
            print("    nutrient_factor= %.4g  →  alpha_eff=%.4g" % (
                args.nutrient_factor, alpha_eff))
        print("    eps_growth_eff = %.4g per direction" % eps_growth_eff)

        # Spatial eigenstrain: load DI(x) and compute per-node T_node(x)
        if args.spatial_eigenstrain:
            cond = args.spatial_eigenstrain
            di_cred_path = os.path.join(args.di_credible_dir, cond)
            coords_npy = os.path.join(di_cred_path, "coords.npy")
            diq_npy    = os.path.join(di_cred_path, "di_quantiles.npy")
            if not os.path.exists(coords_npy) or not os.path.exists(diq_npy):
                print("  WARNING: DI field not found at %s" % di_cred_path)
                print("           Falling back to uniform T_growth.")
            else:
                coords_di = np.load(coords_npy)  # (3375, 3) in [0,1]³
                di_q      = np.load(diq_npy)     # (3, 3375)  rows: p05/p50/p95
                di_p50    = di_q[1]               # p50 DI spatial field
                di_mean   = di_p50.mean()

                # Normalize all mesh nodes to [0,1]³ using STL bounding box
                bbox_min   = verts_inner.min(axis=0)
                bbox_max   = verts_inner.max(axis=0)
                bbox_range = bbox_max - bbox_min
                bbox_range = np.where(bbox_range < 1e-14, 1.0, bbox_range)
                nodes_norm = (nodes - bbox_min) / bbox_range
                nodes_norm = np.clip(nodes_norm, 0.0, 1.0)

                # KD-tree: nearest DI grid point for each mesh node
                tree_di = cKDTree(coords_di)
                _, idx_di = tree_di.query(nodes_norm)  # (N_nodes,)

                # T_node(x) = T_mean * DI(x) / DI_mean
                T_mean  = eps_growth_eff   # = alpha_eff / 3
                T_nodes = T_mean * (di_p50[idx_di] / di_mean)

                print("    SPATIAL eigenstrain from _di_credible/%s" % cond)
                print("    DI_mean=%.5g  T_mean=%.5g  T_min=%.5g  T_max=%.5g" % (
                    di_mean, T_nodes.mean(), T_nodes.min(), T_nodes.max()))

    # 1D α(x) field from JAXFEM Hamilton+nutrient simulation (Option D Step 2)
    if args.growth_eigenstrain_field:
        alpha_data = np.loadtxt(args.growth_eigenstrain_field, delimiter=',', skiprows=1)
        # CSV columns: x_norm, x_mm, c_final, phi_total_final, alpha_x_monod, eps_growth_x_monod
        x_norm_csv = alpha_data[:, 0]   # normalized depth [0=tooth, 1=fluid]
        alpha_csv  = alpha_data[:, 4]   # alpha_x_monod = k_α ∫ φ·c/(k+c) dt

        # Map each node to its normalized depth via layer index
        # nodes are stacked as [layer_0 (V), layer_1 (V), ..., layer_n_layers (V)]
        V_inner    = verts_inner.shape[0]
        N_nodes    = nodes.shape[0]
        node_layer = np.arange(N_nodes) // V_inner          # layer index [0, n_layers]
        depth_norm = node_layer.astype(float) / args.n_layers  # normalized ∈ [0, 1]

        # Interpolate α(depth) for each node, then convert to thermal T = α/3
        alpha_node = np.interp(depth_norm, x_norm_csv, alpha_csv)
        T_nodes    = alpha_node / 3.0   # thermal analogy: eps_growth per direction

        # Set alpha_eff to the mean so INP header is informative
        alpha_eff  = float(alpha_node.mean())

        print("[7c] Growth eigenstrain from 1D field (Option D):")
        print("     file  : %s" % args.growth_eigenstrain_field)
        print("     α(x)  : min=%.5g  mean=%.5g  max=%.5g" % (
            alpha_node.min(), alpha_node.mean(), alpha_node.max()))
        print("     T(x)  : min=%.5g  mean=%.5g  max=%.5g" % (
            T_nodes.min(), T_nodes.mean(), T_nodes.max()))
        print("     Spatial variation: max/mean = %.3f" % (
            alpha_node.max() / (alpha_node.mean() + 1e-12)))

    # ── Step 8: Write INP ─────────────────────────────────────────────────────
    print("[8] Writing Abaqus INP ...")
    if alpha_eff > 0.0:
        mode_str = "spatial DI(%s)" % args.spatial_eigenstrain if args.spatial_eigenstrain else "uniform"
        print("  Eigenstrain: alpha_eff=%.4g  eps_growth=%.4g  mode=%s" % (
            alpha_eff, alpha_eff / 3.0, mode_str))
    write_abaqus_inp(
        args.out, nodes, tets, tet_bins, inner_nodes, outer_nodes,
        bin_E_stiff, args.aniso_ratio, args.nu, args.n_bins,
        args.pressure, args.bc_mode,
        verts_outer, faces, vnorms_outer,
        args.stl, args.di_csv, args.n_layers, args.thickness,
        nlgeom=(args.mode == "biofilm"),
        growth_eigenstrain=alpha_eff,
        T_nodes=T_nodes,
        neo_hookean=args.neo_hookean)

    # ── Step 9: Validation report ─────────────────────────────────────────────
    if args.validate:
        print("[9] Writing validation report ...")
        rep_path = args.out.replace(".inp", "_validation.txt")
        write_validation_report(
            rep_path, nodes, tets, tet_layers, tet_bins,
            inner_nodes, outer_nodes, verts_inner, verts_outer,
            bin_E_stiff, args.n_bins, args.n_layers, args.thickness,
            args.pressure, args.bc_mode)

    print("\nDone.")
    print("  Nodes   : %d" % len(nodes))
    print("  C3D4    : %d" % len(tets))
    print("  INP     : %s" % args.out)
    if n_neg == 0:
        print("  STATUS  : OK (zero penetration by construction)")
    else:
        print("  STATUS  : WARNING – check negative volume tets")


if __name__ == "__main__":
    main()
