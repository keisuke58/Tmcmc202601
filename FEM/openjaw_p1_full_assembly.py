"""
openjaw_p1_full_assembly.py  –  Full assembly: real teeth + hollow biofilm solids
                                  + DI-mapped anisotropic materials + Abaqus jobs.

Runs inside Abaqus Python (noGUI):
  abaqus cae noGUI=openjaw_p1_full_assembly.py -- [options]

Two analysis cases
------------------
CROWN (T23):
  - Hollow ring biofilm solid around P1_Tooth_23 (inner = tooth surface, outer = biofilm)
  - T23 orphan-mesh part imported as rigid reference (E_max)
  - BC: fix bottom cross-section (root end)  →  1 MPa pressure on outer top face

SLIT (T30 + T31):
  - Thin rectangular slit solid in the inter-proximal gap between T30 and T31
  - Both tooth orphan-mesh parts imported (E_max)
  - Slit oriented perpendicular to T30→T31 contact-normal vector
  - BC: fix inner faces (2 sides adjacent to teeth)  →  1 MPa on outer buccal/lingual faces

DI mapping (both cases):
  - Radial / depth distance from tooth surface → normalised depth 0…1
  - Nearest x-value lookup in field CSV (x = depth axis)
  - 20 DI bins → MAT_ANISO_00..19  (engineering constants, transversely isotropic)

Usage (minimal):
  abaqus cae noGUI=openjaw_p1_full_assembly.py -- \\
      --bbox-json p1_tooth_bbox.json \\
      --field-csv abaqus_field_dh_3d.csv

All options:
  --bbox-json    JSON from stl_bbox.py          (required)
  --field-csv    DI field CSV                   (required)
  --case         crown | slit | both            (default: both)
  --biofilm-frac biofilm thickness / R_tooth    (default: 0.15)
  --pocket-depth gingival pocket depth (mm)     (default: 3.0)  [slit only]
  --pocket-width slit width in contact-normal direction (mm) (default: 2.0)
  --aniso-ratio  E_trans / E_stiff              (default: 0.5)
  --n-bins       number of DI bins              (default: 20)
  --e-max        Pa  (default: 10e9)
  --e-min        Pa  (default: 0.5e9)
  --di-scale     DI normalisation               (default: 0.025778)
  --di-exponent                                 (default: 2.0)
  --nu           Poisson ratio                  (default: 0.30)
  --pressure-mpa applied pressure in MPa        (default: 1.0)
  --poly-from-json  use cross_section_polygon from bbox JSON
  --cae-out      output CAE path                (default: OpenJaw_P1_assembly.cae)
  --crown-job    job name for crown case        (default: OJ_Crown_T23)
  --slit-job     job name for slit case         (default: OJ_Slit_T30T31)
"""

from __future__ import print_function, division
import sys
import os
import math
import json

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEF = dict(
    bbox_json=None,
    field_csv=None,
    case="both",
    biofilm_frac=0.15,
    pocket_depth=3.0,  # mm — gingival pocket depth (slit length in contact-normal direction)
    pocket_width=2.0,  # mm — slit width perpendicular to tooth surface
    aniso_ratio=0.5,
    n_bins=20,
    e_max=10.0e9,
    e_min=0.5e9,
    di_scale=0.025778,
    di_exponent=2.0,
    nu=0.30,
    pressure_mpa=1.0,
    poly_from_json=False,
    cae_out="OpenJaw_P1_assembly.cae",
    crown_job="OJ_Crown_T23",
    slit_job="OJ_Slit_T30T31",
)


def _parse(argv):
    cfg = dict(_DEF)
    i = 0
    while i < len(argv):
        a = argv[i]

        def nxt():
            return argv[i + 1] if i + 1 < len(argv) else None

        if a == "--bbox-json" and nxt():
            cfg["bbox_json"] = nxt()
            i += 2
            continue
        if a == "--field-csv" and nxt():
            cfg["field_csv"] = nxt()
            i += 2
            continue
        if a == "--case" and nxt():
            cfg["case"] = nxt()
            i += 2
            continue
        if a == "--biofilm-frac" and nxt():
            cfg["biofilm_frac"] = float(nxt())
            i += 2
            continue
        if a == "--pocket-depth" and nxt():
            cfg["pocket_depth"] = float(nxt())
            i += 2
            continue
        if a == "--pocket-width" and nxt():
            cfg["pocket_width"] = float(nxt())
            i += 2
            continue
        if a == "--aniso-ratio" and nxt():
            cfg["aniso_ratio"] = float(nxt())
            i += 2
            continue
        if a == "--n-bins" and nxt():
            cfg["n_bins"] = int(nxt())
            i += 2
            continue
        if a == "--e-max" and nxt():
            cfg["e_max"] = float(nxt())
            i += 2
            continue
        if a == "--e-min" and nxt():
            cfg["e_min"] = float(nxt())
            i += 2
            continue
        if a == "--di-scale" and nxt():
            cfg["di_scale"] = float(nxt())
            i += 2
            continue
        if a == "--di-exponent" and nxt():
            cfg["di_exponent"] = float(nxt())
            i += 2
            continue
        if a == "--nu" and nxt():
            cfg["nu"] = float(nxt())
            i += 2
            continue
        if a == "--pressure-mpa" and nxt():
            cfg["pressure_mpa"] = float(nxt())
            i += 2
            continue
        if a == "--poly-from-json":
            cfg["poly_from_json"] = True
            i += 1
            continue
        if a == "--cae-out" and nxt():
            cfg["cae_out"] = nxt()
            i += 2
            continue
        if a == "--crown-job" and nxt():
            cfg["crown_job"] = nxt()
            i += 2
            continue
        if a == "--slit-job" and nxt():
            cfg["slit_job"] = nxt()
            i += 2
            continue
        i += 1
    if cfg["bbox_json"] is None:
        raise RuntimeError("--bbox-json required")
    if cfg["field_csv"] is None:
        raise RuntimeError("--field-csv required")
    return cfg


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------


def _di_to_E_stiff(di_val, e_max, e_min, di_scale, exp):
    if di_scale <= 0:
        return e_max
    r = max(0.0, min(1.0, di_val / di_scale))
    return e_max * (1.0 - r) ** exp + e_min * r


def _eng_const(E_stiff, E_trans, nu):
    G_st = E_stiff / (2.0 * (1.0 + nu))
    G_tr = E_trans / (2.0 * (1.0 + nu))
    return (E_stiff, E_trans, E_trans, nu, nu, nu, G_st, G_st, G_tr)


# ---------------------------------------------------------------------------
# DI field reader
# ---------------------------------------------------------------------------


def _read_field_csv(path):
    coords, di_vals = [], []
    xi = yi = di_i = zi = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().lower() for p in line.split(",")]
            if xi is None:
                try:
                    xi = parts.index("x")
                    yi = parts.index("y")
                    di_i = parts.index("di")
                    zi = parts.index("z") if "z" in parts else None
                except ValueError:
                    continue
                continue
            try:
                x = float(line.split(",")[xi])
                y = float(line.split(",")[yi])
                dv = float(line.split(",")[di_i])
                z = float(line.split(",")[zi]) if zi is not None else 0.0
                coords.append((x, y, z))
                di_vals.append(dv)
            except (ValueError, IndexError):
                continue
    return coords, di_vals


def _field_depth_range(coords):
    if not coords:
        return 0.0, 1.0
    xs = [c[0] for c in coords]
    return min(xs), max(xs)


# ---------------------------------------------------------------------------
# Polygon helpers (cross-section geometry)
# ---------------------------------------------------------------------------


def _offset_polygon(pts, cx, cy, delta):
    """Offset polygon points outward from centroid (cx,cy) by delta."""
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        dist = math.sqrt(dx * dx + dy * dy) or 1.0
        out.append((x + delta * dx / dist, y + delta * dy / dist))
    return out


def _default_crown_pts(cx, cy, hx, hy):
    """8-point polygon for crown cross-section (inner edge at tooth surface)."""
    shape = [
        (0.0, -1.0),
        (0.6, -0.9),
        (0.9, -0.2),
        (0.6, 0.9),
        (0.0, 1.0),
        (-0.6, 0.9),
        (-0.9, -0.2),
        (-0.6, -0.9),
    ]
    return [(cx + 0.5 * hx * ux, cy + 0.5 * hy * uy) for ux, uy in shape]


def _draw_closed_polygon(sk, pts):
    """Draw a closed polygon in an Abaqus ConstrainedSketch."""
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        sk.Line(point1=(x1, y1), point2=(x2, y2))


# ---------------------------------------------------------------------------
# Binary STL reader (self-contained; duplicate of openjaw_p1_auto_import.py)
# ---------------------------------------------------------------------------

import struct


def _is_binary_stl(path):
    with open(path, "rb") as f:
        hdr = f.read(80)
        raw = f.read(4)
    if len(raw) < 4:
        return False
    n_tri = struct.unpack("<I", raw)[0]
    if abs(os.path.getsize(path) - (84 + 50 * n_tri)) <= 4:
        return True
    try:
        if hdr.decode("ascii", errors="replace").strip().startswith("solid"):
            return False
    except Exception:
        pass
    return True


def _import_stl_part(model, name, stl_path, max_tri=None):
    from abaqusConstants import THREE_D, DEFORMABLE_BODY, TRI3

    p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    if not os.path.isfile(stl_path):
        print("  [warn] STL not found: %s" % stl_path)
        return p
    binary = _is_binary_stl(stl_path)
    label = 1
    n_elem = 0
    with open(stl_path, "rb") as f:
        if binary:
            f.read(80)
            raw = f.read(4)
            if len(raw) < 4:
                return p
            n_tri = struct.unpack("<I", raw)[0]
            step = max(1, n_tri // max_tri) if max_tri and n_tri > max_tri else 1
            for idx in range(n_tri):
                data = f.read(50)
                if len(data) < 50:
                    break
                if idx % step:
                    continue
                vals = struct.unpack("<12fH", data)
                v1, v2, v3 = vals[3:6], vals[6:9], vals[9:12]
                n1 = p.Node(label=label, coordinates=v1)
                n2 = p.Node(label=label + 1, coordinates=v2)
                n3 = p.Node(label=label + 2, coordinates=v3)
                p.Element(nodes=(n1, n2, n3), elemShape=TRI3)
                label += 3
                n_elem += 1
                if max_tri and n_elem >= max_tri:
                    break
        else:
            verts = []
            for line in f.read().decode("utf-8", errors="replace").splitlines():
                ln = line.strip()
                if not ln.startswith("vertex"):
                    continue
                pts = ln.split()
                if len(pts) == 4:
                    verts.append((float(pts[1]), float(pts[2]), float(pts[3])))
                if len(verts) == 3:
                    n1 = p.Node(label=label, coordinates=verts[0])
                    n2 = p.Node(label=label + 1, coordinates=verts[1])
                    n3 = p.Node(label=label + 2, coordinates=verts[2])
                    p.Element(nodes=(n1, n2, n3), elemShape=TRI3)
                    label += 3
                    n_elem += 1
                    verts = []
                    if max_tri and n_elem >= max_tri:
                        break
    print("  STL '%s': %d triangles" % (name, n_elem))
    return p


# ---------------------------------------------------------------------------
# DI → bin assignment (element centroid → rho_norm → nearest field CSV row)
# ---------------------------------------------------------------------------


def _assign_di_bins(part, cx, cy, R_inner, biofilm_th, coords, di_vals, di_max, n_bins):
    """
    Assign each element of `part` to a DI bin based on radial depth.
    Returns list of label-lists, one per bin.
    """
    bin_w = di_max / n_bins if n_bins > 0 else 1.0
    x_fmin, x_fmax = _field_depth_range(coords)
    x_frange = x_fmax - x_fmin or 1.0
    bin_labels = [[] for _ in range(n_bins)]

    for elem in part.elements:
        nodes = elem.getNodes()
        if not nodes:
            continue
        sx = sy = 0.0
        nc = 0
        for nd in nodes:
            c = nd.coordinates
            sx += c[0]
            sy += c[1]
            nc += 1
        if nc == 0:
            continue
        ex, ey = sx / nc, sy / nc
        rho = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        rho_norm = max(0.0, min(1.0, (rho - R_inner) / max(biofilm_th, 1e-10)))
        x_q = x_fmin + rho_norm * x_frange
        # nearest x in field CSV
        best_di = 0.0
        best_d2 = 1.0e30
        for (xv, yv, zv), dv in zip(coords, di_vals):
            dx2 = (xv - x_q) ** 2
            if dx2 < best_d2:
                best_d2 = dx2
                best_di = dv
        b_idx = max(0, min(n_bins - 1, int(best_di / bin_w)))
        bin_labels[b_idx].append(elem.label)

    return bin_labels


# ---------------------------------------------------------------------------
# Assign sections to elements
# ---------------------------------------------------------------------------


def _apply_section_assignments(part, bin_labels, n_bins):
    from abaqusConstants import FROM_SECTION, MIDDLE_SURFACE
    from regionToolset import Region

    total = 0
    for b in range(n_bins):
        if not bin_labels[b]:
            continue
        seq = part.elements.sequenceFromLabels(labels=bin_labels[b])
        reg = Region(elements=seq)
        part.SectionAssignment(
            region=reg,
            sectionName="SEC_ANISO_%02d" % b,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
        total += len(bin_labels[b])
    return total


# ---------------------------------------------------------------------------
# Material + section creation (shared across models)
# ---------------------------------------------------------------------------


def _create_materials(model, cfg):
    from abaqusConstants import ENGINEERING_CONSTANTS

    n_bins = cfg["n_bins"]
    aniso_ratio = cfg["aniso_ratio"]
    di_max = cfg["_di_max"]
    bin_w = di_max / n_bins if n_bins > 0 else 1.0

    for b in range(n_bins):
        di_b = (b + 0.5) * bin_w
        E_stiff = _di_to_E_stiff(
            di_b, cfg["e_max"], cfg["e_min"], cfg["di_scale"], cfg["di_exponent"]
        )
        E_trans = aniso_ratio * E_stiff
        consts = _eng_const(E_stiff, E_trans, cfg["nu"])
        mname = "MAT_ANISO_%02d" % b
        sname = "SEC_ANISO_%02d" % b
        if mname not in model.materials.keys():
            mat = model.Material(name=mname)
            mat.Elastic(type=ENGINEERING_CONSTANTS, table=(consts,))
        if sname not in model.sections.keys():
            model.HomogeneousSolidSection(name=sname, material=mname, thickness=None)

    # Tooth material (elastic isotropic, E_max)
    if "MAT_TOOTH" not in model.materials.keys():
        mat_t = model.Material(name="MAT_TOOTH")
        mat_t.Elastic(table=((cfg["e_max"], cfg["nu"]),))
    if "SEC_TOOTH" not in model.sections.keys():
        model.HomogeneousSolidSection(name="SEC_TOOTH", material="MAT_TOOTH", thickness=None)


# ============================================================
# CASE 1: Crown biofilm around T23
# ============================================================


def build_crown_case(model, cfg, bb23, coords, di_vals):
    from abaqusConstants import (
        THREE_D,
        DEFORMABLE_BODY,
        CARTESIAN,
        SYSTEM,
        AXIS_1,
        ROTATION_NONE,
        STACK_3,
    )
    from regionToolset import Region

    t_cx, t_cy, t_cz = bb23["center"]
    t_sx, t_sy, t_sz = bb23["size"]
    t_zmin, t_zmax = bb23["min"][2], bb23["max"][2]

    # Inner polygon = tooth cross-section
    poly_pts = None
    if cfg["poly_from_json"] and "cross_section_polygon" in bb23:
        poly_pts = [(p[0], p[1]) for p in bb23["cross_section_polygon"]]
        print("  Crown inner polygon: %d points from JSON" % len(poly_pts))
    if not poly_pts:
        poly_pts = _default_crown_pts(t_cx, t_cy, t_sx, t_sy)

    R_inner = 0.5 * min(t_sx, t_sy)
    biofilm_th = cfg["biofilm_frac"] * R_inner
    outer_pts = _offset_polygon(poly_pts, t_cx, t_cy, biofilm_th)

    print("  Crown: R_inner=%.2f mm  biofilm_th=%.2f mm" % (R_inner, biofilm_th))

    # ── Hollow ring sketch (outer polygon − inner polygon) ─────────────────
    sheet_sz = max(t_sx, t_sy) * 4.0
    sk = model.ConstrainedSketch(name="SkCrown", sheetSize=sheet_sz)
    _draw_closed_polygon(sk, outer_pts)  # outer boundary
    _draw_closed_polygon(sk, poly_pts)  # inner boundary (hole)

    part = model.Part(name="CrownBiofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=t_sz)

    seed_size = max(biofilm_th * 0.6, 0.3)
    part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    print("  Crown mesh: %d elements" % len(part.elements))

    # ── Orientation datum (X = radial outward reference) ────────────────────
    datum = part.DatumCsysByThreePoints(
        coordSysType=CARTESIAN,
        origin=(t_cx, t_cy, 0.0),
        point1=(t_cx + 1.0, t_cy, 0.0),
        point2=(t_cx, t_cy + 1.0, 0.0),
    )
    local_csys = part.datums[datum.id]

    # ── DI bin assignment ────────────────────────────────────────────────────
    di_max = cfg["_di_max"]
    bin_labels = _assign_di_bins(
        part, t_cx, t_cy, R_inner, biofilm_th, coords, di_vals, di_max, cfg["n_bins"]
    )
    n_assigned = _apply_section_assignments(part, bin_labels, cfg["n_bins"])
    print("  Crown: %d elements assigned to DI bins." % n_assigned)

    # ── Material orientation ─────────────────────────────────────────────────
    all_seq = part.elements.sequenceFromLabels(labels=[e.label for e in part.elements])
    part.MaterialOrientation(
        region=Region(elements=all_seq),
        orientationType=SYSTEM,
        axis=AXIS_1,
        localCsys=local_csys,
        fieldName="",
        additionalRotationType=ROTATION_NONE,
        angle=0.0,
        additionalRotationField="",
        stackDirection=STACK_3,
    )

    return part, R_inner, biofilm_th, t_zmin, t_sz


def build_crown_bcs(model, inst, t_zmin, t_sz, pressure_pa, seed_size):
    from regionToolset import Region

    tol = seed_size * 0.3
    bot_z = t_zmin
    top_z = t_zmin + t_sz

    bot_faces = inst.faces.getByBoundingBox(
        xMin=-1e9, xMax=1e9, yMin=-1e9, yMax=1e9, zMin=bot_z - tol, zMax=bot_z + tol
    )
    top_faces = inst.faces.getByBoundingBox(
        xMin=-1e9, xMax=1e9, yMin=-1e9, yMax=1e9, zMin=top_z - tol, zMax=top_z + tol
    )

    model.StaticStep(
        name="LOAD_CROWN",
        previous="Initial",
        maxNumInc=100,
        initialInc=0.1,
        minInc=1e-5,
        maxInc=1.0,
    )
    if bot_faces:
        model.DisplacementBC(
            name="FIX_CROWN_BOT",
            createStepName="Initial",
            region=Region(faces=bot_faces),
            u1=0.0,
            u2=0.0,
            u3=0.0,
        )
        print("  Crown BCs: fixed %d bottom faces" % len(bot_faces))
    if top_faces:
        model.Pressure(
            name="PRESS_CROWN",
            createStepName="LOAD_CROWN",
            region=Region(side1Faces=top_faces),
            magnitude=pressure_pa,
        )
        print("  Crown BCs: pressure %.3g Pa on %d top faces" % (pressure_pa, len(top_faces)))


# ============================================================
# CASE 2: Inter-proximal slit between T30 and T31
# ============================================================


def build_slit_case(model, cfg, bb30, bb31, coords, di_vals):
    """
    Create a thin rectangular biofilm slit solid in the T30-T31 inter-proximal gap.

    The slit cross-section (in the XY plane):
      - Long axis: perpendicular to the T30→T31 contact-normal vector
      - Short axis: along the T30→T31 contact-normal (depth into pocket)
      - Width  (long)  = max(T30_sx, T31_sx)  mm  (full buccal-lingual extent)
      - Depth  (short) = pocket_depth mm (clinically ≈ 2-4 mm from contact point)

    The slit height (Z) = overlap of the two tooth z-ranges.
    """
    from abaqusConstants import (
        THREE_D,
        DEFORMABLE_BODY,
        CARTESIAN,
        SYSTEM,
        AXIS_1,
        ROTATION_NONE,
        STACK_3,
    )
    from regionToolset import Region

    c30 = bb30["center"]
    s30 = bb30["size"]
    c31 = bb31["center"]
    s31 = bb31["size"]

    # Contact normal: T30 → T31 (unit vector in XY)
    dx = c31[0] - c30[0]
    dy = c31[1] - c30[1]
    dxy = math.sqrt(dx * dx + dy * dy) or 1.0
    nx, ny = dx / dxy, dy / dxy  # contact normal (unit)
    tx, ty = -ny, nx  # tangent (perpendicular to contact normal)

    # Slit centre = mid-point between T30 and T31 centres
    mid_x = 0.5 * (c30[0] + c31[0])
    mid_y = 0.5 * (c30[1] + c31[1])

    # Slit extents (in contact-normal frame)
    slit_half_depth = 0.5 * cfg["pocket_depth"]  # mm in ±contact-normal direction
    slit_half_width = 0.5 * max(s30[0], s31[0], s30[1], s31[1])  # full width in tangent dir

    # Z range = overlap of T30 and T31
    z_lo = max(bb30["min"][2], bb31["min"][2])
    z_hi = min(bb30["max"][2], bb31["max"][2])
    if z_hi <= z_lo:
        # No overlap: use midpoint region ±5 mm
        z_lo = 0.5 * (c30[2] + c31[2]) - 5.0
        z_hi = 0.5 * (c30[2] + c31[2]) + 5.0
    slit_h = z_hi - z_lo

    print(
        "  Slit: mid=(%.2f,%.2f)  depth=%.2f  width=%.2f  z=%.2f..%.2f"
        % (mid_x, mid_y, slit_half_depth * 2, slit_half_width * 2, z_lo, z_hi)
    )
    print("  Slit contact-normal dir: (%.3f, %.3f)" % (nx, ny))

    # Four corners of the slit cross-section in XY
    #   n = contact-normal direction, t = tangent direction
    corners = []
    for sn in (-slit_half_depth, slit_half_depth):
        for st in (-slit_half_width, slit_half_width):
            corners.append(
                (
                    mid_x + sn * nx + st * tx,
                    mid_y + sn * ny + st * ty,
                )
            )
    # Order: CCW starting from bottom-left
    c_order = [corners[0], corners[2], corners[3], corners[1]]

    sheet_sz = max(s30[0] + s31[0], s30[1] + s31[1]) * 4.0
    sk = model.ConstrainedSketch(name="SkSlit", sheetSize=sheet_sz)
    _draw_closed_polygon(sk, c_order)

    part = model.Part(name="SlitBiofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=slit_h)

    seed_size = min(cfg["pocket_depth"] * 0.4, 1.0)
    part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    print("  Slit mesh: %d elements" % len(part.elements))

    # Orientation datum: contact-normal as e1 (direction of DI gradient into pocket)
    e1x, e1y = nx, ny
    e2x, e2y = tx, ty
    datum = part.DatumCsysByThreePoints(
        coordSysType=CARTESIAN,
        origin=(mid_x, mid_y, 0.0),
        point1=(mid_x + e1x, mid_y + e1y, 0.0),
        point2=(mid_x + e2x, mid_y + e2y, 0.0),
    )
    local_csys = part.datums[datum.id]

    # DI bin assignment for slit:
    # Use distance from slit centre (in contact-normal direction) as depth
    di_max = cfg["_di_max"]
    n_bins = cfg["n_bins"]
    bin_w = di_max / n_bins if n_bins > 0 else 1.0
    x_fmin, x_fmax = _field_depth_range(coords)
    x_frange = x_fmax - x_fmin or 1.0
    bin_labels = [[] for _ in range(n_bins)]

    for elem in part.elements:
        nodes = elem.getNodes()
        if not nodes:
            continue
        sx = sy = 0.0
        nc = 0
        for nd in nodes:
            c = nd.coordinates
            sx += c[0]
            sy += c[1]
            nc += 1
        if nc == 0:
            continue
        ex, ey = sx / nc, sy / nc
        # Depth = projection onto contact-normal from mid-point
        depth_n = abs((ex - mid_x) * nx + (ey - mid_y) * ny)
        rho_norm = max(0.0, min(1.0, depth_n / slit_half_depth))
        x_q = x_fmin + rho_norm * x_frange
        best_di = 0.0
        best_d2 = 1.0e30
        for (xv, yv, zv), dv in zip(coords, di_vals):
            dx2 = (xv - x_q) ** 2
            if dx2 < best_d2:
                best_d2 = dx2
                best_di = dv
        b_idx = max(0, min(n_bins - 1, int(best_di / bin_w)))
        bin_labels[b_idx].append(elem.label)

    n_assigned = _apply_section_assignments(part, bin_labels, n_bins)
    print("  Slit: %d elements assigned to DI bins." % n_assigned)

    # Material orientation
    all_seq = part.elements.sequenceFromLabels(labels=[e.label for e in part.elements])
    part.MaterialOrientation(
        region=Region(elements=all_seq),
        orientationType=SYSTEM,
        axis=AXIS_1,
        localCsys=local_csys,
        fieldName="",
        additionalRotationType=ROTATION_NONE,
        angle=0.0,
        additionalRotationField="",
        stackDirection=STACK_3,
    )

    return part, mid_x, mid_y, z_lo, slit_h, slit_half_depth, nx, ny


def build_slit_bcs(
    model, inst, z_lo, slit_h, slit_half_depth, nx, ny, mid_x, mid_y, pressure_pa, seed_size
):
    """
    Fix the two inner faces (adjacent to T30 and T31) and apply pressure
    on the two outer faces (buccal/lingual sides).
    """
    from regionToolset import Region

    tol = seed_size * 0.4

    # Inner faces: perpendicular to contact-normal, at ±slit_half_depth from mid
    # Face 1 (near T30): offset = -slit_half_depth in contact-normal direction
    # Face 2 (near T31): offset = +slit_half_depth in contact-normal direction
    # Since the geometry is extruded from z_lo with depth slit_h, we look for faces
    # by bounding box perpendicular to (nx,ny) at ±slit_half_depth from mid.

    # Fix: bottom cross-section (root end) = simplest stable BC
    bot_faces = inst.faces.getByBoundingBox(
        xMin=-1e9, xMax=1e9, yMin=-1e9, yMax=1e9, zMin=z_lo - tol, zMax=z_lo + tol
    )

    # Load: top cross-section (gingival opening)
    top_faces = inst.faces.getByBoundingBox(
        xMin=-1e9,
        xMax=1e9,
        yMin=-1e9,
        yMax=1e9,
        zMin=(z_lo + slit_h) - tol,
        zMax=(z_lo + slit_h) + tol,
    )

    model.StaticStep(
        name="LOAD_SLIT", previous="Initial", maxNumInc=100, initialInc=0.1, minInc=1e-5, maxInc=1.0
    )
    if bot_faces:
        model.DisplacementBC(
            name="FIX_SLIT_BOT",
            createStepName="Initial",
            region=Region(faces=bot_faces),
            u1=0.0,
            u2=0.0,
            u3=0.0,
        )
        print("  Slit BCs: fixed %d bottom faces" % len(bot_faces))
    if top_faces:
        model.Pressure(
            name="PRESS_SLIT",
            createStepName="LOAD_SLIT",
            region=Region(side1Faces=top_faces),
            magnitude=pressure_pa,
        )
        print("  Slit BCs: pressure %.3g Pa on %d top faces" % (pressure_pa, len(top_faces)))


# ============================================================
# Main build & run
# ============================================================


def _build_and_run(cfg):
    from abaqus import mdb
    from abaqusConstants import ON, OFF, CARTESIAN

    # ── Load bbox JSON ──────────────────────────────────────────────────────
    with open(cfg["bbox_json"]) as f:
        bbox_all = json.load(f)

    def _require_key(k):
        if k not in bbox_all:
            raise RuntimeError("'%s' not in bbox JSON (keys: %s)" % (k, list(bbox_all.keys())))
        return bbox_all[k]

    # ── Load DI field CSV ───────────────────────────────────────────────────
    print("Loading field CSV: %s" % cfg["field_csv"])
    coords, di_vals = _read_field_csv(cfg["field_csv"])
    if not coords:
        raise RuntimeError("No data read from field CSV")
    di_max = max(di_vals) if di_vals else 1.0
    cfg["_di_max"] = di_max
    di_scale = cfg["di_scale"] if cfg["di_scale"] > 0 else 1.1 * di_max
    cfg["di_scale"] = di_scale
    print("  %d DI points  di_max=%.5f  di_scale=%.5f" % (len(coords), di_max, di_scale))

    # ── Create/reset Abaqus model ───────────────────────────────────────────
    model_name = "OpenJawAssembly"
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    # ── Shared materials ────────────────────────────────────────────────────
    _create_materials(model, cfg)

    pressure_pa = cfg["pressure_mpa"] * 1.0e6
    run_crown = cfg["case"] in ("crown", "both")
    run_slit = cfg["case"] in ("slit", "both")

    # ── STL root for tooth imports ──────────────────────────────────────────
    base_dir = os.getcwd()
    stl_root = os.path.join(base_dir, "external_tooth_models", "OpenJaw_Dataset", "Patient_1")

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)

    # ── CROWN CASE ─────────────────────────────────────────────────────────
    if run_crown:
        print("\n=== CROWN CASE (T23) ===")
        bb23 = _require_key("P1_Tooth_23")

        crown_part, R_inner, biofilm_th, t_zmin, t_sz = build_crown_case(
            model, cfg, bb23, coords, di_vals
        )

        # Add to assembly (biofilm solid only; tooth STL omitted to avoid S3/solid section conflict)
        inst_crown = asm.Instance(name="CrownBioInst", part=crown_part, dependent=ON)
        # Translate biofilm so it starts at t_zmin
        asm.translate(instanceList=("CrownBioInst",), vector=(0.0, 0.0, t_zmin))

        seed_size = max(biofilm_th * 0.6, 0.3)
        build_crown_bcs(model, inst_crown, t_zmin, t_sz, pressure_pa, seed_size)

        crown_job = mdb.Job(
            name=cfg["crown_job"],
            model=model_name,
            numCpus=1,
            description="OpenJaw crown biofilm T23  beta=%.2f" % cfg["aniso_ratio"],
        )
        crown_job.submit(consistencyChecking=OFF)
        crown_job.waitForCompletion()
        print("[done] Crown job: %s" % cfg["crown_job"])

    # ── SLIT CASE ──────────────────────────────────────────────────────────
    if run_slit:
        print("\n=== SLIT CASE (T30 + T31) ===")
        bb30 = _require_key("P1_Tooth_30")
        bb31 = _require_key("P1_Tooth_31")

        # Need a fresh model for the slit case (separate job)
        slit_model_name = "OpenJawSlit"
        if slit_model_name in mdb.models.keys():
            del mdb.models[slit_model_name]
        slit_model = mdb.Model(name=slit_model_name)
        _create_materials(slit_model, cfg)

        slit_part, mid_x, mid_y, z_lo, slit_h, half_dep, nx, ny = build_slit_case(
            slit_model, cfg, bb30, bb31, coords, di_vals
        )

        slit_asm = slit_model.rootAssembly
        slit_asm.DatumCsysByDefault(CARTESIAN)
        inst_slit = slit_asm.Instance(name="SlitBioInst", part=slit_part, dependent=ON)
        # Translate slit solid to correct Z position (tooth STLs omitted to avoid S3/solid section conflict)
        slit_asm.translate(instanceList=("SlitBioInst",), vector=(0.0, 0.0, z_lo))

        seed_size_slit = min(cfg["pocket_depth"] * 0.4, 1.0)
        build_slit_bcs(
            slit_model,
            inst_slit,
            z_lo,
            slit_h,
            half_dep,
            nx,
            ny,
            mid_x,
            mid_y,
            pressure_pa,
            seed_size_slit,
        )

        slit_job = mdb.Job(
            name=cfg["slit_job"],
            model=slit_model_name,
            numCpus=1,
            description="OpenJaw slit T30-T31  beta=%.2f" % cfg["aniso_ratio"],
        )
        slit_job.submit(consistencyChecking=OFF)
        slit_job.waitForCompletion()
        print("[done] Slit job: %s" % cfg["slit_job"])

    # ── Save CAE ────────────────────────────────────────────────────────────
    cae_path = os.path.join(base_dir, cfg["cae_out"])
    mdb.saveAs(pathName=cae_path)
    print("\n[saved] %s" % cae_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    try:
        sep = sys.argv.index("--")
        user_args = sys.argv[sep + 1 :]
    except ValueError:
        user_args = sys.argv[1:]

    cfg = _parse(user_args)

    print("=" * 65)
    print("OpenJaw Full Assembly  –  Hollow Biofilm + DI Anisotropy")
    print("  case         : %s" % cfg["case"])
    print("  biofilm_frac : %.3f" % cfg["biofilm_frac"])
    print(
        "  pocket_depth : %.1f mm  pocket_width: %.1f mm"
        % (cfg["pocket_depth"], cfg["pocket_width"])
    )
    print("  aniso_ratio  : %.2f" % cfg["aniso_ratio"])
    print("  pressure     : %.2f MPa" % cfg["pressure_mpa"])
    print("=" * 65)

    _build_and_run(cfg)


main()
