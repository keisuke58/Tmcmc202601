"""
openjaw_p1_biofilm_solid.py  –  Build a DI-mapped biofilm solid around a real
                                 OpenJaw tooth and run an Abaqus static job.

Runs inside Abaqus Python (noGUI):
  abaqus cae noGUI=openjaw_p1_biofilm_solid.py -- \\
      --bbox-json   p1_tooth_bbox.json      \\  (from stl_bbox.py OR p1_tooth_bbox_from_cae.json)
      --tooth-key   P1_Tooth_23             \\  (key inside the JSON)
      --field-csv   abaqus_field_dh_3d.csv  \\  (from export_for_abaqus.py)
      [--geometry   crown|slit]             \\  (default: crown)
      [--biofilm-frac 0.15]                 \\  shell thickness / tooth radius
      [--pocket-frac  0.25]                 \\  gingival pocket width / tooth width (slit only)
      [--aniso-ratio  0.5]                  \\  E_trans / E_stiff
      [--e1-dir       radial]               \\  radial | x | y | z  (anisotropy axis)
      [--n-bins       20]                   \\
      [--e-max        10e9]                 \\
      [--e-min        0.5e9]                \\
      [--di-exponent  2.0]                  \\
      [--di-scale     0.025778]             \\
      [--nu           0.30]                 \\
      [--poly-from-json]                    \\  use cross_section_polygon from JSON (if present)
      [--job-name     OpenJawBioJob]

Material model: transverse isotropy (same as abaqus_biofilm_aniso_3d_real.py)
  E_stiff(DI) = E_max*(1-r)^n + E_min*r        (dominant direction)
  E_trans(DI) = aniso_ratio * E_stiff(DI)

DI → element mapping:
  For each biofilm element centroid, compute normalized radial distance from
  the tooth bounding-box centre:

    rho_norm = (rho - R_inner) / biofilm_thickness       (0=tooth surface, 1=outer)

  Then find the DI field CSV row whose x-coordinate (treated as normalised
  depth) is closest to rho_norm, and use that DI value to assign the bin.

  This couples the species-gradient direction (radial outward from tooth) to
  the DI field's dominant spatial axis without requiring coordinate registration.

Geometry modes
--------------
crown : polygonal tooth cross-section extruded along the tooth's long axis.
        Inner surface approximates the tooth STL via its bounding-box.
        Outer surface is offset by biofilm_frac * R_tooth.

slit  : gingival pocket slit — a rectangular pocket of width pocket_frac*dx
        centred on the tooth.  Same DI mapping.

Outputs
-------
  <job_name>.odb  (stress results)
  OpenJaw_<tooth_key>_biofilm.cae
"""

from __future__ import print_function, division
import sys
import os
import math
import json


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEF = {
    "bbox_json": None,
    "tooth_key": "P1_Tooth_23",
    "field_csv": None,
    "geometry": "crown",
    "biofilm_frac": 0.15,  # shell thickness / effective tooth radius
    "pocket_frac": 0.25,  # pocket width / tooth x-extent
    "aniso_ratio": 0.5,
    "e1_dir": "radial",  # 'radial' | 'x' | 'y' | 'z'
    "n_bins": 20,
    "e_max": 10.0e9,
    "e_min": 0.5e9,
    "di_exponent": 2.0,
    "di_scale": 0.025778,
    "nu": 0.30,
    "poly_from_json": False,
    "job_name": "OpenJawBioJob",
}


def _parse_args(argv):
    cfg = dict(_DEF)
    i = 0
    while i < len(argv):
        a = argv[i]

        def _nxt():
            return argv[i + 1] if i + 1 < len(argv) else None

        if a == "--bbox-json" and _nxt():
            cfg["bbox_json"] = _nxt()
            i += 2
            continue
        if a == "--tooth-key" and _nxt():
            cfg["tooth_key"] = _nxt()
            i += 2
            continue
        if a == "--field-csv" and _nxt():
            cfg["field_csv"] = _nxt()
            i += 2
            continue
        if a == "--geometry" and _nxt():
            cfg["geometry"] = _nxt()
            i += 2
            continue
        if a == "--biofilm-frac" and _nxt():
            cfg["biofilm_frac"] = float(_nxt())
            i += 2
            continue
        if a == "--pocket-frac" and _nxt():
            cfg["pocket_frac"] = float(_nxt())
            i += 2
            continue
        if a == "--aniso-ratio" and _nxt():
            cfg["aniso_ratio"] = float(_nxt())
            i += 2
            continue
        if a == "--e1-dir" and _nxt():
            cfg["e1_dir"] = _nxt()
            i += 2
            continue
        if a == "--n-bins" and _nxt():
            cfg["n_bins"] = int(_nxt())
            i += 2
            continue
        if a == "--e-max" and _nxt():
            cfg["e_max"] = float(_nxt())
            i += 2
            continue
        if a == "--e-min" and _nxt():
            cfg["e_min"] = float(_nxt())
            i += 2
            continue
        if a == "--di-exponent" and _nxt():
            cfg["di_exponent"] = float(_nxt())
            i += 2
            continue
        if a == "--di-scale" and _nxt():
            cfg["di_scale"] = float(_nxt())
            i += 2
            continue
        if a == "--nu" and _nxt():
            cfg["nu"] = float(_nxt())
            i += 2
            continue
        if a == "--poly-from-json":
            cfg["poly_from_json"] = True
            i += 1
            continue
        if a == "--job-name" and _nxt():
            cfg["job_name"] = _nxt()
            i += 2
            continue
        i += 1
    if cfg["bbox_json"] is None:
        raise RuntimeError("--bbox-json is required")
    if cfg["field_csv"] is None:
        raise RuntimeError("--field-csv is required")
    return cfg


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------


def _di_to_E_stiff(di_val, e_max, e_min, di_scale, exponent):
    if di_scale <= 0:
        return e_max
    r = max(0.0, min(1.0, di_val / di_scale))
    return e_max * (1.0 - r) ** exponent + e_min * r


def _engineering_constants(E_stiff, E_trans, nu):
    G_stiff = E_stiff / (2.0 * (1.0 + nu))
    G_trans = E_trans / (2.0 * (1.0 + nu))
    return (
        E_stiff,
        E_trans,
        E_trans,
        nu,
        nu,
        nu,
        G_stiff,
        G_stiff,
        G_trans,
    )


# ---------------------------------------------------------------------------
# DI field CSV reader
# ---------------------------------------------------------------------------


def _read_field_csv(path):
    """Return (coords, di_vals) where coords is list of (x,y,z) or (x,y)."""
    coords, di_vals = [], []
    with open(path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None:
                header = [p.strip().lower() for p in parts]
                # detect column indices
                try:
                    xi = header.index("x")
                    yi = header.index("y")
                    di_i = header.index("di")
                    zi = header.index("z") if "z" in header else None
                except ValueError:
                    header = None  # skip malformed header
                continue
            try:
                x = float(parts[xi])
                y = float(parts[yi])
                dv = float(parts[di_i])
                if zi is not None and zi < len(parts):
                    z = float(parts[zi])
                    coords.append((x, y, z))
                else:
                    coords.append((x, y, 0.0))
                di_vals.append(dv)
            except (ValueError, IndexError):
                continue
    return coords, di_vals


def _field_x_range(coords):
    """Return (x_min, x_max) of field coords; x treated as the depth axis."""
    if not coords:
        return 0.0, 1.0
    xs = [c[0] for c in coords]
    return min(xs), max(xs)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def _make_crown_polygon(cx, cy, hx, hy, biofilm_frac, poly_pts=None):
    """
    Return (inner_pts, outer_pts) for a crown cross-section.

    If poly_pts (list of [x,y]) is provided, use it as the inner boundary.
    Otherwise fall back to the 8-point polygon scaled to the bbox.

    The outer boundary is offset outward by biofilm_thickness = biofilm_frac * R_inner.
    R_inner = 0.5 * min(hx, hy).
    """
    R_inner = 0.5 * min(hx, hy)
    biofilm_th = biofilm_frac * R_inner

    if poly_pts and len(poly_pts) >= 3:
        inner = [(p[0], p[1]) for p in poly_pts]
    else:
        # 8-point polygon (same as abaqus_biofilm_aniso_3d_real.py)
        shape_uv = [
            (0.0, -1.0),
            (0.6, -0.9),
            (0.9, -0.2),
            (0.6, 0.9),
            (0.0, 1.0),
            (-0.6, 0.9),
            (-0.9, -0.2),
            (-0.6, -0.9),
        ]
        inner = []
        for ux, uy in shape_uv:
            inner.append((cx + 0.5 * hx * ux, cy + 0.5 * hy * uy))

    # Compute outward offset by biofilm_th (simple centroid-based offset)
    outer = []
    for x, y in inner:
        dx = x - cx
        dy = y - cy
        dist = math.sqrt(dx * dx + dy * dy) or 1.0
        x_out = x + biofilm_th * dx / dist
        y_out = y + biofilm_th * dy / dist
        outer.append((x_out, y_out))

    return inner, outer, R_inner, biofilm_th


def _make_slit_polygon(cx, cy, hx, hy, pocket_frac):
    """
    Return polygon corners for a gingival slit geometry.
    The slit is a thin rectangle of width pocket_frac*hx, full height hy.
    """
    pw = pocket_frac * hx
    x0, x1 = cx - 0.5 * pw, cx + 0.5 * pw
    y0, y1 = cy - 0.5 * hy, cy + 0.5 * hy
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ---------------------------------------------------------------------------
# Core build & run
# ---------------------------------------------------------------------------


def _build_and_run(cfg):
    from abaqus import mdb
    from abaqusConstants import (
        THREE_D,
        DEFORMABLE_BODY,
        ON,
        OFF,
        FROM_SECTION,
        MIDDLE_SURFACE,
        CARTESIAN,
        SYSTEM,
        AXIS_1,
        ROTATION_NONE,
        STACK_3,
        ENGINEERING_CONSTANTS,
    )
    from regionToolset import Region

    # ── Load bbox JSON ───────────────────────────────────────────────────────
    with open(cfg["bbox_json"], "r") as f:
        bbox_all = json.load(f)

    tooth_key = cfg["tooth_key"]
    if tooth_key not in bbox_all:
        raise RuntimeError(
            "tooth_key '%s' not in bbox JSON (available: %s)" % (tooth_key, list(bbox_all.keys()))
        )

    bb = bbox_all[tooth_key]
    t_cx, t_cy, t_cz = bb["center"]
    t_sx, t_sy, t_sz = bb["size"]
    t_zmin, t_zmax = bb["min"][2], bb["max"][2]

    # Use optional cross-section polygon from stl_bbox.py
    poly_pts = None
    if cfg["poly_from_json"] and "cross_section_polygon" in bb:
        poly_pts = bb["cross_section_polygon"]
        print("  Using %d-point cross-section polygon from JSON." % len(poly_pts))

    print(
        "Tooth: %s  center=(%.2f, %.2f, %.2f)  size=(%.2f, %.2f, %.2f)"
        % (tooth_key, t_cx, t_cy, t_cz, t_sx, t_sy, t_sz)
    )

    # ── Load DI field CSV ────────────────────────────────────────────────────
    coords, di_vals = _read_field_csv(cfg["field_csv"])
    if not coords:
        raise RuntimeError("No data in field CSV: %s" % cfg["field_csv"])

    di_max = max(di_vals) if di_vals else 1.0
    di_scale = cfg["di_scale"] if cfg["di_scale"] > 0 else 1.1 * di_max
    n_bins = cfg["n_bins"]
    bin_w = di_max / n_bins if n_bins > 0 else 1.0

    # x range of DI field → normalised depth axis
    x_field_min, x_field_max = _field_x_range(coords)
    x_field_range = x_field_max - x_field_min or 1.0

    print("DI field: %d points  di_max=%.5f  di_scale=%.5f" % (len(coords), di_max, di_scale))

    # ── Abaqus model ─────────────────────────────────────────────────────────
    model_name = "OpenJawBio_%s" % tooth_key.replace("P1_", "").replace("Tooth_", "T")
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    geom = cfg["geometry"]
    bio_frac = cfg["biofilm_frac"]
    pocket_frac = cfg["pocket_frac"]

    # ── Build 2-D sketch ─────────────────────────────────────────────────────
    sheet_size = max(t_sx, t_sy) * 4.0
    sk = model.ConstrainedSketch(name="Sk", sheetSize=sheet_size)

    if geom == "slit":
        corners = _make_slit_polygon(t_cx, t_cy, t_sx, t_sy, pocket_frac)
        for i in range(len(corners)):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % len(corners)]
            sk.Line(point1=(x1, y1), point2=(x2, y2))
        R_inner = 0.5 * min(t_sx, t_sy)
        biofilm_th = bio_frac * R_inner
    else:
        # crown (default)
        inner, outer, R_inner, biofilm_th = _make_crown_polygon(
            t_cx, t_cy, t_sx, t_sy, bio_frac, poly_pts
        )
        # Draw outer polygon as the extruded cross-section
        for i in range(len(outer)):
            x1, y1 = outer[i]
            x2, y2 = outer[(i + 1) % len(outer)]
            sk.Line(point1=(x1, y1), point2=(x2, y2))

    extrude_depth = t_sz  # same as tooth height
    part = model.Part(name="Biofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=extrude_depth)

    # Seed and mesh
    seed_size = max(R_inner * bio_frac * 0.5, 0.5)  # at least half the biofilm thickness
    part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    print("  Mesh: %d elements generated." % len(part.elements))

    # ── Orientation (anisotropy axis) ────────────────────────────────────────
    e1_dir = cfg["e1_dir"]
    if e1_dir == "x":
        e1 = (1.0, 0.0, 0.0)
    elif e1_dir == "y":
        e1 = (0.0, 1.0, 0.0)
    elif e1_dir == "z":
        e1 = (0.0, 0.0, 1.0)
    else:
        # radial: dominant gradient is radially outward from tooth centre in XY-plane
        e1 = (1.0, 0.0, 0.0)  # will be overridden per-element via rho_norm; global datum = X

    # Build a reference CSYS for SYSTEM-type orientation (Abaqus requires one)
    e2_ref = (0.0, 1.0, 0.0) if abs(e1[0]) < 0.9 else (0.0, 0.0, 1.0)
    datum_csys = part.DatumCsysByThreePoints(
        coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(e1[0], e1[1], e1[2]),
        point2=(e2_ref[0], e2_ref[1], e2_ref[2]),
    )
    local_csys = part.datums[datum_csys.id]

    # ── Materials (per DI bin) ────────────────────────────────────────────────
    aniso_ratio = cfg["aniso_ratio"]
    for b in range(n_bins):
        di_b = (b + 0.5) * bin_w
        E_stiff = _di_to_E_stiff(di_b, cfg["e_max"], cfg["e_min"], di_scale, cfg["di_exponent"])
        E_trans = aniso_ratio * E_stiff
        consts = _engineering_constants(E_stiff, E_trans, cfg["nu"])
        mname = "MAT_ANISO_%02d" % b
        sname = "SEC_ANISO_%02d" % b
        mat = model.Material(name=mname)
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=(consts,))
        model.HomogeneousSolidSection(name=sname, material=mname, thickness=None)

    # ── Element → DI bin assignment (radial depth mapping) ───────────────────
    bin_labels = [[] for _ in range(n_bins)]
    n_biofilm = 0

    for elem in part.elements:
        nodes = elem.getNodes()
        if not nodes:
            continue
        sx = sy = sz = 0.0
        nc = 0
        for nd in nodes:
            c = nd.coordinates
            sx += c[0]
            sy += c[1]
            sz += c[2]
            nc += 1
        if nc == 0:
            continue
        ex, ey = sx / nc, sy / nc

        # Radial distance from tooth centre in XY-plane
        dx = ex - t_cx
        dy = ey - t_cy
        rho = math.sqrt(dx * dx + dy * dy)

        if geom == "slit":
            # In slit geometry: depth = distance from tooth surface (approx. from tooth bbox edge)
            # Use distance from the tooth centre normalised by pocket half-width
            pw_half = 0.5 * pocket_frac * t_sx
            rho_norm = max(0.0, min(1.0, rho / (pw_half + biofilm_th)))
        else:
            # crown: depth = (rho - R_inner) / biofilm_th
            rho_norm = max(0.0, min(1.0, (rho - R_inner) / max(biofilm_th, 1e-10)))

        # Map rho_norm to DI field x-axis (x_field_min = surface, x_field_max = outer)
        x_query = x_field_min + rho_norm * x_field_range

        # Nearest neighbour in DI field (by x-coordinate only → 1-D lookup)
        best_di = 0.0
        best_dx2 = 1.0e30
        for (xv, yv, zv), dv in zip(coords, di_vals):
            dx2 = (xv - x_query) ** 2
            if dx2 < best_dx2:
                best_dx2 = dx2
                best_di = dv

        b_idx = int(best_di / bin_w)
        b_idx = max(0, min(n_bins - 1, b_idx))
        bin_labels[b_idx].append(elem.label)
        n_biofilm += 1

    # Assign sections
    total_assigned = 0
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
        total_assigned += len(bin_labels[b])
    print("  Assigned %d / %d elements across %d bins." % (total_assigned, n_biofilm, n_bins))

    # ── Material orientation (global datum CSYS for all elements) ─────────────
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

    # ── Assembly ──────────────────────────────────────────────────────────────
    asm = model.rootAssembly
    inst = asm.Instance(name="BioInst", part=part, dependent=ON)

    # Translate the biofilm solid so that its bottom face is at the actual tooth z_min
    asm.translate(instanceList=("BioInst",), vector=(0.0, 0.0, t_zmin))

    # ── Boundary conditions ───────────────────────────────────────────────────
    # Determine actual model extents after translation
    tol = seed_size * 0.2
    bot_z = t_zmin
    top_z = t_zmin + extrude_depth

    bot_faces = inst.faces.getByBoundingBox(
        xMin=-1e9, xMax=1e9, yMin=-1e9, yMax=1e9, zMin=bot_z - tol, zMax=bot_z + tol
    )
    top_faces = inst.faces.getByBoundingBox(
        xMin=-1e9, xMax=1e9, yMin=-1e9, yMax=1e9, zMin=top_z - tol, zMax=top_z + tol
    )

    model.StaticStep(
        name="LOAD", previous="Initial", maxNumInc=100, initialInc=0.1, minInc=1e-5, maxInc=1.0
    )

    if bot_faces:
        model.DisplacementBC(
            name="FIX_BOT",
            createStepName="Initial",
            region=Region(faces=bot_faces),
            u1=0.0,
            u2=0.0,
            u3=0.0,
        )

    if top_faces:
        model.Pressure(
            name="PRESS",
            createStepName="LOAD",
            region=Region(side1Faces=top_faces),
            magnitude=1.0e6,
        )

    # ── Save CAE ──────────────────────────────────────────────────────────────
    base_dir = os.getcwd()
    cae_path = os.path.join(base_dir, "OpenJaw_%s_biofilm.cae" % tooth_key.replace(" ", "_"))
    mdb.saveAs(pathName=cae_path)
    print("[saved] %s" % cae_path)

    # ── Submit job ────────────────────────────────────────────────────────────
    job = mdb.Job(
        name=cfg["job_name"],
        model=model_name,
        numCpus=1,
        description="OpenJaw biofilm %s  geom=%s  beta=%.2f" % (tooth_key, geom, aniso_ratio),
    )
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("[done] job=%s" % cfg["job_name"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    try:
        sep = sys.argv.index("--")
        user_args = sys.argv[sep + 1 :]
    except ValueError:
        user_args = sys.argv[1:]

    cfg = _parse_args(user_args)

    print("=" * 60)
    print("OpenJaw Biofilm Solid  –  DI-mapped transverse isotropy")
    print("  tooth_key    : %s" % cfg["tooth_key"])
    print("  geometry     : %s" % cfg["geometry"])
    print("  biofilm_frac : %.3f" % cfg["biofilm_frac"])
    print("  aniso_ratio  : %.2f" % cfg["aniso_ratio"])
    print("  e1_dir       : %s" % cfg["e1_dir"])
    print("  n_bins       : %d" % cfg["n_bins"])
    print("=" * 60)

    _build_and_run(cfg)


main()
