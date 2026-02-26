"""
openjaw_p1_all_lower_import.py  –  Import OpenJaw Patient-1 mandible
and all available lower-tooth STL files as orphan-mesh parts into Abaqus CAE.

Runs inside Abaqus Python (noGUI):
  abaqus cae noGUI=openjaw_p1_all_lower_import.py

Produces: OpenJaw_P1_all_lower_withbiofilm.cae
  Parts  : P1_Mandible + P1_Tooth_* for all STL files in the Teeth directory
"""

from abaqus import mdb
from abaqusConstants import THREE_D, DEFORMABLE_BODY, ON, OFF, CARTESIAN, TRI3
from regionToolset import Region
import os
import json
import struct
import sys
import math


def _is_binary_stl(path):
    with open(path, "rb") as f:
        header = f.read(80)
        raw = f.read(4)
    if len(raw) < 4:
        return False
    n_tri = struct.unpack("<I", raw)[0]
    expected = 84 + 50 * n_tri
    actual = os.path.getsize(path)
    if abs(actual - expected) <= 4:
        return True
    try:
        if header.decode("ascii", errors="replace").strip().startswith("solid"):
            return False
    except Exception:
        pass
    return True


def _iter_binary_triangles(path, max_tri=None):
    with open(path, "rb") as f:
        f.read(80)
        raw = f.read(4)
        if len(raw) < 4:
            return
        n_tri = struct.unpack("<I", raw)[0]
        step = 1
        if max_tri and n_tri > max_tri:
            step = max(1, n_tri // max_tri)
        idx = 0
        read_count = 0
        while idx < n_tri:
            data = f.read(50)
            if len(data) < 50:
                break
            if idx % step == 0:
                vals = struct.unpack("<12fH", data)
                yield vals[3:6], vals[6:9], vals[9:12]
                read_count += 1
                if max_tri and read_count >= max_tri:
                    break
            idx += 1


def _iter_ascii_triangles(path, max_tri=None):
    verts = []
    count = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("vertex"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
            if len(verts) == 3:
                yield tuple(verts[0]), tuple(verts[1]), tuple(verts[2])
                verts = []
                count += 1
                if max_tri and count >= max_tri:
                    break


def import_stl_to_part(model, name, stl_path, max_tri=None):
    p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)

    if not os.path.isfile(stl_path):
        print("  [warn] STL not found: %s" % stl_path)
        return p

    binary = _is_binary_stl(stl_path)
    print(
        "  Importing %s  (%s STL%s) ..."
        % (
            os.path.basename(stl_path),
            "binary" if binary else "ASCII",
            "  max_tri=%d" % max_tri if max_tri else "",
        )
    )

    tri_iter = (
        _iter_binary_triangles(stl_path, max_tri)
        if binary
        else _iter_ascii_triangles(stl_path, max_tri)
    )

    label = 1
    n_elem = 0
    for v1, v2, v3 in tri_iter:
        n1 = p.Node(label=label, coordinates=v1)
        n2 = p.Node(label=label + 1, coordinates=v2)
        n3 = p.Node(label=label + 2, coordinates=v3)
        p.Element(nodes=(n1, n2, n3), elemShape=TRI3)
        label += 3
        n_elem += 1

    print("  → %d triangles imported, %d nodes" % (n_elem, label - 1))
    return p


def _part_bbox(part):
    if not part.nodes:
        return None
    xs = [nd.coordinates[0] for nd in part.nodes]
    ys = [nd.coordinates[1] for nd in part.nodes]
    zs = [nd.coordinates[2] for nd in part.nodes]
    return {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
        "center": [0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.5 * (min(zs) + max(zs))],
        "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
        "n_nodes": len(part.nodes),
    }


def build_template_bcs(model, inst, bbox):

    z_min = bbox["min"][2]
    z_size = bbox["size"][2]
    tol = 0.05 * z_size

    bot_faces = inst.faces.getByBoundingBox(
        xMin=-1e9,
        xMax=1e9,
        yMin=-1e9,
        yMax=1e9,
        zMin=z_min - tol,
        zMax=z_min + tol,
    )

    model.StaticStep(
        name="LOAD_TEMPLATE",
        previous="Initial",
        maxNumInc=100,
        initialInc=0.1,
        minInc=1e-5,
        maxInc=1.0,
    )
    if bot_faces:
        model.DisplacementBC(
            name="FIX_MANDIBLE_BASE",
            createStepName="Initial",
            region=Region(faces=bot_faces),
            u1=0.0,
            u2=0.0,
            u3=0.0,
        )


def _offset_polygon(pts, cx, cy, delta):
    out = []
    for x, y in pts:
        dx = x - cx
        dy = y - cy
        dist = math.sqrt(dx * dx + dy * dy) or 1.0
        out.append((x + delta * dx / dist, y + delta * dy / dist))
    return out


def _default_crown_pts(cx, cy, hx, hy):
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


def _load_tooth_polygons(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    out = {}
    for name, rec in data.items():
        poly = rec.get("cross_section_polygon")
        if not poly:
            continue
        pts = []
        for xy in poly:
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                pts.append((float(xy[0]), float(xy[1])))
        if pts:
            out[name] = pts
    return out


def _draw_closed_polygon(sk, pts):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        sk.Line(point1=(x1, y1), point2=(x2, y2))


def _ensure_biofilm_section(model):
    if "MAT_BIOFILM" not in model.materials.keys():
        mat = model.Material(name="MAT_BIOFILM")
        mat.Elastic(table=((1.0e9, 0.3),))
    if "SEC_BIOFILM" not in model.sections.keys():
        model.HomogeneousSolidSection(name="SEC_BIOFILM", material="MAT_BIOFILM")


def build_crown_biofilm(model, bb23, poly_pts=None):
    t_cx, t_cy, _ = bb23["center"]
    t_sx, t_sy, t_sz = bb23["size"]
    t_zmin = bb23["min"][2]

    if not poly_pts:
        poly_pts = _default_crown_pts(t_cx, t_cy, t_sx, t_sy)
    r_inner = 0.5 * min(t_sx, t_sy)
    biofilm_th = 0.15 * r_inner
    outer_pts = _offset_polygon(poly_pts, t_cx, t_cy, biofilm_th)

    sheet_sz = max(t_sx, t_sy) * 4.0
    sk = model.ConstrainedSketch(name="SkCrown", sheetSize=sheet_sz)
    _draw_closed_polygon(sk, outer_pts)
    _draw_closed_polygon(sk, poly_pts)

    part = model.Part(name="CrownBiofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=t_sz)

    seed_size = max(biofilm_th * 0.6, 0.3)
    part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    _ensure_biofilm_section(model)
    if part.elements:
        region = Region(elements=part.elements)
        part.SectionAssignment(region=region, sectionName="SEC_BIOFILM")

    return part, t_zmin, t_sz, seed_size


def build_slit_biofilm(model, bb30, bb31, poly30=None, poly31=None, pocket_depth=3.0):
    c30 = bb30["center"]
    s30 = bb30["size"]
    c31 = bb31["center"]
    s31 = bb31["size"]

    dx = c31[0] - c30[0]
    dy = c31[1] - c30[1]
    dxy = math.sqrt(dx * dx + dy * dy) or 1.0
    nx = dx / dxy
    ny = dy / dxy
    tx = -ny
    ty = nx

    mid_x = 0.5 * (c30[0] + c31[0])
    mid_y = 0.5 * (c30[1] + c31[1])

    vals_n30 = []
    vals_t30 = []
    vals_n31 = []
    vals_t31 = []
    if poly30:
        for x, y in poly30:
            vx = x - mid_x
            vy = y - mid_y
            vals_n30.append(vx * nx + vy * ny)
            vals_t30.append(vx * tx + vy * ty)
    if poly31:
        for x, y in poly31:
            vx = x - mid_x
            vy = y - mid_y
            vals_n31.append(vx * nx + vy * ny)
            vals_t31.append(vx * tx + vy * ty)

    if vals_n30 and vals_n31:
        n30_max = max(vals_n30)
        n31_min = min(vals_n31)
        gap_n = max(n31_min - n30_max, 0.0)
    else:
        gap_n = pocket_depth

    if vals_t30 or vals_t31:
        t_min = min(vals_t30 + vals_t31)
        t_max = max(vals_t30 + vals_t31)
        width_t = t_max - t_min
    else:
        width_t = max(s30[0], s31[0], s30[1], s31[1])

    slit_half_depth = 0.25 * gap_n if gap_n > 0.0 else 0.5 * pocket_depth
    slit_half_width = 0.5 * width_t

    z_lo = max(bb30["min"][2], bb31["min"][2])
    z_hi = min(bb30["max"][2], bb31["max"][2])
    if z_hi <= z_lo:
        z_mid = 0.5 * (c30[2] + c31[2])
        z_lo = z_mid - 5.0
        z_hi = z_mid + 5.0
    slit_h = z_hi - z_lo

    corners = []
    for sn in (-slit_half_depth, slit_half_depth):
        for st in (-slit_half_width, slit_half_width):
            corners.append(
                (
                    mid_x + sn * nx + st * tx,
                    mid_y + sn * ny + st * ty,
                )
            )
    c_order = [corners[0], corners[2], corners[3], corners[1]]

    sheet_sz = max(s30[0] + s31[0], s30[1] + s31[1]) * 4.0
    sk = model.ConstrainedSketch(name="SkSlit", sheetSize=sheet_sz)
    _draw_closed_polygon(sk, c_order)

    part = model.Part(name="SlitBiofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=slit_h)

    seed_size = min(pocket_depth * 0.4, 1.0)
    part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    _ensure_biofilm_section(model)
    if part.elements:
        region = Region(elements=part.elements)
        part.SectionAssignment(region=region, sectionName="SEC_BIOFILM")

    return part, mid_x, mid_y, z_lo, slit_h, slit_half_depth, nx, ny, seed_size


def build_crown_bcs(model, inst, t_zmin, t_sz, pressure_pa, seed_size):
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
    if top_faces:
        model.Pressure(
            name="PRESS_CROWN",
            createStepName="LOAD_CROWN",
            region=Region(side1Faces=top_faces),
            magnitude=pressure_pa,
        )


def build_slit_bcs(
    model, inst, z_lo, slit_h, slit_half_depth, nx, ny, mid_x, mid_y, pressure_pa, seed_size
):
    tol = seed_size * 0.4

    bot_faces = inst.faces.getByBoundingBox(
        xMin=-1e9, xMax=1e9, yMin=-1e9, yMax=1e9, zMin=z_lo - tol, zMax=z_lo + tol
    )

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
    if top_faces:
        model.Pressure(
            name="PRESS_SLIT",
            createStepName="LOAD_SLIT",
            region=Region(side1Faces=top_faces),
            magnitude=pressure_pa,
        )


def build_openjaw_p1_all_lower(max_tri_mandible=200000, max_tri_tooth=None):
    model_name = "OpenJawP1AllLower"
    if model_name in mdb.models.keys():
        model = mdb.models[model_name]
    else:
        model = mdb.Model(name=model_name)

    base_dir = os.getcwd()
    stl_root = os.path.join(base_dir, "external_tooth_models", "OpenJaw_Dataset", "Patient_1")

    mandible_spec = [
        ("P1_Mandible", os.path.join(stl_root, "Mandible", "P1_Mandible.stl"), max_tri_mandible),
    ]

    teeth_dir = os.path.join(stl_root, "Teeth")
    tooth_specs = []
    if os.path.isdir(teeth_dir):
        for fname in sorted(os.listdir(teeth_dir)):
            if not fname.lower().endswith(".stl"):
                continue
            stem, _ = os.path.splitext(fname)
            if not stem.startswith("P1_Tooth_"):
                continue
            tooth_specs.append((stem, os.path.join(teeth_dir, fname), max_tri_tooth))

    parts_spec = mandible_spec + tooth_specs

    if "MAT_TOOTH" not in model.materials.keys():
        mat = model.Material(name="MAT_TOOTH")
        mat.Elastic(table=((1.0e10, 0.3),))
    if "SEC_TOOTH" not in model.sections.keys():
        model.HomogeneousSolidSection(name="SEC_TOOTH", material="MAT_TOOTH")

    imported_parts = []
    for part_name, stl_path, max_tri in parts_spec:
        if part_name in model.parts.keys():
            print("  [skip] part already exists: %s" % part_name)
            imported_parts.append(model.parts[part_name])
            continue
        part = import_stl_to_part(model, part_name, stl_path, max_tri)
        if len(part.elements) > 0:
            region = Region(elements=part.elements)
            part.SectionAssignment(region=region, sectionName="SEC_TOOTH")
        imported_parts.append(part)

    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)

    for part in imported_parts:
        inst_name = part.name + "-1"
        if inst_name not in a.instances.keys():
            a.Instance(name=inst_name, part=part, dependent=ON)

    mandible_part = None
    for part in imported_parts:
        if part.name == "P1_Mandible":
            mandible_part = part
            break
    if mandible_part:
        mandible_bbox = _part_bbox(mandible_part)
        if mandible_bbox and "P1_Mandible-1" in a.instances:
            inst = a.instances["P1_Mandible-1"]
            build_template_bcs(model, inst, mandible_bbox)

    json_poly_path = os.path.join(base_dir, "p1_tooth_bbox.json")
    tooth_polys = _load_tooth_polygons(json_poly_path)

    bbox_for_bio = {}
    for part in imported_parts:
        bb = _part_bbox(part)
        if bb:
            bbox_for_bio[part.name] = bb

    pressure_pa = 1.0e6

    bb23 = bbox_for_bio.get("P1_Tooth_23")
    poly23 = tooth_polys.get("P1_Tooth_23")
    if bb23:
        crown_part, t_zmin, t_sz, seed_crown = build_crown_biofilm(model, bb23, poly23)
        inst_crown = a.Instance(name="CrownBiofilm-1", part=crown_part, dependent=ON)
        a.translate(instanceList=("CrownBiofilm-1",), vector=(0.0, 0.0, t_zmin))
        build_crown_bcs(model, inst_crown, t_zmin, t_sz, pressure_pa, seed_crown)

    bb30 = bbox_for_bio.get("P1_Tooth_30")
    bb31 = bbox_for_bio.get("P1_Tooth_31")
    poly30 = tooth_polys.get("P1_Tooth_30")
    poly31 = tooth_polys.get("P1_Tooth_31")
    if bb30 and bb31:
        slit_part, mid_x, mid_y, z_lo, slit_h, slit_half_depth, nx, ny, seed_slit = (
            build_slit_biofilm(model, bb30, bb31, poly30, poly31)
        )
        inst_slit = a.Instance(name="SlitBiofilm-1", part=slit_part, dependent=ON)
        a.translate(instanceList=("SlitBiofilm-1",), vector=(0.0, 0.0, z_lo))
        build_slit_bcs(
            model,
            inst_slit,
            z_lo,
            slit_h,
            slit_half_depth,
            nx,
            ny,
            mid_x,
            mid_y,
            pressure_pa,
            seed_slit,
        )

    cae_path = os.path.join(base_dir, "OpenJaw_P1_all_lower_withbiofilm.cae")
    mdb.saveAs(pathName=cae_path)
    print("[saved] %s" % cae_path)

    bbox_data = {}
    for part in imported_parts:
        bb = _part_bbox(part)
        if bb:
            bbox_data[part.name] = bb

    bbox_path = os.path.join(base_dir, "p1_tooth_bbox_all_lower_from_cae.json")
    with open(bbox_path, "w") as f:
        json.dump(bbox_data, f, indent=2)
    print("[saved] bbox → %s" % bbox_path)


def _parse_args(argv):
    cfg = {
        "job_name": "OJ_AllLower_WithBiofilm",
        "write_inp": True,
        "run_job": True,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--job-name" and i + 1 < len(argv):
            cfg["job_name"] = argv[i + 1]
            i += 2
            continue
        if a == "--no-run":
            cfg["run_job"] = False
            i += 1
            continue
        if a == "--no-inp":
            cfg["write_inp"] = False
            i += 1
            continue
        i += 1
    return cfg


def main():
    try:
        sep = sys.argv.index("--")
        user_args = sys.argv[sep + 1 :]
    except ValueError:
        user_args = sys.argv[1:]

    cfg = _parse_args(user_args)

    build_openjaw_p1_all_lower()

    model_name = "OpenJawP1AllLower"
    job = mdb.Job(
        name=cfg["job_name"],
        model=model_name,
        numCpus=1,
        description="OpenJaw all-lower with biofilm (crown + slit)",
    )
    if cfg["write_inp"]:
        job.writeInput()
    if cfg["run_job"]:
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()


if __name__ == "__main__":
    main()
