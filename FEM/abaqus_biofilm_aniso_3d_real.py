from __future__ import print_function, division

"""
abaqus_biofilm_aniso_3d_real.py  –  3D transversely isotropic biofilm model (real geometry variants)
=======================================================================
Runs inside the Abaqus Python environment:

  abaqus cae noGUI=abaqus_biofilm_aniso_3d_real.py -- \
      --field-csv abaqus_field_dh_3d.csv       \
      [--aniso-ratio 0.5]   # E_transverse / E_stiff  (1.0 = isotropic) \
      [--e1-x 1.0 --e1-y 0.0 --e1-z 0.0]  # dominant direction         \
      [--mapping power]                                                   \
      [--di-scale 0.025778]                                               \
      [--n-bins 20]                                                       \
      [--e-max 10e9] [--e-min 0.5e9]                                     \
      [--di-exponent 2.0]                                                 \
      [--nu 0.30]                                                         \
      [--geometry cube|crown|slit|real_crown|real_slit]                   \
      [--shell-frac 0.15] [--cap-frac 0.15] [--pocket-frac 0.25]          \
      [--job-name BiofilmAnisoJob]

Material model (C1 – transverse isotropy)
-----------------------------------------
For each DI bin b:
  E_stiff(DI_b) = E_max*(1-r)^n + E_min*r        (along e1, dominant grad dir)
  E_trans(DI_b) = aniso_ratio * E_stiff(DI_b)    (transverse: e2, e3)

  NU_12 = NU_13 = nu         (Poisson: stiff-transverse)
  NU_23 = nu                 (Poisson: within transverse plane)
  G_12  = E_stiff/(2*(1+nu)) (shear: stiff-transverse)
  G_23  = E_trans/(2*(1+nu)) (shear: within transverse plane)

Abaqus uses *ELASTIC, TYPE=ENGINEERING CONSTANTS:
  E1, E2, E3, NU12, NU13, NU23, G12, G13, G23
  (E1 = E_stiff, E2=E3=E_trans)

The orientation (e1 direction) is defined once globally via *ORIENTATION.
"""

import sys
import math


_DEF = {
    "field_csv":    None,
    "mapping":      "power",
    "n_bins":       20,
    "e_max":        10.0e9,
    "e_min":        0.5e9,
    "di_exponent":  2.0,
    "di_scale":     0.025778,
    "nu":           0.30,
    "aniso_ratio":  0.5,
    "e1_x":         1.0,
    "e1_y":         0.0,
    "e1_z":         0.0,
    "job_name":     "BiofilmAnisoJob",
    "geometry":     "cube",
    "shell_frac":   0.15,
    "cap_frac":     0.15,
    "pocket_frac":  0.25,
}


def _parse_args(argv):
    cfg = dict(_DEF)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--field-csv"   and i+1 < len(argv): cfg["field_csv"]   = argv[i+1]; i+=2; continue
        if a == "--mapping"     and i+1 < len(argv): cfg["mapping"]     = argv[i+1]; i+=2; continue
        if a == "--n-bins"      and i+1 < len(argv): cfg["n_bins"]      = int(argv[i+1]);   i+=2; continue
        if a == "--e-max"       and i+1 < len(argv): cfg["e_max"]       = float(argv[i+1]); i+=2; continue
        if a == "--e-min"       and i+1 < len(argv): cfg["e_min"]       = float(argv[i+1]); i+=2; continue
        if a == "--di-exponent" and i+1 < len(argv): cfg["di_exponent"] = float(argv[i+1]); i+=2; continue
        if a == "--di-scale"    and i+1 < len(argv): cfg["di_scale"]    = float(argv[i+1]); i+=2; continue
        if a == "--nu"          and i+1 < len(argv): cfg["nu"]          = float(argv[i+1]); i+=2; continue
        if a == "--aniso-ratio" and i+1 < len(argv): cfg["aniso_ratio"] = float(argv[i+1]); i+=2; continue
        if a == "--e1-x"        and i+1 < len(argv): cfg["e1_x"]        = float(argv[i+1]); i+=2; continue
        if a == "--e1-y"        and i+1 < len(argv): cfg["e1_y"]        = float(argv[i+1]); i+=2; continue
        if a == "--e1-z"        and i+1 < len(argv): cfg["e1_z"]        = float(argv[i+1]); i+=2; continue
        if a == "--job-name"    and i+1 < len(argv): cfg["job_name"]    = argv[i+1]; i+=2; continue
        if a == "--geometry"    and i+1 < len(argv): cfg["geometry"]    = argv[i+1]; i+=2; continue
        if a == "--shell-frac"  and i+1 < len(argv): cfg["shell_frac"]  = float(argv[i+1]); i+=2; continue
        if a == "--cap-frac"    and i+1 < len(argv): cfg["cap_frac"]    = float(argv[i+1]); i+=2; continue
        if a == "--pocket-frac" and i+1 < len(argv): cfg["pocket_frac"] = float(argv[i+1]); i+=2; continue
        i += 1
    if cfg["field_csv"] is None:
        raise RuntimeError("--field-csv required")
    return cfg


def _di_to_E_stiff(di_val, e_max, e_min, di_scale, exponent):
    if di_scale <= 0:
        return e_max
    r = max(0.0, min(1.0, di_val / di_scale))
    return e_max * (1.0 - r) ** exponent + e_min * r


def _engineering_constants(E_stiff, E_trans, nu):
    G_stiff = E_stiff / (2.0 * (1.0 + nu))
    G_trans = E_trans / (2.0 * (1.0 + nu))
    return (
        E_stiff, E_trans, E_trans,
        nu, nu, nu,
        G_stiff, G_stiff, G_trans,
    )


def _read_field_csv(path):
    coords, phi_pg, di_vals = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("x,") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
            phi_pg.append(float(parts[3]))
            di_vals.append(float(parts[4]))
    return coords, phi_pg, di_vals


def _build_and_run(cfg):
    from abaqus import mdb
    from abaqusConstants import (
        THREE_D, DEFORMABLE_BODY, ON, OFF, UNSET,
        FROM_SECTION, MIDDLE_SURFACE, ANALYSIS, SINGLE, DEFAULT,
        CARTESIAN, SYSTEM, AXIS_1, ROTATION_NONE, STACK_3,
        ENGINEERING_CONSTANTS,
    )
    from regionToolset import Region

    coords, phi_pg, di_vals = _read_field_csv(cfg["field_csv"])
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    di_max = max(di_vals) if di_vals else 1.0

    di_scale = cfg["di_scale"] if cfg["di_scale"] > 0 else 1.1 * di_max

    e1x, e1y, e1z = cfg["e1_x"], cfg["e1_y"], cfg["e1_z"]
    norm = math.sqrt(e1x*e1x + e1y*e1y + e1z*e1z) or 1.0
    e1 = [e1x/norm, e1y/norm, e1z/norm]

    if abs(e1[0]) < 0.9:
        rx, ry, rz = 1.0, 0.0, 0.0
    else:
        rx, ry, rz = 0.0, 1.0, 0.0
    e2x = e1[1]*rz - e1[2]*ry
    e2y = e1[2]*rx - e1[0]*rz
    e2z = e1[0]*ry - e1[1]*rx
    n2 = math.sqrt(e2x*e2x + e2y*e2y + e2z*e2z) or 1.0
    e2 = [e2x/n2, e2y/n2, e2z/n2]

    n_bins      = cfg["n_bins"]
    aniso_ratio = cfg["aniso_ratio"]
    bin_w       = di_max / n_bins if n_bins > 0 else 1.0

    nx_est = len(set(round(x, 6) for x in xs))
    ny_est = len(set(round(y, 6) for y in ys))
    nz_est = len(set(round(z, 6) for z in zs))
    elem_size = min(
        (x_max - x_min) / max(nx_est - 1, 1),
        (y_max - y_min) / max(ny_est - 1, 1),
        (z_max - z_min) / max(nz_est - 1, 1),
    )

    print("aniso_ratio=%.2f  e1=[%.3f,%.3f,%.3f]" % (
        aniso_ratio, e1[0], e1[1], e1[2]))
    print("di_scale=%.5f  di_max=%.5f  n_bins=%d" % (di_scale, di_max, n_bins))

    model_name = "BiofilmAniso3DReal"
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    geom = cfg.get("geometry", "cube")
    shell_frac = cfg.get("shell_frac", 0.15)
    cap_frac = cfg.get("cap_frac", 0.15)
    pocket_frac = cfg.get("pocket_frac", 0.25)

    sk = model.ConstrainedSketch(name="Sk", sheetSize=max(x_max-x_min, y_max-y_min)*3)
    if geom in ("crown", "real_crown"):
        hx = x_max - x_min
        hy = y_max - y_min
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
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
        pts = []
        for ux, uy in shape:
            x = cx + 0.5 * hx * ux
            y = cy + 0.5 * hy * uy
            pts.append((x, y))
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            sk.Line(point1=(x1, y1), point2=(x2, y2))
    elif geom in ("slit", "real_slit"):
        hx = x_max - x_min
        hy = y_max - y_min
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        shape_slit = [
            (-1.0, -1.0),
            ( 1.0, -1.0),
            ( 1.0,  0.8),
            ( 0.4,  0.9),
            ( 0.0,  0.6),
            (-0.4,  0.9),
            (-1.0,  0.8),
        ]
        pts_s = []
        for ux, uy in shape_slit:
            x = cx + 0.5 * hx * ux
            y = cy + 0.5 * hy * uy
            pts_s.append((x, y))
        for i in range(len(pts_s)):
            x1, y1 = pts_s[i]
            x2, y2 = pts_s[(i + 1) % len(pts_s)]
            sk.Line(point1=(x1, y1), point2=(x2, y2))
    else:
        sk.rectangle(point1=(x_min, y_min), point2=(x_max, y_max))
    part = model.Part(name="Biofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sk, depth=(z_max - z_min))
    part.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    datum_csys = part.DatumCsysByThreePoints(
        coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(e1[0], e1[1], e1[2]),
        point2=(e2[0], e2[1], e2[2]),
    )
    local_csys = part.datums[datum_csys.id]

    for b in range(n_bins):
        di_b     = (b + 0.5) * bin_w
        E_stiff  = _di_to_E_stiff(di_b, cfg["e_max"], cfg["e_min"],
                                   di_scale, cfg["di_exponent"])
        E_trans  = aniso_ratio * E_stiff
        consts   = _engineering_constants(E_stiff, E_trans, cfg["nu"])

        mname = "MAT_ANISO_%02d" % b
        sname = "SEC_ANISO_%02d" % b
        mat = model.Material(name=mname)
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=(consts,))
        model.HomogeneousSolidSection(name=sname, material=mname, thickness=None)

    bin_labels = [[] for _ in range(n_bins)]
    tooth_labels = []
    for elem in part.elements:
        nodes = elem.getNodes()
        if not nodes:
            continue
        sx = sy = sz = 0.0
        ncnt = 0
        for nd in nodes:
            c = nd.coordinates
            sx += c[0]; sy += c[1]; sz += c[2]; ncnt += 1
        if ncnt == 0:
            continue
        ex, ey, ez = sx / ncnt, sy / ncnt, sz / ncnt

        is_biofilm = True
        if geom in ("slit", "real_slit"):
            hx = x_max - x_min
            if hx > 0.0:
                cx = 0.5 * (x_min + x_max)
                pocket_w = pocket_frac * hx
                pocket_x_min = cx - 0.5 * pocket_w
                pocket_x_max = cx + 0.5 * pocket_w
                if ex < pocket_x_min or ex > pocket_x_max:
                    is_biofilm = False
        elif geom in ("crown", "real_crown"):
            hx = x_max - x_min
            hy = y_max - y_min
            r_outer = 0.5 * min(hx, hy)
            if r_outer > 0.0:
                cx = 0.5 * (x_min + x_max)
                cy = 0.5 * (y_min + y_max)
                shell_th = shell_frac * r_outer
                cap_th_z = cap_frac * (z_max - z_min)
                layer_z = z_max - cap_th_z
                rho = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                on_shell = rho >= (r_outer - shell_th)
                on_cap = ez >= layer_z
                if not (on_shell or on_cap):
                    is_biofilm = False

        if not is_biofilm:
            tooth_labels.append(elem.label)
            continue

        best_di = 0.0
        best_d2 = 1.0e30
        for (xv, yv, zv), dv in zip(coords, di_vals):
            d2 = (xv-ex)**2 + (yv-ey)**2 + (zv-ez)**2
            if d2 < best_d2:
                best_d2 = d2
                best_di = dv
        b_idx = int(best_di / bin_w)
        b_idx = max(0, min(n_bins - 1, b_idx))
        bin_labels[b_idx].append(elem.label)

    total_assigned = 0
    for b in range(n_bins):
        if not bin_labels[b]:
            continue
        seq = part.elements.sequenceFromLabels(labels=bin_labels[b])
        reg = Region(elements=seq)
        part.SectionAssignment(
            region=reg,
            sectionName="SEC_ANISO_%02d" % b,
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
        total_assigned += len(bin_labels[b])
    print("  Assigned %d elements across %d bins." % (total_assigned, n_bins))

    if tooth_labels:
        E_tooth = cfg["e_max"]
        consts_tooth = _engineering_constants(E_tooth, E_tooth, cfg["nu"])
        mat_t = model.Material(name="MAT_TOOTH")
        mat_t.Elastic(type=ENGINEERING_CONSTANTS, table=(consts_tooth,))
        model.HomogeneousSolidSection(name="SEC_TOOTH", material="MAT_TOOTH", thickness=None)
        seq_t = part.elements.sequenceFromLabels(labels=tooth_labels)
        reg_t = Region(elements=seq_t)
        part.SectionAssignment(
            region=reg_t,
            sectionName="SEC_TOOTH",
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField="",
            thicknessAssignment=FROM_SECTION,
        )

    all_elem_seq = part.elements.sequenceFromLabels(
        labels=[e.label for e in part.elements])
    part.MaterialOrientation(
        region=Region(elements=all_elem_seq),
        orientationType=SYSTEM,
        axis=AXIS_1,
        localCsys=local_csys,
        fieldName="",
        additionalRotationType=ROTATION_NONE,
        angle=0.0,
        additionalRotationField="",
        stackDirection=STACK_3,
    )

    asm  = model.rootAssembly
    inst = asm.Instance(name="BioInst", part=part, dependent=ON)

    tol = elem_size * 0.1

    bot_faces = inst.faces.getByBoundingBox(
        xMin=x_min-tol, xMax=x_max+tol,
        yMin=y_min-tol, yMax=y_max+tol,
        zMin=z_min-tol, zMax=z_min+tol)
    top_faces = inst.faces.getByBoundingBox(
        xMin=x_min-tol, xMax=x_max+tol,
        yMin=y_min-tol, yMax=y_max+tol,
        zMin=z_max-tol, zMax=z_max+tol)

    model.StaticStep(name="LOAD", previous="Initial",
                     maxNumInc=100, initialInc=0.1, minInc=1e-5, maxInc=1.0)

    if bot_faces:
        model.DisplacementBC(
            name="FIX_BOT", createStepName="Initial",
            region=Region(faces=bot_faces),
            u1=0.0, u2=0.0, u3=0.0)

    if top_faces:
        region_top = Region(side1Faces=top_faces)
        model.Pressure(
            name="PRESS", createStepName="LOAD",
            region=region_top, magnitude=1.0e6)

    job = mdb.Job(
        name=cfg["job_name"],
        model=model_name,
        numCpus=1,
        description="Aniso biofilm real geometry  beta=%.2f  e1=[%.2f,%.2f,%.2f]" % (
            aniso_ratio, e1[0], e1[1], e1[2]),
    )
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("[done] job=%s" % cfg["job_name"])


def main():
    try:
        sep = sys.argv.index("--")
        user_args = sys.argv[sep+1:]
    except ValueError:
        user_args = sys.argv[1:]

    cfg = _parse_args(user_args)
    print("=" * 55)
    print("Biofilm 3D Transversely Isotropic Model (Real Geometry)")
    print("  aniso_ratio : %.2f  (1.0 = isotropic)" % cfg["aniso_ratio"])
    print("  e1          : [%.3f, %.3f, %.3f]" % (cfg["e1_x"], cfg["e1_y"], cfg["e1_z"]))
    print("  geometry    : %s" % cfg["geometry"])
    print("  shell_frac  : %.3f  cap_frac=%.3f  pocket_frac=%.3f" % (
        cfg["shell_frac"], cfg["cap_frac"], cfg["pocket_frac"]))
    print("=" * 55)
    _build_and_run(cfg)


main()

