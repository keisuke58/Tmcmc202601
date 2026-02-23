from __future__ import print_function, division

import sys
import os


def parse_args(argv):
    field_csv = None
    di_threshold = 0.5
    for i, a in enumerate(argv):
        if a == "--field-csv" and i + 1 < len(argv):
            field_csv = argv[i + 1]
        if a == "--di-threshold" and i + 1 < len(argv):
            di_threshold = float(argv[i + 1])
    if field_csv is None:
        raise RuntimeError(
            "Usage: abaqus cae noGUI=abaqus_biofilm_cohesive.py -- --field-csv path/to/abaqus_field_2d.csv"
        )
    return field_csv, di_threshold


def read_field_csv(path):
    coords = []
    phi_pg = []
    di = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("x,"):
                continue
            parts = line.split(",")
            x = float(parts[0])
            y = float(parts[1])
            pg = float(parts[2])
            dv = float(parts[3])
            coords.append((x, y))
            phi_pg.append(pg)
            di.append(dv)
    return coords, phi_pg, di


def build_model(field_csv, di_threshold):
    from abaqus import mdb
    from abaqusConstants import (
        TWO_D_PLANAR,
        DEFORMABLE_BODY,
        MIDDLE_SURFACE,
        ON,
        OFF,
        UNSET,
        UNIFORM,
        FROM_SECTION,
        ANALYSIS,
        SINGLE,
        DEFAULT,
    )
    from regionToolset import Region

    coords, phi_pg, di_vals = read_field_csv(field_csv)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    print("Field CSV:", field_csv)
    print("  x in [%.3f, %.3f]" % (x_min, x_max))
    print("  y in [%.3f, %.3f]" % (y_min, y_max))
    print("  di in [%.3f, %.3f]" % (min(di_vals), max(di_vals)))

    model_name = "BiofilmCohesive2D"
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    sketch_sub = model.ConstrainedSketch(name="SubSketch", sheetSize=10.0)
    sketch_sub.rectangle(point1=(x_min, y_min - 0.5 * (y_max - y_min)), point2=(x_max, y_min))
    part_sub = model.Part(name="Substrate2D", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part_sub.BaseShell(sketch=sketch_sub)

    sketch_bio = model.ConstrainedSketch(name="BioSketch", sheetSize=10.0)
    sketch_bio.rectangle(point1=(x_min, y_min), point2=(x_max, y_max))
    part_bio = model.Part(name="Biofilm2D", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part_bio.BaseShell(sketch=sketch_bio)

    nx_est = int(round(len(set(xs)))) or 10
    ny_est = int(round(len(set(ys)))) or 10
    size_x = (x_max - x_min) / max(nx_est - 1, 1)
    size_y = (y_max - y_min) / max(ny_est - 1, 1)
    elem_size = min(size_x, size_y)

    part_sub.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    part_sub.generateMesh()
    part_bio.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    part_bio.generateMesh()

    mat_sub = model.Material(name="MAT_SUB")
    mat_sub.Elastic(table=((20.0e9, 0.30),))
    mat_bio_stiff = model.Material(name="MAT_BIO_STIFF")
    mat_bio_stiff.Elastic(table=((10.0e9, 0.30),))
    mat_bio_soft = model.Material(name="MAT_BIO_SOFT")
    mat_bio_soft.Elastic(table=((2.0e9, 0.30),))

    sec_sub = model.HomogeneousSolidSection(name="SEC_SUB", material="MAT_SUB", thickness=1.0)
    sec_bio_stiff = model.HomogeneousSolidSection(
        name="SEC_BIO_STIFF", material="MAT_BIO_STIFF", thickness=1.0
    )
    sec_bio_soft = model.HomogeneousSolidSection(
        name="SEC_BIO_SOFT", material="MAT_BIO_SOFT", thickness=1.0
    )

    region_sub = Region(faces=part_sub.faces)
    part_sub.SectionAssignment(
        region=region_sub,
        sectionName="SEC_SUB",
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField="",
        thicknessAssignment=FROM_SECTION,
    )

    assembly = model.rootAssembly
    inst_sub = assembly.Instance(name="SubInst", part=part_sub, dependent=ON)
    inst_bio = assembly.Instance(name="BioInst", part=part_bio, dependent=ON)

    high_labels = []
    low_labels = []
    for elem in part_bio.elements:
        nodes = elem.getNodes()
        if not nodes:
            continue
        sx = 0.0
        sy = 0.0
        ncnt = 0
        for nd in nodes:
            x_nd, y_nd, _ = nd.coordinates
            sx += x_nd
            sy += y_nd
            ncnt += 1
        if ncnt == 0:
            continue
        cx = sx / ncnt
        cy = sy / ncnt
        best = None
        best_d = 1.0e30
        for (xv, yv), dv in zip(coords, di_vals):
            dx = xv - cx
            dy = yv - cy
            d2 = dx * dx + dy * dy
            if d2 < best_d:
                best_d = d2
                best = dv
        dv = best
        if dv is None:
            low_labels.append(elem.label)
        elif dv >= di_threshold:
            high_labels.append(elem.label)
        else:
            low_labels.append(elem.label)

    if high_labels:
        elem_seq_high = part_bio.elements.sequenceFromLabels(labels=high_labels)
        region_high = Region(elements=elem_seq_high)
        part_bio.SectionAssignment(
            region=region_high,
            sectionName="SEC_BIO_SOFT",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
    if low_labels:
        elem_seq_low = part_bio.elements.sequenceFromLabels(labels=low_labels)
        region_low = Region(elements=elem_seq_low)
        part_bio.SectionAssignment(
            region=region_low,
            sectionName="SEC_BIO_STIFF",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )

    model.ContactProperty("COHESIVE_PROP")
    ip = model.interactionProperties["COHESIVE_PROP"]
    ip.CohesiveBehavior(
        defaultPenalties=ON,
        table=((1.0e12, 1.0e12, 1.0e12),),
    )
    ip.DamageInitiation(
        criterion="QUADS",
        table=((1.0e6, 1.0e6, 1.0e6),),
    )
    ip.DamageEvolution(type="ENERGY", table=((0.5, 0.5, 0.5),))

    top_edges_sub = inst_sub.edges.getByBoundingBox(
        xMin=x_min - 1e-6,
        xMax=x_max + 1e-6,
        yMin=y_min - 1e-6,
        yMax=y_min + 1e-6,
    )
    bottom_edges_bio = inst_bio.edges.getByBoundingBox(
        xMin=x_min - 1e-6,
        xMax=x_max + 1e-6,
        yMin=y_min - 1e-6,
        yMax=y_min + 1e-6,
    )
    surf_sub = assembly.Surface(name="SUB_SURF", side1Edges=top_edges_sub)
    surf_bio = assembly.Surface(name="BIO_SURF", side1Edges=bottom_edges_bio)

    model.SurfaceToSurfaceContactStd(
        name="COHESIVE_CONTACT",
        createStepName="Initial",
        master=surf_sub,
        slave=surf_bio,
        sliding="FINITE",
        interactionProperty="COHESIVE_PROP",
    )

    model.StaticStep(name="ApplyLoad", previous="Initial")

    from regionToolset import Region as BCRegion

    bottom_edges_sub = inst_sub.edges.getByBoundingBox(
        xMin=x_min - 1e-6,
        xMax=x_max + 1e-6,
        yMin=y_min - 0.5 * (y_max - y_min) - 1e-6,
        yMax=y_min - 0.5 * (y_max - y_min) + 1e-6,
    )
    region_bottom = BCRegion(edges=bottom_edges_sub)
    model.DisplacementBC(
        name="BC_FIXED_SUB",
        createStepName="Initial",
        region=region_bottom,
        u1=0.0,
        u2=0.0,
        ur3=UNSET,
        amplitude=UNSET,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )

    top_edges_bio = inst_bio.edges.getByBoundingBox(
        xMin=x_min - 1e-6,
        xMax=x_max + 1e-6,
        yMin=y_max - 1e-6,
        yMax=y_max + 1e-6,
    )
    top_surface = assembly.Surface(name="TOP_SURF", side1Edges=top_edges_bio)
    model.Pressure(
        name="P_TOP",
        createStepName="ApplyLoad",
        region=top_surface,
        magnitude=1.0e6,
    )

    job_name = "BiofilmCohesiveJob"
    if job_name in mdb.jobs.keys():
        del mdb.jobs[job_name]
    job = mdb.Job(
        name=job_name,
        model=model_name,
        type=ANALYSIS,
        explicitPrecision=SINGLE,
        description="Biofilm cohesive debonding demo",
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numDomains=1,
    )

    print("Job created:", job_name)
    print("High-DI elements:", len(high_labels))
    print("Low-DI elements :", len(low_labels))
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("Job finished. Results (ODB etc.) written to working directory.")


def main():
    user_argv = sys.argv[1:]
    field_csv, di_threshold = parse_args(user_argv)
    if not os.path.isfile(field_csv):
        raise RuntimeError("field_csv not found: %s" % field_csv)
    build_model(field_csv, di_threshold)


if __name__ == "__main__":
    main()

