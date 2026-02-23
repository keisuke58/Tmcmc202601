from __future__ import print_function, division

import sys
import os
import math


def parse_args(argv):
    field_csv = None
    for i, a in enumerate(argv):
        if a == "--field-csv" and i + 1 < len(argv):
            field_csv = argv[i + 1]
    return field_csv


def read_field_csv(path):
    coords = []
    phi_pg = []
    di = []
    if path is None:
        return coords, phi_pg, di
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


def build_thread_model(field_csv):
    from abaqus import mdb
    from abaqusConstants import (
        THREE_D,
        DEFORMABLE_BODY,
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

    core_radius = 2.0
    thread_height = 0.5
    pitch = 1.0
    n_turns = 5
    length = n_turns * pitch

    model_name = "ImplantThread3D"
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    sketch = model.ConstrainedSketch(name="ThreadProfile", sheetSize=20.0)
    r_root = core_radius
    r_tip = core_radius + thread_height
    n_segments = 20
    x0 = 0.0
    y0 = r_root
    sketch.Line(point1=(x0, 0.0), point2=(x0, y0))
    for k in range(n_turns):
        x_start = k * pitch
        x_end = (k + 1) * pitch
        for i in range(n_segments):
            t0 = float(i) / float(n_segments)
            t1 = float(i + 1) / float(n_segments)
            x_a = x_start + t0 * (x_end - x_start)
            x_b = x_start + t1 * (x_end - x_start)
            phase_a = 2.0 * math.pi * t0
            phase_b = 2.0 * math.pi * t1
            y_a = r_root + 0.5 * thread_height * (1.0 + math.sin(phase_a))
            y_b = r_root + 0.5 * thread_height * (1.0 + math.sin(phase_b))
            sketch.Line(point1=(x_a, y_a), point2=(x_b, y_b))
    sketch.Line(point1=(length, r_root), point2=(length, 0.0))
    sketch.Line(point1=(length, 0.0), point2=(0.0, 0.0))

    part = model.Part(name="ImplantThread", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidRevolve(sketch=sketch, angle=360.0, flipRevolveDirection=OFF)

    mat_implant = model.Material(name="MAT_IMPLANT")
    mat_implant.Elastic(table=((110.0e9, 0.33),))
    sec_implant = model.HomogeneousSolidSection(
        name="SEC_IMPLANT",
        material="MAT_IMPLANT",
        thickness=1.0,
    )
    region_implant = Region(cells=part.cells)
    part.SectionAssignment(
        region=region_implant,
        sectionName="SEC_IMPLANT",
        offset=0.0,
        offsetType=FROM_SECTION,
        offsetField="",
        thicknessAssignment=FROM_SECTION,
    )

    elem_size = min(pitch, thread_height) / 3.0
    part.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    assembly = model.rootAssembly
    inst = assembly.Instance(name="ImplantInst", part=part, dependent=ON)

    model.StaticStep(name="ApplyLoad", previous="Initial")

    from regionToolset import Region as BCRegion

    bottom_faces = inst.faces.getByBoundingBox(
        xMin=-1e-3,
        xMax=1e-3,
        yMin=-1e-3,
        yMax=1e-3,
        zMin=-1e-3,
        zMax=1e-3,
    )
    if len(bottom_faces) > 0:
        region_bottom = BCRegion(faces=bottom_faces)
        model.DisplacementBC(
            name="BC_FIXED_BASE",
            createStepName="Initial",
            region=region_bottom,
            u1=0.0,
            u2=0.0,
            u3=0.0,
            ur1=UNSET,
            ur2=UNSET,
            ur3=UNSET,
            amplitude=UNSET,
            distributionType=UNIFORM,
            fieldName="",
            localCsys=None,
        )

    top_faces = inst.faces.getByBoundingBox(
        xMin=length - pitch - 1e-3,
        xMax=length + 1e-3,
        yMin=core_radius,
        yMax=core_radius + thread_height + 2.0,
        zMin=-1e-3,
        zMax=1e-3,
    )
    if len(top_faces) > 0:
        top_surface = assembly.Surface(name="TOP_SURF_THREAD", side1Faces=top_faces)
        model.Pressure(
            name="P_TOP_THREAD",
            createStepName="ApplyLoad",
            region=top_surface,
            magnitude=1.0e6,
        )

    job_name = "ImplantThreadJob"
    if job_name in mdb.jobs.keys():
        del mdb.jobs[job_name]
    job = mdb.Job(
        name=job_name,
        model=model_name,
        type=ANALYSIS,
        explicitPrecision=SINGLE,
        description="Simplified threaded implant demo",
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numDomains=1,
    )

    print("Job created:", job_name)
    print("Number of cells:", len(part.cells))
    print("Number of elements:", len(part.elements))
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("Job finished. Results (ODB etc.) written to working directory.")


def main():
    user_argv = sys.argv[1:]
    field_csv = parse_args(user_argv)
    if field_csv is not None and not os.path.isfile(field_csv):
        raise RuntimeError("field_csv not found: %s" % field_csv)
    build_thread_model(field_csv)


if __name__ == "__main__":
    main()

