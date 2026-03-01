#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
abaqus_biofilm_demo.py  –  2D Abaqus model with continuous DI→E(DI) mapping

Runs inside the Abaqus Python environment:

  abaqus cae noGUI=abaqus_biofilm_demo.py -- \\
      --field-csv abaqus_field_dh_2d.csv   \\
      [--mapping power]                     \\
      [--n-bins 20]                         \\
      [--e-max 10e9] [--e-min 0.5e9]       \\
      [--di-exponent 2.0]                   \\
      [--di-scale 0.0]   # 0 → auto        \\
      [--nu 0.30]                           \\
      [--job-name BiofilmDemoJob]

Mapping modes
-------------
power  (default)
    E(DI) = E_max * (1 - r)^n  +  E_min * r
    where r = clamp(DI / di_scale, 0, 1).
    di_scale is set to 1.1 * max(DI) when --di-scale 0 is given.

binary (legacy)
    E = E_min if DI >= di_threshold  else  E_max.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Material mapping
# ---------------------------------------------------------------------------


def di_to_E(di_val, E_max, E_min, di_scale, exponent=2.0):
    """Power-law continuous modulus mapping."""
    if di_scale <= 0.0:
        return E_max
    r = di_val / di_scale
    if r > 1.0:
        r = 1.0
    elif r < 0.0:
        r = 0.0
    return E_max * (1.0 - r) ** exponent + E_min * r


def bin_index(di_val, di_min, di_max, n_bins):
    """Map DI value to bin index in [0, n_bins-1]."""
    if di_max <= di_min:
        return 0
    b = int((di_val - di_min) / (di_max - di_min) * n_bins)
    return max(0, min(n_bins - 1, b))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv):
    field_csv = None
    mapping = "power"
    n_bins = 20
    e_max = 10.0e9
    e_min = 0.5e9
    di_exponent = 2.0
    di_scale = 0.0
    di_threshold = 0.5
    nu = 0.30
    job_name = "BiofilmDemoJob"

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--field-csv" and i + 1 < len(argv):
            field_csv = argv[i + 1]
            i += 2
            continue
        if a == "--mapping" and i + 1 < len(argv):
            mapping = argv[i + 1]
            i += 2
            continue
        if a == "--n-bins" and i + 1 < len(argv):
            n_bins = int(argv[i + 1])
            i += 2
            continue
        if a == "--e-max" and i + 1 < len(argv):
            e_max = float(argv[i + 1])
            i += 2
            continue
        if a == "--e-min" and i + 1 < len(argv):
            e_min = float(argv[i + 1])
            i += 2
            continue
        if a == "--di-exponent" and i + 1 < len(argv):
            di_exponent = float(argv[i + 1])
            i += 2
            continue
        if a == "--di-scale" and i + 1 < len(argv):
            di_scale = float(argv[i + 1])
            i += 2
            continue
        if a == "--di-threshold" and i + 1 < len(argv):
            di_threshold = float(argv[i + 1])
            i += 2
            continue
        if a == "--nu" and i + 1 < len(argv):
            nu = float(argv[i + 1])
            i += 2
            continue
        if a == "--job-name" and i + 1 < len(argv):
            job_name = argv[i + 1]
            i += 2
            continue
        i += 1

    if field_csv is None:
        raise RuntimeError(
            "Usage: abaqus cae noGUI=abaqus_biofilm_demo.py -- "
            "--field-csv path/to/abaqus_field_dh_2d.csv [--mapping power|binary] ..."
        )
    if mapping not in ("power", "binary"):
        raise RuntimeError("--mapping must be 'power' or 'binary'")
    return {
        "field_csv": field_csv,
        "mapping": mapping,
        "n_bins": n_bins,
        "e_max": e_max,
        "e_min": e_min,
        "di_exponent": di_exponent,
        "di_scale": di_scale,
        "di_threshold": di_threshold,
        "nu": nu,
        "job_name": job_name,
    }


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


def read_field_csv(path):
    """Return (coords, phi_pg, di_vals) for 2D field CSV."""
    coords = []
    phi_pg = []
    di_vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("x,"):
                continue
            parts = line.split(",")
            coords.append((float(parts[0]), float(parts[1])))
            phi_pg.append(float(parts[2]))
            di_vals.append(float(parts[3]))
    return coords, phi_pg, di_vals


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(cfg):
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

    field_csv = cfg["field_csv"]
    mapping = cfg["mapping"]
    n_bins = cfg["n_bins"]
    E_max = cfg["e_max"]
    E_min = cfg["e_min"]
    exponent = cfg["di_exponent"]
    di_scale = cfg["di_scale"]
    di_thresh = cfg["di_threshold"]
    nu = cfg["nu"]
    job_name = cfg["job_name"]

    coords, phi_pg, di_vals = read_field_csv(field_csv)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    di_data_min = min(di_vals)
    di_data_max = max(di_vals)

    if di_scale <= 0.0:
        di_scale = 1.1 * di_data_max if di_data_max > 0.0 else 1.0

    print("=== Biofilm 2D Abaqus model ===")
    print("  field_csv  :", field_csv)
    print("  mapping    :", mapping)
    print("  x in [%.4f, %.4f]" % (x_min, x_max))
    print("  y in [%.4f, %.4f]" % (y_min, y_max))
    print("  DI in [%.5f, %.5f]" % (di_data_min, di_data_max))
    if mapping == "power":
        print("  di_scale   : %.5f  (exponent=%.2f)" % (di_scale, exponent))
        E_at_min = di_to_E(di_data_min, E_max, E_min, di_scale, exponent)
        E_at_max = di_to_E(di_data_max, E_max, E_min, di_scale, exponent)
        print("  E at DI_min: %.4g Pa" % E_at_min)
        print("  E at DI_max: %.4g Pa" % E_at_max)
        print("  n_bins     :", n_bins)
    else:
        print("  di_threshold:", di_thresh)

    # ── model ────────────────────────────────────────────────────────────────
    model_name = "BiofilmDemo"
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)

    sketch = model.ConstrainedSketch(name="RectSketch", sheetSize=10.0)
    sketch.rectangle(point1=(x_min, y_min), point2=(x_max, y_max))
    part = model.Part(name="Block2D", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)

    nx_est = max(len(set(xs)), 2)
    ny_est = max(len(set(ys)), 2)
    size_x = (x_max - x_min) / (nx_est - 1)
    size_y = (y_max - y_min) / (ny_est - 1)
    elem_size = min(size_x, size_y)
    part.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    # ── materials ────────────────────────────────────────────────────────────
    if mapping == "power":
        bin_E = []
        for b in range(n_bins):
            di_center = di_data_min + (b + 0.5) * (di_data_max - di_data_min) / n_bins
            E_b = di_to_E(di_center, E_max, E_min, di_scale, exponent)
            bin_E.append(E_b)
            mat = model.Material(name="MAT_BIN_%d" % b)
            mat.Elastic(table=((E_b, nu),))
            model.HomogeneousSolidSection(
                name="SEC_BIN_%d" % b,
                material="MAT_BIN_%d" % b,
                thickness=1.0,
            )
    else:
        mat_stiff = model.Material(name="MAT_STIFF")
        mat_stiff.Elastic(table=((E_max, nu),))
        mat_soft = model.Material(name="MAT_SOFT")
        mat_soft.Elastic(table=((E_min, nu),))
        model.HomogeneousSolidSection(name="SEC_STIFF", material="MAT_STIFF", thickness=1.0)
        model.HomogeneousSolidSection(name="SEC_SOFT", material="MAT_SOFT", thickness=1.0)

    # ── assembly ─────────────────────────────────────────────────────────────
    assembly = model.rootAssembly
    inst = assembly.Instance(name="BlockInst", part=part, dependent=ON)

    # ── element → section assignment ─────────────────────────────────────────
    if mapping == "power":
        bin_labels = [[] for _ in range(n_bins)]
    else:
        high_labels = []
        low_labels = []

    for elem in part.elements:
        nodes = elem.getNodes()
        if not nodes:
            continue
        sx = sy = 0.0
        ncnt = 0
        for nd in nodes:
            x_nd, y_nd, _ = nd.coordinates
            sx += x_nd
            sy += y_nd
            ncnt += 1
        if ncnt == 0:
            continue
        cx, cy = sx / ncnt, sy / ncnt

        best_di = None
        best_d2 = 1.0e30
        for (xv, yv), dv in zip(coords, di_vals):
            d2 = (xv - cx) ** 2 + (yv - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_di = dv

        if mapping == "power":
            b = bin_index(
                best_di if best_di is not None else di_data_min, di_data_min, di_data_max, n_bins
            )
            bin_labels[b].append(elem.label)
        else:
            if best_di is None or best_di < di_thresh:
                low_labels.append(elem.label)
            else:
                high_labels.append(elem.label)

    if mapping == "power":
        for b in range(n_bins):
            if not bin_labels[b]:
                continue
            seq = part.elements.sequenceFromLabels(labels=bin_labels[b])
            reg = Region(elements=seq)
            part.SectionAssignment(
                region=reg,
                sectionName="SEC_BIN_%d" % b,
                offset=0.0,
                offsetType=MIDDLE_SURFACE,
                offsetField="",
                thicknessAssignment=FROM_SECTION,
            )
        total = 0
        for bl in bin_labels:
            total += len(bl)
        print("  Assigned %d elements across %d bins." % (total, n_bins))
    else:
        if high_labels:
            seq = part.elements.sequenceFromLabels(labels=high_labels)
            reg = Region(elements=seq)
            part.SectionAssignment(
                region=reg,
                sectionName="SEC_SOFT",
                offset=0.0,
                offsetType=MIDDLE_SURFACE,
                offsetField="",
                thicknessAssignment=FROM_SECTION,
            )
        if low_labels:
            seq = part.elements.sequenceFromLabels(labels=low_labels)
            reg = Region(elements=seq)
            part.SectionAssignment(
                region=reg,
                sectionName="SEC_STIFF",
                offset=0.0,
                offsetType=MIDDLE_SURFACE,
                offsetField="",
                thicknessAssignment=FROM_SECTION,
            )
        print("  High-DI (soft) elements : %d" % len(high_labels))
        print("  Low-DI  (stiff) elements: %d" % len(low_labels))

    # ── step, BC, load ───────────────────────────────────────────────────────
    model.StaticStep(name="ApplyLoad", previous="Initial")

    bottom_edges = inst.edges.getByBoundingBox(
        xMin=x_min - 1e-6,
        xMax=x_max + 1e-6,
        yMin=y_min - 1e-6,
        yMax=y_min + 1e-6,
    )
    region_bot = Region(edges=bottom_edges)
    model.DisplacementBC(
        name="BC_BOTTOM",
        createStepName="Initial",
        region=region_bot,
        u1=0.0,
        u2=0.0,
        ur3=UNSET,
        amplitude=UNSET,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )

    top_edges = inst.edges.getByBoundingBox(
        xMin=x_min - 1e-6,
        xMax=x_max + 1e-6,
        yMin=y_max - 1e-6,
        yMax=y_max + 1e-6,
    )
    top_surface = assembly.Surface(name="TOP_SURF", side1Edges=top_edges)
    model.Pressure(
        name="P_TOP",
        createStepName="ApplyLoad",
        region=top_surface,
        magnitude=1.0e6,
    )

    # ── job ──────────────────────────────────────────────────────────────────
    if job_name in mdb.jobs.keys():
        del mdb.jobs[job_name]
    job = mdb.Job(
        name=job_name,
        model=model_name,
        type=ANALYSIS,
        explicitPrecision=SINGLE,
        description="Biofilm 2D continuous DI mapping",
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numDomains=1,
    )
    job.writeInput()
    print("Submitting job:", job_name)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("Job finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    user_argv = sys.argv[1:]
    cfg = parse_args(user_argv)
    if not os.path.isfile(cfg["field_csv"]):
        raise RuntimeError("field_csv not found: %s" % cfg["field_csv"])
    build_model(cfg)


if __name__ == "__main__":
    main()
