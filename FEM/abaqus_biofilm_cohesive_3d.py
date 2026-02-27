from __future__ import print_function, division

"""
abaqus_biofilm_cohesive_3d.py  –  3D Cohesive Zone Model (CZM)
==============================================================
Runs inside the Abaqus Python environment:

  abaqus cae noGUI=abaqus_biofilm_cohesive_3d.py -- \\
      --field-csv abaqus_field_dh_3d.csv          \\
      [--job-name BiofilmCZM3DJob]                 \\
      [--t-max 1.0e6]     # Pa  normal strength baseline     \\
      [--gc-max 10.0]     # J/m^2  fracture energy baseline  \\
      [--di-exponent 2.0]                          \\
      [--n-layers-cz 3]   # bottom depth layers for DI mean  \\
      [--u-max 0.01]      # max tensile displacement (m)      \\
      [--n-steps 20]

Physical model
--------------
Biofilm (3D solid, DI-binned elastic, same as demo_3d) sits on a
rigid substrate.  The biofilm–substrate interface is modelled with
**surface-based cohesive behaviour** (Abaqus "cohesive contact").

Cohesive zone parameters are derived from the mean DI value in the
``n-layers-cz`` deepest FEM nodes (closest to the substrate):

  t_max(DI) = t_max_0 * max(0, 1 - DI / di_scale)^n_exp
  G_c(DI)   = G_c_0  * max(0, 1 - DI / di_scale)^n_exp

Higher DI  →  weaker interface  →  lower peel force.

Outputs (all written by abaqus cae / abaqus python):
  <job_name>.odb          – full ODB
  <job_name>_czm_out.csv  – columns: job, di_mean, t_max_eff, gc_eff,
                             U_max, RF_peak, RF_at_Umax
"""

import sys

# ---------------------------------------------------------------------------
# Defaults (overrideable via CLI)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "field_csv": None,
    "job_name": "BiofilmCZM3DJob",
    "t_max_0": 1.0e6,  # Pa  – nominal interface normal strength
    "gc_max": 10.0,  # J/m^2 – baseline fracture energy
    "di_exponent": 2.0,
    "di_scale": 0.025778,  # 1.1 * max(DI) from ensemble
    "n_layers_cz": 3,  # how many bottom z-layers for DI_mean
    "u_max": 5.0e-3,  # m – max applied displacement
    "n_steps": 20,
    "e_sub": 20.0e9,  # Pa – substrate Young's modulus
    "nu_sub": 0.25,
    "e_max": 10.0e9,  # Pa – biofilm E_max (healthy)
    "e_min": 0.5e9,  # Pa – biofilm E_min (dysbiotic)
    "n_exp_bio": 2.0,  # biofilm power-law exponent
    "n_bins": 10,  # number of biofilm material bins
    "nu_bio": 0.30,
    "mesh_size": 0.12,  # element size (fraction of domain)
    "knn_factor": 1.0e4,  # penalty stiffness = t_max * knn_factor
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv):
    cfg = dict(_DEFAULTS)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--field-csv" and i + 1 < len(argv):
            cfg["field_csv"] = argv[i + 1]
            i += 2
            continue
        if a == "--job-name" and i + 1 < len(argv):
            cfg["job_name"] = argv[i + 1]
            i += 2
            continue
        if a == "--t-max" and i + 1 < len(argv):
            cfg["t_max_0"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--gc-max" and i + 1 < len(argv):
            cfg["gc_max"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--di-exponent" and i + 1 < len(argv):
            cfg["di_exponent"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--di-scale" and i + 1 < len(argv):
            cfg["di_scale"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--n-layers-cz" and i + 1 < len(argv):
            cfg["n_layers_cz"] = int(argv[i + 1])
            i += 2
            continue
        if a == "--u-max" and i + 1 < len(argv):
            cfg["u_max"] = float(argv[i + 1])
            i += 2
            continue
        if a == "--n-steps" and i + 1 < len(argv):
            cfg["n_steps"] = int(argv[i + 1])
            i += 2
            continue
        if a == "--n-bins" and i + 1 < len(argv):
            cfg["n_bins"] = int(argv[i + 1])
            i += 2
            continue
        if a == "--mesh-size" and i + 1 < len(argv):
            cfg["mesh_size"] = float(argv[i + 1])
            i += 2
            continue
        i += 1
    if cfg["field_csv"] is None:
        raise RuntimeError(
            "Usage: abaqus cae noGUI=abaqus_biofilm_cohesive_3d.py -- "
            "--field-csv path/to/field.csv [options]"
        )
    return cfg


# ---------------------------------------------------------------------------
# Field CSV reader
# ---------------------------------------------------------------------------


def _read_field_csv(path):
    """Return lists: coords[(x,y,z)], phi_pg, di."""
    coords = []
    phi_pg = []
    di = []
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
            di.append(float(parts[4]))
    return coords, phi_pg, di


# ---------------------------------------------------------------------------
# DI → material properties helpers
# ---------------------------------------------------------------------------


def _di_to_E(di_val, e_max, e_min, di_scale, exponent):
    if di_scale <= 0.0:
        return e_max
    r = di_val / di_scale
    r = max(0.0, min(1.0, r))
    return e_max * (1.0 - r) ** exponent + e_min * r


def _czm_params(di_mean, cfg):
    """Return (t_max_eff, Gc_eff, Knn) based on mean DI of bottom layer."""
    di_scale = cfg["di_scale"]
    n_exp = cfg["di_exponent"]
    r = di_mean / di_scale if di_scale > 0 else 0.0
    r = max(0.0, min(1.0, r))
    factor = (1.0 - r) ** n_exp
    t_max = cfg["t_max_0"] * factor
    gc = cfg["gc_max"] * factor
    knn = t_max * cfg["knn_factor"]  # penalty stiffness (Pa/m)
    return t_max, gc, knn


# ---------------------------------------------------------------------------
# Abaqus model builder
# ---------------------------------------------------------------------------


def _build_and_run(cfg):
    from abaqus import mdb, session
    from abaqusConstants import (
        THREE_D,
        DEFORMABLE_BODY,
        ON,
        OFF,
        FROM_SECTION,
        MIDDLE_SURFACE,
        SMALL,
    )

    # Abaqus symbolic constants for contact/CZM
    import abaqusConstants as _AC

    NONE_AC = getattr(_AC, "NONE", None)

    try:
        from abaqusConstants import (
            COHESIVE,
            TRACTION_SEPARATION,
            BK,
        )

        has_czm_consts = True
    except ImportError:
        has_czm_consts = False

    try:
        from abaqusConstants import MAX_STRESS, ENERGY
    except ImportError:
        from symbolicConstants import SymbolicConstant

        MAX_STRESS = SymbolicConstant("MAX_STRESS")
        ENERGY = SymbolicConstant("ENERGY")

    from regionToolset import Region

    # ── read field ────────────────────────────────────────────────────────
    coords, phi_pg, di_vals = _read_field_csv(cfg["field_csv"])
    print("Field: %d nodes  di=[%.4f, %.4f]" % (len(coords), min(di_vals), max(di_vals)))

    z_all = sorted(set(c[2] for c in coords))
    n_cz = min(cfg["n_layers_cz"], len(z_all))
    z_bot = set(z_all[:n_cz])
    di_bottom = [dv for c, dv in zip(coords, di_vals) if c[2] in z_bot]
    di_mean = sum(di_bottom) / len(di_bottom) if di_bottom else 0.0
    t_max_eff, gc_eff, knn = _czm_params(di_mean, cfg)

    x_all = sorted(set(c[0] for c in coords))
    y_all = sorted(set(c[1] for c in coords))
    x_min, x_max = x_all[0], x_all[-1]
    y_min, y_max = y_all[0], y_all[-1]
    z_min, z_max = z_all[0], z_all[-1]
    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    print("Domain: Lx=%.3f  Ly=%.3f  Lz=%.3f" % (Lx, Ly, Lz))
    print("DI_mean(bottom %d layers)=%.4f" % (n_cz, di_mean))
    print("  t_max_eff=%.4g Pa  Gc_eff=%.4g J/m^2  Knn=%.4g Pa/m" % (t_max_eff, gc_eff, knn))

    # substrate thickness = Lz / 2
    t_sub = Lz * 0.5

    model_name = "BiofilmCZM3D"
    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)
    elem_size = cfg["mesh_size"]

    # ── Part: Substrate (3D solid box) ───────────────────────────────────
    sk_sub = model.ConstrainedSketch(name="SkSub", sheetSize=max(Lx, Ly) * 3)
    sk_sub.rectangle(point1=(x_min, y_min), point2=(x_max, y_max))
    p_sub = model.Part(name="Substrate", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_sub.BaseSolidExtrude(sketch=sk_sub, depth=t_sub)
    p_sub.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    p_sub.generateMesh()

    # ── Part: Biofilm (3D solid box) ─────────────────────────────────────
    sk_bio = model.ConstrainedSketch(name="SkBio", sheetSize=max(Lx, Ly) * 3)
    sk_bio.rectangle(point1=(x_min, y_min), point2=(x_max, y_max))
    p_bio = model.Part(name="Biofilm", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_bio.BaseSolidExtrude(sketch=sk_bio, depth=Lz)
    p_bio.seedPart(size=elem_size, deviationFactor=0.1, minSizeFactor=0.1)
    p_bio.generateMesh()

    # ── Materials: substrate ──────────────────────────────────────────────
    mat_sub = model.Material(name="MAT_SUB")
    mat_sub.Elastic(table=((cfg["e_sub"], cfg["nu_sub"]),))
    sec_sub = model.HomogeneousSolidSection(name="SEC_SUB", material="MAT_SUB", thickness=None)
    p_sub.SectionAssignment(
        region=Region(cells=p_sub.cells),
        sectionName="SEC_SUB",
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField="",
        thicknessAssignment=FROM_SECTION,
    )

    # ── Materials: biofilm bins (power-law DI → E) ────────────────────────
    n_bins = cfg["n_bins"]
    di_max_f = max(di_vals) if di_vals else 1.0
    bin_w = di_max_f / n_bins if n_bins > 0 else 1.0

    for b in range(n_bins):
        di_b = (b + 0.5) * bin_w
        e_b = _di_to_E(di_b, cfg["e_max"], cfg["e_min"], cfg["di_scale"], cfg["n_exp_bio"])
        mname = "MAT_BIO_%02d" % b
        sname = "SEC_BIO_%02d" % b
        mat_b = model.Material(name=mname)
        mat_b.Elastic(table=((e_b, cfg["nu_bio"]),))
        model.HomogeneousSolidSection(name=sname, material=mname, thickness=None)

    # Assign bin sections to biofilm cells by nearest-DI
    all_cells = p_bio.cells
    for ci in range(len(all_cells)):
        cell = all_cells[ci]
        cx, cy, cz = cell.pointOn[0]
        best_di = min(
            di_vals,
            key=lambda dv, xv=cx, yv=cy, zv=cz: (coords[di_vals.index(dv)][0] - xv) ** 2
            + (coords[di_vals.index(dv)][1] - yv) ** 2
            + (coords[di_vals.index(dv)][2] - zv) ** 2,
        )
        b_idx = int(best_di / bin_w)
        b_idx = max(0, min(n_bins - 1, b_idx))
        p_bio.SectionAssignment(
            region=Region(cells=all_cells[ci : ci + 1]),
            sectionName="SEC_BIO_%02d" % b_idx,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )

    # ── Assembly ──────────────────────────────────────────────────────────
    asm = model.rootAssembly
    # substrate at z < 0 (top face at z=0)
    inst_sub = asm.Instance(name="SubInst", part=p_sub, dependent=ON)
    asm.translate(instanceList=("SubInst",), vector=(0.0, 0.0, -t_sub))
    # biofilm at z in [0, Lz]
    inst_bio = asm.Instance(name="BioInst", part=p_bio, dependent=ON)

    # ── Cohesive contact (surface-based) ─────────────────────────────────
    # master = substrate top face (z=0), slave = biofilm bottom face (z=0)
    tol = elem_size * 0.1

    def _find_face(inst, z_target, tol_v):
        faces_ok = []
        for f in inst.faces:
            pts = [
                inst.nodes[nd - 1].coordinates if hasattr(inst, "nodes") else (0, 0, z_target)
                for nd in f.getNodes()
            ]
            # Use face.pointOn instead – simpler
            return None
        return None

    # Use face selection by coordinate
    sub_top = inst_sub.faces.getByBoundingBox(
        xMin=x_min - tol, xMax=x_max + tol, yMin=y_min - tol, yMax=y_max + tol, zMin=-tol, zMax=tol
    )
    bio_bot = inst_bio.faces.getByBoundingBox(
        xMin=x_min - tol, xMax=x_max + tol, yMin=y_min - tol, yMax=y_max + tol, zMin=-tol, zMax=tol
    )

    if sub_top and bio_bot:
        surf_sub = asm.Surface(side1Faces=sub_top, name="SURF_SUB_TOP")
        surf_bio = asm.Surface(side1Faces=bio_bot, name="SURF_BIO_BOT")

        # Cohesive contact property
        model.ContactProperty("CZM_PROP")
        prop = model.interactionProperties["CZM_PROP"]

        # Cohesive behaviour (traction-separation stiffness)
        prop.CohesiveBehavior(
            defaultPenalties=OFF,
            table=((knn, knn * 0.4, knn * 0.4),),  # Knn, Kss, Ktt
        )
        # Damage: initiation (MAX_STRESS) + evolution (energy-based)
        prop.Damage(
            initTable=((t_max_eff, t_max_eff * 0.6, t_max_eff * 0.6),),
            criterion=MAX_STRESS,
            useEvolution=ON,
            evolutionType=ENERGY,
            evolTable=((gc_eff,),),
        )

        model.SurfaceToSurfaceContactStd(
            name="CZM_CONTACT",
            createStepName="Initial",
            main=surf_sub,
            secondary=surf_bio,
            sliding=SMALL,
            thickness=ON,
            interactionProperty="CZM_PROP",
            adjustMethod=NONE_AC,
        )
        print("  [CZM] surface-based cohesive contact defined.")
    else:
        print("  [warn] could not find interface faces; skipping CZM contact.")

    # ── Boundary conditions ───────────────────────────────────────────────
    # Fix substrate bottom
    sub_bot_faces = inst_sub.faces.getByBoundingBox(
        xMin=x_min - tol,
        xMax=x_max + tol,
        yMin=y_min - tol,
        yMax=y_max + tol,
        zMin=-t_sub - tol,
        zMax=-t_sub + tol,
    )
    if sub_bot_faces:
        model.DisplacementBC(
            name="FIX_SUB",
            createStepName="Initial",
            region=Region(faces=sub_bot_faces),
            u1=0.0,
            u2=0.0,
            u3=0.0,
        )

    # Apply displacement to biofilm top face
    bio_top_faces = inst_bio.faces.getByBoundingBox(
        xMin=x_min - tol,
        xMax=x_max + tol,
        yMin=y_min - tol,
        yMax=y_max + tol,
        zMin=Lz - tol,
        zMax=Lz + tol,
    )

    # ── Step: Static displacement ─────────────────────────────────────────
    step = model.StaticStep(
        name="PULL",
        previous="Initial",
        nlgeom=ON,
        maxNumInc=200,
        initialInc=1.0 / cfg["n_steps"],
        minInc=1.0e-6,
        maxInc=0.1,
    )
    # Field output: use default (S, U, RF are included automatically)

    if bio_top_faces:
        region_top = Region(faces=bio_top_faces)
        # Fix lateral (allow z-pull only)
        model.DisplacementBC(
            name="FIX_BIO_LAT",
            createStepName="Initial",
            region=region_top,
            u1=0.0,
            u2=0.0,
        )
        model.DisplacementBC(
            name="PULL_BIO",
            createStepName="PULL",
            region=region_top,
            u3=cfg["u_max"],
        )

    # ── Job ───────────────────────────────────────────────────────────────
    job = mdb.Job(
        name=cfg["job_name"],
        model=model_name,
        numCpus=1,
        numGPUs=0,
        description="Biofilm 3D CZM  di_mean=%.4f" % di_mean,
    )
    print("  [submit] job=%s …" % cfg["job_name"])
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("  [done]  job=%s" % cfg["job_name"])

    # ── Post-process: extract RF peak from history ────────────────────────
    rf_peak = None
    rf_at_end = None
    try:
        odb = session.openOdb(cfg["job_name"] + ".odb")
        step_odb = odb.steps["PULL"]
        # Sum RF3 from top node set
        top_hist = None
        for hkey in step_odb.historyRegions.keys():
            hr = step_odb.historyRegions[hkey]
            if "RF3" in hr.historyOutputs:
                rf_data = hr.historyOutputs["RF3"].data
                rf_vals = [abs(v) for _, v in rf_data]
                if rf_vals:
                    if rf_peak is None or max(rf_vals) > rf_peak:
                        rf_peak = max(rf_vals)
                        rf_at_end = rf_vals[-1]
        odb.close()
    except Exception as e:
        print("  [warn] ODB post-processing failed: %s" % str(e))

    # ── Write summary CSV ─────────────────────────────────────────────────
    out_csv = cfg["job_name"] + "_czm_out.csv"
    with open(out_csv, "w") as f:
        f.write("job,di_mean,t_max_eff,gc_eff,u_max,rf_peak,rf_at_umax\n")
        f.write(
            "%s,%.6f,%.6g,%.6g,%.6g,%.6g,%.6g\n"
            % (
                cfg["job_name"],
                di_mean,
                t_max_eff,
                gc_eff,
                cfg["u_max"],
                rf_peak if rf_peak is not None else -1.0,
                rf_at_end if rf_at_end is not None else -1.0,
            )
        )
    print("  [out] %s" % out_csv)
    print(
        "  di_mean=%.4f  t_max=%.4gPa  Gc=%.4gJ/m2  RF_peak=%.4g"
        % (di_mean, t_max_eff, gc_eff, rf_peak if rf_peak is not None else float("nan"))
    )


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
    print("Biofilm 3D Cohesive Zone Model")
    print("  field_csv   : %s" % cfg["field_csv"])
    print("  job_name    : %s" % cfg["job_name"])
    print("  t_max_0     : %.4g Pa" % cfg["t_max_0"])
    print("  gc_max      : %.4g J/m^2" % cfg["gc_max"])
    print("  di_exponent : %.2f" % cfg["di_exponent"])
    print("  u_max       : %.4g m" % cfg["u_max"])
    print("=" * 60)
    _build_and_run(cfg)


main()
