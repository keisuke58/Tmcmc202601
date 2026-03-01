#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_growth_stress.py
========================
Run under Abaqus Python (abaqus python extract_growth_stress.py):

Extracts from the 4 P30Growth_*.odb files:
  - Per-element centroid coords, von Mises (MISES), principal stresses (S11,S22,S33)
    for both GROWTH and LOAD step last frames
  - Per-node displacement magnitude (U) for LOAD step
  - Summary scalars (max_mises_growth, max_mises_load, max_U, mean_mises)

Outputs in same directory:
  stress_CONDITION.csv   -- per-element data (LOAD step)
  growth_summary.csv     -- one row per condition
"""

from __future__ import print_function
import sys
import os
import math

try:
    from odbAccess import openOdb
except ImportError:
    sys.exit("ERROR: run with  abaqus python extract_growth_stress.py")

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = [
    "commensal_static",
    "commensal_hobic",
    "dh_baseline",
    "dysbiotic_static",
]

ALPHA_EFF = {
    "commensal_static": 0.01717,
    "commensal_hobic": 0.01436,
    "dh_baseline": 0.007055,
    "dysbiotic_static": 0.01759,
}


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def extract_odb(odb_path, condition):
    print("\n[%s] Opening: %s" % (condition, odb_path))
    odb = openOdb(odb_path, readOnly=True)

    # Instance (flat INP → PART-1-1 or ASSEMBLY)
    assembly = odb.rootAssembly
    inst_keys = list(assembly.instances.keys())
    print("  Instances: %s" % inst_keys)
    inst = assembly.instances[inst_keys[0]]
    n_nodes = len(inst.nodes)
    n_elems = len(inst.elements)
    print("  Nodes: %d  Elements: %d" % (n_nodes, n_elems))

    # Node coords map: label -> (x,y,z)
    node_xyz = {}
    for n in inst.nodes:
        c = n.coordinates
        node_xyz[n.label] = (float(c[0]), float(c[1]), float(c[2]))

    # Element connectivity map: label -> (n1,n2,n3,n4)
    elem_conn = {}
    for e in inst.elements:
        elem_conn[e.label] = tuple(int(x) for x in e.connectivity)

    # Centroid per element
    elem_cx = {}
    elem_cy = {}
    elem_cz = {}
    for lbl, conn in elem_conn.items():
        xs = [node_xyz[n][0] for n in conn if n in node_xyz]
        ys = [node_xyz[n][1] for n in conn if n in node_xyz]
        zs = [node_xyz[n][2] for n in conn if n in node_xyz]
        if xs:
            elem_cx[lbl] = sum(xs) / len(xs)
            elem_cy[lbl] = sum(ys) / len(ys)
            elem_cz[lbl] = sum(zs) / len(zs)

    steps = list(odb.steps.keys())
    print("  Steps: %s" % steps)

    results = {}  # step_name -> {elem_lbl: {mises, s11, s22, s33}}
    node_U = {}  # node_lbl -> U_mag (LOAD step)

    for sname in steps:
        step = odb.steps[sname]
        frame = step.frames[-1]
        print("  [%s] last frame: frameId=%d  time=%.4f" % (sname, frame.frameId, frame.frameValue))

        fo = frame.fieldOutputs
        step_res = {}

        if "S" in fo:
            s_fo = fo["S"].getSubset(region=inst)
            for v in s_fo.values:
                lbl = v.elementLabel
                step_res[lbl] = {
                    "mises": float(v.mises),
                    "s11": float(v.data[0]) if len(v.data) >= 1 else 0.0,
                    "s22": float(v.data[1]) if len(v.data) >= 2 else 0.0,
                    "s33": float(v.data[2]) if len(v.data) >= 3 else 0.0,
                }
        results[sname] = step_res
        print("    S values extracted: %d elements" % len(step_res))

        if sname == steps[-1] and "U" in fo:
            u_fo = fo["U"].getSubset(region=inst)
            for v in u_fo.values:
                node_U[v.nodeLabel] = vec_norm(v.data)
            print("    U values extracted: %d nodes" % len(node_U))

    odb.close()

    # Write per-element CSV for LOAD step (last step)
    load_step = steps[-1]
    load_res = results.get(load_step, {})
    grow_step = steps[0] if len(steps) >= 2 else None
    grow_res = results.get(grow_step, {}) if grow_step else {}

    csv_path = os.path.join(HERE, "stress_%s.csv" % condition)
    with open(csv_path, "w") as f:
        f.write("elem,cx,cy,cz,mises_growth,mises_load,s11_load,s22_load,s33_load\n")
        for lbl in sorted(elem_conn.keys()):
            cx = elem_cx.get(lbl, 0.0)
            cy = elem_cy.get(lbl, 0.0)
            cz = elem_cz.get(lbl, 0.0)
            mg = grow_res.get(lbl, {}).get("mises", 0.0)
            ml = load_res.get(lbl, {}).get("mises", 0.0)
            s11 = load_res.get(lbl, {}).get("s11", 0.0)
            s22 = load_res.get(lbl, {}).get("s22", 0.0)
            s33 = load_res.get(lbl, {}).get("s33", 0.0)
            f.write(
                "%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n"
                % (lbl, cx, cy, cz, mg, ml, s11, s22, s33)
            )
    print("  -> %s" % csv_path)

    # Summary scalars
    mises_growth_vals = [v.get("mises", 0) for v in grow_res.values()] if grow_res else [0]
    mises_load_vals = [v.get("mises", 0) for v in load_res.values()] if load_res else [0]
    u_vals = list(node_U.values()) if node_U else [0]

    return {
        "condition": condition,
        "alpha_eff": ALPHA_EFF.get(condition, 0),
        "n_nodes": n_nodes,
        "n_elems": n_elems,
        "max_mises_growth": max(mises_growth_vals),
        "mean_mises_growth": sum(mises_growth_vals) / len(mises_growth_vals),
        "max_mises_load": max(mises_load_vals),
        "mean_mises_load": sum(mises_load_vals) / len(mises_load_vals),
        "max_U_mm": max(u_vals),
    }


def main():
    summaries = []
    for cond in CONDITIONS:
        odb_path = os.path.join(HERE, "P30Growth_%s.odb" % cond)
        if not os.path.isfile(odb_path):
            print("WARNING: ODB not found: %s" % odb_path)
            continue
        s = extract_odb(odb_path, cond)
        summaries.append(s)

    # Write summary CSV
    summ_path = os.path.join(HERE, "growth_summary.csv")
    with open(summ_path, "w") as f:
        f.write(
            "condition,alpha_eff,eps_growth,n_nodes,n_elems,"
            "max_mises_growth_Pa,mean_mises_growth_Pa,"
            "max_mises_load_Pa,mean_mises_load_Pa,max_U_mm\n"
        )
        for s in summaries:
            f.write(
                "%s,%.5g,%.5g,%d,%d,%.4g,%.4g,%.4g,%.4g,%.6g\n"
                % (
                    s["condition"],
                    s["alpha_eff"],
                    s["alpha_eff"] / 3.0,
                    s["n_nodes"],
                    s["n_elems"],
                    s["max_mises_growth"],
                    s["mean_mises_growth"],
                    s["max_mises_load"],
                    s["mean_mises_load"],
                    s["max_U_mm"],
                )
            )
    print("\n=== SUMMARY ===")
    print(
        "%-22s  %7s  %8s  %8s  %8s  %8s"
        % ("Condition", "alpha_eff", "MaxMG[Pa]", "MaxML[Pa]", "MeanML[Pa]", "MaxU[mm]")
    )
    print("-" * 75)
    for s in summaries:
        print(
            "%-22s  %7.4f  %8.4f  %8.4f  %8.4f  %8.6f"
            % (
                s["condition"],
                s["alpha_eff"],
                s["max_mises_growth"],
                s["max_mises_load"],
                s["mean_mises_load"],
                s["max_U_mm"],
            )
        )
    print("\nSummary written: %s" % summ_path)


if __name__ == "__main__":
    main()
