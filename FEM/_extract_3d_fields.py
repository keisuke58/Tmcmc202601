#!/usr/bin/env python
"""Extract node coordinates + displacement from ODB for 3D visualization.
Run with: abq2024 python _extract_3d_fields.py <odb_path> <out_csv>
"""
from __future__ import print_function
import sys
import csv

try:
    from odbAccess import openOdb
except ImportError:
    sys.exit("Must run with: abq2024 python _extract_3d_fields.py <odb> <csv>")

if len(sys.argv) < 3:
    sys.exit("Usage: abq2024 python _extract_3d_fields.py <odb_path> <out_csv>")

odb_path = sys.argv[1]
out_csv = sys.argv[2]

print("Opening:", odb_path)
odb = openOdb(odb_path, readOnly=True)

step_name = list(odb.steps.keys())[-1]
step = odb.steps[step_name]
frame = step.frames[-1]

# Get displacement field
u_field = frame.fieldOutputs["U"] if "U" in frame.fieldOutputs else None
s_field = frame.fieldOutputs["S"] if "S" in frame.fieldOutputs else None

# Build displacement lookup: (instance_name, node_label) -> (ux, uy, uz, mag)
u_lookup = {}
if u_field is not None:
    for val in u_field.values:
        inst = val.instance.name if val.instance else "ASSEMBLY"
        u_lookup[(inst, val.nodeLabel)] = (
            float(val.data[0]),
            float(val.data[1]),
            float(val.data[2]),
            float(val.magnitude),
        )

# Build stress lookup at nodes (ELEMENT_NODAL position for nodal averaging)
mises_lookup = {}
if s_field is not None:
    try:
        s_nodal = s_field.getSubset(position=s_field.values[0].position)
        for val in s_field.values:
            inst = val.instance.name if val.instance else "ASSEMBLY"
            key = (inst, val.nodeLabel if hasattr(val, "nodeLabel") else 0)
            m = float(val.mises) if val.mises is not None else 0.0
            # Keep max mises per node (multiple integration points map to same node)
            if key not in mises_lookup or m > mises_lookup[key]:
                mises_lookup[key] = m
    except Exception as e:
        print("  Warning: stress extraction failed:", str(e))

# Write CSV: iterate all instances and their nodes
print("Writing:", out_csv)
with open(out_csv, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["instance", "node_id", "x", "y", "z", "ux", "uy", "uz", "u_mag"])
    for inst_name in sorted(odb.rootAssembly.instances.keys()):
        inst = odb.rootAssembly.instances[inst_name]
        for node in inst.nodes:
            coords = node.coordinates
            key = (inst_name, node.label)
            u = u_lookup.get(key, (0.0, 0.0, 0.0, 0.0))
            writer.writerow(
                [inst_name, node.label, coords[0], coords[1], coords[2], u[0], u[1], u[2], u[3]]
            )

n_total = sum(len(odb.rootAssembly.instances[i].nodes) for i in odb.rootAssembly.instances.keys())
print("  Exported %d nodes" % n_total)
odb.close()
print("Done.")
