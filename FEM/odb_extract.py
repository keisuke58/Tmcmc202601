#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
odb_extract.py
==============
Run under Abaqus Python:
    abaqus python odb_extract.py [BioFilm3T.odb]

Extracts from the last frame of step LOAD:
  - Nodal coordinates, displacements (U), displacement magnitude
  - Element-averaged von Mises stress (MISES)
  - Tooth membership (T23/T30/T31) and DI bin per element
  - Surface region (INNER/OUTER/BULK) per node

Outputs (same directory as ODB):
  odb_nodes.csv    – one row per node
  odb_elements.csv – one row per element (with centroid coords)
"""
from __future__ import print_function
import sys
import os

try:
    from odbAccess import openOdb
except ImportError:
    sys.exit("ERROR: must run with  abaqus python odb_extract.py")

# ── Paths ─────────────────────────────────────────────────────────────────────
ODB_PATH = sys.argv[1] if len(sys.argv) > 1 else "BioFilm3T.odb"
ODB_PATH = os.path.abspath(ODB_PATH)
OUT_DIR = os.path.dirname(ODB_PATH)

try:
    import numpy as np

    HAS_NP = True
    print("[info] numpy available – fast centroid computation enabled")
except ImportError:
    HAS_NP = False
    print("[warn] numpy not available – using pure-Python fallback")

# ── Open ODB ──────────────────────────────────────────────────────────────────
print("Opening:", ODB_PATH)
odb = openOdb(ODB_PATH, readOnly=True)

assembly = odb.rootAssembly
inst_name = list(assembly.instances.keys())[0]
inst = assembly.instances[inst_name]
print("Instance: %s  |  %d nodes  %d elements" % (inst_name, len(inst.nodes), len(inst.elements)))

# ── Node coordinates ──────────────────────────────────────────────────────────
print("[1/7] Reading node coordinates ...")
node_labels = []
node_xyz = []
for n in inst.nodes:
    node_labels.append(n.label)
    c = n.coordinates
    node_xyz.append((float(c[0]), float(c[1]), float(c[2])))
N_nodes = len(node_labels)
print("       %d nodes" % N_nodes)

# ── Element connectivity ───────────────────────────────────────────────────────
print("[2/7] Reading element connectivity ...")
elem_labels = []
elem_conn = []  # list of tuples (n1,n2,n3,n4) – 1-based node labels
for e in inst.elements:
    elem_labels.append(e.label)
    elem_conn.append(tuple(int(x) for x in e.connectivity))
N_elems = len(elem_labels)
print("       %d elements" % N_elems)

# ── Centroids (numpy fast path) ───────────────────────────────────────────────
print("[3/7] Computing element centroids ...")
if HAS_NP:
    node_xyz_arr = np.array(node_xyz, dtype=np.float64)
    # Build label -> row-index map (labels are 1-based sequential)
    max_nlbl = max(node_labels)
    lbl2idx = np.full(max_nlbl + 1, -1, dtype=np.int64)
    for i, lbl in enumerate(node_labels):
        lbl2idx[lbl] = i
    conn_arr = np.array(elem_conn, dtype=np.int64)  # (E,4)
    idx4 = lbl2idx[conn_arr]  # (E,4)
    centroids = node_xyz_arr[idx4].mean(axis=1)  # (E,3)
    cx_arr = centroids[:, 0]
    cy_arr = centroids[:, 1]
    cz_arr = centroids[:, 2]
else:
    nc = {lbl: xyz for lbl, xyz in zip(node_labels, node_xyz)}
    cx_arr = [sum(nc[n][0] for n in conn) / 4 for conn in elem_conn]
    cy_arr = [sum(nc[n][1] for n in conn) / 4 for conn in elem_conn]
    cz_arr = [sum(nc[n][2] for n in conn) / 4 for conn in elem_conn]

# ── Tooth/bin membership from assembly element sets ───────────────────────────
print("[4/7] Reading element sets (T23/T30/T31, DI bins) ...")
# Sets are stored at the instance level in this ODB
elem_tooth = {}  # label -> 'T23'/'T30'/'T31'
elem_bin = {}  # label -> int bin_id

for setname in list(inst.elementSets.keys()):
    sn = setname.upper()
    if sn.startswith("T23_BIN_"):
        tooth = "T23"
    elif sn.startswith("T30_BIN_"):
        tooth = "T30"
    elif sn.startswith("T31_BIN_"):
        tooth = "T31"
    else:
        continue
    try:
        bin_id = int(sn.split("_")[-1])
    except ValueError:
        bin_id = -1
    for e in inst.elementSets[setname].elements:
        elem_tooth[e.label] = tooth
        elem_bin[e.label] = bin_id

n_tagged = sum(1 for lbl in elem_labels if lbl in elem_tooth)
print("       %d / %d elements tagged with tooth + bin" % (n_tagged, N_elems))

# ── Node surface region (INNER / OUTER) from instance node sets ───────────────
print("[5/7] Reading node sets (INNER/OUTER regions) ...")
node_tooth = {}  # label -> 'T23'/'T30'/'T31'
node_region = {}  # label -> 'INNER'/'OUTER'

# Process INNER/OUTER first, then APPROX (so APPROX overwrites OUTER for slit nodes)
_set_priority = {"INNER": 0, "OUTER": 1, "APPROX": 2}
_sorted_sets = sorted(
    inst.nodeSets.keys(),
    key=lambda s: _set_priority.get(
        (
            "APPROX"
            if "APPROX" in s.upper()
            else "INNER" if "INNER" in s.upper() else "OUTER" if "OUTER" in s.upper() else "ZZZ"
        ),
        99,
    ),
)

for setname in _sorted_sets:
    sn = setname.upper()
    # Expect pattern  T{23|30|31}_{INNER|OUTER|APPROX}
    if not (sn.startswith(("T23_", "T30_", "T31_"))):
        continue
    if "INNER" in sn:
        region = "INNER"
    elif "OUTER" in sn:
        region = "OUTER"
    elif "APPROX" in sn:
        region = "APPROX"
    else:
        continue
    tooth = sn[:3]  # 'T23', 'T30', or 'T31'
    for n in inst.nodeSets[setname].nodes:
        node_tooth[n.label] = tooth
        node_region[n.label] = region

n_tagged_n = sum(1 for lbl in node_labels if lbl in node_tooth)
print("       %d / %d nodes tagged with tooth + region" % (n_tagged_n, N_nodes))

# ── Field outputs – last frame of first step ───────────────────────────────────
step_name = list(odb.steps.keys())[0]
step = odb.steps[step_name]
frame = step.frames[-1]
print(
    "[6/7] Extracting field outputs from step '%s', frame %d (time=%.4f = full load) ..."
    % (step_name, frame.frameId, frame.frameValue)
)

# Displacement (U)
U_field = frame.fieldOutputs["U"]
node_U = {}  # label -> (Ux, Uy, Uz)
for v in U_field.values:
    d = v.data
    node_U[v.nodeLabel] = (float(d[0]), float(d[1]), float(d[2]))
print("       %d nodal displacement values" % len(node_U))

# Stress (S) – von Mises per integration point, then average per element
S_field = frame.fieldOutputs["S"]
elem_mises_sum = {}
elem_mises_cnt = {}
for v in S_field.values:
    lbl = v.elementLabel
    m = float(v.mises)
    if lbl not in elem_mises_sum:
        elem_mises_sum[lbl] = 0.0
        elem_mises_cnt[lbl] = 0
    elem_mises_sum[lbl] += m
    elem_mises_cnt[lbl] += 1
elem_mises = {lbl: elem_mises_sum[lbl] / elem_mises_cnt[lbl] for lbl in elem_mises_sum}
print(
    "       %d element MISES values (avg over %d integration pts each)"
    % (len(elem_mises), max(elem_mises_cnt.values()) if elem_mises_cnt else 0)
)

# ── Write odb_nodes.csv ───────────────────────────────────────────────────────
print("[7/7] Writing CSVs ...")
nodes_csv = os.path.join(OUT_DIR, "odb_nodes.csv")
with open(nodes_csv, "w") as f:
    f.write("label,x,y,z,Ux,Uy,Uz,Umag,tooth,region\n")
    for lbl, (x, y, z) in zip(node_labels, node_xyz):
        Ux, Uy, Uz = node_U.get(lbl, (0.0, 0.0, 0.0))
        Umag = (Ux * Ux + Uy * Uy + Uz * Uz) ** 0.5
        tooth = node_tooth.get(lbl, "BULK")
        region = node_region.get(lbl, "BULK")
        f.write(
            "%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%s,%s\n"
            % (lbl, x, y, z, Ux, Uy, Uz, Umag, tooth, region)
        )
print("  Nodes:    %s  (%d rows)" % (nodes_csv, N_nodes))

# Write odb_elements.csv
elems_csv = os.path.join(OUT_DIR, "odb_elements.csv")
with open(elems_csv, "w") as f:
    f.write("label,cx,cy,cz,mises,bin,tooth\n")
    for i, lbl in enumerate(elem_labels):
        cx = float(cx_arr[i])
        cy = float(cy_arr[i])
        cz = float(cz_arr[i])
        m = elem_mises.get(lbl, 0.0)
        b = elem_bin.get(lbl, -1)
        t = elem_tooth.get(lbl, "UNK")
        f.write("%d,%.8g,%.8g,%.8g,%.8g,%d,%s\n" % (lbl, cx, cy, cz, m, b, t))
print("  Elements: %s  (%d rows)" % (elems_csv, N_elems))

odb.close()

print("\nExtraction complete.")
print("  odb_nodes.csv    →", nodes_csv)
print("  odb_elements.csv →", elems_csv)
print("  Run next:  python3 odb_visualize.py")
