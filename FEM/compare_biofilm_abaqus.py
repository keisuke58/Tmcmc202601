from __future__ import print_function, division

import sys
import os
import math


def _compute_bounds(nodes):
    xs = []
    ys = []
    zs = []
    for n in nodes:
        x, y, z = n.coordinates
        xs.append(x)
        ys.append(y)
        zs.append(z)
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    z_min = min(zs)
    z_max = max(zs)
    return x_min, x_max, y_min, y_max, z_min, z_max


def _select_boundary_nodes(nodes, axis, target, tol):
    boundary = []
    for n in nodes:
        coord = n.coordinates[axis]
        if abs(coord - target) <= tol:
            boundary.append(n)
    return boundary


def _find_center_node_2d(nodes, span_axis, span_min, span_max):
    if not nodes:
        return None
    mid = 0.5 * (span_min + span_max)
    best = None
    best_d2 = 1.0e30
    for n in nodes:
        v = n.coordinates[span_axis]
        d2 = (v - mid) * (v - mid)
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best


def _find_center_node_3d(nodes, axis_a, min_a, max_a, axis_b, min_b, max_b):
    if not nodes:
        return None
    mid_a = 0.5 * (min_a + max_a)
    mid_b = 0.5 * (min_b + max_b)
    best = None
    best_d2 = 1.0e30
    for n in nodes:
        ca = n.coordinates[axis_a]
        cb = n.coordinates[axis_b]
        da = ca - mid_a
        db = cb - mid_b
        d2 = da * da + db * db
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best


def _build_node_to_elems(inst):
    mapping = {}
    for elem in inst.elements:
        label = elem.label
        for nlab in elem.connectivity:
            if nlab in mapping:
                mapping[nlab].append(label)
            else:
                mapping[nlab] = [label]
    return mapping


def _format_vec3(v):
    return "(%.6e, %.6e, %.6e)" % (v[0], v[1], v[2])


def _format_list(vals):
    return "[" + ", ".join("%.6e" % v for v in vals) + "]"


def _detect_geometry(path):
    """Return 'crown', 'slit', or 'generic' from the ODB filename."""
    name = os.path.basename(path).lower()
    if "crown" in name:
        return "crown"
    if "slit" in name:
        return "slit"
    return "generic"


def _compute_centroid_xy(nodes):
    """Mean (x, y) of all node coordinates."""
    sx = sy = 0.0
    n = 0
    for nd in nodes:
        x, y, z = nd.coordinates
        sx += x
        sy += y
        n += 1
    if n == 0:
        return 0.0, 0.0
    return sx / n, sy / n


def _probe_radial(
    nodes, cx, cy, z_min, z_max, fractions, inst_name, u_field, s_field, node_to_elems, odb_name
):
    """
    Sample stress/displacement along the radial direction (r = distance from
    tooth-centre in XY) at mid-height.  fraction=0 is the inner surface
    (smallest r), fraction=1 is the outer surface (largest r).

    Returns rows in the same format as the axis-aligned depth probe.
    """
    z_mid = 0.5 * (z_min + z_max)
    z_tol = 0.15 * max(z_max - z_min, 1.0)  # ±15 % of Z span for mid-height band

    # Restrict to mid-height band so we don't accidentally snap to top/bot caps
    mid_nodes = [nd for nd in nodes if abs(nd.coordinates[2] - z_mid) <= z_tol]
    if not mid_nodes:
        mid_nodes = list(nodes)

    # Compute radial distances
    r_list = []
    for nd in mid_nodes:
        x, y, z = nd.coordinates
        r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_list.append((r, nd))
    if not r_list:
        return []

    r_min = min(r for r, _ in r_list)
    r_max = max(r for r, _ in r_list)
    r_span = r_max - r_min
    if r_span <= 0.0:
        return []

    rows = []
    for frac in fractions:
        target_r = r_min + frac * r_span
        best = min(r_list, key=lambda rv: abs(rv[0] - target_r))[1]

        n_label = best.label
        bx, by, bz = best.coordinates

        u_vec = [float("nan")] * 3
        if u_field is not None:
            for v in u_field.values:
                if v.instance.name == inst_name and v.nodeLabel == n_label:
                    tmp = list(v.data)
                    u_vec = (tmp + [float("nan")] * 3)[:3]
                    break

        s_comps = [float("nan")] * 6
        mises = float("nan")
        for e_lbl in node_to_elems.get(n_label, []):
            if s_field is None:
                break
            for v in s_field.values:
                if v.instance.name == inst_name and v.elementLabel == e_lbl:
                    comps = list(v.data)
                    for k in range(min(6, len(comps))):
                        s_comps[k] = comps[k]
                    m = getattr(v, "mises", None)
                    if m is not None and not (isinstance(m, float) and math.isnan(m)):
                        mises = m
                    break
            break

        rows.append(
            [
                odb_name,
                "depth_profile",
                frac,
                bx,
                by,
                bz,
                u_vec[0],
                u_vec[1],
                u_vec[2],
                s_comps[0],
                s_comps[1],
                s_comps[2],
                s_comps[3],
                s_comps[4],
                s_comps[5],
                mises,
            ]
        )
    return rows


def _nearest_node_to_target(nodes, tx, ty, tz):
    best = None
    best_d2 = 1.0e30
    for n in nodes:
        x, y, z = n.coordinates
        dx = x - tx
        dy = y - ty
        dz = z - tz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best


def extract_odb(path):
    from odbAccess import openOdb

    print("=== ODB: %s ===" % path)
    if not os.path.isfile(path):
        print("  file not found, skipping")
        return []
    odb = openOdb(path)
    try:
        if "ApplyLoad" in odb.steps:
            step = odb.steps["ApplyLoad"]
        else:
            names = sorted(odb.steps.keys())
            if not names:
                print("  no steps found")
                return
            step = odb.steps[names[0]]
        if not step.frames:
            print("  no frames in step")
            return
        frame = step.frames[-1]
        inst_names = list(odb.rootAssembly.instances.keys())
        if not inst_names:
            print("  no instances")
            return
        inst_name = inst_names[0]
        inst = odb.rootAssembly.instances[inst_name]
        nodes = list(inst.nodes)
        if not nodes:
            print("  no nodes")
            return []
        x_min, x_max, y_min, y_max, z_min, z_max = _compute_bounds(nodes)
        z_span = z_max - z_min
        tol_base = max(abs(x_max - x_min), abs(y_max - y_min), abs(z_span), 1.0)
        tol = 1.0e-6 * tol_base
        planar = abs(z_span) <= tol
        geom = _detect_geometry(path)
        if planar:
            load_axis = 1
            load_min = y_min
            load_max = y_max
            span_axis = 0
            span_min = x_min
            span_max = x_max
        else:
            if geom in ("crown", "slit"):
                # Load applied on Z faces (top = loaded, bottom = fixed)
                load_axis = 2
                load_min = z_min
                load_max = z_max
                span_axis_a = 0
                span_axis_b = 1
                span_min_a = x_min
                span_max_a = x_max
                span_min_b = y_min
                span_max_b = y_max
            else:
                load_axis = 0
                load_min = x_min
                load_max = x_max
                span_axis_a = 1
                span_axis_b = 2
                span_min_a = y_min
                span_max_a = y_max
                span_min_b = z_min
                span_max_b = z_max
        if planar:
            depth_axis = load_axis
            depth_min = load_min
            depth_max = load_max
            depth_span = depth_max - depth_min
            span_axis_a = span_axis
            span_axis_b = None
            span_min_a = span_min
            span_max_a = span_max
            span_min_b = 0.0
            span_max_b = 0.0
        else:
            depth_axis = load_axis
            depth_min = load_min
            depth_max = load_max
            depth_span = depth_max - depth_min
        node_to_elems = _build_node_to_elems(inst)
        u_field = frame.fieldOutputs["U"] if "U" in frame.fieldOutputs else None
        s_field = frame.fieldOutputs["S"] if "S" in frame.fieldOutputs else None
        if u_field is None:
            print("  U field not found")
        rows = []
        locations = [
            ("loaded_surface_center", load_max),
            ("support_surface_center", load_min),
        ]
        for label_loc, target in locations:
            b_nodes = _select_boundary_nodes(nodes, load_axis, target, tol)
            if not b_nodes:
                print("  %s: no boundary nodes" % label_loc)
                continue
            if planar:
                node = _find_center_node_2d(b_nodes, span_axis, span_min, span_max)
            else:
                node = _find_center_node_3d(
                    b_nodes,
                    span_axis_a,
                    span_min_a,
                    span_max_a,
                    span_axis_b,
                    span_min_b,
                    span_max_b,
                )
            if node is None:
                print("  %s: center node not found" % label_loc)
                continue
            n_label = node.label
            cx, cy, cz = node.coordinates
            print("  %s: node label=%d coord=(%.6e, %.6e, %.6e)" % (label_loc, n_label, cx, cy, cz))
            if u_field is not None:
                u_vec = None
                for v in u_field.values:
                    if v.instance.name == inst_name and v.nodeLabel == n_label:
                        u_vec = v.data
                        break
                if u_vec is not None and len(u_vec) == 3:
                    print("    U =", _format_vec3(u_vec))
                else:
                    print("    U not available for node")
            elem_labels = node_to_elems.get(n_label, [])
            if not elem_labels or s_field is None:
                print("    S not available for node")
            else:
                e_target = elem_labels[0]
                s_val = None
                for v in s_field.values:
                    if v.instance.name == inst_name and v.elementLabel == e_target:
                        s_val = v
                        break
                if s_val is None:
                    print("    S not found for element %d" % e_target)
                else:
                    comps = list(s_val.data)
                    mises = getattr(s_val, "mises", None)
                    print("    element label=%d" % e_target)
                    print("    S =", _format_list(comps))
                    if mises is not None and not (isinstance(mises, float) and math.isnan(mises)):
                        print("    S_Mises = %.6e" % mises)
        if geom == "crown" and not planar:
            # Radial probe: inner surface (tooth) → outer surface (biofilm)
            cx_c, cy_c = _compute_centroid_xy(nodes)
            print("  Crown radial probe: centroid=(%.4f, %.4f)" % (cx_c, cy_c))
            radial_rows = _probe_radial(
                nodes,
                cx_c,
                cy_c,
                z_min,
                z_max,
                [0.0, 0.5, 1.0],
                inst_name,
                u_field,
                s_field,
                node_to_elems,
                os.path.basename(path),
            )
            rows.extend(radial_rows)
        elif depth_span > 0.0:
            mid_a = 0.5 * (span_min_a + span_max_a)
            mid_b = 0.5 * (span_min_b + span_max_b)
            fractions = [0.0, 0.5, 1.0]
            for frac in fractions:
                d_val = depth_min + frac * depth_span
                if planar:
                    if depth_axis == 0:
                        tx = d_val
                        ty = mid_a
                        tz = 0.0
                    else:
                        tx = mid_a
                        ty = d_val
                        tz = 0.0
                else:
                    if depth_axis == 0:
                        tx = d_val
                        ty = mid_a
                        tz = mid_b
                    elif depth_axis == 1:
                        tx = mid_a
                        ty = d_val
                        tz = mid_b
                    else:
                        tx = mid_a
                        ty = mid_b
                        tz = d_val
                node_p = _nearest_node_to_target(nodes, tx, ty, tz)
                if node_p is None:
                    continue
                n_label = node_p.label
                cx, cy, cz = node_p.coordinates
                u_vec = [float("nan"), float("nan"), float("nan")]
                if u_field is not None:
                    for v in u_field.values:
                        if v.instance.name == inst_name and v.nodeLabel == n_label:
                            tmp = list(v.data)
                            if len(tmp) < 3:
                                tmp = (tmp + [float("nan")] * 3)[:3]
                            else:
                                tmp = tmp[:3]
                            u_vec = tmp
                            break
                elem_labels = node_to_elems.get(n_label, [])
                s_comps = [float("nan")] * 6
                mises = float("nan")
                if elem_labels and s_field is not None:
                    e_target = elem_labels[0]
                    s_val = None
                    for v in s_field.values:
                        if v.instance.name == inst_name and v.elementLabel == e_target:
                            s_val = v
                            break
                    if s_val is not None:
                        comps = list(s_val.data)
                        for i in range(min(6, len(comps))):
                            s_comps[i] = comps[i]
                        m = getattr(s_val, "mises", None)
                        if m is not None and not (isinstance(m, float) and math.isnan(m)):
                            mises = m
                rows.append(
                    [
                        os.path.basename(path),
                        "depth_profile",
                        frac,
                        cx,
                        cy,
                        cz,
                        u_vec[0],
                        u_vec[1],
                        u_vec[2],
                        s_comps[0],
                        s_comps[1],
                        s_comps[2],
                        s_comps[3],
                        s_comps[4],
                        s_comps[5],
                        mises,
                    ]
                )
        return rows
    finally:
        odb.close()


def main(argv):
    if len(argv) < 2:
        print("Usage: abq2024 python compare_biofilm_abaqus.py [OUT.csv] ODB1 [ODB2 ...]")
        return 1
    out_csv = None
    start = 1
    if len(argv) >= 3 and argv[1].lower().endswith(".csv"):
        out_csv = argv[1]
        start = 2
    all_rows = []
    for path in argv[start:]:
        rows = extract_odb(path)
        if rows:
            all_rows.extend(rows)
    if out_csv is not None and all_rows:
        import csv

        with open(out_csv, "w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "odb",
                    "label",
                    "depth_frac",
                    "x",
                    "y",
                    "z",
                    "Ux",
                    "Uy",
                    "Uz",
                    "S11",
                    "S22",
                    "S33",
                    "S12",
                    "S13",
                    "S23",
                    "S_Mises",
                ]
            )
            for r in all_rows:
                w.writerow(r)
        print("Wrote CSV:", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
