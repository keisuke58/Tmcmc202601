#!/usr/bin/env python3
"""
stl_bbox.py  –  Extract bounding-box and 2-D cross-section polygon from STL files.

Outputs a JSON with per-file entries consumed by:
  - openjaw_p1_auto_import.py   (tooth positioning in assembly)
  - openjaw_p1_biofilm_solid.py (crown sketch sizing + DI mapping)

Usage:
  python stl_bbox.py \\
      external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_23.stl \\
      external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_30.stl \\
      external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_31.stl \\
      --out p1_tooth_bbox.json

Optional:
  --units-scale 0.001      multiply coords by factor (mm→m, default 1.0)
  --poly-n 12              number of polygon points for crown sketch (default 12)
  --z-fraction 0.5         Z level (0=bottom, 1=top) for cross-section polygon (default 0.5)
  --no-poly                skip 2-D polygon extraction (faster for large files)
"""

import argparse
import json
import math
import struct
from pathlib import Path

# ---------------------------------------------------------------------------
# STL readers
# ---------------------------------------------------------------------------


def _is_binary_stl(path: Path) -> bool:
    """Heuristic: compare file size to expected binary size."""
    with path.open("rb") as f:
        header = f.read(80)
        raw = f.read(4)
    if len(raw) < 4:
        return False
    n_tri = struct.unpack("<I", raw)[0]
    expected = 84 + 50 * n_tri
    actual = path.stat().st_size
    # Allow small slack (some tools add extra bytes)
    if abs(actual - expected) <= 4:
        return True
    # Check if header starts with "solid" (ASCII indicator)
    try:
        if header.decode("ascii", errors="replace").strip().startswith("solid"):
            return False
    except Exception:
        pass
    return True


def _read_binary_stl_vertices(path: Path) -> list:
    """Return list of (x, y, z) vertex tuples from binary STL."""
    verts = []
    with path.open("rb") as f:
        f.read(80)  # header
        raw = f.read(4)
        if len(raw) < 4:
            return verts
        n_tri = struct.unpack("<I", raw)[0]
        for _ in range(n_tri):
            data = f.read(50)
            if len(data) < 50:
                break
            # 3 floats normal + 3×3 floats vertices + 2-byte attribute
            vals = struct.unpack("<12fH", data)
            for vi in range(3):
                off = 3 + vi * 3
                verts.append((vals[off], vals[off + 1], vals[off + 2]))
    return verts


def _read_ascii_stl_vertices(path: Path) -> list:
    """Return list of (x, y, z) vertex tuples from ASCII STL."""
    verts = []
    with path.open("r", errors="replace") as f:
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
    return verts


def read_stl_vertices(path: Path) -> list:
    if _is_binary_stl(path):
        return _read_binary_stl_vertices(path)
    return _read_ascii_stl_vertices(path)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def compute_bbox(verts: list) -> dict:
    if not verts:
        return {}
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    return {
        "n_vertices": len(verts),
        "min": [xmin, ymin, zmin],
        "max": [xmax, ymax, zmax],
        "center": [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)],
        "size": [xmax - xmin, ymax - ymin, zmax - zmin],
    }


# ---------------------------------------------------------------------------
# 2-D cross-section polygon (convex hull at a given Z slice)
# ---------------------------------------------------------------------------


def _convex_hull_2d(pts):
    """Graham-scan convex hull on 2-D points. Returns hull vertices in CCW order."""
    if len(pts) < 3:
        return list(pts)
    pts = sorted(set(pts))  # sort by (x, y), remove duplicates

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _resample_polygon(hull, n: int) -> list:
    """Resample a polygon to exactly n points by equal arc-length interpolation."""
    if len(hull) <= 1:
        return hull
    # Close the polygon
    closed = hull + [hull[0]]
    # Compute cumulative arc lengths
    arcs = [0.0]
    for i in range(1, len(closed)):
        dx = closed[i][0] - closed[i - 1][0]
        dy = closed[i][1] - closed[i - 1][1]
        arcs.append(arcs[-1] + math.sqrt(dx * dx + dy * dy))
    total = arcs[-1]
    if total == 0.0:
        return hull[:n] if len(hull) >= n else hull

    result = []
    for k in range(n):
        target = total * k / n
        # Find segment
        for i in range(1, len(arcs)):
            if arcs[i] >= target:
                seg_len = arcs[i] - arcs[i - 1]
                t = (target - arcs[i - 1]) / seg_len if seg_len > 0 else 0.0
                x = closed[i - 1][0] + t * (closed[i][0] - closed[i - 1][0])
                y = closed[i - 1][1] + t * (closed[i][1] - closed[i - 1][1])
                result.append((x, y))
                break
    return result


def compute_cross_section_polygon(
    verts: list, z_fraction: float = 0.5, n: int = 12, z_slab: float = 0.05
) -> list:
    """
    Extract 2-D convex hull at a given fractional Z height.

    Parameters
    ----------
    verts      : all STL vertices
    z_fraction : 0.0 = bottom, 1.0 = top of bounding box
    n          : number of output polygon points
    z_slab     : slab thickness as fraction of total height (default 5%)

    Returns
    -------
    List of (x, y) points (n points, CCW order), or empty list if not enough points.
    """
    if not verts:
        return []
    zs = [v[2] for v in verts]
    zmin, zmax = min(zs), max(zs)
    dz = zmax - zmin
    if dz == 0.0:
        return []

    z_center = zmin + z_fraction * dz
    half = 0.5 * z_slab * dz
    slab_verts = [(v[0], v[1]) for v in verts if (z_center - half) <= v[2] <= (z_center + half)]

    if len(slab_verts) < 4:
        # Widen slab
        half = 0.2 * dz
        slab_verts = [(v[0], v[1]) for v in verts if (z_center - half) <= v[2] <= (z_center + half)]

    if len(slab_verts) < 3:
        # Fallback: use bbox rectangle
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        bx = [min(xs), max(xs)]
        by = [min(ys), max(ys)]
        return [(bx[0], by[0]), (bx[1], by[0]), (bx[1], by[1]), (bx[0], by[1])]

    hull = _convex_hull_2d(slab_verts)
    if len(hull) < 3:
        return hull

    return _resample_polygon(hull, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_stl(
    path: Path, units_scale: float, poly_n: int, z_fraction: float, include_poly: bool
) -> dict:
    print(f"  Reading {path.name} ...", end="", flush=True)
    verts = read_stl_vertices(path)
    print(f" {len(verts)} vertices", flush=True)

    info = compute_bbox(verts)
    if units_scale != 1.0:
        for key in ("min", "max", "center", "size"):
            info[key] = [v * units_scale for v in info[key]]
    info["units_scale"] = units_scale

    if include_poly and verts:
        poly = compute_cross_section_polygon(verts, z_fraction=z_fraction, n=poly_n)
        if units_scale != 1.0:
            poly = [(x * units_scale, y * units_scale) for x, y in poly]
        info["cross_section_polygon"] = [list(p) for p in poly]
        info["poly_z_fraction"] = z_fraction
    return info


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stl_files", nargs="+", help="STL files to process")
    ap.add_argument(
        "--out", default="stl_bbox.json", help="Output JSON path (default: stl_bbox.json)"
    )
    ap.add_argument(
        "--units-scale",
        type=float,
        default=1.0,
        help="Multiply all coordinates by this factor (e.g. 0.001 mm→m)",
    )
    ap.add_argument(
        "--poly-n",
        type=int,
        default=12,
        help="Number of polygon vertices for cross-section sketch (default 12)",
    )
    ap.add_argument(
        "--z-fraction",
        type=float,
        default=0.5,
        help="Fractional Z height for cross-section (0=bottom, 1=top, default 0.5)",
    )
    ap.add_argument(
        "--no-poly",
        action="store_true",
        help="Skip 2-D polygon extraction (faster for large files)",
    )
    args = ap.parse_args()

    result = {}
    for s in args.stl_files:
        p = Path(s)
        if not p.exists():
            print(f"[warn] not found: {p}")
            continue
        try:
            info = process_stl(p, args.units_scale, args.poly_n, args.z_fraction, not args.no_poly)
            # Use stem (no extension) as key → matches Abaqus part names
            result[p.stem] = info
        except Exception as exc:
            print(f"[error] {p.name}: {exc}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {len(result)} entries → {out}")


if __name__ == "__main__":
    main()
