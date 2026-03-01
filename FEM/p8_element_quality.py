#!/usr/bin/env python3
"""
p8_element_quality.py  –  [P8] C3D4 element quality report
============================================================

Reads the Abaqus INP file, extracts node coordinates and C3D4 connectivity,
and computes per-element quality metrics:

  - Volume (mm³)
  - Aspect ratio  AR = L_max / L_min   (edge lengths)
  - Minimum Jacobian  J_min  (determinant at each of 4 corners)
  - Shape quality  Q = 12 * (3 * |V|)^(2/3) / sum(edge²)  ∈ (0,1], 1 = equilateral

Flags elements with AR > 10 or J_min ≤ 0 as problematic.

Usage
-----
  python3 p8_element_quality.py [--inp biofilm_3tooth.inp] [--out-dir figures]
"""

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── Tet quality functions ──────────────────────────────────────────────────────


def tet_volumes(nodes, tets):
    """Signed volumes of C3D4 tets. Negative = inverted."""
    v = nodes[tets]  # (E, 4, 3)
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    c = v[:, 3] - v[:, 0]
    return np.einsum("ei,ei->e", a, np.cross(b, c)) / 6.0


def tet_aspect_ratios(nodes, tets):
    """Max/min edge length ratio per tet."""
    v = nodes[tets]  # (E, 4, 3)
    edges = [v[:, j] - v[:, i] for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
    lengths = np.stack([np.linalg.norm(e, axis=1) for e in edges], axis=1)  # (E, 6)
    L_max = lengths.max(axis=1)
    L_min = lengths.min(axis=1)
    return L_max / np.where(L_min < 1e-15, 1e-15, L_min)


def tet_shape_quality(nodes, tets):
    """
    Normalised shape quality Q ∈ (0, 1].
    Q = 12 * (3 * |V|)^(2/3) / sum_of_squared_edges  (1 for equilateral tet)
    """
    v = nodes[tets]  # (E, 4, 3)
    edges = [v[:, j] - v[:, i] for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
    L2 = sum(np.sum(e**2, axis=1) for e in edges)  # (E,)
    vol = np.abs(tet_volumes(nodes, tets))
    num = 12.0 * (3.0 * vol) ** (2.0 / 3.0)
    denom = np.where(L2 < 1e-30, 1e-30, L2)
    return num / denom


# ── INP parser ────────────────────────────────────────────────────────────────


def parse_inp(inp_path):
    """Return nodes (dict label→coords) and tets (list of (label, n1,n2,n3,n4))."""
    nodes = {}
    tets = []
    mode = None
    with open(inp_path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw = line.split(",")[0].strip().upper()
                if kw == "*NODE":
                    mode = "node"
                elif kw == "*ELEMENT":
                    # Only C3D4
                    if "C3D4" in line.upper():
                        mode = "elem"
                    else:
                        mode = None
                else:
                    mode = None
                continue
            if mode == "node":
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        lbl = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[lbl] = (x, y, z)
                    except ValueError:
                        pass
            elif mode == "elem":
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        lbl = int(parts[0])
                        n1, n2, n3, n4 = (int(parts[i]) for i in range(1, 5))
                        tets.append((lbl, n1, n2, n3, n4))
                    except ValueError:
                        pass
    return nodes, tets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(_HERE, "biofilm_3tooth.inp"))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "figures"))
    ap.add_argument(
        "--ar-flag",
        type=float,
        default=10.0,
        help="Aspect ratio threshold for flagging bad elements [10]",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("  P8 Element Quality Report")
    print(f"  INP: {args.inp}")
    print("=" * 60)

    # ── Parse INP ─────────────────────────────────────────────────────────────
    print("\nParsing INP (this may take ~30 s for 437 k elements)...")
    node_dict, tet_list = parse_inp(args.inp)
    print(f"  Nodes: {len(node_dict):,}")
    print(f"  C3D4:  {len(tet_list):,}")

    # Build contiguous arrays
    labels_arr = np.array([t[0] for t in tet_list], dtype=np.int32)
    conn_raw = np.array([[t[1], t[2], t[3], t[4]] for t in tet_list], dtype=np.int32)

    # Map 1-based node labels to 0-based row indices
    node_labels = sorted(node_dict.keys())
    lbl2idx = {lbl: i for i, lbl in enumerate(node_labels)}
    coords = np.array([node_dict[lbl] for lbl in node_labels])  # (N, 3)
    conn = np.array([[lbl2idx[n] for n in row] for row in conn_raw])  # (E, 4)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing quality metrics...")
    vols = tet_volumes(coords, conn)
    AR = tet_aspect_ratios(coords, conn)
    Q = tet_shape_quality(coords, conn)

    n_neg = (vols < 0).sum()
    n_bad_ar = (AR > args.ar_flag).sum()

    print("\nVolume (mm³):")
    for q in [0, 1, 5, 50, 95, 99, 100]:
        print(f"  p{q:3d} = {np.percentile(vols, q):.4e}")
    print(f"  Negative volumes: {n_neg}")

    print("\nAspect ratio (L_max / L_min):")
    for q in [50, 75, 90, 95, 99, 100]:
        print(f"  p{q:3d} = {np.percentile(AR, q):.2f}")
    print(f"  Elements with AR > {args.ar_flag}: {n_bad_ar} ({100.*n_bad_ar/len(AR):.2f}%)")

    print("\nShape quality Q ∈ (0,1]  (1=equilateral):")
    for q in [0, 1, 5, 50, 95, 100]:
        print(f"  p{q:3d} = {np.percentile(Q, q):.4f}")
    print(f"  Q < 0.1 (poor quality): {(Q < 0.1).sum()} ({100.*(Q<0.1).mean():.2f}%)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n── Verdict ──")
    ok = True
    if n_neg > 0:
        print(f"  [FAIL] {n_neg} negative-volume elements — recheck tet orientation fix.")
        ok = False
    if n_bad_ar > 0:
        pct = 100.0 * n_bad_ar / len(AR)
        if pct > 1.0:
            print(f"  [WARN] {n_bad_ar} ({pct:.1f}%) elements have AR > {args.ar_flag}.")
            ok = False
        else:
            print(f"  [OK]   {n_bad_ar} ({pct:.2f}%) elements have AR > {args.ar_flag} (<1%).")
    if ok:
        print("  [PASS] All quality metrics within acceptable range.")

    # ── Figures ───────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.hist(AR, bins=100, range=(0, 50), color="steelblue", edgecolor="none")
    ax.axvline(args.ar_flag, color="red", linestyle="--", label=f"AR={args.ar_flag}")
    ax.set_xlabel("Aspect ratio (L_max / L_min)")
    ax.set_ylabel("Count")
    ax.set_title(f"C3D4 aspect ratio  (flagged: {n_bad_ar})")
    ax.legend()

    ax = axes[1]
    ax.hist(Q, bins=80, range=(0, 1), color="seagreen", edgecolor="none")
    ax.axvline(0.1, color="red", linestyle="--", label="Q=0.1 (poor)")
    ax.set_xlabel("Shape quality Q (1 = equilateral)")
    ax.set_ylabel("Count")
    ax.set_title("C3D4 shape quality Q")
    ax.legend()

    ax = axes[2]
    vol_pos = vols[vols > 0]
    ax.hist(np.log10(vol_pos), bins=80, color="goldenrod", edgecolor="none")
    ax.set_xlabel("log₁₀(volume / mm³)")
    ax.set_ylabel("Count")
    ax.set_title(f"Element volume  (neg: {n_neg})")

    fig.suptitle("P8 Element Quality – biofilm_3tooth.inp")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "P8_element_quality.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
