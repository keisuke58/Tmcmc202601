#!/usr/bin/env python3
"""
p7_tie_diagnostic.py  –  [P7] Tie coverage investigation
==========================================================

Reads odb_nodes.csv (which contains region=APPROX tags) and reports:
  1. Gap distribution between T30_APPROX and T31_APPROX nodes
  2. How many nodes fall within various distance thresholds
  3. Recommended --slit-max-dist value for biofilm_3tooth_assembly.py

Usage
-----
  python3 p7_tie_diagnostic.py [--nodes odb_nodes.csv]
"""
import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=os.path.join(_HERE, "odb_nodes.csv"))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "figures"))
    args = ap.parse_args()

    print("=" * 60)
    print("  P7 Tie Coverage Diagnostic")
    print("=" * 60)

    # ── Load odb_nodes.csv ────────────────────────────────────────────────────
    import csv

    labels, xs, ys, zs, teeth, regions = [], [], [], [], [], []
    with open(args.nodes) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            teeth.append(row["tooth"])
            regions.append(row["region"])

    coords = np.array([xs, ys, zs]).T  # (N, 3)
    teeth = np.array(teeth)
    regions = np.array(regions)

    # ── Extract APPROX nodes for T30 and T31 ─────────────────────────────────
    mask30 = (teeth == "T30") & (regions == "APPROX")
    mask31 = (teeth == "T31") & (regions == "APPROX")
    pts30 = coords[mask30]
    pts31 = coords[mask31]

    print(f"\nT30 APPROX nodes: {mask30.sum()}")
    print(f"T31 APPROX nodes: {mask31.sum()}")

    if mask30.sum() == 0 or mask31.sum() == 0:
        print(
            "[warn] No APPROX nodes found — check that odb_nodes.csv "
            "was generated with the slit Tie constraint enabled."
        )
        return

    # ── Gap distribution: for each T31 APPROX node → nearest T30 APPROX ─────
    from scipy.spatial import cKDTree

    tree30 = cKDTree(pts30)
    dists_31to30, _ = tree30.query(pts31, k=1)

    tree31 = cKDTree(pts31)
    dists_30to31, _ = tree31.query(pts30, k=1)

    print("\nGap T31→T30 (nearest T30 APPROX):")
    for q in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  p{q:3d} = {np.percentile(dists_31to30, q):.3f} mm")

    print("\nGap T30→T31 (nearest T31 APPROX):")
    for q in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  p{q:3d} = {np.percentile(dists_30to31, q):.3f} mm")

    # ── Tie tolerance analysis ────────────────────────────────────────────────
    print("\nT31 nodes within distance threshold (Tie tolerance sweep):")
    print(f"  {'threshold':>10s}  {'n_tied':>8s}  {'% of APPROX':>12s}")
    thresholds = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    for thr in thresholds:
        n = (dists_31to30 <= thr).sum()
        pct = 100.0 * n / len(dists_31to30)
        print(f"  {thr:10.1f}  {n:8d}  {pct:12.1f}%")

    # ── Recommendation ────────────────────────────────────────────────────────
    p5 = np.percentile(dists_31to30, 5)
    p10 = np.percentile(dists_31to30, 10)
    print("\nRecommendation:")
    print(
        f"  The 5th percentile gap is {p5:.2f} mm — only {(dists_31to30 <= p5).sum()} "
        f"T31 nodes are within this distance of a T30 APPROX node."
    )
    print(f"  Suggested --slit-max-dist: {p5:.1f}–{p10:.1f} mm")
    print(
        f"  Re-run assembly: python3 biofilm_3tooth_assembly.py "
        f"--slit-max-dist {p5:.1f} [other args...]"
    )

    # ── Figures ───────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of gap distances
    ax = axes[0]
    ax.hist(dists_31to30, bins=50, color="steelblue", edgecolor="white", alpha=0.8, label="T31→T30")
    ax.hist(dists_30to31, bins=50, color="tomato", edgecolor="white", alpha=0.6, label="T30→T31")
    for thr, ls in [(0.5, "--"), (2.0, ":")]:
        ax.axvline(thr, color="k", linestyle=ls, label=f"d={thr} mm")
    ax.set_xlabel("Distance to nearest APPROX node (mm)")
    ax.set_ylabel("Count")
    ax.set_title("T30/T31 APPROX gap distribution")
    ax.legend()

    # CDF of n_tied vs threshold
    ax = axes[1]
    thr_range = np.linspace(0, 15, 300)
    n31 = np.array([(dists_31to30 <= t).sum() for t in thr_range])
    n30 = np.array([(dists_30to31 <= t).sum() for t in thr_range])
    ax.plot(thr_range, n31, color="steelblue", label="T31 nodes tied")
    ax.plot(thr_range, n30, color="tomato", label="T30 nodes tied")
    ax.axvline(0.5, color="k", linestyle="--", label="Abaqus Tie tol=0.5 mm")
    ax.set_xlabel("Distance threshold (mm)")
    ax.set_ylabel("Number of nodes tied")
    ax.set_title("Cumulative tied nodes vs distance threshold")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "P7_tie_diagnostic.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    # ── 3-D scatter of APPROX nodes coloured by gap ───────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    sc = ax3.scatter(
        pts31[:, 0], pts31[:, 1], pts31[:, 2], c=dists_31to30, cmap="plasma", s=4, vmin=0, vmax=10
    )
    ax3.scatter(
        pts30[:, 0], pts30[:, 1], pts30[:, 2], c="steelblue", s=2, alpha=0.3, label="T30 APPROX"
    )
    plt.colorbar(sc, ax=ax3, label="Gap to nearest T30 node (mm)")
    ax3.set_title("T31 APPROX nodes coloured by gap to T30 APPROX")
    ax3.legend()
    out3d = os.path.join(args.out_dir, "P7_tie_3d_gap.png")
    fig.savefig(out3d, dpi=120)
    print(f"Figure saved: {out3d}")
    plt.close(fig)


if __name__ == "__main__":
    main()
