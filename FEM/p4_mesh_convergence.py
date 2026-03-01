#!/usr/bin/env python3
"""
p4_mesh_convergence.py  –  [P4] Mesh convergence study
========================================================

Sweeps the number of through-thickness biofilm layers (N_layers ∈ {4, 8, 16})
for T23 (crown tooth only — simpler geometry, faster mesh).

For each N_layers:
  1. Generates INP via biofilm_3tooth_assembly.py (T23 only via --no-slit,
     but the assembly includes all teeth; we report T23 stats)
  2. Runs Abaqus
  3. Extracts MISES statistics at reference elements (outer crown surface)

Convergence criterion: median MISES changes by < 2% between successive refinements.

Usage
-----
  # Dry-run:
  python3 p4_mesh_convergence.py --dry-run

  # Full study:
  python3 p4_mesh_convergence.py

  # Custom layer counts:
  python3 p4_mesh_convergence.py --layers 4 8 16 32
"""

import argparse
import os
import subprocess
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSEMBLY = os.path.join(_HERE, "biofilm_3tooth_assembly.py")
_ABAQUS = "/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"
_EXTRACT = os.path.join(_HERE, "odb_extract.py")
_CONV_DIR = os.path.join(_HERE, "_mesh_convergence")
_DI_CSV = os.path.join(_HERE, "abaqus_field_dh_3d.csv")
_STL_ROOT = os.path.join(_HERE, "external_tooth_models", "OpenJaw_Dataset", "Patient_1")


def run_cmd(cmd, cwd=None, dry_run=False, log_file=None):
    print("  $", " ".join(str(c) for c in cmd))
    if dry_run:
        print("  [dry-run] skipped")
        return 0
    if log_file:
        with open(log_file, "w") as lf:
            ret = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    else:
        ret = subprocess.run(cmd, cwd=cwd)
    return ret.returncode


def read_mises_for_tooth(elem_csv, tooth="T23"):
    """Return MISES array for a given tooth from odb_elements CSV."""
    import csv

    mises = []
    with open(elem_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tooth", "").strip() == tooth:
                try:
                    mises.append(float(row["mises"]))
                except (KeyError, ValueError):
                    pass
    return np.array(mises)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--di-csv", default=_DI_CSV)
    ap.add_argument("--stl-root", default=_STL_ROOT)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=_CONV_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  P4 Mesh Convergence Study")
    print(f"  Layers: {args.layers}")
    print(f"  Out dir: {args.out_dir}")
    print("=" * 60)

    results = []

    for nl in args.layers:
        tag = f"Nl{nl:02d}"
        inp_file = os.path.join(args.out_dir, f"biofilm_3tooth_{tag}.inp")
        job_name = f"BioFilm3T_{tag}"
        odb_file = os.path.join(args.out_dir, f"{job_name}.odb")
        elem_csv = os.path.join(args.out_dir, f"odb_elements_{tag}.csv")
        node_csv = os.path.join(args.out_dir, f"odb_nodes_{tag}.csv")
        log_file = os.path.join(args.out_dir, f"{tag}.log")

        print(f"\n── N_layers = {nl} ──")

        # Layer thickness
        layer_thick = 0.5 / nl
        print(f"  Layer thickness: {layer_thick:.4f} mm")

        # 1. Generate INP
        cmd_gen = [
            "python3",
            _ASSEMBLY,
            "--stl-root",
            args.stl_root,
            "--di-csv",
            args.di_csv,
            "--out",
            inp_file,
            "--n-layers",
            str(nl),
        ]
        rc = run_cmd(cmd_gen, dry_run=args.dry_run, log_file=log_file if not args.dry_run else None)
        if rc != 0:
            print("  [error] INP generation failed")
            continue

        # Count elements in generated INP
        n_elements = None
        if not args.dry_run and os.path.exists(inp_file):
            with open(inp_file) as f:
                for line in f:
                    if "Total C3D4:" in line:
                        try:
                            n_elements = int(line.split(":")[1].strip())
                        except Exception:
                            pass

        # 2. Run Abaqus
        cmd_abq = [
            _ABAQUS,
            f"job={job_name}",
            f"input={inp_file}",
            f"cpus={args.cpus}",
            "ask=off",
            "interactive",
        ]
        rc = run_cmd(cmd_abq, cwd=args.out_dir, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            print("  [error] Abaqus failed")
            continue

        # 3. Extract ODB  (odb_extract.py outputs fixed names to ODB dir → rename)
        cmd_ext = [_ABAQUS, "python", _EXTRACT, odb_file]
        rc = run_cmd(cmd_ext, cwd=args.out_dir, dry_run=args.dry_run)
        if not args.dry_run:
            raw_elem = os.path.join(args.out_dir, "odb_elements.csv")
            raw_node = os.path.join(args.out_dir, "odb_nodes.csv")
            if os.path.exists(raw_elem):
                os.rename(raw_elem, elem_csv)
            if os.path.exists(raw_node):
                os.rename(raw_node, node_csv)

        if not args.dry_run and os.path.exists(elem_csv):
            mises_t23 = read_mises_for_tooth(elem_csv, "T23")
            med = float(np.median(mises_t23)) if len(mises_t23) > 0 else float("nan")
            mx = float(mises_t23.max()) if len(mises_t23) > 0 else float("nan")
            print(f"  T23 MISES median={med:.4f}  max={mx:.4f}  n_elem={len(mises_t23)}")
            results.append(
                {
                    "n_layers": nl,
                    "layer_thick": layer_thick,
                    "n_elem_total": n_elements,
                    "mises_median": med,
                    "mises_max": mx,
                    "n_elem_t23": len(mises_t23),
                }
            )
        else:
            results.append(
                {
                    "n_layers": nl,
                    "layer_thick": layer_thick,
                    "n_elem_total": None,
                    "mises_median": None,
                    "mises_max": None,
                    "n_elem_t23": None,
                }
            )

    # ── Convergence check ─────────────────────────────────────────────────────
    data = [r for r in results if r["mises_median"] is not None]
    if len(data) >= 2:
        print("\n── Convergence Summary ──")
        print(f"  {'N_layers':>10s}  {'layer_thick':>12s}  {'MISES_med':>10s}  {'Δ%':>8s}")
        prev = None
        for r in data:
            if prev is not None:
                dpct = 100.0 * abs(r["mises_median"] - prev) / max(abs(prev), 1e-15)
            else:
                dpct = float("nan")
            print(
                f"  {r['n_layers']:>10d}  {r['layer_thick']:>12.4f}  "
                f"{r['mises_median']:>10.4f}  {dpct:>8.2f}%"
            )
            prev = r["mises_median"]

        finest = data[-1]["mises_median"]
        coarsest = data[0]["mises_median"]
        pct_err = 100.0 * abs(finest - coarsest) / max(abs(finest), 1e-15)
        print(f"\n  Coarsest vs finest: Δ = {pct_err:.1f}%")
        if pct_err < 5.0:
            print("  [OK] Mesh is converged to <5% across the layer range.")
        else:
            print("  [WARN] Non-negligible variation — consider using N_layers=16.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(args.out_dir, "mesh_convergence_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if data:
        nl_arr = [r["n_layers"] for r in data]
        med_arr = [r["mises_median"] for r in data]
        max_arr = [r["mises_max"] for r in data]
        lt_arr = [r["layer_thick"] for r in data]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(nl_arr, med_arr, "o-", color="steelblue", label="T23 MISES median")
        ax.plot(nl_arr, max_arr, "s--", color="tomato", label="T23 MISES max")
        ax.set_xlabel("N_layers (through-thickness)")
        ax.set_ylabel("von Mises stress (MPa)")
        ax.set_title("Mesh convergence: MISES vs N_layers")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if len(data) >= 2:
            ref = med_arr[-1]
            dpct = [100.0 * abs(m - ref) / max(abs(ref), 1e-15) for m in med_arr]
            ax.plot(nl_arr, dpct, "D-", color="goldenrod")
            ax.axhline(5.0, color="r", linestyle="--", label="5% threshold")
            ax.axhline(2.0, color="g", linestyle=":", label="2% threshold")
        ax.set_xlabel("N_layers")
        ax.set_ylabel("Δ MISES vs finest mesh (%)")
        ax.set_title("Convergence error vs finest")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("P4 Mesh Convergence – T23 Crown")
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "P4_mesh_convergence.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved: {fig_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
