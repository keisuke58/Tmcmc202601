#!/usr/bin/env python3
"""
p3_pressure_sweep.py  –  [P3] Pressure parameter study
=======================================================

Sweeps applied pressure over a range, regenerates INP Cload values for each,
runs Abaqus, extracts MISES statistics from each ODB, and plots
σ_max, σ_median, |U|_max vs pressure to confirm linear-elastic behaviour.

Usage
-----
  # Dry-run (just regenerate INPs, no Abaqus):
  python3 p3_pressure_sweep.py --dry-run

  # Full sweep:
  python3 p3_pressure_sweep.py

  # Custom pressures:
  python3 p3_pressure_sweep.py --pressures 0.01 0.1 0.5 1.0 5.0 10.0
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
_ABAQUS = "/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"
_ASSEMBLY = os.path.join(_HERE, "biofilm_3tooth_assembly.py")
_EXTRACT = os.path.join(_HERE, "odb_extract.py")
_SWEEP_DIR = os.path.join(_HERE, "_pressure_sweep")

# Default DI CSV and STL root (DH-baseline, same as baseline run)
_DI_CSV = os.path.join(_HERE, "abaqus_field_dh_3d.csv")
_STL_ROOT = os.path.join(_HERE, "external_tooth_models", "OpenJaw_Dataset", "Patient_1")

DEFAULT_PRESSURES = [0.01e6, 0.1e6, 0.5e6, 1.0e6, 5.0e6, 10.0e6]  # Pa


def run_cmd(cmd, cwd=None, dry_run=False):
    print("  $", " ".join(str(c) for c in cmd))
    if dry_run:
        print("  [dry-run] skipped")
        return 0
    ret = subprocess.run(cmd, cwd=cwd)
    return ret.returncode


def extract_mises_from_csv(elem_csv):
    """Read odb_elements CSV and return MISES array."""
    import csv

    mises = []
    with open(elem_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mises.append(float(row["mises"]))
            except (KeyError, ValueError):
                pass
    return np.array(mises)


def extract_umag_from_csv(node_csv):
    """Read odb_nodes CSV and return |U| array."""
    import csv

    umag = []
    with open(node_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                umag.append(float(row["Umag"]))
            except (KeyError, ValueError):
                pass
    return np.array(umag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pressures",
        nargs="+",
        type=float,
        default=[p / 1e6 for p in DEFAULT_PRESSURES],
        help="Applied pressures in MPa (converted to Pa internally)",
    )
    ap.add_argument("--di-csv", default=_DI_CSV)
    ap.add_argument("--stl-root", default=_STL_ROOT)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="Generate INPs but do not run Abaqus")
    ap.add_argument("--out-dir", default=_SWEEP_DIR)
    args = ap.parse_args()

    pressures_pa = [p * 1e6 for p in args.pressures]  # MPa → Pa
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  P3 Pressure Sweep")
    print(f"  Pressures (MPa): {args.pressures}")
    print(f"  Out dir: {args.out_dir}")
    print("=" * 60)

    results = []

    for p_pa in pressures_pa:
        p_mpa = p_pa / 1e6
        tag = f"p{p_mpa:.4g}MPa".replace(".", "p")
        inp = os.path.join(args.out_dir, f"biofilm_3tooth_{tag}.inp")
        job = f"BioFilm3T_{tag}"
        odb = os.path.join(args.out_dir, f"{job}.odb")
        elem_csv = os.path.join(args.out_dir, f"odb_elements_{tag}.csv")
        node_csv = os.path.join(args.out_dir, f"odb_nodes_{tag}.csv")

        print(f"\n{'─'*50}")
        print(f"  pressure = {p_mpa:.4g} MPa")

        # 1. Generate INP
        cmd_gen = [
            "python3",
            _ASSEMBLY,
            "--stl-root",
            args.stl_root,
            "--di-csv",
            args.di_csv,
            "--out",
            inp,
            "--pressure",
            str(p_pa),
        ]
        rc = run_cmd(cmd_gen, dry_run=args.dry_run)
        if rc != 0:
            print(f"  [error] INP generation failed (rc={rc})")
            continue

        # 2. Run Abaqus
        if not args.dry_run:
            cmd_abq = [
                _ABAQUS,
                f"job={job}",
                f"input={inp}",
                f"cpus={args.cpus}",
                "ask=off",
                "interactive",
            ]
            rc = run_cmd(cmd_abq, cwd=args.out_dir, dry_run=False)
            if rc != 0:
                print(f"  [error] Abaqus failed (rc={rc})")
                continue

            # 3. Extract ODB  (outputs fixed names to ODB dir → rename)
            cmd_ext = [_ABAQUS, "python", _EXTRACT, odb]
            run_cmd(cmd_ext, cwd=args.out_dir, dry_run=False)
            raw_elem = os.path.join(args.out_dir, "odb_elements.csv")
            raw_node = os.path.join(args.out_dir, "odb_nodes.csv")
            if os.path.exists(raw_elem):
                os.rename(raw_elem, elem_csv)
            if os.path.exists(raw_node):
                os.rename(raw_node, node_csv)

            # 4. Read results
            if os.path.exists(elem_csv) and os.path.exists(node_csv):
                mises = extract_mises_from_csv(elem_csv)
                umag = extract_umag_from_csv(node_csv)
                results.append(
                    {
                        "pressure_mpa": p_mpa,
                        "mises_median": float(np.median(mises)),
                        "mises_max": float(mises.max()),
                        "umag_max": float(umag.max()),
                        "umag_outer_med": float(np.median(umag[umag > 0])),
                    }
                )
                print(
                    f"  MISES median={results[-1]['mises_median']:.4f} MPa  "
                    f"max={results[-1]['mises_max']:.4f} MPa  "
                    f"|U|_max={results[-1]['umag_max']:.3e} mm"
                )
        else:
            results.append(
                {
                    "pressure_mpa": p_mpa,
                    "mises_median": None,
                    "mises_max": None,
                    "umag_max": None,
                    "umag_outer_med": None,
                }
            )

    # ── Save results JSON ─────────────────────────────────────────────────────
    json_path = os.path.join(args.out_dir, "pressure_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    data = [r for r in results if r["mises_max"] is not None]
    if data:
        p_arr = np.array([r["pressure_mpa"] for r in data])
        med_arr = np.array([r["mises_median"] for r in data])
        max_arr = np.array([r["mises_max"] for r in data])
        umag_arr = np.array([r["umag_max"] for r in data])

        # Linear reference (scale from 1 MPa run if present)
        ref_idx = np.argmin(np.abs(p_arr - 1.0))
        p_lin = np.linspace(p_arr.min(), p_arr.max(), 100)
        lin_scale = med_arr[ref_idx] / p_arr[ref_idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.loglog(p_arr, med_arr, "o-", color="steelblue", label="MISES median")
        ax.loglog(p_arr, max_arr, "s--", color="tomato", label="MISES max")
        ax.loglog(p_lin, lin_scale * p_lin, "k:", alpha=0.5, label="Linear ref")
        ax.set_xlabel("Applied pressure (MPa)")
        ax.set_ylabel("von Mises stress (MPa)")
        ax.set_title("MISES vs pressure")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.loglog(p_arr, umag_arr, "D-", color="seagreen", label="|U|_max")
        ax.loglog(p_lin, umag_arr[ref_idx] / p_arr[ref_idx] * p_lin, "k:", alpha=0.5)
        ax.set_xlabel("Applied pressure (MPa)")
        ax.set_ylabel("|U|_max (mm)")
        ax.set_title("Displacement vs pressure")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ratio_m = med_arr / (lin_scale * p_arr)
        ax.semilogx(p_arr, ratio_m, "o-", color="goldenrod", label="MISES / linear")
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(1.05, color="r", linestyle=":", alpha=0.5, label="±5%")
        ax.axhline(0.95, color="r", linestyle=":", alpha=0.5)
        ax.set_xlabel("Applied pressure (MPa)")
        ax.set_ylabel("MISES / linear-ref")
        ax.set_title("Linearity check (ratio = 1 → linear)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("P3 Pressure Parameter Study – BioFilm3T")
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "P3_pressure_sweep.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved: {fig_path}")
        plt.close(fig)
    else:
        print("\n[info] No Abaqus results to plot (dry-run or all runs failed).")


if __name__ == "__main__":
    main()
