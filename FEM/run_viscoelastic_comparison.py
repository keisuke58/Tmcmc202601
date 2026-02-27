#!/usr/bin/env python3
"""
run_viscoelastic_comparison.py — Compare linear elastic, Mooney-Rivlin,
and Mooney-Rivlin+Prony viscoelastic Abaqus INP files for all 4 conditions.

Usage:
    python run_viscoelastic_comparison.py --stl tooth_T23.stl --di-csv-dir _di_csvs/

Generates INP files under _viscoelastic_comparison/{condition}_{model}/ for:
    - linear     : default linear elastic
    - nh         : Neo-Hookean hyperelastic
    - mr         : Mooney-Rivlin hyperelastic
    - mr_visco   : Mooney-Rivlin + Prony viscoelastic
    - umat_visco : UMAT F=Fe*Fv*Fg
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CONDITIONS = [
    "commensal_static",
    "commensal_hobic",
    "dh_baseline",
    "dysbiotic_static",
]

MODELS = {
    "linear": [],
    "nh": ["--neo-hookean"],
    "mr": ["--mooney-rivlin", "--c01-ratio", "0.15"],
    "mr_visco": [
        "--mooney-rivlin",
        "--c01-ratio",
        "0.15",
        "--viscoelastic",
        "--prony-g1",
        "0.5",
        "--prony-k1",
        "0.0",
        "--prony-tau1",
        "10.0",
    ],
    "umat_nh": [
        "--umat-visco",
        "--viscosity",
        "100.0",
    ],
    "umat_mr": [
        "--umat-visco",
        "--mooney-rivlin",
        "--c01-ratio",
        "0.15",
        "--viscosity",
        "100.0",
    ],
}


def main():
    ap = argparse.ArgumentParser(description="Viscoelastic comparison INP generator")
    ap.add_argument("--stl", required=True, help="Tooth STL file")
    ap.add_argument("--di-csv-dir", required=True, help="Dir with {condition}_di.csv files")
    ap.add_argument("--outdir", default="_viscoelastic_comparison")
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Which models to generate (default: all)",
    )
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITIONS,
        help="Which conditions to run (default: all 4)",
    )
    ap.add_argument("--growth-eigenstrain", type=float, default=0.56)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gen_script = str(_HERE / "biofilm_conformal_tet.py")

    n_total = len(args.conditions) * len(args.models)
    n_done = 0

    for cond in args.conditions:
        di_csv = os.path.join(args.di_csv_dir, f"{cond}_di.csv")
        if not os.path.exists(di_csv):
            print(f"SKIP: {di_csv} not found")
            continue

        for model_name in args.models:
            model_flags = MODELS[model_name]
            out_subdir = os.path.join(args.outdir, f"{cond}_{model_name}")
            os.makedirs(out_subdir, exist_ok=True)
            out_inp = os.path.join(out_subdir, f"biofilm_{cond}_{model_name}.inp")

            cmd = [
                sys.executable,
                gen_script,
                "--stl",
                args.stl,
                "--di-csv",
                di_csv,
                "--out",
                out_inp,
                "--mode",
                "biofilm",
                "--growth-eigenstrain",
                str(args.growth_eigenstrain),
            ] + model_flags

            n_done += 1
            print(f"\n[{n_done}/{n_total}] {cond} / {model_name}")
            print(f"  CMD: {' '.join(cmd)}")

            try:
                subprocess.run(cmd, check=True)
                print(f"  OK: {out_inp}")
            except subprocess.CalledProcessError as e:
                print(f"  FAIL: {e}")

    print(f"\nDone. {n_done} INP files generated under {args.outdir}/")

    # Summary table
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 72)
    print(f"  {'Condition':<25} {'Model':<12} {'INP File'}")
    print("-" * 72)
    for cond in args.conditions:
        for model_name in args.models:
            out_inp = os.path.join(
                args.outdir,
                f"{cond}_{model_name}",
                f"biofilm_{cond}_{model_name}.inp",
            )
            exists = "OK" if os.path.exists(out_inp) else "MISSING"
            print(f"  {cond:<25} {model_name:<12} {exists}")
    print("=" * 72)


if __name__ == "__main__":
    main()
