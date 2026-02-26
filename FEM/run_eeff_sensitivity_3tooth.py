#!/usr/bin/env python3
"""
run_eeff_sensitivity_3tooth.py
==============================

Parameter sweep for the DI→Eeff material model on the 3-tooth BioFilm3T setup.

For a given DI field (CSV) and STL root, this script:
  1. Runs biofilm_3tooth_assembly.py with different (E_max, E_min, DI_EXP, DI_SCALE)
     combinations, each time submitting an Abaqus job.
  2. Calls odb_extract.py (Abaqus Python) to generate odb_nodes.csv and
     odb_elements.csv for each job.
  3. Computes summary statistics and appends them to eeff_sensitivity_summary.csv:
       - Per-tooth median von Mises stress
       - Per-tooth outer-surface median displacement |U|

This is intended to quantify how sensitive key mechanical outputs are to the
DI→Eeff mapping parameters.
"""

from __future__ import print_function, division

import os
import sys
import argparse
import subprocess

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="3-tooth DI→Eeff material-model sensitivity sweep")
    p.add_argument(
        "--stl-root",
        default="external_tooth_models/OpenJaw_Dataset/Patient_1",
        help="Root directory containing Teeth/ STL files",
    )
    p.add_argument(
        "--di-csv", default="abaqus_field_dh_3d.csv", help="DI field CSV used for binning"
    )
    p.add_argument(
        "--abaqus",
        default="/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus",
        help="Path to abaqus executable",
    )
    p.add_argument("--base-e-max-mpa", type=float, default=10.0)
    p.add_argument("--base-e-min-mpa", type=float, default=0.5)
    p.add_argument("--base-di-exp", type=float, default=2.0)
    p.add_argument("--base-di-scale", type=float, default=0.025778)
    p.add_argument(
        "--pressure", type=float, default=1.0e6, help="Applied pressure in Pa (default 1 MPa)"
    )
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--nu", type=float, default=0.30)
    p.add_argument("--slit-threshold", type=float, default=0.30)
    p.add_argument("--slit-max-dist", type=float, default=None)
    p.add_argument("--no-slit", action="store_true")
    p.add_argument("--summary-csv", default="eeff_sensitivity_summary.csv")
    p.add_argument(
        "--dry-run", action="store_true", help="Print planned jobs but do not run Abaqus"
    )
    return p.parse_args()


def build_experiment_grid(base_e_max_mpa, base_e_min_mpa, base_di_exp, base_di_scale):
    exps = []

    emax_values = [0.75 * base_e_max_mpa, base_e_max_mpa, 1.25 * base_e_max_mpa]
    for v in emax_values:
        label = "Emax_%.2f" % v
        exps.append(
            {
                "label": label,
                "param_type": "Emax",
                "E_max_MPa": v,
                "E_min_MPa": base_e_min_mpa,
                "DI_EXP": base_di_exp,
                "DI_SCALE": base_di_scale,
            }
        )

    emin_values = [0.5 * base_e_min_mpa, base_e_min_mpa, 2.0 * base_e_min_mpa]
    for v in emin_values:
        label = "Emin_%.3f" % v
        exps.append(
            {
                "label": label,
                "param_type": "Emin",
                "E_max_MPa": base_e_max_mpa,
                "E_min_MPa": v,
                "DI_EXP": base_di_exp,
                "DI_SCALE": base_di_scale,
            }
        )

    di_exp_values = [0.5 * base_di_exp, base_di_exp, 1.5 * base_di_exp]
    for v in di_exp_values:
        label = "DIexp_%.2f" % v
        exps.append(
            {
                "label": label,
                "param_type": "DI_EXP",
                "E_max_MPa": base_e_max_mpa,
                "E_min_MPa": base_e_min_mpa,
                "DI_EXP": v,
                "DI_SCALE": base_di_scale,
            }
        )

    di_scale_values = [0.75 * base_di_scale, base_di_scale, 1.25 * base_di_scale]
    for v in di_scale_values:
        label = "DIscale_%.5f" % v
        exps.append(
            {
                "label": label,
                "param_type": "DI_SCALE",
                "E_max_MPa": base_e_max_mpa,
                "E_min_MPa": base_e_min_mpa,
                "DI_EXP": base_di_exp,
                "DI_SCALE": v,
            }
        )

    return exps


def run_assembly_and_abaqus(exp, args):
    label = exp["label"]
    job_name = "BioFilm3T_%s" % label
    inp_name = "biofilm_3tooth_%s.inp" % label

    e_max_pa = exp["E_max_MPa"] * 1.0e6
    e_min_pa = exp["E_min_MPa"] * 1.0e6

    assembly_script = os.path.join(SCRIPT_DIR, "biofilm_3tooth_assembly.py")
    cmd = [
        sys.executable,
        assembly_script,
        "--stl-root",
        args.stl_root,
        "--di-csv",
        args.di_csv,
        "--out",
        inp_name,
        "--job-name",
        job_name,
        "--n-layers",
        str(args.n_layers),
        "--n-bins",
        str(args.n_bins),
        "--e-max",
        str(e_max_pa),
        "--e-min",
        str(e_min_pa),
        "--di-scale",
        str(exp["DI_SCALE"]),
        "--di-exp",
        str(exp["DI_EXP"]),
        "--nu",
        str(args.nu),
        "--pressure",
        str(args.pressure),
        "--slit-threshold",
        str(args.slit_threshold),
        "--abaqus",
        args.abaqus,
    ]
    if args.slit_max_dist is not None:
        cmd.extend(["--slit-max-dist", str(args.slit_max_dist)])
    if args.no_slit:
        cmd.append("--no-slit")
    cmd.append("--run")

    print("=" * 72)
    print("Experiment:", label)
    print(
        "  E_max=%.3f MPa  E_min=%.3f MPa  DI_EXP=%.3f  DI_SCALE=%.6f"
        % (exp["E_max_MPa"], exp["E_min_MPa"], exp["DI_EXP"], exp["DI_SCALE"])
    )
    print("  Job name:", job_name)
    print("  INP     :", inp_name)

    if args.dry_run:
        print("  [dry-run] Skipping Abaqus submission")
        return None

    ret = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if ret.returncode != 0:
        print("  [error] Assembly or Abaqus job failed for", label)
        return None

    odb_path = os.path.join(SCRIPT_DIR, "%s.odb" % job_name)
    if not os.path.exists(odb_path):
        print("  [error] ODB not found:", odb_path)
        return None

    print("  ODB:", odb_path)
    return odb_path


def run_odb_extract(odb_path, args, label):
    if args.dry_run:
        return None, None

    cmd = [
        args.abaqus,
        "python",
        os.path.join(SCRIPT_DIR, "odb_extract.py"),
        odb_path,
    ]
    print("  Running odb_extract.py ...")
    ret = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if ret.returncode != 0:
        print("  [error] odb_extract.py failed for", label)
        return None, None

    out_dir = os.path.dirname(odb_path)
    nodes_csv = os.path.join(out_dir, "odb_nodes.csv")
    elems_csv = os.path.join(out_dir, "odb_elements.csv")
    if not (os.path.exists(nodes_csv) and os.path.exists(elems_csv)):
        print("  [error] odb_nodes.csv / odb_elements.csv not found")
        return None, None

    labeled_nodes = os.path.join(out_dir, "odb_nodes_%s.csv" % label)
    labeled_elems = os.path.join(out_dir, "odb_elements_%s.csv" % label)
    os.replace(nodes_csv, labeled_nodes)
    os.replace(elems_csv, labeled_elems)

    return labeled_nodes, labeled_elems


def compute_metrics(nodes_csv, elems_csv):
    nodes = np.genfromtxt(nodes_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    elems = np.genfromtxt(elems_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")

    teeth = ["T23", "T30", "T31"]
    metrics = {}

    for tooth in teeth:
        mask_e = np.array([str(t) == tooth for t in elems["tooth"]])
        m_t = elems["mises"][mask_e]
        if m_t.size == 0:
            med_mises = np.nan
        else:
            med_mises = float(np.median(m_t))

        mask_n_tooth = np.array([str(t) == tooth for t in nodes["tooth"]])
        mask_n_outer = np.array([str(r) == "OUTER" for r in nodes["region"]])
        m_outer = nodes["Umag"][mask_n_tooth & mask_n_outer]
        if m_outer.size == 0:
            med_u_outer = np.nan
        else:
            med_u_outer = float(np.median(m_outer))

        metrics[tooth] = {
            "mises_med": med_mises,
            "u_outer_med": med_u_outer,
        }

    return metrics


def append_summary_row(summary_csv, exp, metrics):
    header = [
        "label",
        "param_type",
        "E_max_MPa",
        "E_min_MPa",
        "DI_EXP",
        "DI_SCALE",
        "T23_mises_med",
        "T30_mises_med",
        "T31_mises_med",
        "T23_u_outer_med",
        "T30_u_outer_med",
        "T31_u_outer_med",
    ]

    need_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a") as f:
        if need_header:
            f.write(",".join(header) + "\n")

        row = [
            exp["label"],
            exp["param_type"],
            "%.6f" % exp["E_max_MPa"],
            "%.6f" % exp["E_min_MPa"],
            "%.6f" % exp["DI_EXP"],
            "%.8f" % exp["DI_SCALE"],
            "%.6f" % metrics["T23"]["mises_med"],
            "%.6f" % metrics["T30"]["mises_med"],
            "%.6f" % metrics["T31"]["mises_med"],
            "%.6e" % metrics["T23"]["u_outer_med"],
            "%.6e" % metrics["T30"]["u_outer_med"],
            "%.6e" % metrics["T31"]["u_outer_med"],
        ]
        f.write(",".join(row) + "\n")


def main():
    args = parse_args()

    exps = build_experiment_grid(
        args.base_e_max_mpa,
        args.base_e_min_mpa,
        args.base_di_exp,
        args.base_di_scale,
    )

    print("Total experiments:", len(exps))
    print("Summary CSV:", os.path.join(SCRIPT_DIR, args.summary_csv))

    for exp in exps:
        odb_path = run_assembly_and_abaqus(exp, args)
        if odb_path is None:
            continue

        nodes_csv, elems_csv = run_odb_extract(odb_path, args, exp["label"])
        if nodes_csv is None or elems_csv is None:
            continue

        metrics = compute_metrics(nodes_csv, elems_csv)
        append_summary_row(
            os.path.join(SCRIPT_DIR, args.summary_csv),
            exp,
            metrics,
        )

        print("  Metrics recorded for", exp["label"])


if __name__ == "__main__":
    main()
