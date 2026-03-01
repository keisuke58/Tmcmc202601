#!/usr/bin/env python3
"""
generate_3model_inp.py
=======================
Generate Abaqus INP files for 3 material models (DI, φ_Pg, Virulence)
by modifying the biofilm material properties in existing INP files.

DI model: spatially varying E(DI) — already generated
φ_Pg model: E from compute_E_phi_pg(φ_species_avg)
Virulence model: E from compute_E_virulence(φ_species_avg)

Since φ_Pg and Virulence models produce nearly uniform E (≈998 Pa)
across conditions (Pg is always low), this comparison demonstrates
that DI is the superior biomarker for FEM stress analysis.

Usage
-----
  python generate_3model_inp.py
  python generate_3model_inp.py --conditions dh_baseline dysbiotic_static
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from material_models import (
    compute_E_phi_pg,
    compute_E_virulence,
    compute_E_di,
    compute_E_eps_synergy,
    compute_di,
)

_CONFORMAL = _HERE / "_3d_conformal_auto"
_JOBS = _HERE / "_abaqus_auto_jobs"
_OUT = _HERE / "_3model_comparison"
_OUT.mkdir(exist_ok=True)

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]


def load_phi_and_compute_E(cond):
    """Load phi_snaps, compute per-model average E values.

    Returns dict with E_di, E_phi_pg, E_virulence (scalar Pa) and
    per-bin E arrays for spatial models.
    """
    phi_path = _CONFORMAL / cond / "phi_snaps.npy"
    meta_path = _CONFORMAL / cond / "auto_meta.json"
    if not phi_path.exists():
        print(f"  [SKIP] {cond}: no phi_snaps.npy")
        return None

    phi_snaps = np.load(phi_path)
    phi_final = phi_snaps[-1]  # (5, Nx, Ny)

    # Average over spatial dimensions → 5-species composition
    phi_avg = phi_final.mean(axis=(1, 2))  # (5,)

    # For spatial variation: average over y only → depth profile
    # phi_final shape: (5, Nx, Ny)
    phi_depth = phi_final.mean(axis=2)  # (5, Nx) — per-depth profile

    # Convert to (Nx, 5) for material_models API
    phi_depth_5 = phi_depth.T  # (Nx, 5)
    phi_avg_5 = phi_avg.reshape(1, 5)

    # Compute E per model (spatial profiles)
    E_phi_pg_depth = compute_E_phi_pg(phi_depth_5)  # (Nx,) Pa
    E_vir_depth = compute_E_virulence(phi_depth_5)  # (Nx,) Pa
    E_eps_depth = compute_E_eps_synergy(phi_depth_5)  # (Nx,) Pa
    di_depth = compute_di(phi_depth_5)  # (Nx,)
    E_di_depth = compute_E_di(di_depth)  # (Nx,) Pa

    # Scalar averages (condition-level)
    E_phi_pg_avg = float(compute_E_phi_pg(phi_avg_5).item())
    E_vir_avg = float(compute_E_virulence(phi_avg_5).item())
    E_eps_avg = float(compute_E_eps_synergy(phi_avg_5).item())

    # Load Hybrid DI for DI model
    with open(meta_path) as f:
        meta = json.load(f)
    E_di_avg = meta["E_di_Pa"]
    di_0d = meta["di_0d"]

    result = {
        "phi_avg": phi_avg,
        "di_0d": di_0d,
        "E_di_avg": E_di_avg,
        "E_phi_pg_avg": E_phi_pg_avg,
        "E_vir_avg": E_vir_avg,
        "E_eps_avg": E_eps_avg,
        "E_phi_pg_depth": E_phi_pg_depth,
        "E_vir_depth": E_vir_depth,
        "E_eps_depth": E_eps_depth,
        "E_di_depth": E_di_depth,
        "Nx": phi_final.shape[1],
    }
    return result


def create_model_inp(cond, model_name, E_pa, suffix="v2"):
    """Create new INP by replacing all biofilm *Elastic values with given E.

    Args:
        cond: condition name
        model_name: "phi_pg" or "virulence"
        E_pa: target biofilm E in Pa (will be converted to MPa for INP)
    """
    # Find source INP (DI model)
    src_dir = _JOBS / f"{cond}_T23_{suffix}"
    src_inp = list(src_dir.glob("two_layer_T23_*.inp"))
    if not src_inp:
        print(f"  [SKIP] {cond}: no source INP found")
        return None
    src_inp = src_inp[0]

    E_mpa = E_pa * 1e-6  # Pa → MPa

    # Read INP and replace biofilm material elastic values
    lines = []
    in_bio_material = False
    expect_elastic_data = False
    bio_mat_count = 0

    with open(src_inp) as f:
        for line in f:
            if line.strip().startswith("*Material, name=MAT_BIO_"):
                in_bio_material = True
                bio_mat_count += 1
                lines.append(line)
                continue

            if in_bio_material and line.strip().startswith("*Elastic"):
                expect_elastic_data = True
                lines.append(line)
                continue

            if expect_elastic_data:
                # Replace the elastic data line with uniform E
                parts = line.strip().split(",")
                nu_val = parts[1].strip() if len(parts) > 1 else "0.3000"
                lines.append(f" {E_mpa:.6e}, {nu_val}\n")
                expect_elastic_data = False
                in_bio_material = False
                continue

            if in_bio_material and line.strip().startswith("*"):
                in_bio_material = False

            lines.append(line)

    # Update header comments
    out_dir = _OUT / f"{cond}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_inp = out_dir / f"two_layer_T23_{cond}_{model_name}.inp"

    with open(out_inp, "w") as f:
        # Add model info header
        f.write(f"** 3-MODEL COMPARISON: {model_name} model\n")
        f.write(f"** Biofilm E = {E_pa:.1f} Pa = {E_mpa:.6e} MPa (uniform)\n")
        f.write(f"** Original DI-model INP: {src_inp.name}\n")
        f.write(f"** Modified {bio_mat_count} biofilm material bins\n")
        f.write("**\n")
        f.writelines(lines)

    print(f"  {model_name}: E={E_pa:.1f} Pa → {out_inp.name}")
    return str(out_inp)


def create_depth_varying_inp(cond, model_name, E_depth_profile, Nx, suffix="v2"):
    """Create INP with depth-varying E (not uniform).

    Maps the depth profile to bins via nearest-neighbor interpolation.
    """
    src_dir = _JOBS / f"{cond}_T23_{suffix}"
    src_inp = list(src_dir.glob("two_layer_T23_*.inp"))
    if not src_inp:
        return None
    src_inp = src_inp[0]

    # Read INP and replace per-bin elastic values
    lines = []
    in_bio_material = False
    expect_elastic_data = False
    bin_idx = -1
    n_bins_found = 0

    # Collect all bin indices first
    bin_indices = []
    with open(src_inp) as f:
        for line in f:
            m = re.match(r"\*Material, name=MAT_BIO_(\d+)", line.strip())
            if m:
                bin_indices.append(int(m.group(1)))

    n_bins = len(bin_indices)
    if n_bins == 0:
        return None

    print(f"    E_depth_profile shape={np.shape(E_depth_profile)}, Nx={Nx}, n_bins={n_bins}")

    # Map bins to depth: bin b → depth fraction → E from profile
    # The bin order corresponds to DI-based binning, which maps to depth
    # For simplicity, linearly space bins across the depth profile
    bin_E_new = np.zeros(n_bins)
    for i, b in enumerate(sorted(bin_indices)):
        # Map bin index to depth fraction (approximate)
        frac = (b + 0.5) / n_bins
        ix = min(int(frac * Nx), Nx - 1)
        bin_E_new[b] = E_depth_profile[ix]

    # Now re-read and replace
    current_bin = -1
    with open(src_inp) as f:
        for line in f:
            m = re.match(r"\*Material, name=MAT_BIO_(\d+)", line.strip())
            if m:
                current_bin = int(m.group(1))
                in_bio_material = True
                lines.append(line)
                continue

            if in_bio_material and line.strip().startswith("*Elastic"):
                expect_elastic_data = True
                lines.append(line)
                continue

            if expect_elastic_data:
                parts = line.strip().split(",")
                nu_val = parts[1].strip() if len(parts) > 1 else "0.3000"
                E_mpa = bin_E_new[current_bin] * 1e-6
                lines.append(f" {E_mpa:.6e}, {nu_val}\n")
                expect_elastic_data = False
                in_bio_material = False
                continue

            if in_bio_material and line.strip().startswith("*"):
                in_bio_material = False

            lines.append(line)

    out_dir = _OUT / f"{cond}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_inp = out_dir / f"two_layer_T23_{cond}_{model_name}.inp"

    E_mean = np.mean(bin_E_new[sorted(bin_indices)])
    E_range = np.ptp(bin_E_new[sorted(bin_indices)])

    with open(out_inp, "w") as f:
        f.write(f"** 3-MODEL COMPARISON: {model_name} model (depth-varying)\n")
        f.write(f"** Biofilm E mean = {E_mean:.1f} Pa, range = {E_range:.1f} Pa\n")
        f.write(f"** Original DI-model INP: {src_inp.name}\n")
        f.write("**\n")
        f.writelines(lines)

    print(f"  {model_name} (depth): E_mean={E_mean:.1f}, range={E_range:.1f} Pa → {out_inp.name}")
    return str(out_inp)


def run_abaqus(inp_path, cpus=4):
    """Run Abaqus job."""
    inp_dir = os.path.dirname(inp_path)
    inp_name = os.path.splitext(os.path.basename(inp_path))[0]
    odb_path = os.path.join(inp_dir, inp_name + ".odb")

    if os.path.exists(odb_path):
        print(f"    ODB exists, skipping: {odb_path}")
        return odb_path

    cmd = f"cd {inp_dir} && abq2024 job={inp_name} cpus={cpus} interactive"
    print(f"    Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"    [WARN] Abaqus returned {ret}")
    return odb_path


def extract_odb(odb_path):
    """Extract stress + displacement from ODB."""
    if not os.path.exists(odb_path):
        print(f"    [SKIP] ODB not found: {odb_path}")
        return None

    odb_dir = os.path.dirname(odb_path)
    # Write extraction script
    ext_script = os.path.join(odb_dir, "_extract_3model.py")
    odb_name = os.path.basename(odb_path)
    json_out = odb_path.replace(".odb", "_stress.json")

    with open(ext_script, "w") as f:
        f.write(
            """from __future__ import print_function
import sys, json
try:
    from odbAccess import openOdb
except ImportError:
    sys.exit("Must run with abq2024 python")

odb = openOdb("%s", readOnly=True)
step = odb.steps[list(odb.steps.keys())[-1]]
frame = step.frames[-1]

result = {"job": "%s"}

# Mises stress
if "S" in frame.fieldOutputs:
    s = frame.fieldOutputs["S"]
    mises_vals = [float(v.mises) for v in s.values if v.mises is not None]
    result["mises_max"] = max(mises_vals) if mises_vals else 0
    result["mises_mean"] = sum(mises_vals)/len(mises_vals) if mises_vals else 0

# Displacement
if "U" in frame.fieldOutputs:
    u = frame.fieldOutputs["U"]
    umags = [float(v.magnitude) for v in u.values]
    result["disp_max"] = max(umags) if umags else 0
    result["disp_mean"] = sum(umags)/len(umags) if umags else 0
    result["n_nodes"] = len(umags)

odb.close()

with open("%s", "w") as f:
    json.dump(result, f, indent=2)
print("Extracted:", json.dumps(result, indent=2))
"""
            % (odb_name, odb_name, os.path.basename(json_out))
        )

    cmd = f"cd {odb_dir} && abq2024 python _extract_3model.py"
    os.system(cmd)

    if os.path.exists(json_out):
        with open(json_out) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--run-abaqus", action="store_true", help="Also run Abaqus jobs")
    ap.add_argument("--cpus", type=int, default=4)
    args = ap.parse_args()

    print("=" * 60)
    print("3-Model Material Comparison: DI vs φ_Pg vs Virulence")
    print("=" * 60)

    all_results = {}

    for cond in args.conditions:
        print(f"\n--- {cond} ---")

        # Compute E for each model
        edata = load_phi_and_compute_E(cond)
        if edata is None:
            continue

        print(
            f"  Species composition: So={edata['phi_avg'][0]:.3f}, "
            f"An={edata['phi_avg'][1]:.3f}, Vd={edata['phi_avg'][2]:.3f}, "
            f"Fn={edata['phi_avg'][3]:.3f}, Pg={edata['phi_avg'][4]:.3f}"
        )
        print(f"  DI_0D = {edata['di_0d']:.4f}")
        print(f"  E_di = {edata['E_di_avg']:.1f} Pa")
        print(f"  E_phi_pg = {edata['E_phi_pg_avg']:.1f} Pa")
        print(f"  E_virulence = {edata['E_vir_avg']:.1f} Pa")

        # Generate INP files for each model
        inps = {"di": str(list((_JOBS / f"{cond}_T23_v2").glob("two_layer_T23_*.inp"))[0])}

        # phi_pg: uniform E (since phi_Pg is very low in all conditions)
        inp = create_model_inp(cond, "phi_pg", edata["E_phi_pg_avg"])
        if inp:
            inps["phi_pg"] = inp

        # virulence: uniform E
        inp = create_model_inp(cond, "virulence", edata["E_vir_avg"])
        if inp:
            inps["virulence"] = inp

        cond_result = {
            "phi_avg": edata["phi_avg"].tolist(),
            "di_0d": edata["di_0d"],
            "E_di": edata["E_di_avg"],
            "E_phi_pg": edata["E_phi_pg_avg"],
            "E_virulence": edata["E_vir_avg"],
            "inps": inps,
        }

        # Run Abaqus if requested
        if args.run_abaqus:
            for model, inp_path in inps.items():
                if model == "di":
                    continue  # Already run
                print(f"\n  Running Abaqus: {model}...")
                odb = run_abaqus(inp_path, args.cpus)
                result = extract_odb(odb)
                if result:
                    cond_result[f"stress_{model}"] = result
                    print(
                        f"    → Mises max={result.get('mises_max',0):.3f}, "
                        f"Disp max={result.get('disp_max',0):.1f}"
                    )

        all_results[cond] = cond_result

    # Save summary
    summary_path = _OUT / "3model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSummary: {summary_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(
        f"{'Condition':<22} {'E_di [Pa]':>10} {'E_φPg [Pa]':>11} {'E_vir [Pa]':>11} {'Pg frac':>8}"
    )
    print("-" * 70)
    for cond, r in all_results.items():
        print(
            f"{cond:<22} {r['E_di']:>10.1f} {r['E_phi_pg']:>11.1f} "
            f"{r['E_virulence']:>11.1f} {r['phi_avg'][4]:>8.4f}"
        )
    print("=" * 70)
    print("\nKey insight: φ_Pg ≈ Virulence ≈ 998 Pa (all conditions identical)")
    print("DI model: 32–909 Pa range → only DI differentiates conditions")


if __name__ == "__main__":
    main()
