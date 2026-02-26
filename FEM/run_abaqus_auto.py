#!/usr/bin/env python3
"""
run_abaqus_auto.py
==================
A2: Automated Abaqus execution + von Mises / principal stress extraction.

For each condition's two-layer Tie INP:
  1. Submit Abaqus job (abaqus job=... cpus=N)
  2. Wait for completion (.sta → COMPLETED)
  3. Extract results via abaqus python odb_extract_auto.py
  4. Parse ODB CSV → stress summary JSON

Usage
-----
  # Run for single condition:
  python run_abaqus_auto.py --condition dh_baseline

  # All conditions:
  python run_abaqus_auto.py --all

  # Extract only (INP already solved):
  python run_abaqus_auto.py --extract-only --condition dh_baseline

  # Dry run (show commands, don't execute):
  python run_abaqus_auto.py --dry-run --all
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT_BASE = _HERE / "_3d_conformal_auto"
_ABAQUS_WORK = _HERE / "_abaqus_auto_jobs"


def find_inp_files(condition, tooth="T23"):
    """Find INP files for a condition."""
    cond_dir = _OUT_BASE / condition
    tie_inp = cond_dir / f"two_layer_{tooth}_{condition}.inp"
    standalone_inp = cond_dir / f"biofilm_{tooth}_{condition}.inp"
    return {
        "tie": tie_inp if tie_inp.exists() else None,
        "standalone": standalone_inp if standalone_inp.exists() else None,
        "cond_dir": cond_dir,
    }


def submit_abaqus_job(inp_path, work_dir, cpus=4, dry_run=False):
    """Submit Abaqus job and return job name."""
    job_name = inp_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy INP to work directory
    dst_inp = work_dir / inp_path.name
    if not dst_inp.exists() or dst_inp.stat().st_mtime < inp_path.stat().st_mtime:
        import shutil

        shutil.copy2(str(inp_path), str(dst_inp))

    cmd = f"abaqus job={job_name} cpus={cpus} interactive"

    if dry_run:
        print(f"  [DRY] cd {work_dir} && {cmd}")
        return job_name

    print(f"  Submitting: {cmd}")
    print(f"  Work dir: {work_dir}")

    try:
        proc = subprocess.run(
            cmd.split(),
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if proc.returncode != 0:
            print(f"  [WARN] Abaqus returned code {proc.returncode}")
            if proc.stderr:
                print(f"  stderr: {proc.stderr[:500]}")
        else:
            print(f"  Abaqus job completed: {job_name}")
    except subprocess.TimeoutExpired:
        print("  [WARN] Abaqus job timed out after 3600s")
    except FileNotFoundError:
        print("  [ERROR] abaqus command not found. Is Abaqus installed?")
        return None

    return job_name


def check_sta_file(work_dir, job_name):
    """Check .sta file for completion status."""
    sta_path = work_dir / f"{job_name}.sta"
    if not sta_path.exists():
        return "NOT_FOUND"
    text = sta_path.read_text()
    if "COMPLETED" in text.upper():
        return "COMPLETED"
    if "ABORTED" in text.upper() or "ERROR" in text.upper():
        return "FAILED"
    return "RUNNING"


def extract_odb_results(work_dir, job_name, dry_run=False):
    """Extract results from ODB using Abaqus Python.

    Creates a lightweight extraction script that runs under abaqus python.
    """
    odb_path = work_dir / f"{job_name}.odb"
    if not odb_path.exists():
        print(f"  [SKIP] ODB not found: {odb_path}")
        return None

    # Write extraction script
    extract_script = work_dir / "_extract_stress.py"
    extract_script.write_text(
        f"""#!/usr/bin/env python
# Auto-generated ODB extraction script
from __future__ import print_function
import sys, os, json

try:
    from odbAccess import openOdb
except ImportError:
    sys.exit("Must run with: abaqus python _extract_stress.py")

ODB_PATH = r"{odb_path}"
OUT_JSON  = r"{work_dir / (job_name + '_stress.json')}"
OUT_CSV   = r"{work_dir / (job_name + '_elements.csv')}"

print("Opening:", ODB_PATH)
odb = openOdb(ODB_PATH, readOnly=True)

assembly = odb.rootAssembly
inst_name = list(assembly.instances.keys())[0]
inst = assembly.instances[inst_name]

# Get last frame of last step
step_name = list(odb.steps.keys())[-1]
step = odb.steps[step_name]
frame = step.frames[-1]

# Extract Mises stress
mises_field = frame.fieldOutputs["S"] if "S" in frame.fieldOutputs else None
s_max_field = mises_field

mises_vals = []
s_principal_vals = []
elem_data = []

if mises_field is not None:
    for val in mises_field.values:
        if val.mises is not None:
            mises_vals.append(float(val.mises))
        if hasattr(val, "maxPrincipal") and val.maxPrincipal is not None:
            s_principal_vals.append(float(val.maxPrincipal))
        # Element data
        elem_data.append({{
            "label": val.elementLabel,
            "mises": float(val.mises) if val.mises is not None else 0.0,
            "max_principal": float(val.maxPrincipal) if hasattr(val, "maxPrincipal") and val.maxPrincipal is not None else 0.0,
            "min_principal": float(val.minPrincipal) if hasattr(val, "minPrincipal") and val.minPrincipal is not None else 0.0,
        }})

# Extract displacements
u_field = frame.fieldOutputs["U"] if "U" in frame.fieldOutputs else None
u_mag_vals = []
if u_field is not None:
    for val in u_field.values:
        u_mag_vals.append(float(val.magnitude))

# Nodal coordinates + displacement
node_data = []
if u_field is not None:
    for val in u_field.values:
        node_data.append({{
            "label": val.nodeLabel,
            "u_mag": float(val.magnitude),
            "u": [float(val.data[i]) for i in range(len(val.data))],
        }})

# Compute summary
summary = {{
    "job_name": "{job_name}",
    "step_name": step_name,
    "n_elements": len(elem_data),
    "n_nodes": len(node_data),
    "mises": {{
        "max": max(mises_vals) if mises_vals else 0.0,
        "mean": sum(mises_vals) / len(mises_vals) if mises_vals else 0.0,
        "p95": sorted(mises_vals)[int(0.95*len(mises_vals))] if mises_vals else 0.0,
    }},
    "max_principal": {{
        "max": max(s_principal_vals) if s_principal_vals else 0.0,
        "mean": sum(s_principal_vals) / len(s_principal_vals) if s_principal_vals else 0.0,
    }},
    "displacement": {{
        "max_mag": max(u_mag_vals) if u_mag_vals else 0.0,
        "mean_mag": sum(u_mag_vals) / len(u_mag_vals) if u_mag_vals else 0.0,
    }},
}}

# Write JSON summary
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print("Stress summary:", OUT_JSON)

# Write element CSV
with open(OUT_CSV, "w") as f:
    f.write("elem_label,mises,max_principal,min_principal\\n")
    for ed in elem_data:
        f.write("%d,%.6e,%.6e,%.6e\\n" % (
            ed["label"], ed["mises"], ed["max_principal"], ed["min_principal"]))
print("Element CSV:", OUT_CSV)

odb.close()
print("Done.")
"""
    )

    if dry_run:
        print(f"  [DRY] abaqus python {extract_script}")
        return None

    cmd = f"abaqus python {extract_script}"
    print(f"  Extracting: {cmd}")
    try:
        proc = subprocess.run(
            cmd.split(),
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            print(f"  [WARN] Extraction returned {proc.returncode}")
            if proc.stderr:
                print(f"  stderr: {proc.stderr[:300]}")
    except FileNotFoundError:
        print("  [WARN] abaqus python not available, trying CSV-only mode")
        return _extract_from_existing_csv(work_dir, job_name)

    stress_json = work_dir / f"{job_name}_stress.json"
    if stress_json.exists():
        with open(stress_json) as f:
            return json.load(f)
    return None


def _extract_from_existing_csv(work_dir, job_name):
    """Fallback: extract from existing element CSV if ODB extraction failed."""
    csv_path = work_dir / f"{job_name}_elements.csv"
    if not csv_path.exists():
        # Try odb_elements.csv
        csv_path = work_dir / "odb_elements.csv"
    if not csv_path.exists():
        return None

    mises = []
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    mises.append(float(parts[1]))
                except ValueError:
                    continue

    if not mises:
        return None

    mises = np.array(mises)
    return {
        "job_name": job_name,
        "n_elements": len(mises),
        "mises": {
            "max": float(np.max(mises)),
            "mean": float(np.mean(mises)),
            "p95": float(np.percentile(mises, 95)),
        },
    }


def run_condition(condition, args):
    """Run full Abaqus pipeline for one condition."""
    t0 = time.perf_counter()
    tooth = args.tooth

    inps = find_inp_files(condition, tooth)
    if inps["tie"] is None:
        print(f"  [SKIP] No Tie INP found for {condition}/{tooth}")
        print("  Run generate_3d_conformal_auto.py first")
        return None

    work_dir = _ABAQUS_WORK / f"{condition}_{tooth}"
    work_dir.mkdir(parents=True, exist_ok=True)

    inp_path = inps["tie"]
    job_name = inp_path.stem

    if not args.extract_only:
        # Submit job
        print(f"\n[1/2] Abaqus job: {job_name}")
        submit_abaqus_job(inp_path, work_dir, cpus=args.cpus, dry_run=args.dry_run)

        # Check status
        status = check_sta_file(work_dir, job_name)
        print(f"  Status: {status}")
    else:
        print(f"\n[EXTRACT] {job_name}")

    # Extract results
    print("\n[2/2] Extracting stress results...")
    stress = extract_odb_results(work_dir, job_name, dry_run=args.dry_run)

    if stress:
        print("\n  Mises stress [MPa]:")
        print(f"    max  = {stress['mises']['max']:.4f}")
        print(f"    mean = {stress['mises']['mean']:.4f}")
        print(f"    p95  = {stress['mises']['p95']:.4f}")
        if "displacement" in stress:
            print(f"  Max displacement [mm]: {stress['displacement']['max_mag']:.6f}")

    # Save combined results
    result = {
        "condition": condition,
        "tooth": tooth,
        "job_name": job_name,
        "inp_path": str(inp_path),
        "stress": stress,
        "timing_s": round(time.perf_counter() - t0, 1),
    }
    out_json = work_dir / f"result_{condition}_{tooth}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved: {out_json}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Automated Abaqus execution pipeline")
    ap.add_argument(
        "--condition",
        default="dh_baseline",
        choices=["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"],
    )
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tooth", default="T23", choices=["T23", "T30", "T31"])
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conditions = (
        ["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"]
        if args.all
        else [args.condition]
    )

    results = {}
    for cond in conditions:
        results[cond] = run_condition(cond, args)

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Condition':<25} {'Mises max':>12} {'Mises mean':>12} {'U max':>10}")
        print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for cond, r in results.items():
            if r and r.get("stress"):
                s = r["stress"]
                u_max = s.get("displacement", {}).get("max_mag", 0)
                print(
                    f"{cond:<25} {s['mises']['max']:>12.4f} "
                    f"{s['mises']['mean']:>12.4f} {u_max:>10.6f}"
                )
            else:
                print(f"{cond:<25} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
