#!/usr/bin/env python3
"""
generate_1kpa_inp.py
====================
Generate 1 kPa (brushing load) INP files from existing 1 MPa INPs.
Scale all *Cload values by 1e-3 (1000 Pa / 1e6 Pa).

Usage:
  python generate_1kpa_inp.py
"""
import re
import shutil
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_JOBS_SRC = _HERE / "_abaqus_auto_jobs"
_JOBS_DST = _HERE / "_abaqus_1kpa_jobs"

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
SCALE = 1e-3  # 1 kPa / 1 MPa


def scale_inp(src_inp: Path, dst_inp: Path):
    """Read INP, scale Cload values by SCALE, write new INP."""
    lines = src_inp.read_text().splitlines()
    out = []
    in_cload = False

    for line in lines:
        stripped = line.strip()

        # Update comment
        if "Pressure = " in line and "Pa" in line:
            line = line.replace("1e+06 Pa (= 1 MPa)",
                                "1000 Pa (= 0.001 MPa)")
            line = line.replace("1e+06", "1000")
        if "Inward pressure" in line and "MPa" in line:
            line = re.sub(r"pressure\s+[\d.e+-]+\s+MPa",
                          "pressure 0.001 MPa", line, flags=re.IGNORECASE)
            line = line.replace("1 MPa", "0.001 MPa")

        if stripped.startswith("*Cload"):
            in_cload = True
            out.append(line)
            continue

        if in_cload:
            # Cload data lines: "node_id, dof, force_value"
            if stripped.startswith("*") or stripped == "":
                in_cload = False
                out.append(line)
                continue
            parts = stripped.split(",")
            if len(parts) == 3:
                node_id = parts[0].strip()
                dof = parts[1].strip()
                force = float(parts[2].strip()) * SCALE
                out.append(f" {node_id}, {dof}, {force:.10g}")
                continue

        out.append(line)

    dst_inp.write_text("\n".join(out) + "\n")
    print(f"  Written: {dst_inp.name}")


def main():
    _JOBS_DST.mkdir(exist_ok=True)

    for cond in CONDITIONS:
        src_dir = _JOBS_SRC / f"{cond}_T23_v2"
        inp_files = list(src_dir.glob("two_layer_T23_*.inp"))
        if not inp_files:
            print(f"  SKIP {cond}: no INP found")
            continue

        src_inp = inp_files[0]
        dst_dir = _JOBS_DST / f"{cond}_T23_1kpa"
        dst_dir.mkdir(exist_ok=True)

        # New job name
        job_name = f"two_layer_T23_{cond}_1kpa"
        dst_inp = dst_dir / f"{job_name}.inp"

        print(f"\n[{cond}]")
        scale_inp(src_inp, dst_inp)


if __name__ == "__main__":
    main()
