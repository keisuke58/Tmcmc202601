#!/usr/bin/env python3
"""
run_oj_beta_sweep.py  –  Generate INP files and run Abaqus jobs for
hollow crown / inter-proximal slit geometries at beta = 0.3, 0.7, 1.0.

The β=0.5 INP files (OJ_Crown_T23_b050.inp, OJ_Slit_T3031_b050.inp)
are used as templates.  Only the transverse stiffness (E2, E3) and
out-of-plane shear (G23) are rescaled; E1, G12, G13, nu, and all
geometry/BC data are unchanged.

Usage:
    python run_oj_beta_sweep.py [--dry-run] [--case crown|slit|both]

Output:
    OJ_Crown_T23_b030.inp / b070.inp / b100.inp
    OJ_Slit_T3031_b030.inp / b070.inp / b100.inp
    run_oj_beta_jobs.sh   (shell script to submit all new jobs)
"""

from __future__ import print_function, division
import re
import os
import sys
import stat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEMPLATES = {
    "crown": {
        "src_inp":  "OJ_Crown_T23_b050.inp",
        "src_beta": 0.50,
        "src_name": "OJ_Crown_T23_b050",
        "tgt_pat":  "OJ_Crown_T23_b{b}",
    },
    "slit": {
        "src_inp":  "OJ_Slit_T3031_b050.inp",
        "src_beta": 0.50,
        "src_name": "OJ_Slit_T3031_b050",
        "tgt_pat":  "OJ_Slit_T3031_b{b}",
    },
}

NEW_BETAS = [0.3, 0.7, 1.0]


def _bstr(beta):
    """0.3 -> '030', 1.0 -> '100'"""
    return "%03d" % int(round(beta * 100))


# ---------------------------------------------------------------------------
# Material rewriter
# ---------------------------------------------------------------------------

_EC_HEADER_RE = re.compile(
    r"^\s*\*Elastic\s*,\s*type\s*=\s*ENGINEERING\s+CONSTANTS\s*$", re.I
)


def _rewrite_materials(text, new_beta):
    """
    Scan every *Elastic, type=ENGINEERING CONSTANTS block and replace
    E2, E3, G23 with values consistent with new_beta while leaving
    E1 (stiff direction), G12, G13 (= E1/2*(1+nu)) untouched.

    ENGINEERING CONSTANTS layout (Abaqus 8-value-per-data-line rule):
      line 1: E1, E2, E3, nu12, nu13, nu23, G12, G13
      line 2: G23
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _EC_HEADER_RE.match(line):
            out.append(line)
            i += 1
            if i >= len(lines):
                break

            # ---- parse elastic-constants data line 1 -------------------
            ec_line = lines[i]
            parts = [p.strip() for p in ec_line.split(",")]
            if len(parts) >= 8:
                try:
                    E1   = float(parts[0])
                    # parts[1] = E2 (old, discarded)
                    # parts[2] = E3 (old, discarded)
                    nu12 = float(parts[3])
                    nu13 = float(parts[4])
                    nu23 = float(parts[5])
                    G12  = float(parts[6])   # E1 / (2*(1+nu)), unchanged
                    G13  = float(parts[7])   # same as G12
                except ValueError:
                    # Unexpected format – keep as-is and skip
                    out.append(ec_line)
                    i += 1
                    continue

                new_E2  = new_beta * E1
                new_E3  = new_beta * E1
                new_G23 = new_beta * G12   # = new_E2 / (2*(1+nu))

                # Reformat to match Abaqus style (space-prefixed, e-notation)
                new_ec = (" %g, %g, %g, %g, %g, %g, %g, %g"
                          % (E1, new_E2, new_E3, nu12, nu13, nu23, G12, G13))
                out.append(new_ec)
                i += 1

                # ---- parse / replace G23 (data line 2) -----------------
                if i < len(lines):
                    # G23 line: " 1.85882e+09,"
                    out.append(" %g," % new_G23)
                    i += 1
            else:
                # Fewer than 8 tokens – keep verbatim
                out.append(ec_line)
                i += 1
        else:
            out.append(line)
            i += 1

    return "\n".join(out)


# ---------------------------------------------------------------------------
# INP file builder
# ---------------------------------------------------------------------------

def make_inp(src_path, src_beta, src_name, new_beta, new_name):
    with open(src_path) as f:
        text = f.read()

    # 1. Update heading description
    text = text.replace("beta=%.2f" % src_beta, "beta=%.2f" % new_beta)
    # 2. Replace job / model name throughout
    text = text.replace(src_name, new_name)
    # 3. Rescale transverse stiffness & G23
    text = _rewrite_materials(text, new_beta)

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv

    case = "both"
    args = sys.argv[1:]
    for k, a in enumerate(args):
        if a == "--case" and k + 1 < len(args):
            case = args[k + 1]
        elif a.startswith("--case="):
            case = a.split("=", 1)[1]

    geoms = []
    if case in ("crown", "both"):
        geoms.append("crown")
    if case in ("slit", "both"):
        geoms.append("slit")
    if not geoms:
        print("ERROR: unknown --case value '%s'" % case)
        sys.exit(1)

    fem_dir = os.path.dirname(os.path.abspath(__file__))
    jobs = []   # list of (new_name, inp_basename)

    for geom in geoms:
        tmpl = TEMPLATES[geom]
        src_path = os.path.join(fem_dir, tmpl["src_inp"])
        if not os.path.isfile(src_path):
            print("ERROR: template INP not found: %s" % src_path)
            continue

        for beta in NEW_BETAS:
            b_str    = _bstr(beta)
            new_name = tmpl["tgt_pat"].format(b=b_str)
            new_inp  = os.path.join(fem_dir, new_name + ".inp")

            if os.path.isfile(new_inp):
                print("skip (exists): %s" % new_inp)
                jobs.append((new_name, new_name + ".inp"))
                continue

            print("generating %s  (beta=%.2f)" % (new_name, beta))
            text = make_inp(src_path, tmpl["src_beta"], tmpl["src_name"],
                            beta, new_name)
            if dry_run:
                print("  [dry-run] would write %s" % new_inp)
            else:
                with open(new_inp, "w") as f:
                    f.write(text)
                print("  wrote %s  (%d lines)" % (
                    new_inp, text.count("\n")))
            jobs.append((new_name, new_name + ".inp"))

    if not jobs:
        print("Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Write shell script to submit all jobs
    # ------------------------------------------------------------------
    sh_path = os.path.join(fem_dir, "run_oj_beta_jobs.sh")
    sh_lines = [
        "#!/bin/bash",
        "# Auto-generated by run_oj_beta_sweep.py",
        "# Submits hollow-crown / slit beta-sweep jobs",
        "set -e",
        "cd " + fem_dir,
        "",
    ]
    for name, inp_base in jobs:
        sh_lines.append("echo '=== %s ==='" % name)
        sh_lines.append(
            "abaqus job=%s inp=%s cpus=1 interactive" % (name, inp_base))
        sh_lines.append("")

    sh_content = "\n".join(sh_lines) + "\n"
    if dry_run:
        print("\n[dry-run] would write shell script: %s" % sh_path)
        print(sh_content)
    else:
        with open(sh_path, "w") as f:
            f.write(sh_content)
        os.chmod(sh_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP |
                 stat.S_IROTH | stat.S_IXOTH)
        print("\nShell script: %s" % sh_path)
        print("Submit with:  bash %s" % sh_path)


if __name__ == "__main__":
    main()
