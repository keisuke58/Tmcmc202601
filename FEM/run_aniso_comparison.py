#!/usr/bin/env python3
"""
run_aniso_comparison.py
=======================
C1: Run isotropic vs anisotropic Abaqus for each condition,
using the dominant ∇φ_pg direction from fem_aniso_analysis.py.

Sweep:
  aniso_ratio ∈ [1.0 (iso), 0.7, 0.5, 0.3]
  × 4 baseline conditions  +  3 OpenJaw real-tooth conditions

Baseline conditions: each job uses p50 DI field CSV + orientation from
  aniso_summary.json  →  abaqus_biofilm_aniso_3d.py

OpenJaw conditions: each job uses a per-tooth bbox JSON + p50 DI field CSV
  →  openjaw_p1_biofilm_solid.py  (real tooth cross-section, radial DI mapping)

Outputs:
  _aniso_sweep/results.csv
  _aniso_sweep/figures/fig_C1_smises_aniso_ratio.png
  _aniso_sweep/figures/fig_C1_aniso_vs_iso.png
  _aniso_sweep/figures/fig_C1_openjaw_vs_idealized.png

Usage:
  python run_aniso_comparison.py
  python run_aniso_comparison.py --plot-only
  python run_aniso_comparison.py --conditions dh_baseline openjaw_t23
  python run_aniso_comparison.py --aniso-ratios 1.0 0.5
  python run_aniso_comparison.py --skip-openjaw   # only baseline conditions
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DI_BASE = _HERE / "_di_credible"
_AN_BASE = _HERE / "_aniso"
_OUT = _HERE / "_aniso_sweep"

# Baseline (idealized geometry) conditions
_BASELINE_CONDITIONS = [
    "dh_baseline",
    "commensal_static",
    "commensal_hobic",
    "dysbiotic_static",
]

# OpenJaw per-tooth conditions (simple solid via openjaw_p1_biofilm_solid.py)
_OPENJAW_CONDITIONS = [
    "openjaw_t23",
    "openjaw_t30",
    "openjaw_t31",
]

# OpenJaw full-assembly conditions (hollow crown / slit via openjaw_p1_full_assembly.py)
_OPENJAW_FULL_CONDITIONS = [
    "openjaw_crown",  # hollow ring around T23
    "openjaw_slit",  # inter-proximal slit between T30 + T31
]

# Mapping: OpenJaw per-tooth condition → (tooth_key, di_condition_to_borrow_field_from)
_OPENJAW_SPEC = {
    "openjaw_t23": ("P1_Tooth_23", "dh_baseline"),
    "openjaw_t30": ("P1_Tooth_30", "dh_baseline"),
    "openjaw_t31": ("P1_Tooth_31", "dh_baseline"),
}

# Full-assembly conditions → (case arg for openjaw_p1_full_assembly.py, di_condition)
_OPENJAW_FULL_SPEC = {
    "openjaw_crown": ("crown", "dh_baseline"),
    "openjaw_slit": ("slit", "dh_baseline"),
}

CONDITIONS = _BASELINE_CONDITIONS + _OPENJAW_CONDITIONS + _OPENJAW_FULL_CONDITIONS

COND_LABELS = {
    "dh_baseline": "dh-baseline",
    "commensal_static": "Comm. Static",
    "commensal_hobic": "Comm. HOBIC",
    "dysbiotic_static": "Dysb. Static",
    "openjaw_t23": "OJ T23 (solid)",
    "openjaw_t30": "OJ T30 (solid)",
    "openjaw_t31": "OJ T31 (solid)",
    "openjaw_crown": "OJ Crown T23",
    "openjaw_slit": "OJ Slit T30-T31",
}

COND_COLORS = {
    "dh_baseline": "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
    "openjaw_t23": "#9467bd",
    "openjaw_t30": "#8c564b",
    "openjaw_t31": "#e377c2",
    "openjaw_crown": "#17becf",
    "openjaw_slit": "#bcbd22",
}

ANISO_RATIOS_DEFAULT = [1.0, 0.7, 0.5, 0.3]

# Bbox JSON produced by stl_bbox.py or openjaw_p1_auto_import.py
_BBOX_JSON_CANDIDATES = [
    _HERE / "p1_tooth_bbox.json",
    _HERE / "p1_tooth_bbox_from_cae.json",
]

GLOBAL_DI_SCALE = 0.025778
E_MAX = 10.0e9
E_MIN = 0.5e9
DI_EXPONENT = 2.0
N_BINS = 20
NU = 0.30

# ---------------------------------------------------------------------------


def _get_field_csv(cond: str) -> Path | None:
    """Use p50 DI field from B1 output."""
    p = _DI_BASE / cond / "p50_field.csv"
    return p if p.exists() else None


def _get_orientation(cond: str) -> dict | None:
    """Load dominant direction from aniso_summary.json."""
    p = _AN_BASE / cond / "aniso_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _find_bbox_json() -> Path | None:
    """Return first existing bbox JSON candidate."""
    for p in _BBOX_JSON_CANDIDATES:
        if p.exists():
            return p
    return None


def _run_abaqus_openjaw(
    tooth_key: str,
    field_csv: Path,
    job_name: str,
    aniso_ratio: float,
    out_dir: Path,
    geometry: str = "crown",
) -> dict | None:
    """
    Run openjaw_p1_biofilm_solid.py for one tooth / aniso_ratio combination.
    Extracts S_Mises from the resulting ODB via compare_biofilm_abaqus.py.
    """
    done_flag = out_dir / "done.flag"
    stress_json = out_dir / "stress.json"
    stress_csv = out_dir / "stress_raw.csv"

    if done_flag.exists() and stress_json.exists():
        with stress_json.open() as f:
            return json.load(f)

    bbox_json = _find_bbox_json()
    if bbox_json is None:
        print("    [skip] no bbox JSON found – run stl_bbox.py first")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cmd = [
        "abaqus",
        "cae",
        "noGUI=%s" % str(_HERE / "openjaw_p1_biofilm_solid.py"),
        "--",
        "--bbox-json",
        str(bbox_json),
        "--tooth-key",
        tooth_key,
        "--field-csv",
        str(field_csv),
        "--geometry",
        geometry,
        "--aniso-ratio",
        "%.3f" % aniso_ratio,
        "--di-scale",
        "%.6f" % GLOBAL_DI_SCALE,
        "--e-max",
        "%.6g" % E_MAX,
        "--e-min",
        "%.6g" % E_MIN,
        "--di-exponent",
        "%.2f" % DI_EXPONENT,
        "--n-bins",
        str(N_BINS),
        "--nu",
        "%.3f" % NU,
        "--e1-dir",
        "radial",
        "--job-name",
        job_name,
        "--poly-from-json",
    ]
    ret = subprocess.run(cmd, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print("    [warn] Abaqus rc=%d (OpenJaw)" % ret.returncode)
        return None

    odb = _HERE / ("%s.odb" % job_name)
    if not odb.exists():
        return None

    cmd2 = [
        "abaqus",
        "python",
        str(_HERE / "compare_biofilm_abaqus.py"),
        str(stress_csv),
        str(odb),
    ]
    ret2 = subprocess.run(cmd2, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret2.returncode != 0 or not stress_csv.exists():
        return None

    substrate = surface = None
    with stress_csv.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 16 or parts[0].startswith("odb"):
                continue
            try:
                df = float(parts[2])
                sm = float(parts[15])
            except ValueError:
                continue
            if abs(df - 0.0) < 1e-6:
                substrate = sm
            elif abs(df - 1.0) < 1e-6:
                surface = sm

    if substrate is None or surface is None:
        return None

    result = {
        "aniso_ratio": aniso_ratio,
        "substrate_smises": substrate,
        "surface_smises": surface,
        "elapsed_s": time.perf_counter() - t0,
        "geometry": "openjaw_%s" % geometry,
    }
    with stress_json.open("w") as f:
        json.dump(result, f, indent=2)
    done_flag.touch()

    print(
        "    β=%.2f  sub=%.3g Pa  surf=%.3g Pa  (%.1fs)"
        % (aniso_ratio, substrate, surface, result["elapsed_s"])
    )
    return result


def _run_abaqus_openjaw_full(
    case: str,
    field_csv: Path,
    job_name: str,
    aniso_ratio: float,
    out_dir: Path,
) -> dict | None:
    """
    Run openjaw_p1_full_assembly.py for crown or slit geometry.
    Extracts S_Mises from resulting ODB via compare_biofilm_abaqus.py.
    """
    done_flag = out_dir / "done.flag"
    stress_json = out_dir / "stress.json"
    stress_csv = out_dir / "stress_raw.csv"

    if done_flag.exists() and stress_json.exists():
        with stress_json.open() as f:
            return json.load(f)

    bbox_json = _find_bbox_json()
    if bbox_json is None:
        print("    [skip] no bbox JSON – run stl_bbox.py first")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cmd = [
        "abaqus",
        "cae",
        "noGUI=%s" % str(_HERE / "openjaw_p1_full_assembly.py"),
        "--",
        "--bbox-json",
        str(bbox_json),
        "--field-csv",
        str(field_csv),
        "--case",
        case,
        "--aniso-ratio",
        "%.3f" % aniso_ratio,
        "--di-scale",
        "%.6f" % GLOBAL_DI_SCALE,
        "--e-max",
        "%.6g" % E_MAX,
        "--e-min",
        "%.6g" % E_MIN,
        "--di-exponent",
        "%.2f" % DI_EXPONENT,
        "--n-bins",
        str(N_BINS),
        "--nu",
        "%.3f" % NU,
        "--poly-from-json",
        "--crown-job",
        "%s_crown" % job_name,
        "--slit-job",
        "%s_slit" % job_name,
    ]
    ret = subprocess.run(cmd, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print("    [warn] Abaqus rc=%d (OpenJaw-full)" % ret.returncode)
        return None

    # Determine ODB name based on case
    odb_name = ("%s_%s" % (job_name, case)) + ".odb"
    odb = _HERE / odb_name
    if not odb.exists():
        # Try crown/slit job names
        for suffix in ("_crown", "_slit"):
            odb = _HERE / (job_name + suffix + ".odb")
            if odb.exists():
                break
    if not odb.exists():
        return None

    cmd2 = [
        "abaqus",
        "python",
        str(_HERE / "compare_biofilm_abaqus.py"),
        str(stress_csv),
        str(odb),
    ]
    ret2 = subprocess.run(cmd2, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret2.returncode != 0 or not stress_csv.exists():
        return None

    substrate = surface = None
    with stress_csv.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 16 or parts[0].startswith("odb"):
                continue
            try:
                df = float(parts[2])
                sm = float(parts[15])
            except ValueError:
                continue
            if abs(df - 0.0) < 1e-6:
                substrate = sm
            elif abs(df - 1.0) < 1e-6:
                surface = sm

    if substrate is None or surface is None:
        return None

    result = {
        "aniso_ratio": aniso_ratio,
        "substrate_smises": substrate,
        "surface_smises": surface,
        "elapsed_s": time.perf_counter() - t0,
        "geometry": "openjaw_full_%s" % case,
    }
    with stress_json.open("w") as f:
        json.dump(result, f, indent=2)
    done_flag.touch()
    print(
        "    β=%.2f  sub=%.3g Pa  surf=%.3g Pa  (%.1fs)"
        % (aniso_ratio, substrate, surface, result["elapsed_s"])
    )
    return result


def _run_abaqus_aniso(
    field_csv: Path,
    job_name: str,
    aniso_ratio: float,
    e1: list[float],
    out_dir: Path,
) -> dict | None:
    done_flag = out_dir / "done.flag"
    stress_json = out_dir / "stress.json"
    stress_csv = out_dir / "stress_raw.csv"

    if done_flag.exists() and stress_json.exists():
        with stress_json.open() as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "abaqus",
        "cae",
        "noGUI=%s" % str(_HERE / "abaqus_biofilm_aniso_3d.py"),
        "--",
        "--field-csv",
        str(field_csv),
        "--job-name",
        job_name,
        "--aniso-ratio",
        "%.3f" % aniso_ratio,
        "--e1-x",
        "%.4f" % e1[0],
        "--e1-y",
        "%.4f" % e1[1],
        "--e1-z",
        "%.4f" % e1[2],
        "--di-scale",
        "%.6f" % GLOBAL_DI_SCALE,
        "--e-max",
        "%.6g" % E_MAX,
        "--e-min",
        "%.6g" % E_MIN,
        "--di-exponent",
        "%.2f" % DI_EXPONENT,
        "--n-bins",
        str(N_BINS),
        "--nu",
        "%.3f" % NU,
    ]
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print("    [warn] Abaqus rc=%d" % ret.returncode)
        return None

    # Extract stress using compare_biofilm_abaqus.py
    odb = _HERE / ("%s.odb" % job_name)
    if not odb.exists():
        return None
    cmd2 = [
        "abaqus",
        "python",
        str(_HERE / "compare_biofilm_abaqus.py"),
        str(stress_csv),
        str(odb),
    ]
    ret2 = subprocess.run(cmd2, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret2.returncode != 0 or not stress_csv.exists():
        return None

    substrate = surface = None
    with stress_csv.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 16 or parts[0].startswith("odb"):
                continue
            try:
                df = float(parts[2])
                sm = float(parts[15])
            except ValueError:
                continue
            if abs(df - 0.0) < 1e-6:
                substrate = sm
            elif abs(df - 1.0) < 1e-6:
                surface = sm

    if substrate is None or surface is None:
        return None

    result = {
        "aniso_ratio": aniso_ratio,
        "substrate_smises": substrate,
        "surface_smises": surface,
        "elapsed_s": time.perf_counter() - t0,
    }
    with stress_json.open("w") as f:
        json.dump(result, f, indent=2)
    done_flag.touch()

    print(
        "    β=%.2f  sub=%.3g Pa  surf=%.3g Pa  (%.1fs)"
        % (aniso_ratio, substrate, surface, result["elapsed_s"])
    )
    return result


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--aniso-ratios", nargs="+", type=float, default=ANISO_RATIOS_DEFAULT)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument(
        "--skip-openjaw",
        action="store_true",
        help="Run only the baseline (idealized geometry) conditions",
    )
    ap.add_argument(
        "--openjaw-geometry",
        default="crown",
        choices=["crown", "slit"],
        help="Biofilm geometry for OpenJaw cases (default: crown)",
    )
    args = ap.parse_args()

    all_oj = _OPENJAW_CONDITIONS + _OPENJAW_FULL_CONDITIONS
    if args.skip_openjaw:
        active_conditions = [c for c in args.conditions if c not in all_oj]
    else:
        active_conditions = list(args.conditions)

    _OUT.mkdir(parents=True, exist_ok=True)
    csv_path = _OUT / "results.csv"
    results = []

    if not args.plot_only:
        for cond in active_conditions:
            print("\n[%s]" % cond)

            # ── OpenJaw full-assembly (hollow crown / slit) ───────────────
            if cond in _OPENJAW_FULL_CONDITIONS:
                case_arg, di_cond = _OPENJAW_FULL_SPEC[cond]
                field_csv = _get_field_csv(di_cond)
                if field_csv is None:
                    print("  [skip] no p50 field CSV for %s" % di_cond)
                    continue
                print("  full-assembly case=%s  DI from %s" % (case_arg, di_cond))
                for beta in args.aniso_ratios:
                    tag = "b%03d" % int(beta * 100)
                    job_name = "ojf_%s_%s" % (cond.replace("openjaw_", ""), tag)
                    out_dir = _OUT / cond / tag
                    rec = _run_abaqus_openjaw_full(case_arg, field_csv, job_name, beta, out_dir)
                    if rec:
                        results.append({"condition": cond, **rec})
                continue

            # ── OpenJaw per-tooth conditions (solid) ──────────────────────
            if cond in _OPENJAW_CONDITIONS:
                tooth_key, di_cond = _OPENJAW_SPEC[cond]
                field_csv = _get_field_csv(di_cond)
                if field_csv is None:
                    print(
                        "  [skip] no p50 field CSV for %s "
                        "(run aggregate_di_credible.py first)" % di_cond
                    )
                    continue
                print("  tooth=%s  DI field from %s" % (tooth_key, di_cond))
                for beta in args.aniso_ratios:
                    tag = "b%03d" % int(beta * 100)
                    job_name = "oj_%s_%s" % (cond.replace("openjaw_", ""), tag)
                    out_dir = _OUT / cond / tag
                    rec = _run_abaqus_openjaw(
                        tooth_key,
                        field_csv,
                        job_name,
                        beta,
                        out_dir,
                        geometry=args.openjaw_geometry,
                    )
                    if rec:
                        results.append({"condition": cond, **rec})
                continue

            # ── Baseline idealized conditions ─────────────────────────────
            field_csv = _get_field_csv(cond)
            orient = _get_orientation(cond)

            if field_csv is None:
                print("  [skip] no p50 field CSV (run aggregate_di_credible.py first)")
                continue
            if orient is None:
                print("  [skip] no aniso_summary.json (run fem_aniso_analysis.py first)")
                continue

            e1 = orient["e1"]
            print(
                "  e1=[%.3f,%.3f,%.3f]  angle=%.1f deg"
                % (e1[0], e1[1], e1[2], orient["angle_x_deg"])
            )

            for beta in args.aniso_ratios:
                tag = "b%03d" % int(beta * 100)
                job_name = "aniso_%s_%s" % (cond[:4], tag)
                out_dir = _OUT / cond / tag

                rec = _run_abaqus_aniso(field_csv, job_name, beta, e1, out_dir)
                if rec:
                    results.append({"condition": cond, **rec})

        if results:
            fieldnames = list(results[0].keys())
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(results)
            print("\n[done] %d results → %s" % (len(results), csv_path))
        elif csv_path.exists():
            with csv_path.open() as f:
                results = [
                    {
                        k: (float(v) if k not in ("condition", "geometry") else v)
                        for k, v in r.items()
                    }
                    for r in csv.DictReader(f)
                ]

    else:
        if not csv_path.exists():
            print("[plot-only] no results.csv found.")
            return
        with csv_path.open() as f:
            results = [
                {k: (float(v) if k not in ("condition", "geometry") else v) for k, v in r.items()}
                for r in csv.DictReader(f)
            ]

    _plot_results(results, args.aniso_ratios)


# ---------------------------------------------------------------------------


def _plot_results(results: list[dict], aniso_ratios: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = _OUT / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Fig C1-1: S_Mises vs aniso_ratio, per condition ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for si, (skey, slabel) in enumerate(
        [
            ("substrate_smises", "Substrate"),
            ("surface_smises", "Surface"),
        ]
    ):
        ax = axes[si]
        for cond in CONDITIONS:
            rows = sorted(
                [r for r in results if r["condition"] == cond],
                key=lambda r: r["aniso_ratio"],
            )
            if not rows:
                continue
            betas = [r["aniso_ratio"] for r in rows]
            vals = [r[skey] / 1e6 for r in rows]
            ax.plot(
                betas,
                vals,
                "o-",
                color=COND_COLORS.get(cond, "gray"),
                label=COND_LABELS.get(cond, cond),
                lw=1.8,
                ms=7,
            )

        ax.axvline(1.0, color="gray", lw=0.8, ls="--", label="isotropic ref (β=1)")
        ax.set_xlabel("Anisotropy ratio β = E_trans/E_stiff", fontsize=10)
        ax.set_ylabel("S_Mises (MPa)", fontsize=10)
        ax.set_title("C1: %s S_Mises vs β" % slabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        ax.invert_xaxis()  # β=1 (iso) on left, most aniso on right

    fig.suptitle(
        "C1: Transverse Isotropy Effect on S_Mises\n"
        "(β=1 isotropic  →  β<1 stiffer in ∇φ_pg direction)",
        fontsize=12,
        fontweight="bold",
    )
    out = fig_dir / "fig_C1_smises_vs_beta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)

    # ── Fig C1-2: Aniso / Iso ratio bar chart per condition ──────────────
    iso_sub = {}
    iso_surf = {}
    for r in results:
        if abs(r["aniso_ratio"] - 1.0) < 0.01:
            iso_sub[r["condition"]] = r["substrate_smises"]
            iso_surf[r["condition"]] = r["surface_smises"]

    # pick one representative aniso point (β=0.5)
    aniso_sub = {}
    aniso_surf = {}
    for r in results:
        if abs(r["aniso_ratio"] - 0.5) < 0.01:
            aniso_sub[r["condition"]] = r["substrate_smises"]
            aniso_surf[r["condition"]] = r["surface_smises"]

    if iso_sub and aniso_sub:
        conds_both = [c for c in CONDITIONS if c in iso_sub and c in aniso_sub]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

        for si, (iso_d, aniso_d, slabel) in enumerate(
            [
                (iso_sub, aniso_sub, "Substrate"),
                (iso_surf, aniso_surf, "Surface"),
            ]
        ):
            ax = axes[si]
            x = np.arange(len(conds_both))
            w = 0.35
            iso_vals = [iso_d.get(c, 0) / 1e6 for c in conds_both]
            aniso_vals = [aniso_d.get(c, 0) / 1e6 for c in conds_both]

            ax.bar(
                x - w / 2,
                iso_vals,
                w,
                label="Isotropic (β=1.0)",
                color="steelblue",
                alpha=0.8,
                edgecolor="k",
                lw=0.5,
            )
            ax.bar(
                x + w / 2,
                aniso_vals,
                w,
                label="Anisotropic (β=0.5)",
                color="tomato",
                alpha=0.8,
                edgecolor="k",
                lw=0.5,
            )

            # ratio text
            for i, (iv, av) in enumerate(zip(iso_vals, aniso_vals)):
                if iv > 0:
                    ratio = av / iv
                    ax.text(
                        i,
                        max(iv, av) * 1.04,
                        "×%.2f" % ratio,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#333",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [COND_LABELS.get(c, c) for c in conds_both], rotation=12, ha="right", fontsize=9
            )
            ax.set_ylabel("S_Mises (MPa)", fontsize=10)
            ax.set_title("C1: Iso vs Aniso  –  %s" % slabel)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.suptitle(
            "C1: Isotropic vs Anisotropic (β=0.5)  –  S_Mises Comparison\n"
            "Anisotropy axis = dominant ∇φ_Pg direction",
            fontsize=12,
            fontweight="bold",
        )
        out = fig_dir / "fig_C1_aniso_vs_iso.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[plot] %s" % out.name)

    # ── Fig C1-3: OpenJaw vs idealised  (real tooth geometry effect) ─────────
    openjaw_conds = [c for c in _OPENJAW_CONDITIONS if any(r["condition"] == c for r in results)]
    baseline_ref = "dh_baseline"
    baseline_rows = [r for r in results if r["condition"] == baseline_ref]

    if openjaw_conds and baseline_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        for si, (skey, slabel) in enumerate(
            [
                ("substrate_smises", "Substrate"),
                ("surface_smises", "Surface"),
            ]
        ):
            ax = axes[si]

            # Baseline reference line
            b_rows = sorted(baseline_rows, key=lambda r: r["aniso_ratio"])
            betas = [r["aniso_ratio"] for r in b_rows]
            vals = [r.get(skey, 0) / 1e6 for r in b_rows]
            ax.plot(
                betas,
                vals,
                "k--o",
                lw=1.5,
                ms=6,
                label="%s (idealized)" % COND_LABELS.get(baseline_ref, baseline_ref),
            )

            # OpenJaw teeth
            for cond in openjaw_conds:
                oj_rows = sorted(
                    [r for r in results if r["condition"] == cond],
                    key=lambda r: r["aniso_ratio"],
                )
                if not oj_rows:
                    continue
                betas_oj = [r["aniso_ratio"] for r in oj_rows]
                vals_oj = [r.get(skey, 0) / 1e6 for r in oj_rows]
                ax.plot(
                    betas_oj,
                    vals_oj,
                    "o-",
                    color=COND_COLORS.get(cond, "gray"),
                    label=COND_LABELS.get(cond, cond),
                    lw=1.8,
                    ms=7,
                )

            ax.axvline(1.0, color="gray", lw=0.8, ls=":", label="isotropic ref (β=1)")
            ax.set_xlabel("Anisotropy ratio β = E_trans/E_stiff", fontsize=10)
            ax.set_ylabel("S_Mises (MPa)", fontsize=10)
            ax.set_title("C1: OpenJaw vs Idealized  –  %s" % slabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, linestyle="--")
            ax.invert_xaxis()

        fig.suptitle(
            "C1: Real Tooth Geometry (OpenJaw P1) vs Idealised Crown\n"
            "DI field = dh_baseline  |  radial DI mapping",
            fontsize=12,
            fontweight="bold",
        )
        out = fig_dir / "fig_C1_openjaw_vs_idealized.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[plot] %s" % out.name)

    print("\n[plot] all → %s/" % fig_dir)


if __name__ == "__main__":
    main()
