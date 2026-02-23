#!/usr/bin/env python3
"""
run_material_sensitivity_sweep.py
==================================
A1 + A2 + A3 combined material-property sensitivity sweep.

A1: E_max × E_min grid  (4×4 = 16 pts),  n_exp = 2 fixed
A2: n_exp ∈ [1, 2, 3],  E_max = 10 GPa / E_min = 0.5 GPa fixed
A3: 3 theta variants  (dh_old / mild_weight / nolambda_baseline)

Pipeline per combination (theta_variant × Abaqus params):
  1. Run 3D FEM with theta_MAP  →  DI field CSV   (cached per theta variant)
  2. abaqus cae  →  ODB  (resume-safe via done.flag)
  3. abaqus python  →  S_Mises (substrate + surface)

Total Abaqus jobs: (16 + 3) × 3 variants ≈ 57  (~10 min)

Outputs:
  _material_sweep/results.csv
  _material_sweep/figures/fig_A1_emax_emin_heatmap.png
  _material_sweep/figures/fig_A2_nexp_bars.png
  _material_sweep/figures/fig_A3_theta_comparison.png

Usage:
  python run_material_sensitivity_sweep.py               # full run
  python run_material_sensitivity_sweep.py --plot-only   # re-plot only
  python run_material_sensitivity_sweep.py --n-macro 20  # quick test (3 min)
  python run_material_sensitivity_sweep.py --variants mild_weight  # single variant
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent

THETA_VARIANTS: dict[str, Path] = {
    "dh_old":      _TMCMC_ROOT / "data_5species/_runs/sweep_pg_20260217_081459/dh_baseline/theta_MAP.json",
    "mild_weight": _TMCMC_ROOT / "_sweeps/K0.05_n4.0/theta_MAP.json",
    "nolambda":    _TMCMC_ROOT / "_sweeps/K0.05_n4.0_baseline/theta_MAP.json",
}

VARIANT_LABELS: dict[str, str] = {
    "dh_old":      "Old dh (a35=21.4)",
    "mild_weight": "Mild-weight (a35=3.56)",
    "nolambda":    "No-λ baseline (a35=20.9)",
}

VARIANT_COLORS: dict[str, str] = {
    "dh_old":      "#d62728",
    "mild_weight": "#2ca02c",
    "nolambda":    "#1f77b4",
}

FEM_CONDITION = "dh_baseline"

# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------
# A1: E_max × E_min  (n = 2 fixed)
E_MAX_VALS: list[float] = [5.0e9, 10.0e9, 15.0e9, 20.0e9]   # Pa
E_MIN_VALS: list[float] = [0.1e9,  0.5e9,  1.0e9,  2.0e9]   # Pa
N_EXP_A1: float = 2.0

# A2: n_exp sweep  (E fixed)
E_MAX_A2: float = 10.0e9
E_MIN_A2: float =  0.5e9
N_EXP_VALS: list[float] = [1.0, 2.0, 3.0]

# Abaqus shared settings
GLOBAL_DI_SCALE: float = 0.025778   # 1.1 × max(DI) from commensal_static
N_BINS: int   = 20
NU:     float = 0.30

_OUT = _HERE / "_material_sweep"

# ---------------------------------------------------------------------------
# FEM helpers
# ---------------------------------------------------------------------------

def _load_theta(json_path: Path) -> np.ndarray:
    d = json.loads(json_path.read_text())
    arr = d.get("theta_full", d.get("theta_sub", []))
    return np.asarray(arr, dtype=float)


def _run_fem(
    theta: np.ndarray,
    n_macro: int,
    n_react_sub: int,
    dt_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import fem_3d_extension as f3d
    sim = f3d.FEM3DBiofilm(
        theta,
        Nx=15, Ny=15, Nz=15,
        Lx=1.0, Ly=1.0, Lz=1.0,
        n_macro=n_macro,
        n_react_sub=n_react_sub,
        dt_h=dt_h,
        save_every=n_macro,       # only final snapshot
        condition=FEM_CONDITION,
    )
    snaps_phi, _ = sim.run()
    return snaps_phi, sim.x_mesh, sim.y_mesh, sim.z_mesh


def _export_field_csv(
    snaps_phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    out_csv: Path,
) -> None:
    phi_last  = snaps_phi[-1]                      # (5, Nx, Ny, Nz)
    phi_nodes = phi_last.transpose(1, 2, 3, 0)    # (Nx, Ny, Nz, 5)
    phi_sum   = phi_nodes.sum(axis=-1)
    safe      = np.where(phi_sum > 0, phi_sum, 1.0)
    p         = phi_nodes / safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H  = -(p * log_p).sum(axis=-1)
    di = 1.0 - H / np.log(5.0)

    phi_pg = phi_nodes[..., 4]
    Nx, Ny, Nz = phi_pg.shape
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("x,y,z,phi_pg,di\n")
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    f.write("%.8e,%.8e,%.8e,%.8e,%.8e\n" % (
                        x[ix], y[iy], z[iz],
                        float(phi_pg[ix, iy, iz]),
                        float(di[ix, iy, iz]),
                    ))

# ---------------------------------------------------------------------------
# Abaqus helpers
# ---------------------------------------------------------------------------

def _run_abaqus_cae(
    field_csv: Path,
    job_name: str,
    e_max: float,
    e_min: float,
    n_exp: float,
) -> int:
    cmd = [
        "abaqus", "cae",
        "noGUI=%s" % str(_HERE / "abaqus_biofilm_demo_3d.py"),
        "--",
        "--field-csv",   str(field_csv),
        "--mapping",     "power",
        "--di-scale",    "%.6f" % GLOBAL_DI_SCALE,
        "--n-bins",      str(N_BINS),
        "--e-max",       "%.6g"  % e_max,
        "--e-min",       "%.6g"  % e_min,
        "--di-exponent", "%.2f"  % n_exp,
        "--nu",          "%.3f"  % NU,
        "--job-name",    job_name,
    ]
    ret = subprocess.run(
        cmd, cwd=str(_HERE),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return ret.returncode


def _extract_stress(job_name: str, stress_csv: Path) -> dict | None:
    odb_path = _HERE / ("%s.odb" % job_name)
    if not odb_path.exists():
        return None
    cmd = [
        "abaqus", "python",
        str(_HERE / "compare_biofilm_abaqus.py"),
        str(stress_csv),
        str(odb_path),
    ]
    ret = subprocess.run(
        cmd, cwd=str(_HERE),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if ret.returncode != 0 or not stress_csv.exists():
        return None
    substrate = surface = None
    with stress_csv.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 16 or parts[0].startswith("odb"):
                continue
            try:
                depth_frac = float(parts[2])
                smises     = float(parts[15])
            except ValueError:
                continue
            if abs(depth_frac - 0.0) < 1e-6:
                substrate = smises
            elif abs(depth_frac - 1.0) < 1e-6:
                surface = smises
    if substrate is None or surface is None:
        return None
    return {"substrate_smises": substrate, "surface_smises": surface}


def _run_one_abaqus(
    field_csv: Path,
    job_name: str,
    e_max: float,
    e_min: float,
    n_exp: float,
    out_dir: Path,
) -> dict | None:
    """Run one Abaqus job, resume-safe via done.flag."""
    done_flag   = out_dir / "done.flag"
    stress_json = out_dir / "stress.json"
    stress_csv  = out_dir / "stress_raw.csv"

    if done_flag.exists() and stress_json.exists():
        with stress_json.open() as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rc = _run_abaqus_cae(field_csv, job_name, e_max, e_min, n_exp)
    if rc != 0:
        print("    [warn] Abaqus rc=%d → %s" % (rc, job_name))
        return None

    stress = _extract_stress(job_name, stress_csv)
    if stress is None:
        print("    [warn] stress extraction failed → %s" % job_name)
        return None

    record = {
        "e_max": e_max,
        "e_min": e_min,
        "n_exp": n_exp,
        **stress,
    }
    with stress_json.open("w") as f:
        json.dump(record, f, indent=2)
    done_flag.touch()

    print("    done in %.1fs  sub=%.3g Pa  surf=%.3g Pa" % (
        time.perf_counter() - t0,
        stress["substrate_smises"],
        stress["surface_smises"],
    ))
    return record

# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "sweep", "variant",
    "e_max", "e_min", "n_exp",
    "substrate_smises", "surface_smises",
]


def _save_results_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(results)


def _load_results_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                k: (float(v) if k not in ("sweep", "variant") else v)
                for k, v in r.items()
            })
    return rows


def _collect_done_results(variants: list[str]) -> list[dict]:
    """Scan done.flag files to rebuild results without re-running."""
    rows = []
    for variant in variants:
        var_dir = _OUT / variant
        for sweep_tag, subdir in [("A1", "a1_"), ("A2", "a2_")]:
            for d in sorted(var_dir.glob(subdir + "*")):
                sj = d / "stress.json"
                if sj.exists():
                    with sj.open() as f:
                        rec = json.load(f)
                    rows.append({
                        "sweep":   sweep_tag,
                        "variant": variant,
                        **rec,
                    })
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only",   action="store_true",
                    help="Skip FEM/Abaqus, re-plot from existing results.csv")
    ap.add_argument("--n-macro",     type=int,   default=100,
                    help="FEM macro steps per theta variant (default 100)")
    ap.add_argument("--n-react-sub", type=int,   default=50,
                    help="FEM reaction sub-steps per macro step (default 50)")
    ap.add_argument("--dt-h",        type=float, default=1e-5,
                    help="FEM Hamilton time step (default 1e-5)")
    ap.add_argument("--variants",    nargs="+",  default=list(THETA_VARIANTS),
                    choices=list(THETA_VARIANTS),
                    help="Which theta variants to run (default: all 3)")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    csv_path = _OUT / "results.csv"

    if args.plot_only:
        if not csv_path.exists():
            # try rebuilding from done flags
            rows = _collect_done_results(list(THETA_VARIANTS))
            if rows:
                _save_results_csv(rows, csv_path)
            else:
                sys.exit("[plot-only] No results.csv and no done flags found.")
        _plot_all(csv_path)
        return

    # ── Run sweep ────────────────────────────────────────────────────────────
    results: list[dict] = []

    for variant in args.variants:
        theta_path = THETA_VARIANTS[variant]
        if not theta_path.exists():
            print("[warn] theta_MAP not found: %s  (skip)" % theta_path)
            continue

        print("\n" + "=" * 60)
        print("Theta variant: %s  (%s)" % (variant, VARIANT_LABELS[variant]))
        print("=" * 60)

        var_dir   = _OUT / variant
        field_csv = var_dir / "field.csv"
        fem_done  = var_dir / "fem_done.flag"

        # ── Step 1: FEM (cached per theta variant) ────────────────────────
        if fem_done.exists() and field_csv.exists():
            print("[FEM] cached → skip  (%s)" % field_csv)
        else:
            theta = _load_theta(theta_path)
            a35   = theta[18] if len(theta) > 18 else float("nan")
            a45   = theta[19] if len(theta) > 19 else float("nan")
            print("[FEM] running  a35=%.3f  a45=%.3f  n_macro=%d …" % (
                a35, a45, args.n_macro))
            t0 = time.perf_counter()
            snaps_phi, mx, my, mz = _run_fem(
                theta, args.n_macro, args.n_react_sub, args.dt_h)
            _export_field_csv(snaps_phi, mx, my, mz, field_csv)
            fem_done.touch()
            print("[FEM] done in %.1fs  →  %s" % (
                time.perf_counter() - t0, field_csv))

        # ── Step 2A: A1 — E_max × E_min grid (n = 2 fixed) ──────────────
        a1_combos = list(product(E_MAX_VALS, E_MIN_VALS))
        print("\n[A1] E_max × E_min sweep  (%d jobs, n=%.0f) …" % (
            len(a1_combos), N_EXP_A1))

        for ei, (emax, emin) in enumerate(a1_combos):
            tag      = "a1_%02d" % ei
            job_name = "ms_%s_%s" % (variant[:5], tag)
            out_dir  = var_dir / tag
            label    = "E_max=%.0fG  E_min=%.1fG" % (emax / 1e9, emin / 1e9)
            print("  [%02d/%02d] %s" % (ei + 1, len(a1_combos), label), end="  ")

            rec = _run_one_abaqus(field_csv, job_name, emax, emin, N_EXP_A1, out_dir)
            if rec:
                results.append({
                    "sweep": "A1", "variant": variant, **rec,
                })

        # ── Step 2B: A2 — n_exp sweep (E fixed) ──────────────────────────
        print("\n[A2] n_exp sweep  (%d jobs, E_max=%.0fG  E_min=%.1fG) …" % (
            len(N_EXP_VALS), E_MAX_A2 / 1e9, E_MIN_A2 / 1e9))

        for ni, nexp in enumerate(N_EXP_VALS):
            tag      = "a2_%02d" % ni
            job_name = "ms_%s_%s" % (variant[:5], tag)
            out_dir  = var_dir / tag
            print("  [%d/%d] n=%.1f" % (ni + 1, len(N_EXP_VALS), nexp), end="  ")

            rec = _run_one_abaqus(field_csv, job_name, E_MAX_A2, E_MIN_A2, nexp, out_dir)
            if rec:
                results.append({
                    "sweep": "A2", "variant": variant, **rec,
                })

    # ── Save results CSV ─────────────────────────────────────────────────────
    if results:
        _save_results_csv(results, csv_path)
        print("\n[done] %d results saved → %s" % (len(results), csv_path))
    else:
        # Try to load existing results from done flags
        results = _collect_done_results(args.variants)
        if results:
            _save_results_csv(results, csv_path)
            print("\n[done] rebuilt %d results from done flags → %s" % (
                len(results), csv_path))
        else:
            print("\n[warn] no results collected.")
            return

    _plot_all(csv_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_all(csv_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        print("[plot] %s not found, skip" % csv_path)
        return

    rows = _load_results_csv(csv_path)
    if not rows:
        print("[plot] results.csv is empty, skip")
        return

    fig_dir = _OUT / "figures"
    fig_dir.mkdir(exist_ok=True)

    variants = [v for v in THETA_VARIANTS if any(r["variant"] == v for r in rows)]

    # ── Fig A1: E_max × E_min heatmap (2 rows × len(variants) cols) ─────────
    a1_rows = [r for r in rows if r["sweep"] == "A1"]
    if a1_rows:
        emax_u = sorted(set(r["e_max"] for r in a1_rows))
        emin_u = sorted(set(r["e_min"] for r in a1_rows))
        n_col  = len(variants)
        fig, axes = plt.subplots(
            2, n_col, figsize=(4.5 * n_col, 8), constrained_layout=True)
        if n_col == 1:
            axes = axes.reshape(2, 1)

        for vi, var in enumerate(variants):
            vrows = [r for r in a1_rows if r["variant"] == var]
            for si, (skey, slabel) in enumerate([
                ("substrate_smises", "Substrate S_Mises"),
                ("surface_smises",   "Surface S_Mises"),
            ]):
                ax = axes[si, vi]
                Z  = np.full((len(emin_u), len(emax_u)), np.nan)
                for r in vrows:
                    ix = emax_u.index(r["e_max"])
                    iy = emin_u.index(r["e_min"])
                    Z[iy, ix] = r[skey] / 1e6   # MPa

                im = ax.imshow(Z, aspect="auto", origin="lower",
                               cmap="RdYlGn_r", vmin=np.nanmin(Z), vmax=np.nanmax(Z))
                ax.set_xticks(range(len(emax_u)))
                ax.set_xticklabels(["%.0f" % (e / 1e9) for e in emax_u], fontsize=9)
                ax.set_yticks(range(len(emin_u)))
                ax.set_yticklabels(["%.1f" % (e / 1e9) for e in emin_u], fontsize=9)
                ax.set_xlabel("E_max (GPa)", fontsize=9)
                ax.set_ylabel("E_min (GPa)", fontsize=9)
                ax.set_title("%s\n%s (MPa)" % (VARIANT_LABELS[var], slabel), fontsize=9)
                plt.colorbar(im, ax=ax)

                for iy in range(len(emin_u)):
                    for ix in range(len(emax_u)):
                        v = Z[iy, ix]
                        if not np.isnan(v):
                            zmid = np.nanmean(Z)
                            ax.text(ix, iy, "%.1f" % v,
                                    ha="center", va="center", fontsize=7,
                                    color="white" if v > zmid else "black")

        fig.suptitle("A1: E_max × E_min Sensitivity  (n_exp = 2,  S_Mises in MPa)",
                     fontsize=13, fontweight="bold")
        out = fig_dir / "fig_A1_emax_emin_heatmap.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[plot] %s" % out.name)

    # ── Fig A2: n_exp bar chart ──────────────────────────────────────────────
    a2_rows = [r for r in rows if r["sweep"] == "A2"]
    if a2_rows:
        nexp_u = sorted(set(r["n_exp"] for r in a2_rows))
        x      = np.arange(len(nexp_u))
        width  = 0.25
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

        for si, (skey, slabel) in enumerate([
            ("substrate_smises", "Substrate"),
            ("surface_smises",   "Surface"),
        ]):
            ax = axes[si]
            for vi, var in enumerate(variants):
                vals = []
                for nexp in nexp_u:
                    matches = [r for r in a2_rows
                               if r["variant"] == var and abs(r["n_exp"] - nexp) < 0.01]
                    vals.append(matches[0][skey] / 1e6 if matches else np.nan)
                bars = ax.bar(x + vi * width, vals, width,
                              label=VARIANT_LABELS[var],
                              color=VARIANT_COLORS[var], alpha=0.85, edgecolor="k",
                              linewidth=0.5)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.5,
                                "%.1f" % v,
                                ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x + width)
            ax.set_xticklabels(["n = %.0f" % n for n in nexp_u])
            ax.set_ylabel("S_Mises (MPa)")
            ax.set_title("A2: %s  (E_max=10 GPa, E_min=0.5 GPa)" % slabel)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.35, linestyle="--")

        fig.suptitle("A2: DI Exponent n Sensitivity  (linear / quadratic / cubic)",
                     fontsize=13, fontweight="bold")
        out = fig_dir / "fig_A2_nexp_bars.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[plot] %s" % out.name)

    # ── Fig A3: θ variant comparison (at fixed E=10/0.5 GPa, n=2) ───────────
    # Use A1 data at (E_max=10 GPa, E_min=0.5 GPa, n=2)
    a3_rows = [
        r for r in rows
        if abs(r["e_max"] - 10.0e9) < 1e6
        and abs(r["e_min"] - 0.5e9) < 1e5
        and abs(r["n_exp"] - 2.0) < 0.01
    ]
    if a3_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        for si, (skey, slabel) in enumerate([
            ("substrate_smises", "Substrate"),
            ("surface_smises",   "Surface"),
        ]):
            ax = axes[si]
            vals  = []
            clrs  = []
            lbls  = []
            for var in variants:
                matches = [r for r in a3_rows if r["variant"] == var]
                vals.append(matches[0][skey] / 1e6 if matches else 0.0)
                clrs.append(VARIANT_COLORS[var])
                lbls.append(VARIANT_LABELS[var])
            bars = ax.bar(range(len(variants)), vals, color=clrs, alpha=0.85,
                          edgecolor="k", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        "%.1f MPa" % v,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels(lbls, rotation=12, ha="right", fontsize=9)
            ax.set_ylabel("S_Mises (MPa)")
            ax.set_title("A3: θ Comparison  –  %s" % slabel)
            ax.grid(axis="y", alpha=0.35, linestyle="--")

            # annotate a35 values
            a35_vals = {"dh_old": 21.4, "mild_weight": 3.56, "nolambda": 20.9}
            for i, var in enumerate(variants):
                ax.text(i, -ax.get_ylim()[1] * 0.07,
                        "a35=%.1f" % a35_vals.get(var, 0),
                        ha="center", va="top", fontsize=7, color="#555555")

        fig.suptitle(
            "A3: θ_MAP Variant Comparison\n"
            "(E_max=10 GPa, E_min=0.5 GPa, n=2)",
            fontsize=13, fontweight="bold",
        )
        out = fig_dir / "fig_A3_theta_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[plot] %s" % out.name)

    # ── Fig A_combined: 2×2 overview panel ───────────────────────────────────
    _plot_combined_overview(rows, variants, fig_dir)

    print("\n[plot] all figures → %s/" % fig_dir)


def _plot_combined_overview(
    rows: list[dict], variants: list[str], fig_dir: Path,
) -> None:
    """4-panel summary: A1 range bar + A2 bars + A3 bars (substrate + surface)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    a1_rows = [r for r in rows if r["sweep"] == "A1"]
    a2_rows = [r for r in rows if r["sweep"] == "A2"]

    for si, (skey, slabel) in enumerate([
        ("substrate_smises", "Substrate S_Mises (MPa)"),
        ("surface_smises",   "Surface S_Mises (MPa)"),
    ]):
        # ── left: A1 E-range (min/mean/max across E grid per variant) ────────
        ax = axes[si, 0]
        x  = np.arange(len(variants))
        for vi, var in enumerate(variants):
            vr = [r[skey] / 1e6 for r in a1_rows if r["variant"] == var]
            if not vr:
                continue
            vmin, vmean, vmax = min(vr), sum(vr) / len(vr), max(vr)
            ax.bar(vi, vmean, color=VARIANT_COLORS[var], alpha=0.8,
                   label=VARIANT_LABELS[var], edgecolor="k", linewidth=0.4)
            ax.errorbar(vi, vmean, yerr=[[vmean - vmin], [vmax - vmean]],
                        fmt="none", color="black", capsize=5, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in variants],
                           rotation=12, ha="right", fontsize=8)
        ax.set_ylabel(slabel)
        ax.set_title("A1: E-grid range (n=2)")
        ax.grid(axis="y", alpha=0.35, linestyle="--")

        # ── right: A2 n_exp comparison ────────────────────────────────────
        ax = axes[si, 1]
        if a2_rows:
            nexp_u = sorted(set(r["n_exp"] for r in a2_rows))
            xn     = np.arange(len(nexp_u))
            w      = 0.25
            for vi, var in enumerate(variants):
                vals = []
                for nexp in nexp_u:
                    matches = [r for r in a2_rows
                               if r["variant"] == var and abs(r["n_exp"] - nexp) < 0.01]
                    vals.append(matches[0][skey] / 1e6 if matches else np.nan)
                ax.bar(xn + vi * w, vals, w, label=VARIANT_LABELS[var],
                       color=VARIANT_COLORS[var], alpha=0.8,
                       edgecolor="k", linewidth=0.4)
            ax.set_xticks(xn + w)
            ax.set_xticklabels(["n=%.0f" % n for n in nexp_u])
            ax.set_ylabel(slabel)
            ax.set_title("A2: n_exp  (E_max=10 GPa, E_min=0.5 GPa)")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.35, linestyle="--")

    fig.suptitle(
        "Material Sensitivity Overview  (A1 E-range + A2 n_exp)\n"
        "Red = Old-dh  Green = Mild-weight  Blue = No-λ baseline",
        fontsize=12, fontweight="bold",
    )
    out = fig_dir / "fig_A_combined_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


if __name__ == "__main__":
    main()
