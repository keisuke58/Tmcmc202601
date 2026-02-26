#!/usr/bin/env python3
"""
aggregate_di_credible.py
========================
B1: DI場の空間的信頼区間（Posterior-uncertainty in the DI field）

後期アンサンブルの各サンプルに保存済みの field.csv を読み込み、
各ノードで DI の p5/p50/p95 を計算する。
その信頼帯フィールドを Abaqus に渡して、θ不確実性起因の
応力空間分布の不確実性を定量化する。

Input  (already computed by run_posterior_abaqus_ensemble.py):
  _posterior_abaqus/{cond}/sample_XXXX/field.csv
    columns: x, y, z, phi_pg, di

Output:
  _di_credible/{cond}/
    di_stack.npy          (n_samples, n_nodes)   raw DI matrix
    phi_pg_stack.npy      (n_samples, n_nodes)
    coords.npy            (n_nodes, 3)
    p{05,50,95}_field.csv  Abaqus-ready field CSV
    fig_di_spatial_ci.png  4-panel spatial uncertainty map
    fig_di_depth_profile.png  depth profile with credible band
  _di_credible/
    fig_di_cross_condition.png  4-condition comparison (depth profiles)
    stress_from_di_ci.csv       Abaqus stress for p05/p50/p95 per condition
    fig_stress_di_uncertainty.png  stress band from DI uncertainty vs θ ensemble

Usage:
  python aggregate_di_credible.py                # all 4 conditions + Abaqus
  python aggregate_di_credible.py --no-abaqus    # fields + plots only (no Abaqus)
  python aggregate_di_credible.py --plot-only    # re-plot from existing outputs
  python aggregate_di_credible.py --conditions commensal_static dh_baseline
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PA_BASE = _HERE / "_posterior_abaqus"  # posterior ensemble root
_OUT_BASE = _HERE / "_di_credible"

CONDITIONS = [
    "dh_baseline",
    "commensal_static",
    "commensal_hobic",
    "dysbiotic_static",
]

COND_LABELS = {
    "dh_baseline": "dh-baseline (dysbiotic Hill)",
    "commensal_static": "Commensal Static",
    "commensal_hobic": "Commensal HOBIC",
    "dysbiotic_static": "Dysbiotic Static",
}

COND_COLORS = {
    "dh_baseline": "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
}

QUANTILES = [0.05, 0.50, 0.95]
QTAGS = ["p05", "p50", "p95"]

# Abaqus settings (same as posterior ensemble)
GLOBAL_DI_SCALE: float = 0.025778
E_MAX: float = 10.0e9
E_MIN: float = 0.5e9
DI_EXPONENT: float = 2.0
N_BINS: int = 20
NU: float = 0.30

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_field_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coords (N,3), phi_pg (N,), di (N,)."""
    coords, phi_pg, di = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
            phi_pg.append(float(row["phi_pg"]))
            di.append(float(row["di"]))
    return np.array(coords), np.array(phi_pg), np.array(di)


def _write_field_csv(
    coords: np.ndarray,
    phi_pg: np.ndarray,
    di: np.ndarray,
    out: Path,
) -> None:
    """Write Abaqus-compatible field CSV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("x,y,z,phi_pg,di\n")
        for i in range(len(di)):
            f.write(
                "%.8e,%.8e,%.8e,%.8e,%.8e\n"
                % (
                    coords[i, 0],
                    coords[i, 1],
                    coords[i, 2],
                    float(phi_pg[i]),
                    float(di[i]),
                )
            )


# ---------------------------------------------------------------------------
# Stack loader
# ---------------------------------------------------------------------------


def _load_condition_stack(cond: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load all sample field.csv for a condition.
    Returns (coords, phi_pg_stack, di_stack) or None if not enough samples.
    coords        : (N_nodes, 3)
    phi_pg_stack  : (N_samples, N_nodes)
    di_stack      : (N_samples, N_nodes)
    """
    cond_dir = _PA_BASE / cond
    if not cond_dir.exists():
        print("  [skip] no posterior dir: %s" % cond_dir)
        return None

    sample_dirs = sorted(cond_dir.glob("sample_*"))
    valid = [d for d in sample_dirs if (d / "field.csv").exists()]
    if len(valid) < 3:
        print("  [skip] only %d valid samples in %s" % (len(valid), cond))
        return None

    print("  loading %d samples …" % len(valid))
    coords_ref = None
    phi_list, di_list = [], []

    for sd in valid:
        coords, phi_pg, di = _load_field_csv(sd / "field.csv")
        if coords_ref is None:
            coords_ref = coords
        phi_list.append(phi_pg)
        di_list.append(di)

    return coords_ref, np.array(phi_list), np.array(di_list)


def _compute_pg_depth_samples(coords: np.ndarray, phi_stack: np.ndarray) -> np.ndarray:
    x = coords[:, 0]
    n_samples = phi_stack.shape[0]
    depths = np.full(n_samples, np.nan, dtype=float)
    for k in range(n_samples):
        phi = phi_stack[k]
        total = float(phi.sum())
        if total <= 0.0:
            continue
        w = phi / total
        depths[k] = float((w * x).sum())
    return depths


# ---------------------------------------------------------------------------
# Abaqus interface
# ---------------------------------------------------------------------------


def _run_abaqus_on_field(
    field_csv: Path,
    job_name: str,
    stress_csv: Path,
) -> dict | None:
    """Run Abaqus CAE + stress extraction. Returns stress dict or None."""
    # CAE
    cmd_cae = [
        "abaqus",
        "cae",
        "noGUI=%s" % str(_HERE / "abaqus_biofilm_demo_3d.py"),
        "--",
        "--field-csv",
        str(field_csv),
        "--mapping",
        "power",
        "--di-scale",
        "%.6f" % GLOBAL_DI_SCALE,
        "--n-bins",
        str(N_BINS),
        "--e-max",
        "%.6g" % E_MAX,
        "--e-min",
        "%.6g" % E_MIN,
        "--di-exponent",
        "%.2f" % DI_EXPONENT,
        "--nu",
        "%.3f" % NU,
        "--job-name",
        job_name,
    ]
    rc = subprocess.run(
        cmd_cae,
        cwd=str(_HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).returncode
    if rc != 0:
        print("    [warn] Abaqus CAE rc=%d" % rc)
        return None

    odb = _HERE / ("%s.odb" % job_name)
    if not odb.exists():
        print("    [warn] ODB not found: %s" % odb)
        return None

    # Extract stress
    cmd_ext = [
        "abaqus",
        "python",
        str(_HERE / "compare_biofilm_abaqus.py"),
        str(stress_csv),
        str(odb),
    ]
    rc2 = subprocess.run(
        cmd_ext,
        cwd=str(_HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).returncode
    if rc2 != 0 or not stress_csv.exists():
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
    return {"substrate_smises": substrate, "surface_smises": surface}


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------


def process_condition(cond: str, run_abaqus: bool) -> dict | None:
    """
    Full B1 pipeline for one condition.
    Returns dict with Abaqus stress results (or None if Abaqus skipped).
    """
    print("\n[%s]" % cond)
    out_dir = _OUT_BASE / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load or reuse stack ────────────────────────────────────────────
    stack_path = out_dir / "di_stack.npy"
    coords_path = out_dir / "coords.npy"
    phi_path = out_dir / "phi_pg_stack.npy"

    if stack_path.exists() and coords_path.exists() and phi_path.exists():
        print("  [cache] loading stacks …")
        coords = np.load(coords_path)
        phi_stack = np.load(phi_path)
        di_stack = np.load(stack_path)
    else:
        result = _load_condition_stack(cond)
        if result is None:
            return None
        coords, phi_stack, di_stack = result
        np.save(coords_path, coords)
        np.save(phi_path, phi_stack)
        np.save(stack_path, di_stack)
        print("  stacks saved  shape=%s" % str(di_stack.shape))

    n_samples, n_nodes = di_stack.shape
    print("  n_samples=%d  n_nodes=%d" % (n_samples, n_nodes))

    pg_depth = _compute_pg_depth_samples(coords, phi_stack)
    np.save(out_dir / "pg_depth_samples.npy", pg_depth)

    # ── 2. Compute quantiles at each node ────────────────────────────────
    di_q = np.quantile(di_stack, QUANTILES, axis=0)  # (3, N_nodes)
    phi_q = np.quantile(phi_stack, QUANTILES, axis=0)
    np.save(out_dir / "di_quantiles.npy", di_q)

    print(
        "  DI quantiles: p05=%.4f  p50=%.4f  p95=%.4f"
        % (di_q[0].mean(), di_q[1].mean(), di_q[2].mean())
    )

    # ── 3. Write quantile field CSVs ──────────────────────────────────────
    for qi, qtag in enumerate(QTAGS):
        fcsv = out_dir / ("%s_field.csv" % qtag)
        if not fcsv.exists():
            _write_field_csv(coords, phi_q[qi], di_q[qi], fcsv)
            print("  wrote %s" % fcsv.name)

    # ── 4. Visualise spatial DI uncertainty ───────────────────────────────
    _plot_spatial_uncertainty(cond, coords, di_q, out_dir)
    _plot_depth_profile(cond, coords, di_q, di_stack, out_dir)

    # ── 5. Run Abaqus on p05/p50/p95 (optional) ─────────────────────────
    stress_by_q = {}
    if run_abaqus:
        short = cond[:4]
        for qi, qtag in enumerate(QTAGS):
            fcsv = out_dir / ("%s_field.csv" % qtag)
            job_name = "dici_%s_%s" % (short, qtag)
            done_flag = out_dir / ("abaqus_%s_done.flag" % qtag)
            scsv = out_dir / ("stress_raw_%s.csv" % qtag)
            sjson = out_dir / ("stress_%s.json" % qtag)

            if done_flag.exists() and sjson.exists():
                with sjson.open() as f:
                    stress_by_q[qtag] = json.load(f)
                print("  [resume] Abaqus %s" % qtag)
                continue

            print("  [Abaqus] %s  (DI_mean=%.4f) …" % (qtag, di_q[qi].mean()), end=" ")
            t0 = time.perf_counter()
            stress = _run_abaqus_on_field(fcsv, job_name, scsv)
            if stress:
                with sjson.open("w") as f:
                    json.dump(stress, f, indent=2)
                done_flag.touch()
                stress_by_q[qtag] = stress
                print(
                    "done in %.1fs  sub=%.3g  surf=%.3g"
                    % (
                        time.perf_counter() - t0,
                        stress["substrate_smises"],
                        stress["surface_smises"],
                    )
                )
            else:
                print("FAILED")

    return stress_by_q if stress_by_q else None


# ---------------------------------------------------------------------------
# Plotting: per-condition
# ---------------------------------------------------------------------------


def _plot_spatial_uncertainty(
    cond: str,
    coords: np.ndarray,
    di_q: np.ndarray,
    out_dir: Path,
) -> None:
    """4-panel: XY slice at z_mid for p05/p50/p95/range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reconstruct grid dimensions from coords
    x_u = np.unique(coords[:, 0])
    y_u = np.unique(coords[:, 1])
    z_u = np.unique(coords[:, 2])
    Nx, Ny, Nz = len(x_u), len(y_u), len(z_u)

    def _to_grid(vals: np.ndarray) -> np.ndarray:
        return vals.reshape(Nx, Ny, Nz)

    iz_mid = Nz // 2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = [
        (di_q[0], "p05 DI"),
        (di_q[1], "p50 DI"),
        (di_q[2], "p95 DI"),
        (di_q[2] - di_q[0], "p95 − p05  (range)"),
    ]
    vmin_di = di_q[0].min()
    vmax_di = di_q[2].max()

    for ax, (vals, title) in zip(axes, panels):
        grid = _to_grid(vals)[:, :, iz_mid]
        cmap = "RdYlGn_r" if "range" not in title else "Oranges"
        vmin = vals.min() if "range" in title else vmin_di
        vmax = vals.max() if "range" in title else vmax_di
        im = ax.imshow(
            grid.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[x_u[0], x_u[-1], y_u[0], y_u[-1]],
        )
        ax.set_xlabel("x (depth)", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "B1: DI Spatial Credible Interval  –  %s  (z=mid slice)" % (COND_LABELS.get(cond, cond)),
        fontsize=11,
        fontweight="bold",
    )
    out = out_dir / "fig_di_spatial_ci.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  [plot] %s" % out.name)


def _plot_depth_profile(
    cond: str,
    coords: np.ndarray,
    di_q: np.ndarray,
    di_stack: np.ndarray,
    out_dir: Path,
) -> None:
    """DI depth profile (x-axis) with p5/p50/p95 band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_u = np.unique(coords[:, 0])
    Nx = len(x_u)
    n_nodes = di_stack.shape[1]
    Ny = Nz = int(round((n_nodes / Nx) ** 0.5))

    def _depth_mean(vals: np.ndarray) -> np.ndarray:
        """Average over y,z for each depth x."""
        g = vals.reshape(Nx, Ny, Nz)
        return g.mean(axis=(1, 2))

    p05_d = _depth_mean(di_q[0])
    p50_d = _depth_mean(di_q[1])
    p95_d = _depth_mean(di_q[2])

    # also compute across-sample IQR for comparison
    sample_depth = np.array([_depth_mean(s) for s in di_stack])  # (N_s, Nx)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    col = COND_COLORS.get(cond, "steelblue")

    ax.fill_between(x_u, p05_d, p95_d, alpha=0.25, color=col, label="p05–p95 (θ uncertainty)")
    ax.plot(x_u, p50_d, color=col, lw=2, label="p50 median")
    ax.plot(x_u, sample_depth.min(axis=0), color=col, lw=0.7, ls="--", alpha=0.5)
    ax.plot(
        x_u, sample_depth.max(axis=0), color=col, lw=0.7, ls="--", alpha=0.5, label="sample min/max"
    )

    ax.set_xlabel("Depth (x, normalised)", fontsize=10)
    ax.set_ylabel("Dysbiotic Index (DI)", fontsize=10)
    ax.set_title(
        "B1: DI Depth Profile with θ Credible Band  –  %s" % (COND_LABELS.get(cond, cond)),
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    out = out_dir / "fig_di_depth_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  [plot] %s" % out.name)


# ---------------------------------------------------------------------------
# Cross-condition comparison
# ---------------------------------------------------------------------------


def _plot_cross_condition(conditions: list[str]) -> None:
    """Overlay depth profiles for all 4 conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for cond in conditions:
        out_dir = _OUT_BASE / cond
        stack_p = out_dir / "di_stack.npy"
        coords_p = out_dir / "coords.npy"
        q_path = out_dir / "di_quantiles.npy"
        if not (stack_p.exists() and q_path.exists()):
            continue

        coords = np.load(coords_p)
        di_q = np.load(q_path)  # (3, N_nodes)
        di_stk = np.load(stack_p)  # (N_s, N_nodes)

        x_u = np.unique(coords[:, 0])
        Nx = len(x_u)
        n_nodes = di_stk.shape[1]
        Ny = Nz = int(round((n_nodes / Nx) ** 0.5))

        def _dm(v):
            return v.reshape(Nx, Ny, Nz).mean(axis=(1, 2))

        col = COND_COLORS.get(cond, "gray")
        lab = COND_LABELS.get(cond, cond)

        for ax, (qi, qtag) in zip(axes, [(1, "Median (p50)"), (None, "p05–p95 band")]):
            if qi is not None:
                ax.plot(x_u, _dm(di_q[qi]), color=col, lw=2, label=lab)
                ax.fill_between(x_u, _dm(di_q[0]), _dm(di_q[2]), alpha=0.15, color=col)
            else:
                ax.fill_between(x_u, _dm(di_q[0]), _dm(di_q[2]), alpha=0.25, color=col, label=lab)
                ax.plot(x_u, _dm(di_q[1]), color=col, lw=1.5, ls="--")

    for ax, title in zip(axes, ["Median DI depth profile", "p05–p95 band width"]):
        ax.set_xlabel("Depth x (normalised)", fontsize=10)
        ax.set_ylabel("Dysbiotic Index (DI)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(
        "B1: Cross-Condition DI Depth Profiles + θ Credible Bands", fontsize=12, fontweight="bold"
    )
    out = _OUT_BASE / "fig_di_cross_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


def _plot_pg_depth_cross_condition(conditions: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for cond in conditions:
        out_dir = _OUT_BASE / cond
        depth_p = out_dir / "pg_depth_samples.npy"
        if not depth_p.exists():
            continue
        depths = np.load(depth_p)
        depths = depths[~np.isnan(depths)]
        if depths.size == 0:
            continue
        col = COND_COLORS.get(cond, "gray")
        lab = COND_LABELS.get(cond, cond)
        q05, q50, q95 = np.quantile(depths, QUANTILES)
        ax.errorbar(
            [lab],
            [q50],
            yerr=[[q50 - q05], [q95 - q50]],
            fmt="o",
            color=col,
            capsize=6,
            lw=1.5,
        )

    ax.set_ylabel("P. gingivalis depth (x, centre-of-mass)", fontsize=10)
    ax.set_title("Pg depth at t_final – posterior samples", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")
    out = _OUT_BASE / "fig_pg_depth_cross_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


def _plot_stress_di_uncertainty(conditions: list[str]) -> None:
    """
    Compare:
    - θ-ensemble stress band (from stress_all.npy)
    - DI-credible stress band (from p05/p50/p95 Abaqus runs)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(conditions), figsize=(4.5 * len(conditions), 8), constrained_layout=True
    )
    if len(conditions) == 1:
        axes = axes.reshape(2, 1)

    for ci, cond in enumerate(conditions):
        out_dir = _OUT_BASE / cond
        pa_dir = _PA_BASE / cond

        for si, (skey, slabel) in enumerate(
            [
                ("substrate_smises", "Substrate"),
                ("surface_smises", "Surface"),
            ]
        ):
            ax = axes[si, ci]
            col = COND_COLORS.get(cond, "gray")

            # θ-ensemble credible interval
            stress_all_p = pa_dir / "stress_all.npy"
            if stress_all_p.exists():
                sa = np.load(stress_all_p)  # (N_valid, 2)
                idx = 0 if "substrate" in skey else 1
                theta_vals = sa[:, idx] / 1e6
                ax.bar(0, theta_vals.mean(), color=col, alpha=0.7, label="θ-ensemble mean")
                ax.errorbar(
                    0,
                    theta_vals.mean(),
                    yerr=[
                        [theta_vals.mean() - np.percentile(theta_vals, 5)],
                        [np.percentile(theta_vals, 95) - theta_vals.mean()],
                    ],
                    fmt="none",
                    color="black",
                    capsize=8,
                    lw=1.5,
                    label="θ p05–p95",
                )

            # DI-credible interval stress
            di_vals = []
            for qi, qtag in enumerate(QTAGS):
                sj = out_dir / ("stress_%s.json" % qtag)
                if sj.exists():
                    with sj.open() as f:
                        d = json.load(f)
                    di_vals.append(d[skey] / 1e6)

            if len(di_vals) == 3:
                ax.bar(1, di_vals[1], color="orange", alpha=0.7, label="DI-CI p50")
                ax.errorbar(
                    1,
                    di_vals[1],
                    yerr=[[abs(di_vals[1] - di_vals[0])], [abs(di_vals[2] - di_vals[1])]],
                    fmt="none",
                    color="darkorange",
                    capsize=8,
                    lw=1.5,
                    label="DI p05–p95",
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["θ uncertainty", "DI-field CI"], fontsize=8)
            ax.set_ylabel("S_Mises (MPa)", fontsize=9)
            ax.set_title("%s\n%s" % (COND_LABELS.get(cond, cond), slabel), fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "B1: theta-Ensemble Stress  vs  DI-Field Credible Stress\n"
        "(Biological uncertainty vs Field uncertainty)",
        fontsize=12,
        fontweight="bold",
    )
    out = _OUT_BASE / "fig_stress_di_uncertainty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITIONS,
        choices=CONDITIONS,
        help="Which conditions to process (default: all 4)",
    )
    ap.add_argument("--no-abaqus", action="store_true", help="Skip Abaqus runs (field + plot only)")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip everything, re-draw figures from saved stacks",
    )
    args = ap.parse_args()

    _OUT_BASE.mkdir(parents=True, exist_ok=True)
    run_abaqus = not (args.no_abaqus or args.plot_only)

    if not args.plot_only:
        # ── Per-condition pipeline ─────────────────────────────────────────
        for cond in args.conditions:
            process_condition(cond, run_abaqus=run_abaqus)

    # ── Cross-condition plots ──────────────────────────────────────────────
    _plot_cross_condition(args.conditions)
    _plot_pg_depth_cross_condition(args.conditions)

    if run_abaqus or args.plot_only:
        _plot_stress_di_uncertainty(args.conditions)

    print("\n[done]  Output: %s" % _OUT_BASE)


if __name__ == "__main__":
    main()
