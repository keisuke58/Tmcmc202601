#!/usr/bin/env python3
"""
run_posterior_abaqus_ensemble.py
================================
Posterior Abaqus ensemble for stress credible bands.

For each condition and each posterior sample:
  1. Build theta from TMCMC samples.npy
  2. Run 3D FEM  →  phi_pg / DI field at final snapshot
  3. Export field CSV
  4. abaqus cae  →  global-scale power-law model  →  ODB
  5. abaqus python  →  extract substrate + surface S_Mises
  6. Save per-sample stress.json  (resume-safe via done.flag)

Aggregate: stress_all.npy  (n_valid, 2)  [substrate, surface]
           stress_p{05,50,95}.npy  (2,)

Usage
-----
  # Full run (20 samples per condition):
  python run_posterior_abaqus_ensemble.py

  # Quick test (3 samples, 20 FEM macro steps):
  python run_posterior_abaqus_ensemble.py --n-samples 3 --n-macro 20

  # Plot only (from existing outputs):
  python run_posterior_abaqus_ensemble.py --plot-only
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths and global parameters
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_DATA_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"

CONDITION_RUNS: dict = {
    "dh_baseline": _DATA_ROOT / "sweep_pg_20260217_081459" / "dh_baseline",
    "commensal_static": _DATA_ROOT / "Commensal_Static_20260208_002100",
    "commensal_hobic": _DATA_ROOT / "Commensal_HOBIC_20260208_002100",
    "dysbiotic_static": _DATA_ROOT / "Dysbiotic_Static_20260207_203752",
}
COND_SHORT = {
    "dh_baseline": "dh",
    "commensal_static": "cs",
    "commensal_hobic": "ch",
    "dysbiotic_static": "ds",
}

GLOBAL_DI_SCALE = 0.025778  # 1.1 * max(DI_cs_max=0.02344)
E_MAX = 10.0e9
E_MIN = 0.5e9
DI_EXPONENT = 2.0
N_BINS = 20

_OUT_BASE = _HERE / "_posterior_abaqus"


# ---------------------------------------------------------------------------
# FEM helpers (imported at runtime to avoid Numba warm-up at import time)
# ---------------------------------------------------------------------------


def _run_fem(
    theta: np.ndarray, n_macro: int, n_react_sub: int, dt_h: float, condition: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run FEM3DBiofilm and return (snaps_phi, mesh_x) at last snapshot."""
    import fem_3d_extension as f3d

    sim = f3d.FEM3DBiofilm(
        theta,
        Nx=15,
        Ny=15,
        Nz=15,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        n_macro=n_macro,
        n_react_sub=n_react_sub,
        dt_h=dt_h,
        save_every=n_macro,  # only save final snapshot (+ t=0)
        condition=condition,
    )
    snaps_phi, snaps_t = sim.run()
    return snaps_phi, sim.x_mesh, sim.y_mesh, sim.z_mesh


def _export_field_csv(
    snaps_phi: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, out_csv: Path
) -> None:
    """Export phi_pg and DI at the last FEM snapshot to a 3D CSV."""
    phi_last = snaps_phi[-1]  # (5, Nx, Ny, Nz)
    phi_nodes = phi_last.transpose(1, 2, 3, 0)  # (Nx, Ny, Nz, 5)
    phi_sum = phi_nodes.sum(axis=-1)
    phi_sum_safe = np.where(phi_sum > 0, phi_sum, 1.0)
    p = phi_nodes / phi_sum_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=-1)
    di = 1.0 - H / np.log(5.0)

    phi_pg = phi_nodes[..., 4]
    Nx, Ny, Nz = phi_pg.shape
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("x,y,z,phi_pg,di\n")
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    f.write(
                        "%.8e,%.8e,%.8e,%.8e,%.8e\n"
                        % (
                            x[ix],
                            y[iy],
                            z[iz],
                            float(phi_pg[ix, iy, iz]),
                            float(di[ix, iy, iz]),
                        )
                    )


def _run_abaqus_cae(field_csv: Path, job_name: str) -> int:
    """Run Abaqus CAE with power-law mapping. Return return code."""
    cmd = [
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
        "--job-name",
        job_name,
    ]
    ret = subprocess.run(cmd, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return ret.returncode


def _extract_stress(job_name: str, stress_csv: Path) -> dict | None:
    """Run compare_biofilm_abaqus.py via abaqus python and parse the output."""
    odb_path = _HERE / ("%s.odb" % job_name)
    if not odb_path.exists():
        print("  [warn] ODB not found: %s" % odb_path)
        return None
    cmd = [
        "abaqus",
        "python",
        str(_HERE / "compare_biofilm_abaqus.py"),
        str(stress_csv),
        str(odb_path),
    ]
    ret = subprocess.run(cmd, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ret.returncode != 0 or not stress_csv.exists():
        print("  [warn] stress extraction failed (rc=%d)" % ret.returncode)
        return None
    # parse CSV: odb,label,depth_frac,x,y,z,Ux,Uy,Uz,S11,S22,S33,S12,S13,S23,S_Mises
    substrate = surface = None
    with stress_csv.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 16 or parts[0].startswith("odb"):
                continue
            try:
                depth_frac = float(parts[2])
                smises = float(parts[15])
            except ValueError:
                continue
            if abs(depth_frac - 0.0) < 1e-6:
                substrate = smises
            elif abs(depth_frac - 1.0) < 1e-6:
                surface = smises
    if substrate is None or surface is None:
        print("  [warn] could not parse substrate/surface from %s" % stress_csv)
        return None
    return {"substrate_smises": substrate, "surface_smises": surface}


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------


def run_sample(
    enum_k: int,
    theta: np.ndarray,
    cond: str,
    sample_dir: Path,
    n_macro: int,
    n_react_sub: int,
    dt_h: float,
) -> dict | None:
    """Complete FEM + Abaqus pipeline for one posterior sample. Resume-safe."""
    done_flag = sample_dir / "done.flag"
    stress_json = sample_dir / "stress.json"
    field_csv = sample_dir / "field.csv"
    stress_csv = sample_dir / "stress_raw.csv"
    cshort = COND_SHORT[cond]
    job_name = "BiofilmDemo3DJob_%s_%04d" % (cshort, enum_k)

    # ── resume: already fully done ──────────────────────────────────────────
    if done_flag.exists() and stress_json.exists():
        with stress_json.open() as f:
            d = json.load(f)
        print(
            "  [resume] sample %04d  sub=%.3g  surf=%.3g"
            % (enum_k, d["substrate_smises"], d["surface_smises"])
        )
        return d

    sample_dir.mkdir(parents=True, exist_ok=True)
    t_sample = time.perf_counter()
    print("\n  === sample %04d ===" % enum_k)

    # ── Step 1: FEM ──────────────────────────────────────────────────────────
    fem_done = sample_dir / "fem_done.flag"
    if fem_done.exists() and field_csv.exists():
        print("  [resume] FEM already done, skipping FEM step.")
    else:
        print("  [FEM] running 3D FEM …")
        np.save(sample_dir / "theta.npy", theta)
        t0 = time.perf_counter()
        snaps_phi, mesh_x, mesh_y, mesh_z = _run_fem(theta, n_macro, n_react_sub, dt_h, cond)
        print("  [FEM] done in %.1fs" % (time.perf_counter() - t0))

        # ── Step 2: export field CSV ─────────────────────────────────────────
        _export_field_csv(snaps_phi, mesh_x, mesh_y, mesh_z, field_csv)
        print("  [export] wrote %s" % field_csv.name)
        fem_done.touch()

    # ── Step 3: Abaqus CAE ───────────────────────────────────────────────────
    odb_path = _HERE / ("%s.odb" % job_name)
    abaqus_done = sample_dir / "abaqus_done.flag"
    if abaqus_done.exists() and odb_path.exists():
        print("  [resume] Abaqus already done, skipping CAE step.")
    else:
        print("  [Abaqus] building model & running job %s …" % job_name)
        t0 = time.perf_counter()
        rc = _run_abaqus_cae(field_csv, job_name)
        if rc != 0:
            print("  [warn] Abaqus CAE exited with rc=%d, skipping sample." % rc)
            return None
        print("  [Abaqus] done in %.1fs" % (time.perf_counter() - t0))
        abaqus_done.touch()

    # ── Step 4: extract stress ───────────────────────────────────────────────
    print("  [stress] extracting from ODB …")
    stress = _extract_stress(job_name, stress_csv)
    if stress is None:
        return None

    # ── save & done ─────────────────────────────────────────────────────────
    with stress_json.open("w") as f:
        json.dump(stress, f, indent=2)
    done_flag.touch()

    print(
        "  [done] sample %04d  sub=%.4g Pa  surf=%.4g Pa  (%.1fs total)"
        % (
            enum_k,
            stress["substrate_smises"],
            stress["surface_smises"],
            time.perf_counter() - t_sample,
        )
    )
    return stress


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(out_dir: Path) -> dict:
    """Load all stress.json files and compute percentile bands."""
    subs, surfs = [], []
    for d in sorted(out_dir.glob("sample_*/stress.json")):
        with d.open() as f:
            s = json.load(f)
        subs.append(s["substrate_smises"])
        surfs.append(s["surface_smises"])
    if not subs:
        return {}
    sub_arr = np.array(subs)
    surf_arr = np.array(surfs)
    all_arr = np.stack([sub_arr, surf_arr], axis=1)  # (n, 2)
    np.save(out_dir / "stress_all.npy", all_arr)
    for pct, tag in [(5, "p05"), (50, "p50"), (95, "p95")]:
        np.save(out_dir / ("stress_%s.npy" % tag), np.percentile(all_arr, pct, axis=0))
    meta = {
        "n_valid": len(subs),
        "substrate_p05": float(np.percentile(sub_arr, 5)),
        "substrate_p50": float(np.percentile(sub_arr, 50)),
        "substrate_p95": float(np.percentile(sub_arr, 95)),
        "surface_p05": float(np.percentile(surf_arr, 5)),
        "surface_p50": float(np.percentile(surf_arr, 50)),
        "surface_p95": float(np.percentile(surf_arr, 95)),
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ensemble(results: dict[str, dict], out_path: Path) -> None:
    """Bar chart with posterior credible bands for substrate and surface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    colors = {
        "dh_baseline": "#d62728",
        "commensal_static": "#1f77b4",
        "commensal_hobic": "#2ca02c",
        "dysbiotic_static": "#ff7f0e",
    }
    labels = {
        "dh_baseline": "DH Baseline",
        "commensal_static": "Commensal Static",
        "commensal_hobic": "Commensal HOBIC",
        "dysbiotic_static": "Dysbiotic Static",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, (depth_key, depth_label) in zip(
        axes, [("substrate", "Substrate (x=0)"), ("surface", "Surface (x=1)")]
    ):
        x = np.arange(len(conds))
        for k, cond in enumerate(conds):
            m = results[cond]
            p50 = m["%s_p50" % depth_key]
            p05 = m["%s_p05" % depth_key]
            p95 = m["%s_p95" % depth_key]
            bar = ax.bar(k, p50 / 1e6, color=colors[cond], alpha=0.8, label=labels[cond], width=0.5)
            ax.errorbar(
                k,
                p50 / 1e6,
                yerr=[[(p50 - p05) / 1e6], [(p95 - p50) / 1e6]],
                fmt="none",
                color="black",
                capsize=6,
                linewidth=1.5,
            )
            ax.text(
                k, p95 / 1e6 + 0.008, "%.3f" % (p50 / 1e6), ha="center", va="bottom", fontsize=8
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in conds], fontsize=8, rotation=15, ha="right")
        ax.set_title(depth_label, fontsize=10)
        ax.set_ylabel("von Mises stress [MPa]")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Posterior Abaqus ensemble  –  von Mises stress\n"
        "(N=%d samples, global DI scale s=%.4f, 5th–95th pct.)"
        % (results[conds[0]]["n_valid"], GLOBAL_DI_SCALE),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot →", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Posterior Abaqus stress ensemble")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-macro", type=int, default=100)
    ap.add_argument("--n-react-sub", type=int, default=50)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITION_RUNS.keys()))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=_OUT_BASE)
    args = ap.parse_args()

    ensemble_results = {}

    for cond in args.conditions:
        if cond not in CONDITION_RUNS:
            print("[warn] unknown condition %r, skipping" % cond)
            continue
        run_dir = CONDITION_RUNS[cond]
        out_dir = args.out_dir / cond
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n%s" % ("=" * 65))
        print("Condition: %s" % cond)
        print("TMCMC run: %s" % run_dir)
        print("Output   : %s" % out_dir)
        print("=" * 65)

        # ── load posterior samples ───────────────────────────────────────────
        with (run_dir / "theta_MAP.json").open() as f:
            theta_map_data = json.load(f)
        samples = np.load(run_dir / "samples.npy")
        active_indices = theta_map_data.get("active_indices", list(range(20)))
        theta_template = np.array(theta_map_data["theta_full"], dtype=np.float64)
        n_total = samples.shape[0]
        n_use = min(args.n_samples, n_total)
        rng = np.random.default_rng(args.seed)
        indices = (
            np.arange(n_total)
            if n_total == n_use
            else rng.choice(n_total, size=n_use, replace=False)
        )

        meta_path = out_dir / "run_meta.json"
        with meta_path.open("w") as f:
            json.dump(
                {
                    "condition": cond,
                    "n_samples": int(n_use),
                    "n_macro": args.n_macro,
                    "seed": args.seed,
                    "global_di_scale": GLOBAL_DI_SCALE,
                    "sample_indices": indices.tolist(),
                },
                f,
                indent=2,
            )

        if not args.plot_only:
            t_cond = time.perf_counter()
            for enum_k, idx in enumerate(indices):
                theta_s = samples[idx]
                theta = theta_template.copy()
                if theta_s.shape[0] == len(active_indices):
                    theta[active_indices] = theta_s
                elif theta_s.shape[0] == 20:
                    theta = theta_s.astype(np.float64)
                else:
                    print("  [skip] unexpected shape %s" % str(theta_s.shape))
                    continue

                sample_dir = out_dir / ("sample_%04d" % enum_k)
                run_sample(
                    enum_k, theta, cond, sample_dir, args.n_macro, args.n_react_sub, args.dt_h
                )

            print("\n[cond done] %s  total %.1fm" % (cond, (time.perf_counter() - t_cond) / 60))

        # ── aggregate ────────────────────────────────────────────────────────
        meta = aggregate(out_dir)
        if meta:
            ensemble_results[cond] = meta
            print("\n[%s] posterior stress:" % cond)
            print(
                "  substrate: p50=%.4g  [%.4g, %.4g] Pa"
                % (meta["substrate_p50"], meta["substrate_p05"], meta["substrate_p95"])
            )
            print(
                "  surface  : p50=%.4g  [%.4g, %.4g] Pa"
                % (meta["surface_p50"], meta["surface_p05"], meta["surface_p95"])
            )

    # ── plot ─────────────────────────────────────────────────────────────────
    if len(ensemble_results) >= 1:
        out_plot = _HERE / "_posterior_abaqus" / "stress_posterior_bands.png"
        plot_ensemble(ensemble_results, out_plot)

    print("\n[pipeline complete]  Output: %s" % args.out_dir)


if __name__ == "__main__":
    main()
