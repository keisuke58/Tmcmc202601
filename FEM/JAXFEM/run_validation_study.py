#!/usr/bin/env python3
"""
run_validation_study.py — Comprehensive validation of staggered coupling
=========================================================================

Runs 4 validation studies for the quasi-static staggered solver:

  1. Mesh convergence:   15×15, 25×25, 50×50, 75×75 (fixed DH condition)
  2. Time-step convergence: dt_growth = 0.05, 0.1, 0.2, 0.5
  3. σ_crit sensitivity: 0, 50, 100, 200, 500 Pa
  4. Posterior UQ:        50 posterior θ samples → E, σ_vm CI bands

Generates a single multi-panel validation figure + JSON summary.

Usage:
    python run_validation_study.py --study mesh       # mesh convergence only
    python run_validation_study.py --study all         # everything
    python run_validation_study.py --study uq --n-samples 50
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TMCMC = _HERE.parent.parent
_RUNS = _TMCMC / "data_5species" / "_runs"


def _run_single(
    condition, nx, ny, dt_growth, n_steps, sigma_crit, alpha_max, outdir, extra_args=None
):
    """Run staggered solver as subprocess, return npz path."""
    cmd = [
        sys.executable,
        str(_HERE / "run_coupled_staggered.py"),
        "--condition",
        condition,
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--dt-h",
        "1e-5",
        "--ode-init-steps",
        "2500",
        "--ode-adjust-steps",
        "100",
        "--dt-growth",
        str(dt_growth),
        "--n-growth-steps",
        str(n_steps),
        "--k-alpha",
        "0.05",
        "--e-model",
        "phi_pg",
        "--sigma-crit",
        str(sigma_crit),
        "--stress-type",
        "plane_stress",
        "--nutrient-bc",
        "mixed",
        "--alpha-max",
        str(alpha_max),
        "--outdir",
        str(outdir),
        "--save-npz",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  FAILED: {' '.join(cmd[:6])}")
        print(proc.stderr[-500:] if proc.stderr else "no stderr")
        return None
    npz = Path(outdir) / f"coupled_snaps_{condition}.npz"
    return npz if npz.exists() else None


def _load_final(npz_path):
    """Load final-step quantities from npz."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        "E_mean": float(d["E"][-1].mean()),
        "E_min": float(d["E"][-1].min()),
        "E_max": float(d["E"][-1].max()),
        "sigma_vm_max": float(d["sigma_vm_max"][-1]),
        "sigma_vm_mean": float(d["sigma_vm_mean"][-1]),
        "u_max": float(d["u_max"][-1]),
        "alpha_max": float(d["alpha"][-1].max()),
        "DI_min": float(d["DI"][-1].min()),
        "DI_max": float(d["DI"][-1].max()),
        "geom_nonlin": float(d["geom_nonlin"][-1]) if "geom_nonlin" in d else 0.0,
    }


# ============================================================================
# Study 1: Mesh convergence
# ============================================================================


def study_mesh_convergence(condition="dh_baseline", outdir=None):
    """Run mesh refinement: 15, 25, 50, 75 nodes."""
    print("\n" + "=" * 60)
    print("  STUDY 1: Mesh Convergence")
    print("=" * 60)

    meshes = [15, 25, 50, 75]
    results = {}
    for n in meshes:
        tag = f"mesh_{n}x{n}"
        sub_dir = outdir / tag
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  --- {n}×{n} ---")
        t0 = time.perf_counter()
        npz = _run_single(condition, n, n, 0.1, 50, 100, 0.3, sub_dir)
        elapsed = time.perf_counter() - t0
        if npz:
            r = _load_final(npz)
            r["elapsed_s"] = round(elapsed, 1)
            r["nx"] = n
            results[tag] = r
            print(
                f"    E={r['E_mean']:.1f} Pa, σ_vm={r['sigma_vm_max']:.1f} Pa, "
                f"gnl={r['geom_nonlin']:.4f}, t={elapsed:.1f}s"
            )
        else:
            print(f"    FAILED")

    return results


# ============================================================================
# Study 2: Time-step convergence
# ============================================================================


def study_timestep_convergence(condition="dh_baseline", outdir=None):
    """Run dt_growth refinement: 0.05, 0.1, 0.2, 0.5."""
    print("\n" + "=" * 60)
    print("  STUDY 2: Time-Step Convergence")
    print("=" * 60)

    dt_values = [0.05, 0.1, 0.2, 0.5]
    # Keep total growth time = 5.0 hours for all
    results = {}
    for dt in dt_values:
        n_steps = int(round(5.0 / dt))
        tag = f"dt_{dt:.3f}"
        sub_dir = outdir / tag
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  --- dt={dt}, n_steps={n_steps} ---")
        t0 = time.perf_counter()
        npz = _run_single(condition, 25, 25, dt, n_steps, 100, 0.3, sub_dir)
        elapsed = time.perf_counter() - t0
        if npz:
            r = _load_final(npz)
            r["elapsed_s"] = round(elapsed, 1)
            r["dt_growth"] = dt
            r["n_steps"] = n_steps
            results[tag] = r
            print(
                f"    E={r['E_mean']:.1f} Pa, σ_vm={r['sigma_vm_max']:.1f} Pa, "
                f"α_max={r['alpha_max']:.4f}, t={elapsed:.1f}s"
            )

    return results


# ============================================================================
# Study 3: σ_crit sensitivity
# ============================================================================


def study_sigma_crit_sensitivity(condition="dh_baseline", outdir=None):
    """Run σ_crit = 0, 50, 100, 200, 500 Pa."""
    print("\n" + "=" * 60)
    print("  STUDY 3: σ_crit Sensitivity")
    print("=" * 60)

    sigma_values = [0, 50, 100, 200, 500]
    results = {}
    for sc in sigma_values:
        tag = f"sigma_crit_{sc}"
        sub_dir = outdir / tag
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  --- σ_crit={sc} Pa ---")
        t0 = time.perf_counter()
        npz = _run_single(condition, 25, 25, 0.1, 50, sc, 0.3, sub_dir)
        elapsed = time.perf_counter() - t0
        if npz:
            r = _load_final(npz)
            r["elapsed_s"] = round(elapsed, 1)
            r["sigma_crit"] = sc
            results[tag] = r
            print(
                f"    E={r['E_mean']:.1f} Pa, σ_vm={r['sigma_vm_max']:.1f} Pa, "
                f"α_max={r['alpha_max']:.4f}"
            )

    return results


# ============================================================================
# Study 4: Posterior UQ propagation
# ============================================================================


def study_posterior_uq(condition="dh_baseline", outdir=None, n_samples=50):
    """Run posterior θ samples through staggered solver, compute CI bands."""
    print("\n" + "=" * 60)
    print(f"  STUDY 4: Posterior UQ ({n_samples} samples)")
    print("=" * 60)

    # Find posterior samples
    posterior_dirs = sorted(_RUNS.glob(f"*{condition}*posterior*")) + sorted(
        _RUNS.glob(f"*{condition}*1000p*")
    )
    samples = []
    for pd in posterior_dirs:
        chain_files = sorted(pd.glob("chain_*.json"))
        for cf in chain_files:
            with open(cf) as f:
                chain = json.load(f)
            if isinstance(chain, list):
                for s in chain:
                    theta = s.get("theta_full", s.get("theta_sub", s.get("theta")))
                    if theta and len(theta) >= 20:
                        samples.append(np.array(theta[:20], dtype=np.float64))

    if not samples:
        # Try loading from posterior npz
        for pd in sorted(_RUNS.glob(f"*{condition.replace('_', '*')}*")):
            for npz_f in pd.glob("posterior_samples*.npz"):
                d = np.load(npz_f)
                if "samples" in d:
                    for s in d["samples"]:
                        if len(s) >= 20:
                            samples.append(np.array(s[:20], dtype=np.float64))

    if not samples:
        # Fallback: load MAP and perturb
        map_path = _RUNS / condition / "theta_MAP.json"
        if map_path.exists():
            with open(map_path) as f:
                d = json.load(f)
            theta_map = np.array(d.get("theta_full", d.get("theta_sub")), dtype=np.float64)
            rng = np.random.RandomState(42)
            for _ in range(n_samples):
                # 10% Gaussian perturbation
                noise = rng.normal(0, 0.1, 20) * np.abs(theta_map[:20])
                samples.append(theta_map[:20] + noise)
            print(f"  Using MAP ± 10% perturbation ({n_samples} samples)")
        else:
            print("  ERROR: No posterior samples found")
            return {}

    # Subsample if too many
    if len(samples) > n_samples:
        idx = np.linspace(0, len(samples) - 1, n_samples, dtype=int)
        samples = [samples[i] for i in idx]

    print(f"  Using {len(samples)} posterior samples")

    # Save theta samples temporarily
    sub_dir = outdir / "uq_samples"
    sub_dir.mkdir(parents=True, exist_ok=True)

    results_list = []
    for i, theta in enumerate(samples):
        # Write theta to temp file
        theta_tmp = sub_dir / f"theta_{i:03d}.json"
        with open(theta_tmp, "w") as f:
            json.dump({"theta_full": theta.tolist()}, f)

        out_tmp = sub_dir / f"out_{i:03d}"
        out_tmp.mkdir(exist_ok=True)

        # Run solver with --theta-json (direct path, no condition lookup)
        cmd = [
            sys.executable,
            str(_HERE / "run_coupled_staggered.py"),
            "--condition",
            f"uq_sample_{i:03d}",
            "--theta-json",
            str(theta_tmp),
            "--nx",
            "25",
            "--ny",
            "25",
            "--dt-h",
            "1e-5",
            "--ode-init-steps",
            "2500",
            "--ode-adjust-steps",
            "100",
            "--dt-growth",
            "0.1",
            "--n-growth-steps",
            "50",
            "--k-alpha",
            "0.05",
            "--e-model",
            "phi_pg",
            "--sigma-crit",
            "100",
            "--stress-type",
            "plane_stress",
            "--nutrient-bc",
            "mixed",
            "--alpha-max",
            "0.3",
            "--outdir",
            str(out_tmp),
            "--save-npz",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Find the npz file
        npz_files = list(out_tmp.glob("*.npz"))
        if npz_files:
            r = _load_final(npz_files[0])
            results_list.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i+1}/{len(samples)}] E={r['E_mean']:.1f} Pa, "
                    f"σ_vm={r['sigma_vm_max']:.1f} Pa"
                )
        else:
            if proc.returncode != 0:
                print(f"  [{i+1}/{len(samples)}] FAILED: {proc.stderr[-200:]}")

    if not results_list:
        print("  ERROR: No UQ results")
        return {}

    # Compute statistics
    E_vals = np.array([r["E_mean"] for r in results_list])
    svm_vals = np.array([r["sigma_vm_max"] for r in results_list])
    u_vals = np.array([r["u_max"] for r in results_list])
    di_vals = np.array([r["DI_max"] for r in results_list])

    uq_summary = {
        "n_samples": len(results_list),
        "E_mean": float(E_vals.mean()),
        "E_std": float(E_vals.std()),
        "E_ci_5": float(np.percentile(E_vals, 5)),
        "E_ci_95": float(np.percentile(E_vals, 95)),
        "sigma_vm_max_mean": float(svm_vals.mean()),
        "sigma_vm_max_std": float(svm_vals.std()),
        "sigma_vm_ci_5": float(np.percentile(svm_vals, 5)),
        "sigma_vm_ci_95": float(np.percentile(svm_vals, 95)),
        "u_max_mean": float(u_vals.mean()),
        "u_max_ci_5": float(np.percentile(u_vals, 5)),
        "u_max_ci_95": float(np.percentile(u_vals, 95)),
        "DI_max_mean": float(di_vals.mean()),
        "DI_max_ci_5": float(np.percentile(di_vals, 5)),
        "DI_max_ci_95": float(np.percentile(di_vals, 95)),
        "all_E": E_vals.tolist(),
        "all_sigma_vm": svm_vals.tolist(),
    }

    print(f"\n  UQ Summary:")
    print(
        f"    E:     {uq_summary['E_mean']:.1f} ± {uq_summary['E_std']:.1f} Pa "
        f"[{uq_summary['E_ci_5']:.1f}, {uq_summary['E_ci_95']:.1f}]"
    )
    print(
        f"    σ_vm:  {uq_summary['sigma_vm_max_mean']:.1f} ± {uq_summary['sigma_vm_max_std']:.1f} Pa "
        f"[{uq_summary['sigma_vm_ci_5']:.1f}, {uq_summary['sigma_vm_ci_95']:.1f}]"
    )

    return uq_summary


# ============================================================================
# Comprehensive validation figure
# ============================================================================


def plot_validation(mesh_results, dt_results, sigma_results, uq_results, outdir):
    """Generate 2×3 validation summary figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Mesh convergence: E vs mesh size
    if mesh_results:
        ns = [r["nx"] for r in mesh_results.values()]
        Es = [r["E_mean"] for r in mesh_results.values()]
        svms = [r["sigma_vm_max"] for r in mesh_results.values()]
        gnls = [r["geom_nonlin"] for r in mesh_results.values()]

        ax = axes[0, 0]
        ax.plot(ns, Es, "bo-", lw=2, ms=8)
        ax.set_xlabel("Mesh size N×N")
        ax.set_ylabel("E_mean [Pa]")
        ax.set_title("(a) Mesh convergence: E")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(ns, svms, "ro-", lw=2, ms=8)
        ax.set_xlabel("Mesh size N×N")
        ax.set_ylabel("σ_vm,max [Pa]")
        ax.set_title("(b) Mesh convergence: σ_vm")
        ax.grid(True, alpha=0.3)

        # Geometric nonlinearity
        ax = axes[0, 2]
        ax.bar(range(len(ns)), gnls, tick_label=[str(n) for n in ns], color="orange")
        ax.axhline(0.05, ls="--", color="red", label="5% threshold")
        ax.set_xlabel("Mesh N")
        ax.set_ylabel("|∇u|/|ε| ratio")
        ax.set_title("(c) Geometric nonlinearity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # (d) Time-step convergence
    if dt_results:
        dts = [r["dt_growth"] for r in dt_results.values()]
        Es_dt = [r["E_mean"] for r in dt_results.values()]
        alphas_dt = [r["alpha_max"] for r in dt_results.values()]

        ax = axes[1, 0]
        ax.plot(dts, Es_dt, "go-", lw=2, ms=8)
        ax.set_xlabel("dt_growth [hours]")
        ax.set_ylabel("E_mean [Pa]")
        ax.set_title("(d) Time-step convergence")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # (e) σ_crit sensitivity
    if sigma_results:
        scs = [r["sigma_crit"] for r in sigma_results.values()]
        alphas_sc = [r["alpha_max"] for r in sigma_results.values()]
        svms_sc = [r["sigma_vm_max"] for r in sigma_results.values()]

        ax = axes[1, 1]
        ax2 = ax.twinx()
        l1 = ax.plot(scs, alphas_sc, "bs-", lw=2, ms=8, label="α_max")
        l2 = ax2.plot(scs, svms_sc, "r^--", lw=2, ms=8, label="σ_vm,max")
        ax.set_xlabel("σ_crit [Pa]")
        ax.set_ylabel("α_max", color="blue")
        ax2.set_ylabel("σ_vm,max [Pa]", color="red")
        ax.set_title("(e) σ_crit sensitivity")
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="center right")
        ax.grid(True, alpha=0.3)

    # (f) Posterior UQ
    if uq_results and "all_E" in uq_results:
        ax = axes[1, 2]
        E_all = uq_results["all_E"]
        ax.hist(E_all, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(uq_results["E_ci_5"], ls="--", color="red", lw=2, label="5%/95% CI")
        ax.axvline(uq_results["E_ci_95"], ls="--", color="red", lw=2)
        ax.axvline(uq_results["E_mean"], ls="-", color="black", lw=2, label="mean")
        ax.set_xlabel("E_mean [Pa]")
        ax.set_ylabel("Count")
        ax.set_title(f"(f) Posterior UQ (N={uq_results['n_samples']})")
        ax.legend()

    fig.suptitle("Staggered Coupling Validation Study (v3)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = outdir / "validation_study.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\n  Validation figure: {outpath}")
    return str(outpath)


# ============================================================================
# CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="Staggered coupling validation study")
    ap.add_argument(
        "--study",
        default="all",
        choices=["all", "mesh", "dt", "sigma", "uq"],
        help="Which study to run",
    )
    ap.add_argument("--condition", default="dh_baseline")
    ap.add_argument(
        "--n-samples", type=int, default=50, help="Number of posterior samples for UQ study"
    )
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else _TMCMC / "FEM" / "figures" / "validation_study"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mesh_results = {}
    dt_results = {}
    sigma_results = {}
    uq_results = {}

    if args.study in ("all", "mesh"):
        mesh_results = study_mesh_convergence(args.condition, outdir)

    if args.study in ("all", "dt"):
        dt_results = study_timestep_convergence(args.condition, outdir)

    if args.study in ("all", "sigma"):
        sigma_results = study_sigma_crit_sensitivity(args.condition, outdir)

    if args.study in ("all", "uq"):
        uq_results = study_posterior_uq(args.condition, outdir, args.n_samples)

    elapsed = time.perf_counter() - t0

    # Save JSON summary
    summary = {
        "condition": args.condition,
        "total_elapsed_s": round(elapsed, 1),
        "mesh_convergence": mesh_results,
        "timestep_convergence": dt_results,
        "sigma_crit_sensitivity": sigma_results,
        "posterior_uq": {k: v for k, v in uq_results.items() if k not in ("all_E", "all_sigma_vm")},
    }
    summary_path = outdir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    # Plot
    plot_validation(mesh_results, dt_results, sigma_results, uq_results, outdir)

    print(f"\n  Total validation time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
