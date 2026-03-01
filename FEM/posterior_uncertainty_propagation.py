#!/usr/bin/env python3
"""
posterior_uncertainty_propagation.py
=====================================
C3: Sensitivity analysis and uncertainty propagation.

Propagates TMCMC posterior uncertainty through the full pipeline:
  theta (posterior samples) → Hamilton 2D ODE → DI field → E_eff → sigma (CI bands)

For N posterior samples per condition:
  1. Load posterior samples from TMCMC run
  2. For each sample, run 2D Hamilton+nutrient solver
  3. Compute DI field, E_eff field
  4. (Optional) Export to Abaqus for stress computation
  5. Aggregate: DI credible bands, E_eff credible bands
  6. Compute Sobol-like sensitivity indices (first-order)

Outputs:
  _uncertainty_propagation/{cond}/
    sample_{k:04d}/
      di_field.npy, phi_snaps.npy, theta.npy
    aggregated/
      di_mean.npy, di_p05.npy, di_p50.npy, di_p95.npy
      eeff_mean.npy, eeff_p05.npy, eeff_p95.npy
      sensitivity_indices.json
    figures/
      uncertainty_bands.png
      sensitivity_spider.png
      sobol_bar.png

Usage
-----
  python posterior_uncertainty_propagation.py
  python posterior_uncertainty_propagation.py --n-samples 50 --condition dh_baseline
  python posterior_uncertainty_propagation.py --plot-only  # replot from existing data
  python posterior_uncertainty_propagation.py --quick      # 5 samples, 10x10 grid
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    from material_models import compute_E_phi_pg, compute_E_virulence
except ImportError:
    compute_E_phi_pg = None
    compute_E_virulence = None

_TMCMC_ROOT = _HERE.parent
_RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
_OUT_BASE = _HERE / "_uncertainty_propagation"

CONDITION_RUNS = {
    "dh_baseline": _RUNS_ROOT / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "commensal_static",
    "commensal_hobic": _RUNS_ROOT / "commensal_hobic",
    "dysbiotic_static": _RUNS_ROOT / "dysbiotic_static",
}

E_MAX = 10.0e9
E_MIN = 0.5e9
DI_SCALE = 0.025778
DI_EXP = 2.0

PARAM_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]


def di_to_eeff(di):
    r = np.clip(di / DI_SCALE, 0, 1)
    return E_MAX * (1 - r) ** DI_EXP + E_MIN * r


def load_posterior_samples(run_dir, n_samples, seed=42):
    """Load N posterior samples from TMCMC run.

    Falls back to generating perturbations around theta_MAP if samples.npy
    is not available.
    """
    samples_path = run_dir / "samples.npy"
    if samples_path.exists():
        all_samples = np.load(samples_path)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_samples), size=min(n_samples, len(all_samples)), replace=False)
        return all_samples[idx]

    # Fallback: generate samples around theta_MAP
    theta_path = run_dir / "theta_MAP.json"
    if theta_path.exists():
        import json as _json

        with open(theta_path) as f:
            d = _json.load(f)
        if "theta_full" in d:
            theta_map = np.array(d["theta_full"], dtype=np.float64)
        elif "theta_sub" in d:
            theta_map = np.array(d["theta_sub"], dtype=np.float64)
        else:
            theta_map = np.array([d[k] for k in PARAM_NAMES], dtype=np.float64)
        rng = np.random.default_rng(seed)
        scale = np.maximum(np.abs(theta_map) * 0.1, 0.01)
        samples = theta_map[None, :] + rng.normal(0, scale, size=(n_samples, len(theta_map)))
        print(f"  [INFO] Generated {n_samples} perturbation samples around theta_MAP")
        return samples

    return None


def compute_di_from_phi(phi_final):
    """Compute DI from phi (5, Nx, Ny)."""
    phi_t = phi_final.transpose(1, 2, 0)
    phi_sum = phi_t.sum(axis=-1)
    phi_sum_safe = np.where(phi_sum > 0, phi_sum, 1.0)
    p = phi_t / phi_sum_safe[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=-1)
    return 1.0 - H / np.log(5.0)


def run_single_sample(theta, cfg_dict):
    """Run single forward model: theta → DI field."""
    from JAXFEM.core_hamilton_2d_nutrient import run_simulation, Config2D

    cfg = Config2D(**cfg_dict)
    result = run_simulation(theta, cfg)
    phi_snaps = np.array(result["phi_snaps"])
    c_snaps = np.array(result["c_snaps"])
    di = compute_di_from_phi(phi_snaps[-1])
    # phi_final: (5, Nx, Ny) -> (Nx, Ny, 5) for material_models API
    phi_final_nxy5 = phi_snaps[-1].transpose(1, 2, 0)
    e_phi_pg = compute_E_phi_pg(phi_final_nxy5)  # (Nx, Ny) [Pa]
    e_virulence = compute_E_virulence(phi_final_nxy5)  # (Nx, Ny) [Pa]
    return {
        "phi_snaps": phi_snaps,
        "c_snaps": c_snaps,
        "di": di,
        "e_phi_pg": e_phi_pg,
        "e_virulence": e_virulence,
        "di_mean": float(np.mean(di)),
        "di_max": float(np.max(di)),
        "phi_pg_max": float(np.max(phi_snaps[-1][4])),
    }


def run_condition(condition, args):
    """Run uncertainty propagation for one condition."""
    run_dir = CONDITION_RUNS.get(condition)
    if run_dir is None or not run_dir.exists():
        print(f"[SKIP] {condition}: run directory not found")
        return None

    out_dir = _OUT_BASE / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = out_dir / "aggregated"
    agg_dir.mkdir(exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load samples
    samples = load_posterior_samples(run_dir, args.n_samples, seed=args.seed)
    if samples is None:
        print(f"[SKIP] {condition}: no posterior samples found")
        return None

    n_actual = len(samples)
    print(f"\n{'='*60}")
    print(f"Uncertainty Propagation: {condition}")
    print(f"  Samples: {n_actual}")
    print(f"  Grid: {args.nx}x{args.ny}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    cfg_dict = {
        "Nx": args.nx,
        "Ny": args.ny,
        "n_macro": args.n_macro,
        "n_react_sub": args.n_react_sub,
        "dt_h": args.dt_h,
        "save_every": args.save_every,
        "K_hill": args.k_hill,
        "n_hill": args.n_hill,
    }

    # Run samples
    di_fields = []
    e_phi_pg_fields = []
    e_virulence_fields = []
    sample_metas = []
    t0 = time.perf_counter()

    for k in range(n_actual):
        sample_dir = out_dir / f"sample_{k:04d}"
        done_flag = sample_dir / "done.flag"

        if done_flag.exists() and not args.force:
            # Validate grid size matches
            meta_path = sample_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    cached_meta = json.load(f)
                cached_nx = cached_meta.get("nx", -1)
                cached_ny = cached_meta.get("ny", -1)
                if (cached_nx, cached_ny) != (-1, -1) and (cached_nx, cached_ny) != (
                    args.nx,
                    args.ny,
                ):
                    print(
                        f"  [{k+1}/{n_actual}] grid mismatch ({cached_nx}x{cached_ny} vs {args.nx}x{args.ny}), recomputing..."
                    )
                    import shutil

                    shutil.rmtree(sample_dir)
                    done_flag = sample_dir / "done.flag"  # reset

            # Load existing
            if done_flag.exists():
                di_path = sample_dir / "di_field.npy"
            else:
                di_path = None

            if di_path is not None and di_path.exists():
                di = np.load(di_path)
                di_fields.append(di)
                # Load or recompute E_phi_pg / E_virulence from phi_snaps
                epg_path = sample_dir / "e_phi_pg.npy"
                evir_path = sample_dir / "e_virulence.npy"
                if epg_path.exists() and evir_path.exists():
                    e_phi_pg_fields.append(np.load(epg_path))
                    e_virulence_fields.append(np.load(evir_path))
                else:
                    phi_path = sample_dir / "phi_snaps.npy"
                    if phi_path.exists():
                        phi_snaps = np.load(phi_path)
                        phi_final_nxy5 = phi_snaps[-1].transpose(1, 2, 0)
                        epg = compute_E_phi_pg(phi_final_nxy5)
                        evir = compute_E_virulence(phi_final_nxy5)
                        np.save(epg_path, epg)
                        np.save(evir_path, evir)
                        e_phi_pg_fields.append(epg)
                        e_virulence_fields.append(evir)
                    else:
                        # fallback: zeros (will not happen in practice)
                        e_phi_pg_fields.append(np.full_like(di, np.nan))
                        e_virulence_fields.append(np.full_like(di, np.nan))
                meta_path = sample_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        sample_metas.append(json.load(f))
                print(f"  [{k+1}/{n_actual}] loaded (cached)")
                continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        theta = samples[k]
        np.save(sample_dir / "theta.npy", theta)

        print(f"  [{k+1}/{n_actual}] running...", end="", flush=True)
        tk = time.perf_counter()

        try:
            result = run_single_sample(theta, cfg_dict)
            di_fields.append(result["di"])
            e_phi_pg_fields.append(result["e_phi_pg"])
            e_virulence_fields.append(result["e_virulence"])
            np.save(sample_dir / "di_field.npy", result["di"])
            np.save(sample_dir / "e_phi_pg.npy", result["e_phi_pg"])
            np.save(sample_dir / "e_virulence.npy", result["e_virulence"])
            np.save(sample_dir / "phi_snaps.npy", result["phi_snaps"])

            meta = {
                "di_mean": result["di_mean"],
                "di_max": result["di_max"],
                "phi_pg_max": result["phi_pg_max"],
                "timing_s": round(time.perf_counter() - tk, 1),
                "nx": args.nx,
                "ny": args.ny,
            }
            with (sample_dir / "meta.json").open("w") as f:
                json.dump(meta, f, indent=2)
            sample_metas.append(meta)
            done_flag.touch()

            print(f" DI={result['di_mean']:.4f}, {time.perf_counter()-tk:.1f}s")

            # Free JAX compilation cache to prevent memory leak
            try:
                import jax

                jax.clear_caches()
            except Exception:
                pass
            import gc

            gc.collect()

        except Exception as e:
            print(f" ERROR: {e}")
            continue

    if not di_fields:
        print("  No valid samples!")
        return None

    total_time = time.perf_counter() - t0

    # Aggregate
    print(f"\nAggregating {len(di_fields)} samples...")
    di_arr = np.array(di_fields)  # (N, Nx, Ny)
    di_mean = np.mean(di_arr, axis=0)
    di_p05 = np.percentile(di_arr, 5, axis=0)
    di_p50 = np.percentile(di_arr, 50, axis=0)
    di_p95 = np.percentile(di_arr, 95, axis=0)

    # E_eff credible bands (DI-based)
    eeff_arr = di_to_eeff(di_arr) * 1e-9  # GPa
    eeff_mean = np.mean(eeff_arr, axis=0)
    eeff_p05 = np.percentile(eeff_arr, 5, axis=0)
    eeff_p95 = np.percentile(eeff_arr, 95, axis=0)

    # E_phi_pg credible bands (Pg-based Hill sigmoid)
    epg_arr = np.array(e_phi_pg_fields)  # (N, Nx, Ny) [Pa]
    epg_arr_gpa = epg_arr * 1e-9  # NOT used; keep in Pa for this model
    epg_mean = np.mean(epg_arr, axis=0)
    epg_p05 = np.percentile(epg_arr, 5, axis=0)
    epg_p95 = np.percentile(epg_arr, 95, axis=0)

    # E_virulence credible bands (Pg+Fn weighted)
    evir_arr = np.array(e_virulence_fields)  # (N, Nx, Ny) [Pa]
    evir_mean = np.mean(evir_arr, axis=0)
    evir_p05 = np.percentile(evir_arr, 5, axis=0)
    evir_p95 = np.percentile(evir_arr, 95, axis=0)

    # Save aggregated
    np.save(agg_dir / "di_mean.npy", di_mean)
    np.save(agg_dir / "di_p05.npy", di_p05)
    np.save(agg_dir / "di_p50.npy", di_p50)
    np.save(agg_dir / "di_p95.npy", di_p95)
    np.save(agg_dir / "eeff_mean.npy", eeff_mean)
    np.save(agg_dir / "eeff_p05.npy", eeff_p05)
    np.save(agg_dir / "eeff_p95.npy", eeff_p95)
    np.save(agg_dir / "epg_mean.npy", epg_mean)
    np.save(agg_dir / "epg_p05.npy", epg_p05)
    np.save(agg_dir / "epg_p95.npy", epg_p95)
    np.save(agg_dir / "evir_mean.npy", evir_mean)
    np.save(agg_dir / "evir_p05.npy", evir_p05)
    np.save(agg_dir / "evir_p95.npy", evir_p95)

    # Sensitivity indices (first-order, variance-based)
    sensitivity = compute_sensitivity(samples[: len(di_fields)], di_arr)

    with (agg_dir / "sensitivity_indices.json").open("w") as f:
        json.dump(sensitivity, f, indent=2)

    # Summary
    summary = {
        "condition": condition,
        "n_samples": len(di_fields),
        "di_mean_global": float(np.mean(di_mean)),
        "di_ci_width": float(np.mean(di_p95 - di_p05)),
        "eeff_mean_gpa": float(np.mean(eeff_mean)),
        "eeff_ci_width_gpa": float(np.mean(eeff_p95 - eeff_p05)),
        "epg_mean_pa": float(np.mean(epg_mean)),
        "epg_ci_width_pa": float(np.mean(epg_p95 - epg_p05)),
        "evir_mean_pa": float(np.mean(evir_mean)),
        "evir_ci_width_pa": float(np.mean(evir_p95 - evir_p05)),
        "timing_s": round(total_time, 1),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Generate figures
    _plot_uncertainty_bands(di_arr, eeff_arr, epg_arr, evir_arr, fig_dir, condition)
    _plot_sensitivity(sensitivity, fig_dir, condition)

    print(f"\n  DI mean: {np.mean(di_mean):.6f}")
    print(f"  DI 90% CI width: {np.mean(di_p95 - di_p05):.6f}")
    print(f"  E_eff (DI) mean: {np.mean(eeff_mean):.4f} GPa")
    print(f"  E_eff (DI) 90% CI width: {np.mean(eeff_p95 - eeff_p05):.4f} GPa")
    print(f"  E_phi_pg mean: {np.mean(epg_mean):.1f} Pa")
    print(f"  E_phi_pg 90% CI width: {np.mean(epg_p95 - epg_p05):.1f} Pa")
    print(f"  E_virulence mean: {np.mean(evir_mean):.1f} Pa")
    print(f"  E_virulence 90% CI width: {np.mean(evir_p95 - evir_p05):.1f} Pa")
    print(f"  Total time: {total_time:.0f}s")

    return summary


def compute_sensitivity(theta_samples, di_arr):
    """Compute first-order Sobol-like sensitivity indices.

    For each parameter theta_i, compute correlation with DI field statistics.
    Uses Spearman rank correlation as a robust sensitivity measure.
    """
    from scipy.stats import spearmanr

    n_params = theta_samples.shape[1]
    di_means = np.mean(di_arr.reshape(len(di_arr), -1), axis=1)
    di_maxs = np.max(di_arr.reshape(len(di_arr), -1), axis=1)

    sensitivity = {}
    for i in range(min(n_params, 20)):
        name = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"theta_{i}"
        theta_i = theta_samples[:, i]

        rho_mean, p_mean = spearmanr(theta_i, di_means)
        rho_max, p_max = spearmanr(theta_i, di_maxs)

        sensitivity[name] = {
            "spearman_di_mean": float(rho_mean) if not np.isnan(rho_mean) else 0.0,
            "p_value_mean": float(p_mean) if not np.isnan(p_mean) else 1.0,
            "spearman_di_max": float(rho_max) if not np.isnan(rho_max) else 0.0,
            "p_value_max": float(p_max) if not np.isnan(p_max) else 1.0,
        }

    # Sort by absolute correlation
    sorted_params = sorted(
        sensitivity.items(), key=lambda x: abs(x[1]["spearman_di_mean"]), reverse=True
    )
    sensitivity["_ranking"] = [p[0] for p in sorted_params[:5]]

    return sensitivity


def _plot_uncertainty_bands(di_arr, eeff_arr, epg_arr, evir_arr, fig_dir, condition):
    """Plot DI, E_eff (DI), E_phi_pg, and E_virulence uncertainty bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Panel (a): DI mean with CI (spatial)
    ax = axes[0, 0]
    di_mean = np.mean(di_arr, axis=0)
    im = ax.imshow(di_mean.T, origin="lower", cmap="RdYlGn_r", aspect="equal")
    plt.colorbar(im, ax=ax, label="DI mean")
    ax.set_title("(a) DI Mean Field", fontsize=12)

    # Panel (b): DI CI width
    ax = axes[0, 1]
    di_ci = np.percentile(di_arr, 95, axis=0) - np.percentile(di_arr, 5, axis=0)
    im = ax.imshow(di_ci.T, origin="lower", cmap="hot", aspect="equal")
    plt.colorbar(im, ax=ax, label="DI 90% CI width")
    ax.set_title("(b) DI Uncertainty Width", fontsize=12)

    # Panel (c): DI histogram with CI
    ax = axes[1, 0]
    di_means = np.mean(di_arr.reshape(len(di_arr), -1), axis=1)
    ax.hist(di_means, bins=20, color="#2ca02c", alpha=0.7, density=True)
    ax.axvline(np.mean(di_means), color="k", ls="--", lw=2, label=f"mean={np.mean(di_means):.4f}")
    ax.axvline(
        np.percentile(di_means, 5), color="r", ls=":", label=f"5%={np.percentile(di_means, 5):.4f}"
    )
    ax.axvline(
        np.percentile(di_means, 95),
        color="r",
        ls=":",
        label=f"95%={np.percentile(di_means, 95):.4f}",
    )
    ax.set_xlabel("DI (spatial mean)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(c) DI Mean Distribution", fontsize=12)
    ax.legend(fontsize=8)

    # Panel (d): E_eff (DI-based) histogram with CI
    ax = axes[1, 1]
    eeff_means = np.mean(eeff_arr.reshape(len(eeff_arr), -1), axis=1)
    ax.hist(eeff_means, bins=20, color="#d62728", alpha=0.7, density=True)
    ax.axvline(
        np.mean(eeff_means), color="k", ls="--", lw=2, label=f"mean={np.mean(eeff_means):.2f} GPa"
    )
    ax.axvline(np.percentile(eeff_means, 5), color="r", ls=":")
    ax.axvline(np.percentile(eeff_means, 95), color="r", ls=":")
    ax.set_xlabel("$E_{eff}$ (DI) [GPa] (spatial mean)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(d) $E_{eff}$ (DI-based) Distribution", fontsize=12)
    ax.legend(fontsize=8)

    # Panel (e): E_phi_pg histogram with CI
    ax = axes[2, 0]
    epg_means = np.mean(epg_arr.reshape(len(epg_arr), -1), axis=1)
    ax.hist(epg_means, bins=20, color="#9467bd", alpha=0.7, density=True)
    ax.axvline(
        np.mean(epg_means), color="k", ls="--", lw=2, label=f"mean={np.mean(epg_means):.1f} Pa"
    )
    ax.axvline(
        np.percentile(epg_means, 5),
        color="r",
        ls=":",
        label=f"5%={np.percentile(epg_means, 5):.1f}",
    )
    ax.axvline(
        np.percentile(epg_means, 95),
        color="r",
        ls=":",
        label=f"95%={np.percentile(epg_means, 95):.1f}",
    )
    ax.set_xlabel("$E_{\\phi_{Pg}}$ [Pa] (spatial mean)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(e) $E_{\\phi_{Pg}}$ (Pg Hill) Distribution", fontsize=12)
    ax.legend(fontsize=8)

    # Panel (f): E_virulence histogram with CI
    ax = axes[2, 1]
    evir_means = np.mean(evir_arr.reshape(len(evir_arr), -1), axis=1)
    ax.hist(evir_means, bins=20, color="#8c564b", alpha=0.7, density=True)
    ax.axvline(
        np.mean(evir_means), color="k", ls="--", lw=2, label=f"mean={np.mean(evir_means):.1f} Pa"
    )
    ax.axvline(
        np.percentile(evir_means, 5),
        color="r",
        ls=":",
        label=f"5%={np.percentile(evir_means, 5):.1f}",
    )
    ax.axvline(
        np.percentile(evir_means, 95),
        color="r",
        ls=":",
        label=f"95%={np.percentile(evir_means, 95):.1f}",
    )
    ax.set_xlabel("$E_{vir}$ [Pa] (spatial mean)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(f) $E_{vir}$ (Pg+Fn) Distribution", fontsize=12)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Uncertainty Propagation: {condition} (N={len(di_arr)})", fontsize=14, weight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = fig_dir / "uncertainty_bands.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Uncertainty bands: {out}")


def _plot_sensitivity(sensitivity, fig_dir, condition):
    """Plot sensitivity indices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter out metadata keys
    params = {k: v for k, v in sensitivity.items() if not k.startswith("_")}
    names = list(params.keys())
    rhos = [abs(params[n]["spearman_di_mean"]) for n in names]
    p_vals = [params[n]["p_value_mean"] for n in names]

    # Sort by importance
    order = np.argsort(rhos)[::-1]
    names_sorted = [names[i] for i in order]
    rhos_sorted = [rhos[i] for i in order]
    p_sorted = [p_vals[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Bar chart of |rho|
    ax = axes[0]
    colors = ["#d62728" if p < 0.05 else "#888888" for p in p_sorted]
    ax.barh(range(len(names_sorted)), rhos_sorted, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("|Spearman $\\rho$|", fontsize=11)
    ax.set_title("(a) Parameter Sensitivity (DI mean)\nRed = p < 0.05", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Top-5 scatter plots
    ax = axes[1]
    ax.axis("off")
    top5 = sensitivity.get("_ranking", names_sorted[:5])
    lines = ["Top-5 most influential parameters:", "=" * 35, ""]
    for i, name in enumerate(top5):
        p = params.get(name, {})
        rho = p.get("spearman_di_mean", 0)
        pv = p.get("p_value_mean", 1)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        lines.append(f"  {i+1}. {name:5s}: rho={rho:+.3f} (p={pv:.4f}) {sig}")
    lines.extend(["", "Significance: *** p<0.001, ** p<0.01, * p<0.05"])
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        va="top",
    )

    fig.suptitle(f"Sensitivity Analysis: {condition}", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = fig_dir / "sensitivity_spider.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Sensitivity: {out}")


def main():
    ap = argparse.ArgumentParser(description="Posterior uncertainty propagation")
    ap.add_argument("--conditions", nargs="+", default=["dh_baseline"])
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute even if cached")
    ap.add_argument("--quick", action="store_true")
    # Simulation
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--n-macro", type=int, default=60)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=60)
    ap.add_argument("--k-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    args = ap.parse_args()

    if args.quick:
        args.n_samples = 5
        args.nx = 10
        args.ny = 10
        args.n_macro = 10
        args.n_react_sub = 5
        args.save_every = 10

    print("=" * 60)
    print("Posterior Uncertainty Propagation")
    print(f"  Conditions: {args.conditions}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Grid: {args.nx}x{args.ny}")
    print("=" * 60)

    results = {}
    for cond in args.conditions:
        results[cond] = run_condition(cond, args)

    # Cross-condition summary
    valid = {k: v for k, v in results.items() if v}
    if valid:
        print(f"\n{'='*60}")
        print("Cross-condition summary:")
        for cond, r in valid.items():
            print(f"  {cond}:")
            print(f"    DI mean: {r['di_mean_global']:.6f} " f"(CI width: {r['di_ci_width']:.6f})")
            print(
                f"    E_phi_pg: {r['epg_mean_pa']:.1f} Pa "
                f"(CI width: {r['epg_ci_width_pa']:.1f})"
            )
            print(
                f"    E_virulence: {r['evir_mean_pa']:.1f} Pa "
                f"(CI width: {r['evir_ci_width_pa']:.1f})"
            )
        print(f"{'='*60}")

    if len(valid) >= 2:
        _plot_cross_condition(valid, _OUT_BASE)

    # Save final master summary
    master = {k: v for k, v in valid.items()}
    with (_OUT_BASE / "master_summary.json").open("w") as f:
        json.dump(master, f, indent=2)
    print(f"\nMaster summary: {_OUT_BASE / 'master_summary.json'}")


def _plot_cross_condition(results, out_base):
    """Cross-condition CI comparison figure (publication quality)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    n = len(conds)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    x = np.arange(n)

    # (a) DI mean + CI
    ax = axes[0]
    means = [results[c]["di_mean_global"] for c in conds]
    widths = [results[c]["di_ci_width"] for c in conds]
    ax.bar(
        x,
        means,
        color=[colors[i % len(colors)] for i in range(n)],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.errorbar(
        x,
        means,
        yerr=[w / 2 for w in widths],
        fmt="none",
        ecolor="k",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("DI (mean)", fontsize=11)
    ax.set_title("(a) Dysbiosis Index", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) E_phi_pg + CI
    ax = axes[1]
    means = [results[c]["epg_mean_pa"] for c in conds]
    widths = [results[c]["epg_ci_width_pa"] for c in conds]
    ax.bar(
        x,
        means,
        color=[colors[i % len(colors)] for i in range(n)],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.errorbar(
        x,
        means,
        yerr=[w / 2 for w in widths],
        fmt="none",
        ecolor="k",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("$E_{\\phi_{Pg}}$ [Pa]", fontsize=11)
    ax.set_title("(b) Young's Modulus (Pg Hill)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) E_virulence + CI
    ax = axes[2]
    means = [results[c]["evir_mean_pa"] for c in conds]
    widths = [results[c]["evir_ci_width_pa"] for c in conds]
    ax.bar(
        x,
        means,
        color=[colors[i % len(colors)] for i in range(n)],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.errorbar(
        x,
        means,
        yerr=[w / 2 for w in widths],
        fmt="none",
        ecolor="k",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("$E_{vir}$ [Pa]", fontsize=11)
    ax.set_title("(c) Young's Modulus (Pg+Fn)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Cross-Condition Uncertainty Comparison (90% CI)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_base / "cross_condition_ci_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nCross-condition comparison: {out}")


if __name__ == "__main__":
    main()
