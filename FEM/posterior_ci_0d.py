#!/usr/bin/env python3
"""
posterior_ci_0d.py
==================
0D-based posterior uncertainty propagation.

For each condition:
  1. Load posterior samples (theta) from TMCMC run
  2. Run 0D Hamilton ODE for each sample → DI_0D, φ_final, E(DI), E(φ_Pg), E(Vir)
  3. Basin filter: discard samples that jump to wrong ODE attractor
  4. Compute 90% CI for DI_0D and all E models
  5. Save summary + per-sample results

This is the CORRECT approach: 0D ODE captures condition-level differences (DI_0D),
unlike 2D PDE which homogenizes species via diffusion (DI_2D ≈ 0.05 for all).

Important: DI_SCALE=1.0 for 0D DI (not 0.026 which was calibrated for 2D DI).

Usage:
  python posterior_ci_0d.py                     # all conditions, 50 samples
  python posterior_ci_0d.py --n-samples 200     # more samples
  python posterior_ci_0d.py --conditions dh_baseline dysbiotic_static
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_RUNS_ROOT = _HERE.parent / "data_5species" / "_runs"
_OUT_BASE = _HERE / "_ci_0d_results"

CONDITION_RUNS = {
    "dh_baseline": _RUNS_ROOT / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "commensal_static",
    "commensal_hobic": _RUNS_ROOT / "commensal_hobic",
    "dysbiotic_static": _RUNS_ROOT / "dysbiotic_static",
}

PARAM_NAMES = [
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "a12",
    "a13",
    "a14",
    "a21",
    "a23",
    "a24",
    "a31",
    "a32",
    "a34",
    "a41",
    "a42",
    "a43",
    "a51",
    "a52",
    "a53",
    "a54",
]

ALL_CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]

# Basin filter threshold: samples with |DI - DI_MAP| > this are discarded
# (ODE has multiple attractors; small perturbation can flip basins)
BASIN_THRESHOLD = 0.25

# Use di_scale=1.0 for 0D DI (the default 0.026 was calibrated for 2D DI fields)
DI_SCALE_0D = 1.0


def load_posterior_samples(run_dir, n_samples, seed=42):
    """Load posterior samples from TMCMC run directory.

    Returns (samples, is_real_posterior).
    """
    samples_path = run_dir / "samples.npy"
    if samples_path.exists():
        all_samples = np.load(samples_path)
        rng = np.random.default_rng(seed)
        if len(all_samples) > n_samples:
            idx = rng.choice(len(all_samples), n_samples, replace=False)
            return all_samples[idx], True
        return all_samples, True

    # Fallback: perturb around MAP
    for jf in run_dir.glob("theta_MAP*.json"):
        with open(jf) as f:
            d = json.load(f)
        if "theta_full" in d:
            theta_map = np.array(d["theta_full"], dtype=np.float64)
        elif "theta_sub" in d:
            theta_map = np.array(d["theta_sub"], dtype=np.float64)
        else:
            theta_map = np.array([d[k] for k in PARAM_NAMES], dtype=np.float64)
        rng = np.random.default_rng(seed)
        # Small perturbation (2%) — Hamilton ODE has multiple attractors;
        # 10% noise causes bimodal DI (jumps between commensal/dysbiotic basins).
        scale = np.maximum(np.abs(theta_map) * 0.02, 0.002)
        samples = theta_map[None, :] + rng.normal(0, scale, size=(n_samples, len(theta_map)))
        samples = np.clip(samples, 0, None)
        # Prepend the exact MAP as sample 0
        samples = np.vstack([theta_map[None, :], samples])
        print(f"  Generated {n_samples} perturbation samples around MAP (2% noise)")
        return samples, False

    return None, False


def load_theta_map(run_dir):
    """Load MAP theta from run directory."""
    for jf in run_dir.glob("theta_MAP*.json"):
        with open(jf) as f:
            d = json.load(f)
        if "theta_full" in d:
            return np.array(d["theta_full"], dtype=np.float64)
        elif "theta_sub" in d:
            return np.array(d["theta_sub"], dtype=np.float64)
    return None


def solve_0d_single(theta_np, n_steps=2500, dt=0.01):
    """Run 0D Hamilton ODE for a single theta → DI, phi, E values."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from JAXFEM.core_hamilton_1d import theta_to_matrices, newton_step, make_initial_state

    theta_jax = jnp.array(theta_np, dtype=jnp.float64)
    A, b_diag = theta_to_matrices(theta_jax)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5, dtype=jnp.float64),
        "EtaPhi": jnp.ones(5, dtype=jnp.float64),
        "c": 100.0,
        "alpha": 100.0,
        "K_hill": jnp.array(0.05, dtype=jnp.float64),
        "n_hill": jnp.array(4.0, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
        "newton_steps": 6,
    }

    g0 = make_initial_state(1, active_mask)[0]

    # Use Python loop instead of lax.scan to avoid LLVM memory issues
    step_fn = jax.jit(newton_step)
    g = g0
    for _ in range(n_steps):
        g = step_fn(g, params)

    phi_final = np.array(g[0:5])

    # DI = 1 - H/H_max
    phi_sum = phi_final.sum()
    p = phi_final / max(phi_sum, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum()
    di_0d = float(1.0 - H / np.log(5.0))

    # Material models — use di_scale=1.0 for 0D DI values
    from material_models import (
        compute_E_di,
        compute_E_phi_pg,
        compute_E_virulence,
        compute_E_eps_synergy,
    )

    phi_2d = phi_final.reshape(1, 5)
    E_di = float(compute_E_di(di_0d * np.ones(1), di_scale=DI_SCALE_0D)[0])
    E_pg = float(compute_E_phi_pg(phi_2d)[0])
    E_vir = float(compute_E_virulence(phi_2d)[0])
    E_eps = float(compute_E_eps_synergy(phi_2d)[0])

    return {
        "di_0d": di_0d,
        "phi_final": phi_final.tolist(),
        "phi_So": float(phi_final[0]),
        "phi_An": float(phi_final[1]),
        "phi_Vd": float(phi_final[2]),
        "phi_Fn": float(phi_final[3]),
        "phi_Pg": float(phi_final[4]),
        "E_di": E_di,
        "E_phi_pg": E_pg,
        "E_virulence": E_vir,
        "E_eps_synergy": E_eps,
    }


def run_condition(condition, n_samples, seed=42):
    """Run 0D CI propagation for one condition."""
    run_dir = CONDITION_RUNS.get(condition)
    if run_dir is None or not run_dir.exists():
        print(f"[SKIP] {condition}: run dir not found at {run_dir}")
        return None

    out_dir = _OUT_BASE / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # First compute MAP DI for basin filtering reference
    theta_map = load_theta_map(run_dir)
    if theta_map is None:
        print(f"[SKIP] {condition}: no theta_MAP")
        return None

    print(f"\n{'='*60}")
    print(f"0D CI Propagation: {condition}")
    print(f"{'='*60}")

    print("  Computing MAP reference...")
    map_result = solve_0d_single(theta_map)
    di_map = map_result["di_0d"]
    print(
        f"  MAP: DI={di_map:.4f}  E_di={map_result['E_di']:.1f} Pa  "
        f"phi={[f'{v:.3f}' for v in map_result['phi_final']]}"
    )

    samples, is_real = load_posterior_samples(run_dir, n_samples, seed)
    if samples is None:
        print(f"[SKIP] {condition}: no posterior samples")
        return None

    n_actual = len(samples)
    src_label = "real posterior" if is_real else "MAP + 2% perturbation"
    print(f"  Samples: {n_actual} ({src_label})")

    results = []
    n_filtered = 0
    t0 = time.time()

    for i in range(n_actual):
        theta = samples[i]
        try:
            r = solve_0d_single(theta)
            r["sample_idx"] = i

            # Basin filter: discard if DI jumped to different attractor
            if not is_real and abs(r["di_0d"] - di_map) > BASIN_THRESHOLD:
                n_filtered += 1
                if n_filtered <= 3:
                    print(
                        f"  [{i+1}/{n_actual}] FILTERED: DI={r['di_0d']:.4f} "
                        f"(MAP={di_map:.4f}, delta={abs(r['di_0d']-di_map):.3f})"
                    )
                continue

            results.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_actual - i - 1) / rate
                print(
                    f"  [{i+1}/{n_actual}] DI={r['di_0d']:.4f} E_di={r['E_di']:.1f} Pa "
                    f"({elapsed:.0f}s, ETA {eta:.0f}s)"
                )
        except Exception as e:
            print(f"  [{i+1}/{n_actual}] ERROR: {e}")

    elapsed = time.time() - t0

    if n_filtered > 0:
        print(
            f"  Basin filter: {n_filtered}/{n_actual} samples discarded "
            f"(jumped attractor, threshold={BASIN_THRESHOLD})"
        )

    if not results:
        print(f"  No valid results for {condition}")
        return None

    # Save per-sample results
    with open(out_dir / "samples_0d.json", "w") as f:
        json.dump(results, f, indent=1)

    # Compute statistics
    di_vals = np.array([r["di_0d"] for r in results])
    e_di_vals = np.array([r["E_di"] for r in results])
    e_pg_vals = np.array([r["E_phi_pg"] for r in results])
    e_vir_vals = np.array([r["E_virulence"] for r in results])
    phi_pg_vals = np.array([r["phi_Pg"] for r in results])

    def ci90(arr):
        lo = np.percentile(arr, 5)
        hi = np.percentile(arr, 95)
        return float(lo), float(hi), float(hi - lo)

    di_lo, di_hi, di_w = ci90(di_vals)
    edi_lo, edi_hi, edi_w = ci90(e_di_vals)
    epg_lo, epg_hi, epg_w = ci90(e_pg_vals)
    evir_lo, evir_hi, evir_w = ci90(e_vir_vals)

    summary = {
        "condition": condition,
        "n_samples_total": n_actual,
        "n_samples_kept": len(results),
        "n_filtered": n_filtered,
        "is_real_posterior": is_real,
        "method": "0D Hamilton ODE",
        "di_scale_0d": DI_SCALE_0D,
        "di_0d_map": di_map,
        "E_di_map": map_result["E_di"],
        "di_0d_mean": float(np.mean(di_vals)),
        "di_0d_median": float(np.median(di_vals)),
        "di_0d_std": float(np.std(di_vals)),
        "di_0d_ci90": [di_lo, di_hi],
        "di_0d_ci_width": di_w,
        "E_di_mean": float(np.mean(e_di_vals)),
        "E_di_ci90": [edi_lo, edi_hi],
        "E_di_ci_width": edi_w,
        "E_phi_pg_mean": float(np.mean(e_pg_vals)),
        "E_phi_pg_ci90": [epg_lo, epg_hi],
        "E_phi_pg_ci_width": epg_w,
        "E_vir_mean": float(np.mean(e_vir_vals)),
        "E_vir_ci90": [evir_lo, evir_hi],
        "E_vir_ci_width": evir_w,
        "phi_Pg_mean": float(np.mean(phi_pg_vals)),
        "timing_s": round(elapsed, 1),
    }

    with open(out_dir / "summary_0d.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results for {condition} ({len(results)} valid samples):")
    print(f"    DI_0D MAP:  {di_map:.4f}")
    print(
        f"    DI_0D mean: {summary['di_0d_mean']:.4f} "
        f"(90% CI: [{di_lo:.4f}, {di_hi:.4f}], width={di_w:.4f})"
    )
    print(f"    E_di MAP:   {map_result['E_di']:.1f} Pa")
    print(f"    E_di mean:  {summary['E_di_mean']:.1f} Pa " f"(CI: [{edi_lo:.1f}, {edi_hi:.1f}])")
    print(
        f"    E_Pg mean:  {summary['E_phi_pg_mean']:.1f} Pa " f"(CI: [{epg_lo:.1f}, {epg_hi:.1f}])"
    )
    print(
        f"    E_vir mean: {summary['E_vir_mean']:.1f} Pa " f"(CI: [{evir_lo:.1f}, {evir_hi:.1f}])"
    )
    print(f"    Time:       {elapsed:.1f}s")

    return summary


def plot_comparison(summaries):
    """Generate cross-condition comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(summaries.keys())
    n = len(conds)

    COND_LABELS = {
        "commensal_static": "Comm.\nStatic",
        "commensal_hobic": "Comm.\nHOBIC",
        "dh_baseline": "Dysb.\nHOBIC",
        "dysbiotic_static": "Dysb.\nStatic",
    }
    COND_COLORS = {
        "commensal_static": "#2ca02c",
        "commensal_hobic": "#17becf",
        "dh_baseline": "#d62728",
        "dysbiotic_static": "#ff7f0e",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    x = np.arange(n)
    colors = [COND_COLORS.get(c, "#333") for c in conds]
    labels = [COND_LABELS.get(c, c) for c in conds]

    def safe_err(means, ci_lo, ci_hi):
        lo = [max(m - l, 0) for m, l in zip(means, ci_lo)]
        hi = [max(h - m, 0) for m, h in zip(means, ci_hi)]
        return lo, hi

    # (a) DI_0D with CI + MAP diamond
    ax = axes[0, 0]
    means = [summaries[c]["di_0d_mean"] for c in conds]
    ci_lo = [summaries[c]["di_0d_ci90"][0] for c in conds]
    ci_hi = [summaries[c]["di_0d_ci90"][1] for c in conds]
    err_lo, err_hi = safe_err(means, ci_lo, ci_hi)
    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        x,
        means,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="k",
        capsize=6,
        capthick=1.5,
        linewidth=1.5,
    )
    map_vals = [summaries[c]["di_0d_map"] for c in conds]
    ax.scatter(x, map_vals, marker="D", color="red", s=40, zorder=5, label="MAP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$DI_{0D}$", fontsize=12)
    ax.set_title("(a) Dysbiosis Index (0D ODE)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    for i, m in enumerate(means):
        ax.text(i, m + max(err_hi) * 0.15 + 0.02, f"{m:.3f}", ha="center", fontsize=9)

    # (b) E_di with CI + MAP diamond
    ax = axes[0, 1]
    means = [summaries[c]["E_di_mean"] for c in conds]
    ci_lo = [summaries[c]["E_di_ci90"][0] for c in conds]
    ci_hi = [summaries[c]["E_di_ci90"][1] for c in conds]
    err_lo, err_hi = safe_err(means, ci_lo, ci_hi)
    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        x,
        means,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="k",
        capsize=6,
        capthick=1.5,
        linewidth=1.5,
    )
    map_vals = [summaries[c]["E_di_map"] for c in conds]
    ax.scatter(x, map_vals, marker="D", color="red", s=40, zorder=5, label="MAP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$E_{DI}$ [Pa]", fontsize=12)
    ax.set_title("(b) Stiffness — DI Model (scale=1.0)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # (c) E_phi_pg with CI
    ax = axes[0, 2]
    means = [summaries[c]["E_phi_pg_mean"] for c in conds]
    ci_lo = [summaries[c]["E_phi_pg_ci90"][0] for c in conds]
    ci_hi = [summaries[c]["E_phi_pg_ci90"][1] for c in conds]
    err_lo, err_hi = safe_err(means, ci_lo, ci_hi)
    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        x,
        means,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="k",
        capsize=6,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$E_{\\phi_{Pg}}$ [Pa]", fontsize=12)
    ax.set_title("(c) Stiffness — $\\phi_{Pg}$ Model", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (d) DI CI width + sample info
    ax = axes[1, 0]
    widths = [summaries[c]["di_0d_ci_width"] for c in conds]
    ax.bar(x, widths, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("90% CI width", fontsize=12)
    ax.set_title("(d) DI Uncertainty (CI width)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, c in enumerate(conds):
        s = summaries[c]
        lbl = f"{s['n_samples_kept']}/{s['n_samples_total']}"
        if s.get("is_real_posterior"):
            lbl += "\n(real)"
        else:
            lbl += "\n(perturb)"
        ax.text(i, widths[i] + max(widths) * 0.05, lbl, ha="center", fontsize=8)

    # (e) E_di CI width
    ax = axes[1, 1]
    widths = [summaries[c]["E_di_ci_width"] for c in conds]
    ax.bar(x, widths, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$E_{DI}$ CI width [Pa]", fontsize=12)
    ax.set_title("(e) Stiffness Uncertainty", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (f) Summary table
    ax = axes[1, 2]
    ax.axis("off")
    col_labels = ["Condition", "$DI_{MAP}$", "DI mean", "DI CI", "$E_{DI}$ [Pa]", "Samples"]
    cell_text = []
    for c in conds:
        s = summaries[c]
        src = "posterior" if s.get("is_real_posterior") else "perturb"
        cell_text.append(
            [
                c.replace("_", " "),
                f"{s['di_0d_map']:.4f}",
                f"{s['di_0d_mean']:.4f}",
                f"[{s['di_0d_ci90'][0]:.3f}, {s['di_0d_ci90'][1]:.3f}]",
                f"{s['E_di_mean']:.1f}",
                f"{s['n_samples_kept']}/{s['n_samples_total']} ({src})",
            ]
        )
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    ax.set_title("(f) Summary", fontsize=12, weight="bold", pad=20)

    fig.suptitle(
        "Posterior → 0D Hamilton ODE → DI / E Uncertainty (90% CI)\n"
        "DI_SCALE=1.0 for 0D | Basin filter (threshold=0.25) | MAP marked as ◆",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = _OUT_BASE / "cross_condition_ci_0d.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _OUT_BASE.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for cond in args.conditions:
        s = run_condition(cond, args.n_samples, args.seed)
        if s is not None:
            summaries[cond] = s

    # Save master summary
    master_path = _OUT_BASE / "master_summary_0d.json"
    with open(master_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nMaster summary: {master_path}")

    # Plot
    if len(summaries) >= 2:
        plot_comparison(summaries)

    print("\nDone.")


if __name__ == "__main__":
    main()
