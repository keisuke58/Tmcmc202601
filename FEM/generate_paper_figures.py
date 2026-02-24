#!/usr/bin/env python3
"""
generate_paper_figures.py
==========================
C2: Paper-quality figures (Fig 8-15) — final versions.

Generates all manuscript figures for the biofilm multiscale mechanics paper.

Figure layout:
  Fig 8:  TMCMC posterior parameter distributions (20 params, 2 conditions)
  Fig 9:  2D Hamilton+Nutrient: species volume fractions + nutrient field
  Fig 10: DI field spatial distribution (commensal vs dysbiotic)
  Fig 11: Material model E(DI) with biofilm data overlay
  Fig 12: 3D conformal mesh + tooth assembly visualization
  Fig 13: von Mises stress comparison (commensal vs dysbiotic)
  Fig 14: Klempt 2024 benchmark comparison
  Fig 15: Posterior uncertainty propagation (theta → DI → sigma CI bands)

Data sources:
  - TMCMC posterior samples: data_5species/_runs/
  - 2D simulation results: _3d_conformal_auto/ or _results_2d_nutrient/
  - Abaqus stress results: _posterior_abaqus/ or _abaqus_auto_jobs/
  - Klempt benchmark: _klempt_benchmark/

Usage
-----
  python generate_paper_figures.py              # all figures
  python generate_paper_figures.py --figs 9 10  # specific figures only
  python generate_paper_figures.py --dpi 300    # high-res for print
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

_HERE = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_RUNS_ROOT = _TMCMC_ROOT / "data_5species" / "_runs"
_FIG_DIR = _HERE / "figures" / "paper_final"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

SPECIES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
SP_SHORT = ["S.or", "A.na", "V.di", "F.nu", "P.gi"]

COND_META = {
    "commensal_static": {"label": "Commensal", "color": "#2ca02c", "ls": "-"},
    "dh_baseline": {"label": "Dysbiotic (DH)", "color": "#d62728", "ls": "--"},
    "commensal_hobic": {"label": "Comm-HOBIC", "color": "#17becf", "ls": "-."},
    "dysbiotic_static": {"label": "Dysb-Static", "color": "#ff7f0e", "ls": ":"},
}

PARAM_NAMES = [
    "a11", "a12", "a22", "b1", "b2",
    "a33", "a34", "a44", "b3", "b4",
    "a13", "a14", "a23", "a24",
    "a55", "b5", "a15", "a25", "a35", "a45",
]
PARAM_TEX = [
    r"$a_{11}$", r"$a_{12}$", r"$a_{22}$", r"$b_1$", r"$b_2$",
    r"$a_{33}$", r"$a_{34}$", r"$a_{44}$", r"$b_3$", r"$b_4$",
    r"$a_{13}$", r"$a_{14}$", r"$a_{23}$", r"$a_{24}$",
    r"$a_{55}$", r"$b_5$", r"$a_{15}$", r"$a_{25}$", r"$a_{35}$", r"$a_{45}$",
]

CONDITION_RUNS = {
    "dh_baseline":      _RUNS_ROOT / "sweep_pg_20260217_081459" / "dh_baseline",
    "commensal_static": _RUNS_ROOT / "Commensal_Static_20260208_002100",
}

E_MAX = 10.0e9; E_MIN = 0.5e9; DI_SCALE = 0.025778; DI_EXP = 2.0


def di_to_eeff(di):
    r = np.clip(di / DI_SCALE, 0, 1)
    return E_MAX * (1-r)**DI_EXP + E_MIN * r


def _load_samples(run_dir):
    """Load posterior samples from TMCMC run."""
    samples_path = run_dir / "samples.npy"
    if samples_path.exists():
        return np.load(samples_path)
    # Try CSV fallback
    csv_path = run_dir / "posterior_samples.csv"
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return None


def _load_theta_map(run_dir):
    """Load MAP theta from TMCMC run."""
    tp = run_dir / "theta_MAP.json"
    if not tp.exists():
        return None
    with open(tp) as f:
        d = json.load(f)
    if "theta_full" in d:
        return np.array(d["theta_full"])
    elif "theta_sub" in d:
        return np.array(d["theta_sub"])
    return None


def fig08_posterior(dpi=200):
    """Fig 8: TMCMC posterior distributions."""
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    axes = axes.flatten()

    for cond, meta in COND_META.items():
        run_dir = CONDITION_RUNS.get(cond)
        if run_dir is None or not run_dir.exists():
            continue
        samples = _load_samples(run_dir)
        theta_map = _load_theta_map(run_dir)
        if samples is None:
            continue

        n_params = min(samples.shape[1], 20)
        for i in range(n_params):
            ax = axes[i]
            ax.hist(samples[:, i], bins=30, alpha=0.4,
                    color=meta["color"], density=True, label=meta["label"])
            if theta_map is not None and i < len(theta_map):
                ax.axvline(theta_map[i], color=meta["color"], ls="--", lw=1.5)

    for i in range(20):
        axes[i].set_xlabel(PARAM_TEX[i], fontsize=10)
        if i == 0:
            axes[i].legend(fontsize=7)
        axes[i].tick_params(labelsize=7)

    fig.suptitle("Fig 8: TMCMC Posterior Parameter Distributions", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _FIG_DIR / "Fig08_posterior.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 8: {out}")


def fig09_species_nutrient(dpi=200):
    """Fig 9: 2D Hamilton species + nutrient field."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.35)

    for ci, cond in enumerate(["commensal_static", "dh_baseline"]):
        meta = COND_META[cond]
        # Try loading simulation data
        for search_dir in [
            _HERE / "_3d_conformal_auto" / cond,
            _HERE / "_results_2d_nutrient" / cond,
        ]:
            phi_path = search_dir / "phi_snaps.npy"
            c_path = search_dir / "c_snaps.npy"
            if phi_path.exists() and c_path.exists():
                phi_snaps = np.load(phi_path)
                c_snaps = np.load(c_path)
                break
        else:
            # No data, skip this condition
            for j in range(4):
                ax = fig.add_subplot(gs[ci, j])
                ax.text(0.5, 0.5, f"No data\n({cond})", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
            continue

        phi_final = phi_snaps[-1]  # (5, Nx, Ny)
        c_final = c_snaps[-1]     # (Nx, Ny)
        Nx, Ny = c_final.shape

        # Panel: 3 species + nutrient
        show_species = [0, 3, 4]  # S.oralis, F.nucleatum, P.gingivalis
        for j, sp in enumerate(show_species):
            ax = fig.add_subplot(gs[ci, j])
            im = ax.imshow(phi_final[sp].T, origin="lower", aspect="equal",
                          cmap="hot", vmin=0, vmax=max(0.3, phi_final[sp].max()))
            plt.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(f"{meta['label']}: {SP_SHORT[sp]}", fontsize=10)

        # Nutrient field
        ax = fig.add_subplot(gs[ci, 3])
        im = ax.imshow(c_final.T, origin="lower", aspect="equal",
                       cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.7, label="c")
        ax.set_title(f"{meta['label']}: Nutrient", fontsize=10)

    fig.suptitle("Fig 9: 2D Hamilton+Nutrient Species and Nutrient Fields",
                 fontsize=14, weight="bold")
    out = _FIG_DIR / "Fig09_species_nutrient.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 9: {out}")


def fig10_di_field(dpi=200):
    """Fig 10: DI field spatial distribution comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    di_data = {}
    for ci, cond in enumerate(["commensal_static", "dh_baseline"]):
        meta = COND_META[cond]
        for search_dir in [
            _HERE / "_3d_conformal_auto" / cond,
            _HERE / "_results_2d_nutrient" / cond,
        ]:
            di_path = search_dir / "di_field.npy"
            if di_path.exists():
                di = np.load(di_path)
                di_data[cond] = di
                break

        ax = axes[ci]
        if cond in di_data:
            im = ax.imshow(di_data[cond].T, origin="lower", aspect="equal",
                          cmap="RdYlGn_r", vmin=0, vmax=max(0.03, di_data[cond].max()))
            plt.colorbar(im, ax=ax, shrink=0.8, label="DI")
        else:
            ax.text(0.5, 0.5, f"No data\n({cond})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{meta['label']}", fontsize=12)

    # Panel 3: DI histogram comparison
    ax = axes[2]
    for cond, meta in COND_META.items():
        if cond in di_data:
            ax.hist(di_data[cond].flatten(), bins=50, alpha=0.5,
                    color=meta["color"], density=True, label=meta["label"])
    ax.set_xlabel("DI"); ax.set_ylabel("Density")
    ax.set_title("DI Distribution", fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle("Fig 10: Dysbiotic Index (DI) Spatial Distribution",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIG_DIR / "Fig10_di_field.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 10: {out}")


def fig11_material_model(dpi=200):
    """Fig 11: E(DI) material model with data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: E(DI) curve
    ax = axes[0]
    di = np.linspace(0, DI_SCALE * 2, 200)
    e = di_to_eeff(di) * 1e-9
    ax.plot(di, e, "k-", lw=3, label=r"$E(DI) = E_{max}(1-r)^n + E_{min}\cdot r$")
    ax.axvline(DI_SCALE, color="gray", ls=":", label=f"DI_scale={DI_SCALE:.4f}")

    # Overlay condition data points
    for cond in ["commensal_static", "dh_baseline"]:
        meta = COND_META.get(cond)
        if meta is None:
            continue
        for search_dir in [
            _HERE / "_3d_conformal_auto" / cond,
            _HERE / "_results_2d_nutrient" / cond,
        ]:
            di_path = search_dir / "di_field.npy"
            if di_path.exists():
                di_field = np.load(di_path).flatten()
                e_field = di_to_eeff(di_field) * 1e-9
                ax.scatter(di_field[::10], e_field[::10], s=5, alpha=0.3,
                          color=meta["color"], label=meta["label"])
                break

    ax.set_xlabel("DI", fontsize=12)
    ax.set_ylabel("$E_{eff}$ [GPa]", fontsize=12)
    ax.set_title("(a) Power-Law Material Model", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel 2: E parameters
    ax = axes[1]
    ax.axis("off")
    params_text = [
        "Material Model Parameters",
        "="*30,
        f"E_max = {E_MAX*1e-9:.1f} GPa (healthy)",
        f"E_min = {E_MIN*1e-9:.1f} GPa (dysbiotic)",
        f"DI_scale = {DI_SCALE:.6f}",
        f"n (exponent) = {DI_EXP:.1f}",
        f"nu = 0.30",
        "",
        "E_eff = E_max*(1-r)^n + E_min*r",
        "r = clamp(DI / DI_scale, 0, 1)",
        "",
        "Reference:",
        "  Billings et al. 2015",
        "  Klempt et al. 2024",
    ]
    ax.text(0.1, 0.95, "\n".join(params_text), transform=ax.transAxes,
            fontsize=10, family="monospace", va="top")

    fig.suptitle("Fig 11: DI-Dependent Material Model", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIG_DIR / "Fig11_material_model.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 11: {out}")


def fig12_mesh_assembly(dpi=200):
    """Fig 12: 3D conformal mesh schematic."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Schematic of mesh generation
    ax = axes[0]
    ax.axis("off")
    schematic = [
        "Mesh Generation Pipeline",
        "="*30,
        "",
        "1. Read STL (tooth surface)",
        "   -> Deduplicate vertices",
        "   -> Area-weighted normals",
        "",
        "2. Offset surface (biofilm)",
        "   -> N-layer interpolation",
        "   -> Laplacian smoothing",
        "",
        "3. Prism -> C3D4 tet split",
        "   -> 3 tets per prism",
        "   -> Volume check + fix",
        "",
        "4. DI bin assignment",
        "   -> E_eff per material bin",
        "",
        "5. Two-layer Tie assembly",
        "   -> Tooth S3 + Biofilm C3D4",
        "   -> *Tie constraint",
    ]
    ax.text(0.05, 0.95, "\n".join(schematic), transform=ax.transAxes,
            fontsize=9, family="monospace", va="top")
    ax.set_title("(a) Mesh Pipeline", fontsize=12)

    # Panel 2: Cross-section schematic
    ax = axes[1]
    # Draw simplified cross-section
    theta = np.linspace(0, 2*np.pi, 100)
    # Tooth surface
    r_tooth = 1.0
    x_tooth = r_tooth * np.cos(theta) + 2
    y_tooth = r_tooth * np.sin(theta) + 2
    ax.plot(x_tooth, y_tooth, "b-", lw=2, label="Tooth (S3)")
    # Biofilm surface (offset)
    r_bio = 1.3
    x_bio = r_bio * np.cos(theta) + 2
    y_bio = r_bio * np.sin(theta) + 2
    ax.plot(x_bio, y_bio, "r-", lw=2, label="Biofilm outer")
    # Fill biofilm layer
    ax.fill_between(x_bio, y_bio, y_tooth, alpha=0.2, color="red")
    # Layers
    for k in range(1, 4):
        r_k = r_tooth + k * (r_bio - r_tooth) / 4
        x_k = r_k * np.cos(theta) + 2
        y_k = r_k * np.sin(theta) + 2
        ax.plot(x_k, y_k, "r:", lw=0.5, alpha=0.5)
    ax.fill_between(x_tooth, y_tooth, 0, alpha=0.15, color="blue")
    ax.set_xlim(0.3, 3.7); ax.set_ylim(0.3, 3.7)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.set_title("(b) Cross-Section Schematic", fontsize=12)

    # Panel 3: Assembly stats
    ax = axes[2]
    ax.axis("off")
    # Check for auto results
    stats_text = ["Assembly Statistics", "="*25, ""]
    for cond in ["dh_baseline", "commensal_static"]:
        meta_path = _HERE / "_3d_conformal_auto" / cond / "auto_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for tooth, mesh in meta.get("meshes", {}).items():
                stats_text.append(f"{cond} / {tooth}:")
                stats_text.append(f"  Tooth nodes: {mesh.get('n_tooth_nodes', 'N/A')}")
                stats_text.append(f"  Bio nodes:   {mesh.get('n_bio_nodes', 'N/A')}")
                stats_text.append(f"  Bio tets:    {mesh.get('n_bio_tets', 'N/A')}")
                stats_text.append("")
    if len(stats_text) <= 3:
        stats_text.append("(Run generate_3d_conformal_auto.py first)")

    ax.text(0.05, 0.95, "\n".join(stats_text), transform=ax.transAxes,
            fontsize=9, family="monospace", va="top")
    ax.set_title("(c) Mesh Statistics", fontsize=12)

    fig.suptitle("Fig 12: 3D Conformal Mesh Assembly", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIG_DIR / "Fig12_mesh_assembly.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 12: {out}")


def fig13_stress_comparison(dpi=200):
    """Fig 13: von Mises stress comparison."""
    # Delegate to plot_stress_comparison.py logic
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ci, cond in enumerate(["commensal_static", "dh_baseline"]):
        meta = COND_META[cond]
        ax = axes[ci]

        # Try loading Abaqus results
        auto_dir = _HERE / "_abaqus_auto_jobs" / f"{cond}_T23"
        post_dir = _HERE / "_posterior_abaqus" / cond

        stress_data = None
        if auto_dir.exists():
            for f in auto_dir.glob("*_elements.csv"):
                mises = []
                with open(f) as fp:
                    fp.readline()
                    for line in fp:
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            try:
                                mises.append(float(parts[1]))
                            except ValueError:
                                pass
                if mises:
                    stress_data = np.array(mises)

        if stress_data is not None:
            ax.hist(stress_data, bins=50, color=meta["color"], alpha=0.7)
            ax.axvline(np.mean(stress_data), color="k", ls="--",
                      label=f"mean={np.mean(stress_data):.3f}")
            ax.legend(fontsize=8)
        elif post_dir.exists():
            subs = []
            for sd in sorted(post_dir.glob("sample_*")):
                sj = sd / "stress.json"
                df = sd / "done.flag"
                if sj.exists() and df.exists():
                    with open(sj) as fp:
                        s = json.load(fp)
                    if "substrate_mises_mean" in s:
                        subs.append(s["substrate_mises_mean"])
            if subs:
                ax.hist(subs, bins=15, color=meta["color"], alpha=0.7)
                ax.axvline(np.mean(subs), color="k", ls="--",
                          label=f"mean={np.mean(subs):.3f}")
                ax.legend(fontsize=8)
        else:
            # Synthetic from DI
            for sd in [_HERE / "_3d_conformal_auto" / cond,
                       _HERE / "_results_2d_nutrient" / cond]:
                dp = sd / "di_field.npy"
                if dp.exists():
                    di = np.load(dp)
                    e = di_to_eeff(di) * 1e-6  # MPa
                    ax.hist(e.flatten(), bins=50, color=meta["color"], alpha=0.7)
                    ax.axvline(np.mean(e), color="k", ls="--",
                              label=f"E_mean={np.mean(e):.1f}")
                    ax.set_xlabel("$E_{eff}$ [MPa]")
                    ax.legend(fontsize=8)
                    break
            else:
                ax.text(0.5, 0.5, f"No data", ha="center", va="center",
                        transform=ax.transAxes)

        ax.set_title(f"{meta['label']}", fontsize=12)

    # Panel 3: ratio
    ax = axes[2]
    ax.axis("off")
    ax.text(0.5, 0.5, "Stress ratio plot\n(requires Abaqus results)",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("(c) Stress Ratio", fontsize=12)

    fig.suptitle("Fig 13: von Mises Stress Distribution",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIG_DIR / "Fig13_stress.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 13: {out}")


def fig14_klempt(dpi=200):
    """Fig 14: Klempt benchmark (delegates to klempt_benchmark.py output)."""
    bench_fig = _HERE / "_klempt_benchmark" / "klempt_benchmark.png"
    if bench_fig.exists():
        # Just copy/link
        import shutil
        out = _FIG_DIR / "Fig14_klempt_benchmark.png"
        shutil.copy2(str(bench_fig), str(out))
        print(f"  Fig 14: {out} (from klempt_benchmark.py)")
    else:
        # Generate inline
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Run klempt_benchmark.py first\nto generate this figure",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        fig.suptitle("Fig 14: Klempt 2024 Benchmark", fontsize=14, weight="bold")
        out = _FIG_DIR / "Fig14_klempt_benchmark.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  Fig 14: {out} (placeholder)")


def fig15_uncertainty(dpi=200):
    """Fig 15: Posterior uncertainty propagation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ci, cond in enumerate(["commensal_static", "dh_baseline"]):
        meta = COND_META[cond]
        ax = axes[ci]

        # Try loading posterior ensemble
        post_dir = _HERE / "_posterior_abaqus" / cond
        unc_dir = _HERE / "_uncertainty_propagation" / cond

        if unc_dir.exists():
            # Load from uncertainty propagation results
            di_samples = []
            for sp in sorted(unc_dir.glob("sample_*")):
                dp = sp / "di_field.npy"
                if dp.exists():
                    di_samples.append(np.load(dp).flatten())
            if di_samples:
                di_arr = np.array(di_samples)
                di_mean = np.mean(di_arr, axis=0)
                di_p05 = np.percentile(di_arr, 5, axis=0)
                di_p95 = np.percentile(di_arr, 95, axis=0)

                x = np.arange(len(di_mean))
                ax.plot(x, np.sort(di_mean), color=meta["color"], lw=2, label="Median")
                ax.fill_between(x, np.sort(di_p05), np.sort(di_p95),
                               alpha=0.3, color=meta["color"], label="90% CI")
                ax.legend(fontsize=8)
        elif post_dir.exists():
            subs = []
            for sd in sorted(post_dir.glob("sample_*")):
                sj = sd / "stress.json"
                df = sd / "done.flag"
                if sj.exists() and df.exists():
                    with open(sj) as fp:
                        s = json.load(fp)
                    if "substrate_mises_mean" in s:
                        subs.append(s["substrate_mises_mean"])
            if subs:
                ax.hist(subs, bins=15, color=meta["color"], alpha=0.7,
                       density=True)
                ax.axvline(np.mean(subs), color="k", ls="--")
                ax.axvline(np.percentile(subs, 5), color="k", ls=":")
                ax.axvline(np.percentile(subs, 95), color="k", ls=":")
        else:
            ax.text(0.5, 0.5, f"No data\n({cond})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)

        ax.set_title(f"{meta['label']}", fontsize=12)
        ax.set_xlabel("DI / Stress")

    # Panel 3: Combined CI comparison
    ax = axes[2]
    ax.axis("off")
    summary = [
        "Uncertainty Propagation Summary",
        "="*35,
        "",
        "Pipeline: posterior theta",
        "  -> Hamilton 2D ODE",
        "  -> DI field",
        "  -> E_eff material model",
        "  -> FEM stress",
        "",
        "90% credible intervals",
        "propagated through full chain",
        "",
        "Run posterior_uncertainty_propagation.py",
        "for full results",
    ]
    ax.text(0.05, 0.95, "\n".join(summary), transform=ax.transAxes,
            fontsize=9, family="monospace", va="top")
    ax.set_title("(c) Pipeline", fontsize=12)

    fig.suptitle("Fig 15: Posterior Uncertainty Propagation",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _FIG_DIR / "Fig15_uncertainty.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Fig 15: {out}")


# ── Main dispatch ────────────────────────────────────────────────────────────

FIG_DISPATCH = {
    8:  fig08_posterior,
    9:  fig09_species_nutrient,
    10: fig10_di_field,
    11: fig11_material_model,
    12: fig12_mesh_assembly,
    13: fig13_stress_comparison,
    14: fig14_klempt,
    15: fig15_uncertainty,
}


def main():
    ap = argparse.ArgumentParser(description="Generate paper figures (Fig 8-15)")
    ap.add_argument("--figs", nargs="+", type=int, default=list(range(8, 16)),
                    help="Figure numbers to generate (default: 8-15)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    print("="*60)
    print("Paper Figure Generation (Fig 8-15)")
    print(f"  Output: {_FIG_DIR}")
    print(f"  DPI: {args.dpi}")
    print("="*60)

    for fig_num in sorted(args.figs):
        if fig_num in FIG_DISPATCH:
            print(f"\n[Fig {fig_num}]")
            try:
                FIG_DISPATCH[fig_num](dpi=args.dpi)
            except Exception as e:
                print(f"  [ERROR] Fig {fig_num}: {e}")
        else:
            print(f"  [SKIP] Fig {fig_num}: not defined")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {_FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
