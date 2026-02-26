#!/usr/bin/env python3
"""
plot_stress_3d_unified.py
==========================
Publication-quality 3D stress visualization with:
  - Unified colorbar across all 4 conditions
  - DI spatial field overlay (2D heatmap inset)
  - Log-scale displacement for cross-condition comparison
  - E_bio annotation per condition

Usage
-----
  python plot_stress_3d_unified.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)
_JOBS = _HERE / "_abaqus_auto_jobs"
_CONFORMAL = _HERE / "_3d_conformal_auto"

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]

COND_LABELS = {
    "commensal_static": "Commensal Static",
    "commensal_hobic": "Commensal HOBIC",
    "dh_baseline": "Dysbiotic HOBIC",
    "dysbiotic_static": "Dysbiotic Static",
}
COND_COLORS = {
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#17becf",
    "dh_baseline": "#d62728",
    "dysbiotic_static": "#ff7f0e",
}


def load_nodes_csv(csv_path):
    """Load node CSV: instance, node_id, x, y, z, ux, uy, uz, u_mag."""
    xs, ys, zs, umags = [], [], [], []
    with open(csv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            xs.append(float(parts[2]))
            ys.append(float(parts[3]))
            zs.append(float(parts[4]))
            umags.append(float(parts[8]))
    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "u_mag": np.array(umags),
    }


def load_condition_data(cond):
    """Load all data for one condition."""
    # Node CSV
    csv_path = _JOBS / f"{cond}_T23_v2" / "nodes_3d.csv"
    if not csv_path.exists():
        return None
    data = load_nodes_csv(csv_path)

    # Stress summary
    job_dir = _JOBS / f"{cond}_T23_v2"
    for jf in job_dir.glob("*_stress.json"):
        with open(jf) as f:
            data["stress"] = json.load(f)
        break

    # E_bio from INP
    inp_path = list(job_dir.glob("two_layer_T23_*.inp"))
    if inp_path:
        count = 0
        with open(inp_path[0]) as f:
            for line in f:
                if line.strip().startswith("*Elastic"):
                    count += 1
                    nxt = next(f, "").strip()
                    if count == 2:
                        data["E_bio_mpa"] = float(nxt.split(",")[0].strip())
                        data["E_bio_pa"] = data["E_bio_mpa"] * 1e6
                        break

    # DI field + meta from conformal auto
    meta_path = _CONFORMAL / cond / "auto_meta.json"
    di_path = _CONFORMAL / cond / "di_field.npy"
    if meta_path.exists():
        with open(meta_path) as f:
            data["meta"] = json.load(f)
    if di_path.exists():
        data["di_field"] = np.load(di_path)

    return data


def fig_unified_3d(all_data):
    """4-panel 3D scatter with unified log-scale colorbar + DI inset."""
    conds = [c for c in CONDITIONS if c in all_data]
    n = len(conds)
    if n < 2:
        print("Need >= 2 conditions")
        return

    fig = plt.figure(figsize=(18, 16))

    # Compute global displacement range for unified colorbar
    all_umag = np.concatenate([all_data[c]["u_mag"] for c in conds])
    vmin = max(all_umag.min(), 1.0)  # avoid log(0)
    vmax = all_umag.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    max_pts = 6000
    rng = np.random.default_rng(42)

    for idx, cond in enumerate(conds):
        d = all_data[cond]

        # 3D scatter (left 70% of each panel)
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        # Subsample
        n_nodes = len(d["x"])
        if n_nodes > max_pts:
            sel = rng.choice(n_nodes, max_pts, replace=False)
        else:
            sel = np.arange(n_nodes)

        sc = ax.scatter(
            d["x"][sel],
            d["y"][sel],
            d["z"][sel],
            c=np.clip(d["u_mag"][sel], vmin, vmax),
            cmap="inferno",
            norm=norm,
            s=3,
            alpha=0.6,
            rasterized=True,
        )

        # Annotations
        label = COND_LABELS[cond]
        e_pa = d.get("E_bio_pa", d.get("meta", {}).get("E_di_Pa", 0))
        di_0d = d.get("meta", {}).get("di_0d", 0)
        disp_max = d.get("stress", {}).get("displacement", {}).get("max_mag", np.max(d["u_mag"]))

        title = f"{label}\n$DI_{{0D}}$={di_0d:.3f}, $E_{{bio}}$={e_pa:.0f} Pa\n$U_{{max}}$={disp_max:.0f} mm"
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("X [mm]", fontsize=7)
        ax.set_ylabel("Y [mm]", fontsize=7)
        ax.set_zlabel("Z [mm]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-60)

    # Unified colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm = cm.ScalarMappable(norm=norm, cmap="inferno")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Displacement magnitude [mm] (log scale)", fontsize=11)

    fig.suptitle(
        "3D Abaqus: Biofilm Displacement — Unified Colorbar (Log Scale)\n"
        "Hybrid DI: 0D condition scale × 2D spatial pattern → $E(DI)$",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05, wspace=0.15, hspace=0.25)

    out = _FIG_DIR / "stress_3d_unified_log.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_di_overlay(all_data):
    """2×4 panel: Row 1 = DI spatial field (2D heatmap), Row 2 = 3D displacement."""
    conds = [c for c in CONDITIONS if c in all_data]
    n = len(conds)

    fig = plt.figure(figsize=(20, 12))

    # Global displacement range
    all_umag = np.concatenate([all_data[c]["u_mag"] for c in conds])
    vmin_u = max(all_umag.min(), 1.0)
    vmax_u = all_umag.max()
    norm_u = LogNorm(vmin=vmin_u, vmax=vmax_u)

    max_pts = 5000
    rng = np.random.default_rng(42)

    for idx, cond in enumerate(conds):
        d = all_data[cond]

        # Row 1: DI spatial heatmap
        ax_di = fig.add_subplot(2, n, idx + 1)
        di_field = d.get("di_field", None)
        if di_field is not None:
            im = ax_di.imshow(
                di_field.T, origin="lower", cmap="RdYlGn_r", aspect="equal", extent=[0, 1, 0, 1]
            )
            cb = fig.colorbar(im, ax=ax_di, shrink=0.8, pad=0.02)
            cb.set_label("DI", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            ax_di.text(0.5, 0.5, "No DI data", ha="center", va="center", transform=ax_di.transAxes)

        label = COND_LABELS[cond]
        di_0d = d.get("meta", {}).get("di_0d", 0)
        e_pa = d.get("meta", {}).get("E_di_Pa", 0)
        ax_di.set_title(
            f"{label}\n$DI_{{0D}}$={di_0d:.3f}, $E$={e_pa:.0f} Pa", fontsize=9, fontweight="bold"
        )
        ax_di.set_xlabel("x/L", fontsize=8)
        ax_di.set_ylabel("y/L", fontsize=8)
        ax_di.tick_params(labelsize=7)

        # Row 2: 3D displacement
        ax_3d = fig.add_subplot(2, n, n + idx + 1, projection="3d")
        n_nodes = len(d["x"])
        sel = rng.choice(n_nodes, min(max_pts, n_nodes), replace=False)

        sc = ax_3d.scatter(
            d["x"][sel],
            d["y"][sel],
            d["z"][sel],
            c=np.clip(d["u_mag"][sel], vmin_u, vmax_u),
            cmap="inferno",
            norm=norm_u,
            s=3,
            alpha=0.6,
            rasterized=True,
        )

        disp_max = d.get("stress", {}).get("displacement", {}).get("max_mag", np.max(d["u_mag"]))
        ax_3d.set_title(f"$U_{{max}}$ = {disp_max:.0f} mm", fontsize=9)
        ax_3d.set_xlabel("X", fontsize=7)
        ax_3d.set_ylabel("Y", fontsize=7)
        ax_3d.set_zlabel("Z", fontsize=7)
        ax_3d.tick_params(labelsize=6)
        ax_3d.view_init(elev=25, azim=-60)

    # Unified colorbar for displacement
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.40])
    sm = cm.ScalarMappable(norm=norm_u, cmap="inferno")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Displacement [mm] (log)", fontsize=9)

    fig.suptitle(
        "DI Spatial Field (top) → 3D Biofilm Displacement (bottom)\n"
        "Row 1: Hybrid DI from 0D ODE + 2D Hamilton+PDE   |   "
        "Row 2: Abaqus displacement on T23 conformal mesh",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.04, wspace=0.20, hspace=0.30)

    out = _FIG_DIR / "stress_3d_di_overlay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_bar_summary(all_data):
    """Summary bar chart: E_bio, U_max, DI_0D across conditions."""
    conds = [c for c in CONDITIONS if c in all_data]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(conds))
    colors = [COND_COLORS[c] for c in conds]
    labels = [COND_LABELS[c].replace(" ", "\n") for c in conds]

    # (a) DI_0D
    ax = axes[0]
    vals = [all_data[c].get("meta", {}).get("di_0d", 0) for c in conds]
    ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$DI_{0D}$", fontsize=11)
    ax.set_title("(a) Dysbiosis Index (0D)", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (b) E_bio [Pa]
    ax = axes[1]
    vals = [all_data[c].get("meta", {}).get("E_di_Pa", 0) for c in conds]
    ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$E_{bio}$ [Pa]", fontsize=11)
    ax.set_title("(b) Biofilm Stiffness", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) U_max [mm] (log scale)
    ax = axes[2]
    vals = [
        all_data[c]
        .get("stress", {})
        .get("displacement", {})
        .get("max_mag", np.max(all_data[c]["u_mag"]))
        for c in conds
    ]
    ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$U_{max}$ [mm] (log)", fontsize=11)
    ax.set_title("(c) Max Displacement", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (d) E vs U scatter (log-log)
    ax = axes[3]
    for i, c in enumerate(conds):
        e = all_data[c].get("meta", {}).get("E_di_Pa", 0)
        u = (
            all_data[c]
            .get("stress", {})
            .get("displacement", {})
            .get("max_mag", np.max(all_data[c]["u_mag"]))
        )
        ax.scatter(e, u, color=COND_COLORS[c], s=120, edgecolor="k", zorder=5, label=COND_LABELS[c])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$E_{bio}$ [Pa]", fontsize=11)
    ax.set_ylabel("$U_{max}$ [mm]", fontsize=11)
    ax.set_title("(d) Stiffness vs Displacement", fontsize=11, weight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Condition Biofilm Mechanics Summary", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = _FIG_DIR / "stress_3d_bar_summary.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("Loading condition data...")
    all_data = {}
    for cond in CONDITIONS:
        d = load_condition_data(cond)
        if d is not None:
            all_data[cond] = d
            n = len(d["x"])
            disp = np.max(d["u_mag"])
            print(f"  {cond}: {n} nodes, U_max={disp:.0f} mm")

    if not all_data:
        print("No data available")
        return

    print(f"\n{len(all_data)} conditions loaded")

    # Figure 1: Unified 3D with log colorbar
    print("\nGenerating unified 3D figure...")
    fig_unified_3d(all_data)

    # Figure 2: DI overlay (2D heatmap + 3D displacement)
    print("\nGenerating DI overlay figure...")
    fig_di_overlay(all_data)

    # Figure 3: Bar summary
    print("\nGenerating bar summary...")
    fig_bar_summary(all_data)

    print(f"\nAll figures saved in: {_FIG_DIR}")


if __name__ == "__main__":
    main()
