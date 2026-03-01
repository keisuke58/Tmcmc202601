#!/usr/bin/env python3
"""
plot_stress_3d.py
==================
3D visualization of Abaqus stress analysis results.
Reads node CSV from ODB extraction and creates 3D figures.

Usage
-----
  python plot_stress_3d.py
  python plot_stress_3d.py --conditions commensal_static dh_baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]

COND_INFO = {
    "commensal_static": {"short": "Comm-Static", "color": "#2ca02c"},
    "commensal_hobic": {"short": "Comm-HOBIC", "color": "#17becf"},
    "dh_baseline": {"short": "Dysb-HOBIC", "color": "#d62728"},
    "dysbiotic_static": {"short": "Dysb-Static", "color": "#ff7f0e"},
}


def load_nodes_csv(csv_path):
    """Load node CSV: instance, node_id, x, y, z, ux, uy, uz, u_mag."""
    data = {
        "instance": [],
        "node_id": [],
        "x": [],
        "y": [],
        "z": [],
        "ux": [],
        "uy": [],
        "uz": [],
        "u_mag": [],
    }
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            data["instance"].append(parts[0])
            data["node_id"].append(int(parts[1]))
            for i, key in enumerate(["x", "y", "z", "ux", "uy", "uz", "u_mag"]):
                data[key].append(float(parts[i + 2]))
    for key in ["x", "y", "z", "ux", "uy", "uz", "u_mag"]:
        data[key] = np.array(data[key])
    data["instance"] = np.array(data["instance"])
    return data


def load_stress_json(jobs_dir, cond, suffix="v2"):
    """Load stress summary JSON."""
    job_dir = jobs_dir / f"{cond}_T23_{suffix}"
    json_path = job_dir / f"two_layer_T23_{cond}_stress.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def extract_e_biofilm(jobs_dir, cond, suffix="v2"):
    """Extract biofilm E [MPa] from INP (second *Elastic)."""
    inp_path = jobs_dir / f"{cond}_T23_{suffix}" / f"two_layer_T23_{cond}.inp"
    if not inp_path.exists():
        return None
    count = 0
    with open(inp_path) as f:
        for line in f:
            if line.strip().startswith("*Elastic"):
                count += 1
                nxt = next(f, "").strip()
                if count == 2:
                    return float(nxt.split(",")[0].strip())
    return None


def fig_3d_displacement_comparison(all_data, stress_info, suffix="v2"):
    """4-panel 3D scatter: deformed shape colored by displacement."""
    conds = list(all_data.keys())
    n = len(conds)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(8 * ncols, 7 * nrows))

    # Global displacement range for consistent colorbar
    all_umag = np.concatenate([all_data[c]["u_mag"] for c in conds])
    vmin_global = 0
    vmax_global = np.percentile(all_umag, 99)

    for idx, cond in enumerate(conds):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        d = all_data[cond]

        # Separate tooth vs biofilm by instance name
        instances = np.unique(d["instance"])
        tooth_mask = np.array(
            ["TOOTH" in inst.upper() or "DENTIN" in inst.upper() for inst in d["instance"]]
        )
        bio_mask = ~tooth_mask

        # Deformation scale (normalize so max deformation is visible)
        u_mag = d["u_mag"]
        scale = 0.0  # Don't deform -- show undeformed with color

        x = d["x"] + scale * d["ux"]
        y = d["y"] + scale * d["uy"]
        z = d["z"] + scale * d["uz"]

        # Subsample for performance (plot at most 5000 points per part)
        max_pts = 5000

        # Plot tooth nodes in gray
        if tooth_mask.any():
            idx_t = np.where(tooth_mask)[0]
            if len(idx_t) > max_pts:
                idx_t = np.random.choice(idx_t, max_pts, replace=False)
            ax.scatter(
                x[idx_t], y[idx_t], z[idx_t], c="lightgray", s=1, alpha=0.15, rasterized=True
            )

        # Plot biofilm nodes colored by displacement
        if bio_mask.any():
            idx_b = np.where(bio_mask)[0]
            if len(idx_b) > max_pts:
                idx_b = np.random.choice(idx_b, max_pts, replace=False)
            sc = ax.scatter(
                x[idx_b],
                y[idx_b],
                z[idx_b],
                c=u_mag[idx_b],
                cmap="hot_r",
                vmin=0,
                vmax=max(u_mag[bio_mask]) * 1.0 if bio_mask.any() else 1,
                s=4,
                alpha=0.7,
                rasterized=True,
            )
            cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08)
            cb.set_label("Displacement [mm]", fontsize=9)
        else:
            # All nodes are one instance - color all by displacement
            idx_all = np.arange(len(x))
            if len(idx_all) > max_pts * 2:
                idx_all = np.random.choice(idx_all, max_pts * 2, replace=False)
            sc = ax.scatter(
                x[idx_all],
                y[idx_all],
                z[idx_all],
                c=u_mag[idx_all],
                cmap="hot_r",
                vmin=0,
                vmax=np.percentile(u_mag, 99),
                s=2,
                alpha=0.6,
                rasterized=True,
            )
            cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08)
            cb.set_label("Displacement [mm]", fontsize=9)

        # Axis labels and title
        info = COND_INFO.get(cond, {"short": cond})
        e_bio = stress_info.get(cond, {}).get("E_Pa", None)
        title = f"{info['short']}"
        if e_bio:
            title += f"\n$E_{{bio}}$ = {e_bio:.0f} Pa"
        si = stress_info.get(cond, {})
        if "disp_max" in si:
            title += f", $U_{{max}}$ = {si['disp_max']:.0f} mm"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X [mm]", fontsize=8)
        ax.set_ylabel("Y [mm]", fontsize=8)
        ax.set_zlabel("Z [mm]", fontsize=8)
        ax.tick_params(labelsize=7)

        # Set consistent view angle
        ax.view_init(elev=25, azim=-60)

    fig.suptitle(
        "3D Abaqus: Biofilm Displacement by Condition\n"
        "(Hybrid DI: 0D Hamilton scale + 2D spatial pattern)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = _FIG_DIR / "stress_3d_displacement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def fig_3d_multiview(all_data, stress_info, cond="dh_baseline"):
    """Multi-view (3 angles) of a single condition."""
    if cond not in all_data:
        print(f"  [SKIP] {cond} not available for multi-view")
        return None

    d = all_data[cond]
    fig = plt.figure(figsize=(18, 6))

    angles = [(25, -60), (90, 0), (0, 0)]  # Perspective, Top, Front
    angle_names = ["Perspective", "Top", "Front"]

    u_mag = d["u_mag"]
    x, y, z = d["x"], d["y"], d["z"]

    # Instance classification
    tooth_mask = np.array(
        ["TOOTH" in inst.upper() or "DENTIN" in inst.upper() for inst in d["instance"]]
    )
    bio_mask = ~tooth_mask
    max_pts = 5000

    for i, ((elev, azim), name) in enumerate(zip(angles, angle_names)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # Tooth
        if tooth_mask.any():
            idx_t = np.where(tooth_mask)[0]
            if len(idx_t) > max_pts:
                idx_t = np.random.choice(idx_t, max_pts, replace=False)
            ax.scatter(
                x[idx_t], y[idx_t], z[idx_t], c="lightgray", s=1, alpha=0.15, rasterized=True
            )

        # Biofilm
        if bio_mask.any():
            idx_b = np.where(bio_mask)[0]
            if len(idx_b) > max_pts:
                idx_b = np.random.choice(idx_b, max_pts, replace=False)
            sc = ax.scatter(
                x[idx_b],
                y[idx_b],
                z[idx_b],
                c=u_mag[idx_b],
                cmap="hot_r",
                vmin=0,
                vmax=max(u_mag[bio_mask]),
                s=4,
                alpha=0.7,
                rasterized=True,
            )
            fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.08)
        else:
            idx_all = np.arange(len(x))
            if len(idx_all) > max_pts * 2:
                idx_all = np.random.choice(idx_all, max_pts * 2, replace=False)
            sc = ax.scatter(
                x[idx_all],
                y[idx_all],
                z[idx_all],
                c=u_mag[idx_all],
                cmap="hot_r",
                vmin=0,
                vmax=np.percentile(u_mag, 99),
                s=2,
                alpha=0.6,
                rasterized=True,
            )
            fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.08)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} view", fontsize=11)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(labelsize=7)

    info = COND_INFO.get(cond, {"short": cond})
    fig.suptitle(
        f"3D Views: {info['short']} — Displacement Field [mm]", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = _FIG_DIR / f"stress_3d_multiview_{cond}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--suffix", default="v2")
    args = ap.parse_args()

    jobs_dir = _HERE / "_abaqus_auto_jobs"

    # Load node CSVs
    all_data = {}
    stress_info = {}
    for cond in args.conditions:
        csv_path = jobs_dir / f"{cond}_T23_{args.suffix}" / "nodes_3d.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {cond}: no nodes_3d.csv")
            continue
        print(f"  Loading {cond}...")
        all_data[cond] = load_nodes_csv(csv_path)
        n = len(all_data[cond]["x"])
        print(f"    {n} nodes loaded")

        # Stress summary
        sj = load_stress_json(jobs_dir, cond, args.suffix)
        e_bio = extract_e_biofilm(jobs_dir, cond, args.suffix)
        si = {}
        if e_bio:
            si["E_Pa"] = e_bio * 1e6
        if sj:
            si["disp_max"] = sj["displacement"]["max_mag"]
            si["mises_max"] = sj["mises"]["max"]
        stress_info[cond] = si

    if not all_data:
        print("No data available")
        return

    # Figure 1: 4-panel comparison
    print("\nGenerating 3D comparison figure...")
    fig_3d_displacement_comparison(all_data, stress_info, args.suffix)

    # Figure 2: Multi-view of most interesting condition
    for cond in ["dysbiotic_static", "dh_baseline"]:
        if cond in all_data:
            print(f"\nGenerating multi-view for {cond}...")
            fig_3d_multiview(all_data, stress_info, cond)

    print(f"\nAll figures in: {_FIG_DIR}")


if __name__ == "__main__":
    main()
