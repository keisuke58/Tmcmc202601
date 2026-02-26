#!/usr/bin/env python3
"""
fem_aniso_analysis.py
=====================
C1: Anisotropic elasticity from biofilm concentration gradients.

Computes ∇φ_pg (P.gingivalis volume-fraction gradient) at each FEM node
from the posterior field data.  The dominant gradient direction is used
as the local stiffness axis for transverse isotropy in Abaqus.

Physical motivation (Wriggers & Junker 2024)
--------------------------------------------
In the Hamilton principle framework the free energy Ψ can depend on the
deformation gradient AND on the current microstructural state.  For a
layered / columnar biofilm the microstructure is anisotropic: species
grow preferentially perpendicular to the substrate (depth direction x).
The observed gradient ∇φ_pg captures this structural anisotropy:

  E_1  = E(DI)             (stiff direction, along gradient)
  E_2  = E_3 = β·E(DI)    (transverse, β = aniso_ratio < 1)

β = 1 → isotropic (standard model)
β < 1 → stiffer in gradient direction (typical biofilm column structure)

Outputs per condition:
  _aniso/{cond}/
    grad_phi.npy          (Nx, Ny, Nz, 3)  gradient field (median sample)
    aniso_strength.npy    (Nx, Ny, Nz)     |∇φ_pg| at each node
    dominant_e1.npy       (3,)             mean dominant direction
    aniso_angle_deg.npy   scalar           angle of e1 w.r.t. x-axis (deg)
    fig_aniso_field.png   gradient magnitude + direction map
    aniso_summary.json    {e1, angle_deg, mean_strength, condition}

  _aniso/
    fig_aniso_cross_condition.png   4-cond comparison
    aniso_table.csv                 per-condition summary

Usage:
  python fem_aniso_analysis.py                  # all 4 conditions
  python fem_aniso_analysis.py --conditions dh_baseline commensal_static
  python fem_aniso_analysis.py --plot-only
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DI_BASE = _HERE / "_di_credible"
_OUT = _HERE / "_aniso"

CONDITIONS = [
    "dh_baseline",
    "commensal_static",
    "commensal_hobic",
    "dysbiotic_static",
]

COND_LABELS = {
    "dh_baseline": "dh-baseline",
    "commensal_static": "Comm. Static",
    "commensal_hobic": "Comm. HOBIC",
    "dysbiotic_static": "Dysb. Static",
}

COND_COLORS = {
    "dh_baseline": "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
}

# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------


def _compute_gradient_field(
    coords: np.ndarray,
    phi_pg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ∇φ_pg on the 3D grid using np.gradient (central differences).

    Parameters
    ----------
    coords  : (N_nodes, 3)  x, y, z coordinates
    phi_pg  : (N_nodes,)    P.g volume fraction (median sample)

    Returns
    -------
    grad_field   : (Nx, Ny, Nz, 3)
    strength     : (Nx, Ny, Nz)   |∇φ_pg|
    x_u, y_u, z_u : unique coordinate arrays
    """
    x_u = np.unique(coords[:, 0])
    y_u = np.unique(coords[:, 1])
    z_u = np.unique(coords[:, 2])
    Nx, Ny, Nz = len(x_u), len(y_u), len(z_u)

    phi_3d = phi_pg.reshape(Nx, Ny, Nz)

    dx = float(x_u[1] - x_u[0]) if Nx > 1 else 1.0
    dy = float(y_u[1] - y_u[0]) if Ny > 1 else 1.0
    dz = float(z_u[1] - z_u[0]) if Nz > 1 else 1.0

    gx = np.gradient(phi_3d, dx, axis=0)
    gy = np.gradient(phi_3d, dy, axis=1)
    gz = np.gradient(phi_3d, dz, axis=2)

    grad_field = np.stack([gx, gy, gz], axis=-1)  # (Nx, Ny, Nz, 3)
    strength = np.sqrt(gx**2 + gy**2 + gz**2)  # (Nx, Ny, Nz)

    return grad_field, strength, x_u, y_u, z_u


def _dominant_direction(
    grad_field: np.ndarray,
    strength: np.ndarray,
    weight_threshold: float = 0.5,
) -> np.ndarray:
    """
    Weighted mean gradient direction (weighted by |∇φ_pg|).
    Only nodes with strength > threshold * max_strength contribute.

    Returns e1 : (3,) unit vector
    """
    s_flat = strength.ravel()
    g_flat = grad_field.reshape(-1, 3)
    thresh = weight_threshold * s_flat.max()
    mask = s_flat > thresh
    if mask.sum() == 0:
        return np.array([1.0, 0.0, 0.0])

    w = s_flat[mask]
    g = g_flat[mask]
    e1 = (w[:, None] * g).sum(axis=0)
    norm = np.linalg.norm(e1)
    return e1 / norm if norm > 0 else np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------


def process_condition(cond: str) -> dict | None:
    """Compute anisotropy field for one condition. Returns summary dict."""
    di_dir = _DI_BASE / cond
    out_dir = _OUT / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load median phi_pg field ──────────────────────────────────────────
    coords_p = di_dir / "coords.npy"
    phi_q_p = di_dir / "phi_pg_stack.npy"  # (n_samples, n_nodes)
    di_q_p = di_dir / "di_quantiles.npy"  # (3, n_nodes) p05/50/95

    if not (coords_p.exists() and phi_q_p.exists()):
        print("  [skip] no B1 data for %s" % cond)
        return None

    coords = np.load(coords_p)  # (N_nodes, 3)
    phi_stack = np.load(phi_q_p)  # (N_samples, N_nodes)
    phi_med = np.median(phi_stack, axis=0)  # (N_nodes,)  median sample

    print("  coords=%s  phi shape=%s" % (coords.shape, phi_stack.shape))

    # ── Gradient field ────────────────────────────────────────────────────
    grad_field, strength, x_u, y_u, z_u = _compute_gradient_field(coords, phi_med)

    np.save(out_dir / "grad_phi.npy", grad_field)
    np.save(out_dir / "aniso_strength.npy", strength)
    np.save(out_dir / "x_u.npy", x_u)
    np.save(out_dir / "y_u.npy", y_u)
    np.save(out_dir / "z_u.npy", z_u)

    # ── Dominant direction ────────────────────────────────────────────────
    e1 = _dominant_direction(grad_field, strength)
    np.save(out_dir / "dominant_e1.npy", e1)

    # Angle between e1 and x-axis (depth direction)
    cos_angle = float(np.clip(np.dot(e1, [1.0, 0.0, 0.0]), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(abs(cos_angle))))
    np.save(out_dir / "aniso_angle_deg.npy", np.array(angle_deg))

    mean_strength = float(strength.mean())

    summary = {
        "condition": cond,
        "e1": e1.tolist(),
        "angle_x_deg": angle_deg,
        "mean_strength": mean_strength,
        "max_strength": float(strength.max()),
    }
    with (out_dir / "aniso_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        "  e1=[%.3f, %.3f, %.3f]  angle_x=%.1f deg  mean_|∇φ|=%.4f"
        % (e1[0], e1[1], e1[2], angle_deg, mean_strength)
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_aniso_field(cond, grad_field, strength, x_u, y_u, z_u, e1, out_dir)

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_aniso_field(
    cond: str,
    grad_field: np.ndarray,
    strength: np.ndarray,
    x_u: np.ndarray,
    y_u: np.ndarray,
    z_u: np.ndarray,
    e1: np.ndarray,
    out_dir: Path,
) -> None:
    """3-panel: XY / XZ / YZ slice of |∇φ_pg| with quiver for direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nx, Ny, Nz = strength.shape
    iz_mid = Nz // 2
    iy_mid = Ny // 2
    ix_mid = Nx // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    slices = [
        (
            strength[:, :, iz_mid],
            grad_field[:, :, iz_mid, 0],
            grad_field[:, :, iz_mid, 1],
            x_u,
            y_u,
            "XY (z=mid)",
        ),
        (
            strength[:, iy_mid, :],
            grad_field[:, iy_mid, :, 0],
            grad_field[:, iy_mid, :, 2],
            x_u,
            z_u,
            "XZ (y=mid)",
        ),
        (
            strength[ix_mid, :, :],
            grad_field[ix_mid, :, :, 1],
            grad_field[ix_mid, :, :, 2],
            y_u,
            z_u,
            "YZ (x=mid)",
        ),
    ]

    for ax, (S, Gx, Gy, u_ax, v_ax, title) in zip(axes, slices):
        im = ax.imshow(
            S.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[u_ax[0], u_ax[-1], v_ax[0], v_ax[-1]],
        )
        # Quiver every 3rd node for readability
        step = max(1, S.shape[0] // 8)
        ug = np.linspace(u_ax[0], u_ax[-1], S.shape[0])
        vg = np.linspace(v_ax[0], v_ax[-1], S.shape[1])
        UG, VG = np.meshgrid(ug[::step], vg[::step], indexing="ij")
        qx = Gx[::step, ::step]
        qy = Gy[::step, ::step]
        norm = np.sqrt(qx**2 + qy**2 + 1e-15)
        ax.quiver(UG, VG, qx / norm, qy / norm, alpha=0.6, color="white", scale=20, width=0.006)
        plt.colorbar(im, ax=ax, label="|∇φ_pg|")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(title.split("(")[0][0], fontsize=9)

    fig.suptitle(
        "C1: Anisotropy Field |∇φ_pg| with Direction Arrows  –  %s\n"
        "Dominant e1=[%.2f, %.2f, %.2f]  angle_x=%.1f deg"
        % (
            COND_LABELS.get(cond, cond),
            e1[0],
            e1[1],
            e1[2],
            float(np.degrees(np.arccos(abs(float(np.clip(np.dot(e1, [1, 0, 0]), -1, 1)))))),
        ),
        fontsize=10,
        fontweight="bold",
    )
    out = out_dir / "fig_aniso_field.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  [plot] %s" % out.name)


def _plot_cross_condition(summaries: list[dict]) -> None:
    """2-panel: angle + mean strength bar chart, 4 conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summaries:
        return

    conds = [s["condition"] for s in summaries]
    angles = [s["angle_x_deg"] for s in summaries]
    strengths = [s["mean_strength"] for s in summaries]
    colors = [COND_COLORS.get(c, "gray") for c in conds]
    labels = [COND_LABELS.get(c, c) for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Angle w.r.t. x (depth) axis
    ax = axes[0]
    bars = ax.bar(range(len(conds)), angles, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(90, color="gray", lw=0.8, ls="--", label="perpendicular to x")
    for bar, v in zip(bars, angles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            "%.1f°" % v,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Angle of dominant ∇φ_pg w.r.t. x-axis (deg)")
    ax.set_title("C1: Anisotropy Direction\n(0° = depth, 90° = lateral)")
    ax.set_ylim(0, 95)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Mean gradient strength
    ax = axes[1]
    bars = ax.bar(
        range(len(conds)), strengths, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5
    )
    for bar, v in zip(bars, strengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            "%.4f" % v,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean |∇φ_pg| (domain average)")
    ax.set_title("C1: Anisotropy Strength\n(larger = more heterogeneous)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "C1: Gradient-Based Anisotropy Analysis\n"
        "(∇φ_Pg direction → local stiffness axis for Abaqus)",
        fontsize=12,
        fontweight="bold",
    )
    out = _OUT / "fig_aniso_cross_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


def _save_table(summaries: list[dict]) -> None:
    path = _OUT / "aniso_table.csv"
    if not summaries:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "angle_x_deg",
                "mean_strength",
                "max_strength",
                "e1_x",
                "e1_y",
                "e1_z",
            ],
        )
        w.writeheader()
        for s in summaries:
            e1 = s["e1"]
            w.writerow(
                {
                    "condition": s["condition"],
                    "angle_x_deg": "%.3f" % s["angle_x_deg"],
                    "mean_strength": "%.6f" % s["mean_strength"],
                    "max_strength": "%.6f" % s["max_strength"],
                    "e1_x": "%.4f" % e1[0],
                    "e1_y": "%.4f" % e1[1],
                    "e1_z": "%.4f" % e1[2],
                }
            )
    print("[save] %s" % path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    summaries = []

    if not args.plot_only:
        for cond in args.conditions:
            print("\n[%s]" % cond)
            s = process_condition(cond)
            if s:
                summaries.append(s)
        _save_table(summaries)
    else:
        for cond in args.conditions:
            sj = _OUT / cond / "aniso_summary.json"
            if sj.exists():
                with sj.open() as f:
                    summaries.append(json.load(f))

    _plot_cross_condition(summaries)

    # Print Abaqus-ready orientation summary
    if summaries:
        print("\n── Abaqus ORIENTATION args (for run_aniso_comparison.py) ──")
        for s in summaries:
            e1 = s["e1"]
            print(
                "  %-20s  e1=[%+.3f,%+.3f,%+.3f]  angle=%.1f deg"
                % (s["condition"], e1[0], e1[1], e1[2], s["angle_x_deg"])
            )

    print("\n[done] %s" % _OUT)


if __name__ == "__main__":
    main()
