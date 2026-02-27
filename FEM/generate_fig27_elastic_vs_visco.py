#!/usr/bin/env python3
"""
generate_fig27_elastic_vs_visco.py — Paper Fig 27: Elastic vs viscoelastic 2D σ_vm map
======================================================================================

DH baseline condition. 3 columns (t=10, 60, 300 s).
Top row: viscoelastic σ_vm, Bottom row: difference (elastic - VE).

Output: FEM/figures/paper_final/Fig27_elastic_vs_visco.{png,pdf}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "JAXFEM"))

from material_models import compute_viscoelastic_params_di, compute_E_di, DI_SCALE
from solve_stress_2d import solve_2d_fem, solve_2d_fem_viscoelastic_sls

OUT_DIR = _HERE / "figures" / "paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NX, NY = 15, 15
NU = 0.3


def _make_dh_fields():
    """Generate DH-like 2D fields with spatial DI gradient."""
    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Egg-shaped biofilm mask (simplified)
    cx, cy = 0.5, 0.5
    ax, ay = 0.35, 0.25
    dist = ((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2
    mask = (dist < 1.0).astype(float)

    # DI field: higher in center (nutrient depleted → dysbiotic)
    di_base = 0.55  # DH mean DI
    di_field = di_base + 0.25 * (1.0 - dist) * mask
    di_field = np.clip(di_field * mask, 0, 1)

    # Eigenstrain
    eps_growth = 0.005 * mask

    return di_field, eps_growth, mask


def main():
    di_field, eps_growth, mask = _make_dh_fields()

    # VE parameters from DI
    ve = compute_viscoelastic_params_di(di_field, di_scale=1.0)
    E_inf_field = ve["E_inf"]
    E_1_field = ve["E_1"]
    tau_field = ve["tau"]
    E_0_field = ve["E_0"]

    # Time points for snapshots
    t_snaps = [10.0, 60.0, 300.0]
    t_array = np.array([0.0] + t_snaps)

    # Solve VE
    print("  Solving VE FEM...")
    res_ve = solve_2d_fem_viscoelastic_sls(
        E_inf_field,
        E_1_field,
        tau_field,
        NU,
        eps_growth,
        NX,
        NY,
        t_array,
    )

    # Solve elastic with E_inf (long-term)
    print("  Solving elastic (E_inf)...")
    res_el = solve_2d_fem(E_inf_field, NU, eps_growth, NX, NY)

    n_ex, n_ey = NX - 1, NY - 1

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    extent = [0, 1, 0, 1]

    for col, t_snap in enumerate(t_snaps):
        ti = col + 1  # t_array index (0=initial, 1=10s, ...)

        svm_ve = res_ve["sigma_vm_history"][ti].reshape(n_ex, n_ey)
        svm_el = res_el["sigma_vm"].reshape(n_ex, n_ey)

        # Top: VE σ_vm
        ax = axes[0, col]
        vmax = max(svm_ve.max(), 0.01)
        im = ax.imshow(svm_ve.T, origin="lower", cmap="jet", extent=extent, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label="σ_vm [Pa]")
        ax.set_title(f"VE σ_vm at t = {t_snap:.0f} s\n" f"max = {svm_ve.max():.2f} Pa", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Bottom: difference (elastic_Einf - VE)
        ax = axes[1, col]
        diff = svm_el - svm_ve
        vlim = max(abs(diff).max(), 0.01)
        im = ax.imshow(diff.T, origin="lower", cmap="RdBu_r", extent=extent, vmin=-vlim, vmax=vlim)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Δσ_vm [Pa]")
        ax.set_title(f"Δ = Elastic(E∞) − VE\n" f"mean = {diff.mean():.2f} Pa", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        "Fig 27: Elastic vs Viscoelastic — DH Baseline (2D FEM)\n"
        "Top: VE σ_vm(t), Bottom: difference from elastic E∞ solution",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ["png", "pdf"]:
        path = OUT_DIR / f"Fig27_elastic_vs_visco.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 27 saved: {OUT_DIR / 'Fig27_elastic_vs_visco.png'}")


if __name__ == "__main__":
    main()
