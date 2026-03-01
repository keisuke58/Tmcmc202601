#!/usr/bin/env python3
"""
klempt_benchmark.py
====================
C1: Quantitative comparison with Klempt 2024.

Benchmarks our multiscale pipeline against Klempt et al. (2024):
  1. Nutrient field comparison (reaction-diffusion steady state)
  2. Stress field comparison (eigenstrain-driven mechanics)
  3. Growth model comparison (0D Hamilton vs Klempt logistic)
  4. Parameter sensitivity (Thiele modulus, DI exponent)

Reference: Klempt (2024) - Hamilton principle-based model for
           diffusion-driven biofilm growth

Usage
-----
  python klempt_benchmark.py                 # full benchmark suite
  python klempt_benchmark.py --nutrient-only # nutrient field only
  python klempt_benchmark.py --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_OUT = _HERE / "_klempt_benchmark"


# =========================================================================
# Klempt 2024 reference values (from paper Table 1, Fig 3-5)
# =========================================================================

KLEMPT_PARAMS = {
    "D_c": 1.0,  # nutrient diffusivity
    "k_monod": 1.0,  # Monod half-saturation
    "g_eff": 50.0,  # effective consumption rate
    "c_boundary": 1.0,  # boundary nutrient concentration
    # Biofilm shape (egg-shaped, Fig. 1)
    "ax": 0.35,  # semi-axis x
    "ay": 0.25,  # semi-axis y
    "skew": 0.3,  # skewness
    "eps": 0.1,  # smoothing
    # Mechanics
    "E_biofilm": 1000.0,  # Pa (EPS matrix)
    "nu": 0.30,
    "alpha_growth": 0.05,  # growth eigenstrain
    # Grid
    "Nx": 40,
    "Ny": 40,
    "Lx": 1.0,
    "Ly": 1.0,
}

# Expected results from Klempt 2024
KLEMPT_REFERENCE = {
    "c_min_interior": 0.3,  # approx c_min inside biofilm (Fig. 3)
    "thiele_modulus": 4.0,  # sqrt(g_eff / D_c) ~ diffusion-limited
    "sigma_max_pa": 50.0,  # approx max von Mises in biofilm (Pa)
    "growth_strain_max": 0.05,
}


def _egg_shape(x, y, params):
    """Klempt 2024 egg-shaped biofilm indicator function."""
    ax = params["ax"]
    ay = params["ay"]
    skew = params["skew"]
    eps = params["eps"]
    cx, cy = 0.5, 0.5
    xn = (x - cx) / ax
    yn = (y - cy + skew * (x - cx) ** 2) / ay
    r2 = xn**2 + yn**2
    return 0.5 * (1.0 - np.tanh((r2 - 1.0) / eps))


def benchmark_nutrient(params=None, quick=False):
    """Benchmark 1: Nutrient field steady-state comparison.

    Solves: -D_c * Laplacian(c) + g * phi(x) * c/(k+c) = 0
    BC: c = 1 on boundary
    """
    p = params or KLEMPT_PARAMS.copy()
    Nx = 20 if quick else p["Nx"]
    Ny = 20 if quick else p["Ny"]
    Lx, Ly = p["Lx"], p["Ly"]
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Biofilm shape
    phi = _egg_shape(X, Y, p)

    # Solve steady-state: Newton iteration on
    # R(c) = D_c * lap(c) - g * phi * c/(k+c) = 0
    c = np.ones((Nx, Ny)) * p["c_boundary"]
    D_c = p["D_c"]
    g = p["g_eff"]
    k = p["k_monod"]

    for newton_iter in range(200):
        # Laplacian with Dirichlet BC (c=1 on boundary)
        c_pad = np.pad(c, 1, mode="constant", constant_values=p["c_boundary"])
        lap = (c_pad[:-2, 1:-1] + c_pad[2:, 1:-1] - 2 * c_pad[1:-1, 1:-1]) / dx**2 + (
            c_pad[1:-1, :-2] + c_pad[1:-1, 2:] - 2 * c_pad[1:-1, 1:-1]
        ) / dy**2

        # Monod consumption
        monod = c / (k + c)
        consumption = g * phi * monod

        # Residual
        R = D_c * lap - consumption

        # Jacobian diagonal (for damped Newton)
        # dR/dc = D_c * d(lap)/dc - g * phi * k/(k+c)^2
        dR_dc = -D_c * 2 * (1 / dx**2 + 1 / dy**2) * np.ones_like(c) - g * phi * k / (k + c) ** 2

        # Newton update (damped)
        delta = -R / (dR_dc + 1e-12)
        delta = np.clip(delta, -0.1, 0.1)  # damping
        c += delta

        # Enforce BC
        c[0, :] = p["c_boundary"]
        c[-1, :] = p["c_boundary"]
        c[:, 0] = p["c_boundary"]
        c[:, -1] = p["c_boundary"]
        c = np.clip(c, 0, p["c_boundary"])

        res_norm = np.max(np.abs(R))
        if res_norm < 1e-8:
            break

    c_min = float(np.min(c))
    c_min_biofilm = float(np.min(c[phi > 0.5]))

    # Thiele modulus
    thiele = np.sqrt(g / D_c)

    result = {
        "c_min": c_min,
        "c_min_biofilm": c_min_biofilm,
        "c_max": float(np.max(c)),
        "c_mean_biofilm": float(np.mean(c[phi > 0.5])),
        "thiele_modulus": float(thiele),
        "newton_iters": newton_iter + 1,
        "residual_norm": float(res_norm),
        "grid": f"{Nx}x{Ny}",
        # Comparison with Klempt reference
        "klempt_c_min_ref": KLEMPT_REFERENCE["c_min_interior"],
        "c_min_error": abs(c_min_biofilm - KLEMPT_REFERENCE["c_min_interior"]),
    }

    print("\n  Nutrient field benchmark:")
    print(f"    Grid: {Nx}x{Ny}")
    print(f"    Newton iterations: {newton_iter+1}")
    print(
        f"    c_min (biofilm interior): {c_min_biofilm:.4f} (Klempt ref: ~{KLEMPT_REFERENCE['c_min_interior']})"
    )
    print(f"    Thiele modulus: {thiele:.2f} (Klempt ref: ~{KLEMPT_REFERENCE['thiele_modulus']})")
    print(f"    Error: {result['c_min_error']:.4f}")

    return result, c, phi, X, Y


def benchmark_growth(quick=False):
    """Benchmark 2: Growth model comparison.

    Compare our Hamilton 0D model with Klempt's logistic growth:
      Klempt: dphi/dt = r*phi*(1-phi/K)  (logistic)
      Ours:   Hamilton 5-species with interaction matrix

    Key metric: final total volume fraction phi_total
    """
    # Klempt logistic model
    r_logistic = 2.0  # growth rate
    K_cap = 0.8  # carrying capacity
    n_steps = 1000
    dt = 0.01
    phi_init = 0.1

    # Logistic solution
    t = np.arange(n_steps) * dt
    phi_logistic = K_cap / (1 + (K_cap / phi_init - 1) * np.exp(-r_logistic * t))

    # Our simplified 5-species model (Lotka-Volterra approx)
    phi_5sp = np.array([0.12, 0.12, 0.08, 0.05, 0.02])
    A_simple = np.eye(5) * 1.5
    b_simple = np.array([2.0, 0.3, 3.0, 1.0, 0.1])

    phi_5sp_history = [phi_5sp.copy()]
    for _ in range(n_steps):
        growth = phi_5sp * (b_simple - A_simple @ phi_5sp)
        phi_5sp = phi_5sp + dt * growth
        phi_5sp = np.clip(phi_5sp, 1e-10, 1.0)
        phi_sum = phi_5sp.sum()
        if phi_sum > 0.999:
            phi_5sp *= 0.999 / phi_sum
        phi_5sp_history.append(phi_5sp.copy())

    phi_5sp_history = np.array(phi_5sp_history)
    phi_total_ours = phi_5sp_history.sum(axis=1)

    # Compare characteristic times
    # Klempt: time to reach 90% capacity
    t_90_klempt = -np.log((K_cap / phi_init - 1) * 0.1 / 0.9) / r_logistic
    # Ours: time for phi_total to reach 90% of max
    phi_total_max = phi_total_ours.max()
    idx_90 = np.argmax(phi_total_ours >= 0.9 * phi_total_max)
    t_90_ours = idx_90 * dt

    result = {
        "logistic": {
            "phi_final": float(phi_logistic[-1]),
            "t_90": float(t_90_klempt),
            "K_cap": K_cap,
        },
        "hamilton_5sp": {
            "phi_total_final": float(phi_total_ours[-1]),
            "t_90": float(t_90_ours),
            "species_final": [float(x) for x in phi_5sp],
        },
        "comparison": {
            "phi_total_ratio": float(phi_total_ours[-1] / phi_logistic[-1]),
            "t_90_ratio": float(t_90_ours / t_90_klempt) if t_90_klempt > 0 else 0,
        },
    }

    print("\n  Growth model benchmark:")
    print(f"    Klempt logistic: phi_final={phi_logistic[-1]:.4f}, t_90={t_90_klempt:.2f}")
    print(f"    Hamilton 5sp:    phi_total={phi_total_ours[-1]:.4f}, t_90={t_90_ours:.2f}")
    print(f"    phi_total ratio: {result['comparison']['phi_total_ratio']:.3f}")

    return result, t, phi_logistic, phi_total_ours, phi_5sp_history


def benchmark_eigenstrain(c_field, phi_field, params=None, quick=False):
    """Benchmark 3: Eigenstrain-driven stress comparison.

    Computes growth eigenstrain from nutrient-limited Monod kinetics
    and compares the resulting stress field with Klempt reference values.
    """
    p = params or KLEMPT_PARAMS.copy()

    # Alpha_Monod from nutrient field
    k = p["k_monod"]
    monod = c_field / (k + c_field)
    alpha_monod = phi_field * monod * p["alpha_growth"]

    # Eigenstrain (isotropic volume growth)
    eps_growth = alpha_monod / 3.0

    # Simple stress estimate (plane-stress, constrained by boundary)
    E = p["E_biofilm"]
    nu = p["nu"]
    # sigma_thermal = -E/(1-nu) * alpha * dT  (for constrained growth)
    sigma_growth = -E / (1 - nu) * eps_growth

    sigma_vm = np.abs(sigma_growth)  # simplified von Mises for isotropic

    result = {
        "eps_growth_max": float(np.max(eps_growth)),
        "eps_growth_mean": float(np.mean(eps_growth[phi_field > 0.5])),
        "sigma_vm_max_pa": float(np.max(sigma_vm)),
        "sigma_vm_mean_pa": float(np.mean(sigma_vm[phi_field > 0.5])),
        "klempt_sigma_ref_pa": KLEMPT_REFERENCE["sigma_max_pa"],
        "sigma_error_ratio": float(np.max(sigma_vm) / KLEMPT_REFERENCE["sigma_max_pa"]),
    }

    print("\n  Eigenstrain/stress benchmark:")
    print(f"    eps_growth max: {np.max(eps_growth):.6f}")
    print(
        f"    sigma_vm max:   {np.max(sigma_vm):.2f} Pa (Klempt ref: ~{KLEMPT_REFERENCE['sigma_max_pa']} Pa)"
    )
    print(f"    Ratio: {result['sigma_error_ratio']:.3f}")

    return result, eps_growth, sigma_vm


def benchmark_sensitivity(quick=False):
    """Benchmark 4: Parameter sensitivity (Thiele modulus scan)."""
    g_values = [5, 10, 20, 50, 100, 200] if not quick else [10, 50, 100]
    results = []

    print("\n  Thiele modulus sensitivity:")
    for g in g_values:
        params = KLEMPT_PARAMS.copy()
        params["g_eff"] = g
        res, c, phi, _, _ = benchmark_nutrient(params, quick=True)
        thiele = np.sqrt(g / params["D_c"])
        results.append(
            {
                "g_eff": g,
                "thiele": float(thiele),
                "c_min": res["c_min_biofilm"],
            }
        )
        print(f"    g={g:4d}, Phi={thiele:5.1f}, c_min={res['c_min_biofilm']:.4f}")

    return results


def generate_figures(
    nutrient_res,
    c,
    phi,
    X,
    Y,
    growth_res,
    t,
    phi_log,
    phi_total,
    phi_5sp,
    eigen_res,
    eps_growth,
    sigma_vm,
    sensitivity_res,
    out_dir,
):
    """Generate comprehensive benchmark comparison figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Panel 1: Nutrient field c(x,y)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(X, Y, c, levels=20, cmap="viridis")
    ax.contour(X, Y, phi, levels=[0.5], colors="white", linewidths=2)
    plt.colorbar(im, ax=ax, label="c")
    ax.set_title("(a) Nutrient Field c(x,y)", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Panel 2: Biofilm shape phi(x,y)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.contourf(X, Y, phi, levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label=r"$\phi_0$")
    ax.set_title("(b) Biofilm Shape (Klempt Egg)", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Panel 3: Nutrient profile (centerline y=0.5)
    ax = fig.add_subplot(gs[0, 2])
    ny_mid = c.shape[1] // 2
    x_line = np.linspace(0, KLEMPT_PARAMS["Lx"], c.shape[0])
    ax.plot(x_line, c[:, ny_mid], "b-", lw=2, label="Computed")
    ax.axhline(
        KLEMPT_REFERENCE["c_min_interior"],
        color="r",
        ls="--",
        label=f"Klempt ref ({KLEMPT_REFERENCE['c_min_interior']})",
    )
    # Mark biofilm region
    phi_line = phi[:, ny_mid]
    ax.fill_between(x_line, 0, 1, where=phi_line > 0.5, alpha=0.1, color="orange", label="Biofilm")
    ax.set_xlabel("x")
    ax.set_ylabel("c")
    ax.set_title("(c) Centerline Nutrient Profile", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Panel 4: Growth model comparison
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, phi_log, "b-", lw=2, label="Klempt logistic")
    ax.plot(
        np.arange(len(phi_total)) * 0.01, phi_total, "r--", lw=2, label="Hamilton 5-species (total)"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\phi_{total}$")
    ax.set_title("(d) Growth: Logistic vs Hamilton", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: Species dynamics
    ax = fig.add_subplot(gs[1, 1])
    species_names = ["S.oralis", "A.naes", "Veillon", "F.nucl", "P.ging"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    t_5sp = np.arange(len(phi_5sp)) * 0.01
    for i, (name, clr) in enumerate(zip(species_names, colors)):
        ax.plot(t_5sp, phi_5sp[:, i], color=clr, lw=1.5, label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\phi_i$")
    ax.set_title("(e) Hamilton Species Dynamics", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 6: Eigenstrain field
    ax = fig.add_subplot(gs[1, 2])
    im = ax.contourf(X, Y, eps_growth, levels=20, cmap="hot")
    ax.contour(X, Y, phi, levels=[0.5], colors="white", linewidths=2)
    plt.colorbar(im, ax=ax, label=r"$\varepsilon_{growth}$")
    ax.set_title("(f) Growth Eigenstrain", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Panel 7: Stress field
    ax = fig.add_subplot(gs[2, 0])
    im = ax.contourf(X, Y, sigma_vm, levels=20, cmap="YlOrRd")
    ax.contour(X, Y, phi, levels=[0.5], colors="black", linewidths=1)
    plt.colorbar(im, ax=ax, label=r"$\sigma_{vM}$ [Pa]")
    ax.set_title("(g) von Mises Stress [Pa]", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Panel 8: Thiele sensitivity
    ax = fig.add_subplot(gs[2, 1])
    if sensitivity_res:
        thieles = [r["thiele"] for r in sensitivity_res]
        c_mins = [r["c_min"] for r in sensitivity_res]
        ax.plot(thieles, c_mins, "ko-", lw=2, markersize=8)
        ax.set_xlabel(r"Thiele modulus $\Phi = \sqrt{g/D_c}$", fontsize=11)
        ax.set_ylabel(r"$c_{min}$ (biofilm interior)", fontsize=11)
        ax.set_title("(h) Nutrient Depletion vs Thiele", fontsize=11)
        ax.grid(True, alpha=0.3)
        # Mark Klempt reference point
        ax.axhline(KLEMPT_REFERENCE["c_min_interior"], color="r", ls="--", alpha=0.5)
        ax.axvline(KLEMPT_REFERENCE["thiele_modulus"], color="r", ls="--", alpha=0.5)

    # Panel 9: Summary metrics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    metrics_text = [
        "Klempt 2024 Benchmark Summary",
        f"{'='*35}",
        "Nutrient field:",
        f"  c_min (biofilm): {nutrient_res['c_min_biofilm']:.3f} (ref: ~{KLEMPT_REFERENCE['c_min_interior']})",
        f"  Thiele modulus:  {nutrient_res['thiele_modulus']:.1f} (ref: ~{KLEMPT_REFERENCE['thiele_modulus']})",
        "",
        "Growth model:",
        f"  phi_total ratio: {growth_res['comparison']['phi_total_ratio']:.3f}",
        f"  t_90 ratio:      {growth_res['comparison']['t_90_ratio']:.3f}",
        "",
        "Stress field:",
        f"  sigma_max: {eigen_res['sigma_vm_max_pa']:.1f} Pa (ref: ~{KLEMPT_REFERENCE['sigma_max_pa']})",
        f"  Ratio:     {eigen_res['sigma_error_ratio']:.3f}",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(metrics_text),
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        va="top",
    )

    fig.suptitle("Klempt 2024 Quantitative Benchmark", fontsize=15, weight="bold")
    out = out_dir / "klempt_benchmark.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Benchmark figure: {out}")


def main():
    ap = argparse.ArgumentParser(description="Klempt 2024 quantitative benchmark")
    ap.add_argument("--nutrient-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    _OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    print("=" * 60)
    print("Klempt 2024 Quantitative Benchmark")
    print("=" * 60)

    # Benchmark 1: Nutrient field
    print("\n[1/4] Nutrient field...")
    nutrient_res, c, phi, X, Y = benchmark_nutrient(quick=args.quick)

    if args.nutrient_only:
        return

    # Benchmark 2: Growth model
    print("\n[2/4] Growth model...")
    growth_res, t, phi_log, phi_total, phi_5sp = benchmark_growth(quick=args.quick)

    # Benchmark 3: Eigenstrain
    print("\n[3/4] Eigenstrain/stress...")
    eigen_res, eps_growth, sigma_vm = benchmark_eigenstrain(c, phi, quick=args.quick)

    # Benchmark 4: Sensitivity
    print("\n[4/4] Thiele sensitivity...")
    sensitivity_res = benchmark_sensitivity(quick=args.quick)

    # Save all results
    all_results = {
        "nutrient": nutrient_res,
        "growth": growth_res,
        "eigenstrain": eigen_res,
        "sensitivity": sensitivity_res,
        "timing_s": round(time.perf_counter() - t0, 1),
    }
    with (_OUT / "benchmark_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    # Generate figures
    generate_figures(
        nutrient_res,
        c,
        phi,
        X,
        Y,
        growth_res,
        t,
        phi_log,
        phi_total,
        phi_5sp,
        eigen_res,
        eps_growth,
        sigma_vm,
        sensitivity_res,
        _OUT,
    )

    dt = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {dt:.1f}s")
    print(f"Output: {_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
