#!/usr/bin/env python3
"""
compare_3models_3d.py — DEM / PI-FNO / PINN 3D elasticity comparison.

Loads all 3 trained models, evaluates on the same E(x,y,z) field,
and produces a comparison figure.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

jax.config.update("jax_enable_x64", False)

# Import model classes
from dem_elasticity_3d import (
    ElasticityNetwork,
    generate_E_field_3d as dem_gen_E,
    W,
    H,
    D,
    E_MAX,
    E_MIN,
    P_APPLIED,
)
from pinn_elasticity_3d import (
    ElasticityPINN3D,
    pde_residual_3d,
)
from pifo_elasticity_3d import (
    FNO3d,
    get_grid,
)


def load_models():
    """Load all 3 trained models."""
    key = jr.PRNGKey(0)
    k1, k2, k3 = jr.split(key, 3)

    models = {}

    # 1) DEM
    dem = ElasticityNetwork(key=k1)
    try:
        dem = eqx.tree_deserialise_leaves("dem_3d.eqx", dem)
        models["DEM"] = dem
        print("[OK] DEM loaded (dem_3d.eqx)")
    except Exception as e:
        print(f"[SKIP] DEM: {e}")

    # 2) PI-FNO
    fno = FNO3d(in_channels=4, out_channels=3, modes=8, width=20, key=k2)
    try:
        fno = eqx.tree_deserialise_leaves("pifno_3d.eqx", fno)
        models["PI-FNO"] = fno
        print("[OK] PI-FNO loaded (pifno_3d.eqx)")
    except Exception as e:
        print(f"[SKIP] PI-FNO: {e}")

    # 3) PINN
    pinn = ElasticityPINN3D(key=k3)
    try:
        pinn = eqx.tree_deserialise_leaves("pinn_checkpoints_3d/best.eqx", pinn)
        models["PINN"] = pinn
        print("[OK] PINN loaded (pinn_checkpoints_3d/best.eqx)")
    except Exception as e:
        print(f"[SKIP] PINN: {e}")

    return models


def predict_pointwise(model, X, Y, Z, E_norm):
    """Predict displacement for point-wise models (DEM / PINN)."""

    def predict_single(x, y, z, e):
        return model(x, y, z, e)

    return jax.vmap(jax.vmap(predict_single))(X, Y, Z, E_norm)


def predict_fno(model, E_grid_3d, resolution):
    """Predict displacement for PI-FNO (grid-based)."""
    X, Y, Z = get_grid(resolution)
    inputs = jnp.stack([E_grid_3d / E_MAX, X, Y, Z], axis=0)  # (4, N, N, N)
    u = model(inputs)  # (3, N, N, N)
    return u, X, Y, Z


def compute_analytical_1d(E_vals_1d, y_vals):
    """
    Analytical solution for 1D column under uniform pressure P.
    u_y(y) = integral_y^H P/E dy  (compression, top-loaded)
    For uniform E: u_y(y) = P*(H-y)/E
    For varying E: integrate numerically.
    """
    ny = len(y_vals)
    uy = np.zeros(ny)
    for i in range(ny - 1, -1, -1):
        # Integrate from y to H
        dy_seg = np.diff(y_vals[i:])
        E_seg = 0.5 * (E_vals_1d[i:-1] + E_vals_1d[i + 1 :])
        if len(dy_seg) > 0:
            uy[i] = np.sum(P_APPLIED / E_seg * dy_seg)
    return uy


def main():
    models = load_models()
    if not models:
        print("No models loaded!")
        return

    # ============================================================
    # Common test E field (use same seed for all)
    # ============================================================
    E_fn = dem_gen_E(jr.PRNGKey(99))

    # Slice parameters
    z_slice = D / 2
    nx, ny = 80, 40
    x = jnp.linspace(0, W, nx)
    y = jnp.linspace(0, H, ny)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.full_like(X, z_slice)

    E_vals = jax.vmap(jax.vmap(E_fn))(X, Y, Z)
    E_norm = E_vals / E_MAX

    # ============================================================
    # Predict
    # ============================================================
    results = {}
    param_counts = {}

    for name, model in models.items():
        n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        param_counts[name] = n_params

        if name in ("DEM", "PINN"):
            U = predict_pointwise(model, X, Y, Z, E_norm)  # (ny, nx, 3)
            results[name] = {
                "Ux": np.array(U[:, :, 0]),
                "Uy": np.array(U[:, :, 1]),
                "Uz": np.array(U[:, :, 2]),
            }
        elif name == "PI-FNO":
            # PI-FNO works on full 3D grid
            res = 32
            X3, Y3, Z3 = get_grid(res)
            E3 = jax.vmap(
                jax.vmap(jax.vmap(E_fn)),
            )(
                X3, Y3, Z3
            )  # (res, res, res)
            u_3d, _, _, _ = predict_fno(model, E3, res)
            # Extract z-slice (nearest index)
            iz = res // 2
            results[name] = {
                "Ux": np.array(u_3d[0, :, :, iz]),
                "Uy": np.array(u_3d[1, :, :, iz]),
                "Uz": np.array(u_3d[2, :, :, iz]),
                "X": np.array(X3[:, :, iz]),
                "Y": np.array(Y3[:, :, iz]),
            }

    # ============================================================
    # 1D column profile (x = W/2, z = D/2)
    # ============================================================
    ny_prof = 100
    y_prof = np.linspace(0, H, ny_prof)
    x_mid = W / 2
    z_mid = D / 2
    E_prof = np.array([float(E_fn(x_mid, yy, z_mid)) for yy in y_prof])
    uy_analytical = compute_analytical_1d(E_prof, y_prof)

    profiles = {}
    for name, model in models.items():
        if name in ("DEM", "PINN"):
            uy_vals = []
            for yy in y_prof:
                u = model(
                    jnp.float32(x_mid),
                    jnp.float32(yy),
                    jnp.float32(z_mid),
                    jnp.float32(E_fn(x_mid, yy, z_mid) / E_MAX),
                )
                uy_vals.append(float(u[1]))
            profiles[name] = np.array(uy_vals)
        elif name == "PI-FNO":
            # Extract column from 3D prediction
            res = 32
            ix_mid = res // 2
            iz_mid = res // 2
            u_col = np.array(u_3d[1, ix_mid, :, iz_mid])
            y_col = np.linspace(0, H, res)
            profiles[name] = (y_col, u_col)

    # ============================================================
    # PDE residual check (pointwise models only)
    # ============================================================
    n_test = 200
    key_test = jr.PRNGKey(777)
    k1, k2, k3 = jr.split(key_test, 3)
    x_t = jr.uniform(k1, (n_test,), minval=0.05 * W, maxval=0.95 * W)
    y_t = jr.uniform(k2, (n_test,), minval=0.05 * H, maxval=0.95 * H)
    z_t = jr.uniform(k3, (n_test,), minval=0.05 * D, maxval=0.95 * D)
    E_t = jax.vmap(E_fn)(x_t, y_t, z_t)

    residuals = {}
    for name in ("DEM", "PINN"):
        if name not in models:
            continue
        model = models[name]

        def single_res(x, y, z, E):
            div_sig, _, _ = pde_residual_3d(model, x, y, z, E)
            return jnp.sum(div_sig**2)

        res_vals = jax.vmap(single_res)(x_t, y_t, z_t, E_t)
        residuals[name] = np.array(res_vals)

    # ============================================================
    # Figure
    # ============================================================
    n_models = len(results)
    colors = {"DEM": "#2196F3", "PI-FNO": "#FF9800", "PINN": "#4CAF50"}

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "3D Elasticity: DEM vs PI-FNO vs PINN Comparison\n"
        f"Domain {W}x{H}x{D} mm, E=[{E_MIN},{E_MAX}] Pa, "
        f"P={P_APPLIED} Pa, slice z={z_slice:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    # --- Row 1: E field + Uy for each model ---
    gs = fig.add_gridspec(
        3, n_models + 1, hspace=0.35, wspace=0.3, left=0.06, right=0.96, top=0.90, bottom=0.06
    )

    # E field
    ax_e = fig.add_subplot(gs[0, 0])
    c = ax_e.contourf(np.array(X), np.array(Y), np.array(E_vals), levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax_e, label="E [Pa]")
    ax_e.set_title("Stiffness E(x,y)")
    ax_e.set_ylabel("y [mm]")
    ax_e.set_aspect("auto")

    # Uy for each model
    for col, (name, data) in enumerate(results.items(), 1):
        ax = fig.add_subplot(gs[0, col])
        if name == "PI-FNO":
            Xp, Yp = data["X"], data["Y"]
        else:
            Xp, Yp = np.array(X), np.array(Y)
        c = ax.contourf(Xp, Yp, data["Uy"], levels=20, cmap="coolwarm")
        plt.colorbar(c, ax=ax, label="Uy [mm]")
        ax.set_title(f"{name}  Uy  ({param_counts[name]:,} params)")
        ax.set_aspect("auto")

    # --- Row 2: 1D Profile + Residual comparison ---
    ax_prof = fig.add_subplot(gs[1, :2])
    ax_prof.plot(y_prof * 1000, uy_analytical * 1000, "k--", lw=2, label="Analytical (1D approx)")
    for name, prof in profiles.items():
        if isinstance(prof, tuple):
            yy, uu = prof
            ax_prof.plot(yy * 1000, uu * 1000, "o-", color=colors[name], ms=3, label=name)
        else:
            ax_prof.plot(y_prof * 1000, prof * 1000, "-", color=colors[name], lw=2, label=name)
    ax_prof.set_xlabel("y [um]")
    ax_prof.set_ylabel("Uy [um]")
    ax_prof.set_title("Vertical Displacement Profile (x=W/2, z=D/2)")
    ax_prof.legend()
    ax_prof.grid(True, alpha=0.3)

    # Residual histogram
    ax_res = fig.add_subplot(gs[1, 2:])
    for name, rv in residuals.items():
        ax_res.hist(
            np.log10(rv + 1e-30),
            bins=30,
            alpha=0.6,
            color=colors[name],
            label=f"{name} (med={np.median(rv):.2e})",
        )
    ax_res.set_xlabel("log10(PDE residual)")
    ax_res.set_ylabel("Count")
    ax_res.set_title("PDE Residual Distribution (interior)")
    ax_res.legend()
    ax_res.grid(True, alpha=0.3)

    # --- Row 3: Deformed mesh for each model + summary table ---
    for col, (name, data) in enumerate(results.items()):
        ax = fig.add_subplot(gs[2, col])
        if name == "PI-FNO":
            Xp, Yp = data["X"], data["Y"]
        else:
            Xp, Yp = np.array(X), np.array(Y)
        Ux, Uy = data["Ux"], data["Uy"]
        U_mag = np.sqrt(Ux**2 + Uy**2 + data["Uz"] ** 2)

        max_u = np.max(np.abs(U_mag)) + 1e-20
        scale = 0.15 / max_u

        Xd = Xp + Ux * scale
        Yd = Yp + Uy * scale

        c = ax.contourf(Xd, Yd, U_mag, levels=20, cmap="magma")
        plt.colorbar(c, ax=ax, label="|U| [mm]")
        # Grid lines
        step = max(1, Xd.shape[0] // 10)
        ax.plot(Xd[::step].T, Yd[::step].T, "k-", alpha=0.15, lw=0.5)
        ax.plot(Xd.T[::step].T, Yd.T[::step].T, "k-", alpha=0.15, lw=0.5)

        ax.set_title(f"{name} Deformed (scale {scale:.0f}x)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("auto")

    # Summary table
    ax_tab = fig.add_subplot(gs[2, n_models])
    ax_tab.axis("off")
    table_data = []
    headers = ["Model", "Params", "Max |Uy|", "Med Res"]
    for name in results:
        max_uy = np.max(np.abs(results[name]["Uy"]))
        med_res = np.median(residuals[name]) if name in residuals else "N/A"
        if isinstance(med_res, float):
            med_str = f"{med_res:.2e}"
        else:
            med_str = str(med_res)
        table_data.append([name, f"{param_counts[name]:,}", f"{max_uy:.4e}", med_str])
    tab = ax_tab.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.2, 1.5)
    ax_tab.set_title("Summary", fontweight="bold")

    out_path = "comparison_3models_3d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison figure: {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    for name in results:
        max_uy = np.max(np.abs(results[name]["Uy"]))
        med_res = np.median(residuals[name]) if name in residuals else None
        print(
            f"  {name:8s}: params={param_counts[name]:>8,}  "
            f"max|Uy|={max_uy:.4e}  "
            f"med_res={med_res:.2e}"
            if med_res
            else f"  {name:8s}: params={param_counts[name]:>8,}  "
            f"max|Uy|={max_uy:.4e}  "
            f"med_res=N/A (grid-based)"
        )


if __name__ == "__main__":
    main()
