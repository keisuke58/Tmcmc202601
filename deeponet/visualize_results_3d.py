import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from dem_elasticity_3d import ElasticityNetwork, W, H, D, E_MAX, generate_E_field_3d

# Config
jax.config.update("jax_enable_x64", False)


def visualize():
    # Load Model
    key = jr.PRNGKey(0)
    model = ElasticityNetwork(key=key)
    try:
        model = eqx.tree_deserialise_leaves("dem_3d.eqx", model)
        print("Loaded dem_3d.eqx")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # Generate Grid for Visualization (Slice at z = D/2)
    nx, ny = 100, 20
    x = jnp.linspace(0, W, nx)
    y = jnp.linspace(0, H, ny)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.full_like(X, D / 2)

    # E field for this slice
    E_fn = generate_E_field_3d(jr.PRNGKey(99))
    E_vals = jax.vmap(jax.vmap(E_fn))(X, Y, Z)
    E_norm = E_vals / E_MAX

    # Predict Displacement
    def predict(x, y, z, e):
        return model(x, y, z, e)

    U = jax.vmap(jax.vmap(predict))(X, Y, Z, E_norm)
    # U shape: (ny, nx, 3)

    Ux = U[:, :, 0]
    Uy = U[:, :, 1]
    Uz = U[:, :, 2]
    U_mag = jnp.sqrt(Ux**2 + Uy**2 + Uz**2)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. E field (Stiffness)
    c1 = axes[0].contourf(X, Y, E_vals, levels=20, cmap="viridis")
    plt.colorbar(c1, ax=axes[0], label="Stiffness E [Pa]")
    axes[0].set_title("Stiffness Distribution (z=D/2)")
    axes[0].set_ylabel("y [mm]")

    # 2. Vertical Displacement Uy
    c2 = axes[1].contourf(X, Y, Uy, levels=20, cmap="coolwarm")
    plt.colorbar(c2, ax=axes[1], label="Uy [mm]")
    axes[1].set_title("Vertical Displacement (Uy)")
    axes[1].set_ylabel("y [mm]")

    # 3. Deformed Shape (scaled)
    scale = 0.2 / jnp.max(jnp.abs(U_mag))  # Scale for visibility
    X_def = X + Ux * scale
    Y_def = Y + Uy * scale

    axes[2].plot(X_def.T, Y_def.T, "k-", alpha=0.3, linewidth=0.5)
    axes[2].plot(X_def, Y_def, "k-", alpha=0.3, linewidth=0.5)
    c3 = axes[2].contourf(X_def, Y_def, U_mag, levels=20, cmap="magma")
    plt.colorbar(c3, ax=axes[2], label="|U| [mm]")
    axes[2].set_title(f"Deformed Mesh (Scale factor: {scale:.1f})")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("y [mm]")

    plt.tight_layout()
    plt.savefig("result_dem_3d_slice.png", dpi=150)
    print("Saved visualization to result_dem_3d_slice.png")


if __name__ == "__main__":
    visualize()
