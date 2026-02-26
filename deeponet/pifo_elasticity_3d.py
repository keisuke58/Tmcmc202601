#!/usr/bin/env python3
"""
pifo_elasticity_3d.py — Physics-Informed Fourier Neural Operator (PI-FNO) for 3D elasticity.

Approximates the operator:
    E(x,y,z) ↦ u(x,y,z)

Solves:
    ∇·σ = 0  in Ω
    u = 0    on Bottom
    σ·n = t  on Top

Method:
    - Input: Stiffness field E on a 3D grid.
    - Output: Displacement field u on the same grid.
    - Loss: Physics-based (PDE residual) calculated via Spectral Derivatives.
      No data required (unsupervised / physics-informed).

Usage:
    python pifo_elasticity_3d.py train --resolution 32
"""

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

# ============================================================
# Config & Constants
# ============================================================
jax.config.update("jax_enable_x64", False)

W, H, D = 1.0, 0.2, 1.0  # Domain size
E_MAX = 1000.0
E_MIN = 10.0
NU = 0.30
P_APPLIED = 1.0


# ============================================================
# Spectral Utilities
# ============================================================
def get_grid(resolution):
    """Generate normalized grid (0..1) for W, H, D."""
    # We use cell centers or nodes? Let's use nodes including endpoints for simplicity in BCs,
    # but FFT assumes periodic.
    # For PI-FNO on non-periodic domains, we usually use padding or specific BC handling.
    # Here we will use a simple grid and assume the field is periodic-ish for the spectral derivatives
    # OR (better) use finite differences for the PDE loss if boundaries are sharp.
    # However, "Fourier" Neural Operator implies Fourier modes.
    # Let's use Spectral Derivatives but mask the boundaries for the loss.

    nx, ny, nz = resolution, resolution, resolution
    x = jnp.linspace(0, W, nx)
    y = jnp.linspace(0, H, ny)
    z = jnp.linspace(0, D, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def spectral_derivative_3d(field, Ls):
    """
    Compute derivatives (d/dx, d/dy, d/dz) using FFT.
    field: (Nx, Ny, Nz) or (C, Nx, Ny, Nz)
    Ls: (Lx, Ly, Lz) domain lengths
    """
    is_vector = field.ndim == 4
    if not is_vector:
        field = field[None, ...]

    C, Nx, Ny, Nz = field.shape
    Lx, Ly, Lz = Ls

    # Wavenumbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=Ly / Ny)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=Lz / Nz)

    Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing="ij")

    # FFT
    field_hat = jnp.fft.fftn(field, axes=(1, 2, 3))

    # Derivatives in Fourier space: d/dx -> * i*kx
    du_dx_hat = 1j * Kx * field_hat
    du_dy_hat = 1j * Ky * field_hat
    du_dz_hat = 1j * Kz * field_hat

    # IFFT
    du_dx = jnp.real(jnp.fft.ifftn(du_dx_hat, axes=(1, 2, 3)))
    du_dy = jnp.real(jnp.fft.ifftn(du_dy_hat, axes=(1, 2, 3)))
    du_dz = jnp.real(jnp.fft.ifftn(du_dz_hat, axes=(1, 2, 3)))

    if not is_vector:
        return du_dx[0], du_dy[0], du_dz[0]
    return du_dx, du_dy, du_dz


# ============================================================
# 3D FNO Model
# ============================================================
class SpectralConv3d(eqx.Module):
    weight: jnp.ndarray
    n_modes: tuple

    def __init__(self, in_channels, out_channels, n_modes, *, key):
        self.n_modes = n_modes  # (k_x, k_y, k_z)
        scale = 1 / (in_channels * out_channels)
        # Complex weights for the corners of the Fourier spectrum
        # We keep the lowest n_modes frequency components
        self.weight = (
            jr.uniform(
                key,
                (in_channels, out_channels, n_modes[0], n_modes[1], n_modes[2], 2),
                minval=-1,
                maxval=1,
            )
            * scale
        )

    def __call__(self, x):
        # x: (channels, Nx, Ny, Nz)
        C, Nx, Ny, Nz = x.shape
        x_ft = jnp.fft.rfftn(x, axes=(1, 2, 3))

        # Multiply relevant modes
        # We handle the "corners" (low freq)
        # Implementation simplified: standard FNO usually takes lower corner
        # (modes, modes, modes) from (Nx, Ny, Nz/2+1)

        mx, my, mz = self.n_modes

        # Slice weights as complex
        w = jax.lax.complex(self.weight[..., 0], self.weight[..., 1])

        # Slice input spectrum
        # We only keep the low frequency corner
        out_ft = jnp.zeros((self.weight.shape[1], Nx, Ny, Nz // 2 + 1), dtype=jnp.complex64)

        # Contract: O_{klmn} = \sum_j W_{jklmn} * I_{jmnp} ... simplified einsum
        # x_ft corner: [:mx, :my, :mz]

        corner_in = x_ft[:, :mx, :my, :mz]
        corner_w = w

        # (in, out, x, y, z) * (in, x, y, z) -> (out, x, y, z)
        corner_out = jnp.einsum("ioxyz,ixyz->oxyz", corner_w, corner_in)

        # Place back
        out_ft = out_ft.at[:, :mx, :my, :mz].set(corner_out)

        x = jnp.fft.irfftn(out_ft, s=(Nx, Ny, Nz), axes=(1, 2, 3))
        return x


class FNO3d(eqx.Module):
    lift: eqx.nn.Conv3d
    fno_blocks: list
    conv_blocks: list
    proj: eqx.nn.Conv3d

    def __init__(self, in_channels, out_channels, modes=8, width=20, depth=4, *, key):
        k1, *ks = jr.split(key, 10)
        self.lift = eqx.nn.Conv3d(in_channels, width, kernel_size=1, key=k1)

        self.fno_blocks = []
        self.conv_blocks = []

        for i in range(depth):
            k_fno, k_conv = jr.split(ks[i], 2)
            self.fno_blocks.append(SpectralConv3d(width, width, (modes, modes, modes), key=k_fno))
            self.conv_blocks.append(eqx.nn.Conv3d(width, width, kernel_size=1, key=k_conv))

        self.proj = eqx.nn.Conv3d(width, out_channels, kernel_size=1, key=ks[-1])

    def __call__(self, x):
        # x: (in, Nx, Ny, Nz)
        x = self.lift(x)

        for fno, conv in zip(self.fno_blocks, self.conv_blocks):
            x1 = fno(x)
            x2 = conv(x)
            x = jax.nn.gelu(x1 + x2)

        x = self.proj(x)
        return x


# ============================================================
# Physics Loss
# ============================================================
def compute_loss(model, E_grid, resolution):
    """
    E_grid: (1, Nx, Ny, Nz) — Stiffness field
    Returns physics loss (PDE + BCs)
    """
    # 1. Forward Pass
    # Input to model: E_grid concatenated with coordinate grid?
    # FNO learns mapping E -> u. Coordinates are implicit in the grid structure,
    # but providing them as channels helps.
    Nx, Ny, Nz = E_grid.shape[1:]
    X, Y, Z = get_grid(Nx)

    # Input features: [E, x, y, z]
    # shape: (4, Nx, Ny, Nz)
    inputs = jnp.stack([E_grid[0], X, Y, Z], axis=0)

    # Predict Displacement u: (3, Nx, Ny, Nz)
    u = model(inputs)

    # 2. Compute Strains & Stresses
    # Derivatives via Spectral Method (or Finite Difference)
    # Spectral is differentiable and matches FNO inductive bias
    du_dx, du_dy, du_dz = spectral_derivative_3d(u, (W, H, D))

    # u is (3, ...), so du_dx is (3, ...)
    # u_x, u_y, u_z = u[0], u[1], u[2]

    eps_xx = du_dx[0]
    eps_yy = du_dy[1]
    eps_zz = du_dz[2]

    # Shear strains (engineering strain: gamma = 2 * epsilon)
    # But tensor strain is usually used for Hooke's law: sigma = ...
    # Let's stick to tensor components: ε_xy = 0.5(∂ux/∂y + ∂uy/∂x)
    eps_xy = 0.5 * (du_dy[0] + du_dx[1])
    eps_yz = 0.5 * (du_dz[1] + du_dy[2])
    eps_zx = 0.5 * (du_dx[2] + du_dz[0])

    trace_eps = eps_xx + eps_yy + eps_zz

    # Material properties
    E = E_grid[0]
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))

    # Stress Tensor
    # σ_ij = λ tr(ε) δ_ij + 2μ ε_ij
    sig_xx = lam * trace_eps + 2 * mu * eps_xx
    sig_yy = lam * trace_eps + 2 * mu * eps_yy
    sig_zz = lam * trace_eps + 2 * mu * eps_zz
    sig_xy = 2 * mu * eps_xy
    sig_yz = 2 * mu * eps_yz
    sig_zx = 2 * mu * eps_zx

    # 3. Equilibrium (Divergence of Stress)
    # ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z = 0
    # ...

    # Pack stresses for derivative
    # shape (6, Nx, Ny, Nz) -> [xx, yy, zz, xy, yz, zx]
    # Actually, let's compute derivatives of each component

    d_sig_xx_dx, _, _ = spectral_derivative_3d(sig_xx[None], (W, H, D))
    _, d_sig_yy_dy, _ = spectral_derivative_3d(sig_yy[None], (W, H, D))
    _, _, d_sig_zz_dz = spectral_derivative_3d(sig_zz[None], (W, H, D))

    d_sig_xy_dx, d_sig_xy_dy, _ = spectral_derivative_3d(sig_xy[None], (W, H, D))
    _, d_sig_yz_dy, d_sig_yz_dz = spectral_derivative_3d(sig_yz[None], (W, H, D))
    d_sig_zx_dx, _, d_sig_zx_dz = spectral_derivative_3d(sig_zx[None], (W, H, D))

    # Equilibrium Residuals
    res_x = d_sig_xx_dx + d_sig_xy_dy + d_sig_zx_dz
    res_y = d_sig_xy_dx + d_sig_yy_dy + d_sig_yz_dz
    res_z = d_sig_zx_dx + d_sig_yz_dy + d_sig_zz_dz

    # Loss: Interior Residual
    # We might want to mask boundaries if using finite difference, but for spectral
    # it's global. Let's just minimize sum of squares.
    loss_pde = jnp.mean(res_x**2 + res_y**2 + res_z**2)

    # 4. Boundary Conditions
    # Indices for boundaries
    # Bottom: y=0 -> index j=0
    # Top: y=H -> index j=-1

    # BC Bottom: u = 0
    u_bot = u[:, :, 0, :]  # (3, Nx, Nz)
    loss_bc_bot = jnp.mean(u_bot**2)

    # BC Top: Traction
    # n = (0, 1, 0). Traction t = σ · n = (σ_xy, σ_yy, σ_yz)
    # Target t = (0, -P, 0)
    sig_xy_top = sig_xy[:, -1, :]
    sig_yy_top = sig_yy[:, -1, :]
    sig_yz_top = sig_yz[:, -1, :]

    loss_bc_top = jnp.mean(sig_xy_top**2 + (sig_yy_top - (-P_APPLIED)) ** 2 + sig_yz_top**2)

    return loss_pde + 10.0 * loss_bc_bot + 10.0 * loss_bc_top


# ============================================================
# Training
# ============================================================
def generate_random_E(key, resolution):
    """Generate a random E field for training."""
    Nx, Ny, Nz = resolution, resolution, resolution
    # Generate low freq noise
    k1, k2 = jr.split(key)
    noise = jr.normal(k1, (Nx, Ny, Nz))

    # Smooth via FFT (low pass filter)
    noise_ft = jnp.fft.fftn(noise)
    kx = jnp.fft.fftfreq(Nx)
    ky = jnp.fft.fftfreq(Ny)
    kz = jnp.fft.fftfreq(Nz)
    Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing="ij")

    # Gaussian filter
    mask = jnp.exp(-100 * (Kx**2 + Ky**2 + Kz**2))
    smooth = jnp.real(jnp.fft.ifftn(noise_ft * mask))

    # Normalize to 0..1
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())

    # Map to E
    E = E_MAX * (1.0 - smooth) ** 2 + E_MIN * smooth
    return E[None]  # (1, Nx, Ny, Nz)


def train(resolution=32, epochs=1000, batch_size=4):
    print(f"Training PI-FNO 3D with resolution {resolution}^3...")

    key = jr.PRNGKey(0)
    k_model, k_opt = jr.split(key)

    # Model: 4 inputs (E, x, y, z), 3 outputs (u_x, u_y, u_z)
    model = FNO3d(in_channels=4, out_channels=3, modes=8, width=20, key=k_model)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, E_batch):
        # Loss averaged over batch
        def loss_fn(m, e):
            return compute_loss(m, e, resolution)

        loss, grads = eqx.filter_value_and_grad(
            lambda m, eb: jnp.mean(jax.vmap(lambda e: loss_fn(m, e))(eb))
        )(model, E_batch)

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    t0 = time.time()
    for ep in range(epochs):
        k_data = jr.fold_in(k_opt, ep)
        # Generate batch of random E fields
        keys = jr.split(k_data, batch_size)
        E_batch = jax.vmap(lambda k: generate_random_E(k, resolution))(keys)

        model, opt_state, loss = make_step(model, opt_state, E_batch)

        if ep % 10 == 0:
            print(f"Epoch {ep:4d} | Loss: {loss:.4e}")

    print(f"Done in {time.time() - t0:.1f}s")
    # Save model
    eqx.tree_serialise_leaves("pifno_3d.eqx", model)
    print("Saved to pifno_3d.eqx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    if args.command == "train":
        train(resolution=args.resolution, epochs=args.epochs)
