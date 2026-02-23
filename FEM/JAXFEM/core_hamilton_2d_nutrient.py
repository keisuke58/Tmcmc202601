"""
JAXFEM/core_hamilton_2d_nutrient.py
====================================
Hamilton 2D + 栄養場 c(x,y,t) の連成シミュレーション (Issue #3: P2).

物理モデル
----------
core_hamilton_1d_nutrient.py の 2D 拡張:

  [Hamilton PDE — φᵢ, ψᵢ, γ]
    Operator splitting: 反応 (Newton, vmap) + 拡散 (2D Laplacian)
    反応ステップでは c(x,y) がノードごとにスカラーとして入る

  [栄養拡散-反応 PDE — c]
    ∂c/∂t = D_c (∂²c/∂x² + ∂²c/∂y²) - g_eff · φ_total · c / (k_monod + c)
    BC: c(x=Lx, y, t) = 1.0   (saliva side, Dirichlet)
         Neumann (no-flux) on other 3 edges

  座標系:
    x 方向 = 深さ (i=0: 歯面, i=Nx-1: 唾液)
    y 方向 = 横方向 (歯面に沿った方向, 対称)
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# 1D nutrient版の Newton ステップを再利用 (グリッド非依存)
from .core_hamilton_1d_nutrient import (
    newton_step_c,
    _newton_vmap_c,
)

# 2D版の拡散・初期状態を再利用
from .core_hamilton_2d import (
    make_initial_state_2d,
    diffusion_step_2d,
)

# パラメータ変換
from .core_hamilton_1d import (
    THETA_DEMO,
    theta_to_matrices,
)


# ---------------------------------------------------------------------------
# 反応ステップ (c フィールド付き, 2D flat)
# ---------------------------------------------------------------------------

def reaction_step_2d_c(G_flat, c_flat, params):
    """Hamilton 反応ステップ (2D flat). c_flat (Nx*Ny,) をノードごとに渡す。

    _newton_vmap_c は in_axes=(0, 0, None) で定義されており
    (Nx*Ny, 12) と (Nx*Ny,) を同時に vmap する。1D と同一。
    """
    n_sub = params["n_react_sub"]

    def body(carry, _):
        G_local, c_local = carry
        G_new = _newton_vmap_c(G_local, c_local, params)
        return (G_new, c_local), None

    (G_final, _), _ = jax.lax.scan(body, (G_flat, c_flat), jnp.arange(n_sub))
    return G_final


# ---------------------------------------------------------------------------
# 2D 栄養場 c(x,y,t) の拡散-反応ステップ
# ---------------------------------------------------------------------------

def nutrient_step_2d(c_flat, phi_total_flat, params):
    """
    栄養 c(x,y) を 1 マクロステップ更新する。

    PDE: ∂c/∂t = D_c (∂²c/∂x² + ∂²c/∂y²) - g_eff · φ_total · c / (k + c)
    BC:  c(x=Lx, y) = 1.0   (Dirichlet: saliva, i=Nx-1)
         Neumann (no-flux) on x=0, y=0, y=Ly

    Parameters
    ----------
    c_flat : (Nx*Ny,)
    phi_total_flat : (Nx*Ny,)
    params : dict with D_c, g_eff, k_monod, dx, dy, Nx, Ny, dt_h, n_react_sub, n_sub_c
    """
    D_c = params["D_c"]
    g_eff = params["g_eff"]
    k_monod = params["k_monod"]
    dx = params["dx"]
    dy = params["dy"]
    Nx = params["Nx"]
    Ny = params["Ny"]

    dt_macro = params["dt_h"] * params["n_react_sub"]
    n_sub_c = params["n_sub_c"]
    dt_c = dt_macro / n_sub_c

    c = c_flat.reshape((Nx, Ny))
    phi_total = phi_total_flat.reshape((Nx, Ny))

    def sub_step(c_grid, _):
        # --- 2D Laplacian ---
        lap = jnp.zeros_like(c_grid)

        # Interior: standard 5-point stencil
        lap_x = (c_grid[:-2, 1:-1] + c_grid[2:, 1:-1]
                 - 2.0 * c_grid[1:-1, 1:-1]) / (dx * dx)
        lap_y = (c_grid[1:-1, :-2] + c_grid[1:-1, 2:]
                 - 2.0 * c_grid[1:-1, 1:-1]) / (dy * dy)
        lap = lap.at[1:-1, 1:-1].set(lap_x + lap_y)

        # --- Neumann BC (ghost-point approach) ---
        # x=0 (tooth, i=0): ∂c/∂x = 0 → c[-1,j] = c[1,j]
        # Only interior y (j=1..Ny-2)
        lap_x0 = (c_grid[1, 1:-1] - c_grid[0, 1:-1]) / (dx * dx)
        lap_y0 = (c_grid[0, :-2] + c_grid[0, 2:]
                  - 2.0 * c_grid[0, 1:-1]) / (dy * dy)
        lap = lap.at[0, 1:-1].set(lap_x0 + lap_y0)

        # y=0 (j=0): ∂c/∂y = 0, interior x (i=1..Nx-2)
        lap_x_y0 = (c_grid[:-2, 0] + c_grid[2:, 0]
                    - 2.0 * c_grid[1:-1, 0]) / (dx * dx)
        lap_y_y0 = (c_grid[1:-1, 1] - c_grid[1:-1, 0]) / (dy * dy)
        lap = lap.at[1:-1, 0].set(lap_x_y0 + lap_y_y0)

        # y=Ly (j=Ny-1): ∂c/∂y = 0, interior x
        lap_x_yL = (c_grid[:-2, -1] + c_grid[2:, -1]
                    - 2.0 * c_grid[1:-1, -1]) / (dx * dx)
        lap_y_yL = (c_grid[1:-1, -2] - c_grid[1:-1, -1]) / (dy * dy)
        lap = lap.at[1:-1, -1].set(lap_x_yL + lap_y_yL)

        # Corner (0,0): Neumann in both x and y
        lap_corner_00 = ((c_grid[1, 0] - c_grid[0, 0]) / (dx * dx)
                         + (c_grid[0, 1] - c_grid[0, 0]) / (dy * dy))
        lap = lap.at[0, 0].set(lap_corner_00)

        # Corner (0, Ny-1): Neumann x + Neumann y
        lap_corner_0L = ((c_grid[1, -1] - c_grid[0, -1]) / (dx * dx)
                         + (c_grid[0, -2] - c_grid[0, -1]) / (dy * dy))
        lap = lap.at[0, -1].set(lap_corner_0L)

        # x=Lx (i=Nx-1): Dirichlet c=1 → lap row is irrelevant (overwritten)

        # --- Monod consumption ---
        consumption = g_eff * phi_total * c_grid / (k_monod + c_grid + 1e-12)

        c_new = c_grid + dt_c * (D_c * lap - consumption)
        c_new = jnp.clip(c_new, 0.0, None)

        # Dirichlet BC: c(x=Lx, y) = 1.0
        c_new = c_new.at[-1, :].set(1.0)

        return c_new, None

    c_final, _ = jax.lax.scan(sub_step, c, jnp.arange(n_sub_c))
    return c_final.reshape(Nx * Ny)


# ---------------------------------------------------------------------------
# メインシミュレーション関数
# ---------------------------------------------------------------------------

def simulate_hamilton_2d_nutrient(
    theta,
    D_eff,
    D_c=1.0,
    g_eff=50.0,
    k_monod=1.0,
    k_alpha=0.05,
    n_macro=60,
    n_react_sub=20,
    n_sub_c=10,
    Nx=20,
    Ny=20,
    Lx=1.0,
    Ly=1.0,
    dt_h=1e-5,
):
    """
    Hamilton 2D + 栄養場 c(x,y,t) の連成シミュレーション。

    座標系: x=深さ方向 (0=歯面, Lx=唾液), y=横方向

    Parameters
    ----------
    theta        : jnp.ndarray (20,)   TMCMC パラメータ
    D_eff        : jnp.ndarray (5,)    各菌種の有効拡散係数
    D_c          : float               栄養の拡散係数
    g_eff        : float               栄養消費係数
    k_monod      : float               Monod 半飽和定数
    k_alpha      : float               成長-固有ひずみ結合 k_α
    n_macro      : int                 マクロタイムステップ数
    n_react_sub  : int                 反応サブステップ数
    n_sub_c      : int                 栄養 PDE サブステップ数
    Nx, Ny       : int                 空間ノード数 (x, y)
    Lx, Ly       : float               ドメインサイズ
    dt_h         : float               Hamilton タイムステップ

    Returns
    -------
    G_all   : jnp.ndarray (n_macro+1, Nx*Ny, 12)
    c_all   : jnp.ndarray (n_macro+1, Nx*Ny)
    """
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    A, b_diag = theta_to_matrices(theta)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h": dt_h,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5),
        "EtaPhi": jnp.ones(5),
        "alpha": 100.0,
        "K_hill": 0.05,
        "n_hill": 4.0,
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
        "n_react_sub": n_react_sub,
        "D_eff": D_eff,
        "dx": dx,
        "dy": dy,
        "Nx": Nx,
        "Ny": Ny,
        # 栄養パラメータ
        "D_c": D_c,
        "g_eff": g_eff,
        "k_monod": k_monod,
        "n_sub_c": n_sub_c,
    }

    G0_flat = make_initial_state_2d(Nx, Ny, active_mask)   # (Nx*Ny, 12)
    c0_flat = jnp.ones(Nx * Ny, dtype=jnp.float64)        # c=1 everywhere

    def body(carry, _):
        G, c = carry
        phi_total = G[:, 0:5].sum(axis=1)                  # (Nx*Ny,)
        G = reaction_step_2d_c(G, c, params)                # Hamilton reaction
        G = diffusion_step_2d(G, params)                    # species diffusion
        c = nutrient_step_2d(c, phi_total, params)          # nutrient PDE
        return (G, c), (G, c)

    _, (G_traj, c_traj) = jax.lax.scan(body, (G0_flat, c0_flat), jnp.arange(n_macro))

    G_all = jnp.concatenate([G0_flat[jnp.newaxis], G_traj], axis=0)
    c_all = jnp.concatenate([c0_flat[jnp.newaxis], c_traj], axis=0)

    return G_all, c_all
