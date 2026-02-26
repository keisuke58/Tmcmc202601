"""
JAXFEM/core_hamilton_1d_nutrient.py
====================================
Option D Step 1 (物理版) — Hamilton 1D + 栄養場 c(x,t) の連成シミュレーション。

物理モデル
----------
Klempt et al. (2024) の式 (8)-(11) に基づく 1D 版:

  [Hamilton PDE — φᵢ, ψᵢ, γ]  ← core_hamilton_1d.py と同一
    ∂tφᵢ : Euler-Lagrange（Newton 法で各ノードを解く）
    ∂tφᵢ には c(x) が局所的に入る

  [栄養拡散-反応 PDE — c]
    ∂c/∂t = D_c · ∂²c/∂x² - g_eff · φ_total · c / (k_monod + c)
    BC: c(x=L, t) = 1.0      (外側 = 歯肉溝液、Dirichlet)
        ∂c/∂x|_{x=0} = 0     (内側 = 歯面、Neumann)
    IC: c(x, 0) = 1.0

  [成長膨張 — α]
    α̇(x, t) = k_α · φ_total(x, t)
    → α(x) = k_α ∫₀^{t_end} φ_total(x, t) dt

Klempt 2024 パラメータ目安
--------------------------
  D_c       = 1.0    (非次元, Klempt Fig.1 と同値)
  g_eff     = 50.0   (消費係数, Thiele 数 ~4 を与える)
  k_monod   = 1.0    (半飽和定数)
  k_alpha   = 0.05   (成長-膨張結合)

core_hamilton_1d.py との違い
-----------------------------
  - `c` がノードごとのスカラー c_node として渡される
  - 栄養 PDE ステップ (nutrient_step) を追加
  - vmap の軸が (0, 0, None) — (g, c_node, params)
  - シミュレーション関数が (G_all, c_all) を返す
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# 共有ユーティリティを core_hamilton_1d から再利用
from .core_hamilton_1d import (
    theta_to_matrices,
    clip_state,
    diffusion_step,
    make_initial_state,
)


# ---------------------------------------------------------------------------
# 残差関数: c をノードごとのスカラーとして受け取る
# ---------------------------------------------------------------------------


def residual_c(g_new, g_prev, c_node, params):
    """
    Hamilton PDE の残差 Q(g_new) = 0.
    core_hamilton_1d.residual と同一だが c はスカラー c_node で受け取る。
    """
    dt = params["dt_h"]
    Kp1 = params["Kp1"]
    Eta = params["Eta"]
    EtaPhi = params["EtaPhi"]
    alpha = params["alpha"]
    K_hill = params["K_hill"]
    n_hill = params["n_hill"]
    A = params["A"]
    b_diag = params["b_diag"]
    active_mask = params["active_mask"]
    eps = 1e-12

    phi_new = g_new[0:5]
    phi0_new = g_new[5]
    psi_new = g_new[6:11]
    gamma_new = g_new[11]
    phi_old = g_prev[0:5]
    phi0_old = g_prev[5]
    psi_old = g_prev[6:11]

    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Ia = A @ (phi_new * psi_new)

    # Hill gate (Pg, 種インデックス 4)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[4] == 1).astype(jnp.float64)
    fn = jnp.maximum(phi_new[3] * psi_new[3], 0.0)
    num = fn**n_hill
    den = K_hill**n_hill + num
    factor = jnp.where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[4].set(Ia[4] * factor)

    Q = jnp.zeros(12, dtype=jnp.float64)

    def body_phi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1

        def active_branch():
            t1 = Kp1 * (2.0 - 4.0 * phi_new[i]) / ((phi_new[i] - 1.0) ** 3 * phi_new[i] ** 3)
            t2 = (1.0 / Eta[i]) * (
                gamma_new
                + (EtaPhi[i] + Eta[i] * psi_new[i] ** 2) * phidot[i]
                + Eta[i] * phi_new[i] * psi_new[i] * psidot[i]
            )
            t3 = (c_node / Eta[i]) * psi_new[i] * Ia[i]  # ← c_node (スカラー)
            return Q_local.at[i].set(t1 + t2 - t3)

        def inactive_branch():
            return Q_local.at[i].set(phi_new[i])

        return jax.lax.cond(active, active_branch, inactive_branch), None

    Q, _ = jax.lax.scan(body_phi, Q, jnp.arange(5))
    Q = Q.at[5].set(
        gamma_new + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new**3) + phi0dot
    )

    def body_psi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1

        def active_branch():
            t1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0) ** 2 * psi_new[i] ** 3) - (2.0 * Kp1) / (
                (psi_new[i] - 1.0) ** 3 * psi_new[i] ** 2
            )
            t2 = (b_diag[i] * alpha / Eta[i]) * psi_new[i]
            t3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i] ** 2 * psidot[i]
            t4 = (c_node / Eta[i]) * phi_new[i] * Ia[i]  # ← c_node
            return Q_local.at[6 + i].set(t1 + t2 + t3 - t4)

        def inactive_branch():
            return Q_local.at[6 + i].set(psi_new[i])

        return jax.lax.cond(active, active_branch, inactive_branch), None

    Q, _ = jax.lax.scan(body_psi, Q, jnp.arange(5))
    Q = Q.at[11].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


# ---------------------------------------------------------------------------
# Newton 法: ノードごと (g_prev, c_node) → g_new
# ---------------------------------------------------------------------------


def newton_step_c(g_prev, c_node, params):
    """1 ノードの Hamilton Newton ステップ (c_node はスカラー)。"""
    active_mask = params["active_mask"]
    n_steps = 6

    def body(carry, _):
        g = carry
        g = clip_state(g, active_mask)

        def F(gg):
            return residual_c(gg, g_prev, c_node, params)

        Q = F(g)
        J = jax.jacfwd(F)(g)
        delta = jnp.linalg.solve(J, -Q)
        g_next = g + delta
        g_next = clip_state(g_next, active_mask)
        return g_next, None

    g0 = clip_state(g_prev, active_mask)
    g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_steps))
    return g_final


# in_axes=(0, 0, None): ノード方向に g_prev と c_field を同時に vmap
_newton_vmap_c = jax.jit(jax.vmap(newton_step_c, in_axes=(0, 0, None)))


# ---------------------------------------------------------------------------
# 反応ステップ (c フィールド付き)
# ---------------------------------------------------------------------------


def reaction_step_c(G, c_field, params):
    """Hamilton 反応ステップ。c_field (N,) をノードごとに渡す。"""
    n_sub = params["n_react_sub"]

    def body(carry, _):
        G_local, c_local = carry
        G_new = _newton_vmap_c(G_local, c_local, params)
        return (G_new, c_local), None

    (G_final, _), _ = jax.lax.scan(body, (G, c_field), jnp.arange(n_sub))
    return G_final


# ---------------------------------------------------------------------------
# 栄養場 c(x,t) の拡散-反応ステップ
# ---------------------------------------------------------------------------


def nutrient_step(c_field, phi_total, params):
    """
    栄養 c(x) を 1 ステップ更新する。

    PDE: ∂c/∂t = D_c Δc - g_eff · φ_total · c / (k_monod + c)
    BC:  c(x=L) = 1.0          (Dirichlet: 外側 = 歯肉溝液)
         dc/dx|_{x=0} = 0      (Neumann:  内側 = 歯面)
    """
    D_c = params["D_c"]
    g_eff = params["g_eff"]
    k_monod = params["k_monod"]
    dx = params["dx"]
    # マクロステップ幅 = dt_h * n_react_sub
    dt = params["dt_h"] * params["n_react_sub"]
    n_sub_c = params["n_sub_c"]  # 数値安定性のための栄養サブステップ数
    dt_c = dt / n_sub_c

    def sub_step(c, _):
        N = c.shape[0]
        # ラプラシアン (2 次中心差分, 内部ノードのみ)
        lap = jnp.zeros_like(c)
        interior = (c[:-2] + c[2:] - 2.0 * c[1:-1]) / (dx * dx)
        lap = lap.at[1:-1].set(interior)
        # Neumann BC at x=0: ghost node approach → lap[0] = (c[1]-c[0])/(dx²)
        lap = lap.at[0].set((c[1] - c[0]) / (dx * dx))
        # Dirichlet at x=L: c[-1] は更新しない

        # 反応項: Monod 型消費
        consumption = g_eff * phi_total * c / (k_monod + c + 1e-12)

        c_new = c + dt_c * (D_c * lap - consumption)
        c_new = jnp.clip(c_new, 0.0, None)
        # Dirichlet BC 適用
        c_new = c_new.at[-1].set(1.0)
        return c_new, None

    c_final, _ = jax.lax.scan(sub_step, c_field, jnp.arange(n_sub_c))
    return c_final


# ---------------------------------------------------------------------------
# メインシミュレーション関数
# ---------------------------------------------------------------------------


def simulate_hamilton_1d_nutrient(
    theta,
    D_eff,
    D_c=1.0,
    g_eff=50.0,
    k_monod=1.0,
    k_alpha=0.05,
    n_macro=60,
    n_react_sub=20,
    n_sub_c=10,
    N=30,
    L=1.0,
    dt_h=1e-5,
):
    """
    Hamilton 1D + 栄養場 c(x,t) の連成シミュレーション。

    Parameters
    ----------
    theta        : jnp.ndarray (20,)   TMCMC パラメータ
    D_eff        : jnp.ndarray (5,)    各菌種の有効拡散係数
    D_c          : float               栄養の拡散係数
    g_eff        : float               栄養消費係数 (Thiele 数を決める)
    k_monod      : float               Monod 半飽和定数
    k_alpha      : float               成長-固有ひずみ結合 k_α
    n_macro      : int                 マクロタイムステップ数
    n_react_sub  : int                 反応サブステップ数
    n_sub_c      : int                 栄養 PDE サブステップ数 (安定性のため)
    N            : int                 空間ノード数
    L            : float               正規化ドメイン長
    dt_h         : float               Hamilton タイムステップ

    Returns
    -------
    G_all   : jnp.ndarray (n_macro+1, N, 12)  Hamilton 変数の全軌跡
    c_all   : jnp.ndarray (n_macro+1, N)       栄養場の全軌跡
    """
    dx = L / (N - 1)
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
        # 栄養パラメータ
        "D_c": D_c,
        "g_eff": g_eff,
        "k_monod": k_monod,
        "n_sub_c": n_sub_c,
    }

    G0 = make_initial_state(N, active_mask)
    c0 = jnp.ones(N, dtype=jnp.float64)  # 初期栄養: 全ノード = 1.0

    def body(carry, _):
        G, c = carry
        phi_total = G[:, 0:5].sum(axis=1)  # (N,) 全種合計
        G = reaction_step_c(G, c, params)  # Hamilton 反応
        G = diffusion_step(G, params)  # φ 拡散
        c = nutrient_step(c, phi_total, params)  # 栄養 PDE
        return (G, c), (G, c)

    _, (G_traj, c_traj) = jax.lax.scan(body, (G0, c0), jnp.arange(n_macro))

    # 初期状態を先頭に付加して (n_macro+1, ...) にする
    G_all = jnp.concatenate([G0[jnp.newaxis], G_traj], axis=0)
    c_all = jnp.concatenate([c0[jnp.newaxis], c_traj], axis=0)

    return G_all, c_all
