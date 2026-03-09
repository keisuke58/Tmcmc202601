# Klempt 2024 モデル解析 — 連成構造と我々のパイプラインの問題点

## 1. Klempt の連成構造（正しい姿）

### 場の変数（全て FEM 内で同時に解く）

| 変数 | PDE | 意味 |
|------|-----|------|
| **φ(x,t)** | Allen-Cahn 型 + 栄養勾配駆動 | バイオフィルム phase field |
| **c(x,t)** | 反応拡散: ∂c/∂t = D∇²c − g·φ·c/(k+c) | 栄養濃度 |
| **α(x,t)** | α̇ = k_α · φ · c/(k+c) | 体積成長変数 |
| **u(x,t)** | ∇·P = 0 (準静的釣り合い) | 変位 |

### 変形の乗法分解

```
F = Fe · Fg
Fg = (1 + α)^{1/3} · I   （等方成長）
Fe = F · Fg^{-1}          （弾性変形）
σ = ∂Ψ/∂Fe               （応力は弾性部分のみから）
```

### 連成ループ（モノリシック Newton で一括解法）

```
時刻 t_n → t_{n+1}:

  ┌─ c(x) 拡散 + Monod 消費 ─────────────────┐
  │       ↓                                     │
  │  φ(x) 成長（栄養勾配 ∇c に従って前進）      │
  │       ↓                                     │
  │  α(x) 蓄積: α̇ = k_α · φ · c/(k+c)         │
  │       ↓                                     │
  │  Fg(α) → F = Fe·Fg → σ(Fe) → ∇·P = 0      │
  │       ↓                                     │
  │  u(x) 更新 → メッシュ変形                    │
  │       ↓                                     │
  │  変形メッシュ上で c の拡散を再計算 ←──────────┘
  └─────────────────────────────────────────────┘
  全て同一 FEM フレーム内、同一タイムステップ
```

**ポイント**: 栄養拡散・成長・力学が**同じメッシュ上で同時に進行**する。
メッシュが膨張すれば拡散距離が変わり、栄養到達パターンも変わる。

### Hamilton 原理の役割

全支配方程式を**一つの変分原理**から導出：
```
δ ∫ (Ψ_elastic + Ψ_chemical + Ψ_interface − W_ext − Q − B) dV dt = 0
```
- 熱力学第1・第2法則が**自動的に満たされる**
- 散逸不等式を個別にチェックする必要なし
- 数値的に安定・堅牢

---

## 2. 我々の現行パイプライン（問題あり）

### 現状の流れ（一方向・分離型）

```
Step 1: Hamilton ODE → φ_i(t)  [0D, 空間なし]
Step 2: DI = −Σ φ_i ln φ_i / ln 5
Step 3: E(DI) = E_max(1−r)^n + E_min·r
Step 4: FEM（静的）: −∇·(C:ε) = 0 → σ, u
```

### 何が問題か

| 問題 | 説明 | Klempt との乖離 |
|------|------|-----------------|
| **① 成長が FEM の外** | α を ODE から事前計算して eigenstrain として貼り付け | Klempt: α は FEM 内で時間発展 |
| **② 栄養拡散がメッシュ非依存** | c(x) の拡散は固定格子上、変形の影響なし | Klempt: 変形メッシュ上で拡散 |
| **③ 静的 FEM** | 最終状態の E, ε_g だけで 1 回解く | Klempt: 各タイムステップで解く |
| **④ φ が空間場でない（0D）** | 条件ごとに 1 つの DI 値 | Klempt: φ(x,t) は空間分布を持つ |
| **⑤ フィードバックなし** | σ → growth への逆影響ゼロ | Klempt: メッシュ変形 → 拡散経路変化 |

### 「部分的に正しい」点

- 2D Hamilton+Nutrient (`core_hamilton_2d_nutrient.py`) は φ_i(x,y,t) と c(x,y,t) を空間的に解いている → **④は 2D パイプラインでは解消済み**
- eigenstrain ε_g(x,y) も空間的に計算している
- **ただし FEM と ODE が分離している**（ODE 完了後に FEM を 1 回だけ実行）

---

## 3. Klempt 風にするには（提案）

### 3.1 最小限の修正（推奨：論文に間に合う現実的レベル）

**Staggered coupling（交互連成）**:

```python
for t in timesteps:
    # (a) 栄養 + 種構成を 1 ステップ進める
    phi, c = hamilton_nutrient_step(phi, c, dt, mesh_current)

    # (b) 成長変数を更新
    alpha += k_alpha * phi_total * c / (k + c) * dt

    # (c) FEM で力学を解く（現在の alpha で）
    u, sigma = solve_fem(E_field=E(DI(phi)), eps_growth=alpha/3)

    # (d) メッシュを更新（Optional: ALE or updated Lagrangian）
    mesh_current = update_mesh(mesh_ref, u)
```

**変更点**:
- `solve_stress_2d.py` のタイムループ化（現在は 1 回だけ）
- 各ステップで α を更新して FEM を再解法
- メッシュ更新は optional（小変形なら省略可）

**利点**: 既存コードの構造を活かしつつ、「成長の時間発展と力学の同時進行」を実現

### 3.2 フル連成（将来）

**Monolithic Newton（Klempt 完全再現）**:

```
残差ベクトル R = [R_u, R_φ, R_c, R_α]^T = 0

接線剛性:
K = ∂R/∂[u, φ, c, α] = [K_uu  K_uφ  0     K_uα ]
                         [0     K_φφ  K_φc  0    ]
                         [0     K_cφ  K_cc  0    ]
                         [0     K_αφ  K_αc  K_αα ]
```

- K_uα: 成長 → 変位の連成（eigenstrain 経由）
- K_φc: 栄養 → 種の連成（Monod 項）
- K_cφ: 種 → 栄養の連成（消費項）
- JAX-FEM (jax-fem) のカスタム問題として実装可能

### 3.3 実装優先度

| 段階 | 内容 | 効果 | 工数 |
|------|------|------|------|
| **A** | Staggered time loop (3.1) | 成長過程の可視化、応力の時間発展 | 1-2 日 |
| **B** | メッシュ更新 (ALE) | 変形による拡散経路変化 | 3-5 日 |
| **C** | Monolithic Newton (3.2) | 完全連成、熱力学的整合性 | 1-2 週 |
| **D** | Hamilton 原理から導出 | 散逸不等式の自動保証 | 理論的作業 |

---

## 4. 論文での書き方

### 現状の正当化（もし修正しない場合）

> "We employ a **sequential (staggered) multiscale approach**: the Hamilton ODE system is first solved
> to obtain species equilibrium composition φ_i, from which the dysbiosis index DI and the
> spatially-varying Young's modulus E(DI) are derived. The quasi-static FEM problem is then solved
> with these pre-computed material fields. This one-way coupling is justified under the assumption
> that **biological timescales (hours–days) are much larger than mechanical relaxation timescales
> (seconds–minutes)**, such that the biofilm is always in mechanical equilibrium for its current
> growth state."

### Klempt 風に修正した場合

> "Following Klempt et al. (2024), we solve the coupled reaction-diffusion-mechanics system in a
> **staggered time-stepping scheme**: at each macro time step, the 5-species Hamilton dynamics and
> nutrient diffusion are advanced, followed by an update of the growth eigenstrain α and a
> quasi-static FEM solve. This ensures that the evolving biofilm composition and the resulting
> mechanical state are **temporally consistent**."

---

## 5. 結論

**現行パイプラインは「間違い」ではないが、Klempt の枠組みに比べて不完全**。

核心的な差は:
1. **時間的連成がない**: 成長の過程が見えず、最終状態しか分からない
2. **FEM が 1 回きり**: 成長途中の応力状態（バイオフィルムが歯面を押す力の発展）が計算できない
3. **Hamilton 原理からの導出でない**: 熱力学的整合性が保証されていない

**最も impact が大きい修正は 3.1 の Staggered coupling**。
既存の `core_hamilton_2d_nutrient.py` + `solve_stress_2d.py` を時間ループで接続するだけで、
Klempt 論文と同等の「成長+力学の連成シミュレーション」が実現でき、
Fig 9-13 を時間発展アニメーションに格上げできる。

---

## 6. 実装結果 (2026-03-09)

### 実装ファイル

- **Solver**: `FEM/JAXFEM/run_coupled_staggered.py`
- **PBS script**: `FEM/JAXFEM/coupled_staggered_job.sh`
- **Output**: `FEM/figures/coupled_staggered/`

### アーキテクチャ

```python
for step in range(1, n_macro + 1):
    # (1) Hamilton reaction: Newton over all (Nx*Ny) nodes
    G = _reaction_step(G, params)

    # (2) Species diffusion: explicit Euler, Neumann BCs
    phi_2d = diffusion_step_species_2d(phi_2d, D_eff, dt_macro, dx, dy)

    # (3) Nutrient PDE: CFL-stable sub-stepping (30 sub-steps)
    c = _nutrient_stable(c, phi_2d, D_c, k_M, g_cons, c_bc, dx, dy, dt_macro)

    # (4) Growth accumulation (Klempt の α)
    alpha_field += k_alpha * phi_total * c / (k_M + c) * dt_macro

    # (5-6) DI → E → FEM solve (at snapshot intervals)
    if step % fem_every == 0:
        DI = compute_di(phi_2d)
        E_field = compute_E_phi_pg(phi_2d)
        eps_growth = alpha_field / 3.0
        fem = solve_2d_fem(E_field, nu, eps_growth, Nx, Ny)
```

### JAX LLVM メモリ回避

- 4 条件を subprocess で分離実行（条件間の JAX 再コンパイルを避ける）
- PBS ジョブとして qsub 投入（研究室共有サーバー安全ルール準拠）
- `--save-npz` で結果を中間保存 → 最後に比較図を生成

### 出力図

| ファイル | 内容 |
|----------|------|
| `coupled_evolution_{COND}.png` | 6行×N列 時間発展（φ, c, DI, E, α, σ_vm）|
| `coupled_timeseries_{COND}.png` | σ_vm_max(t), |u|_max(t), α_max(t) |
| `coupled_comparison_4cond.png` | 4条件の時系列比較 |
| `coupled_final_state_4cond.png` | 最終状態の DI/E/σ_vm 4条件並べ |
