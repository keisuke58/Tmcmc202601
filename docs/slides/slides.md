---
theme: academic
title: "5菌種バイオフィルム相互作用パラメータの逐次TMCMCベイズ推定"
info: |
  口腔バイオフィルムモデルにおける相互作用パラメータのベイズ推定
author: Keisuke Nishioka
paginationX: ~
paginationY: ~
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
math: katex
fonts:
  sans: 'Noto Sans JP'
  serif: 'Noto Serif JP'
  mono: 'Fira Code'
  weights: '300,400,500,700'
---

# 5菌種バイオフィルムモデルにおける<br>相互作用パラメータの逐次TMCMCベイズ推定

<div class="mt-8"></div>

**Keisuke Nishioka**, Felix Klempt, Hendrik Geisler, Meisam Soleimani, Philipp Junker

Institute of Continuum Mechanics (IKM), Leibniz Universität Hannover

---

# 研究背景

<div class="mt-2"></div>

### インプラント周囲炎と口腔バイオフィルム

- **インプラント周囲炎** (peri-implantitis) は歯科インプラント表面の多菌種バイオフィルムが主因
- Heine et al. (2025) による **in-vitro 時系列データ**: 5菌種 × 4条件 × 6時点

<div class="mt-4"></div>

### 実験条件

| 条件 | 菌叢状態 | 培養方式 | 推定パラメータ数 |
|:---:|:---:|:---:|:---:|
| CS | Commensal | Static | 11 |
| CH | Commensal | HOBIC (流れ) | 12 |
| DS | Dysbiotic | Static | 15 |
| DH | Dysbiotic | HOBIC (流れ) | 20 |

<div class="mt-3"></div>

- **5菌種**: *S. oralis* (So), *A. naeslundii* (An), *Veillonella* (Vei), *F. nucleatum* (Fn), *P. gingivalis* (Pg)
- **6時点**: Day 1, 3, 7, 10, 15, 21

---
layout: two-cols
---

# 課題設定

<div class="mt-3"></div>

### データと未知数のバランス

$$\underbrace{30 \text{ データ点}}_{6 \text{ 時点} \times 5 \text{ 菌種}} \;\longleftrightarrow\; \underbrace{20 \text{ パラメータ}}_{A_{5 \times 5} + \mathbf{b}_5}$$

<div class="mt-4"></div>

### 困難な点

1. **同定不良**: データ数 < パラメータ数
2. **多峰性事後分布**: 標準MCMCでは非効率
3. **生物学的に非現実的**な局所解の存在

::right::

<div class="mt-12"></div>

### 我々の戦略

<div class="mt-3"></div>

**① 生物学的制約**

不活性ペア5組を $A_{ij} = 0$ に固定

$$20 \to 15 \text{ 自由パラメータ}$$

<div class="mt-4"></div>

**② 逐次TMCMC**

生態学的遷移に沿った4段階分解

$$\leq 6 \text{ 次元の部分問題}$$

---

# 数学モデル — 支配方程式

<div class="mt-1"></div>

### 一般化 Lotka-Volterra 型 ODE

活性体積分率 $u_i = \phi_i \psi_i$ の時間発展:

$$\frac{du_i}{dt} = \left(A_{ii} + \sum_{j \neq i} A_{ij}\, u_j \right) u_i - b_i\, u_i, \qquad i = 0, \dots, 4$$

<div class="grid grid-cols-2 gap-4 mt-2 text-sm">
<div>

- $A_{ii}$: 自己調節係数 (self-regulation)
- $A_{ij}$ ($j \neq i$): 菌種 $j$ → $i$ への相互作用
</div>
<div>

- $b_i$: 減衰率 (decay rate)
- 空隙拘束: $\sum \phi_i + \phi_\text{void} = 1$
</div>
</div>

<div class="mt-4"></div>

### 変分原理からの導出

- **拡張 Hamilton 原理** (Junker & Balzani 2021) の停留条件 $\delta\Pi = 0$ から導出
- エネルギー汎関数: $\Pi = \Psi + \Delta + W_\text{ext}$ (自由エネルギー + 散逸 + 外部仕事)
- **対称性** $A_{ij} = A_{ji}$ は熱力学的整合性から **自然に帰結**（仮定ではない）

---

# 数学モデル — パラメータ構造

<div class="mt-1"></div>

### 対称行列 $\mathbf{A} \in \mathbb{R}^{5 \times 5}$

対称性 $A_{ij} = A_{ji}$ → 非対角10 + 対角5 + 減衰5 = **20パラメータ**

<div class="text-sm mt-2">

$$\boldsymbol{\theta} = \underbrace{(a_{11}, a_{12}, a_{22}, b_1, b_2)}_{\text{So--An}} \oplus \underbrace{(a_{33}, a_{34}, a_{44}, b_3, b_4)}_{\text{Vei--Fn}} \oplus \underbrace{(a_{13}, a_{14}, a_{23}, a_{24})}_{\text{交差}} \oplus \underbrace{(a_{55}, b_5)}_{\text{Pg自己}} \oplus \underbrace{(a_{15}, a_{25}, a_{35}, a_{45})}_{\text{Pg交差}}$$

</div>

<div class="mt-3"></div>

### 生物学的制約 — ロック集合 $\mathcal{L}$

代謝経路が存在しない5ペアを恒等的に零に固定:

$$\mathcal{L} = \{6, 12, 13, 16, 17\}, \qquad \theta_k = 0 \;\; \forall k \in \mathcal{L}$$

<div class="grid grid-cols-2 gap-4 mt-2">
<div>

| ロックペア | 根拠 |
|:---|:---|
| An--Vei, An--Fn, Vei--Fn | 直接的代謝経路なし |
| So--Pg, An--Pg | 共凝集・交差供給なし |
</div>
<div class="flex items-center justify-center">

$$n_\text{free} = 20 - |\mathcal{L}| = 15$$

</div>
</div>

---

# 相互作用ネットワーク

<div class="grid grid-cols-[auto_1fr] gap-4 mt-1">
<div class="flex items-center justify-center">
<img src="/img/fig1_network.png" class="h-88" />
</div>
<div>

### 活性ペア (5組)

<div class="text-sm">

| ペア | メカニズム | タイプ |
|:---|:---|:---|
| So -- An | 共凝集 | 双方向 |
| So -- Vd | 乳酸交換 | 双方向 |
| So -- Fn | ギ酸共生 | 双方向 |
| Vd → Pg | pH上昇 (Hill gate) | 一方向 |
| Fn -- Pg | ペプチド供給 | 双方向 |

</div>

<div class="mt-2 text-sm">

**ロック ($A_{ij} = 0$)**: An-Vd, An-Fn, Vd-Fn, So-Pg, An-Pg

→ Fig. 4C で直接的な代謝パスウェイが存在しない

</div>

<div class="mt-3 text-sm">

**生態学的遷移**:

$$\underbrace{\text{So, An}}_{\text{Pioneer}} \xrightarrow{\text{代謝産物}} \underbrace{\text{Vd, Fn}}_{\text{Bridge}} \xrightarrow{\text{pH・ペプチド}} \underbrace{\text{Pg}}_{\text{Pathogen}}$$

</div>

</div>
</div>

---

# ベイズ推定フレームワーク

<div class="mt-1"></div>

### 事後分布

$$p(\boldsymbol{\theta} \mid \mathbf{y}_\text{obs}) = \frac{p(\mathbf{y}_\text{obs} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta})}{p(\mathbf{y}_\text{obs})}$$

### 尤度関数

菌種・時点間で独立なガウス誤差を仮定:

$$p(\mathbf{y}_\text{obs} \mid \boldsymbol{\theta}) = \prod_{k=1}^{N_t} \prod_{i=0}^{4} \frac{1}{\sqrt{2\pi}\,\sigma_i} \exp\!\left(-\frac{(y_{\text{obs},i}(t_k) - \hat{y}_i(t_k; \boldsymbol{\theta}))^2}{2\sigma_i^2}\right)$$

<div class="text-sm mt-1">

観測ノイズ $\sigma$ はダイナミックレンジに基づく: $\sigma_\text{CS} = 0.11$, $\sigma_\text{CH} = 0.16$, $\sigma_\text{DS} = 0.25$, $\sigma_\text{DH} = 0.23$

</div>

### 制約付き事前分布

$$p(\theta_k) = \begin{cases} \delta(\theta_k) & k \in \mathcal{L} \quad \text{(Diracデルタ → ロック)} \\ \frac{1}{u_k - l_k}\, \mathbf{1}_{[l_k, u_k]}(\theta_k) & k \notin \mathcal{L} \quad \text{(一様分布, } [l_k,u_k]=[-1,1] \text{)} \end{cases}$$

---

# TMCMC アルゴリズム

<div class="mt-1"></div>

### テンパリング尤度列 (Ching & Chen 2007)

事前分布 ($\beta=0$) から事後分布 ($\beta=1$) へ段階的に移行:

$$p_m(\boldsymbol{\theta}) \propto p(\mathbf{y}_\text{obs} \mid \boldsymbol{\theta})^{\beta_m} \, p(\boldsymbol{\theta}), \qquad 0 = \beta_0 < \beta_1 < \cdots < \beta_M = 1$$

<div class="mt-3"></div>

### 各テンパリングステージ

<div class="text-sm">

1. **重み計算**: $\;w_j^{(m)} = p(\mathbf{y}_\text{obs} \mid \boldsymbol{\theta}_j^{(m)})^{\beta_{m+1} - \beta_m}$

2. **$\beta_{m+1}$ の適応決定**: $\;\text{CoV}[\{w_j^{(m)}\}] = \delta_\text{target}$ を二分法で解く

3. **リサンプリング**: $\;N$ 粒子を重み $\propto w_j^{(m)}$ で再抽出

4. **MH 変異**: $\;q(\boldsymbol{\theta}^* \mid \boldsymbol{\theta}_j) = \mathcal{N}(\boldsymbol{\theta}_j, \gamma^2 \boldsymbol{\Sigma}^{(m)})$, 受容確率:

$$\alpha = \min\!\left(1, \; \frac{p(\mathbf{y}_\text{obs} \mid \boldsymbol{\theta}^*)^{\beta_{m+1}} \, p(\boldsymbol{\theta}^*)}{p(\mathbf{y}_\text{obs} \mid \boldsymbol{\theta}_j)^{\beta_{m+1}} \, p(\boldsymbol{\theta}_j)}\right)$$

5. **制約強制**: ロックインデックスを零にリセット、自由パラメータを事前分布範囲にクリップ

</div>

---

# 4段階逐次分解

<div class="mt-1"></div>

### 生態学的遷移に沿った段階的推定

| ステージ | グループ | パラメータ | 次元 | 生態学的役割 |
|:---:|:---|:---|:---:|:---|
| 1 | So -- An ブロック | $a_{11}, a_{12}, a_{22}, b_1, b_2$ | 5 | 初期定着菌 |
| 2 | Vei -- Fn ブロック | $a_{33}, a_{34}, a_{44}, b_3, b_4$ | $\leq 5$ | 橋渡し菌 |
| 3 | 交差ブロック + Pg自己 | $a_{13}, a_{14}, a_{23}, a_{24}, a_{55}, b_5$ | $\leq 6$ | ブロック間結合 |
| 4 | Pg 交差供給 | $a_{15}, a_{25}, a_{35}, a_{45}$ | $\leq 4$ | 後期定着菌 |

<div class="mt-4"></div>

### 手順

1. ステージ $s$ で活性パラメータ $\mathcal{A}_s = \mathcal{P}_s \setminus \mathcal{L}$ のみを TMCMC で推定
2. 推定した **MAP 値を固定** → 次ステージの条件付き低次元部分問題
3. 分解順序は **生態学的遷移** (初期 → 橋渡し → 後期) に対応
4. DH 条件: 全20パラメータを同時推定（「**探索モード**」）
5. 副産物: モデルエビデンス $\hat{p}(\mathbf{y}_\text{obs}) = \prod_{m} \frac{1}{N}\sum_j w_j^{(m)}$

---

# 実験条件とロッキング戦略

<div class="mt-2"></div>

### 条件別パラメータロッキング

| 条件 | 培養 | $N_\text{locked}$ | $N_\text{est}$ | 根拠 |
|:---:|:---:|:---:|:---:|:---|
| CS | Static | 9 | 11 | Pg, Fn が qPCR 検出限界以下 |
| CH | Flow | 8 | 12 | Blue bloom, Pg 抑制 |
| DS | Static | 5 | 15 | 病原菌相互作用が活性化 |
| DH | Flow | 0 | 20 | サージ発見のため全開放 |

<div class="mt-4"></div>

### DH「探索モード」の妥当性検証

- DH 条件では $N_\text{locked} = 0$ として全パラメータを推定
- 構造的にロックすべきパラメータの事後 MAP 値は自動的に零近傍に収束:

<div class="text-sm mt-1">

> 例: $\hat{A}_{01}^\text{DH} \approx 0.03$ (So--Pg) → **データ駆動で生物学的制約を検証**

</div>

- $\theta_{18}$ (Vei→Pg) の事前分布範囲は条件特異的: CS/CH ではロック, DS では $[-3, 0]$, DH では $[0, 25]$

---

# 結果: 適合度と収束

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

### MAP RMSE ($N = 1000$ 粒子)

| 条件 | RMSE | $N_\text{est}$ | $\ln \hat{Z}$ |
|:---:|:---:|:---:|:---:|
| CS | 0.055 | 11 | $-1.97$ |
| CH | 0.063 | 12 | $-12.66$ |
| DS | 0.054 | 15 | $-8.89$ |
| DH | 0.075 | 20 | $-17.67$ |

<div class="text-sm mt-2">

全条件で RMSE < 0.08 を達成

テンパリング段数: $M = 3$--$5$

</div>

</div>
<div>

### 収束診断

<div class="text-sm mt-2">

| 指標 | 値 |
|:---|:---|
| Gelman-Rubin $\hat{R}$ | < 1.05 (CS, CH, DH) |
| DS $\hat{R}_\text{max}$ | $\approx 1.11$ (許容範囲) |
| 有効サンプルサイズ | > 59% |
| MCMC 受容率 | 49--71% |
| 壁時計時間 | 40h (CS) -- 90h (DH) |
| Forward solve 数 | $\sim$25,000/条件 |

</div>

<div class="text-sm mt-2">

DH の per-species 最大残差:
Vei (0.107), Fn (0.081)

</div>

</div>
</div>

---

# 事後予測フィット: Commensal Static (CS)

<div class="grid grid-cols-[1fr_auto] gap-2">
<div class="flex justify-center">
<img src="/img/fig2_cs.png" class="h-90" />
</div>
<div class="flex items-end">
<img src="/img/ab_map_cs.png" class="w-72" />
</div>
</div>

<div class="text-sm">

So, An の急成長 + 定常状態を再現. Fn, Pg は検出限界以下 → 関連パラメータをロック

</div>

---

# 事後予測フィット: Commensal HOBIC (CH)

<div class="grid grid-cols-[1fr_auto] gap-2">
<div class="flex justify-center">
<img src="/img/fig2_ch.png" class="h-90" />
</div>
<div class="flex items-end">
<img src="/img/ab_map_ch.png" class="w-72" />
</div>
</div>

<div class="text-sm">

So 支配の blue bloom を捕捉. 流れ条件下でも共生構造は安定. Pg は抑制されロック

</div>

---

# 事後予測フィット: Dysbiotic Static (DS)

<div class="grid grid-cols-[1fr_auto] gap-2">
<div class="flex justify-center">
<img src="/img/fig2_ds.png" class="h-90" />
</div>
<div class="flex items-end">
<img src="/img/ab_map_ds.png" class="w-72" />
</div>
</div>

<div class="text-sm">

病原菌相互作用が活性化するも中程度. サージなし. 構造的ロック5組のみ

</div>

---

# 事後予測フィット: Dysbiotic HOBIC (DH)

<div class="grid grid-cols-[1fr_auto] gap-2">
<div class="flex justify-center">
<img src="/img/fig2_dh.png" class="h-90" />
</div>
<div class="flex items-end">
<img src="/img/ab_map_dh.png" class="w-72" />
</div>
</div>

<div class="text-sm">

全20パラメータ開放. $\theta_{18}$ (Vd→Pg) = 17.3 が突出. Vei, Pg の非線形サージを捕捉

</div>

---

# 病原菌サージメカニズム

<div class="mt-1"></div>

### DH 条件で Pg サージを駆動するパラメータ

| パラメータ | DH | DS | CS/CH |
|:---|:---:|:---:|:---:|
| $\theta_{18}$ (Vei → Pg, pH 媒介) | **17.3** | $-0.39$ | ロック |
| $\theta_{19}$ (Fn -- Pg, ペプチド) | **4.6** | $0.93$ | ロック |
| $\theta_{14}$ (Pg 自己成長) | **1.7** | $0.97$ | ロック |

<div class="mt-3"></div>

### Frobenius 解析 — 健康 → 疾患の構造的転換

<div class="text-sm">

$$\rho^{(c_1,c_2)} = \text{corr}\bigl(\text{vec}(\mathbf{A}^{(c_1)}), \text{vec}(\mathbf{A}^{(c_2)})\bigr), \quad \Delta_F^{(c_1,c_2)} = \frac{\|\mathbf{A}^{(c_1)} - \mathbf{A}^{(c_2)}\|_F}{\max\{\|\mathbf{A}^{(c_1)}\|_F, \|\mathbf{A}^{(c_2)}\|_F\}}$$

</div>

| 比較 | $\rho$ | $\Delta_F$ |
|:---|:---:|:---:|
| 同一健康状態 (C-C, D-D) | $\approx +0.46$ | 0.42--0.52 |
| 異なる健康状態 (C-D) | $\approx -0.15$ | 0.57--0.71 |

> 健康 → 疾患は $\mathbf{A}$ の**構造的再編成**であり、単なるスケーリングではない

---

# 結論

<div class="mt-2"></div>

1. **生物学的制約の導入**: Dirac デルタ事前分布で不在相互作用を固定 → $20 \to 15$ 次元, 同定性の改善

2. **逐次 TMCMC**: 生態学的遷移に沿った4段階分解 → $\leq 6$ 次元の部分問題

3. **4条件の再現**: 共生ホメオスタシス (CS, CH), 中程度の病原性 (DS), 非線形サージ (DH)

4. **Frobenius 解析**: 健康 → 疾患 = $\mathbf{A}$ の構造的再編成 ($\rho_\text{cross} \approx -0.15$, $\Delta_F \geq 0.57$)

5. **DH 探索モード**: 全20パラメータ推定でも、ロック対象は自動的に零近傍に収束 → データ駆動で制約を検証

<div class="mt-5"></div>

### 今後の予定

- **マルチスケール拡張**: Dysbiosis Index (DI) → 材料モデル $E(\text{DI})$ → 3D FEM への接続
- **事後 UQ 伝播**: ベイズ不確実性を ODE → DI → FEM パイプライン全体に伝播
- **状態依存型相互作用行列**: $A_{ij}(\text{pH}, c^*)$ による DH サージの完全再現
- **サロゲート加速**: DeepONet による ${\sim}100\times$ 高速化 TMCMC
