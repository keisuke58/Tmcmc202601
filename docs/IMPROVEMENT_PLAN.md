# 改善計画: 文献調査に基づくアクションプラン (2026-03-12)

36リポジトリの分析に基づく、優先度順の改善項目。

---

## A. DeepONet 改善 (CS 62%→~15%, DS 52%→~15% 目標)

| # | 改善 | 難度 | 工数 | 期待効果 | ソース |
|---|------|------|------|----------|--------|
| A1 | **出力の StandardScaler** | Easy | 20行 | 低存在量種 (Pg,Fn) 15-25%↓ | deeponet-fno |
| A2 | **Trunk に Sine activation** | Easy | 3行 | 全条件 10-20% relative ↓ | separable-operator-networks |
| A3 | **POD-DeepONet** (SVD trunk) | Medium | 50行 | CS/DS MAP err 62→15% | deeponet-fno, deepxde |
| A4 | **条件統合モデル** (4→1) | Medium | 30行 | データ4倍、CS/DS大幅改善 | conceptual (latent-deeponet) |
| A5 | **Multi-fidelity ASMC** | Hard | 150行 | Pg params overlap 0.95→0.98+ | ASMC-SURR |
| A6 | Latent-space AE+DeepONet | Hard | 200行 | ~5% (500次元では効果小) | latent-deeponet |

**推奨順: A1 → A2 → A3 → A4 → A5**

### A1: 出力 StandardScaler (20min)
- `load_data()` で phi の per-species mean/std 計算
- 学習時に正規化、`predict_trajectory()` で逆変換
- Pg (0.001-0.03) と So (0.2-0.8) のスケール差を解消

### A2: Sine trunk (10min)
- `TrunkNet.__call__` で `jax.nn.gelu` → `jnp.sin` に変更 (BranchNet は GELU 維持)
- ODE軌道の non-monotonic dynamics に適合

### A3: POD-DeepONet (半日)
- 学習データ phi `(N, T, 5)` → `(N, T*5)` で SVD、top-k モード抽出
- Trunk を固定 POD basis に置換、Branch は POD 係数 `(k,)` を出力
- **最大インパクト**: trunk の学習負担を劇的削減

---

## B. TMCMC 改善 (Quick Wins)

| # | 改善 | 難度 | 工数 | 期待効果 | ソース |
|---|------|------|------|----------|--------|
| B1 | **Systematic resampling** | Easy | 15行 | unique ratio +15-30% | Bayesian-Struct-Dynamics |
| B2 | **Weighted covariance** (重み付き) | Easy | 3行 | acceptance +5-15% | transitional-mcmc, Korali |
| B3 | **Adaptive step count** | Easy | 5行 | wall-time -20-40% | transitional-mcmc |
| B4 | **scalem = 1/9 + 8/9·R** | Easy | 3行 | チューニングパラメータ削減 | transitional-mcmc |
| B5 | **CoV-based beta schedule** | Medium | 30行 | evidence精度 +15% | Korali, ASMC |
| B6 | DREAMzs proposal | Hard | 200行 | 多峰性探索向上 | ASMC |

**推奨順: B1 → B2 → B3 → B4 → B5**

### B1: Systematic resampling (15min)
```python
def systematic_resample(weights, rng):
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cs = np.cumsum(weights)
    return np.clip(np.searchsorted(cs, positions), 0, N-1)
```
- Multinomial の O(N) variance → O(1) に削減
- Particle degeneracy を直接改善

### B2: Weighted covariance (5min)
- L715 `np.cov(theta.T)` → `np.cov(theta.T, aweights=weights)` (resample 前)
- 重要度重みの情報を保存

---

## C. VEM 改善 (Future Work 品質向上)

| # | 改善 | 難度 | 工数 | 期待効果 | ソース |
|---|------|------|------|----------|--------|
| C1 | **Two-way Picard coupling** | Medium | 100行 | 真の mechano-biological feedback | vem_stress_assisted_diffusion |
| C2 | **Sparse assembly** | Easy | 20行/solver | 10k+ nodes 対応 | mVEM |
| C3 | **Mixed (u,p) formulation** | Medium | 80行 | 非圧縮性 biofilm の locking 回避 | mVEM, stress-diffusion |
| C4 | IKM式 stabilization | Easy | 10行 | E(DI) 変動時の安定性向上 | MinhNguyenIKM/vem |
| C5 | Space-time VEM ベンチマーク | Medium | 比較のみ | Wriggers 論文との対比 | VEM-spcae-time-dynamic |
| C6 | Mass matrix 追加 | Medium | 40行 | 動的 VE time-stepping | mVEM (heatVEM) |

**推奨順: C2 → C1 → C3**

### C1: Two-way Picard coupling (最重要)
- 現在: species ODE → DI → E → VEM (one-way)
- 改善: stress → species growth rate 修正 → ODE → DI → E → VEM (Picard loop, tol=1e-8)
- ソース: `vem_stress_assisted_diffusion/bin/main_k21_convergence.m` L105-170
- **バイオフィルムの mechano-sensing を表現**

### C5: Space-time VEM 比較 (論文ネタ)
- Xu, Junker, **Wriggers** (CMAME 2024) — **IKMの教授の論文**
- 彼らの3D extrusion式 vs 我々の2D anisotropic Voronoi 式を比較
- 研究室内での位置づけを明確化

---

## 実行ロードマップ

### Phase 1: Quick Wins (1日) ✅ 完了 2026-03-12
- [x] A1: DeepONet 出力正規化 → `deeponet_hamilton.py` load_data() + predict_trajectory()
- [x] A2: Sine trunk activation → `TrunkNet.__call__` jnp.sin
- [x] B1: Systematic resampling → `tmcmc.py` systematic_resample() (unique +34.5%)
- [x] B2: Weighted covariance → `tmcmc.py` aweights=weights_before_resample
- [x] B3: Adaptive step count → `tmcmc.py` ceil(log(0.01)/log(1-acc))
- [x] B4: scalem formula → `tmcmc.py` 1/9 + 8/9·R (smooth, threshold-free)

**ベンチマーク**: 2D Gaussian N=500×10 trials: mean error -30.3%, unique ratio +34.5%

### Phase 2: High Impact (3日) ✅ 完了 2026-03-12
- [x] A3: POD-DeepONet → `deeponet_hamilton.py` PODDeepONet + train_pod() + CLI `train-pod`
- [ ] A4: 条件統合モデル (未着手)
- [x] B5: CoV-based beta schedule → `tmcmc.py` beta_schedule="cov", target_cov=1.0
- [x] C2: VEM sparse assembly → `vem_elasticity.py` COO triplet + scipy.sparse

### Phase 3: Advanced (1週間) ✅ 完了 2026-03-12
- [x] A5: Multi-fidelity ASMC correction → `tmcmc.py` multifidelity_correct()
  - Surrogate TMCMC → HF importance weight 修正 → systematic_resample
  - correction_fraction で HF 評価コスト制御
  - テスト: バイアス 0.226→0.075 (3× 改善)
- [x] C1: Two-way Picard coupling → `vem_elasticity.py` picard_coupled_solve()
  - stress_dependent_diffusivity(): M = m₀·exp(-m₁·σ_vol)
  - compute_element_stress(): VEM 変位場→要素応力
  - Picard loop: tol=1e-8, max_iter=20, 2 iters で収束確認
- [x] C3: Mixed (u,p) formulation → `vem_elasticity.py` vem_elasticity_mixed()
  - Saddle-point [K,G; G^T,-M_p], P0 圧力, 体積ロッキング回避

### Phase 4: Paper/Future
- [ ] A4: 条件統合モデル (theta_dim 20→24, condition one-hot)
- [ ] C4: IKM式 stabilization
- [ ] C5: Space-time VEM benchmark vs Wriggers
- [ ] C6: Mass matrix for dynamic VE
- [ ] B6: DREAMzs proposal
