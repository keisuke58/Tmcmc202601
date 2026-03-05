# Dysbiotic_HOBIC (DH) TMCMC 精度改善プラン

**現状**: MAP RMSE DH=0.075（他条件 0.054–0.063）、20 free params、DeepONet overlap 17/20

---

## 1. 現状の課題

| 項目 | DH | 他条件 |
|------|-----|--------|
| MAP RMSE | **0.075** | 0.054–0.063 |
| Free params | **20**（最多） | 9–15 |
| Prior locks | なし | あり |
| 事後平均 vs θ_MAP | MAE 1.65（乖離大） | MAE 0.1–0.2 |
| DeepONet overlap | 17/20（a32, a35 低） | — |

**根本原因**:
- 20次元探索空間が広く、prior が一様で制約が弱い
- θ_MAP と事後平均が乖離 → 多峰性の可能性
- Pg 関連パラメータ（a32, a35, a45）の事後が広い

---

## 2. 改善策（優先度順）

### A. ODE-based TMCMC の強化（即効性あり）

#### A1. 粒子数・ステージ数の増加

```bash
# 現行: dysbiotic_hobic_1000p は 1000p×12 stages、他条件は 150p でも収束実績あり
# DH 推奨: 300–500 particles（150 で回した実績を踏まえ、500 程度で十分な場合も）
cd data_5species/main
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init \
  --checkpoint-every 5
```

- **根拠**: CS/CH/DS は 150p で収束実績あり。DH は 20 free のため 300–500p を試す。2000 は過剰の可能性。
- **計算時間**: 500p×30 stages で ODE なら 1–2 時間程度

#### A2. NUTS/HMC mutation の使用

**JAX ODE 版（推奨）** — DeepONet 不要、サロゲート誤差ゼロ:

```bash
cd data_5species/main
PYTHON=$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python
$PYTHON estimate_reduced_nishioka_jax.py --condition Dysbiotic --cultivation HOBIC \
  --n-particles 200 --use-exp-init --mutation nuts
```

**DeepONet 版**:

```bash
cd deeponet
python gradient_tmcmc_nuts.py --condition Dysbiotic_HOBIC \
  --compare-all --n-particles 500 --real
```

- **効果**: PAPER_OUTLINE より DH で RW 0.45 → NUTS 0.97（**2.2× acceptance**）
- **JAX ODE**: `hamilton_ode_jax.py` + `tmcmc_nuts_engine.py` で ODE 直接 NUTS が可能。

**JAX ODE テスト／本番**:

```bash
cd data_5species/main
bash run_jax_ode_nuts.sh --test              # 50p, 500 steps（数分）
bash run_jax_ode_nuts.sh --production        # 200p, 2500 steps（30分〜1時間）
```

**GPU**: `jax[cuda12]` 入りなら自動で GPU 使用。CPU より数倍〜10倍速くなる場合あり。

#### A3. 実験初期値の使用（use_exp_init）

```bash
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --use-exp-init
```

- DH の `phi_init_exp` は実験 Day 1 の組成。初期値がデータに近いと収束が早く、局所解への陥りが減る。

---

### B. Prior の絞り込み

#### B1. 2段階アプローチ（粗探索 → 精密化）

1. **Phase 1**: 現行 prior で TMCMC 実行 → 事後サンプル取得
2. **Phase 2**: 事後の 95% HDI で prior bounds を narrow に設定 → 再実行

```python
# prior_bounds_narrow.json は既存だが DH は wide のまま
# dh_baseline の事後から bounds を自動生成するスクリプトを追加可能
```

#### B2. GNN prior の利用

```bash
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --gnn-prior-json gnn/data/gnn_prior_predictions_v2.json
```

- GNN が interaction パラメータを制約し、初期粒子を高尤度領域に集中させる。
- Fig 23: GNN prior で active edges σ が 4.47× 向上。

---

### C. DeepONet パイプラインの改善（overlap 向上）

#### C1. 事後周辺のさらなる濃縮

```bash
# posterior_frac を 0.5 → 0.7–0.8 に増加
python generate_training_data.py --condition Dysbiotic_HOBIC \
  --n-samples 50000 --posterior-dir ../data_5species/_runs/dh_baseline \
  --posterior-frac 0.7
```

- a32, a35 周辺のサンプルを増やし、DeepONet の学習を強化。

#### C2. Active learning（将来）

- ODE 事後で overlap が低いパラメータ（a32, a35）周辺を重点的にサンプリングし、DeepONet を再学習。

---

### D. モデル・尤度の調整

#### D1. Pg の重み付け（λ_Pg）

- 論文: λ_Pg=5 で Pg の残差を重視。DH は Pg の終末 surge が重要。
- `build_likelihood_weights` で λ_Pg を 5–10 に増やすと Pg の fit が改善する可能性。

#### D2. sigma_obs の再推定

- DH の `sigma_obs` が他条件と異なる場合、尤度のスケールが変わる。条件別の sigma 推定を検討。

#### D3. 種別 sigma（--species-sigma）

- V. dispar は栄養枯渇でモデルが追いきれず RMSE が大きくなりやすい。`--species-sigma` で Vd に 2× の noise を割り当て、過学習を抑える。
- `diagnose_fit_gap` で Vd RMSE > 0.20 が指摘されている条件向け。

```bash
python estimate_reduced_nishioka.py ... --species-sigma
```

#### D4. λ_Pg / λ_late の増加

- 現行: `--lambda-pg 5 --lambda-late 3`（論文値）
- DH/CH: Pg の終末 surge が重要 → `--lambda-pg 8 --lambda-late 4` で試す。

```bash
python estimate_reduced_nishioka.py ... --lambda-pg 8 --lambda-late 4
```

#### D5. 多チェーン・多シード

- `--n-chains 3` で R-hat の安定性を確認。
- 複数 seed（`--seed 42` / `--seed 101`）で実行し、MAP のばらつきを確認。

#### D6. dh_v3_tight_bounds の利用

- 既存の `dh_v3_tight_bounds` は prior を絞った run。同様の narrow prior で再実行すると収束が早くなる可能性。

---

## 2.5 推定手法の使い分け

| 手法 | 用途 | 速度 | 精度 | NUTS | 分散実行 |
|------|------|------|------|------|----------|
| **NumPy ODE** (`estimate_reduced_nishioka.py`) | 本番・ハイパラ sweep | 中 | 正確 | 不可 | 可（複数サーバ） |
| **JAX ODE** (`estimate_reduced_nishioka_jax.py`) | NUTS で acceptance 向上 | CPU:遅 / GPU:速 | 正確 | 可 | 要 JAX（GPU 可） |
| **DeepONet** (`gradient_tmcmc_nuts.py`) | 高速・大量サンプル | 最速 | 近似（overlap 17/20） | 可 | 要 GPU 推奨 |

**推奨**:
- **ハイパラ探索・分散実行**: NumPy ODE（`run_dh_ch_hyperparam_distributed.sh`）
- **NUTS で acceptance 向上したい**: JAX ODE（`run_jax_ode_nuts.sh --production`）
- **サロゲート誤差を避けたい**: JAX ODE または NumPy ODE
- **大量サンプル・高速**: DeepONet

---

## 3. 推奨実行順序

| 順序 | 施策 | 期待効果 | 所要時間 |
|------|------|----------|----------|
| 1 | A1 + A3: 500p, 30 stages, use_exp_init | MAP RMSE 0.075 → 0.06–0.07 | 1–2 時間 |
| 2 | B2: GNN prior | 収束安定化、ESS 向上 | 数時間 |
| 3 | C1: posterior_frac=0.7 で再学習 | overlap 17/20 → 18–19/20 | 1日 |
| 4 | A2: NUTS（DeepONet 経由） | acceptance 向上 | 数時間 |

---

## 4. 実行コマンド例（最小構成）

```bash
# DH のみ再推定（ODE、500 particles — 150p 実績を踏まえ 300–500 で十分な場合も）
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init \
  --checkpoint-every 5 \
  --out-dir _runs/dh_500p30_$(date +%Y%m%d_%H%M%S)
```

---

## 5. RMSE に効くハイパラ（σ 以外）

| ハイパラ | 効果 | 現状 |
|----------|------|------|
| **σ (sigma_obs)** | 小→厳しめ fit、大→緩め | IQR/1.35 で条件別推定。0.5× で感度解析中 |
| **λ_Pg** | Pg 残差の重み | デフォルト 1、論文 5 |
| **λ_late** | 終末時間点の重み | デフォルト 1、論文 3 |
| **use_exp_init** | 初期値＝実験 Day1 | 推奨 |
| **n_particles, n_stages** | 探索の十分性 | 500×30 推奨 |
| **species_sigma** | Vd 過学習抑制 | --species-sigma |
| **GNN prior** | 初期粒子の質 | --gnn-prior-json |
| **prior bounds** | 探索範囲 | tight で収束早い |

---

## 6. σ の現状とこれ以上下げる必要

**現状の sigma_obs（条件別）**:
| 条件 | sigma_obs | 備考 |
|------|-----------|------|
| CS | 0.111 | IQR/1.35 から推定 |
| CH | 0.157 | 同上 |
| DS | 0.247 | 同上 |
| DH | 0.232 | 同上 |

**sigma_scale=0.5 適用後**: 上記の半分（0.056–0.124）

**これ以上下げる必要**: まず 0.5× の結果を確認。0.25× などは過学習リスク大。測定ノイズの実態を超えて σ を下げるのは妥当でない。

---

## 7. 追加オプション一覧（試す価値あり）

| オプション | 効果 | 例 |
|------------|------|-----|
| `--species-sigma` | Vd の過学習抑制 | CS/DS で Vd RMSE 大のとき |
| `--lambda-pg 8` | Pg の fit 重視 | DH/CH |
| `--lambda-late 4` | 終末時間点の重み増 | DH/CH |
| `--n-chains 3` | R-hat 安定性 | 全条件 |
| `--gnn-prior-json` | 初期粒子の質向上 | DH |
| `--sigma-obs 0.02` | 尤度スケール調整 | 条件別チューニング |

---

## 8. DH/CH ハイパラ探索

DH と CH に限定したハイパラグリッド探索:

```bash
cd data_5species/main

# 分散実行（推奨）: 7台に12本を振り分け、並列で短縮
bash run_dh_ch_hyperparam_distributed.sh

# 試行用（150p×15st）
bash run_dh_ch_hyperparam_distributed.sh --quick

# 単一サーバ
bash run_dh_ch_hyperparam_sweep.sh              # 逐次
bash run_dh_ch_hyperparam_sweep.sh --parallel   # DH/CH 並列
```

探索するハイパラ: `sigma_scale` (0.5, 1.0), `lambda_pg` (5, 8), `lambda_late` (3, 4)

### 実行状況（2026-03-05）

| 種別 | 状態 | 出力 |
|------|------|------|
| テスト (--quick) | 完了 | `_runs/dh_ch_hyperparam_20260305_204944/` |
| 本番 (500p×30st) | 実行中 | `_runs/dh_ch_hyperparam_20260305_210702/` |

**結果が出たら**: 本ドキュメントと論文に RMSE 比較表を追記予定。

---

## 9. スコア定義と満点達成へのアクション

### 満点の定義（100点満点）

| 項目 | 配点 | 満点基準 | 現状 | 現状得点 |
|------|------|----------|------|----------|
| **MAP RMSE (DH)** | 30 | ≤ 0.060（他条件並み） | 0.075 | 約 12 |
| **事後較正 (MAE)** | 25 | 事後平均 vs θ_MAP の MAE ≤ 0.2 | 1.65 | 約 0 |
| **DeepONet overlap** | 25 | 20/20 パラメータで > 0.95 | 17/20 | 約 21 |
| **収束診断** | 10 | R-hat < 1.05, ESS > 0.3N | 要確認 | 仮 8 |
| **パイプライン一貫性** | 10 | DI credible interval が他条件と同程度 | DH が 35× 広い | 約 0 |

**現状合計: 約 41/100 点**

### 満点達成のためのアクション（項目別）

| 項目 | やること | コマンド・手順 |
|------|----------|----------------|
| **MAP RMSE** | A1+A3+D4: 500p×30st, use_exp_init, λ_Pg=8, λ_late=4 | 下記 Step 1 |
| **事後較正** | A1+A3+B2: 500p×30st, use_exp_init, GNN prior | 下記 Step 1 + Step 2 |
| **DeepONet overlap** | C1: posterior_frac=0.7 で再学習 | 下記 Step 3 |
| **収束診断** | D5: n-chains=3, 複数 seed で R-hat/ESS 確認 | `--n-chains 3 --seed 42` 等 |
| **パイプライン一貫性** | 事後較正の改善に伴い DI CI が縮小 | Step 1–2 の副産物 |

### 実行ステップ（優先順）

**Step 1（即効・1–2時間）**: ODE-TMCMC の強化

```bash
cd data_5species/main
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init --lambda-pg 8 --lambda-late 4 \
  --checkpoint-every 5 \
  --out-dir _runs/dh_step1_$(date +%Y%m%d_%H%M%S)
```

→ MAP RMSE, 事後較正 (MAE) の改善を確認。fit_metrics.json, theta_MEAN vs theta_MAP を比較。

**Step 2（数時間）**: GNN prior を追加

```bash
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init --gnn-prior-json gnn/data/gnn_prior_predictions_v2.json \
  --lambda-pg 8 --lambda-late 4 \
  --out-dir _runs/dh_step2_$(date +%Y%m%d_%H%M%S)
```

→ 事後較正・ESS のさらなる改善。Step 1 の事後を `--posterior-dir` に渡して 2 段階 prior も検討可。

**Step 3（1日）**: DeepONet 再学習

```bash
cd deeponet
python generate_training_data.py --condition Dysbiotic_HOBIC \
  --n-samples 50000 --posterior-dir ../data_5species/_runs/dh_step2_* \
  --posterior-frac 0.7
# posterior-dir は Step 2 の出力ディレクトリを指定。学習 → TMCMC → overlap 評価（Fig22）
```

→ overlap 17/20 → 18–20/20 を目指す。

**Step 4（オプション）**: NUTS で acceptance 向上

```bash
python gradient_tmcmc_nuts.py --condition Dysbiotic_HOBIC \
  --compare-all --n-particles 500 --real
```

### チェックリスト（満点に向けて）

- [ ] Step 1 実行 → MAP RMSE ≤ 0.065 を確認
- [ ] Step 1 実行 → 事後平均 vs θ_MAP の MAE ≤ 0.5 を確認
- [ ] Step 2 実行 → MAE ≤ 0.2 を確認
- [ ] Step 3 実行 → overlap ≥ 18/20 を確認
- [ ] DI credible interval が CS の 5 倍以内に縮小
- [ ] R-hat < 1.05, ESS > 0.3N を全条件で確認

---

## 10. 参照

| 項目 | パス |
|------|------|
| 推定スクリプト | `data_5species/main/estimate_reduced_nishioka.py` |
| JAX ODE + NUTS | `data_5species/main/estimate_reduced_nishioka_jax.py` |
| JAX Hamilton ODE | `data_5species/main/hamilton_ode_jax.py` |
| TMCMC NUTS engine | `data_5species/main/tmcmc_nuts_engine.py` |
| DH/CH ハイパラ探索 | `data_5species/main/run_dh_ch_hyperparam_sweep.sh` |
| Prior bounds | `data_5species/model_config/prior_bounds.json` |
| DH 事後 | `data_5species/_runs/dh_baseline/` |
| NUTS 比較 | `deeponet/gradient_tmcmc_nuts.py` |
| Stuttgart (GPU) 分散実行 | `deeponet/run_nuts_stuttgart.sh` |
| Importance-weighted | `deeponet/run_importance_weighted_pipeline.sh` |
| DH MAE 悪化対策 | `project_e/docs/ISSUE_Dysbiotic_HOBIC_MAE_degradation.md` |
