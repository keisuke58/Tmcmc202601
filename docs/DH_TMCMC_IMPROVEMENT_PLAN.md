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

```bash
cd deeponet
python gradient_tmcmc_nuts.py --condition Dysbiotic_HOBIC \
  --compare-all --n-particles 500 --real
```

- **効果**: PAPER_OUTLINE より DH で RW 0.45 → NUTS 0.97（**2.2× acceptance**）
- **注意**: ODE ベースでは JAX 化が必要。現状は DeepONet 経由で NUTS が使える。

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

---

## 9. 参照

| 項目 | パス |
|------|------|
| 推定スクリプト | `data_5species/main/estimate_reduced_nishioka.py` |
| DH/CH ハイパラ探索 | `data_5species/main/run_dh_ch_hyperparam_sweep.sh` |
| Prior bounds | `data_5species/model_config/prior_bounds.json` |
| DH 事後 | `data_5species/_runs/dh_baseline/` |
| NUTS 比較 | `deeponet/gradient_tmcmc_nuts.py` |
| Importance-weighted | `deeponet/run_importance_weighted_pipeline.sh` |
| DH MAE 悪化対策 | `project_e/docs/ISSUE_Dysbiotic_HOBIC_MAE_degradation.md` |
