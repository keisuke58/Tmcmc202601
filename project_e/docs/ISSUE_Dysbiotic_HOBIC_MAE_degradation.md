# Dysbiotic_HOBIC Phase 2 MAE 悪化の原因と対策

## 現象

Phase 2（合成データ拡張）実行後、Dysbiotic_HOBIC 条件のみ MAE が悪化する。

| 条件 | Phase 1 MAE | Phase 2 MAE |
|------|-------------|-------------|
| Commensal_Static | 0.82 | 0.55 ✓ |
| Commensal_HOBIC | 1.00 | 0.64 ✓ |
| Dysbiotic_Static | 0.88 | 0.55 ✓ |
| Dysbiotic_HOBIC | 2.27 | **2.64** ✗ |

## 診断結果

`python diagnose_theta_distribution.py --synthetic data/synthetic_all_N8000.npz` の出力より：

### Dysbiotic_HOBIC の θ 分布のずれ

| 指標 | TMCMC posterior | Synthetic | θ_MAP |
|------|-----------------|-----------|-------|
| mean[0:5] | [2.45, 1.31, 1.92, 1.45, 1.50] | [1.96, 1.40, 1.98, 1.69, 1.77] | [0.52, 1.82, 0.74, 2.08, 2.42] |
| MAE(post, syn) | — | **1.76** | — |
| MAE(post, MAP) | — | — | 1.65 |
| ~95% CI 幅 | 8.27 | 4.70 | — |

### 根本原因

1. **合成データの θ 分布が TMCMC 事後と大きくずれている**
   - 合成: 70% prior（一様）+ 30% MAP 周辺（σ=0.15×range）
   - TMCMC 事後: 実測データに基づく事後分布
   - Dysbiotic_HOBIC は **prior が広く**（全20次元 free）、かつ **事後平均が θ_MAP から大きくシフト**している
   - その結果、合成 mean と事後 mean の MAE が **1.76** と他条件（0.1〜0.2）より桁違いに大きい

2. **θ_MAP と事後平均の乖離**
   - θ_MAP は最尤点だが、事後平均は別の位置（MAE 1.65）
   - 合成の 30% が MAP 周辺 → 事後の中心から外れた分布を学習

3. **他条件で改善した理由**
   - Commensal/Static 条件は **locks が多く** prior が狭い
   - 事後と prior/MAP のずれが小さい（MAE 0.1〜0.2）

## 対策案

### A. 合成データ生成の改善（実装済み ✓）

**A1. 事後サンプルを活用した合成データ**

`generate_synthetic_data.py` に `posterior_frac` を追加済み。Dysbiotic_HOBIC は preset で `posterior_frac=0.5` を自動適用：

```bash
# 4条件一括生成（Dysbiotic_HOBIC は事後 50% を自動使用）
python generate_synthetic_data.py --n-samples 5000 --all-conditions

# 明示的に指定する場合
python generate_synthetic_data.py --n-samples 5000 --all-conditions --posterior-frac 0.5
```

- 事後サンプルを bootstrap で抽出し、ODE で y_obs を再生成
- 事後分布に忠実な (y_obs, θ) ペアを学習に使う

**A2. Dysbiotic_HOBIC 専用の map_frac 増加**

```python
# generate_synthetic_data.py の CONDITION_PRESETS に追加
"Dysbiotic_HOBIC": {"map_frac": 0.6, "map_std_frac": 0.08}
```

- MAP 周辺の割合を増やし、広がりを狭める
- ただし θ_MAP ≠ 事後平均 のため、根本解決にはならない

**A3. 事後平均を中心にサンプリング**

- `load_theta_map` の代わりに `theta_mean`（事後平均）を読み、それを中心にガウスサンプリング
- `dh_baseline` に `theta_mean.json` があれば利用可能

### B. 学習時の重み付け

- TMCMC 事後データに重みを付与（例: 2〜3倍）
- 合成データが 87% を占める現状では、事後が埋もれやすい
- `WeightedRandomSampler` 等で事後サンプルの抽出確率を上げる

### C. 条件別の合成戦略

- Dysbiotic_HOBIC のみ「事後ベース」の合成を使用
- 他 3 条件は現行の prior+MAP で継続

## 診断コマンド

```bash
cd project_e
python diagnose_theta_distribution.py --synthetic data/synthetic_all_N8000.npz
```

## 参考

- `generate_synthetic_data.py`: 合成データ生成（prior/MAP サンプリング）
- `load_posterior_data.py`: TMCMC 事後ロード（Dysbiotic_HOBIC: y_obs←dysbiotic_hobic_1000p, θ←dh_baseline）
- `prior_bounds.json`: Dysbiotic_HOBIC は locks=[] で prior が広い
