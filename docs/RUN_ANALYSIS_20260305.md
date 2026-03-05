# 実行結果の考察（2026-03-05）

**ステータス**: ✅ DONE（2026-03-05 セッション完了）

---

## 1. Hill gate K, n 感度解析

### 結果

| theta ソース | DI 範囲 | 変動幅 |
|-------------|---------|--------|
| THETA_DEMO（dysbiotic 型） | 0.8763 | 0.00% |
| dh_baseline MAP（commensal 型） | 0.1612–0.1761 | 1.48% |

### 考察

- **dysbiotic 型**: Veillonella 支配（φ_Vei ≈ 96%）、Pg ≈ 0.8%。Hill gate は Fn→Pg を制御するが、Fn が低いため gate はほぼ閉じており、K, n を変えても平衡が変わらない。
- **commensal 型**: 多種共存（DI ≈ 0.16–0.18）。K, n を変えると DI が 1.5% 程度変動するが、結論（CS > DH の順序、E の 28 倍差）には影響しない。
- **結論**: 論文の K=0.05, n=4 の選択は結論に頑健。感度解析を Methods に 1 文追加する価値あり。

---

## 2. DI→E 実験検証パイプライン

### 結果（合成データ）

| 指標 | 真値 | フィット値 | 相対誤差 |
|------|------|------------|----------|
| e_max | 909 Pa | 954 Pa | +5.0% |
| e_min | 32 Pa | 31.6 Pa | −1.1% |
| di_scale | 1.0 | 1.03 | +3% |
| exponent | 2.0 | 2.17 | +8.5% |
| R² | — | 0.989 | — |
| RMSE | — | 24.3 Pa | — |

### 考察

- 40 サンプル・12% ノイズで e_max, e_min は 5% 以内で回復。パイプラインは妥当。
- 実データ取得後は `run_validation --data data/di_e_pairs.csv` で同様にフィット可能。
- 文献オーバーレイ（Pattem, Gloag）は `plot_E_DI_with_literature` で確認済み。

---

## 3. サンプルサイズ推定（実験デザイン）

### 結果

- target: RMSE ≤ 100 Pa, R² ≥ 0.9、noise 15%
- n=15 で RMSE 成功率 100%、R² 成功率 100%
- 推奨 n_samples: **15**

### 考察

- 15% ノイズを仮定すると、**n ≥ 15** で十分な精度が期待できる。
- 実際の 16S+AFM では 15–20% 程度の変動が想定されるため、**n ≥ 20** を推奨（余裕を持たせる）。
- 4 条件 × 5 サンプル = 20 など、条件ごとのサンプル配分も検討可能。

---

## 4. DeepONet importance-weighted データ生成

### 結果

```
Loaded 300 posterior samples from dh_baseline
Importance sampling: 50 posterior + 15 MAP-centered + 35 uniform
Done: 100 samples in 2.1s (47/s, 0 failed)
```

### 考察

- `--posterior-dir` で事後サンプルを読み込み、50% を事後周辺からサンプリング可能。
- a32, a35 の overlap 改善には、**50k サンプル × posterior_frac=0.5** で再学習するのが効果的。
- 実行例: `python generate_training_data.py --condition Dysbiotic_HOBIC --n-samples 50000 --posterior-dir ../data_5species/_runs/dh_baseline --posterior-frac 0.5`

---

## 5. 総合考察

| 項目 | 状態 | 次のアクション |
|------|------|----------------|
| Hill gate 感度 | ✓ 論文に 1 文追加済み | — |
| DI→E 検証 | 合成データでパイプライン検証済み | 実データ取得後に再実行 |
| サンプルサイズ | n≥15 で十分 | 実験計画では n≥20 を推奨 |
| Importance-weighted | ✓ 完了（overlap 17/20 維持） | active learning 等で将来改善 |

---

## 6. 10/10 に向けた残りタスク

1. **実験検証**: 16S + AFM の同時測定による DI→E の直接検証（最重要）
2. ~~**DeepONet 再学習**~~: 完了（overlap 17/20 維持、改善は限定的）
3. **状態依存 A**: 実データ（pH 等）があればパイロット実装

---

## 7. 実施済みアクション（2026-03-05 追記）

### 論文への追記
- **Hill gate 感度**: Methods 3.9.1 に 1 文追加済み
  「A follow-up sensitivity analysis over K and n shows that the Dysbiosis Index (DI) varies by less than 2% across the grid for both dysbiotic and commensal equilibria, confirming that the main conclusions (CS > DH ordering, 28-fold stiffness range) are robust to the choice of Hill gate parameters.」

### Importance-weighted パイプライン
- **データ生成**: `generate_training_data.py --posterior-dir ../data_5species/_runs/dh_baseline --posterior-frac 0.5` で 50k サンプル生成（50% 事後周辺）
- **実行スクリプト**: `deeponet/run_importance_weighted_pipeline.sh`
  1. 50k importance-weighted データで DeepONet 再学習 → `checkpoints_DH_50k_importance/`
  2. TMCMC（DH のみ）→ `_runs/deeponet_DH_50k_importance/`
  3. Fig22 overlap 比較 → `--don-dir` で新事後を指定可能

```bash
# データ生成（50k、約 15–20 分）
cd deeponet && python generate_training_data.py --condition Dysbiotic_HOBIC \
  --n-samples 50000 --posterior-dir ../data_5species/_runs/dh_baseline --posterior-frac 0.5

# パイプライン実行（学習 + TMCMC + overlap 評価）
./run_importance_weighted_pipeline.sh
```

### Importance-weighted パイプライン実行結果（2026-03-05 完了）

| フェーズ | 状態 | 備考 |
|----------|------|------|
| Phase 1 データ | ✓ | 50k importance-weighted（50% 事後周辺） |
| Phase 2 DeepONet | ✓ | best val loss 0.000020、checkpoints_DH_50k_importance |
| Phase 3 TMCMC | ✓ | 1000 particles × 30 stages、_runs/deeponet_DH_50k_importance |
| Phase 4 overlap | ✓ | Fig22 生成済み |

**Overlap 結果（ODE vs importance-weighted DeepONet）**:
- High (>0.95): **17/20**
- Medium (0.5–0.95): 1/20
- Low (<0.5): 2/20
- Mean overlap: **0.901**

**考察**: importance-weighted 再学習後も overlap は 17/20 で従来と同程度。a32, a35 の改善は限定的。ODE 事後が広く、サロゲートの学習限界に近い可能性。今後の改善案: active learning、事後周辺のさらなる濃縮（posterior_frac 増）、または ODE 直接評価とのハイブリッド。

### 修正（ImportError）
- **BiofilmTSM5S**: `evaluator.py` で `tmcmc_5species_tsm.BiofilmTSM5S` が存在しないため、`BiofilmTSM_Analytical` を 5-species 用に使用するよう修正（`paper_mode=False`）。

### 実験検証の準備（2026-03-05 追記）
- **実験プロトコル**: `experimental_verification_DI/EXPERIMENTAL_PROTOCOL.md` — 16S + AFM 同時測定の手順
- **チェックリスト**: `experimental_verification_DI/CHECKLIST.md` — 実験担当者向けタスク
- **文献整合性**: `make exp-verify-literature` — レポート + fig_E_DI_literature.png
- **実データ取得後**: `run_validation --data data/di_e_pairs.csv`

### Phase 2 パイロット（2026-03-05 追記）
- **対称性検証**: `tools/verify_symmetric_A.py` — 全条件で $\|A - A^\top\|_F = 0$ を確認（`make verify-symmetric-A`）
- **時間依存 A テスト**: `tools/test_time_dependent_A.py` — A35(t), A45(t) の簡易検証（`make test-time-dependent-A`）
