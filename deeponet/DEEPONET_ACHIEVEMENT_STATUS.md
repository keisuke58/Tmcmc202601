# DeepONet 達成状況サマリ

**更新日**: 2026-02-26

---

## 1. 高速化の根拠（正式値: ~80× per-sample, ~29× E2E）

### 実測値（Wiki / ベンチマークより）

| 計測種別 | ODE | DeepONet | Speedup |
|----------|-----|----------|---------|
| **単一 eval** (per-sample) | ~10 ms | ~0.03 ms | **78–90×** (条件別) |
| **E2E TMCMC** (200 particles) | 44 s | 1.5 s | **29×** |

### 条件別 単一 eval speedup（Wiki）

| Condition | Speedup |
|-----------|---------|
| Commensal Static | 78× |
| Commensal HOBIC | 88× |
| Dysbiotic Static | 90× |
| Dysbiotic HOBIC | 84× |

### ドキュメント統一（2026-02-26 実施）

- README, estimate_reduced_nishioka.py, evaluator.py, surrogate_tmcmc.py, docs/index.html を
  **「~80× per-sample, ~29× E2E TMCMC」** に統一済み。

---

## 2. TMCMC への組み込み

| 項目 | 状態 |
|------|------|
| `--use-deeponet` フラグ | ✅ 実装済み |
| DeepONetEvaluator | ✅ drop-in replacement |
| チェックポイント自動検出 | ✅ 条件名 → v2 優先 |
| 4条件での本番実行 | ⚠️ 要検証（JAX/Equinox 環境が必要） |

**実行例**:
```bash
# klempt_fem 等の JAX 環境で
python data_5species/main/estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --use-deeponet --n-particles 1000
```

---

## 3. MAP θ での精度評価

### 評価スクリプト

`deeponet/eval_map_accuracy.py` で 4 条件の θ_MAP について ODE vs DeepONet を比較。

### 結果（2026-02-26）

| Condition | MSE | MAE | Rel.Err |
|-----------|-----|-----|---------|
| Commensal_Static | 2.13e-02 | 0.120 | **61.6%** |
| Commensal_HOBIC | 8.85e-03 | 0.086 | **44.1%** |
| Dysbiotic_Static | 1.42e-02 | 0.101 | **51.8%** (v2) |
| Dysbiotic_HOBIC | 1.46e-03 | 0.022 | **11.2%** (50k) |

- **Dysbiotic_HOBIC**: 11% rel err（50k モデル）→ TMCMC 本番に適している
- **CS/CH/DS**: θ_MAP が学習データ範囲外の可能性があり、誤差が大きい
- v2 (MAP-centered) で DS は 70% → 52% に改善

### チェックポイント推奨

| Condition | 推奨 checkpoint |
|-----------|----------------|
| Dysbiotic_HOBIC | checkpoints_Dysbiotic_HOBIC_50k |
| Dysbiotic_Static | checkpoints_DS_v2 |
| Commensal_* | デフォルト（MAP 周辺 importance sampling 強化で改善） |

### CS/CH importance sampling 強化（実装済み）

`generate_training_data.py` に条件別 preset を追加:

| Condition | map_frac | map_std_frac |
|-----------|----------|--------------|
| Commensal_Static | 0.5 | 0.15 |
| Commensal_HOBIC | 0.5 | 0.15 |
| Dysbiotic_Static | 0.4 | 0.12 |
| Dysbiotic_HOBIC | 0.3 | 0.1 |

再学習例:
```bash
python generate_training_data.py --condition Commensal_Static --n-samples 10000
# → preset で map_frac=0.5, map_std=0.15 が自動適用
```

---

## 4. 次のアクション

1. [ ] 4条件で `--use-deeponet` 付き TMCMC を本番実行し、ODE 版と posterior を比較
2. [x] 384× の記載を 80× / 29× に統一（README 等）
3. [x] CS/CH の MAP 精度改善（importance sampling preset 追加）
