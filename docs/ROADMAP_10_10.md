# 10/10 を目指すためのロードマップ

**参照**: 評価 8.5/10 を踏まえた改善プラン（2026-03-05）

**現時点の見積もり**: **約 9.0/10**（Phase 1 完了、実験検証は未達）

---

## 1. 実験検証の追加（最重要）★

**フォルダ**: `experimental_verification_DI/`

| 課題 | 現状 | 10にするためのプラン |
|------|------|----------------------|
| **DI→E の constitutive law** | 現象論的、文献との整合性のみ | **同一サンプルでの 16S rRNA + AFM 測定**（Ohmura et al. 2024 のような手法）で直接検証 |
| **予測の妥当性** | 数値シミュレーションの検証のみ | **新規実験データ**での out-of-sample 予測（例：未使用の HOBIC 条件）で評価 |

**GitHub Issue**: `.github/ISSUE_TEMPLATE/experimental_verification_DI.md` から作成

---

## 2. 方法論の強化

| 課題 | 現状 | 10にするためのプラン |
|------|------|----------------------|
| **Hill gate の \(K\), \(n\)** | 固定値の根拠はあるが感度解析が少ない | **\(K\), \(n\) の感度解析**を本文に追加 |
| **DeepONet の 2 パラメータ** | \(a_{32}\), \(a_{35}\) の overlap が低い | **importance-weighted retraining** や **active learning** で 17/20 → 19/20 以上に改善 |
| **対称行列仮定** | 変分原理の帰結として説明 | **非対称モデルとの比較**を検討 |

---

## 3. 結果の説得力を高める

| 課題 | 現状 | 10にするためのプラン |
|------|------|----------------------|
| **P. gingivalis の終末 surge** | 過小評価の傾向 | **状態依存相互作用** \(\mathbf{A}(\mathrm{pH}, \mathrm{metabolites})\) を導入 |
| **実験データの記述** | 簡潔 | **データセットの詳細**（サンプル数、時間点、測定誤差等）を Methods に明記 |
| **ベイズ因子の解釈** | pseudo BF は大きい | **標準的な Bayes factor** を可能な範囲で計算 |

---

## 4. 文章・構成の改善

| 課題 | 現状 | 10にするためのプラン |
|------|------|----------------------|
| **Abstract** | 情報が多め | 主要結論を 2–3 点に絞り、簡潔に |
| **Limitations** | 6 項目に整理 | 各項目を 1–2 文で簡潔に |
| **図のキャプション** | 十分 | 主要メッセージを 1 文で示す形に統一 |

---

## 5. 優先順位付きロードマップ

```
Phase 1（短期・3–6ヶ月）
├── ✓ Hill gate の K, n 感度解析（論文に 1 文追加済み）
├── ✓ 実験データの詳細記述
├── ✓ Abstract の簡潔化
└── ✓ DeepONet の importance-weighted retraining（完了、overlap 17/20 維持）

Phase 2（中期・6–12ヶ月）
├── 状態依存相互作用の導入（終末 surge の改善）
├── 非対称モデルとの比較
└── 新規実験データでの out-of-sample 検証

Phase 3（長期・1–2年）
├── 16S + AFM の同時測定による DI→E の実験検証  ★
└── 臨床応用のパイロット検討
```

---

## 6. 10/10 達成条件

| 項目 | 達成条件 |
|------|----------|
| **実験検証** | DI→E の constitutive law が、同一サンプルでの実験データで検証されている |
| **予測の妥当性** | 未使用データでの予測精度が定量的に示されている |
| **方法論の完全性** | 主要なハイパーパラメータの感度が評価され、結論が妥当である |

---

## 7. 実装済み（2026-03-05）

| 項目 | 実装 |
|------|------|
| Hill gate K, n 感度解析 | `make hill-sensitivity` |
| DeepONet importance-weighted | `generate_training_data.py --posterior-dir path` |
| 実験データの詳細記述 | Methods Table~\ref{tab:exp_data_detail} |
| Abstract 簡潔化 | 実施済み |
| Limitations 簡潔化 | 各 1–2 文に |
| 状態依存相互作用 | 設計ドラフト `docs/state_dependent_interaction_draft.md` |
| 非対称モデル比較 | 設計ドラフト `docs/asymmetric_model_comparison.md` |
