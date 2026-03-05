# GitHub Issue 作成用テンプレート

以下をコピーして GitHub の New Issue で使用してください。

---

## Title

`[10/10 最重要] DI→E constitutive law の実験検証`

---

## Body

### 概要

Dysbiosis Index (DI) から弾性率 \(E\) への constitutive law を、**同一サンプルでの 16S rRNA + AFM 測定**により実験的に検証する。現状は現象論的で文献との整合性のみ。10/10 評価の最重要項目。

### 課題

| 課題 | 現状 | 目標 |
|------|------|------|
| DI→E の constitutive law | 現象論的、文献との整合性のみ | 同一サンプルでの 16S + AFM で直接検証 |
| 予測の妥当性 | 数値シミュレーションの検証のみ | 新規実験データでの out-of-sample 予測 |

### 手法

- **参照**: Ohmura et al. (2024) — 生きた 3D バイオフィルムで多糖類分布と局所弾性率の同時測定
- **提案**: 5 種口腔バイオフィルム（HOBIC 条件）で 16S rRNA → DI と AFM ナノインデンテーション → E を同一サンプルで取得

### ロードマップ

- **Phase 1 (3–6 ヶ月)**: 文献調査、実験デザイン、共同研究者確保
- **Phase 2 (6–12 ヶ月)**: パイロット実験、予備的フィッティング
- **Phase 3 (1–2 年)**: 本格測定、論文反映

### 関連

- 詳細プラン: `Tmcmc202601/experimental_verification_DI/`
- 論文 Limitations: DI→E mapping の実験検証が Future Work に記載済み

### ラベル

`enhancement`, `paper`, `experimental`
