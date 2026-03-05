# DI→E 実験検証 詳細プラン

**作成日**: 2026-03-05
**参照**: 10/10 評価に向けた改善プラン

---

## 1. 課題の整理

| 課題 | 現状 | 10にするためのプラン |
|------|------|----------------------|
| **DI→E の constitutive law** | 現象論的、文献との整合性のみ | **同一サンプルでの 16S rRNA + AFM 測定**（Ohmura et al. 2024 のような手法）で、組成と弾性率の同時測定データを取得し、DI→E を直接検証する |
| **予測の妥当性** | 数値シミュレーションの検証のみ | **新規実験データ**での out-of-sample 予測（例：未使用の HOBIC 条件）で、モデルの予測精度を評価する |

---

## 2. 手法: 16S rRNA + AFM 同時測定

### 2.1 参照手法（Ohmura et al. 2024）

- **論文**: Ohmura, Wakita, Nishimura (2024). "Spatially resolved correlation between polysaccharide distribution and local elastic modulus in living 3D biofilms." *Soft Matter*.
- **手法**: 生きた 3D バイオフィルムにおいて、多糖類分布と局所弾性率の空間相関を同時測定。
- **応用**: 5 種口腔バイオフィルム（So, An, Vd, Fn, Pg）に対して、16S rRNA による種組成 → DI と、AFM ナノインデンテーションによる弾性率を同一サンプルで取得。

### 2.2 測定フロー（想定）

```
サンプル調製（HOBIC 条件: CS, CH, DS, DH）
    ↓
同一サンプルを 2 経路で処理:
  (A) 16S rRNA シーケンス → 種組成 φ_i → DI = f(φ)
  (B) AFM ナノインデンテーション → 弾性率 E
    ↓
同一サンプルでの (DI, E) ペアを複数取得
    ↓
E(DI) constitutive law のフィッティング・検証
```

### 2.3 必要なリソース

- 16S rRNA シーケンス（既存 HOBIC データの拡張 or 新規測定）
- AFM ナノインデンテーション（Pattem 2018, 2021 と同様のプロトコル）
- 同一サンプルからの両測定を可能にする実験デザイン

---

## 3. Phase 別ロードマップ

### Phase 1（短期・3–6 ヶ月）

- [x] 文献データの整理（Pattem 2018/2021, Gloag 2019 → `literature_data.py`）
- [x] 実験デザイン案の作成（サンプル数推定: `make exp-verify-design`）
- [x] **実験プロトコル書**（`EXPERIMENTAL_PROTOCOL.md`）
- [x] **チェックリスト**（`CHECKLIST.md`）
- [x] **文献整合性レポート**（`make exp-verify-literature`）
- [ ] 文献調査の深化（Ohmura 2024, Pattem 2018/2021 のプロトコル詳細）
- [ ] 既存 HOBIC データとの整合性確認
- [ ] 共同研究者・実験施設の確保

### Phase 2（中期・6–12 ヶ月）

- [ ] パイロット実験（少数サンプルでの 16S + AFM 同時測定）
- [ ] DI→E の予備的フィッティング
- [ ] 新規 HOBIC 条件での out-of-sample 予測実験

### Phase 3（長期・1–2 年）

- [ ] 本格的な 16S + AFM 同時測定データの取得
- [ ] DI→E constitutive law の実験的検証
- [ ] 論文 Methods / Results への反映

---

## 4. 成果物

| 成果物 | 説明 |
|--------|------|
| 実験プロトコル | 16S + AFM 同時測定の手順書 |
| データセット | (DI, E) ペアの測定データ |
| 検証レポート | E(DI) モデルとの比較・適合度 |
| 論文更新 | Methods に実験検証を追加、Limitations を更新 |

---

## 5. 関連ドキュメント

- 論文 LaTeX: `data_5species/docs/nishioka_latex20260218.tex`（Limitations, Future Work）
- Wiki: `.wiki/Viscoelastic-SLS-Model.md`（AFM creep + 16S の推奨）
- 文献: `data_5species/docs/references_ikm.bib`（Ohmura2024, Pattem2018, Pattem2021）
