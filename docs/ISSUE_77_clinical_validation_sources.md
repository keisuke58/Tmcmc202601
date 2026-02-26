# 臨床検証 — データソース調査

> **Issue:** [#77](https://github.com/keisuke58/Tmcmc202601/issues/77)
> **参照:** Heine et al. 2025 (F1), [GNN README](https://github.com/keisuke58/Tmcmc202601/blob/master/gnn/README.md)

---

## 目的

患者サンプル（唾液中 16S）から DI を推定し、インプラント周囲炎リスク（BOP, PPD 等）と相関させる。

---

## データソース候補

### 1. HMP (Human Microbiome Project)

- **URL:** https://www.hmpdacc.org/
- **内容:** 口腔・腸管等の 16S データ。健康人中心。
- **本研究での利用:** GNN の Phase 2 で HMP oral 16S を使用予定（`gnn/download_hmp.py`）
- **臨床検証向き:** 健康 vs 疾患の比較には追加の臨床メタデータが必要

### 2. HOMD (Human Oral Microbiome Database)

- **URL:** https://www.homd.org/
- **内容:** 16S RefSeq、abundance データ、系統樹
- **用途:** 種同定・分類の参照。生データではなく参照 DB

### 3. ENA / SRA (European Nucleotide Archive / Sequence Read Archive)

- **検索例:** "periodontitis 16S", "peri-implantitis microbiome"
- **論文例:**
  - Kim et al. 2023: ペリインプラント炎 vs 歯周病の 16S 比較 (JPIS)
  - Sanz-Martin et al.: 健康 vs 疾患インプラント部位の Illumina 16S
  - Koyanagi et al.: 健康・インプラント周囲粘膜炎・ペリインプラント炎の pyrosequencing
- **課題:** 各論文の Data Availability を確認し、SRA アクcession を取得

### 4. 公開データセット（論文付随）

| 論文 | 疾患 | データ | 備考 |
|------|------|--------|------|
| Abusleme et al. 2013 | 歯周病 vs 健康 | 16S | 歯肉縁下プラーク |
| Koyanagi et al. 2013 | インプラント周囲炎 | 16S | SRA に登録の可能性 |
| Frontiers 2020 (FCIMB) | ペリインプラント炎 | Metagenomic + 16S | 統合解析 |
| Kim et al. 2023 | ペリインプラント炎 vs 歯周病 | 16S | JPIS |

### 5. MHH (Hannover Medical School) 共同研究

- **Stiesch グループ:** Heine et al. 2025 の実験データ提供元
- **利点:** 5 種組成・HOBIC 条件と直接対応。臨床サンプルへのアクセス可能性
- **進め方:** IKM–MHH 連携で患者コホートの 16S + 臨床指標を取得

---

## パイプライン案

```
患者唾液/歯肉縁下プラーク
    → 16S シーケンス
    → 5 菌種 (So, An, Vd, Fn, Pg) へのマッピング
    → DI = -Σ φ_i log φ_i を計算
    → BOP, PPD, インプラント周囲炎診断と相関解析
```

### 5 菌種マッピング

- HMP/GNN で使用する oral taxa と本研究 5 種の対応表が必要
- `gnn/scripts/extract_hmp_oral.R` を拡張して臨床データ用マッピングを追加

---

## タスクチェックリスト

- [ ] SRA で "peri-implantitis 16S" を検索し、利用可能なデータセットをリスト化
- [ ] Abusleme 2013, Koyanagi 2013 の Data Availability を確認
- [ ] HMP oral の臨床メタデータ（疾患有無）の有無を確認
- [ ] MHH 共同研究の可能性を IKM 経由で打診
- [ ] 5 菌種マッピング表を `gnn/` に追加
