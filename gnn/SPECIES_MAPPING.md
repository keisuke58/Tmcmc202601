# 5 菌種マッピング表

HMP 16S / 臨床 16S データから本研究の 5 菌種モデル (Hamilton ODE, Klempt et al. 2024) へのマッピング。

**参照:** [ORAL_EXAM_AND_ISSUES_MERGED.md](../docs/ORAL_EXAM_AND_ISSUES_MERGED.md) §9, [ISSUE_77](../docs/ISSUE_77_clinical_validation_sources.md), [HMP_PHASE2.md](HMP_PHASE2.md)

---

## 1. モデル 5 菌種一覧

| Index | 略称 | 学名 | 役割 | Genus | Species |
|-------|------|------|------|-------|---------|
| 0 | So | *Streptococcus oralis* | 初期定着菌 (pioneer) | Streptococcus | oralis |
| 1 | An | *Actinomyces naeslundii* | 初期定着菌 (commensal) | Actinomyces | naeslundii |
| 2 | Vd | *Veillonella dispar* | ブリッジ菌 (pH modulation, lactate) | Veillonella | dispar |
| 3 | Fn | *Fusobacterium nucleatum* | ブリッジ菌 (coaggregation) | Fusobacterium | nucleatum |
| 4 | Pg | *Porphyromonas gingivalis* | 後期定着菌・病原体 (keystone pathogen) | Porphyromonas | gingivalis |

---

## 2. HMP 16S へのマッピング

### 2.1 HMP16SData (R) — CONSENSUS_LINEAGE パターン

`scripts/extract_hmp_oral.R` で使用。`rowData(d)$CONSENSUS_LINEAGE` に `Genus Species` を含む OTU を部分一致でマッピング。

| モデル菌種 | HMP 検索パターン | 出力 CSV 列名 |
|------------|------------------|---------------|
| So | Streptococcus oralis | S_oralis |
| An | Actinomyces naeslundii | A_naeslundii |
| Vd | Veillonella dispar | V_dispar |
| Fn | Fusobacterium nucleatum | F_nucleatum |
| Pg | Porphyromonas gingivalis | P_gingivalis |

### 2.2 R スクリプト用 species_target 配列

```r
species_target <- c(
  "Streptococcus_oralis", "Actinomyces_naeslundii", "Veillonella_dispar",
  "Fusobacterium_nucleatum", "Porphyromonas_gingivalis"
)
```

### 2.3 Python 用マッピング (download_hmp.py, predict_hmp.py)

```python
SPECIES_MAPPING = {
    "S. oralis": ["Streptococcus", "oralis"],
    "A. naeslundii": ["Actinomyces", "naeslundii"],
    "V. dispar": ["Veillonella", "dispar"],
    "F. nucleatum": ["Fusobacterium", "nucleatum"],
    "P. gingivalis": ["Porphyromonas", "gingivalis"],
}
# predict_hmp.py の期待列名
COLS = ["S_oralis", "A_naeslundii", "V_dispar", "F_nucleatum", "P_gingivalis"]
```

---

## 3. 代替 taxa（Dysbiotic 条件）

Heine et al. 実験系では **Commensal** と **Dysbiotic** で Veillonella 株が異なる場合あり。

| モデル | 主な taxa | 備考 |
|--------|-----------|------|
| Vd (Index 2) | *V. dispar* | Commensal 条件 |
| Vd (Index 2) | *V. parvula* | Dysbiotic 条件で Orange 株としてマージ可能 |

`estimate_reduced_nishioka.py` の `SPECIES_MAP` では `"Orange": 2` (V. parvula) を Vd にマージ。

---

## 4. パイプライン

```
患者唾液 / 歯肉縁下プラーク
    → 16S シーケンス (QIIME2, DADA2 等)
    → OTU/ASV の taxonomy 照合
    → 本表に従い 5 菌種へ集約
    → φ = (φ_So, φ_An, φ_Vd, φ_Fn, φ_Pg) 正規化 (sum=1)
    → DI = -Σ φ_i log φ_i 計算
    → GNN 入力 or TMCMC prior
```

---

## 5. 参照

- **HOMD:** https://www.homd.org/ — 口腔細菌 RefSeq, 系統樹
- **Heine et al. (2025):** HOBIC 5 菌種 in vitro 実験（本研究のデータソース）
- **Klempt et al. (2024):** Hamilton ODE 5 菌種モデル
- **interaction_graph.json:** `data_5species/interaction_graph.json` — 5 菌種の相互作用構造
