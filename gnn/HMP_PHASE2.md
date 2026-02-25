# Phase 2: HMP データ統合 メモ

## データソース

| ソース | 形式 | 口腔サンプル | 取得方法 |
|--------|------|-------------|----------|
| **HMP16SData** | SummarizedExperiment | V35: 4,743 samples 中 oral あり | R: `BiocManager::install("HMP16SData")` |
| **HM16STR** | FASTA + メタデータ | 14,000+ samples, body site で絞り込み | hmpdacc.org から直接 DL |
| **curatedMetagenomicData** | ExperimentHub | 20,000+ サンプル | Bioconductor |

## HMP16SData の oral フィルタ例 (R)

```r
library(HMP16SData)
d <- V35()
# colData(d)$HMP_BODY_SUBSITE で oral 関連を抽出
oral_idx <- grep("oral|tongue|throat|saliva", 
                 colData(d)$HMP_BODY_SUBSITE, ignore.case=TRUE)
d_oral <- d[, oral_idx]
# 5 菌種への OTU マッピング: rowData(d)$CONSENSUS_LINEAGE で Genus/Species を照合
```

## 5 菌種マッピング

| 菌種 | Genus | Species | 備考 |
|------|-------|---------|------|
| S. oralis | Streptococcus | oralis |  |
| A. naeslundii | Actinomyces | naeslundii |  |
| V. dispar | Veillonella | dispar |  |
| F. nucleatum | Fusobacterium | nucleatum |  |
| P. gingivalis | Porphyromonas | gingivalis |  |

## Python からの利用

1. **rpy2**: R を Python から呼び出し、HMP16SData を実行
2. **R で CSV export**: R スクリプトで前処理 → CSV 出力 → Python で読む
3. **SRA/ENA**: NCBI から raw 16S を取得し、QIIME2 等で処理

## 実行スクリプト

```bash
cd gnn
Rscript scripts/extract_hmp_oral.R data/hmp_oral
# → data/hmp_oral/species_abundance.csv, sample_metadata.csv
```

## 次のアクション

- [ ] R 環境で `Rscript scripts/extract_hmp_oral.R` を実行
- [ ] oral サブセットのサンプル数確認
- [ ] OTU → 5 菌種の abundance 変換ロジック
- [ ] co-occurrence 行列の計算
- [ ] GNN 入力形式への変換
