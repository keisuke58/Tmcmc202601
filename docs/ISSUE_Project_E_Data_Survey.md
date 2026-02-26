# Issue: Project E (VAE × TMCMC) — Heine 2025 類似データの公開状況調査

**GitHub**: [Issue #71](https://github.com/keisuke58/Tmcmc202601/issues/71)

**関連**: Issue #39 (公開データ × ML プロジェクトアイデア), Project E

---

## GitHub Issue 用（コピーして新規 Issue 作成）

**Title**: `[Project E] Heine 2025 類似データの公開状況調査結果`

**Labels**: `enhancement`, `documentation`

**Body** (以下をそのまま貼り付け):

---

Project E (VAE で TMCMC posterior を近似) を検討するにあたり、Heine et al. (2025) の実験データに近い形式の公開データを調査した。

**結論**: 5菌種・qPCR 種組成・4条件・21日・チタン基質と完全一致する公開データは**見つかっていない**。部分的に類似するデータ（Dryad, HMP, curatedMetagenomicData 等）はあるが、前処理・マッピングが必要。

**推奨**: (1) 既存 TMCMC posterior で VAE 学習を開始 (2) 合成データ拡張で amortized inference を訓練 (3) Heine 著者へ生データ提供を問い合わせ

詳細は `docs/ISSUE_Project_E_Data_Survey.md` を参照。

---

## 本文

**目的**: Project E (VAE で TMCMC posterior を近似 → amortized inference) を検討するにあたり、Heine et al. (2025) の実験データに近い形式の公開データが存在するか調査した結果をまとめる。

---

## 1. Heine 2025 データの仕様

### 1.1 論文情報

| 項目 | 内容 |
|------|------|
| **著者** | Heine et al. |
| **タイトル** | Influence of species composition and cultivation condition on peri-implant biofilm dysbiosis in vitro |
| **雑誌** | Frontiers in Oral Health, Vol. 6, 2025 |
| **DOI** | [10.3389/froh.2025.1649419](https://doi.org/10.3389/froh.2025.1649419) |

### 1.2 データ形式（TMCMC 入力に必要な形式）

| 項目 | 内容 |
|------|------|
| **菌種** | 5種: *S. oralis*, *A. naeslundii*, *V. dispar/parvula*, *F. nucleatum*, *P. gingivalis* |
| **条件** | 4条件: Commensal/Dysbiotic × Static/HOBIC |
| **時系列** | Day 1, 3, 6, 10, 15, 21 |
| **測定** | qRT-PCR（種組成 %）, バイオフィルム体積（CLSM）, 生存率 |
| **出力形式** | 種組成の時系列（median, IQR 等の boxplot 統計） |
| **基質** | チタンインプラント（HOBIC 流路 / 6-well 静置） |

### 1.3 Heine 2025 のデータ公開状況

- **Supplementary**: 論文に Supplementary Tables S1–S9, Figures S1–S2 が記載
- **Data Sheet 1.pdf**: プロジェクト内に存在（再調査 2026-02-26 確認）
  - `Tmcmc202601/data_5species/docs/Data Sheet 1.pdf`
  - `Tmcmc202601/tmcmc/program2602/Data Sheet 1.pdf`
- **本プロジェクトでの利用**: 論文 Figure の画像に含まれるデータを抽出し、`experiment_data/` に CSV として保存済み（後述 §1.4）

---

### 1.4 Data Sheet 1 の内容と、論文 Figure からのデータ抽出

**Data Sheet 1.pdf** の内容（メソッド・統計表のみ。各 replicate の individual 生値は含まない）:

| 表・図 | 内容 |
|--------|------|
| **Table S1–S5** | qRT-PCR プライマー、反応条件、ゲノムサイズ、FISH プローブ |
| **Table S6–S9** | Biofilm volume / viability / species distribution の統計結果（adjusted p-values） |
| **Figure S1, S2** | 種組成 box plot の説明、pH・conditioned medium の影響 |

**Table S8** は Nishioka Algorithm の根拠として引用：「P. gingivalis と F. nucleatum が Commensal 条件で検出限界以下」。

---

**論文 Figure に含まれるデータと、本プロジェクトでの抽出結果**

生データは論文の **Figure 画像** に可視化されており、本プロジェクトではこれを抽出して CSV 化している。

| 元 Figure | 抽出データ | 出力 CSV |
|-----------|------------|----------|
| **Figure 2** | Biofilm volume（boxplot）, 膜分布（intact/damaged %） | `fig2_biofilm_volume_replicates.csv`, `fig2_membrane_distribution.csv`, `fig2_combined_data.csv` |
| **Figure 3** | 種組成 %（4条件 × 5菌種 × 6日、median, Q1, Q3, whisker） | `fig3_species_distribution_replicates.csv`, `fig3_species_distribution_summary.csv` |
| **Figure 4A–4C** | pH 時系列、Gingipain 濃度、代謝相互作用ネットワーク | `fig4A_pH_timeseries.csv`, `fig4B_gingipain_concentration.csv`, `fig4C_metabolic_interactions.csv` |
| **Figure 1B** | OD600 増殖曲線 | `fig1B_OD600_growth_curves.csv` |

抽出スクリプト: `extract_and_plot_fig2.py`, `extract_and_plot_fig3.py`, `extract_and_plot_fig1_fig4.py`

これらを集約した TMCMC 入力用データ: `species_distribution_data.csv`, `biofilm_boxplot_data.csv`, `expected_species_volumes.csv`（Fig2 × Fig3 の積）

---

## 2. 類似公開データの調査結果

### 2.1 完全一致するデータ: **見つかっていない**

Heine 2025 と同形式（5菌種・qPCR 種組成・4条件・21日・チタン基質）の公開データは**見つかっていない**。

### 2.2 部分的に類似するデータ（公開済み）

| データソース | 類似点 | 相違点 | Project E での利用可能性 |
|-------------|--------|--------|--------------------------|
| **Dryad: Zhou et al. (2022)**<br>doi:10.25349/D9P02H | 口腔 in vitro バイオフィルム、時系列、16S | 5菌種制御ではない（唾液 microcosm）、3 hosts、FASTQ 生データ | 16S 解析→種組成抽出が必要。5菌種へのマッピングは要検討 |
| **HMP (hmpdacc.org)** | 口腔 16S/metagenome、大規模 | 5菌種限定ではない、cross-sectional 中心、longitudinal は限定的 | 種組成の抽出・前処理は可能。5菌種・時系列・in vitro ではない |
| **curatedMetagenomicData**<br>(Bioconductor) | 22,588 サンプル、口腔含む、種組成（相対度） | 5菌種・時系列・in vitro ではない | 種組成の分布学習には使えるが、Hamilton ODE 形式とは乖離 |
| **MDPI IJERPH (2022)**<br>peri-implant 30日モデル | チタンインプラント、in vitro、30日、16S、156種 | 5菌種制御ではない、健康→mucositis→peri-implantitis の 3 段階 | 16S 解析→5菌種への集約が必要 |
| **Zenodo: Oral Microbiome Catalog (2025)** | 口腔菌叢の包括的カタログ、689 metagenomes | 5菌種・時系列・in vitro ではない | 参照用。直接の TMCMC 入力には不向き |

### 2.3 データ形式の違い

| 形式 | Heine 2025 | 公開データの多く |
|------|-------------|------------------|
| 測定法 | qRT-PCR（種特異的定量） | 16S amplicon / metagenome |
| 種数 | 5種に固定 | 数十〜数百種 |

16S データから 5 菌種へのマッピングは可能だが、qPCR の定量精度・特異性とは異なる。

---

## 3. Project E への示唆

### 3.1 Project E のデータ要件

| 用途 | 必要なデータ |
|------|-------------|
| **Phase 1: 既存 posterior の圧縮** | 既存 TMCMC の posterior samples（dh_baseline 等）→ **既に利用可能** |
| **Phase 2: Amortized inference** | 多様な (y_obs, θ_samples) ペア → 新データ y_obs に対して 1 回の forward pass で posterior を近似 |

### 3.2 結論

1. **Heine 2025 のデータ**: 生データは論文 Figure 画像に可視化されており、本プロジェクトで **Figure 2, 3, 4, 1B から抽出済み**（`experiment_data/` の CSV）。Data Sheet 1 はメソッド・統計表を補足。

2. **類似公開データ**: 5菌種・時系列・in vitro の完全一致はない。Dryad の Zhou et al. や HMP の口腔データは、前処理・マッピングを経れば「種組成時系列」として利用できるが、Hamilton ODE の 5 菌種への直接対応は要検討。

3. **Project E の現実的な進め方**:
   - **短期**: 既存 TMCMC posterior（4条件分）を VAE/Normalizing Flow で学習。データは十分。
   - **中期**: 合成データ拡張（θ を変えて ODE 解を生成し、ノイズ付き y_obs と θ のペアを大量に作成）で amortized inference を訓練。
   - **長期**: Heine 著者へのデータ提供依頼、または Dryad/HMP 等の前処理済み種組成を 5 菌種にマッピングしたデータセットを構築。

---

## 4. 推奨アクション

- [x] **Data Sheet 1 確認**: プロジェクト内に存在（`data_5species/docs/`, `tmcmc/program2602/`）
- [x] **Figure からのデータ抽出**: 論文 Figure 2, 3, 4, 1B から抽出したデータを CSV 化済み（`experiment_data/`）
- [ ] **Heine 2025 著者**（任意）: 各 replicate の individual 生値の提供を問い合わせると、解析の自由度がさらに向上
- [ ] **Dryad Zhou et al.**: 16S 解析パイプラインで 5 菌種相当の割合を抽出し、TMCMC 入力形式に変換できるか検証する
- [ ] **合成データ**: `generate_training_data.py`（GNN 用）と同様のフローで、Hamilton ODE から (θ, φ(t)) を大量生成し、Project E の訓練データとして拡張する
- [x] **Project E Phase 1 着手**: `project_e/` に VAE 学習パイプラインを実装（`load_posterior_data.py`, `vae_model.py`, `train.py`, `eval.py`）

---

## 5. 参考文献

- Heine et al. (2025) Frontiers in Oral Health. doi:10.3389/froh.2025.1649419
- Zhou et al. (2022) Dryad. doi:10.25349/D9P02H
- HMP: https://hmpdacc.org
- curatedMetagenomicData: https://bioconductor.org/packages/curatedMetagenomicData
- Zenodo Oral Microbiome Catalog: https://zenodo.org/records/16983006

---

*調査日: 2026-02-26*
*再調査（Data Sheet 1 精査）: 2026-02-26*
