# NIFE 面接対策 (2026-04-10)

## 基本情報

- **日時**: 2026年4月10日 10:00 AM (対面)
- **場所**: Stadtfelddamm 34, 30625 Hannover (NIFE)
  - 到着後: 正面玄関から内線 1415 (Dr. Szafranski) に電話
  - アクセス: 4番線 MHH駅 → バス「Neue-Land-Straße」or「Stadtfelddamm」停
- **面接官**:
  - Dr. Szymon Szafranski (B7: Biotechnology, Microbiology)
  - Dr. Rumjhum Mukherjee (Zentrumsmanagement)
  - Dr. Amruta Arun Joshi (B2: Clinic, NGS, Biomarker)
- **希望期間**: 2026年5月〜6月開始、12週以上、最長2027年3月まで可

---

## SIIRI プロジェクト

**Sicherheitsintegrierte und infektionsreaktive Implantate**
(Safety-Integrated and Infection-Reactive Implants)

- **資金**: DFG 1000万ユーロ以上 (3.5年)
- **機関**: MHH, LUH, Helmholtz, TU Braunschweig, HMTMH (150名以上)
- **目標**: 航空安全工学の手法を医療インプラントに適用し、感染の早期検知・反応型インプラントを開発
- **対象**: 歯科インプラント、整形外科インプラント、補聴器など
- **関連サブプロジェクト**:
  - **B7** (Szafranski): Biotechnology, Microbiology — バイオフィルム形成・dysbiosis
  - **B2** (Joshi): NGS, バイオマーカー — 感染の遺伝子レベル診断

---

## COMETS とは

**Computation Of Microbial Ecosystems in Time and Space**
— Daniel Segrè (Boston University) 開発

### コア技術
- **dFBA (dynamic Flux Balance Analysis)**: ゲノムスケール代謝モデル (GEM) を時間発展させる制約ベース手法
- **空間構造**: グリッド上でバイオマス・代謝物の拡散・消費を計算
- **cometspy**: Python ラッパー (COBRApy と統合)

```
GEM (genome-scale metabolic model)
    ↓ FBA (線形計画法) → 増殖速度, 代謝フラックス
    ↓ × 空間グリッド (拡散方程式)
空間的バイオマス分布 + 代謝物濃度場 (時系列)
```

### Daniel Segrè 研究のキーワード
- Quantitative principles of microbial metabolism across scales (Nature Microbiology 2024)
- Metabolic complexity driving community divergence (Nature Ecology & Evolution 2024)
- 生物ネットワークの力学・進化、微生物生態系の理論モデル

---

## Szafranski 研究の要点

- インプラント周囲バイオフィルム (transmucosal abutment) の 12患者コホート研究
- **371 菌種**同定、対象菌種に So, An, Aa, Fn, Pg 含む
- 口腔粘膜 + インプラント材料 + バイオフィルムの in vitro organotypic モデル構築
- commensal (S. oralis) vs. pathogenic (A. actinomycetemcomitans) の組織応答比較
- salivary flow 条件下での dysbiosis 組成変化モデリング

---

## 自分の研究との接続点

| 自分の手法 | COMETS / SIIRI | 補完性 |
|---|---|---|
| Hamilton ODE (phenomenological) | FBA (mechanistic stoichiometry) | 機構的根拠を補強 |
| TMCMC (Bayesian UQ) | deterministic FBA (不確実性なし) | UQ を COMETS に付加可能 |
| DI → E(DI) → FEM (力学) | 代謝・増殖のみ | 力学連成は COMETS に未実装 |
| 5菌種 oral biofilm | 371菌種 口腔バイオフィルム | 直接応用可能 |
| GPU TMCMC (200× 高速化) | 計算コスト高 | 大規模 GEM への適用可 |
| DI (Dysbiosis Index) | SIIRI 感染モニタリング | リアルタイム指標として提案可能 |

---

## 面接で使えるフレーズ (英語)

1. "My TMCMC framework can quantify parameter uncertainty in COMETS-style dFBA models — something that is currently missing from purely deterministic FBA approaches."

2. "The Dysbiosis Index I defined aligns directly with your SIIRI goal of early infection detection. It could serve as a real-time, model-based indicator for implant monitoring."

3. "I can bridge your NGS biomarker data (B2) with mechanistic ODE and FEM models — from genome-scale metabolism to tissue-level mechanical response."

4. "My GPU-accelerated pipeline achieves a 200× speedup, making Bayesian calibration of large microbial interaction networks computationally feasible."

5. "I am very open to extending the internship beyond 12 weeks, as I will remain in Germany until March 2027."

---

## 取得済みデータ・論文

### 論文 PDF (nife/data/ に保存済み)

| ファイル | 論文 | データ |
|---|---|---|
| `dieckow2024_npj_biofilms.pdf` | Dieckow, Szafrański et al. (2024) *npj Biofilms Microbiomes* 10, 155 | **ENA: PRJEB71108** (30サンプル, metagenomics) |
| `dieckow2024_supplementary.pdf` | 上記 Supplementary | Table S4: 菌種間代謝相互作用 DB (351 taxa) |
| `joshi2025_npj_peri-implantitis.pdf` | Joshi, Szafrański et al. (2025) *npj Biofilms Microbiomes* 11, 175 | **NCBI SRA: PRJNA1192962** (48サンプル, 16S + metatranscriptomics) |

### Joshi 2025 の詳細
- **対象**: 32患者 48バイオフィルム (健常 vs peri-implantitis)
- **方法**: 全長 16S rRNA + メタトランスクリプトーム (RNAseq)
- **結果**: AUC = 0.85 / 健常: *Streptococcus*, *Rothia* ↑ / 病態: 嫌気性グラム陰性菌 ↑
- **データ**: NCBI SRA PRJNA1192962 (公開済み)

### Joshi 2026 の詳細 (Journal of Dental Research)
- **論文**: "The Submucosal Microbiome Correlates with Peri-implantitis Severity"
  doi:10.1177/00220345251352809 / PMID: 40719760
- **対象**: 34患者 49インプラント
- **結果**: Pseudoramibacter ↑ → 重症化 / 中心炭素代謝経路が重症度と相関
- **データ公開**: 未確認 (JDR は有料誌、PMC12861548 で確認要)

---

## 事前に読んでおく論文

- [ ] Segrè lab: COMETS 原著論文 (eLife 2021, doi:10.7554/eLife.63372)
- [x] Dieckow, Szafrański et al. (2024) doi:10.1038/s41522-024-00624-3 **→ PDF取得済み**
- [x] Joshi, Szafrański et al. (2025) doi:10.1038/s41522-025-00807-6 **→ PDF取得済み**
- [ ] Joshi, Szafrański et al. (2026) doi:10.1177/00220345251352809 (JDR, 有料)
- [ ] Szafranski: mucosa-biofilm interaction モデル (Cell Microbiol 2020)

---

## 質問候補 (自分から聞く)

1. SIIRI の B7 サブプロジェクトで COMETS を具体的にどのように使っているか？
2. NGS データ (B2) と力学モデルを統合する計画はあるか？
3. Fachpraktikum 期間中に論文共著の可能性はあるか？
4. どのプログラミング環境を主に使っているか (Python, MATLAB, R)?
