# 口頭試問対策・補強方向 — 統合ドキュメント

> **最終更新:** 2026-02-26  
> **Wiki 版（推奨）:** [口頭試問対策](https://github.com/keisuke58/Tmcmc202601/wiki/Oral-Exam-Preparation) — 本内容は Wiki にマージ済み  
> **参照:** [IKM 研究者・論文一覧 §8](https://github.com/keisuke58/Tmcmc202601/wiki/IKM-Researchers-and-Publications#8-教授が本研究を見たときの予想される指摘補強方向)  
> **Issues:** [#77](https://github.com/keisuke58/Tmcmc202601/issues/77)–[#90](https://github.com/keisuke58/Tmcmc202601/issues/90)

---

## 目次

1. [口頭試問 Q&A メモ](#1-口頭試問-qa-メモ)
2. [村松先生視点](#2-村松先生視点--研究全体)
3. [機械工学科教授の一般的な質問](#3-機械工学科教授の一般的な質問)
4. [査読で聞かれそうな質問](#4-査読で聞かれそうな質問)
5. [学会で聞かれそうな質問](#5-学会で聞かれそうな質問)
6. [E(DI) 文献裏付け](#6-edi-文献裏付け--discussion-案)
7. [TSM サロゲートとのハイブリッド](#7-tsm-サロゲートとのハイブリッド)
8. [E(DI) の変分導出](#8-edi-の変分導出)
9. [臨床検証](#9-臨床検証--データソース)
10. [GNN × 口腔菌叢](#10-gnn--口腔菌叢)

※ 詳細は [Wiki: 口頭試問対策](https://github.com/keisuke58/Tmcmc202601/wiki/Oral-Exam-Preparation) を参照

---

## 1. 口頭試問 Q&A メモ

### 応用 (Application)

| # | 質問 | 想定質問者 | 回答の骨子 |
|---|------|------------|------------|
| Q1 | 臨床検証はどうするか？ | Stiesch | 患者 16S → DI → BOP/PPD 相関。HMP, ENA, MHH 共同研究。§5 参照 |
| Q2 | TSM と DeepONet の使い分けは？ | Geisler | 低次元は TSM、非線形は DeepONet。相補的。§3 参照 |

### 理論 (Theory)

| # | 質問 | 想定質問者 | 回答の骨子 |
|---|------|------------|------------|
| Q3 | E(DI) を Hamilton 原理から導出できるか？ | Junker | 現状は経験的。Ψ(DI) 導入で変分導出を検討。§4 参照 |
| Q4 | Hill ゲートは変分整合的か？ | Junker | 散逸汎関数に飽和型抵抗。変分再定式化は今後の課題 |

### 補強 (Reinforcement)

| # | 質問 | 想定質問者 | 回答の骨子 |
|---|------|------------|------------|
| Q5 | E(DI) の文献的根拠は？ | Junker | 3 段階因果チェーン。§2 参照 |
| Q6 | 0D ODE で 3D を近似する根拠は？ | Soleimani | 3D Reaction-Diffusion Evaluation で 18× 差以内 |
| Q7 | Pg < 3% は臨床と矛盾しないか？ | Stiesch | in vitro 0.1–5% 標準。Keystone pathogen で説明 |
| Q8 | 49/51 basin jump の意味は？ | 全員 | Parametric basin stability。Marsh 2003 の ecological catastrophe |

### 次点（余裕があれば）

| # | テーマ | 一言回答 |
|---|--------|----------|
| 84 | 電磁拡張 | Wolf & Junker 2025 の枠組みで拡張可能 |
| 85 | 動脈硬化類推 | Haverich outside-in と DI 勾配の類推 |
| 86 | Space-time 定式化 | Junker & Wick 2024 に栄養 PDE を統合 |
| 87 | 栄養 PDE | c*(x,t), α*(x,t) を陽にモデル化 |
| 88 | Model evidence | 複数シード・prior で Bayes factor ロバスト性検証 |
| 89 | Gradient enhancement | Bensel et al. でチェッカーボード対策 |

---

## 2. 村松先生視点 — 研究全体

> **上司:** 村松眞由（慶應義塾大学 理工学部 機械工学科 准教授）

| 領域 | 想定質問例 | 回答の骨子 |
|------|------------|------------|
| **TMCMC** | 収束判定・不確実性伝播 | R-hat, ESS。事後 1000 サンプルで φ, DI, E の CI 算出。 |
| **DeepONet** | サロゲートの検証・外挿リスク | 条件別 RMSE。posterior が training 域内に収まるよう設計。 |
| **E(DI)** | 28 倍差の根拠・構成則の検証 | Literature Reinforcement §1。3 段階因果チェーン。 |
| **3D FEM** | 境界条件・メッシュ感度・応力 CI | 事後サンプルから DI→E→FEM で 90% CI。 |
| **GNN** | 検証・不確実性・転移の根拠 | §7 参照。 |

**詳細:** [Wiki: 口頭試問対策 §2](https://github.com/keisuke58/Tmcmc202601/wiki/Oral-Exam-Preparation#2-村松先生視点--研究全体)

---

## 3. 機械工学科教授の一般的な質問

動機・手法・数値・学びに関する 16 問。**詳細:** [Wiki §3](https://github.com/keisuke58/Tmcmc202601/wiki/Oral-Exam-Preparation#3-機械工学科教授の一般的な質問)

| カテゴリ | 例 |
|----------|-----|
| 動機・背景 | なぜこの研究？機械工学との接点？社会貢献？ |
| 手法・新規性 | 既存との違い？なぜ TMCMC？20 パラメータの意味？ |
| 数値・検証 | 単位の整合性？収束？実験との比較？計算コスト？ |
| 学び・発展 | 何を学んだ？苦労した点？今後の発展？ |

---

## 4. 査読で聞かれそうな質問

新規性・妥当性・統計・解釈・形式に関する 20 問。**詳細:** [Wiki §4](https://github.com/keisuke58/Tmcmc202601/wiki/Oral-Exam-Preparation#4-査読で聞かれそうな質問)

| カテゴリ | 例 |
|----------|-----|
| 新規性 | 1 文で述べよ。先行研究との違い？28 倍差は既知？ |
| 妥当性 | 0D 近似の根拠？E(DI) の正当化？サンプルサイズ？ |
| 統計 | 多重比較補正？CI の解釈？感度解析？再現性？ |
| 解釈 | Pg < 3% の矛盾？限界の明確化？因果の過大解釈？ |
| 形式 | 図表キャプション？データ公開？倫理？ |

---

## 5. E(DI) 文献裏付け — Discussion 案

**Issue:** [#81](https://github.com/keisuke58/Tmcmc202601/issues/81)

### パラグラフ 1: 因果チェーンの明示

> Our constitutive law E(DI) maps Shannon diversity to elastic modulus. While no study has directly measured Shannon diversity and elastic modulus simultaneously on the same biofilm sample, the literature supports a three-step causal chain: (i) community composition determines EPS composition (Flemming & Wingender 2010), (ii) EPS composition determines biofilm mechanics (Billings et al. 2015; Peterson et al. 2015), and (iii) sucrose-driven ecological shifts (diversity loss) correlate with 10–80× stiffness reduction (Pattem et al. 2018). Our E(DI) mapping is thus a well-motivated constitutive hypothesis that formalizes this chain.

### パラグラフ 2: E のレンジの文献的根拠

> The elastic modulus range E ∈ [E_min, E_max] = [10, 1000] Pa is consistent with reported biofilm stiffness: Billings et al. (2015) cite 0.1–100,000 Pa across biofilm types; Pattem et al. (2018) report 14 kPa vs 0.55 kPa (≈25×) for high- vs low-diversity oral biofilms; our fitted range 30–900 Pa falls within these bounds. The variational structure of the underlying Hamilton-ODE model ensures thermodynamic consistency automatically (Junker & Balzani 2021), eliminating the need for separate entropy inequality verification.

### パラグラフ 3: Low diversity = soft の直接証拠

> Koo et al. (2013) showed that S. mutans-dominated biofilms exhibit porous, structurally weaker EPS compared to diverse communities. Houry et al. (2012) demonstrated that second-species infiltration dramatically alters matrix mechanics. These findings support our hypothesis that diversity loss (dysbiosis) reduces effective stiffness, as captured by E(DI).

### BibTeX 追加推奨

```bibtex
@article{Flemming2010EPS,
  author = {Flemming, Hans-Curt and Wingender, Jost},
  title = {The biofilm matrix},
  journal = {Nature Reviews Microbiology},
  volume = {8}, number = {9}, pages = {623--633}, year = {2010},
  doi = {10.1038/nrmicro2415}
}
@article{Billings2015BiofilmMechanics,
  author = {Billings, Nicole and others},
  title = {Material properties of biofilms — a review},
  journal = {Reports on Progress in Physics},
  volume = {78}, number = {3}, pages = {036601}, year = {2015},
  doi = {10.1088/0034-4885/78/3/036601}
}
@article{Koo2013SucroseBiofilm,
  author = {Koo, Hyun and Falsetta, Megan L. and Klein, Marlise I.},
  title = {The exopolysaccharide matrix: a virulence determinant of cariogenic biofilm},
  journal = {Journal of Dental Research},
  volume = {92}, number = {12}, pages = {1065--1073}, year = {2013},
  doi = {10.1177/0022034513504218}
}
@article{Houry2012SecondSpecies,
  author = {Houry, A. and others},
  title = {Bacterial swimmers that infiltrate and take over the biofilm matrix},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {109}, number = {32}, pages = {13088--13093}, year = {2012},
  doi = {10.1073/pnas.1200791109}
}
```

### チェックリスト

- [ ] Pattem 2018 の正確な出典を確認
- [ ] 3 パラグラフを paper LaTeX Discussion に挿入
- [ ] BibTeX を .bib に追加

---

## 7. TSM サロゲートとのハイブリッド

**Issue:** [#78](https://github.com/keisuke58/Tmcmc202601/issues/78)

### 現状の比較

| 項目 | Fritsch et al. 2025 | 本研究 |
|------|---------------------|--------|
| サロゲート | TSM | DeepONet |
| 種数 | 2種・4種 | 5種 |
| 利点 | 理論的保証、追加コストほぼゼロ | 3.8M× 高速化、勾配取得可能 |
| 課題 | 高次元・強非線形で精度低下の可能性 | ブラックボックス、外挿リスク |

### ハイブリッド設計案

**案 A: 階層的 ROM**
```
θ → [低次元・線形] → TSM → φ_lin
  → [残差・非線形] → DeepONet → φ_res
  → φ = φ_lin + φ_res
```

**案 B: 条件分岐** — prior 中心付近は TSM、裾は DeepONet

**案 C: アンサンブル** — log L = α·log L_TSM + (1-α)·log L_DeepONet

### 実装タスク（案 A）

1. TSM 部分の切り出し（`BiofilmTSM5S`, `BiofilmTSM_Analytical`）
2. DeepONet の残差学習（ターゲット: φ_full - φ_TSM_lin）
3. TMCMC 統合

### 参照コード

- TSM: `data_5species/core/` の `BiofilmTSM5S`, `BiofilmTSM_Analytical`
- DeepONet: `deeponet/surrogate_tmcmc.py`

---

## 7. E(DI) の変分導出

**Issue:** [#79](https://github.com/keisuke58/Tmcmc202601/issues/79)

### 現状

- E(DI) = E_max(1-r)² + E_min·r, r = DI/DI_scale
- 経験的構成則として仮定。変分原理からの導出は未実施。

### 候補 1: DI を内部変数とする自由エネルギー

$$\Psi(\varepsilon, \mathrm{DI}) = \frac{1}{2} E(\mathrm{DI}) \varepsilon^2 + \Psi_{\mathrm{DI}}(\mathrm{DI})$$

**課題:** DI は生態学 ODE の結果であり、独立した発展則が不明。

### 候補 2: 有効媒質としての解釈

$$E_{\mathrm{eff}} = \sum_i f(\varphi_i) E_i, \quad E(\varphi) = E(\mathrm{DI}(\varphi))$$

Hamilton 原理では φ が状態変数。E を弾性項の係数として Lagrangian に組み込む。

### 候補 3: 散逸ポテンシャル

$$D = D(\dot{\varphi}, \dot{\psi}, \mathrm{DI})$$

**要検討:** 低 DI = soft と散逸の関係は未整理。

### 次のステップ

1. Junker & Balzani 2021 Section 2–3 を精読
2. Klempt et al. 2025 で E(DI) の挿入点を検討
3. 1D 例で変分から E(DI) 型の関係を試算

---

## 9. 臨床検証 — データソース

**Issue:** [#77](https://github.com/keisuke58/Tmcmc202601/issues/77)

### 目的

患者 16S → DI 推定 → インプラント周囲炎リスク（BOP, PPD）と相関

### データソース候補

| ソース | URL | 内容 |
|--------|-----|------|
| HMP | hmpdacc.org | 口腔 16S。GNN Phase 2 で使用予定 |
| HOMD | homd.org | 16S RefSeq、abundance、系統樹 |
| ENA/SRA | ebi.ac.uk/ena | periodontitis, peri-implantitis 16S |
| MHH 共同研究 | Stiesch グループ | 5 種組成・HOBIC と直接対応 |

### 公開データセット（論文付随）

| 論文 | 疾患 | 備考 |
|------|------|------|
| Abusleme et al. 2013 | 歯周病 vs 健康 | 歯肉縁下プラーク |
| Koyanagi et al. 2013 | インプラント周囲炎 | SRA 登録の可能性 |
| Kim et al. 2023 | ペリインプラント炎 vs 歯周病 | JPIS |

### パイプライン案

```
患者唾液/歯肉縁下プラーク → 16S → 5 菌種マッピング → DI 計算 → BOP/PPD 相関
```

### タスクチェックリスト

- [ ] SRA で "peri-implantitis 16S" を検索
- [ ] Abusleme 2013, Koyanagi 2013 の Data Availability を確認
- [ ] HMP oral の臨床メタデータの有無を確認
- [ ] MHH 共同研究の可能性を IKM 経由で打診
- [ ] 5 菌種マッピング表を `gnn/` に追加

---

## 10. GNN × 口腔菌叢 — 口頭試問対策

**Project B, Issue #39**  
**上司:** 村松眞由（慶應義塾大学 理工学部 機械工学科 准教授）

### 詳細は WIKI を参照

- **[gnn/WIKI.md](../gnn/WIKI.md)** — 査読・口頭試問対策の完全版

### 村松先生視点からの想定質問（抜粋）

| # | カテゴリ | 質問 | 回答の骨子 |
|---|----------|------|------------|
| M1 | 検証 | GNN の a_ij 予測の ground truth は？ | 合成データでは θ が真値。HMP では co-occurrence で間接検証。 |
| M2 | 検証 | 合成→実データ転移の根拠は？ | 同一 Hamilton ODE に基づく domain-invariant 仮定。Phase 2 で検証。 |
| M4 | 不確実性 | 予測誤差の TMCMC への伝播は？ | σ で感度制御。系統的誤差伝播の解析は今後の課題。 |
| M5 | 検証 | 構成則の検証に相当する検証は？ | 合成データでは解析解比較。実データでは TMCMC 収束速度で間接評価。 |
| M6 | 不確実性 | 予測に信頼区間は付与しているか？ | 現状は点推定。MC dropout やアンサンブルで拡張可能。 |
| M13 | 位置づけ | 材料力学のどの文脈と対応するか？ | パラメータ同定の informed prior。破壊力学の事前分布設計と類比。 |

### 技術的・方法論的質問（抜粋）

| # | 質問 | 回答の骨子 |
|---|------|------------|
| T1 | なぜ GNN か？ | 菌種間相互作用がグラフ構造。GCN の message passing が a_ij 予測に適する。 |
| T12 | GNN prior の数式は？ | p(θ) ∝ U(bounds) × N(θ_active \| μ_gnn, σ²)。μ_gnn = GNN(φ_features)。 |
| T14 | JAX と PyTorch の分離はなぜ？ | gradient_tmcmc_nuts は JAX。GNN は PyTorch。JSON で事前計算済みを渡す。 |

### 必須で答えられる項目

1. GNN の入力 (φ_mean, φ_std, φ_final) × 5 菌種 → 出力 5 本の a_ij
2. 検証の 3 段階: Phase 1 合成→Phase 2 HMP 転移→Phase 3 TMCMC 統合
3. 限界: ground truth 欠如、domain gap、不確実性定量化は未実装
4. 村松先生への説明: 構成則のパラメータ同定との類比、検証の階層、過学習対策
