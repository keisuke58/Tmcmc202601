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
| Joshi et al. 2025/2026 (JDR) | ペリインプラント炎（重症度差） | full-length 16S + RNAseq（サブセット） | BioProject: PRJNA1192962（SIIRI cohort） |

#### Joshi et al. (JDR) のデータでできそうなこと（時系列なし前提）

この論文は横断データなので「同一個体の時系列フィット（動的 ODE/状態方程式の校正）」は基本できない。一方で、臨床重症度（PD など）を目的変数にして、組成・機能特徴量を説明変数にしたフィットは十分できる。

- **データの中身（最低限）**
  - SRA（PRJNA1192962）に raw reads があり、サンプル属性に臨床指標が入っている（例: peri_implant_pocket_depth_(mm), bleeding_on_probing, plaque_index, gingival_index, suppuration, smoking, host_age/sex, host_subject_id）。
  - この BioProject はコホート全体なので、論文の解析対象（例: 49 インプラント、RNAseq 27 サンプル）に一致するサブセット抽出が必要。

- **1) DI を「臨床指標と相関するか」で検証**
  - 16S から 5 菌種へマッピング（または近縁分類群への集約）して DI を計算。
  - PD（連続）との相関、PD の閾値での群比較、BOP/排膿などとの関連を評価。
  - 患者内で複数インプラントがある可能性が高いので、患者（host_subject_id）で階層化した回帰（混合効果）を基本形にする。

- **2) 「重症度予測」モデルのベースライン作り**
  - 目的変数: PD（回帰）/ 重症度カテゴリ（分類）。
  - 説明変数: DI 単独 vs DI+主要臨床指標 vs 16S 由来の低次元特徴（上位属/種、PCoA 座標など）。
  - 何が効いているかを明確化して、後段のモデル（GNN/機構モデル）評価のベンチマークにする。

- **3) 時系列の代わりに「擬似進行軸」を作る**
  - PD を単純な代理時間として扱うのは粗いが、横断でも「PD が増える方向に DI が単調に変化するか」を検証できる。
  - 例: 単調回帰（isotonic）や潜在重症度スコア（1 次元の latent severity）を推定して、DI/機能と整合するかを見る。

- **4) 機能（PICRUSt2/RNAseq）を使った拡張**
  - 16S→機能予測（MetaCyc 等）で、DI と独立に PD を説明できるかを評価。
  - RNAseq サブセットでは、予測機能と実測機能の一致度を見て「予測機能を DI 推定に混ぜると得か」を判断する。

- **5) こちらの機構モデル側への落とし込み（できる範囲）**
  - 動的校正ではなく、各サンプルを「定常状態に近い断面」と見なして、DI↔重症度の静的マッピング（観測モデル）を作る。
  - 出力が DI のみのモデルでも、PD などの臨床指標へのリンク（回帰層）を追加して検証できる。

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
