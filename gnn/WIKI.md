# GNN × 口腔菌叢相互作用ネットワーク — WIKI

> **Project B, Issue #39**  
> **上司:** 村松眞由（慶應義塾大学 理工学部 機械工学科 准教授）  
> **参照:** [README.md](README.md)

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [口頭試問対策 — 査読・質問想定](#2-口頭試問対策--査読質問想定)
3. [村松先生（上司）視点からの想定質問](#3-村松先生上司視点からの想定質問)
4. [技術的・方法論的質問](#4-技術的方法論的質問)
5. [生物学的・臨床的質問](#5-生物学的臨床的質問)
6. [回答の骨子・準備メモ](#6-回答の骨子準備メモ)

---

## 1. プロジェクト概要

HMP 16S データから菌種間相互作用 \(a_{ij}\) を GNN で予測し、TMCMC の informed prior として活用する。

```
HMP 16S (菌叢組成) → co-occurrence network → GNN → a_ij 予測
                                                      ↓
                                              Hamilton ODE の prior として使う
                                              → informed TMCMC (faster convergence)
```

**5 菌種:** So (S. oralis), An (A. naeslundii), Vd (V. dispar), Fn (F. nucleatum), Pg (P. gingivalis)  
**Active edges (5本):** a01(So→An), a02(So→Vd), a03(So→Fn), a24(Vd→Pg), a34(Fn→Pg)

---

## 2. 口頭試問対策 — 査読・質問想定

以下、査読者・口頭試問で想定される質問をカテゴリ別に整理。**村松先生（材料力学・破壊力学・検証手法の専門家）の視点**を特に重視。

---

## 3. 村松先生（上司）視点からの想定質問

> 村松先生の専門: 材料力学、機械材料、強弾性材料、SOFC、形状記憶合金、破壊力学、結晶塑性シミュレーション、位相場モデル  
> → **検証手法・統計的 rigor・不確実性の定量化・モデル仮定・数値安定性**に強い関心を持つと想定

### 検証・妥当性

| # | 質問 | 回答の骨子 |
|---|------|------------|
| M1 | GNN の \(a_{ij}\) 予測の ground truth は何か？検証方法は？ | 合成データでは θ が真値。HMP では co-occurrence や文献値との整合性で間接検証。Phase 1 で RMSE/MAE を報告。 |
| M2 | 合成データで学習した GNN を実データ（HMP）に転移する根拠は？ | 同一 Hamilton ODE に基づくため、組成統計 (φ_mean, φ_std, φ_final) と a_ij の関係は domain-invariant と仮定。Phase 2 で転移学習 or 再学習で検証。 |
| M3 | 過学習をどう防いだか？ | N=10k サンプル、dropout=0.2、weight_decay=1e-2、early stopping (patience=100)、train/val 分離。 |
| M4 | GNN の予測誤差が TMCMC の事後分布にどう伝播するか定量化しているか？ | 現状は σ (prior の標準偏差) で感度を制御。系統的誤差伝播の解析は今後の課題。 |
| M5 | 材料力学でいう「構成則の検証」に相当する検証をしているか？ | 合成データでは解析解との比較。実データでは TMCMC 収束速度・事後幅の改善で間接評価。直接測定との比較は今後の課題。 |

### 不確実性・感度

| # | 質問 | 回答の骨子 |
|---|------|------------|
| M6 | GNN の予測に信頼区間や不確実性は付与しているか？ | 現状は点推定。MC dropout やアンサンブルで epistemic uncertainty を推定可能。実装は今後の拡張。 |
| M7 | σ (prior の幅) を変えたとき TMCMC の収束・事後にどう影響するか？ | σ 小 → 収束早いが prior に強く縛られる。σ 大 → 一様 prior に近づく。sensitivity analysis で報告予定。 |
| M8 | ハイパーパラメータ（hidden, layers, dropout）の感度は？ | 過学習対策で dropout, weight_decay を調整。アーキテクチャ感度は ablation で検証可能。 |

### モデル仮定・数値

| # | 質問 | 回答の骨子 |
|---|------|------------|
| M9 | ノード特徴が (φ_mean, φ_std, φ_final) の 3 次元のみで十分な根拠は？ | Hamilton ODE の解の統計量として情報量が高い。時間系列全体を使う拡張（LSTM, attention）は検討可能。 |
| M10 | complete graph を仮定しているが、生物学的に非対称・疎な相互作用は？ | interaction_graph.json で 5 本の active edge のみ有効。locked は 0。生物学的構造は反映済み。 |
| M11 | 数値安定性（オーバーフロー、勾配消失）は考慮しているか？ | LayerNorm、residual、grad_clip=1.0。ODE ソルバーは既存 TMCMC パイプラインと同一。 |
| M12 | 位相場モデルや結晶塑性と同様の「物理的制約」を GNN に組み込めるか？ | 現状はデータ駆動。物理情報ニューラルネット（PINN）や Hamiltonian 構造の組み込みは今後の検討。 |

### 全体設計・位置づけ

| # | 質問 | 回答の骨子 |
|---|------|------------|
| M13 | この研究は材料力学のどの文脈と対応するか？ | パラメータ同定における「informed prior」は、材料定数の事前知識に相当。破壊力学の事前分布設計と類比可能。 |
| M14 | TMCMC の収束改善は定量的にどの程度か？ | generate_fig23_gnn_prior_effect.py で stage 数・ESS を比較。数値は要報告。 |
| M15 | 本研究の限界と今後の拡張は？ | 5 菌種・4 条件に限定。HMP 転移、不確実性定量化、臨床データ統合が次のステップ。 |

---

## 4. 技術的・方法論的質問

### 手法選択

| # | 質問 | 回答の骨子 |
|---|------|------------|
| T1 | なぜ GNN か？MLP や RNN では駄目か？ | 菌種間の相互作用がグラフ構造で表現される。GCN の message passing が a_ij の予測に適している。 |
| T2 | なぜ co-occurrence network か？ | 16S から直接 a_ij は測定不可。co-occurrence が相互作用の proxy。文献 (Kolenbrander 等) と整合。 |
| T3 | DeepONet との役割分担は？ | DeepONet: θ→φ の ODE サロゲート。GNN: φ→a_ij の逆問題。TMCMC では GNN が prior を、DeepONet が likelihood を提供。 |
| T4 | なぜ edge regression で node ではなく edge を予測するのか？ | a_ij は相互作用強度であり、エッジ属性。5 本の active edge に対応する 5 スカラーを出力。 |

### データ・学習

| # | 質問 | 回答の骨子 |
|---|------|------------|
| T5 | 合成データの θ サンプリングは一様 prior か？ | condition (Dysbiotic_HOBIC 等) に応じた prior_bounds.json を使用。MAP 周辺の正規分布サンプリングも可能。 |
| T6 | 5 菌種への HMP マッピングはどう行うか？ | scripts/extract_hmp_oral.R で HMP oral 16S を抽出。属レベルで So, An, Vd, Fn, Pg にマッピング。マッピング表は要整備。 |
| T7 | train/val の分離方法は？ | 85/15 の random split。seed 固定で再現性確保。 |
| T8 | 損失関数は MSE のみか？ | 現状 MSE。スケール差がある場合は MAE や重み付き損失を検討可能。 |

### アーキテクチャ

| # | 質問 | 回答の骨子 |
|---|------|------------|
| T9 | GCN の層数・hidden 次元の選定根拠は？ | 5 ノードの小グラフ。hidden=64–128, layers=3–4 で十分。過学習を避けるため控えめに設定。 |
| T10 | residual connection の役割は？ | 勾配の流れを改善。深い GCN での performance 向上。 |
| T11 | なぜ global_mean_pool ではなく flatten か？ | 各グラフは固定 5 ノード。batch_size × (5*hidden) として MLP に入力。 |

### TMCMC 統合

| # | 質問 | 回答の骨子 |
|---|------|------------|
| T12 | GNN prior の数式は？ | p(θ) ∝ U(bounds) × N(θ_active \| μ_gnn, σ²)。μ_gnn = GNN(φ_features)。 |
| T13 | 粒子初期化は GNN から行うか？ | はい。gnn_prior.sample(rng, bounds) で N(μ_gnn, σ²) からサンプリング。 |
| T14 | JAX 環境と PyTorch の分離はなぜ？ | gradient_tmcmc_nuts.py は JAX。GNN は PyTorch。predict_for_tmcmc.py で JSON に事前計算し、JAX 側で読み込み。 |

---

## 5. 生物学的・臨床的質問

| # | 質問 | 回答の骨子 |
|---|------|------------|
| B1 | a_ij の生物学的解釈は？ | a01: co-aggregation, a02: lactate handover, a03: formate symbiosis, a24: pH trigger, a34: co-aggregation/peptides。Klempt et al. に準拠。 |
| B2 | HMP データと Heine et al. in vitro の対応は？ | HMP はヒト口腔 16S。Heine は HOBIC 培養。種組成の対応は 5 菌種マッピングで近似。完全一致は課題。 |
| B3 | 臨床応用の想定は？ | 患者 16S → GNN → a_ij prior → TMCMC → DI → 弾性率 → インプラント周囲炎リスク。Phase 2–3 で検証。 |
| B4 | 5 菌種に限定する理由は？ | Heine et al. の実験系と Klempt モデルに合わせた。拡張は 7 菌種等で検討可能。 |

---

## 6. 回答の骨子・準備メモ

### 必須で答えられるようにする項目

1. **GNN の入力・出力**
   - 入力: (φ_mean, φ_std, φ_final) × 5 菌種 = 15 次元
   - 出力: 5 本の active edge の a_ij

2. **検証の 3 段階**
   - Phase 1: 合成データで RMSE/MAE、過学習チェック
   - Phase 2: HMP 転移、co-occurrence との整合性
   - Phase 3: TMCMC 収束速度・事後幅の改善

3. **限界の明示**
   - Ground truth の欠如（実データでは a_ij 直接測定不可）
   - 合成→実データの domain gap
   - 不確実性の定量化は未実装

4. **村松先生への説明のポイント**
   - 材料力学における「構成則のパラメータ同定」との類比
   - 検証の階層（合成→転移→統合）
   - 過学習対策・数値安定性の具体的措置
   - 今後の拡張（不確実性定量化、物理制約の組み込み）

### 参考文献

- Klempt et al. (2024) — Hamilton ODE, 5 菌種モデル
- HMP16SData — https://github.com/waldronlab/HMP16SData
- Heine et al. (2025) — HOBIC, 5 菌種実験
- Issue #39 — 公開データ × ML プロジェクト

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-02-26 | 初版作成。村松先生視点・査読想定・口頭試問対策を追加 |
