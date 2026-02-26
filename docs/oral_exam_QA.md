# 口頭試問 Q&A メモ

> **参照:** [IKM 研究者・論文一覧 §8](https://github.com/keisuke58/Tmcmc202601/wiki/IKM-Researchers-and-Publications#8-教授が本研究を見たときの予想される指摘補強方向)
> **Issues:** [#77](https://github.com/keisuke58/Tmcmc202601/issues/77)–[#90](https://github.com/keisuke58/Tmcmc202601/issues/90)

---

## 応用 (Application)

### Q1: 臨床検証はどうするか？ [#77]

**想定質問者:** Stiesch

**回答の骨子:**
- 現状は Heine et al. (2025) の in vitro 4 条件のみ
- 次のステップ: 患者唾液中 16S から DI を推定し、BOP/PPD 等の臨床指標と相関解析
- データソース: 公開 DB (HMP, EBI ENA)、MHH 共同研究、自施設コホート
- 課題: 16S と in vitro の種組成の対応、サンプル数

---

### Q2: TSM と DeepONet の使い分けは？ [#78]

**想定質問者:** Geisler

**回答の骨子:**
- Fritsch et al. (2025) は TSM を ROM としてベイズ推定に使用
- 本研究は DeepONet で 3.8M× 高速化、勾配も取れる
- ハイブリッド案: 低次元・線形部分は TSM（解析的）、非線形・高次元は DeepONet
- 両者は相補的。TSM は UQ の理論的保証が強い、DeepONet は柔軟性が高い

---

## 理論 (Theory)

### Q3: E(DI) を Hamilton 原理から導出できるか？ [#79]

**想定質問者:** Junker

**回答の骨子:**
- 現状は経験的構成則として E(DI) を仮定
- 課題: DI を内部変数として Lagrangian に組み込む設計
- 展望: 自由エネルギー Ψ(DI) を導入し、E = ∂²Ψ/∂ε² のような関係を変分から導出
- Junker & Balzani (2021) の拡張ハミルトン原理が枠組み

---

### Q4: Hill ゲートは変分整合的か？ [#80]

**想定質問者:** Junker

**回答の骨子:**
- Hill 型飽和項は生物学的に標準（酵素反応、酸素結合）
- Hamilton 原理での正当化: 散逸汎関数に飽和型抵抗を入れる等
- Klempt et al. (2025) のモデルでは Monod 型が既に含まれる
- 変分的再定式化は今後の課題（HIWI_ATTENTION_POINTS に記載）

---

## 補強 (Reinforcement)

### Q5: E(DI) の文献的根拠は？ [#81]

**想定質問者:** Junker

**回答の骨子:**
- 3 段階因果チェーン: composition → EPS → mechanics
- Flemming, Billings, Peterson, Koo, Pattem の引用
- 直接測定はないが、チェーン全体で支持
- [paper_discussion_EDI_draft.md](paper_discussion_EDI_draft.md) に Discussion 案あり

---

### Q6: 0D ODE で 3D バイオフィルムを近似する根拠は？ [#82]

**想定質問者:** Soleimani

**回答の骨子:**
- [3D Reaction-Diffusion Evaluation](https://github.com/keisuke58/Tmcmc202601/wiki/3D-Reaction-Diffusion-Evaluation) で 0D/3D 比較実施
- 18× 差以内で実用上許容
- 条件: 栄養拡散が支配的、空間勾配が緩やか
- 限界: 強い勾配・異方性がある場合は 3D が必要

---

### Q7: Pg < 3% は臨床と矛盾しないか？ [#83]

**想定質問者:** Stiesch

**回答の骨子:**
- in vitro 5–7 種では 0.1–5% が標準（Bradshaw, Zijnge）
- 臨床歯周病 10–30% は months–years の経過
- Keystone pathogen: 低存在量で community-wide effect
- 培養期間・嫌気条件・宿主免疫の違いで説明

---

### Q8: 49/51 basin jump の生態学的意味は？ [#90]

**想定質問者:** 全員

**回答の骨子:**
- 「Parametric basin stability」— Menck et al. の basin stability をパラメータ空間に拡張
- Commensal Static が tipping point 直近にいる
- Marsh (2003): 歯科疾患は ecological catastrophe
- 予防的介入のタイミングに示唆

---

## 次点（余裕があれば）

| # | テーマ | 一言回答 |
|---|--------|----------|
| 84 | 電磁拡張 | Wolf & Junker 2025 の枠組みで DI→E を電磁-熱-力学に拡張可能 |
| 85 | 動脈硬化類推 | Haverich outside-in と DI 勾配での「外側崩壊」の類推 |
| 86 | Space-time 定式化 | Junker & Wick 2024 の時空間汎関数に栄養 PDE を統合 |
| 87 | 栄養 PDE | c*(x,t), α*(x,t) を陽にモデル化。Klempt 2024 に準拠 |
| 88 | Model evidence | 複数シード・複数 prior で Bayes factor のロバスト性検証 |
| 89 | Gradient enhancement | Bensel et al. の gradient-enhanced 成長でチェッカーボード対策 |
