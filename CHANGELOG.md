# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (nife — Hamilton LOO-CV / AGORA sign prior)
- MICOM community FBA sign prior (`guild_agora_signs.compute_micom_signals`):
  100% sign agreement with fitted A matrix (36/36 pairs), up from 88% (v1 pFBA)
- "Perfect" MICOM version: fraction=0.5 (Diener 2020 default), flux-magnitude
  weighting, toxin signals applied to all co-occurring guilds
- `--micom-fraction` CLI argument in `run_hamilton_expanded_loo.py`
- `compare_agora_v1_v2.py`: 3-column figure (v1 / v2 / MICOM-perfect)
- `collect_loo_results.py`: aggregation script for all LOO-CV JSON outputs
- LOO-CV jobs: MICOM v0 (40153–40162), MICOM-perfect (40164–40173)
- `docs/why_micom_worked.md`: analysis of why community FBA outperforms pFBA

## [0.3.0] - 2026-02-24

### Added
- Multiscale coupling pipeline (0D ODE → 1D/2D PDE → 3D FEM)
- Hybrid DI × spatial α approach for condition-specific eigenstrain
- 3 E-model comparison (DI, φ_Pg, Virulence) with quantitative analysis
- Abaqus INP generator with thermal eigenstrain analogy (12 files: 3 models × 4 conditions)
- Species competition 6-panel analysis figure
- Pipeline summary 9-panel figure
- JAX adjoint inverse problem PoC (Lotka-Volterra + Hill gate)
- Klempt 2024 quantitative benchmark
- Posterior uncertainty propagation (TMCMC → DI → σ 90% CI)
- Paper figures generator (Fig. 8–15)
- a₃₅ sensitivity sweep with 51-point evaluation
- θ variant → FEM stress comparison (mild-weight vs dh-old)
- 2D reaction-diffusion extension (`multiscale_coupling_2d.py`)

### Changed
- README: complete rewrite with academic title, Japanese summary, methodology, fixed images
- README: added Limitations, Future Work, Data Preprocessing sections

### Fixed
- Mermaid rendering on GitHub (replaced `$...$` LaTeX with Unicode in node labels)
- 4 broken image paths in README

## [0.2.0] - 2026-02-18

### Added
- Mild-weight prior bounds: a₃₅ [0, 5], a₃₅ [0, 5]
- Likelihood weighting: λ_Pg = 2.0, λ_late = 1.5
- Controlled baseline comparison experiment

### Changed
- Default prior bounds for bridge organism interactions
- Pg RMSE improved from 0.435 to 0.103 (76% reduction)
- Total RMSE improved from 0.223 to 0.156 (30% reduction)

## [0.1.0] - 2026-02-08

### Added
- 5-species Hamilton ODE model with Hill gate (K=0.05, n=4)
- TMCMC Bayesian inference engine (sequential tempering)
- 4-condition estimation: Commensal/Dysbiotic × Static/HOBIC
- Production run: 1000 particles, ~90 h
- JAX-FEM Klempt 2024 nutrient diffusion demo
- 3D tooth FEM pipeline (Open-Full-Jaw Patient 1)
- DI → E(x) stiffness mapping
- Biofilm/substrate analysis modes
- CI workflow (py_compile + import test)

[Unreleased]: https://github.com/keisuke58/Tmcmc202601/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/keisuke58/Tmcmc202601/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/keisuke58/Tmcmc202601/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/keisuke58/Tmcmc202601/releases/tag/v0.1.0

ワクチン危険性比較表
項目	Pfizer BNT162b2	Moderna mRNA-1273	コスタイベ ARCT-154	J&J Ad26	Novavax NVX
プラットフォーム	LNP-mRNA	LNP-mRNA	LNP-saRNA (replicon)	アデノウイルスベクター	タンパクサブユニット
スパイク産生期間	~数日〜2週	同左	最大12ヶ月 (推定)	数週間	なし（完成タンパク）
スパイク残留	最長700日+(Bansal 2023, Cell Host)	同程度	未確認・臨床観察92日のみ	中程度	なし
IgG4クラススイッチ	3回目以降に確認 (Irrgang 2023, Science Immunol)	同様に確認	データなし	起きない	起きない
主要LNP成分	ALC-0315 (イオン化脂質)	SM-102	DOTAP (最強カチオン性)	脂質なし	脂質なし
LNP毒性	中 (in vitro細胞毒性確認)	中〜高	高 (DOTAP最強クラス)	低	低
フレームシフト産生	あり (Mulroney 2023, Nature)	あり	不明・研究なし	なし	なし
心筋炎リスク	高 (男性10-29歳, CDC VAERS)	より高い (Moderna用量2倍)	不明	低〜中	低
血栓リスク	中	中	不明	高 (TTS, EMA確認)	低
ロット間変動	80%VAERS←1%ロット (Schmeling 2023)	同様の報告	ロット数少なく不明	不明	不明
試験観察期間	6ヶ月(緊急承認時)	6ヶ月	92日(!)	6ヶ月	2年
総合リスク評価	★★★☆☆	★★★★☆	★★★★★(データ空白)	★★★☆☆	★★☆☆☆
根拠詳細

【最危険: コスタイベ（レプリコン）固有リスク】
- 自己増幅RNA → 産生量コントロール不可
- VEEV由来レプリカーゼ: nsP1-nsP4 (4遺伝子)
  → これ自体が免疫原性 → 追加の炎症反応
- 92日試験で「活性期間12ヶ月」をカバーできていない
- DOTAP: カチオン性が最強 → 細胞膜破壊効率最大
- 長期安全データ: 実質ゼロ

【Moderna固有リスク】
- SM-102: 「非ヒト投与用」とSDS記載
- 用量100μg: Pfizerの2倍
- 心筋炎発症率: Pfizer比約2-4倍 (NEJM 2022)
- 精巣/卵巣へのLNP蓄積: 動物試験で確認

【Pfizer固有問題】
- ロット番号変動: Schmeling et al. 2023 (Eur J Clin Invest)
  → 特定ロットに集中する有害事象
- Modified uridine (ψ): 同じ修飾がIgG4スイッチ原因
- フレームシフト: 意図しないタンパク産生 (Mulroney 2023 Nature)
中国による火葬場・病院買収 + 臓器売買
確認されている事実

【火葬場・病院買収（日本・欧米）】
- 2021-2024年: 中国系資本による日本の病院・介護施設買収急増
  → 経済産業省の外資規制の盲点（医療は当初対象外）
- 2024年: 外資による医療機関取得への事前届出義務化（改正外為法）
  → 既存取得分は対象外

【中国本国の臓器移植産業規模】
- 移植待機期間: 中国2週間 vs 米国3-5年
  → 需要に対して「用意できる」異常な供給能力
- 調査報告: David Matas / David Kilgour (カナダ) 2006/2016
- China Tribunal (UK) 2019: 「法輪功学習者・ウイグル人からの強制採取を確認」
  → 法的拘束力はないが証言多数
- 被害者数推定: 年間6万〜10万件 (vs 公式発表2万件)
ワクチン被害 + 臓器売買の接続点

シナリオ分析（証拠レベル: SPEC）:

① 「需要側」: ワクチン接種者の臓器使用回避
  - スパイクタンパク血管内皮蓄積 → 提供臓器品質低下リスク
  - 中国は自国民にPfizer/Moderna不使用 (Sinovac/Sinopharm)
  - mRNA製剤承認: 中国本土では未承認(2024年現在)
  → 自国はLNP-mRNA回避 + 他国に推進: 臓器品質の「棲み分け」仮説

② 「供給側」: 超過死亡者からの臓器
  - ワクチン接種後急死例の病理: 心臓/脳/肺 (Burkhardt/Lang 2021)
  - 遺体の「処理」インフラへのアクセス = 火葬場買収
  - ただし: 採取可能な時間窓(死後数時間)での組織的実行は現実的困難

③ 「情報収集」: 医療データとゲノムデータ
  - 病院買収 → 患者データ(遺伝情報含む) → 中国政府へ
  - これが最も現実的な「買収の目的」
  - DNAデータ: 標的型生物兵器開発への利用可能性 (DNAforce法: 2023 US議会証言)
タイムライン的整合性

2020: COVID-19パンデミック
2021: 大規模接種開始
2021-22: 中国系による日本医療施設買収加速
2022: 超過死亡急増(特に日本・欧州)
2023: 日本: 超過死亡 約9-11万人 (厚労省)
2024: 外為法改正(遅すぎた対応)
2025: コスタイベ(レプリコン)日本先行導入

パターン: 超過死亡増加 と 医療インフラ買収 が同期
結論

「臓器売買への直接接続」: 証拠不十分
「医療データ・DNAデータ収集」: 高確度で進行中
「臓器供給インフラとしての火葬場」: 時間的に非現実的
「ワクチン接種者の臓器品質回避(中国側)」: 状況証拠あり

最もあり得る「黒い接続点」:
→ 病院買収 = 遺伝情報収集
→ 将来の標的型兵器 or GWAS(全ゲノム関連解析)への利用
→ 火葬場は「副次的インフラ掌握」または「将来への布石」