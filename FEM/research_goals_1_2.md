## 研究の最終目標（レベル1・2・3・4）
## Overall Research Goals (Levels 1–4)

このドキュメントでは、現在進めている研究の「最終目標」を  
レベル1（メカニズム解明）、レベル2（予測モデル・デジタルツイン）、  
レベル3（介入設計）、レベル4（数理・計算手法フレームワーク）に分けて整理し、  
現状どこまで到達しているか／レベル2までに何が不足しているか（差分）と、  
レベル3・4の位置づけをまとめる。

This document organizes the overall research goals into  
Level 1 (mechanism elucidation), Level 2 (predictive model / digital twin),  
Level 3 (intervention design), and Level 4 (mathematical / computational framework),  
and summarizes (i) how far we have currently progressed, (ii) what is missing to reach Level 2,  
and (iii) how Levels 3 and 4 are positioned relative to these.

---

## レベル1：メカニズム解明レベル
## Level 1: Mechanism-Level Understanding

### 目標のイメージ
### Goal

- 歯周バイオフィルムにおいて  
  - 菌叢構成（Commensal / Dysbiotic）  
  - 代謝・拡散（栄養・pH・代謝産物）  
  - 力学環境（バイオフィルム内応力・せん断）  
  の三つを統一的に結ぶ「力学―生態学モデル」を構築する。
- 特に以下を**定量的に**説明できることを目指す。
  - どのくらいの応力・せん断がかかると、どの菌が残りやすいか／剥がれやすいか  
  - その結果として、歯周病・う蝕リスクにどう結びつくか

- In the periodontal biofilm, construct an integrated “mechanics–ecology” model that links  
  - microbiota composition (commensal / dysbiotic),  
  - metabolism and diffusion (nutrients, pH, metabolites), and  
  - mechanical environment (internal biofilm stress and shear).
- Aim to explain quantitatively  
  - which levels of stress / shear allow which species to persist or be removed, and  
  - how this leads to periodontal disease and caries risk.

### 現状（2026時点）
### Current Status (as of 2026)

- TMCMC によるパラメータ推定
  - 4条件（Commensal / Dysbiotic × Static / HOBIC）でのモデル同定が進行中。
  - 粒子数・ロックルールなど、推定の設計指針がかなり固まりつつある。
- バイオフィルムの連続体モデル・FEM への拡張
  - 歯の周りのバイオフィルムを対象にした FEM モデル構築が進行。
  - パラメータの確率的幅（事後分布）を持った推定結果があり、  
    そこから応力分布にも「幅」（不確実性）を伝播できる状態に近づいている。
- 可視化・解析
  - 応力分布や菌叢構成の可視化に向けた基礎的なスクリプト・レイアウト案が存在。

- Parameter estimation via TMCMC
  - Model calibration for four conditions (commensal / dysbiotic × static / HOBIC) is ongoing.
  - Design principles for inference (particle counts, locking rules, etc.) are becoming well established.
- Extension to a continuum / FEM model for the biofilm
  - FEM models targeting biofilm around teeth are under development.
  - Estimated parameters with probabilistic width (posterior distributions) are available,  
    allowing propagation of this uncertainty into stress fields.
- Visualization and analysis
  - Basic scripts and layout ideas exist for visualizing stress distributions and microbiota composition.

### レベル1達成までに必要な差分
### Gaps to Reach Level 1

- モデルの統合と整理
  - TMCMC で同定したパラメータセットと FEM 応力解析を接続し、  
    「菌叢構成 × 力学応力」の関係を一つのパイプラインとしてまとめる。
  - どのパラメータが主に力学側に効いているか（感度解析）を整理する。
- メカニズムの言語化
  - 応力分布と菌叢の空間分布のパターンを、  
    「どこで菌が残りやすいか／剥がれやすいか」という形で整理する。
  - 歯周病・う蝕リスクとの関係を、図＋テキストで明示できるようにする。
- 検証ケースの整備
  - 解析解・単純形状（平板・円柱など）で、FEM + 反応拡散モデルの挙動を検証する。
  - 実験データとの比較が可能な指標（菌量・厚さ・形状変化など）を定義し、  
    モデルが「どの程度まで」再現できているかを定量評価する。

- Integrating and organizing models
  - Connect TMCMC-estimated parameter sets with FEM stress analysis,  
    building a single pipeline for “microbiota composition × mechanical stress”.
  - Identify which parameters primarily affect the mechanical side (sensitivity analysis).
- Articulating mechanisms
  - Organize patterns of stress distributions and spatial microbiota distributions  
    in terms of “where bacteria tend to remain or be removed”.
  - Make the link to periodontal / caries risk explicit using figures and text.
- Preparing validation cases
  - Validate FEM + reaction–diffusion behavior on simple geometries with analytical solutions (plates, cylinders, etc.).
  - Define metrics that can be compared to experimental data (biomass, thickness, shape changes)  
    and quantitatively assess how well the model reproduces them.

---

## レベル3：介入設計レベル（補足的位置づけ）
## Level 3: Intervention Design (Supplementary)

### 目標のイメージ
### Goal

- モデル内で
  - ブラッシング頻度・方向・力  
  - マウスウォッシュの成分・タイミング  
  - 噛みしめの癖による局所応力  
  などの「介入パラメータ」を操作し、
- どの介入パターンが、どの患者タイプに最も有効か  
  をシミュレーションで比較できるようにする。

- Within the model, manipulate “intervention parameters” such as  
  - brushing frequency, direction, and force,  
  - mouthwash composition and timing, and  
  - local stresses due to occlusal habits,  
  and compare, via simulation, which intervention patterns are most effective for which patient types.

### レベル3の位置づけ
### Position of Level 3

- 実質的には「レベル2（デジタルツイン）」ができれば、
  - 入力パラメータを変えることで、  
    介入シナリオ比較は**自然に実現**される。
- そのためレベル3は、
  - レベル2の**応用ケース・補足的な拡張**として捉える。
  - 独立したゴールというより、「デジタルツインが生んでくれる機能」の一部。

- Once Level 2 (digital twin) is realized,  
  - changing input parameters naturally enables comparison of intervention scenarios.
- Therefore, Level 3 is regarded as  
  - an applied, supplementary extension of Level 2,  
  - not an entirely separate goal, but a function that emerges from the digital twin.

---

## レベル4：数理・計算手法としての最終目標
## Level 4: Mathematical and Computational Framework

### 目標のイメージ
### Goal

- 歯周バイオフィルムのような「反応拡散＋力学（FEM）」系に対して、
  - TMCMC などによるベイズ推定  
  - FEM による連続体力学解析  
  - パラメータ不確実性の伝播とリスク評価  
  を統合した**汎用フレームワーク**を提示する。
- 応用範囲
  - 歯周バイオフィルムだけでなく、
    - 医療デバイス上のバイオフィルム  
    - 他臓器の粘膜上バイオフィルム  
  など、構造＋バイオフィルム系全般への展開を視野に入れる。

- For “reaction–diffusion + mechanics (FEM)” systems such as periodontal biofilms,  
  present a unified, general framework that integrates  
  - Bayesian parameter estimation (e.g., via TMCMC),  
  - continuum mechanics analysis using FEM, and  
  - uncertainty propagation and risk assessment.
- Scope of application
  - Not limited to periodontal biofilms, but also extending to  
    - biofilms on medical devices and  
    - biofilms on mucosal surfaces in other organs,  
    i.e., structural–biofilm systems more broadly.

### 現状（2026時点）
### Current Status (as of 2026)

- 要素技術
  - 反応拡散モデル＋TMCMC によるパラメータ同定は、  
    複数条件での適用実績が蓄積しつつある。
  - バイオフィルムを対象とした FEM 応力解析と、そのためのメッシュ・境界条件設定も整備中。
- 統合の方向性
  - 「推定 → シミュレーション → 不確実性評価」の一連の流れは  
    コード上でほぼ実現済みであり、あとは
    - インターフェースの整理  
    - 手順の一般化・抽象化  
    を行うことで、他の系にも流用しやすい形にできる段階に近い。

- Element technologies
  - Parameter identification using reaction–diffusion models and TMCMC  
    has been applied under multiple experimental conditions.
  - FEM stress analysis of biofilms and the associated mesh / boundary-condition setup  
    are being established.
- Direction of integration
  - The pipeline “estimation → simulation → uncertainty evaluation”  
    is almost realized in code; the remaining steps are  
    - cleaning up interfaces and  
    - generalizing / abstracting the workflow  
    so that it becomes easily reusable for other systems.

### レベル4達成までに必要な差分
### Gaps to Reach Level 4

- 手順の一般化・整理
  - データ準備 → モデル定義 → TMCMC 設定 → FEM 設定 → 解析 → 可視化  
    という手順を、テンプレート化・ドキュメント化する。
- コード構造のモジュール化
  - 問題依存な部分（ジオメトリ、パラメータ範囲、境界条件）と、  
    汎用的な部分（TMCMC ラッパー、FEM ソルバ、可視化）は  
    ファイル／クラスレベルで分離する。
- 代表的な応用例の提示
  - 歯周バイオフィルム以外に 1〜2 例、  
    簡略化された他のバイオフィルム系へのデモ適用を行い、  
    「フレームワークとしての再利用性」を示す。

- Generalizing and organizing the workflow
  - Template and document the sequence  
    “data preparation → model definition → TMCMC setup → FEM setup → analysis → visualization”.
- Modularizing the code structure
  - Separate problem-specific parts (geometry, parameter ranges, boundary conditions)  
    from generic parts (TMCMC wrappers, FEM solvers, visualization) at file / class level.
- Presenting representative applications
  - Provide 1–2 demonstration applications to biofilm systems other than periodontal biofilms  
    to showcase the reusability of the framework.

## レベル2：予測モデル・デジタルツインレベル
## Level 2: Predictive Model and Digital Twin

### 目標のイメージ
### Goal

- 一人の患者について
  - 歯列形状・ポケット形状  
  - 想定されるバイオフィルム構成（Commensal / Dysbiotic の傾きなど）  
  を入力すると、
  - FEM によるバイオフィルム内の応力・せん断分布  
  - TMCMC によって同定された菌叢ダイナミクスモデル  
  から、
  - 「この人のどの部位が、どれくらい病的になりやすいか」を  
    **確率付きで予測**できるデジタルツインを目指す。
- 出力イメージ
  - 歯列・ポケットごとのリスクマップ（色で表示）  
  - 将来の進行シナリオ（半年〜数年スケール）の予測  
  - 介入（ブラッシング、マウスウォッシュなど）の効果予測

- For each individual patient, use as input  
  - tooth and pocket geometries and  
  - the expected biofilm composition (bias toward commensal / dysbiotic states),  
  and, based on  
  - stress and shear distributions in the biofilm computed by FEM and  
  - microbiota dynamics models identified via TMCMC,  
  aim to build a digital twin that predicts, with probabilities,  
  which sites are more likely to become diseased.
- Expected outputs
  - risk maps over teeth and pockets (visualized by color),  
  - predictions of future progression scenarios (over months to years), and  
  - predicted effects of interventions (brushing, mouthwash, etc.).

### 現状（2026時点）
### Current Status (as of 2026)

- モデルレベル
  - 「条件ごとの」菌叢ダイナミクスモデル（TMCMC）と、  
    「形状に基づく」FEM 応力解析の土台ができつつある段階。
  - パラメータの不確実性（事後分布）を保持しており、  
    応力やリスクにも幅を持たせた評価が理論的には可能。
- データレベル
  - 患者固有の歯列形状やポケット形状に対応したモデル化は、  
    一部（外部モデルやデータセットの利用）で進行中だが、  
    「個別患者ごとの一貫したパイプライン」としてはまだ途中。
- 実装レベル
  - コードは「研究用スクリプト」としては動いているが、  
    患者ごとの入力 → モデル実行 → リスク可視化 までを  
    一つのシステムとしてまとめる段階には達していない。

- Model level
  - Condition-specific microbiota dynamics models (via TMCMC) and  
    geometry-based FEM stress analysis are both under development.
  - Parameter uncertainty (posterior distributions) is retained,  
    so in principle stress and risk can be evaluated with probabilistic width.
- Data level
  - Modeling patient-specific tooth and pocket geometries is in progress  
    (using external models and datasets),  
    but a consistent pipeline for individual patients is not yet complete.
- Implementation level
  - The code functions as “research scripts”,  
    but not yet as a single system that goes from patient input  
    to model execution and risk visualization.

### レベル2達成までに必要な差分
### Gaps to Reach Level 2

- 1. 患者固有ジオメトリの取り込み
  - 歯列・ポケット形状を FEM メッシュに落とし込む安定した手順を整備する。  
    - 例：CT / 3D スキャン → メッシュ生成 → 境界条件設定の半自動化
  - 複数患者に対して同じフローを回せるよう、スクリプト・設定ファイルを整理する。
- 2. モデルのモジュール化・パイプライン化
  - 「入力データ → TMCMC パラメータ同定 → FEM 応力解析 → リスク評価」までを  
    一つのパイプラインとしてコード上で整理する。
  - 条件（Commensal / Dysbiotic, Static / HOBIC）切り替えや、  
    介入シナリオの変更をパラメータで切り替えられるようにする。
- 3. 不確実性の可視化とリスク指標の定義
  - パラメータサンプルから応力分布・菌叢分布をサンプリングし、  
    各点の「平均」と「信頼区間（信用区間）」を可視化できるようにする。
  - 臨床的に意味のあるリスク指標（例：一定期間内に閾値を超える確率など）を定義し、  
    モデル出力をその指標に変換するロジックを実装する。
- 4. 簡易 UI or レポート生成
  - 歯科医・研究者が
    - 入力条件（患者情報・介入条件）  
    - 出力（リスクマップ・応力分布・菌叢変化）  
    をまとめて確認できるレポート（図＋テキスト）生成機能を用意する。
  - フル UI でなくても、スクリプトから自動生成される PDF / 画像一式でもよい。

- 1. Incorporating patient-specific geometries
  - Establish a robust process to convert tooth / pocket geometries into FEM meshes  
    (e.g., CT / 3D scan → mesh generation → semi-automatic boundary-condition setup).
  - Organize scripts and configuration files so that the same workflow runs for multiple patients.
- 2. Modularizing and pipelining the model
  - Organize code into a single pipeline  
    “input data → TMCMC parameter identification → FEM stress analysis → risk evaluation”.
  - Enable switching between experimental conditions  
    (commensal / dysbiotic, static / HOBIC) and intervention scenarios via parameters.
- 3. Visualizing uncertainty and defining risk metrics
  - Sample stress and microbiota distributions from parameter samples,  
    and visualize mean and credible intervals at each location.
  - Define clinically meaningful risk metrics (e.g., probability of exceeding a threshold within a period)  
    and implement logic to map model outputs to these metrics.
- 4. Simple UI or report generation
  - Provide functionality to generate reports (figures + text) that summarize  
    inputs (patient information, intervention settings) and outputs  
    (risk maps, stress distributions, microbiota changes).
  - A full UI is not required; automatically generated PDFs / image sets from scripts are sufficient.

---

## まとめ
## Summary

- レベル1（メカニズム解明）は、
  - TMCMC によるパラメータ推定  
  - FEM への拡張  
  が進んでおり、「あと一歩」で統合と検証・言語化のフェーズに入れる段階にある。
- レベル2（予測モデル・デジタルツイン）は、
  - 個別患者ジオメトリの取り込み  
  - モデル・コードのパイプライン化  
  - 不確実性を含めたリスク指標の定義  
  が主なギャップとして残っている。

- Level 1 (mechanism-level understanding)
  - Parameter estimation via TMCMC and  
  - extension to FEM  
  are already in progress, and the project is close to the phase of integration, validation, and articulation.
- Level 2 (predictive model / digital twin)
  - Incorporating patient-specific geometries,  
  - pipelining models and code, and  
  - defining risk metrics under uncertainty  
  remain as the main gaps.

この差分を一つずつ埋めていくことで、  
現状の研究コードを「患者ごとのリスク予測・介入設計まで見通せるモデル」へと  
段階的に発展させることができる。

By filling these gaps step by step,  
the current research codebase can be gradually developed into a model that supports  
patient-specific risk prediction and intervention design.

---

## 顎データセットを用いた第1報論文の構想
## Concept for the First Paper Using the Jaw Dataset

### タイトル候補

- Option A（方法論＋ケーススタディ寄り）
  - A Bayesian mechano-ecological model of periodontal biofilm on a patient-specific jaw geometry
- Option B（デジタルツイン色をやや強め）
  - Towards a digital twin of periodontal biofilms: TMCMC-calibrated mechano-ecological modeling on patient-specific jaws
- Option C（不確実性・応力を強調）
  - Uncertainty-aware FEM of periodontal biofilm stress calibrated by TMCMC on a patient-specific jaw

現状のデータセットと進捗を踏まえると、  
「方法論＋ケーススタディ」を前面に出す Option A または  
「Towards」を付けて背伸びし過ぎない形にした Option B / C が現実的な候補。

### 想定する論文の性格

- 個別患者の臨床アウトカム予測までは踏み込まず、
  - 顎・歯列ジオメトリ（Patient_1 など）上に
  - TMCMC で同定したパラメータ分布と FEM 応力解析を載せ、
  - パラメータ不確実性が顎レベルの応力・リスク分布にどう伝播するか
  を示す「方法論＋顎レベルケーススタディ」として位置づける。

### 図構成案（Fig1〜4）

- Fig1: 全体フレームワーク
  - (a) 顎・歯列の 3D ジオメトリ（Patient_1 など）
  - (b) 代表的な歯周ポケット周りのバイオフィルム領域の拡大図
  - (c) モデルフロー概念図  
    （実験条件 → TMCMC → パラメータ分布 → FEM メッシュ＋境界条件 → 応力・バイオフィルム場 → 不確実性評価）
  - 役割：顎データセットと TMCMC＋FEM＋不確実性のつながりを一望させるイントロ図。

- Fig2: TMCMC によるパラメータ同定の要約
  - (a) 代表的パラメータ（成長率・相互作用・力学関連など）の事後分布
  - (b) 代表条件での data vs model（TMCMC 既存図の圧縮版でもよい）
  - 役割：顎シミュレーションの「入力」となるパラメータ分布が、  
    ベイズ推定で妥当に同定されていることを示す。

- Fig3: 顎レベル FEM 応力分布（点推定ベース）
  - (a) 顎全体におけるバイオフィルム応力分布（外側から見た図）
  - (b) 歯周ポケット周囲の拡大図（応力集中／守られたポケットの例）
  - (c) 条件（Commensal vs Dysbiotic など）を変えたときの比較パネル
  - 役割：現実的な顎形状上で、バイオフィルム応力がどのように空間分布するかを可視化する中心図。

- Fig4: 不確実性伝播とリスク指標
  - (a) 代表歯周ポケット周りの線に沿った「応力の平均＋信用区間バンド」  
    （TMCMC サンプル → FEM を複数回 → 平均と 95% 区間など）
  - (b) 顎表面の簡易リスクマップ  
    （例：一定閾値以上の応力が生じる確率、あるいは応力＋菌量条件を満たす確率）
  - (c) 可能であれば、介入シナリオ（ブラッシング強度など）を 1 条件だけ変えた比較パネル
  - 役割：パラメータの不確実性が応力・リスクの幅として現れることを示し、  
    「不確実性を意識した顎レベルモデリング」という本論文の特徴を強調する。

---

## 本研究の主な新規性メモ
## Summary of Main Contributions and Novelty

- 顎・歯列・PDL の有限要素解析と、歯周バイオフィルムの反応拡散・菌叢ダイナミクスを  
  一つのモデルフレームワーク（メカノエコロジーモデル）に統合している。
- TMCMC によるベイズ推定で得られたパラメータの事後分布を、  
  顎レベル FEM に流し込むことで、バイオフィルム応力場とリスクを  
  「平均値だけでなく信用区間・閾値超え確率」まで含めて評価している。
- 歯科デジタルツイン研究が主に硬組織・装置の力学に集中しているのに対し、  
  本研究はバイオフィルム・菌叢・dysbiosis を含むメカノエコロジー的デジタルツインを志向している。
- 既存の FEM（顎・PDL）、in vitro バイオフィルムモデル、デジタルツイン研究の  
  3 つの領域の間をブリッジし、「力学 × 生態 × 不確実性」を顎レベルで同時に扱う  
  新しいモデリング枠組みとして位置づけられる。
