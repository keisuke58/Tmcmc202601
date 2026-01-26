# tmcmc/ に追加された論文PDFと「今回の実装」の対応関係（整理）

このメモは、`tmcmc/` 配下に置かれている論文PDFが **現在の実装（TMCMC × TSM-ROM + 線形化点更新）** のどの部分に対応しているか、また **未実装/実装方針の差分** がどこにあるかを短く整理したものです。

## どの実装を指しているか（ここでの「今回の実装」）

- **推論パイプライン**: `tmcmc/case2_tmcmc_linearization.py`
  - TMCMC（β-tempering / ESSターゲット / resampling / mutation）
  - TSM-ROM（平均・分散ベースの尤度、`cov_rel` で aleatory を表現）
  - **線形化点（θ₀）更新**（ROM精度維持のための実装拡張）
- **TSM-ROM本体**: `tmcmc/demo_analytical_tsm_with_linearization_jit.py` + `tmcmc/analytical_derivatives_jit.py`
- **FOM（基準モデル）**: `tmcmc/improved1207_paper_jit.py`
- 実行入口: `tmcmc/run_pipeline.py`（実験→レポート生成）

## PDFごとの関係性（結論）

| PDF | 役割（実装に対して） | 実装上の対応箇所 | ギャップ/注意 |
|---|---|---|---|
| `Bayesian updating of bacterial microfilms under hybrid uncertainties with a novel surrogate model - Kopie.pdf` | **アルゴリズムの“骨格”**（hybrid uncertainty + TSM-ROM + TMCMC） | `case2_tmcmc_linearization.py`（TMCMCと、平均・分散の尤度、`cov_rel`、single-loop指向） | 論文は **p-box（未知平均 + CoV固定）** を明示。実装は **一様境界prior（bounds）** で近い解釈は可能だが、p-boxそのものの表現/出力はしていない（＝「平均の推定」中心）。 |
| `biofilm_simulation.pdf`（A continuum multi-species biofilm model…） | **前向きモデル（biofilm方程式）の仕様/背景**（multi-species・相互作用行列・抗生物質項・Hamilton原理） | FOM: `improved1207_paper_jit.py`、そのROM化: `demo_analytical_tsm_with_linearization_jit.py` | これは主に **モデル側** の参照。TMCMCの手順自体（β/ESSなど）はこのPDFからは直接来ない。 |
| `hamiltonian.pdf`（extended Hamilton principle…） | **理論的背景**（Hamilton原理を「散逸/連成問題」に拡張する一般論） | `biofilm_simulation.pdf` → `improved1207_paper_jit.py` の「エネルギー原理からの導出」の根拠側 | 推論アルゴリズム（TMCMC/TSM）には直接は関係しない。モデルの正当化・説明のための引用候補。 |
| `Influence of species composition … peri-implant biofilm dysbiosis in vitro.pdf` | **実験・ドメイン背景**（peri-implant/口腔biofilm、species composition、flow vs static、dysbiosis） | `config.py` の `MODEL_CONFIGS`（M1/M2/M3の「種の組」「条件」をどう置くかの動機付けに使える） | これは **“何をデータとして同定するか”** の設計に効く。現状の実装は「合成データでの検証」が主なので、論文の測定量（pH、種分布、flow条件など）を使うなら尤度・観測モデル設計が追加で必要。 |

## 実装が論文（Bayesian updating…）と一致している点（高確度）

- **hybrid uncertaintyの計算コスト問題**に対して、TSM-ROMで **single-loop** を目指す設計
- **尤度が平均・分散（共分散）に依存**する構成（合成データ・ノイズ込みの比較）
- **TMCMC（β-tempering）**で prior→posterior を段階的に遷移し、ESSを指標に進める基本形
- 係数変動（CoV）を固定して平均（または中心パラメータ）を推定する、という意味での整合

## 実装が論文から「意図的に拡張/簡略化」している点（ギャップ候補）

- **p-boxを“箱（区間）として更新して可視化する”** ところは、現状は明示的にやっていない  
  - 代わりに `prior_bounds`（一様）+ posteriorサンプルで「平均の不確かさ」を表現している
- 論文側が比較する可能性がある **full covariance**（観測共分散）について、現状は **対角中心**の扱いになっている可能性が高い  
  - ここは「どの観測をベクトル化しているか」「相関を入れると安定するか」で要検討
- 本実装固有の要素として **線形化点更新（θ₀更新）** と **ROM誤差に基づくガード/重み付け** が入っている  
  - これは「TSMのTaylor展開点が posterior 移動に追従しないと崩れる」問題への実務的対策

## 次にやると良い“実装に効く”読み方（おすすめ）

- `Bayesian updating…` からは、特に以下を重点的に確認すると実装改善に直結します
  - TMCMCの **βスケジュール決定法**（ESSターゲットの式・Δβ探索）
  - mutation（提案分布）設計（tempered covariance の定義、スケール係数）
  - 尤度の共分散を **対角に落とす根拠** と、その影響（図6付近）
- `Influence…` は「将来、実データで校正する」段で効きます（観測量・条件・時間スケール）

