# 2〜4 番目の投稿先用（BMB で落ちたときの予備・軽量版）

**同時投稿は不可**（二重投稿は出版倫理違反。どの誌も投稿時に「他誌で審査中でない」と宣言させる）。
BMB の結果が出てから、下の順に 1 誌ずつ出す。

`manuscript_format_free.tex` は BMB 版（`../BMB_submission/BMB_manuscript_nishioka.tex`）と**本文が同じ**で、
article クラス・行番号・1.5 行送りにしただけ。3 誌とも初回投稿はテンプレート不要の想定（**投稿前に各誌の
Instructions for Authors で要確認**）。本文を直すときは BMB 版を先に直し、ここに写す。

| 順 | 誌 | 出す前に足すもの |
|---|---|---|
| 2 | **Journal of Biological Dynamics**（Taylor & Francis、Gold OA、APC USD 1,680、**DEAL 対象外**） | keywords、cover letter（BMB 版の宛先と誌名を差し替え）。APC の扱いを Junker 先生に確認 |
| 3 | **Journal of Theoretical Biology**（Elsevier、DEAL 対象） | **Highlights 3〜5 行（各 ≤85 字）必須**、keywords 1〜7、CRediT 形式の著者貢献、Declaration of competing interest |
| 4 | **IJNMBE**（Wiley、DEAL 対象） | **Novelty statement（≤100 語）**、図付き要旨、keywords ≤7、**収束・精度の確認が必須**（数値手法の誌なので GPU 実装と収束判定を前に出す）、コード・データを Data Files でアップロード |

下書き（ult の後に数字を入れて仕上げる）:

**JTB Highlights 案**
- Bayesian inference of the full interaction matrix of a five-species oral biofilm model
- No interaction is fixed a priori; every stage is checked across independent seeds
- The Fn–Pg coefficient is positive only under dysbiotic dynamic cultivation
- Removing F. nucleatum suppresses the late P. gingivalis increase in silico
- The prediction can be tested by a leave-one-species-out co-culture

**IJNMBE Novelty statement 案**
We infer all 15 interaction parameters of a variational multi-species biofilm model from in-vitro
time courses with a GPU-parallel TMCMC sampler, and we verify convergence with explicit multi-seed
criteria and identifiability checks on a wide prior box. The inferred model predicts that removing
F. nucleatum suppresses the late P. gingivalis increase, a prediction testable by experiment.
