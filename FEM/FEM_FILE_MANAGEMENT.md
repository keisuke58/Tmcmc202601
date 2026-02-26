# FEM ファイル管理ガイド

最終更新: 2026-02-23
対象ディレクトリ: `Tmcmc202601/FEM`

---

## 1. 目的

- Abaqus や FEM の **生成物（.odb, .inp, .csv, .npy, 図など）が増えすぎないように整理する**
- 「どれがソースで、どれが再生成可能な成果物か」をはっきり分ける
- 将来、新しいスイープや解析を追加しても **同じルールで迷わず置ける** ようにする

---

## 2. 大分類ルール

FEM 直下のファイル/フォルダは、次の 7 区分に整理することを基本方針とする。

1. **ソースコード**: 解析ロジック、Abaqus スクリプト、USDFLD など
2. **入力・ジオメトリ**: CAE, INP, 歯モデル、bbox JSON など
3. **実験スクリプト（ランナー）**: `run_*.py`, `p*_*.py` など
4. **軽量な結果（再生成可）**: `_results*`, `_posterior_plots` など
5. **重い結果・ジョブアーカイブ**: `_posterior_abaqus`, `_job_archive`, `_pressure_sweep` など
6. **図・論文用ファイル**: `figures/`, `fem_report.*`, `docs/` など
7. **一時ファイル・デバッグ**: `_tmp_*` 系, 中間的なメモなど

---

## 3. 既存ディレクトリの位置づけ

### 3.1 ソースコード

- `abaqus_*.py` — Abaqus スクリプト
- `fem_*.py`, `fem_visualize.py`, `compare_biofilm_abaqus.py` など — FEM 本体・解析
- `posterior_sensitivity*.py`, `plot_*.py`, `analyze_abaqus_profiles.py`, `odb_extract.py`, `odb_visualize.py`
- `usdfld_biofilm.f` — USDFLD 実装

これらは **「常に残すべきファイル」** として FEM 直下に置く。

### 3.2 入力・ジオメトリ

- `abaqus_cae/` — CAE モデル
- `abaqus_inp/` — 手動 or 自動生成した `.inp`
- `abaqus_jnl/`, `abaqus_replay/` — CAE 操作ログ
- `external_tooth_models/` — 外部歯モデル
- `p*_tooth_bbox*.json` — 歯の bbox 情報

ここは **「手で編集する or 重要な入力」** を集める。
新しいジオメトリ を追加する場合は、原則としてこのグループに入れる。

### 3.3 実験スクリプト（ランナー）

- `run_posterior_pipeline.py`
- `run_posterior_abaqus_ensemble.py`
- `run_material_sensitivity_sweep.py`
- `run_aniso_comparison.py`
- `run_czm3d_sweep.py`
- `run_oj_beta_jobs.sh`, `run_oj_beta_sweep.py`
- `p3_pressure_sweep.py`, `p4_mesh_convergence.py`, `p7_tie_diagnostic.py`, `p8_element_quality.py` など

**新しいスイープや実行シナリオ** を追加するときは、`run_*.py` か `p*_*.py` 形式にし、FEM 直下に置く。

### 3.4 軽量な結果（再生成可）

主に **プロット・集計結果・軽い numpy/csv** を含むもの:

- `_results/`, `_results_2d/`, `_results_3d/`, `_results_3d_long/`
- `_posterior_plots/`, `_posterior_sensitivity/`, `_posterior_uncertainty/`
- `_material_sweep/`
- `_di_credible/`
- `_aniso/`, `_aniso_sweep/`
- `_benchmarks/`
- `_mesh_convergence/`

方針:

- これらは **すべて「再生成可能」** とみなし、必要に応じて削除してよい
- 新しい解析を追加する場合も、**原則として `_experiment_name/` 形式の軽量出力ディレクトリを作る**

### 3.5 重い結果・ジョブアーカイブ

- `_posterior_abaqus/`, `_posterior_abaqus_test/`
- `_posterior_abaqus_ALL_DONE.flag` — 完了フラグ
- `_pressure_sweep/`
- `_job_archive/` 以下
  - `aniso/`, `biofilm_demo/`, `di_credible/`, `material_sweep/`, `old_field_csv/` など

方針:

- `.odb`, `.inp`, `.sta`, `.dat` など **Abaqus ジョブ関連ファイルは基本的に `_job_archive/` 以下に集約** する
- 新しいジョブを大量に投げる場合は、
  - `_job_archive/new_experiment_name/` を作り、そこに Abaqus のジョブディレクトリをまとめる
- `_posterior_abaqus/` や `_pressure_sweep/` のような **大きな結果ディレクトリ** は、
  - 一定期間ごとに外部ストレージに移すか、
  - 生成条件（θ_MAP、条件名等）が明確なら再実行前提で削除してもよい

### 3.6 図・論文用ファイル

- `figures/` — 論文・スライド用に厳選した図を保存
- `fem_report.pdf/.tex`, `abaqus_implementation_report.pdf/.tex`
- `docs/` 以下の `fem_pipeline.*` など

方針:

- **最終成果物として残す図** は `figures/` に集約する
- `_posterior_plots/` など中間的な大量の図は、「後で選別するバッファ」として扱う

### 3.7 一時ファイル・デバッグ

- `_tmp_2d_*`, `_tmp_3d_*` など `_tmp_*` 系
- `_job_archive/old_field_csv/` のような一時的な退避

方針:

- `_tmp_*` は **完全に一時領域** とし、「見終わったら削除してよい」ことを前提に使う
- 新しく一時的な実験をする際は、`_tmp_experiment_name/` を作ってまとめる

---

## 4. ディレクトリ命名ルール（新規追加時）

新しいスクリプトや解析を増やすときの命名ポリシー:

1. **スクリプト名**: `run_XXX.py` または `pN_XXX.py`
2. **軽量出力ディレクトリ**: `_<短い実験名>/`
3. **Abaqus ジョブアーカイブ**: `_job_archive/<短い実験名>/`
4. **一時用途**: `_tmp_<説明>/`

例:

- 新しい Cohesive モデルのスイープ
  - スクリプト: `run_czm3d_beta_sweep.py`
  - 軽量結果: `_czm3d_beta_sweep/`
  - ジョブファイル: `_job_archive/czm3d_beta_sweep/`

このように **「スクリプト名」「軽量結果ディレクトリ」「ジョブアーカイブディレクトリ」** を 1 セットとして揃えておくと、

- どのフォルダがどのスクリプト由来かすぐ追える
- 不要になった実験結果を安全に削除できる

---

## 5. ログ・実行記録の扱い

- `abaqus_logs/` — Abaqus 実行ログ（.log, .msg など）
- TMCMC → FEM パイプラインやリモート実行のログ（`*_run.log` など）は、
  - 原則としてプロジェクト直下の `data_5species/_runs/...` にあるが、
  - FEM 特有のログを増やす場合は `abaqus_logs/` または `logs/` ディレクトリを作り、そこに集約する

方針:

- **長期的に残したいログ** は、条件名・日付を含めたファイル名にする:
  `dysbiotic_static_20260205_abaqus.log` など
- 一時的なデバッグログは `_tmp_*` と同様、確認後に削除してよい

---

## 6. クリーンアップの目安

ディスクがいっぱいになってきたときの削除優先度の目安:

1. `_tmp_*` ディレクトリ
2. `_results*`, `_posterior_plots`, `_posterior_sensitivity`, `_posterior_uncertainty` など **軽量出力**
3. `_pressure_sweep`, `_material_sweep`, `_aniso*`, `_di_credible`, `_benchmarks`, `_mesh_convergence`
4. `_job_archive/` 以下のジョブファイル（必要なら再実行で復元）
5. `_posterior_abaqus*`（計算時間と相談しつつ、バックアップ後に削除）

削除前に、必要であれば:

- 図だけ `figures/` に残す
- 実験条件（θ_MAP, 条件名, 日付）を `docs/` かメモに記録する

---

## 7. 新しい実験を追加する際のテンプレ

1. `run_new_experiment.py` を FEM 直下に追加
2. 出力先を `"_new_experiment/"` に統一
3. Abaqus ジョブを `_job_archive/new_experiment/` に保存
4. プロット・最終図は `figures/` へコピー（または移動）
5. 不要になった `_tmp_*` や中間結果は適宜削除

このファイルは **運用しながら少しずつ更新していく前提** のメモとして使う。
ルールを変えたくなった場合も、まずここに追記してから実際の整理を行う。
