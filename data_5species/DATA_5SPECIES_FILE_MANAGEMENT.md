# data_5species ファイル管理ガイド

最終更新: 2026-02-23  
対象ディレクトリ: `Tmcmc202601/data_5species`

---

## 1. 目的

- 5 種モデル TMCMC 推定の **ラン・スイープ・ログ・図** を整理する
- 「コード」「設定」「実験結果」「ログ」「図」をはっきり分けて、  
  後から「どの結果がどの run だったか」を追いやすくする
- サーバー上の `_runs` が増えても、**安全にクリーンアップできる状態** を保つ

---

## 2. 大分類ルール

`data_5species/` 以下は、次の 7 区分で考える。

1. **コアコード・設定**: `main/`, `species_config.json`, Python テスト等
2. **ランナー・スクリプト**: 各条件・スイープ用の `.sh` / `.py`
3. **観測データ・静的ファイル**: 実験データ、真値、ネットワーク定義など
4. **ラン結果 `_runs/`**: TMCMC 実行ごとの run ディレクトリ
5. **サマリ・集計**: `results`, スイープ集計、ベストランのメモ
6. **図・可視化**: `bounds_visualization.png`, `interaction_network.png` など
7. **ログ・一時ファイル**: `*.log`, 古い sweep ログ、クリーンアップ対象

---

## 3. コアコード・設定

### 3.1 main/

`data_5species/main/` には、5 種モデル専用のメインスクリプトを置く:

- `refine_5species.py` などの **本体ロジック**
- `generate_extra_figures_generic.py` など **run ディレクトリに対して図を生成するスクリプト**

方針:

- 5 種モデルのロジックは原則 `main/` 配下に集約する
- run ごとに実行する補助スクリプトも、コード自体は `main/` に置き、
  引数として run ディレクトリを与える

### 3.2 設定ファイル

- `species_config.json` — 種のラベルやカラー、図示用設定
- `interaction_graph.json` — ネットワーク構造（エッジと種類）

これらは **「観測データではなくモデル設定」** として扱う。  
変更する場合は commit 対象として残す。

---

## 4. ランナー・スクリプト

root 直下にある `.sh` / `.py` は、主に **実行オーケストレーション** 用:

- 条件別ランナー
  - `run_all_4conditions.sh`
  - `run_commensal_absolute.sh`
  - `run_improved_estimation.sh`
  - `run_hobic_estimation.sh`
  - `run_tight_decay_estimation.sh`
- スイープ系
  - `run_pg_weighted_sweep.sh`
  - `check_sweep.sh`
  - `check_jobs.sh`
- 管理・ユーティリティ
  - `cleanup_runs.py`  — 不完全な run を `_runs/archive_incomplete/` などに移すスクリプトを想定
  - `test_bounds.py`, `compare_nishioka_standard.py` などの検証用スクリプト

方針:

- 新しいスイープを追加する場合は、`run_*.sh` 命名で root に置く
- Python ベースの一括処理は `main/` 以下に置き、ここから呼び出す

---

## 5. 観測データ・静的ファイル

主に **実験から得られたデータやモデル可視化のベース** になるもの:

- `interaction_graph.json` — ネットワーク構造
- `bounds_visualization.png`, `interaction_network.png` — パラメータ境界やネットワークの図（準静的）

方針:

- 実験データや真値が増えてきた場合は、`data/` サブディレクトリを切ってまとめる
- 可視化用のベース図も、更新頻度が低ければここ（root または `data/`）に置いてよい

---

## 6. ラン結果 `_runs/`

TMCMC の結果は **必ず `_runs/` 以下に集約する**。

- 例: `/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs`
- ランナーの標準構造:
  - `_runs/<run_id>/` — 1 つの run
  - `_runs/sweep_pg_YYYYMMDD_HHMMSS/` — sweep 用の親フォルダ
    - その中に `<tag>/`, `<tag>.log`, `fit_metrics.json` など

`cleanup_runs.py` の想定する標準構造:

- 完了した run: `fit_metrics.json`, `theta_MAP*.json`, 図などが揃っている
- 不完全な run: `fit_metrics.json` がない → アーカイブ or 削除候補

方針:

- **新しい run は必ず `_runs/` の下に作る**
- sweep やテストでも `_runs/sweep_xxx/` を親にし、その下に run を作る
- 手動 run でも、原則 `_runs/manual_xxx/` を使う

---

## 7. サマリ・集計

- `results` — ベストランや集計結果をまとめるためのファイルを置くことを想定
  - 例: `results/best_run.txt`, `results/sweep_summary.csv` など

方針:

- 「多数の run から選んだ結果のまとめ」をここに置く
- 生の run ディレクトリとは分離しておくことで、後から `_runs` をクリーンアップしても  
  サマリだけ見れば論文に必要な数字が再現できる状態を目指す

---

## 8. 図・可視化

root 直下に既にいくつかの図がある:

- `bounds_visualization.png`
- `interaction_network.png`

また、各 run ディレクトリ内でも、`generate_extra_figures_generic.py` などで図を生成する。

方針:

- **run 固有の図** は原則として `"_runs/<run_id>/figures/"` にまとめる
- その中から「論文採用」などの重要な図だけを root か `figures/`（将来作成する場合）にコピーする
- root 直下の図は「最終版」扱いで、頻繁には増やさない

---

## 9. ログ・一時ファイル

root 直下の `*.log` は、主に長期実行やサーバー別の記録:

- `frontale02_tight_decay.log`
- `frontale03_hobic.log`
- `frontale_improved.log`
- `improved_run.log`
- `improved_estimation*.log`
- `dysbiotic_static_1000.log`
- `test_100.log`
- `rapid_test_frontale04_v*.log` など

方針:

- サーバー別・条件別の **代表的なログ** は残してよいが、
  同種の古いログは適宜削除する
- sweep 専用のログは `_runs/sweep_xxx/*.log` にまとまるようにして、
  root 直下には増やさない
- サーバー全体の状態スナップショットは `server.md` に残す

---

## 10. 命名ルール（新規追加時）

新しい run や sweep を追加するときの推奨ルール:

1. **run ID**  
   - 条件 + 日付を含める:  
     `Commensal_Static_20260205_034318` など
   - sweep の場合は:  
     `_runs/sweep_pg_YYYYMMDD_HHMMSS/<tag>/`

2. **ログファイル**  
   - run ID と同じプレフィックスを用いる:  
     `_runs/sweep_pg_.../<tag>.log`

3. **サマリ**  
   - sweep 共通のサマリ: `_runs/sweep_pg_.../sweep_summary.csv`
   - プロジェクト全体のサマリ: `results/*.csv`, `results/*.md`

4. **図**  
   - run 固有: `_runs/<run_id>/figures/*.png`
   - 採用図: `figures/`（将来作る場合）または root にコピー

---

## 11. クリーンアップの優先度

ディスク使用量が増えたときの削除優先度の目安:

1. `_runs/` 内の不完全な run  
   - `cleanup_runs.py` で自動判定（`fit_metrics.json` がない run など）
2. `_runs/sweep_*` のうち、既にサマリを `results/` に反映済みのもの
3. 古い `*.log`（特に日付の古いもの）
4. 中間的な図（各 run の `figures/` 内）、再生成可能なもの

削除前に必要であれば:

- ベストランの run ID と主要な指標（RMSE, log evidence など）を `results/` にメモ
- 採用する図を root or `figures/` 側にコピー

---

## 12. 新しい実験を追加する際のテンプレ

1. `run_new_experiment.sh` を `data_5species/` 直下に作成
2. その中で `_runs/<prefix>/` を作り、run をそこに格納
3. 必要なら `main/` 以下に Python スクリプトを追加し、run ディレクトリを引数で受け取る
4. 結果のサマリを `results/` 以下に保存
5. 不要な run は `cleanup_runs.py` で整理

このファイルは、実際の運用に合わせて随時アップデートしてよい。  
ルールを変更したくなった場合も、まずここに追記してから `_runs` やログを整理する。

