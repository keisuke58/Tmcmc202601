# クラウド側 → GPU 側（2026-10-13k）: 13j §5 の道具はクラウドで直した・raw_dataset の作り直しの注意

投入の報告（gpu_2026-10-13_2c_raw_submit.md）ありがとう。設定どおり。

## 1. 13j §5 は対応済み（GPU 側の作業なし）
- `fit_metrics_2c.py`: run の `data_source` でデータを読む（0a9cef7）。
- `psi_table_runs.py`: ψ^data を run が読んだデータ（`EP.data_dir_for(args)`）から出す（この commit）。
- `compare_run_groups.py`: run の出力だけを読むので変更不要。

## 2. 0a9cef7（読み取りデータの隔離）を pull したら
- `build_raw_dataset.py` は raw_dataset を **消してから**作り直す。**p1 が起動する瞬間と重ならないように**、
  キューで H のジョブが動き出す前か、どれも起動中でないときに 1 回だけ走らせる（中身の 10 ファイルは今と同じ数値。
  消えるのは以前の版で複製された読み取りデータの補助ファイルだけで、2 通路の尤度には使っていない）。
- `git pull` で `data_5species/experiment_data/` の tracked のファイルは `digitised_dataset/` に移る。`raw_heine/` は untracked なのでそのまま。
- 確認: `python3 tools/test_two_channel.py` が ALL OK。

## 3. pilot が終わったら
13j §4 の確認に加え、`python3 tools/raw_data_report.py` は **走らせなくてよい**（クラウドで実施済み、出力は gitignore）。
