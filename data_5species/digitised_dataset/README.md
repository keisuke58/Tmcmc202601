# digitised_dataset — 論文の図から読み取ったデータ（**新しい推定には使わない**）

2026-10-13、ユーザー指示で `data_5species/experiment_data/` からここへ隔離した。
中身は Heine et al. の図（box plot・棒グラフ）から読み取った値と、そこから作った合成の「反復」（`generate_replicates`）。
生データ（MHH、Heine 提供）と比べると、dysbiotic の組成で最大 42 ポイントずれる（DH の生きた菌の Vei など）。

- **新しい推定は生データ**（`data_5species/raw_dataset/`、`tools/build_raw_dataset.py` で作る。estimator の既定 `--data-source raw`）。
- ここは **過去の run（BMB 版・2026-10-13 までの EML の run）を再現するときだけ**使う。
  `run_record.json` の args に `data_source` が無い run は、`EP.data_dir_for()` がここを返す。
- 消さない（過去の結果の記録）。
