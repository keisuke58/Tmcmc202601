# GPU 側報告 2026-10-19（19b: CH の 1 日外し予測を投入）

ユーザー指示で回収テストの終了を待たず、空いている 3090 に投入した（GPU は同時 13 本、15 本の内）。

| 系列 | ノード | pilot | p1 | p2 | 報告（notify） |
|---|---|---|---|---|---|
| Day 10 外し（`RUNTAG=2c_raw_a33w_ho10`） | stuttgart03 | 3667 | 3668 | 3669 | 3670 |
| Day 15 外し（`RUNTAG=2c_raw_a33w_ho15`） | stuttgart02 | 3671 | 3672 | 3673 | 3674 |

- seed 42 のみ。afterok でつないだ。walltime は pilot 10h・p1 14h・p2 16h。
- 投入前に `tools/submit_13f.sh` の holdout を 19b に合わせた（59854d5）: `DATA_SOURCE=raw` を追加、RUNTAG を `2c_raw_a33w_ho<日>`、
  p2 の OVERRIDE を本線 CH p2 a33w と同じ箱に。dry-run で pilot・p1・p2 の引数が本線（`CH_*_2c_raw*_seed42/run_record.json` の argv）と
  `--holdout-day` 以外一致することを確認（pilot 1000×40、p1 2000×150、p2 2000×150）。
- 3090 なので 1 系列 20h 前後の見込み（4090 で約 14h）。終わったら判定（0〜2b）と final_runs へのコピー（p2 は polish.json も）。
