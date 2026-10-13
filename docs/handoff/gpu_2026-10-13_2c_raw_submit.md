# GPU 側 → クラウド側（2026-10-13）: 13j 実施。digitised 2c を全部止め、raw の 2 通路 pilot 12 本＋p1 12 本を投入

## 1. 止めた（13j §1 と追記 2）
- 先に gate・通知を qdel（3455・3458・3463・3464 → 3454・3456・3457・3459）。どれも C／消滅で、起動していないことを確認してから上流を消した
- p2: DS 3460-3462、DH 3448-3450、CH 3445-3447、CS 3451-3453。polish: 3465-3467
  （3449・3452・3461・3462・3466 は 1 回目に「Server could not connect to MOM」で残り、再度の qdel で消えた）
- p2 の出力ディレクトリはどれも作られていなかった（estimator は終了時に書く）ので、`_killed` を付けるものはない。
  polish の出力（polish.json）も p1 の run には書かれていない

## 2. 生データの配置（13i・13j 追記）
- `b2751e9` から xlsx 2 つを取り出し → `convert_raw_heine.py`（qpcr_long 2955・livedead_long 670・volume_long 715 行）
  → `build_raw_dataset.py`（replicates 1470 / 1485・membrane 24・boxplot 24）。数はクラウドと一致
- 追記（55b972b）の版で `build_raw_dataset.py` を走らせ直し、`boxplot_<条件>_<培養>.csv` 4 つも生データの版になった
- `git status` に raw_heine・raw_dataset は出ない

## 3. 投入（git deeaff4、`DATA_SOURCE=raw TWO_CHANNEL=1 RUNTAG=2c_raw`）
設定は 13b の表どおり（digitised の `*_2c_seed42` の run_record.json と粒子数・mutation・OVERRIDE が一致することを確認）。
p1 は `-W depend=afterok:<pilot>`、PREV=`_runs/paper_gateoff/<TAG>_pilot_2c_raw_seed<SEED>`。

| 条件 | seed | ノード | pilot | p1 | pilot 設定 | p1 設定 | walltime (pilot / p1) |
|---|---|---|---|---|---|---|---|
| CH | 42 | vancouver01 | 3468 | 3469 | 1000×40 | 2000×150 | 10h / 14h |
| CH | 7 | vancouver01 | 3470 | 3471 | 〃 | 〃 | 〃 |
| CH | 123 | vancouver02 | 3472 | 3473 | 〃 | 〃 | 〃 |
| CS | 42 | vancouver02 | 3474 | 3475 | 1000×40 | 2000×120 | 10h / 14h |
| CS | 7 | vancouver03 | 3476 | 3477 | 〃 | 〃 | 〃 |
| CS | 123 | vancouver03 | 3478 | 3479 | 〃 | 〃 | 〃 |
| DS | 42 | stuttgart01 | 3480 | 3481 | 1000×40、`5:-15:20;6:-15:20;19:-15:20` | 3000×100、同じ OVERRIDE | 10h / 20h |
| DS | 7 | stuttgart01 | 3482 | 3483 | 〃 | 〃 | 〃 |
| DS | 123 | stuttgart01 | 3484 | 3485 | 〃 | 〃 | 〃 |
| DH | 42 | stuttgart02 | 3486 | 3487 | 1000×40 | 2000×80 | 10h / 16h |
| DH | 7 | stuttgart02 | 3488 | 3489 | 〃 | 〃 | 〃 |
| DH | 123 | stuttgart02 | 3490 | 3491 | 〃 | 〃 | 〃 |

- pilot 12 本は全部 R（4090 の空き 6 枚を CH・CS に）。p1 12 本は H（依存待ち）。`qstat -f` の Variable_List で 3480/3481 を確認済み

## 4. これから
- 最初の pilot が終わったら 13j §4 を確認（`data_source: "raw"`・`two_channel: true`、読み込み先が raw_dataset）
- 13j §5: `psi_table_runs.py`・`compare_run_groups.py`・`fit_metrics_2c.py` に `EP.data_dir_for(rec["args"])`（未着手）
- p1 → p2 は判定を見てから（13b どおり）
