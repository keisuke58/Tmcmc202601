# GPU 側 → クラウド側（2026-10-13）: 13b 実施。古い尤度のジョブを止め、2 通路の pilot 12 本＋p1 12 本を投入

## 1. 止めた
- gate 3407・3408 を先に qdel（消えたことを確認してから上流を消した）
- 3395-3397（DH 本線 p2 w4 p20k）、3400-3402（DS 予備 ult p18k m250）、3403-3405（DH 予備 p2 p25k）、
  3409-3412・3419・3420（回収テスト）
- 出力ディレクトリは**どれも作られていなかった**（estimator は終了時に書く）。`_killed` を付ける対象なし。
  ログは `data_5species/main/logs/pbs/{paper_gateoff,recovery}_<jobid>.copaam.log` に残っている

## 2. 投入（git 1163398、`TWO_CHANNEL=1`・`RUNTAG=2c`、`LAMBDA_CH2_2C` 既定 1.0）
p1 は `-W depend=afterok:<pilot>`、PREV=`_runs/paper_gateoff/<TAG>_pilot_2c_seed<SEED>`。

| 条件 | seed | ノード | pilot | p1 | pilot 設定 | p1 設定 | walltime (pilot / p1) |
|---|---|---|---|---|---|---|---|
| CH | 42 | vancouver01 | 3421 | 3422 | 1000×40 | 2000×150 | 10h / 14h |
| CH | 7 | vancouver01 | 3423 | 3424 | 〃 | 〃 | 〃 |
| CH | 123 | vancouver02 | 3425 | 3426 | 〃 | 〃 | 〃 |
| CS | 42 | vancouver02 | 3427 | 3428 | 1000×40 | 2000×120 | 10h / 14h |
| CS | 7 | vancouver03 | 3429 | 3430 | 〃 | 〃 | 〃 |
| CS | 123 | vancouver03 | 3431 | 3432 | 〃 | 〃 | 〃 |
| DS | 42 | stuttgart01 | 3433 | 3434 | 1000×40、`5:-15:20;6:-15:20;19:-15:20` | 3000×100、同じ OVERRIDE | 10h / 20h |
| DS | 7 | stuttgart01 | 3435 | 3436 | 〃 | 〃 | 〃 |
| DS | 123 | stuttgart01 | 3437 | 3438 | 〃 | 〃 | 〃 |
| DH | 42 | stuttgart02 | 3439 | 3440 | 1000×40 | 2000×80 | 10h / 16h |
| DH | 7 | stuttgart02 | 3441 | 3442 | 〃 | 〃 | 〃 |
| DH | 123 | stuttgart02 | 3443 | 3444 | 〃 | 〃 | 〃 |

- pilot 12 本は全部 R（04:24 JST 開始）。p1 12 本は H（依存待ち）。`qstat -f` の Variable_List で 3433/3434 を確認済み
- walltime は過去の実測（pilot 2.5〜4.5 h、p1 4〜5 h）に余裕を持たせた
- `tools/test_two_channel.py` は GPU 側でも ALL OK

## 3. これから
- 最初の pilot が終わったら `run_record.json`（`two_channel: true`、`lambda_ch_effective[1]=1.0`、`psi_fixed` (5,5)）を確認して報告
- p1 → p2 は判定を見てから（13b どおり）
