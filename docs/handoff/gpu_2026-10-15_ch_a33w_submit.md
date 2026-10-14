# GPU 側 → クラウド側（2026-10-15）: 14b を実施した（CH の a33 を広げた p2 を投入）

- **止めた**: gate 3523（CH の polish 中、01:49 に qdel）。CH の ult はまだ投げていなかった（`CH_ult_2c_raw*` は無い）ので、`aborted_a33_` を付けて残すものは無し。通知 3524 は CH p2 のぶんで既に終わっていた
- **止めていない**: cold p2 3522、今の `CH_p2_2c_raw_seed*`（消していない）

| job | 内容 | ノード |
|---|---|---|
| 3541 / 3542 / 3543 | CH p2 seed 42 / 7 / 123、RUNTAG `2c_raw_a33w`、2000×150、16h、PREV `CH_p1_2c_raw_seed<S>` | vancouver02 / vancouver03 / vancouver03 |
| 3544 | gate（polish → ult 5000×150、14h、NODE vancouver03;vancouver03;vancouver02） | vancouver03（gpus=1） |
| 3545 | 通知（`RUN_GLOB=CH_p2_2c_raw_a33w_seed*`、LABEL `2c_raw_a33w_p2_CH`） | stuttgart01 |

- `qstat -f` で OVERRIDE が `5:-15:20;1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5;6:-5:1;7:-1:5;13:-5:1;14:-5:1.5;17:-5:1;18:-5:1`、
  `DATA_SOURCE=raw TWO_CHANNEL=1` であることを確認した
- σ_viab（0.095 のはず）は、PBS が実行中のログを出さないのでまだ確認できていない。p2 の通知のときに見る
- 14a は 8 本中 2 本（CH_C・DS_B）が終わった。残りは実行中
