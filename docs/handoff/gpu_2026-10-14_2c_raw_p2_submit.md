# GPU 側 → クラウド側（2026-10-14）: raw の p1 判定と p2 12 本・cold p2 1 本・gate 4 本の投入（13o・13p・13l）

## 1. p1（2c_raw）の判定（`check_paper_runs.py`、粒子の最良値。p1 の polish.json は run に書かれていない）
| 群 | 判定 | maxlogL 幅 | 箱の端の印 |
|---|---|---|---|
| CH_p1_2c_raw | 全 PASS | 0.11 | なし |
| CS_p1_2c_raw | 全 PASS | 0.23 | なし |
| DH_p1_2c_raw | 全 PASS | 0.64 | なし |
| DS_p1_2c_raw | 全 PASS | 0.79 | なし |

- 端の印は出なかったので、13o・13p の OVERRIDE に足したものはない。
- 13m の作り直しは済み（`fig2_membrane_distribution.csv` の Day 3–21 の error_pct 平均: CS 11.3・CH 9.5・DS 5.86・DH 7.51）。
  p2 のログの `σ_viab` は PBS が stdout を溜めるので、終了後に確認する。

## 2. p2 の投入（`DATA_SOURCE=raw TWO_CHANNEL=1 RUNTAG=2c_raw`、PREV=`<TAG>_p1_2c_raw_seed<S>`）
粒子数・mutation・walltime は 13b、OVERRIDE は 13o（CH・DS・DH）・13p（CS）。

| 条件 | seed 42 / 7 / 123 | 設定 | ノード | walltime |
|---|---|---|---|---|
| DH | 3510 / 3511 / 3512 | 20000×120、`12:-15:20;5:-15:20;6:-15:20;18:-25:20;13:-15:20;11:-8:3;19:-5:20;2:-10:5;7:-10:5;10:-8:3;0:-10:5;1:-10:3;17:-5:2.5` | vancouver01, vancouver01, vancouver02 | 48h |
| CH | 3513 / 3514 / 3515 | 2000×150、`1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5;6:-5:1;7:-1:5;13:-5:1;14:-5:1.5;17:-5:1;18:-5:1` | vancouver02, vancouver03, vancouver03 | 16h |
| CS | 3516 / 3517 / 3518 | 2000×160、`5:-15:20;10:-1:20;0:-10:2;6:-5:1;11:-5:1;12:-5:3;13:-5:1;16:-5:1;17:-5:1;18:-5:1`（LAMBDA_CH3 は渡さない） | stuttgart03 ×3 | 16h |
| DS | 3519 / 3520 / 3521 | 18000×150、`5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5;7:-5:2.5;10:-5:1.5` | stuttgart02 ×3 | 40h |

- cold p2（13l）: 3522、CH seed 42、`P2_COLD=1 RUNTAG=2c_raw_cold`、CH と同じ OVERRIDE、2000×150、stuttgart01、16h。gate・ult なし。
- `qstat -f` の Variable_List で全ジョブの STAGE・TAG・SEED・RUNTAG・N_PART・N_MUT・OVERRIDE・PREV・P2_COLD を確認済み。13 本とも R。

## 3. gate（p2 → ult、`gpus=1`、判定前に polish）と通知
| 条件 | gate | gate のノード / walltime | ult | 通知（p2） |
|---|---|---|---|---|
| CH | 3523 | vancouver03 / 4h | 5000×150、`vancouver03;vancouver03;vancouver02`、14h | 3524 |
| DH | 3525 | vancouver01 / 8h | N_PART 20000×150、`vancouver01;vancouver01;vancouver02`、30h | 3526 |
| CS | 3527 | stuttgart03 / 4h | 5000×150、stuttgart03 ×3、18h | 3528 |
| DS | 3529 | stuttgart02 / 24h | N_PART 18000×150、stuttgart02 ×3、36h | 3530 |

- gate はすべて `TWO_CHANNEL=1 DATA_SOURCE=raw PREV_RUNTAG=2c_raw ULT_RUNTAG=2c_raw`、OVERRIDE は p2 と同じ、`MODES` は空。
- cold p2 の通知: 3531（`LABEL=2c_raw_p2_CH_cold`）。warm の CH seed 42 と `compare_run_groups.py` で比べて `gpu_2026-10-13_p2_cold.md` 相当を push する。
- 箱を広げたので段数が増えて walltime が足りない可能性がある。足りなければ報告する。
