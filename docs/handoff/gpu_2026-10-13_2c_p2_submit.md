# GPU 側 → クラウド側（2026-10-13）: 13b の続き。2 通路の p1 判定と p2 12 本・gate 4 本の投入

## 1. p1（2c）の判定（`check_paper_runs.py`、粒子の最良値。p1 の polish は回していない）
| 群 | 判定 | 箱で切られている印 |
|---|---|---|
| CH_p1_2c | 全 PASS（maxlogL 幅 0.30） | なし |
| CS_p1_2c | 全 PASS（幅 0.02） | なし |
| DH_p1_2c | 全 PASS（幅 0.62） | なし |
| DS_p1_2c | **判定 4 FAIL**（a33 中央値幅/sd 1.69、a45 0.64、a23 0.61）、幅 0.53 | **a44（idx 7）seed 42**: 論文の箱の下限 −1 に 19%（seed 7 も 15%） |

- pilot 4 群の 2b FAIL は想定内（13b）。
- 13e の「p1 → p2 の前に polish」は省いた（判定 3 は粒子の最良値で PASS）。p2 の gate では磨いてから判定する。

## 2. p2 の投入（`TWO_CHANNEL=1`、`RUNTAG=2c`、PREV=`<TAG>_p1_2c_seed<S>`、粒子数・mutation・OVERRIDE は 13b の表）
| 条件 | seed 42 / 7 / 123 | 設定 | ノード | walltime |
|---|---|---|---|---|
| CH | 3445 / 3446 / 3447 | 2000×150、`1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5` | vancouver03, vancouver03, vancouver02 | 16h |
| DH | 3448 / 3449 / 3450 | 20000×120、13b の OVERRIDE（12 成分） | vancouver01, vancouver01, vancouver02 | 48h |
| CS | 3451 / 3452 / 3453 | 2000×160、`5:-15:20;0:-5:2;10:-1:20` | stuttgart03 ×3 | 16h |
| DS | 3460 / 3461 / 3462 | 18000×150、13b の OVERRIDE **＋ `7:-5:2.5`**（a44 の下限 −1 → −5、ユーザー承認） | stuttgart02 ×3 | 40h |

- DS は p1 が判定 4 FAIL だが、ユーザー判断で p2 に進めた（古い尤度でも DS は p2 で粒子を増やして詰めた）。
- `qstat -f` の Variable_List で全ジョブの STAGE・TAG・SEED・RUNTAG・N_PART・N_MUT・OVERRIDE・PREV を確認済み。

## 3. gate（p2 → ult、`gpus=1`、判定前に polish）と通知
| 条件 | gate | gate のノード / walltime | ult | 通知（p2） |
|---|---|---|---|---|
| CH | 3454 | vancouver03 / 4h | 5000×150、`vancouver03;vancouver03;vancouver02`、14h | 3457 |
| DH | 3455 | vancouver01 / 8h | N_PART 20000×150、`vancouver01;vancouver01;vancouver02`、30h | 3458 |
| CS | 3456 | stuttgart03 / 4h | 5000×150、stuttgart03 ×3、18h | 3459 |
| DS | 3463 | stuttgart02 / **24h**（DS の polish は 3 run で 8h を超えた実績あり） | N_PART 18000×150、stuttgart02 ×3、36h | 3464 |

- gate はすべて `TWO_CHANNEL=1`、`PREV_RUNTAG=2c`、`ULT_RUNTAG=2c`、OVERRIDE は p2 と同じ。`MODES` は空（2 通路で多峰になる成分はまだ不明）。
- 13b §4 の ψ_i と ψ_i^data の表は、p2 が出たらこちらで付ける。

## 4. 13f・13g
13b の本線のあとなので、まだ投入していない。
