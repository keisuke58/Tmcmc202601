# GPU 側報告 2026-10-16（DS p2 を vancouver・72h で出し直した）

## 何が起きたか
- DS p2（3519〜3521、stuttgart02、18000×150）は 3 本とも **walltime 40h で kill**（`job killed: walltime 144027 exceeded limit 144000`）。
  出力ディレクトリ `DS_p2_2c_raw_seed*` は残っていない。ログに stage の行は無い（1 stage も終わらなかったのか、kill で stdout が失われたのかは未確認）。
- 後ろの磨き 3597〜3599 は「対象の run が無い」で空振り、gate 3600 は「run 0/3」で ult を投入しなかった（175ddc5）。
- 見積もり: DS p1（3000 粒子）は 1 stage 約 2700 s・9〜10 stage。p2 は粒子 6 倍で 1 stage 約 4.5h、3090 では 50h 超になる。

## 出し直し（ユーザー決定: 設定は同じ、ノードを 4090 に・walltime を延ばす）
設定は 10/14 の投入（gpu_2026-10-14_2c_raw_p2_submit.md）と同じ: `STAGE=p2 TAG=DS DATA_SOURCE=raw TWO_CHANNEL=1 RUNTAG=2c_raw N_PART=18000 N_MUT=150`、
OVERRIDE `5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5;7:-5:2.5;10:-5:1.5`、PREV=`DS_p1_2c_raw_seed<S>`。

| jobid | 内容 | 依存 | ノード・walltime |
|---|---|---|---|
| 3601 | DS p2 seed 42 | — | vancouver03・72h |
| 3602 | DS p2 seed 7 | — | vancouver03・72h |
| 3603 | DS p2 seed 123 | — | vancouver02・72h |
| 3604 | 磨き DS_p2_2c_raw_seed42（`TOP_K=3 MAXITER=3000 WRITE_RUN=1`） | afterany:3601 | vancouver03・24h |
| 3605 | 磨き DS_p2_2c_raw_seed7 | afterany:3602 | vancouver03・24h |
| 3606 | 磨き DS_p2_2c_raw_seed123 | afterany:3603 | vancouver02・24h |
| 3607 | DS の gate（`POLISH=0`、N_PART 18000×150、同じ OVERRIDE、ult は `vancouver03;vancouver03;vancouver02`・**60h**） | afterany:3604:3605:3606 | stuttgart01・2h |

- 3601〜3603 は Variable_List を確認済みで、3 本とも R。
- ult の walltime は 3600 の 36h から 60h に延ばした（p2 と同じく足りなくなるのを避けるため）。
