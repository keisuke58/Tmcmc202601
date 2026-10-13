# GPU 側 → クラウド側（2026-10-14）: raw 2c の p1 判定（CH・CS・DH PASS）と polish の投入。p2 は polish の後（ユーザー決定）

## 1. p1 の判定（`check_paper_runs.py --glob '*_p1_2c_raw_seed*'`、粒子の最良値）

| 条件 | 判定 | maxlogL 幅 | rmse | 端の印 |
|---|---|---|---|---|
| CH | 全 PASS | 0.11 | 0.097–0.100 | なし |
| CS | 全 PASS | 0.23 | 0.114–0.116 | なし |
| DH | 全 PASS | 0.64 | 0.084–0.092 | a45 上側 0.27–0.28（3 seed、箱上限 6）、a35 下側 0.21（seed 42） |
| DS | p1 実行中（seed 7 は完了、42・123 は 3481・3485） | — | — | — |

- DH の a35（idx 18）・a45（idx 19）は 13o の DH の p2 の箱ですでに広がっている（`18:-25:20`、`19:-5:20`）→ 箱の追加なし。
- pilot は DH・DS で判定 2b FAIL だったが、DH の p1 では PASS。

## 2. polish（13e: 判定 3 は磨いた max logL）
`gpu_2026-10-14_2c_raw_p1_status.md` §2 では「判定 3 が PASS なら polish は省く」としていたが、
**ユーザー決定で polish を回してから判定し直し、p2 はその後**にする。

`tools/polish_max_logL_job.sh`、seed ごと、`TOP_K=10 WRITE_RUN=1`、walltime 16h。

| 条件 | seed 42 | seed 7 | seed 123 | ノード |
|---|---|---|---|---|
| CH | 3496 | 3497 | 3498 | vancouver01 / 01 / 02 |
| CS | 3499 | 3500 | 3501 | vancouver02 / 03 / 03 |
| DH | 3502 | 3503 | 3504 | stuttgart02 ×3 |
| DS | — | 3505 | — | stuttgart03 |

- 生ログで `JAX devices: [CudaDevice]` と `σ_viab`（CH 0.095・DH 0.075・DS 0.059、13m の値）を確認済み。
- DS の seed 42・123 は p1 が終わってから同じ設定で polish を投げる。

## 3. これから
- polish がそろったら `check_paper_runs.py` で判定し直す → PASS の群は `gpu_2026-10-14_2c_raw_p1_status.md` §2 の表どおり p2
  （CH・DH は 13o、CS は 13p の箱、CH の cold p2 も）。
