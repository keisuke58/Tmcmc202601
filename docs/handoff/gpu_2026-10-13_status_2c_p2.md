# GPU 側 → クラウド側（2026-10-13）: 2 通路の p2 投入、p1 の polish、ψ の表、13f の準備（中間報告）

投入の詳細は [gpu_2026-10-13_2c_p2_submit.md](gpu_2026-10-13_2c_p2_submit.md)。ここは全体のまとめ。

## 1. 13b: p1 → p2（実施）
- 2c の pilot 12 本・p1 12 本はすべて完走。p1 の判定: CH・CS・DH は全 PASS。**DS は判定 4 FAIL**（a33 の中央値幅/sd 1.69）で、a44 が論文の箱の下限 −1 で切られていた（seed 42 で 19%）。
- p2 12 本を投入、すべて R（18 時頃 JST 開始）。gate 4 本（`gpus=1`、判定前に polish、`TWO_CHANNEL=1`）と通知 4 本は H。

| 条件 | p2 | 設定 | gate → ult | 通知 |
|---|---|---|---|---|
| CH | 3445-3447（vancouver） | 2000×150 | 3454 → 5000×150 | 3457 |
| CS | 3451-3453（stuttgart03） | 2000×160 | 3456 → 5000×150 | 3459 |
| DH | 3448-3450（vancouver） | 20000×120 | 3455 → 20000×150 | 3458 |
| DS | 3460-3462（stuttgart02） | 18000×150、OVERRIDE に **`7:-5:2.5`（a44 の下限 −1 → −5）を追加** | 3463 → 18000×150 | 3464 |

- **DS を FAIL のまま p2 に進めたのはユーザー判断**（古い尤度でも DS は p2 で粒子を増やして詰めた）。a44 を広げたのもユーザー承認。
- 見込み: CH・CS の p2 は 5〜8 h、DH・DS は 20 h 前後。

## 2. 13e: p1 の polish（実施中）
- p1 → p2 は粒子の最良値の判定 3（PASS）で進めた。13e の「先に polish」は後追いで回している:
  3465（CH+CS）・3466（DH）・3467（DS、walltime 24h）、`TOP_K=10`、`--write-run`。
- `tools/polish_max_logL_job.sh` に `WRITE_RUN=1`（`--write-run` を渡す）を追加した（12e0029）。
- 結果は各ジョブが `gpu_2026-10-13_polish_p1_2c_*.md` を push する。polish 後に判定 3 が FAIL になった群があれば、その p2 をどうするか相談する。

## 3. 13b §4: ψ_i と ψ_i^data の表（道具を追加、9e73cee）
- `tools/psi_table_runs.py`: 群ごとに ψ^data（`psi_from_data.py` と同じ式）、MAP（max logL の seed）、事後（3 seed の粒子 200 本）の Day 3–21 の中央値。モデルの ψ は 2 通路の尤度と同じ `g[:, 6:11]` を観測日で読む。ψ 固定の段（pilot・p1）には印を付ける。
- `tools/notify_cloud_runs.sh`: RUN_GLOB に `_2c` を含むときは、この表を通知に入れる。待っている p2 の通知と、gate が出す ult の通知に入る。
- 試走（CH_p1_2c、ψ 固定の段なのでモデル側は参考値）: data は So 0.42・An 0.37・Vei 0.39・Fn 0.04・Pg 0.05。

## 4. 13f の準備（未投入、a8f544a）
- `tools/submit_13f.sh`（`DRY_RUN=1` 可）:
  - `PART=box TAG=CH|CS NODES=...`: 2c の p2（seed 42）の箱を `widen_box.py` で 1.5 倍にして p2 を 3 seed（RUNTAG `2c_box15`）。p2 が無いうちは止まる。
  - `PART=holdout DAY=10|15 NODE=...`: CH seed 42 で pilot → p1 → p2 を afterok で連鎖（RUNTAG `2c_ho10` / `2c_ho15`）。
- `tools/compare_run_groups.py`: 2 群の 15 成分の対照表（id / s.o. / n.i. は自分の箱で、S4 と同じ規則、中央値と 90% 区間、分類・符号が同じ成分の数）。13g の対照表にも使える。
- いまは GPU が上限の 15 本なので投入していない。holdout は CH・CS の本線が通ってからの予定（13f どおり）。早めるならユーザーが決める。

## 5. 13g
13f のあと。未着手。
