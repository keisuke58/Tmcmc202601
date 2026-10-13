# GPU 側 → クラウド側（2026-10-14）: raw 2c の p1 実行中。13o を受けて p1 判定 PASS なら p2 を投入する計画

## 1. いまの状態
- pilot 12 本は完了（報告は `gpu_2026-10-13_2c_raw_pilot_*.md`）。4 群とも判定 0・1・2・3・4 は PASS、2b は FAIL（2.7〜4.2/次元、pilot では想定内）。
- p1 12 本が実行中（すべて R、`DATA_SOURCE=raw TWO_CHANNEL=1 RUNTAG=2c_raw`、Variable_List を全文で確認済み）。

| 条件 | job | ノード | walltime |
|---|---|---|---|
| CH | 3469 / 3471 / 3473 | vancouver01, vancouver01, vancouver02 | 14h |
| CS | 3475 / 3477 / 3479 | vancouver02, vancouver03, vancouver03 | 14h |
| DS | 3481 / 3483 / 3485 | stuttgart01 ×3 | 20h |
| DH | 3487 / 3489 / 3491 | stuttgart02 ×3 | 16h |

- 13m の作り直しは済んでいる: `fig2_membrane_distribution.csv` の Day 3–21 の error_pct の平均は CS 11.3・CH 9.5・DS 5.9・DH 7.5。

## 2. p1 が終わったら（ユーザー指示: 判定が PASS なら p2 まで投入）
- `check_paper_runs.py --glob '<TAG>_p1_2c_raw_seed*'` で判定する。判定 3 が粒子の最良値で PASS なら、p1 の polish は省く（13b と同じ）。FAIL なら GPU で polish してから判定し直す。
- **PASS の場合**: 13o の OVERRIDE で p2 を 3 seed（設定は 13b と同じ）。gate（`gpus=1`、`TWO_CHANNEL=1 DATA_SOURCE=raw`）と通知を付ける。

| 条件 | p2 | walltime | 追加 |
|---|---|---|---|
| CH | 2000×150 | 16h | cold p2 1 本（13l、`P2_COLD=1`、`2c_raw_cold`、seed 42、13o の CH の箱） |
| DH | 20000×120 | 48h | gate に `N_PART=20000` |
| DS | 18000×150 | 40h | gate に `N_PART=18000`（gate の walltime は 24h） |
| CS | 投入しない（13p 待ち） | — | — |

- p1 で 13o 以外の成分に端の印が出たら、13o の規則でその側を広げて OVERRIDE に足す（報告に書く）。
- FAIL を含む群は投入せず報告する。
- 投入後は p2 のログで `σ_viab`（CH 0.095・DS 0.059・DH 0.075）を確認し、投入表を `gpu_2026-10-14_2c_raw_p2_submit.md` で push する。
