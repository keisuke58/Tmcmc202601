# GPU 側報告 2026-10-12（12g の予備 6 本＋DH gate を投入した）

3090 が空いていたので（他ユーザーは放置 jupyterlab の HELD-IDLE だけ、`gpualloc` で確認）、
12g の `afterany:3388:3389:3390` は付けずに**すぐ投入した**。本数は計算 9 ＋ 予備 6 = 15（上限ちょうど、gate・通知は数えない）。
本線（DS ult 3384-3386、DH p2 w4 3395-3397）はそのまま。

| jobid | 中身 | seed | ノード | walltime | 状態 |
|---|---|---|---|---|---|
| 3400 | DS ult `wide80_p18k_wide2_sd4_p18k_m250`（18000 粒子・N_MUT 250） | 42 | stuttgart02 | 72h | R |
| 3401 | 同 | 7 | stuttgart02 | 72h | R |
| 3402 | 同 | 123 | stuttgart02 | 72h | R |
| 3403 | DH p2 `nonarrow_w4_noph_p25k`（25000 粒子・N_MUT 120） | 42 | stuttgart03 | 96h | R |
| 3404 | 同 | 7 | stuttgart03 | 96h | R |
| 3405 | 同 | 123 | stuttgart03 | 96h | R |
| 3406 | gate（`afterany:3403:3404:3405`、ult `nonarrow_w4_noph_p25k_sd4`・N_PART 25000・N_MUT 150・vancouver01・72h） | — | — | — | H |

- OVERRIDE・PREV は 12g のとおり（`qstat -f` の Variable_List を全文で確認済み）。
- **walltime は 12g の指示（DS 48h・DH 72h）より長くした。** 4090 の実測から換算すると:
  - DS p2 p18k（18000×150）は vancouver で 17.6h → ult は N_MUT 250 で約 ×1.7 = 29h。3090 は 1.5〜2 倍遅いので 44〜59h → 48h では足りない恐れがあり 72h。
  - DH p2 p14k（14000×120）は vancouver で 16.6h → 25000 粒子で約 30h、3090 で 45〜60h → 96h。
  - 走り出したら延ばせないので長めに取った。queue に walltime 上限は無い（`qmgr`）。
- DH gate の ult は vancouver01 に投げる（その頃には本線 3395/3396 が終わっている見込み）。
