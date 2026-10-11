# GPU 側報告 2026-10-11（11l: DH p2 p20k＋gate と、p14k からの仮の ult を投入した）

## 結論

- **本線**: DH p2 `nonarrow_w3_noph_p20k` を 3 本（3375-3377）と gate（3378）を投入した。
- **仮**: DH ult `nonarrow_w3_noph_p14k_sd4_prov` を 3 本（3379-3381）と通知（3382）を投入した。
- Variable_List は全文確認済み（RUNTAG・OVERRIDE 末尾の `7:-10:5` まで・PREV）。OVERRIDE はどれも 11l と同じ。

## 1. DH p2 `nonarrow_w3_noph_p20k`（本線）

```
STAGE=p2 TAG=DH SEED=$S N_PART=20000 N_MUT=120 RUNTAG=nonarrow_w3_noph_p20k
PREV=_runs/paper_gateoff/DH_p1_mut80_seed$S   walltime 48h
```

| jobid | 内容 | ノード | 状態（投入時） |
|---|---|---|---|
| 3375 | seed 42 | vancouver01 | R |
| 3376 | seed 123 | vancouver01 | R |
| 3377 | seed 7 | vancouver03 | Q |
| 3378 | gate_ult（09c 版、MODES なし） | — | H（`afterany:3375:3376:3377`） |

- 3377 が Q なのは、4090 の空きが vancouver01 の 2 枚だけだったため。fit_ceiling 3369 が終われば動く。完了は他の 2 本より約 2h 遅れる見込み。
- gate が PASS なら ult は `N_MUT=150 RUNTAG=nonarrow_w3_noph_p20k_sd4`、vancouver01・walltime 12h で出る。
- DS ult（3371-3373）とは枠が重なっていない。

## 2. DH ult `nonarrow_w3_noph_p14k_sd4_prov`（仮、11l 追記）

```
STAGE=ult TAG=DH SEED=$S N_MUT=150 RUNTAG=nonarrow_w3_noph_p14k_sd4_prov
PREV=_runs/paper_gateoff/DH_p2_nonarrow_w3_noph_p14k_seed$S   walltime 12h
```

| jobid | 内容 | ノード | 状態（投入時） |
|---|---|---|---|
| 3379 | seed 42 | stuttgart01 | R |
| 3380 | seed 123 | stuttgart02 | R |
| 3381 | seed 7 | stuttgart03 | R |
| 3382 | notify_cloud（`DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed*`） | — | H（`afterany:3379:3380:3381`） |

- 優先順位（DS ult → DH p20k → prov）に従い、4090 は使わず stuttgart（3090）に置いた。
- **3090 で 12h に収まるかは未確認**（実行中の walltime は延ばせない）。足りなければ報告して投げ直す。

## 3. 未着手（prov の ult が終わってから）

- 判定を流して報告
- `final_runs/` に `_prov` の名前のまま上げる（README に「仮・判定 4 FAIL の p2 から」）
- 仮の図（DH の glob を `DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed*`）と `final_theta_MAP/DH.json`（状態欄に「仮」）

## 4. 本数

| 種類 | jobid | 本数 |
|---|---|---|
| 計算 | fit_ceiling 3369・DS ult 3371-3373・DH p20k 3375-3377・DH prov 3379-3381 | 10 |
| gate | 3378 | 1 |
| 通知 | 3374・3382 | 2 |

上限 15 本以内。
