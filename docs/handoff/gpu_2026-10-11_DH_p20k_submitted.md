# GPU 側報告 2026-10-11（11l: DH p2 p20k と gate、p14k からの仮の ult を投入した）

11l と追記のとおり投入した。Variable_List は全文確認済み（RUNTAG・OVERRIDE 末尾の `7:-10:5` まで・PREV）。

## DH p2 `nonarrow_w3_noph_p20k`（本線）

`STAGE=p2 TAG=DH N_PART=20000 N_MUT=120 RUNTAG=nonarrow_w3_noph_p20k`、OVERRIDE は 11l と同じ、`PREV=_runs/paper_gateoff/DH_p1_mut80_seed$S`、walltime 48h。

| jobid | seed | ノード | 状態（投入時） |
|---|---|---|---|
| 3375 | 42 | vancouver01 | R |
| 3376 | 123 | vancouver01 | R |
| 3377 | 7 | vancouver03 | Q（4090 の空きが vancouver01 の 2 枚だけだった。fit_ceiling 3369 が終われば動く。完了は他の 2 本より約 2h 遅れる見込み） |
| 3378 | gate_ult（09c 版、MODES なし） | — | H（`afterany:3375:3376:3377`） |

gate が PASS なら ult は `N_MUT=150 RUNTAG=nonarrow_w3_noph_p20k_sd4`、OVERRIDE 同じ、vancouver01・walltime 12h で出る。
DS ult（3371-3373）とは枠が重なっていない。

## DH ult `nonarrow_w3_noph_p14k_sd4_prov`（仮、11l 追記）

`STAGE=ult TAG=DH N_MUT=150 RUNTAG=nonarrow_w3_noph_p14k_sd4_prov`、OVERRIDE 同じ、`PREV=_runs/paper_gateoff/DH_p2_nonarrow_w3_noph_p14k_seed$S`、walltime 12h。
優先順位（DS ult → DH p20k → prov）に従い、4090 は使わず stuttgart（3090）に置いた。

| jobid | seed | ノード | 状態（投入時） |
|---|---|---|---|
| 3379 | 42 | stuttgart01 | R |
| 3380 | 123 | stuttgart02 | R |
| 3381 | 7 | stuttgart03 | R |
| 3382 | notify_cloud（`DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed*`） | — | H（`afterany:3379:3380:3381`） |

- 3090 で 12h に収まるかは未確認（実行中の walltime は延ばせない）。足りなければ報告して投げ直す。
- 終わったら判定を流し、`final_runs/` に `_prov` のまま上げる・仮の図と `final_theta_MAP/DH.json`（README に「仮」）は未着手。

## 本数

自分の計算ジョブ 10 本（fit_ceiling 3369・DS ult 3371-3373・DH p20k 3375-3377・DH prov 3379-3381）＋ gate 3378 ＋ 通知 3374・3382。上限 15 本以内。
