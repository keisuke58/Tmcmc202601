# クラウド側 → GPU 側（2026-10-13f）: 質の確認 2 つ（ユーザー指示「4 と 5」、EML_PLAN §22）

**2 通路の本線（13b）を優先。** どちらも 2 通路の CH・CS の段が通ってから、空いた枠で。

## 4. 箱への依存の確認（CH・CS、各 3 seed）
2 通路の p2 の箱を中心のまわりに **1.5 倍**に広げて p2 を回し直し、同定される成分が変わらないかを見る。
```
OVR=$(python3 tools/widen_box.py _runs/paper_gateoff/CH_p2_<2c の RUNTAG>_seed42 --factor 1.5)
qsub ... -v STAGE=p2,TAG=CH,SEED=$S,TWO_CHANNEL=1,RUNTAG=<2c の RUNTAG>_box15,OVERRIDE="$OVR",PREV=<2c の p1>_seed$S paper_gateoff_job.sh
```
- 粒子数・mutation は 2c の p2 と同じ。判定 (a)–(e)（(d) は磨いた値）も同じく。
- 比べるもの: 同定 / 符号だけ / n.i. の分類（`tools/ident_threshold_table.py` と同じ規則、それぞれ自分の箱で）、中央値と 90% 区間。報告は 2c の p2 との対照表で。

## 5. 予測の確認（leave-one-day-out、CH、seed 42 の 1 系列ずつ）
`--holdout-day`（job では `HOLDOUT_DAY`、今回追加）でその日を尤度から外し、pilot → p1 → p2 を 2 通路で回す。
- **外す日: 10 と 15**（中の日。21 日目を外すとモデル時間の目盛り（day_scale）が変わる別の問題になるので今回はやらない）
- RUNTAG に `_ho10` / `_ho15`。粒子数・箱は 2c の CH と同じ。
- 評価（CPU で数分）:
```
python3 tools/holdout_eval.py data_5species/main/_runs/paper_gateoff \
  --holdout 'CH_p2_<2c>_ho10_seed*' --full 'CH_p2_<2c>_seed*' --day 10 --out docs/revision/generated/holdout/CH_day10.json
```
  外した日の生きた菌・全菌の組成と生きた割合について、事後予測の中央値の RMSE と 90% 区間の被覆を、その日も使った推定（当てはめ）と並べて出す。

## 確認済み（クラウド、CPU）
- `--holdout-day 10 --two-channel --fix-psi` の小さな run（20 粒子）が通り、当てる日 [3, 6, 15, 21]、ψ 固定 (4, 5)、logL 照合 OK。
- `polish_max_logL.py` が holdout の run を組み直せる（logL 再現 OK）。
- `widen_box.py`・`holdout_eval.py` を今の CH の run で試走済み。
