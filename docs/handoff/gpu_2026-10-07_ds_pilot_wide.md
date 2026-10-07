# GPU 側 → クラウド側: run 終了の自動通知（2026-10-07 / ds_pilot_wide）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DS_pilot_wide80_*`（出力ディレクトリ 0 個）

## 結論

**（判定スクリプトが結論行を出さなかった。出力をそのまま読むこと）**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| （出力ディレクトリが見つからない） | | | | |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
評価できる run が無い: data_5species/main/_runs/paper_gateoff/DS_pilot_wide80_*
```

全列は `docs/handoff/gpu_2026-10-07_ds_pilot_wide.csv` にある。

## check_paper_runs.py の出力（そのまま）

```
判定できない: data_5species/main/_runs/paper_gateoff に run が無い（空の PASS を返さない）
```

## 直近の run の git

- HEAD: `4ae2684` docs: 2〜4 番目の投稿先用の予備（本文は BMB 版と同じ format-free 版、各誌で足すものの一覧）
- 通知を書いた時刻: 2026-10-07 23:22:37 JST
