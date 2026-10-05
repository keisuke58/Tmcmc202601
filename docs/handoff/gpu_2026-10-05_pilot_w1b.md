# GPU 側 → クラウド側: run 終了の自動通知（2026-10-05 / pilot_w1b）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/*pilot*mut80*`（出力ディレクトリ 4 個）

## 結論

**（判定スクリプトが結論行を出さなかった。出力をそのまま読むこと）**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DH_pilot_gateon_mut80_seed42 | 9 | 0.123 | 9.8/stage (5.9/次元) | -4.09 |
| DH_pilot_mut80_seed42 | 10 | 0.123 | 9.8/stage (6.6/次元) | -4.44 |
| DS_pilot_gateon_mut80_seed42 | 8 | 0.134 | 10.7/stage (5.7/次元) | -1.10 |
| DS_pilot_mut80_seed42 | 9 | 0.132 | 10.5/stage (6.3/次元) | -1.24 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（eval_gateoff_runs.py）

RMSE・chi・Pg D21/15・a35/a45 の MAP と事後。箱の端は run 自身の箱で判定している。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
Traceback (most recent call last):
  File "eval_gateoff_runs.py", line 27, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

全列は `docs/handoff/gpu_2026-10-05_pilot_w1b.csv` にある。

## check_paper_runs.py の出力（そのまま）

```
Traceback (most recent call last):
  File "tools/check_paper_runs.py", line 36, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

## 直近の run の git

- HEAD: `18d1bff` feat: 通知ジョブが数字まで自動回収する（波ごとに 1 本）
- 通知を書いた時刻: 2026-10-05 23:18:13 JST
