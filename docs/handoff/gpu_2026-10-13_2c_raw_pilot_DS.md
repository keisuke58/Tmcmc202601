# GPU 側 → クラウド側: run 終了の自動通知（2026-10-13 / 2c_raw_pilot_DS）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DS_pilot_2c_raw_seed*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DS_pilot_2c_raw_seed123 | 11 | 0.139 | 5.6/stage (4.1/次元) | -63.91 |
| DS_pilot_2c_raw_seed42 | 12 | 0.128 | 5.1/stage (4.1/次元) | -63.91 |
| DS_pilot_2c_raw_seed7 | 12 | 0.131 | 5.3/stage (4.2/次元) | -63.37 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## データの出どころ（13j §4）

| run | data_source | two_channel | data_dir_for（experiment_data） | ファイル数 |
|---|---|---|---|---|
| DS_pilot_2c_raw_seed123 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| DS_pilot_2c_raw_seed42 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| DS_pilot_2c_raw_seed7 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-13 23:59:27,366:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
Traceback (most recent call last):
  File "/home/nishioka/miniforge3/envs/klempt_fem2/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 442, in discover_pjrt_plugins
    plugin_module.initialize()
  File "/home/nishioka/miniforge3/envs/klempt_fem2/lib/python3.10/site-packages/jax_plugins/xla_cuda12/__init__.py", line 324, in initialize
    _check_cuda_versions(raise_on_first_error=True)
  File "/home/nishioka/miniforge3/envs/klempt_fem2/lib/python3.10/site-packages/jax_plugins/xla_cuda12/__init__.py", line 281, in _check_cuda_versions
    local_device_count = cuda_versions.cuda_device_count()
RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE
ERROR:jax._src.xla_bridge:Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
Traceback (most recent call last):
  File "/home/nishioka/miniforge3/envs/klempt_fem2/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 442, in discover_pjrt_plugins
    plugin_module.initialize()
  File "/home/nishioka/miniforge3/envs/klempt_fem2/lib/python3.10/site-packages/jax_plugins/xla_cuda12/__init__.py", line 324, in initialize
    _check_cuda_versions(raise_on_first_error=True)
  File "/home/nishioka/miniforge3/envs/klempt_fem2/lib/python3.10/site-packages/jax_plugins/xla_cuda12/__init__.py", line 281, in _check_cuda_versions
    local_device_count = cuda_versions.cuda_device_count()
RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE
run                                         K  maxlogL    RMSE  照合  D21/15    実測          a33 [5,50,95]%    端          a45 [5,50,95]%    端
DS_pilot_2c_raw_seed123                  0.00   -63.91  0.0967  OK    0.92  2.53  [ +1.21, +1.98, +2.77]   0%  [ +1.76, +3.06, +5.03]   0%
DS_pilot_2c_raw_seed42                   0.00   -63.91  0.0986  OK    0.92  2.53  [ +1.23, +1.98, +2.75]   0%  [ +1.73, +3.05, +5.09]   0%
DS_pilot_2c_raw_seed7                    0.00   -63.37  0.0968  OK    0.93  2.53  [ +1.10, +1.91, +2.70]   0%  [ +1.79, +3.20, +5.08]   0%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-13_2c_raw_pilot_DS.csv
```

全列は `docs/handoff/gpu_2026-10-13_2c_raw_pilot_DS.csv` にある。

## a33 の下の山の重み（DS の二峰判定・2026-10-08h）

DS の a33 は +1 付近と −10 付近の二峰で、seed ごとに山の配分が違う。
**w(a33<-5) が 3 seed で ±0.10 以内にそろえば、判定 4 が二峰のせいで FAIL しても先へ進めてよい**
（原稿では a33 を「二峰・弱く同定」と書く）。そろわなければ止めて報告。
参考: pilot（1000 粒子）は 0.139 / 0.443 / 0.657 で幅 0.52 だった。

```
run                                                       n  w(a33<-5)   median       5%      95%    maxlogL
DS_pilot_2c_raw_seed123                                1000      0.000     1.98     1.21     2.77     -63.91
DS_pilot_2c_raw_seed42                                 1000      0.000     1.98     1.23     2.75     -63.91
DS_pilot_2c_raw_seed7                                  1000      0.000     1.91     1.10     2.70     -63.37

run                                                  山                 n   a45 5%  a45 50%  a45 95%    maxlogL
DS_pilot_2c_raw_seed123                              a33>=-5        1000    +1.76    +3.06    +5.03     -63.91
DS_pilot_2c_raw_seed123                              a33<-5            0        -        -        -          -
DS_pilot_2c_raw_seed42                               a33>=-5        1000    +1.73    +3.05    +5.09     -63.91
DS_pilot_2c_raw_seed42                               a33<-5            0        -        -        -          -
DS_pilot_2c_raw_seed7                                a33>=-5        1000    +1.79    +3.20    +5.08     -63.37
DS_pilot_2c_raw_seed7                                a33<-5            0        -        -        -          -
```

## ψ_i とデータの ψ_i^data（tools/psi_table_runs.py、13b §4）

run: `DS_pilot_2c_raw_seed*`。各セルは ψ の Day 3–21 の中央値。data = ψ^data（括弧は使えた日数）、MAP = max logL の seed、事後 = 3 seed の粒子 200 本の中央値 [5%, 95%]。

| 条件 | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|
| DS data | -- (0) | 0.74 (4) | 0.65 (5) | 0.60 (5) | 1.06 (5) |
| DS MAP（ψ 固定の段） | 0.16 | 0.96 | 0.98 | 0.97 | 0.97 |
| DS 事後 | 0.80 [0.10, 0.92] | 0.95 [0.93, 0.96] | 0.98 [0.98, 0.98] | 0.97 [0.97, 0.98] | 0.96 [0.96, 0.97] |

seed ごとの MAP（日の中央値）:

| run | max logL | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|---|
| DS_pilot_2c_raw_seed123 | -63.91 | 0.11 | 0.96 | 0.98 | 0.97 | 0.97 |
| DS_pilot_2c_raw_seed42 | -63.91 | 0.12 | 0.95 | 0.98 | 0.97 | 0.96 |
| DS_pilot_2c_raw_seed7 | -63.37 | 0.16 | 0.96 | 0.98 | 0.97 | 0.97 |

## check_paper_runs.py の出力（そのまま）

```

=== DS_pilot_2c_raw  (3 seeds)
  seed123 beta=1.000 st=11 acc=0.14 maxlogL=   -63.91 lnZ=   -88.28 rmse=0.0945  a35=+2.10[+0.94,+2.88]  a45=+3.06[+1.76,+5.03]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.1, 'a22': 0.11, 'a44': 0.34, 'a24': 0.12, 'a15': 0.12, 'a35': 0.15}
          片側 5%: 成分 箱 下側/上側
            a44  [-1,2.5] 0.34/0.00  ← 箱で切られている
            a35  [-3,3] 0.00/0.15
            a24  [-0.5,3] 0.01/0.11
  seed 42 beta=1.000 st=12 acc=0.13 maxlogL=   -63.91 lnZ=   -92.97 rmse=0.0959  a35=+2.16[+1.12,+2.88]  a45=+3.05[+1.73,+5.09]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.1, 'a44': 0.32, 'a23': 0.11, 'a24': 0.12, 'a15': 0.1, 'a25': 0.11, 'a35': 0.14}
          片側 5%: 成分 箱 下側/上側
            a44  [-1,2.5] 0.32/0.00  ← 箱で切られている
            a35  [-3,3] 0.00/0.14
            a24  [-0.5,3] 0.01/0.12
            a23  [-1,3] 0.01/0.10
  seed  7 beta=1.000 st=12 acc=0.13 maxlogL=   -63.37 lnZ=   -88.63 rmse=0.0946  a35=+1.95[+0.69,+2.82]  a45=+3.20[+1.79,+5.08]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a44': 0.31, 'a24': 0.14, 'a15': 0.12, 'a25': 0.13}
          片側 5%: 成分 箱 下側/上側
            a44  [-1,2.5] 0.31/0.00  ← 箱で切られている
            a24  [-0.5,3] 0.00/0.14
            a25  [-0.5,2.5] 0.02/0.11
       1 粒子の移動: seed123 5.6/stage (4.1/次元), seed42 5.1/stage (4.1/次元), seed7 5.3/stage (4.2/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1（粒子の最良値、polish.json なし）
  PASS 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.54、中央値幅/sd 上位: a35(0.34), a34(0.28), a55(0.19)
       同定されていない（判定 4 から外した）: a11(sd/一様sd 0.98, 幅/sd 0.25), a12(sd/一様sd 0.98, 幅/sd 0.16), a22(sd/一様sd 0.96, 幅/sd 0.28), a13(sd/一様sd 0.93, 幅/sd 0.59), a14(sd/一様sd 0.99, 幅/sd 0.09), a15(sd/一様sd 0.94, 幅/sd 0.18), a25(sd/一様sd 0.94, 幅/sd 0.23)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `8c5311d` docs: run 終了の自動通知（2026-10-13）— FAIL を含む群は回し直すまで解釈しない
- 通知を書いた時刻: 2026-10-14 00:02:02 JST
