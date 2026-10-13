# GPU 側 → クラウド側: run 終了の自動通知（2026-10-13 / 2c_raw_pilot_CH）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/CH_pilot_2c_raw_seed*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CH_pilot_2c_raw_seed123 | 8 | 0.129 | 5.2/stage (2.8/次元) | -2.09 |
| CH_pilot_2c_raw_seed42 | 8 | 0.131 | 5.3/stage (2.8/次元) | -2.00 |
| CH_pilot_2c_raw_seed7 | 8 | 0.128 | 5.1/stage (2.7/次元) | -1.96 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## データの出どころ（13j §4）

| run | data_source | two_channel | data_dir_for（experiment_data） | ファイル数 |
|---|---|---|---|---|
| CH_pilot_2c_raw_seed123 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| CH_pilot_2c_raw_seed42 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| CH_pilot_2c_raw_seed7 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-13 22:33:56,814:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CH_pilot_2c_raw_seed123                  0.00    -2.09  0.0986  OK    0.98  0.00  [ -0.95, -0.38, +0.91]  13%  [ -0.85, +0.54, +1.83]  10%
CH_pilot_2c_raw_seed42                   0.00    -2.00  0.1040  OK    0.99  0.00  [ -0.95, -0.45, +0.80]  13%  [ -0.82, +0.55, +1.86]  10%
CH_pilot_2c_raw_seed7                    0.00    -1.96  0.0981  OK    0.99  0.00  [ -0.94, -0.38, +0.80]  11%  [ -0.83, +0.68, +1.87]  10%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-13_2c_raw_pilot_CH.csv
```

全列は `docs/handoff/gpu_2026-10-13_2c_raw_pilot_CH.csv` にある。

## ψ_i とデータの ψ_i^data（tools/psi_table_runs.py、13b §4）

run: `CH_pilot_2c_raw_seed*`。各セルは ψ の Day 3–21 の中央値。data = ψ^data（括弧は使えた日数）、MAP = max logL の seed、事後 = 3 seed の粒子 200 本の中央値 [5%, 95%]。

| 条件 | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|
| CH data | 0.35 (5) | 0.24 (5) | 0.53 (5) | -- (0) | -- (0) |
| CH MAP（ψ 固定の段） | 0.98 | 0.97 | 0.97 | 0.85 | 0.79 |
| CH 事後 | 0.98 [0.97, 0.98] | 0.97 [0.96, 0.97] | 0.97 [0.96, 0.97] | 0.26 [0.09, 0.91] | 0.22 [0.10, 0.91] |

seed ごとの MAP（日の中央値）:

| run | max logL | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|---|
| CH_pilot_2c_raw_seed123 | -2.09 | 0.98 | 0.97 | 0.97 | 0.88 | 0.13 |
| CH_pilot_2c_raw_seed42 | -2.00 | 0.98 | 0.97 | 0.97 | 0.27 | 0.12 |
| CH_pilot_2c_raw_seed7 | -1.96 | 0.98 | 0.97 | 0.97 | 0.85 | 0.79 |

## check_paper_runs.py の出力（そのまま）

```

=== CH_pilot_2c_raw  (3 seeds)
  seed123 beta=1.000 st= 8 acc=0.13 maxlogL=    -2.09 lnZ=   -13.30 rmse=0.0986  a35=-0.15[-0.91,+0.88]  a45=+0.54[-0.85,+1.83]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a22': 0.22, 'a33': 0.13, 'a24': 0.11, 'a25': 0.11}
          片側 5%: 成分 箱 下側/上側
            a22  [-1,2.5] 0.00/0.22  ← 箱で切られている
            a33  [-1,1.5] 0.12/0.01
  seed 42 beta=1.000 st= 8 acc=0.13 maxlogL=    -2.00 lnZ=   -13.28 rmse=0.1040  a35=-0.03[-0.90,+0.90]  a45=+0.55[-0.82,+1.86]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a22': 0.24, 'a33': 0.13, 'a24': 0.1, 'a35': 0.1}
          片側 5%: 成分 箱 下側/上側
            a22  [-1,2.5] 0.00/0.24  ← 箱で切られている
            a33  [-1,1.5] 0.13/0.00
  seed  7 beta=1.000 st= 8 acc=0.13 maxlogL=    -1.96 lnZ=   -13.49 rmse=0.0981  a35=+0.01[-0.88,+0.88]  a45=+0.68[-0.83,+1.87]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a22': 0.25, 'a33': 0.11, 'a55': 0.1, 'a15': 0.1, 'a45': 0.1}
          片側 5%: 成分 箱 下側/上側
            a22  [-1,2.5] 0.00/0.25  ← 箱で切られている
            a33  [-1,1.5] 0.11/0.00
       1 粒子の移動: seed123 5.2/stage (2.8/次元), seed42 5.3/stage (2.8/次元), seed7 5.1/stage (2.7/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1（粒子の最良値、polish.json なし）
  PASS 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.13、中央値幅/sd 上位: a12(0.14), a33(0.13), a13(0.11)
       同定されていない（判定 4 から外した）: a34(sd/一様sd 0.99, 幅/sd 0.10), a44(sd/一様sd 0.99, 幅/sd 0.14), a14(sd/一様sd 0.99, 幅/sd 0.05), a24(sd/一様sd 1.00, 幅/sd 0.09), a55(sd/一様sd 0.99, 幅/sd 0.10), a15(sd/一様sd 0.98, 幅/sd 0.18), a25(sd/一様sd 0.99, 幅/sd 0.13), a35(sd/一様sd 0.99, 幅/sd 0.27), a45(sd/一様sd 0.99, 幅/sd 0.16)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `4815604` docs: run 終了の自動通知（2026-10-13）— FAIL を含む群は回し直すまで解釈しない
- 通知を書いた時刻: 2026-10-13 22:36:32 JST
