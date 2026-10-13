# GPU 側 → クラウド側: run 終了の自動通知（2026-10-13 / 2c_raw_pilot_CS）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/CS_pilot_2c_raw_seed*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CS_pilot_2c_raw_seed123 | 8 | 0.138 | 5.5/stage (2.9/次元) | -4.26 |
| CS_pilot_2c_raw_seed42 | 8 | 0.140 | 5.6/stage (3.0/次元) | -4.15 |
| CS_pilot_2c_raw_seed7 | 8 | 0.142 | 5.7/stage (3.0/次元) | -4.17 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## データの出どころ（13j §4）

| run | data_source | two_channel | data_dir_for（experiment_data） | ファイル数 |
|---|---|---|---|---|
| CS_pilot_2c_raw_seed123 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| CS_pilot_2c_raw_seed42 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| CS_pilot_2c_raw_seed7 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-13 22:31:22,094:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CS_pilot_2c_raw_seed123                  0.00    -4.26  0.1154  OK    1.00  1.82  [ -0.99, -0.83, -0.41]  40%  [ -0.92, -0.06, +0.87]  11%
CS_pilot_2c_raw_seed42                   0.00    -4.15  0.1151  OK    0.99  1.82  [ -0.99, -0.83, -0.41]  37%  [ -0.91, -0.06, +0.86]   8%
CS_pilot_2c_raw_seed7                    0.00    -4.17  0.1150  OK    1.00  1.82  [ -0.99, -0.83, -0.41]  39%  [ -0.86, +0.09, +0.91]  10%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-13_2c_raw_pilot_CS.csv
```

全列は `docs/handoff/gpu_2026-10-13_2c_raw_pilot_CS.csv` にある。

## ψ_i とデータの ψ_i^data（tools/psi_table_runs.py、13b §4）

run: `CS_pilot_2c_raw_seed*`。各セルは ψ の Day 3–21 の中央値。data = ψ^data（括弧は使えた日数）、MAP = max logL の seed、事後 = 3 seed の粒子 200 本の中央値 [5%, 95%]。

| 条件 | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|
| CS data | 0.80 (5) | 0.59 (5) | 0.50 (5) | 0.09 (1) | -- (0) |
| CS MAP（ψ 固定の段） | 0.98 | 0.97 | 0.97 | 0.45 | 0.85 |
| CS 事後 | 0.98 [0.97, 0.98] | 0.97 [0.96, 0.97] | 0.97 [0.97, 0.97] | 0.23 [0.10, 0.90] | 0.26 [0.10, 0.91] |

seed ごとの MAP（日の中央値）:

| run | max logL | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|---|
| CS_pilot_2c_raw_seed123 | -4.26 | 0.98 | 0.97 | 0.97 | 0.84 | 0.85 |
| CS_pilot_2c_raw_seed42 | -4.15 | 0.98 | 0.97 | 0.97 | 0.45 | 0.85 |
| CS_pilot_2c_raw_seed7 | -4.17 | 0.98 | 0.97 | 0.97 | 0.15 | 0.18 |

## check_paper_runs.py の出力（そのまま）

```

=== CS_pilot_2c_raw  (3 seeds)
  seed123 beta=1.000 st= 8 acc=0.14 maxlogL=    -4.26 lnZ=   -16.63 rmse=0.1154  a35=-0.09[-0.90,+0.86]  a45=-0.06[-0.92,+0.87]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a12': 0.33, 'a33': 0.41, 'a13': 0.33, 'a24': 0.13, 'a45': 0.11}
          片側 5%: 成分 箱 下側/上側
            a33  [-1,1.5] 0.41/0.00  ← 箱で切られている
            a12  [-0.5,2] 0.00/0.33  ← 箱で切られている
            a13  [-1,3] 0.00/0.33  ← 箱で切られている
  seed 42 beta=1.000 st= 8 acc=0.14 maxlogL=    -4.15 lnZ=   -16.50 rmse=0.1151  a35=-0.12[-0.91,+0.80]  a45=-0.06[-0.91,+0.86]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a12': 0.32, 'a33': 0.37, 'a34': 0.13, 'a13': 0.35, 'a55': 0.1, 'a25': 0.1}
          片側 5%: 成分 箱 下側/上側
            a33  [-1,1.5] 0.37/0.00  ← 箱で切られている
            a13  [-1,3] 0.00/0.35  ← 箱で切られている
            a12  [-0.5,2] 0.00/0.32  ← 箱で切られている
  seed  7 beta=1.000 st= 8 acc=0.14 maxlogL=    -4.17 lnZ=   -16.37 rmse=0.1150  a35=-0.01[-0.91,+0.87]  a45=+0.09[-0.86,+0.91]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a12': 0.29, 'a33': 0.39, 'a44': 0.1, 'a13': 0.35, 'a24': 0.11, 'a55': 0.1, 'a25': 0.1}
          片側 5%: 成分 箱 下側/上側
            a33  [-1,1.5] 0.39/0.00  ← 箱で切られている
            a13  [-1,3] 0.00/0.35  ← 箱で切られている
            a12  [-0.5,2] 0.00/0.29  ← 箱で切られている
       1 粒子の移動: seed123 5.5/stage (2.9/次元), seed42 5.6/stage (3.0/次元), seed7 5.7/stage (3.0/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1（粒子の最良値、polish.json なし）
  PASS 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.11、中央値幅/sd 上位: a13(0.20), a23(0.17), a22(0.11)
       同定されていない（判定 4 から外した）: a34(sd/一様sd 0.99, 幅/sd 0.30), a44(sd/一様sd 1.00, 幅/sd 0.27), a14(sd/一様sd 0.98, 幅/sd 0.09), a24(sd/一様sd 1.01, 幅/sd 0.10), a55(sd/一様sd 0.99, 幅/sd 0.12), a15(sd/一様sd 0.98, 幅/sd 0.04), a25(sd/一様sd 1.00, 幅/sd 0.13), a35(sd/一様sd 0.95, 幅/sd 0.21), a45(sd/一様sd 0.98, 幅/sd 0.27)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `6b69e0c` Manuscript Methods: two-channel likelihood (viable vs phi*psi, total vs phi, live fraction), psi fixed to data in stages (i)-(ii); supplement weights table/algorithm; results flagged for raw 2c rewrite. tools/viab_sigma_options.py
- 通知を書いた時刻: 2026-10-13 22:33:58 JST
