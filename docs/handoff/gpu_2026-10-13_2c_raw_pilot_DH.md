# GPU 側 → クラウド側: run 終了の自動通知（2026-10-13 / 2c_raw_pilot_DH）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DH_pilot_2c_raw_seed*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DH_pilot_2c_raw_seed123 | 10 | 0.122 | 4.9/stage (3.2/次元) | -15.34 |
| DH_pilot_2c_raw_seed42 | 11 | 0.120 | 4.8/stage (3.5/次元) | -14.45 |
| DH_pilot_2c_raw_seed7 | 11 | 0.123 | 4.9/stage (3.6/次元) | -14.44 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## データの出どころ（13j §4）

| run | data_source | two_channel | data_dir_for（experiment_data） | ファイル数 |
|---|---|---|---|---|
| DH_pilot_2c_raw_seed123 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| DH_pilot_2c_raw_seed42 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |
| DH_pilot_2c_raw_seed7 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-13 23:41:50,289:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DH_pilot_2c_raw_seed123                  0.00   -15.34  0.0975  OK    1.37 14.94  [ +0.14, +0.58, +1.10]   0%  [ +0.16, +4.66, +5.92]  20%
DH_pilot_2c_raw_seed42                   0.00   -14.45  0.0926  OK    1.28 14.94  [ +0.14, +0.63, +1.15]   0%  [ +0.51, +4.85, +5.89]  16%
DH_pilot_2c_raw_seed7                    0.00   -14.44  0.0917  OK    1.24 14.94  [ +0.22, +0.65, +1.09]   0%  [ +2.03, +5.09, +5.93]  21%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-13_2c_raw_pilot_DH.csv
```

全列は `docs/handoff/gpu_2026-10-13_2c_raw_pilot_DH.csv` にある。

## ψ_i とデータの ψ_i^data（tools/psi_table_runs.py、13b §4）

run: `DH_pilot_2c_raw_seed*`。各セルは ψ の Day 3–21 の中央値。data = ψ^data（括弧は使えた日数）、MAP = max logL の seed、事後 = 3 seed の粒子 200 本の中央値 [5%, 95%]。

| 条件 | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|
| DH data | 0.35 (4) | 0.26 (5) | 0.58 (5) | 0.63 (2) | 1.19 (1) |
| DH MAP（ψ 固定の段） | 0.10 | 0.97 | 0.98 | 0.93 | 0.91 |
| DH 事後 | 0.16 [0.07, 0.92] | 0.97 [0.96, 0.97] | 0.97 [0.96, 0.98] | 0.93 [0.89, 0.95] | 0.90 [0.21, 0.93] |

seed ごとの MAP（日の中央値）:

| run | max logL | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|---|
| DH_pilot_2c_raw_seed123 | -15.34 | 0.85 | 0.97 | 0.98 | 0.95 | 0.92 |
| DH_pilot_2c_raw_seed42 | -14.45 | 0.16 | 0.97 | 0.98 | 0.93 | 0.90 |
| DH_pilot_2c_raw_seed7 | -14.44 | 0.10 | 0.97 | 0.98 | 0.93 | 0.91 |

## check_paper_runs.py の出力（そのまま）

```

=== DH_pilot_2c_raw  (3 seeds)
  seed123 beta=1.000 st=10 acc=0.12 maxlogL=   -15.34 lnZ=   -36.17 rmse=0.0980  a35=-0.12[-0.46,+0.47]  a45=+4.66[+0.16,+5.92]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a24': 0.29, 'a35': 0.39, 'a45': 0.2}
          片側 5%: 成分 箱 下側/上側
            a35  [-0.5,5] 0.39/0.00  ← 箱で切られている
            a24  [-0.5,3] 0.00/0.29  ← 箱で切られている
            a45  [-0.5,6] 0.02/0.19
  seed 42 beta=1.000 st=11 acc=0.12 maxlogL=   -14.45 lnZ=   -36.31 rmse=0.0934  a35=-0.21[-0.47,+0.49]  a45=+4.85[+0.51,+5.89]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a14': 0.12, 'a24': 0.33, 'a35': 0.48, 'a45': 0.16}
          片側 5%: 成分 箱 下側/上側
            a35  [-0.5,5] 0.48/0.00  ← 箱で切られている
            a24  [-0.5,3] 0.00/0.33  ← 箱で切られている
            a45  [-0.5,6] 0.01/0.15
  seed  7 beta=1.000 st=11 acc=0.12 maxlogL=   -14.44 lnZ=   -35.43 rmse=0.0924  a35=-0.23[-0.47,+0.37]  a45=+5.09[+2.03,+5.93]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.1, 'a24': 0.38, 'a35': 0.5, 'a45': 0.21}
          片側 5%: 成分 箱 下側/上側
            a35  [-0.5,5] 0.50/0.00  ← 箱で切られている
            a24  [-0.5,3] 0.00/0.38  ← 箱で切られている
            a45  [-0.5,6] 0.01/0.20  ← 箱で切られている
       1 粒子の移動: seed123 4.9/stage (3.2/次元), seed42 4.8/stage (3.5/次元), seed7 4.9/stage (3.6/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1（粒子の最良値、polish.json なし）
  PASS 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.90、中央値幅/sd 上位: a13(0.50), a35(0.36), a44(0.31)
       同定されていない（判定 4 から外した）: a11(sd/一様sd 0.96, 幅/sd 0.43, 符号は決まる), a12(sd/一様sd 0.94, 幅/sd 0.22, 符号は決まる), a14(sd/一様sd 0.97, 幅/sd 0.29), a55(sd/一様sd 0.93, 幅/sd 0.39), a15(sd/一様sd 0.97, 幅/sd 0.21), a25(sd/一様sd 0.85, 幅/sd 0.78), a45(sd/一様sd 0.86, 幅/sd 0.27, 符号は決まる)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `5d7b4ee` Cover letter and highlights for the measured data and the two-channel likelihood; result claims flagged until the raw runs
- 通知を書いた時刻: 2026-10-13 23:44:25 JST
