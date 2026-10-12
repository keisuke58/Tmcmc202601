# GPU 側 → クラウド側: run 終了の自動通知（2026-10-13 / DH_ult_prov）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed123 | 11 | 0.108 | 16.2/stage (11.9/次元) | -14.71 |
| DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed42 | 11 | 0.125 | 18.8/stage (13.8/次元) | -14.79 |
| DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed7 | 11 | 0.107 | 16.0/stage (11.8/次元) | -14.86 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-13 01:50:15,397:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed123 0.00   -14.71  0.0830  OK    3.00  2.74  [ -2.40, -1.20, -0.72]   0%  [ -1.20, +1.61, +5.20]   0%
DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed42 0.00   -14.79  0.0847  OK    2.49  2.74  [ -2.34, -1.38, -0.81]   0%  [ -2.18, +0.60, +3.59]   0%
DH_ult_nonarrow_w3_noph_p14k_sd4_prov_seed7 0.00   -14.86  0.0818  OK    2.81  2.74  [ -2.80, -1.30, -0.75]   0%  [ -1.49, +1.48,+10.88]   0%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-13_DH_ult_prov.csv
```

全列は `docs/handoff/gpu_2026-10-13_DH_ult_prov.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== DH_ult_nonarrow_w3_noph_p14k_sd4_prov  (3 seeds)
  seed123 beta=1.000 st=11 acc=0.11 maxlogL=   -14.71 lnZ=   -33.75 rmse=0.0830  a35=-7.50[-13.96,-2.05]  a45=+1.61[-1.20,+5.20]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a55': 0.1}
  seed 42 beta=1.000 st=11 acc=0.13 maxlogL=   -14.79 lnZ=   -31.24 rmse=0.0847  a35=-6.22[-13.70,-1.83]  a45=+0.60[-2.18,+3.59]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.11, 'a22': 0.11, 'a44': 0.11, 'a25': 0.11}
          片側 5%: 成分 箱 下側/上側（前段 DH_p2_nonarrow_w3_noph_p14k_seed42 の 下側/上側）
            a44  [-10,2.68464] 0.11/0.00 (前段 0.18/0.00)
            a22  [-5,1.81036] 0.11/0.00 (前段 0.06/0.00)
  seed  7 beta=1.000 st=11 acc=0.11 maxlogL=   -14.86 lnZ=   -34.44 rmse=0.0818  a35=-7.35[-14.70,-1.82]  a45=+1.48[-1.49,+10.88]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a15': 0.15, 'a35': 0.13}
          片側 5%: 成分 箱 下側/上側（前段 DH_p2_nonarrow_w3_noph_p14k_seed7 の 下側/上側）
            a35  [-15,8.29445] 0.13/0.00 (前段 0.09/0.00)
            a15  [-0.5,2.5] 0.11/0.04 (前段 0.03/0.04)
       1 粒子の移動: seed123 16.2/stage (11.9/次元), seed42 18.8/stage (13.8/次元), seed7 16.0/stage (11.8/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.14、中央値幅/sd 上位: a24(0.63), a13(0.58), a34(0.55)
       同定されていない（判定 4 から外した）: a11(sd/一様sd 0.96, 幅/sd 0.64), a12(sd/一様sd 0.82, 幅/sd 0.46), a14(sd/一様sd 0.91, 幅/sd 0.33), a55(sd/一様sd 0.95, 幅/sd 0.24), a15(sd/一様sd 0.97, 幅/sd 0.10), a25(sd/一様sd 0.91, 幅/sd 0.29)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `88c188e` gate_submit_ult: NODE accepts a ';'-separated per-seed node list
- 通知を書いた時刻: 2026-10-13 01:50:23 JST
