# GPU 側 → クラウド側: run 終了の自動通知（2026-10-09 / ch_p2_noph）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/CH_p2_mut150_wide_noph_seed*`（出力ディレクトリ 3 個）

## 結論

**全群 PASS**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CH_p2_mut150_wide_noph_seed123 | 7 | 0.133 | 19.9/stage (9.3/次元) | -8.79 |
| CH_p2_mut150_wide_noph_seed42 | 7 | 0.131 | 19.7/stage (9.2/次元) | -8.82 |
| CH_p2_mut150_wide_noph_seed7 | 7 | 0.133 | 19.9/stage (9.3/次元) | -8.77 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-09 06:56:08,862:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CH_p2_mut150_wide_noph_seed123           0.00    -8.79  0.0751  OK    1.00  1.00  [ -0.95, -0.42, +0.39]  11%  [ -0.85, +0.44, +1.84]   9%
CH_p2_mut150_wide_noph_seed42            0.00    -8.82  0.0763  OK    0.98  1.00  [ -0.93, -0.41, +0.36]  10%  [ -0.86, +0.45, +1.84]  10%
CH_p2_mut150_wide_noph_seed7             0.00    -8.77  0.0752  OK    0.96  1.00  [ -0.92, -0.40, +0.38]  10%  [ -0.87, +0.47, +1.85]  11%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-09_ch_p2_noph.csv
```

全列は `docs/handoff/gpu_2026-10-09_ch_p2_noph.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== CH_p2_mut150_wide_noph  (3 seeds)
  seed123 beta=1.000 st= 7 acc=0.13 maxlogL=    -8.79 lnZ=   -20.54 rmse=0.0751  a35=-0.27[-0.92,+0.74]  a45=+0.44[-0.85,+1.84]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.22, 'a22': 0.11, 'a33': 0.11, 'a44': 0.1, 'a55': 0.1}
          片側 5%: 成分 箱 下側/上側（前段 CH_p1_mut150_seed123 の 下側/上側・尤度が違う段なので増加は見ない）
            a11  [-1,1.5] 0.22/0.00 (前段 0.00/0.04)  ← 箱で切られている
            a33  [-1,1.5] 0.10/0.00 (前段 0.08/0.00)
  seed 42 beta=1.000 st= 7 acc=0.13 maxlogL=    -8.82 lnZ=   -20.53 rmse=0.0763  a35=-0.26[-0.93,+0.78]  a45=+0.45[-0.86,+1.84]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.21, 'a22': 0.11, 'a33': 0.1, 'a44': 0.1, 'a35': 0.1, 'a45': 0.1}
          片側 5%: 成分 箱 下側/上側（前段 CH_p1_mut150_seed42 の 下側/上側・尤度が違う段なので増加は見ない）
            a11  [-1,1.5] 0.21/0.00 (前段 0.00/0.04)  ← 箱で切られている
            a33  [-1,1.5] 0.10/0.00 (前段 0.07/0.01)
  seed  7 beta=1.000 st= 7 acc=0.13 maxlogL=    -8.77 lnZ=   -20.80 rmse=0.0752  a35=-0.30[-0.94,+0.78]  a45=+0.47[-0.87,+1.85]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.21, 'a22': 0.12, 'a55': 0.1, 'a45': 0.11}
          片側 5%: 成分 箱 下側/上側（前段 CH_p1_mut150_seed7 の 下側/上側・尤度が違う段なので増加は見ない）
            a11  [-1,1.5] 0.21/0.00 (前段 0.00/0.04)  ← 箱で切られている
            a22  [-1,2.5] 0.11/0.01 (前段 0.05/0.01)
       1 粒子の移動: seed123 19.9/stage (9.3/次元), seed42 19.7/stage (9.2/次元), seed7 19.9/stage (9.3/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.05、中央値幅/sd 上位: a44(0.1), a11(0.1), a24(0.1)

全群 PASS
```

## 直近の run の git

- HEAD: `1812b74` handoff: gate CH p2 PASS → ult 投入（2026-10-09）
- 通知を書いた時刻: 2026-10-09 06:56:16 JST
