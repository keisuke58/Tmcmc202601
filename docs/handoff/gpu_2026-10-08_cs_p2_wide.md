# GPU 側 → クラウド側: run 終了の自動通知（2026-10-08 / cs_p2_wide）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/CS_p2_mut160_wide_*`（出力ディレクトリ 3 個）

## 結論

**全群 PASS**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CS_p2_mut160_wide_seed123 | 7 | 0.135 | 21.6/stage (10.1/次元) | -16.08 |
| CS_p2_mut160_wide_seed42 | 7 | 0.137 | 21.9/stage (10.2/次元) | -16.05 |
| CS_p2_mut160_wide_seed7 | 7 | 0.138 | 22.1/stage (10.3/次元) | -16.47 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-08 19:02:40,341:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CS_p2_mut160_wide_seed123                0.00   -16.08  0.1848  OK    0.87  1.02  [ -3.73, -2.43, -1.35]   0%  [ -0.91, -0.02, +0.89]  10%
CS_p2_mut160_wide_seed42                 0.00   -16.05  0.1846  OK    0.87  1.02  [ -3.80, -2.45, -1.35]   0%  [ -0.91, -0.01, +0.88]   9%
CS_p2_mut160_wide_seed7                  0.00   -16.47  0.1780  OK    0.85  1.02  [ -3.82, -2.43, -1.31]   0%  [ -0.90, -0.06, +0.89]  10%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-08_cs_p2_wide.csv
```

全列は `docs/handoff/gpu_2026-10-08_cs_p2_wide.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== CS_p2_mut160_wide  (3 seeds)
  seed123 beta=1.000 st= 7 acc=0.14 maxlogL=   -16.08 lnZ=   -27.22 rmse=0.1848  a35=-0.41[-0.94,+0.64]  a45=-0.02[-0.91,+0.89]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a34': 0.1, 'a44': 0.1, 'a13': 0.24, 'a24': 0.1}
  seed 42 beta=1.000 st= 7 acc=0.14 maxlogL=   -16.05 lnZ=   -26.73 rmse=0.1846  a35=-0.39[-0.95,+0.65]  a45=-0.01[-0.91,+0.88]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a13': 0.23, 'a24': 0.1, 'a55': 0.1, 'a35': 0.11}
  seed  7 beta=1.000 st= 7 acc=0.14 maxlogL=   -16.47 lnZ=   -26.32 rmse=0.1780  a35=-0.39[-0.93,+0.63]  a45=-0.06[-0.90,+0.89]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a13': 0.23, 'a55': 0.11, 'a15': 0.11}
       1 粒子の移動: seed123 21.6/stage (10.1/次元), seed42 21.9/stage (10.2/次元), seed7 22.1/stage (10.3/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.43、中央値幅/sd 上位: a11(0.1), a14(0.1), a13(0.1)

全群 PASS
```

## 直近の run の git

- HEAD: `f793124` docs: run 終了の自動通知（2026-10-08）— FAIL を含む群は回し直すまで解釈しない
- 通知を書いた時刻: 2026-10-08 19:02:50 JST
