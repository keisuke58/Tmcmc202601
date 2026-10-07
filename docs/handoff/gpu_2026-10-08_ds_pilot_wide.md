# GPU 側 → クラウド側: run 終了の自動通知（2026-10-08 / ds_pilot_wide）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DS_pilot_wide80_*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DS_pilot_wide80_seed123 | 10 | 0.116 | 9.3/stage (6.2/次元) | -0.91 |
| DS_pilot_wide80_seed42 | 10 | 0.111 | 8.9/stage (5.9/次元) | -1.63 |
| DS_pilot_wide80_seed7 | 10 | 0.112 | 8.9/stage (6.0/次元) | -1.49 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-08 05:16:31,871:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DS_pilot_wide80_seed123                  0.00    -0.91  0.0191  OK    1.17  1.31  [-11.37, +0.87, +1.80]   2%  [ +0.43, +2.17, +4.27]   0%
DS_pilot_wide80_seed42                   0.00    -1.63  0.0276  OK    1.20  1.31  [-14.22, -8.10, +1.71]  12%  [ -0.20, +0.95, +4.24]   0%
DS_pilot_wide80_seed7                    0.00    -1.49  0.0250  OK    1.23  1.31  [-13.61, -0.07, +1.95]   6%  [ +0.27, +1.51, +4.27]   0%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-08_ds_pilot_wide.csv
```

全列は `docs/handoff/gpu_2026-10-08_ds_pilot_wide.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== DS_pilot_wide80  (3 seeds)
  seed123 beta=1.000 st=10 acc=0.12 maxlogL=    -0.91 lnZ=   -20.89 rmse=0.0191  a35=+1.56[-0.16,+2.67]  a45=+2.17[+0.43,+4.27]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a22': 0.1, 'a44': 0.11, 'a13': 0.14, 'a55': 0.17}
  seed 42 beta=1.000 st=10 acc=0.11 maxlogL=    -1.63 lnZ=   -22.30 rmse=0.0276  a35=+1.42[-0.79,+2.72]  a45=+0.95[-0.20,+4.24]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.15, 'a12': 0.1, 'a22': 0.14, 'a33': 0.12, 'a13': 0.12, 'a15': 0.17}
  seed  7 beta=1.000 st=10 acc=0.11 maxlogL=    -1.49 lnZ=   -21.25 rmse=0.0250  a35=+1.43[-1.23,+2.74]  a45=+1.51[+0.27,+4.27]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.12, 'a22': 0.11, 'a44': 0.1, 'a55': 0.14, 'a15': 0.14}
       1 粒子の移動: seed123 9.3/stage (6.2/次元), seed42 8.9/stage (5.9/次元), seed7 8.9/stage (6.0/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.72、中央値幅/sd 上位: a33(1.6), a13(1.4), a25(1.4)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `27bb52d` handoff: DH λ 感度は 2b 境界 FAIL → N_MUT=100 で回し直し（2026-10-08g）
- 通知を書いた時刻: 2026-10-08 05:16:41 JST
