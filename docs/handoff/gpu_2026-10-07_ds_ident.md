# GPU 側 → クラウド側: run 終了の自動通知（2026-10-07 / ds_ident）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DS_ident*mut80*`（出力ディレクトリ 6 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DS_ident_prior0_mut80_seed123 | 10 | 0.125 | 10.0/stage (6.6/次元) | -1.06 |
| DS_ident_prior0_mut80_seed42 | 9 | 0.118 | 9.4/stage (5.7/次元) | -1.63 |
| DS_ident_prior0_mut80_seed7 | 10 | 0.119 | 9.5/stage (6.4/次元) | -1.10 |
| DS_ident_prior6_mut80_seed123 | 9 | 0.143 | 11.5/stage (6.9/次元) | -1.15 |
| DS_ident_prior6_mut80_seed42 | 9 | 0.142 | 11.3/stage (6.8/次元) | -1.06 |
| DS_ident_prior6_mut80_seed7 | 9 | 0.141 | 11.3/stage (6.8/次元) | -1.06 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-07 17:39:38,167:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DS_ident_prior0_mut80_seed123            0.00    -1.06  0.0209  OK    1.24  1.31  [-14.49,-10.61, -1.48]  19%  [ +2.68, +7.60,+12.64]   1%
DS_ident_prior0_mut80_seed42             0.00    -1.63  0.0235  OK    1.33  1.31  [-12.88, +1.27, +8.24]   4%  [ +0.01, +9.99,+18.32]   5%
DS_ident_prior0_mut80_seed7              0.00    -1.10  0.0199  OK    1.24  1.31  [-14.62,-11.48, +2.78]  24%  [ +4.38, +9.53,+14.07]   1%
DS_ident_prior6_mut80_seed123            0.00    -1.15  0.0207  OK    1.28  1.31  [-12.99, -7.02, -1.45]   4%  [ +0.48, +3.79, +7.87]   0%
DS_ident_prior6_mut80_seed42             0.00    -1.06  0.0205  OK    1.29  1.31  [-13.15, -6.90, -1.22]   5%  [ +0.15, +3.94, +8.21]   0%
DS_ident_prior6_mut80_seed7              0.00    -1.06  0.0210  OK    1.25  1.31  [-12.98, -7.01, -1.50]   4%  [ +0.78, +4.14, +8.39]   0%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-07_ds_ident.csv
```

全列は `docs/handoff/gpu_2026-10-07_ds_ident.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== DS_ident_prior0_mut80  (3 seeds)
  seed123 beta=1.000 st=10 acc=0.12 maxlogL=    -1.06 lnZ=   -18.34 rmse=0.0209  a35=+3.01[-1.12,+9.18]  a45=+7.60[+2.68,+12.64]
          箱の端 5% に 10% 超: {'a22': 0.19, 'a33': 0.19, 'a44': 0.18, 'a13': 0.12, 'a15': 0.12}
  seed 42 beta=1.000 st= 9 acc=0.12 maxlogL=    -1.63 lnZ=   -20.43 rmse=0.0235  a35=+8.12[+2.87,+18.64]  a45=+9.99[+0.01,+18.32]
          箱の端 5% に 10% 超: {'a22': 0.11, 'a44': 0.26, 'a23': 0.12, 'a55': 0.1}
  seed  7 beta=1.000 st=10 acc=0.12 maxlogL=    -1.10 lnZ=   -18.32 rmse=0.0199  a35=+4.86[+0.35,+11.75]  a45=+9.53[+4.38,+14.07]
          箱の端 5% に 10% 超: {'a22': 0.1, 'a33': 0.24, 'a44': 0.24, 'a13': 0.11}
       1 粒子の移動: seed123 10.0/stage (6.6/次元), seed42 9.4/stage (5.7/次元), seed7 9.5/stage (6.4/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.57、中央値幅/sd 上位: a33(1.8), a34(1.5), a23(1.4)

=== DS_ident_prior6_mut80  (3 seeds)
  seed123 beta=1.000 st= 9 acc=0.14 maxlogL=    -1.15 lnZ=   -16.73 rmse=0.0207  a35=+1.93[-1.26,+6.08]  a45=+3.79[+0.48,+7.87]
  seed 42 beta=1.000 st= 9 acc=0.14 maxlogL=    -1.06 lnZ=   -17.21 rmse=0.0205  a35=+2.33[-1.18,+6.29]  a45=+3.94[+0.15,+8.21]
  seed  7 beta=1.000 st= 9 acc=0.14 maxlogL=    -1.06 lnZ=   -16.43 rmse=0.0210  a35=+1.84[-1.31,+5.48]  a45=+4.14[+0.78,+8.39]
       1 粒子の移動: seed123 11.5/stage (6.9/次元), seed42 11.3/stage (6.8/次元), seed7 11.3/stage (6.8/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.09、中央値幅/sd 上位: a22(0.2), a14(0.2), a25(0.2)

PASS 5 DS_ident_prior0_mut80 の max logL -1.06 >= DS_ident_prior6_mut80 の -1.06 − 0.5

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `b961ba8` docs: 返答下書き — A1 の原因 (ii) は未確認と明記
- 通知を書いた時刻: 2026-10-07 17:39:57 JST
