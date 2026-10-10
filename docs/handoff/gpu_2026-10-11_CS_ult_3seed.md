# GPU 側 → クラウド側: run 終了の自動通知（2026-10-11 / CS_ult_3seed）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/CS_ult_wide2_sd4_seed*`（出力ディレクトリ 3 個）

## 結論

**全群 PASS**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CS_ult_wide2_sd4_seed123 | 6 | 0.132 | 19.8/stage (7.9/次元) | -16.05 |
| CS_ult_wide2_sd4_seed42 | 7 | 0.131 | 19.6/stage (9.2/次元) | -15.81 |
| CS_ult_wide2_sd4_seed7 | 7 | 0.130 | 19.5/stage (9.1/次元) | -15.97 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-11 01:08:54,834:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CS_ult_wide2_sd4_seed123                 0.00   -16.05  0.1720  OK    0.80  1.02  [-13.44, -7.00, -2.59]   3%  [ -0.91, -0.05, +0.90]  10%
CS_ult_wide2_sd4_seed42                  0.00   -15.81  0.1794  OK    0.88  1.02  [-13.46, -6.59, -2.45]   3%  [ -0.91, -0.04, +0.91]  11%
CS_ult_wide2_sd4_seed7                   0.00   -15.97  0.1791  OK    0.88  1.02  [-13.42, -6.65, -2.58]   3%  [ -0.93, -0.06, +0.89]  11%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-11_CS_ult_3seed.csv
```

全列は `docs/handoff/gpu_2026-10-11_CS_ult_3seed.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== CS_ult_wide2_sd4  (3 seeds)
  seed123 beta=1.000 st= 6 acc=0.13 maxlogL=   -16.05 lnZ=   -25.65 rmse=0.1720  a35=-0.33[-0.94,+0.73]  a45=-0.05[-0.91,+0.90]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.12, 'a34': 0.1, 'a44': 0.1, 'a14': 0.1, 'a15': 0.11, 'a25': 0.1, 'a45': 0.1}
          片側 5%: 成分 箱 下側/上側（前段 CS_p2_mut160_wide2_seed123 の 下側/上側）
            a11  [-5,1.36735] 0.12/0.00 (前段 0.12/0.00)
            a15  [-1,1] 0.10/0.01 (前段 0.10/0.01)
  seed 42 beta=1.000 st= 7 acc=0.13 maxlogL=   -15.81 lnZ=   -25.93 rmse=0.1794  a35=-0.34[-0.95,+0.75]  a45=-0.04[-0.91,+0.91]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.12, 'a44': 0.11, 'a14': 0.11, 'a55': 0.11, 'a15': 0.1, 'a35': 0.11, 'a45': 0.11}
          片側 5%: 成分 箱 下側/上側（前段 CS_p2_mut160_wide2_seed42 の 下側/上側）
            a11  [-5,1.2777] 0.12/0.00 (前段 0.14/0.00)
  seed  7 beta=1.000 st= 7 acc=0.13 maxlogL=   -15.97 lnZ=   -25.52 rmse=0.1791  a35=-0.30[-0.94,+0.75]  a45=-0.06[-0.93,+0.89]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.12, 'a34': 0.11, 'a44': 0.1, 'a14': 0.11, 'a55': 0.1, 'a25': 0.12, 'a35': 0.1, 'a45': 0.11}
          片側 5%: 成分 箱 下側/上側（前段 CS_p2_mut160_wide2_seed7 の 下側/上側）
            a11  [-5,1.29478] 0.12/0.00 (前段 0.12/0.00)
       1 粒子の移動: seed123 19.8/stage (7.9/次元), seed42 19.6/stage (9.2/次元), seed7 19.5/stage (9.1/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.24、中央値幅/sd 上位: a13(0.13), a33(0.12), a11(0.07)
       同定されていない（判定 4 から外した）: a12(sd/一様sd 0.90, 幅/sd 0.01), a34(sd/一様sd 0.93, 幅/sd 0.06), a44(sd/一様sd 1.00, 幅/sd 0.16), a14(sd/一様sd 0.85, 幅/sd 0.10), a23(sd/一様sd 0.87, 幅/sd 0.10), a24(sd/一様sd 0.95, 幅/sd 0.02), a55(sd/一様sd 1.00, 幅/sd 0.11), a15(sd/一様sd 0.81, 幅/sd 0.04), a25(sd/一様sd 0.96, 幅/sd 0.14), a35(sd/一様sd 0.91, 幅/sd 0.07), a45(sd/一様sd 1.01, 幅/sd 0.04)

全群 PASS
```

## 直近の run の git

- HEAD: `431c16e` P3 設計: 点モデルの全量 Σφ で α̇ を駆動（Klempt の局所残差の形、κ は 4 条件共通、s=0.2375・T*=1↔Day21、段階 0〜2 と全桁一致の検証）
- 通知を書いた時刻: 2026-10-11 01:09:04 JST
