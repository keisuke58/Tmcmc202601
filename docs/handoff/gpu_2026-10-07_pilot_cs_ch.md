# GPU 側 → クラウド側: run 終了の自動通知（2026-10-07 / pilot_cs_ch）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/*pilot*mut120*`（出力ディレクトリ 12 個）

## 結論

**全群 PASS**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CH_pilot_gateon_mut120_seed123 | 5 | 0.148 | 17.8/stage (5.9/次元) | -3.18 |
| CH_pilot_gateon_mut120_seed42 | 5 | 0.143 | 17.2/stage (5.7/次元) | -3.18 |
| CH_pilot_gateon_mut120_seed7 | 5 | 0.142 | 17.1/stage (5.7/次元) | -3.18 |
| CH_pilot_mut120_seed123 | 5 | 0.146 | 17.5/stage (5.8/次元) | -3.17 |
| CH_pilot_mut120_seed42 | 5 | 0.144 | 17.2/stage (5.7/次元) | -3.09 |
| CH_pilot_mut120_seed7 | 5 | 0.149 | 17.9/stage (6.0/次元) | -3.14 |
| CS_pilot_gateon_mut120_seed123 | 6 | 0.140 | 16.8/stage (6.7/次元) | -5.93 |
| CS_pilot_gateon_mut120_seed42 | 6 | 0.140 | 16.8/stage (6.7/次元) | -5.71 |
| CS_pilot_gateon_mut120_seed7 | 6 | 0.141 | 16.9/stage (6.8/次元) | -5.94 |
| CS_pilot_mut120_seed123 | 6 | 0.139 | 16.7/stage (6.7/次元) | -5.88 |
| CS_pilot_mut120_seed42 | 6 | 0.141 | 16.9/stage (6.8/次元) | -5.82 |
| CS_pilot_mut120_seed7 | 6 | 0.141 | 16.9/stage (6.8/次元) | -5.95 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-07 11:03:13,410:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CH_pilot_gateon_mut120_seed123           0.05    -3.18  0.0712  OK    1.00  1.00  [ -0.92, -0.17, +1.09]   9%  [ -0.83, +0.46, +1.82]   8%
CH_pilot_gateon_mut120_seed42            0.05    -3.18  0.0711  OK    1.00  1.00  [ -0.90, -0.18, +0.98]   8%  [ -0.85, +0.49, +1.83]  10%
CH_pilot_gateon_mut120_seed7             0.05    -3.18  0.0712  OK    1.00  1.00  [ -0.92, -0.23, +0.98]   9%  [ -0.87, +0.52, +1.85]  11%
CH_pilot_mut120_seed123                  0.00    -3.17  0.0721  OK    0.99  1.00  [ -0.90, -0.11, +1.06]   7%  [ -0.88, +0.48, +1.84]  11%
CH_pilot_mut120_seed42                   0.00    -3.09  0.0677  OK    1.00  1.00  [ -0.92, -0.16, +1.11]   9%  [ -0.86, +0.50, +1.83]  10%
CH_pilot_mut120_seed7                    0.00    -3.14  0.0680  OK    1.00  1.00  [ -0.93, -0.17, +1.06]   9%  [ -0.83, +0.48, +1.84]   9%
CS_pilot_gateon_mut120_seed123           0.05    -5.93  0.1142  OK    0.99  1.02  [ -0.97, -0.61, +0.17]  18%  [ -0.88, -0.00, +0.90]   9%
CS_pilot_gateon_mut120_seed42            0.05    -5.71  0.1144  OK    0.99  1.02  [ -0.96, -0.60, +0.15]  18%  [ -0.90, -0.03, +0.90]  10%
CS_pilot_gateon_mut120_seed7             0.05    -5.94  0.1144  OK    0.99  1.02  [ -0.96, -0.57, +0.18]  16%  [ -0.91, +0.02, +0.89]   9%
CS_pilot_mut120_seed123                  0.00    -5.88  0.1183  OK    0.99  1.02  [ -0.96, -0.58, +0.17]  18%  [ -0.90, -0.03, +0.88]   9%
CS_pilot_mut120_seed42                   0.00    -5.82  0.1110  OK    0.94  1.02  [ -0.96, -0.62, +0.15]  16%  [ -0.89, -0.03, +0.91]  10%
CS_pilot_mut120_seed7                    0.00    -5.95  0.1137  OK    1.00  1.02  [ -0.96, -0.59, +0.20]  16%  [ -0.89, +0.00, +0.90]   9%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-07_pilot_cs_ch.csv
```

全列は `docs/handoff/gpu_2026-10-07_pilot_cs_ch.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== CH_pilot_gateon_mut120  (3 seeds)
  seed123 beta=1.000 st= 5 acc=0.15 maxlogL=    -3.18 lnZ=    -9.59 rmse=0.0712  a35=+0.04[-0.88,+0.89]  a45=+0.46[-0.83,+1.82]
          箱の端 5% に 10% 超: {'a44': 0.11, 'a14': 0.11, 'a23': 0.1, 'a15': 0.12}
  seed 42 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.18 lnZ=    -9.72 rmse=0.0711  a35=-0.04[-0.91,+0.89]  a45=+0.49[-0.85,+1.83]
          箱の端 5% に 10% 超: {'a44': 0.1, 'a23': 0.1, 'a55': 0.1, 'a15': 0.11, 'a35': 0.11}
  seed  7 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.18 lnZ=    -9.51 rmse=0.0712  a35=-0.01[-0.89,+0.89]  a45=+0.52[-0.87,+1.85]
          箱の端 5% に 10% 超: {'a23': 0.12, 'a24': 0.1, 'a55': 0.1, 'a25': 0.11, 'a45': 0.11}
       1 粒子の移動: seed123 17.8/stage (5.9/次元), seed42 17.2/stage (5.7/次元), seed7 17.1/stage (5.7/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.01、中央値幅/sd 上位: a35(0.1), a55(0.1), a44(0.1)

=== CH_pilot_mut120  (3 seeds)
  seed123 beta=1.000 st= 5 acc=0.15 maxlogL=    -3.17 lnZ=    -9.68 rmse=0.0721  a35=-0.10[-0.90,+0.88]  a45=+0.48[-0.88,+1.84]
          箱の端 5% に 10% 超: {'a34': 0.11, 'a44': 0.13, 'a23': 0.14, 'a25': 0.11, 'a45': 0.11}
  seed 42 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.09 lnZ=    -9.74 rmse=0.0677  a35=-0.03[-0.88,+0.91]  a45=+0.50[-0.86,+1.83]
          箱の端 5% に 10% 超: {'a14': 0.11, 'a23': 0.12, 'a24': 0.11, 'a25': 0.1, 'a35': 0.1}
  seed  7 beta=1.000 st= 5 acc=0.15 maxlogL=    -3.14 lnZ=    -9.51 rmse=0.0680  a35=-0.06[-0.91,+0.87]  a45=+0.48[-0.83,+1.84]
          箱の端 5% に 10% 超: {'a34': 0.11, 'a23': 0.12, 'a55': 0.1, 'a25': 0.11}
       1 粒子の移動: seed123 17.5/stage (5.8/次元), seed42 17.2/stage (5.7/次元), seed7 17.9/stage (6.0/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.08、中央値幅/sd 上位: a22(0.2), a11(0.2), a24(0.2)

=== CS_pilot_gateon_mut120  (3 seeds)
  seed123 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.93 lnZ=   -15.19 rmse=0.1142  a35=+0.02[-0.89,+0.88]  a45=-0.00[-0.88,+0.90]
          箱の端 5% に 10% 超: {'a11': 0.13, 'a12': 0.13, 'a33': 0.18, 'a34': 0.1, 'a44': 0.11, 'a13': 0.31, 'a55': 0.1}
  seed 42 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.71 lnZ=   -15.21 rmse=0.1144  a35=-0.04[-0.89,+0.91]  a45=-0.03[-0.90,+0.90]
          箱の端 5% に 10% 超: {'a11': 0.1, 'a12': 0.11, 'a33': 0.18, 'a13': 0.31, 'a55': 0.1, 'a25': 0.11, 'a35': 0.1, 'a45': 0.1}
  seed  7 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.94 lnZ=   -15.03 rmse=0.1144  a35=+0.03[-0.92,+0.89]  a45=+0.02[-0.91,+0.89]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a12': 0.12, 'a33': 0.16, 'a13': 0.3, 'a24': 0.1, 'a35': 0.11}
       1 粒子の移動: seed123 16.8/stage (6.7/次元), seed42 16.8/stage (6.7/次元), seed7 16.9/stage (6.8/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.23、中央値幅/sd 上位: a24(0.2), a25(0.2), a55(0.1)

=== CS_pilot_mut120  (3 seeds)
  seed123 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.88 lnZ=   -15.22 rmse=0.1183  a35=-0.05[-0.91,+0.88]  a45=-0.03[-0.90,+0.88]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a12': 0.15, 'a33': 0.18, 'a13': 0.29, 'a24': 0.12, 'a25': 0.1}
  seed 42 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.82 lnZ=   -15.04 rmse=0.1110  a35=+0.01[-0.92,+0.92]  a45=-0.03[-0.89,+0.91]
          箱の端 5% に 10% 超: {'a11': 0.12, 'a12': 0.13, 'a33': 0.16, 'a34': 0.1, 'a13': 0.29, 'a24': 0.11, 'a55': 0.1, 'a35': 0.13, 'a45': 0.1}
  seed  7 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.95 lnZ=   -15.10 rmse=0.1137  a35=-0.07[-0.91,+0.91]  a45=+0.00[-0.89,+0.90]
          箱の端 5% に 10% 超: {'a11': 0.12, 'a12': 0.14, 'a33': 0.17, 'a34': 0.11, 'a44': 0.1, 'a13': 0.3, 'a14': 0.1, 'a15': 0.11, 'a25': 0.1, 'a35': 0.12}
       1 粒子の移動: seed123 16.7/stage (6.7/次元), seed42 16.9/stage (6.8/次元), seed7 16.9/stage (6.8/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.12、中央値幅/sd 上位: a14(0.1), a35(0.1), a33(0.1)

全群 PASS
```

## 直近の run の git

- HEAD: `ffa7047` docs: pilot と Fn ノックアウトの結果の要点（ゲートなしでも予測は a45 から出る）
- 通知を書いた時刻: 2026-10-07 11:03:50 JST
