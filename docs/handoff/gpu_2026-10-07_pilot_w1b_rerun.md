# GPU 側 → クラウド側: run 終了の自動通知（2026-10-07 / pilot_w1b_rerun）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/*pilot*mut80*`（出力ディレクトリ 24 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CH_pilot_gateon_mut80_seed123 | 5 | 0.147 | 11.8/stage (3.9/次元) | -3.19 |
| CH_pilot_gateon_mut80_seed42 | 5 | 0.139 | 11.2/stage (3.7/次元) | -3.21 |
| CH_pilot_gateon_mut80_seed7 | 5 | 0.143 | 11.4/stage (3.8/次元) | -3.18 |
| CH_pilot_mut80_seed123 | 5 | 0.144 | 11.5/stage (3.8/次元) | -3.26 |
| CH_pilot_mut80_seed42 | 5 | 0.143 | 11.5/stage (3.8/次元) | -3.29 |
| CH_pilot_mut80_seed7 | 5 | 0.145 | 11.6/stage (3.9/次元) | -3.19 |
| CS_pilot_gateon_mut80_seed123 | 6 | 0.139 | 11.1/stage (4.5/次元) | -5.91 |
| CS_pilot_gateon_mut80_seed42 | 6 | 0.139 | 11.1/stage (4.4/次元) | -5.86 |
| CS_pilot_gateon_mut80_seed7 | 6 | 0.138 | 11.1/stage (4.4/次元) | -5.75 |
| CS_pilot_mut80_seed123 | 6 | 0.138 | 11.0/stage (4.4/次元) | -5.86 |
| CS_pilot_mut80_seed42 | 6 | 0.139 | 11.1/stage (4.4/次元) | -5.97 |
| CS_pilot_mut80_seed7 | 6 | 0.140 | 11.2/stage (4.5/次元) | -5.97 |
| DH_pilot_gateon_mut80_seed123 | 9 | 0.123 | 9.8/stage (5.9/次元) | -4.40 |
| DH_pilot_gateon_mut80_seed42 | 9 | 0.123 | 9.8/stage (5.9/次元) | -4.09 |
| DH_pilot_gateon_mut80_seed7 | 9 | 0.120 | 9.6/stage (5.7/次元) | -3.96 |
| DH_pilot_mut80_seed123 | 10 | 0.124 | 9.9/stage (6.6/次元) | -4.32 |
| DH_pilot_mut80_seed42 | 10 | 0.123 | 9.8/stage (6.6/次元) | -4.44 |
| DH_pilot_mut80_seed7 | 10 | 0.126 | 10.0/stage (6.7/次元) | -3.92 |
| DS_pilot_gateon_mut80_seed123 | 8 | 0.139 | 11.1/stage (5.9/次元) | -1.41 |
| DS_pilot_gateon_mut80_seed42 | 8 | 0.134 | 10.7/stage (5.7/次元) | -1.10 |
| DS_pilot_gateon_mut80_seed7 | 9 | 0.130 | 10.4/stage (6.2/次元) | -1.23 |
| DS_pilot_mut80_seed123 | 9 | 0.134 | 10.7/stage (6.4/次元) | -1.09 |
| DS_pilot_mut80_seed42 | 9 | 0.132 | 10.5/stage (6.3/次元) | -1.24 |
| DS_pilot_mut80_seed7 | 9 | 0.131 | 10.5/stage (6.3/次元) | -0.85 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
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
CH_pilot_gateon_mut80_seed123            0.05    -3.19  0.0715  OK    1.00  1.00  [ -0.93, -0.16, +1.02]  10%  [ -0.85, +0.53, +1.85]  10%
CH_pilot_gateon_mut80_seed42             0.05    -3.21  0.0744  OK    1.00  1.00  [ -0.91, -0.21, +0.98]   8%  [ -0.85, +0.44, +1.84]  10%
CH_pilot_gateon_mut80_seed7              0.05    -3.18  0.0668  OK    1.00  1.00  [ -0.90, -0.19, +0.95]   7%  [ -0.82, +0.66, +1.88]  10%
CH_pilot_mut80_seed123                   0.00    -3.26  0.0689  OK    1.01  1.00  [ -0.91, -0.15, +1.06]   7%  [ -0.84, +0.42, +1.83]   9%
CH_pilot_mut80_seed42                    0.00    -3.29  0.0720  OK    0.98  1.00  [ -0.93, -0.18, +1.08]   9%  [ -0.86, +0.48, +1.85]  11%
CH_pilot_mut80_seed7                     0.00    -3.19  0.0688  OK    1.00  1.00  [ -0.92, -0.19, +0.99]   8%  [ -0.86, +0.49, +1.88]  12%
CS_pilot_gateon_mut80_seed123            0.05    -5.91  0.1124  OK    0.98  1.02  [ -0.96, -0.59, +0.16]  16%  [ -0.89, +0.05, +0.92]  10%
CS_pilot_gateon_mut80_seed42             0.05    -5.86  0.1149  OK    0.99  1.02  [ -0.97, -0.60, +0.16]  16%  [ -0.92, +0.01, +0.87]  10%
CS_pilot_gateon_mut80_seed7              0.05    -5.75  0.1159  OK    0.98  1.02  [ -0.97, -0.60, +0.21]  17%  [ -0.91, -0.01, +0.90]  11%
CS_pilot_mut80_seed123                   0.00    -5.86  0.1181  OK    0.99  1.02  [ -0.96, -0.62, +0.19]  16%  [ -0.88, +0.07, +0.89]   9%
CS_pilot_mut80_seed42                    0.00    -5.97  0.1166  OK    0.95  1.02  [ -0.97, -0.60, +0.13]  16%  [ -0.90, -0.07, +0.88]   9%
CS_pilot_mut80_seed7                     0.00    -5.97  0.1200  OK    0.99  1.02  [ -0.96, -0.62, +0.13]  19%  [ -0.90, -0.04, +0.90]  10%
DH_pilot_gateon_mut80_seed123            0.05    -4.40  0.0590  OK    2.31  2.74  [ -0.01, +0.50, +0.96]   0%  [ +0.94, +2.86, +5.12]   2%
DH_pilot_gateon_mut80_seed42             0.05    -4.09  0.0565  OK    2.66  2.74  [ +0.02, +0.49, +0.97]   0%  [ +0.76, +2.89, +4.96]   1%
DH_pilot_gateon_mut80_seed7              0.05    -3.96  0.0624  OK    2.34  2.74  [ +0.06, +0.56, +0.99]   0%  [ +0.74, +2.99, +5.03]   2%
DH_pilot_mut80_seed123                   0.00    -4.32  0.0651  OK    2.87  2.74  [ +0.04, +0.54, +1.00]   0%  [ +0.47, +3.06, +5.47]   4%
DH_pilot_mut80_seed42                    0.00    -4.44  0.0576  OK    2.43  2.74  [ +0.04, +0.54, +1.00]   0%  [ +1.10, +3.15, +5.32]   3%
DH_pilot_mut80_seed7                     0.00    -3.92  0.0618  OK    2.83  2.74  [ -0.13, +0.53, +0.98]   0%  [ +0.94, +2.90, +5.22]   2%
DS_pilot_gateon_mut80_seed123            0.05    -1.41  0.0207  OK    1.13  1.31  [ +1.00, +1.05, +1.20]  74%  [ +1.44, +2.16, +2.47]  26%
DS_pilot_gateon_mut80_seed42             0.05    -1.10  0.0169  OK    1.18  1.31  [ +1.00, +1.07, +1.22]  66%  [ +1.33, +2.16, +2.47]  28%
DS_pilot_gateon_mut80_seed7              0.05    -1.23  0.0210  OK    1.14  1.31  [ +1.00, +1.06, +1.21]  70%  [ +1.33, +2.14, +2.47]  27%
DS_pilot_mut80_seed123                   0.00    -1.09  0.0196  OK    1.18  1.31  [ +1.00, +1.06, +1.23]  66%  [ +1.65, +2.22, +2.47]  34%
DS_pilot_mut80_seed42                    0.00    -1.24  0.0217  OK    1.32  1.31  [ +1.00, +1.06, +1.21]  67%  [ +1.59, +2.21, +2.48]  34%
DS_pilot_mut80_seed7                     0.00    -0.85  0.0185  OK    1.32  1.31  [ +1.01, +1.06, +1.22]  70%  [ +1.60, +2.22, +2.48]  33%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-07_pilot_w1b_rerun.csv
```

全列は `docs/handoff/gpu_2026-10-07_pilot_w1b_rerun.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== CH_pilot_gateon_mut80  (3 seeds)
  seed123 beta=1.000 st= 5 acc=0.15 maxlogL=    -3.19 lnZ=    -9.72 rmse=0.0715  a35=+0.06[-0.87,+0.89]  a45=+0.53[-0.85,+1.85]
          箱の端 5% に 10% 超: {'a44': 0.11, 'a14': 0.1, 'a23': 0.11, 'a55': 0.1}
  seed 42 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.21 lnZ=    -9.65 rmse=0.0744  a35=+0.02[-0.88,+0.91]  a45=+0.44[-0.85,+1.84]
          箱の端 5% に 10% 超: {'a34': 0.11, 'a23': 0.13, 'a55': 0.11, 'a25': 0.1}
  seed  7 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.18 lnZ=    -9.44 rmse=0.0668  a35=-0.06[-0.89,+0.85]  a45=+0.66[-0.82,+1.88]
          箱の端 5% に 10% 超: {'a44': 0.11, 'a23': 0.11, 'a15': 0.1}
       1 粒子の移動: seed123 11.8/stage (3.9/次元), seed42 11.2/stage (3.7/次元), seed7 11.4/stage (3.8/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.04、中央値幅/sd 上位: a45(0.2), a34(0.2), a35(0.2)

=== CH_pilot_mut80  (3 seeds)
  seed123 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.26 lnZ=    -9.66 rmse=0.0689  a35=-0.09[-0.92,+0.88]  a45=+0.42[-0.84,+1.83]
          箱の端 5% に 10% 超: {'a44': 0.1, 'a14': 0.1, 'a23': 0.1, 'a55': 0.1}
  seed 42 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.29 lnZ=    -9.71 rmse=0.0720  a35=-0.01[-0.92,+0.91]  a45=+0.48[-0.86,+1.85]
          箱の端 5% に 10% 超: {'a14': 0.11, 'a23': 0.13, 'a35': 0.11, 'a45': 0.11}
  seed  7 beta=1.000 st= 5 acc=0.14 maxlogL=    -3.19 lnZ=    -9.47 rmse=0.0688  a35=-0.04[-0.91,+0.88]  a45=+0.49[-0.86,+1.88]
          箱の端 5% に 10% 超: {'a14': 0.1, 'a23': 0.11, 'a55': 0.11, 'a25': 0.1, 'a45': 0.12}
       1 粒子の移動: seed123 11.5/stage (3.8/次元), seed42 11.5/stage (3.8/次元), seed7 11.6/stage (3.9/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.10、中央値幅/sd 上位: a55(0.1), a35(0.1), a14(0.1)

=== CS_pilot_gateon_mut80  (3 seeds)
  seed123 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.91 lnZ=   -15.31 rmse=0.1124  a35=+0.00[-0.89,+0.89]  a45=+0.05[-0.89,+0.92]
          箱の端 5% に 10% 超: {'a12': 0.12, 'a33': 0.16, 'a34': 0.1, 'a13': 0.26, 'a24': 0.11, 'a55': 0.11, 'a45': 0.1}
  seed 42 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.86 lnZ=   -15.28 rmse=0.1149  a35=-0.03[-0.92,+0.88]  a45=+0.01[-0.92,+0.87]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a12': 0.14, 'a33': 0.16, 'a13': 0.28, 'a14': 0.11, 'a25': 0.11, 'a45': 0.1}
  seed  7 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.75 lnZ=   -15.08 rmse=0.1159  a35=-0.02[-0.91,+0.87]  a45=-0.01[-0.91,+0.90]
          箱の端 5% に 10% 超: {'a11': 0.13, 'a12': 0.13, 'a33': 0.17, 'a44': 0.11, 'a13': 0.29, 'a14': 0.1, 'a55': 0.1, 'a25': 0.1, 'a45': 0.11}
       1 粒子の移動: seed123 11.1/stage (4.5/次元), seed42 11.1/stage (4.4/次元), seed7 11.1/stage (4.4/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.16、中央値幅/sd 上位: a55(0.2), a44(0.1), a15(0.1)

=== CS_pilot_mut80  (3 seeds)
  seed123 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.86 lnZ=   -15.21 rmse=0.1181  a35=-0.09[-0.89,+0.89]  a45=+0.07[-0.88,+0.89]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a12': 0.13, 'a33': 0.16, 'a13': 0.31, 'a24': 0.1}
  seed 42 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.97 lnZ=   -15.21 rmse=0.1166  a35=-0.10[-0.90,+0.87]  a45=-0.07[-0.90,+0.88]
          箱の端 5% に 10% 超: {'a11': 0.13, 'a12': 0.14, 'a33': 0.16, 'a44': 0.11, 'a13': 0.29}
  seed  7 beta=1.000 st= 6 acc=0.14 maxlogL=    -5.97 lnZ=   -15.06 rmse=0.1200  a35=+0.01[-0.89,+0.90]  a45=-0.04[-0.90,+0.90]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a12': 0.12, 'a33': 0.19, 'a34': 0.1, 'a44': 0.11, 'a13': 0.3, 'a14': 0.11, 'a25': 0.1, 'a45': 0.1}
       1 粒子の移動: seed123 11.0/stage (4.4/次元), seed42 11.1/stage (4.4/次元), seed7 11.2/stage (4.5/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  FAIL 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.11、中央値幅/sd 上位: a45(0.3), a35(0.2), a24(0.2)

=== DH_pilot_gateon_mut80  (3 seeds)
  seed123 beta=1.000 st= 9 acc=0.12 maxlogL=    -4.40 lnZ=   -20.75 rmse=0.0590  a35=+0.70[-0.34,+2.96]  a45=+2.86[+0.94,+5.12]
          箱の端 5% に 10% 超: {'a11': 0.1, 'a22': 0.25, 'a34': 0.24, 'a44': 0.2, 'a55': 0.11, 'a15': 0.12, 'a35': 0.1}
  seed 42 beta=1.000 st= 9 acc=0.12 maxlogL=    -4.09 lnZ=   -20.61 rmse=0.0565  a35=+0.74[-0.34,+2.89]  a45=+2.89[+0.76,+4.96]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a22': 0.26, 'a34': 0.25, 'a44': 0.2, 'a35': 0.11}
  seed  7 beta=1.000 st= 9 acc=0.12 maxlogL=    -3.96 lnZ=   -20.99 rmse=0.0624  a35=+0.85[-0.30,+3.30]  a45=+2.99[+0.74,+5.03]
          箱の端 5% に 10% 超: {'a12': 0.11, 'a22': 0.23, 'a34': 0.26, 'a44': 0.17, 'a55': 0.1, 'a15': 0.11}
       1 粒子の移動: seed123 9.8/stage (5.9/次元), seed42 9.8/stage (5.9/次元), seed7 9.6/stage (5.7/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.44、中央値幅/sd 上位: a11(0.5), a24(0.3), a12(0.3)

=== DH_pilot_mut80  (3 seeds)
  seed123 beta=1.000 st=10 acc=0.12 maxlogL=    -4.32 lnZ=   -20.89 rmse=0.0651  a35=+0.33[-0.38,+1.31]  a45=+3.06[+0.47,+5.47]
          箱の端 5% に 10% 超: {'a22': 0.3, 'a34': 0.26, 'a44': 0.24, 'a35': 0.13}
  seed 42 beta=1.000 st=10 acc=0.12 maxlogL=    -4.44 lnZ=   -21.64 rmse=0.0576  a35=+0.32[-0.42,+1.32]  a45=+3.15[+1.10,+5.32]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a12': 0.12, 'a22': 0.3, 'a34': 0.25, 'a44': 0.19, 'a35': 0.15}
  seed  7 beta=1.000 st=10 acc=0.13 maxlogL=    -3.92 lnZ=   -21.54 rmse=0.0618  a35=+0.31[-0.39,+1.25]  a45=+2.90[+0.94,+5.22]
          箱の端 5% に 10% 超: {'a22': 0.29, 'a34': 0.23, 'a44': 0.2, 'a35': 0.15}
       1 粒子の移動: seed123 9.9/stage (6.6/次元), seed42 9.8/stage (6.6/次元), seed7 10.0/stage (6.7/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.51、中央値幅/sd 上位: a45(0.2), a13(0.2), a23(0.2)

=== DS_pilot_gateon_mut80  (3 seeds)
  seed123 beta=1.000 st= 8 acc=0.14 maxlogL=    -1.41 lnZ=   -18.27 rmse=0.0207  a35=+1.36[+0.88,+1.94]  a45=+2.16[+1.44,+2.47]
          箱の端 5% に 10% 超: {'a11': 0.11, 'a22': 0.1, 'a33': 0.73, 'a34': 0.54, 'a14': 0.11, 'a24': 0.18, 'a15': 0.1, 'a25': 0.15, 'a45': 0.26}
  seed 42 beta=1.000 st= 8 acc=0.13 maxlogL=    -1.10 lnZ=   -25.34 rmse=0.0169  a35=+1.36[+0.90,+1.96]  a45=+2.16[+1.33,+2.47]
          箱の端 5% に 10% 超: {'a11': 0.1, 'a12': 0.11, 'a33': 0.66, 'a34': 0.53, 'a14': 0.12, 'a24': 0.19, 'a15': 0.1, 'a25': 0.17, 'a45': 0.28}
  seed  7 beta=1.000 st= 9 acc=0.13 maxlogL=    -1.23 lnZ=   -25.50 rmse=0.0210  a35=+1.34[+0.88,+1.96]  a45=+2.14[+1.33,+2.47]
          箱の端 5% に 10% 超: {'a11': 0.1, 'a33': 0.7, 'a34': 0.54, 'a14': 0.11, 'a24': 0.18, 'a25': 0.16, 'a45': 0.27}
       1 粒子の移動: seed123 11.1/stage (5.9/次元), seed42 10.7/stage (5.7/次元), seed7 10.4/stage (6.2/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.32、中央値幅/sd 上位: a33(0.2), a22(0.2), a12(0.2)

=== DS_pilot_mut80  (3 seeds)
  seed123 beta=1.000 st= 9 acc=0.13 maxlogL=    -1.09 lnZ=   -17.79 rmse=0.0196  a35=+1.16[+0.72,+1.69]  a45=+2.22[+1.65,+2.47]
          箱の端 5% に 10% 超: {'a33': 0.66, 'a34': 0.48, 'a14': 0.11, 'a24': 0.14, 'a25': 0.12, 'a45': 0.34}
  seed 42 beta=1.000 st= 9 acc=0.13 maxlogL=    -1.24 lnZ=   -20.19 rmse=0.0217  a35=+1.18[+0.76,+1.71]  a45=+2.21[+1.59,+2.48]
          箱の端 5% に 10% 超: {'a33': 0.67, 'a34': 0.5, 'a24': 0.16, 'a15': 0.11, 'a25': 0.13, 'a45': 0.34}
  seed  7 beta=1.000 st= 9 acc=0.13 maxlogL=    -0.85 lnZ=   -18.97 rmse=0.0185  a35=+1.13[+0.70,+1.70]  a45=+2.22[+1.60,+2.48]
          箱の端 5% に 10% 超: {'a11': 0.1, 'a33': 0.7, 'a34': 0.46, 'a14': 0.11, 'a24': 0.15, 'a25': 0.14, 'a45': 0.33}
       1 粒子の移動: seed123 10.7/stage (6.4/次元), seed42 10.5/stage (6.3/次元), seed7 10.5/stage (6.3/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  PASS 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.38、中央値幅/sd 上位: a11(0.3), a13(0.3), a14(0.2)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `726b39d` docs: run 終了の自動通知（2026-10-07）— FAIL を含む群は回し直すまで解釈しない
- 通知を書いた時刻: 2026-10-07 02:15:04 JST
