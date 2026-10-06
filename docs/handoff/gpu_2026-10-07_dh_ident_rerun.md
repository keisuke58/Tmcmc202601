# GPU 側 → クラウド側: run 終了の自動通知（2026-10-07 / dh_ident_rerun）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/*ident*mut80*`（出力ディレクトリ 6 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DH_ident_prior0_mut80_seed123 | 10 | 0.114 | 9.1/stage (6.1/次元) | -5.03 |
| DH_ident_prior0_mut80_seed42 | 11 | 0.105 | 8.4/stage (6.2/次元) | -5.81 |
| DH_ident_prior0_mut80_seed7 | 11 | 0.111 | 8.9/stage (6.5/次元) | -5.05 |
| DH_ident_prior6_mut80_seed123 | 10 | 0.134 | 10.7/stage (7.2/次元) | -4.35 |
| DH_ident_prior6_mut80_seed42 | 10 | 0.130 | 10.4/stage (6.9/次元) | -4.51 |
| DH_ident_prior6_mut80_seed7 | 10 | 0.128 | 10.3/stage (6.8/次元) | -4.43 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-07 02:13:43,304:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DH_ident_prior0_mut80_seed123            0.00    -5.03  0.0507  OK    2.20  2.74  [ +0.60, +5.54, +9.69]   0%  [ -0.28,+12.27,+19.07]  10%
DH_ident_prior0_mut80_seed42             0.00    -5.81  0.0668  OK    2.28  2.74  [-13.93, +5.59, +9.85]  10%  [ +0.15, +8.34,+17.05]   2%
DH_ident_prior0_mut80_seed7              0.00    -5.05  0.0656  OK    3.12  2.74  [ -1.13, +6.58,+11.60]   0%  [ +0.18,+11.02,+19.02]   9%
DH_ident_prior6_mut80_seed123            0.00    -4.35  0.0554  OK    2.72  2.74  [ -7.17, -1.82, +3.03]   0%  [ -1.92, +1.46,+10.71]   0%
DH_ident_prior6_mut80_seed42             0.00    -4.51  0.0641  OK    3.03  2.74  [ -8.39, -0.18, +2.77]   1%  [ -3.26, +2.31, +7.96]   0%
DH_ident_prior6_mut80_seed7              0.00    -4.43  0.0572  OK    2.73  2.74  [-10.64, -0.41, +3.13]   1%  [ -2.33, +3.66,+10.43]   0%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-07_dh_ident_rerun.csv
```

全列は `docs/handoff/gpu_2026-10-07_dh_ident_rerun.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== DH_ident_prior0_mut80  (3 seeds)
  seed123 beta=1.000 st=10 acc=0.11 maxlogL=    -5.03 lnZ=   -27.79 rmse=0.0507  a35=+3.73[-1.05,+8.51]  a45=+12.27[-0.28,+19.07]
          箱の端 5% に 10% 超: {'a44': 0.1, 'a24': 0.24, 'a55': 0.15}
  seed 42 beta=1.000 st=11 acc=0.10 maxlogL=    -5.81 lnZ=   -29.34 rmse=0.0668  a35=+2.73[-8.70,+8.29]  a45=+8.34[+0.15,+17.05]
          箱の端 5% に 10% 超: {'a11': 0.12, 'a12': 0.12, 'a44': 0.19, 'a24': 0.21, 'a55': 0.22}
  seed  7 beta=1.000 st=11 acc=0.11 maxlogL=    -5.05 lnZ=   -28.45 rmse=0.0656  a35=+4.30[-5.06,+10.32]  a45=+11.02[+0.18,+19.02]
          箱の端 5% に 10% 超: {'a24': 0.17, 'a55': 0.15}
       1 粒子の移動: seed123 9.1/stage (6.1/次元), seed42 8.4/stage (6.2/次元), seed7 8.9/stage (6.5/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.78、中央値幅/sd 上位: a22(1.1), a14(0.8), a45(0.7)

=== DH_ident_prior6_mut80  (3 seeds)
  seed123 beta=1.000 st=10 acc=0.13 maxlogL=    -4.35 lnZ=   -23.34 rmse=0.0554  a35=-2.78[-11.28,+2.01]  a45=+1.46[-1.92,+10.71]
  seed 42 beta=1.000 st=10 acc=0.13 maxlogL=    -4.51 lnZ=   -24.96 rmse=0.0641  a35=-1.78[-6.52,+1.63]  a45=+2.31[-3.26,+7.96]
  seed  7 beta=1.000 st=10 acc=0.13 maxlogL=    -4.43 lnZ=   -25.37 rmse=0.0572  a35=-1.01[-6.55,+2.70]  a45=+3.66[-2.33,+10.43]
       1 粒子の移動: seed123 10.7/stage (7.2/次元), seed42 10.4/stage (6.9/次元), seed7 10.3/stage (6.8/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.16、中央値幅/sd 上位: a55(1.2), a15(1.1), a25(0.8)

FAIL 5 DH_ident_prior0_mut80 の max logL -5.03 >= DH_ident_prior6_mut80 の -4.35 − 0.5  → 事前分布なしは探索不足。解釈しない

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `8e2e685` docs: run 終了の自動通知（2026-10-06）— （判定スクリプトが結論行を出さなかった。出力をそのまま読むこと）
- 通知を書いた時刻: 2026-10-07 02:14:06 JST
