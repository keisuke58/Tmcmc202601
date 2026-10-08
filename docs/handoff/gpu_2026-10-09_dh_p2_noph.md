# GPU 側 → クラウド側: run 終了の自動通知（2026-10-09 / dh_p2_noph）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DH_p2_nonarrow_a45_noph_seed*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DH_p2_nonarrow_a45_noph_seed123 | 13 | 0.106 | 8.5/stage (7.4/次元) | -15.40 |
| DH_p2_nonarrow_a45_noph_seed42 | 13 | 0.109 | 8.7/stage (7.5/次元) | -16.99 |
| DH_p2_nonarrow_a45_noph_seed7 | 13 | 0.096 | 7.7/stage (6.6/次元) | -15.94 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-09 07:39:16,470:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DH_p2_nonarrow_a45_noph_seed123          0.00   -15.40  0.0863  OK    3.02  2.74  [ -1.13, -0.81, -0.55]   0%  [ -0.16, +0.73, +7.53]  37%
DH_p2_nonarrow_a45_noph_seed42           0.00   -16.99  0.0887  OK    3.04  2.74  [ -1.12, -0.99, -0.57]   0%  [ +0.71, +3.38, +6.06]   4%
DH_p2_nonarrow_a45_noph_seed7            0.00   -15.94  0.0867  OK    3.16  2.74  [ -1.24, -0.82, -0.60]   0%  [ +0.16, +1.16,+13.00]  21%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-09_dh_p2_noph.csv
```

全列は `docs/handoff/gpu_2026-10-09_dh_p2_noph.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== DH_p2_nonarrow_a45_noph  (3 seeds)
  seed123 beta=1.000 st=13 acc=0.11 maxlogL=   -15.40 lnZ=   -46.60 rmse=0.0863  a35=-4.31[-8.86,-1.84]  a45=+0.73[-0.16,+7.53]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a12': 0.37, 'a22': 0.16, 'a44': 0.36, 'a15': 0.18, 'a45': 0.37}
          片側 5%: 成分 箱 下側/上側（前段 DH_p1_mut80_seed123 の 下側/上側・尤度が違う段なので増加は見ない）
            a12  [0.5,3] 0.37/0.00 (前段 0.08/0.02)  ← 箱で切られている
            a45  [-0.5,20] 0.37/0.00 (前段 0.01/0.02)  ← 箱で切られている
            a44  [0,5] 0.34/0.02 (前段 0.15/0.00)  ← 箱で切られている
            a15  [-0.5,2.5] 0.18/0.00 (前段 0.06/0.03)
            a22  [0,5] 0.16/0.00 (前段 0.11/0.00)
  seed 42 beta=1.000 st=13 acc=0.11 maxlogL=   -16.99 lnZ=   -49.09 rmse=0.0887  a35=-10.18[-12.11,-1.41]  a45=+3.38[+0.71,+6.06]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a22': 0.37}
          片側 5%: 成分 箱 下側/上側（前段 DH_p1_mut80_seed42 の 下側/上側・尤度が違う段なので増加は見ない）
            a22  [0,5] 0.37/0.00 (前段 0.12/0.00)  ← 箱で切られている
  seed  7 beta=1.000 st=13 acc=0.10 maxlogL=   -15.94 lnZ=   -48.00 rmse=0.0867  a35=-4.42[-8.26,-3.05]  a45=+1.16[+0.16,+13.00]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a22': 0.58, 'a44': 0.23, 'a14': 0.58, 'a55': 0.21, 'a45': 0.21}
          片側 5%: 成分 箱 下側/上側（前段 DH_p1_mut80_seed7 の 下側/上側・尤度が違う段なので増加は見ない）
            a22  [0,5] 0.58/0.00 (前段 0.10/0.00)  ← 箱で切られている
            a14  [-3,3] 0.58/0.00 (前段 0.07/0.01)  ← 箱で切られている
            a44  [0,5] 0.22/0.00 (前段 0.15/0.00)  ← 箱で切られている
            a45  [-0.5,20] 0.21/0.00 (前段 0.01/0.02)  ← 箱で切られている
            a55  [-1.5,5] 0.02/0.19 (前段 0.08/0.02)
       1 粒子の移動: seed123 8.5/stage (7.4/次元), seed42 8.7/stage (7.5/次元), seed7 7.7/stage (6.6/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  FAIL 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 1.59、中央値幅/sd 上位: a15(2.3), a12(2.2), a14(1.9)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `52782f6` docs: run 終了の自動通知（2026-10-09）— 全群 PASS
- 通知を書いた時刻: 2026-10-09 07:39:24 JST
