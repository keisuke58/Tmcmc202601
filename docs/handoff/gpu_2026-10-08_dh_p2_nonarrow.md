# GPU 側 → クラウド側: run 終了の自動通知（2026-10-08 / dh_p2_nonarrow）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/DH_p2_nonarrow_*`（出力ディレクトリ 3 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| DH_p2_nonarrow_seed123 | 12 | 0.114 | 9.1/stage (7.3/次元) | -103.33 |
| DH_p2_nonarrow_seed42 | 12 | 0.119 | 9.5/stage (7.6/次元) | -103.91 |
| DH_p2_nonarrow_seed7 | 12 | 0.120 | 9.6/stage (7.7/次元) | -103.85 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-08 08:01:30,996:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
DH_p2_nonarrow_seed123                   0.00  -103.33  0.1447  OK    2.70  2.74  [ -0.82, -0.54, -0.39]   0%  [ +1.81, +4.65, +5.88]  12%
DH_p2_nonarrow_seed42                    0.00  -103.91  0.1416  OK    2.32  2.74  [ -0.74, -0.50, -0.36]   0%  [ +0.52, +3.62, +5.74]   8%
DH_p2_nonarrow_seed7                     0.00  -103.85  0.1423  OK    2.61  2.74  [ -0.76, -0.52, -0.36]   0%  [ +1.44, +4.36, +5.84]  10%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-08_dh_p2_nonarrow.csv
```

全列は `docs/handoff/gpu_2026-10-08_dh_p2_nonarrow.csv` にある。

## check_paper_runs.py の出力（そのまま）

```

=== DH_p2_nonarrow  (3 seeds)
  seed123 beta=1.000 st=12 acc=0.11 maxlogL=  -103.33 lnZ=  -130.31 rmse=0.1447  a35=-5.31[-12.48,-1.21]  a45=+4.65[+1.81,+5.88]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.86, 'a12': 0.15, 'a14': 0.16, 'a45': 0.12}
  seed 42 beta=1.000 st=12 acc=0.12 maxlogL=  -103.91 lnZ=  -131.78 rmse=0.1416  a35=-6.63[-13.19,-1.86]  a45=+3.62[+0.52,+5.74]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.88, 'a12': 0.14, 'a14': 0.18}
  seed  7 beta=1.000 st=12 acc=0.12 maxlogL=  -103.85 lnZ=  -131.02 rmse=0.1423  a35=-6.53[-13.84,-1.26]  a45=+4.36[+1.44,+5.84]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a11': 0.87, 'a12': 0.12, 'a14': 0.21, 'a45': 0.1}
       1 粒子の移動: seed123 9.1/stage (7.3/次元), seed42 9.5/stage (7.6/次元), seed7 9.6/stage (7.7/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  PASS 3 maxlogL 幅<=1
  FAIL 4 中央値幅<=0.5sd
       maxlogL 幅 = 0.58、中央値幅/sd 上位: a45(0.7), a15(0.7), a55(0.6)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `7f0b4e3` docs: run 終了の自動通知（2026-10-08）— FAIL を含む群は回し直すまで解釈しない
- 通知を書いた時刻: 2026-10-08 08:01:41 JST
