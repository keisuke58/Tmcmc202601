# GPU 側 → クラウド側: run 終了の自動通知（2026-10-15 / 2c_raw_p2_CH_cold）

`tools/notify_cloud_runs.sh` が PBS の依存ジョブとして自動で書いた。
監視対象: `data_5species/main/_runs/paper_gateoff/CH_p2_2c_raw_cold_seed*`（出力ディレクトリ 1 個）

## 結論

**FAIL を含む群は回し直すまで解釈しない**

## 今回の run

| run | stages | 受理率 | 1 粒子の移動 | max logL |
|---|---|---|---|---|
| CH_p2_2c_raw_cold_seed42 | 10 | 0.136 | 20.3/stage (13.6/次元) | -26.74 |

判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。

## データの出どころ（13j §4）

| run | data_source | two_channel | data_dir_for（experiment_data） | ファイル数 |
|---|---|---|---|---|
| CH_p2_2c_raw_cold_seed42 | raw | True | `data_5species/raw_dataset/experiment_data` | 10 |

## 回収した数字（tools/eval_paper_runs.py）

RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。
**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。

```
ERROR:2026-10-15 04:24:31,363:jax._src.xla_bridge:444: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
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
CH_p2_2c_raw_cold_seed42                 0.00   -26.74  0.0640  OK    0.79  0.00  [ -0.97, -0.65, +0.07]  20%  [ -0.81, +0.57, +1.86]   9%

CSV: /home/nishioka/Tmcmc202601/docs/handoff/gpu_2026-10-15_2c_raw_p2_CH_cold.csv
```

全列は `docs/handoff/gpu_2026-10-15_2c_raw_p2_CH_cold.csv` にある。

## ψ_i とデータの ψ_i^data（tools/psi_table_runs.py、13b §4）

run: `CH_p2_2c_raw_cold_seed*`。各セルは ψ の Day 3–21 の中央値。data = ψ^data（括弧は使えた日数）、MAP = max logL の seed、事後 = 3 seed の粒子 200 本の中央値 [5%, 95%]。

| 条件 | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|
| CH data | 0.35 (5) | 0.24 (5) | 0.53 (5) | -- (0) | -- (0) |
| CH MAP | 0.38 | 0.88 | 0.90 | 0.06 | 0.05 |
| CH 事後 | 0.38 [0.29, 0.46] | 0.89 [0.85, 0.90] | 0.90 [0.87, 0.91] | 0.06 [0.05, 0.09] | 0.06 [0.05, 0.08] |

seed ごとの MAP（日の中央値）:

| run | max logL | So | An | Vei | Fn | Pg |
|---|---|---|---|---|---|---|
| CH_p2_2c_raw_cold_seed42 | -26.74 | 0.38 | 0.88 | 0.90 | 0.06 | 0.05 |

## check_paper_runs.py の出力（そのまま）

```

=== CH_p2_2c_raw_cold  (1 seeds)
  seed 42 beta=1.000 st=10 acc=0.14 maxlogL=   -26.74 lnZ=   -44.75 rmse=0.1036  a35=-2.56[-4.75,+0.50]  a45=+0.57[-0.81,+1.86]
          箱の端 5% に 10% 超（一様なら 0.10）: {'a33': 0.2, 'a55': 0.11}
          片側 5%: 成分 箱 下側/上側
            a33  [-1,1.5] 0.20/0.00
       seed が足りない（1 run だけ。判定 3・4 は seed がそろうまで FAIL）
       1 粒子の移動: seed42 20.3/stage (13.6/次元)
  PASS 0 logL と粒子の対応
  PASS 1 beta=1
  PASS 2 粒子の移動(>=2/stage)
  PASS 2b 次元あたりの移動(>=5)
  FAIL 3 maxlogL 幅<=1（粒子の最良値、polish.json なし）
  FAIL 4 中央値幅<=0.5sd（同定された成分・多峰は山ごと）
       maxlogL 幅 = 0.00、中央値幅/sd 上位: a11(0.00), a12(0.00), a22(0.00)
       同定されていない（判定 4 から外した）: a34(sd/一様sd 0.94, 幅/sd 0.00), a44(sd/一様sd 0.99, 幅/sd 0.00), a14(sd/一様sd 0.89, 幅/sd 0.00), a24(sd/一様sd 0.97, 幅/sd 0.00), a55(sd/一様sd 1.01, 幅/sd 0.00), a15(sd/一様sd 0.95, 幅/sd 0.00), a25(sd/一様sd 0.97, 幅/sd 0.00), a35(sd/一様sd 0.95, 幅/sd 0.00), a45(sd/一様sd 0.99, 幅/sd 0.00)

FAIL を含む群は回し直すまで解釈しない
```

## 直近の run の git

- HEAD: `f60ce8a` docs: run 終了の自動通知（2026-10-15）— 全群 PASS
- 通知を書いた時刻: 2026-10-15 04:26:58 JST
