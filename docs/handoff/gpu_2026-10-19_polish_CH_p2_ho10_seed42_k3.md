# GPU 側 → クラウド側: max logL の磨き直し（2026-10-19 / CH_p2_ho10_seed42_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CH_p2_2c_raw_a33w_ho10_seed42`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CH_p2_2c_raw_a33w_ho10_seed42               -21.497    -20.487   1.010   575  yes


JSON: docs/handoff/gpu_2026-10-19_polish_CH_p2_ho10_seed42_k3.json
```

- HEAD: `5cbe865`、ホスト vancouver01、2026-10-20 02:37:07 JST
