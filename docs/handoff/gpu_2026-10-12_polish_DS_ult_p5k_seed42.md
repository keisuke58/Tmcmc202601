# GPU 側 → クラウド側: max logL の磨き直し（2026-10-12 / DS_ult_p5k_seed42）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_ult_wide80_p18k_wide2_sd4_seed42`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_ult_wide80_p18k_wide2_sd4_seed42          -7.302     -5.597   1.705   300   no


JSON: docs/handoff/gpu_2026-10-12_polish_DS_ult_p5k_seed42.json
```

- HEAD: `53666ba`、ホスト stuttgart01、2026-10-12 12:49:40 JST
