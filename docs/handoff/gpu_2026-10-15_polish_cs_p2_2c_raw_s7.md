# GPU 側 → クラウド側: max logL の磨き直し（2026-10-15 / cs_p2_2c_raw_s7）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CS_p2_2c_raw_seed7`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CS_p2_2c_raw_seed7                          -33.023    -32.735   0.288   300   no


JSON: docs/handoff/gpu_2026-10-15_polish_cs_p2_2c_raw_s7.json
```

- HEAD: `9d4aca4`、ホスト vancouver02、2026-10-15 21:42:59 JST
