# GPU 側 → クラウド側: max logL の磨き直し（2026-10-15 / cs_p2_2c_raw_s42）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CS_p2_2c_raw_seed42`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CS_p2_2c_raw_seed42                         -33.037    -32.585   0.452   300   no


JSON: docs/handoff/gpu_2026-10-15_polish_cs_p2_2c_raw_s42.json
```

- HEAD: `852b02f`、ホスト vancouver01、2026-10-15 21:55:45 JST
