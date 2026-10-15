# GPU 側 → クラウド側: max logL の磨き直し（2026-10-15 / cs_p2_2c_raw_s123）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CS_p2_2c_raw_seed123`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CS_p2_2c_raw_seed123                        -32.999    -32.529   0.470   300   no


JSON: docs/handoff/gpu_2026-10-15_polish_cs_p2_2c_raw_s123.json
```

- HEAD: `ab36297`、ホスト stuttgart03、2026-10-15 22:42:45 JST
