# GPU 側 → クラウド側: max logL の磨き直し（2026-10-14 / CS_p1_2c_raw_seed7）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CS_p1_2c_raw_seed7`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CS_p1_2c_raw_seed7                           -4.007     -3.866   0.142   300   no


JSON: docs/handoff/gpu_2026-10-14_polish_CS_p1_2c_raw_seed7.json
```

- HEAD: ``、ホスト vancouver03、2026-10-14 09:31:51 JST
