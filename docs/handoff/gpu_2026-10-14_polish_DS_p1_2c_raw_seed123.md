# GPU 側 → クラウド側: max logL の磨き直し（2026-10-14 / DS_p1_2c_raw_seed123）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p1_2c_raw_seed123`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p1_2c_raw_seed123                        -63.450    -59.070   4.380   300   no


JSON: docs/handoff/gpu_2026-10-14_polish_DS_p1_2c_raw_seed123.json
```

- HEAD: `2c2a062`、ホスト stuttgart03、2026-10-14 10:53:22 JST
