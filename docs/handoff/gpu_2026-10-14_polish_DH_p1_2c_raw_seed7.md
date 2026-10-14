# GPU 側 → クラウド側: max logL の磨き直し（2026-10-14 / DH_p1_2c_raw_seed7）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DH_p1_2c_raw_seed7`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DH_p1_2c_raw_seed7                          -13.740    -12.928   0.812   300   no


JSON: docs/handoff/gpu_2026-10-14_polish_DH_p1_2c_raw_seed7.json
```

- HEAD: `a81cbc3`、ホスト stuttgart02、2026-10-14 10:28:51 JST
