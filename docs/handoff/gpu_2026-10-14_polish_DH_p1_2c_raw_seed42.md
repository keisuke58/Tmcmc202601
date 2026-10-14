# GPU 側 → クラウド側: max logL の磨き直し（2026-10-14 / DH_p1_2c_raw_seed42）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DH_p1_2c_raw_seed42`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DH_p1_2c_raw_seed42                         -13.289    -12.576   0.713   300   no


JSON: docs/handoff/gpu_2026-10-14_polish_DH_p1_2c_raw_seed42.json
```

- HEAD: `238ddae`、ホスト stuttgart02、2026-10-14 10:25:13 JST
