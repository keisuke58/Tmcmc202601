# GPU 側 → クラウド側: max logL の磨き直し（2026-10-15 / dh_p2_2c_raw_s42）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DH_p2_2c_raw_seed42`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DH_p2_2c_raw_seed42                         -51.413    -50.617   0.797   300   no


JSON: docs/handoff/gpu_2026-10-15_polish_dh_p2_2c_raw_s42.json
```

- HEAD: `8653569`、ホスト vancouver01、2026-10-15 21:57:27 JST
