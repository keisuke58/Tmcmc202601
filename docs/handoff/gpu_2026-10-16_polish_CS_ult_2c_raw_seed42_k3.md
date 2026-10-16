# GPU 側 → クラウド側: max logL の磨き直し（2026-10-16 / CS_ult_2c_raw_seed42_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CS_ult_2c_raw_seed42`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CS_ult_2c_raw_seed42                        -32.912    -32.342   0.571   488  yes


JSON: docs/handoff/gpu_2026-10-16_polish_CS_ult_2c_raw_seed42_k3.json
```

- HEAD: `811f10b`、ホスト vancouver01、2026-10-17 03:21:12 JST
