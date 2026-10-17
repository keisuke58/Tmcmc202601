# GPU 側 → クラウド側: max logL の磨き直し（2026-10-17 / DS_p2_2c_raw_seed42_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p2_2c_raw_seed42`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p2_2c_raw_seed42                         -97.107    -96.685   0.422   141  yes


JSON: docs/handoff/gpu_2026-10-17_polish_DS_p2_2c_raw_seed42_k3.json
```

- HEAD: `95d0c8a`、ホスト vancouver03、2026-10-17 13:43:22 JST
