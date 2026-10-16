# GPU 側 → クラウド側: max logL の磨き直し（2026-10-16 / DS_p1_2c_raw_seed42_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p1_2c_raw_seed42`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p1_2c_raw_seed42                         -62.661    -60.957   1.704   363  yes


JSON: docs/handoff/gpu_2026-10-16_polish_DS_p1_2c_raw_seed42_k3.json
```

- HEAD: `3496518`、ホスト stuttgart03、2026-10-16 09:28:20 JST
