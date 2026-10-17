# GPU 側 → クラウド側: max logL の磨き直し（2026-10-17 / DS_p2_2c_raw_seed7_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p2_2c_raw_seed7`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p2_2c_raw_seed7                          -96.994    -96.531   0.463   105  yes


JSON: docs/handoff/gpu_2026-10-17_polish_DS_p2_2c_raw_seed7_k3.json
```

- HEAD: `f732ce6`、ホスト vancouver03、2026-10-17 13:05:11 JST
