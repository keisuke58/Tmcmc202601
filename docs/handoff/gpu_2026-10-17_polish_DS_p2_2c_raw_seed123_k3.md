# GPU 側 → クラウド側: max logL の磨き直し（2026-10-17 / DS_p2_2c_raw_seed123_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p2_2c_raw_seed123`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p2_2c_raw_seed123                        -96.995    -96.457   0.538   408  yes


JSON: docs/handoff/gpu_2026-10-17_polish_DS_p2_2c_raw_seed123_k3.json
```

- HEAD: `b2d8030`、ホスト vancouver02、2026-10-17 20:07:12 JST
