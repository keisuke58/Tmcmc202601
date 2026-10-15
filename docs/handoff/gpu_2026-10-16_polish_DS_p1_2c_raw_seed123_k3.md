# GPU 側 → クラウド側: max logL の磨き直し（2026-10-16 / DS_p1_2c_raw_seed123_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p1_2c_raw_seed123`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p1_2c_raw_seed123                        -63.450    -61.248   2.202    60  yes


JSON: docs/handoff/gpu_2026-10-16_polish_DS_p1_2c_raw_seed123_k3.json
```

- HEAD: `2178e88`、ホスト stuttgart03、2026-10-16 05:24:37 JST
