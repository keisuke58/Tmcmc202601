# GPU 側 → クラウド側: max logL の磨き直し（2026-10-14 / DS_p1_2c_raw_seed42）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p1_2c_raw_seed42`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_p1_2c_raw_seed42                         -62.661    -60.742   1.919   300   no


JSON: docs/handoff/gpu_2026-10-14_polish_DS_p1_2c_raw_seed42.json
```

- HEAD: `733ef8b`、ホスト stuttgart03、2026-10-14 11:31:40 JST
