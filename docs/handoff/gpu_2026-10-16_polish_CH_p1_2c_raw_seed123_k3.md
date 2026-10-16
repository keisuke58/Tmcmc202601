# GPU 側 → クラウド側: max logL の磨き直し（2026-10-16 / CH_p1_2c_raw_seed123_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CH_p1_2c_raw_seed123`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CH_p1_2c_raw_seed123                         -1.863     -1.711   0.152   654  yes


JSON: docs/handoff/gpu_2026-10-16_polish_CH_p1_2c_raw_seed123_k3.json
```

- HEAD: `3319a8b`、ホスト stuttgart01、2026-10-17 01:02:57 JST
