# GPU 側 → クラウド側: max logL の磨き直し（2026-10-14 / CH_p1_2c_raw_seed123）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `CH_p1_2c_raw_seed123`、上位 10 粒子、L-BFGS-B 最大 300 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
CH_p1_2c_raw_seed123                         -1.863     -1.769   0.094   122  yes


JSON: docs/handoff/gpu_2026-10-14_polish_CH_p1_2c_raw_seed123.json
```

- HEAD: `1ae763b`、ホスト vancouver02、2026-10-14 07:37:51 JST
