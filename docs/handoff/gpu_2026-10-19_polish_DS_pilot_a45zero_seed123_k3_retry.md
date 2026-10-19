# GPU 側 → クラウド側: max logL の磨き直し（2026-10-19 / DS_pilot_a45zero_seed123_k3_retry）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_pilot_2c_raw_a45zero_m120_seed123`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_pilot_2c_raw_a45zero_m120_seed123        -64.911    -57.781   7.130   780  yes


JSON: docs/handoff/gpu_2026-10-19_polish_DS_pilot_a45zero_seed123_k3_retry.json
```

- HEAD: `212d54f`、ホスト stuttgart02、2026-10-19 11:57:23 JST
