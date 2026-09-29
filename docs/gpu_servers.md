# IKM GPU Server Overview

Last updated: 2026-09-30

## Server Specs

| Node | GPU | VRAM | CPU cores | RAM | SSH |
|------|-----|------|-----------|-----|-----|
| vancouver01 | 4× RTX 4090 | 24 GB each | 12 | 251 GB | `ssh vancouver01` |
| vancouver02 | 4× RTX 4090 | 24 GB each | 12 | 251 GB | `ssh vancouver02` |
| stuttgart01 | 4× RTX 3090 | 24 GB each | 10 | 187 GB | `ssh stuttgart01` |
| stuttgart02 | 4× RTX 3090 | 24 GB each | 10 | 187 GB | `ssh stuttgart02` |
| stuttgart03 | 4× RTX 3090 | 24 GB each | 10 | 187 GB | `ssh stuttgart03` |
| celtic01 | 4× RTX 2080 Ti | 11 GB each | 4 | 125 GB | `ssh celtic01` |
| celtic02 | 4× RTX 2080 Ti | 11 GB each | 4 | 125 GB | `ssh celtic02` |
| celtic03 | (driver error) | — | — | — | nvidia-smi 壊れ |
| celtic04 | 4× RTX 2080 Ti | 11 GB each | — | — | `ssh celtic04` |

**Total: 28 GPUs** (8× 4090 + 12× 3090 + 8× 2080 Ti) — celtic03 除くと24枚稼働

## Quick Check

```bash
gpustat    # ~/bin/gpustat — 全ノード一括確認 (SSH 1回/ノード)
```

## Notes

- **PBS (qsub) はこれらの GPU ノードも管理している**（server: copaam）。
  2026-09-30 訂正: 以前ここには「PBS は GPU ノードを管理していない／直接 SSH で使う」と
  書いてあったが誤り。GPU ジョブは ssh して直接バックグラウンド実行せず、必ず `qsub` を使う。
  `-l nodes=1:ppn=N:gpus=1:<hostname>` で要求すると Torque が空き GPU を自動割当する
  （`qstat -f <jobid>` の `exec_gpus` で確認）
- **PBS バッチジョブは `LD_LIBRARY_PATH=/usr/local/cuda/lib64` を自動設定する。**
  pip の `nvidia-cusparse-cu12` と衝突して JAX が GPU を見失う
  (`RuntimeError: Unable to load cuSPARSE`)。対話 ssh では未設定なので再現せず気づきにくい。
  ジョブスクリプト内で `unset LD_LIBRARY_PATH` してから python を呼ぶこと
- ジョブスクリプトの雛形: `data_5species/main/dh_prior_check_job.sh` (JAX/GPU),
  `data_5species/main/jax_gpu_job_template.sh`, `data_5species/main/tmcmc_job.sh` (numba/CPU)
- `gpustat` は空き確認には使えるが、これだけに頼って直接実行しない（他ユーザーのジョブと衝突する）
- **ホームは全ノードで NFS 共有。** リポジトリの実体は `~/Tmcmc202601`。
  `~/IKM_Hiwi/Tmcmc202601` は 2026-03-05 で止まった古いコピーで git 管理外 — 混同注意
- CUDA_VISIBLE_DEVICES で GPU 指定: `CUDA_VISIBLE_DEVICES=1,3 python ...`
- vancouver は 4090 で最速。TMCMC JAX GPU ジョブはここがベスト
- stuttgart は 3090 × 12枚で並列バッチ向き
- celtic は 2080 Ti (11GB) なので大きいモデルは入らない
- celtic03 は nvidia-smi ドライバエラー（要管理者対応）
- 全ノード ProxyJump copaam 経由（~/.ssh/config 設定済み）
