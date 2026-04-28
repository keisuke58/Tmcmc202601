# IKM GPU Server Overview

Last updated: 2026-03-19

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

- PBS (qsub) はこれらの GPU ノードを管理していない。直接 SSH で使う
- PBS 管轄は frontale01-04, marinos01 (CPU のみ)
- GPU ノードにはジョブスケジューラなし → `gpustat` で空きを確認してから使う
- CUDA_VISIBLE_DEVICES で GPU 指定: `CUDA_VISIBLE_DEVICES=1,3 python ...`
- vancouver は 4090 で最速。TMCMC JAX GPU ジョブはここがベスト
- stuttgart は 3090 × 12枚で並列バッチ向き
- celtic は 2080 Ti (11GB) なので大きいモデルは入らない
- celtic03 は nvidia-smi ドライバエラー（要管理者対応）
- 全ノード ProxyJump copaam 経由（~/.ssh/config 設定済み）
