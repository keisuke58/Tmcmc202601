# Stuttgart (GPU) での DeepONet + NUTS 実行

## 前提

- stuttgart01, stuttgart02, stuttgart03 は RTX 3090 GPU 搭載
- プロジェクトは rsync で同期（`run_nuts_stuttgart.sh` の Phase 0）

## 初回セットアップ（stuttgart 各サーバで 1 回）

```bash
# stuttgart01 に SSH
ssh stuttgart01

# conda または pyenv で klempt_fem2 環境を作成
# 例: conda
conda create -n klempt_fem2 python=3.10 -y
conda activate klempt_fem2
pip install jax[cuda12] equinox matplotlib numpy pandas requests numba

# パスを確認（miniforge3 の場合）
which python3  # /home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

# CUDA が CPU フォールバックする場合
python data_5species/main/check_cuda_jax.py   # 診断
# JAX_PLATFORMS=cuda は run スクリプトに組み込み済み
# LD_LIBRARY_PATH 競合時: unset LD_LIBRARY_PATH
```

## 実行

```bash
# ローカルから（プロジェクト同期 + 4条件分散実行）
cd Tmcmc202601
bash deeponet/run_nuts_stuttgart.sh

# stuttgart 用 Python を明示指定する場合（miniforge3）
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3 bash deeponet/run_nuts_stuttgart.sh
```

## JAX ODE × 4 GPU（data_5species/main）

```bash
# ローカルから実行（プロジェクト同期 + stuttgart01 で 4 GPU 並列）
cd Tmcmc202601/data_5species/main
REMOTE_PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3 bash run_jax_ode_4gpu_stuttgart.sh

# クイックテスト（50p×500st）
REMOTE_PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3 bash run_jax_ode_4gpu_stuttgart.sh --quick
```

## Git 管理

- `run_nuts_stuttgart.sh`, `check_stuttgart.sh`, `run_jax_ode_4gpu_stuttgart.sh` は Git 管理推奨
- rsync で同期するため、コミット済みのファイルが stuttgart に反映される
