# Stuttgart (GPU) での DeepONet + NUTS 実行

## 前提

- stuttgart01, stuttgart02, stuttgart03 は RTX 3090 GPU 搭載
- プロジェクトは rsync で同期（`run_nuts_stuttgart.sh` の Phase 0）

## 初回セットアップ（stuttgart 各サーバで 1 回）

```bash
# stuttgart01 に SSH
ssh stuttgart01

# conda または pyenv で klempt_fem 環境を作成
# 例: conda
conda create -n klempt_fem python=3.10 -y
conda activate klempt_fem
pip install jax[cuda12] equinox matplotlib numpy

# パスを確認
which python  # 例: /home/nishioka/miniconda3/envs/klempt_fem/bin/python
```

## 実行

```bash
# ローカルから（プロジェクト同期 + 4条件分散実行）
cd Tmcmc202601
bash deeponet/run_nuts_stuttgart.sh

# stuttgart 用 Python を明示指定する場合
PYTHON=/home/nishioka/miniconda3/envs/klempt_fem/bin/python bash deeponet/run_nuts_stuttgart.sh
```

## Git 管理

- `run_nuts_stuttgart.sh`, `check_stuttgart.sh` は Git 管理推奨
- rsync で同期するため、コミット済みのファイルが stuttgart に反映される
