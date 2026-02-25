#!/bin/bash
# N=1k vs N=10k の eval 結果を比較
# 学習終了後に実行: ./compare_n1k_n10k.sh

set -e
cd "$(dirname "$0")"

echo "=== GNN: N=1k vs N=10k 比較 ==="
echo ""

if [ -f data/train_gnn_N1000.npz ] && [ -f data/checkpoints/best.pt ]; then
  echo "--- N=1k データで評価 ---"
  python train.py eval --data data/train_gnn_N1000.npz --checkpoint data/checkpoints/best.pt \
    --hidden 128 --layers 4 2>/dev/null | head -15
  echo ""
fi

if [ -f data/train_gnn_N10000.npz ] && [ -f data/checkpoints/best.pt ]; then
  echo "--- N=10k データで評価 ---"
  python train.py eval --data data/train_gnn_N10000.npz --checkpoint data/checkpoints/best.pt \
    --hidden 128 --layers 4 2>/dev/null | head -15
fi
