#!/bin/bash
# Train GNN with N=10k data + stronger regularization + shorter patience.
# Overfitting: train=0.59 vs val=1.38 with N=1k → 対策適用

set -e
cd "$(dirname "$0")"

DATA="data/train_gnn_N10000.npz"
CKPT="data/checkpoints/best.pt"

echo "=== GNN Train: N=10k, dropout=0.2, weight_decay=1e-2, patience=100 ==="
python train.py \
  --data "$DATA" \
  --checkpoint "$CKPT" \
  --epochs 1000 \
  --batch-size 64 \
  --hidden 128 \
  --layers 4 \
  --dropout 0.2 \
  --weight-decay 1e-2 \
  --patience 100

echo ""
echo "=== Eval ==="
./eval_detailed.sh "$DATA" "$CKPT"
