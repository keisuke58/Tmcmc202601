#!/bin/bash
# Eval retrained GNN with detailed per-edge metrics.
# Usage: ./eval_detailed.sh [data] [checkpoint]
#        ./eval_detailed.sh data.npz ckpt.pt
# Architecture: --hidden 128 --layers 4 (edit below if your checkpoint differs)

set -e
cd "$(dirname "$0")"

DATA="${1:-data/train_gnn_N1000.npz}"
CKPT="${2:-data/checkpoints/best.pt}"

echo "=== GNN Eval: per-edge metrics ==="
echo "Data:       $DATA"
echo "Checkpoint: $CKPT"
echo ""

python train.py eval --data "$DATA" --checkpoint "$CKPT" --hidden 128 --layers 4
