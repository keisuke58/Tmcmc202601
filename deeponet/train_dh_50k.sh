#!/bin/bash
# Train DeepONet for Dysbiotic_HOBIC with 50k samples + enhanced architecture
# Target: <1% relative error (from current 2.7% with 10k/small arch)

set -e
cd "$(dirname "$0")"

DATA=data/train_Dysbiotic_HOBIC_N50000.npz
CKPT=checkpoints_Dysbiotic_HOBIC_50k

echo "=== Training DH with 50k samples ==="
echo "  Architecture: hidden=256, p=128, n_layers=4 (residual)"
echo "  Epochs: 800, batch_size=512, lr=1e-3"
echo ""

python deeponet_hamilton.py train \
    --data "$DATA" \
    --epochs 800 \
    --batch-size 512 \
    --lr 1e-3 \
    --p 128 \
    --hidden 256 \
    --n-layers 4 \
    --checkpoint-dir "$CKPT"

echo ""
echo "=== Evaluating ==="
python deeponet_hamilton.py eval \
    --checkpoint "$CKPT/best.eqx" \
    --data "$DATA" \
    --p 128 --hidden 256 --n-layers 4

echo ""
echo "=== Benchmark ==="
python deeponet_hamilton.py benchmark \
    --checkpoint "$CKPT/best.eqx" \
    --data "$DATA" \
    --n-bench 1000 \
    --p 128 --hidden 256 --n-layers 4

echo ""
echo "=== Plot ==="
python deeponet_hamilton.py plot \
    --checkpoint "$CKPT/best.eqx" \
    --data "$DATA" \
    --p 128 --hidden 256 --n-layers 4

echo ""
echo "=== Done ==="
