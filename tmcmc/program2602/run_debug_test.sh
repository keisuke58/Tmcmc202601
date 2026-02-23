#!/bin/bash
# Debug mode execution script for case2_main.py
# This runs a quick test with minimal particles and stages

echo "========================================"
echo "Running case2_main.py in DEBUG mode"
echo "========================================"
echo ""

cd "$(dirname "$0")"

python -m main.case2_main \
    --mode debug \
    --models M1 \
    --seed 42 \
    --n-particles 100 \
    --n-stages 10 \
    --n-chains 1 \
    --debug-level VERBOSE

echo ""
echo "========================================"
echo "Debug execution completed"
echo "========================================"
