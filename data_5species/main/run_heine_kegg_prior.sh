#!/bin/bash
# Run Hamilton TMCMC with KEGG/HMDB sign prior for all 4 Heine conditions.
# Execute on GPU server: ssh vancouver01 "cd /path && bash run_heine_kegg_prior.sh"
set -e

PYTHON=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python
SCRIPT=estimate_reduced_nishioka.py
N_PARTICLES=500
SIGMA_KEGG=0.15
GPU=${GPU:-0}

CONDITIONS=(
    "Commensal Static"
    "Commensal HOBIC"
    "Dysbiotic Static"
    "Dysbiotic HOBIC"
)

for COND in "${CONDITIONS[@]}"; do
    read -r CONDITION CULTIVATION <<< "$COND"
    echo "=== ${CONDITION} / ${CULTIVATION} ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
        --fit \
        --condition "$CONDITION" \
        --cultivation "$CULTIVATION" \
        --n-particles $N_PARTICLES \
        --mutation rw \
        --kegg-prior-sigma $SIGMA_KEGG \
        --device gpu \
        2>&1 | tee "logs/heine_kegg_${CONDITION}_${CULTIVATION}.log"
    echo ""
done

echo "All conditions done."
