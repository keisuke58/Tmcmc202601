#!/bin/bash
#PBS -N dh_prior_check
#PBS -l nodes=1:ppn=4:gpus=1:stuttgart01
#PBS -l walltime=03:00:00
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# DH (Dysbiotic HOBIC) gate-off, weak-prior sigma check.
# 2 runs (no prior / sigma=6) decide whether a35 (theta[18], Vei->Pg)
# is unidentifiable (piles at the box edge) without a prior.
# See instruction doc 2026-09-29 "copaam 側の Claude への指示書".
#
# Usage:
#   qsub -v RUNTAG=noprior,PRIOR_SCALE=0,GPUIDX=0 dh_prior_check_job.sh
#   qsub -v RUNTAG=sigma6,PRIOR_SCALE=6,GPUIDX=1 dh_prior_check_job.sh
# ============================================================

set -euo pipefail

RUNTAG="${RUNTAG:?RUNTAG must be set (noprior|sigma6)}"
PRIOR_SCALE="${PRIOR_SCALE:?PRIOR_SCALE must be set (0 or 6)}"
GPUIDX="${GPUIDX:-0}"
SEED="${SEED:-42}"
NPART="${NPART:-50}"
MUTATION="${MUTATION:-nuts}"
# 既に投入済みのジョブと出力先が衝突しないよう、任意の接尾辞を付けられるようにする。
OUTTAG="${OUTTAG:-}"
# 他条件 (CS/CH/DS) も同じジョブで回せるようにする。既定は DH で従来と同じ出力先。
CODE="${CODE:-dh}"
COND="${COND:-Dysbiotic}"
CULT="${CULT:-HOBIC}"
# EXTDATA を指定すると 2月時点のレガシーデータ (_extdata/*_legacy.json) を使う。
# 既定（未指定）の loader は fig3_*.csv を読み、こちらが論文 MAP と同じ側。
# 両者は DH の正規化後で最大 0.30 ずれる。詳細は tools/build_condition_data.py。
EXTDATA="${EXTDATA:-}"
if [ -n "${EXTDATA}" ]; then
    DATA_ARGS="--external-data ${EXTDATA}"
else
    DATA_ARGS="--condition ${COND} --cultivation ${CULT} --use-exp-init"
fi

cd /home/nishioka/Tmcmc202601/data_5species/main
mkdir -p logs/pbs
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

# --- GPU pinning: prefer PBS's own GPU assignment (cgroup-isolated),
#     fall back to explicit index if PBS_GPUFILE is absent. Both branches
#     are logged so we can confirm in the .log which one was taken and
#     that the two runs never picked the same physical device.
if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
    ASSIGNED=$(grep -oE 'gpu[0-9]+' "${PBS_GPUFILE}" | head -1 | grep -oE '[0-9]+')
    if [ -n "${ASSIGNED:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${ASSIGNED}"
        echo "GPU pinning: PBS_GPUFILE -> CUDA_VISIBLE_DEVICES=${ASSIGNED}"
    else
        export CUDA_VISIBLE_DEVICES="${GPUIDX}"
        echo "GPU pinning: PBS_GPUFILE present but unparsable -> fallback CUDA_VISIBLE_DEVICES=${GPUIDX}"
    fi
else
    export CUDA_VISIBLE_DEVICES="${GPUIDX}"
    echo "GPU pinning: no PBS_GPUFILE -> explicit CUDA_VISIBLE_DEVICES=${GPUIDX}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=

# PBS batch jobs on this cluster auto-set LD_LIBRARY_PATH=/usr/local/cuda/lib64
# (system CUDA), which shadows the pip-installed nvidia-cusparse-cu12 package
# and breaks jaxlib's cuSPARSE discovery (interactive ssh sessions don't hit
# this because LD_LIBRARY_PATH is unset there). Unset it so pip's nvidia-*
# libs win, per check_cuda_jax.py's own diagnostic hint #5.
unset LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH unset (was system CUDA under PBS batch env)"

OUTDIR="_runs/${CODE}_gateoff_${RUNTAG}_${NPART}p_seed${SEED}${OUTTAG}_cpualign_20260930"

echo "=============================================="
echo "prior-check job: ${CODE} (${COND}/${CULT}) RUNTAG=${RUNTAG} PRIOR_SCALE=${PRIOR_SCALE} NPART=${NPART} SEED=${SEED} MUTATION=${MUTATION}"
echo "  Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PBS Job ID: ${PBS_JOBID:-local}"
echo "  Output: ${OUTDIR}"
echo "  Data: ${EXTDATA:-loader ${COND}/${CULT}}"
echo "  Start: $(date)"
echo "  git HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

$PYTHON estimate_reduced_nishioka_jax.py \
    ${DATA_ARGS} \
    --K-hill 0.0 --n-hill 2.0 \
    --dt 1e-4 --n-steps 2500 \
    --box -15 20 \
    --prior-scale "${PRIOR_SCALE}" \
    --n-particles "${NPART}" --max-stages 30 --seed "${SEED}" \
    --mutation "${MUTATION}" --device gpu \
    --lambda-pg 1 --lambda-late 1 \
    --no-polish \
    --output-dir "${OUTDIR}"

echo "=============================================="
echo "DH prior-check job finished: $(date)"
echo "Results: ${OUTDIR}"
echo "=============================================="
