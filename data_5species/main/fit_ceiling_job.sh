#!/bin/bash
#PBS -N fit_ceiling
#PBS -l walltime=12:00:00
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log

# ============================================================
# Fit-ceiling diagnosis (handoff 2026-10-11f): tools/fit_ceiling.py on GPU.
# Usage:
#   qsub -l nodes=1:ppn=1:gpus=1:vancouver03 -v TAG=CS fit_ceiling_job.sh
# ============================================================

set -euo pipefail

TAG="${TAG:?TAG must be set (CS|CH|DS|DH)}"

cd /home/nishioka/Tmcmc202601
mkdir -p data_5species/main/logs/pbs
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
    ASSIGNED=$(grep -oE 'gpu[0-9]+' "${PBS_GPUFILE}" | head -1 | grep -oE '[0-9]+')
    export CUDA_VISIBLE_DEVICES="${ASSIGNED:-0}"
fi
echo "GPU pinning: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=
# See dh_prior_check_job.sh: PBS's system-CUDA LD_LIBRARY_PATH breaks jaxlib.
unset LD_LIBRARY_PATH

PART="${PART:-}"
ONLY=""
SUFFIX=""
if [ -n "${PART}" ]; then ONLY="--only ${PART}"; SUFFIX="_part${PART}"; fi
OUT="docs/handoff/gpu_2026-10-11_fit_ceiling_${TAG}${SUFFIX}.txt"
echo "fit_ceiling ${TAG}  Node: $(hostname)  Job: ${PBS_JOBID:-local}  git: $(/usr/bin/git rev-parse --short HEAD)  Start: $(date)"

$PYTHON tools/fit_ceiling.py --tag "${TAG}" --device gpu --lo -15 --hi 20 \
    --restarts "${RESTARTS:-3}" --maxiter 300 ${ONLY} | tee "${OUT}"

echo "finished: $(date)  -> ${OUT}"
