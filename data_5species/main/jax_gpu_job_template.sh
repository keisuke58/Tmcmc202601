#!/bin/bash
#PBS -N jax_gpu_job
#PBS -l nodes=1:ppn=4:gpus=1:stuttgart01
#PBS -l walltime=04:00:00
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# Generic PBS/Torque template for JAX-on-GPU jobs on this cluster
# (copaam scheduler, stuttgart/vancouver/celtic GPU nodes).
#
# Copy this file, rename, and replace the `python3 ...` line at the
# bottom. See claude.md "GPU クラスタ運用" for the two gotchas this
# template already handles:
#   1. Never ssh + background a GPU job — use qsub so Torque tracks
#      GPU allocation and avoids colliding with other users' jobs.
#   2. PBS batch env sets LD_LIBRARY_PATH=/usr/local/cuda/lib64, which
#      breaks jaxlib's pip-installed cuSPARSE. Unset it (done below).
#
# Usage:
#   qsub -v GPUIDX=0 jax_gpu_job_template.sh
#   qsub -l nodes=1:ppn=4:gpus=1:vancouver01 -v GPUIDX=0 jax_gpu_job_template.sh
# ============================================================

set -euo pipefail

GPUIDX="${GPUIDX:-0}"

cd /home/nishioka/Tmcmc202601/data_5species/main
mkdir -p logs/pbs
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
    ASSIGNED=$(grep -oE 'gpu[0-9]+' "${PBS_GPUFILE}" | head -1 | grep -oE '[0-9]+')
    export CUDA_VISIBLE_DEVICES="${ASSIGNED:-$GPUIDX}"
else
    export CUDA_VISIBLE_DEVICES="${GPUIDX}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=
unset LD_LIBRARY_PATH

echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  PBS Job ID: ${PBS_JOBID:-local}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

# --- replace below with the actual job ---
$PYTHON check_cuda_jax.py
