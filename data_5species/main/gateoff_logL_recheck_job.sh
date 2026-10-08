#!/bin/bash
#PBS -N gateoff_logL_recheck
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l walltime=03:00:00
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# 2026-10-02: 9/30 のゲート OFF run で logL.npy と samples.npy が
# 対応していない件の診断。
#
# Part A: 24 run の samples.npy 全粒子で logL を再計算し、記録値との
#         相関・真の最良粒子・その RMSE を出す（tools/recheck_gateoff_logL.py）
# Part B: 同じ条件を少数粒子で再実行し、logL.npy と samples.npy が
#         その場で自己整合するかを見る。再現すればエンジン側のバグが確定する。
#
# 使い方:
#   qsub data_5species/main/gateoff_logL_recheck_job.sh
# ============================================================
set -euo pipefail

cd "$HOME/Tmcmc202601"
mkdir -p logs/pbs
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

# GPU は PBS の割り当てに従う（PBS_GPUFILE）。無ければ 0。
if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
  export CUDA_VISIBLE_DEVICES="$(grep -oE 'gpu[0-9]+' "$PBS_GPUFILE" | head -1 | grep -oE '[0-9]+')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# PBS バッチは LD_LIBRARY_PATH にシステム CUDA を入れ、pip の nvidia-* と衝突して
# JAX が GPU を見失う（対話 ssh では再現しない）。dh_prior_check_job.sh と同じ対処。
unset LD_LIBRARY_PATH

echo "=============================================="
echo "logL/samples 不整合の診断"
echo "  Node: $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  git HEAD: $(git rev-parse --short HEAD)"
echo "  Start: $(date)"
echo "=============================================="

echo
echo "########## Part A: 24 run の全粒子 logL 再計算"
"$PYTHON" tools/recheck_gateoff_logL.py \
    --glob '*_gateoff_*_cpualign_20260930' \
    --out "data_5species/main/_runs/logL_recheck_$(date +%Y%m%d)"

echo
echo "########## Part B: 少数粒子での再現テスト"
cd data_5species/main
REPRO=_runs/logL_repro_300p_seed7_$(date +%Y%m%d)
"$PYTHON" estimate_reduced_nishioka_jax.py \
    --condition Dysbiotic --cultivation HOBIC --use-exp-init \
    --K-hill 0.0 --n-hill 2.0 \
    --dt 1e-4 --n-steps 2500 \
    --box -15 20 \
    --prior-scale 0 \
    --n-particles 300 --max-stages 30 --seed 7 \
    --mutation rw --device gpu \
    --lambda-pg 1 --lambda-late 1 \
    --no-polish \
    --output-dir "$REPRO"

echo
echo "--- 再現テストの自己整合チェック ---"
"$PYTHON" ../../tools/recheck_gateoff_logL.py \
    --glob "$(basename "$REPRO")" \
    --out "_runs/logL_recheck_repro_$(date +%Y%m%d)"

echo
echo "=============================================="
echo "done: $(date)"
echo "=============================================="
