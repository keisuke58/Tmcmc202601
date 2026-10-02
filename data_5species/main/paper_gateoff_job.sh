#!/bin/bash
#PBS -N paper_gateoff
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# 論文パイプライン（estimate_paper_jax.py）をゲート OFF で回す PBS ジョブ。
#
# 段階（STAGE）:
#   pilot : Phase 1 の出発点。ψ 固定・論文の箱・1000 粒子・40 mutation・DE-MC
#   p1    : Phase 1 本番。pilot から warm start、箱を MAP ± 4σ に絞る、2000 粒子
#   p2    : Phase 2。ψ 自由・多チャネル（ch1 1.0 / ch2 0 / ch3 2.0 / ch5 0.3）、
#           p1 から warm start、MAP ± 3σ、2000 粒子
#   ult   : 本番。p2 から warm start、MAP ± 2σ、10000 粒子・50 mutation
#   ident : 識別性の検証。ψ 固定・箱 [-15, 20]・2000 粒子・40 mutation・DE-MC。
#           PRIOR_SCALE=0 で事前分布なし、6 で N(0, 6^2)
# pilot〜ult は 2026-03 の論文の手順（run_production_2000p.sh / run_phase2_free_psi.sh /
# run_ultimate_10000p.sh）と同じ設定で、違いは K_hill=0（ゲート OFF）と b=0 固定だけ。
#
# 使い方（例: DH, seed 42。前段の出力を PREV に渡す）:
#   qsub -l walltime=02:00:00 -v STAGE=pilot,TAG=DH,SEED=42 paper_gateoff_job.sh
#   qsub -l walltime=03:00:00 -v STAGE=p1,TAG=DH,SEED=42,PREV=<pilot の出力> paper_gateoff_job.sh
#   qsub -l walltime=04:00:00 -v STAGE=p2,TAG=DH,SEED=42,PREV=<p1 の出力> paper_gateoff_job.sh
#   qsub -l walltime=08:00:00 -v STAGE=ult,TAG=DH,SEED=42,PREV=<p2 の出力> paper_gateoff_job.sh
#   qsub -l walltime=03:00:00 -v STAGE=ident,TAG=DH,SEED=42,PRIOR_SCALE=0 paper_gateoff_job.sh
# 特定ノードに載せるときは -l nodes=1:ppn=1:gpus=1:stuttgart01 を上書きする。
# ============================================================
set -euo pipefail

STAGE="${STAGE:?STAGE must be pilot|p1|p2|ult|ident}"
TAG="${TAG:?TAG must be CS|CH|DS|DH}"
SEED="${SEED:-42}"
PREV="${PREV:-}"
PRIOR_SCALE="${PRIOR_SCALE:-0}"

case "$TAG" in
  CS) COND=Commensal; CULT=Static ;;
  CH) COND=Commensal; CULT=HOBIC ;;
  DS) COND=Dysbiotic; CULT=Static ;;
  DH) COND=Dysbiotic; CULT=HOBIC ;;
  *) echo "unknown TAG $TAG"; exit 1 ;;
esac
LAMBDA_CH5=0.0
[ "$CULT" = "HOBIC" ] && LAMBDA_CH5=0.3

cd "$HOME/Tmcmc202601/data_5species/main"
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

# 共通: ゲート OFF（論文は K=0.05, n=4）、実験 Day1 の初期値、DE-MC
COMMON=(--condition "$COND" --cultivation "$CULT" --K-hill 0.0 --n-hill 4.0
        --use-exp-init --use-de-mc --mutation rw --seed "$SEED" --device gpu)

need_prev() { [ -n "$PREV" ] && [ -f "$PREV/samples.npy" ] || { echo "PREV が無い: '$PREV'"; exit 1; }; }

case "$STAGE" in
  pilot) ARGS=(--fix-psi --n-particles 1000 --n-mutation-steps 40) ;;
  p1)    need_prev; ARGS=(--fix-psi --n-particles 2000 --n-mutation-steps 40
                          --init-from-dir "$PREV" --posterior-prior-nsigma 4) ;;
  p2)    need_prev; ARGS=(--multichannel --lambda-ch1 1.0 --lambda-ch2 0.0 --lambda-ch3 2.0
                          --lambda-ch5 "$LAMBDA_CH5" --n-particles 2000 --n-mutation-steps 40
                          --init-from-dir "$PREV" --posterior-prior-nsigma 3) ;;
  ult)   need_prev; ARGS=(--multichannel --lambda-ch1 1.0 --lambda-ch2 0.0 --lambda-ch3 2.0
                          --lambda-ch5 "$LAMBDA_CH5" --n-particles 10000 --n-mutation-steps 50
                          --init-from-dir "$PREV" --posterior-prior-nsigma 2) ;;
  ident) ARGS=(--fix-psi --n-particles 2000 --n-mutation-steps 40 --box -15 20
               --prior-scale "$PRIOR_SCALE") ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

SUFFIX=""
[ "$STAGE" = "ident" ] && SUFFIX="_prior${PRIOR_SCALE}"
OUTDIR="_runs/paper_gateoff/${TAG}_${STAGE}${SUFFIX}_seed${SEED}"

# GPU の割り当ては PBS に任せる（PBS_GPUFILE）。無ければ 0。
if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
  export CUDA_VISIBLE_DEVICES="$(grep -oE 'gpu[0-9]+' "$PBS_GPUFILE" | head -1 | grep -oE '[0-9]+')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# PBS バッチは LD_LIBRARY_PATH にシステム CUDA を入れ、pip の nvidia-* と衝突して
# JAX が GPU を見失う（対話 ssh では再現しない）。
unset LD_LIBRARY_PATH

echo "=== $STAGE $TAG seed=$SEED prior=$PRIOR_SCALE  $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "    PREV=$PREV  OUT=$OUTDIR  git=$(git rev-parse --short HEAD)  $(date)"

"$PYTHON" estimate_paper_jax.py "${COMMON[@]}" "${ARGS[@]}" --output-dir "$OUTDIR"

echo "=== done $(date)  -> $OUTDIR"
