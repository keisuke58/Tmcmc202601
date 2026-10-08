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
#   p2    : Phase 2。ψ 自由・多チャネル（ch1 1.0 / ch2 0 / ch3 2.0、pH の ch5 は 2026-10-08 から 0）、
#           p1 から warm start、**箱は絞らない**（論文の箱＋OVERRIDE）、2000 粒子
#           （2026-10-08: p1 の事後で絞ると、多チャネル尤度の事後が箱の外に出て端に張り付いた。
#            DH p2 で a23 の粒子が 100% 端。P2_NSIGMA を渡したときだけ絞る）
#   ult   : 本番。p2 から warm start、平均 ± ULT_NSIGMA σ（既定 4）。
#           （2026-10-08: ± 2σ だと、よく決まる成分ごとに事後の約 5% を切り落とし、10 成分で 3〜4 割になる。
#            同じデータの事後で箱を作るので査読で指摘されやすい。± 4σ なら成分ごとに 0.01% 未満）
#           粒子数は N_PART（既定 5000）、mutation は N_MUT（既定 80）。2026-10-08 の実測で決めた
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
# ノードは必ず明示する: -l nodes=1:ppn=1:gpus=1:stuttgart01（stuttgart01-03 / celtic01 / celtic02）。
# celtic03 は使わない。GPU が 1 枚落ちていて、割り当てられると起動時に死ぬ
# （cuInit(0) failed: CUDA_ERROR_NO_DEVICE、2026-10-05）。省略すると Torque が celtic03 にも割り当てる。
# ============================================================
set -euo pipefail

STAGE="${STAGE:?STAGE must be pilot|p1|p2|ult|ident}"
TAG="${TAG:?TAG must be CS|CH|DS|DH}"
SEED="${SEED:-42}"
PREV="${PREV:-}"
PRIOR_SCALE="${PRIOR_SCALE:-0}"
# GATE=off（既定）: K_hill=0（ゲート OFF）／ GATE=on: 論文と同じ K_hill=0.05, n_hill=4（対照）
GATE="${GATE:-off}"
case "$GATE" in
  off) K_HILL=0.0 ;;
  on)  K_HILL=0.05 ;;
  *) echo "unknown GATE $GATE (off|on)"; exit 1 ;;
esac
# mutation 回数の上書き（空なら STAGE ごとの既定値）
N_MUT="${N_MUT:-}"
# 粒子数の上書き（空なら STAGE ごとの既定値）
N_PART="${N_PART:-}"
# p2 の箱の絞り込み（空 = 絞らない、2026-10-08 の既定）
P2_NSIGMA="${P2_NSIGMA:-}"
# ult の箱の絞り込み（平均 ± ULT_NSIGMA σ）。2026-10-08 に 2 → 4
ULT_NSIGMA="${ULT_NSIGMA:-4}"
# 尤度の重みの上書き（空なら estimator の既定 λPg=5・λlate=3）。重みの感度を見る run 用（2026-10-08f）
LAMBDA_PG="${LAMBDA_PG:-}"
LAMBDA_LATE="${LAMBDA_LATE:-}"
# 低 beta で mutation 回数を絞る下限。1.0 = 絞らない（2026-10-05 の修正、既定）
THROTTLE_FLOOR="${THROTTLE_FLOOR:-1.0}"
# 出力ディレクトリ名の末尾に付ける識別子。過去の run を上書きしないために使う
RUNTAG="${RUNTAG:-}"
# 成分ごとの箱の上書き（estimator の --override-bounds にそのまま渡す。"idx:lo:hi,idx:lo:hi"）。
# 論文の箱が事後を切っている成分に使う（2026-10-07: DS の a33 は [1, 3] だが ident DS の事後は [−13, −1]）。
# p1 以降の絞り込みは元の箱で clip されるので、連鎖の全段で同じ値を渡すこと。
OVERRIDE="${OVERRIDE:-}"
# qsub -v はカンマで変数を区切るので、OVERRIDE の区切りは ";" でも渡せるようにする（"5:-15:20;6:-15:20"）
OVERRIDE="${OVERRIDE//;/,}"

case "$TAG" in
  CS) COND=Commensal; CULT=Static ;;
  CH) COND=Commensal; CULT=HOBIC ;;
  DS) COND=Dysbiotic; CULT=Static ;;
  DH) COND=Dysbiotic; CULT=HOBIC ;;
  *) echo "unknown TAG $TAG"; exit 1 ;;
esac
# pH（ch5）は推定に使わない（2026-10-08 ユーザー判断）。pH は全種の回帰式による答え合わせにだけ使う。
# 以前の値（HOBIC で 0.3）に戻すときは LAMBDA_CH5=0.3 を渡す。
LAMBDA_CH5="${LAMBDA_CH5:-0.0}"

cd "$HOME/Tmcmc202601/data_5species/main"
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

# 共通: ゲートは GATE で切り替え（既定 OFF）、実験 Day1 の初期値、DE-MC
COMMON=(--condition "$COND" --cultivation "$CULT" --K-hill "$K_HILL" --n-hill 4.0
        --use-exp-init --use-de-mc --mutation rw --seed "$SEED" --device gpu
        --mutation-throttle-floor "$THROTTLE_FLOOR")

need_prev() { [ -n "$PREV" ] && [ -f "$PREV/samples.npy" ] || { echo "PREV が無い: '$PREV'"; exit 1; }; }

case "$STAGE" in
  pilot) ARGS=(--fix-psi --n-particles 1000 --n-mutation-steps 40) ;;
  p1)    need_prev; ARGS=(--fix-psi --n-particles 2000 --n-mutation-steps 40
                          --init-from-dir "$PREV" --posterior-prior-nsigma 4) ;;
  p2)    need_prev; ARGS=(--multichannel --lambda-ch1 1.0 --lambda-ch2 0.0 --lambda-ch3 2.0
                          --lambda-ch5 "$LAMBDA_CH5" --n-particles 2000 --n-mutation-steps 40
                          --init-from-dir "$PREV")
         [ -n "$P2_NSIGMA" ] && ARGS+=(--posterior-prior-nsigma "$P2_NSIGMA") ;;
  ult)   need_prev; ARGS=(--multichannel --lambda-ch1 1.0 --lambda-ch2 0.0 --lambda-ch3 2.0
                          --lambda-ch5 "$LAMBDA_CH5" --n-particles 5000 --n-mutation-steps 80
                          --init-from-dir "$PREV" --posterior-prior-nsigma "$ULT_NSIGMA") ;;
  ident) ARGS=(--fix-psi --n-particles 2000 --n-mutation-steps 40 --box -15 20
               --prior-scale "$PRIOR_SCALE") ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

# N_MUT が指定されていれば --n-mutation-steps を差し替える
if [ -n "$N_MUT" ]; then
  for i in "${!ARGS[@]}"; do
    [ "${ARGS[$i]}" = "--n-mutation-steps" ] && ARGS[$((i + 1))]="$N_MUT"
  done
fi

if [ -n "$N_PART" ]; then
  for i in "${!ARGS[@]}"; do
    [ "${ARGS[$i]}" = "--n-particles" ] && ARGS[$((i + 1))]="$N_PART"
  done
fi

[ -n "$OVERRIDE" ] && ARGS+=(--override-bounds "$OVERRIDE")
[ -n "$LAMBDA_PG" ] && ARGS+=(--lambda-pg "$LAMBDA_PG")
[ -n "$LAMBDA_LATE" ] && ARGS+=(--lambda-late "$LAMBDA_LATE")

SUFFIX=""
[ "$STAGE" = "ident" ] && SUFFIX="_prior${PRIOR_SCALE}"
[ "$GATE" = "on" ] && SUFFIX="${SUFFIX}_gateon"
[ -n "$RUNTAG" ] && SUFFIX="${SUFFIX}_${RUNTAG}"
OUTDIR="_runs/paper_gateoff/${TAG}_${STAGE}${SUFFIX}_seed${SEED}"

# 既存の run は上書きしない（生データ上書き禁止）。意図的なら OVERWRITE=1 を渡す
if [ -d "$OUTDIR" ] && [ "${OVERWRITE:-0}" != "1" ]; then
  echo "既に存在する: $OUTDIR  （RUNTAG を変えるか OVERWRITE=1 を渡す）"
  exit 1
fi

# GPU の割り当ては PBS に任せる（PBS_GPUFILE）。無ければ 0。
if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
  export CUDA_VISIBLE_DEVICES="$(grep -oE 'gpu[0-9]+' "$PBS_GPUFILE" | head -1 | grep -oE '[0-9]+')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# PBS バッチは LD_LIBRARY_PATH にシステム CUDA を入れ、pip の nvidia-* と衝突して
# JAX が GPU を見失う（対話 ssh では再現しない）。
unset LD_LIBRARY_PATH

echo "=== $STAGE $TAG seed=$SEED prior=$PRIOR_SCALE gate=$GATE(K=$K_HILL)  $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "    PREV=$PREV  OUT=$OUTDIR  git=$(git rev-parse --short HEAD)  $(date)"

"$PYTHON" estimate_paper_jax.py "${COMMON[@]}" "${ARGS[@]}" --output-dir "$OUTDIR"

echo "=== done $(date)  -> $OUTDIR"
