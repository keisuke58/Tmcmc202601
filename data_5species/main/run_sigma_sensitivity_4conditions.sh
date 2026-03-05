#!/bin/bash
# Sigma sensitivity analysis: run all 4 conditions with sigma_obs scaled by 0.5
# Usage: bash run_sigma_sensitivity_4conditions.sh [--parallel]
#   or:  ssh frontale04 "cd ~/IKM_Hiwi/Tmcmc202601/data_5species/main && nohup bash run_sigma_sensitivity_4conditions.sh > sigma05_batch.log 2>&1 &"

set -e
PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
cd "$PROJECT_ROOT/data_5species/main"

PARALLEL=false
[ "$1" = "--parallel" ] && PARALLEL=true && shift

TS=$(date +%Y%m%d_%H%M%S)
OUT_BASE="_runs/sigma05_4cond_${TS}"
mkdir -p "$OUT_BASE"

COMMON="--n-particles 500 --n-stages 30 --use-exp-init --checkpoint-every 5 --sigma-scale 0.5"

echo "Sigma sensitivity (sigma_scale=0.5): running all 4 conditions (parallel=$PARALLEL)"
echo "Output base: $OUT_BASE"
echo ""

run_one() {
  local cond="$1" cult="$2" name="${1}_${2}"
  echo "[$name] Starting..."
  python estimate_reduced_nishioka.py \
    --condition "$cond" --cultivation "$cult" \
    $COMMON \
    --output-dir "${OUT_BASE}/${name}" \
    > "${OUT_BASE}/${name}.log" 2>&1
  echo "[$name] Done."
}

if $PARALLEL; then
  run_one Commensal Static &
  run_one Commensal HOBIC &
  run_one Dysbiotic Static &
  run_one Dysbiotic HOBIC &
  wait
else
  for cond_cult in "Commensal:Static" "Commensal:HOBIC" "Dysbiotic:Static" "Dysbiotic:HOBIC"; do
    cond="${cond_cult%%:*}"
    cult="${cond_cult##*:}"
    echo "=== ${cond}_${cult} ==="
    run_one "$cond" "$cult"
    echo ""
  done
fi

echo "Done. Results in $OUT_BASE/"
ls -la "$OUT_BASE"
