#!/bin/bash
# =============================================================================
# DH/CH ハイパラ探索（sigma_scale, lambda_pg, lambda_late）
# DH と CH に限定して RMSE に効くハイパラをグリッド探索
# =============================================================================
# Usage:
#   bash run_dh_ch_hyperparam_sweep.sh              # 逐次実行
#   bash run_dh_ch_hyperparam_sweep.sh --parallel   # DH/CH 並列（各条件内は逐次）
#   ssh frontale04 "cd ~/IKM_Hiwi/Tmcmc202601/data_5species/main && nohup bash run_dh_ch_hyperparam_sweep.sh --parallel > dh_ch_sweep.log 2>&1 &"
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
cd "$PROJECT_ROOT/data_5species/main"

PARALLEL=false
[ "${1:-}" = "--parallel" ] && PARALLEL=true && shift

TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="_runs/dh_ch_hyperparam_${TS}"
mkdir -p "$SWEEP_DIR"

COMMON="--n-particles 500 --n-stages 30 --use-exp-init --checkpoint-every 5"

# ハイパラ設定: sigma_scale lambda_pg lambda_late tag
CONFIGS=(
    "1.0 5 3 baseline"
    "0.5 5 3 sigma05"
    "1.0 8 3 wpg8"
    "1.0 5 4 wlate4"
    "1.0 8 4 wpg8_wlate4"
    "0.5 8 4 sigma05_wpg8_wlate4"
)

echo "=========================================="
echo " DH/CH Hyperparameter Sweep"
echo " Started: $(date)"
echo " Sweep dir: $SWEEP_DIR"
echo " Configs: ${#CONFIGS[@]} per condition"
echo " Parallel: $PARALLEL"
echo "=========================================="

run_one() {
    local cond="$1" cult="$2" sig="$3" lpg="$4" llate="$5" tag="$6"
    local name="${cond}_${cult}_${tag}"
    local out="${SWEEP_DIR}/${name}"
    echo "[$(date +%H:%M:%S)] $name (sigma=$sig, lambda_pg=$lpg, lambda_late=$llate)"
    python estimate_reduced_nishioka.py \
        --condition "$cond" --cultivation "$cult" \
        $COMMON \
        --sigma-scale "$sig" --lambda-pg "$lpg" --lambda-late "$llate" \
        --output-dir "$out" \
        > "${SWEEP_DIR}/${name}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $name done."
}

run_condition() {
    local cond="$1" cult="$2"
    for cfg in "${CONFIGS[@]}"; do
        read -r sig lpg llate tag <<< "$cfg"
        run_one "$cond" "$cult" "$sig" "$lpg" "$llate" "$tag"
    done
}

if $PARALLEL; then
    run_condition Dysbiotic HOBIC &
    run_condition Commensal HOBIC &
    wait
else
    run_condition Dysbiotic HOBIC
    run_condition Commensal HOBIC
fi

echo ""
echo "=========================================="
echo " Sweep finished at $(date)"
echo "=========================================="

# Summary: RMSE from fit_metrics.json
echo ""
echo "=== Results Summary (MAP RMSE) ==="
for d in "$SWEEP_DIR"/Dysbiotic_HOBIC_* "$SWEEP_DIR"/Commensal_HOBIC_*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ -f "$d/fit_metrics.json" ]; then
        rmse=$(python3 -c "import json; print(json.load(open('$d/fit_metrics.json')).get('rmse_total','N/A'))" 2>/dev/null || echo "N/A")
        echo "  $name: RMSE=$rmse"
    else
        echo "  $name: (no fit_metrics.json)"
    fi
done

echo ""
echo "Results in $SWEEP_DIR/"
