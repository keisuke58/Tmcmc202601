#!/bin/bash
# =============================================================================
# DH/CH ハイパラ探索 - 複数サーバに分散実行
# marinos01-03, frontale01-04 に 12 本を振り分け、並列実行で短縮
# =============================================================================
# Usage:
#   bash run_dh_ch_hyperparam_distributed.sh
#   bash run_dh_ch_hyperparam_distributed.sh --quick          # 150p×15st（試行用）
#   bash run_dh_ch_hyperparam_distributed.sh --servers "marinos01 marinos02"
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
MAIN_DIR="${PROJECT_ROOT}/data_5species/main"
cd "$MAIN_DIR"

# サーバ一覧（デフォルト: 7台）
SERVERS="${SERVERS:-marinos01 marinos02 marinos03 frontale01 frontale02 frontale03 frontale04}"
QUICK=false
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --servers) shift; SERVERS="${1:-$SERVERS}"; shift ;;
        --quick)   QUICK=true; shift ;;
        *) shift ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="_runs/dh_ch_hyperparam_${TS}"
if $QUICK; then
    COMMON="--n-particles 150 --n-stages 15 --use-exp-init --checkpoint-every 3"
else
    COMMON="--n-particles 500 --n-stages 30 --use-exp-init --checkpoint-every 5"
fi

# 12 jobs: cond cult sig lpg llate tag
JOBS=(
    "Dysbiotic HOBIC 1.0 5 3 baseline"
    "Dysbiotic HOBIC 0.5 5 3 sigma05"
    "Dysbiotic HOBIC 1.0 8 3 wpg8"
    "Dysbiotic HOBIC 1.0 5 4 wlate4"
    "Dysbiotic HOBIC 1.0 8 4 wpg8_wlate4"
    "Dysbiotic HOBIC 0.5 8 4 sigma05_wpg8_wlate4"
    "Commensal HOBIC 1.0 5 3 baseline"
    "Commensal HOBIC 0.5 5 3 sigma05"
    "Commensal HOBIC 1.0 8 3 wpg8"
    "Commensal HOBIC 1.0 5 4 wlate4"
    "Commensal HOBIC 1.0 8 4 wpg8_wlate4"
    "Commensal HOBIC 0.5 8 4 sigma05_wpg8_wlate4"
)

# サーバ配列
read -ra SRV_ARRAY <<< "$SERVERS"
N_SRV=${#SRV_ARRAY[@]}
N_JOBS=${#JOBS[@]}

echo "=========================================="
echo " DH/CH Hyperparameter Sweep (Distributed)"
echo " Started: $(date)"
echo " Sweep dir: $SWEEP_DIR"
echo " Servers: $N_SRV ($SERVERS)"
echo " Jobs: $N_JOBS"
$QUICK && echo " Mode: --quick (150p×15st)"
echo "=========================================="

# 各ジョブをサーバに割り当てて実行
PIDS=()
for i in "${!JOBS[@]}"; do
    read -r cond cult sig lpg llate tag <<< "${JOBS[$i]}"
    srv="${SRV_ARRAY[$((i % N_SRV))]}"
    name="${cond}_${cult}_${tag}"
    out="${SWEEP_DIR}/${name}"
    log="${SWEEP_DIR}/${name}.log"

    echo "[$((i+1))/$N_JOBS] $name -> $srv"
    ssh "$srv" "cd $MAIN_DIR && mkdir -p $SWEEP_DIR && python estimate_reduced_nishioka.py \
        --condition $cond --cultivation $cult \
        $COMMON \
        --sigma-scale $sig --lambda-pg $lpg --lambda-late $llate \
        --output-dir $out \
        > $log 2>&1" &
    PIDS+=($!)
done

echo ""
echo "All $N_JOBS jobs launched. Waiting..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo " All jobs finished at $(date)"
echo "=========================================="

# 結果をローカルに集約（NFS でない場合）
echo ""
echo "Syncing results from servers..."
for srv in ${SRV_ARRAY[@]}; do
    if ssh -o ConnectTimeout=2 "$srv" "[ -d $MAIN_DIR/$SWEEP_DIR ]" 2>/dev/null; then
        rsync -avz --ignore-existing "$srv:$MAIN_DIR/$SWEEP_DIR/" "./$SWEEP_DIR/" 2>/dev/null || true
    fi
done

# Summary
echo ""
echo "=== Results Summary (MAP RMSE) ==="
for d in ./"$SWEEP_DIR"/Dysbiotic_HOBIC_* ./"$SWEEP_DIR"/Commensal_HOBIC_*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ -f "$d/fit_metrics.json" ]; then
        rmse=$(python3 -c "import json; print(json.load(open('$d/fit_metrics.json')).get('rmse_total','N/A'))" 2>/dev/null || echo "N/A")
        echo "  $name: RMSE=$rmse"
    else
        echo "  $name: (no fit_metrics.json yet)"
    fi
done

echo ""
echo "Results in $MAIN_DIR/$SWEEP_DIR/"
