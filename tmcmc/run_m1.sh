#!/usr/bin/env bash
set -euo pipefail

# Single execution script for M1 model with improved settings.
# This script runs a single TMCMC execution (not a parameter sweep).
#
# Usage:
#   ./tmcmc/run_m1.sh                    # Use default settings
#   ./tmcmc/run_m1.sh --n-particles 3000 # Override n_particles
#   ./tmcmc/run_m1.sh --n-stages 40      # Override n_stages
#
# Default settings (improved for better accuracy):
#   - n_particles: 2000 (increased from 300)
#   - n_stages: 30 (increased from 20)
#   - max_delta_beta: 0.05 (reduced from 0.2, set in config.py)
#   - models: M1 only
#   - mode: debug

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Load environment variables from .env file if it exists
if [[ -f "${ROOT_DIR}/.env" ]]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

# Default settings (improved for better ESS and convergence)
N_PARTICLES="${N_PARTICLES:-2000}"
N_STAGES="${N_STAGES:-30}"
N_MUTATION_STEPS="${N_MUTATION_STEPS:-5}"
N_CHAINS="${N_CHAINS:-1}"
MODE="${MODE:-debug}"
DEBUG_LEVEL="${DEBUG_LEVEL:-MINIMAL}"
MODELS="${MODELS:-M1}"
SIGMA_OBS="${SIGMA_OBS:-0.01}"
COV_REL="${COV_REL:-0.005}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-particles)
            N_PARTICLES="$2"
            shift 2
            ;;
        --n-stages)
            N_STAGES="$2"
            shift 2
            ;;
        --n-mutation-steps)
            N_MUTATION_STEPS="$2"
            shift 2
            ;;
        --n-chains)
            N_CHAINS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --debug-level)
            DEBUG_LEVEL="$2"
            shift 2
            ;;
        --sigma-obs)
            SIGMA_OBS="$2"
            shift 2
            ;;
        --cov-rel)
            COV_REL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-particles N      Number of particles (default: 2000)"
            echo "  --n-stages N         Number of stages (default: 30)"
            echo "  --n-mutation-steps N Number of mutation steps (default: 5)"
            echo "  --n-chains N         Number of chains (default: 1)"
            echo "  --mode MODE          Execution mode: debug, paper, sanity (default: debug)"
            echo "  --debug-level LEVEL  Debug level: OFF, ERROR, MINIMAL, VERBOSE (default: MINIMAL)"
            echo "  --sigma-obs VAL      Observation noise std (default: 0.01)"
            echo "  --cov-rel VAL        Relative covariance (default: 0.005)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Environment variables (override defaults):"
            echo "  N_PARTICLES, N_STAGES, N_MUTATION_STEPS, N_CHAINS"
            echo "  MODE, DEBUG_LEVEL, MODELS, SIGMA_OBS, COV_REL"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Generate run_id with timestamp
RUN_ID="${RUN_ID:-m1_$(date +%Y%m%d_%H%M%S)}"
START_TIME=$(date +%s)

echo "================================================================================"
echo "Running M1 TMCMC with improved settings"
echo "================================================================================"
echo "Run ID: ${RUN_ID}"
echo "Mode: ${MODE}"
echo "Models: ${MODELS}"
echo "Settings:"
echo "  n_particles: ${N_PARTICLES}"
echo "  n_stages: ${N_STAGES}"
echo "  n_mutation_steps: ${N_MUTATION_STEPS}"
echo "  n_chains: ${N_CHAINS}"
echo "  sigma_obs: ${SIGMA_OBS}"
echo "  cov_rel: ${COV_REL}"
echo "  debug_level: ${DEBUG_LEVEL}"
echo "  max_delta_beta: 0.05 (from config.py)"
echo "================================================================================"
echo ""

# Run the pipeline
python tmcmc/run_pipeline.py \
    --mode "${MODE}" \
    --run-id "${RUN_ID}" \
    --models "${MODELS}" \
    --n-particles "${N_PARTICLES}" \
    --n-stages "${N_STAGES}" \
    --n-mutation-steps "${N_MUTATION_STEPS}" \
    --n-chains "${N_CHAINS}" \
    --sigma-obs "${SIGMA_OBS}" \
    --cov-rel "${COV_REL}" \
    --debug-level "${DEBUG_LEVEL}"

EXIT_CODE=$?

# Calculate execution time
END_TIME=$(date +%s)
if [[ -n "${START_TIME:-}" ]]; then
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    ELAPSED_TIME="${ELAPSED_MIN}分 ${ELAPSED_SEC}秒"
else
    ELAPSED_TIME="不明"
fi

echo ""
echo "================================================================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "✓ Execution completed successfully"
    echo "Results saved in: tmcmc/_runs/${RUN_ID}/"
    echo "Report: tmcmc/_runs/${RUN_ID}/REPORT.md"
    echo "Execution time: ${ELAPSED_TIME}"
    
    # Send notifications (Email and/or Slack)
    if command -v python3 &> /dev/null; then
        python3 << EOF
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, "${ROOT_DIR}")

# Try to import email notifier
try:
    from tmcmc.utils.email_notifier import notify_tmcmc_completion
    EMAIL_ENABLED = True
except ImportError:
    EMAIL_ENABLED = False

# Try to import Slack notification
SLACK_ENABLED = False
try:
    stranger_path = Path("${ROOT_DIR}") / "stranger"
    if stranger_path.exists():
        sys.path.insert(0, str(stranger_path))
        from d import notify_slack
        SLACK_ENABLED = bool(os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_BOT_TOKEN"))
except:
    pass

# Determine status from REPORT.md
report_path = Path("${ROOT_DIR}") / "tmcmc" / "_runs" / "${RUN_ID}" / "REPORT.md"
status = "SUCCESS"
if report_path.exists():
    try:
        content = report_path.read_text()
        if "**status**: **PASS" in content:
            status = "PASS"
        elif "**status**: **WARN" in content:
            status = "WARN"
        elif "**status**: **FAIL" in content:
            status = "FAIL"
    except:
        pass

# Send email notification
if EMAIL_ENABLED:
    notify_tmcmc_completion(
        run_id="${RUN_ID}",
        status=status,
        elapsed_time="${ELAPSED_TIME}",
        run_dir="tmcmc/_runs/${RUN_ID}",
        report_path="tmcmc/_runs/${RUN_ID}/REPORT.md" if report_path.exists() else None,
    )

# Send Slack notification (if enabled)
if SLACK_ENABLED:
    notify_slack(
        f"✅ M1 TMCMC計算完了\n"
        f"   Run ID: ${RUN_ID}\n"
        f"   実行時間: ${ELAPSED_TIME}\n"
        f"   ステータス: ${status}\n"
        f"   結果: tmcmc/_runs/${RUN_ID}/\n"
        f"   レポート: tmcmc/_runs/${RUN_ID}/REPORT.md",
        raise_on_error=False
    )
EOF
    fi
else
    echo "✗ Execution failed with exit code: ${EXIT_CODE}"
    echo "Check logs in: tmcmc/_runs/${RUN_ID}/"
    
    # Send notifications on failure
    if command -v python3 &> /dev/null; then
        python3 << EOF
import os
import sys
from pathlib import Path

sys.path.insert(0, "${ROOT_DIR}")

# Email notification
try:
    from tmcmc.utils.email_notifier import notify_tmcmc_completion
    notify_tmcmc_completion(
        run_id="${RUN_ID}",
        status="FAIL",
        elapsed_time="${ELAPSED_TIME}",
        run_dir="tmcmc/_runs/${RUN_ID}",
        error_message=f"終了コード: ${EXIT_CODE}",
    )
except:
    pass

# Slack notification
try:
    stranger_path = Path("${ROOT_DIR}") / "stranger"
    if stranger_path.exists():
        sys.path.insert(0, str(stranger_path))
        from d import notify_slack
        SLACK_ENABLED = bool(os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_BOT_TOKEN"))
        if SLACK_ENABLED:
            notify_slack(
                f"❌ M1 TMCMC計算失敗\n"
                f"   Run ID: ${RUN_ID}\n"
                f"   終了コード: ${EXIT_CODE}\n"
                f"   ログ: tmcmc/_runs/${RUN_ID}/",
                raise_on_error=False
            )
except:
    pass
EOF
    fi
fi
echo "================================================================================"

exit ${EXIT_CODE}
