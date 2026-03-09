#!/bin/bash
# =============================================================================
# Vancouver で CPU vs GPU ベンチマークを実行
# vancouver が空いていればそちらで実行、接続不可ならエラー
# =============================================================================
# Usage:
#   cd Tmcmc202601/data_5species/main
#   bash run_benchmark_vancouver.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="${SERVER:-vancouver01}"

echo "Vancouver で CPU vs GPU ベンチマークを実行します..."
echo "  (接続できない場合は SSH 設定を確認してください)"
echo ""

bash "$SCRIPT_DIR/benchmark_jax_cpu_vs_gpu.sh" "$SERVER"
