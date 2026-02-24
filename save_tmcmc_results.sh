#!/bin/bash
# Save TMCMC results to git (with LFS for .npy files)
# Usage: bash save_tmcmc_results.sh [run_dir_pattern]
# Example: bash save_tmcmc_results.sh "*_1000p"
#          bash save_tmcmc_results.sh "commensal_static_1000p"

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"

PATTERN="${1:-*_1000p}"
RUNS_DIR="data_5species/_runs"

echo "=== Saving TMCMC results to git ==="
echo "Pattern: $RUNS_DIR/$PATTERN"
echo ""

# Find matching run directories
found=0
for d in $RUNS_DIR/$PATTERN; do
    [ -d "$d" ] || continue
    found=$((found + 1))
    echo "--- $d ---"

    # Key files to track
    for ext in npy json csv png; do
        count=$(find "$d" -name "*.$ext" 2>/dev/null | wc -l)
        [ "$count" -gt 0 ] && echo "  .$ext: $count files"
    done

    # Add all important outputs
    git add "$d"/*.npy "$d"/*.json "$d"/*.csv "$d"/*.png 2>/dev/null
    git add "$d"/checkpoints/*.npy 2>/dev/null
    git add "$d"/figures/*.png 2>/dev/null
    git add "$d"/diagnostics_tables/*.csv 2>/dev/null
done

if [ "$found" -eq 0 ]; then
    echo "No directories found matching $RUNS_DIR/$PATTERN"
    exit 1
fi

# Also add logs
git add $RUNS_DIR/${PATTERN}.log 2>/dev/null

echo ""
echo "Staged files:"
git diff --cached --stat | tail -5

echo ""
echo "To commit: git commit -m 'data: TMCMC 1000p results for $PATTERN'"
echo "To push:   git push origin master"
