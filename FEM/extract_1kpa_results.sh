#!/bin/bash
# Extract node data + stress from 1kPa ODB files
set -e
FEM_DIR="$(cd "$(dirname "$0")" && pwd)"
JOBS_DIR="$FEM_DIR/_abaqus_1kpa_jobs"

CONDITIONS="commensal_static commensal_hobic dh_baseline dysbiotic_static"

for cond in $CONDITIONS; do
    JOB_DIR="$JOBS_DIR/${cond}_T23_1kpa"
    ODB="$JOB_DIR/two_layer_T23_${cond}_1kpa.odb"
    CSV="$JOB_DIR/nodes_3d.csv"

    if [ ! -f "$ODB" ]; then
        echo "SKIP $cond: ODB not found"
        continue
    fi

    echo "=== Extracting: $cond ==="
    cd "$JOB_DIR"
    abq2024 python "$FEM_DIR/_extract_3d_fields.py" "$ODB" "$CSV"

    # Also extract stress summary using the run_abaqus_auto.py extractor
    abq2024 python "$FEM_DIR/run_abaqus_auto.py" --extract-only "$ODB" 2>/dev/null || true
done

echo ""
echo "=== Summary ==="
for cond in $CONDITIONS; do
    CSV="$JOBS_DIR/${cond}_T23_1kpa/nodes_3d.csv"
    if [ -f "$CSV" ]; then
        echo "$cond: $(wc -l < "$CSV") lines"
    fi
done
