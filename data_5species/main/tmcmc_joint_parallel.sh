#!/bin/bash
# ============================================================
# Submit joint 4-condition TMCMC across multiple nodes
# Each chain runs on a separate node → ~4x speedup
# After all finish, merge with merge_joint_chains.py
#
# Usage:
#   bash tmcmc_joint_parallel.sh          # 200 particles
#   bash tmcmc_joint_parallel.sh 500      # 500 particles
# ============================================================

set -euo pipefail
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main

NPART="${1:-200}"
NSTAGES="${2:-30}"
N_MUTATION_STEPS="${3:-10}"
TS=$(date +%Y%m%d_%H%M%S)
OUTBASE="_runs/joint_4cond_${NPART}p_${TS}"

# Available nodes (skip frontale01/02 = sasaki_s)
NODES=(frontale03 frontale04 marinos01 marinos03)
NCHAINS=${#NODES[@]}

echo "=============================================="
echo "Joint TMCMC: ${NCHAINS} chains on ${NCHAINS} nodes"
echo "  Particles: ${NPART}, Stages: ${NSTAGES}"
echo "  Output base: ${OUTBASE}"
echo "=============================================="

mkdir -p "${OUTBASE}"

JOBIDS=()
for i in $(seq 0 $((NCHAINS-1))); do
    NODE=${NODES[$i]}
    SEED=$((42 + i * 1000))
    CHAIN_OUTDIR="${OUTBASE}/chain_${i}"

    echo "Submitting chain $i on $NODE (seed=$SEED)..."

    JOBID=$(qsub -l nodes=${NODE}:ppn=24 -N "jt-ch${i}" \
        -v "NPART=${NPART},NSTAGES=${NSTAGES},NCHAINS=1,NJOBS=24,SEED=${SEED},N_MUTATION_STEPS=${N_MUTATION_STEPS},MULTICHANNEL=0,WEIGHT_CS=1.0,WEIGHT_CH=1.0,WEIGHT_DS=1.0,WEIGHT_DH=1.0,OUTDIR=${CHAIN_OUTDIR}" \
        -o "${OUTBASE}/chain_${i}.log" \
        tmcmc_joint_job.sh 2>&1)

    echo "  -> Job $JOBID"
    JOBIDS+=("$JOBID")
done

echo ""
echo "All ${NCHAINS} chains submitted:"
for i in $(seq 0 $((NCHAINS-1))); do
    echo "  Chain $i: ${JOBIDS[$i]} on ${NODES[$i]}"
done
echo ""
echo "Monitor: qstat | grep jt-ch"
echo "After completion, merge with:"
echo "  python3 merge_joint_chains.py ${OUTBASE}"
echo "=============================================="

# Save job info for merge script
python3 -c "
import json
info = {
    'n_chains': ${NCHAINS},
    'n_particles': ${NPART},
    'n_stages': ${NSTAGES},
    'timestamp': '${TS}',
    'nodes': $(printf "'%s'," "${NODES[@]}" | sed "s/,$//" | sed "s/^/[/" | sed "s/$/]/"),
    'seeds': [$(for i in $(seq 0 $((NCHAINS-1))); do echo -n "$((42 + i * 1000))"; [ $i -lt $((NCHAINS-1)) ] && echo -n ","; done)]
}
with open('${OUTBASE}/job_info.json', 'w') as f:
    json.dump(info, f, indent=2)
"
