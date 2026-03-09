#!/bin/bash
#PBS -N coupled-stag
#PBS -l nodes=1:ppn=4
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/Tmcmc202601/FEM/figures/coupled_staggered/pbs_output.log

cd /home/nishioka/IKM_Hiwi/Tmcmc202601

# Use the venv with JAX installed
export PATH="/home/nishioka/IKM_Hiwi/.venv_jax/bin:$PATH"

echo "=== Staggered Coupled Growth-Mechanics (Klempt-style) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo ""

# Run all 4 conditions with subprocess isolation
# dt_h=1e-5, n_sub=200 → dt_macro = 2e-3
# n_macro=500 → t = 1.0 Hamilton time (near equilibrium for most conditions)
# Python loop Newton (not lax.scan) to avoid LLVM memory leak
python FEM/JAXFEM/run_coupled_staggered.py \
    --condition all \
    --nx 20 --ny 20 \
    --n-macro 500 \
    --save-every 50 \
    --n-react-sub 200 \
    --dt-h 1e-5 \
    --k-alpha 0.05 \
    --e-model phi_pg

echo ""
echo "=== Completed: $(date) ==="
