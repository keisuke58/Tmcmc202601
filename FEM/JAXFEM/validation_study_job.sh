#!/bin/bash
#PBS -N val-stag
#PBS -l nodes=1:ppn=4
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/Tmcmc202601/FEM/figures/validation_study/pbs_output.log

cd /home/nishioka/IKM_Hiwi/Tmcmc202601

# Use the venv with JAX installed
export PATH="/home/nishioka/IKM_Hiwi/.venv_jax/bin:$PATH"

echo "=== Staggered Coupling Validation Study ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo ""

# Comprehensive validation:
#   Study 1: Mesh convergence (15, 25, 50, 75)
#   Study 2: Time-step convergence (dt = 0.05, 0.1, 0.2, 0.5)
#   Study 3: σ_crit sensitivity (0, 50, 100, 200, 500 Pa)
#   Study 4: Posterior UQ propagation (50 samples)
python FEM/JAXFEM/run_validation_study.py \
    --study all \
    --condition dh_baseline \
    --n-samples 50

echo ""
echo "=== Completed: $(date) ==="
