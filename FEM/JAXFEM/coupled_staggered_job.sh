#!/bin/bash
#PBS -N adiab-stag
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/Tmcmc202601/FEM/figures/coupled_staggered/pbs_output.log

cd /home/nishioka/IKM_Hiwi/Tmcmc202601

# Use the venv with JAX installed
export PATH="/home/nishioka/IKM_Hiwi/.venv_jax/bin:$PATH"

echo "=== Adiabatic Staggered Coupled Growth-Mechanics ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo ""

# Adiabatic approach:
#   - 0D Numba solver for initial equilibration (2500 steps, matching TMCMC)
#   - No JAX ODE needed (species at quasi-equilibrium)
#   - Growth loop: nutrient PDE + growth accumulation + FEM
#   - dt_growth=0.1 × 50 steps = 5.0 growth time units
python FEM/JAXFEM/run_coupled_staggered.py \
    --condition all \
    --nx 25 --ny 25 \
    --dt-h 1e-5 \
    --ode-init-steps 2500 \
    --ode-adjust-steps 100 \
    --dt-growth 0.1 \
    --n-growth-steps 50 \
    --k-alpha 0.05 \
    --e-model phi_pg

echo ""
echo "=== Completed: $(date) ==="
