#!/bin/bash
#PBS -N stag-v2
#PBS -l nodes=1:ppn=4
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/Tmcmc202601/FEM/figures/coupled_staggered/pbs_output.log

cd /home/nishioka/IKM_Hiwi/Tmcmc202601

# Use the venv with JAX installed
export PATH="/home/nishioka/IKM_Hiwi/.venv_jax/bin:$PATH"

echo "=== Quasi-Static Staggered Coupled Growth-Mechanics (v2) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo ""

# Quasi-static staggered coupling (v2):
#   - 0D Numba solver for initial equilibration (2500 steps = TMCMC calibration)
#   - Hamilton reaction at EACH growth step with LOCAL nutrient c(x,y)
#     c_effective = c_hamilton * c_nutrient(x,y) / c_boundary
#   - Species-specific growth weights (So=1.0, An=0.8, Vei=0.6, Fn=0.5, Pg=0.3)
#   - Mixed nutrient BCs (Dirichlet top=saliva, Neumann bottom=tooth)
#   - Stress-dependent growth inhibition: f(σ) = max(0, 1 - σ_vm/100)
#   - Plane stress (thin biofilm assumption)
python FEM/JAXFEM/run_coupled_staggered.py \
    --condition all \
    --nx 25 --ny 25 \
    --dt-h 1e-5 \
    --ode-init-steps 2500 \
    --ode-adjust-steps 100 \
    --dt-growth 0.1 \
    --n-growth-steps 50 \
    --k-alpha 0.05 \
    --e-model phi_pg \
    --sigma-crit 100 \
    --stress-type plane_stress \
    --nutrient-bc mixed

echo ""
echo "=== Completed: $(date) ==="
