#!/bin/bash
#PBS -N val-uq
#PBS -l nodes=1:ppn=4
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/Tmcmc202601/FEM/figures/validation_study/pbs_uq_output.log

cd /home/nishioka/IKM_Hiwi/Tmcmc202601

# Use the venv with JAX installed
export PATH="/home/nishioka/IKM_Hiwi/.venv_jax/bin:$PATH"

echo "=== Staggered Coupling: UQ Study ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"

# Force NFS cache refresh
stat FEM/JAXFEM/run_validation_study.py > /dev/null
stat FEM/JAXFEM/run_coupled_staggered.py > /dev/null
echo "Validation script md5: $(md5sum FEM/JAXFEM/run_validation_study.py | cut -d' ' -f1)"
echo "Staggered script md5:  $(md5sum FEM/JAXFEM/run_coupled_staggered.py | cut -d' ' -f1)"
echo ""

# Check that the fix is present
if grep -q "theta-json" FEM/JAXFEM/run_validation_study.py; then
    echo "OK: --theta-json fix present"
else
    echo "ERROR: --theta-json fix NOT found! Aborting."
    exit 1
fi
echo ""

# Run UQ study only (Studies 1-3 already complete)
python FEM/JAXFEM/run_validation_study.py \
    --study uq \
    --condition dh_baseline \
    --n-samples 50

echo ""
echo "=== Completed: $(date) ==="
