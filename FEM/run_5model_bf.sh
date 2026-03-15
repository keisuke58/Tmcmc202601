#!/bin/bash
#PBS -N 5model_bf
#PBS -l nodes=frontale01:ppn=12
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/Tmcmc202601/FEM/5model_bf.log

cd /home/nishioka/IKM_Hiwi/Tmcmc202601
source ~/.bashrc
conda activate base 2>/dev/null || true

python3 FEM/compute_3model_bayes_factor.py --n-samples 100 --workers 12
