#!/bin/bash
# Learning rate and momentum of the multi-perceptron on binned Poisson data:
# one array task per (bins, lr factor), 2 x 5 = 10 tasks, each runs 2 momenta x 10 datasets.
#
# Submit:   from the project folder:  mkdir -p logs && sbatch cluster/mpc_binned_tune.sbatch
# Merge:    when all tasks are done:  python -m experiments.mpc_binned_tune --merge

#SBATCH --job-name=mpc_tune
#SBATCH --array=0-9
#SBATCH --cpus-per-task=20          # = momenta x datasets, one core each
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%a.out

BINS=(50 16)
LR_FACTORS=(0.25 0.5 1.0 2.0 4.0)

B=${BINS[$((SLURM_ARRAY_TASK_ID / 5))]}
F=${LR_FACTORS[$((SLURM_ARRAY_TASK_ID % 5))]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load python-3.10.2-gcc-8.5.0-f6bu3ie
source venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: bins = $B, lr factor = $F on $(hostname)"
python -u -m experiments.mpc_binned_tune --bins "$B" --lr-factors "$F" --tag "B${B}_lr${F}"