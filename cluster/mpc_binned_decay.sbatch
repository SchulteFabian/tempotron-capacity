#!/bin/bash
# Multi-perceptron on binned Poisson data with a decaying learning rate (as the tempotron):
# one array task per (bins, alpha), 4 x 6 = 24 tasks, each runs 20 datasets in parallel.
#
# Submit:   from the project folder:  mkdir -p logs && sbatch cluster/mpc_binned_decay.sbatch
# Merge:    when all tasks are done:  python -m experiments.mpc_binned_scan --tau-decay 2000 --merge

#SBATCH --job-name=mpc_decay
#SBATCH --array=0-17
#SBATCH --cpus-per-task=20          # = datasets per (bins, alpha), one core each
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%a.out

BINS=(4 3 2)
ALPHAS=(3.0 2.8 2.6 2.4 2.2 2.0)

B=${BINS[$((SLURM_ARRAY_TASK_ID / 6))]}
ALPHA=${ALPHAS[$((SLURM_ARRAY_TASK_ID % 6))]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load python-3.10.2-gcc-8.5.0-f6bu3ie
source venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: bins = $B, alpha = $ALPHA, lr decay 2000 on $(hostname)"
python -u -m experiments.mpc_binned_scan --data poisson --bins "$B" --alphas "$ALPHA" \
    --tau-decay 2000 --tag "B${B}_alpha${ALPHA}"