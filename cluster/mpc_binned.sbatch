#!/bin/bash
# Multi-perceptron on binned Poisson spike data: one array task per (bins, alpha).
# 4 x 6 = 24 tasks, each runs the 20 datasets of its (bins, alpha) in parallel.
#
# Setup:    as in cluster/grid.sbatch (project folder on the cluster, venv with requirements)
# Submit:   from the project folder:  mkdir -p logs && sbatch cluster/mpc_binned.sbatch
# Status:   squeue -u $USER
# Restart:  just sbatch again - finished datasets are skipped.
# Merge:    when all tasks are done:  python -m experiments.mpc_binned_scan --merge

#SBATCH --job-name=mpc_binned
#SBATCH --array=0-23
#SBATCH --cpus-per-task=20          # = datasets per (bins, alpha), one core each
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%a.out

# Most bins and largest alpha first (index 0 = 50 bins, alpha 3.3), so the slowest tasks start earliest
BINS=(40 35 30 20)
ALPHAS=(3.3 3.2 3.1 3.0 2.8 2.6)

B=${BINS[$((SLURM_ARRAY_TASK_ID / 6))]}
ALPHA=${ALPHAS[$((SLURM_ARRAY_TASK_ID % 6))]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load python-3.10.2-gcc-8.5.0-f6bu3ie
source venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: bins = $B, alpha = $ALPHA on $(hostname)"
python -u -m experiments.mpc_binned_scan --data poisson --bins "$B" --alphas "$ALPHA" --tag "B${B}_alpha${ALPHA}"