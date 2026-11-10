#!/bin/bash
# Multi-perceptron capacity grid: one array task per grid point (K, mean).
# 7 x 7 = 49 tasks, each runs all datasets of its grid point in parallel.
#
# Setup:    copy the whole project folder to the cluster (or git clone the repository), then
#           in the project folder:  python3 -m venv venv && source venv/bin/activate
#                                   pip install -r cluster/requirements.txt
# Submit:   from the project folder:  mkdir -p logs && sbatch cluster/grid.sbatch
# Status:   squeue -u $USER
# Restart:  just sbatch again - finished datasets are skipped.
# Merge:    when all tasks are done:  python -m experiments.mpc_grid_scan --merge

#SBATCH --job-name=mpc_grid
#SBATCH --array=0-48
#SBATCH --cpus-per-task=20          # = datasets per grid point, one core each
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a.out

# Largest K first (index 0-6 = K 200), so the slowest tasks start earliest
KS=(200 150 100 50 20 10 1)
MEANS=(1 10 20 50 100 150 200)

K=${KS[$((SLURM_ARRAY_TASK_ID / 7))]}
MEAN=${MEANS[$((SLURM_ARRAY_TASK_ID % 7))]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Activate the Python environment (adjust if needed).
# The job runs in the folder where sbatch was called, so submit from the project folder.
source venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: K = $K, mean = $MEAN on $(hostname)"
python -u -m experiments.mpc_grid_scan --Ks "$K" --means "$MEAN" --tag "K${K}_mean${MEAN}"
