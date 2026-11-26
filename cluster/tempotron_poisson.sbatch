#!/bin/bash
# Tempotron capacity with Poisson input data: one array task per load alpha.
# 8 tasks, each runs the 20 datasets of its alpha in parallel (one core each).
#
# Setup:    as in cluster/grid.sbatch (project folder on the cluster, venv with requirements)
# Submit:   from the project folder:  mkdir -p logs && sbatch cluster/tempotron_poisson.sbatch
# Status:   squeue -u $USER
# Restart:  just sbatch again - finished datasets are skipped.
# Merge:    when all tasks are done:  python -m experiments.tempotron_capacity --data poisson --merge

#SBATCH --job-name=tp_poisson
#SBATCH --array=0-7
#SBATCH --cpus-per-task=20          # = datasets per alpha, one core each
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%a.out

ALPHAS=(3.4 3.35 3.3 3.25 3.2 3.15 3.1 3.0)
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

source venv/bin/activate

echo "Task $SLURM_ARRAY_TASK_ID: alpha = $ALPHA on $(hostname)"
python -u -m experiments.tempotron_capacity --data poisson --alphas "$ALPHA" --tag "alpha${ALPHA}"
