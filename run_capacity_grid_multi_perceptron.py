"""Multi-perceptron capacity on a grid of K and input sparsity, run from the terminal.

For every triple (K, mean, dataset), the capacity is found by bisection over P
(config.mpc_capacity_grid_task). All triples are independent tasks that run in
parallel; each finished task is saved immediately.

Usage (from the project folder):

    caffeinate -i python -u run_capacity_grid_multi_perceptron.py 2>&1 | tee logs/capacity_grid_multi_perceptron.log

Optional arguments:

    --Ks 1 10            scan only these K (default: config.MPC_GRID_KS)
    --means 20 50        scan only these sparsities (default: config.MPC_GRID_MEANS)
    --n-jobs 4           number of parallel processes (default: all cores but one)

Results are written to results/<config.mpc_capacity_grid_name()>.csv. Tasks
already in the file are skipped, so an interrupted scan continues where it stopped.
"""

import argparse
import time

import config as cfg
import experiments as ex


def main():
    parser = argparse.ArgumentParser(description="Multi-perceptron capacity on a (K, sparsity) grid")
    parser.add_argument("--Ks", type=int, nargs="+", default=cfg.MPC_GRID_KS,
                        help="numbers of sub-patterns K")
    parser.add_argument("--means", type=float, nargs="+", default=cfg.MPC_GRID_MEANS,
                        help="mean numbers of active afferents per sub-pattern")
    parser.add_argument("--n-jobs", type=int, default=-2,
                        help="parallel processes (-1 = all cores, -2 = all but one)")
    args = parser.parse_args()

    name = cfg.mpc_capacity_grid_name()
    print(f"Multi-perceptron capacity grid -> results/{name}.csv")
    print(f"Ks = {args.Ks}, means = {args.means}, {cfg.MPC_GRID_N_DATASETS} datasets, "
          f"n_jobs = {args.n_jobs}")
    print(f"N = {cfg.MPC_N}, std = sqrt(mean), lr * mean = {cfg.MPC_GRID_LR_X_MEAN:.3g}, "
          f"mu = {cfg.MPC_MU}, credit = {cfg.MPC_CREDIT}, max_epochs = {cfg.MPC_GRID_MAX_EPOCHS}\n")

    # Largest K first: the forward pass costs ~K * N per pattern, so these tasks take longest
    tasks = [dict(K=K, mean=float(m), dataset=s)
             for K in sorted(args.Ks, reverse=True)
             for m in args.means
             for s in range(cfg.MPC_GRID_N_DATASETS)]

    t_start = time.time()
    df = ex.run_tasks(name, tasks, cfg.mpc_capacity_grid_task, n_jobs=args.n_jobs)

    summary = df.groupby(["K", "mean"])["alpha_c"].mean().unstack("mean")
    print("\nMean capacity alpha_c (rows: K, columns: mean active afferents):")
    print(summary.round(2).to_string())
    print(f"\nTotal runtime: {(time.time() - t_start) / 3600:.2f} h")


# Required for parallel processes on macOS: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()