"""Multi-perceptron capacity as a function of K, run from the terminal on several CPU cores.

For every pair (K, dataset), the capacity is found by bisection over P
(multi_perceptron.capacity_bisection). All pairs are independent tasks that
run in parallel; each finished task is saved immediately.

Usage (from the project folder):

    caffeinate -i python -u run_capacity_multi_perceptron.py 2>&1 | tee logs/capacity_multi_perceptron.log

Optional arguments:

    --Ks 8 16 25         scan only these K (default: config.MPC_K_LIST)
    --n-jobs 4           number of parallel processes (default: all cores but one)

Results are written to results/<config.mpc_capacity_vs_K_name()>.csv. Pairs
(K, dataset) that are already in the file are skipped, so an interrupted scan
continues where it stopped. The presentation notebook reads the same file.
"""

import argparse
import time

import config as cfg
import experiments as ex


def main():
    parser = argparse.ArgumentParser(description="Multi-perceptron capacity vs. K")
    parser.add_argument("--Ks", type=int, nargs="+", default=cfg.MPC_K_LIST,
                        help="numbers of sub-patterns K to scan")
    parser.add_argument("--n-jobs", type=int, default=-2,
                        help="parallel processes (-1 = all cores, -2 = all but one)")
    args = parser.parse_args()

    name = cfg.mpc_capacity_vs_K_name()
    print(f"Multi-perceptron capacity vs. K -> results/{name}.csv")
    print(f"Ks = {args.Ks}, {cfg.MPC_K_N_DATASETS} datasets per K, n_jobs = {args.n_jobs}")
    print(f"N = {cfg.MPC_N}, active per sub-pattern = N({cfg.MPC_MEAN_ACTIVE}, {cfg.MPC_STD_ACTIVE}^2), "
          f"lr = {cfg.MPC_LR:g}, mu = {cfg.MPC_MU}, credit = {cfg.MPC_CREDIT}, "
          f"max_epochs = {cfg.MPC_K_MAX_EPOCHS}\n")

    # Largest K first: these tasks take longest, so starting them early balances the workers
    tasks = [dict(K=K, dataset=s)
             for K in sorted(args.Ks, reverse=True)
             for s in range(cfg.MPC_K_N_DATASETS)]

    t_start = time.time()
    df = ex.run_tasks(name, tasks, cfg.mpc_capacity_task, n_jobs=args.n_jobs)

    summary = df.groupby("K")["alpha_c"].agg(["mean", "std", "count"])
    summary["sem"] = summary["std"] / summary["count"] ** 0.5
    print("\nSummary (capacity alpha_c per K):")
    print(summary.round(3).to_string())
    print(f"\nTotal runtime: {(time.time() - t_start) / 3600:.2f} h")


# Required for parallel processes on macOS: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()