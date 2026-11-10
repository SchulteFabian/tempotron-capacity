"""Learning rate / momentum tuning of the multi-perceptron, run from the terminal on several CPU cores.

For every triple (lr, mu, dataset), one dataset with P = alpha * N patterns
(K = config.MPC_TUNE_K, alpha = config.MPC_TUNE_ALPHA) is trained once with
learning rate lr and momentum mu. All triples are independent tasks that run
in parallel; each finished task is saved immediately.

Usage (from the project folder):

    caffeinate -i python -u run_tune_multi_perceptron.py --n-jobs 7 2>&1 | tee logs/tune_multi_perceptron.log

Optional arguments:

    --n-jobs 7           number of parallel processes (default: all cores but one)

K, alpha, the learning rates, momenta, number of datasets and max_epochs are set
in config.py (MPC_TUNE_*). Results are written to results/<config.mpc_tune_name()>.csv.
Triples (lr, mu, dataset) that are already in the file are skipped, so an
interrupted scan continues where it stopped. The presentation notebook reads the
same file.
"""

import argparse
import time

import config as cfg
import experiments as ex


def main():
    parser = argparse.ArgumentParser(description="Multi-perceptron learning rate / momentum tuning")
    parser.add_argument("--n-jobs", type=int, default=-2,
                        help="parallel processes (-1 = all cores, -2 = all but one)")
    args = parser.parse_args()

    name = cfg.mpc_tune_name()
    print(f"Multi-perceptron lr / mu tuning -> results/{name}.csv")
    print(f"K = {cfg.MPC_TUNE_K}, alpha = {cfg.MPC_TUNE_ALPHA}, "
          f"{cfg.MPC_TUNE_N_DATASETS} datasets per (lr, mu), "
          f"max_epochs = {cfg.MPC_TUNE_MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    print(f"lrs = {cfg.MPC_TUNE_LRS}")
    print(f"mus = {cfg.MPC_TUNE_MUS}\n")

    tasks = [dict(lr=lr, mu=mu, dataset=s)
             for lr in cfg.MPC_TUNE_LRS for mu in cfg.MPC_TUNE_MUS
             for s in range(cfg.MPC_TUNE_N_DATASETS)]

    t_start = time.time()
    df = ex.run_tasks(name, tasks, cfg.mpc_tune_task, n_jobs=args.n_jobs)

    summary = ex.summarize(df, ["lr", "mu"])
    print("\nSummary (n datasets, fraction converged, median epochs):")
    print(summary[["n", "frac_converged", "median"]].round(2).to_string())
    print(f"\nTotal runtime: {(time.time() - t_start) / 3600:.2f} h")


# Required for parallel processes on macOS: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()