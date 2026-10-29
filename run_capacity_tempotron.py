"""Tempotron capacity scan, run from the terminal on several CPU cores.

Usage (from the project folder):

    caffeinate -i python run_capacity_tempotron.py | tee logs/capacity_tempotron.log

Optional arguments:

    --alphas 3.4 3.6     scan only these loads (default: config.TP_CAP_ALPHAS)
    --n-jobs 4           number of parallel processes (default: all cores but one)

Results are written to results/<config.tempotron_capacity_name()>.csv after
every load alpha. Loads that are already in the file are skipped, so an
interrupted scan continues where it stopped. The presentation notebook reads
the same file.
"""

import argparse
import time
from functools import partial

import config as cfg
import experiments as ex


def main():
    parser = argparse.ArgumentParser(description="Tempotron capacity scan")
    parser.add_argument("--alphas", type=float, nargs="+", default=cfg.TP_CAP_ALPHAS,
                        help="loads alpha = P/N to scan")
    parser.add_argument("--n-jobs", type=int, default=-2,
                        help="parallel processes (-1 = all cores, -2 = all but one)")
    args = parser.parse_args()

    name = cfg.tempotron_capacity_name()
    print(f"Tempotron capacity scan -> results/{name}.csv")
    print(f"alphas = {args.alphas}, n_jobs = {args.n_jobs}")
    print(f"N = {cfg.TP_N}, tau_m = {cfg.TP_TAU_M} ms, lr = {cfg.TP_LR:.2e}, mu = {cfg.TP_MU}, "
          f"tau_decay = {cfg.TP_TAU_DECAY}, {cfg.TP_CAP_N_DATASETS} datasets, "
          f"max_epochs = {cfg.TP_CAP_MAX_EPOCHS}\n", flush=True)

    t_start = time.time()
    df = ex.run_scan(
        name,
        [dict(alpha=a) for a in args.alphas],
        partial(cfg.tempotron_capacity_run, n_jobs=args.n_jobs, print_progress=True),
    )

    summary = ex.summarize(df, "alpha")
    print("\nSummary:")
    print(summary[["n", "frac_converged", "median"]].round(3).to_string())
    _, note = ex.estimate_crossing(summary.index, summary["frac_converged"])
    print(f"\nCapacity estimate: {note}")
    print(f"Total runtime: {(time.time() - t_start) / 3600:.2f} h")


# Required for parallel processes on macOS: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()