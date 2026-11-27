"""Learning rate and momentum of the multi-perceptron on binned Poisson spike data.

LR and MU of the multi-perceptron were tuned on "gauss" data (independent sub-patterns,
21.9 active afferents on average). Before comparing it with the tempotron on binned
spike data (experiments/mpc_binned_scan.py), this scan checks whether other values learn
better there. For a few bin numbers B, at a load alpha near the measured capacity (where
about half of the datasets are learned, so changes are easiest to see), the learning rate
is varied by a factor around the standard scaling lr = LR_X_MEAN / (spikes per bin), and
the momentum is varied as well.

Result: one row per (bins, alpha, lr_factor, mu, dataset) with epochs, converged, min_errors.

Usage (from the project folder):

    python -m experiments.mpc_binned_tune --n-jobs 7
    python -m experiments.mpc_binned_tune --bins 16 --lr-factors 0.5 1 2 --n-jobs 7

On the cluster: cluster/mpc_binned_tune.sbatch, then  python -m experiments.mpc_binned_tune --merge
"""

from functools import partial

import numpy as np

from experiments import runner
from experiments.mpc_binned_scan import N, T, THETA, CREDIT, W_STD, LR_X_MEAN, bin_spikes, make_spike_data
from models import multi_perceptron as mpc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DATA = "poisson"
ALPHA_OF_BINS = {16: 3.1, 50: 2.7}           # loads near the capacities measured with the standard values
LR_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]      # lr = factor * LR_X_MEAN / (spikes per bin)
MUS = [0.9, 0.99]
N_DATASETS = 10
MAX_EPOCHS = 20000


def result_name():
    return runner.make_name("tune_binned", model="multi_perceptron", data=DATA, N=N, T=T,
                            lr_x_mean=round(LR_X_MEAN, 4), credit=CREDIT, n=N_DATASETS,
                            max_epochs=MAX_EPOCHS)


def tasks(bins, lr_factors, mus):
    return [dict(bins=B, alpha=ALPHA_OF_BINS[B], lr_factor=f, mu=mu, dataset=s)
            for B in sorted(bins, reverse=True) for f in lr_factors for mu in mus
            for s in range(N_DATASETS)]


def task(bins, alpha, lr_factor, mu, dataset):
    """Train one binned dataset (same data as mpc_binned_scan) with a given learning rate and momentum."""
    step = T / bins
    P = runner.even_P(alpha, N)
    times, affs, y = make_spike_data(DATA, P, np.random.default_rng([dataset, 0]))
    X = bin_spikes(times, affs, N, T, step)
    lr = lr_factor * LR_X_MEAN / (N / bins)
    w0 = mpc.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = mpc.train(X, y, THETA, lr, MAX_EPOCHS, np.random.default_rng([dataset, 2]), w0,
                           mu=mu, credit=CREDIT, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load():
    return runner.load_results(result_name(), "mpc_binned_tune")


def _arguments(parser):
    parser.add_argument("--bins", type=int, nargs="+", default=list(ALPHA_OF_BINS),
                        choices=list(ALPHA_OF_BINS))
    parser.add_argument("--lr-factors", type=float, nargs="+", default=LR_FACTORS)
    parser.add_argument("--mus", type=float, nargs="+", default=MUS)


def main():
    args = runner.parse_args("Learning rate and momentum of the multi-perceptron on binned data",
                             _arguments)
    print(f"Tuning on binned {DATA} data: bins = {args.bins} (alpha {[ALPHA_OF_BINS[B] for B in args.bins]}), "
          f"lr factors = {args.lr_factors}, mu = {args.mus}, {N_DATASETS} datasets, "
          f"max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(), tasks(args.bins, args.lr_factors, args.mus), task)
    print("\nFraction learned (rows: bins, lr factor; columns: mu):")
    print(runner.summarize(df, ["bins", "lr_factor", "mu"])["frac_converged"].unstack("mu").round(2).to_string())


# Required for parallel processes: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()