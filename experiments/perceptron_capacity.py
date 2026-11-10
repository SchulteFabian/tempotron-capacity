"""Perceptron capacity: fraction of learnable datasets as a function of the load alpha = P/N.

For every load alpha and dataset s, P = alpha * N random patterns are trained
(seeds [s, 0] data, [s, 1] initial weights, [s, 2] pattern order). The result
is whether the dataset was learned within MAX_EPOCHS and how many epochs it took.

Usage (from the project folder):

    python -m experiments.perceptron_capacity --n-jobs 7
"""

import numpy as np

from experiments import runner
from models import perceptron as pc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, F, THETA, LR, W_STD = pc.N, pc.F, pc.THETA, pc.LR, pc.W_STD
ALPHAS = [1.6, 1.8, 1.9, 1.96, 2.0, 2.04, 2.1, 2.2, 2.4]
N_DATASETS = 10
MAX_EPOCHS = 20000


def result_name():
    return runner.make_name("capacity", model="perceptron", N=N, f=F,
                            n=N_DATASETS, max_epochs=MAX_EPOCHS)


def tasks():
    return [dict(alpha=a, dataset=s) for a in ALPHAS for s in range(N_DATASETS)]


def task(alpha, dataset):
    """Train one dataset at load alpha."""
    P = runner.even_P(alpha, N)
    X, y = pc.make_dataset(N, P, np.random.default_rng([dataset, 0]), f=F)
    w0 = pc.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = pc.train(X, y, LR, MAX_EPOCHS, np.random.default_rng([dataset, 2]), w0,
                          theta=THETA, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load():
    return runner.load_results(result_name(), "perceptron_capacity")


def main():
    args = runner.parse_args("Perceptron capacity scan")
    print(f"Perceptron capacity: N = {N}, f = {F}, alphas = {ALPHAS}, "
          f"{N_DATASETS} datasets, max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(), tasks(), task)
    print(runner.summarize(df, "alpha")[["n", "frac_converged", "median"]].round(3).to_string())


# Required for parallel processes: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()
