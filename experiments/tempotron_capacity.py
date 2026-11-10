"""Tempotron capacity: fraction of learnable datasets as a function of the load alpha = P/N.

For every load alpha and dataset s, P = alpha * N random spike patterns are
trained with the decaying learning rate (seeds [s, 0] data, [s, 1] initial
weights, [s, 2] pattern order).

Usage (from the project folder):

    python -m experiments.tempotron_capacity --n-jobs 7
"""

import numpy as np

from experiments import runner
from models import tempotron as tp

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, T, THETA, W_STD = tp.N, tp.T, tp.THETA, tp.W_STD
TAU_M, TAU_S = tp.TAU_M, tp.TAU_S
LR, MU, TAU_DECAY = tp.LR, tp.MU, tp.TAU_DECAY
ALPHAS = [2.4, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8]
N_DATASETS = 20
MAX_EPOCHS = 20000
ALPHA_PAPER = 3.68                # Rubin et al. (2010) for T = 500 ms, tau_m = 10 ms


def result_name():
    return runner.make_name("capacity", model="tempotron", N=N, tau_m=TAU_M, lr=LR, mu=MU,
                            decay=TAU_DECAY, n=N_DATASETS, max_epochs=MAX_EPOCHS)


def tasks():
    # Largest alpha first: these runs take longest
    return [dict(alpha=a, dataset=s) for a in sorted(ALPHAS, reverse=True)
            for s in range(N_DATASETS)]


def task(alpha, dataset):
    """Train one dataset at load alpha."""
    P = runner.even_P(alpha, N)
    X, y = tp.make_dataset(N, P, T, np.random.default_rng([dataset, 0]))
    data = tp.prepare_exact(X, T, TAU_M, TAU_S)
    w0 = tp.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = tp.train_decay_exact(data, y, THETA, lr_ini=LR, mu=MU, max_epochs=MAX_EPOCHS,
                                      rng=np.random.default_rng([dataset, 2]), w_init=w0,
                                      tau_decay=TAU_DECAY, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load():
    return runner.load_results(result_name(), "tempotron_capacity")


def main():
    args = runner.parse_args("Tempotron capacity scan")
    print(f"Tempotron capacity: N = {N}, tau_m = {TAU_M}, alphas = {ALPHAS}, "
          f"{N_DATASETS} datasets, max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(), tasks(), task)
    print(runner.summarize(df, "alpha")[["n", "frac_converged", "median"]].round(3).to_string())


if __name__ == "__main__":
    main()
