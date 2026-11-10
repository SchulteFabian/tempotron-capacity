"""Single multi-perceptron training run, used in the Summary to show the learning curve.

Trains one dataset of P patterns with K sub-patterns ("gauss" data) and saves
the trained weights and the number of errors per epoch.

Usage (from the project folder):

    python -m experiments.mpc_single
"""

import numpy as np

from experiments import runner
from models import multi_perceptron as mpc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, THETA, W_STD = mpc.N, mpc.THETA, mpc.W_STD
MEAN_ACTIVE, STD_ACTIVE = mpc.MEAN_ACTIVE, mpc.STD_ACTIVE
LR, MU, CREDIT = mpc.LR, mpc.MU, mpc.CREDIT
K = 8                             # sub-patterns per pattern
P = 1000                          # patterns (alpha = 2)
MAX_EPOCHS = 5000
SEEDS = dict(data=100, init=110, order=120)


def result_name():
    return runner.make_name("single", model="multi_perceptron", N=N, P=P, K=K, mean=MEAN_ACTIVE,
                            lr=LR, mu=MU, credit=CREDIT)


def make_data():
    """The dataset of this run (cheap to regenerate, so it is not saved)."""
    return mpc.make_dataset(N, P, K, np.random.default_rng(SEEDS["data"]),
                            mean_active=MEAN_ACTIVE, std_active=STD_ACTIVE)


def compute():
    X, y, _ = make_data()
    w0 = mpc.init_weights(N, np.random.default_rng(SEEDS["init"]), std=W_STD)
    w, history, converged = mpc.train(X, y, THETA, LR, MAX_EPOCHS,
                                      np.random.default_rng(SEEDS["order"]), w0,
                                      mu=MU, credit=CREDIT, verbose=100)
    return dict(w=w, history=np.array(history), converged=converged)


def load():
    return runner.load_results(result_name(), "mpc_single", kind="npz")


def main():
    res = runner.cached_run(result_name(), compute)
    print(f"Converged: {bool(res['converged'])} after {len(res['history'])} epochs")


if __name__ == "__main__":
    main()
