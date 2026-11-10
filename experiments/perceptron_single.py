"""Single perceptron training run, used in the Summary to show the learning curve.

Trains one dataset of P patterns and saves the trained weights and the number
of errors per epoch.

Usage (from the project folder):

    python -m experiments.perceptron_single
"""

import numpy as np

from experiments import runner
from models import perceptron as pc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, F, THETA, LR, W_STD = pc.N, pc.F, pc.THETA, pc.LR, pc.W_STD
P = 800                           # patterns (alpha = 1.6)
MAX_EPOCHS = 20000
SEEDS = dict(data=0, init=1, order=2)


def result_name():
    return runner.make_name("single", model="perceptron", N=N, P=P, f=F)


def make_data():
    """The dataset of this run (cheap to regenerate, so it is not saved)."""
    return pc.make_dataset(N, P, np.random.default_rng(SEEDS["data"]), f=F)


def compute():
    X, y = make_data()
    w0 = pc.init_weights(N, np.random.default_rng(SEEDS["init"]), std=W_STD)
    w, history, converged = pc.train(X, y, LR, MAX_EPOCHS, np.random.default_rng(SEEDS["order"]),
                                     w0, theta=THETA, verbose=100)
    return dict(w=w, history=np.array(history), converged=converged)


def load():
    return runner.load_results(result_name(), "perceptron_single", kind="npz")


def main():
    res = runner.cached_run(result_name(), compute)
    print(f"Converged: {bool(res['converged'])} after {len(res['history'])} epochs")


if __name__ == "__main__":
    main()
