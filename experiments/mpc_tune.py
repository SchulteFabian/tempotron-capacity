"""Learning rate and momentum of the multi-perceptron, tuned near capacity.

For every K in KS and every (lr, mu, dataset), one dataset of P = ALPHA * N
patterns ("gauss" data) is trained once. Criteria: fraction of learned
datasets and learning time. One result file per K.

Usage (from the project folder):

    python -m experiments.mpc_tune --n-jobs 7
    python -m experiments.mpc_tune --Ks 100 --n-jobs 7      # only K = 100
"""

from functools import partial

import numpy as np

from experiments import runner
from models import multi_perceptron as mpc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, THETA, W_STD, CREDIT = mpc.N, mpc.THETA, mpc.W_STD, mpc.CREDIT
MEAN_ACTIVE, STD_ACTIVE = mpc.MEAN_ACTIVE, mpc.STD_ACTIVE
KS = [25, 100]
ALPHA = 2.8
LRS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]
MUS = [0.9, 0.99]
N_DATASETS = 10
MAX_EPOCHS = 5000


def result_name(K):
    return runner.make_name("tune", model="multi_perceptron", N=N, K=K, alpha=ALPHA,
                            mean=MEAN_ACTIVE, std=STD_ACTIVE, credit=CREDIT, max_epochs=MAX_EPOCHS)


def tasks():
    return [dict(lr=lr, mu=mu, dataset=s) for lr in LRS for mu in MUS for s in range(N_DATASETS)]


def task(lr, mu, dataset, K):
    """Train one dataset with one (lr, mu) pair."""
    P = runner.even_P(ALPHA, N)
    X, y, _ = mpc.make_dataset(N, P, K, np.random.default_rng([dataset, 0]),
                               mean_active=MEAN_ACTIVE, std_active=STD_ACTIVE)
    w0 = mpc.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = mpc.train(X, y, THETA, lr, MAX_EPOCHS, np.random.default_rng([dataset, 2]), w0,
                           mu=mu, credit=CREDIT, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load(K):
    """Results for one K, restricted to the scanned momenta MUS."""
    df = runner.load_results(result_name(K), "mpc_tune")
    return df[df["mu"].isin(MUS)]


def main():
    args = runner.parse_args(
        "Multi-perceptron learning rate / momentum tuning",
        lambda p: p.add_argument("--Ks", type=int, nargs="+", default=KS, help="values of K"),
    )
    for K in args.Ks:
        print(f"\nMulti-perceptron tuning: K = {K}, alpha = {ALPHA}, {N_DATASETS} datasets, "
              f"max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
        df = runner.run_experiment(args, result_name(K), tasks(), partial(task, K=K))
        print(runner.summarize(df, ["lr", "mu"])[["n", "frac_converged", "median"]].round(2).to_string())


if __name__ == "__main__":
    main()
