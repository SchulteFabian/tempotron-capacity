"""Multi-perceptron capacity on a grid of K and input sparsity ("gauss" data).

For every grid point (K, mean number of active afferents per sub-pattern) and
every dataset, the capacity is found by bisection over P. The number of active
afferents per sub-pattern is drawn from a Gaussian with std = sqrt(mean). The
learning rate is scaled as lr = LR_X_MEAN / mean, so that a learning step
changes the potential by the same amount relative to the threshold at every
sparsity (LR_X_MEAN = tuned lr 0.02 at mean 21.9).

Usage (from the project folder):

    python -m experiments.mpc_grid_scan --n-jobs 7
    python -m experiments.mpc_grid_scan --Ks 200 --means 50 --tag K200_mean50   # one grid point

On the cluster, one job per grid point (cluster/grid.sbatch); afterwards merge
the part files with

    python -m experiments.mpc_grid_scan --merge
"""

from functools import partial

import numpy as np

from experiments import runner
from models import multi_perceptron as mpc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, THETA, W_STD = mpc.N, mpc.THETA, mpc.W_STD
MU, CREDIT = mpc.MU, mpc.CREDIT
LR_X_MEAN = mpc.LR * mpc.MEAN_ACTIVE         # lr * mean is kept fixed across the grid (0.438)
KS = [1, 10, 20, 50, 100, 150, 200]
MEANS = [1, 10, 20, 50, 100, 150, 200]
N_DATASETS = 20
MAX_EPOCHS = 20000
RESOLUTION = 10                              # P resolution of the bisection (alpha: 0.02)


def result_name(max_epochs=MAX_EPOCHS):
    return runner.make_name("capacity_grid", model="multi_perceptron", N=N, std="sqrt",
                            lr_x_mean=round(LR_X_MEAN, 4), mu=MU, credit=CREDIT, res=RESOLUTION,
                            max_epochs=max_epochs)


def tasks(Ks=KS, means=MEANS, n_datasets=N_DATASETS):
    # Largest K first: these tasks take longest, so the cores stay busy until the end
    return [dict(K=K, mean=float(m), dataset=s)
            for K in sorted(Ks, reverse=True) for m in means for s in range(n_datasets)]


def task(K, mean, dataset, max_epochs=MAX_EPOCHS):
    """Capacity of one dataset at one grid point (K, mean)."""
    lr = LR_X_MEAN / mean
    P_ok, P_fail = mpc.capacity_bisection(
        N, K, THETA, lr, seed=dataset, resolution=RESOLUTION, max_epochs=max_epochs, w_std=W_STD,
        data="gauss", mean_active=mean, std_active=float(np.sqrt(mean)), mu=MU, credit=CREDIT,
    )
    return dict(P_ok=P_ok, P_fail=P_fail, alpha_c=(P_ok + P_fail) / 2 / N, lr=lr)


def load(max_epochs=MAX_EPOCHS):
    return runner.load_results(result_name(max_epochs), "mpc_grid_scan")


def _arguments(parser):
    parser.add_argument("--Ks", type=int, nargs="+", default=KS, help="values of K")
    parser.add_argument("--means", type=float, nargs="+", default=MEANS,
                        help="mean numbers of active afferents per sub-pattern")
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)


def main():
    args = runner.parse_args("Multi-perceptron capacity on a (K, sparsity) grid", _arguments)
    print(f"Multi-perceptron capacity grid: Ks = {args.Ks}, means = {args.means}, "
          f"{args.n_datasets} datasets, max_epochs = {args.max_epochs}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(args.max_epochs),
                               tasks(args.Ks, args.means, args.n_datasets),
                               partial(task, max_epochs=args.max_epochs))
    print("\nMean capacity alpha_c (rows: K, columns: mean active afferents):")
    print(df.groupby(["K", "mean"])["alpha_c"].mean().unstack("mean").round(2).to_string())


if __name__ == "__main__":
    main()
