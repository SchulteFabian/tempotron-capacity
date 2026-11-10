"""Multi-perceptron capacity for two kinds of input data as a function of K.

Both kinds have on average N / K active afferents per sub-pattern:

  once   Every afferent is active in exactly one of the K sub-patterns, chosen at
         random: the tempotron input with the spike times binned into K bins.
  gauss  Independent sub-patterns, active count ~ Gaussian(N / K, sqrt(N / K)),
         as in the grid scan.

For every (data, K, dataset), the capacity is found by bisection over P. The
learning rate is scaled as in the grid scan: lr = LR_X_MEAN / (N / K).

Usage (from the project folder):

    python -m experiments.mpc_data_scan --n-jobs 7
    python -m experiments.mpc_data_scan --data once --Ks 10 20 --n-jobs 7

On the cluster, one job per (data, K), each with its own --tag; afterwards
merge the part files with --merge (see experiments/mpc_grid_scan.py).
"""

from functools import partial

from experiments import runner
from models import multi_perceptron as mpc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, THETA, W_STD = mpc.N, mpc.THETA, mpc.W_STD
MU, CREDIT = mpc.MU, mpc.CREDIT
LR_X_MEAN = mpc.LR * mpc.MEAN_ACTIVE         # lr * mean is kept fixed across K (0.438)
DATA_TYPES = ["once", "gauss"]
KS = [10, 20, 50, 100, 150, 200]
N_DATASETS = 5
MAX_EPOCHS = 5000
RESOLUTION = 10                              # P resolution of the bisection (alpha: 0.02)


def result_name(max_epochs=MAX_EPOCHS):
    return runner.make_name("capacity_data", model="multi_perceptron", N=N, mean="N_over_K",
                            std="sqrt", lr_x_mean=round(LR_X_MEAN, 4), mu=MU, credit=CREDIT,
                            res=RESOLUTION, max_epochs=max_epochs)


def tasks(data_types=DATA_TYPES, Ks=KS, n_datasets=N_DATASETS):
    # Largest K first: these tasks take longest, so the cores stay busy until the end
    return [dict(data=d, K=K, dataset=s)
            for K in sorted(Ks, reverse=True) for d in data_types for s in range(n_datasets)]


def task(data, K, dataset, max_epochs=MAX_EPOCHS):
    """Capacity of one dataset for one data type and one K."""
    mean = N / K
    lr = LR_X_MEAN / mean
    P_ok, P_fail = mpc.capacity_bisection(
        N, K, THETA, lr, seed=dataset, resolution=RESOLUTION, max_epochs=max_epochs, w_std=W_STD,
        data=data, mean_active=mean, std_active=mean ** 0.5, mu=MU, credit=CREDIT,
    )
    return dict(mean=mean, P_ok=P_ok, P_fail=P_fail, alpha_c=(P_ok + P_fail) / 2 / N, lr=lr)


def load(max_epochs=MAX_EPOCHS):
    return runner.load_results(result_name(max_epochs), "mpc_data_scan")


def _arguments(parser):
    parser.add_argument("--data", nargs="+", default=DATA_TYPES, choices=DATA_TYPES,
                        help="data types to scan")
    parser.add_argument("--Ks", type=int, nargs="+", default=KS, help="values of K")
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)


def main():
    args = runner.parse_args("Multi-perceptron capacity for two data types vs. K", _arguments)
    print(f"Multi-perceptron capacity: data = {args.data}, Ks = {args.Ks}, "
          f"{args.n_datasets} datasets, max_epochs = {args.max_epochs}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(args.max_epochs),
                               tasks(args.data, args.Ks, args.n_datasets),
                               partial(task, max_epochs=args.max_epochs))
    print("\nCapacity alpha_c (rows: K, columns: data type):")
    print(runner.summarize_capacity(df, ["K", "data"])["mean"].unstack("data").round(3).to_string())


if __name__ == "__main__":
    main()
