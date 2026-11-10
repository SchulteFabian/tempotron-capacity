"""Multi-perceptron capacity as a function of the number of sub-patterns K ("gauss" data).

For every (K, dataset), the capacity is found by bisection over the number of
patterns P (multi_perceptron.capacity_bisection), with the window statistics
of the tempotron as data.

Usage (from the project folder):

    python -m experiments.mpc_capacity_vs_K --n-jobs 7
    python -m experiments.mpc_capacity_vs_K --Ks 8 16 --n-jobs 7   # only these K
"""

from experiments import runner
from models import multi_perceptron as mpc

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, THETA, W_STD = mpc.N, mpc.THETA, mpc.W_STD
MEAN_ACTIVE, STD_ACTIVE = mpc.MEAN_ACTIVE, mpc.STD_ACTIVE
LR, MU, CREDIT = mpc.LR, mpc.MU, mpc.CREDIT
KS = [1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 200, 300]
N_DATASETS = 5
MAX_EPOCHS = 5000
RESOLUTION = 10                   # P resolution of the bisection (alpha: 10 / N = 0.02)


def result_name():
    return runner.make_name("capacity_vs_K", model="multi_perceptron", N=N, mean=MEAN_ACTIVE,
                            std=STD_ACTIVE, lr=LR, mu=MU, credit=CREDIT, res=RESOLUTION,
                            max_epochs=MAX_EPOCHS)


def tasks(Ks=KS):
    # Largest K first: these tasks take longest, so the cores stay busy until the end
    return [dict(K=K, dataset=s) for K in sorted(Ks, reverse=True) for s in range(N_DATASETS)]


def task(K, dataset):
    """Capacity of one dataset for one K, by bisection over P."""
    P_ok, P_fail = mpc.capacity_bisection(
        N, K, THETA, LR, seed=dataset, resolution=RESOLUTION, max_epochs=MAX_EPOCHS, w_std=W_STD,
        data="gauss", mean_active=MEAN_ACTIVE, std_active=STD_ACTIVE, mu=MU, credit=CREDIT,
    )
    return dict(P_ok=P_ok, P_fail=P_fail, alpha_c=(P_ok + P_fail) / 2 / N)


def load():
    return runner.load_results(result_name(), "mpc_capacity_vs_K")


def main():
    args = runner.parse_args(
        "Multi-perceptron capacity vs. K",
        lambda p: p.add_argument("--Ks", type=int, nargs="+", default=KS, help="values of K"),
    )
    print(f"Multi-perceptron capacity vs. K: Ks = {args.Ks}, {N_DATASETS} datasets per K, "
          f"lr = {LR}, mu = {MU}, max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(), tasks(args.Ks), task)
    print(runner.summarize_capacity(df, "K").round(3).to_string())


if __name__ == "__main__":
    main()
