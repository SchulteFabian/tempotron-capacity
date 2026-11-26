"""Tempotron capacity: fraction of learnable datasets as a function of the load alpha = P/N.

For every load alpha and dataset s, P = alpha * N random spike patterns are
trained with the decaying learning rate (seeds [s, 0] data, [s, 1] initial
weights, [s, 2] pattern order). Two kinds of input data (models/tempotron.py):

  once     every afferent fires exactly once (Gütig & Sompolinsky 2006)
  poisson  every afferent fires as a Poisson process with rate 1/T (Rubin et al. 2010)

Each data type has its own result file; the file of the once data has the same
name as before the poisson data were added.

Result: one row per (alpha, dataset) with epochs, converged, min_errors.

Usage (from the project folder):

    python -m experiments.tempotron_capacity --n-jobs 7                  # once data
    python -m experiments.tempotron_capacity --data poisson --n-jobs 7
    python -m experiments.tempotron_capacity --data poisson --alphas 3.15 3.25 --n-jobs 7
"""

from functools import partial

import numpy as np

from experiments import runner
from models import tempotron as tp

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, T, THETA, W_STD = tp.N, tp.T, tp.THETA, tp.W_STD
TAU_M, TAU_S = tp.TAU_M, tp.TAU_S
LR, MU, TAU_DECAY = tp.LR, tp.MU, tp.TAU_DECAY
ALPHAS = {                        # default loads for each data type
    "once": [2.4, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8],
    "poisson": [3.0, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4],
}
N_DATASETS = 20
MAX_EPOCHS = 20000
ALPHA_PAPER = 3.25                # Rubin et al. (2010), Fig. 2a: theory curve at K = 100 (with K/8);
                                  # their simulations for N = 500 give about 3.0


def result_name(data="once"):
    params = dict(model="tempotron", N=N, tau_m=TAU_M, lr=LR, mu=MU, decay=TAU_DECAY,
                  n=N_DATASETS, max_epochs=MAX_EPOCHS)
    if data != "once":            # the once file keeps its old name (no data in it)
        params["data"] = data
    return runner.make_name("capacity", **params)


def tasks(alphas):
    # Largest alpha first: these runs take longest
    return [dict(alpha=a, dataset=s) for a in sorted(alphas, reverse=True)
            for s in range(N_DATASETS)]


def make_data(data, P, rng):
    """Dataset of one task as (ExactData, labels)."""
    if data == "once":
        X, y = tp.make_dataset(N, P, T, rng)
        return tp.prepare_exact(X, T, TAU_M, TAU_S), y
    if data == "poisson":
        times, affs, y = tp.make_dataset_poisson(N, P, T, rng)
        return tp.prepare_spikes(times, affs, N, T, TAU_M, TAU_S), y
    raise ValueError(f'Unknown data type "{data}" (use "once" or "poisson")')


def task(alpha, dataset, data="once"):
    """Train one dataset at load alpha."""
    P = runner.even_P(alpha, N)
    exact_data, y = make_data(data, P, np.random.default_rng([dataset, 0]))
    w0 = tp.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = tp.train_decay_exact(exact_data, y, THETA, lr_ini=LR, mu=MU, max_epochs=MAX_EPOCHS,
                                      rng=np.random.default_rng([dataset, 2]), w_init=w0,
                                      tau_decay=TAU_DECAY, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load(data="once"):
    return runner.load_results(result_name(data), "tempotron_capacity")


def _arguments(parser):
    parser.add_argument("--data", default="once", choices=list(ALPHAS),
                        help="input data: once (default) or poisson")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="loads to scan (default: ALPHAS of the data type)")


def main():
    args = runner.parse_args("Tempotron capacity scan", _arguments)
    alphas = args.alphas if args.alphas is not None else ALPHAS[args.data]
    print(f"Tempotron capacity: data = {args.data}, N = {N}, tau_m = {TAU_M}, alphas = {alphas}, "
          f"{N_DATASETS} datasets, max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(args.data), tasks(alphas),
                               partial(task, data=args.data))
    print(runner.summarize(df, "alpha")[["n", "frac_converged", "median"]].round(3).to_string())


# Required for parallel processes: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()