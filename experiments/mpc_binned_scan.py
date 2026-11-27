"""Multi-perceptron on binned tempotron spike data: how many bins is the tempotron worth?

The spike patterns of the tempotron capacity scan (same seeds, so the same patterns
and labels as in experiments/tempotron_capacity.py) are cut into B time windows.
Sub-pattern k of a pattern counts, for every afferent, its spikes in window k:

    X[p, k, i] = number of spikes of afferent i in window k of pattern p

With the default window = T/B the windows are B disjoint bins; with --window W (ms)
there are still B windows ending at T/B, 2T/B, ..., T, but each integrates the last
W ms, so neighbouring windows overlap. The multi-perceptron (models/multi_perceptron.py)
is then trained on these sub-patterns with the same initial weights, pattern order and
epoch budget as the tempotron. The learning rate is scaled as in the data scan:
lr = LR_X_MEAN / (expected spikes per window).

Result: one row per (bins, alpha, dataset) with epochs, converged, min_errors, i.e. the
same format as the tempotron capacity scan, so the fractions learned can be compared
directly (hypothesis: B = K / 2 pi = 16 bins match the tempotron with K = 100).

Usage (from the project folder):

    python -m experiments.mpc_binned_scan --n-jobs 7                      # poisson, disjoint bins
    python -m experiments.mpc_binned_scan --bins 16 --alphas 3.0 3.2 --n-jobs 7
    python -m experiments.mpc_binned_scan --bins 100 --window 20 --n-jobs 7  # 20 ms windows, step 5 ms

On the cluster: cluster/mpc_binned.sbatch (one array task per (bins, alpha)), then
    python -m experiments.mpc_binned_scan --merge
"""

from functools import partial

import numpy as np

from experiments import runner
from models import multi_perceptron as mpc
from models import tempotron as tp

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, T = tp.N, tp.T                            # as in the tempotron capacity scan
THETA, MU, CREDIT, W_STD = mpc.THETA, mpc.MU, mpc.CREDIT, mpc.W_STD
LR_X_MEAN = mpc.LR * mpc.MEAN_ACTIVE         # lr * mean is kept fixed (0.438), as in the data scan
BINS = [10, 16, 25, 50]
ALPHAS = [2.6, 2.8, 3.0, 3.1, 3.2, 3.3]
N_DATASETS = 20
MAX_EPOCHS = 20000


def result_name(data="poisson", window=None, tau_decay=None):
    params = dict(model="multi_perceptron", data=data, N=N, T=T,
                  window="bin" if window is None else window,
                  lr_x_mean=round(LR_X_MEAN, 4), mu=MU, credit=CREDIT,
                  n=N_DATASETS, max_epochs=MAX_EPOCHS)
    if tau_decay is not None:     # files with a constant learning rate keep their old names
        params["decay"] = tau_decay
    return runner.make_name("capacity_binned", **params)

def tasks(bins, alphas):
    # Largest alpha and most bins first: these runs take longest
    return [dict(bins=B, alpha=a, dataset=s) for a in sorted(alphas, reverse=True)
            for B in sorted(bins, reverse=True) for s in range(N_DATASETS)]


def make_spike_data(data, P, rng):
    """Tempotron spike data as lists (times, affs, y): "poisson" or "once".

    With the same rng seed, these are exactly the datasets of experiments/tempotron_capacity.py.
    """
    if data == "poisson":
        return tp.make_dataset_poisson(N, P, T, rng)
    if data == "once":
        X, y = tp.make_dataset(N, P, T, rng)
        return list(X), [np.arange(N)] * P, y
    raise ValueError(f'Unknown data type "{data}" (use "poisson" or "once")')


def bin_spikes(times, affs, N, T, step, window=None):
    """Spike counts per afferent in time windows, as multi-perceptron data.

    Sub-pattern k counts the spikes in [end_k - window, end_k), with end_k = (k + 1) * step,
    k = 0, ..., T/step - 1. window = step (default) gives disjoint bins; window > step gives
    overlapping windows (parts before t = 0 are empty).

    Parameters
    ----------
    times, affs : lists of P arrays: spike times and afferents (as from make_spike_data)
    N           : number of afferents
    T           : pattern duration in ms
    step        : time between the ends of consecutive windows in ms
    window      : window length in ms (default: step)

    Returns
    -------
    X : (P, K, N) int8 array, K = T / step; X[p, k, i] = spikes of afferent i in window k
    """
    window = step if window is None else window
    ends = np.arange(1, int(round(T / step)) + 1) * step
    K = len(ends)
    n_win = int(np.ceil(window / step - 1e-9))          # windows that contain one spike
    X = np.zeros((len(times), K, N), dtype=np.int8)
    for p, (t, a) in enumerate(zip(times, affs)):
        first = np.floor(t / step).astype(int)          # first window that contains the spike
        for d in range(n_win):
            k = first + d
            ok = k < K
            ok[ok] &= ends[k[ok]] - window <= t[ok]     # spike still inside window k
            np.add.at(X[p], (k[ok], a[ok]), 1)
    return X


def task(bins, alpha, dataset, data="poisson", window=None, tau_decay=None):
    """Bin one tempotron dataset into `bins` windows and train the multi-perceptron on it.

    tau_decay : None = constant learning rate; otherwise decaying as for the tempotron.
    """
    step = T / bins
    window = step if window is None else window
    P = runner.even_P(alpha, N)
    times, affs, y = make_spike_data(data, P, np.random.default_rng([dataset, 0]))
    X = bin_spikes(times, affs, N, T, step, window)
    lr = LR_X_MEAN / (N * window / T)                   # expected spikes per window: N * window / T
    w0 = mpc.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = mpc.train(X, y, THETA, lr, MAX_EPOCHS, np.random.default_rng([dataset, 2]), w0,
                           mu=MU, credit=CREDIT, verbose=0, tau_decay=tau_decay)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load(data="poisson", window=None, tau_decay=None):
    return runner.load_results(result_name(data, window, tau_decay), "mpc_binned_scan")


def _arguments(parser):
    parser.add_argument("--data", default="poisson", choices=["poisson", "once"])
    parser.add_argument("--bins", type=int, nargs="+", default=BINS, help="numbers of windows B")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--window", type=float, default=None,
                        help="window length in ms (default: T / bins, i.e. disjoint bins)")
    parser.add_argument("--tau-decay", type=float, default=None,
                        help="learning rate decay in epochs (default: constant learning rate)")


def main():
    args = runner.parse_args("Multi-perceptron on binned tempotron spike data", _arguments)
    print(f"Binned multi-perceptron: data = {args.data}, bins = {args.bins}, alphas = {args.alphas}, "
          f"window = {args.window or 'T/bins'}, lr decay = {args.tau_decay or 'none'}, "
          f"{N_DATASETS} datasets, max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(args.data, args.window, args.tau_decay),
                               tasks(args.bins, args.alphas),
                               partial(task, data=args.data, window=args.window,
                                       tau_decay=args.tau_decay))
    print("\nFraction learned (rows: alpha, columns: bins):")
    print(runner.summarize(df, ["alpha", "bins"])["frac_converged"].unstack("bins").round(2).to_string())


# Required for parallel processes: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()