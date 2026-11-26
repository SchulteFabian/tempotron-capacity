"""Single tempotron training run and analysis of what makes the trained tempotron fire.

1. Trains one dataset of P spike patterns and saves the initial and trained
   weights and the number of errors per epoch.
2. For every output spike of the trained tempotron, looks at the input spikes
   before it: how much of V(t_spike) comes from the last L ms (for all L), and
   how many inputs arrive in the last W ms, for every window length W in WINDOWS.
   As a baseline, the inputs in random windows of the same length are counted.

Results:
    results/single_...npz           weights and learning curve
    results/window_analysis_..._window-W.npz  arrays for the window figures (one file per W)
    results/window_stats_..._window-W.json    summary numbers (one file per W)

Usage (from the project folder):

    python -m experiments.tempotron_single
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
P = 1000                          # patterns (alpha = 2)
MAX_EPOCHS = 5000
SEEDS = dict(data=100, init=110, order=120)

WINDOW = 20.0                     # ms before an output spike: window of the multi-perceptron data
WINDOWS = [20.0, 5.0]             # all window lengths analysed (5 ms = T / K with K = 100)
LAGS = np.linspace(1, 100, 100)   # window lengths L for the explained fraction of V(t_spike)
N_RANDOM = 100                    # random windows per pattern (baseline)
SEED_BASELINE = 0


def result_name():
    return runner.make_name("single", model="tempotron", N=N, P=P, tau_m=TAU_M, mu=MU,
                            decay=TAU_DECAY)


def window_stats_name(window=WINDOW):
    return runner.make_name("window_stats", model="tempotron", N=N, P=P, tau_m=TAU_M, lr=LR,
                            mu=MU, decay=TAU_DECAY, window=window)


def window_analysis_name(window=WINDOW):
    return runner.make_name("window_analysis", model="tempotron", N=N, P=P, tau_m=TAU_M, lr=LR,
                            mu=MU, decay=TAU_DECAY, window=window)


def make_data():
    """The dataset of this run (cheap to regenerate, so it is not saved)."""
    X, y = tp.make_dataset(N, P, T, np.random.default_rng(SEEDS["data"]))
    return X, y, tp.prepare_exact(X, T, TAU_M, TAU_S)


def train(X, y, data):
    w0 = tp.init_weights(N, np.random.default_rng(SEEDS["init"]), std=W_STD)
    w, history, converged = tp.train_decay_exact(
        data, y, THETA, lr_ini=LR, mu=MU, max_epochs=MAX_EPOCHS,
        rng=np.random.default_rng(SEEDS["order"]), w_init=w0, tau_decay=TAU_DECAY, verbose=25,
    )
    return dict(w=w, w0=w0, history=np.array(history), converged=converged)


def window_analysis(X, data, w, window=WINDOW):
    """Input statistics in the `window` ms before the output spikes of the trained tempotron."""
    rng_baseline = np.random.default_rng(SEED_BASELINE)
    count_spike, count_random, explained = [], [], []
    lags_in_window, w_in_window = [], []
    for p in range(P):
        # Baseline: random windows of the same length in the same pattern
        t_r = rng_baseline.uniform(window, T, size=N_RANDOM)
        in_window = (X[p][None, :] > t_r[:, None] - window) & (X[p][None, :] <= t_r[:, None])
        count_random.extend(in_window.sum(axis=1))

        t_sp = tp.spike_time_exact(data, p, w, THETA)
        if t_sp is None:
            continue
        lag = t_sp - X[p]
        before = lag > 0
        in_window = before & (lag <= window)
        count_spike.append(np.sum(in_window))
        lags_in_window.append(lag[in_window])
        w_in_window.append(w[in_window])

        # Fraction of V(t_spike) contributed by inputs within the last L ms, for every L
        contrib = w[before] * tp.kernel(lag[before], TAU_M, TAU_S)
        total = contrib.sum()                          # = V(t_spike) = theta
        explained.append([contrib[lag[before] <= L].sum() / total for L in LAGS])

    return dict(
        lags=LAGS, explained=np.array(explained),
        count_spike=np.array(count_spike, dtype=np.int16),
        count_random=np.array(count_random, dtype=np.int16),
        lags_in_window=np.concatenate(lags_in_window), w_in_window=np.concatenate(w_in_window),
    )


def load(window=WINDOW):
    """Training run, window analysis arrays and window statistics for one window length, as one dict."""
    res = runner.load_results(result_name(), "tempotron_single", kind="npz")
    res.update(runner.load_results(window_analysis_name(window), "tempotron_single", kind="npz"))
    res["stats"] = runner.load_results(window_stats_name(window), "tempotron_single", kind="json")
    return res


def main():
    X, y, data = make_data()
    res = runner.cached_run(result_name(), lambda: train(X, y, data))
    print(f"Converged: {bool(res['converged'])} after {len(res['history'])} epochs")

    for window in WINDOWS:
        print(f"\nWindow analysis of the trained tempotron, {window:g} ms ...")
        wa = window_analysis(X, data, res["w"], window)
        runner.save_arrays(wa, window_analysis_name(window))

        i_window = np.argmin(np.abs(LAGS - window))
        stats = dict(
            mean=float(wa["count_spike"].mean()), std=float(wa["count_spike"].std()),
            n_spikes=int(len(wa["count_spike"])), window=window,
            random_mean=float(wa["count_random"].mean()), random_std=float(wa["count_random"].std()),
            explained_fraction=float(wa["explained"][:, i_window].mean()),
        )
        runner.save_json(stats, window_stats_name(window))
        print(f"Inputs in the {window:g} ms before an output spike: {stats['mean']:.2f} ± {stats['std']:.2f} "
              f"(random windows: {stats['random_mean']:.2f} ± {stats['random_std']:.2f}), "
              f"fraction of V(t_spike): {stats['explained_fraction']:.2f}")


if __name__ == "__main__":
    main()