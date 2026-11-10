"""Tempotron learning time as a function of the membrane time constant tau_m (tau_s = tau_m / 4).

Since V0 depends only on the ratio tau_m / tau_s, the learning rate
1e-4 * V0 is the same for all tau_m: only the time scale of the kernel changes.
Every (tau_m, dataset) is trained once at load ALPHA.

Usage (from the project folder):

    python -m experiments.tempotron_tau_scan --n-jobs 7
"""

import numpy as np

from experiments import runner
from models import tempotron as tp

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, T, THETA, W_STD = tp.N, tp.T, tp.THETA, tp.W_STD
LR, MU, TAU_DECAY = tp.LR, tp.MU, tp.TAU_DECAY
TAUS_M = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 50.0, 75.0]
ALPHA = 2.0
N_DATASETS = 10
MAX_EPOCHS = 3000


def result_name():
    return runner.make_name("tau_scan", model="tempotron", N=N, alpha=ALPHA, lr=LR, mu=MU,
                            decay=TAU_DECAY, n=N_DATASETS, max_epochs=MAX_EPOCHS)


def tasks():
    return [dict(tau_m=t, dataset=s) for t in TAUS_M for s in range(N_DATASETS)]


def task(tau_m, dataset):
    """Train one dataset with membrane time constant tau_m."""
    P = runner.even_P(ALPHA, N)
    X, y = tp.make_dataset(N, P, T, np.random.default_rng([dataset, 0]))
    data = tp.prepare_exact(X, T, tau_m, tau_m / 4)
    w0 = tp.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = tp.train_decay_exact(data, y, THETA, lr_ini=LR, mu=MU, max_epochs=MAX_EPOCHS,
                                      rng=np.random.default_rng([dataset, 2]), w_init=w0,
                                      tau_decay=TAU_DECAY, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load():
    return runner.load_results(result_name(), "tempotron_tau_scan")


def main():
    args = runner.parse_args("Tempotron time constant scan")
    print(f"Tempotron tau_m scan: tau_m = {TAUS_M}, alpha = {ALPHA}, {N_DATASETS} datasets, "
          f"max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(), tasks(), task)
    print(runner.summarize(df, "tau_m")[["n", "frac_converged", "median"]].round(2).to_string())


if __name__ == "__main__":
    main()
