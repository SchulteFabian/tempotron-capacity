"""Tempotron learning time as a function of the initial learning rate.

The initial learning rate is LR_FACTOR times the paper value 1e-4 * V0
(Gütig & Sompolinsky 2006). Every (lr_factor, dataset) is trained once at load ALPHA.

Usage (from the project folder):

    python -m experiments.tempotron_lr_scan --n-jobs 7
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
LR_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
ALPHA = 2.0
N_DATASETS = 10
MAX_EPOCHS = 3000


def result_name():
    return runner.make_name("lr_scan", model="tempotron", N=N, alpha=ALPHA, tau_m=TAU_M, mu=MU,
                            decay=TAU_DECAY, n=N_DATASETS, max_epochs=MAX_EPOCHS)


def tasks():
    return [dict(lr_factor=f, dataset=s) for f in LR_FACTORS for s in range(N_DATASETS)]


def task(lr_factor, dataset):
    """Train one dataset with initial learning rate lr_factor * LR."""
    P = runner.even_P(ALPHA, N)
    X, y = tp.make_dataset(N, P, T, np.random.default_rng([dataset, 0]))
    data = tp.prepare_exact(X, T, TAU_M, TAU_S)
    w0 = tp.init_weights(N, np.random.default_rng([dataset, 1]), std=W_STD)
    _, h, conv = tp.train_decay_exact(data, y, THETA, lr_ini=lr_factor * LR, mu=MU,
                                      max_epochs=MAX_EPOCHS, rng=np.random.default_rng([dataset, 2]),
                                      w_init=w0, tau_decay=TAU_DECAY, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))


def load():
    return runner.load_results(result_name(), "tempotron_lr_scan")


def main():
    args = runner.parse_args("Tempotron learning rate scan")
    print(f"Tempotron learning rate scan: factors = {LR_FACTORS} x {LR:.3g}, alpha = {ALPHA}, "
          f"{N_DATASETS} datasets, max_epochs = {MAX_EPOCHS}, n_jobs = {args.n_jobs}")
    df = runner.run_experiment(args, result_name(), tasks(), task)
    print(runner.summarize(df, "lr_factor")[["n", "frac_converged", "median"]].round(2).to_string())


if __name__ == "__main__":
    main()
