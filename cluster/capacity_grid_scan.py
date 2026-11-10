"""Standalone capacity scan of the multi-perceptron on a grid of K and input sparsity.

The multi-perceptron is a perceptron with an OR rule over K binary sub-patterns:
it fires if w . x_k >= theta for at least one sub-pattern k. For every grid
point (K, mean number of active afferents per sub-pattern) and every dataset,
the capacity alpha_c = P/N is found by bisection over the number of patterns P.
This file needs no other project files.

Default grid: K = 1, 10, 20, 50, 100, 150, 200 and mean = 1, 10, 20, 50, 100,
150, 200 (7 x 7 points), 20 datasets per point, max_epochs = 20000.

Usage
-----
    pip install -r requirements.txt               # numpy, pandas, joblib

    python capacity_grid_scan.py --selftest       # quick check (a few seconds)

    nohup python -u capacity_grid_scan.py --n-jobs 128 > grid.log 2>&1 &

Optional arguments:
    --Ks 1 10 ...         numbers of sub-patterns K
    --means 1 10 ...      mean numbers of active afferents per sub-pattern
    --n-datasets 20       datasets per grid point
    --max-epochs 20000    maximum learning epochs per training run
    --n-jobs 128          parallel processes (default: all cores but one)
    --tag K200            file-name suffix, needed when several jobs run at the same
                          time (e.g. one cluster node per K): each job its own tag

Results
-------
One CSV in ./results/ (the file name encodes all parameters; with --tag, each
job writes ..._part-<tag>.csv). Every finished task (one dataset at one grid
point) is saved immediately; tasks already in the CSV are skipped, so the scan
can be stopped and restarted at any time without losing finished work.

Model details: the number of active afferents per sub-pattern is drawn from a
Gaussian with mean = mean and std = sqrt(mean), rounded and clipped to [0, N].
The learning rate is scaled as lr = 0.438 / mean, so that a learning step
changes the potential by the same amount relative to the threshold at every
sparsity (0.438 = tuned lr 0.02 at mean 21.9).
"""

import argparse
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RESULTS = Path(__file__).resolve().parent / "results"


# ===========================================================================
# Parameters
# ===========================================================================

N = 500                           # afferents
THETA = 1.0                       # threshold
MU = 0.99                         # momentum
CREDIT = "first"                  # sub-pattern held responsible for a spike
W_STD = 1e-3                      # std of the initial weights
LR_REF, MEAN_REF = 0.02, 21.9     # tuned learning rate at this mean number of active afferents
LR_X_MEAN = LR_REF * MEAN_REF     # lr * mean is kept fixed across the grid
RESOLUTION = 10                   # P resolution of the bisection (alpha: 0.02)

KS = [1, 10, 20, 50, 100, 150, 200]
MEANS = [1, 10, 20, 50, 100, 150, 200]
N_DATASETS = 20
MAX_EPOCHS = 20000


# ===========================================================================
# Multi-perceptron
# ===========================================================================

def make_dataset(N, P, K, rng, mean_active, std_active):
    """P patterns of K binary sub-patterns; active count per sub-pattern ~ N(mean, std), clipped."""
    n_active = np.clip(np.rint(rng.normal(mean_active, std_active, size=(P, K))), 0, N).astype(int)
    X = np.zeros((P, K, N), dtype=bool)
    for p in range(P):
        for k in range(K):
            X[p, k, rng.choice(N, n_active[p, k], replace=False)] = True
    y = np.repeat([0, 1], P // 2)
    rng.shuffle(y)
    return X, y, n_active


def init_weights(N, rng, std=1e-3):
    """Initial weights from a Gaussian with mean 0 and standard deviation std."""
    return rng.normal(0, std, size=N)


def forward(w, Xp, theta, credit="first"):
    """Spike decision and responsible sub-pattern for one pattern.

    credit = "first": the first sub-pattern reaching theta (analogous to shunting);
    if silent, the sub-pattern with the highest potential.
    """
    V = Xp @ w
    above = np.flatnonzero(V >= theta)
    spiked = above.size > 0
    if spiked and credit == "first":
        k_star = int(above[0])
    else:
        k_star = int(np.argmax(V))
    return spiked, k_star


def run_epoch(w, dw, X, y, theta, lr, mu, rng, credit="first"):
    """One online cycle over all patterns in random order; returns the number of errors.

    On an error: dw = mu * dw +- lr * x_{k_star}, w += dw.
    """
    n_errors = 0
    for p in rng.permutation(len(y)):
        spiked, k = forward(w, X[p], theta, credit)
        if spiked == bool(y[p]):
            continue
        n_errors += 1
        sign = -1.0 if spiked else 1.0
        dw *= mu
        dw += sign * lr * X[p, k]
        w += dw
    return n_errors


def train(X, y, theta, lr, max_epochs, rng, w_init, mu, credit):
    """Train until an epoch is error-free or max_epochs is reached; returns converged."""
    w = w_init.copy()
    dw = np.zeros_like(w)
    for _ in range(max_epochs):
        if run_epoch(w, dw, X, y, theta, lr, mu, rng, credit) == 0:
            return True
    return False


def capacity_bisection(N, K, theta, lr, seed, resolution, max_epochs, w_std,
                       mean_active, std_active, mu, credit, P_start=1000, P_limit=40000):
    """Largest learnable P for one dataset, by bisection.

    Uses nested datasets (the problem with P patterns = the first P patterns of
    one large dataset), so "learnable" is monotone in P. Labels alternate
    0, 1, 0, 1, ... so every prefix is balanced. "Not learnable" means "not
    learned within max_epochs", so this is an operational capacity (a lower bound).

    Returns (P_ok, P_fail): largest P learned, smallest P not learned.
    """
    rng_data = np.random.default_rng([seed, 0])
    w0 = init_weights(N, np.random.default_rng([seed, 1]), std=w_std)
    cache = {}

    def dataset(P):
        if P > cache.get("P", 0):
            X_new, _, _ = make_dataset(N, P - cache.get("P", 0), K, rng_data,
                                       mean_active=mean_active, std_active=std_active)
            cache["X"] = X_new if "X" not in cache else np.concatenate([cache["X"], X_new])
            cache["P"] = P
        return cache["X"][:P], np.arange(P) % 2

    def learnable(P):
        X, y = dataset(P)
        return train(X, y, theta, lr, max_epochs, np.random.default_rng([seed, 2]), w0, mu, credit)

    lo, hi = 0, P_start
    while learnable(hi):
        lo, hi = hi, 2 * hi
        if hi > P_limit:
            raise RuntimeError(f"Still learnable at P = {lo}; increase P_limit")
    while hi - lo > resolution:
        mid = (lo + hi) // 4 * 2
        if learnable(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def grid_task(K, mean, dataset, max_epochs):
    """Capacity of one dataset at one grid point (K, mean)."""
    lr = LR_X_MEAN / mean
    P_ok, P_fail = capacity_bisection(
        N, K, THETA, lr, seed=dataset, resolution=RESOLUTION, max_epochs=max_epochs,
        w_std=W_STD, mean_active=mean, std_active=float(np.sqrt(mean)), mu=MU, credit=CREDIT,
    )
    return dict(P_ok=P_ok, P_fail=P_fail, alpha_c=(P_ok + P_fail) / 2 / N, lr=lr)


# ===========================================================================
# Result file and resumable parallel tasks
# ===========================================================================

def result_name(max_epochs, tag=""):
    """Result name; same naming scheme as config.mpc_capacity_grid_name() in the project."""
    name = (f"capacity_grid_model-multi_perceptron_N-{N}_std-sqrt_lr_x_mean-{round(LR_X_MEAN, 4):g}"
            f"_mu-{MU:g}_credit-{CREDIT}_res-{RESOLUTION}_max_epochs-{max_epochs}")
    return f"{name}_part-{tag}" if tag else name


def _key(values):
    """Hashable key for a task (floats rounded against CSV round-off)."""
    return tuple(round(float(v), 10) if isinstance(v, (float, np.floating)) else v for v in values)


def _call(task_fn, task):
    return {**task, **task_fn(**task)}


def run_tasks(name, tasks, task_fn, n_jobs):
    """Run tasks in parallel, skipping those already in results/<name>.csv; save after each task."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv"
    key_cols = list(tasks[0].keys())
    df = pd.read_csv(path) if path.exists() else None
    rows = [] if df is None else df.to_dict("records")
    done = set() if df is None else {_key(r) for r in df[key_cols].itertuples(index=False)}
    todo = [t for t in tasks if _key(t[k] for k in key_cols) not in done]
    print(f"Results -> {path}")
    print(f"{len(tasks) - len(todo)} of {len(tasks)} tasks already done, {len(todo)} to run\n",
          flush=True)

    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_call)(task_fn, t) for t in todo
    )
    for i, row in enumerate(results, 1):
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{i}/{len(todo)}] {row}  (total {(time.time() - t_start) / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


# ===========================================================================
# Command line
# ===========================================================================

def selftest():
    """Small, fast run to check the installation."""
    t0 = time.time()
    P_ok, P_fail = capacity_bisection(100, 5, THETA, LR_X_MEAN / 10, seed=0, resolution=10,
                                      max_epochs=200, w_std=W_STD, mean_active=10,
                                      std_active=np.sqrt(10), mu=MU, credit=CREDIT, P_start=100)
    print(f"Multi-perceptron (N = 100, K = 5): capacity between P = {P_ok} and {P_fail}")
    print(f"Parallel test: {Parallel(n_jobs=2)(delayed(np.sqrt)(i) for i in range(4))}")
    print(f"Selftest OK ({time.time() - t0:.0f} s)")


def main():
    parser = argparse.ArgumentParser(description="Multi-perceptron capacity on a (K, sparsity) grid")
    parser.add_argument("--Ks", type=int, nargs="+", default=KS, help="numbers of sub-patterns K")
    parser.add_argument("--means", type=float, nargs="+", default=MEANS,
                        help="mean numbers of active afferents per sub-pattern")
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", -2)),
                            help="parallel processes (default: CPUs given by SLURM, otherwise all cores but one)")
    parser.add_argument("--tag", default="",
                        help="suffix for the result file when several jobs run at the same time")
    parser.add_argument("--selftest", action="store_true", help="quick check of the installation")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    print(f"Multi-perceptron capacity grid: Ks = {args.Ks}, means = {args.means}")
    print(f"N = {N}, std = sqrt(mean), lr * mean = {LR_X_MEAN:.3g}, mu = {MU}, credit = {CREDIT}, "
          f"{args.n_datasets} datasets, max_epochs = {args.max_epochs}, n_jobs = {args.n_jobs}",
          flush=True)

    # Largest K first: these tasks take longest, so the cores stay busy until the end
    tasks = [dict(K=K, mean=float(m), dataset=s)
             for K in sorted(args.Ks, reverse=True) for m in args.means
             for s in range(args.n_datasets)]

    t0 = time.time()
    df = run_tasks(result_name(args.max_epochs, args.tag), tasks,
                   partial(grid_task, max_epochs=args.max_epochs), args.n_jobs)

    summary = df.groupby(["K", "mean"])["alpha_c"].mean().unstack("mean")
    print("\nMean capacity alpha_c (rows: K, columns: mean active afferents):")
    print(summary.round(2).to_string())
    print(f"\nTotal runtime: {(time.time() - t0) / 3600:.2f} h")


# Required for parallel processes: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()
