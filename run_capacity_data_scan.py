"""Capacity of the multi-perceptron for two kinds of input data, as a function of K.

The multi-perceptron is a perceptron with an OR rule over K binary sub-patterns:
it fires if w . x_k >= theta for at least one sub-pattern k. Two kinds of data
are compared, both with on average N / K active afferents per sub-pattern:

  once   Every afferent is active in exactly one of the K sub-patterns, chosen at
         random. This is the tempotron input (each afferent spikes exactly once)
         with the spike times binned into K time bins. The number of active
         afferents per sub-pattern varies around N / K (binomial), and the
         sub-patterns of one pattern never share an afferent.

  gauss  As in the grid scan: the sub-patterns are independent, and the number
         of active afferents per sub-pattern is drawn from a Gaussian with
         mean = N / K and std = sqrt(N / K), rounded and clipped to [0, N].
         An afferent can be active in several sub-patterns or in none.

For every (data, K, dataset), the capacity alpha_c = P / N is found by bisection
over the number of patterns P. This file needs no other project files.

Default (small local run): K = 10, 20, 50, 100, 150, 200, both data types,
5 datasets per point, max_epochs = 5000.

Usage
-----
    pip install numpy pandas joblib

    python capacity_data_scan.py --selftest       # quick check (a few seconds)

    # Locally on 7 cores (macOS: caffeinate keeps the Mac awake)
    caffeinate -i python -u capacity_data_scan.py --n-jobs 7 2>&1 | tee logs/capacity_data.log

Optional arguments:
    --data once gauss     data types to scan
    --Ks 10 20 ...        numbers of sub-patterns K
    --n-datasets 5        datasets per (data, K)
    --max-epochs 5000     maximum learning epochs per training run
    --n-jobs 7            parallel processes (default: CPUs given by SLURM, otherwise all cores but one)
    --tag once_K200       file-name suffix, needed when several jobs run at the same time
                          (e.g. one cluster job per data type and K): each job its own tag

Results
-------
One CSV in ./results/ (the file name encodes all parameters; with --tag, each
job writes ..._part-<tag>.csv). Every finished task (one dataset at one data
type and K) is saved immediately; tasks already in the CSV are skipped, so the
scan can be stopped and restarted at any time without losing finished work.

Model details: as in capacity_grid_scan.py. The learning rate is scaled as
lr = 0.438 / mean with mean = N / K, so that a learning step changes the
potential by the same amount relative to the threshold for every K.
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
LR_X_MEAN = LR_REF * MEAN_REF     # lr * mean is kept fixed across K
RESOLUTION = 10                   # P resolution of the bisection (alpha: 0.02)

DATA_TYPES = ["once", "gauss"]
KS = [10, 20, 50, 100, 150, 200]
N_DATASETS = 5
MAX_EPOCHS = 5000


# ===========================================================================
# Data
# ===========================================================================

def make_dataset_once(N, P, K, rng):
    """P patterns of K sub-patterns; every afferent is active in exactly one random sub-pattern."""
    k_of = rng.integers(0, K, size=(P, N))                      # sub-pattern of each afferent
    X = np.zeros((P, K, N), dtype=bool)
    X[np.arange(P)[:, None], k_of, np.arange(N)[None, :]] = True
    return X


def make_dataset_gauss(N, P, K, rng, mean_active, std_active):
    """P patterns of K independent sub-patterns; active count per sub-pattern ~ N(mean, std), clipped."""
    n_active = np.clip(np.rint(rng.normal(mean_active, std_active, size=(P, K))), 0, N).astype(int)
    X = np.zeros((P, K, N), dtype=bool)
    for p in range(P):
        for k in range(K):
            X[p, k, rng.choice(N, n_active[p, k], replace=False)] = True
    return X


def make_dataset(data, N, P, K, rng):
    """Dataset of the given type with on average N / K active afferents per sub-pattern."""
    if data == "once":
        return make_dataset_once(N, P, K, rng)
    if data == "gauss":
        mean = N / K
        return make_dataset_gauss(N, P, K, rng, mean_active=mean, std_active=np.sqrt(mean))
    raise ValueError(f"Unknown data type: {data}")


# ===========================================================================
# Multi-perceptron (same as in capacity_grid_scan.py)
# ===========================================================================

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


def capacity_bisection(data, N, K, theta, lr, seed, resolution, max_epochs, w_std, mu, credit,
                       P_start=1000, P_limit=40000):
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
            X_new = make_dataset(data, N, P - cache.get("P", 0), K, rng_data)
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


def scan_task(data, K, dataset, max_epochs):
    """Capacity of one dataset for one data type and one K."""
    mean = N / K
    lr = LR_X_MEAN / mean
    P_ok, P_fail = capacity_bisection(
        data, N, K, THETA, lr, seed=dataset, resolution=RESOLUTION, max_epochs=max_epochs,
        w_std=W_STD, mu=MU, credit=CREDIT,
    )
    return dict(mean=mean, P_ok=P_ok, P_fail=P_fail, alpha_c=(P_ok + P_fail) / 2 / N, lr=lr)


# ===========================================================================
# Result file and resumable parallel tasks
# ===========================================================================

def result_name(max_epochs, tag=""):
    """Result name; encodes all parameters that are not task columns."""
    name = (f"capacity_data_model-multi_perceptron_N-{N}_mean-N_over_K_std-sqrt"
            f"_lr_x_mean-{round(LR_X_MEAN, 4):g}_mu-{MU:g}_credit-{CREDIT}"
            f"_res-{RESOLUTION}_max_epochs-{max_epochs}")
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
    """Small, fast run to check the installation and both data types."""
    t0 = time.time()
    rng = np.random.default_rng(0)

    X = make_dataset_once(100, 20, 5, rng)
    assert (X.sum(axis=1) == 1).all(), "once: every afferent must be active in exactly one sub-pattern"
    print(f"once:  active per sub-pattern {X.sum(axis=2).mean():.1f} "
          f"(expected 100/5 = 20), every afferent exactly once: OK")
    X = make_dataset("gauss", 100, 20, 5, rng)
    print(f"gauss: active per sub-pattern {X.sum(axis=2).mean():.1f} (expected 20)")

    for data in DATA_TYPES:
        P_ok, P_fail = capacity_bisection(data, 100, 5, THETA, LR_X_MEAN / 20, seed=0,
                                          resolution=10, max_epochs=200, w_std=W_STD, mu=MU,
                                          credit=CREDIT, P_start=100)
        print(f"{data:5s}: capacity (N = 100, K = 5) between P = {P_ok} and {P_fail}")
    print(f"Parallel test: {Parallel(n_jobs=2)(delayed(np.sqrt)(i) for i in range(4))}")
    print(f"Selftest OK ({time.time() - t0:.0f} s)")


def main():
    parser = argparse.ArgumentParser(description="Multi-perceptron capacity for two data types vs. K")
    parser.add_argument("--data", nargs="+", default=DATA_TYPES, choices=DATA_TYPES,
                        help="data types to scan")
    parser.add_argument("--Ks", type=int, nargs="+", default=KS, help="numbers of sub-patterns K")
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--n-jobs", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", -2)),
                        help="parallel processes (default: CPUs given by SLURM, "
                             "otherwise all cores but one)")
    parser.add_argument("--tag", default="",
                        help="suffix for the result file when several jobs run at the same time")
    parser.add_argument("--selftest", action="store_true", help="quick check of the installation")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    print(f"Multi-perceptron capacity: data = {args.data}, Ks = {args.Ks}")
    print(f"N = {N}, mean = N / K, lr * mean = {LR_X_MEAN:.3g}, mu = {MU}, credit = {CREDIT}, "
          f"{args.n_datasets} datasets, max_epochs = {args.max_epochs}, n_jobs = {args.n_jobs}",
          flush=True)

    # Largest K first: these tasks take longest, so the cores stay busy until the end
    tasks = [dict(data=d, K=K, dataset=s)
             for K in sorted(args.Ks, reverse=True) for d in args.data
             for s in range(args.n_datasets)]

    t0 = time.time()
    df = run_tasks(result_name(args.max_epochs, args.tag), tasks,
                   partial(scan_task, max_epochs=args.max_epochs), args.n_jobs)

    summary = df.groupby(["K", "data"])["alpha_c"].agg(["mean", "std", "count"]).unstack("data")
    print("\nCapacity alpha_c (rows: K, columns: data type):")
    print(summary.round(3).to_string())
    print(f"\nTotal runtime: {(time.time() - t0) / 3600:.2f} h")


# Required for parallel processes: worker processes import this file,
# and only the main process may start the scan.
if __name__ == "__main__":
    main()