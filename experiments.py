"""Shared infrastructure for experiments: file names, saving, loading and resumable scans.

Model-agnostic: a scan calls a user-supplied function that returns, for one
parameter set, the arrays (epochs, converged, min_errors) with one entry per
dataset, as returned by learning_time_over_datasets in tempotron.py,
perceptron.py and multi_perceptron.py.

All results are stored in the folder results/ next to this file, so every
notebook in the project reads and writes the same files.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import json

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------

def _fmt(value):
    """Short, file-name friendly representation of a parameter value."""
    if isinstance(value, (float, np.floating)):
        return f"{value:g}"                       # 10.0 -> '10', 0.005 -> '0.005', inf -> 'inf'
    return str(value)


def make_name(base, **params):
    """Readable result name from a base name and the parameters that define the experiment.

    Example: make_name("capacity", model="tempotron", N=500, tau_m=10.0)
             -> 'capacity_model-tempotron_N-500_tau_m-10'

    Different parameters automatically give different files, so results of
    different settings are never mixed.
    """
    return "_".join([base] + [f"{k}-{_fmt(v)}" for k, v in params.items()])


def even_P(alpha, N):
    """Number of patterns for load alpha, rounded to an even number (exact 50/50 labels)."""
    return int(round(alpha * N / 2)) * 2


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def save_table(df, name):
    """Save a results table as results/<name>.csv."""
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_table(name):
    """Load results/<name>.csv, or return None if it does not exist yet."""
    path = RESULTS / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def delete_results(name):
    """Delete all result files results/<name>.* (to rerun an experiment from scratch)."""
    for path in RESULTS.glob(f"{name}.*"):
        path.unlink()
        print(f"Deleted {path}")


def cached_run(name, compute_fn):
    """Run compute_fn once and cache its result as results/<name>.npz.

    compute_fn : function without arguments returning a dict of arrays/numbers,
                 e.g. dict(w=w_trained, history=np.array(history), converged=conv)

    On later calls the stored result is loaded instead of recomputed.
    """
    path = RESULTS / f"{name}.npz"
    if path.exists():
        with np.load(path) as data:
            print(f"Loaded {path.name}")
            return {k: data[k] for k in data.files}
    result = compute_fn()
    np.savez(path, **result)
    print(f"Saved {path.name}")
    return result


def save_json(data, name):
    """Save a dict of numbers/strings as results/<name>.json."""
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {path.name}")


def load_json(name):
    """Load results/<name>.json, or return None if it does not exist yet."""
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# Resumable scans
# ---------------------------------------------------------------------------

def _key(values):
    """Hashable key for a parameter combination (floats rounded against CSV round-off)."""
    return tuple(round(float(v), 10) if isinstance(v, (float, np.floating)) else v for v in values)


def run_scan(name, param_list, run_fn, verbose=True):
    """Run run_fn for every parameter set that is not yet in results/<name>.csv.

    Parameters
    ----------
    name       : result name (e.g. from make_name)
    param_list : list of dicts with the scanned parameters, all with the same keys,
                 e.g. [dict(alpha=2.0), dict(alpha=2.2)]
    run_fn     : function(**params) -> (epochs, converged, min_errors), one entry per dataset

    The table has one row per (parameter set, dataset) with the parameters as
    columns. It is saved after every parameter set, so an interrupted scan
    continues where it stopped.

    Returns
    -------
    df : pandas DataFrame with all results (old and new)
    """
    key_cols = list(param_list[0].keys())
    df = load_table(name)
    rows = [] if df is None else df.to_dict("records")
    done = set() if df is None else {_key(r) for r in df[key_cols].itertuples(index=False)}

    t_start = time.time()
    for params in param_list:
        if _key(params[k] for k in key_cols) in done:
            if verbose:
                print(f"{params}: already done, skipped")
            continue
        if verbose:
            print(f"=== {params} ===")
        t0 = time.time()
        epochs, converged, min_errors = run_fn(**params)
        for s in range(len(epochs)):
            rows.append({**params, "dataset": s, "epochs": int(epochs[s]),
                         "converged": bool(converged[s]), "min_errors": int(min_errors[s])})
        df = pd.DataFrame(rows)
        save_table(df, name)
        if verbose:
            print(f"  {np.mean(converged):.0%} converged, saved ({time.time() - t0:.0f} s, "
                  f"total {(time.time() - t_start) / 60:.1f} min)")
    return df


def _call(task_fn, task):
    """Run one task and return its row (task parameters plus results); used by run_tasks."""
    return {**task, **task_fn(**task)}


def run_tasks(name, tasks, task_fn, n_jobs=1, verbose=True):
    """Run independent tasks in parallel, skipping those already in results/<name>.csv.

    Parameters
    ----------
    name    : result name (e.g. from make_name)
    tasks   : list of dicts with the task parameters, all with the same keys,
              e.g. [dict(K=8, dataset=0), dict(K=8, dataset=1)]
    task_fn : function(**task) -> dict of results, e.g. dict(P_ok=..., P_fail=...);
              must be defined in a module (not in a notebook) for n_jobs != 1
    n_jobs  : parallel processes (1 = sequential, -2 = all cores but one)

    Every finished task is saved immediately, so an interruption loses only the
    tasks that are running at that moment.

    Returns
    -------
    df : pandas DataFrame with one row per task (old and new)
    """
    from joblib import Parallel, delayed

    key_cols = list(tasks[0].keys())
    df = load_table(name)
    rows = [] if df is None else df.to_dict("records")
    done = set() if df is None else {_key(r) for r in df[key_cols].itertuples(index=False)}
    todo = [t for t in tasks if _key(t[k] for k in key_cols) not in done]
    if verbose:
        print(f"{len(tasks) - len(todo)} of {len(tasks)} tasks already done, {len(todo)} to run")

    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_call)(task_fn, t) for t in todo
    )
    for i, row in enumerate(results, 1):
        rows.append(row)
        save_table(pd.DataFrame(rows), name)
        if verbose:
            print(f"[{i}/{len(todo)}] {row}  (total {(time.time() - t_start) / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def summarize(df, by):
    """Fraction converged and learning time statistics (converged runs only) per value of `by`."""
    conv = df[df["converged"]]
    summary = df.groupby(by).agg(n=("converged", "size"), frac_converged=("converged", "mean"))
    stats = conv.groupby(by)["epochs"].agg(["median", "mean", "std", "count"])
    summary["median"] = stats["median"]
    summary["mean"] = stats["mean"]
    summary["sem"] = stats["std"] / np.sqrt(stats["count"])
    return summary


def estimate_crossing(x, frac, level=0.5):
    """Where frac (decreasing in x) crosses `level`, by linear interpolation.

    Returns
    -------
    estimate : float, or nan if the crossing lies outside the scanned range
    note     : short text describing the result
    """
    x, frac = np.asarray(x, dtype=float), np.asarray(frac, dtype=float)
    below = np.flatnonzero(frac < level)
    if below.size == 0:
        return np.nan, f"crossing above the scanned range (> {x[-1]:g})"
    if below[0] == 0:
        return np.nan, f"crossing below the scanned range (< {x[0]:g})"
    i = below[0]
    est = x[i - 1] + (frac[i - 1] - level) / (frac[i - 1] - frac[i]) * (x[i] - x[i - 1])
    return est, f"crossing at {est:.3f}"