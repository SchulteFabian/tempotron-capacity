"""Shared infrastructure for the experiments: file names, saving, loading and resumable scans.

Every experiment script in this folder uses these functions, and Summary.ipynb
uses load_results to read what the scripts computed. All results are stored in
the folder results/ of the project, all figures in figures/.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


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


def part_name(name, tag):
    """Name of the part file of one job, when several jobs run at the same time (e.g. on a cluster)."""
    return f"{name}_part-{tag}" if tag else name


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


def save_arrays(arrays, name):
    """Save a dict of arrays/numbers as results/<name>.npz."""
    path = RESULTS / f"{name}.npz"
    np.savez(path, **arrays)
    print(f"Saved {path.name}")


def load_arrays(name):
    """Load results/<name>.npz as a dict, or return None if it does not exist yet."""
    path = RESULTS / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_json(data, name):
    """Save a dict of numbers/strings as results/<name>.json."""
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {path.name}")


def load_json(name):
    """Load results/<name>.json, or return None if it does not exist yet."""
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cached_run(name, compute_fn):
    """Run compute_fn once and cache its result as results/<name>.npz.

    compute_fn : function without arguments returning a dict of arrays/numbers,
                 e.g. dict(w=w_trained, history=np.array(history), converged=conv)

    On later calls the stored result is loaded instead of recomputed.
    """
    result = load_arrays(name)
    if result is not None:
        print(f"Loaded {name}.npz")
        return result
    result = compute_fn()
    save_arrays(result, name)
    return result


def delete_results(name):
    """Delete all result files results/<name>.* (to rerun an experiment from scratch)."""
    for path in RESULTS.glob(f"{name}.*"):
        path.unlink()
        print(f"Deleted {path}")


def load_results(name, script, kind="csv"):
    """Load a result for the Summary notebook; if it is missing, say which script computes it.

    kind : "csv" (table), "npz" (dict of arrays) or "json" (dict)
    """
    loader = {"csv": load_table, "npz": load_arrays, "json": load_json}[kind]
    result = loader(name)
    if result is None:
        raise FileNotFoundError(
            f"results/{name}.{kind} does not exist yet. Compute it from the project folder with:\n"
            f"    python -m experiments.{script}"
        )
    return result


def merge_parts(name, folder=None):
    """Merge the part files <name>_part-*.csv of several jobs into results/<name>.csv.

    folder : where the part files are (default: results/); the merged file
             always goes to results/. Returns the merged DataFrame.
    """
    folder = RESULTS if folder is None else Path(folder).expanduser()
    parts = sorted(folder.glob(f"{name}_part-*.csv"))
    if not parts:
        raise FileNotFoundError(f"No files {name}_part-*.csv in {folder}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    path = save_table(df, name)
    print(f"{len(parts)} part files, {len(df)} rows -> {path}")
    return df


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(description, add_arguments=None):
    """Common command-line options of all experiment scripts.

    --n-jobs : parallel processes (default: CPUs given by SLURM, otherwise all cores but one)
    --tag    : suffix of the result file when several jobs run at the same time
               (e.g. one cluster job per grid point): every job gets its own tag
    --merge  : merge the part files of several jobs into one result file instead of computing

    add_arguments : optional function(parser) that adds script-specific options
    """
    parser = argparse.ArgumentParser(description=description)
    if add_arguments is not None:
        add_arguments(parser)
    parser.add_argument("--n-jobs", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", -2)),
                        help="parallel processes (default: CPUs given by SLURM, "
                             "otherwise all cores but one)")
    parser.add_argument("--tag", default="",
                        help="suffix of the result file when several jobs run at the same time")
    parser.add_argument("--merge", nargs="?", const="", default=None, metavar="FOLDER",
                        help="merge the part files (in FOLDER, default results/) instead of computing")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resumable parallel tasks
# ---------------------------------------------------------------------------

def _key(values):
    """Hashable key for a task (floats rounded against CSV round-off)."""
    return tuple(round(float(v), 10) if isinstance(v, (float, np.floating)) else v for v in values)


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
    task_fn : function(**task) -> dict of results, e.g. dict(P_ok=..., P_fail=...)
    n_jobs  : parallel processes (1 = sequential, -2 = all cores but one)

    Every finished task is saved immediately, so an interruption loses only the
    tasks that are running at that moment, and a restart continues where it stopped.

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
        print(f"Results -> results/{name}.csv")
        print(f"{len(tasks) - len(todo)} of {len(tasks)} tasks already done, {len(todo)} to run\n",
              flush=True)

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


def run_experiment(args, name, tasks, task_fn):
    """What every scan script does in main(): merge part files, or run the tasks.

    args    : result of parse_args (uses --merge, --tag, --n-jobs)
    name    : result name of the full experiment
    tasks, task_fn : as in run_tasks

    Returns the DataFrame of results (merged or computed).
    """
    if args.merge is not None:
        return merge_parts(name, args.merge or None)
    t_start = time.time()
    df = run_tasks(part_name(name, args.tag), tasks, task_fn, n_jobs=args.n_jobs)
    print(f"\nTotal runtime: {(time.time() - t_start) / 3600:.2f} h")
    return df


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


def summarize_capacity(df, by):
    """Mean, std, count and SEM of alpha_c per value of `by` (for bisection scans)."""
    summary = df.groupby(by)["alpha_c"].agg(["mean", "std", "count"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
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
