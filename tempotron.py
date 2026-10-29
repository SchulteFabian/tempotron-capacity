"""Tempotron (Gütig & Sompolinsky, 2006).

Classification of random spike patterns: each of N afferents fires exactly one
spike, uniformly distributed in [0, T). The tempotron should fire at least once
for patterns with label 1 and stay silent for patterns with label 0.

The membrane potential is integrated exactly (event-based): between two input
spikes, V is a sum of two exponentials, so its maximum and the first
threshold crossing are found analytically, without a time grid.

Learning: tempotron rule evaluated at t_max with a momentum term, either with a
constant learning rate (train_exact, as in the 2006 paper) or with a decaying
learning rate (train_decay_exact, as in Gütig & Sompolinsky 2009).

Reference: R. Gütig and H. Sompolinsky, "The tempotron: a neuron that learns
spike timing-based decisions", Nat. Neurosci. 9, 420-428 (2006).
"""

import time

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Dataset and initial weights
# ---------------------------------------------------------------------------

def make_dataset(N, P, T, rng):
    """Generate random spike patterns with balanced labels.

    Each afferent fires exactly once, at a time drawn uniformly from [0, T).
    Exactly P/2 patterns get label 1, the rest label 0.

    Parameters
    ----------
    N   : number of afferents
    P   : number of patterns (must be even)
    T   : pattern duration in ms
    rng : numpy random Generator

    Returns
    -------
    X : (P, N) array of spike times in ms
    y : (P,) array of labels in {0, 1}
    """
    assert P % 2 == 0, "P must be even for an exact 50/50 label split"
    X = rng.uniform(0, T, size=(P, N))
    y = np.repeat([0, 1], P // 2)
    rng.shuffle(y)
    return X, y


def init_weights(N, rng, std=1e-3):
    """Draw initial weights from a Gaussian with mean 0 and standard deviation std.

    The default std = 1e-3 follows Gütig & Sompolinsky (2006).
    """
    return rng.normal(0, std, size=N)


# ---------------------------------------------------------------------------
# Kernel and membrane potential
# ---------------------------------------------------------------------------

def kernel_norm(tau_m, tau_s):
    """Normalization factor V0 such that the kernel's maximum equals 1.

    The maximum of exp(-s/tau_m) - exp(-s/tau_s) lies at
    s* = tau_m * tau_s / (tau_m - tau_s) * ln(tau_m / tau_s).
    """
    r = tau_m / tau_s
    return r ** (r / (r - 1)) / (r - 1)


def kernel(s, tau_m, tau_s):
    """Postsynaptic potential kernel.

    K(s) = V0 * (exp(-s/tau_m) - exp(-s/tau_s)) for s >= 0, and 0 for s < 0.
    """
    s = np.asarray(s, dtype=float)
    sp = np.maximum(s, 0.0)  # avoids exp of large positive numbers for s < 0
    k = kernel_norm(tau_m, tau_s) * (np.exp(-sp / tau_m) - np.exp(-sp / tau_s))
    return np.where(s >= 0, k, 0.0)


def decayed_lr(lr_ini, epoch, tau_decay):
    """Learning rate in a given epoch (1-based).

    lr(l) = lr_ini / (1 + (l - 1) / tau_decay).
    The 2009 paper uses tau_decay = 1e4, i.e. lr_ini / (1 + 1e-4 (l - 1)).
    tau_decay = None or float("inf") gives a constant learning rate.
    """
    if tau_decay is None:
        return lr_ini
    return lr_ini / (1 + (epoch - 1) / tau_decay)


class ExactData:
    """Per-dataset quantities for exact integration, computed once by prepare_exact.

    Attributes
    ----------
    X           : (P, N) original spike times
    order       : (P, N) indices that sort each pattern's spike times
    ts          : (P, N) sorted spike times
    gaps        : (P, N) length of the interval after each sorted spike (last one up to T)
    em, es      : (P, N) exp((ts - T/2) / tau) for tau_m and tau_s (scaled for numerical range)
    T, tau_m, tau_s, V0
    """

    def __init__(self, X, T, tau_m, tau_s):
        assert T / (2 * tau_s) < 600, "T / tau_s too large for the exponential scaling"
        self.X = X
        self.T, self.tau_m, self.tau_s = T, tau_m, tau_s
        self.V0 = kernel_norm(tau_m, tau_s)
        self.order = np.argsort(X, axis=1)
        self.ts = np.take_along_axis(X, self.order, axis=1)
        self.gaps = np.diff(self.ts, axis=1, append=T)
        self.em = np.exp((self.ts - T / 2) / tau_m)
        self.es = np.exp((self.ts - T / 2) / tau_s)


def prepare_exact(X, T, tau_m, tau_s):
    """Sort the spikes of all patterns and precompute the exponentials (once per dataset).

    Memory is about 5 * P * N * 8 bytes, e.g. 20 MB for N = 500, P = 1000.
    """
    return ExactData(X, T, tau_m, tau_s)


def _interval_values(a, b, u, tau_m, tau_s, V0):
    """V(t_k + u) for interval coefficients a, b (all arrays broadcast)."""
    return V0 * (a * np.exp(-u / tau_m) - b * np.exp(-u / tau_s))


def _interval_max(a, b, D, tau_m, tau_s, V0):
    """Maximum of V and its position u in each interval [0, D].

    Returns
    -------
    v_max : maximal value of V in each interval
    u_max : position of the maximum within the interval
    """
    a, b, D = np.atleast_1d(a, b, D)
    c = 1 / tau_s - 1 / tau_m
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (b * tau_m) / (a * tau_s)
        u_star = np.where((a > 0) & (b > 0) & (ratio > 1), np.log(ratio) / c, np.nan)
    inside = (u_star > 0) & (u_star < D)

    v0 = V0 * (a - b)                                   # at the start of the interval
    vD = _interval_values(a, b, D, tau_m, tau_s, V0)    # at the end of the interval
    vs = np.where(inside, _interval_values(a, b, np.where(inside, u_star, 0.0), tau_m, tau_s, V0), -np.inf)

    cand_v = np.stack([v0, vD, vs])
    cand_u = np.stack([np.zeros_like(D), D, np.where(inside, u_star, 0.0)])
    i = np.argmax(cand_v, axis=0)
    idx = np.arange(cand_v.shape[1])
    return cand_v[i, idx], cand_u[i, idx]


def forward_exact(data, p, w, theta, shunt=True):
    """Exact output decision, t_max and learning gradient for pattern p.

    Parameters
    ----------
    data  : ExactData from prepare_exact
    p     : pattern index
    w     : (N,) weights
    theta : firing threshold
    shunt : ignore inputs after the first output spike

    Returns
    -------
    spiked : bool, whether V reaches the threshold
    grad   : (N,) K(t_max - t_i) for all non-shunted inputs, i.e. dV(t_max)/dw
    t_max  : time of the maximal (shunted) potential
    V_max  : V(t_max)
    """
    tau_m, tau_s, V0 = data.tau_m, data.tau_s, data.V0
    ts, order, D = data.ts[p], data.order[p], data.gaps[p]
    ws = w[order]

    # Interval coefficients just after each sorted input spike
    a = np.cumsum(ws * data.em[p]) / data.em[p]
    b = np.cumsum(ws * data.es[p]) / data.es[p]

    v_max, u_max = _interval_max(a, b, D, tau_m, tau_s, V0)
    above = np.flatnonzero(v_max >= theta)
    spiked = above.size > 0

    if spiked and shunt:
        k = above[0]                                    # interval containing the output spike
        # No further inputs after the output spike: the interval extends up to T
        D_ext = data.T - ts[k]
        v_k, u_k = _interval_max(a[k], b[k], D_ext, tau_m, tau_s, V0)
        t_max, V_max = ts[k] + u_k[0], v_k[0]
        keep = np.arange(len(ts)) <= k                  # inputs before the output spike
    else:
        k = np.argmax(v_max)
        t_max, V_max = ts[k] + u_max[k], v_max[k]
        keep = None

    g_sorted = kernel(t_max - ts, tau_m, tau_s)         # 0 for inputs after t_max
    if keep is not None:
        g_sorted = g_sorted * keep
    grad = np.empty_like(g_sorted)
    grad[order] = g_sorted
    return spiked, grad, float(t_max), float(V_max)


def potential_exact(data, p, w, t_eval, theta, shunt=True):
    """V(t) of pattern p evaluated at arbitrary times t_eval, with exact shunting.

    For plotting and validation only; learning uses forward_exact.
    """
    spiked, _, _, _ = forward_exact(data, p, w, theta, shunt=False)
    X_p = data.X[p]
    w_eff = w
    if spiked and shunt:
        # Inputs up to and including the interval with the output spike are kept
        ts, order, D = data.ts[p], data.order[p], data.gaps[p]
        ws = w[order]
        a = np.cumsum(ws * data.em[p]) / data.em[p]
        b = np.cumsum(ws * data.es[p]) / data.es[p]
        v_max, _ = _interval_max(a, b, D, data.tau_m, data.tau_s, data.V0)
        k = np.flatnonzero(v_max >= theta)[0]
        w_eff = np.where(X_p <= ts[k], w, 0.0)
    K_eval = kernel(np.asarray(t_eval)[None, :] - X_p[:, None], data.tau_m, data.tau_s)
    return w_eff @ K_eval


def run_epoch_exact(w, dw, data, y, theta, lr, mu, rng):
    """One online learning cycle over all patterns in random order.

    On an error, dw = mu * dw ± lr * grad and w += dw, with + for a missed
    label-1 pattern and - for a label-0 pattern that fired (grad from
    forward_exact). Correct patterns cause no update. w and dw are modified
    in place.

    Returns
    -------
    n_errors : number of misclassified patterns during this epoch
    """
    n_errors = 0
    for p in rng.permutation(len(y)):
        spiked, grad, _, _ = forward_exact(data, p, w, theta)
        if spiked == bool(y[p]):
            continue
        n_errors += 1
        sign = -1.0 if spiked else 1.0
        dw *= mu
        dw += sign * lr * grad
        w += dw
    return n_errors


def count_errors_exact(w, data, y, theta):
    """Number of misclassified patterns with frozen weights (exact integration)."""
    return sum(forward_exact(data, p, w, theta)[0] != bool(y[p]) for p in range(len(y)))


def train_exact(data, y, theta, lr, mu, max_epochs, rng, w_init, verbose=25):
    """Train with a constant learning rate (2006) until an epoch is error-free.

    An error-free online epoch means no update happened, so all patterns are
    correctly classified with the final weights.

    Parameters
    ----------
    data       : ExactData from prepare_exact
    y          : (P,) labels in {0, 1}
    theta      : firing threshold
    lr         : learning rate (Gütig & Sompolinsky 2006: 1e-4 * V0)
    mu         : momentum (Gütig & Sompolinsky 2006: 0.99)
    max_epochs : maximum number of learning cycles
    rng        : random Generator for the pattern order
    w_init     : initial weights (not modified)
    verbose    : print progress every `verbose` epochs; 0 = silent

    Returns
    -------
    w         : trained weights
    history   : list of errors per epoch
    converged : bool
    """
    w = w_init.copy()
    dw = np.zeros_like(w)
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        n_err = run_epoch_exact(w, dw, data, y, theta, lr, mu, rng)
        history.append(n_err)
        if verbose and epoch % verbose == 0:
            print(f"Epoch {epoch:5d}: {n_err:4d} errors  ({time.time() - t0:.0f} s)")
        if n_err == 0:
            if verbose:
                print(f"Converged after {epoch} epochs ({time.time() - t0:.0f} s)")
            return w, history, True
    if verbose:
        print(f"Not converged after {max_epochs} epochs (min. {min(history)} errors)")
    return w, history, False


def train_decay_exact(data, y, theta, lr_ini, mu, max_epochs, rng, w_init,
                      tau_decay=1e4, verbose=25):
    """Like train_exact, but with a decaying learning rate (Gütig & Sompolinsky 2009).

    The learning rate in epoch l is lr_ini / (1 + (l - 1) / tau_decay), see
    decayed_lr; tau_decay = float("inf") gives a constant learning rate.
    """
    w = w_init.copy()
    dw = np.zeros_like(w)
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        lr = decayed_lr(lr_ini, epoch, tau_decay)
        n_err = run_epoch_exact(w, dw, data, y, theta, lr, mu, rng)
        history.append(n_err)
        if verbose and epoch % verbose == 0:
            print(f"Epoch {epoch:5d}: {n_err:4d} errors  lr={lr:.2e}  ({time.time() - t0:.0f} s)")
        if n_err == 0:
            if verbose:
                print(f"Converged after {epoch} epochs ({time.time() - t0:.0f} s)")
            return w, history, True
    if verbose:
        print(f"Not converged after {max_epochs} epochs (min. {min(history)} errors)")
    return w, history, False


def _run_one_dataset(s, train_fn, N, P, T, tau_m, tau_s, theta, max_epochs, w_std, train_kwargs):
    """Train on dataset s (seeds [s, 0], [s, 1], [s, 2]); used by learning_time_over_datasets."""
    t0 = time.time()
    X_s, y_s = make_dataset(N, P, T, np.random.default_rng([s, 0]))
    data_s = prepare_exact(X_s, T, tau_m, tau_s)
    w0_s = init_weights(N, np.random.default_rng([s, 1]), std=w_std)
    _, h, conv = train_fn(data_s, y_s, theta, max_epochs=max_epochs,
                          rng=np.random.default_rng([s, 2]), w_init=w0_s, verbose=0,
                          **train_kwargs)
    return len(h), conv, min(h), time.time() - t0


def learning_time_over_datasets(train_fn, N, P, T, tau_m, tau_s, theta,
                                n_datasets=10, max_epochs=3000, w_std=1e-3,
                                n_jobs=1, print_progress=True, **train_kwargs):
    """Train on n_datasets independent datasets and report the learning times.

    Dataset s uses the seeds [s, 0] (data), [s, 1] (initial weights) and
    [s, 2] (pattern order), so different learning rules are evaluated on
    exactly the same datasets.

    Parameters
    ----------
    train_fn       : train_exact or train_decay_exact
    N, P, T        : afferents, patterns, pattern duration in ms
    tau_m, tau_s   : membrane and synaptic time constants in ms
    theta          : firing threshold
    n_datasets     : number of independent datasets
    max_epochs     : maximum number of learning cycles per dataset
    w_std          : std of the initial weights
    n_jobs         : datasets trained in parallel (1 = sequential, -1 = all cores)
    print_progress : print per-dataset results and a summary
    train_kwargs   : learning parameters passed to train_fn
                     (lr, mu for train_exact; lr_ini, mu, tau_decay for train_decay_exact)

    Returns
    -------
    epochs     : (n_datasets,) number of epochs per run
    converged  : (n_datasets,) bool
    min_errors : (n_datasets,) minimum number of errors during training
    """
    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10 if print_progress else 0)(
        delayed(_run_one_dataset)(s, train_fn, N, P, T, tau_m, tau_s, theta,
                                  max_epochs, w_std, train_kwargs)
        for s in range(n_datasets)
    )
    epochs = np.array([r[0] for r in results])
    converged = np.array([r[1] for r in results])
    min_errors = np.array([r[2] for r in results])

    if print_progress:
        for s, (e, c, m, dt) in enumerate(results):
            print(f"Dataset {s}: converged = {c!s:5}  epochs = {e:5d}  "
                  f"min errors = {m:4d}  ({dt:.0f} s)")
        ep = epochs[converged]
        print(f"\nN = {N}, P = {P}, alpha = {P / N:.2f}, {train_kwargs}")
        print(f"Converged: {converged.sum()} / {n_datasets}")
        if len(ep) > 1:
            print(f"Learning time (converged runs): mean = {ep.mean():.1f} ± "
                  f"{ep.std(ddof=1) / np.sqrt(len(ep)):.1f} (SEM), "
                  f"median = {np.median(ep):.0f}, range = {ep.min()}–{ep.max()}")
        print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")
    return epochs, converged, min_errors


def spike_time_exact(data, p, w, theta):
    """Exact time of the first output spike (first threshold crossing) of pattern p.

    Returns None if V never reaches theta.
    """
    tau_m, tau_s, V0 = data.tau_m, data.tau_s, data.V0
    ts, order, D = data.ts[p], data.order[p], data.gaps[p]
    ws = w[order]
    a = np.cumsum(ws * data.em[p]) / data.em[p]
    b = np.cumsum(ws * data.es[p]) / data.es[p]

    v_max, u_max = _interval_max(a, b, D, tau_m, tau_s, V0)
    above = np.flatnonzero(v_max >= theta)
    if above.size == 0:
        return None
    k = above[0]                                   # first interval in which V reaches theta

    def f(u):
        return _interval_values(a[k], b[k], u, tau_m, tau_s, V0) - theta

    if f(0.0) >= 0:                                # threshold reached exactly at the input spike
        return float(ts[k])
    return float(ts[k] + brentq(f, 0.0, u_max[k]))
