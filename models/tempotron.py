"""Tempotron (Gütig & Sompolinsky, 2006).

Classification of random spike patterns: the tempotron should fire at least once
for patterns with label 1 and stay silent for patterns with label 0. Two kinds of
input patterns:

  once     every afferent fires exactly one spike, uniformly distributed in [0, T)
           (Gütig & Sompolinsky 2006); make_dataset, data as a (P, N) array
  poisson  every afferent fires as a Poisson process with rate 1/T, i.e. on average
           one spike, but 0, 1, 2, ... spikes per pattern (Rubin et al. 2010);
           make_dataset_poisson, data as lists of spikes (time, afferent)

The membrane potential is integrated exactly (event-based): between two input
spikes, V is a sum of two exponentials, so its maximum and the first threshold
crossing are found analytically, without a time grid. Internally, every pattern
is a list of spikes (time, afferent), so both kinds of data use the same code.

Learning: tempotron rule evaluated at t_max with a momentum term, either with a
constant learning rate (train_exact, as in the 2006 paper) or with a decaying
learning rate (train_decay_exact, as in Gütig & Sompolinsky 2009).
"""

import time

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Standard parameters (defaults for the experiments in experiments/)
# ---------------------------------------------------------------------------

N = 500                           # afferents
T = 500.0                         # pattern duration in ms
TAU_M = 10.0                      # membrane time constant in ms
TAU_S = TAU_M / 4                 # synaptic time constant in ms
THETA = 1.0                       # firing threshold (V_rest = 0)
MU = 0.99                         # momentum
TAU_DECAY = 2000                  # learning rate decay in epochs (float("inf") = constant, as in 2006)
W_STD = 1e-3                      # std of the initial weights
# V0 and the learning rate LR = 1e-4 * V0 (Gütig & Sompolinsky 2006) are defined below kernel_norm


# ---------------------------------------------------------------------------
# Dataset and initial weights
# ---------------------------------------------------------------------------

def make_dataset(N, P, T, rng):
    """Generate random spike patterns with balanced labels (Gütig et al. 2006).

    Parameters
    ----------
    N   : number of afferents
    P   : number of patterns (must be even)
    T   : pattern duration in ms
    rng : numpy random Generator (first the spike times are drawn, then the labels
          are shuffled)

    Returns
    -------
    X : (P, N) array of spike times in ms; X[p, i] is the spike time of afferent i
        in pattern p (unsorted, in the order of the afferents)
    y : (P,) array of labels in {0, 1}
    """
    assert P % 2 == 0, "P must be even for an exact 50/50 label split"
    X = rng.uniform(0, T, size=(P, N))
    y = np.repeat([0, 1], P // 2)
    rng.shuffle(y)
    return X, y


def make_dataset_poisson(N, P, T, rng, rate=None):
    """Generate Poisson spike patterns with balanced labels (Rubin et al. 2010).

    Parameters
    ----------
    N, P, T, rng : as in make_dataset (P must be even)
    rate         : spikes per ms per afferent (default 1 / T)

    Returns
    -------
    times : list of P arrays, the spike times of each pattern in ms (unsorted);
            the arrays have different lengths
    affs  : list of P integer arrays of the same lengths: affs[p][s] is the afferent
            that emitted spike s of pattern p
    y     : (P,) array of labels in {0, 1}

    The result is meant for prepare_spikes, which turns it into an ExactData object.
    """
    assert P % 2 == 0, "P must be even for an exact 50/50 label split"
    rate = 1 / T if rate is None else rate
    times, affs = [], []
    for _ in range(P):
        counts = rng.poisson(rate * T, size=N)           # number of spikes of each afferent
        aff = np.repeat(np.arange(N), counts)            # e.g. counts [2, 0, 1] -> [0, 0, 2]
        times.append(rng.uniform(0, T, size=aff.size))
        affs.append(aff)
    y = np.repeat([0, 1], P // 2)
    rng.shuffle(y)
    return times, affs, y


def init_weights(N, rng, std=1e-3):
    """Draw initial weights from a Gaussian with mean 0 and standard deviation std.

    Parameters
    ----------
    N   : number of afferents
    rng : numpy random Generator
    std : standard deviation of the weights

    Returns
    -------
    w : (N,) array of initial weights
    """
    return rng.normal(0, std, size=N)


# ---------------------------------------------------------------------------
# Kernel and membrane potential
# ---------------------------------------------------------------------------

def kernel_norm(tau_m, tau_s):
    """Normalization factor V0 such that the kernel's maximum equals 1.

    Parameters
    ----------
    tau_m : membrane time constant in ms
    tau_s : synaptic time constant in ms (must differ from tau_m)

    Returns
    -------
    V0 : float, the normalization factor
    """
    r = tau_m / tau_s
    return r ** (r / (r - 1)) / (r - 1)


V0 = kernel_norm(TAU_M, TAU_S)    # kernel normalization for the standard time constants
LR = 1e-4 * V0                    # initial learning rate (Gütig & Sompolinsky 2006)


def kernel(s, tau_m, tau_s):
    """Postsynaptic potential kernel.

    Parameters
    ----------
    s     : time since the input spike in ms (number or array of any shape)
    tau_m : membrane time constant in ms
    tau_s : synaptic time constant in ms

    Returns
    -------
    K : array of the same shape as s
    """
    s = np.asarray(s, dtype=float)
    sp = np.maximum(s, 0.0)  # avoids exp of large positive numbers for s < 0
    k = kernel_norm(tau_m, tau_s) * (np.exp(-sp / tau_m) - np.exp(-sp / tau_s))
    return np.where(s >= 0, k, 0.0)


def decayed_lr(lr_ini, epoch, tau_decay):
    """Learning rate in a given epoch (1-based).

    Parameters
    ----------
    lr_ini    : learning rate in the first epoch
    epoch     : current epoch, starting at 1
    tau_decay : decay time in epochs; None or float("inf") gives a constant
                learning rate (as in the 2006 paper)

    Returns
    -------
    lr : float, the learning rate in this epoch
    """
    if tau_decay is None:
        return lr_ini
    return lr_ini / (1 + (epoch - 1) / tau_decay)


# ---------------------------------------------------------------------------
# Exact (event-based) integration
# ---------------------------------------------------------------------------

class ExactData:
    """Per-dataset quantities for exact integration, computed once by prepare_exact/prepare_spikes.

    Parameters (of ExactData(...); usually called through prepare_exact or prepare_spikes)
    ----------
    times : list of P arrays, the spike times of each pattern (any order)
    affs  : list of P integer arrays of the same lengths, the afferent of each spike
    N     : number of afferents (length of the weight vector)
    T     : pattern duration in ms
    tau_m : membrane time constant in ms
    tau_s : synaptic time constant in ms

    Attributes (lists with one array per pattern)
    ----------
    ts          : sorted spike times
    aff         : afferent of each sorted spike
    gaps        : length of the interval after each sorted spike (the last one up to T)
    em, es      : exp((ts - T/2) / tau) for tau_m and tau_s (scaled for numerical range)
    N, T, tau_m, tau_s, V0
    """

    def __init__(self, times, affs, N, T, tau_m, tau_s):
        assert T / (2 * tau_s) < 600, "T / tau_s too large for the exponential scaling"
        self.N, self.T, self.tau_m, self.tau_s = N, T, tau_m, tau_s
        self.V0 = kernel_norm(tau_m, tau_s)
        self.ts, self.aff, self.gaps, self.em, self.es = [], [], [], [], []
        for t, a in zip(times, affs):
            order = np.argsort(t)
            ts = np.asarray(t)[order]
            self.ts.append(ts)
            self.aff.append(np.asarray(a)[order])
            self.gaps.append(np.diff(ts, append=T))
            self.em.append(np.exp((ts - T / 2) / tau_m))
            self.es.append(np.exp((ts - T / 2) / tau_s))


def prepare_exact(X, T, tau_m, tau_s):
    """ExactData for once data (every afferent fires exactly once).

    Turns every row X[p] into a spike list: the spike times X[p] with the
    afferents 0, 1, ..., N-1.

    Parameters
    ----------
    X     : (P, N) array of spike times; X[p, i] is the spike time of afferent i in
            pattern p (as returned by make_dataset)
    T     : pattern duration in ms
    tau_m : membrane time constant in ms
    tau_s : synaptic time constant in ms

    Returns
    -------
    data : ExactData
    """
    P, N = X.shape
    return ExactData(list(X), [np.arange(N)] * P, N, T, tau_m, tau_s)


def prepare_spikes(times, affs, N, T, tau_m, tau_s):
    """ExactData for spike lists with any number of spikes per afferent.

    Parameters
    ----------
    times : list of P arrays of spike times (as returned by make_dataset_poisson)
    affs  : list of P integer arrays of the same lengths, the afferent of each spike
    N     : number of afferents
    T     : pattern duration in ms
    tau_m : membrane time constant in ms
    tau_s : synaptic time constant in ms

    Returns
    -------
    data : ExactData
    """
    return ExactData(times, affs, N, T, tau_m, tau_s)


def _interval_values(a, b, u, tau_m, tau_s, V0):
    """V(t_k + u) inside an interval between two input spikes.

    Between input spike k and input spike k + 1 no new input arrives, so the
    potential is V(t_k + u) = V0 * (a_k exp(-u/tau_m) - b_k exp(-u/tau_s)),
    with the interval coefficients a_k, b_k from _coefficients.

    Parameters
    ----------
    a, b         : interval coefficients (numbers or arrays)
    u            : time since the start of the interval in ms (number or array)
    tau_m, tau_s : time constants in ms
    V0           : kernel normalization

    Returns
    -------
    V : array, broadcast from a, b and u
    """
    return V0 * (a * np.exp(-u / tau_m) - b * np.exp(-u / tau_s))


def _interval_max(a, b, D, tau_m, tau_s, V0):
    """Maximum of V and its position u in each interval [0, D].

    The maximum of V(t_k + u) = V0 (a e^{-u/tau_m} - b e^{-u/tau_s}) on [0, D] lies
    at the start (u = 0), at the end (u = D), or where dV/du = 0, i.e. at
    u* = ln(b tau_m / (a tau_s)) / (1/tau_s - 1/tau_m). u* is a maximum only if
    a > 0 and b > 0, and lies in the interval only if 0 < u* < D. The function
    evaluates all candidates and takes the largest; it works on all intervals of a
    pattern at once.

    Parameters
    ----------
    a, b         : interval coefficients, numbers or arrays of the same length
    D            : interval lengths in ms (number or array of the same length)
    tau_m, tau_s : time constants in ms
    V0           : kernel normalization

    Returns
    -------
    v_max : array, maximal value of V in each interval
    u_max : array, position of the maximum within each interval (0 <= u_max <= D)
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


def _coefficients(data, p, w):
    """Interval coefficients a, b just after each sorted input spike of pattern p.

    Between spike k and spike k+1: V(t_k + u) = V0 * (a_k exp(-u/tau_m) - b_k exp(-u/tau_s)),
    with a_k = sum_{j <= k} w_j exp(-(t_k - t_j)/tau_m), the contributions of all spikes
    up to k decayed until t_k (b_k the same with tau_s). They are computed for all k at
    once as cumsum(w_j e^{t_j/tau}) / e^{t_k/tau}, with the precomputed exponentials
    data.em and data.es.

    Parameters
    ----------
    data : ExactData
    p    : pattern index
    w    : (N,) weights

    Returns
    -------
    a, b : arrays with one entry per spike of pattern p (in time order)
    """
    ws = w[data.aff[p]]                                 # weight of each spike
    a = np.cumsum(ws * data.em[p]) / data.em[p]
    b = np.cumsum(ws * data.es[p]) / data.es[p]
    return a, b


def forward_exact(data, p, w, theta, shunt=True):
    """Exact output decision, t_max and learning gradient for pattern p.

    Finds the maximum of V in every interval between input spikes (_interval_max).
    The tempotron fires if one of them reaches theta. With shunting, all inputs after
    the first output spike are ignored, so t_max is the maximum of the potential
    continued from that interval up to T. Without an output spike, t_max is the
    global maximum of V. The gradient of V(t_max) with respect to the weights is
    what the tempotron learning rule uses.

    Parameters
    ----------
    data  : ExactData from prepare_exact or prepare_spikes
    p     : pattern index
    w     : (N,) weights
    theta : firing threshold
    shunt : ignore inputs after the first output spike

    Returns
    -------
    spiked : bool, whether V reaches the threshold
    grad   : (N,) dV(t_max)/dw: for every afferent, the sum of K(t_max - t_s) over its
             non-shunted spikes s (one term per spike, 0 for afferents without spikes)
    t_max  : time of the maximal (shunted) potential
    V_max  : V(t_max)
    """
    tau_m, tau_s, V0 = data.tau_m, data.tau_s, data.V0
    ts, aff, D = data.ts[p], data.aff[p], data.gaps[p]
    if len(ts) == 0:                                    # no input spikes: V = 0 throughout
        return False, np.zeros(data.N), 0.0, 0.0
    a, b = _coefficients(data, p, w)

    v_max, u_max = _interval_max(a, b, D, tau_m, tau_s, V0)
    above = np.flatnonzero(v_max >= theta)
    spiked = above.size > 0

    if spiked and shunt:
        k = above[0]                                    # interval containing the output spike
        # No further inputs after the output spike: the interval extends up to T
        v_k, u_k = _interval_max(a[k], b[k], data.T - ts[k], tau_m, tau_s, V0)
        t_max, V_max = ts[k] + u_k[0], v_k[0]
        n_keep = k + 1                                  # spikes up to the output spike
    else:
        k = np.argmax(v_max)
        t_max, V_max = ts[k] + u_max[k], v_max[k]
        n_keep = len(ts)

    g = kernel(t_max - ts[:n_keep], tau_m, tau_s)      # 0 for spikes after t_max
    grad = np.bincount(aff[:n_keep], weights=g, minlength=data.N)
    return spiked, grad, float(t_max), float(V_max)


def potential_exact(data, p, w, t_eval, theta, shunt=True):
    """V(t) of pattern p evaluated at arbitrary times t_eval, with exact shunting.

    Sums w_i K(t - t_s) directly over the spikes. With shunting, only the spikes up
    to the first output spike count (found as in forward_exact). Slower than the
    interval formulas, but gives V at any times; for plotting and validation only,
    learning uses forward_exact.

    Parameters
    ----------
    data   : ExactData
    p      : pattern index
    w      : (N,) weights
    t_eval : times in ms at which V is evaluated (number or 1D array)
    theta  : firing threshold (determines where shunting starts)
    shunt  : ignore inputs after the first output spike

    Returns
    -------
    V : array of the same length as t_eval
    """
    ts, aff = data.ts[p], data.aff[p]
    n_keep = len(ts)
    if shunt and len(ts) > 0:
        a, b = _coefficients(data, p, w)
        v_max, _ = _interval_max(a, b, data.gaps[p], data.tau_m, data.tau_s, data.V0)
        above = np.flatnonzero(v_max >= theta)
        if above.size > 0:
            n_keep = above[0] + 1                       # spikes up to the output spike
    t_eval = np.asarray(t_eval, dtype=float)
    K_eval = kernel(t_eval[None, :] - ts[:n_keep, None], data.tau_m, data.tau_s)
    return w[aff[:n_keep]] @ K_eval


def spike_time_exact(data, p, w, theta):
    """Exact time of the first output spike (first threshold crossing) of pattern p.

    Finds the first interval in which V reaches theta. In this interval V starts
    below theta and reaches it at the latest at the interval maximum u_max, so
    V(t_k + u) - theta changes sign on [0, u_max]; its zero is found with Brent's
    method (scipy.optimize.brentq). Used for the window analysis (which inputs arrive
    shortly before the output spike).

    Parameters
    ----------
    data  : ExactData
    p     : pattern index
    w     : (N,) weights
    theta : firing threshold

    Returns
    -------
    t_spike : float, time of the first threshold crossing in ms, or None if V
              never reaches theta
    """
    tau_m, tau_s, V0 = data.tau_m, data.tau_s, data.V0
    ts, D = data.ts[p], data.gaps[p]
    if len(ts) == 0:
        return None
    a, b = _coefficients(data, p, w)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch_exact(w, dw, data, y, theta, lr, mu, rng):
    """One online learning cycle over all patterns in random order.

    On an error, dw = mu * dw ± lr * grad and w += dw, with + for a missed
    label-1 pattern and - for a label-0 pattern that fired (grad from
    forward_exact). Correct patterns cause no update. w and dw are modified
    in place.

    Parameters
    ----------
    w     : (N,) weights, updated in place
    dw    : (N,) last weight change (momentum memory), updated in place
    data  : ExactData
    y     : (P,) labels in {0, 1}
    theta : firing threshold
    lr    : learning rate in this epoch
    mu    : momentum: fraction of the previous update added to the new one
    rng   : random Generator for the order of the patterns

    Returns
    -------
    n_errors : number of misclassified patterns during this epoch (each counted
               when it was presented, i.e. with the weights at that moment)
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
    """Number of misclassified patterns with frozen weights (exact integration).

    Unlike run_epoch_exact, the weights are not changed, so every pattern is
    evaluated with the same w (e.g. to check a trained tempotron).

    Parameters
    ----------
    w     : (N,) weights
    data  : ExactData
    y     : (P,) labels in {0, 1}
    theta : firing threshold

    Returns
    -------
    n_errors : int, number of patterns for which "fires" differs from the label
    """
    return sum(forward_exact(data, p, w, theta)[0] != bool(y[p]) for p in range(len(y)))


def train_exact(data, y, theta, lr, mu, max_epochs, rng, w_init, verbose=25):
    """Train with a constant learning rate (2006) until an epoch is error-free.

    An error-free online epoch means no update happened, so all patterns are
    correctly classified with the final weights.

    Parameters
    ----------
    data       : ExactData from prepare_exact or prepare_spikes
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
    """Train with a decaying learning rate (Gütig & Sompolinsky 2009) until an epoch is error-free.

    Like train_exact, but the learning rate in epoch l is
    lr_ini / (1 + (l - 1) / tau_decay), see decayed_lr. An error-free online
    epoch means no update happened, so all patterns are correctly classified
    with the final weights.

    Parameters
    ----------
    data       : ExactData from prepare_exact or prepare_spikes
    y          : (P,) labels in {0, 1}
    theta      : firing threshold
    lr_ini     : learning rate in the first epoch (Gütig & Sompolinsky 2006: 1e-4 * V0)
    mu         : momentum (Gütig & Sompolinsky 2006: 0.99)
    max_epochs : maximum number of learning cycles
    rng        : random Generator for the pattern order
    w_init     : initial weights (not modified)
    tau_decay  : decay time of the learning rate in epochs (halved after tau_decay
                 epochs); float("inf") gives a constant learning rate
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