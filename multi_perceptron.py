"""Perceptron with an OR rule over K binary sub-patterns.

Each pattern consists of K binary sub-patterns over N afferents. In each
sub-pattern, a random number of afferents is active (normally distributed,
mean 21.9, std 4.3, clipped at 0). The perceptron computes V_k = w . x_k for
every sub-pattern and fires if V_k >= theta for at least one k (OR rule).
It should fire for patterns with label 1 and stay silent for label 0.

Learning (tempotron-like):
- fired, but label 0: lower the weights of the afferents active in the
  sub-pattern that caused the spike
- silent, but label 1: raise the weights of the afferents active in the
  sub-pattern with the highest potential
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Dataset and initial weights
# ---------------------------------------------------------------------------

def make_dataset(N, P, K, rng, mean_active=21.9, std_active=4.3):
    """Generate P patterns of K binary sub-patterns with balanced labels.

    The number of active afferents per sub-pattern is drawn from a normal
    distribution, rounded and clipped to [0, N]; the active afferents are
    chosen uniformly at random without replacement.

    Returns
    -------
    X        : (P, K, N) bool array, X[p, k, i] = afferent i active in sub-pattern k of pattern p
    y        : (P,) labels in {0, 1}
    n_active : (P, K) number of active afferents per sub-pattern
    """
    assert P % 2 == 0, "P must be even for an exact 50/50 label split"
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


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(w, Xp, theta, credit="first"):
    """Output decision and responsible sub-pattern for one pattern.

    Parameters
    ----------
    w      : (N,) weights
    Xp     : (K, N) binary sub-patterns of this pattern
    theta  : firing threshold
    credit : which sub-pattern is held responsible for a spike:
             "first" = first sub-pattern reaching theta (analog to shunting),
             "max"   = sub-pattern with the highest potential

    Returns
    -------
    spiked : bool, whether V_k >= theta for at least one k
    k_star : index of the sub-pattern used for learning
             (the one causing the spike, or the one with maximal V if silent)
    V      : (K,) potentials of all sub-patterns
    """
    V = Xp @ w
    above = np.flatnonzero(V >= theta)
    spiked = above.size > 0
    if spiked and credit == "first":
        k_star = int(above[0])
    else:
        k_star = int(np.argmax(V))
    return spiked, k_star, V


def count_errors(w, X, y, theta):
    """Number of misclassified patterns with frozen weights."""
    V = X @ w                                     # (P, K)
    return int(np.sum((V.max(axis=1) >= theta) != y.astype(bool)))


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def run_epoch(w, dw, X, y, theta, lr, mu, rng, credit="first"):
    """One online learning cycle over all patterns in random order.

    On an error, dw = mu * dw ± lr * x_{k_star} and w += dw, with - for a
    label-0 pattern that fired and + for a missed label-1 pattern.
    w and dw are modified in place.

    Returns
    -------
    n_errors : number of misclassified patterns during this epoch
    """
    n_errors = 0
    for p in rng.permutation(len(y)):
        spiked, k, _ = forward(w, X[p], theta, credit)
        if spiked == bool(y[p]):
            continue
        n_errors += 1
        sign = -1.0 if spiked else 1.0
        dw *= mu
        dw += sign * lr * X[p, k]
        w += dw
    return n_errors


def train(X, y, theta, lr, max_epochs, rng, w_init, mu=0.0, credit="first", verbose=25):
    """Train until an epoch is error-free or max_epochs is reached.

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
        n_err = run_epoch(w, dw, X, y, theta, lr, mu, rng, credit)
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

# --------------------------------------------------------------------------------
# Helper functions for capacity curve and K-dependence
# --------------------------------------------------------------------------------
def learning_time_over_datasets(N, P, K, theta, lr, n_datasets=10, max_epochs=2000,
                                w_std=1e-3, print_progress=True, 
                                mean_active=21.9, std_active=4.3, **train_kwargs):
    """Train on n_datasets independent datasets of size P.

    Dataset s uses the seeds [s, 0] (data), [s, 1] (initial weights), [s, 2] (pattern order).
    train_kwargs are passed to train (e.g. mu, credit).

    Returns
    -------
    epochs, converged, min_errors : (n_datasets,) arrays
    """
    epochs, converged, min_errors = [], [], []
    for s in range(n_datasets):
        X_s, y_s, _ = make_dataset(N, P, K, np.random.default_rng([s, 0]),
                                   mean_active=mean_active, std_active=std_active)
        w0_s = init_weights(N, np.random.default_rng([s, 1]), std=w_std)
        _, h, conv = train(X_s, y_s, theta, lr, max_epochs, np.random.default_rng([s, 2]),
                           w0_s, verbose=0, **train_kwargs)
        epochs.append(len(h))
        converged.append(conv)
        min_errors.append(min(h))
        if print_progress:
            print(f"  dataset {s}: converged = {conv!s:5}  epochs = {len(h):5d}  min errors = {min(h)}")
    return np.array(epochs), np.array(converged), np.array(min_errors)


def capacity_bisection(N, K, theta, lr, seed, P_start=1000, resolution=10,
                       max_epochs=2000, w_std=1e-3, P_limit=40000,
                       mean_active=21.9, std_active=4.3, **train_kwargs):
    """Largest learnable number of patterns for one dataset, found by bisection.

    Uses nested datasets: the problem with P patterns consists of the first P
    patterns of one large dataset. Since a subset of a learnable set is
    learnable, "converges" is monotone in P and bisection applies. Labels
    alternate 0, 1, 0, 1, ... so every prefix is balanced (the patterns are
    i.i.d., so the label order carries no information).

    Note: "not learnable" means "not learned within max_epochs", so this is an
    operational capacity, slightly below the true one.

    Returns
    -------
    P_ok   : largest P that was learned
    P_fail : smallest P that was not learned (P_fail - P_ok <= resolution)
    """
    rng_data = np.random.default_rng([seed, 0])
    w0 = init_weights(N, np.random.default_rng([seed, 1]), std=w_std)
    cache = {}                                     # the dataset is grown on demand

    def dataset(P):
        if P > cache.get("P", 0):
            X_new, _, _ = make_dataset(N, P - cache.get("P", 0), K, rng_data,
                                       mean_active=mean_active, std_active=std_active)
            cache["X"] = X_new if "X" not in cache else np.concatenate([cache["X"], X_new])
            cache["P"] = P
        return cache["X"][:P], np.arange(P) % 2

    def learnable(P):
        X, y = dataset(P)
        _, _, conv = train(X, y, theta, lr, max_epochs, np.random.default_rng([seed, 2]),
                           w0, verbose=0, **train_kwargs)
        return conv

    # Bracket the capacity: grow the upper end until learning fails
    lo, hi = 0, P_start
    while learnable(hi):
        lo, hi = hi, 2 * hi
        if hi > P_limit:
            raise RuntimeError(f"Still learnable at P = {lo}; increase P_limit")

    # Bisection on even P
    while hi - lo > resolution:
        mid = (lo + hi) // 4 * 2
        if learnable(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi