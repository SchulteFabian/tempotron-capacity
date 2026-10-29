"""Classic perceptron (Rosenblatt) as a reference model.

Random dense patterns x in {0, 1}^N with balanced labels. The perceptron
outputs 1 if w . x >= theta (default theta = 0, no bias).

For this setting the capacity is known exactly: at P = 2N, half of all random
problems are linearly separable (Cover 1965), and the perceptron rule is
guaranteed to find a solution whenever one exists (perceptron convergence
theorem). It therefore serves as a check of the capacity measurement.
"""

import time

import numpy as np


def make_dataset(N, P, rng, f=0.5):
    """P random binary patterns with balanced labels.

    Each input is 1 with probability f (coding level) and 0 otherwise,
    independently. Exactly P/2 patterns get label 1, the rest label 0.

    Returns
    -------
    X : (P, N) array with entries in {0, 1} (as floats)
    y : (P,) labels in {0, 1}
    """
    assert P % 2 == 0, "P must be even for an exact 50/50 label split"
    X = (rng.random((P, N)) < f).astype(float)
    y = np.repeat([0, 1], P // 2)
    rng.shuffle(y)
    return X, y


def init_weights(N, rng, std=1e-3):
    """Initial weights from a Gaussian with mean 0 and standard deviation std."""
    return rng.normal(0, std, size=N)


def count_errors(w, X, y, theta=0.0):
    """Number of misclassified patterns with frozen weights."""
    return int(np.sum((X @ w >= theta) != y.astype(bool)))


def run_epoch(w, X, y, theta, lr, rng):
    """One online cycle of the classic perceptron rule: w += ± lr * x on errors.

    w is modified in place. Returns the number of errors during the epoch.
    """
    n_errors = 0
    for p in rng.permutation(len(y)):
        fired = X[p] @ w >= theta
        if fired == bool(y[p]):
            continue
        n_errors += 1
        w += (-lr if fired else lr) * X[p]
    return n_errors


def train(X, y, lr, max_epochs, rng, w_init, theta=0.0, verbose=25):
    """Train until an epoch is error-free or max_epochs is reached.

    Returns
    -------
    w, history, converged
    """
    w = w_init.copy()
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        n_err = run_epoch(w, X, y, theta, lr, rng)
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


def learning_time_over_datasets(N, P, lr, n_datasets=10, max_epochs=5000, w_std=1e-3,
                                theta=0.0, f=0.5, print_progress=True):
    """Train on n_datasets independent datasets (seeds [s, 0], [s, 1], [s, 2]).

    Returns
    -------
    epochs, converged, min_errors : (n_datasets,) arrays
    """
    epochs, converged, min_errors = [], [], []
    for s in range(n_datasets):
        X_s, y_s = make_dataset(N, P, np.random.default_rng([s, 0]), f=f)
        w0_s = init_weights(N, np.random.default_rng([s, 1]), std=w_std)
        _, h, conv = train(X_s, y_s, lr, max_epochs, np.random.default_rng([s, 2]), w0_s,
                           theta=theta, verbose=0)
        epochs.append(len(h))
        converged.append(conv)
        min_errors.append(min(h))
        if print_progress:
            print(f"  dataset {s}: converged = {conv!s:5}  epochs = {len(h):5d}  min errors = {min(h)}")
    return np.array(epochs), np.array(converged), np.array(min_errors)