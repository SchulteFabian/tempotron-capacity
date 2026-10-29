"""Shared parameters for scripts and notebooks.

Scripts (e.g. run_capacity_tempotron.py) and the presentation notebook import
their parameters from here. Since result file names are built from these
parameters (experiments.make_name), both always read and write the same files.

Change a parameter here, and both the script and the notebook automatically
use a new result file.
"""

import numpy as np

import experiments as ex
import perceptron as pc
import tempotron as tp
import multi_perceptron as mpc


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

PC_N = 500                        # input dimension
PC_F = 0.5                        # coding level: probability that an input is active
PC_THETA = 0.0                    # threshold (0 = classic perceptron without bias)
PC_LR = 1.0                       # learning rate (irrelevant for theta = 0: it only rescales w)
PC_W_STD = 1e-3                   # std of the initial weights

# Single training run (section 1)
PC_SINGLE_P = 800                 # patterns (alpha = 1.6)
PC_SINGLE_MAX_EPOCHS = 20000
PC_SINGLE_SEEDS = dict(data=0, init=1, order=2)

# Capacity scan
PC_CAP_ALPHAS = [1.6, 1.8, 1.9, 1.96, 2.0, 2.04, 2.1, 2.2, 2.4]
PC_CAP_N_DATASETS = 10
PC_CAP_MAX_EPOCHS = 20000


def pc_single_name():
    """File name of the single perceptron training run."""
    return ex.make_name("single", model="perceptron", N=PC_N, P=PC_SINGLE_P, f=PC_F)


def pc_capacity_name():
    """File name of the perceptron capacity scan (shared by script and notebook)."""
    return ex.make_name("capacity", model="perceptron", N=PC_N, f=PC_F,
                        n=PC_CAP_N_DATASETS, max_epochs=PC_CAP_MAX_EPOCHS)


def pc_capacity_run(alpha, print_progress=False):
    """Train the perceptron on all datasets for one load alpha (one step of the capacity scan)."""
    return pc.learning_time_over_datasets(
        PC_N, ex.even_P(alpha, PC_N), PC_LR, n_datasets=PC_CAP_N_DATASETS,
        max_epochs=PC_CAP_MAX_EPOCHS, w_std=PC_W_STD, theta=PC_THETA, f=PC_F,
        print_progress=print_progress,
    )


# ---------------------------------------------------------------------------
# Tempotron
# ---------------------------------------------------------------------------

TP_N = 500                        # afferents
TP_T = 500.0                      # pattern duration in ms
TP_TAU_M = 10.0                   # membrane time constant in ms
TP_TAU_S = TP_TAU_M / 4           # synaptic time constant in ms
TP_THETA = 1.0                    # firing threshold (V_rest = 0)
TP_V0 = tp.kernel_norm(TP_TAU_M, TP_TAU_S)

TP_LR = 1e-4 * TP_V0              # initial learning rate (Gütig & Sompolinsky 2006)
TP_MU = 0.99                      # momentum
TP_TAU_DECAY = 2000                # learning rate decay in epochs (float("inf") = constant, as in 2006)
TP_W_STD = 1e-3                   # std of the initial weights

# Time constant scan (tau_s = tau_m / 4 fixed; V0 and thus lr do not depend on tau_m)
TP_TAU_SCAN_TAUS = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 50.0, 75.0]
TP_TAU_SCAN_ALPHA = 2.0
TP_TAU_SCAN_N_DATASETS = 10
TP_TAU_SCAN_MAX_EPOCHS = 3000

# Learning rate scan (multiples of the paper value 1e-4 * V0, at TP_TAU_M)
TP_LR_SCAN_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TP_LR_SCAN_ALPHA = 2.0
TP_LR_SCAN_N_DATASETS = 10
TP_LR_SCAN_MAX_EPOCHS = 3000

# Single training run (section 2), also the basis for the window analysis
TP_SINGLE_P = 1000                # patterns (alpha = 2)
TP_SINGLE_MAX_EPOCHS = 5000
TP_SINGLE_SEEDS = dict(data=100, init=110, order=120)
TP_WINDOW = 20.0                  # ms before an output spike, for the input statistics

# Capacity scan
TP_CAP_ALPHAS = [2.4, 2.8, 3.0, 3.2, 3.4, 3.5]
TP_CAP_N_DATASETS = 20
TP_CAP_MAX_EPOCHS = 20000


def tempotron_tau_scan_name():
    """File name of the tempotron time constant scan."""
    return ex.make_name("tau_scan", model="tempotron", N=TP_N, alpha=TP_TAU_SCAN_ALPHA,
                        lr=TP_LR, mu=TP_MU, decay=TP_TAU_DECAY,
                        n=TP_TAU_SCAN_N_DATASETS, max_epochs=TP_TAU_SCAN_MAX_EPOCHS)


def tempotron_tau_scan_run(tau_m, n_jobs=1, print_progress=False):
    """Train the tempotron on all datasets for one membrane time constant (tau_s = tau_m / 4)."""
    return tp.learning_time_over_datasets(
        tp.train_decay_exact, TP_N, ex.even_P(TP_TAU_SCAN_ALPHA, TP_N), TP_T,
        tau_m, tau_m / 4, TP_THETA,
        n_datasets=TP_TAU_SCAN_N_DATASETS, max_epochs=TP_TAU_SCAN_MAX_EPOCHS, w_std=TP_W_STD,
        n_jobs=n_jobs, print_progress=print_progress,
        lr_ini=TP_LR, mu=TP_MU, tau_decay=TP_TAU_DECAY,
    )

def tempotron_lr_scan_name():
    """File name of the tempotron learning rate scan."""
    return ex.make_name("lr_scan", model="tempotron", N=TP_N, alpha=TP_LR_SCAN_ALPHA,
                        tau_m=TP_TAU_M, mu=TP_MU, decay=TP_TAU_DECAY,
                        n=TP_LR_SCAN_N_DATASETS, max_epochs=TP_LR_SCAN_MAX_EPOCHS)


def tempotron_lr_scan_run(lr_factor, n_jobs=1, print_progress=False):
    """Train the tempotron on all datasets for one initial learning rate lr_factor * TP_LR."""
    return tp.learning_time_over_datasets(
        tp.train_decay_exact, TP_N, ex.even_P(TP_LR_SCAN_ALPHA, TP_N), TP_T,
        TP_TAU_M, TP_TAU_S, TP_THETA,
        n_datasets=TP_LR_SCAN_N_DATASETS, max_epochs=TP_LR_SCAN_MAX_EPOCHS, w_std=TP_W_STD,
        n_jobs=n_jobs, print_progress=print_progress,
        lr_ini=lr_factor * TP_LR, mu=TP_MU, tau_decay=TP_TAU_DECAY,
    )


def window_stats_name():
    """File name of the input statistics before output spikes (written by the notebook)."""
    return ex.make_name("window_stats", model="tempotron", N=TP_N, P=TP_SINGLE_P,
                        tau_m=TP_TAU_M, lr=TP_LR, mu=TP_MU, decay=TP_TAU_DECAY, window=TP_WINDOW)


def tempotron_capacity_name():
    """File name of the tempotron capacity scan (shared by script and notebook)."""
    return ex.make_name("capacity", model="tempotron", N=TP_N, tau_m=TP_TAU_M, lr=TP_LR,
                        mu=TP_MU, decay=TP_TAU_DECAY, n=TP_CAP_N_DATASETS,
                        max_epochs=TP_CAP_MAX_EPOCHS)


def tempotron_capacity_run(alpha, n_jobs=1, print_progress=False):
    """Train the tempotron on all datasets for one load alpha (one step of the capacity scan)."""
    return tp.learning_time_over_datasets(
        tp.train_decay_exact, TP_N, ex.even_P(alpha, TP_N), TP_T, TP_TAU_M, TP_TAU_S, TP_THETA,
        n_datasets=TP_CAP_N_DATASETS, max_epochs=TP_CAP_MAX_EPOCHS, w_std=TP_W_STD,
        n_jobs=n_jobs, print_progress=print_progress,
        lr_ini=TP_LR, mu=TP_MU, tau_decay=TP_TAU_DECAY,
    )


# ---------------------------------------------------------------------------
# Multi-perceptron
# ---------------------------------------------------------------------------

MPC_N = 500                       # afferents
MPC_THETA = 1.0                   # threshold

# Window statistics measured in section 2 of the notebook; default values until they exist
_window_stats = ex.load_json(window_stats_name())
if _window_stats is not None:
    MPC_MEAN_ACTIVE = round(_window_stats["mean"], 1)
    MPC_STD_ACTIVE = round(_window_stats["std"], 1)
    MPC_WINDOW_SOURCE = "measured (section 2)"
else:
    MPC_MEAN_ACTIVE, MPC_STD_ACTIVE = 21.9, 4.3
    MPC_WINDOW_SOURCE = "default values (section 2 not yet run)"

MPC_LR = 0.02                     # learning rate: set to tuned value
MPC_MU = 0.99                     # momentum: set to tuned value
MPC_CREDIT = "first"              # sub-pattern held responsible for a spike
MPC_W_STD = 1e-3

# Parameter tuning (learning rate and momentum) at one K near capacity
MPC_TUNE_K = 25
MPC_TUNE_ALPHA = 2.8
MPC_TUNE_LRS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]
MPC_TUNE_MUS = [0.9, 0.99]
MPC_TUNE_N_DATASETS = 10
MPC_TUNE_MAX_EPOCHS = 5000


# Capacity vs. K (bisection per K and dataset)
MPC_K_LIST = [1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 200, 300]
MPC_K_N_DATASETS = 5
MPC_K_MAX_EPOCHS = 5000
MPC_K_RESOLUTION = 10             # P resolution of the bisection (alpha: 10 / N = 0.02)


def mpc_tune_name():
    """File name of the learning rate / momentum tuning of the multi-perceptron."""
    mean, std = MPC_MEAN_ACTIVE, MPC_STD_ACTIVE
    return ex.make_name("tune", model="multi_perceptron", N=MPC_N, K=MPC_TUNE_K,
                        alpha=MPC_TUNE_ALPHA, mean=mean, std=std, credit=MPC_CREDIT,
                        max_epochs=MPC_TUNE_MAX_EPOCHS)


def mpc_tune_task(lr, mu, dataset):
    """Train one dataset with one (lr, mu) pair (one task of the tuning scan)."""
    P = ex.even_P(MPC_TUNE_ALPHA, MPC_N)
    X, y, _ = mpc.make_dataset(MPC_N, P, MPC_TUNE_K, np.random.default_rng([dataset, 0]),
                               mean_active=MPC_MEAN_ACTIVE, std_active=MPC_STD_ACTIVE)
    w0 = mpc.init_weights(MPC_N, np.random.default_rng([dataset, 1]), std=MPC_W_STD)
    _, h, conv = mpc.train(X, y, MPC_THETA, lr, MPC_TUNE_MAX_EPOCHS,
                           np.random.default_rng([dataset, 2]), w0,
                           mu=mu, credit=MPC_CREDIT, verbose=0)
    return dict(epochs=len(h), converged=bool(conv), min_errors=int(min(h)))

def mpc_capacity_vs_K_name():
    """File name of the multi-perceptron K scan (shared by script and notebook)."""
    return ex.make_name("capacity_vs_K", model="multi_perceptron", N=MPC_N,
                        mean=MPC_MEAN_ACTIVE, std=MPC_STD_ACTIVE, lr=MPC_LR, mu=MPC_MU,
                        credit=MPC_CREDIT, res=MPC_K_RESOLUTION, max_epochs=MPC_K_MAX_EPOCHS)


def mpc_capacity_task(K, dataset):
    """Capacity of one dataset for one K, by bisection over P (one task of the K scan)."""
    P_ok, P_fail = mpc.capacity_bisection(
        MPC_N, K, MPC_THETA, MPC_LR, seed=dataset, resolution=MPC_K_RESOLUTION,
        max_epochs=MPC_K_MAX_EPOCHS, w_std=MPC_W_STD,
        mean_active=MPC_MEAN_ACTIVE, std_active=MPC_STD_ACTIVE,
        mu=MPC_MU, credit=MPC_CREDIT,
    )
    return dict(P_ok=P_ok, P_fail=P_fail, alpha_c=(P_ok + P_fail) / 2 / MPC_N)