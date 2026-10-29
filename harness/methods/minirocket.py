"""MiniRocket, wrapped (D75, D76 item 4, D81; D10 mode A, D19).

The official code lives unmodified in external/minirocket (git submodule, commit 0b1c245);
this file uses its univariate version, code/minirocket.py, for one channel (FordA, D81), and its
multivariate version, code/minirocket_multivariate.py, for several channels (D150):
    pretrain = MiniRocket's fit: sets dilations and biases from the training inputs (no labels)
    encode   = MiniRocket's transform: about 10,000 features per series
Its own probe (ridge classifier) is 'ridge_minirocket' in harness/evaluate.py.

The official functions are compiled by numba when they are imported (they have explicit
type signatures). That time is measured here as COMPILE_S and recorded separately by
scripts/pretrain.py, so it is not counted as fit time (D79). Both versions are compiled
at import, so COMPILE_S covers both.
"""

import os
import time

import numpy as np
from numba import njit

from harness.methods.external import import_official

_start = time.perf_counter()
official = import_official(os.path.join("minirocket", "code"), "minirocket", local_names=("minirocket",))
official_mv = import_official(os.path.join("minirocket", "code"), "minirocket_multivariate",
                              local_names=("minirocket_multivariate",))


@njit("void(int64)")
def _seed_numba(seed):
    """Seed numba's own random generator (D81).

    MiniRocket draws the training examples for its biases inside numba-compiled code,
    which uses numba's generator; np.random.seed in runs.set_seed does not reach it.
    """
    np.random.seed(seed)


COMPILE_S = time.perf_counter() - _start  # compiling (or loading numba's cache) of the functions above


def _univariate(X):
    """(series, time, 1) -> (series, time) as float32, the input of the official univariate code."""
    if X.shape[2] != 1:
        raise ValueError("code/minirocket.py is univariate")
    return np.ascontiguousarray(X[:, :, 0], dtype=np.float32)  # conversion to 32-bit at training (D23)


def _multivariate(X):
    """(series, time, channels) -> (series, channels, time) as float32, the input of the official multivariate code."""
    return np.ascontiguousarray(np.transpose(X, (0, 2, 1)), dtype=np.float32)


def pretrain(X, s, device):
    """MiniRocket's fit on the training inputs (no labels). MiniRocket runs on the CPU; device is not used.

    s: the 'minirocket' section of the config (official settings, D76).
    Returns the fitted parameters and an empty loss log: one channel (dilations, features per
    dilation, biases); several channels (channels per combination, channel indices, dilations,
    features per dilation, biases). The multivariate fit draws its channel combinations from
    NumPy's generator (seeded by runs.set_seed) and the bias examples from numba's (seeded here).
    """
    # numba's generator gets a seed drawn from NumPy's, which runs.set_seed has seeded,
    # so the run's seed fixes MiniRocket's random choices.
    _seed_numba(np.random.randint(2**31 - 1))
    if X.shape[2] == 1:
        params = official.fit(_univariate(X), num_features=s["num_features"],
                              max_dilations_per_kernel=s["max_dilations_per_kernel"])
    else:  # several channels (D150)
        params = official_mv.fit(_multivariate(X), num_features=s["num_features"],
                                 max_dilations_per_kernel=s["max_dilations_per_kernel"])
    return params, []


def save(params, path):
    """Save the fitted parameters (3 arrays for one channel, 5 for several)."""
    if len(params) == 3:
        dilations, num_features_per_dilation, biases = params
        np.savez(path + ".npz", dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)
    else:
        channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = params
        np.savez(path + ".npz", num_channels_per_combination=channels_per_combination, channel_indices=channel_indices,
                 dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)


def encode(params, X):
    """MiniRocket's transform: one feature vector per series (the version the parameters were fitted with)."""
    if len(params) == 3:
        return official.transform(_univariate(X), params)
    return official_mv.transform(_multivariate(X), params)
