"""Baseline: the 14 MSSP statistics as features (D49, D76 item 3, D82); rig only (D85).

The statistics are computed by the MSSP code itself (harness/methods/mssp_features.py,
copied unchanged). For now only the broadband set: each channel as loaded, no band filter.
Since D85 this baseline is used on the Herning rig only (the domain baseline); the
octave bands for the rig (D53-D61) are added with the Herning step.

Nothing is fitted. Same interface as the other methods (D77).
"""

import numpy as np

from harness.methods.mssp_features import FEATURE_NAMES, extract_features


def pretrain(X, s, device):
    """Nothing to fit. The 'model' only carries the sampling rate needed by encode().

    s: the 'mssp' section of the config, with fs (sampling rate in Hz; FordA: 1, i.e.
       frequencies in cycles per sample, D82).
    """
    return {"fs": s["fs"]}, []


def save(model, path):
    """Nothing to save (the sampling rate is in the run's config)."""


def encode(model, X):
    """(series, time, channels) -> (series, channels * 14): the statistics per channel, channels one after the other.

    Column order: channel 0's statistics in FEATURE_NAMES order, then channel 1's, and so on.
    """
    rows = []
    for series in X:
        row = []
        for c in range(X.shape[2]):
            stats = extract_features(series[:, c], model["fs"])
            row += [stats[name] for name in FEATURE_NAMES]
        rows.append(row)
    return np.array(rows)
