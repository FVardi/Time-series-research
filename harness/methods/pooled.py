"""Baseline: a feature sequence summarised over time, ignoring the order of the windows (D166, D187).

Given per-window features (series, windows, channels), e.g. the Herning band features (D165), the
mean and the standard deviation of each channel over the windows give one vector per series:
everything about the windows except their order. Tests whether a method gains anything from the
order of the windows. Nothing is fitted. Same interface as the other methods (D77).
"""

import numpy as np


def pretrain(X, s, device):
    """Nothing to fit. Returns (model, loss log) like the other methods."""
    return None, []


def save(model, path):
    """Nothing to save."""


def encode(model, X):
    """(series, windows, channels) -> (series, 2 * channels): the mean of each channel over the windows,
    then its standard deviation (population SD, numpy's default ddof=0) (D187)."""
    return np.hstack([X.mean(axis=1), X.std(axis=1)])
