"""Baseline: a feature sequence averaged over time (D166).

Given per-window features (series, windows, channels), e.g. the Herning band features (D165),
the mean over the windows gives one vector per series. Tests whether a method gains anything
from the order of the windows. Nothing is fitted. Same interface as the other methods (D77).
"""


def pretrain(X, s, device):
    """Nothing to fit. Returns (model, loss log) like the other methods."""
    return None, []


def save(model, path):
    """Nothing to save."""


def encode(model, X):
    """(series, windows, channels) -> (series, channels): the mean over the windows."""
    return X.mean(axis=1)
