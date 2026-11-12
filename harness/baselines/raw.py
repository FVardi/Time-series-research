"""Baseline: the raw samples as features (D75, D76 item 1).

Nothing is fitted. Values are used as loaded (as recorded, D5), with no
per-series normalisation. Same interface as the other methods (D77).
"""


def pretrain(X, s, device):
    """Nothing to fit. Returns (model, loss log) like the other methods."""
    return None, []


def save(model, path):
    """Nothing to save."""


def encode(model, X):
    """(series, time, channels) -> (series, channels * time): the channels one after the other."""
    return X.transpose(0, 2, 1).reshape(len(X), -1)
