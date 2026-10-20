"""Baseline: the |FFT| magnitude spectrum as features (D75, D76 item 2).

One-sided spectrum (numpy.fft.rfft: bins 0 ... N/2), linear magnitude, per channel.
Nothing is fitted. Same interface as the other methods (D77).
(D46's two-sided spectrum for FD-B is in scripts/probe_spectrum.py and stays as recorded.)
"""

import numpy as np


def pretrain(X, s, device):
    """Nothing to fit. Returns (model, loss log) like the other methods."""
    return None, []


def save(model, path):
    """Nothing to save."""


def encode(model, X):
    """(series, time, channels) -> (series, channels * (time // 2 + 1)): |rfft| per channel, channels one after the other."""
    spectrum = np.abs(np.fft.rfft(X, axis=1))  # (series, bins, channels)
    return spectrum.transpose(0, 2, 1).reshape(len(X), -1)
