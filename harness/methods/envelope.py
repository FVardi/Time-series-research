"""Baseline: the squared envelope spectrum as features (D161).

Identical to the |FFT| baseline (fft.py, D76 item 2) except for one step: each channel is
first demodulated over the whole band (no band-pass filter) into its squared envelope
|x + i H(x)|^2, H the Hilbert transform (scipy.signal.hilbert). Then, as fft.py: one-sided
spectrum (numpy.fft.rfft: bins 0 ... N/2), linear magnitude, no taper, frequencies in Hz
(no order normalisation), DC bin kept, channels one after the other. Nothing is fitted.
"""

import numpy as np
from scipy.signal import hilbert


def pretrain(X, s, device):
    """Nothing to fit. Returns (model, loss log) like the other methods."""
    return None, []


def save(model, path):
    """Nothing to save."""


def encode(model, X):
    """(series, time, channels) -> (series, channels * (time // 2 + 1)): |rfft| of the squared envelope per channel."""
    squared_envelope = np.abs(hilbert(X, axis=1)) ** 2  # (series, time, channels)
    spectrum = np.abs(np.fft.rfft(squared_envelope, axis=1))  # (series, bins, channels)
    return spectrum.transpose(0, 2, 1).reshape(len(X), -1)
