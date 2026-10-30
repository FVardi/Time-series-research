"""The MSSP statistics per band over whole Herning sweeps: the domain baseline (D49-D61, D181).

Per channel: the broadband set, then each octave band. For each, the whole sweep is filtered
(order-5 zero-phase Butterworth, the MSSP code's `bandpass_filter`; D50) and the 14 statistics of
the MSSP code (`extract_features`, FEATURE_NAMES order; D49, D54) are computed over the whole sweep.
Bands: the same octaves as the windowed band features (D60/D61, harness/band_features.py).
Output per sweep: one window of 14 x (1 + bands) statistics per channel, channels one after the
other: (series, 1, 196) for AE and UL. Stored once by scripts/derive_features.py (D181).
"""

import numpy as np

from harness.band_features import BANDS
from harness.methods.mssp_features import FEATURE_NAMES, bandpass_filter, extract_features

# Broadband set per sensor (D50): AE 50-560 kHz, the range its bands cover (D61); UL from 0 Hz (D54)
# up to the top of its bands, 178 kHz (D181). f_lo = 0 makes bandpass_filter a low-pass.
BROADBAND = {"AE": (50e3, 560e3), "UL": (0.0, BANDS["UL"][-1][1])}


def bands_of(ch):
    """The broadband set, then the octave bands, of one channel, as (f_lo, f_hi) in Hz."""
    return [BROADBAND[ch]] + BANDS[ch]


def compute(X, channels, fs):
    """Sweeps (series, samples, channels) -> statistics (series, 1, sum over channels of 14 x (1 + bands))."""
    rows = []
    for s in X:
        row = []
        for c, ch in enumerate(channels):
            for f_lo, f_hi in bands_of(ch):
                stats = extract_features(bandpass_filter(s[:, c], fs, f_lo, f_hi), fs)
                row += [stats[name] for name in FEATURE_NAMES]
        rows.append(row)
    return np.array(rows)[:, None, :]


def settings(channels, fs):
    """What a stored file must match (D181): everything that determines the statistics."""
    return {"channels": list(channels), "fs": float(fs), "features": FEATURE_NAMES, "window": "whole sweep",
            "bands_hz": {ch: [list(b) for b in bands_of(ch)] for ch in channels},
            "filter": "order-5 zero-phase Butterworth per band on the whole capture (mssp_features.bandpass_filter)"}


def names(channels):
    """Names of the output channels, e.g. 'AE broadband 50-560 kHz rms', 'UL 0.708-1.41 kHz kurtosis'."""
    out = []
    for ch in channels:
        for i, (lo, hi) in enumerate(bands_of(ch)):
            label = f"{ch} {'broadband ' if i == 0 else ''}{lo / 1e3:.3g}-{hi / 1e3:.3g} kHz"
            out += [f"{label} {name}" for name in FEATURE_NAMES]
    return out
