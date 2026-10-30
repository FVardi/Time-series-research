"""Per-window band features of the Herning sweeps: the SSL input (D165).

For each band, the whole sweep is band-pass filtered (order-5 zero-phase Butterworth, the MSSP
code's `bandpass_filter`), cut into consecutive non-overlapping windows (the incomplete last
window is dropped) and four features are computed per window:
    log_energy  natural log of the mean square of the band signal
    mobility    Hjorth mobility (antropy, samples^-1), as the MSSP code
    complexity  Hjorth complexity (antropy), as the MSSP code
    kurtosis    Pearson kurtosis (3 for Gaussian data), as the MSSP code
Channels of the output: band by band (AE bands, then UL bands), the four features per band.
Standardisation with training statistics is done by the caller (harness.windows, D137).
"""

import antropy as ant
import numpy as np

from harness.methods.mssp_features import bandpass_filter

FEATURES = ["log_energy", "mobility", "complexity", "kurtosis"]

# Octave bands in Hz: AE from 50 kHz, top band ending at 560 kHz (D61); UL base-10 octaves with
# centres 1.0 ... 126 kHz and edges at centre * 10**(+-0.15) (D60).
BANDS = {
    "AE": [(50e3, 100e3), (100e3, 200e3), (200e3, 400e3), (400e3, 560e3)],
    "UL": [(1e3 * 10 ** (0.3 * k - 0.15), 1e3 * 10 ** (0.3 * k + 0.15)) for k in range(8)],
}
WINDOW = 2500  # samples: 1 ms at 2.5 MHz (D165)


def window_features(W):
    """The four features of each row of W (windows, samples), as columns (windows, 4).

    Kurtosis and the Hjorth parameters follow harness/methods/mssp_features.py exactly,
    including its guard: both are 0 for a window with standard deviation below 1e-30.
    """
    energy = np.mean(W ** 2, axis=1)
    dev = W - W.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(dev ** 2, axis=1))
    ok = std >= 1e-30
    safe = np.where(ok, std, 1.0)[:, None]  # avoids dividing by 0 where the guard applies
    kurt = np.where(ok, np.mean((dev / safe) ** 4, axis=1), 0.0)
    mob, com = ant.hjorth_params(W, axis=1)
    return np.stack([np.log(energy), np.where(ok, mob, 0.0), np.where(ok, com, 0.0), kurt], axis=1)


def channel_features(x, fs, bands, window=WINDOW):
    """One channel of one sweep (samples,) -> (windows, len(bands) * 4)."""
    n = len(x) // window
    cols = []
    for f_lo, f_hi in bands:
        xb = bandpass_filter(x, fs, f_lo, f_hi)  # whole sweep first, then windows
        cols.append(window_features(xb[: n * window].reshape(n, window)))
    return np.concatenate(cols, axis=1)


def sweep_features(X, channels, fs, window=WINDOW):
    """Sweeps (series, samples, channels) -> feature sequences (series, windows, n_bands * 4).

    channels: the names of X's channels in order (e.g. ["AE", "UL"]); each uses its BANDS.
    """
    return np.stack([np.concatenate([channel_features(s[:, c], fs, BANDS[ch], window) for c, ch in enumerate(channels)],
                                    axis=1) for s in X])


def feature_names(channels):
    """Names of the output channels, in order, e.g. 'AE 50-100 kHz log_energy'."""
    return [f"{ch} {lo / 1e3:.3g}-{hi / 1e3:.3g} kHz {feat}" for ch in channels for lo, hi in BANDS[ch] for feat in FEATURES]
