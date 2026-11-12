"""Per-window sub-band power of the Herning sweeps: the stored spectrogram (D186).

Each D165 octave band signal (the same order-5 zero-phase Butterworth band-pass on the whole sweep as
harness/features/band_features.py) is cut into the D165 windows (consecutive, non-overlapping, 1 ms; incomplete
last window dropped). Each window is Fourier transformed (rectangular window) and the power of its bins
is summed per sub-band:
    power = 2 * sum |X_k|^2 / W^2   over the bins k with s_lo <= f_k < s_hi
which is the part of the window's mean square in that sub-band (Parseval, one-sided spectrum).
Filtering before cutting keeps the power of other octaves out of each band: cutting the raw signal
spreads UL's strong low-frequency power into the upper UL bands (D186 check).

Sub-bands: 1/12 of an octave within each D165 band, on grids aligned with the band edges, AE base-2
(step 2**(1/12)) and UL base-10 (step 10**0.025), counted from each band's lower edge. The AE top band
(400-560 kHz) is not a full octave: its last sub-band is cut at 560 kHz. Sub-bands without an FFT bin
are dropped; that happens only on UL below about 17 kHz, where 1/12 octave is narrower than the 1 kHz
bin spacing of a 1 ms window (the lowest UL octaves keep 1-3 bins).

Output: linear power per window and sub-band, channels AE sub-bands then UL sub-bands, in frequency
order. Stored as float32 (STORE_DTYPE); the log is taken by whoever uses it. The sub-bands of one octave
add up to about the mean square of its band signal (exp of the D165 log_energy), less the small part of
the band's power that the cut spreads outside the band.
Computed separately from harness/features/band_features.py, so the band-pass filters run once per feature set.
"""

import numpy as np

from harness.features.band_features import BANDS, WINDOW
from harness.baselines.mssp_features import bandpass_filter

STEP = {"AE": 2 ** (1 / 12), "UL": 10 ** 0.025}  # ratio between neighbouring sub-band edges
STORE_DTYPE = "float32"  # D186


def band_edges(f_lo, f_hi, step):
    """Sub-band edges from f_lo by the ratio step, the last one at f_hi."""
    n = int(np.ceil(np.log(f_hi / f_lo) / np.log(step) - 1e-9))  # number of sub-bands; 1e-9: rounding
    return [f_lo * step ** k for k in range(n)] + [f_hi]


def subbands(channel, fs, window=WINDOW):
    """[(band index, s_lo, s_hi)] of the channel's kept sub-bands: those containing at least one FFT bin."""
    f = np.fft.rfftfreq(window, 1 / fs)
    out = []
    for b, (f_lo, f_hi) in enumerate(BANDS[channel]):
        e = band_edges(f_lo, f_hi, STEP[channel])
        out += [(b, lo, hi) for lo, hi in zip(e[:-1], e[1:]) if np.any((f >= lo) & (f < hi))]
    return out


def window_power(frames, fs, edges):
    """frames (windows, W) -> (windows, len(edges)): power of each window in each [lo, hi)."""
    W = frames.shape[1]
    P = 2 * np.abs(np.fft.rfft(frames, axis=1)) ** 2 / W ** 2  # one-sided; DC and Nyquist not in any band
    f = np.fft.rfftfreq(W, 1 / fs)
    return np.stack([P[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in edges], axis=1)


def channel_power(x, fs, channel, window=WINDOW):
    """One channel of one sweep (samples,) -> (windows, number of the channel's sub-bands)."""
    n = len(x) // window
    sb = subbands(channel, fs, window)
    cols = []
    for b, (f_lo, f_hi) in enumerate(BANDS[channel]):
        xb = bandpass_filter(x, fs, f_lo, f_hi)  # whole sweep first, then windows (as D165)
        cols.append(window_power(xb[: n * window].reshape(n, window), fs, [(lo, hi) for i, lo, hi in sb if i == b]))
    return np.concatenate(cols, axis=1)


def compute(X, channels, fs, window=WINDOW):
    """Sweeps (series, samples, channels) -> sub-band power (series, windows, sub-bands of all channels)."""
    return np.stack([np.concatenate([channel_power(s[:, c], fs, ch, window) for c, ch in enumerate(channels)],
                                    axis=1) for s in X])


def settings(channels, fs, window=WINDOW):
    """What a stored file must match (D172): everything that determines the values."""
    return {"channels": list(channels), "fs": float(fs), "window": int(window),
            "bands_hz": {ch: [list(b) for b in BANDS[ch]] for ch in channels},
            "subbands_hz": {ch: [[lo, hi] for _, lo, hi in subbands(ch, fs, window)] for ch in channels},
            "filter": "order-5 zero-phase Butterworth per band on the whole capture (mssp_features.bandpass_filter)",
            "transform": "rectangular-window FFT of each window of the band signal; power 2|X_k|^2/W^2 summed "
                         "over the bins in [lo, hi)",
            "dtype": STORE_DTYPE}


def names(channels, fs=2.5e6, window=WINDOW):
    """Names of the output channels, in order, e.g. 'AE 50-52.97 kHz power'.

    fs defaults to the Herning rate because callers pass only the channels; the kept sub-bands depend on it.
    """
    return [f"{ch} {lo / 1e3:.4g}-{hi / 1e3:.4g} kHz power" for ch in channels for _, lo, hi in subbands(ch, fs, window)]
