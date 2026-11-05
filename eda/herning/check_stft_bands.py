"""D186 check: band energies from a rectangular-window STFT of the raw signal vs the Butterworth band energies of D165.

Usage:
    uv run python eda/herning/check_stft_bands.py [n_sweeps] [output.csv]
    (defaults: 10 sweeps, no file written)

Takes n_sweeps Herning sweeps evenly spaced in capture order (so they spread over the rig's
schedule of speeds and temperatures). For each sweep, channel (AE, UL) and D165 band, the band
energy (mean square of the band signal) of every 1 ms window is computed in two ways:
    butterworth  as D165: order-5 zero-phase Butterworth on the whole sweep, then windows
    stft         rectangular, non-overlapping 1 ms windows, one FFT each, power of the bins
                 with f_lo <= f < f_hi summed (Parseval, one-sided)
Both use the D165 band edges, so only the way of computing differs (not the D186 sub-band grid).

Printed per channel and band, over all windows of the chosen sweeps:
    median_db, p05_db, p95_db   10*log10(stft / butterworth): level difference in dB
    corr_all                    Pearson correlation of the two log energies, all windows pooled
    corr_within                 median over sweeps of the same correlation within one sweep
                                (the within-sweep dynamics D165 is meant to keep)
No threshold is fixed here: the user judges whether the agreement is close. Result and outcome: D186,
runs/20261102_stft_check/ (the raw-signal FFT leaks on UL, so D186 filters first).
"""

import sys

import numpy as np
import pandas as pd

from harness.band_features import BANDS, WINDOW
from harness.data import canonical
from harness.methods.mssp_features import bandpass_filter

ROOT, DATASET = "data/canonical", "herning"


def butterworth_energy(x, fs, f_lo, f_hi, n):
    """Mean square of the Butterworth band signal in each of the n windows (D165)."""
    xb = bandpass_filter(x, fs, f_lo, f_hi)  # whole sweep first, then windows
    return np.mean(xb[: n * WINDOW].reshape(n, WINDOW) ** 2, axis=1)


def stft_energy(x, fs, f_lo, f_hi, n):
    """Mean square of each window's content in [f_lo, f_hi), from its FFT.

    Parseval: mean(x^2) = (1/W^2) * sum over all two-sided bins of |X_k|^2. A bin strictly
    between DC and Nyquist appears twice in the two-sided sum, hence the factor 2.
    """
    X = np.fft.rfft(x[: n * WINDOW].reshape(n, WINDOW), axis=1)
    f = np.fft.rfftfreq(WINDOW, 1 / fs)
    sel = (f >= f_lo) & (f < f_hi)
    return 2 * np.sum(np.abs(X[:, sel]) ** 2, axis=1) / WINDOW ** 2


def main(n_sweeps=10, out_path=None):
    meta = canonical.load_metadata(ROOT, DATASET)
    caps = meta["captures"]
    fs = float(meta["channels"].sample_rate_hz.unique()[0])
    picks = np.linspace(0, len(caps) - 1, n_sweeps).round().astype(int)
    print(f"fs = {fs:g} Hz, window = {WINDOW} samples; sweeps: {list(caps.capture_id.iloc[picks])}", flush=True)

    rows = []  # one row per sweep, channel, band and window
    for i in picks:
        c = caps.iloc[i]
        x = canonical.load_captures(ROOT, DATASET, c.run_id, [c.capture_id], list(BANDS))[0]  # (samples, channels)
        n = x.shape[0] // WINDOW  # incomplete last window dropped, as D165
        for j, ch in enumerate(BANDS):
            for f_lo, f_hi in BANDS[ch]:
                e_bw = butterworth_energy(x[:, j], fs, f_lo, f_hi, n)
                e_st = stft_energy(x[:, j], fs, f_lo, f_hi, n)
                rows.append(pd.DataFrame({"sweep": c.capture_id, "channel": ch,
                                          "band": f"{f_lo / 1e3:.3g}-{f_hi / 1e3:.3g} kHz",
                                          "window": np.arange(n), "butterworth": e_bw, "stft": e_st}))
        print(f"  {c.capture_id} done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["log_bw"], df["log_st"] = np.log(df.butterworth), np.log(df.stft)
    df["diff_db"] = 10 * np.log10(df.stft / df.butterworth)

    summary = []
    for (ch, band), g in df.groupby(["channel", "band"], sort=False):
        within = g.groupby("sweep")[["log_bw", "log_st"]].apply(lambda s: np.corrcoef(s.log_bw, s.log_st)[0, 1])
        summary.append({"channel": ch, "band": band,
                         "median_db": g.diff_db.median(), "p05_db": g.diff_db.quantile(0.05),
                         "p95_db": g.diff_db.quantile(0.95),
                         "corr_all": np.corrcoef(g.log_bw, g.log_st)[0, 1], "corr_within": within.median()})
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if out_path:
        df.drop(columns=["log_bw", "log_st"]).to_csv(out_path, index=False)
        print(f"per-window values written to {out_path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10, sys.argv[2] if len(sys.argv) > 2 else None)
