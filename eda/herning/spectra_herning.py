"""Diagnostic: average Welch power spectra per operating-point group (Herning source file).

Usage:
    uv run python eda/herning/spectra_herning.py configs/spectra_herning.yaml

Writes runs/<date_time>_<name>/spectra.csv: one row per group, channel and frequency
bin (log-spaced bins up to Nyquist; mean power spectral density within the bin, V^2/Hz),
and sweeps.csv with the sweeps used per group.
"""

import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

from harness import runs


def log_bins(freqs, psd, n):
    """Average a spectrum into n log-spaced frequency bins (bins without a frequency are dropped)."""
    edges = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n + 1)
    idx = np.digitize(freqs, edges) - 1
    out = [(np.sqrt(edges[i] * edges[i + 1]), psd[idx == i].mean()) for i in range(n) if np.any(idx == i)]
    return np.array(out)


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    rows, used = [], []
    with h5py.File(cfg["source"], "r") as f:
        sweeps = f["sweeps"]
        attrs = {n: (float(sweeps[n].attrs["telem_rpm_target"]), float(sweeps[n].attrs["telem_temp_target"]))
                 for n in sweeps}
        for group, (rpm_t, temp_t) in cfg["groups"].items():
            names = [n for n, a in attrs.items() if a == (rpm_t, temp_t)]
            acc = {ch: [] for ch in cfg["channels"]}
            for n in names:
                fs = float(sweeps[n][cfg["channels"][0]].attrs["sample_rate"])
                for ch in cfg["channels"]:
                    freqs, psd = welch(sweeps[n][ch]["voltage"][:], fs=fs, window=cfg["welch_window"],
                                       nperseg=cfg["welch_nperseg"], noverlap=cfg["welch_noverlap"])
                    acc[ch].append(psd)
                used.append({"group": group, "sweep": n})
            for ch in cfg["channels"]:
                for fc, p in log_bins(freqs, np.mean(acc[ch], axis=0), cfg["n_freq_points"]):
                    rows.append({"group": group, "channel": ch, "freq_hz": fc, "psd": p, "n_sweeps": len(names)})
            print(group, len(names), "sweeps", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(folder, "spectra.csv"), index=False)
    pd.DataFrame(used).to_csv(os.path.join(folder, "sweeps.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
