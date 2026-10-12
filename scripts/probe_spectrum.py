"""Baseline (D46): the harness probes trained directly on |FFT| spectra, no encoder.

Question: how easy is the target task? If probes on the plain spectra already score
as high as the encoders, the dataset cannot separate the methods.
The input is the full FFT magnitude spectrum of each window, as TF-C's frequency
branch receives it (D41 item 2); several channels are concatenated.

Usage:
    uv run python scripts/probe_spectrum.py configs/probe_spectrum_fd.yaml

Writes results.csv (one row per label fraction and probe) into runs/<date_time>_<name>/.
The probes are deterministic, so there are no seeds.
"""

import os
import sys

import numpy as np
import pandas as pd

from harness import evaluate, runs, transfer
from harness.data import canonical


def spectra(X):
    """(series, time, channels) -> (series, channels * time): |FFT| per channel, concatenated."""
    return np.abs(np.fft.fft(X, axis=1)).transpose(0, 2, 1).reshape(len(X), -1)


def main(config_path):
    cfg = runs.load_config(config_path)
    ev = cfg["evaluation"]
    folder = runs.new_run_dir(config_path, cfg)

    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    X_tr, y_tr, _ = transfer.load_set(cfg, meta, cfg["target_run"], "train")
    X_te, y_te, _ = transfer.load_set(cfg, meta, cfg["target_run"], "test")
    Z_tr, Z_te = spectra(X_tr), spectra(X_te)

    rows = []
    for fraction in ev["label_fractions"]:
        idx = evaluate.label_subset(y_tr, fraction, seed=0)  # all labels at 100 %; seed unused then
        for probe in ev["probes"]:
            clf = evaluate.PROBES[probe](Z_tr[idx], y_tr[idx])
            rows.append({"input": "fft_magnitude", "label_fraction": fraction, "n_labels": len(idx),
                         "probe": probe, **evaluate.scores(clf, Z_te, y_te)})
            print(rows[-1], flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(folder, "results.csv"), index=False)
    print(results)


if __name__ == "__main__":
    main(sys.argv[1])
