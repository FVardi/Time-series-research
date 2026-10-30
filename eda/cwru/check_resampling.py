"""Diagnostic: does CWRU data recorded at 48 kHz and resampled to 12 kHz differ consistently from data
recorded at 12 kHz? (Decides whether a healthy class cut from the 48 kHz normal files can be recognised
by how it was recorded and resampled instead of by the bearing's condition; D163, D167.)

Usage:
    uv run python eda/cwru/check_resampling.py configs/resampling_cwru.yaml

CWRU recorded every drive-end fault condition (location, size, clock hour) at every load twice: at
12 kHz and at 48 kHz. The two are separate recordings (different lengths and levels), so differences
can come from the recording sessions as well as from the resampling; either kind, if consistent across
bearings, would also mark the resampled healthy class. Both versions go through the pipeline's own
resampling and windowing (harness/windows.cut, D136, D137).

Two measures, per channel:
  1. Spectrum: mean power per band (band_hz wide) of each recording, in dB; resampled minus native for
     each pair (same condition and load). 'shape' compares the spectra after removing each recording's
     mean level over the bands below edge_hz (so the large band-edge difference does not shift the others).
     A consistent fingerprint shows as a band where nearly all pairs differ in one direction.
  2. Classifier: native (0) or resampled (1), from each window's log band powers. Outer folds hold out
     whole conditions (bearings), so the classifier must generalise to unseen bearings; inside, C is
     chosen by the harness probe with folds grouped by condition (harness/evaluate.logreg_grouped).
     Score: balanced accuracy (the classes differ in size), 0.5 = no difference found.
     Feature sets: 'level+shape' (log band powers), 'shape' (each window's mean log power removed),
     'shape_below_edge' (the same, only bands below edge_hz).
Writes runs/<date_time>_<name>/spectrum_bands.csv and classifier.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from harness import evaluate, runs, windows
from harness.data import canonical

CONDITION = ["fault_location", "fault_size_mil", "clock_hour"]  # one drive-end bearing each


def band_power_db(X, rate_hz, band_hz):
    """Mean power per band of each window (Hann window, one-sided |FFT|^2), in dB.
    X: (windows, samples). Returns (windows, bands) and the bands' lower edges in Hz."""
    spec = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], 1 / rate_hz)
    edges = np.arange(0, rate_hz / 2, band_hz)
    band = np.digitize(freqs, edges) - 1  # band index of each frequency bin
    power = np.stack([spec[:, band == b].mean(axis=1) for b in range(len(edges))], axis=1)
    return 10 * np.log10(power), edges


def pairs(meta, cfg):
    """Drive-end fault captures recorded at both rates: one row per (condition, load) with both captures."""
    caps = windows.select(meta, cfg["captures"])
    key = CONDITION + ["load_hp"]
    caps = caps.assign(clock_hour=caps.clock_hour.fillna(-1))  # no clock hour (inner, ball): -1, so it joins
    native = caps[caps.sample_rate_hz == cfg["rate_hz"]]
    other = caps[caps.sample_rate_hz == cfg["other_rate_hz"]]
    both = native.merge(other, on=key, suffixes=("_native", "_resampled"))
    if len(both) != len(other):  # every 48 kHz capture should have a 12 kHz partner
        raise ValueError(f"{len(other) - len(both)} {cfg['other_rate_hz']} Hz captures without a partner")
    return both, key


def main(config_path):
    cfg = runs.load_config(config_path)
    folder = runs.new_run_dir(config_path, cfg)
    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    both, key = pairs(meta, cfg)
    print(f"{len(both)} pairs, {both.groupby(CONDITION).ngroups} conditions")

    spectra, feats = [], []
    for p in both.itertuples():
        for version in ("native", "resampled"):
            cap = pd.DataFrame([{"run_id": getattr(p, f"run_id_{version}"), "capture_id": getattr(p, f"capture_id_{version}"),
                                 "sample_rate_hz": getattr(p, f"sample_rate_hz_{version}")}])
            X, _ = windows.cut(cfg["data_root"], cfg["dataset"], cap, cfg["channels"], cfg["window_length"], cfg["rate_hz"])
            for k, ch in enumerate(cfg["channels"]):
                db, edges = band_power_db(X[:, :, k], cfg["rate_hz"], cfg["band_hz"])
                rec = 10 * np.log10(np.mean(10 ** (db / 10), axis=0))  # the recording's mean power per band
                spectra.append(pd.DataFrame({**{c: getattr(p, c) for c in key}, "channel": ch, "version": version,
                                             "band_hz": edges, "db": rec,
                                             "shape_db": rec - rec[edges < cfg["edge_hz"]].mean()}))  # level from bands below the edge
                feats.append({"condition": f"{p.fault_location}_{p.fault_size_mil:g}_{p.clock_hour:g}", "channel": ch,
                              "y": int(version == "resampled"), "Z": db})
    spectra = pd.concat(spectra, ignore_index=True)

    # 1. Spectrum: resampled minus native per pair and band
    wide = spectra.pivot_table(index=key + ["channel", "band_hz"], columns="version", values=["db", "shape_db"])
    diff = pd.DataFrame({"diff_db": wide["db"]["resampled"] - wide["db"]["native"],
                         "shape_diff_db": wide["shape_db"]["resampled"] - wide["shape_db"]["native"]}).reset_index()
    diff.to_csv(os.path.join(folder, "spectrum_bands.csv"), index=False)
    summary = diff.groupby(["channel", "band_hz"]).agg(
        median_shape_diff_db=("shape_diff_db", "median"),
        share_resampled_higher=("shape_diff_db", lambda d: (d > 0).mean()))
    print("\nSpectrum shape, resampled minus native (median over pairs; share of pairs where resampled is higher):")
    print(summary.round(2).to_string())

    # 2. Classifier: native or resampled, outer folds hold out whole conditions
    rows = []
    for ch in cfg["channels"]:
        f = [r for r in feats if r["channel"] == ch]
        Z = np.concatenate([r["Z"] for r in f])
        y = np.concatenate([np.full(len(r["Z"]), r["y"]) for r in f])
        groups = np.concatenate([np.full(len(r["Z"]), r["condition"]) for r in f])
        below = edges < cfg["edge_hz"]
        sets = {"level+shape": Z,
                "shape": Z - Z.mean(axis=1, keepdims=True),
                "shape_below_edge": Z[:, below] - Z[:, below].mean(axis=1, keepdims=True)}
        for name, F in sets.items():
            for fold, (tr, te) in enumerate(GroupKFold(n_splits=cfg["outer_folds"]).split(F, y, groups)):
                clf, _, _ = evaluate.logreg_grouped(F[tr], y[tr], groups[tr], cfg["seed"])
                rows.append({"channel": ch, "features": name, "fold": fold,
                             "test_conditions": ";".join(sorted(set(groups[te]))),
                             "balanced_accuracy": balanced_accuracy_score(y[te], clf.predict(F[te]))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(folder, "classifier.csv"), index=False)
    print("\nClassifier, native vs resampled on unseen conditions (balanced accuracy; 0.5 = no difference):")
    print(res.groupby(["channel", "features"]).balanced_accuracy.agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
