"""Spectral probe (D244, checklist task A): can a linear probe read the spectrum of the input from the
stored representations?

For every run listed in a config (one config per dataset, in studies/spectral_probe/configs/):
  1. The windows the encoder received are rebuilt from the canonical data with the run's own config,
     as scripts/pretrain.py builds them (windows.prepare, resampling, the run's saved standardisation),
     and checked row by row against the run's windows.csv. Nothing is retrained or re-encoded.
  2. Targets per window and channel (D244), from the power |rfft|^2 of each bin, DC bin left out:
       absolute : log E_b, E_b = power summed over the bins of band b
       relative : log (E_b / E), E = sum of E_b over the bands
       total    : log E
       centroid : sum of f * power / sum of power, f in cycles per sample (not logged)
     Bands: n_bands equal-width bands from 0 to Nyquist; a bin at frequency f is in band b if
     edge_b < f <= edge_(b+1). The config's 'targets' says which of the four kinds are used.
  3. Probes on the stored Z of seed_<seed>: ridge, one per target (standardised features, alpha per
     target by cross-validation over the config's alphas, as harness/evaluate.ridge), and floor_mean
     (the training mean). At each label fraction the labelled windows are an unstratified random
     subset of the training windows (harness/evaluate.label_subset, as for regression, D170).
     Score: R² on all test windows, per target.
  4. Reference (config fft_reference: true): the |rfft| spectrum of the same windows
     (harness/baselines/fft.py, the FFT baseline) probed in the same way.
  5. Per test bearing (config per_bearing: true; run-to-failure data, XJTU-SY, D265): Pearson's correlation
     between the ridge probe's predictions and the targets within each test bearing, so a constant offset
     between bearings does not count; written to results/<config name>_per_bearing.csv.

Usage (from the repo root):
    uv run --no-sync python studies/spectral_probe/probe_spectral.py studies/spectral_probe/configs/forda.yaml
Writes studies/spectral_probe/results/<config name>.csv, one row per run (or reference), label fraction,
probe, channel and target (and <config name>_per_bearing.csv with per_bearing: true).
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import evaluate, runs, windows
from harness.data import canonical
from harness.baselines import fft

OUT = os.path.join(os.path.dirname(__file__), "results")
CHUNK = 256  # windows per FFT call (keeps memory low on long windows); does not change any value
KINDS = ("absolute", "relative", "total", "centroid")


# ---------------------------------------------------------------- the encoder's inputs

def run_config(folder):
    """The run's copy of its config; for a protocol run (Paderborn, D216) with the filled-in split from split.json."""
    cfg = runs.load_config(os.path.join(folder, "config.yaml"))
    split_path = os.path.join(folder, "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            cfg["split"] = json.load(f)["split"]
    return cfg


def window_set_key(folder):
    """Identifies a set of encoder inputs: runs with the same key received exactly the same windows,
    so their targets (and the FFT reference) are computed once."""
    h = hashlib.sha1()
    for name in ("windows.csv", "standardisation.json", "split.json"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(name.encode() + f.read())
    h.update(json.dumps(run_config(folder).get("windows")).encode())  # e.g. standardise: per_window
    return h.hexdigest()


def encoder_inputs(folder):
    """Training and test windows (n, length, channels) as the run's encoder received them, the bearing
    (group) of each training window (None without groups), and the run's config.

    Stops if the rebuilt windows are not the run's windows.csv rows, in the same order (the rows of Z, D33)."""
    cfg = run_config(folder)
    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    if "split" in cfg:  # windowed configs (D147): as windowed_inputs() in scripts/pretrain.py
        if cfg["windows"].get("features"):  # e.g. Herning's band features: not signal windows
            raise ValueError(f"{folder}: the encoder received features, not signal windows")
        parts = windows.prepare(cfg, meta)
        X = {p: parts[p][0] for p in ("train", "test")}
        built = {p: parts[p][1] for p in ("train", "test")}
        mode = cfg["windows"]["standardise"]
        if mode == "per_window":  # each window by its own mean and SD (D232)
            X = {p: windows.standardise_per_window(x) for p, x in X.items()}
        elif mode:  # the statistics the run computed from its pretraining windows and saved (D137)
            with open(os.path.join(folder, "standardisation.json")) as f:
                saved = json.load(f)
            if saved["channels"] != cfg["channels"]:
                raise ValueError(f"{folder}: standardisation.json is for channels {saved['channels']}")
            stats = {"mean": np.array(saved["mean"]), "std": np.array(saved["std"])}
            X = {p: windows.standardise(x, stats) for p, x in X.items()}
    else:  # whole captures with an official split (FordA): as load_split() in scripts/pretrain.py
        X, built = {}, {}
        for p in ("train", "test"):
            caps = meta["captures"][meta["captures"].official_split == p]
            X[p] = canonical.load_captures(cfg["data_root"], cfg["dataset"], cfg["run_id"], caps.capture_id,
                                           cfg["channels"])
            built[p] = pd.DataFrame({"run_id": cfg["run_id"], "capture_id": caps.capture_id.to_numpy(),
                                     "start": 0, "stop": X[p].shape[1]})
    # IDs read as text: pandas would otherwise turn run IDs such as CWRU's '0097' into the number 97
    table = pd.read_csv(os.path.join(folder, "windows.csv"), dtype={"run_id": str, "capture_id": str, "group": str})
    cols = ["run_id", "capture_id", "start", "stop"]
    for p in ("train", "test"):
        stored = table[table.split == p].sort_values("row")[cols].reset_index(drop=True).astype(str)
        if not stored.equals(built[p][cols].reset_index(drop=True).astype(str)):
            raise ValueError(f"{folder}: the rebuilt {p} windows are not the rows of windows.csv")
    train = table[table.split == "train"].sort_values("row")
    groups = train.group.to_numpy() if "group" in train else None
    return X, groups, cfg


def test_bearings(folder):
    """Bearing (group) of each test window, in the row order of Z_test (D33)."""
    table = pd.read_csv(os.path.join(folder, "windows.csv"), dtype={"run_id": str, "capture_id": str, "group": str})
    return table[table.split == "test"].sort_values("row").group.to_numpy()


# ---------------------------------------------------------------- targets

def spectral_targets(X, n_bands, kinds, channels, rate_hz):
    """Targets of every window (D244): Y (n_windows, n_targets) and one description per column.

    rate_hz: sample rate of the windows, only to state the band edges in Hz as well (None if unknown)."""
    f = np.fft.rfftfreq(X.shape[1])[1:]  # bin frequencies in cycles per sample, DC bin left out
    edges = np.linspace(0.0, 0.5, n_bands + 1)  # equal-width bands from 0 to Nyquist
    band = np.searchsorted(edges, f, side="left") - 1  # band b holds edges[b] < f <= edges[b + 1]
    E = np.empty((len(X), n_bands, X.shape[2]))  # band energy per window, band, channel
    centroid = np.empty((len(X), X.shape[2]))
    for i in range(0, len(X), CHUNK):
        P = np.abs(np.fft.rfft(X[i:i + CHUNK], axis=1))[:, 1:] ** 2  # power per bin (chunk, bins, channels)
        for b in range(n_bands):
            E[i:i + CHUNK, b] = P[:, band == b].sum(axis=1)
        centroid[i:i + CHUNK] = (P * f[None, :, None]).sum(axis=1) / P.sum(axis=1)
    if (E <= 0).any():  # the log would be undefined; no small constant is added silently
        raise ValueError("a band has zero energy in some window")
    total = E.sum(axis=1)  # (n_windows, channels)

    columns, info = [], []
    hz = lambda x: x * rate_hz if rate_hz else np.nan
    for c, channel in enumerate(channels):
        for kind in kinds:
            if kind in ("absolute", "relative"):
                for b in range(n_bands):
                    values = np.log(E[:, b, c]) if kind == "absolute" else np.log(E[:, b, c] / total[:, c])
                    columns.append(values)
                    info.append({"channel": channel, "target": kind, "band": b, "f_lo": edges[b],
                                 "f_hi": edges[b + 1], "f_lo_hz": hz(edges[b]), "f_hi_hz": hz(edges[b + 1])})
            else:  # one value per window and channel
                columns.append(np.log(total[:, c]) if kind == "total" else centroid[:, c])
                info.append({"channel": channel, "target": kind, "band": None, "f_lo": np.nan, "f_hi": np.nan,
                             "f_lo_hz": np.nan, "f_hi_hz": np.nan})
    return np.stack(columns, axis=1), info


# ---------------------------------------------------------------- probes

def cv_folds(n, groups, study, seed):
    """Folds for choosing alpha: grouped by bearing on datasets split by bearing (as the benchmark, D129),
    else plain shuffled folds (as harness/evaluate.ridge); shuffled with the seed."""
    if study["cv_groups"]:
        return list(GroupKFold(n_splits=study["cv_folds"], shuffle=True, random_state=seed).split(np.zeros(n), groups=groups))
    return list(KFold(n_splits=study["cv_folds"], shuffle=True, random_state=seed).split(np.zeros(n)))


def ridge_per_target(Z, Y, Z_test, folds, alphas):
    """A ridge probe for every column of Y, each with its own alpha; returns the alphas and the test predictions.

    The same as harness/evaluate.ridge (StandardScaler + Ridge, GridSearchCV: R² averaged over the folds,
    the first best alpha on ties) run once per column with these folds. Ridge fits every column of Y
    separately, so all columns are fitted together for each alpha; this only saves time."""
    cv = np.zeros((len(alphas), Y.shape[1]))  # mean validation R² per alpha and target
    for a, alpha in enumerate(alphas):
        for tr, va in folds:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Z[tr], Y[tr])
            cv[a] += r2_score(Y[va], model.predict(Z[va]), multioutput="raw_values") / len(folds)
    best = cv.argmax(axis=0)  # first maximum per target
    pred = np.empty((len(Z_test), Y.shape[1]))
    for a in np.unique(best):  # refit on all labelled windows, once per chosen alpha
        cols = best == a
        model = make_pipeline(StandardScaler(), Ridge(alpha=alphas[a])).fit(Z, Y[:, cols])
        pred[:, cols] = model.predict(Z_test).reshape(len(Z_test), -1)
    return np.asarray(alphas)[best], pred


def per_bearing_rows(pred, Y_test, bearings, info, about):
    """Pearson's correlation between the predictions and the targets within each test bearing (D265):
    does the representation follow how a bearing's spectrum changes over its life, whatever its level?"""
    rows = []
    for b in pd.unique(bearings[pd.notna(bearings)]):  # windows without a bearing (e.g. CWRU's healthy file) are left out
        m = bearings == b
        for j, target in enumerate(info):
            y, p = Y_test[m, j], pred[m, j]
            r = np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan  # undefined if either is constant
            rows.append({**about, **target, "probe": "ridge", "test_bearing": b, "n_windows": int(m.sum()), "pearson_r": r})
    return rows


def probe_rows(Z_train, Z_test, Y_train, Y_test, info, groups, study, about, bearings=None):
    """Result rows of one representation (a run's Z or the reference) at every label fraction, and the
    per-bearing rows (D265) when the test bearings are given."""
    rows, bearing_rows = [], []
    seed = study["seed"]
    for fraction in study["label_fractions"]:
        idx = evaluate.label_subset(Y_train[:, 0], fraction, seed, stratify=False)  # unstratified (D170)
        folds = cv_folds(len(idx), groups[idx] if groups is not None else None, study, seed)
        alpha, pred = ridge_per_target(Z_train[idx], Y_train[idx], Z_test, folds, study["alphas"])
        r2 = r2_score(Y_test, pred, multioutput="raw_values")
        floor = r2_score(Y_test, np.tile(Y_train[idx].mean(axis=0), (len(Y_test), 1)), multioutput="raw_values")
        if bearings is not None:
            bearing_rows += per_bearing_rows(pred, Y_test, bearings, info, {**about, "label_fraction": fraction})
        for j, target in enumerate(info):
            common = {**about, "label_fraction": fraction, "n_labels": len(idx), **target}
            rows.append({**common, "probe": "ridge", "alpha": alpha[j], "r2": r2[j]})
            rows.append({**common, "probe": "floor_mean", "alpha": np.nan, "r2": floor[j]})
        banded = [t["band"] is not None for t in info]  # the band targets (absolute, relative)
        print(f"  {about['name']}, {fraction:.0%} of the labels: mean R² over the band targets "
              f"{np.mean(r2[banded]):.3f}", flush=True)
    return rows, bearing_rows


# ---------------------------------------------------------------- main

def main(config_path):
    study = runs.load_config(config_path)
    if not set(study["targets"]) <= set(KINDS):
        raise ValueError(f"targets must be among {KINDS}")
    rows, bearing_rows, key = [], [], None
    for folder in study["runs"]:
        folder = folder.rstrip("/")
        print(folder, flush=True)
        if window_set_key(folder) != key:  # new encoder inputs: targets (and the reference) computed once
            key = window_set_key(folder)
            X, groups, cfg = encoder_inputs(folder)
            rate = cfg.get("windows", {}).get("sample_rate_hz")  # FordA: none given
            (Y_train, info), (Y_test, _) = (spectral_targets(X[p], study["n_bands"], study["targets"],
                                                             cfg["channels"], rate) for p in ("train", "test"))
            combination = None  # protocol runs (Paderborn, D220): the combination k of split.json
            split_path = os.path.join(folder, "split.json")
            if os.path.exists(split_path):
                with open(split_path) as f:
                    combination = json.load(f)["combination"]
            bearings = test_bearings(folder) if study["per_bearing"] else None  # D265
            if study["fft_reference"]:  # the FFT baseline's features of the same windows, in chunks
                F_train, F_test = (np.concatenate([fft.encode(None, X[p][i:i + CHUNK]) for i in range(0, len(X[p]), CHUNK)])
                                   for p in ("train", "test"))
                about = {"run": os.path.basename(folder), "name": "fft_reference", "combination": combination}
                r, b = probe_rows(F_train, F_test, Y_train, Y_test, info, groups, study, about, bearings)
                rows, bearing_rows = rows + r, bearing_rows + b
                del F_train, F_test
            del X  # only the targets are needed from here on
        cfg = run_config(folder)
        seed_dir = os.path.join(folder, f"seed_{study['seed']}")
        Z_train, Z_test = np.load(os.path.join(seed_dir, "Z_train.npy")), np.load(os.path.join(seed_dir, "Z_test.npy"))
        if (len(Z_train), len(Z_test)) != (len(Y_train), len(Y_test)):
            raise ValueError(f"{folder}: Z has {len(Z_train)} / {len(Z_test)} rows, the windows {len(Y_train)} / {len(Y_test)}")
        about = {"run": os.path.basename(folder), "name": cfg["name"], "combination": combination}
        r, b = probe_rows(Z_train, Z_test, Y_train, Y_test, info, groups, study, about, bearings)
        rows, bearing_rows = rows + r, bearing_rows + b

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{study['name']}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Done: {path}")
    if bearing_rows:  # D265
        path = os.path.join(OUT, f"{study['name']}_per_bearing.csv")
        pd.DataFrame(bearing_rows).to_csv(path, index=False)
        print(f"Done: {path}")


if __name__ == "__main__":
    main(sys.argv[1])
