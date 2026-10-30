"""Features computed once per capture and stored, read back by the pipeline (D172, D181).

One HDF5 file per feature set and dataset: 'features' (captures, windows, feature channels), the
run and capture ID of each row, the feature names, and the settings that determine the features.
load() stops if the stored settings differ from the code's, so a changed definition needs the
file to be made again (scripts/derive_features.py).
"""

import json

import h5py
import numpy as np


def create(path, n_captures, row_shape, settings, names):
    """An empty file for n_captures rows of shape row_shape (windows, feature channels); fill it row by row."""
    f = h5py.File(path, "w")
    f.create_dataset("features", (n_captures,) + tuple(row_shape), dtype="float64")
    f.create_dataset("run_id", (n_captures,), dtype=h5py.string_dtype())
    f.create_dataset("capture_id", (n_captures,), dtype=h5py.string_dtype())
    f.attrs["settings"] = json.dumps(settings)
    f.attrs["feature_names"] = json.dumps(names)
    return f


def load(path, run_ids, capture_ids, settings):
    """Stored features (n, windows, feature channels) of the given captures, in the given order.

    Stops if the file was made with other settings, or lacks a capture.
    """
    with h5py.File(path, "r") as f:
        stored = json.loads(f.attrs["settings"])
        if stored != json.loads(json.dumps(settings)):
            raise ValueError(f"{path} was made with other settings: {stored}")
        keys = list(zip(f["run_id"].asstr()[:], f["capture_id"].asstr()[:]))
        row = {k: i for i, k in enumerate(keys)}
        missing = [k for k in zip(run_ids, capture_ids) if k not in row]
        if missing:
            raise ValueError(f"{len(missing)} captures are not in {path}, e.g. {missing[:3]}")
        idx = np.array([row[k] for k in zip(run_ids, capture_ids)])
        order = np.argsort(idx)  # h5py reads rows in increasing order; put them back afterwards
        X = np.empty((len(idx),) + f["features"].shape[1:])
        X[order] = f["features"][np.sort(idx)]
    return X
