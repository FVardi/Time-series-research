"""The canonical data format, shared by all datasets (D2, D3, D21, D23).

Layout on disk, one folder per run:

    <root>/<dataset>/<run_id>/
        runs.parquet  captures.parquet  channels.parquet  telemetry.parquet
        signals.h5    one dataset per capture and channel: /<capture_id>/<channel>

Signals are stored as integer codes; each dataset carries its own 'scale',
'offset' and 'units', so physical value = code * scale + offset.
Canonical data is written once and never modified (D5).
"""

import os

import h5py
import numpy as np
import pandas as pd

TABLES = ("runs", "captures", "channels", "telemetry")
CHUNK = 16_384  # samples per HDF5 chunk (D21)


def run_dir(root, dataset, run_id):
    return os.path.join(root, dataset, run_id)


def write_run(root, dataset, run_id, tables, signals):
    """Write one run: the four tables and all its signals.

    signals: iterable of (capture_id, channel, codes, scale, offset, units).
    Refuses to overwrite an existing run, because canonical data is immutable (D5).
    """
    folder = run_dir(root, dataset, run_id)
    if os.path.exists(folder):
        raise FileExistsError(f"{folder} already exists; delete it by hand to reconvert.")
    os.makedirs(folder)

    with h5py.File(os.path.join(folder, "signals.h5"), "w") as f:
        for capture_id, channel, codes, scale, offset, units in signals:
            ds = f.create_dataset(
                f"{capture_id}/{channel}",
                data=codes,
                chunks=(min(CHUNK, len(codes)),),
                compression="gzip",
                compression_opts=4,  # D23
            )
            ds.attrs["scale"] = scale
            ds.attrs["offset"] = offset
            ds.attrs["units"] = units

    # Tables last: a run folder without tables means the conversion did not finish.
    for name in TABLES:
        tables[name].to_parquet(os.path.join(folder, f"{name}.parquet"), index=False)


def load_metadata(root, dataset):
    """The four tables for all converted runs of a dataset, as a dict of DataFrames."""
    base = os.path.join(root, dataset)
    runs = sorted(os.listdir(base))
    return {
        name: pd.concat(
            [pd.read_parquet(os.path.join(base, r, f"{name}.parquet")) for r in runs],
            ignore_index=True,
        )
        for name in TABLES
    }


def load_signal(root, dataset, run_id, capture_id, channel, start=0, stop=None):
    """One channel of one capture, samples [start, stop), in physical units.

    Returned as 64-bit floats; conversion to 32-bit happens at training (D23).
    """
    path = os.path.join(run_dir(root, dataset, run_id), "signals.h5")
    with h5py.File(path, "r") as f:
        ds = f[f"{capture_id}/{channel}"]
        codes = ds[start:stop]
        return codes.astype(np.float64) * ds.attrs["scale"] + ds.attrs["offset"]


def load_captures(root, dataset, run_id, capture_ids, channels, start=0, stop=None):
    """Several captures at once, as one array of shape (n_captures, n_samples, n_channels).

    Same values as load_signal, but opens the file only once. All requested
    captures must have the same length between start and stop.
    """
    path = os.path.join(run_dir(root, dataset, run_id), "signals.h5")
    out = []
    with h5py.File(path, "r") as f:
        for capture_id in capture_ids:
            per_channel = []
            for channel in channels:
                ds = f[f"{capture_id}/{channel}"]
                per_channel.append(ds[start:stop].astype(np.float64) * ds.attrs["scale"] + ds.attrs["offset"])
            out.append(np.stack(per_channel, axis=-1))
    return np.stack(out)
