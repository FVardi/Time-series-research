"""Reader for AU Herning test-rig files (the source format).

One source file holds one run: a series of short oscilloscope "sweeps"
(0.2 s each, taken about every 12 s). Each sweep records several channels at
the same rate; we use only acoustic emission (AE) and ultrasound (UL) (D25).
Speed, temperature and controller state are stored as attributes on each sweep.

This module only *reads*. Writing the canonical format is in canonical.py.
"""

import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd

DATASET = "herning"
STREAM = "scope"  # AE and UL come from the same oscilloscope, so they share one clock (D1)
CHANNELS = {"AE": "acoustic_emission", "UL": "ultrasound"}  # source name -> sensor type (D22, D25)
TIMING_ACCURACY = "host clock; exact trigger time unknown"  # D22 item 4


def run_id_from_path(path):
    """'scope_20260901_112732.h5' -> 'herning_20260901_112732'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{DATASET}_{stem.removeprefix('scope_')}"


def _plain(value):
    """Convert an HDF5 attribute value to a plain Python value (for tables and JSON)."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs(obj):
    """All attributes of an HDF5 group or dataset as a plain dict."""
    return {k: _plain(v) for k, v in obj.attrs.items()}


def sweep_names(f):
    """Source sweep names sorted by acquisition time.

    The names themselves do not sort correctly as text ('sweep_1000' < 'sweep_999'),
    so we sort by the 'tick' attribute (seconds since the run started).
    """
    sweeps = f["sweeps"]
    return sorted(sweeps.keys(), key=lambda name: sweeps[name].attrs["tick"])


def read_tables(path, max_captures=None):
    """Build the four metadata tables (D2) for one source file.

    Returns a dict of DataFrames: runs, captures, channels, telemetry.
    max_captures limits the number of sweeps (used by the tests only).
    """
    run_id = run_id_from_path(path)
    stat = os.stat(path)

    with h5py.File(path, "r") as f:
        names = sweep_names(f)[:max_captures]
        meta = f["metadata"]

        # --- runs: one row. Everything in the source metadata goes into 'extras', as recorded (D22 item 7).
        extras = {key: _attrs(meta[key]) for key in ("bearing", "lubricant", "test_parameters")}
        extras["recording"] = _attrs(meta)
        extras["scope_settings"] = {"common": _attrs(meta["scope_settings"])}
        for ch in meta["scope_settings"]:
            extras["scope_settings"][ch] = _attrs(meta["scope_settings"][ch])
        extras["note"] = ("test_parameters contradicts the actual schedule "
                          "(60 min / 500 rpm / 60 C vs 13 h, 0-3000 rpm, 25-100 C).")
        bearing = extras["bearing"]
        runs = pd.DataFrame([{
            "dataset": DATASET,
            "run_id": run_id,
            "status": "validation",  # D7, D22 item 2
            "bearing": f"{bearing['manufacturer']} {bearing['model']}",
            "lubricant": extras["lubricant"]["product_name"],
            "source_name": os.path.basename(path),  # D24: name, size and modification time
            "source_size_bytes": stat.st_size,
            "source_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "extras": json.dumps(extras),  # stored as a JSON string: different datasets have different extras
        }])

        # --- channels: one row per channel. Taken from the first sweep;
        # the conversion script checks that every sweep has the same values.
        first = f["sweeps"][names[0]]
        channels = pd.DataFrame([{
            "run_id": run_id,
            "stream": STREAM,
            "channel": ch,
            "sensor_type": sensor,
            "sample_rate_hz": float(first[ch].attrs["sample_rate"]),
            "units": "V",  # scope reports 'VOLT'
            "scale": float(first[ch].attrs["y_increment"]),  # volts per integer code
            "offset": float(first[ch].attrs["y_origin"]),  # volts at code 0
            "x_origin_s": float(first[ch].attrs["x_origin"]),  # time of sample 0 relative to the trigger
            "dtype": "int16",  # D20
        } for ch, sensor in CHANNELS.items()])

        # --- captures and telemetry: one row per sweep each.
        captures, telemetry = [], []
        for i, name in enumerate(names):
            sweep = f["sweeps"][name]
            attrs = _attrs(sweep)
            capture_id = f"sweep_{i:04d}"  # zero-padded so the IDs sort correctly (D22 item 3)
            captures.append({
                "run_id": run_id,
                "capture_id": capture_id,
                "stream": STREAM,
                "source_name": name,
                "tick_s": attrs["tick"],  # seconds since run start (host clock)
                "timestamp_utc": attrs["timestamp_utc"],
                # Start of the recorded signal. AE's trigger offset is used; UL's differs
                # by 1 ns (0.0025 of a sample) and is kept exactly in the channels table.
                "start_s": attrs["tick"] + float(sweep["AE"].attrs["x_origin"]),
                "n_samples": int(sweep["AE"]["voltage"].shape[0]),
                "timing_accuracy": TIMING_ACCURACY,
            })
            # Every telemetry field exactly as recorded; nothing derived here (D22 item 6).
            row = {"run_id": run_id, "capture_id": capture_id}
            row.update({k: v for k, v in attrs.items() if k.startswith("telem_")})
            telemetry.append(row)

    return {
        "runs": runs,
        "channels": channels,
        "captures": pd.DataFrame(captures),
        "telemetry": pd.DataFrame(telemetry),
    }


def read_codes(f, source_name, channel, scale, offset):
    """Return one channel of one sweep as int16 codes (D6, D20).

    The source stores volts as 64-bit floats, but every value is an exact
    integer code times 'scale' plus 'offset'. We recover the codes and stop
    with an error if any value is not an integer code, so nothing is lost silently.
    """
    volts = f["sweeps"][source_name][channel]["voltage"][:]
    exact = (volts - offset) / scale
    codes = np.round(exact)
    if np.abs(exact - codes).max() > 1e-6:
        raise ValueError(f"{source_name}/{channel}: values are not integer codes")
    if codes.min() < -32768 or codes.max() > 32767:
        raise ValueError(f"{source_name}/{channel}: codes do not fit in int16")
    return codes.astype(np.int16)
