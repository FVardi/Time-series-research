"""Reader for FordA from the UCR time-series classification archive (D27).

Expects the two files FordA_TRAIN and FordA_TEST (extension .tsv or .txt) in one
folder. Each line is one series: the class label first, then 500 values,
separated by tabs or spaces. The archive has already normalised every series
to mean 0 and standard deviation 1; we use the values as they are.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATASET = "forda"
RUN_ID = "forda"
STREAM = "series"
CHANNEL = "x"


def find_file(folder, part):
    """Path of FordA_TRAIN / FordA_TEST, whichever text extension is present."""
    for ext in (".tsv", ".txt"):
        path = os.path.join(folder, f"FordA_{part}{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"FordA_{part}.tsv or .txt not found in {folder}")


def read(folder):
    """Return (tables, signals) for the canonical writer.

    tables: dict with runs, captures, channels, telemetry (D2).
    signals: list of (capture_id, channel, values, scale, offset, units).
    """
    captures, signals, files = [], [], {}
    for part in ("TRAIN", "TEST"):
        path = find_file(folder, part)
        stat = os.stat(path)
        files[os.path.basename(path)] = {
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        data = np.loadtxt(path)  # works for tab- and space-separated files
        split = part.lower()
        for i, row in enumerate(data):
            capture_id = f"{split}_{i:04d}"
            values = row[1:]
            captures.append({
                "run_id": RUN_ID,
                "capture_id": capture_id,
                "stream": STREAM,
                "source_name": f"{os.path.basename(path)}:line {i + 1}",
                "start_s": np.nan,  # no time information in the archive
                "n_samples": len(values),
                "timing_accuracy": "not applicable",
                # Class label and official split stored on the captures table (D30).
                "label": int(row[0]),  # as in the file: -1 or 1
                "official_split": split,
            })
            # Values are floats in the archive, so they are stored as floats (D6): scale 1, offset 0.
            signals.append((capture_id, CHANNEL, values, 1.0, 0.0, "normalised (unitless)"))

    runs = pd.DataFrame([{
        "dataset": DATASET,
        "run_id": RUN_ID,
        "status": "final",  # public benchmark, not a validation run (D7, D30)
        "bearing": None,
        "lubricant": None,
        "source_name": "; ".join(files),  # D24, per file in extras
        "source_size_bytes": sum(f["size_bytes"] for f in files.values()),
        "source_mtime_utc": max(f["mtime_utc"] for f in files.values()),
        "extras": json.dumps({"files": files, "note": "UCR archive; every series z-normalised by the archive (D27)."}),
    }])
    channels = pd.DataFrame([{
        "run_id": RUN_ID,
        "stream": STREAM,
        "channel": CHANNEL,
        "sensor_type": "unknown",  # the archive describes it as engine noise; sensor not specified
        "sample_rate_hz": np.nan,  # not given in the archive
        "units": "normalised (unitless)",
        "scale": 1.0,
        "offset": 0.0,
        "x_origin_s": np.nan,
        "dtype": "float64",
    }])
    telemetry = pd.DataFrame(columns=["run_id", "capture_id"])  # FordA has no telemetry
    tables = {"runs": runs, "captures": pd.DataFrame(captures), "channels": channels, "telemetry": telemetry}
    return tables, signals
