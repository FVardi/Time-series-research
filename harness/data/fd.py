"""Reader for FD-A and FD-B as distributed by the TF-C authors (D39).

FD-A and FD-B are windows cut by the TF-C authors from the Paderborn bearing
dataset, one operating condition each (Zhang et al., 2022; figshare articles
19930205 and 19930226). Each folder holds train.pt, val.pt and test.pt: a dict
with 'samples' (n, 1, 5120) and 'labels' (n,) with classes 0, 1, 2.

The files say nothing about which recording or bearing a window comes from, so
leakage between the splits cannot be checked (D39). Values are used as they are.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

DATASET = "fd"
STREAM = "window"
CHANNEL = "x"
SPLITS = ("train", "val", "test")


def read(folder, run_id):
    """Return (tables, signals) for the canonical writer.

    folder: the folder with train.pt, val.pt and test.pt.
    run_id: 'fd_a' or 'fd_b' (one run per operating condition).
    """
    captures, signals, files = [], [], {}
    for split in SPLITS:
        path = os.path.join(folder, f"{split}.pt")
        stat = os.stat(path)
        files[os.path.basename(path)] = {
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        data = torch.load(path, weights_only=True)  # plain tensors only, no code is run
        samples = data["samples"].numpy()
        labels = data["labels"].numpy()
        assert samples.ndim == 3 and samples.shape[1] == 1, f"unexpected shape {samples.shape} in {path}"
        for i in range(len(samples)):
            capture_id = f"{split}_{i:05d}"
            values = samples[i, 0]
            captures.append({
                "run_id": run_id,
                "capture_id": capture_id,
                "stream": STREAM,
                "source_name": f"{os.path.basename(path)}:row {i}",
                "start_s": np.nan,  # position in the original recording is not given
                "n_samples": len(values),
                "timing_accuracy": "not applicable",
                # Class label and official split on the captures table (D30, D31, D39).
                "label": int(labels[i]),  # 0, 1, 2 as in the file (undamaged / inner / outer damage per TF-C)
                "official_split": split,
            })
            # Values are floats in the files, so they are stored as floats (D6): scale 1, offset 0.
            signals.append((capture_id, CHANNEL, values, 1.0, 0.0, "unknown (as distributed)"))

    runs = pd.DataFrame([{
        "dataset": DATASET,
        "run_id": run_id,
        "status": "final",  # public benchmark (D7, D39)
        "bearing": None,  # not given per window
        "lubricant": None,
        "source_name": "; ".join(files),  # D24, per file in extras
        "source_size_bytes": sum(f["size_bytes"] for f in files.values()),
        "source_mtime_utc": max(f["mtime_utc"] for f in files.values()),
        "extras": json.dumps({"files": files, "note": "TF-C preprocessing of the Paderborn data; "
                              "recording and bearing per window unknown (D39)."}),
    }])
    channels = pd.DataFrame([{
        "run_id": run_id,
        "stream": STREAM,
        "channel": CHANNEL,
        "sensor_type": "unknown",  # PLACEHOLDER: TF-C does not say which Paderborn signal (vibration or motor current)
        "sample_rate_hz": 64000.0,  # TF-C README, dataset table
        "units": "unknown (as distributed)",
        "scale": 1.0,
        "offset": 0.0,
        "x_origin_s": np.nan,
        "dtype": str(samples.dtype),
    }])
    telemetry = pd.DataFrame(columns=["run_id", "capture_id"])  # no telemetry
    tables = {"runs": runs, "captures": pd.DataFrame(captures), "channels": channels, "telemetry": telemetry}
    return tables, signals
