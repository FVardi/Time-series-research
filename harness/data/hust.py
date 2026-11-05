"""Reader for the HUST bearing data (Hanoi University of Science and Technology; Thuan & Hong, 2023).

Decisions: D189, D190 (adopted), D194 (download), D223 (conversion).
Expects the 99 .mat files in one folder (as downloaded, e.g.
data/raw/hust/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset/).

File names: <fault><bearing digit>0<load digit>.mat, e.g. I402.mat = inner fault, bearing 6204, 200 W
(the data note explains the bearing and load digits; the middle 0 is in every file name).
One run per file, one capture, one stream, one channel (the steady-state vibration signal `data`,
as used by the dataset authors). The run-up recordings (`ru`, `ru_raw`, `rpm`) are not converted
(D223). Values are stored as recorded (64-bit floats, scale 1, offset 0; D5, D6). Nothing is
derived, filled in or corrected here; quirks are recorded in the run's conversion notes.

Sources: Thuan & Hong (2023), BMC Research Notes 16:138, doi 10.1186/s13104-023-06400-4
(data note: variables, sensor, bearings, loads) and arXiv 2302.12533 (the authors' experiments).
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io

DATASET = "hust"
STREAM = "accel"
CHANNEL = "vibration"  # the files' variable `data`
SAMPLE_RATE_HZ = 51200.0  # data note: "51,200 samples per second for 10 s"
EXPECTED_SAMPLES = 512000  # 10 s at 51.2 kHz (data note); other lengths are recorded as quirks
EXPECTED_VARS = {"data", "fs", "rpm", "ru", "ru_raw"}  # data note; only `data` and `fs` are converted

# Fault code -> damaged components (data note: wire-cut cracks of 0.2 mm width).
FAULTS = {
    "N": (), "I": ("inner",), "O": ("outer",), "B": ("ball",),
    "IO": ("inner", "outer"), "IB": ("inner", "ball"), "OB": ("outer", "ball"),
}
# Bearing digit -> type and (inner diameter mm, outer diameter mm, ball diameter mm, number of balls),
# as given in the data note (KG Bearing India).
BEARINGS = {
    "4": ("6204", 20, 47, 7.6, 8),
    "5": ("6205", 25, 52, 7.8, 9),
    "6": ("6206", 30, 62, 9.0, 9),
    "7": ("6207", 35, 72, 11.0, 9),
    "8": ("6208", 40, 80, 12.0, 9),
}
LOADS_W = {"0": 0, "2": 200, "4": 400}  # load digit -> powder-brake load in W (data note)

# All 99 files: B and IB are missing for 6204 (a limitation the data note states).
FILES = [f"{fault}{b}0{load}" for fault in FAULTS for b in BEARINGS for load in LOADS_W
         if not (fault in ("B", "IB") and b == "4")]
assert len(FILES) == 99

SENSOR = "PCB 325C33 accelerometer, vertical, on the bearing housing; NI-9234 module, NI CompactDAQ, LabVIEW"

# Quirks found by inspecting the files (2026-11-05) that a per-file check cannot see (D223).
KNOWN_NOTES = {
    "IO602": "fs = 22.89 Hz, identical to IO604; the other 200 W files have 23.96-24.42 Hz",
}


def parse(name):
    """File name (without .mat) -> (fault code, bearing digit, load digit)."""
    return name[:-3], name[-3], name[-1]


def run_id(name):
    return name  # the file name, e.g. "I402"


def read(folder, name):
    """Return (tables, signals, notes) of one file; notes lists the quirks found."""
    path = os.path.join(folder, f"{name}.mat")
    mat = scipy.io.loadmat(path)
    variables = {k for k in mat if not k.startswith("__")}
    unexpected = variables - EXPECTED_VARS
    if unexpected or "data" not in variables or "fs" not in variables:
        raise ValueError(f"{name}.mat: variables {sorted(variables)}")

    x = mat["data"].ravel()  # float64 as recorded
    notes = []
    if len(x) != EXPECTED_SAMPLES:
        notes.append(f"data has {len(x)} samples, not {EXPECTED_SAMPLES} (10 s, data note)")
    if name in KNOWN_NOTES:
        notes.append(KNOWN_NOTES[name])

    fault, b, load = parse(name)
    btype, d_in, d_out, d_ball, n_balls = BEARINGS[b]
    bearing = f"{fault}{btype}"  # one test bearing per fault type and bearing type, recorded at three loads
    rid = run_id(name)
    capture = {
        "run_id": rid, "capture_id": "capture_0", "stream": STREAM,
        "source_name": f"{name}.mat", "start_s": np.nan, "n_samples": len(x),
        "timing_accuracy": "not applicable",
        # per-capture columns (D31): labels from the file name and bearing data from the data note
        "fault": fault,
        "inner_fault": "inner" in FAULTS[fault],
        "outer_fault": "outer" in FAULTS[fault],
        "ball_fault": "ball" in FAULTS[fault],
        "bearing": bearing, "bearing_type": btype, "load_w": LOADS_W[load],
        "inner_diameter_mm": float(d_in), "outer_diameter_mm": float(d_out),
        "ball_diameter_mm": d_ball, "n_balls": n_balls,
    }
    telemetry = {"run_id": rid, "capture_id": "capture_0",
                 "shaft_freq_hz": float(mat["fs"].ravel()[0])}  # `fs`: shaft frequency in Hz (data note), as recorded

    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7)
        "bearing": bearing, "lubricant": None,
        "source_name": f"{name}.mat",  # D24
        "source_size_bytes": os.path.getsize(path),
        "source_mtime_utc": datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(),
        "extras": json.dumps({
            "sensor": SENSOR,
            "units_note": "not stated in the data note or the files",
            "conversion_notes": notes,
            "not_converted": sorted(variables - {"data", "fs"}),  # run-up recordings (D223)
        }),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": CHANNEL, "sensor_type": "accelerometer",
        "sample_rate_hz": SAMPLE_RATE_HZ, "units": "unknown", "scale": 1.0, "offset": 0.0,
        "x_origin_s": np.nan, "dtype": "float64",
    }])
    tables = {"runs": runs, "captures": pd.DataFrame([capture]),
              "channels": channels, "telemetry": pd.DataFrame([telemetry])}
    signals = [("capture_0", CHANNEL, x, 1.0, 0.0, "unknown")]
    return tables, signals, notes
