"""Reader for the University of Ottawa ball-bearing vibration and acoustic fault data under constant
load and speed (UODS-VAFDC; Sehri, Dumond & Bouchard, 2023, Data in Brief 49, 109327,
doi 10.1016/j.dib.2023.109327; Mendeley Data y2px5tg92h; CC BY 4.0).

Decisions: D189 (adopted), D194 (download), D249 (conversion), D250 (task and split).
Expects the 60 .mat files in their class folders, as downloaded (data/raw/ottawa_vafdc/University of
Ottawa .../1_Healthy/H_1_0.mat ... 5_Cage_Faults/C_20_2.mat). File name <condition>_<bearing>_<stage>:
condition H, I, O, B, C (healthy, inner race, outer race, ball, cage); bearing 1-20; stage 0 healthy,
1 fault developing, 2 faulty (data note).

One run per bearing (one run-to-failure test, D1): its healthy file H_<k>_0 and its two fault files
<X>_<k>_1 and <X>_<k>_2 are three captures (stages 0, 1, 2). That H_<k>_0 belongs to bearing k is
taken from the data note's naming (first number = bearing ID); the measured speeds agree (D249).
One stream (accelerometer and microphone on one NI USB-6212), two channels, 42 kHz, 10 s. Values
are stored as recorded (64-bit floats, scale 1, offset 0; D5, D6: not on an exact ADC grid).
Speed and load are single values in the first row of columns 3 and 4; they go into the telemetry
table. Nothing is derived here; quirks are recorded in the run's conversion notes.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io

DATASET = "ottawa_vafdc"
STREAM = "daq"  # accelerometer and microphone share the NI USB-6212 (data note)
SAMPLE_RATE_HZ = 42000.0  # data note
EXPECTED_SAMPLES = 420000  # 10 s at 42 kHz (data note)
# Columns of each file's one variable, in order (data note lists five: accelerometer, acoustic,
# speed, load, temperature; the files hold the first four).
CHANNELS = {"accelerometer": (0, "m/s^2", "PCB 623C01 accelerometer, magnetically mounted on the drive-end bearing inside the motor casing"),
            "microphone": (1, "V", "PCB 130F20 microphone, within 2 cm of the drive-end bearing")}
SPEED_COL, LOAD_COL = 2, 3

FOLDERS = {"H": "1_Healthy", "I": "2_Inner_Race_Faults", "O": "3_Outer_Race_Faults",
           "B": "4_Ball_Faults", "C": "5_Cage_Faults"}
FAULT_OF_BEARING = {k: "I" if k <= 5 else "O" if k <= 10 else "B" if k <= 15 else "C" for k in range(1, 21)}  # data note
BEARING_MODEL = {k: "NSK 6203ZZ" if k <= 5 else "FAFNIR 203KD" for k in range(1, 21)}  # tests 1-5 / 6-20 (data note)
GEOMETRY = {"pitch_diameter_mm": 28.50, "ball_diameter_mm": 6.77, "n_balls": 8}  # both models (data note)
STAGES = {0: "healthy", 1: "fault developing", 2: "faulty"}  # data note
SETUP = ("single-phase motor, 1,750 rpm nominal; SKF E22206 spherical roller bearing loaded by a cantilever beam "
         "and lead screw, nominal 400 N (ball-fault data: no load); drive-end test bearing degreased, seals removed; "
         "stage files chosen from 50 per bearing: healthy 1-5, developing 25-30, faulty 45-50 (data note)")
BEARINGS = list(range(1, 21))


def files_of(bearing):
    """The three file names of a bearing: [H_k_0, X_k_1, X_k_2]."""
    x = FAULT_OF_BEARING[bearing]
    return [f"H_{bearing}_0", f"{x}_{bearing}_1", f"{x}_{bearing}_2"]


def path_of(folder, name):
    return os.path.join(folder, FOLDERS[name[0]], f"{name}.mat")


def run_id(bearing):
    return f"bearing_{bearing:02d}"


def _one_value(column, name, what):
    """The value in the first row of a column that is zero everywhere else (speed, load)."""
    if np.any(column[1:] != 0):
        raise ValueError(f"{name}: {what} column is not zero after the first row")  # layout not as checked: stop
    return float(column[0])


def read(folder, bearing):
    """Return (tables, signals, notes) of one bearing; notes lists the quirks found."""
    rid = run_id(bearing)
    names = files_of(bearing)
    paths = [path_of(folder, n) for n in names]
    notes, captures, telemetry, signals, hashes = [], [], [], [], {}
    for stage, (name, path) in enumerate(zip(names, paths)):
        mat = scipy.io.loadmat(path)
        variables = [k for k in mat if not k.startswith("__")]
        if variables != [name]:
            raise ValueError(f"{name}.mat: variables {variables}")
        v = mat[name]
        if v.ndim != 2 or v.shape[1] != 4:
            raise ValueError(f"{name}.mat: shape {v.shape}, expected (n, 4)")
        if v.shape[0] != EXPECTED_SAMPLES:
            notes.append(f"{name}: {v.shape[0]} samples, not {EXPECTED_SAMPLES}")  # stored as recorded
        if np.isnan(v).any():
            notes.append(f"{name}: {int(np.isnan(v).sum())} NaN values")  # stored as recorded
        capture_id = f"stage_{stage}"
        h = hashlib.sha256(v[:, :2].tobytes()).hexdigest()
        copy_of = hashes.get(h)
        hashes.setdefault(h, capture_id)
        if copy_of:
            notes.append(f"{capture_id} ({name}): signals identical to {copy_of}")
        captures.append({
            "run_id": rid, "capture_id": capture_id, "stream": STREAM,
            "source_name": f"{FOLDERS[name[0]]}/{name}.mat", "n_samples": v.shape[0],
            "start_s": np.nan,  # no timestamps; the stages are in time order (data note)
            "timing_accuracy": "not available: stages of one run-to-failure test, order only",
            # per-capture columns (D31), from the file name and the data note
            "condition": name[0], "stage": stage, "stage_name": STAGES[stage],
            "bearing_id": bearing, "bearing_fault": FAULT_OF_BEARING[bearing],
            "bearing_model": BEARING_MODEL[bearing], "identical_to": copy_of,
        })
        telemetry.append({"run_id": rid, "capture_id": capture_id,
                          "speed_rpm": _one_value(v[:, SPEED_COL], name, "speed"),  # as recorded, first row
                          "load_n": _one_value(v[:, LOAD_COL], name, "load")})
        for ch, (col, units, _) in CHANNELS.items():
            signals.append((capture_id, ch, np.ascontiguousarray(v[:, col]), 1.0, 0.0, units))

    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7)
        "bearing": f"{FAULT_OF_BEARING[bearing]}-{bearing}", "lubricant": "none (degreased, data note)",
        # D24 for a run made of three files: names, total size, latest modification time
        "source_name": ", ".join(f"{n}.mat" for n in names),
        "source_size_bytes": sum(os.path.getsize(p) for p in paths),
        "source_mtime_utc": datetime.fromtimestamp(max(os.path.getmtime(p) for p in paths), timezone.utc).isoformat(),
        "extras": json.dumps({
            "bearing_model": BEARING_MODEL[bearing], "geometry": GEOMETRY, "setup": SETUP,
            "sensors": {ch: d for ch, (_, _, d) in CHANNELS.items()},
            "telemetry_note": "speed (rpm) and load (N): the single non-zero value in the first row of columns 3 and 4; "
                              "the data note's temperature column is not in the files",
            "conversion_notes": notes,
        }),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": ch, "sensor_type": ch,
        "sample_rate_hz": SAMPLE_RATE_HZ, "units": units, "scale": 1.0, "offset": 0.0,
        "x_origin_s": np.nan, "dtype": "float64",
    } for ch, (_, units, _) in CHANNELS.items()])
    tables = {"runs": runs, "captures": pd.DataFrame(captures),
              "channels": channels, "telemetry": pd.DataFrame(telemetry)}
    return tables, signals, notes
