"""Reader for the XJTU-SY run-to-failure bearing data (Xi'an Jiaotong University and
Changxing Sumyoung Technology; Wang et al., 2020; Lei et al., 2019).

Decisions: D189 (adopted), D191 (tasks, derived labels), D193 (download), D213 (unpacked), D224 (conversion).
Expects the unpacked dataset folder (data/raw/xjtu_sy/XJTU-SY_Bearing_Datasets/) with
<condition>/<bearing>/<n>.csv, e.g. 35Hz12kN/Bearing1_1/1.csv, n = 1..N.

One run per bearing (one run-to-failure test), each CSV (one 1.28 s sample, taken once a minute)
one capture, one stream (both accelerometers, 25.6 kHz), two channels (D224).
Values are stored as recorded (64-bit floats, scale 1, offset 0; D5, D6). Nothing is derived here:
first prediction time, remaining useful life and health stages are derived in the pipeline (D191).
Quirks are recorded, not fixed (D224 item 9).

Sources:
- Introduction_to_XJTU-SY_Bearing_Dataset.pdf (in the dataset folder): Tables 1 and 2, Sections 2.3, 2.4.
- Lei et al. (2019), J. Mech. Eng. 55(16), doi 10.3901/JME.2019.16.001 (in "Important References"):
  Table 3 (L10), Section 1.2 (DT9837 acquisition, magnetic mounts), Section 3.2 (amplitudes in g).
- Wang, Lei, Li, Li (2020), IEEE Trans. Reliability 69(1), 401-412, doi 10.1109/TR.2018.2882682 (to be cited).
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATASET = "xjtu_sy"
STREAM = "accel"  # the two accelerometers share one acquisition clock
SAMPLE_RATE_HZ = 25600.0  # intro Section 2.4
EXPECTED_SAMPLES = 32768  # 1.28 s per sample (intro Section 2.4); other lengths are recorded as quirks
SAMPLING_PERIOD_S = 60.0  # "the sampling period is equal to 1 min" (intro Section 2.4)
# Channel names as in the CSV header (D224 item 1): first column horizontal, second vertical.
CHANNELS = ("Horizontal_vibration_signals", "Vertical_vibration_signals")
UNITS = "g"  # Lei et al. (2019) Section 3.2: amplitudes "up to 50 g"; intro Fig. 4 axes (D224 item 7)

# Condition folder -> (speed rpm, shaft frequency Hz, radial force kN), nominal (intro Section 2.3).
CONDITIONS = {
    "35Hz12kN": (2100, 35.0, 12.0),
    "37.5Hz11kN": (2250, 37.5, 11.0),
    "40Hz10kN": (2400, 40.0, 10.0),
}
# Basic rating life L10 per condition, as given in Lei et al. (2019) Table 3.
L10 = {"35Hz12kN": "5.600-9.677 h", "37.5Hz11kN": "6.786-11.726 h", "40Hz10kN": "8.468-14.632 h"}

# Bearing -> (condition, number of files, lifetime, fault element), as in intro Table 2 (verbatim).
BEARINGS = {
    "Bearing1_1": ("35Hz12kN", 123, "2 h 3 min", "Outer race"),
    "Bearing1_2": ("35Hz12kN", 161, "2 h 41 min", "Outer race"),
    "Bearing1_3": ("35Hz12kN", 158, "2 h 38 min", "Outer race"),
    "Bearing1_4": ("35Hz12kN", 122, "2 h 2 min", "Cage"),
    "Bearing1_5": ("35Hz12kN", 52, "52 min", "Inner race and outer race"),
    "Bearing2_1": ("37.5Hz11kN", 491, "8 h 11 min", "Inner race"),
    "Bearing2_2": ("37.5Hz11kN", 161, "2 h 41 min", "Outer race"),
    "Bearing2_3": ("37.5Hz11kN", 533, "8 h 53 min", "Cage"),
    "Bearing2_4": ("37.5Hz11kN", 42, "42 min", "Outer race"),
    "Bearing2_5": ("37.5Hz11kN", 339, "5 h 39 min", "Outer race"),
    "Bearing3_1": ("40Hz10kN", 2538, "42 h 18 min", "Outer race"),
    "Bearing3_2": ("40Hz10kN", 2496, "41 h 36 min", "Inner race, ball, cage and outer race"),
    "Bearing3_3": ("40Hz10kN", 371, "6 h 11 min", "Inner race"),
    "Bearing3_4": ("40Hz10kN", 1515, "25 h 15 min", "Inner race"),
    "Bearing3_5": ("40Hz10kN", 114, "1 h 54 min", "Outer race"),
}

# Run-level information kept in the runs table's extras (D224 item 6).
BEARING_TYPE = "LDK UER204"
GEOMETRY = {  # intro Table 1
    "outer_race_diameter_mm": 39.80, "inner_race_diameter_mm": 29.30, "mean_diameter_mm": 34.55,
    "ball_diameter_mm": 7.92, "n_balls": 8, "contact_angle_deg": 0.0,
    "static_load_rating_kn": 6.65, "dynamic_load_rating_kn": 12.82,
}
SENSORS = ("2 x PCB 352C33 single-axis accelerometers on magnetic mounts, at 90 degrees on the bearing "
           "housing (horizontal and vertical); DT9837 portable dynamic signal acquisition unit")
STOP_CRITERION = ("test stopped when the maximum amplitude of the horizontal or vertical signal exceeded "
                  "10 x A_h, A_h = maximum amplitude in the normal operating stage")


def run_id(bearing):
    return bearing


def _read_csv(path):
    """(header, values) of one CSV, values exactly as written (D224 item 8).

    Parsed with Python's float(), which is correctly rounded on every platform. pandas' default
    parser is not exact in every version (pandas 3.0.6 differs by up to thousands of ulp), and
    the values are written with 17 significant digits, so the parser matters for "as recorded" (D5).
    """
    with open(path) as f:
        header = f.readline().strip()
        rows = [[float(v) for v in line.split(",")] for line in f.read().splitlines()]
    return header, np.array(rows, dtype=np.float64)


def read(folder, bearing):
    """Return (tables, signals, notes) of one bearing; notes lists the quirks found."""
    condition, n_files_doc, lifetime, fault = BEARINGS[bearing]
    bdir = os.path.join(folder, condition, bearing)
    rid = run_id(bearing)

    # File numbers present; gaps and a count differing from Table 2 are recorded, not fixed.
    numbers = sorted(int(n[:-4]) for n in os.listdir(bdir) if n.endswith(".csv"))
    notes = []
    if numbers != list(range(1, len(numbers) + 1)):
        notes.append(f"file numbers not 1..{len(numbers)}: missing {sorted(set(range(1, max(numbers) + 1)) - set(numbers))}")
    if len(numbers) != n_files_doc:
        notes.append(f"{len(numbers)} files, intro Table 2 lists {n_files_doc}")
    files = [os.path.join(bdir, f"{n}.csv") for n in numbers]

    captures, telemetry, signals = [], [], []
    hashes = {}  # hash of both channels -> capture, to find exact copies within the run
    for n, path in zip(numbers, files):
        capture_id = f"m{n:04d}"
        header, x = _read_csv(path)
        if header != ",".join(CHANNELS):
            raise ValueError(f"{path}: unexpected header {header!r}")  # different layout: stop
        if x.shape[1] != len(CHANNELS):
            raise ValueError(f"{path}: {x.shape[1]} columns")
        if x.shape[0] != EXPECTED_SAMPLES:
            notes.append(f"{capture_id}: {x.shape[0]} samples")  # stored as recorded
        if np.isnan(x).any():
            notes.append(f"{capture_id}: {int(np.isnan(x).sum())} NaN values")  # stored as recorded
        h = hashlib.sha256(x.tobytes()).hexdigest()
        copy_of = hashes.get(h)
        hashes.setdefault(h, capture_id)
        if copy_of:
            notes.append(f"{capture_id}: identical to {copy_of}")

        captures.append({
            "run_id": rid, "capture_id": capture_id, "stream": STREAM,
            "source_name": f"{condition}/{bearing}/{n}.csv", "n_samples": x.shape[0],
            # nominal: files are named in sampling order, one sample per minute; no timestamps in the files
            "start_s": (n - 1) * SAMPLING_PERIOD_S,
            "timing_accuracy": "nominal: 1 min sampling period from the documentation; no timestamps in the files",
            # per-capture columns (D31, D224 item 5)
            "bearing": bearing, "condition": condition, "file_number": n,
            "fault_element": fault, "identical_to": copy_of,
        })
        rpm, shaft_hz, force_kn = CONDITIONS[condition]
        telemetry.append({"run_id": rid, "capture_id": capture_id, "rpm_nominal": float(rpm),
                          "shaft_freq_hz_nominal": shaft_hz, "radial_force_kn_nominal": force_kn})
        for i, ch in enumerate(CHANNELS):
            signals.append((capture_id, ch, np.ascontiguousarray(x[:, i]), 1.0, 0.0, UNITS))

    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7, D224 item 8)
        "bearing": bearing, "lubricant": None,
        # D24 for a run made of many files: name pattern, total size and latest modification time
        "source_name": f"{condition}/{bearing}/*.csv",
        "source_size_bytes": sum(os.path.getsize(f) for f in files),
        "source_mtime_utc": datetime.fromtimestamp(max(os.path.getmtime(f) for f in files), timezone.utc).isoformat(),
        "extras": json.dumps({
            "condition": condition,
            "lifetime_documented": lifetime,  # intro Table 2
            "l10_documented": L10[condition],  # Lei et al. (2019) Table 3
            "bearing_type": BEARING_TYPE, "geometry": GEOMETRY,
            "sensors": SENSORS, "stop_criterion": STOP_CRITERION,
            "units_source": "Lei et al. (2019) Section 3.2 ('up to 50 g'); intro Fig. 4 axes 'Amplitude (g)'",
            "conversion_notes": notes,
        }),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": ch, "sensor_type": "accelerometer",
        "sample_rate_hz": SAMPLE_RATE_HZ, "units": UNITS, "scale": 1.0, "offset": 0.0,
        "x_origin_s": np.nan, "dtype": "float64",
    } for ch in CHANNELS])
    tables = {"runs": runs, "captures": pd.DataFrame(captures),
              "channels": channels, "telemetry": pd.DataFrame(telemetry)}
    return tables, signals, notes
