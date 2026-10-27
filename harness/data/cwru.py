"""Reader for the CWRU bearing data (Case Western Reserve University Bearing Data Center).

Decisions: D116 (all files converted), D118/D120 (outer-race clock hour), D139 (conversion).
Expects the data center's .mat files, unchanged, in one folder: <folder>/<number>.mat.

One run per source file (one test: one bearing pair at one load), one capture, one stream;
channels DE, FE and BA as present in the file. Values are stored as recorded (64-bit floats,
scale 1, offset 0; D5, D6). Nothing is derived or filled in here.

The file table below is transcribed from the data center's pages (12k Drive End, 12k Fan End,
48k Drive End and Normal Baseline Bearing Data); fault depth and manufacturer come from its
Fault Specifications page, bearing models from its Bearing Information page.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io

DATASET = "cwru"
STREAM = "accel"
CHANNELS = ("DE", "FE", "BA")  # drive end, fan end, base plate; stored as present in each file
TABLE_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}  # approximate speed per load (hp), data center tables

# One row per fault: (data center table, sample rate in Hz, fault end, fault location,
# fault size in mil, outer-race clock hour or None, files at loads 0, 1, 2, 3 hp; None = not available).
FAULT_FILES = [
    ("12k Drive End", 12000, "DE", "inner", 7, None, [105, 106, 107, 108]),
    ("12k Drive End", 12000, "DE", "inner", 14, None, [169, 170, 171, 172]),
    ("12k Drive End", 12000, "DE", "inner", 21, None, [209, 210, 211, 212]),
    ("12k Drive End", 12000, "DE", "inner", 28, None, [3001, 3002, 3003, 3004]),
    ("12k Drive End", 12000, "DE", "ball", 7, None, [118, 119, 120, 121]),
    ("12k Drive End", 12000, "DE", "ball", 14, None, [185, 186, 187, 188]),
    ("12k Drive End", 12000, "DE", "ball", 21, None, [222, 223, 224, 225]),
    ("12k Drive End", 12000, "DE", "ball", 28, None, [3005, 3006, 3007, 3008]),
    ("12k Drive End", 12000, "DE", "outer", 7, 6, [130, 131, 132, 133]),
    ("12k Drive End", 12000, "DE", "outer", 14, 6, [197, 198, 199, 200]),
    ("12k Drive End", 12000, "DE", "outer", 21, 6, [234, 235, 236, 237]),
    ("12k Drive End", 12000, "DE", "outer", 7, 3, [144, 145, 146, 147]),
    ("12k Drive End", 12000, "DE", "outer", 21, 3, [246, 247, 248, 249]),
    ("12k Drive End", 12000, "DE", "outer", 7, 12, [156, 158, 159, 160]),
    ("12k Drive End", 12000, "DE", "outer", 21, 12, [258, 259, 260, 261]),
    ("12k Fan End", 12000, "FE", "inner", 7, None, [278, 279, 280, 281]),
    ("12k Fan End", 12000, "FE", "inner", 14, None, [274, 275, 276, 277]),
    ("12k Fan End", 12000, "FE", "inner", 21, None, [270, 271, 272, 273]),
    ("12k Fan End", 12000, "FE", "ball", 7, None, [282, 283, 284, 285]),
    ("12k Fan End", 12000, "FE", "ball", 14, None, [286, 287, 288, 289]),
    ("12k Fan End", 12000, "FE", "ball", 21, None, [290, 291, 292, 293]),
    ("12k Fan End", 12000, "FE", "outer", 7, 6, [294, 295, 296, 297]),
    ("12k Fan End", 12000, "FE", "outer", 14, 6, [313, None, None, None]),
    ("12k Fan End", 12000, "FE", "outer", 21, 6, [315, None, None, None]),
    ("12k Fan End", 12000, "FE", "outer", 7, 3, [298, 299, 300, 301]),
    ("12k Fan End", 12000, "FE", "outer", 14, 3, [310, 309, 311, 312]),
    ("12k Fan End", 12000, "FE", "outer", 21, 3, [None, 316, 317, 318]),
    ("12k Fan End", 12000, "FE", "outer", 7, 12, [302, 305, 306, 307]),
    ("48k Drive End", 48000, "DE", "inner", 7, None, [109, 110, 111, 112]),
    ("48k Drive End", 48000, "DE", "inner", 14, None, [174, 175, 176, 177]),
    ("48k Drive End", 48000, "DE", "inner", 21, None, [213, 214, 215, 217]),  # 217 as published (not 216)
    ("48k Drive End", 48000, "DE", "ball", 7, None, [122, 123, 124, 125]),
    ("48k Drive End", 48000, "DE", "ball", 14, None, [189, 190, 191, 192]),
    ("48k Drive End", 48000, "DE", "ball", 21, None, [226, 227, 228, 229]),
    ("48k Drive End", 48000, "DE", "outer", 7, 6, [135, 136, 137, 138]),
    ("48k Drive End", 48000, "DE", "outer", 14, 6, [201, 202, 203, 204]),
    ("48k Drive End", 48000, "DE", "outer", 21, 6, [238, 239, 240, 241]),
    ("48k Drive End", 48000, "DE", "outer", 7, 3, [148, 149, 150, 151]),
    ("48k Drive End", 48000, "DE", "outer", 21, 3, [250, 251, 252, 253]),
    ("48k Drive End", 48000, "DE", "outer", 7, 12, [161, 162, 163, 164]),
    ("48k Drive End", 48000, "DE", "outer", 21, 12, [262, 263, 264, 265]),
]
NORMAL_FILES = [97, 98, 99, 100]  # loads 0, 1, 2, 3 hp
NORMAL_RATE = 48000  # not stated by the data center; from eda/cwru/check_normal_fs.py (D126, D139 item 4)

# Source-file quirks (D139 item 6).
# Files whose variables carry another number than the file: the 28 mil files (one drive-end
# variable each) and 174.mat (holds X173_*, only 63,788 samples). That variable set is converted
# and its original name recorded (D139 item 6, D140).
VARIABLE_NUMBER = {3001: 56, 3002: 57, 3003: 58, 3004: 59, 3005: 48, 3006: 49, 3007: 50, 3008: 51, 174: 173}
# Variables that are not converted (D139 item 6, D140): copies of other files' data and
# leftovers from the MATLAB session ('ans', 'i').
IGNORED = {
    99: {"X098_DE_time", "X098_FE_time", "ans"},  # file 98's data, byte-identical to 98.mat
    175: {"X217", "X217_DE_time"},  # rescaled copies of file 217's channels (correlation 1.000, about 0.2x amplitude)
    217: {"X215_DE_time", "X215_FE_time", "X215RPM"},  # file 215's data, byte-identical to 215.mat
    277: {"i"},
    278: {"i"},
}
# Notes stored in the run's extras.
NOTES = {
    99: "also holds file 98's variables (byte-identical to 98.mat); not converted",
    174: "holds X173_* instead of X174_*; only 63,788 samples (1.3 s) versus about 10 s for its neighbours; "
         "may not be the intended recording",
    175: "own recording 381,890 samples (8 s); also holds X217 and X217_DE_time, rescaled copies of "
         "file 217's channels (not converted)",
    217: "also holds file 215's variables (byte-identical to 215.mat); not converted",
}


def fault_spec(end, location, size):
    """Fault depth (inches) and bearing manufacturer, data center's Fault Specifications page."""
    if size in (7, 14, 21):
        return 0.011, "SKF"
    if (end, location, size) == ("DE", "inner", 28):
        return 0.050, "NTN"
    if (end, location, size) == ("DE", "ball", 28):
        return 0.150, "NTN"
    raise ValueError(f"no fault specification for {end} {location} {size} mil")


def bearing_id(end, location, size, clock):
    """ID of a faulty bearing, e.g. DE_IR_007, FE_B_014, DE_OR6_021 (outer race with clock hour).

    12 and 48 kHz drive-end files of the same fault get the same ID: the data center gives one
    specification per fault, and the same ID keeps them on the same side of a split by bearing.
    """
    code = {"inner": "IR", "ball": "B", "outer": f"OR{clock}"}[location]
    return f"{end}_{code}_{size:03d}"


def file_table():
    """One dict per source file: number, table, sample rate, load and fault information."""
    rows = []
    for load, n in enumerate(NORMAL_FILES):
        rows.append({"file": n, "table": "Normal Baseline", "sample_rate_hz": NORMAL_RATE, "load_hp": load,
                     "fault_end": "none", "fault_location": "none", "fault_size_mil": np.nan,
                     "clock_hour": np.nan, "fault_depth_in": np.nan, "manufacturer": None,
                     "de_bearing": "DE_healthy", "fe_bearing": "FE_healthy"})
    for table, rate, end, location, size, clock, files in FAULT_FILES:
        depth, maker = fault_spec(end, location, size)
        faulty = bearing_id(end, location, size, clock)
        for load, n in enumerate(files):
            if n is None:
                continue
            rows.append({"file": n, "table": table, "sample_rate_hz": rate, "load_hp": load,
                         "fault_end": end, "fault_location": location, "fault_size_mil": size,
                         "clock_hour": np.nan if clock is None else clock,
                         "fault_depth_in": depth, "manufacturer": maker,
                         # the bearing at the other end is the healthy one of that end
                         "de_bearing": faulty if end == "DE" else "DE_healthy",
                         "fe_bearing": faulty if end == "FE" else "FE_healthy"})
    numbers = [r["file"] for r in rows]
    assert len(numbers) == len(set(numbers)), "a file number appears twice in the table"
    return rows


def run_id(n):
    """Run ID of a source file, zero-padded so the IDs sort correctly."""
    return f"{n:04d}"


def read(folder, entry):
    """Return (tables, signals) of one source file for the canonical writer.

    entry: one row of file_table(). Stops with an error if the file holds a variable
    that is neither converted nor listed in IGNORED, so nothing is skipped unnoticed.
    """
    n = entry["file"]
    path = os.path.join(folder, f"{n}.mat")
    mat = scipy.io.loadmat(path)
    present = {k for k in mat if not k.startswith("__")}
    prefix = f"X{VARIABLE_NUMBER.get(n, n):03d}"

    # Channels as present in the file, each a column vector in the source.
    source_vars = {ch: f"{prefix}_{ch}_time" for ch in CHANNELS if f"{prefix}_{ch}_time" in present}
    values = {ch: mat[var].ravel() for ch, var in source_vars.items()}
    lengths = {len(v) for v in values.values()}
    if len(lengths) != 1:
        raise ValueError(f"{n}.mat: channels differ in length {lengths}")
    rpm_var = f"{prefix}RPM" if f"{prefix}RPM" in present else None

    unexpected = present - set(source_vars.values()) - {rpm_var} - IGNORED.get(n, set())
    if unexpected:
        raise ValueError(f"{n}.mat: unexpected variables {sorted(unexpected)}; add them to IGNORED or convert them")

    rid, capture_id = run_id(n), "capture_0"
    stat = os.stat(path)
    runs = pd.DataFrame([{
        "dataset": DATASET,
        "run_id": rid,
        "status": "final",  # public benchmark (D7, D139 item 7)
        "bearing": None,  # two bearings per run: see de_bearing / fe_bearing on the captures table
        "lubricant": None,
        "source_name": os.path.basename(path),  # D24: name, size and modification time
        "source_size_bytes": stat.st_size,
        "source_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "extras": json.dumps({
            "data_center_table": entry["table"],
            "source_variables": source_vars,
            "rpm_variable": rpm_var,
            "ignored_variables": sorted(IGNORED.get(n, set())),
            "note": NOTES.get(n),
            "sample_rate_source": ("eda/cwru/check_normal_fs.py; not stated by the data center"
                                   if entry["table"] == "Normal Baseline" else "data center file table"),
            "units_note": "not stated by the data center; the literature treats the values as g (D139 item 5)",
            "bearing_models": {"DE": "6205-2RS JEM SKF", "FE": "6203-2RS JEM SKF",
                               "note": "data center Bearing Information page; 28 and 40 mil faults in NTN equivalent bearings"},
            "sensor_positions": {"DE": "12 o'clock, drive end of the motor housing",
                                 "FE": "12 o'clock, fan end of the motor housing",
                                 "BA": "motor supporting base plate (some runs only)"},
        }),
    }])
    captures = pd.DataFrame([{
        "run_id": rid,
        "capture_id": capture_id,
        "stream": STREAM,
        "source_name": os.path.basename(path),
        "start_s": np.nan,  # no time information in the files
        "n_samples": lengths.pop(),
        "timing_accuracy": "not applicable",
        # per-capture columns (D31, D139 item 2)
        "source_file": n,
        "load_hp": entry["load_hp"],
        "fault_end": entry["fault_end"],
        "fault_location": entry["fault_location"],
        "fault_size_mil": entry["fault_size_mil"],
        "clock_hour": entry["clock_hour"],
        "fault_depth_in": entry["fault_depth_in"],
        "manufacturer": entry["manufacturer"],
        "de_bearing": entry["de_bearing"],
        "fe_bearing": entry["fe_bearing"],
    }])
    channels = pd.DataFrame([{
        "run_id": rid,
        "stream": STREAM,
        "channel": ch,
        "sensor_type": "accelerometer",
        "sample_rate_hz": float(entry["sample_rate_hz"]),
        "units": "unknown",  # D139 item 5
        "scale": 1.0,
        "offset": 0.0,
        "x_origin_s": np.nan,
        "dtype": "float64",
    } for ch in source_vars])
    telemetry = pd.DataFrame([{
        "run_id": rid,
        "capture_id": capture_id,
        "rpm": float(mat[rpm_var].ravel()[0]) if rpm_var else np.nan,  # as recorded; missing stays missing (D139 item 3)
        "rpm_table": float(TABLE_RPM[entry["load_hp"]]),  # data center's approximate speed for this load
    }])
    # Values are floats in the source, so they are stored as floats (D6): scale 1, offset 0.
    signals = [(capture_id, ch, v, 1.0, 0.0, "unknown") for ch, v in values.items()]
    tables = {"runs": runs, "captures": captures, "channels": channels, "telemetry": telemetry}
    return tables, signals
