"""Reader for the Paderborn run-to-failure bearing data under time-varying load and speed
(Chair of Dynamics and Mechatronics (LDM), Paderborn University; Aimiyekagbon, 2024).

Decisions: D234 (adopted), D235 (download and storage), D236 (conversion).
Reads the Zenodo ZIP archives directly (record 10868257): <experiment>.zip, holding
<experiment>/vibrationData/data_<experiment>_M<n>.mat and <experiment>/<experiment>_log.pdf etc.,
or <experiment>_part<k>.zip, whose top folder is <experiment>_part<k>/ (e.g. B06_part1/
... B06_part7/; the CSV and log files are in one of the parts). Member names are normalised
to <experiment>/... here; the captures table keeps the archive's own member name.

One run per experiment (one bearing run to failure), each .mat file (one 1.6 s measurement)
one capture, one stream (accelerometers A and C share the acquisition clock), two channels.
The signals are stored as int16 ADC codes with a per-channel step (scale) and offset (D6, D20):
the values in the files lie on the fixed grid of the 16-bit NI-9223. The time array `measTime`
is not stored; it equals sample index / sample rate (checked per file). Telemetry (operating
conditions, temperatures) is stored per capture with the source's column names. Nothing is
derived here: transition time, remaining useful life and conversion to g are pipeline steps.

Sources:
- LDM_UPB_Run-to-failure_data_set_of_ball_bearings.pdf (v2, 2024-03-25; "description"):
  Table 1 (bearing), Table 2 (operating-condition ranges), Table 3 (files, threshold, defects),
  Section 2 (sensors, sampling), Section 2.2 (file layout and variables).
- <experiment>_log.pdf in each experiment folder: conversion factor (g/V), gain, log.
- Javanmardi et al. (2024), PHM Society European Conference 8(1), doi 10.36001/phme.2024.v8i1.4101
  (to be cited).
"""

import hashlib
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io

from harness.data import quantized

DATASET = "paderborn_rtf"
STREAM = "accel"
CHANNELS = ("accHorizRear_A", "accHorizFrontal_C")  # variable names in the .mat files (description 2.2)
UNITS = "V"  # "The signals are given in Voltage (V)" (description 2.2); g/V factor in each log
DURATION_S = 1.6  # "sampling duration of 1.6 s" (description Section 2)
ZENODO = "https://doi.org/10.5281/zenodo.10868257 (record 10868257, version of 2024-04-02)"

EXPERIMENTS = [f"B{i:02d}" for i in range(1, 18)]


def sample_rate_hz(experiment):
    # "sampled at 128 kHz for experiments till B09 and ... at 64 kHz for experiments from B010"
    return 128000.0 if int(experiment[1:]) <= 9 else 64000.0


# Description Table 3, verbatim: (approx. duration, number of files, threshold criterion, defect).
TABLE3 = {
    "B01": ("2h", 377, "-", "IR, OR, B"), "B02": ("6h", 1116, "Tvibration", "IR"),
    "B03": ("3h", 614, "Ttemperature", "IR"), "B04": ("6h", 1114, "Tvibration", "IR, B"),
    "B05": ("3h", 572, "-", "B"), "B06": ("67 h", 12965, "Tvibration", "IR, B"),
    "B07": ("37 h", 6393, "Ttemperature", "IR"), "B08": ("11 h", 1827, "Ttemperature", "IR, B"),
    "B09": ("13 h", 1813, "Tvibration", "OR, B"), "B10": ("19 h", 3224, "Tvibration", "OR, B"),
    "B11": ("8h", 1953, "Tvibration", "IR, OR, B"), "B12": ("3h", 767, "Tvibration", "IR, B"),
    "B13": ("66 h", 19417, "Tvibration", "IR, B"), "B14": ("41 h", 12317, "Tvibration", "B"),
    "B15": ("76 h", 22567, "Ttemperature", "IR, B"), "B16": ("20 h", 5891, "-", "IR, OR, B"),
    "B17": ("17 h", 2733, "Tvibration", "B"),
}
# Description Table 2, verbatim: (speed rpm, static load N, dynamic load amplitude N, excitation).
_T2 = {
    ("B01", "B02", "B03"): ("[2400, 3000]", "[3300, 3800]", "[500, 1400]", "Sinusoidal"),
    ("B04", "B05"): ("[1500, 3000]", "[2500, 3800]", "[500, 1500]", "Sinusoidal"),
    ("B06",): ("[1500, 3600]", "[3300, 3800]", "[750, 1700]", "Sinusoidal"),
    ("B07",): ("[1500, 3000]", "[3250, 4000]", "[250, 750]", "Gaussian noise"),
    ("B08",): ("[1500, 3000]", "[3250, 4000]", "[500, 1000]", "Gaussian noise"),
    ("B09",): ("[1500, 3000]", "[2500, 3800]", "[750, 1000]", "Gaussian noise"),
    ("B10",): ("[1500, 2700]", "[2000, 3250]", "[750, 1500]", "Gaussian noise"),
    ("B11", "B12", "B13"): ("[1500, 2700]", "3000", "1000", "Gaussian noise"),
    ("B14", "B15"): ("2700", "2500", "[700, 1500]", "Gaussian noise"),
    ("B16",): ("2700", "2500", "[1000, 1500]", "Gaussian noise"),
    ("B17",): ("2700", "[1500, 2500]", "1500", "Gaussian noise"),
}
TABLE2 = {e: v for group, v in _T2.items() for e in group}

BEARING = {  # description Table 1
    "type": "61806-2RS", "inner_diameter_mm": 30, "outer_diameter_mm": 42, "width_mm": 7,
    "rolling_circle_diameter_mm": 35.5, "rolling_element_diameter_mm": 3.18, "n_balls": 19,
    "seal": "NBR+08", "static_load_rating_kn": 3.15, "dynamic_load_rating_kn": 4.00,
}
SENSORS = ("PCB 603C01 accelerometers (100 mV/g, +-50 g, 0.5-10,000 Hz at 3 dB, resonance 25 kHz) on the "
           "test-bearing housing, A horizontal rear and C horizontal frontal (B, on the lever, not provided); "
           "MMF M208B conditioner, NI-9223, cDAQ-9189. Pt100 (MC 745691-02) at T1, T2 and ambient, 10 Hz, "
           "given as means per measurement.")
STOP_CRITERION = ("experiment ends when T_vibration (within [6, 10] g) or T_temperature (110 degC) is exceeded; "
                  "no criterion pre-set for B01, B05 interrupted deliberately (description Section 1)")

MAX_ULP = 2  # reconstructed values (code * scale + offset) must lie within 2 ulp of the file's largest |value|
_MAT = re.compile(r"^(B\d\d)/vibrationData/data_\1_M(\d+)\.mat$")  # on normalised names
_TOP = re.compile(r"^(B\d\d)(_part\d+)?/")  # top folder: B06/ or B06_part1/
_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def run_id(experiment):
    return experiment


def zip_parts(folder, experiment):
    """The experiment's ZIP files in folder: [<exp>.zip] or [<exp>_part1.zip, ...], in part order."""
    single = os.path.join(folder, f"{experiment}.zip")
    if os.path.exists(single):
        return [single]
    parts = sorted((f for f in os.listdir(folder) if re.fullmatch(rf"{experiment}_part\d+\.zip", f)),
                   key=lambda f: int(f.split("_part")[1][:-4]))
    return [os.path.join(folder, f) for f in parts]


def normalise(name):
    """'B06_part1/vibrationData/x.mat' -> 'B06/vibrationData/x.mat'; 'B01/...' unchanged."""
    return _TOP.sub(lambda m: m.group(1) + "/", name, count=1)


def open_members(paths):
    """Map normalised member name -> (open ZipFile, member name in that archive), over all parts.

    A normalised name found twice (in two parts) stops the conversion.
    """
    members = {}
    for p in paths:
        z = zipfile.ZipFile(p)
        for name in z.namelist():
            if name.endswith("/"):  # folder entries (each part has its own vibrationData/)
                continue
            key = normalise(name)
            if key in members:
                raise ValueError(f"{name} appears more than once across the parts")
            members[key] = (z, name)
    return members


def _parse_time(s):
    """'19-Oct-2021 08:46:54' -> datetime (month names parsed by hand, independent of the locale)."""
    day, mon, rest = s.split("-")
    year, hms = rest.split(" ")
    h, m, sec = (int(v) for v in hms.split(":"))
    return datetime(int(year), _MONTHS[mon], int(day), h, m, sec)


def _read_csv(members, name):
    """(header, rows) of one CSV; numbers parsed with Python's float() (correctly rounded, D224 note)."""
    z, member = members[name]
    text = z.read(member).decode("utf-8")
    lines = text.splitlines()
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:] if line]
    return header, rows


def _load_mat(members, name):
    z, member = members[name]
    return scipy.io.loadmat(io.BytesIO(z.read(member)))


def to_codes(x, step, offset, where):
    """int16 codes of x on the given grid (harness/data/quantized.py, D236 item 3)."""
    return quantized.to_codes(x, step, offset, where, np.int16, MAX_ULP)


def read_tables(paths, experiment, members):
    """The four tables of one experiment, from the CSV files and the first measurement file."""
    rid = run_id(experiment)
    fs = sample_rate_hz(experiment)
    n_samples = int(round(DURATION_S * fs))
    notes = []

    # Measurement files present; gaps and a count differing from Table 3 are recorded, not fixed.
    mats = {}
    for name in members:
        m = _MAT.match(name)
        if m and m.group(1) == experiment:
            mats[int(m.group(2))] = name
    numbers = sorted(mats)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{experiment}: measurement numbers not 1..{len(numbers)}")  # rows could not be matched
    if len(numbers) != TABLE3[experiment][1]:
        notes.append(f"{len(numbers)} measurement files, description Table 3 lists {TABLE3[experiment][1]}")
    other = sorted(n for n in members if not _MAT.match(n))
    expected_other = {f"{experiment}/{experiment}_{s}" for s in ("log.pdf", "meanTemperatures.csv", "operatingConditions.csv")}
    if not expected_other <= set(other):
        raise ValueError(f"{experiment}: missing {sorted(expected_other - set(other))}")
    if set(other) - expected_other:
        notes.append(f"other files: {sorted(set(other) - expected_other)}")

    # Telemetry: one row per measurement in both CSVs, matched by order and checked by timestamp.
    oc_head, oc = _read_csv(members, f"{experiment}/{experiment}_operatingConditions.csv")
    t_head, temp = _read_csv(members, f"{experiment}/{experiment}_meanTemperatures.csv")
    if len(oc) != len(numbers) or len(temp) != len(numbers):
        raise ValueError(f"{experiment}: {len(numbers)} files, {len(oc)} and {len(temp)} CSV rows")
    if [r[0] for r in oc] != [r[0] for r in temp]:
        raise ValueError(f"{experiment}: timestamps differ between the two CSV files")
    times = [_parse_time(r[0]) for r in oc]
    if any(b < a for a, b in zip(times, times[1:])):
        notes.append("timestamps not in increasing order")

    captures, telemetry = [], []
    for n, oc_row, t_row, t in zip(numbers, oc, temp, times):
        capture_id = f"m{n:05d}"
        captures.append({
            "run_id": rid, "capture_id": capture_id, "stream": STREAM,
            "source_name": members[mats[n]][1], "n_samples": n_samples,  # name in the archive
            "start_s": (t - times[0]).total_seconds(),
            "timing_accuracy": "1 s: dateTime in the operating-conditions CSV (local time of the rig)",
            # per-capture columns (D31)
            "experiment": experiment, "measurement": n, "date_time": oc_row[0],
            "defect": TABLE3[experiment][3], "threshold_criterion": TABLE3[experiment][2],
        })
        row = {"run_id": rid, "capture_id": capture_id}
        row.update({h: float(v) for h, v in zip(oc_head[1:], oc_row[1:])})
        row.update({h: float(v) for h, v in zip(t_head[1:], t_row[1:])})
        telemetry.append(row)

    # Grid constants per channel from the first measurement; every file is checked against them.
    first = _load_mat(members, mats[numbers[0]])
    grid = {ch: quantized.grid_constants(first[ch].ravel()) for ch in CHANNELS}
    for ch in CHANNELS:  # the chosen constants must pass the checks on the file they came from
        to_codes(first[ch].ravel(), *grid[ch], f"{mats[numbers[0]]}/{ch}")

    speed, stat, dyn, excitation = TABLE2[experiment]
    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7)
        "bearing": experiment, "lubricant": None,
        # D24 for a run made of archives: archive names, total size, latest modification time
        "source_name": ", ".join(os.path.basename(p) for p in paths),
        "source_size_bytes": sum(os.path.getsize(p) for p in paths),
        "source_mtime_utc": datetime.fromtimestamp(max(os.path.getmtime(p) for p in paths), timezone.utc).isoformat(),
        "extras": json.dumps({
            "source": ZENODO,
            "duration_documented": TABLE3[experiment][0], "n_files_documented": TABLE3[experiment][1],
            "threshold_criterion": TABLE3[experiment][2], "defect": TABLE3[experiment][3],
            "speed_rpm_range": speed, "static_load_n_range": stat,
            "dynamic_load_amplitude_n_range": dyn, "dynamic_load_excitation": excitation,
            "bearing": BEARING, "sensors": SENSORS, "stop_criterion": STOP_CRITERION,
            "units_note": f"volts as recorded; g/V conversion factor and gain in {experiment}/{experiment}_log.pdf "
                          "(kept in data/raw/paderborn_rtf/, D235)",
            "measTime": "not stored: checked equal to sample index / sample rate within 1e-9 s in every file",
            "conversion_notes": notes,
        }),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": ch, "sensor_type": "accelerometer",
        "sample_rate_hz": fs, "units": UNITS, "scale": grid[ch][0], "offset": grid[ch][1],
        "x_origin_s": 0.0, "dtype": "int16",
    } for ch in CHANNELS])
    tables = {"runs": runs, "captures": pd.DataFrame(captures),
              "channels": channels, "telemetry": pd.DataFrame(telemetry)}
    return tables, notes


def signals(members, tables, hashes):
    """Yield (capture_id, channel, codes, scale, offset, units) for every measurement and channel.

    Every file is checked: variables, sample count, time array, ADC grid and reconstruction
    (anything unexpected stops the conversion). hashes[(capture_id, channel)] receives the SHA-256
    of the codes, so the written file can be checked afterwards.
    """
    ch_table = tables["channels"].set_index("channel")
    caps = tables["captures"]
    by_member = {member: key for key, (_, member) in members.items()}  # archive name -> normalised
    fs = ch_table.sample_rate_hz.iloc[0]
    for i, row in enumerate(caps.itertuples()):
        d = _load_mat(members, by_member[row.source_name])
        if set(k for k in d if not k.startswith("__")) != set(CHANNELS) | {"measTime"}:
            raise ValueError(f"{row.source_name}: unexpected variables {sorted(d)}")
        t = d["measTime"].ravel()
        if len(t) != row.n_samples:
            raise ValueError(f"{row.source_name}: {len(t)} samples, expected {row.n_samples}")
        if np.max(np.abs(t - np.arange(len(t)) / fs)) > 1e-9:
            raise ValueError(f"{row.source_name}: measTime is not sample index / {fs} Hz")
        for ch in CHANNELS:
            x = d[ch].ravel()
            if len(x) != row.n_samples:
                raise ValueError(f"{row.source_name}/{ch}: {len(x)} samples")
            c = ch_table.loc[ch]
            codes = to_codes(x, c.scale, c.offset, f"{row.source_name}/{ch}")
            hashes[(row.capture_id, ch)] = hashlib.sha256(codes.tobytes()).hexdigest()
            yield row.capture_id, ch, codes, c.scale, c.offset, c.units
        if i % 1000 == 0:
            print(f"  {i} / {len(caps)} measurements", flush=True)
