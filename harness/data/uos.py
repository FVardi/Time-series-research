"""Reader for the University of Seoul multi-domain vibration dataset with various bearing types under
compound machine fault scenarios (UOS; Data in Brief 57 (2024) 110940, doi 10.1016/j.dib.2024.110940;
Mendeley Data 53vtnjy6c6, 7trwzz77xh, 2cygy6y4rk for subsets 1-3).

Decisions: D189, D190 (adopted), D194 (download), D248 (conversion).
Expects the three subset folders as downloaded under one folder (data/raw/uos/), each with
BearingType_<type>/SamplingRate_<Hz>/RotatingSpeed_<rpm>/<file>.mat. File name
<machine>_<bearing>_<kHz>_<model>_<rpm>.mat, e.g. H_IR_8_6204_1000.mat (data paper):
machine (rotating component) H, L, M1-M3, U1-U3; bearing H, B, IR, OR; model 6204, N204/NJ204, 30204.

One run per file (one recording at one speed and sampling rate, as HUST D223), one capture, one stream,
one channel: the raw vibration `Data` (g). The authors' STFT products (`Spectrogram`, `STFTFreq`,
`STFTTime`) are derived and not converted (D5). The values lie on a fixed grid (digital accelerometer),
so they are stored as int32 codes with a per-file step and offset (D6; harness/data/quantized.py);
every file is checked. Nothing is derived here; quirks are recorded in the run's conversion notes.
"""

import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io

from harness.data import quantized

DATASET = "uos"
STREAM = "accel"
CHANNEL = "Data"  # the variable name in the files (data paper)
UNITS = "g"  # data paper: raw vibration in g
EXPECTED_SAMPLES = 1280000  # 160 s at 8 kHz, 80 s at 16 kHz (data paper)
EXPECTED_VARS = {"Data", "Spectrogram", "STFTFreq", "STFTTime"}  # data paper; only Data is converted
MAX_ULP = 8  # reconstruction within 8 ulp of the file's largest |value| (B01-style check, D248; observed up to 4)

MACHINE = {  # rotating-component condition (data paper)
    "H": "healthy", "L": "looseness (motor-end housing screw loosened half a turn)",
    "M1": "misalignment 0.6 mm", "M2": "misalignment 0.8 mm", "M3": "misalignment 1.0 mm",
    "U1": "unbalance 3 g", "U2": "unbalance 4 g", "U3": "unbalance 5 g",
}
BEARING_CONDITION = {"H": "healthy", "B": "ball", "IR": "inner raceway", "OR": "outer raceway"}  # ground defects
# Bearing model -> (bearing type, pitch diameter mm, roller diameter mm, contact angle deg, rolling elements), Table 2
MODELS = {
    "6204": ("deep groove ball", 34.57, 7.94, 0.0, 8),
    "N204": ("cylindrical roller", 34.0, 7.50, 0.0, 11),  # B and OR faults
    "NJ204": ("cylindrical roller", 34.0, 7.50, 0.0, 11),  # IR faults (separable raceway, data paper)
    "30204": ("tapered roller", 35.8, 6.20, 12.6, 15),
}
SPEEDS_RPM = (600, 800, 1000, 1200, 1400, 1600)
RATES_HZ = (8000, 16000)
SENSOR = ("PCB Piezotronics 333D01 USB digital accelerometer, uniaxial, +-20 g, magnetic stud on top of the "
          "shaft-end bearing housing; AST fault simulator, 40 W BLDC motor, two rotor disks (data paper)")

_NAME = re.compile(r"^(H|L|M[123]|U[123])_(H|B|IR|OR)_(8|16)_(6204|N204|NJ204|30204)_(\d+)$")


def parse(name):
    """File name (without .mat) -> (machine, bearing condition, rate Hz, model, rpm); stops if unexpected."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"unexpected file name {name}")
    machine, bearing, khz, model, rpm = m.groups()
    return machine, bearing, int(khz) * 1000, model, int(rpm)


def expected_files():
    """All 1,152 file names: 8 machine x 4 bearing conditions x 2 rates x 6 speeds per subset."""
    names = []
    for subset_models in (("6204",), ("N204", "NJ204"), ("30204",)):
        for machine in MACHINE:
            for bearing in BEARING_CONDITION:
                model = subset_models[0] if len(subset_models) == 1 else ("NJ204" if bearing == "IR" else "N204")
                for rate in RATES_HZ:
                    for rpm in SPEEDS_RPM:
                        names.append(f"{machine}_{bearing}_{rate // 1000}_{model}_{rpm}")
    return names


def find_files(folder):
    """{file name: path} of every .mat file below folder; stops on a name found twice."""
    found = {}
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".mat"):
                name = f[:-4]
                if name in found:
                    raise ValueError(f"{name} found twice")
                found[name] = os.path.join(dirpath, f)
    return found


def run_id(name):
    return name  # the file name, e.g. "H_IR_8_6204_1000"


def read(folder, path, name):
    """Return (tables, signals, notes) of one file (path below folder); notes lists the quirks found."""
    machine, bearing, rate, model, rpm = parse(name)
    # The folders repeat the rate and speed: check they agree with the file name.
    parts = path.replace("\\", "/").split("/")
    if f"SamplingRate_{rate}" not in parts or f"RotatingSpeed_{rpm}" not in parts:
        raise ValueError(f"{path}: folder does not match the file name")
    mat = scipy.io.loadmat(path)
    variables = {k for k in mat if not k.startswith("__")}
    if variables != EXPECTED_VARS:
        raise ValueError(f"{name}.mat: variables {sorted(variables)}")
    x = mat["Data"].ravel()  # float64, g
    notes = []
    if len(x) != EXPECTED_SAMPLES:
        notes.append(f"Data has {len(x)} samples, not {EXPECTED_SAMPLES}")  # stored as recorded
    if np.isnan(x).any():
        raise ValueError(f"{name}.mat: NaN values")  # the grid check cannot handle them: stop
    step, offset = quantized.grid_constants(x)
    try:
        values = quantized.to_codes(x, step, offset, name, np.int32, MAX_ULP)
        dtype, scale, offset_stored = "int32", step, offset
        tiny = np.flatnonzero((x != 0) & (np.abs(x) < step / 2))  # e.g. -6.4e-302 in U1_IR_8_6204_600
        if len(tiny):
            notes.append(f"{len(tiny)} non-zero values smaller than half a step stored as code 0 (= 0.0): "
                         f"samples {tiny[:5].tolist()}, values {x[tiny[:5]].tolist()}")
    except ValueError:
        # Not every value on the grid (e.g. one sample 0.05 steps off in L_OR_8_30204_1400): the file is
        # stored as recorded, 64-bit floats (D6: codes only for exactly quantized data; D248 note).
        values, dtype, scale, offset_stored = x, "float64", 1.0, 0.0
        k = (x - offset) / step
        off = np.flatnonzero(np.abs(k - np.rint(k)) > quantized.GRID_TOLERANCE_STEPS)
        notes.append(f"stored as float64: {len(off)} values off the ADC grid (step {step!r} g): "
                     f"samples {off[:5].tolist()}, values {x[off[:5]].tolist()}")

    btype, pitch, roller, angle, n_el = MODELS[model]
    rid = run_id(name)
    capture = {
        "run_id": rid, "capture_id": "capture_0", "stream": STREAM,
        "source_name": os.path.relpath(path, folder).replace("\\", "/"),  # path below data/raw/uos/
        "start_s": np.nan, "n_samples": len(x), "timing_accuracy": "not applicable",
        # per-capture columns (D31), from the file name and the data paper
        "machine_condition": machine, "bearing_condition": bearing,
        "bearing_model": model, "bearing_type": btype, "speed_rpm": rpm,
    }
    telemetry = {"run_id": rid, "capture_id": "capture_0", "speed_rpm_nominal": float(rpm)}  # set speed (file name)
    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7)
        "bearing": f"{model}_{bearing}", "lubricant": None,
        "source_name": f"{name}.mat",  # D24 (path below data/raw/uos/ in the captures table)
        "source_size_bytes": os.path.getsize(path),
        "source_mtime_utc": datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(),
        "extras": json.dumps({
            "machine_condition": MACHINE[machine], "bearing_condition": BEARING_CONDITION[bearing],
            "bearing_geometry": {"pitch_diameter_mm": pitch, "roller_diameter_mm": roller,
                                 "contact_angle_deg": angle, "rolling_elements": n_el},
            "sensor": SENSOR,
            "not_converted": sorted(EXPECTED_VARS - {"Data"}),  # the authors' STFT products (D5)
            "conversion_notes": notes,
        }),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": CHANNEL, "sensor_type": "accelerometer",
        "sample_rate_hz": float(rate), "units": UNITS, "scale": scale, "offset": offset_stored,
        "x_origin_s": np.nan, "dtype": dtype,
    }])
    tables = {"runs": runs, "captures": pd.DataFrame([capture]),
              "channels": channels, "telemetry": pd.DataFrame([telemetry])}
    return tables, [("capture_0", CHANNEL, values, scale, offset_stored, UNITS)], notes
