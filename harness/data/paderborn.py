"""Reader for the Paderborn bearing data (KAt DataCenter, Paderborn University; Lessmeier et al., 2016).

Decisions: D47 (download), D48 (signals), D212 (conversion), D213 (unpacked archives).
Expects the unpacked archives in one folder: <folder>/<bearing>/<condition>_<bearing>_<n>.mat
(e.g. data/raw/paderborn/K001/N15_M07_F10_K001_1.mat) for 4 conditions and n = 1..20.

One run per bearing and operating condition, its 20 measurements as 20 captures, one stream
(the 64 kHz channels). Values are stored as recorded (64-bit floats, scale 1, offset 0; D5, D6).
Nothing is derived, filled in or corrected here; quirks are recorded (D212 item 7).

The bearing table below is transcribed from each bearing's damage sheet (<bearing>.pdf in its
folder) and cross-checked with Lessmeier et al. (2016), Tables 4, 5 and 7 (D212 item 4).
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io

DATASET = "paderborn"
STREAM = "fast"  # the 64 kHz channels (raster "HostService" in the files)
SAMPLE_RATE_HZ = 64000.0
CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2")  # D48; names as in the files
SENSOR_TYPE = {"vibration_1": "accelerometer", "phase_current_1": "current", "phase_current_2": "current"}
# All variables in the files' Y structure; those not in CHANNELS are not converted (D48).
EXPECTED_Y = {"force", "phase_current_1", "phase_current_2", "speed", "temp_2_bearing_module", "torque", "vibration_1"}

# Operating conditions: code -> (speed rpm, load torque Nm, radial force N), nominal (D212 item 3).
CONDITIONS = {
    "N15_M07_F10": (1500, 0.7, 1000),
    "N09_M07_F10": (900, 0.7, 1000),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
}
N_MEASUREMENTS = 20

# One entry per bearing, from its damage sheet. Fields with one value per damage are joined
# with "; " in sheet order. Abbreviations as on the sheets: S single, R repetitive, M multiple damage.
# lessmeier_class: Lessmeier et al. (2016) Sections 7.2-7.4 (D212 item 4).
def _healthy(code, run_in_h):
    return dict(group="healthy", lessmeier_class="healthy", damaged_rings="none", n_damages=0,
                symptom=None, component=None, combination=None, arrangement=None, extent=None,
                characteristic=None, method=None, manufacturer="IBU", run_in_h=run_in_h, lifetime=None)

def _damaged(group, cls, manufacturer, component, symptom, combination, arrangement, extent,
             characteristic, method, lifetime=None):
    rings = sorted(set(component.replace("AR", "OR").split("; ")), key=["IR", "OR"].index)  # "AR" = Außenring (outer ring)
    return dict(group=group, lessmeier_class=cls, damaged_rings="+".join(rings),
                n_damages=len(component.split("; ")), symptom=symptom, component=component,
                combination=combination, arrangement=arrangement, extent=extent,
                characteristic=characteristic, method=method, manufacturer=manufacturer,
                run_in_h=None, lifetime=lifetime)

NO_REP = "without repetitive damage"
BEARINGS = {
    # healthy (sheet "Lifetime h" = run-in time; Lessmeier et al. Table 7)
    "K001": _healthy("K001", ">50"), "K002": _healthy("K002", "19"), "K003": _healthy("K003", "1"),
    "K004": _healthy("K004", "5"), "K005": _healthy("K005", "10"), "K006": _healthy("K006", "16"),
    # artificial damage (Table 4); symptom n/a on the sheets
    "KA01": _damaged("artificial", "OR", "MTK", "OR", "n/a", "S", NO_REP, "1", "single point", "EDM machining"),
    "KA03": _damaged("artificial", "OR", "MTK", "OR", "n/a", "S", NO_REP, "2", "single point", "electric engraver"),
    "KA05": _damaged("artificial", "OR", "IBU", "OR", "n/a", "S", NO_REP, "1", "single point", "electric engraver"),
    "KA06": _damaged("artificial", "OR", "IBU", "OR", "n/a", "S", NO_REP, "2", "single point", "electric engraver"),
    "KA07": _damaged("artificial", "OR", "IBU", "OR", "n/a", "S", NO_REP, "1", "single point", "drilled"),
    "KA08": _damaged("artificial", "OR", "IBU", "AR", "n/a", "S", NO_REP, "2", "single point", "drilled"),  # sheet says "AR"
    "KA09": _damaged("artificial", "OR", "IBU", "OR", "n/a", "S", NO_REP, "2", "single point", "drilled"),
    "KI01": _damaged("artificial", "IR", "MTK", "IR", "n/a", "S", NO_REP, "1", "single point", "n/a"),  # paper: EDM
    "KI03": _damaged("artificial", "IR", "MTK", "IR", "n/a", "S", NO_REP, "1", "single point", "electric engraver"),
    "KI05": _damaged("artificial", "IR", "IBU", "IR", "n/a", "S", NO_REP, "1", "single point", "electric engraver"),
    "KI07": _damaged("artificial", "IR", "IBU", "IR", "n/a", "S", NO_REP, "2", "single point", "electric engraver"),
    "KI08": _damaged("artificial", "IR", "IBU", "IR", "n/a", "S", NO_REP, "2", "single point", "electric engraver"),
    # real damage from accelerated lifetime tests (Table 5); lifetime as on the sheet (h:min)
    "KA04": _damaged("real", "OR", "FAG", "OR", "pitting", "S", NO_REP, "1", "single point", "lifetime test", "12:01"),
    "KA15": _damaged("real", "OR", "FAG", "OR", "particle-caused indentations", "S", NO_REP, "1", "single point", "lifetime test", "6:01"),
    "KA16": _damaged("real", "OR", "MTK", "OR; OR", "pitting; pitting", "R; R", "random; random", "2; 2", "single point; single point", "lifetime test; lifetime test", "9:01"),
    "KA22": _damaged("real", "OR", "IBU / IBB", "OR", "pitting", "S", NO_REP, "1", "single point", "lifetime test", "14:01"),
    "KA30": _damaged("real", "OR", "MTK", "OR", "particle-caused indentations", "R", "random", "1", "distributed", "lifetime test", "21:01"),
    "KB23": _damaged("real", "IR", "MTK", "IR; IR; OR", "pitting; pitting; pitting", "R; R; R", f"random; random; {NO_REP}", "1; 2; 2", "single point; single point; single point", "lifetime test; lifetime test; lifetime test", "21:01"),
    "KB24": _damaged("real", "IR", "FAG", "IR; OR", "pitting; particle-caused indentations", "M; M", f"{NO_REP}; random", "3; n/a", "distributed; single point", "lifetime test; lifetime test", "15:01"),
    "KB27": _damaged("real", "OR", "FAG", "OR; IR", "particle-caused indentations; particle-caused indentations", "M; M", "random; random", "1; 1", "distributed; distributed", "lifetime test; lifetime test", "21:01"),
    "KI04": _damaged("real", "IR", "MTK", "IR; OR", "pitting; particle-caused indentations", "M; M", f"{NO_REP}; random", "1; n/a", "single point; single point", "lifetime test; lifetime test", "12:01"),
    "KI14": _damaged("real", "IR", "MTK", "IR; OR", "pitting; particle-caused indentations", "M; M", f"{NO_REP}; random", "1; n/a", "single point; single point", "lifetime test; lifetime test", "21:01"),
    "KI16": _damaged("real", "IR", "FAG", "IR", "pitting", "S", NO_REP, "3", "single point", "lifetime test", "3:01"),
    "KI17": _damaged("real", "IR", "MTK", "IR; IR", "pitting; pitting", "R; R", "random; random", "1; 1", "single point; single point", "lifetime test; lifetime test", "19:01"),
    "KI18": _damaged("real", "IR", "MTK", "IR", "pitting", "S", NO_REP, "2", "single point", "lifetime test", "7:01"),
    "KI21": _damaged("real", "IR", "FAG", "IR", "pitting", "S", NO_REP, "1", "single point", "lifetime test", "20:01"),
}
assert len(BEARINGS) == 32

# Sensor details from the measuring logs (D212 item 5).
SENSORS = {
    "vibration_1": "PCB 336C04 (100 mV/g), Kistler 5015 charge amplifier, 30 kHz filter (2nd order)",
    "phase_current_1": "LEM CKSR 15-NP current transducer",
    "phase_current_2": "LEM CKSR 15-NP current transducer",
}
# Remarks in the measuring logs that concern the data (D212 item 7).
LOG_NOTES = {
    "KA04": "measuring log: N09_M07_F10_KA04_17.mat held faulty data and was replaced by a copy of "
            "N09_M07_F10_KA04_18.mat; some measurements have too few samples",
    "KI04": "measuring log: faulty files (vibration sensor not switched on) were moved to a folder "
            "'Fehlerhaft'; that folder is not in the download",
}


def run_id(bearing, condition):
    return f"{bearing}_{condition}"


def _y_channels(mat_struct):
    """The Y variables of one measurement: name -> (raster, data)."""
    return {e.Name: (e.Raster, np.asarray(e.Data, dtype=np.float64).ravel()) for e in mat_struct.Y}


def read(folder, bearing, condition):
    """Return (tables, signals, notes) of one bearing at one condition; notes lists the quirks found."""
    info = BEARINGS[bearing]
    rid = run_id(bearing, condition)
    files = [os.path.join(folder, bearing, f"{condition}_{bearing}_{n}.mat") for n in range(1, N_MEASUREMENTS + 1)]

    captures, telemetry, signals, notes = [], [], [], []
    hashes = {}  # vibration hash -> capture, to find exact copies within the run
    for n in range(1, N_MEASUREMENTS + 1):
        name = f"{condition}_{bearing}_{n}"
        member = f"{bearing}/{name}.mat"  # path relative to <folder>
        try:
            mat = scipy.io.loadmat(files[n - 1], squeeze_me=True, struct_as_record=False)
        except (TypeError, ValueError) as e:  # unreadable file: recorded and left out (D212 item 7)
            notes.append(f"m{n:02d}: {member} cannot be read ({type(e).__name__}: {e}); not converted")
            continue
        y = _y_channels(mat[name])
        unexpected = set(y) - EXPECTED_Y
        if unexpected:
            raise ValueError(f"{member}: unexpected variables {sorted(unexpected)}")
        rasters = {y[ch][0] for ch in CHANNELS}
        if rasters != {"HostService"}:
            raise ValueError(f"{member}: 64 kHz channels on rasters {rasters}")
        lengths = {ch: len(y[ch][1]) for ch in CHANNELS}
        if len(set(lengths.values())) != 1:
            notes.append(f"m{n:02d}: channels differ in length {lengths}")  # recorded, not fixed

        capture_id = f"m{n:02d}"
        h = hashlib.sha256(y["vibration_1"][1].tobytes()).hexdigest()
        copy_of = hashes.get(h)
        hashes.setdefault(h, capture_id)
        if copy_of:
            notes.append(f"{capture_id}: vibration identical to {copy_of}")
        captures.append({
            "run_id": rid, "capture_id": capture_id, "stream": STREAM,
            "source_name": member, "start_s": np.nan, "n_samples": min(lengths.values()),
            "timing_accuracy": "not applicable",
            # per-capture columns (D31, D212 items 3, 4, 7)
            "bearing": bearing, "condition": condition, "measurement": n,
            "identical_to": copy_of,
            **{k: v for k, v in info.items()},
        })
        speed, torque, force = CONDITIONS[condition]
        telemetry.append({"run_id": rid, "capture_id": capture_id,
                          "rpm_nominal": float(speed), "torque_nm_nominal": torque, "force_n_nominal": float(force)})
        for ch in CHANNELS:
            signals.append((capture_id, ch, y[ch][1], 1.0, 0.0, "unknown"))

    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7, D212 item 6)
        "bearing": bearing, "lubricant": None,
        # D24 for a run made of 20 files: the files' name pattern, total size and latest modification time
        "source_name": f"{bearing}/{condition}_{bearing}_*.mat",
        "source_size_bytes": sum(os.path.getsize(f) for f in files),
        "source_mtime_utc": datetime.fromtimestamp(max(os.path.getmtime(f) for f in files), timezone.utc).isoformat(),
        "extras": json.dumps({
            "condition": condition,
            "sensors": SENSORS,
            "units_note": "not stated in the files (D212 item 5)",
            "log_note": LOG_NOTES.get(bearing),
            "conversion_notes": notes,
            "not_converted": sorted(EXPECTED_Y - set(CHANNELS)),  # D48
        }),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": ch, "sensor_type": SENSOR_TYPE[ch],
        "sample_rate_hz": SAMPLE_RATE_HZ, "units": "unknown", "scale": 1.0, "offset": 0.0,
        "x_origin_s": np.nan, "dtype": "float64",
    } for ch in CHANNELS])
    tables = {"runs": runs, "captures": pd.DataFrame(captures),
              "channels": channels, "telemetry": pd.DataFrame(telemetry)}
    return tables, signals, notes
