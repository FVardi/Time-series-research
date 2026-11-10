"""Reader for the DCASE 2022 Challenge Task 2 development data, machine type "bearing"
(unsupervised anomalous sound detection under domain shift; https://zenodo.org/record/6355122;
task page: dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring).

Decisions: D189 (adopted), D194 (download), D251 (conversion and split).
Expects the development data as downloaded (data/raw/dcase2022_bearing/dev_bearing/bearing/ with
train/, test/ and attributes_00.csv ... attributes_02.csv). File name
section_<NN>_<source|target>_<train|test>_<normal|anomaly>_<NNNN>_<attributes>.wav, e.g.
section_01_source_train_normal_0003_vel_12_loc_A.wav (task page).

One run per section (00 velocity shift, 01 microphone-location shift, 02 factory-noise shift), each
10 s clip one capture, one stream, one channel. The clips are 16-bit PCM, mono, 16 kHz; stored as the
int16 sample values exactly as in the WAV file, scale 1/32768, offset 0 (D6, D20), units "full scale"
(uncalibrated). Domain, split, label and attributes come from the file name and the attribute CSVs.
"""

import json
import os
import re
import wave
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATASET = "dcase2022_bearing"
STREAM = "audio"
CHANNEL = "microphone"
SAMPLE_RATE_HZ = 16000
EXPECTED_FRAMES = 160000  # 10 s at 16 kHz (task page: 10-s clips)
SCALE = 1.0 / 32768  # 16-bit PCM to full scale [-1, 1)
UNITS = "full scale (uncalibrated)"
SECTIONS = {"00": "different rotation velocity between domains",
            "01": "different microphone location between domains",
            "02": "mixing of different factory noise between domains"}  # task page
SOURCE = "DCASE 2022 Task 2 development dataset, bearing (https://zenodo.org/record/6355122)"

_NAME = re.compile(r"^section_(\d\d)_(source|target)_(train|test)_(normal|anomaly)_(\d{4})_(.*)\.wav$")


def parse(file):
    """File name -> dict of section, domain, split, label, index and the attributes (verbatim strings)."""
    m = _NAME.match(file)
    if not m:
        raise ValueError(f"unexpected file name {file}")
    section, domain, split, label, index, attr = m.groups()
    tokens = attr.split("_")
    if len(tokens) % 2:
        raise ValueError(f"{file}: attributes are not name/value pairs")
    attributes = dict(zip(tokens[0::2], tokens[1::2]))  # e.g. {"vel": "12", "loc": "A"}
    return {"section": section, "domain": domain, "split": split, "label": label,
            "clip_index": int(index), "attributes": attributes}


def run_id(section):
    return f"section_{section}"


def read_wav(path):
    """int16 samples of a mono 16-bit 16 kHz WAV file; stops on any other format."""
    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, SAMPLE_RATE_HZ, "NONE"):
            raise ValueError(f"{path}: not 16-bit mono PCM at {SAMPLE_RATE_HZ} Hz")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


def read(folder, section):
    """Return (tables, signals, notes) of one section; notes lists the quirks found.

    signals is a generator (about 1,200 clips per section), read one clip at a time.
    """
    rid = run_id(section)
    files = sorted(f for part in ("train", "test") for f in os.listdir(os.path.join(folder, part))
                   if f.startswith(f"section_{section}_") and f.endswith(".wav"))
    csv = pd.read_csv(os.path.join(folder, f"attributes_{section}.csv"), dtype=str)
    csv_attr = {os.path.basename(r.file_name): (r.d1p, r.d1v) for r in csv.itertuples()}
    notes = []
    counts = pd.Series([tuple(parse(f)[k] for k in ("domain", "split", "label")) for f in files]).value_counts()
    notes += [f"{'/'.join(k)}: {n} clips" for k, n in sorted(counts.items())]  # e.g. section 01 has 9 target train clips (D194)
    missing_csv = [f for f in files if f not in csv_attr]
    if missing_csv:
        notes.append(f"{len(missing_csv)} clips not in attributes_{section}.csv")

    captures, telemetry, paths = [], [], []
    for f in files:
        p = parse(f)
        path = os.path.join(folder, p["split"], f)
        paths.append(path)
        d1p, d1v = csv_attr.get(f, (None, None))
        if d1p is not None and p["attributes"].get(d1p) != d1v:
            raise ValueError(f"{f}: attribute CSV ({d1p}={d1v}) disagrees with the file name")
        capture_id = f[:-4]  # the file name without .wav
        captures.append({
            "run_id": rid, "capture_id": capture_id, "stream": STREAM,
            "source_name": f"{p['split']}/{f}", "n_samples": EXPECTED_FRAMES,  # checked when read
            "start_s": np.nan, "timing_accuracy": "not applicable",
            # per-capture columns (D31): official labels and attributes (task page; file names, attribute CSV)
            "section": section, "domain": p["domain"], "split": p["split"], "label": p["label"],
            "clip_index": p["clip_index"], "attributes": json.dumps(p["attributes"]),
            "d1p": d1p, "d1v": d1v,  # the attribute CSV's columns, verbatim
        })
        telemetry.append({"run_id": rid, "capture_id": capture_id})  # no operating quantities beyond the attributes

    def signals():
        for c, path in zip(captures, paths):
            x = read_wav(path)
            if len(x) != EXPECTED_FRAMES:
                raise ValueError(f"{path}: {len(x)} samples, expected {EXPECTED_FRAMES}")
            yield c["capture_id"], CHANNEL, x, SCALE, 0.0, UNITS

    runs = pd.DataFrame([{
        "dataset": DATASET, "run_id": rid,
        "status": "final",  # public benchmark (D7)
        "bearing": None, "lubricant": None,
        # D24 for a run made of many files: name pattern, total size, latest modification time
        "source_name": f"{{train,test}}/section_{section}_*.wav",
        "source_size_bytes": sum(os.path.getsize(p) for p in paths),
        "source_mtime_utc": datetime.fromtimestamp(max(os.path.getmtime(p) for p in paths), timezone.utc).isoformat(),
        "extras": json.dumps({"source": SOURCE, "domain_shift": SECTIONS[section], "conversion_notes": notes}),
    }])
    channels = pd.DataFrame([{
        "run_id": rid, "stream": STREAM, "channel": CHANNEL, "sensor_type": "microphone",
        "sample_rate_hz": float(SAMPLE_RATE_HZ), "units": UNITS, "scale": SCALE, "offset": 0.0,
        "x_origin_s": np.nan, "dtype": "int16",
    }])
    tables = {"runs": runs, "captures": pd.DataFrame(captures),
              "channels": channels, "telemetry": pd.DataFrame(telemetry)}
    return tables, signals(), notes
