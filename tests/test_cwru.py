"""Check that the CWRU conversion reproduces the source values exactly (D139).

Converts a few files, chosen for their quirks, into a temporary folder and compares
every sample with the source. Skipped if the source files are not present.
"""

import os

import numpy as np
import pytest
import scipy.io

from harness.data import canonical, cwru
from scripts.convert_cwru import convert

SOURCE = "data/raw/cwru"
# normal (no RPM in 98; 99 also holds 98's data), DE and FE faults, stray variable (277), 28 mil (other variable number),
# 48 kHz files with quirks: 174 (holds X173_*), 175 (copies of 217's channels), 217 (also holds 215's data)
FILES = [98, 99, 105, 174, 175, 217, 246, 270, 277, 3001]


@pytest.mark.skipif(not all(os.path.exists(f"{SOURCE}/{n}.mat") for n in FILES), reason="source files not available")
def test_conversion_is_exact(tmp_path):
    convert(SOURCE, tmp_path, files=FILES)
    meta = canonical.load_metadata(tmp_path, cwru.DATASET)
    caps = meta["captures"].set_index("source_file")
    assert sorted(caps.index) == sorted(FILES)

    for n in FILES:
        mat = scipy.io.loadmat(f"{SOURCE}/{n}.mat")
        prefix = f"X{cwru.VARIABLE_NUMBER.get(n, n):03d}"
        chans = meta["channels"][meta["channels"].run_id == cwru.run_id(n)].channel.tolist()
        for ch in chans:
            stored = canonical.load_signal(tmp_path, cwru.DATASET, cwru.run_id(n), "capture_0", ch)
            assert np.array_equal(stored, mat[f"{prefix}_{ch}_time"].ravel())

    # quirks
    assert np.isnan(meta["telemetry"].set_index("run_id").loc["0098", "rpm"])  # no RPM in 98.mat: left missing
    assert meta["channels"][meta["channels"].run_id == "3001"].channel.tolist() == ["DE"]  # 28 mil: DE only
    assert caps.loc[105, "de_bearing"] == "DE_IR_007" and caps.loc[105, "fe_bearing"] == "FE_healthy"
    assert caps.loc[270, "fe_bearing"] == "FE_IR_021" and caps.loc[270, "de_bearing"] == "DE_healthy"
    assert caps.loc[246, "clock_hour"] == 3 and caps.loc[246, "de_bearing"] == "DE_OR3_021"
    assert caps.loc[98, "fault_end"] == "none"
    assert caps.loc[174, "n_samples"] == 63788  # converted from X173_*
    assert caps.loc[217, "n_samples"] == 489125 and caps.loc[175, "n_samples"] == 381890  # own data, not the copies
    sr = meta["channels"].set_index("run_id").sample_rate_hz
    assert (sr.loc["0098"] == 48000).all() and (sr.loc["0105"] == 12000).all() and (sr.loc["0217"] == 48000).all()


def test_file_table_is_consistent():
    rows = cwru.file_table()
    assert len(rows) == len({r["file"] for r in rows})  # every file number once
    assert len(rows) == 4 + 157  # normal + fault files listed on the data center's pages
