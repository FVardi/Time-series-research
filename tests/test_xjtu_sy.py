"""Check that the XJTU-SY conversion reproduces the source values exactly (D224).

Converts the two shortest bearings (two conditions) into a temporary folder and compares every
sample with the CSV text parsed by Python's float() (correctly rounded, so the reference for
"exactly as written"); checks the whole path from CSV through HDF5 and back.
Skipped if the source files are not present.
"""

import os

import numpy as np
import pytest

from harness.data import canonical, xjtu_sy
from scripts.convert_xjtu_sy import convert

SOURCE = "data/raw/xjtu_sy/XJTU-SY_Bearing_Datasets"
BEARINGS = ["Bearing1_5", "Bearing2_4"]  # 52 and 42 files


@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="source files not available")
def test_conversion_is_exact(tmp_path):
    convert(SOURCE, tmp_path, bearings=BEARINGS)
    meta = canonical.load_metadata(tmp_path, xjtu_sy.DATASET)
    caps = meta["captures"]
    assert caps.groupby("run_id").size().to_dict() == {"Bearing1_5": 52, "Bearing2_4": 42}

    for bearing in BEARINGS:
        condition = xjtu_sy.BEARINGS[bearing][0]
        for n in range(1, xjtu_sy.BEARINGS[bearing][1] + 1):
            with open(f"{SOURCE}/{condition}/{bearing}/{n}.csv") as f:
                rows = f.read().splitlines()[1:]
            expected = np.array([[float(v) for v in r.split(",")] for r in rows])
            stored = canonical.load_captures(tmp_path, xjtu_sy.DATASET, bearing, [f"m{n:04d}"], xjtu_sy.CHANNELS)[0]
            assert np.array_equal(stored, expected), f"{bearing} {n}"

    c = caps.set_index(["run_id", "capture_id"])
    assert c.loc[("Bearing1_5", "m0001"), "start_s"] == 0.0
    assert c.loc[("Bearing1_5", "m0052"), "start_s"] == 51 * 60.0
    assert c.loc[("Bearing2_4", "m0010"), "fault_element"] == "Outer race"
    tel = meta["telemetry"].set_index(["run_id", "capture_id"])
    assert tel.loc[("Bearing2_4", "m0001"), "rpm_nominal"] == 2250.0
    assert (meta["channels"].units == "g").all()
    assert caps.identical_to.isna().all()


def test_bearing_table():
    assert len(xjtu_sy.BEARINGS) == 15
    assert sum(v[1] for v in xjtu_sy.BEARINGS.values()) == 9216  # intro Table 2
