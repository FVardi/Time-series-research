"""Check that the Ottawa UODS-VAFDC conversion reproduces the source values exactly (D249).

Converts bearings 1 and 19 (19 has two identical stage files) into a temporary folder and compares
every sample of both channels with the .mat files. Skipped if the source files are not present.
"""

import os

import numpy as np
import pytest
import scipy.io

from harness.data import canonical, ottawa_vafdc
from scripts.convert_ottawa_vafdc import convert

SOURCE = ("data/raw/ottawa_vafdc/University of Ottawa Ball-bearing Vibration and Acoustic Fault Data "
          "under Constant Load and Speed Conditions (UODS-VAFDC)")


@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="source files not available")
def test_conversion_is_exact(tmp_path):
    convert(SOURCE, str(tmp_path), bearings=[1, 19])
    meta = canonical.load_metadata(str(tmp_path), ottawa_vafdc.DATASET)
    caps = meta["captures"].set_index(["run_id", "capture_id"])
    assert len(caps) == 6
    assert caps.loc[("bearing_19", "stage_2"), "identical_to"] == "stage_1"
    assert caps.loc[("bearing_01", "stage_0"), "condition"] == "H" and caps.loc[("bearing_01", "stage_2"), "condition"] == "I"
    for (rid, cid), row in caps.iterrows():
        name = row.source_name.split("/")[-1][:-4]
        v = scipy.io.loadmat(os.path.join(SOURCE, row.source_name))[name]
        stored = canonical.load_captures(str(tmp_path), ottawa_vafdc.DATASET, rid, [cid], list(ottawa_vafdc.CHANNELS))[0]
        assert np.array_equal(stored, v[:, :2]), f"{rid} {cid}"
    tel = meta["telemetry"].set_index("capture_id")
    assert (meta["telemetry"].load_n == 400.0).all()  # bearings 1 and 19 run under the nominal 400 N
