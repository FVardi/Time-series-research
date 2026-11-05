"""Check that the HUST conversion reproduces the source values exactly (D223).

Converts a few files, chosen for their quirks, into a temporary folder and compares
every sample with the source. Skipped if the source files are not present.
"""

import os

import numpy as np
import pytest
import scipy.io

from harness.data import canonical, hust
from scripts.convert_hust import convert

SOURCE = "data/raw/hust/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset"
# healthy with ru_raw (N400), run-up with rpm/ru (I600), 512,001 samples (IO400), longest (IB600),
# shortest (OB402), no run-up at all (OB400), fs copied from IO604 (IO602)
FILES = ["N400", "I600", "IO400", "IB600", "OB402", "OB400", "IO602"]


@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="source files not available")
def test_conversion_is_exact(tmp_path):
    convert(SOURCE, tmp_path, files=FILES)
    meta = canonical.load_metadata(tmp_path, hust.DATASET)
    caps = meta["captures"].set_index("run_id")
    assert sorted(caps.index) == sorted(FILES)

    for name in FILES:
        mat = scipy.io.loadmat(f"{SOURCE}/{name}.mat")
        stored = canonical.load_signal(tmp_path, hust.DATASET, name, "capture_0", hust.CHANNEL)
        assert np.array_equal(stored, mat["data"].ravel())
        tel = meta["telemetry"].set_index("run_id")
        assert tel.loc[name, "shaft_freq_hz"] == mat["fs"].ravel()[0]

    assert caps.loc["OB402", "n_samples"] == 121082 and caps.loc["IB600", "n_samples"] == 562001
    assert caps.loc["IO602", "fault"] == "IO" and caps.loc["IO602", "bearing_type"] == "6206"
    assert caps.loc["IO602", "load_w"] == 200 and caps.loc["IO602", "bearing"] == "IO6206"
    assert caps.loc["N400", ["inner_fault", "outer_fault", "ball_fault"]].tolist() == [False, False, False]
    assert caps.loc["OB400", ["inner_fault", "outer_fault", "ball_fault"]].tolist() == [False, True, True]


def test_file_list():
    assert len(hust.FILES) == 99 and "B400" not in hust.FILES and "IB404" not in hust.FILES
