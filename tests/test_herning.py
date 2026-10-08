"""Check that the Herning conversion reproduces the source values exactly.

Converts the first 20 sweeps of the source file into a temporary folder and
compares every sample with the source. Skipped if the source file is not present.
"""

import os

import h5py
import numpy as np
import pytest

from harness.data import canonical, herning
from scripts.convert_herning import convert

SOURCE = "data/raw/herning/scope_20260901_112732.h5"
N = 20


@pytest.mark.skipif(not os.path.exists(SOURCE), reason="source file not available")
def test_conversion_is_exact(tmp_path):
    convert(SOURCE, tmp_path, max_captures=N)
    meta = canonical.load_metadata(tmp_path, herning.DATASET)

    # Four tables, N captures, one telemetry row per capture, two channels.
    assert len(meta["captures"]) == N
    assert len(meta["telemetry"]) == N
    assert list(meta["channels"].channel) == ["AE", "UL"]
    # Captures are in time order.
    assert meta["captures"].tick_s.is_monotonic_increasing

    run_id = meta["runs"].run_id[0]
    with h5py.File(SOURCE, "r") as f:
        for row in meta["captures"].itertuples():
            for ch in herning.CHANNELS:
                source = f["sweeps"][row.source_name][ch]["voltage"][:]
                loaded = canonical.load_signal(tmp_path, herning.DATASET, run_id, row.capture_id, ch)
                # Same values up to floating-point rounding (far below one code step).
                assert np.allclose(loaded, source, rtol=0, atol=1e-9)
                # A slice gives the same samples as the full signal.
                part = canonical.load_signal(tmp_path, herning.DATASET, run_id, row.capture_id, ch, 1000, 5000)
                assert np.array_equal(part, loaded[1000:5000])
