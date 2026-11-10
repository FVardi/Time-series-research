"""Checks of the UOS conversion (D248).

Synthetic files in the dataset's folder layout, with values on an ADC grid as in the real files:
the whole path .mat -> int32 codes -> HDF5 -> load_signal must stay within the allowed few ulp, and
an off-grid value must stop the conversion. If two real files are present, they are checked too.
"""

import os

import numpy as np
import pytest
import scipy.io

from harness.data import canonical, uos

STEP = 3.132768703465217e-06  # the real files' grid step (g), checked 2026-11-10
REAL = ("data/raw/uos/Multi-domain vibration dataset with various bearing types under compound machine fault "
        "scenarios subset 1 (deep groove ball bearing)/BearingType_DeepGrooveBall")


def _write(folder, name, x):
    machine, bearing, rate, model, rpm = uos.parse(name)
    d = folder / "subset" / "BearingType_X" / f"SamplingRate_{rate}" / f"RotatingSpeed_{rpm}"
    d.mkdir(parents=True)
    scipy.io.savemat(d / f"{name}.mat", {"Data": x[:, None], "Spectrogram": np.zeros((2, 2, 2)),
                                         "STFTFreq": np.zeros((2, 1)), "STFTTime": np.zeros((1, 2))})
    return str(d / f"{name}.mat")


def test_synthetic_file(tmp_path):
    x = np.random.default_rng(0).integers(-900_000, 900_000, uos.EXPECTED_SAMPLES) * STEP
    x[0] = STEP  # code 1 present, as in the real files: the step is then taken exactly (D248 note)
    path = _write(tmp_path, "U2_OR_8_30204_1200", x)
    tables, signals, notes = uos.read(str(tmp_path), path, "U2_OR_8_30204_1200")
    root = str(tmp_path / "canonical")
    canonical.write_run(root, uos.DATASET, "U2_OR_8_30204_1200", tables, signals)
    y = canonical.load_signal(root, uos.DATASET, "U2_OR_8_30204_1200", "capture_0", uos.CHANNEL)
    assert np.array_equal(y, x)  # bit for bit, as on the real files
    cap = tables["captures"].iloc[0]
    assert (cap.machine_condition, cap.bearing_condition, cap.bearing_type, cap.speed_rpm) == ("U2", "OR", "tapered roller", 1200)
    assert tables["channels"].iloc[0].sample_rate_hz == 8000.0 and notes == []


def test_off_grid_value_stops(tmp_path):
    x = np.random.default_rng(1).integers(-1000, 1000, uos.EXPECTED_SAMPLES) * STEP
    x[7] += 0.3 * STEP
    path = _write(tmp_path, "H_H_16_6204_600", x)
    with pytest.raises(ValueError, match="not on the ADC grid"):
        uos.read(str(tmp_path), path, "H_H_16_6204_600")


@pytest.mark.skipif(not os.path.isdir(REAL), reason="source files not available")
def test_real_files(tmp_path):
    for name in ("H_IR_8_6204_600", "H_IR_16_6204_600"):
        rate = uos.parse(name)[2]
        path = os.path.join(REAL, f"SamplingRate_{rate}", "RotatingSpeed_600", f"{name}.mat")
        tables, signals, _ = uos.read("data/raw/uos", path, name)
        canonical.write_run(str(tmp_path), uos.DATASET, name, tables, signals)
        x = scipy.io.loadmat(path)["Data"].ravel()
        y = canonical.load_signal(str(tmp_path), uos.DATASET, name, "capture_0", uos.CHANNEL)
        assert np.array_equal(y, x)  # bit for bit (D248 note)
