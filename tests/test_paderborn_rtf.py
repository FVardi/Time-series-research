"""Checks of the Paderborn run-to-failure conversion (D236) on a small synthetic experiment.

The real archives are deleted after conversion (D235), so the test builds its own: an experiment
B10 (64 kHz) with two measurements whose values lie on an ADC grid with round-decimal constants,
as in the real files, in two ZIP parts. It checks the whole path ZIP -> HDF5 -> load_signal,
and that values off the grid stop the conversion.
"""

import io
import zipfile

import numpy as np
import pytest
import scipy.io

from harness.data import canonical, paderborn_rtf
from scripts.convert_paderborn_rtf import convert

STEP = {"accHorizRear_A": 0.000323335, "accHorizFrontal_C": 0.000323356}
OFFSET = {"accHorizRear_A": -7.2113e-05, "accHorizFrontal_C": -0.0001208}
N = 102_400  # 1.6 s at 64 kHz


def _mat(values):
    buf = io.BytesIO()
    scipy.io.savemat(buf, {**{ch: v[:, None] for ch, v in values.items()},
                           "measTime": (np.arange(N) / 64000.0)[:, None]})
    return buf.getvalue()


def _experiment(folder, off_grid=False):
    rng = np.random.default_rng(0)
    signals = []
    for m in (1, 2):
        codes = {ch: rng.integers(-3000 * m, 3000 * m, N) for ch in STEP}
        signals.append({ch: codes[ch] * STEP[ch] + OFFSET[ch] for ch in STEP})
    if off_grid:
        signals[1]["accHorizRear_A"][5] += 0.4 * STEP["accHorizRear_A"]
    oc = ("Time,setDynLoad / N,peak_dynLoad / N,setStatLoad / N,meanAbs_statLoad / N,setSpeed / rpm,meanAbs_speed / rpm\n"
          "01-Mar-2022 10:00:00,0,10.5,2500,2501.25,750,751.5\n01-Mar-2022 10:00:12,1000,1012.5,3000,3001.5,2700,2700.75\n")
    temp = ("Time,Mean Abs. Temp. T1 / °C,Mean Abs. Temp. T2 / °C,Mean Room Temp. / °C\n"
            "01-Mar-2022 10:00:00,21.5,21.25,21.0\n01-Mar-2022 10:00:12,22.5,22.25,21.0\n")
    # Two parts with the real multi-part layout (as B06): top folder B10_part<k>/, documents in part 1.
    with zipfile.ZipFile(folder / "B10_part1.zip", "w") as z:
        z.writestr("B10_part1/vibrationData/", b"")  # folder entry, as in the real parts
        z.writestr("B10_part1/vibrationData/data_B10_M00001.mat", _mat(signals[0]))
        z.writestr("B10_part1/B10_operatingConditions.csv", oc)
        z.writestr("B10_part1/B10_meanTemperatures.csv", temp)
        z.writestr("B10_part1/B10_log.pdf", b"%PDF")
    with zipfile.ZipFile(folder / "B10_part2.zip", "w") as z:
        z.writestr("B10_part2/vibrationData/", b"")  # folder entry, as in the real parts
        z.writestr("B10_part2/vibrationData/data_B10_M00002.mat", _mat(signals[1]))
    return signals


def test_conversion_is_exact(tmp_path):
    raw, root = tmp_path / "raw", tmp_path / "canonical"
    raw.mkdir()
    signals = _experiment(raw)
    convert(str(raw), str(root), ["B10"])

    meta = canonical.load_metadata(str(root), paderborn_rtf.DATASET)
    ch = meta["channels"].set_index("channel")
    assert ch.loc["accHorizRear_A", "scale"] == STEP["accHorizRear_A"]
    assert ch.loc["accHorizRear_A", "offset"] == OFFSET["accHorizRear_A"]
    assert (ch.sample_rate_hz == 64000.0).all() and (ch.dtype == "int16").all()
    caps = meta["captures"].set_index("capture_id")
    assert caps.loc["m00002", "start_s"] == 12.0 and caps.loc["m00001", "defect"] == "OR, B"
    assert caps.loc["m00002", "source_name"] == "B10_part2/vibrationData/data_B10_M00002.mat"
    assert meta["telemetry"].set_index("capture_id").loc["m00002", "setSpeed / rpm"] == 2700.0

    for i, cid in enumerate(["m00001", "m00002"]):
        for c in STEP:
            stored = canonical.load_signal(str(root), paderborn_rtf.DATASET, "B10", cid, c)
            assert np.array_equal(stored, signals[i][c]), f"{cid} {c}"


def test_off_grid_value_stops(tmp_path):
    raw, root = tmp_path / "raw", tmp_path / "canonical"
    raw.mkdir()
    _experiment(raw, off_grid=True)
    with pytest.raises(ValueError, match="not on the ADC grid"):
        convert(str(raw), str(root), ["B10"])
