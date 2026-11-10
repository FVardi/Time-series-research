"""Checks of the DCASE 2022 bearing conversion (D251) on synthetic clips in the dataset's layout:
WAV -> int16 codes -> HDF5 -> load_signal is exact (sample / 32768), labels and attributes come
from the file names, and a disagreeing attribute CSV stops the conversion."""

import wave

import numpy as np
import pytest

from harness.data import canonical, dcase2022_bearing as dcase
from scripts.convert_dcase2022_bearing import convert


def _wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(x.astype("<i2").tobytes())


def _dataset(folder, csv_value="C"):
    (folder / "train").mkdir(parents=True); (folder / "test").mkdir()
    rng = np.random.default_rng(0)
    clips = {"train/section_01_source_train_normal_0000_vel_12_loc_A.wav": rng.integers(-32768, 32767, 160000),
             "test/section_01_target_test_anomaly_0003_vel_4_loc_C.wav": rng.integers(-32768, 32767, 160000)}
    for rel, x in clips.items():
        _wav(folder / rel, x)
    (folder / "attributes_01.csv").write_text(
        "file_name,d1p,d1v\n"
        "bearing/train/section_01_source_train_normal_0000_vel_12_loc_A.wav,loc,A\n"
        f"bearing/test/section_01_target_test_anomaly_0003_vel_4_loc_C.wav,loc,{csv_value}\n")
    return clips


def test_conversion_is_exact(tmp_path):
    clips = _dataset(tmp_path / "bearing")
    convert(str(tmp_path / "bearing"), str(tmp_path / "canonical"), ["01"])
    meta = canonical.load_metadata(str(tmp_path / "canonical"), dcase.DATASET)
    caps = meta["captures"].set_index("capture_id")
    c = caps.loc["section_01_target_test_anomaly_0003_vel_4_loc_C"]
    assert (c.domain, c.split, c.label, c.clip_index, c.d1p, c.d1v) == ("target", "test", "anomaly", 3, "loc", "C")
    for rel, x in clips.items():
        cid = rel.split("/")[1][:-4]
        y = canonical.load_signal(str(tmp_path / "canonical"), dcase.DATASET, "section_01", cid, dcase.CHANNEL)
        assert np.array_equal(y, x / 32768.0)


def test_disagreeing_csv_stops(tmp_path):
    _dataset(tmp_path / "bearing", csv_value="B")
    with pytest.raises(ValueError, match="disagrees"):
        dcase.read(str(tmp_path / "bearing"), "01")
