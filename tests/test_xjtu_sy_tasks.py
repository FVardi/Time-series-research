"""XJTU-SY tasks (D228-D232): labels, FPT (Li et al., 2015), per-window standardisation,
time-window embedding and the remaining-useful-life metrics."""

import os

import numpy as np
import pandas as pd
import pytest

from harness import evaluate, fpt, labels, windows
from harness.data import canonical, xjtu_sy

ROOT = "data/canonical"


def test_rul_and_health_stage_labels():
    caps = pd.DataFrame({"bearing": ["Bearing1_5", "Bearing1_5", "Bearing1_5", "Bearing3_5"],
                         "file_number": [1, 36, 52, 10]})
    rul, groups = labels.xjtu_sy_rul(caps)
    assert rul[:3] == [51 / 52, 16 / 52, 0.0] and groups == list(caps.bearing)  # (N - n) / N, N = 52 (D228)
    stage, _ = labels.xjtu_sy_health_stage(caps)
    assert stage == ["healthy", "degrading", "degrading", None]  # FPT of 1_5 = 36; 3_5 left out (D230)


def test_li2015_trigger_ignores_single_spikes():
    x = 3 + 0.01 * np.tile([1.0, -1.0], 50)  # reference: tiny noise
    x[24] = 10  # single spike: FPT_0, not the final FPT
    x[40:] = 10  # lasting change from index 40 (minute 41)
    assert fpt.li2015(x) == 41


def test_standardise_per_window():
    X = np.random.default_rng(0).normal(5, 3, size=(4, 1000, 2))
    Z = windows.standardise_per_window(X.copy())
    assert np.allclose(Z.mean(axis=1), 0) and np.allclose(Z.std(axis=1), 1)


def test_time_window_embedding():
    Z = np.arange(10, dtype=float)[:, None] * [1, 100]  # row i = [i, 100 i]
    runs = np.array(["a"] * 6 + ["b"] * 4)
    minutes = np.array([3, 1, 2, 4, 6, 5, 1, 2, 3, 4])  # shuffled within run a
    E, kept = evaluate.time_window_embedding(Z, runs, minutes, 3)
    # run a in time order: rows 1, 2, 0, 3, 5, 4; kept from the 3rd minute on
    assert kept.tolist() == [0, 3, 5, 4, 8, 9]
    assert E[0].tolist() == [1, 100, 2, 200, 0, 0]  # oldest first, current last (DSCN Eq. 16)
    with pytest.raises(ValueError):
        evaluate.time_window_embedding(Z[:3], ["a"] * 3, [1, 2, 4], 2)


def test_rul_scores():
    N = 100
    n = np.arange(1, N + 1)
    y = (N - n) / N
    r = evaluate.rul_scores(y, y, n, N)  # perfect: Eq. 15 gives back N - n exactly
    assert r["n_scored"] == 51 and r["score"] == pytest.approx(0) and r["rmse_min"] == pytest.approx(0)
    r = evaluate.rul_scores(y + 0.05, y, n, N)  # late predictions
    assert r["score"] > 0 and r["mae"] == pytest.approx(0.05)
    assert evaluate.phm08_score([-13.0, 10.0]) == pytest.approx(2 * (np.e - 1))
    assert evaluate.rul_scores(np.full(N, 1.5), y, n, N)["n_clipped"] == 51


@pytest.mark.skipif(not os.path.isdir(f"{ROOT}/xjtu_sy"), reason="converted XJTU-SY data not available")
def test_fpt_table_reproduced():
    """Recomputes D230 from the converted data: horizontal kurtosis per minute, Li et al. (2015)."""
    expected = {**fpt.XJTU_SY, "Bearing2_4": None, "Bearing3_5": None, "Bearing3_4": 52, "Bearing3_2": 1230}
    for bearing, (_, n_files, *_) in xjtu_sy.BEARINGS.items():
        ids = [f"m{n:04d}" for n in range(1, n_files + 1)]
        k = []
        for i in range(0, n_files, 500):  # in chunks: one file open per chunk, bounded memory
            x = canonical.load_captures(ROOT, "xjtu_sy", bearing, ids[i:i + 500], xjtu_sy.CHANNELS[:1])[..., 0]
            k += [fpt.kurtosis(row) for row in x]
        assert fpt.li2015(k) == expected[bearing], bearing
