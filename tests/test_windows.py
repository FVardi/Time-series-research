"""Tests for the windowed inputs (harness/windows.py, harness/labels.py) and the grouped probe folds (D129, D147)."""

import os

import numpy as np
import pandas as pd
import pytest

from harness import evaluate, runs, windows
from harness.data import canonical
from harness.labels import cwru as cwru_labels


def test_standardise_keeps_ratios_between_windows():
    """D137: a fixed change of units per channel; amplitude ratios between windows stay."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2048, 2)) * np.array([0.3, 0.1])
    X[5:] *= 4  # five 'loud' windows
    Z = windows.standardise(X, windows.standardisation(X))
    rms = lambda A: np.sqrt((A ** 2).mean(axis=1))
    assert np.allclose(rms(Z[5:]).mean(0) / rms(Z[:5]).mean(0), rms(X[5:]).mean(0) / rms(X[:5]).mean(0), rtol=0.01)
    assert np.allclose(Z.mean(axis=(0, 1)), 0) and np.allclose(Z.std(axis=(0, 1)), 1)


def test_cwru_labels():
    caps = pd.DataFrame({
        "fault_location": ["inner", "outer", "none"], "clock_hour": [np.nan, 3.0, np.nan],
        "fault_end": ["DE", "FE", "none"], "de_bearing": ["DE_IR_007", "DE_healthy", "DE_healthy"],
        "fe_bearing": ["FE_healthy", "FE_OR3_014", "FE_healthy"]})
    labels, groups = cwru_labels(caps)
    assert labels == ["inner", "outer@3", None]
    assert groups == ["DE_IR_007", "FE_OR3_014", None]


def test_check_separate_stops_on_shared_bearing():
    table = pd.DataFrame({"split": ["train", "test"], "run_id": ["a", "b"], "capture_id": ["c", "c"],
                          "group": ["DE_IR_007", "DE_IR_007"]})
    with pytest.raises(ValueError, match="groups in both"):
        windows.check_separate(table)


def test_grouped_folds_hold_whole_groups_and_every_class():
    """D129: whole groups in each validation part, every class in every training part, same folds for the same seed."""
    y = np.repeat([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], 20)
    groups = np.repeat(np.arange(10), 20)
    folds, draw, single = evaluate.grouped_folds(y, groups, seed=5)
    assert single == []
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
        assert set(y[train]) == set(y)
    again, _, _ = evaluate.grouped_folds(y, groups, seed=5)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_grouped_folds_accept_a_class_with_one_group():
    """D148: a class whose windows all come from one group is reported, not a reason to stop."""
    y = np.repeat([0, 0, 1, 1, 2, 2, 3, 3, 4], 20)  # class 4: one group only
    groups = np.repeat(np.arange(9), 20)
    folds, _, single = evaluate.grouped_folds(y, groups, seed=1)
    assert single == [4]
    assert sum(4 not in set(y[train]) for train, _ in folds) == 1  # missing from exactly one training part
    assert all({0, 1, 2, 3} <= set(y[train]) for train, _ in folds)


def test_check_windows_separate_stops_on_shared_window():
    """D176: windows of one capture may be on both sides, the same window may not."""
    table = pd.DataFrame({"split": ["train", "test", "test"], "run_id": ["a", "a", "a"], "capture_id": ["c", "c", "c"],
                          "start": [0, 2048, 0]})
    windows.check_windows_separate(table.iloc[:2])  # same capture, different windows: fine
    with pytest.raises(ValueError, match="windows are in both"):
        windows.check_windows_separate(table)


def test_cwru_zhao_labels():
    """D167: healthy + location x size; outer race only at 6 o'clock."""
    from harness.labels import cwru_zhao
    caps = pd.DataFrame({"run_id": ["0097", "0105", "0234"], "fault_location": ["none", "inner", "outer"],
                         "fault_size_mil": [np.nan, 7.0, 21.0], "clock_hour": [np.nan, np.nan, 6.0],
                         "fault_end": ["none", "DE", "DE"], "de_bearing": ["DE_healthy", "DE_IR_007", "DE_OR6_021"],
                         "fe_bearing": ["FE_healthy"] * 3})
    assert cwru_zhao(caps)[0] == ["healthy", "inner_7", "outer_21"]
    with pytest.raises(ValueError, match="6 o'clock"):
        cwru_zhao(caps.assign(clock_hour=[np.nan, np.nan, 3.0]))


def test_per_class_accuracy():
    """D168, D179: share of each class's test windows predicted correctly."""
    class Fixed:  # predicts class 0 for everything
        def predict(self, Z):
            return np.zeros(len(Z), dtype=int)
    rows = evaluate.per_class(Fixed(), np.zeros((5, 1)), np.array([0, 0, 1, 1, 1]), ["a", "b"])
    assert rows == [{"class": "a", "n_windows": 2, "accuracy": 1.0}, {"class": "b", "n_windows": 3, "accuracy": 0.0}]


@pytest.mark.skipif(not os.path.exists("data/canonical/cwru/0097"), reason="converted CWRU data not available")
def test_cwru_zhao_split():
    """D167: Zhao et al.'s ten files, drive end, 1,024-sample windows; the healthy file resampled (59 windows)."""
    cfg = runs.load_config("configs/fft_cwru.yaml")  # the CWRU protocol (D240)
    parts = windows.prepare(cfg, canonical.load_metadata(cfg["data_root"], cfg["dataset"]))
    (X_train, t_train), (X_test, t_test), (X_only, _) = parts["train"], parts["test"], parts["pretrain_only"]
    assert X_train.shape[1:] == (1024, 1) and len(X_only) == 0
    counts = pd.concat([t_train, t_test]).label.value_counts()
    assert len(counts) == 10 and counts["healthy"] == 59 and counts.drop("healthy").between(118, 119).all()
    assert set(t_test.label) == set(counts.index)


def test_cut_with_hop_overlaps_windows():
    """D226: with a hop, window i holds samples [i * hop, i * hop + length); without, as before (D137)."""
    root = "data/canonical"
    if not os.path.isdir(f"{root}/hust"):
        pytest.skip("HUST not converted")
    meta = canonical.load_metadata(root, "hust")
    caps = windows.select(meta, {"run_id": ["N400", "OB402"]})  # 512,000 and 121,082 samples
    X, t = windows.cut(root, "hust", caps, ["vibration"], 2048, 51200, hop=512)
    assert t.groupby("run_id").size().to_dict() == {"N400": 997, "OB402": 233}  # (n - 2048) // 512 + 1
    x = canonical.load_signal(root, "hust", "N400", "capture_0", "vibration")
    assert np.array_equal(X[5, :, 0], x[5 * 512: 5 * 512 + 2048]) and t.start[5] == 2560 and t.stop[5] == 4608
    X0, t0 = windows.cut(root, "hust", caps, ["vibration"], 2048, 51200)  # no hop: consecutive windows
    assert len(X0) == 512000 // 2048 + 121082 // 2048
    assert np.array_equal(X0[3, :, 0], x[3 * 2048: 4 * 2048]) and t0.start[3] == 3 * 2048


def test_hust_split_per_class():
    """D226 on a subset (one bearing type, one load, one recording per class): per_class windows of
    each class, stratified 80/20, the same draw every time, no window on both sides."""
    cfg = runs.load_config("configs/tfc_hust.yaml")
    if not os.path.isdir(f"{cfg['data_root']}/hust"):
        pytest.skip("HUST not converted")
    cfg["split"]["captures"] = {"bearing_type": "6205", "load_w": 0}
    cfg["split"]["random_windows"]["per_class"] = 500
    meta = canonical.load_metadata(cfg["data_root"], cfg["dataset"])
    parts = windows.prepare(cfg, meta)
    (X_train, t_train), (X_test, t_test) = parts["train"], parts["test"]
    assert X_train.shape == (2800, 2048, 1) and len(X_test) == 700
    assert (t_train.label.value_counts() == 400).all() and (t_test.label.value_counts() == 100).all()
    assert set(t_train.label) == {"N", "I", "O", "B", "IO", "IB", "OB"}
    windows.check_windows_separate(pd.concat([t_train, t_test]))
    again = windows.prepare(cfg, meta)["test"][1]
    assert again[["run_id", "start"]].equals(t_test[["run_id", "start"]])  # fixed draw and split (random_state)
