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


CONFIG = "configs/fft_cwru.yaml"


@pytest.mark.skipif(not os.path.exists("data/canonical/cwru"), reason="converted CWRU data not available")
def test_cwru_split_windows():
    """D128, D136, D140: windows of the CWRU split; no test bearing in the pretraining data."""
    cfg = runs.load_config(CONFIG)
    parts = windows.prepare(cfg, canonical.load_metadata(cfg["data_root"], cfg["dataset"]))
    table = pd.concat([t for _, t in parts.values()], ignore_index=True)
    windows.check_separate(table)
    (X_train, t_train), (X_test, t_test), (X_only, t_only) = parts["train"], parts["test"], parts["pretrain_only"]
    assert X_train.shape[1:] == (2048, 2) and len(X_train) == len(t_train)
    assert t_train.group.nunique() == 17 and t_test.group.nunique() == 9
    assert set(t_train.label) == set(t_test.label) == {"inner", "ball", "outer@3", "outer@6", "outer@12"}
    assert t_only.label.isna().all() and t_only.run_id.nunique() == 4  # the normal files, resampled
    # windows are consecutive and start at the first sample of their capture
    first = t_train.groupby(["run_id", "capture_id"]).start.min()
    assert (first == 0).all()
    # window k of a capture equals samples [start, stop) of the stored signal (12 kHz files)
    r = t_train.iloc[3]
    x = canonical.load_signal(cfg["data_root"], "cwru", r.run_id, r.capture_id, "DE", r.start, r.stop)
    assert np.array_equal(X_train[3, :, 0], x)
