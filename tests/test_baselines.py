"""Checks of the baselines (D75-D83): feature layouts, the two new probes, MiniRocket's seeding,
and an end-to-end run of every baseline config on a tiny synthetic stand-in for FordA."""

import glob
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import RidgeClassifierCV

from harness import evaluate
import pycatch22

from harness.methods import catch22, envelope, fft, minirocket, mssp, raw
from harness.methods.mssp_features import FEATURE_NAMES
from scripts import convert_forda
from scripts import evaluate as evaluate_script
from scripts import pretrain
from tests.test_slice import write_fake_forda

rng = np.random.default_rng(0)
X2 = rng.standard_normal((5, 64, 2))  # 5 series, 64 samples, 2 channels


def test_raw_layout():
    Z = raw.encode(None, X2)
    assert Z.shape == (5, 128)
    assert np.array_equal(Z[:, :64], X2[:, :, 0]) and np.array_equal(Z[:, 64:], X2[:, :, 1])


def test_fft_is_one_sided_magnitude_per_channel():
    Z = fft.encode(None, X2)
    assert Z.shape == (5, 2 * 33)  # 64 // 2 + 1 bins per channel
    assert np.allclose(Z[:, 33:], np.abs(np.fft.rfft(X2[:, :, 1], axis=1)))


def test_envelope_layout_and_modulation_peak():
    """Squared envelope spectrum (D161): same layout as fft; a carrier at bin 100 modulated at
    bin 8 gives its largest non-DC envelope-spectrum value at bin 8, not near the carrier."""
    Z = envelope.encode(None, X2)
    assert Z.shape == (5, 2 * 33)
    t = np.arange(1024)
    x = (1 + 0.5 * np.cos(2 * np.pi * 8 * t / 1024)) * np.cos(2 * np.pi * 100 * t / 1024)
    Z = envelope.encode(None, x[None, :, None])
    assert Z.shape == (1, 513)
    assert np.argmax(Z[0, 1:]) + 1 == 8

def test_mssp_statistics_order_and_values():
    fs = 1000.0
    t = np.arange(1000) / fs
    X = np.stack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 120 * t)], axis=1)[None]  # 1 series, 2 channels
    model, _ = mssp.pretrain(X, {"fs": fs}, "cpu")
    Z = mssp.encode(model, X)
    assert Z.shape == (1, 2 * 14)
    dom = FEATURE_NAMES.index("dominant_frequency")
    assert Z[0, dom] == pytest.approx(50) and Z[0, 14 + dom] == pytest.approx(120)
    assert Z[0, FEATURE_NAMES.index("rms")] == pytest.approx(np.sqrt(0.5), rel=1e-3)


def test_catch24_per_channel():
    X = rng.standard_normal((5, 128, 2))  # >= 100 samples: below that one feature (DFA) cannot be estimated
    model, _ = catch22.pretrain(X, {"catch24": True, "mask_nans": False}, "cpu")
    Z = catch22.encode(model, X)
    assert Z.shape == (5, 2 * 24)
    assert np.allclose(Z[3, 24:], pycatch22.catch22_all(X[3, :, 1], catch24=True, mask_nans=False)["values"])


def test_catch24_stops_on_features_that_cannot_be_estimated():
    model, _ = catch22.pretrain(None, {"catch24": True, "mask_nans": False}, "cpu")
    with pytest.raises(ValueError, match="could not be estimated"):
        catch22.encode(model, np.ones((1, 64, 1)))  # constant series: most features are undefined


def test_ridge_minirocket_equals_old_normalize():
    """Standardising and scaling alpha by n gives the same model as the removed normalize=True:
    centre each feature, divide it by its L2 norm (training statistics), fit RidgeClassifierCV."""
    n = 60
    Z = rng.standard_normal((n, 8)) * rng.uniform(0.1, 10, 8)
    y = rng.integers(0, 3, n)
    Z_test = rng.standard_normal((20, 8))
    alphas = np.logspace(-3, 3, 10)

    mean, norm = Z.mean(axis=0), np.linalg.norm(Z - Z.mean(axis=0), axis=0)
    old = RidgeClassifierCV(alphas=alphas).fit((Z - mean) / norm, y)
    new = evaluate.ridge_minirocket(Z, y)

    assert new[-1].alpha_ == pytest.approx(old.alpha_ * n)
    assert np.allclose(new.decision_function(Z_test), old.decision_function((Z_test - mean) / norm))


def test_floor_predicts_most_frequent_class():
    y = np.array([0, 1, 1, 1, 2])
    assert (evaluate.floor(np.zeros((5, 3)), y).predict(np.ones((4, 3))) == 1).all()


def test_minirocket_seed_fixes_biases():
    X = rng.standard_normal((20, 64, 1))
    s = {"num_features": 168, "max_dilations_per_kernel": 32}

    def biases(seed):
        np.random.seed(seed)  # as runs.set_seed does
        return minirocket.pretrain(X, s, "cpu")[0][2]

    assert np.array_equal(biases(0), biases(0))
    assert not np.array_equal(biases(0), biases(1))


TINY = {"minirocket": {"num_features": 168}}  # small so the test runs in seconds


@pytest.mark.parametrize("name", ["raw", "fft", "catch22", "minirocket"])
def test_baseline_end_to_end(tmp_path, monkeypatch, name):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_fake_forda(raw_dir, n=200, length=128)  # catch24 needs >= 100 samples (D90)
    convert_forda.convert(str(raw_dir), str(tmp_path / "canonical"))

    with open(f"configs/{name}_forda.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = str(tmp_path / "canonical")
    cfg["seeds"] = [0, 1]
    cfg["evaluation"]["label_fractions"] = [0.1, 1.0]  # 1 % of 200 is too few for 5-fold cross-validation
    cfg[name].update(TINY.get(name, {}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.chdir(tmp_path)  # runs/ is created in the current folder
    pretrain.main(str(config_path))
    run = glob.glob("runs/*")[0]
    timing = json.load(open(f"{run}/seed_0/timing.json"))
    assert set(timing) == {"pretrain_s", "encode_train_s", "encode_test_s", "encode_series_per_s"}
    assert json.load(open(f"{run}/hardware.json"))["device"] == "cpu"
    assert (len(glob.glob(f"{run}/compile.json")) == 1) == (name == "minirocket")

    evaluate_script.main(run)
    results = pd.read_csv(f"{run}/results.csv")
    assert len(results) == 2 * 2 * len(cfg["evaluation"]["probes"])  # seeds x label fractions x probes
    assert results.accuracy.between(0, 1).all() and (results.probe_fit_s >= 0).all()
    # Same seed -> same label subset for every method (D78): n_labels per fraction is the same everywhere.
    assert results.groupby("label_fraction").n_labels.nunique().eq(1).all()


def test_minirocket_several_channels_use_the_multivariate_code():
    """D150: two channels -> official multivariate fit (5 parameter arrays); one channel unchanged (3)."""
    X = rng.standard_normal((20, 64, 2))
    s = {"num_features": 168, "max_dilations_per_kernel": 32}
    np.random.seed(0)
    params, _ = minirocket.pretrain(X, s, "cpu")
    assert len(params) == 5 and minirocket.encode(params, X).shape[0] == 20
    np.random.seed(0)
    assert len(minirocket.pretrain(X[:, :, :1], s, "cpu")[0]) == 3
