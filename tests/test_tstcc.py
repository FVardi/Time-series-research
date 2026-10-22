"""Tests for the TS-TCC adapter (D104-D107).

They need the submodule external/tstcc and are skipped if it is missing.
"""

import os

import numpy as np
import pytest
import torch
import yaml

from harness.methods.external import EXTERNAL

HAVE_OFFICIAL = os.path.isdir(os.path.join(EXTERNAL, "tstcc", "models"))
pytestmark = pytest.mark.skipif(not HAVE_OFFICIAL, reason="external/tstcc not present")
if HAVE_OFFICIAL:
    from harness.methods import tstcc

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "tstcc_forda.yaml")


def settings():
    """The 'tstcc' section of the FordA config."""
    with open(CONFIG) as f:
        return yaml.safe_load(f)["tstcc"]


def all_names(obj, prefix=""):
    """Every setting of an official config object, as 'name' or 'section.name'."""
    names = []
    for key, value in vars(obj).items():
        if hasattr(value, "__dict__"):  # a section such as augmentation or TC
            names += all_names(value, f"{prefix}{key}.")
        else:
            names.append(prefix + key)
    return names


def test_yaml_lists_every_official_setting():
    """Each HAR setting is either checked (official_config) or changed (changed), none is left implicit,
    and the checked values equal the official file."""
    s = settings()
    assert set(all_names(tstcc.HARConfig())) == set(s["official_config"]) | set(s["changed"])
    tstcc.official_config(s)  # raises if a value differs from the official file


def test_difference_from_official_config_stops():
    s = settings()
    s["official_config"]["lr"] = 0.001
    with pytest.raises(ValueError):
        tstcc.official_config(s)


def test_features_len_fits_forda():
    """65 is the encoder's output length for FordA's 500 samples (check_sizes raises otherwise)."""
    configs = tstcc.official_config(settings())
    model = tstcc.official_model.base_Model(configs)
    tstcc.check_sizes(configs, model, np.zeros((2, 500, 1)))
    with pytest.raises(ValueError):
        tstcc.check_sizes(configs, model, np.zeros((2, 400, 1)))


def test_permutation_fix_gives_the_official_order():
    """The fix shuffles segment indices; on old NumPy the official line shuffled a 1-d object
    array of the segments. Both, from the same seed, must give the same segment order."""
    for seed in range(20):
        splits = np.split(np.arange(500), [37, 120, 121, 400])  # unequal lengths
        objects = np.empty(len(splits), dtype=object)
        objects[:] = splits
        np.random.seed(seed)
        old = np.concatenate(np.random.permutation(objects))
        np.random.seed(seed)
        new = np.concatenate([splits[k] for k in np.random.permutation(len(splits))])
        assert np.array_equal(old, new)


def test_pretrain_and_encode(monkeypatch):
    """Two epochs on random data: finite losses, one 320-value vector per series,
    and the values do not depend on how the series are batched when encoding."""
    s = settings()
    s["official_config"]["num_epoch"] = 2

    class TwoEpochs(tstcc.HARConfig):  # the official config, shortened for the test
        def __init__(self):
            super().__init__()
            self.num_epoch = 2

    monkeypatch.setattr(tstcc, "HARConfig", TwoEpochs)
    X = np.random.default_rng(0).standard_normal((300, 500, 1))
    torch.manual_seed(0)
    np.random.seed(0)
    model, losses = tstcc.pretrain(X, s, "cpu")
    assert len(losses) == 2 and np.all(np.isfinite(losses))
    Z = tstcc.encode(model, X[:50])
    assert Z.shape == (50, 320)
    assert np.allclose(Z[:10], tstcc.encode(model, X[:10]), atol=1e-6)
