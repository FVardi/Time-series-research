"""Checks of the random-init baseline (D130): configs, batch-norm statistics, untrained TF-C."""

import glob
import os

import numpy as np
import pytest
import torch
import yaml

from harness import random_init
from harness.methods import tfc, tfc_frozen
from tests.test_diagnostics import TFC_SETTINGS

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_random_init_configs_match_trained():
    """Each configs/<base>_random.yaml / _random_bn.yaml equals configs/<base>.yaml apart from
    name, random_init and published_accuracy, so the untrained encoder is the trained run's encoder."""
    paths = sorted(glob.glob(os.path.join(CONFIGS, "*_random.yaml")) + glob.glob(os.path.join(CONFIGS, "*_random_bn.yaml")))
    assert paths
    for path in paths:
        name = os.path.basename(path)[:-len(".yaml")]
        base = name.rsplit("_random", 1)[0]
        r, t = load(path), load(os.path.join(CONFIGS, base + ".yaml"))
        assert r["name"] == name, path
        assert r["random_init"] == ("bn_from_train" if name.endswith("_bn") else "as_initialised"), path
        assert r["evaluation"]["published_accuracy"] is None, path
        assert set(r) == set(t) | {"random_init"}, path
        for key in t:
            if key == "evaluation":
                drop = lambda ev: {k: v for k, v in ev.items() if k != "published_accuracy"}
                assert drop(r[key]) == drop(t[key]), path
            elif key != "name":
                assert r[key] == t[key], f"{path}: {key}"


def toy_model():
    """Two conv + batch-norm blocks, as in the CNN encoders; batch norm on (batch, channels, time)."""
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv1d(2, 4, 3), torch.nn.BatchNorm1d(4), torch.nn.ReLU(),
                               torch.nn.Conv1d(4, 3, 3), torch.nn.BatchNorm1d(3))


def toy_encode(model, X):
    """Like the adapters' encode(): evaluation mode, in batches (batching must not matter)."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x) for x in torch.split(torch.from_numpy(X), 7)]).numpy()


def test_batchnorm_statistics_are_those_of_the_layer_inputs():
    """After recompute_batchnorm, each layer stores the mean and population variance of its own input,
    where the second layer's input is computed with the first layer's new statistics."""
    X = 3.0 + 2.0 * np.random.default_rng(0).standard_normal((50, 2, 20))
    model = toy_model()
    assert random_init.recompute_batchnorm(model, toy_encode, X) == 2
    model.eval()
    with torch.no_grad():
        x1 = model[0](torch.from_numpy(X))
        x2 = model[3](model[2](model[1](x1)))
    for layer, x in ((model[1], x1), (model[4], x2)):
        assert torch.allclose(layer.running_mean, x.mean(dim=(0, 2)), atol=1e-4)
        assert torch.allclose(layer.running_var, x.var(dim=(0, 2), unbiased=False), atol=1e-4)
    weights = toy_model().state_dict()  # same seed: the weights themselves are unchanged
    for k in ("0.weight", "1.weight", "1.bias", "3.weight", "4.weight", "4.bias"):
        assert torch.equal(model.state_dict()[k], weights[k]), k


def test_bn_from_train_refuses_models_without_batchnorm():
    with pytest.raises(ValueError, match="no batch-norm layers|has none"):
        random_init.recompute_batchnorm(torch.nn.Linear(3, 2), lambda m, X: m(X), torch.zeros(4, 3))


def test_untrained_tfc_has_the_trained_runs_initial_weights(monkeypatch):
    """With the same seed, pretrain(train=False) returns the weights the trained run starts from
    (captured when the trained run creates its optimiser, i.e. before its first step)."""
    X = np.random.default_rng(0).standard_normal((32, 128, 1))
    torch.manual_seed(7)
    np.random.seed(7)
    untrained, log = tfc.pretrain(X, TFC_SETTINGS, "cpu", train=False)
    assert len(log) == 0

    start = {}
    original_adam = tfc.adam

    def capture(params, o):
        params = list(params)
        start["weights"] = [p.detach().clone() for p in params]
        return original_adam(params, o)

    monkeypatch.setattr(tfc, "adam", capture)
    torch.manual_seed(7)
    np.random.seed(7)
    trained, _ = tfc.pretrain(X, TFC_SETTINGS, "cpu")
    for w0, w in zip(start["weights"], untrained.parameters()):
        assert torch.equal(w0, w)
    assert not all(torch.equal(w0, w) for w0, w in zip(start["weights"], trained.parameters()))  # training moved them


def test_frozen_tfc_adapter_untrained():
    """The pretrain.py adapter passes train=False on and returns an empty loss list."""
    X = np.random.default_rng(1).standard_normal((16, 128, 1))
    model, log = tfc_frozen.pretrain(X, {**TFC_SETTINGS, "loss": {**TFC_SETTINGS["loss"], "variant": "positives"}},
                                     "cpu", train=False)
    assert log == []
    assert len(random_init.batchnorm_layers(model)) == 8  # 3 encoder + 1 projector layer per branch
