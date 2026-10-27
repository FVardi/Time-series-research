"""Tests for the TF-JEPA adapter. Skipped if the submodule external/tfjepa is missing."""

import os

import numpy as np
import pytest
import torch

from harness.methods.external import EXTERNAL, import_official

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(EXTERNAL, "tfjepa", "src")),
                                reason="external/tfjepa not present")


def settings(views="official"):
    """Small settings for fast tests (not the experiment's settings)."""
    return {"views": views, "jitter_ratio": 0.8, "batch_size": 8, "drop_last": True, "epochs": 2,
            "ema_momentum": 0.995,
            "encoder": {"embed_dim": 16, "nhead": 2, "nlayers": 1, "ff_mult": 2},
            "optimizer": {"lr": 3e-4, "betas": [0.9, 0.99], "eps": 1e-8, "weight_decay": 3e-4},
            "representation": "z", "encode_batch_size": 5}


@pytest.mark.parametrize("views", ["official", "augmented", "single"])
def test_pretrain_and_encode(views):
    from harness.methods import tfjepa
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(30, 40, 1))
    model, log = tfjepa.pretrain(X, settings(views), "cpu")
    assert len(log) == 2 and np.all(np.isfinite(log))
    for rep, dim in [("z", 2 * 16), ("h", 2 * 40), ("z_raw", 2 * 16 + 1)]:
        model.representation = rep
        Z = tfjepa.encode(model, X)
        assert Z.shape == (30, dim)
        Z_one = np.concatenate([tfjepa.encode(model, X[i:i + 1]) for i in range(30)])
        assert np.allclose(Z, Z_one, atol=1e-5)  # a series' vector does not depend on its batch


def test_encode_matches_official_forward():
    """encode('z') equals the online projector outputs of the official forward pass (eval mode)."""
    from harness.methods import tfjepa
    torch.manual_seed(0)
    X = np.random.default_rng(1).normal(size=(16, 40, 1))
    model, _ = tfjepa.pretrain(X, settings(), "cpu")
    model.eval()
    x_t = torch.from_numpy(np.transpose(X, (0, 2, 1)).astype(np.float32))
    with torch.no_grad():
        out = model(x_t, torch.fft.fft(x_t).abs())
    expected = torch.cat([out["z_time_online"], out["z_freq_online"]], dim=1).numpy()
    assert np.allclose(tfjepa.encode(model, X), expected, atol=1e-5)


def test_placeholders_and_channels():
    from harness.methods import tfjepa
    s = settings()
    s["batch_size"] = None
    with pytest.raises(ValueError, match="placeholder"):
        tfjepa.pretrain(np.zeros((10, 40, 1)), s, "cpu")
    with pytest.raises(ValueError, match="channel 0"):
        tfjepa.pretrain(np.zeros((10, 40, 2)), settings(), "cpu")


def test_training_loop_equals_official_function(monkeypatch):
    """Our written-out loop (views: official) gives exactly the weights of the official
    src/train.py:pretrain_tf_jepa, called with the same model, data loader and seed.

    The official function picks the GPU itself if there is one; ours ran on the CPU, and GPU
    arithmetic differs slightly, so it is made to see no GPU for this comparison.
    """
    from harness.methods import tfjepa
    official_train = import_official("tfjepa", "src.train", local_names=("src",))
    s = settings()
    X = np.random.default_rng(2).normal(size=(37, 40, 1))  # 37 series: the last batch is dropped

    torch.manual_seed(0)
    ours, _ = tfjepa.pretrain(X, s, "cpu")

    torch.manual_seed(0)
    cfg, loader = tfjepa.loader(X, s)  # same order of random draws as in pretrain()
    model = official_train.TF_JEPA(cfg, momentum=s["ema_momentum"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    theirs = official_train.pretrain_tf_jepa(model, loader, cfg, lr=s["optimizer"]["lr"], num_epochs=s["epochs"])

    a, b = ours.state_dict(), theirs.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
