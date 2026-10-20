"""Tests for the PatchTST adapter. Skipped if the submodule external/patchtst is missing.

They use small settings for speed (not the experiment's settings) and the device the official
Learner picks (the GPU if there is one).
"""

import os

import numpy as np
import pytest
import torch

from harness.methods.external import EXTERNAL

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(EXTERNAL, "patchtst", "PatchTST_self_supervised", "src")),
    reason="external/patchtst not present")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def settings(lr=1e-3, validation_fraction=0.25, pooling="mean"):
    return {"patch_len": 8, "stride": 8, "mask_ratio": 0.4, "revin": True, "revin_eps": 1e-5,
            "batch_size": 4, "epochs": 2, "lr": lr, "pct_start": 0.3, "validation_fraction": validation_fraction,
            "model": {"n_layers": 1, "n_heads": 2, "d_model": 8, "d_ff": 16, "dropout": 0.2, "head_dropout": 0.2,
                      "attn_dropout": 0.0, "shared_embedding": True, "norm": "BatchNorm", "act": "relu",
                      "res_attention": False, "pre_norm": False, "pe": "zeros", "learn_pe": True},
            "pooling": pooling, "encode_batch_size": 3}


def test_patching_like_official_script():
    """Number of patches as in the script; uneven lengths lose their FIRST samples (official create_patch)."""
    from harness.methods.patchtst import n_patches, official
    assert n_patches(512, 12, 12) == 42  # the paper's 42 patches
    assert n_patches(500, 12, 12) == 41
    x = torch.arange(20, dtype=torch.float).reshape(1, 20, 1)
    patches, n = official.callback.patch_mask.create_patch(x, 8, 8)
    assert n == 2 and patches[0, 0, 0, 0].item() == 4  # samples 0..3 dropped


@pytest.mark.parametrize("lr, validation_fraction", [(1e-3, 0.25), (1e-3, 0.0), ("finder", 0.25)])
def test_pretrain_and_encode(lr, validation_fraction):
    from harness.methods import patchtst
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(16, 40, 2))  # 2 channels, 5 patches of 8
    model, log = patchtst.pretrain(X, settings(lr, validation_fraction), DEVICE)
    rec = model.training_record
    assert len(log) == 2 and all(np.isfinite(log))
    assert len(rec["validation_rows"]) == round(16 * validation_fraction)
    assert len(rec["valid_loss"]) == (2 if validation_fraction else 0)
    assert rec["lr"] > 0
    Z = patchtst.encode(model, X)
    assert Z.shape == (16, 2 * 8)  # channels x d_model
    Z_one = np.concatenate([patchtst.encode(model, X[i:i + 1]) for i in range(16)])
    assert np.allclose(Z, Z_one, atol=1e-5)  # a series' vector does not depend on the other series in its batch


@pytest.mark.parametrize("pooling, size", [("last", 2 * 8), ("mean", 2 * 8), ("flatten", 2 * 8 * 5)])
def test_pooling(pooling, size):
    from harness.methods import patchtst
    torch.manual_seed(0)
    X = np.random.default_rng(1).normal(size=(8, 40, 2))
    model, _ = patchtst.pretrain(X, settings(validation_fraction=0.0, pooling=pooling), DEVICE)
    Z = patchtst.encode(model, X)
    assert Z.shape == (8, size)
    if pooling == "last":  # same vector as the official classification head takes
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(next(model.parameters()).device)
            x = patchtst.official.models.layers.revin.RevIN(2, eps=1e-5, affine=False)(x, "norm")
            p, _ = patchtst.official.callback.patch_mask.create_patch(x, 8, 8)
            z = model.backbone(p)[:, :, :, -1].flatten(1)  # = ClassificationHead before its dropout and linear layer
        assert np.allclose(Z, z.cpu().numpy(), atol=1e-6)


def test_save(tmp_path):
    from harness.methods import patchtst
    torch.manual_seed(0)
    X = np.random.default_rng(2).normal(size=(8, 40, 1))
    model, _ = patchtst.pretrain(X, settings(validation_fraction=0.25), DEVICE)
    patchtst.save(model, str(tmp_path / "model"))
    fresh = patchtst.build_model(1, 40, settings())
    fresh.load_state_dict(torch.load(tmp_path / "model.pth"))  # official checkpoint format: plain state_dict
    assert (tmp_path / "model_training.json").exists()


def test_load_with_other_pooling(tmp_path):
    """A saved model, loaded with another pooling, gives what the trained model gives with that pooling."""
    from harness.methods import patchtst
    torch.manual_seed(0)
    X = np.random.default_rng(3).normal(size=(8, 40, 2))
    model, _ = patchtst.pretrain(X, settings(validation_fraction=0.0, pooling="last"), DEVICE)
    patchtst.save(model, str(tmp_path / "model"))
    for pooling in ("mean", "flatten"):
        loaded = patchtst.load(str(tmp_path / "model"), 2, 40, settings(pooling=pooling), DEVICE)
        model.encode_settings["pooling"] = pooling
        assert np.allclose(patchtst.encode(loaded, X), patchtst.encode(model, X), atol=1e-6)


def test_placeholders_and_device():
    from harness.methods import patchtst
    X = np.zeros((4, 40, 1))
    s = settings()
    s["model"]["d_ff"] = None
    with pytest.raises(ValueError, match="d_ff"):
        patchtst.pretrain(X, s, DEVICE)
    with pytest.raises(ValueError, match="official Learner"):
        patchtst.pretrain(X, settings(), "cpu" if DEVICE == "cuda" else "cuda")
