"""Tests for the FreRA adapter. Skipped if the submodule external/frera is missing."""

import os
import types

import numpy as np
import pytest
import torch

from harness.methods.external import EXTERNAL

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(EXTERNAL, "frera", "autoaug")),
                                reason="external/frera not present")


def settings():
    """Small settings for fast tests (not the experiment's settings)."""
    return {"batch_size": 8, "drop_last": True, "epochs": 2, "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4,
            "temperature": 0.2, "low_dim": 128, "f_lr": 0.001, "f_temperature": 0.1, "l1_weight": 0.003,
            "f_optimizer": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}, "encode_batch_size": 5}


def test_fcn_out_len_matches_official_values():
    """The formula gives the lengths hard-coded in the official FCN (UCIHAR 128 -> 18, WISDM/MS 200 -> 27)
    and 65 for FordA's 500 samples."""
    from harness.methods import frera
    assert frera.fcn_out_len(128) == 18 and frera.fcn_out_len(200) == 27 and frera.fcn_out_len(500) == 65


def test_get_sampling_fix_gives_the_official_draws():
    """The copied get_sampling draws the same numbers as the official one (which needs a GPU),
    shown by running the official lines on the CPU without the .cuda() call."""
    from harness.methods import frera
    aug = frera.FreRA(len_sw=40)
    torch.manual_seed(0)
    ours = aug.get_sampling(aug.weight, temperature=0.1)
    torch.manual_seed(0)
    bias = 0.0001
    eps = (bias - (1 - bias)) * torch.rand(aug.weight.size()) + (1 - bias)
    expected = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + aug.weight) / 0.1)
    assert torch.equal(ours, expected)


def test_pretrain_and_encode():
    from harness.methods import frera
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(30, 40, 2))  # FreRA takes several channels
    model, log = frera.pretrain(X, settings(), "cpu")
    assert len(log) == 2 and np.all(np.isfinite(log))
    Z = frera.encode(model, X)
    assert Z.shape == (30, frera.fcn_out_len(40) * 128)
    Z_one = np.concatenate([frera.encode(model, X[i:i + 1]) for i in range(30)])
    assert np.allclose(Z, Z_one, atol=1e-5)


def test_placeholder_stops():
    from harness.methods import frera
    s = settings()
    s["l1_weight"] = None
    with pytest.raises(ValueError, match="placeholder"):
        frera.pretrain(np.zeros((30, 40, 1)), s, "cpu")


def test_pretraining_equals_official_main_worker(tmp_path, monkeypatch):
    """Two epochs of the adapter give exactly the weights of the official main_worker()
    (framework 'simclr', CPU), run with the same data loader and seed.

    main_worker() gets its data from setup_dataloaders(), which in the official code returns
    nothing (so the released script cannot run); here it returns our loader. fitlog is replaced
    by an object that ignores the calls, and FCN gets the output length as in the adapter.
    """
    from harness.methods import frera
    s = settings()
    X = np.random.default_rng(1).normal(size=(37, 40, 1))  # 37 series: the last batch is dropped

    torch.manual_seed(0)
    ours = frera.pretrain(X, s, "cpu")[0]["model"].state_dict()

    torch.manual_seed(0)
    x = torch.from_numpy(X.astype(np.float32))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, torch.zeros(37), torch.zeros(37)),
                                         batch_size=8, shuffle=True, drop_last=True)
    ignore = types.SimpleNamespace(set_log_dir=lambda *a, **k: None, add_hyper=lambda *a, **k: None,
                                   add_hyper_in_file=lambda *a, **k: None, add_loss=lambda *a, **k: None,
                                   add_metric=lambda *a, **k: None)
    monkeypatch.setattr(frera.official_main, "fitlog", ignore, raising=False)
    monkeypatch.setattr(frera.official_main, "setup_dataloaders", lambda args: (loader, None, loader), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(frera.FCN, "out_len", frera.fcn_out_len(40), raising=False)
    args = types.SimpleNamespace(gpu=None, framework="simclr", dataset="harness", n_feature=1, len_sw=40,
                                 batch_size=8, arch="FCN", low_dim=128, temperature=0.2, lr=0.01, momentum=0.9,
                                 weight_decay=1e-4, logdir=str(tmp_path), f_aug_mode="FreRA", f_lr=0.001,
                                 start_epoch=0, epochs=2, cos=True, f_temperature=0.1, l1_weight=0.003)
    theirs = frera.official_main.main_worker(None, args)
    assert set(ours) == set(theirs)
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)
