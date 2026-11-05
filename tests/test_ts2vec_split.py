"""Tests for the split-representation TS2Vec (harness/methods/ts2vec_split.py; D182, D183, D185)."""

import numpy as np
import pytest
import torch

from harness.methods import ts2vec_split

SMALL = {"repr_dims": 16, "hidden_dims": 8, "depth": 2, "lr": 0.001, "batch_size": 16, "max_train_length": 3000,
         "temporal_unit": 0, "n_iters": 4, "z_op_dims": 4, "head_hidden": 8, "op_weight": 1.0,
         "removal_weight": 1.0, "ramp_max": 1.0}


def data(n=40, length=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, length, 2)).astype(np.float32), rng.integers(0, 4, size=n)


@pytest.mark.parametrize("removal", ["none", "adversary", "hsic"])
def test_variants_train_encode_save(removal, tmp_path):
    X, c = data()
    model, losses = ts2vec_split.pretrain(X, {**SMALL, "removal": removal}, "cpu", c)
    assert len(losses) == SMALL["n_iters"] and np.isfinite(losses).all()
    assert len(model.loss_parts) == SMALL["n_iters"]
    assert (model.adversaries is not None) == (removal == "adversary")
    Z = ts2vec_split.encode(model, X)
    assert Z.shape == (len(X), SMALL["repr_dims"])
    ts2vec_split.save(model, str(tmp_path / "model"))
    assert (tmp_path / "model_heads").exists() and (tmp_path / "model_loss_parts.json").exists()


def test_placeholder_stops():
    with pytest.raises(ValueError, match="removal_weight"):
        ts2vec_split.check_settings({**SMALL, "removal": "hsic", "removal_weight": None})


def test_gradient_reversal():
    x = torch.ones(3, requires_grad=True)
    ts2vec_split.GradReverse.apply(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_dann_ramp():
    assert ts2vec_split.dann_ramp(0.0, 1.0) == 0.0
    assert 0.99 < ts2vec_split.dann_ramp(1.0, 1.0) <= 1.0


def test_hsic_larger_when_dependent():
    torch.manual_seed(0)
    c = torch.randint(0, 4, (64,))
    independent = torch.randn(64, 8)
    dependent = torch.nn.functional.one_hot(c, 4).float() * 3 + 0.1 * torch.randn(64, 4)
    assert ts2vec_split.hsic(dependent, c) > 5 * ts2vec_split.hsic(independent, c)


def test_same_as_official_when_added_losses_are_off():
    """With op_weight 0 and no removal, the encoder trains exactly as the official fit()."""
    from harness.methods import ts2vec as plain
    X, c = data()
    s = {**SMALL, "removal": "none", "op_weight": 0.0}
    torch.manual_seed(1); np.random.seed(1)
    official, official_losses = plain.pretrain(X, {k: s[k] for k in ("repr_dims", "hidden_dims", "depth", "lr",
                                                "batch_size", "max_train_length", "temporal_unit", "n_iters")}, "cpu")
    torch.manual_seed(1); np.random.seed(1)
    ours, ours_losses = ts2vec_split.pretrain(X, s, "cpu", c)
    assert np.allclose([p["ts2vec"] for p in ours.loss_parts], official_losses)
    assert np.allclose(ts2vec_split.encode(ours, X), plain.encode(official, X), atol=1e-6)


@pytest.mark.parametrize("removal", ["adversary", "hsic"])
def test_two_factors(removal):
    """One head (and one adversary) per factor (D197): conditions (n, 2) with 4 and 3 classes."""
    X, c = data()
    rng = np.random.default_rng(1)
    conditions = np.stack([c, rng.integers(0, 3, size=len(c))], axis=1)
    model, losses = ts2vec_split.pretrain(X, {**SMALL, "removal": removal}, "cpu", conditions)
    assert [h[-1].out_features for h in model.op_heads] == [4, 3]
    if removal == "adversary":
        assert [h[-1].out_features for h in model.adversaries] == [4, 3]
    assert np.isfinite(losses).all()


def test_condition_codes_follow_pretraining_order():
    """D198: codes in the order of X_pre ('train' rows, then 'pretrain_only'), classes sorted per factor."""
    import pandas as pd
    from harness import windows
    table = pd.DataFrame({"split": ["test", "train", "pretrain_only", "train"],
                          "run_id": ["a", "b", "c", "a"], "capture_id": ["k"] * 4})
    meta = {"captures": pd.DataFrame({"run_id": ["a", "b", "c"], "capture_id": ["k"] * 3, "stream": ["s"] * 3,
                                      "speed_rpm": [600.0, 2400.0, 1200.0], "temperature_c": [40.0, 40.0, 80.0]}),
            "channels": pd.DataFrame({"run_id": ["a", "b", "c"], "stream": ["s"] * 3, "sample_rate_hz": [1.0] * 3}),
            "telemetry": pd.DataFrame({"run_id": ["a", "b", "c"], "capture_id": ["k"] * 3})}
    codes, classes = windows.condition_codes(table, meta, ["speed_rpm", "temperature_c"])
    # pretraining order: train b, train a, pretrain_only c
    assert codes.tolist() == [[2, 0], [0, 0], [1, 1]]
    assert classes == {"speed_rpm": [600.0, 1200.0, 2400.0], "temperature_c": [40.0, 80.0]}


def test_live_encoder(tmp_path):
    """D207: the live encoder differs from the official averaged copy; both can encode, and save() stores it."""
    X, c = data()
    model, _ = ts2vec_split.pretrain(X, {**SMALL, "removal": "adversary", "save_live_encoder": True}, "cpu", c)
    Z, Z_live = ts2vec_split.encode(model, X), ts2vec_split.encode_live(model, X)
    assert Z.shape == Z_live.shape and not np.allclose(Z, Z_live)
    assert np.allclose(ts2vec_split.encode(model, X), Z)  # the swap is undone after encode_live
    ts2vec_split.save(model, str(tmp_path / "model"))
    assert (tmp_path / "model_live").exists()
