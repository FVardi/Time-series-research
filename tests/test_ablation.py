"""Tests for the ablation stages of the TF-C study (studies/tfc_ablation/ablation.py, D96, D97)."""

import numpy as np
import torch

from harness.methods import tfc
from studies.tfc_ablation import ablation, collapse
from tests.test_diagnostics import DIAG, TFC_SETTINGS

X = np.random.default_rng(0).standard_normal((32, 128, 1))


def run_stage(stage, representations):
    torch.manual_seed(7)
    np.random.seed(7)
    rows = collapse.diagnostic_rows(len(X), DIAG["windows"], DIAG["seed"])
    return ablation.pretrain(stage, X, X[rows], TFC_SETTINGS, {**DIAG, "representations": representations}, "cpu")


def test_stage_E_equals_tfc_pretrain():
    """Stage E is exactly our TF-C: same weights as harness tfc.pretrain with the same seed."""
    torch.manual_seed(7)
    np.random.seed(7)
    ref, _ = tfc.pretrain(X, TFC_SETTINGS, "cpu")
    model, _, _ = run_stage("E", ["h_t", "h_f", "h", "z_t", "z_f", "z"])
    for k, v in ref.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k


def test_stages_A_and_B_build_one_branch():
    model_a, log_a, mon_a = run_stage("A", ["h_t"])
    model_b, log_b, _ = run_stage("B", ["h_f"])
    assert hasattr(model_a, "time") and not hasattr(model_a, "freq")
    assert hasattr(model_b, "freq") and not hasattr(model_b, "time")
    assert {r["representation"] for r in mon_a.rows} == {"h_t"}
    assert list(log_a.columns) == ["epoch", "loss", "loss_t"]


def test_stage_C_projectors_get_no_gradient():
    """Without S_TF or L_C nothing trains the projectors: their weights stay as initialised; the encoders change.
    (Only the BatchNorm running statistics of the projectors move, since the projectors still run forward.)"""
    torch.manual_seed(7)
    init = {k: v.clone() for k, v in ablation.build_model("C", 128, 1, TFC_SETTINGS["encoder"]).named_parameters()}
    model, _, _ = run_stage("C", ["h_t", "h_f", "h"])
    after = dict(model.named_parameters())
    for k in init:
        if ".projector." in k:
            assert torch.equal(init[k], after[k]), k
    assert not torch.equal(init["time.encoder.0.0.weight"], after["time.encoder.0.0.weight"])


def test_stage_D_loss_is_alignment_formula():
    """D = λ(L_T + L_F) + (1 − λ) S_TF on the same views as tfc.augmented_step."""
    torch.manual_seed(1)
    model = ablation.build_model("D", 128, 1, TFC_SETTINGS["encoder"])
    x = tfc.to_device(X[:8], "cpu")
    torch.manual_seed(3)
    np.random.seed(3)
    loss, _ = ablation.stage_loss("D", model, x, TFC_SETTINGS)
    torch.manual_seed(3)
    np.random.seed(3)
    (h_t, z_t, h_f, z_f), (h_ta, _, h_fa, _) = tfc.augmented_step(model, x, TFC_SETTINGS)
    l = TFC_SETTINGS["loss"]
    expected = (l["lambda"] * (tfc.nt_xent(h_t, h_ta, l["temperature"]) + tfc.nt_xent(h_f, h_fa, l["temperature"]))
                + (1 - l["lambda"]) * tfc.nt_xent(z_t, z_f, l["temperature"]))
    assert torch.allclose(loss, expected)
