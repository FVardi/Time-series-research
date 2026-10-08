"""Tests for our TF-C (D41 item 10).

The comparisons with the official loss functions need the submodule external/tfc;
they are skipped if it is missing.
"""

import os

import numpy as np
import pytest
import torch

from harness.methods import tfc
from harness.methods.external import EXTERNAL, import_official

HAVE_OFFICIAL = os.path.isdir(os.path.join(EXTERNAL, "tfc", "code", "TFC"))
ENCODER = {"channels": [32, 64, 128], "kernel_size": 8, "strides": [8, 1, 1], "padding": 4,
           "pool_padding": 1, "conv_bias": False, "dropout_block1": 0.35, "projector": [256, 128]}


def official_loss_module():
    return import_official("tfc/code/TFC", "loss", local_names=("loss",))


@pytest.mark.skipif(not HAVE_OFFICIAL, reason="external/tfc not present")
def test_nt_xent_equals_official():
    """Our NT-Xent gives the value of the official (plain) NTXentLoss."""
    torch.manual_seed(0)
    a, b = torch.randn(16, 40), torch.randn(16, 40)
    official = official_loss_module().NTXentLoss("cpu", 16, 0.2, True)
    assert torch.allclose(tfc.nt_xent(a, b, 0.2), official(a, b), atol=1e-5)


@pytest.mark.skipif(not HAVE_OFFICIAL, reason="external/tfc not present")
def test_consistency_loss_equals_2022_formula():
    """L_C equals the 2022 code's loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
    (with plain NT-Xent), and the total is λ(L_T + L_F) + (1 - λ)L_C."""
    torch.manual_seed(1)
    out = [torch.randn(8, 30), torch.randn(8, 12), torch.randn(8, 30), torch.randn(8, 12)]
    aug = [torch.randn(8, 30), torch.randn(8, 12), torch.randn(8, 30), torch.randn(8, 12)]
    d = official_loss_module().NTXentLoss("cpu", 8, 0.2, True)
    l_tf = d(out[1], out[3])
    loss_c = (1 + l_tf - d(out[1], aug[3])) + (1 + l_tf - d(aug[1], out[3])) + (1 + l_tf - d(aug[1], aug[3]))
    expected = 0.5 * (d(out[0], aug[0]) + d(out[2], aug[2])) + 0.5 * loss_c
    ours = tfc.tfc_loss(out, aug, {"temperature": 0.2, "lambda": 0.5, "delta": 1.0})
    assert torch.allclose(ours, expected, atol=1e-5)


def test_freq_augment_removes_adds_symmetrically():
    """One bin pair set to 0, one other (weak) bin pair set to alpha*max, mirror bins equal, rest unchanged."""
    torch.manual_seed(2)
    N, alpha = 64, 0.5
    xf = tfc.spectrum(torch.randn(32, 1, N))
    aug = tfc.freq_augment(xf, n_components=1, alpha=alpha)
    assert torch.allclose(aug[..., 1:N // 2], aug[..., N // 2 + 1:].flip(-1))  # still symmetric
    half = slice(1, N // 2)
    for i in range(32):
        a, x = aug[i, 0, half], xf[i, 0, half]
        top = alpha * xf[i, 0].max()
        changed = torch.nonzero(~torch.isclose(a, x)).flatten()
        removed = [k for k in changed if a[k] == 0]
        added = [k for k in changed if torch.isclose(a[k], top)]
        assert len(removed) == 1 and len(added) == 1 and removed[0] != added[0]
        assert x[added[0]] < top  # only a weak component is raised
        assert len(changed) == 2  # nothing else changed
    assert torch.equal(aug[..., 0], xf[..., 0]) and torch.equal(aug[..., N // 2], xf[..., N // 2])


def test_freq_augment_is_spectrum_of_a_real_signal():
    """A symmetric magnitude with the original (Hermitian) phase gives a real inverse FFT."""
    torch.manual_seed(3)
    x = torch.randn(4, 1, 128)
    X = torch.fft.fft(x)
    aug = tfc.freq_augment(X.abs(), 1, 0.5)
    signal = torch.fft.ifft(aug * torch.exp(1j * X.angle()))
    assert signal.imag.abs().max() < 1e-4


def test_time_augmentations_keep_shape_and_values():
    torch.manual_seed(4)
    np.random.seed(4)
    x = torch.randn(10, 1, 50)
    a = {"jitter_sigma": 0.1, "scaling_mean": 1.0, "scaling_sigma": 0.1, "max_segments": 5}
    assert tfc.time_augment(x, a).shape == x.shape
    # Permutation only reorders samples within each window.
    out = tfc.permutation(x, 5)
    assert torch.allclose(out.sort(-1).values, x.sort(-1).values)
    assert torch.equal(tfc.permutation(x, 1), x)  # M = 1: always one segment, window unchanged


def test_permutation_reaches_M_segments():
    """With M = 3, some windows are cut into 3 segments (the text's maximum), never more."""
    np.random.seed(5)
    x = torch.arange(30, dtype=torch.float32).repeat(200, 1, 1)  # increasing ramp per window
    out = tfc.permutation(x, 3)[:, 0]
    # Number of segments = 1 + number of places where the next value is not "previous + 1".
    n = 1 + (out[:, 1:] != out[:, :-1] + 1).sum(1)
    assert n.max() == 3 and n.min() >= 1


def test_encoder_size_for_fd_windows():
    """For 5,120-sample windows the flattened encoder output has 128 x 82 values."""
    model = tfc.TFC(5120, 1, ENCODER)
    h_t, z_t, h_f, z_f = model(torch.zeros(2, 1, 5120), torch.zeros(2, 1, 5120))
    assert h_t.shape == (2, 128 * 82) and z_t.shape == (2, 128) and z_f.shape == (2, 128)


def test_placeholders_stop_the_run():
    with pytest.raises(ValueError, match="placeholder"):
        tfc.check_settings({"time_aug": {"jitter_sigma": None}})
