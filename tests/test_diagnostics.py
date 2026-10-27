"""Tests for the collapse diagnostics (harness/diagnostics.py, D93) and the collapse study's
training loop (studies/tfc_ablation/collapse.py, D95)."""

import math

import numpy as np
import torch

from harness import diagnostics
from harness.methods import tfc
from studies.tfc_ablation import collapse

SETTINGS = {"rankme_epsilon": 1e-7, "std_threshold_relative": 0.01, "sigreg": {"t_max": 3, "n_points": 17}}


def test_rankme_extremes():
    """Rank 1 for windows on one line, about d for an isotropic cloud with many windows."""
    torch.manual_seed(0)
    line = torch.randn(500, 1) * torch.randn(1, 20)
    assert abs(diagnostics.rankme(line, 1e-12) - 1.0) < 1e-3
    cloud = torch.randn(20000, 20)
    assert diagnostics.rankme(cloud, 1e-12) > 19.5


def test_covariance_eigenvalues_match_numpy():
    torch.manual_seed(1)
    Z = torch.randn(300, 8) @ torch.randn(8, 8)
    expected = np.sort(np.linalg.eigvalsh(np.cov(Z.numpy(), rowvar=False)))[::-1]
    assert np.allclose(diagnostics.covariance_eigenvalues(Z).numpy(), expected)


def test_mean_pairwise_cosine_equals_direct_formula():
    torch.manual_seed(2)
    Z = torch.randn(50, 6)
    u = torch.nn.functional.normalize(Z.double(), dim=1)
    S = u @ u.T
    direct = (S.sum() - S.diagonal().sum()) / (50 * 49)
    assert math.isclose(diagnostics.mean_pairwise_cosine(Z), float(direct), abs_tol=1e-12)
    assert math.isclose(diagnostics.mean_pairwise_cosine(torch.ones(10, 4)), 1.0, abs_tol=1e-12)  # complete collapse


def test_sigreg_small_for_gaussian_large_for_collapse():
    torch.manual_seed(3)
    g = torch.Generator().manual_seed(0)
    A = diagnostics.sigreg_directions(16, 256, g)
    gaussian = diagnostics.sigreg(torch.randn(4000, 16), A, 3, 17)
    collapsed = diagnostics.sigreg(torch.zeros(4000, 16) + 0.5, A, 3, 17)
    assert collapsed > 100 * gaussian  # a Gaussian sample gives a value of order 1 (sampling noise), collapse grows with N


def test_sigreg_equals_official_formula():
    """Same value as the official EppsPulley + SlicingUnivariateTest computation (lejepa c293d29), written out here."""
    torch.manual_seed(4)
    Z, A = torch.randn(200, 10).double(), diagnostics.sigreg_directions(10, 32, torch.Generator().manual_seed(1))
    t = torch.linspace(0, 3, 17, dtype=torch.float64)
    w = torch.full((17,), 2 * 3 / 16, dtype=torch.float64)
    w[[0, -1]] = 3 / 16
    phi = (-(t ** 2) / 2).exp()
    x_t = (Z @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    official = ((err @ (w * phi)) * 200).mean()
    assert math.isclose(diagnostics.sigreg(Z, A, 3, 17), float(official), rel_tol=1e-12)


def test_dead_dimensions_relative_to_largest_std():
    Z = torch.randn(1000, 4) * torch.tensor([1.0, 0.5, 0.001, 0.0])
    _, frac = diagnostics.dimension_std(Z, 0.01)
    assert frac == 0.5  # the last two dimensions are below 1 % of the largest std
    assert diagnostics.dimension_std(torch.ones(10, 3), 0.01) == (0.0, 1.0)  # complete collapse


def test_measures_nan_when_not_finite():
    Z = torch.randn(20, 4)
    Z[0, 0] = float("nan")
    row, eig = diagnostics.measures(Z, SETTINGS, diagnostics.sigreg_directions(4, 8, torch.Generator().manual_seed(0)))
    assert not row["finite"] and math.isnan(row["rankme"]) and eig is None


# ----------------------------------------------------------------------------- collapse study

TFC_SETTINGS = {
    "encoder": {"channels": [4, 8, 8], "kernel_size": 8, "strides": [8, 1, 1], "padding": 4, "pool_padding": 1,
                "conv_bias": False, "dropout_block1": 0.35, "projector": [16, 8]},
    "loss": {"variant": "paper", "temperature": 0.2, "lambda": 0.5, "delta": 1.0},
    "freq_aug": {"n_components": 1, "alpha": 0.5},
    "time_aug": {"jitter_sigma": 0.1, "scaling_mean": 1.0, "scaling_sigma": 0.1, "max_segments": 5},
    "pretrain": {"epochs": 2, "batch_size": 8, "drop_last": True, "lr": 3e-4, "betas": [0.9, 0.99],
                 "weight_decay": 5e-4},
    "eval_chunk": 7,
}
DIAG = {**SETTINGS, "representations": ["h_t", "h_f", "z_t", "z_f", "z"], "windows": 12, "seed": 5,
        "alignment": True, "sigreg": {"t_max": 3, "n_points": 17, "n_slices": 16, "same_directions_every_epoch": True}}


def test_study_training_equals_tfc_pretrain():
    """With the same seed, the study's loop (with measurements and fixed augmented views) gives exactly
    the weights and losses of harness tfc.pretrain: the measurements do not change the training."""
    X = np.random.default_rng(0).standard_normal((32, 128, 1))
    torch.manual_seed(7)
    np.random.seed(7)
    ref, ref_log = tfc.pretrain(X, TFC_SETTINGS, "cpu")
    torch.manual_seed(7)
    np.random.seed(7)
    rows = collapse.diagnostic_rows(len(X), DIAG["windows"], DIAG["seed"])
    model, log, monitor = collapse.pretrain(X, X[rows], TFC_SETTINGS, DIAG, "cpu")
    for k, v in ref.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    assert np.allclose(ref_log.loss, log.loss)
    assert len(monitor.rows) == 3 * 5  # epochs 0, 1, 2 x five representations
    assert all(r["n"] == 12 for r in monitor.rows)


def test_h_dimension_for_fd_windows():
    """The note in collapse_fd.yaml: h has 128 x 82 = 10,496 dimensions for 5,120-sample windows (FordA: 128 x 10)."""
    e = {**TFC_SETTINGS["encoder"], "channels": [32, 64, 128], "projector": [256, 128]}
    assert tfc.Branch(5120, 1, e).encoder(torch.zeros(1, 1, 5120)).numel() == 10496
    assert tfc.Branch(500, 1, e).encoder(torch.zeros(1, 1, 500)).numel() == 1280
