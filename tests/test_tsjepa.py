"""Tests for the TS-JEPA adapter. Skipped if the submodule external/tsjepa is missing."""

import os
import random

import numpy as np
import pytest
import torch

from harness.methods.external import EXTERNAL

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(EXTERNAL, "tsjepa", "src")),
                                reason="external/tsjepa not present")


def settings(masking="per_sample"):
    """Small settings for fast tests (not the experiment's settings)."""
    return {"n_patches": 5, "mask_ratio": 0.6, "masking": masking, "batch_size": 4, "epochs": 2,
            "ema_momentum": 0.998, "ipe_scale": 1.25,
            "encoder": {"embed_dim": 16, "nhead": 2, "num_layers": 1, "kernel_size": 3, "embed_bias": True},
            "predictor": {"embed_dim": 16, "nhead": 2, "num_layers": 1},
            "optimizer": {"lr": 1e-3, "lr_end_factor": 0.5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            "pooling": "mean", "encode_batch_size": 3, "print_every": 1}


def test_official_apply_mask_takes_tokens_from_first_series():
    """Documents the official bug and the per-sample call that avoids it."""
    from harness.methods.tsjepa import official
    x = torch.arange(3 * 5 * 2, dtype=torch.float).reshape(3, 5, 2)
    idx = torch.tensor([[0, 1], [2, 3], [1, 4]])
    expected = torch.stack([x[b, idx[b]] for b in range(3)])
    assert torch.equal(official.apply_mask(x, [idx]), expected)  # per_sample: correct
    assert torch.equal(official.apply_mask(x, idx), torch.stack([x[0, idx[b]] for b in range(3)]))  # official: from series 0


def test_random_masks_like_official_loader():
    from harness.methods.tsjepa import random_masks
    random.seed(0)
    masked, context = random_masks(batch_size=6, n_patches=10, n_masked=7)
    assert masked.shape == (6, 7) and context.shape == (6, 3)
    for m, c in zip(masked.tolist(), context.tolist()):
        assert sorted(m + c) == list(range(10)) and c == sorted(c)  # disjoint, complete, context ascending


@pytest.mark.parametrize("masking", ["official", "per_sample"])
def test_pretrain_and_encode(masking):
    from harness.methods import tsjepa
    torch.manual_seed(0)
    random.seed(0)
    X = np.random.default_rng(0).normal(size=(10, 50, 1))
    model, log = tsjepa.pretrain(X, settings(masking), "cpu")
    assert len(log) == 2 and all(np.isfinite(log))
    Z = tsjepa.encode(model, X)
    assert Z.shape == (10, 16)
    Z_one = np.concatenate([tsjepa.encode(model, X[i:i + 1]) for i in range(10)])
    assert np.allclose(Z, Z_one, atol=1e-5)  # a series' vector does not depend on the other series in its batch


def test_placeholders_and_patching():
    """D154: patches of length // n_patches samples per channel (the rest unused); with several
    channels each patch holds the same time span of every channel, one after the other."""
    from harness.methods import tsjepa
    with pytest.raises(ValueError, match="placeholder"):
        tsjepa.check_settings({"encoder": {"num_layers": None}})
    X = np.random.default_rng(0).standard_normal((2, 51, 2))
    P = tsjepa.to_patches(X, 5).numpy()  # 10 samples per channel and patch; sample 50 unused
    assert P.shape == (2, 5, 20)
    assert np.allclose(P[1, 3, :10], X[1, 30:40, 0]) and np.allclose(P[1, 3, 10:], X[1, 30:40, 1])
    one = X[:, :50, :1]  # one channel, divisible: exactly the previous patching
    assert np.array_equal(tsjepa.to_patches(one, 5).numpy(), one[:, :, 0].reshape(2, 5, 10).astype(np.float32))


def test_training_loop_equals_official_script(tmp_path):
    """Our written-out loop (masking: official) gives exactly the weights of the official pretrain.py.

    The official script is run as distributed, with two pieces replaced because they only work on
    the weather CSV: its argument parsing (settings and seed set here) and its data loader (replaced
    by one that patches and masks the same way, item by item, as the official CSVDataLoader).
    pretrain.py reads dataset[0] three times for array sizes before training; those masks are
    discarded, so the replacement does not draw them (keeps the random sequences aligned).
    """
    import runpy
    import sys
    import warnings

    from harness.methods import tsjepa
    from harness.methods.external import EXTERNAL

    s = settings("official")
    s.update(n_patches=10, mask_ratio=0.7, batch_size=8, epochs=3)
    X = np.random.default_rng(1).normal(size=(37, 320, 1))  # 37 series: the last batch is smaller
    P = tsjepa.to_patches(X, 10)

    torch.manual_seed(0)
    random.seed(0)
    ours = tsjepa.pretrain(X, s, "cpu")[0].state_dict()

    off = os.path.join(EXTERNAL, "tsjepa")
    sys.path.insert(0, off)
    import main.utils as mu
    import src.data_loaders.data_loader as dl

    class Items(torch.utils.data.Dataset):
        calls = 0

        def __len__(self):
            return len(P)

        def __getitem__(self, i):
            Items.calls += 1
            if Items.calls <= 3:  # the three size look-ups
                return P[i], torch.zeros(7, dtype=torch.long), torch.zeros(3, dtype=torch.long)
            m = random.sample(range(10), 7)
            return P[i], torch.tensor(m), torch.tensor([j for j in range(10) if j not in m])

    def args(config):
        torch.manual_seed(0)
        random.seed(0)
        e, p = s["encoder"], s["predictor"]
        config.update(lr=s["optimizer"]["lr"], num_epochs=3, batch_size=8, ema_momentum=0.998, ipe_scale=1.25,
                      encoder_embed_dim=e["embed_dim"], encoder_nhead=e["nhead"], encoder_num_layers=e["num_layers"],
                      encoder_kernel_size=e["kernel_size"], predictor_embed=p["embed_dim"], predictor_nhead=p["nhead"],
                      predictor_num_layers=p["num_layers"], path_data="unused", ratio_patches=10, mask_ratio=0.7,
                      checkpoint_save=2, checkpoint_print=100, path_save=str(tmp_path / "ckpt"))
        return config

    dl.get_jepa_loaders = lambda *a, **kw: torch.utils.data.DataLoader(Items(), batch_size=8, shuffle=True)
    mu.prepare_args_pretrain = args
    cwd, filters, cuda_available = os.getcwd(), warnings.filters[:], torch.cuda.is_available
    os.chdir(off)
    # The script picks the GPU itself if there is one; ours ran on the CPU, and GPU arithmetic
    # differs slightly, so the script is made to see no GPU for this comparison.
    torch.cuda.is_available = lambda: False
    try:
        runpy.run_path("pretrain.py", run_name="__main__")
    finally:  # undo the script's side effects: working folder, warnings switched off, imported modules
        torch.cuda.is_available = cuda_available
        os.chdir(cwd)
        warnings.filters[:] = filters
        sys.path.remove(off)
        for name in [n for n in sys.modules if n.split(".")[0] in ("src", "main", "config")]:
            del sys.modules[name]
    theirs = torch.load(tmp_path / "ckpt_epoch_2.pt", map_location="cpu")["encoder"]
    assert set(ours) == set(theirs)
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_two_channels_pretrain_and_encode():
    """D154: two channels train and encode; the tokenizer check and the channel check stop wrong inputs."""
    from harness.methods import tsjepa
    X = np.random.default_rng(1).standard_normal((12, 64, 2))  # 5 patches of 12 samples per channel (+4 unused)
    torch.manual_seed(0)
    random.seed(0)
    model, losses = tsjepa.pretrain(X, settings(), "cpu")
    assert len(losses) == 2 and np.all(np.isfinite(losses))
    assert tsjepa.encode(model, X).shape == (12, 16)
    with pytest.raises(ValueError, match="channel"):
        tsjepa.encode(model, X[:, :, :1])
    with pytest.raises(ValueError, match="kernel_size"):
        tsjepa.pretrain(np.zeros((12, 70, 2)), settings(), "cpu")  # 14 samples per channel and patch, kernel 3
