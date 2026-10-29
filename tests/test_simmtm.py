"""Tests for the SimMTM adapter. Skipped if the submodule external/simmtm is missing."""

import os

import numpy as np
import pytest
import torch
import yaml

from harness.methods.external import EXTERNAL

CODE = os.path.join(EXTERNAL, "simmtm", "SimMTM_Classification", "code")
HAVE_OFFICIAL = os.path.isdir(CODE)
pytestmark = pytest.mark.skipif(not HAVE_OFFICIAL, reason="external/simmtm not present")
if HAVE_OFFICIAL:
    from harness.methods import simmtm

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "simmtm_forda.yaml")


def settings():
    """The 'simmtm' section of the FordA config, with the open batch size filled in for testing."""
    with open(CONFIG) as f:
        s = yaml.safe_load(f)["simmtm"]
    s["changed"]["batch_size"] = 16  # test value only
    return s


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
    """Each SleepEEG setting is either checked (official_config) or changed (changed), and the
    checked values equal the official file."""
    s = settings()
    assert set(all_names(simmtm.SleepEEGConfig())) == set(s["official_config"]) | set(s["changed"])
    simmtm.official_config(s)  # raises if a value differs from the official file


def test_difference_from_official_config_stops():
    s = settings()
    s["official_config"]["kernel_size"] = 8
    with pytest.raises(ValueError):
        simmtm.official_config(s)


def test_open_placeholder_stops():
    s = settings()
    s["changed"]["batch_size"] = None
    with pytest.raises(ValueError, match="placeholder"):
        simmtm.pretrain(np.zeros((40, 500, 1)), s, "cpu")


def test_sizes_fit_forda():
    """23 is the encoder's output length for 500 samples; the decoder is resized to 23 x 128 -> 500."""
    s = settings()
    model = simmtm.build_model(simmtm.official_config(s), simmtm.main_args(s), "cpu")
    assert (model.head.in_features, model.head.out_features) == (23 * 128, 500)
    s["changed"]["CNNoutput_channel"] = 24
    with pytest.raises(ValueError, match="CNNoutput_channel"):
        simmtm.build_model(simmtm.official_config(s), simmtm.main_args(s), "cpu")


def test_pretrain_and_encode():
    """Two epochs on random FordA-sized data: finite losses, one 23 x 128 vector per series,
    and the values do not depend on how the series are batched when encoding."""
    s = settings()
    s["main_args"]["pretrain_epoch"] = 2
    s["encode_batch_size"] = 7
    X = np.random.default_rng(0).standard_normal((40, 500, 1))
    torch.manual_seed(0)
    np.random.seed(0)
    model, losses = simmtm.pretrain(X, s, "cpu")
    assert len(losses) == 2 and np.all(np.isfinite(losses))
    Z = simmtm.encode(model, X[:20])
    assert Z.shape == (20, 23 * 128)
    model.encode_batch_size = 1
    assert np.allclose(Z, simmtm.encode(model, X[:20]), atol=1e-5)
    with pytest.raises(ValueError, match="channel"):
        simmtm.pretrain(np.zeros((40, 500, 2)), s, "cpu")


def test_pretraining_equals_official_main(tmp_path, monkeypatch):
    """One pretraining epoch of the adapter gives exactly the weights of the official main.py
    (SleepEEG sizes: 178 samples, so the official decoder is kept; CPU).

    main.py reads ./dataset/<name>/{train,test}.pt, so random series are written there.
    It shuffles the datasets while loading and then seeds NumPy and PyTorch (set_seed);
    the adapter is seeded before it loads. The datasets are shuffled with the same NumPy seed
    in both, and the adapter's Load_Dataset is wrapped to re-seed NumPy afterwards as main.py
    does, so the masks are drawn from the same random sequence (the re-seeding is done just
    before the model is built, the adapter's next step after loading).
    """
    import runpy
    import sys
    import warnings

    seed, n = 2023, 40
    rng = np.random.default_rng(3)
    for name in ("SleepEEG", "Epilepsy"):  # Epilepsy: main.py also loads a fine-tuning target
        os.makedirs(tmp_path / "dataset" / name)
        for split in ("train", "test"):
            torch.save({"samples": torch.tensor(rng.standard_normal((n, 1, 178)), dtype=torch.float32),
                        "labels": torch.zeros(n, dtype=torch.long)}, tmp_path / "dataset" / name / f"{split}.pt")
    X = torch.load(tmp_path / "dataset" / "SleepEEG" / "train.pt")["samples"].numpy().transpose(0, 2, 1).astype(np.float64)

    # Ours: SleepEEG sizes, everything else as in the FordA config, one epoch.
    s = settings()
    s["changed"].update(TSlength_aligned=178, CNNoutput_channel=10, batch_size=32)  # SleepEEG values
    s["main_args"]["pretrain_epoch"] = 1
    original = simmtm.build_model

    def reseed_then_build(*args, **kwargs):
        np.random.seed(seed)  # as main.py's set_seed after data_generator
        return original(*args, **kwargs)

    monkeypatch.setattr(simmtm, "build_model", reseed_then_build)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ours = simmtm.pretrain(X, s, "cpu")[0].state_dict()
    monkeypatch.undo()

    # Theirs: the official script, pretraining for one epoch (fine-tuning starts only at epoch 2).
    cwd, filters, path, argv = os.getcwd(), warnings.filters[:], sys.path[:], sys.argv[:]
    names = ("model", "loss", "trainer", "utils", "layers", "dataloader", "config_files")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in names}
    os.chdir(tmp_path)
    sys.path.insert(0, CODE)
    sys.argv = ["main.py", "--pretrain_epoch", "1", "--device", "cpu", "--seed", str(seed)]
    np.random.seed(seed)
    try:
        runpy.run_path(os.path.join(CODE, "main.py"), run_name="__main__")
    finally:  # undo the script's side effects
        os.chdir(cwd)
        warnings.filters[:], sys.path[:], sys.argv = filters, path, argv
        for k in [k for k in sys.modules if k.split(".")[0] in names]:
            del sys.modules[k]
        sys.modules.update(saved)
    found = [os.path.join(r, f) for r, _, fs in os.walk(tmp_path / "experiments_logs") for f in fs if f == "ckp_ep1.pt"]
    theirs = torch.load(found[0], map_location="cpu")["model_state_dict"]
    assert set(ours) == set(theirs)
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_two_channels_are_encoded_and_rebuilt():
    """D159: the decoder rebuilds both channels (2 x length values) and pretraining runs."""
    s = settings()
    s["changed"].update({"TSlength_aligned": 2048, "CNNoutput_channel": 87, "input_channels": 2})
    s["main_args"]["pretrain_epoch"] = 1
    X = np.random.default_rng(4).standard_normal((32, 2048, 2))
    torch.manual_seed(0)
    np.random.seed(0)
    model, losses = simmtm.pretrain(X, s, "cpu")
    assert model.head.out_features == 2 * 2048 and np.all(np.isfinite(losses))
    assert simmtm.encode(model, X[:4]).shape == (4, 87 * 128)
    # the training set keeps both channels, in Load_Dataset's shuffled order
    np.random.seed(7)
    ds = simmtm.training_set(X, simmtm.official_config(s))
    np.random.seed(7)
    order = list(range(len(X)))
    np.random.shuffle(order)  # the same draw as Load_Dataset's np.random.shuffle on the list of pairs
    assert np.allclose(ds[0][0].numpy(), X[order[0]].T.astype(np.float32))
