"""End-to-end check of the first slice on a tiny synthetic stand-in for FordA.

Writes two small FordA-format files (two classes: sine vs noise), converts
them, pretrains TS2Vec for a few iterations and runs both probes. This checks
that the pieces fit together; it says nothing about TS2Vec's accuracy.
"""

import glob
import os

import numpy as np
import pandas as pd
import yaml

from harness.data import canonical, forda
from scripts import convert_forda, evaluate, pretrain


def write_fake_forda(folder, n, length=64, seed=0):
    rng = np.random.default_rng(seed)
    for part in ("TRAIN", "TEST"):
        labels = np.repeat([-1, 1], n // 2)
        t = np.linspace(0, 4 * np.pi, length)
        x = np.where(labels[:, None] == 1, np.sin(t), 0) + 0.3 * rng.standard_normal((n, length))
        np.savetxt(os.path.join(folder, f"FordA_{part}.tsv"), np.column_stack([labels, x]), delimiter="\t")


def test_slice_end_to_end(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fake_forda(raw, n=200)
    convert_forda.convert(str(raw), str(tmp_path / "canonical"))

    # Converted values equal the file values.
    meta = canonical.load_metadata(tmp_path / "canonical", forda.DATASET)
    assert len(meta["captures"]) == 400
    x0 = canonical.load_signal(tmp_path / "canonical", "forda", "forda", "train_0000", "x")
    assert np.allclose(x0, np.loadtxt(raw / "FordA_TRAIN.tsv")[0, 1:])

    # A config like configs/ts2vec_forda.yaml, but tiny.
    with open("configs/ts2vec_forda.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = str(tmp_path / "canonical")
    cfg["seeds"] = [0, 1]
    cfg["ts2vec"].update(repr_dims=16, hidden_dims=8, depth=2, n_iters=5)
    cfg["evaluation"]["label_fractions"] = [0.1, 1.0]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.chdir(tmp_path)  # runs/ is created in the current folder
    pretrain.main(str(config_path))
    run = glob.glob("runs/*")[0]
    assert np.load(f"{run}/seed_0/Z_train.npy").shape == (200, 16)
    evaluate.main(run)
    results = pd.read_csv(f"{run}/results.csv")
    assert len(results) == 2 * 2 * 2  # seeds x label fractions x probes
    assert results.accuracy.between(0, 1).all()
