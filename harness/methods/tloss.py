"""T-Loss (Franceschi et al., 2019), wrapped (D10 mode A, D19, D32).

The official code lives unmodified in external/tloss (git submodule,
commit 4aff592). This file only calls it:
    pretrain(X, settings, device) -> trained model
    encode(model, X)              -> one vector per series
    save(model, path)

Only the encoder training and encoding of the official code are used. Its
classifier code is not: evaluation runs through harness/evaluate.py, whose SVM
probe is the same protocol T-Loss used for its published results.
"""

import numpy as np

from harness.methods.external import import_official

# Fix needed to run on Python 3.12 (D19: fixes go here, the official files stay unedited):
# T-Loss's losses/__init__.py and networks/__init__.py load their submodules with import
# functions that Python 3.12 removed. We skip those two __init__ files and import the
# two submodules T-Loss actually uses directly. Their code is unchanged.
CausalCNNEncoderClassifier = import_official(
    "tloss", "scikit_wrappers",
    local_names=("scikit_wrappers", "utils", "losses", "networks"),
    stub_packages=("losses", "networks"),
    preload=("losses.triplet_loss", "networks.causal_cnn"),
).CausalCNNEncoderClassifier


def _to_tloss(X):
    """Our layout is (series, time, channels); T-Loss expects (series, channels, time).

    T-Loss trains in 64-bit floats (its encoder is converted with .double()),
    so the data stays 64-bit here, as the official code requires.
    """
    return np.ascontiguousarray(np.transpose(X, (0, 2, 1)), dtype=np.float64)


def pretrain(X, s, device):
    """Train the T-Loss encoder without labels.

    X: array (n_series, n_samples, n_channels), training inputs only.
    s: the 'tloss' section of the config; every setting is given explicitly (D32).
    Returns the trained model and an empty loss log (the official code does not record losses).
    """
    model = CausalCNNEncoderClassifier(
        compared_length=s["compared_length"],  # None = compare full-length series
        nb_random_samples=s["nb_random_samples"],  # K, the number of negative samples
        negative_penalty=s["negative_penalty"],
        batch_size=s["batch_size"],
        nb_steps=s["nb_steps"],
        lr=s["lr"],
        penalty=s["penalty"],  # only used by T-Loss's own classifier, which we do not use
        early_stopping=s["early_stopping"],
        channels=s["channels"],
        depth=s["depth"],
        reduced_size=s["reduced_size"],
        out_channels=s["out_channels"],
        kernel_size=s["kernel_size"],
        in_channels=X.shape[2],
        cuda=(device == "cuda"),
        gpu=0,
    )
    model.fit_encoder(_to_tloss(X))  # no labels: early stopping is off (D32)
    return model, []


def encode(model, X):
    """One vector per series (T-Loss max-pools over time inside its encoder)."""
    return model.encode(_to_tloss(X))


def save(model, path):
    """Save the encoder weights (official method; writes <path>_CausalCNN_encoder.pth)."""
    model.save_encoder(path)
