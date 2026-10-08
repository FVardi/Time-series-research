"""TS2Vec, wrapped (D10 mode A, D19).

The official code lives unmodified in external/ts2vec (git submodule,
commit b0088e1). This file only calls it:
    pretrain(X, settings, device) -> trained model
    encode(model, X)              -> one vector per series
    save(model, path)
"""

import numpy as np

from harness.methods.external import import_official

TS2Vec = import_official("ts2vec", "ts2vec", local_names=("ts2vec", "models", "utils")).TS2Vec


def pretrain(X, s, device):
    """Train TS2Vec without labels.

    X: array (n_series, n_samples, n_channels), training inputs only.
    s: the 'ts2vec' section of the config; every setting is given explicitly (D28).
    Returns the trained model and the loss per epoch.
    """
    model = TS2Vec(
        input_dims=X.shape[2],
        output_dims=s["repr_dims"],
        hidden_dims=s["hidden_dims"],
        depth=s["depth"],
        device=device,
        lr=s["lr"],
        batch_size=s["batch_size"],
        max_train_length=s["max_train_length"],
        temporal_unit=s["temporal_unit"],
    )
    # Conversion to 32-bit happens here, at training (D23).
    loss_log = model.fit(X.astype(np.float32), n_iters=s["n_iters"])
    return model, loss_log


def save(model, path):
    """Save the model weights (official method)."""
    model.save(path)


def encode(model, X):
    """One vector per series: max-pooling over the whole series, as in TS2Vec's classification evaluation."""
    return model.encode(X.astype(np.float32), encoding_window="full_series")
