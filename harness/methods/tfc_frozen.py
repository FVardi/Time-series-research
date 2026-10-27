"""Our TF-C (harness/methods/tfc.py) in the frozen-probe pipeline (scripts/pretrain.py, D123).

Only adapts the interface that scripts/pretrain.py expects; pretraining is tfc.pretrain
and the representation is tfc.encode, i.e. [z_T; z_F] per series (D41 item 10, D123).
The loss variant comes from the config (tfc.loss.variant, D122).
"""

from harness.methods import tfc

check_settings = tfc.check_settings  # pretrain.py stops before any work if a setting is still null


def pretrain(X, s, device, train=True):
    """tfc.pretrain; returns the model (with what encode needs) and the loss per epoch as a list.
    train: False returns the model as built, untrained (random-init baseline, D130)."""
    model, log = tfc.pretrain(X, s, device, train)
    return {"tfc": model, "chunk": s["eval_chunk"], "device": device}, log["loss"].tolist()


def encode(model, X):
    """[z_T; z_F] per series, in evaluation mode (tfc.encode)."""
    return tfc.encode(model["tfc"], X, model["chunk"], model["device"])


def save(model, path):
    tfc.save(model["tfc"], path + ".pt")
